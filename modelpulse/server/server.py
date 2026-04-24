"""
Control plane : WebSocket /ws  (signals only — no binary blobs)
Data plane    : HTTP  /manifest, /shards/*, /models/upload,
                      /models/delta, /shards/delta/<model_id>/*
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import struct
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from rich.console import Console

from modelpulse.shared.ws_protocol import MsgType, WsMessage

cli = typer.Typer(name="server", add_completion=False)
_console = Console(highlight=False)
log = logging.getLogger("modelpulse.server")

PING_INTERVAL = 20.0

# Delta manifest filename
DELTA_MANIFEST_FILE = "delta_manifest.json"


# SHA-256 helpers

def _sha256_of_upload(upload: UploadFile, data: bytes) -> str:
    """Return hex SHA-256 of raw shard payload (strip SHRD header if present)."""
    if len(data) >= 12 and data[:4] == b"SHRD":
        hdr_len = struct.unpack("<I", data[8:12])[0]
        payload = data[12 + hdr_len:]
    else:
        payload = data
    return hashlib.sha256(payload).hexdigest()


def _sha256_of_file(path: Path) -> str:
    """Return hex SHA-256 of the tensor payload stored in a .shard file."""
    data = path.read_bytes()
    if len(data) >= 12 and data[:4] == b"SHRD":
        hdr_len = struct.unpack("<I", data[8:12])[0]
        payload = data[12 + hdr_len:]
    else:
        payload = data
    return hashlib.sha256(payload).hexdigest()


# ConnectionManager

class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[WebSocket, str] = {}
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, client_id: str) -> None:
        async with self._lock:
            self._connections[ws] = client_id
        log.info("WS connected  client_id=%s  total=%d", client_id, self.count)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            cid = self._connections.pop(ws, "?")
        log.info("WS disconnected  client_id=%s  total=%d", cid, self.count)

    async def send(self, ws: WebSocket, msg: WsMessage) -> bool:
        try:
            await ws.send_text(msg.to_json())
            return True
        except Exception as exc:
            log.warning("send failed → %s: %s", self._connections.get(ws, "?"), exc)
            return False

    async def broadcast(self, msg: WsMessage) -> int:
        async with self._lock:
            targets = list(self._connections.items())

        ok, dead = 0, []
        for ws, cid in targets:
            try:
                await ws.send_text(msg.to_json())
                ok += 1
            except Exception as exc:
                log.warning("broadcast failed → %s: %s", cid, exc)
                dead.append(ws)

        async with self._lock:
            for ws in dead:
                self._connections.pop(ws, None)
        return ok

    @property
    def count(self) -> int:
        return len(self._connections)

    def client_ids(self) -> list[str]:
        return list(self._connections.values())


# App

def create_app(
    shard_dir: Path,
    metrics_log: Path,
    *,
    port: int = 8000,
    ping_interval: float = PING_INTERVAL,
) -> FastAPI:

    # active model state — mutable box so closures can write it
    _state: dict[str, Any] = {
        "model_id": "",
        # iteration counter: how many delta uploads have been applied to the
        # current active model lineage.  Reset to 0 on a full upload.
        "iteration": 0,
    }

    # helpers

    def _model_dir(model_id: str) -> Path:
        return shard_dir / model_id

    def _active_dir() -> Path:
        mid = _state["model_id"]
        if mid:
            d = _model_dir(mid)
            if d.exists():
                return d
        return shard_dir          # legacy flat layout fallback

    def _load_manifest() -> dict | None:
        p = _active_dir() / "manifest.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def _load_delta_manifest(model_id: str) -> dict | None:
        """Load delta_manifest.json for the given model_id, if it exists."""
        p = _model_dir(model_id) / DELTA_MANIFEST_FILE
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def _log_metrics(payload: dict) -> None:
        with open(metrics_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    # lifespan

    manager: ConnectionManager | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal manager
        manager = ConnectionManager()
        app.state.manager = manager

        async def _ping_task():
            while True:
                await asyncio.sleep(ping_interval)
                if manager.count:
                    await manager.broadcast(WsMessage.ping())

        task = asyncio.create_task(_ping_task())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="modelpulse / Device A", version="0.3.0", lifespan=lifespan)

    def _mgr() -> ConnectionManager:
        return app.state.manager

    # HTTP — health / manifest / shards

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "active_model_id": _state["model_id"],
            "active_iteration": _state["iteration"],
            "shard_dir": str(shard_dir),
            "ws_clients": _mgr().count,
        }

    @app.get("/manifest")
    def get_manifest():
        m = _load_manifest()
        if m is None:
            raise HTTPException(404, "No manifest.json for the active model")
        return JSONResponse(content=m)

    @app.get("/shards/{filename}")
    def get_shard(filename: str):
        if "/" in filename or ".." in filename or "\\" in filename:
            raise HTTPException(400, "Invalid filename")
        if not filename.endswith(".shard"):
            raise HTTPException(400, "Only .shard files are served")
        p = _active_dir() / filename
        if not p.exists():
            raise HTTPException(404, f"Not found: {filename}")
        return FileResponse(str(p), media_type="application/octet-stream", filename=filename)

    # HTTP — delta manifest + delta shards

    @app.get("/shards/delta/{model_id}/manifest")
    def get_delta_manifest(model_id: str):
        """
        Return the delta_manifest.json for *model_id*.

        The delta manifest lists only the shards that changed relative to the
        base model.  Clients fetch this first, then fetch only those shards
        from  GET /shards/delta/{model_id}/{filename}.
        """
        dm = _load_delta_manifest(model_id)
        if dm is None:
            raise HTTPException(
                404,
                f"No delta manifest for model_id={model_id!r}. "
                "This model was uploaded as a full model, not a delta.",
            )
        return JSONResponse(content=dm)

    @app.get("/shards/delta/{model_id}/{filename}")
    def get_delta_shard(model_id: str, filename: str):
        """
        Serve a single changed shard file that belongs to a delta update.

        Delta shard files live in  {shard_dir}/{model_id}/  alongside the
        full-model shards that were *not* changed.  The client is responsible
        for merging them into its cached shard dict.
        """
        if "/" in filename or ".." in filename or "\\" in filename:
            raise HTTPException(400, "Invalid filename")
        if not filename.endswith(".shard"):
            raise HTTPException(400, "Only .shard files are served")

        dm = _load_delta_manifest(model_id)
        if dm is None:
            raise HTTPException(404, f"No delta manifest for model_id={model_id!r}")

        changed = dm.get("changed_shards", {})
        # Check if the requested filename is either a key in changed_shards
        # or the 'file' field of one of the entries.
        is_changed = (filename in changed) or any(s.get("file") == filename for s in changed.values())

        if not is_changed:
            raise HTTPException(
                404,
                f"{filename!r} is not listed as a changed shard for {model_id!r}",
            )

        p = _model_dir(model_id) / filename
        if not p.exists():
            raise HTTPException(404, f"Shard file missing on server: {filename}")

        return FileResponse(str(p), media_type="application/octet-stream", filename=filename)

    # HTTP — metrics

    @app.post("/metrics")
    async def post_metrics(request: Request):
        _log_metrics(await request.json())
        return {"status": "received"}

    @app.get("/results")
    def get_all_results():
        if not metrics_log.exists():
            return []
        return [
            json.loads(l)
            for l in metrics_log.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]

    @app.get("/results/latest")
    def get_latest_result():
        if not metrics_log.exists():
            raise HTTPException(404, "No metrics yet")
        lines = [l for l in metrics_log.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            raise HTTPException(404, "No metrics yet")
        return json.loads(lines[-1])

    # HTTP — full model upload

    @app.post("/models/upload")
    async def upload_model(
        model_id: Annotated[str, Form()],
        manifest_file: Annotated[UploadFile, File(alias="manifest")],
        shard_files: Annotated[list[UploadFile], File(alias="shards")],
    ):
        """
        Upload a complete model (manifest.json + *.shard files).

        multipart/form-data
          model_id   str          unique slug, e.g. "qwen2.5-0.5b-q4"
          manifest   file         must be named manifest.json
          shards     file[]       one or more .shard files

        On success
          • Files written to  {shard_dir}/{model_id}/
          • Active model pointer switched atomically
          • Iteration counter reset to 0
          • model_ready broadcast (update_type="full") to all WS clients
        """
        # validate inputs
        if not model_id or any(c in model_id for c in (".", "/", "\\", "..")):
            raise HTTPException(400, f"Invalid model_id: {model_id!r}")

        for upload in [manifest_file, *shard_files]:
            name = upload.filename or ""
            if not name or any(c in name for c in ("/", "\\", "..")):
                raise HTTPException(400, f"Unsafe filename: {name!r}")

        if manifest_file.filename != "manifest.json":
            raise HTTPException(
                400,
                f"manifest field must be named 'manifest.json', got {manifest_file.filename!r}",
            )

        bad = [u.filename for u in shard_files if not (u.filename or "").endswith(".shard")]
        if bad:
            raise HTTPException(400, f"Non-.shard file(s): {bad}")

        # write to disk
        dest = _model_dir(model_id)
        dest.mkdir(parents=True, exist_ok=True)

        written: list[str] = []
        try:
            for upload in [manifest_file, *shard_files]:
                dest_path = dest / upload.filename
                with open(dest_path, "wb") as fh:
                    while chunk := await upload.read(65_536):
                        fh.write(chunk)
                written.append(upload.filename)
                log.info(
                    "upload  model=%s  file=%s  size=%d B",
                    model_id, upload.filename, dest_path.stat().st_size,
                )
        except Exception as exc:
            log.exception("write error for model=%s: %s", model_id, exc)
            raise HTTPException(500, f"Write error: {exc}") from exc

        # validate manifest
        manifest_path = dest / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(422, f"manifest.json is not valid JSON: {exc}") from exc

        if not manifest.get("model_id"):
            manifest["model_id"] = model_id
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # switch active model
        previous = _state["model_id"]
        _state["model_id"] = model_id
        _state["iteration"] = 0          # full upload → reset iteration counter
        log.info("active model  %s → %s  (full upload, iteration reset to 0)", previous or "(none)", model_id)

        # broadcast model_ready (full)
        msg = WsMessage.model_ready(
            {},
            model_id=model_id,
            update_type="full",
            base_model_id=None,
        )
        reached = await _mgr().broadcast(msg)
        log.info("model_ready[full] broadcast  model=%s  clients=%d", model_id, reached)

        return {
            "status": "uploaded",
            "update_type": "full",
            "model_id": model_id,
            "iteration": 0,
            "files_written": written,
            "clients_notified": reached,
            "dest_dir": str(dest),
        }

    # HTTP — delta upload

    @app.post("/models/delta")
    async def upload_delta(
        model_id: Annotated[str, Form()],
        base_model_id: Annotated[str, Form()],
        shard_files: Annotated[list[UploadFile], File(alias="shards")],
    ):
        """
        Upload only the changed (quantised) tensor shard blocks on top of an
        existing full model.

        multipart/form-data
          model_id       str        slug for *this delta*, e.g. "qwen2.5-0.5b-q4-d1"
          base_model_id  str        slug of the full model this patches
          shards         file[]     only the changed .shard files

        Server behaviour
          1. Validate that base_model_id exists on disk.
          2. Copy the base model directory to a new directory for model_id
             (so the delta model is self-contained — unchanged shards come
             from the copy, changed shards are overwritten).
          3. Compute SHA-256 of every incoming shard and compare it to the
             corresponding file in base_model_id to confirm they really differ.
          4. Write the changed shards into the new model_id directory,
             overwriting the copied baseline versions.
          5. Build and persist delta_manifest.json, listing only the changed
             shards with their new SHA-256 digests and byte counts.
          6. Switch the active model pointer to model_id.
          7. Broadcast model_ready (update_type="delta") to WS clients.

        On success clients can:
          • Call GET /shards/delta/{model_id}/manifest  to learn which shards changed.
          • Call GET /shards/delta/{model_id}/{filename} for each changed shard.
          • Patch their in-memory shard cache and reassemble without re-downloading
            the unchanged shards.
        """
        # validate
        for slug, label in [(model_id, "model_id"), (base_model_id, "base_model_id")]:
            if not slug or any(c in slug for c in (".", "/", "\\", "..")):
                raise HTTPException(400, f"Invalid {label}: {slug!r}")

        base_dir = _model_dir(base_model_id)
        if not base_dir.exists():
            raise HTTPException(
                404,
                f"base_model_id={base_model_id!r} not found on server. "
                "Upload the full model first.",
            )

        if not shard_files:
            raise HTTPException(400, "No shard files provided")

        bad = [u.filename for u in shard_files if not (u.filename or "").endswith(".shard")]
        if bad:
            raise HTTPException(400, f"Non-.shard file(s): {bad}")

        for upload in shard_files:
            name = upload.filename or ""
            if not name or any(c in name for c in ("/", "\\", "..")):
                raise HTTPException(400, f"Unsafe filename: {name!r}")

        # copy base → new model dir
        dest = _model_dir(model_id)
        if dest.exists():
            shutil.rmtree(dest)            # clean slate for idempotent re-uploads
        shutil.copytree(base_dir, dest)
        log.info("delta  copied base %s → %s", base_model_id, model_id)

        # read, verify, write changed shards
        # Load manifest to map filenames to tensor names
        manifest_path = dest / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            file_to_tensor = {s["file"]: name for name, s in manifest.get("shards", {}).items()}
        except Exception as exc:
            log.exception("failed to load manifest for delta: %s", exc)
            file_to_tensor = {}

        changed_shards: dict[str, dict] = {}
        written: list[str] = []

        try:
            for upload in shard_files:
                fname = upload.filename
                raw   = await upload.read()                 # read entire shard into RAM

                # Compare against the baseline copy we just wrote
                baseline_path = dest / fname
                if baseline_path.exists():
                    old_sha = _sha256_of_file(baseline_path)
                else:
                    old_sha = ""

                new_sha = _sha256_of_upload(upload, raw)

                if old_sha == new_sha:
                    log.warning(
                        "delta  shard %s is identical to baseline — "
                        "including anyway (caller said it changed)", fname
                    )

                # Overwrite the copied baseline with the new version
                dest_path = dest / fname
                dest_path.write_bytes(raw)
                written.append(fname)

                # Record in the delta index
                # Use the tensor name (from manifest) as the key, falling back to filename
                tensor_key = file_to_tensor.get(fname, fname)
                changed_shards[tensor_key] = {
                    "file":    fname,
                    "sha256":  new_sha,
                    "nbytes":  len(raw),
                    "old_sha": old_sha,
                }

                # If the shard has a SHRD header, update the manifest with new metadata
                if len(raw) >= 12 and raw[:4] == b"SHRD":
                    try:
                        hdr_len = struct.unpack("<I", raw[8:12])[0]
                        hdr_json = json.loads(raw[12:12+hdr_len].decode("utf-8"))
                        if tensor_key in manifest.get("shards", {}):
                            s_info = manifest["shards"][tensor_key]
                            s_info["nbytes"]         = hdr_json.get("nbytes", s_info["nbytes"])
                            s_info["ggml_type"]      = hdr_json.get("ggml_type", s_info["ggml_type"])
                            s_info["ggml_type_name"] = hdr_json.get("ggml_type_name", s_info.get("ggml_type_name"))
                            s_info["dims"]           = hdr_json.get("dims", s_info["dims"])
                            s_info["sha256"]         = hdr_json.get("sha256", s_info["sha256"])
                    except Exception as exc:
                        log.warning("failed to parse SHRD header for %s: %s", fname, exc)
                log.info(
                    "delta  model=%s  shard=%s  size=%d B  sha=%s…",
                    model_id, fname, len(raw), new_sha[:12],
                )

        except Exception as exc:
            log.exception("delta write error for model=%s: %s", model_id, exc)
            shutil.rmtree(dest, ignore_errors=True)
            raise HTTPException(500, f"Write error: {exc}") from exc

        # patch manifest.json
        # (manifest was already loaded above)

        manifest["model_id"]      = model_id
        manifest["base_model_id"] = base_model_id
        manifest["total_bytes"]   = sum(s["nbytes"] for s in manifest.get("shards", {}).values())
        manifest["tensor_count"]  = len(manifest.get("shards", {}))
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # determine iteration number
        # If we are chaining deltas from the same lineage, increment; else start at 1.
        prev_iteration = _state["iteration"] if _state["model_id"] == base_model_id else 0
        new_iteration  = prev_iteration + 1

        # write delta_manifest.json
        delta_manifest = {
            "base_model_id":  base_model_id,
            "delta_model_id": model_id,
            "iteration":      new_iteration,
            "changed_shards": changed_shards,
            "timestamp":      time.time(),
        }
        (dest / DELTA_MANIFEST_FILE).write_text(
            json.dumps(delta_manifest, indent=2), encoding="utf-8"
        )
        log.info(
            "delta_manifest written  model=%s  changed=%d  iteration=%d",
            model_id, len(changed_shards), new_iteration,
        )

        # switch active model
        previous = _state["model_id"]
        _state["model_id"]  = model_id
        _state["iteration"] = new_iteration
        log.info(
            "active model  %s → %s  (delta, iteration=%d)",
            previous or "(none)", model_id, new_iteration,
        )

        # broadcast model_ready (delta)
        msg = WsMessage.model_ready(
            {},
            model_id=model_id,
            update_type="delta",
            base_model_id=base_model_id,
        )
        reached = await _mgr().broadcast(msg)
        log.info(
            "model_ready[delta] broadcast  model=%s  base=%s  clients=%d",
            model_id, base_model_id, reached,
        )

        return {
            "status":          "delta_uploaded",
            "update_type":     "delta",
            "model_id":        model_id,
            "base_model_id":   base_model_id,
            "iteration":       new_iteration,
            "changed_shards":  list(changed_shards.keys()),
            "files_written":   written,
            "clients_notified": reached,
            "dest_dir":        str(dest),
        }

    # HTTP — admin

    @app.post("/models/notify")
    async def notify_clients():
        """Re-broadcast the current active model to all WS clients."""
        m = _load_manifest()
        if m is None:
            raise HTTPException(404, "No active model")
        model_id      = _state["model_id"] or m.get("model_id", "")
        dm            = _load_delta_manifest(model_id)
        update_type   = "delta" if dm is not None else "full"
        base_model_id = dm.get("base_model_id") if dm else None

        msg = WsMessage.model_ready(
            {},
            model_id=model_id,
            update_type=update_type,
            base_model_id=base_model_id,
        )
        reached = await _mgr().broadcast(msg)
        return {
            "status":        "broadcast",
            "update_type":   update_type,
            "clients_reached": reached,
            "model_id":      model_id,
            "base_model_id": base_model_id,
        }

    @app.get("/models")
    def list_models():
        models = []
        for d in sorted(shard_dir.iterdir()):
            if not d.is_dir():
                continue
            dm = None
            if (d / DELTA_MANIFEST_FILE).exists():
                dm = json.loads((d / DELTA_MANIFEST_FILE).read_text(encoding="utf-8"))
            models.append({
                "model_id":      d.name,
                "active":        d.name == _state["model_id"],
                "shards":        [f.name for f in sorted(d.glob("*.shard"))],
                "has_manifest":  (d / "manifest.json").exists(),
                "is_delta":      dm is not None,
                "base_model_id": dm.get("base_model_id") if dm else None,
                "iteration":     dm.get("iteration") if dm else 0,
                "changed_shards": list(dm["changed_shards"].keys()) if dm else [],
            })
        return {"active_model_id": _state["model_id"], "models": models}

    @app.get("/ws/clients")
    def ws_clients():
        return {"count": _mgr().count, "client_ids": _mgr().client_ids()}

    # WebSocket

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        client_id = "unknown"
        try:
            await ws.accept()

            raw   = await ws.receive_text()
            hello = WsMessage.from_json(raw)
            if hello.type != MsgType.HELLO:
                await ws.send_text(WsMessage.error("Expected hello first").to_json())
                await ws.close(code=1002)
                return

            client_id = hello.payload.get("client_id", "unknown")
            await _mgr().connect(ws, client_id)
            log.info("hello  client=%s  caps=%s", client_id, hello.payload.get("capabilities"))

            # if a model is already active, tell the client immediately
            if _state["model_id"]:
                mid           = _state["model_id"]
                dm            = _load_delta_manifest(mid)
                update_type   = "delta" if dm is not None else "full"
                base_model_id = dm.get("base_model_id") if dm else None

                ok = await _mgr().send(
                    ws,
                    WsMessage.model_ready(
                        {},
                        model_id=mid,
                        update_type=update_type,
                        base_model_id=base_model_id,
                    ),
                )
                if ok:
                    log.info(
                        "model_ready[%s] sent on connect  client=%s  model=%s",
                        update_type, client_id, mid,
                    )
            else:
                manifest = _load_manifest()
                if manifest:
                    model_id = manifest.get("model_id", shard_dir.name)
                    ok = await _mgr().send(
                        ws,
                        WsMessage.model_ready(
                            {},
                            model_id=model_id,
                            update_type="full",
                            base_model_id=None,
                        ),
                    )
                    if ok:
                        log.info(
                            "model_ready[full] sent (auto-detect)  client=%s  model=%s",
                            client_id, model_id,
                        )

            # message loop
            async for raw_msg in ws.iter_text():
                msg = WsMessage.from_json(raw_msg)
                log.debug("← %s  from=%s", msg.type, client_id)

                if msg.type == MsgType.METRICS:
                    _log_metrics(msg.payload.get("data", {}))
                    await _mgr().send(ws, WsMessage.ack(msg.msg_id))
                    log.info(
                        "metrics stored  client=%s  model=%s",
                        client_id, msg.payload.get("model_id", ""),
                    )

                elif msg.type == MsgType.PONG:
                    rtt = (time.time() - msg.payload.get("ping_ts", 0)) * 1000
                    log.debug("pong  client=%s  rtt=%.1f ms", client_id, rtt)

                elif msg.type == MsgType.ACK:
                    log.debug("ack  ref=%s  from=%s", msg.payload.get("ref"), client_id)

                elif msg.type == MsgType.BYE:
                    log.info("bye  client=%s  reason=%s", client_id, msg.payload.get("reason", ""))
                    break

                else:
                    await _mgr().send(ws, WsMessage.error(f"Unhandled type: {msg.type!r}"))

        except WebSocketDisconnect as exc:
            log.info("WS disconnect  client=%s  code=%s", client_id, exc.code)
        except Exception as exc:
            log.exception("WS error  client=%s: %s", client_id, exc)
        finally:
            await _mgr().disconnect(ws)

    return app


# CLI

@cli.command()
def run(
    shard_dir: Path = typer.Option(
        Path("models-storage"),
        "--shard-dir",
        "-d",
        help="Directory to store model shards. Created if it doesn't exist.",
        file_okay=False,
        dir_okay=True,
    ),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    metrics_log: Path = typer.Option(Path("metrics.jsonl")),
    ping_interval: float = typer.Option(PING_INTERVAL),
    reload: bool = typer.Option(False, "--reload"),
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    shard_dir = shard_dir.resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(shard_dir, metrics_log, port=port, ping_interval=ping_interval)
    uvicorn.run(app, host=host, port=port, reload=reload, log_level="info")


def main() -> None:
    cli()