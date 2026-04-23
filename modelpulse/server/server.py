"""
Control plane : WebSocket /ws  (signals only — no binary blobs)
Data plane    : HTTP  /manifest, /shards/*, /models/upload
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
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


#   App

def create_app(
    shard_dir: Path,
    metrics_log: Path,
    *,
    port: int = 8000,
    ping_interval: float = PING_INTERVAL,
) -> FastAPI:

    # active model state — mutable box so closures can write it
    _state: dict[str, str] = {"model_id": ""}

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

    def _log_metrics(payload: dict) -> None:
        with open(metrics_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    #   lifespan (manager lives inside the running event loop)

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

    app = FastAPI(title="modelpulse / Device A", version="0.2.0", lifespan=lifespan)

    def _mgr() -> ConnectionManager:
        return app.state.manager

    #  HTTP — health / manifest / shards    

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "active_model_id": _state["model_id"],
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

    #  HTTP — metrics  

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

    #   HTTP — model upload  

    @app.post("/models/upload")
    async def upload_model(
        model_id: Annotated[str, Form()],
        manifest_file: Annotated[UploadFile, File(alias="manifest")],
        shard_files: Annotated[list[UploadFile], File(alias="shards")],
    ):
        """
        Upload a complete model (manifest.json + *.shard files).

        multipart/form-data
        ───────────────────
          model_id   str          unique slug, e.g. "qwen2.5-0.5b-q4"
          manifest   file         must be named manifest.json
          shards     file[]       one or more .shard files

        On success
        ──────────
          • Files written to  {shard_dir}/{model_id}/
          • Active model pointer switched atomically
          • model_ready broadcast to all connected WS clients
            (carries model_id only — clients fetch /manifest + /shards via HTTP)
        """
        #   validate inputs  
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

        #   write to disk  
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

        #   validate manifest  
        manifest_path = dest / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(422, f"manifest.json is not valid JSON: {exc}") from exc

        # embed model_id so GET /manifest is self-describing
        if not manifest.get("model_id"):
            manifest["model_id"] = model_id
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        #   switch active model  
        previous = _state["model_id"]
        _state["model_id"] = model_id
        log.info("active model  %s → %s", previous or "(none)", model_id)

        #   broadcast to WS clients (signal only, no binary payload)  
        msg = WsMessage.model_ready({}, model_id=model_id)
        reached = await _mgr().broadcast(msg)
        log.info("model_ready broadcast  model=%s  clients=%d", model_id, reached)

        return {
            "status": "uploaded",
            "model_id": model_id,
            "files_written": written,
            "clients_notified": reached,
            "dest_dir": str(dest),
        }

    #   HTTP — admin  

    @app.post("/models/notify")
    async def notify_clients():
        """Re-broadcast the current active model to all WS clients."""
        m = _load_manifest()
        if m is None:
            raise HTTPException(404, "No active model")
        model_id = _state["model_id"] or m.get("model_id", "")
        msg = WsMessage.model_ready({}, model_id=model_id)
        reached = await _mgr().broadcast(msg)
        return {"status": "broadcast", "clients_reached": reached, "model_id": model_id}

    @app.get("/models")
    def list_models():
        models = []
        for d in sorted(shard_dir.iterdir()):
            if not d.is_dir():
                continue
            models.append({
                "model_id": d.name,
                "active": d.name == _state["model_id"],
                "shards": [f.name for f in sorted(d.glob("*.shard"))],
                "has_manifest": (d / "manifest.json").exists(),
            })
        return {"active_model_id": _state["model_id"], "models": models}

    @app.get("/ws/clients")
    def ws_clients():
        return {"count": _mgr().count, "client_ids": _mgr().client_ids()}

    #   WebSocket    

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        client_id = "unknown"
        try:
            await ws.accept()

            # expect hello as the very first message
            raw = await ws.receive_text()
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
                ok = await _mgr().send(ws, WsMessage.model_ready({}, model_id=_state["model_id"]))
                if ok:
                    log.info("model_ready sent on connect  client=%s  model=%s", client_id, _state["model_id"])
            else:
                # No active model set; auto-detect if manifest exists in shard_dir
                manifest = _load_manifest()
                if manifest:
                    # Found manifest.json in the flat layout or currently active dir
                    model_id = manifest.get("model_id", shard_dir.name)
                    ok = await _mgr().send(ws, WsMessage.model_ready({}, model_id=model_id))
                    if ok:
                        log.info("model_ready sent (auto-detect)  client=%s  model=%s", client_id, model_id)

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


#   CLI  

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