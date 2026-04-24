"""
Two distinct layers:

ShardClient
    Thin async HTTP client.
    Fetches manifest, streams individual shards, POSTs metrics over HTTP.
    New in v0.3.0:
      • fetch_delta_manifest(model_id) → dict
      • fetch_delta_shards(model_id, delta_manifest) → dict[str, bytes]

ShardWebSocketSession
    Persistent WebSocket session (control-plane).
    Handles full and delta model_ready signals.
    Auto-reconnects with exponential back-off.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import struct
import time
import uuid
from typing import Awaitable, Callable, Optional

import httpx
import websockets
import websockets.exceptions

from modelpulse.shared.models import InferenceMetrics, ShardManifest
from modelpulse.shared.ws_protocol import MsgType, WsMessage

log = logging.getLogger("modelpulse.client")

# ShardClient (HTTP)

class ShardClient:
    """
    Async HTTP client for the Device A shard server.

    Usage (async context manager):
        async with ShardClient("http://192.168.1.10:8000") as c:
            manifest = await c.fetch_manifest()
            data     = await c.fetch_shard("blk.0.attn_q.weight.shard")

            # Delta workflow:
            dm       = await c.fetch_delta_manifest("qwen2.5-0.5b-q4-d1")
            patches  = await c.fetch_delta_shards("qwen2.5-0.5b-q4-d1", dm)
    """

    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url  = base_url.rstrip("/")
        self._timeout  = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                follow_redirects=True,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "ShardClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()

    # API calls

    async def ping(self) -> float:
        """GET /health — returns round-trip time in ms."""
        client = await self._get_client()
        t0     = time.perf_counter()
        resp   = await client.get("/health")
        resp.raise_for_status()
        return (time.perf_counter() - t0) * 1000.0

    async def fetch_manifest(self) -> ShardManifest:
        """GET /manifest — returns a parsed ShardManifest."""
        client = await self._get_client()
        resp   = await client.get("/manifest")
        resp.raise_for_status()
        return ShardManifest.from_dict(resp.json())

    async def fetch_shard(
        self,
        filename: str,
        expected_sha256: Optional[str] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> bytes:
        """
        GET /shards/{filename} — stream-download a shard file.

        SHA-256 verification (when expected_sha256 is provided) is performed
        over the raw tensor payload, not the SHRD container header.
        """
        client   = await self._get_client()
        chunks: list[bytes] = []
        received = 0

        async with client.stream("GET", f"/shards/{filename}") as resp:
            resp.raise_for_status()
            content_length = int(resp.headers.get("content-length", 0))
            async for chunk in resp.aiter_bytes(chunk_size=65536):
                chunks.append(chunk)
                received += len(chunk)
                if on_progress:
                    on_progress(received, content_length)

        data = b"".join(chunks)

        if expected_sha256:
            data = self._verify_shard(data, expected_sha256, filename)

        return data

    # Delta API calls

    async def fetch_delta_manifest(self, model_id: str) -> dict:
        """
        GET /shards/delta/{model_id}/manifest

        Returns the raw delta manifest dict:
        {
          "base_model_id":  "...",
          "delta_model_id": "...",
          "iteration":      N,
          "changed_shards": {
            "<shard_name>": {"file": "...", "sha256": "...", "nbytes": N},
            ...
          }
        }

        Raises httpx.HTTPStatusError if no delta manifest exists for the
        model (i.e. it was a full upload, not a delta).
        """
        client = await self._get_client()
        resp   = await client.get(f"/shards/delta/{model_id}/manifest")
        resp.raise_for_status()
        return resp.json()

    async def fetch_delta_shards(
        self,
        model_id: str,
        delta_manifest: dict,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> dict[str, bytes]:
        """
        Download only the changed shards listed in *delta_manifest*.

        Parameters
        model_id        The delta model_id (used to build the URL path).
        delta_manifest  The dict returned by fetch_delta_manifest().
        on_progress     Optional callback(shard_name, bytes_received, total_bytes).

        Returns
        A dict mapping shard_name → raw bytes (same format as full shards).
        Callers merge this into their existing shard cache:
            shard_cache[base_model_id].update(patches)

        SHA-256 is verified against the digest stored in delta_manifest for
        every shard.  Raises ValueError on mismatch.
        """
        changed: dict[str, dict] = delta_manifest.get("changed_shards", {})
        patches: dict[str, bytes] = {}

        for shard_name, entry in changed.items():
            filename       = entry["file"]
            expected_sha   = entry.get("sha256")

            log.info(
                "delta fetch  model=%s  shard=%s  expected_bytes=%s",
                model_id, shard_name, entry.get("nbytes", "?"),
            )

            client   = await self._get_client()
            chunks: list[bytes] = []
            received = 0

            async with client.stream(
                "GET", f"/shards/delta/{model_id}/{filename}"
            ) as resp:
                resp.raise_for_status()
                content_length = int(resp.headers.get("content-length", 0))
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    chunks.append(chunk)
                    received += len(chunk)
                    if on_progress:
                        on_progress(shard_name, received, content_length)

            raw = b"".join(chunks)

            if expected_sha:
                raw = self._verify_shard(raw, expected_sha, shard_name)

            patches[shard_name] = raw
            log.info("delta fetched  shard=%s  size=%d B", shard_name, len(raw))

        return patches

    # shared helper

    @staticmethod
    def _verify_shard(data: bytes, expected_sha256: str, label: str) -> bytes:
        """
        Verify SHA-256 over the tensor payload (strips SHRD header if present).
        Returns *data* unchanged on success; raises ValueError on mismatch.
        """
        if len(data) >= 12 and data[:4] == b"SHRD":
            hdr_len = struct.unpack("<I", data[8:12])[0]
            payload = data[12 + hdr_len:]
        else:
            payload = data
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {label!r}\n"
                f"  expected : {expected_sha256[:16]}…\n"
                f"  actual   : {actual[:16]}…"
            )
        return data

    async def post_metrics(self, metrics: InferenceMetrics) -> dict:
        """POST /metrics — send InferenceMetrics to Device A (HTTP fallback)."""
        client = await self._get_client()
        resp   = await client.post("/metrics", json=metrics.to_dict(), timeout=10.0)
        resp.raise_for_status()
        return resp.json()


# ShardWebSocketSession (control-plane)

_BACKOFF_BASE   = 2.0
_BACKOFF_MAX    = 60.0
_BACKOFF_FACTOR = 1.5


class ShardWebSocketSession:
    """
    Persistent WebSocket session with Device A.

    The caller provides two coroutines:

    on_model_ready(manifest)           → InferenceMetrics
        Full model: fetch all shards, load, infer, return metrics.

    on_model_delta(manifest, patches)  → InferenceMetrics
        Delta update: patches is a dict[shard_name, bytes] of the changed
        shards only.  The caller merges them, reassembles, and infers.
        Optional — if not provided, delta signals fall back to on_model_ready
        with a freshly fetched full manifest (safe but wasteful).

    Parameters
    base_url        HTTP base URL of Device A.
    on_model_ready  async (ShardManifest) → InferenceMetrics
    on_model_delta  Optional async (ShardManifest, dict[str, bytes]) → InferenceMetrics
    client_id       Unique identifier for this device.
    ping_timeout    Seconds before a stale connection is declared.
    reconnect_delay Base reconnect back-off (seconds).
    """

    def __init__(
        self,
        base_url: str,
        on_model_ready: Callable[[ShardManifest], Awaitable[InferenceMetrics]],
        *,
        on_model_delta: Optional[
            Callable[[ShardManifest, dict[str, bytes]], Awaitable[InferenceMetrics]]
        ] = None,
        client_id: Optional[str] = None,
        ping_timeout: float = 30.0,
        reconnect_delay: float = _BACKOFF_BASE,
    ) -> None:
        self.base_url       = base_url.rstrip("/")
        self.ws_url         = (
            self.base_url
            .replace("http://", "ws://")
            .replace("https://", "wss://")
            + "/ws"
        )
        self.client_id      = client_id or str(uuid.uuid4())
        self.on_model_ready = on_model_ready
        self.on_model_delta = on_model_delta
        self.ping_timeout   = ping_timeout
        self._reconnect_delay = reconnect_delay

        self._stop     = asyncio.Event()
        self._ws       = None
        self._model_id = ""

    # public API

    async def run(self) -> None:
        delay = self._reconnect_delay
        while not self._stop.is_set():
            try:
                await self._connect_and_loop()
                delay = self._reconnect_delay
            except (
                OSError,
                websockets.exceptions.WebSocketException,
                asyncio.TimeoutError,
            ) as exc:
                if self._stop.is_set():
                    break
                log.warning("WS connection lost (%s). Reconnecting in %.1fs…", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * _BACKOFF_FACTOR, _BACKOFF_MAX)
            except Exception as exc:
                if self._stop.is_set():
                    break
                log.exception("Unexpected WS error: %s. Reconnecting in %.1fs…", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * _BACKOFF_FACTOR, _BACKOFF_MAX)

    async def stop(self, reason: str = "client shutdown") -> None:
        self._stop.set()
        if self._ws is not None:
            try:
                await self._ws.send(WsMessage.bye(reason).to_json())
                await self._ws.close()
            except Exception:
                pass

    # internals

    async def _connect_and_loop(self) -> None:
        log.info("Connecting to %s  client_id=%s", self.ws_url, self.client_id)

        async with websockets.connect(
            self.ws_url,
            ping_interval=None,
            ping_timeout=self.ping_timeout,
            close_timeout=5,
            max_size=2 * 1024 * 1024,
        ) as ws:
            self._ws = ws
            log.info("WS connected")

            await ws.send(
                WsMessage.hello(
                    self.client_id, capabilities={"version": "0.3.0"}
                ).to_json()
            )

            async for raw in ws:
                if self._stop.is_set():
                    break
                try:
                    msg = WsMessage.from_json(raw)
                except Exception as exc:
                    log.warning("Malformed message: %s", exc)
                    continue
                log.debug("→ %s", msg.type)
                await self._dispatch(ws, msg)

        self._ws = None

    async def _dispatch(self, ws, msg: WsMessage) -> None:
        if msg.type == MsgType.MODEL_READY:
            await self._handle_model_ready(ws, msg)
        elif msg.type == MsgType.PING:
            try:
                await ws.send(WsMessage.pong(msg.payload.get("ts", 0.0)).to_json())
            except Exception:
                pass
        elif msg.type == MsgType.ACK:
            pass
        elif msg.type == MsgType.ERROR:
            log.error("Server error: %s", msg.payload.get("detail", "?"))
        elif msg.type == MsgType.BYE:
            log.info("Server sent BYE — closing session")

    async def _handle_model_ready(self, ws, msg: WsMessage) -> None:
        await ws.send(WsMessage.ack(msg.msg_id).to_json())

        model_id      = msg.payload.get("model_id", "")
        update_type   = msg.payload.get("update_type", "full")
        base_model_id = msg.payload.get("base_model_id")

        if model_id and model_id == self._model_id:
            log.info("model_ready: already loaded %s — skipping", model_id)
            return

        if update_type == "delta" and base_model_id and self.on_model_delta:
            await self._handle_delta(ws, msg, model_id, base_model_id)
        else:
            await self._handle_full(ws, msg, model_id)

    async def _handle_full(self, ws, msg: WsMessage, model_id: str) -> None:
        log.info("model_ready[full]  model_id=%s", model_id)

        try:
            async with ShardClient(self.base_url, timeout=30.0) as http:
                manifest = await http.fetch_manifest()
        except Exception as exc:
            log.error("manifest fetch failed: %s", exc)
            await ws.send(WsMessage.error(f"manifest fetch error: {exc}", ref_msg_id=msg.msg_id).to_json())
            return

        t0 = time.perf_counter()
        try:
            metrics = await self.on_model_ready(manifest)
        except Exception as exc:
            log.exception("on_model_ready raised: %s", exc)
            await ws.send(WsMessage.error(f"inference error: {exc}", ref_msg_id=msg.msg_id).to_json())
            return

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "on_model_ready[full] complete  total_elapsed=%.0f ms  "
            "load=%.2f s  ttft=%.3f s  tok/s=%.1f  model=%s",
            elapsed_ms, metrics.load_time_s, metrics.time_to_first_tok_s,
            metrics.tokens_per_sec, model_id,
        )
        self._model_id = model_id
        await self._send_metrics(ws, metrics, model_id)

    async def _handle_delta(
        self,
        ws,
        msg: WsMessage,
        model_id: str,
        base_model_id: str,
    ) -> None:
        log.info(
            "model_ready[delta]  model_id=%s  base=%s",
            model_id, base_model_id,
        )

        try:
            async with ShardClient(self.base_url, timeout=30.0) as http:
                manifest      = await http.fetch_manifest()
                delta_manifest = await http.fetch_delta_manifest(model_id)
                patches       = await http.fetch_delta_shards(model_id, delta_manifest)
        except Exception as exc:
            log.error("delta fetch failed: %s", exc)
            await ws.send(
                WsMessage.error(f"delta fetch error: {exc}", ref_msg_id=msg.msg_id).to_json()
            )
            return

        t0 = time.perf_counter()
        try:
            metrics = await self.on_model_delta(manifest, patches)
        except Exception as exc:
            log.exception("on_model_delta raised: %s", exc)
            await ws.send(
                WsMessage.error(f"delta inference error: {exc}", ref_msg_id=msg.msg_id).to_json()
            )
            return

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "on_model_delta complete  total_elapsed=%.0f ms  "
            "load=%.2f s  ttft=%.3f s  tok/s=%.1f  model=%s  changed_shards=%d",
            elapsed_ms, metrics.load_time_s, metrics.time_to_first_tok_s,
            metrics.tokens_per_sec, model_id, len(patches),
        )
        self._model_id = model_id
        await self._send_metrics(ws, metrics, model_id)

    async def _send_metrics(self, ws, metrics: InferenceMetrics, model_id: str) -> None:
        try:
            metrics_msg = WsMessage.metrics(metrics.to_dict(), model_id=model_id)
            await ws.send(metrics_msg.to_json())
            log.info("metrics sent  msg_id=%s", metrics_msg.msg_id)
        except Exception as exc:
            log.error("metrics send failed: %s", exc)