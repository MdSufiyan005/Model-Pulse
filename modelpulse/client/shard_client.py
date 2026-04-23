"""
modelpulse.client.shard_client
────────────────────────────────
Two distinct layers:

ShardClient
    Thin async HTTP client — unchanged from v1.
    Fetches manifest, streams individual shards, POSTs metrics over HTTP.
    Used internally by ShardWebSocketSession for the data-plane transfers.

ShardWebSocketSession
    Persistent WebSocket session (control-plane).
    • Connects to ws[s]://<host>/ws
    • Sends    hello       on connect
    • Receives model_ready → calls your on_model_ready coroutine
                              (which should fetch + run inference via ShardClient)
    • Sends    metrics     back after inference
    • Handles  ping/pong   keepalive
    • Auto-reconnects with exponential back-off on any disconnect

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

# ── ShardClient (HTTP)  ───

class ShardClient:
    """
    Async HTTP client for the Device A shard server.

    Usage (async context manager):
        async with ShardClient("http://192.168.1.10:8000") as c:
            manifest = await c.fetch_manifest()
            data     = await c.fetch_shard("blk.0.attn_q.weight.shard")
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
        GET /shards/{filename} — stream-downloads a shard file.

        Parameters
        ----------
        filename         e.g. "blk.0.attn_q.weight.shard"
        expected_sha256  if given, verifies the raw tensor payload (after
                         stripping the SHRD container header, if present).
        on_progress      callback(bytes_received, total_bytes)

        Notes
        -----
        SHA-256 is computed over the raw tensor *payload*, not the full SHRD
        container, matching the digest stored in the manifest.
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
            # Unwrap SHRD container before hashing so the digest matches the
            # manifest (which is computed over raw tensor data only).
            if len(data) >= 12 and data[:4] == b"SHRD":
                hdr_len = struct.unpack("<I", data[8:12])[0]
                payload = data[12 + hdr_len:]
            else:
                payload = data
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected_sha256:
                raise ValueError(
                    f"SHA-256 mismatch for {filename!r}\n"
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


#  ShardWebSocketSession (control-plane)

_BACKOFF_BASE   = 2.0    # seconds
_BACKOFF_MAX    = 60.0   # seconds
_BACKOFF_FACTOR = 1.5


class ShardWebSocketSession:
    """
    Persistent WebSocket session with Device A.

    The caller provides an ``on_model_ready`` coroutine that receives a
    ShardManifest and returns an InferenceMetrics object.  The session
    calls it whenever the server announces a (new) model is available,
    then sends the resulting metrics back over the WebSocket.

    Parameters
    ----------
    base_url        HTTP base URL of Device A, e.g. "http://100.64.0.1:8000".
                    The WS URL is derived automatically (http→ws, https→wss).
    on_model_ready  async (ShardManifest) → InferenceMetrics
                    Your inference callback.  Fetch shards, run the model,
                    return metrics.  Exceptions are caught and logged; the
                    session continues.
    client_id       Unique identifier for this device.  Defaults to a UUID.
    ping_timeout    Seconds to wait for a pong before declaring the connection
                    stale (passed through to the websockets library).
    reconnect_delay Base delay (seconds) before the first reconnect attempt.
                    Subsequent attempts use exponential back-off up to 60 s.
    """

    def __init__(
        self,
        base_url: str,
        on_model_ready: Callable[[ShardManifest], Awaitable[InferenceMetrics]],
        *,
        client_id: Optional[str] = None,
        ping_timeout: float = 30.0,
        reconnect_delay: float = _BACKOFF_BASE,
    ) -> None:
        self.base_url  = base_url.rstrip("/")
        self.ws_url    = (
            self.base_url
            .replace("http://", "ws://")
            .replace("https://", "wss://")
            + "/ws"
        )
        self.client_id       = client_id or str(uuid.uuid4())
        self.on_model_ready  = on_model_ready
        self.ping_timeout    = ping_timeout
        self._reconnect_delay = reconnect_delay

        self._stop     = asyncio.Event()
        self._ws       = None   # current websockets connection
        self._model_id = ""     # last model_id received from server

    # ── public API  ───────

    async def run(self) -> None:
        """
        Connect and loop forever, reconnecting on any failure.
        Returns only after stop() is called.
        """
        delay = self._reconnect_delay
        while not self._stop.is_set():
            try:
                await self._connect_and_loop()
                delay = self._reconnect_delay   # reset back-off on clean exit
            except (
                OSError,
                websockets.exceptions.WebSocketException,
                asyncio.TimeoutError,
            ) as exc:
                if self._stop.is_set():
                    break
                log.warning(
                    "WS connection lost (%s). Reconnecting in %.1fs…", exc, delay
                )
                await asyncio.sleep(delay)
                delay = min(delay * _BACKOFF_FACTOR, _BACKOFF_MAX)
            except Exception as exc:
                if self._stop.is_set():
                    break
                log.exception(
                    "Unexpected WS error: %s. Reconnecting in %.1fs…", exc, delay
                )
                await asyncio.sleep(delay)
                delay = min(delay * _BACKOFF_FACTOR, _BACKOFF_MAX)

    async def stop(self, reason: str = "client shutdown") -> None:
        """Signal the session to stop and close the current connection."""
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
                    self.client_id, capabilities={"version": "0.2.0"}
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
        # Acknowledge immediately so the server does not time out waiting.
        await ws.send(WsMessage.ack(msg.msg_id).to_json())

        model_id = msg.payload.get("model_id", "")

        if model_id and model_id == self._model_id:
            log.info("model_ready: already loaded %s — skipping", model_id)
            return

        log.info("model_ready  model_id=%s  fetching manifest over HTTP…", model_id)

        #  Fetch manifest via HTTP (not from the WS payload)
        try:
            async with ShardClient(self.base_url, timeout=30.0) as http:
                manifest = await http.fetch_manifest()
        except Exception as exc:
            log.error("manifest fetch failed: %s", exc)
            await ws.send(
                WsMessage.error(
                    f"manifest fetch error: {exc}", ref_msg_id=msg.msg_id
                ).to_json()
            )
            return

        # Call the user-supplied inference coroutine
        # elapsed_ms covers shard download + assembly + load + inference.
        # It is a coarse operational diagnostic, NOT an inference latency figure.
        # Fine-grained timings (TTFT, tok/s, load_time_s) live inside the
        # InferenceMetrics object returned by on_model_ready().
        t0 = time.perf_counter()
        try:
            metrics = await self.on_model_ready(manifest)
        except Exception as exc:
            log.exception("on_model_ready raised: %s", exc)
            await ws.send(
                WsMessage.error(
                    f"inference error: {exc}", ref_msg_id=msg.msg_id
                ).to_json()
            )
            return

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "on_model_ready complete  total_elapsed=%.0f ms  "
            "load=%.2f s  ttft=%.3f s  tok/s=%.1f  model=%s",
            elapsed_ms,
            metrics.load_time_s,
            metrics.time_to_first_tok_s,
            metrics.tokens_per_sec,
            model_id,
        )
        self._model_id = model_id

        # Send metrics back to Device A
        try:
            metrics_msg = WsMessage.metrics(metrics.to_dict(), model_id=model_id)
            await ws.send(metrics_msg.to_json())
            log.info("metrics sent  msg_id=%s", metrics_msg.msg_id)
        except Exception as exc:
            log.error("metrics send failed: %s", exc)