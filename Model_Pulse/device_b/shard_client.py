"""
modelpulse.device_b.shard_client
Async HTTP client — pulls manifest and individual shards from Device A,
and POSTs metrics back.
"""
from __future__ import annotations

import hashlib
import time
from typing import Callable, Optional

import httpx

from modelpulse.shared.models import InferenceMetrics, ShardManifest


class ShardClient:
    """
    Async client for the Device A shard server.

    Usage (async context manager):
        async with ShardClient("http://192.168.1.10:8000") as c:
            manifest = await c.fetch_manifest()
            data = await c.fetch_shard("blk.0.attn_q.weight.shard")
    """

    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                follow_redirects=True,
            )
        return self._client

    async def aclose(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "ShardClient":
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    # ── API calls ─────────────────────────────────────────────────────────────

    async def ping(self) -> float:
        """
        GET /health — returns round-trip time in ms.
        Raises httpx.HTTPError on failure.
        """
        client = await self._get_client()
        t0 = time.perf_counter()
        resp = await client.get("/health")
        resp.raise_for_status()
        return (time.perf_counter() - t0) * 1000.0

    async def fetch_manifest(self) -> ShardManifest:
        """GET /manifest — returns a parsed ShardManifest."""
        client = await self._get_client()
        resp = await client.get("/manifest")
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

        Args:
            filename:        e.g. "blk.0.attn_q.weight.shard"
            expected_sha256: if given, verifies integrity after download
            on_progress:     callback(bytes_received, total_bytes)
        """
        client = await self._get_client()
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
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected_sha256:
                raise ValueError(
                    f"SHA-256 mismatch for {filename!r}\n"
                    f"  expected : {expected_sha256[:16]}…\n"
                    f"  actual   : {actual[:16]}…"
                )

        return data

    async def post_metrics(self, metrics: InferenceMetrics) -> dict:
        """POST /metrics — send InferenceMetrics to Device A."""
        client = await self._get_client()
        resp = await client.post(
            "/metrics",
            json=metrics.to_dict(),
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()