"""
modelpulse.shared.ws_protocol
──────────────────────────────
Shared WebSocket message envelope and typed factories used by both
Device A (server) and Device B (client).

All messages are JSON-encoded dicts with this top-level shape:

    {
        "type":    "<MsgType>",
        "msg_id":  "<8-char hex>",     # for ack correlation
        "ts":      1713000000.123,     # unix epoch, sender wall-clock
        "payload": { ... }             # type-specific body
    }

Direction legend
    S→C   server (Device A) → client (Device B)
    C→S   client (Device B) → server (Device A)
    ↔     bidirectional
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Message types ─────────────────────────────────────────────────────────────

class MsgType(str, Enum):
    # S→C
    MODEL_READY   = "model_ready"   # server has a (new) sharded model ready
    PING          = "ping"          # server-initiated keepalive
    ERROR         = "error"         # server-side problem notification

    # C→S
    HELLO         = "hello"         # first message after WS handshake
    METRICS       = "metrics"       # inference results
    PONG          = "pong"          # response to PING

    # ↔
    ACK           = "ack"           # acknowledge a message by msg_id
    BYE           = "bye"           # graceful close announcement


# ── Envelope ──────────────────────────────────────────────────────────────────

@dataclass
class WsMessage:
    """Universal WebSocket envelope.  Use the class-method factories below."""

    type:    str
    payload: dict[str, Any]
    msg_id:  str   = field(default_factory=lambda: uuid.uuid4().hex[:8])
    ts:      float = field(default_factory=time.time)

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(
            {
                "type":    self.type,
                "msg_id":  self.msg_id,
                "ts":      self.ts,
                "payload": self.payload,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> "WsMessage":
        d: dict = json.loads(raw)
        return cls(
            type    = d["type"],
            payload = d.get("payload", {}),
            msg_id  = d.get("msg_id", uuid.uuid4().hex[:8]),
            ts      = d.get("ts", time.time()),
        )

    # ── S→C factories ─────────────────────────────────────────────────────────

    @classmethod
    def model_ready(cls, manifest: dict, *, model_id: str = "") -> "WsMessage":
        """Server announces a sharded model is available for download."""
        return cls(
            type    = MsgType.MODEL_READY,
            payload = {"manifest": manifest, "model_id": model_id or manifest.get("model_id", "")},
        )

    @classmethod
    def ping(cls) -> "WsMessage":
        return cls(type=MsgType.PING, payload={"ts": time.time()})

    @classmethod
    def error(cls, detail: str, *, ref_msg_id: str = "") -> "WsMessage":
        return cls(type=MsgType.ERROR, payload={"detail": detail, "ref": ref_msg_id})

    # ── C→S factories ─────────────────────────────────────────────────────────

    @classmethod
    def hello(cls, client_id: str, *, capabilities: dict | None = None) -> "WsMessage":
        """First message the client sends after the WS handshake is complete."""
        return cls(
            type    = MsgType.HELLO,
            payload = {"client_id": client_id, "capabilities": capabilities or {}},
        )

    @classmethod
    def metrics(cls, data: dict, *, model_id: str = "") -> "WsMessage":
        """Inference metrics pushed from client after a completed run."""
        return cls(type=MsgType.METRICS, payload={"data": data, "model_id": model_id})

    @classmethod
    def pong(cls, ping_ts: float) -> "WsMessage":
        return cls(type=MsgType.PONG, payload={"ping_ts": ping_ts, "ts": time.time()})

    # ── bidirectional factories ────────────────────────────────────────────────

    @classmethod
    def ack(cls, ref_msg_id: str) -> "WsMessage":
        return cls(type=MsgType.ACK, payload={"ref": ref_msg_id})

    @classmethod
    def bye(cls, reason: str = "") -> "WsMessage":
        return cls(type=MsgType.BYE, payload={"reason": reason})

    # ── helpers ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return f"WsMessage(type={self.type!r}, msg_id={self.msg_id!r})"