"""
modelpulse.shared.ws_protocol
Shared WebSocket message types and (de)serialisation helpers.

Changes in v0.3.0
  WsMessage.model_ready() now accepts two extra keyword arguments:
    update_type   : "full" | "delta"  (default "full")
    base_model_id : str | None        (None for full updates)

  These fields travel inside the existing ``payload`` dict so the wire
  format remains backwards-compatible — old clients that do not read
  them simply ignore the extra keys.
"""
from __future__ import annotations

import json
import time
import uuid
from enum import Enum
from typing import Any, Optional


class MsgType(str, Enum):
    HELLO       = "hello"
    MODEL_READY = "model_ready"
    METRICS     = "metrics"
    ACK         = "ack"
    PING        = "ping"
    PONG        = "pong"
    BYE         = "bye"
    ERROR       = "error"


class WsMessage:
    """
    Thin wrapper around a JSON WebSocket frame.

    Wire format
    {
      "type":    "<MsgType>",
      "msg_id":  "<uuid4>",
      "ts":      <unix float>,
      "payload": { ... }
    }
    """

    def __init__(
        self,
        type: MsgType,
        payload: dict[str, Any],
        msg_id: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> None:
        self.type    = type
        self.payload = payload
        self.msg_id  = msg_id or str(uuid.uuid4())
        self.ts      = ts    or time.time()

    # serialisation

    def to_json(self) -> str:
        return json.dumps(
            {
                "type":    self.type.value,
                "msg_id":  self.msg_id,
                "ts":      self.ts,
                "payload": self.payload,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "WsMessage":
        d = json.loads(raw)
        return cls(
            type    = MsgType(d["type"]),
            payload = d.get("payload", {}),
            msg_id  = d.get("msg_id"),
            ts      = d.get("ts"),
        )

    # factory helpers

    @classmethod
    def hello(cls, client_id: str, *, capabilities: Optional[dict] = None) -> "WsMessage":
        return cls(MsgType.HELLO, {"client_id": client_id, "capabilities": capabilities or {}})

    @classmethod
    def model_ready(
        cls,
        extra: dict,
        *,
        model_id: str = "",
        update_type: str = "full",
        base_model_id: Optional[str] = None,
    ) -> "WsMessage":
        """
        Construct a MODEL_READY signal.

        Parameters
        extra          Additional keys merged into the payload (pass {} if none).
        model_id       The model being announced.
        update_type    "full"  → client must fetch all shards from scratch.
                       "delta" → client fetches only the changed shards and
                                 merges them into its cached shard dict.
        base_model_id  For delta updates: the model_id of the full model that
                       this delta patches.  None for full updates.
        """
        payload: dict[str, Any] = {
            "model_id":      model_id,
            "update_type":   update_type,
            "base_model_id": base_model_id,
            **extra,
        }
        return cls(MsgType.MODEL_READY, payload)

    @classmethod
    def metrics(cls, data: dict, *, model_id: str = "") -> "WsMessage":
        return cls(MsgType.METRICS, {"model_id": model_id, "data": data})

    @classmethod
    def ack(cls, ref_msg_id: str) -> "WsMessage":
        return cls(MsgType.ACK, {"ref": ref_msg_id})

    @classmethod
    def ping(cls) -> "WsMessage":
        return cls(MsgType.PING, {"ts": time.time()})

    @classmethod
    def pong(cls, ping_ts: float) -> "WsMessage":
        return cls(MsgType.PONG, {"ping_ts": ping_ts})

    @classmethod
    def bye(cls, reason: str = "") -> "WsMessage":
        return cls(MsgType.BYE, {"reason": reason})

    @classmethod
    def error(cls, detail: str, *, ref_msg_id: Optional[str] = None) -> "WsMessage":
        payload: dict[str, Any] = {"detail": detail}
        if ref_msg_id:
            payload["ref"] = ref_msg_id
        return cls(MsgType.ERROR, payload)

    def __repr__(self) -> str:
        return f"WsMessage(type={self.type!r}, msg_id={self.msg_id!r})"