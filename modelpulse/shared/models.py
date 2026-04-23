"""
modelpulse.shared.models
Shared dataclasses used by both Device A and Device B.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


#  Manifest  

@dataclass
class ShardManifest:
    version: int
    source_model: str
    gguf_version: int
    alignment: int
    tensor_count: int
    total_bytes: int
    gguf_metadata_kvs: list[dict[str, Any]]
    shards: dict[str, dict[str, Any]]   # name → {file, ggml_type, dims, nbytes, sha256, …}

    @classmethod
    def from_dict(cls, d: dict) -> "ShardManifest":
        return cls(
            version=d["version"],
            source_model=d["source_model"],
            gguf_version=d["gguf_version"],
            alignment=d["alignment"],
            tensor_count=d["tensor_count"],
            total_bytes=d["total_bytes"],
            gguf_metadata_kvs=d.get("gguf_metadata_kvs", []),
            shards=d["shards"],
        )


# Metrics 

@dataclass
class InferenceMetrics:
    # Timing
    load_time_s: float = 0.0
    time_to_first_tok_s: float = 0.0
    # Throughput
    tokens_per_sec: float = 0.0
    tokens_generated: int = 0
    # Memory
    ram_delta_mb: float = 0.0
    ram_used_mb: float = 0.0
    # Hardware telemetry
    cpu_temp_c: Optional[float] = None
    cpu_percent: float = 0.0
    device_hw: str = ""
    os_info: str = ""
    # Optional quality proxy
    perplexity: Optional[float] = None
    # Session context
    timestamp: float = field(default_factory=time.time)
    prompt: str = ""
    output: str = ""
    server_url: str = ""
    source_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "load_time_s":          self.load_time_s,
            "time_to_first_tok_s":  self.time_to_first_tok_s,
            "tokens_per_sec":       self.tokens_per_sec,
            "tokens_generated":     self.tokens_generated,
            "ram_delta_mb":         self.ram_delta_mb,
            "ram_used_mb":          self.ram_used_mb,
            "cpu_temp_c":           self.cpu_temp_c,
            "cpu_percent":          self.cpu_percent,
            "device_hw":            self.device_hw,
            "os_info":              self.os_info,
            "perplexity":           self.perplexity,
            "timestamp":            self.timestamp,
            "prompt":               self.prompt,
            "output":               self.output,
            "server_url":           self.server_url,
            "source_model":         self.source_model,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InferenceMetrics":
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj