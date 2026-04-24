"""
Converts a monolithic GGUF file into a dynamic shard store.
"""

import json
import hashlib
import os
import struct
import time
from pathlib import Path
from typing import Callable, Optional

from modelpulse.server.sharder.parser import GGUFReader, GGUFTensorInfo

SHARD_MAGIC = b"SHRD"
SHARD_VERSION = 1


def _json_safe(v):
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    return str(v)


def write_shard(path: Path, tensor: GGUFTensorInfo, data: bytes) -> str:
    """
    Write a single tensor as a self-contained .shard file.
    Returns the sha256 hex digest of the WHOLE shard file.
    """
    tensor_sha256 = hashlib.sha256(data).hexdigest()

    header = {
        "name": tensor.name,
        "ggml_type": tensor.ggml_type,
        "ggml_type_name": tensor.type_name,
        "dims": tensor.dims,
        "nbytes": tensor.nbytes,
        "sha256": tensor_sha256,
    }
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    full_data = bytearray()
    full_data.extend(SHARD_MAGIC)
    full_data.extend(struct.pack("<I", SHARD_VERSION))
    full_data.extend(struct.pack("<I", len(hdr_bytes)))
    full_data.extend(hdr_bytes)
    full_data.extend(data)

    path.write_bytes(full_data)
    return hashlib.sha256(full_data).hexdigest()


def tensor_name_to_filename(name: str) -> str:
    safe = name.replace("/", "_").replace(" ", "_").replace("\\", "_")
    return safe + ".shard"


def convert(
    gguf_path: str | Path,
    output_dir: str | Path,
    *,
    on_status: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    Convert a GGUF file to a dynamic shard store.
    Returns the manifest dict (also written to manifest.json).
    """
    gguf_path = Path(gguf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    with GGUFReader(gguf_path) as reader:
        gguf = reader.parse(on_status=on_status)

        shards: dict[str, dict] = {}
        total_tensors = len(gguf.tensors)

        for i, tensor in enumerate(gguf.tensors, 1):
            filename = tensor_name_to_filename(tensor.name)
            shard_path = output_dir / filename

            if on_progress:
                on_progress(i, total_tensors, tensor.name)

            data = reader.read_tensor_data(gguf, tensor)
            sha256 = write_shard(shard_path, tensor, data)

            shards[tensor.name] = {
                "file": filename,
                "ggml_type": tensor.ggml_type,
                "ggml_type_name": tensor.type_name,
                "dims": tensor.dims,
                "nbytes": tensor.nbytes,
                "sha256": sha256,
            }

    elapsed = time.time() - t_start

    manifest = {
        "version": 1,
        "source_model": gguf_path.name,
        "gguf_version": gguf.version,
        "alignment": gguf.alignment,
        "tensor_count": len(shards),
        "total_bytes": sum(s["nbytes"] for s in shards.values()),
        "gguf_metadata_kvs": [
            {
                "key": e.key,
                "type_id": e.type_id,
                "array_elem_type": e.array_elem_type,
                "value": _json_safe(e.value),
            }
            for e in gguf.metadata_entries
        ],
        "shards": shards,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
