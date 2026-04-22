"""
gguf_to_shards.py
Converts a monolithic GGUF file into a dynamic shard store.

Output layout:
  <output_dir>/
    manifest.json
    blk.0.attn_q.weight.shard
    blk.0.attn_k.weight.shard
    ...

Shard file binary format:
  [4]  magic   = b"SHRD"
  [4]  version = uint32 LE = 1
  [4]  hdr_len = uint32 LE
  [hdr_len] JSON header
  [nbytes] raw tensor data
"""

import json
import hashlib
import os
import struct
import time
from pathlib import Path

from tools.gguf_parser import GGUFReader, GGUFTensorInfo

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


def read_shard(path: Path) -> tuple[dict, bytes]:
    """
    Read a .shard file. Returns (header_dict, raw_data).
    Raises on magic mismatch or sha256 failure.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != SHARD_MAGIC:
            raise ValueError(f"Not a shard file: {path} (magic={magic!r})")
        version = struct.unpack("<I", f.read(4))[0]
        if version != SHARD_VERSION:
            raise ValueError(f"Unsupported shard version {version} in {path}")
        hdr_len = struct.unpack("<I", f.read(4))[0]
        header = json.loads(f.read(hdr_len).decode("utf-8"))
        data = f.read(header["nbytes"])

    actual = hashlib.sha256(data).hexdigest()
    if actual != header["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}!\n"
            f"  expected: {header['sha256']}\n"
            f"  actual:   {actual}"
        )
    return header, data


def tensor_name_to_filename(name: str) -> str:
    safe = name.replace("/", "_").replace(" ", "_").replace("\\", "_")
    return safe + ".shard"


def convert(
    gguf_path: str | Path,
    output_dir: str | Path,
    *,
    verbose: bool = True,
) -> dict:
    """
    Convert a GGUF file to a dynamic shard store.
    Returns the manifest dict (also written to manifest.json).
    """
    gguf_path = Path(gguf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        size_mb = gguf_path.stat().st_size / 1e6
        print(f"\n{'═'*62}")
        print(f"  Source : {gguf_path.name}  ({size_mb:.1f} MB)")
        print(f"  Output : {output_dir}")
        print(f"{'═'*62}")

    t_start = time.time()

    with GGUFReader(gguf_path) as reader:
        gguf = reader.parse()

        shards: dict[str, dict] = {}
        total_tensors = len(gguf.tensors)

        for i, tensor in enumerate(gguf.tensors, 1):
            filename = tensor_name_to_filename(tensor.name)
            shard_path = output_dir / filename

            if verbose:
                print(
                    f"  [{i:>3}/{total_tensors}]  {tensor.name:<42} "
                    f"{tensor.type_name:<8} "
                    f"{tensor.nbytes/1e6:>7.2f} MB",
                    end="  ", flush=True
                )

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

            if verbose:
                print("✓")

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

    total_mb = sum(s["nbytes"] for s in shards.values()) / 1e6
    shard_files_mb = sum((output_dir / s["file"]).stat().st_size for s in shards.values()) / 1e6

    if verbose:
        print(f"\n{'─'*62}")
        print(f"  Tensors written : {len(shards)}")
        print(f"  Tensor data     : {total_mb:.1f} MB")
        print(f"  Shard files     : {shard_files_mb:.1f} MB  (includes per-file headers)")
        print(f"  Manifest        : {manifest_path.name}")
        print(f"  Time elapsed    : {elapsed:.1f}s")
        print(f"{'═'*62}\n")

    return manifest


def _write_gguf_value(buf: bytearray, type_id: int, value, array_elem_type: int | None = None):
    def _append_str(s: str):
        b = s.encode("utf-8")
        buf.extend(struct.pack("<Q", len(b)))
        buf.extend(b)

    scalar_formats = {
        0: "B",
        1: "b",
        2: "H",
        3: "h",
        4: "I",
        5: "i",
        6: "f",
        7: "?",
        10: "Q",
        11: "q",
        12: "d",
    }

    if type_id == 8:
        _append_str(value)
        return

    if type_id == 9:
        if array_elem_type is None:
            raise ValueError("ARRAY metadata entry missing array_elem_type")
        if not isinstance(value, list):
            raise ValueError("ARRAY metadata value must be a list")

        buf.extend(struct.pack("<I", array_elem_type))
        buf.extend(struct.pack("<Q", len(value)))
        for item in value:
            _write_gguf_value(buf, array_elem_type, item)
        return

    fmt = scalar_formats.get(type_id)
    if fmt is None:
        raise ValueError(f"Unsupported metadata type_id: {type_id}")

    buf.extend(struct.pack("<" + fmt, value))


def assemble(
    shard_dir: str | Path,
    output_path: str | Path,
    *,
    verbose: bool = True,
) -> None:
    """
    Reassemble a shard store back into a valid GGUF file.
    Writes atomically: temp file -> verify -> os.replace().
    """
    shard_dir = Path(shard_dir)
    output_path = Path(output_path)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with open(shard_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    metadata_kvs = manifest.get("gguf_metadata_kvs", [])
    arch = next((kv["value"] for kv in metadata_kvs if kv["key"] == "general.architecture"), None)
    if not arch:
        raise ValueError(
            "manifest is missing general.architecture; "
            "re-run convert from the original GGUF so typed metadata is preserved."
        )

    tensor_data: dict[str, tuple[dict, bytes]] = {}
    for name, entry in manifest["shards"].items():
        path = shard_dir / entry["file"]
        if verbose:
            print(f"  reading {entry['file']:<50}", end="  ", flush=True)
        header, data = read_shard(path)
        tensor_data[name] = (header, data)
        if verbose:
            print("✓ verified")

    alignment = manifest["alignment"]
    ordered_names = list(manifest["shards"].keys())
    gguf_version = manifest["gguf_version"]

    # Recompute absolute offsets fresh
    running_offset = 0
    offsets: dict[str, int] = {}
    for name in ordered_names:
        pad = (alignment - (running_offset % alignment)) % alignment if running_offset % alignment else 0
        running_offset += pad
        offsets[name] = running_offset
        running_offset += manifest["shards"][name]["nbytes"]

    if verbose:
        print(f"\n  assembling → {output_path.name}")

    try:
        with open(temp_path, "wb") as out:
            buf = bytearray()

            def _append_str(s: str):
                b = s.encode("utf-8")
                buf.extend(struct.pack("<Q", len(b)))
                buf.extend(b)

            # magic + version
            buf.extend(struct.pack("<I", 0x46554747))  # GGUF
            buf.extend(struct.pack("<I", gguf_version))

            tensor_count = len(ordered_names)
            kv_count = len(metadata_kvs)

            if gguf_version == 1:
                buf.extend(struct.pack("<I", tensor_count))
                buf.extend(struct.pack("<I", kv_count))
            else:
                buf.extend(struct.pack("<Q", tensor_count))
                buf.extend(struct.pack("<Q", kv_count))

            # typed metadata KVs
            for kv in metadata_kvs:
                _append_str(kv["key"])
                buf.extend(struct.pack("<I", kv["type_id"]))
                _write_gguf_value(
                    buf,
                    kv["type_id"],
                    kv["value"],
                    kv.get("array_elem_type"),
                )

            # tensor info array
            for name in ordered_names:
                entry = manifest["shards"][name]
                _append_str(name)
                dims = entry["dims"]
                buf.extend(struct.pack("<I", len(dims)))
                for d in dims:
                    buf.extend(struct.pack("<Q", d))
                buf.extend(struct.pack("<I", entry["ggml_type"]))
                buf.extend(struct.pack("<Q", offsets[name]))

            # align to data section
            pad = (alignment - (len(buf) % alignment)) % alignment
            buf.extend(b"\x00" * pad)

            out.write(buf)

            # write tensor data in order, with alignment padding between tensors
            written = 0
            for name in ordered_names:
                _, data = tensor_data[name]
                pad = (alignment - (written % alignment)) % alignment if written % alignment else 0
                if pad:
                    out.write(b"\x00" * pad)
                    written += pad
                out.write(data)
                written += len(data)

        os.replace(temp_path, output_path)

        if verbose:
            size_mb = output_path.stat().st_size / 1e6
            print(f"  wrote {size_mb:.1f} MB  →  {output_path}")

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


if __name__ == "__main__":
    import sys

    usage = (
        "Usage:\n"
        "  convert:   python gguf_to_shards.py convert <model.gguf> <output_dir/>\n"
        "  assemble:  python gguf_to_shards.py assemble <shard_dir/> <output.gguf>\n"
    )

    if len(sys.argv) < 4:
        print(usage)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "convert":
        convert(sys.argv[2], sys.argv[3])
    elif cmd == "assemble":
        assemble(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown command: {cmd!r}\n{usage}")
        sys.exit(1)