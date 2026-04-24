"""
In-memory GGUF assembly + llama.cpp inference wrapper.

Zero-disk strategy
  1.  Shard bytes live in a Python dict (RAM).
  2.  assemble_gguf_bytes() builds the complete GGUF file as a bytes object.
  3.  The bytes are written to /dev/shm (tmpfs — RAM-backed, never touches the
      physical disk).  Fallback: /run/shm → /tmp.
  4.  llama-cpp-python loads (mmaps) the tmpfs file.
  5.  The in-memory bytes object is immediately deleted; GC reclaims it.
  6.  cleanup() unlinks the tmpfs file when done.

Delta support (v0.3.0)
  ShardBridge now exposes its shard_data dict publicly so callers can
  patch individual entries before reassembling.

  apply_delta(patches) merges a dict[shard_name, bytes] into the live
  shard_data, unloads the current llama.cpp model, and reloads from a
  freshly assembled GGUF — all in-process with no network I/O.

  Typical lifecycle for a delta update:
      patches = await http.fetch_delta_shards(model_id, delta_manifest)
      load_time = bridge.apply_delta(patches, n_ctx=2048, on_status=print)
      output, metrics = bridge.infer(prompt)
"""
from __future__ import annotations

import os
import gc
import struct
import json
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

from modelpulse.shared.models import InferenceMetrics, ShardManifest


# Tmpfs discovery

_TMPFS_CANDIDATES = ["/dev/shm", "/run/shm"]


def _find_tmpfs() -> Path:
    for p in _TMPFS_CANDIDATES:
        path = Path(p)
        if path.exists() and os.access(p, os.W_OK):
            return path
    return Path(os.environ.get("TMPDIR", "/tmp"))


# GGUF value writer

def _write_gguf_value(
    buf: bytearray,
    type_id: int,
    value,
    array_elem_type: Optional[int] = None,
) -> None:
    """Serialise one typed metadata value into buf (little-endian)."""

    _SCALAR = {
        0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
        6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
    }

    def _str(s: str) -> None:
        b = s.encode("utf-8")
        buf.extend(struct.pack("<Q", len(b)))
        buf.extend(b)

    if type_id == 8:
        _str(str(value))
        return

    if type_id == 9:
        if array_elem_type is None:
            raise ValueError("ARRAY metadata entry is missing array_elem_type")
        if not isinstance(value, list):
            raise ValueError(f"ARRAY value must be a list, got {type(value)}")
        buf.extend(struct.pack("<I", array_elem_type))
        buf.extend(struct.pack("<Q", len(value)))
        for item in value:
            _write_gguf_value(buf, array_elem_type, item)
        return

    fmt = _SCALAR.get(type_id)
    if fmt is None:
        raise ValueError(f"Unsupported metadata type_id: {type_id}")
    buf.extend(struct.pack("<" + fmt, value))


# In-memory GGUF assembly

def assemble_gguf_bytes(
    manifest: ShardManifest,
    shard_data: dict[str, bytes],
) -> bytes:
    """
    Assemble a complete, spec-compliant GGUF file from shard bytes.
    Returns the raw bytes ready to be written to tmpfs.
    """
    alignment     = manifest.alignment
    ordered_names = list(manifest.shards.keys())
    gguf_version  = manifest.gguf_version
    metadata_kvs  = manifest.gguf_metadata_kvs

    running_offset = 0
    offsets: dict[str, int] = {}
    for name in ordered_names:
        if running_offset % alignment:
            running_offset += alignment - (running_offset % alignment)
        offsets[name] = running_offset
        running_offset += manifest.shards[name]["nbytes"]

    buf = bytearray()

    def _str(s: str) -> None:
        b = s.encode("utf-8")
        buf.extend(struct.pack("<Q", len(b)))
        buf.extend(b)

    buf.extend(struct.pack("<I", 0x46554747))
    buf.extend(struct.pack("<I", gguf_version))

    tensor_count = len(ordered_names)
    kv_count     = len(metadata_kvs)

    if gguf_version == 1:
        buf.extend(struct.pack("<II", tensor_count, kv_count))
    else:
        buf.extend(struct.pack("<QQ", tensor_count, kv_count))

    for kv in metadata_kvs:
        _str(kv["key"])
        buf.extend(struct.pack("<I", kv["type_id"]))
        _write_gguf_value(buf, kv["type_id"], kv["value"], kv.get("array_elem_type"))

    for name in ordered_names:
        entry = manifest.shards[name]
        _str(name)
        dims = entry["dims"]
        buf.extend(struct.pack("<I", len(dims)))
        for d in dims:
            buf.extend(struct.pack("<Q", d))
        buf.extend(struct.pack("<I", entry["ggml_type"]))
        buf.extend(struct.pack("<Q", offsets[name]))

    if len(buf) % alignment:
        buf.extend(b"\x00" * (alignment - len(buf) % alignment))

    written = 0
    for name in ordered_names:
        if written % alignment:
            pad = alignment - (written % alignment)
            buf.extend(b"\x00" * pad)
            written += pad

        full_shard = shard_data[name]

        if len(full_shard) > 12 and full_shard[:4] == b"SHRD":
            hdr_len  = struct.unpack("<I", full_shard[8:12])[0]
            raw_data = full_shard[12 + hdr_len:]
        else:
            raw_data = full_shard

        nbytes   = manifest.shards[name]["nbytes"]
        raw_data = raw_data[:nbytes]

        buf.extend(raw_data)
        written += len(raw_data)

    return bytes(buf)


# Hardware helpers

def _read_cpu_temp() -> Optional[float]:
    try:
        import psutil
        temps = psutil.sensors_temperatures()
        for key in ("cpu_thermal", "cpu-thermal", "coretemp", "k10temp", "acpitz", "soc_thermal"):
            if key in temps and temps[key]:
                return float(temps[key][0].current)
    except Exception:
        pass
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            ttype = (zone / "type").read_text().strip().lower()
            if any(k in ttype for k in ("cpu", "soc", "arm")):
                raw = (zone / "temp").read_text().strip()
                return float(raw) / 1000.0
        except Exception:
            continue
    return None


def _detect_hw() -> str:
    for candidate in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            return Path(candidate).read_bytes().decode("utf-8", errors="replace").strip("\x00 ")
        except Exception:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            for prefix in ("Model name", "model name", "Hardware"):
                if line.startswith(prefix):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    import platform
    return platform.processor() or platform.machine() or "unknown"


def _os_info() -> str:
    import platform
    return f"{platform.system()} {platform.release()} {platform.machine()}"


def _proc_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1_048_576
    except Exception:
        return 0.0


def _system_ram_used_mb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().used / 1_048_576
    except Exception:
        return 0.0


# ShardBridge

class ShardBridge:
    """
    Full lifecycle: assemble → write to tmpfs → load via llama.cpp → infer → cleanup.

    Delta support
    Call apply_delta(patches) to hot-swap updated tensor shards without
    re-downloading unchanged data.  The method:
      1. Merges patches into self.shard_data (in-place).
      2. Unloads the current llama.cpp model and removes the old tmpfs file.
      3. Reassembles a new GGUF from the patched shard_data.
      4. Writes and loads the new GGUF via llama.cpp.
      5. Updates internal RAM and timing bookkeeping.

    After apply_delta() returns the bridge is ready for infer() calls
    exactly as after a normal load().

    Usage
        bridge = ShardBridge(manifest, shard_data)
        bridge.load(on_status=print)
        output, metrics = bridge.infer("Explain edge computing.")

        # Later, when a delta arrives:
        patches = await http.fetch_delta_shards(model_id, delta_manifest)
        bridge.apply_delta(patches, on_status=print)
        output2, metrics2 = bridge.infer("What changed?")

        bridge.cleanup()
    """

    def __init__(
        self,
        manifest: ShardManifest,
        shard_data: dict[str, bytes],
        compute_perplexity: bool = True,
    ):
        self.manifest           = manifest
        # Exposed publicly so callers can inspect or patch entries.
        self.shard_data         = shard_data
        self.compute_perplexity = compute_perplexity

        self._tmpfs_path: Optional[Path] = None
        self._llm                        = None
        self._load_time_s: float         = 0.0

        self._rss_before_load_mb: float  = 0.0
        self._rss_after_load_mb: float   = 0.0

        # n_ctx stored at load() time so apply_delta() can reuse it.
        self._n_ctx: int = 2048

    # Perplexity

    def perplexity(self, text: str) -> float:
        if self._llm is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        import math

        resp = self._llm.create_completion(
            prompt=text, max_tokens=1, temperature=0.0,
            echo=True, logprobs=1, stream=False,
        )

        choice        = resp["choices"][0]
        usage         = resp.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))

        all_logprobs = choice["logprobs"]["token_logprobs"]
        vals = [lp for lp in all_logprobs[:prompt_tokens] if lp is not None]

        if not vals:
            return float("nan")

        avg_nll = -sum(vals) / len(vals)
        return math.exp(avg_nll)

    # Load

    def load(
        self,
        n_ctx: int = 2048,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> float:
        """
        Assemble GGUF in RAM, write to tmpfs, load llama.cpp.
        Returns load_time_s.
        """
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed.\n"
                "  pip install llama-cpp-python"
            ) from exc

        self._n_ctx = n_ctx
        tmpfs       = _find_tmpfs()
        filename    = f"sb_{os.getpid()}.gguf"
        self._tmpfs_path = tmpfs / filename

        t0 = time.perf_counter()

        _emit(on_status, "assembling GGUF in memory")
        gguf_bytes    = assemble_gguf_bytes(self.manifest, self.shard_data)
        gguf_size_mb  = len(gguf_bytes) / 1_048_576

        _emit(on_status, f"writing {gguf_size_mb:.0f} MB → {self._tmpfs_path}")
        self._tmpfs_path.write_bytes(gguf_bytes)

        del gguf_bytes
        gc.collect()

        self._rss_before_load_mb = _proc_rss_mb()

        _emit(on_status, f"loading model via llama.cpp  (n_ctx={n_ctx})")
        self._llm = Llama(
            model_path=str(self._tmpfs_path),
            n_ctx=n_ctx,
            n_threads=os.cpu_count() or 4,
            logits_all=self.compute_perplexity,
            verbose=False,
        )

        self._load_time_s       = time.perf_counter() - t0
        self._rss_after_load_mb = _proc_rss_mb()
        return self._load_time_s

    # Delta hot-swap

    def apply_delta(
        self,
        patches: dict[str, bytes],
        *,
        n_ctx: Optional[int] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> float:
        """
        Merge *patches* into shard_data, then reload the model from a freshly
        assembled GGUF — without re-downloading unchanged shards.

        Parameters
        patches    dict[shard_name, bytes] — only the changed shards.
                   Must be a subset of the keys already in self.shard_data.
                   Unknown keys are accepted and logged as warnings (they may
                   represent newly added tensors in extended quantisation runs).
        n_ctx      Context window to use for the reloaded model.  Defaults to
                   the value passed at the last load() call.
        on_status  Optional progress callback (same signature as for load()).

        Returns
        load_time_s   Wall-clock seconds from patch application to model ready.

        Notes
        • The old tmpfs GGUF is deleted before the new one is written to
          avoid running out of /dev/shm when the model is large.
        • self.shard_data is mutated in-place; the caller does not need to
          keep a separate reference.
        """
        if self._llm is None:
            raise RuntimeError(
                "No model is currently loaded. Call load() before apply_delta()."
            )

        effective_n_ctx = n_ctx if n_ctx is not None else self._n_ctx

        # 1. Log any unexpected shard names
        # The patches dict may use filenames (e.g. blk.0.attn_k.weight.shard)
        # instead of tensor names (e.g. blk.0.attn_k.weight). Normalize them.
        file_to_name = {s["file"]: name for name, s in self.manifest.shards.items()}
        normalized = {}
        for k, v in patches.items():
            if k in self.manifest.shards:
                normalized[k] = v
            elif k in file_to_name:
                normalized[file_to_name[k]] = v
            else:
                normalized[k] = v
        patches = normalized

        unknown = set(patches) - set(self.manifest.shards)
        if unknown:
            for name in sorted(unknown):
                _emit(on_status, f"[warn] delta patch contains unknown shard: {name!r} — skipping")
            patches = {k: v for k, v in patches.items() if k not in unknown}

        if not patches:
            raise ValueError("apply_delta() called with no valid shard patches.")

        _emit(
            on_status,
            f"applying delta  ({len(patches)} shard(s) changed: "
            + ", ".join(sorted(patches)) + ")",
        )

        # 2. Merge patches into shard_data and update manifest
        for name, data in patches.items():
            old_size = len(self.shard_data.get(name, b""))
            self.shard_data[name] = data

            # If the patch is a SHRD container, extract metadata and update manifest
            if len(data) >= 12 and data[:4] == b"SHRD":
                try:
                    hdr_len = struct.unpack("<I", data[8:12])[0]
                    hdr_json = json.loads(data[12:12 + hdr_len].decode("utf-8"))

                    # Update manifest entry for this tensor
                    if name in self.manifest.shards:
                        entry = self.manifest.shards[name]
                        entry["nbytes"]         = hdr_json.get("nbytes", entry["nbytes"])
                        entry["ggml_type"]      = hdr_json.get("ggml_type", entry["ggml_type"])
                        entry["ggml_type_name"] = hdr_json.get("ggml_type_name", entry.get("ggml_type_name"))
                        entry["dims"]           = hdr_json.get("dims", entry["dims"])
                        entry["sha256"]         = hdr_json.get("sha256", entry["sha256"])
                except Exception as exc:
                    _emit(on_status, f"[warn] failed to parse SHRD header for {name}: {exc}")

            _emit(
                on_status,
                f"  patched {name}  "
                f"({old_size // 1024} KB → {len(data) // 1024} KB)",
            )

        # Update total model bytes in manifest
        self.manifest.total_bytes = sum(s["nbytes"] for s in self.manifest.shards.values())

        # 3. Unload current model and free tmpfs
        _emit(on_status, "unloading current model")
        self._llm = None
        gc.collect()

        if self._tmpfs_path and self._tmpfs_path.exists():
            self._tmpfs_path.unlink(missing_ok=True)
            _emit(on_status, f"removed old tmpfs file  {self._tmpfs_path}")

        # 4. Reassemble + reload (reuse load() internals)
        return self.load(n_ctx=effective_n_ctx, on_status=on_status)

    # RAM delta

    @property
    def ram_delta_mb(self) -> float:
        return max(0.0, self._rss_after_load_mb - self._rss_before_load_mb)

    # Infer

    def infer(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, InferenceMetrics]:
        if self._llm is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        import psutil

        psutil.cpu_percent(interval=None)

        t_start: float        = time.perf_counter()
        t_first_token: float  = 0.0
        t_second_token: float = 0.0
        tokens: list[str]     = []
        token_count: int      = 0

        stream = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        for chunk in stream:
            tok = chunk["choices"][0]["text"]
            now = time.perf_counter()

            if token_count == 0:
                t_first_token = now - t_start
            elif token_count == 1:
                t_second_token = now

            tokens.append(tok)
            token_count += 1

            if on_token:
                on_token(tok)

        t_end     = time.perf_counter()
        cpu_pct   = psutil.cpu_percent(interval=None)

        full_output   = "".join(tokens)
        total_elapsed = t_end - t_start

        if token_count > 1 and t_second_token > 0.0:
            decode_tokens  = token_count - 1
            decode_elapsed = t_end - t_second_token
            tokens_per_sec = decode_tokens / decode_elapsed if decode_elapsed > 0 else 0.0
        elif total_elapsed > 0:
            tokens_per_sec = token_count / total_elapsed
        else:
            tokens_per_sec = 0.0

        metrics = InferenceMetrics(
            load_time_s         = self._load_time_s,
            time_to_first_tok_s = t_first_token,
            tokens_per_sec      = tokens_per_sec,
            tokens_generated    = token_count,
            ram_delta_mb        = self.ram_delta_mb,
            ram_used_mb         = _system_ram_used_mb(),
            cpu_temp_c          = _read_cpu_temp(),
            cpu_percent         = cpu_pct,
            device_hw           = _detect_hw(),
            os_info             = _os_info(),
            prompt              = prompt,
            output              = full_output,
            source_model        = self.manifest.source_model,
        )

        return full_output, metrics

    # Cleanup

    def cleanup(self) -> None:
        """Unlink the tmpfs model file and release llama.cpp resources."""
        self._llm = None
        if self._tmpfs_path and self._tmpfs_path.exists():
            self._tmpfs_path.unlink(missing_ok=True)
        self._tmpfs_path = None

    def __enter__(self) -> "ShardBridge":
        return self

    def __exit__(self, *_) -> None:
        self.cleanup()


# util

def _emit(cb: Optional[Callable[[str], None]], msg: str) -> None:
    if cb:
        cb(msg)