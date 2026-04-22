"""
modelpulse.client.bridge
In-memory GGUF assembly + llama.cpp inference wrapper.

Zero-disk strategy
──────────────────
  1.  Shard bytes live in a Python dict (RAM).
  2.  assemble_gguf_bytes() builds the complete GGUF file as a bytes object.
  3.  The bytes are written to /dev/shm (tmpfs — RAM-backed, never touches the
      physical disk).  Fallback: /run/shm → /tmp.
  4.  llama-cpp-python loads (mmaps) the tmpfs file.
  5.  The in-memory bytes object is immediately deleted; GC reclaims it.
  6.  cleanup() unlinks the tmpfs file when done.


"""
from __future__ import annotations

import os
import gc
import struct
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

from modelpulse.shared.models import InferenceMetrics, ShardManifest


# ── Tmpfs discovery ──────────────────────────────────────────────────────────

_TMPFS_CANDIDATES = ["/dev/shm", "/run/shm"]


def _find_tmpfs() -> Path:
    for p in _TMPFS_CANDIDATES:
        path = Path(p)
        if path.exists() and os.access(p, os.W_OK):
            return path
    return Path(os.environ.get("TMPDIR", "/tmp"))


# ── GGUF value writer (mirrors gguf_to_shards._write_gguf_value) ─────────────

def _write_gguf_value(
    buf: bytearray,
    type_id: int,
    value,
    array_elem_type: Optional[int] = None,
) -> None:
    """Serialise one typed metadata value into buf (little-endian)."""

    _SCALAR = {
        0: "B",  # UINT8
        1: "b",  # INT8
        2: "H",  # UINT16
        3: "h",  # INT16
        4: "I",  # UINT32
        5: "i",  # INT32
        6: "f",  # FLOAT32
        7: "?",  # BOOL
        10: "Q", # UINT64
        11: "q", # INT64
        12: "d", # FLOAT64
    }

    def _str(s: str) -> None:
        b = s.encode("utf-8")
        buf.extend(struct.pack("<Q", len(b)))
        buf.extend(b)

    if type_id == 8:  # STRING
        _str(str(value))
        return

    if type_id == 9:  # ARRAY
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


# ── In-memory GGUF assembly ──────────────────────────────────────────────────

def assemble_gguf_bytes(
    manifest: ShardManifest,
    shard_data: dict[str, bytes],
) -> bytes:
    """
    Assemble a complete, spec-compliant GGUF file from shard bytes.
    Returns the raw bytes ready to be written to tmpfs.
    """
    alignment      = manifest.alignment
    ordered_names  = list(manifest.shards.keys())
    gguf_version   = manifest.gguf_version
    metadata_kvs   = manifest.gguf_metadata_kvs

    # ── Compute data-section offsets (relative to start of data section) ──────
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

    # ── GGUF header ───────────────────────────────────────────────────────────
    buf.extend(struct.pack("<I", 0x46554747))       # magic "GGUF"
    buf.extend(struct.pack("<I", gguf_version))

    tensor_count = len(ordered_names)
    kv_count     = len(metadata_kvs)

    if gguf_version == 1:
        buf.extend(struct.pack("<II", tensor_count, kv_count))
    else:
        buf.extend(struct.pack("<QQ", tensor_count, kv_count))

    # ── Metadata KV pairs ─────────────────────────────────────────────────────
    for kv in metadata_kvs:
        _str(kv["key"])
        buf.extend(struct.pack("<I", kv["type_id"]))
        _write_gguf_value(buf, kv["type_id"], kv["value"], kv.get("array_elem_type"))

    # ── Tensor info array ─────────────────────────────────────────────────────
    for name in ordered_names:
        entry = manifest.shards[name]
        _str(name)
        dims = entry["dims"]
        buf.extend(struct.pack("<I", len(dims)))
        for d in dims:
            buf.extend(struct.pack("<Q", d))
        buf.extend(struct.pack("<I", entry["ggml_type"]))
        buf.extend(struct.pack("<Q", offsets[name]))

    # ── Alignment padding before data section ─────────────────────────────────
    if len(buf) % alignment:
        buf.extend(b"\x00" * (alignment - len(buf) % alignment))

    # ── Tensor data ───────────────────────────────────────────────────────────
    written = 0
    for name in ordered_names:
        if written % alignment:
            pad = alignment - (written % alignment)
            buf.extend(b"\x00" * pad)
            written += pad

        # shard_data[name] contains the FULL .shard container (headers + payload)
        full_shard = shard_data[name]
        
        # Unwrap the SHRD container: skip 12-byte preamble + JSON header
        if len(full_shard) > 12 and full_shard[:4] == b"SHRD":
            hdr_len = struct.unpack("<I", full_shard[8:12])[0]
            raw_data = full_shard[12 + hdr_len :]
        else:
            # Fallback for unwrapped shards
            raw_data = full_shard
            
        # Ensure we only take exactly nbytes as defined in manifest
        # (ignoring any potential padding in the shard file)
        nbytes = manifest.shards[name]["nbytes"]
        raw_data = raw_data[:nbytes]

        buf.extend(raw_data)
        written += len(raw_data)

    return bytes(buf)


# ── Hardware helpers ─────────────────────────────────────────────────────────

def _read_cpu_temp() -> Optional[float]:
    """Read CPU temperature from Linux thermal zones or psutil."""
    try:
        import psutil
        temps = psutil.sensors_temperatures()
        for key in ("cpu_thermal", "cpu-thermal", "coretemp", "k10temp",
                     "acpitz", "soc_thermal"):
            if key in temps and temps[key]:
                return float(temps[key][0].current)
    except Exception:
        pass
    # Fallback: walk /sys/class/thermal directly (works on RPi / Jetson)
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
    """Return a human-readable hardware string."""
    # Raspberry Pi / Jetson expose the board model here
    for candidate in (
        "/proc/device-tree/model",
        "/sys/firmware/devicetree/base/model",
    ):
        try:
            return Path(candidate).read_bytes().decode("utf-8", errors="replace").strip("\x00 ")
        except Exception:
            pass
    # Generic Linux CPU info
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


def _ram_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1_048_576
    except Exception:
        return 0.0


# ── ShardBridge ──────────────────────────────────────────────────────────────

class ShardBridge:
    """
    Full lifecycle: assemble → write to tmpfs → load via llama.cpp → infer → cleanup.

    Usage:
        bridge = ShardBridge(manifest, shard_data)
        bridge.load(on_status=print)
        output, metrics = bridge.infer("Explain edge computing.")
        bridge.cleanup()
    """

    def __init__(
        self,
        manifest: ShardManifest,
        shard_data: dict[str, bytes],
    ):
        self.manifest = manifest
        self.shard_data = shard_data
        self._tmpfs_path: Optional[Path] = None
        self._llm = None
        self._load_time_s = 0.0
        self._ram_before_mb = 0.0
        self._ram_after_mb = 0.0

    def load(
        self,
        n_ctx: int = 2048,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> float:
        """
        Assemble GGUF in RAM, write to tmpfs, load llama.cpp.
        Returns full load_time_s covering assembly + write + model init.
        """
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed.\n"
                "  pip install llama-cpp-python"
            ) from exc

        tmpfs = _find_tmpfs()
        filename = f"sb_{os.getpid()}.gguf"
        self._tmpfs_path = tmpfs / filename

        t0 = time.perf_counter()

        # ── Assemble ──────────────────────────────────────────────────────────
        _emit(on_status, "assembling GGUF in memory")
        gguf_bytes = assemble_gguf_bytes(self.manifest, self.shard_data)
        gguf_size_mb = len(gguf_bytes) / 1_048_576

        # ── Write to tmpfs ────────────────────────────────────────────────────
        _emit(on_status, f"writing {gguf_size_mb:.0f} MB → {self._tmpfs_path}")
        self._tmpfs_path.write_bytes(gguf_bytes)

        # Free Python-side copy before measuring RAM
        del gguf_bytes
        gc.collect()

        # Baseline RAM after temporary assembly bytes are gone
        self._ram_before_mb = _ram_mb()

        # ── Load ──────────────────────────────────────────────────────────────
        _emit(on_status, f"loading model via llama.cpp  (n_ctx={n_ctx})")
        self._llm = Llama(
            model_path=str(self._tmpfs_path),
            n_ctx=n_ctx,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )

        self._load_time_s = time.perf_counter() - t0
        self._ram_after_mb = _ram_mb()
        return self._load_time_s

    @property
    def ram_delta_mb(self) -> float:
        return max(0.0, self._ram_after_mb - self._ram_before_mb)

    def infer(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, InferenceMetrics]:
        """
        Run inference on the loaded model.
        cpu_percent is measured across the inference window, not after it.
        """
        if self._llm is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        import psutil  # type: ignore

        # Prime CPU counter so the next call measures the inference window
        psutil.cpu_percent(interval=None)

        t_start = time.perf_counter()
        first_token_time: Optional[float] = None
        tokens: list[str] = []

        stream = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        for chunk in stream:
            tok = chunk["choices"][0]["text"]
            if first_token_time is None:
                first_token_time = time.perf_counter() - t_start
            tokens.append(tok)
            if on_token:
                on_token(tok)

        elapsed = time.perf_counter() - t_start
        full_output = "".join(tokens)
        n_tokens = len(tokens)

        # Measure CPU usage over the whole inference window
        cpu_percent = psutil.cpu_percent(interval=None)

        metrics = InferenceMetrics(
            load_time_s=self._load_time_s,
            time_to_first_tok_s=first_token_time or 0.0,
            tokens_per_sec=n_tokens / elapsed if elapsed > 0 else 0.0,
            tokens_generated=n_tokens,
            ram_delta_mb=self.ram_delta_mb,
            ram_used_mb=psutil.virtual_memory().used / 1_048_576,
            cpu_temp_c=_read_cpu_temp(),
            cpu_percent=cpu_percent,
            device_hw=_detect_hw(),
            os_info=_os_info(),
            prompt=prompt,
            output=full_output,
            source_model=self.manifest.source_model,
        )

        return full_output, metrics

    # ── Cleanup ───────────────────────────────────────────────────────────────

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


# ── util ─────────────────────────────────────────────────────────────────────

def _emit(cb: Optional[Callable[[str], None]], msg: str) -> None:
    if cb:
        cb(msg)