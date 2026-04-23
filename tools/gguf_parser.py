"""
Low-level GGUF binary reader. 
Supports GGUF versions 1, 2, 3.
Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# GGML type registry 
# (type_id, block_size, block_bytes)
# nbytes(tensor) = (n_elements / block_size) * block_bytes
_GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0:  ("F32",   1,   4),
    1:  ("F16",   1,   2),
    2:  ("Q4_0",  32,  18),
    3:  ("Q4_1",  32,  20),
    6:  ("Q5_0",  32,  22),
    7:  ("Q5_1",  32,  24),
    8:  ("Q8_0",  32,  34),
    9:  ("Q8_1",  32,  36),
    10: ("Q2_K",  256, 84),
    11: ("Q3_K",  256, 110),
    12: ("Q4_K",  256, 144),
    13: ("Q5_K",  256, 176),
    14: ("Q6_K",  256, 210),
    15: ("Q8_K",  256, 292),
    16: ("I8",    1,   1),
    17: ("I16",   1,   2),
    18: ("I32",   1,   4),
    30: ("BF16",  1,   2),
    31: ("Q4_0_4_4",  32, 18),
    32: ("Q4_0_4_8",  32, 18),
    33: ("Q4_0_8_8",  32, 18),
}

# GGUF metadata value type IDs
_GGUF_VALUE_TYPES = {
    0:  ("UINT8",   "B"),
    1:  ("INT8",    "b"),
    2:  ("UINT16",  "H"),
    3:  ("INT16",   "h"),
    4:  ("UINT32",  "I"),
    5:  ("INT32",   "i"),
    6:  ("FLOAT32", "f"),
    7:  ("BOOL",    "?"),
    8:  ("STRING",  None),
    9:  ("ARRAY",   None),
    10: ("UINT64",  "Q"),
    11: ("INT64",   "q"),
    12: ("FLOAT64", "d"),
}

GGUF_MAGIC = 0x46554747  # b"GGUF"
DEFAULT_ALIGN = 32


def ggml_type_name(type_id: int) -> str:
    return _GGML_TYPES.get(type_id, (f"UNKNOWN_{type_id}", 1, 0))[0]


def ggml_tensor_nbytes(type_id: int, dims: list[int]) -> int:
    """Compute raw byte size of a tensor given its ggml type and shape."""
    entry = _GGML_TYPES.get(type_id)
    if entry is None:
        raise ValueError(f"Unknown ggml type id: {type_id}")
    _, block_size, block_bytes = entry
    n_elements = 1
    for d in dims:
        n_elements *= d
    if n_elements % block_size != 0:
        raise ValueError(
            f"n_elements ({n_elements}) not divisible by block_size ({block_size}) "
            f"for type {ggml_type_name(type_id)}"
        )
    return (n_elements // block_size) * block_bytes


@dataclass
class GGUFTensorInfo:
    name: str
    dims: list[int]
    ggml_type: int
    offset: int          # relative to start of data section
    nbytes: int          # computed from type + dims

    @property
    def type_name(self) -> str:
        return ggml_type_name(self.ggml_type)


@dataclass
class GGUFMetadataEntry:
    key: str
    type_id: int
    value: Any
    array_elem_type: int | None = None


@dataclass
class GGUFFile:
    path: Path
    version: int
    metadata: dict[str, Any]
    metadata_entries: list[GGUFMetadataEntry]
    tensors: list[GGUFTensorInfo]
    data_offset: int
    alignment: int

    def tensor_by_name(self, name: str) -> GGUFTensorInfo | None:
        for t in self.tensors:
            if t.name == name:
                return t
        return None


class GGUFReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._f = open(self.path, "rb")

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # primitive readers 
    def _read(self, fmt: str) -> tuple:
        size = struct.calcsize(fmt)
        data = self._f.read(size)
        if len(data) < size:
            raise EOFError(f"Unexpected EOF reading {fmt!r} at offset {self._f.tell()}")
        return struct.unpack_from("<" + fmt, data)

    def _u8(self) -> int: return self._read("B")[0]
    def _u16(self) -> int: return self._read("H")[0]
    def _u32(self) -> int: return self._read("I")[0]
    def _u64(self) -> int: return self._read("Q")[0]
    def _i8(self) -> int: return self._read("b")[0]
    def _i16(self) -> int: return self._read("h")[0]
    def _i32(self) -> int: return self._read("i")[0]
    def _i64(self) -> int: return self._read("q")[0]
    def _f32(self) -> float: return self._read("f")[0]
    def _f64(self) -> float: return self._read("d")[0]
    def _bool(self) -> bool: return bool(self._u8())

    def _string(self) -> str:
        length = self._u64()
        raw = self._f.read(length)
        return raw.decode("utf-8", errors="replace")

    def _value(self, type_id: int) -> tuple[Any, int | None]:
        """Read a single metadata value of the given type_id.
        Returns: (value, array_elem_type)
        """
        readers = {
            0: self._u8, 1: self._i8,
            2: self._u16, 3: self._i16,
            4: self._u32, 5: self._i32,
            6: self._f32, 7: self._bool,
            8: self._string,
            10: self._u64, 11: self._i64, 12: self._f64,
        }

        if type_id == 9:  # ARRAY
            elem_type = self._u32()
            count = self._u64()
            read_elem = readers.get(elem_type)
            if read_elem is None:
                raise ValueError(f"Nested array or unsupported array elem type: {elem_type}")
            return [read_elem() for _ in range(count)], elem_type

        reader = readers.get(type_id)
        if reader is None:
            raise ValueError(f"Unknown value type_id: {type_id}")
        return reader(), None

    def parse(self) -> GGUFFile:
        f = self._f
        f.seek(0)

        magic = self._u32()
        if magic != GGUF_MAGIC:
            raise ValueError(
                f"Not a GGUF file (magic={magic:#010x}, expected {GGUF_MAGIC:#010x})"
            )

        version = self._u32()
        if version not in (1, 2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")

        if version == 1:
            tensor_count = self._u32()
            kv_count = self._u32()
        else:
            tensor_count = self._u64()
            kv_count = self._u64()

        print(f"  GGUF v{version}  |  {tensor_count} tensors  |  {kv_count} metadata entries")

        metadata: dict[str, Any] = {}
        metadata_entries: list[GGUFMetadataEntry] = []

        for _ in range(kv_count):
            key = self._string()
            type_id = self._u32()
            value, array_elem_type = self._value(type_id)
            metadata[key] = value
            metadata_entries.append(
                GGUFMetadataEntry(
                    key=key,
                    type_id=type_id,
                    value=value,
                    array_elem_type=array_elem_type,
                )
            )

        alignment = int(metadata.get("general.alignment", DEFAULT_ALIGN))

        tensor_infos: list[GGUFTensorInfo] = []
        for _ in range(tensor_count):
            name = self._string()
            n_dims = self._u32()
            dims = [self._u64() for _ in range(n_dims)]
            ttype = self._u32()
            offset = self._u64()
            nbytes = ggml_tensor_nbytes(ttype, dims)
            tensor_infos.append(
                GGUFTensorInfo(
                    name=name,
                    dims=dims,
                    ggml_type=ttype,
                    offset=offset,
                    nbytes=nbytes,
                )
            )

        pos = f.tell()
        pad = (alignment - (pos % alignment)) % alignment
        data_offset = pos + pad

        return GGUFFile(
            path=self.path,
            version=version,
            metadata=metadata,
            metadata_entries=metadata_entries,
            tensors=tensor_infos,
            data_offset=data_offset,
            alignment=alignment,
        )

    def read_tensor_data(self, gguf: GGUFFile, tensor: GGUFTensorInfo) -> bytes:
        abs_offset = gguf.data_offset + tensor.offset
        self._f.seek(abs_offset)
        data = self._f.read(tensor.nbytes)
        if len(data) != tensor.nbytes:
            raise IOError(
                f"Short read for tensor '{tensor.name}': "
                f"expected {tensor.nbytes} bytes, got {len(data)}"
            )
        return data


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gguf_parser.py <model.gguf>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"\nParsing: {path}  ({os.path.getsize(path) / 1e6:.1f} MB)\n")

    with GGUFReader(path) as reader:
        gguf = reader.parse()

    print(f"\n{'─'*60}")
    print(f"{'Tensor':<45} {'Type':<8} {'Dims':<28} {'Bytes':>12}")
    print(f"{'─'*60}")
    total = 0
    for t in gguf.tensors:
        dims_str = "×".join(str(d) for d in t.dims)
        print(f"  {t.name:<43} {t.type_name:<8} {dims_str:<28} {t.nbytes:>12,}")
        total += t.nbytes
    print(f"{'─'*60}")
    print(f"  {'TOTAL DATA':<43} {'':8} {'':28} {total:>12,}")
    print(f"  data section starts at byte: {gguf.data_offset:,}")
    print(f"  alignment: {gguf.alignment}")
    print(f"\n  metadata keys ({len(gguf.metadata)}):")
    for k, v in list(gguf.metadata.items())[:20]:
        vr = repr(v)
        print(f"    {k}: {vr[:60]}")
    if len(gguf.metadata) > 20:
        print(f"    ... and {len(gguf.metadata)-20} more")
    