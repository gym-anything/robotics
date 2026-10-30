"""Lossless array storage for local traces; JSON expansion only on retrieval."""

import json
import struct
import zlib
import zstandard
import numpy as np


def encode(value):
    buffers = []
    offset = 0

    def pack(item, path="root"):
        nonlocal offset
        if isinstance(item, np.ndarray):
            a = np.ascontiguousarray(item)
            if a.dtype.hasobject or not np.isfinite(a).all():
                raise ValueError(f"Trace arrays must be finite numeric values at {path}")
            data = a.tobytes()
            descriptor = {
                "__array__": True,
                "dtype": a.dtype.str,
                "shape": a.shape,
                "offset": offset,
                "bytes": len(data),
            }
            buffers.append(data)
            offset += len(data)
            return descriptor
        if isinstance(item, dict):
            return {k: pack(v, f"{path}.{k}") for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [pack(v, f"{path}[{i}]") for i, v in enumerate(item)]
        return item

    header = json.dumps(pack(value), allow_nan=False, separators=(",", ":")).encode()
    return b"WBT2" + zstandard.ZstdCompressor(level=1).compress(
        struct.pack("!I", len(header)) + header + b"".join(buffers)
    )


def decode(payload, *, arrays=False):
    data = (
        zstandard.ZstdDecompressor().decompress(payload[4:])
        if payload.startswith(b"WBT2")
        else zlib.decompress(payload)
    )
    size = struct.unpack("!I", data[:4])[0]
    header = json.loads(data[4 : 4 + size])
    array_bytes = memoryview(data)[4 + size :]

    def unpack(item):
        if isinstance(item, dict):
            if item.get("__array__") is True:
                view = array_bytes[item["offset"] : item["offset"] + item["bytes"]]
                value = np.frombuffer(view, dtype=item["dtype"]).reshape(item["shape"])
                return value.copy() if arrays else value.tolist()
            return {k: unpack(v) for k, v in item.items()}
        if isinstance(item, list):
            return [unpack(v) for v in item]
        return item

    return unpack(header)
