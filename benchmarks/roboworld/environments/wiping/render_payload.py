"""Lossless bounded array compression for the local wiping renderer only.

Keep the shared 64MiB IPC limit and 32MiB decoded-array limit. This does not
resample geometry or images, change float32 precision, or accept executable data.
Legacy raw-base64 arrays remain readable. Zlib output is bounded while decoding,
not checked only after allocating an arbitrarily large result.
"""
import base64
import math
import zlib
import numpy as np
from titrate.newton15.render_payload import DTYPES, MAX_ARRAY_BYTES
from titrate.newton15.render_payload import encode as raw_encode, decode as raw_decode


def encode(values, dtype):
    value = raw_encode(values, dtype)
    raw = base64.b64decode(value['data'], validate=True)
    packed = zlib.compress(raw, level=1)
    if len(packed) < len(raw):
        value['data'] = base64.b64encode(packed).decode('ascii')
        value['codec'] = 'zlib'
    return value


def decode(value, dtype, width=None):
    if not isinstance(value, dict) or 'codec' not in value:
        return raw_decode(value, dtype, width)
    if (set(value) != {'dtype', 'shape', 'data', 'codec'} or
            value['codec'] != 'zlib' or dtype not in DTYPES or value['dtype'] != dtype):
        raise ValueError('Compressed render array schema/dtype mismatch')
    shape = value['shape']
    if (not isinstance(shape, list) or len(shape) != (1 if width is None else 2) or
            any(type(n) is not int or n < 0 for n in shape) or
            (width is not None and shape[1] != width)):
        raise ValueError('Invalid compressed render array shape')
    size = math.prod(shape)*DTYPES[dtype].itemsize
    data = value['data']
    if size > MAX_ARRAY_BYTES or not isinstance(data, str) or len(data) > 4*((size+2)//3):
        raise ValueError('Oversized compressed render array')
    packed = base64.b64decode(data, validate=True)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(packed, size+1)
    if (len(raw) != size or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data):
        raise ValueError('Truncated, oversized or trailing compressed render data')
    result = np.frombuffer(raw, DTYPES[dtype]).reshape(shape)
    if not np.isfinite(result).all():raise ValueError('Non-finite compressed render geometry')
    return result
