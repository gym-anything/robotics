"""Bounded, lossless typed arrays inside the authenticated JSON render channel.

No pickle, executable serialization, paths, or compressed archives. Array bytes
have the same float32/int32 precision ultimately stored by USD; the base64
envelope only avoids converting every coordinate to a JSON decimal string.
"""

import base64
import math
import numpy as np

MAX_ARRAY_BYTES = 32*1024*1024
DTYPES = {'float32': np.dtype('<f4'), 'int32': np.dtype('<i4')}


def encode(values, dtype):
    if dtype not in DTYPES:
        raise ValueError('Unsupported render array dtype')
    incoming = np.asarray(values)
    if incoming.ndim not in (1, 2) or not np.isfinite(incoming).all():
        raise ValueError('Finite one/two-dimensional render array required')
    if dtype == 'int32' and (np.any(incoming != np.floor(incoming)) or
            np.any(incoming < -2**31) or np.any(incoming >= 2**31)):
        raise ValueError('Render indices must be exactly representable int32 values')
    with np.errstate(over='ignore'):
        array = np.asarray(incoming, dtype=DTYPES[dtype], order='C')
    if not np.isfinite(array).all() or array.nbytes > MAX_ARRAY_BYTES:
        raise ValueError('Invalid or oversized render array')
    return dict(dtype=dtype, shape=list(array.shape), data=base64.b64encode(array.tobytes()).decode('ascii'))


def decode(value, dtype, width=None):
    if not isinstance(value, dict) or set(value) != {'dtype', 'shape', 'data'} or value['dtype'] != dtype:
        raise ValueError('Render array schema/dtype mismatch')
    shape = value['shape']
    if not isinstance(shape, list) or len(shape) != (1 if width is None else 2) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError('Invalid render array shape')
    if width is not None and shape[1] != width:
        raise ValueError('Invalid render array vector width')
    size = math.prod(shape)*DTYPES[dtype].itemsize
    data = value['data']
    if size > MAX_ARRAY_BYTES or not isinstance(data, str) or len(data) != 4*((size+2)//3):
        raise ValueError('Invalid render array byte count')
    raw = base64.b64decode(data, validate=True)
    if len(raw) != size:
        raise ValueError('Truncated render array')
    array = np.frombuffer(raw, DTYPES[dtype]).reshape(shape)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite render geometry')
    return array


def mesh_arrays(payload):
    required = {'vertices', 'triangles', 'normals'}
    if not isinstance(payload, dict) or not required <= set(payload) or set(payload)-required-{'pink_molar'}:
        raise ValueError('Invalid render mesh fields')
    points = decode(payload['vertices'], 'float32', 3)
    triangles = decode(payload['triangles'], 'int32', 3)
    normals = decode(payload['normals'], 'float32', 3)
    if normals.shape != points.shape or np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise ValueError('Invalid render mesh topology/normals')
    pink = decode(payload['pink_molar'], 'float32') if 'pink_molar' in payload else None
    if pink is not None and (pink.shape != (len(points),) or np.any(pink < 0)):
        raise ValueError('Invalid optical concentration')
    return points, triangles, normals, pink
