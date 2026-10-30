"""Small fast geometry helpers (quaternions are xyzw, as Newton stores them)."""

import numpy as np


def quat_to_R(q):
    """(n, 4) xyzw -> (n, 3, 3)."""
    q = np.asarray(q, float)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (yy + zz); R[:, 0, 1] = 2 * (xy - wz); R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz); R[:, 1, 1] = 1 - 2 * (xx + zz); R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy); R[:, 2, 1] = 2 * (yz + wx); R[:, 2, 2] = 1 - 2 * (xx + yy)
    return R


def rotate(q, v):
    """Rotate vectors v (n, 3) by quaternions q (n, 4) xyzw."""
    q = np.asarray(q, float); v = np.asarray(v, float)
    u, w = q[:, :3], q[:, 3:4]
    t = 2.0 * cross(u, v)
    return v + w * t + cross(u, t)


def cross(a, b):
    """np.cross for (..., 3) arrays without numpy's axis bookkeeping (5-10x faster on small arrays)."""
    a = np.asarray(a); b = np.asarray(b)
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0), axis=-1)


def cross3(a, b):
    """Cross product of two length-3 sequences as a numpy array (scalar arithmetic)."""
    return np.array((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))
