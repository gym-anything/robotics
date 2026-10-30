"""Analytic contact geometry for the paint applicators (roller nap samples, brush tuft tips, spray parcels, drips).

The MuJoCo world resolves rigid contact of every collider; the applicators' compliant layers (nap, bristles) and the
liquid need distances to the same surfaces. Each primitive mirrors a collider of the scene exactly (same centre, half
extents, body) and names the paint charts that lie on its faces and the sink that receives paint elsewhere on it.
Oriented-box signed distance (negative inside) with its outward normal; planes; plates with apertures (the stencil:
a tip above an aperture sees the substrate beneath) and palette wells.
"""

import numpy as np

from .film import _quat_rot


class Box:
    def __init__(self, name, body, centre, quat, half, *, charts=(), sink=None, holes=None):
        self.name, self.body = name, int(body)
        self.c = np.asarray(centre, float)
        self.Rl = _quat_rot(quat)
        self.h = np.asarray(half, float)
        self.charts = tuple(charts)
        self.sink = sink or name
        self.holes = holes                  # callable(local xy) -> bool: points over an aperture pass through
        self.radius = float(np.linalg.norm(self.h))

    def frame(self, body_q):
        if self.body < 0:
            return self.c, self.Rl
        q = body_q[self.body]
        Rb = _quat_rot(q[3:7])
        return np.asarray(q[:3], float) + Rb @ self.c, Rb @ self.Rl

    def sdf(self, pts, body_q):
        """(distance, outward normal (world)) for world points (N, 3)."""
        c, R = self.frame(body_q)
        q = (pts - c) @ R
        d = np.abs(q) - self.h
        outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
        inside = np.minimum(d.max(axis=1), 0.0)
        dist = outside + inside
        # normal: outside -> direction of the clamped offset; inside -> the face of least penetration
        g = np.where(d > 0, np.sign(q)*np.maximum(d, 0.0), 0.0)
        gn = np.linalg.norm(g, axis=1)
        k = np.argmax(d, axis=1)
        nin = np.zeros_like(q)
        nin[np.arange(len(q)), k] = np.sign(q[np.arange(len(q)), k])
        nl = np.where((gn > 1e-12)[:, None], g/np.maximum(gn, 1e-12)[:, None], nin)
        if self.holes is not None:
            over = self.holes(q[:, :2]) & (np.abs(q[:, 2]) < self.h[2] + 0.02)
            dist = np.where(over, 1.0, dist)
        return dist, nl @ R.T


class Plane:
    def __init__(self, name, point, normal, *, charts=(), sink=None):
        self.name, self.body = name, -1
        self.p = np.asarray(point, float)
        self.n = np.asarray(normal, float)/np.linalg.norm(normal)
        self.charts, self.sink = tuple(charts), sink or name
        self.radius = np.inf

    def frame(self, body_q):
        return self.p, np.eye(3)

    def sdf(self, pts, body_q):
        d = (pts - self.p) @ self.n
        return d, np.broadcast_to(self.n, pts.shape)


class Cylinder:
    """Upright (local z) solid cylinder."""

    def __init__(self, name, body, centre, quat, radius, half_height, *, charts=(), sink=None):
        self.name, self.body = name, int(body)
        self.c, self.Rl = np.asarray(centre, float), _quat_rot(quat)
        self.r, self.hh = float(radius), float(half_height)
        self.charts, self.sink = tuple(charts), sink or name
        self.radius = float(np.hypot(radius, half_height))

    frame = Box.frame

    def sdf(self, pts, body_q):
        c, R = self.frame(body_q)
        q = (pts - c) @ R
        rr = np.hypot(q[:, 0], q[:, 1])
        dr, dz = rr - self.r, np.abs(q[:, 2]) - self.hh
        outside = np.hypot(np.maximum(dr, 0), np.maximum(dz, 0))
        dist = outside + np.minimum(np.maximum(dr, dz), 0.0)
        radial = np.stack([q[:, 0], q[:, 1], np.zeros(len(q))], 1)/np.maximum(rr, 1e-12)[:, None]
        axial = np.stack([np.zeros(len(q)), np.zeros(len(q)), np.sign(q[:, 2])], 1)
        use_r = (dr > dz)
        nl = np.where(use_r[:, None], radial, axial)
        both = (dr > 0) & (dz > 0)
        if np.any(both):
            v = radial[both]*dr[both][:, None] + axial[both]*dz[both][:, None]
            nl[both] = v/np.linalg.norm(v, axis=1, keepdims=True)
        return dist, nl @ R.T


def nearest(prims, pts, body_q, centre=None, radius=None):
    """Signed distance, normal and primitive index of the closest surface for each point (over the candidate
    primitives whose bounding sphere reaches the query sphere)."""
    n = len(pts)
    best = np.full(n, np.inf); normal = np.zeros((n, 3)); idx = np.full(n, -1)
    for k, pr in enumerate(prims):
        if centre is not None and np.isfinite(pr.radius):
            c, _ = pr.frame(body_q)
            if np.linalg.norm(c - centre) > pr.radius + radius:
                continue
        d, nn = pr.sdf(pts, body_q)
        better = d < best
        best = np.where(better, d, best)
        normal[better] = nn[better]
        idx[better] = k
    return best, normal, idx


def candidates(prims, body_q, centre, radius):
    """Primitives whose surface comes within `radius` of `centre` (exact signed distance after a bounding-sphere
    pre-test; big flat boxes such as a table top would pass any sphere test)."""
    out = []
    c0 = np.asarray(centre, float)[None]
    for pr in prims:
        if np.isfinite(pr.radius):
            c, _ = pr.frame(body_q)
            if np.linalg.norm(c - c0[0]) > pr.radius + radius:
                continue
        holes = getattr(pr, "holes", None)
        if holes is not None:
            pr.holes = None
        d, _ = pr.sdf(c0, body_q)
        if holes is not None:
            pr.holes = holes
        if d[0] < radius:
            out.append(pr)
    return out
