"""Bulk paint in containers: a quasi-static pool whose free surface is the horizontal plane enclosing its
volume inside the container's cavity (cavity sampled once as volume-weighted points in the container frame; the
level-volume table is rebuilt only when the container's tilt changes by more than 0.5 deg). The liquid's mass and
centroid are lumped into the container body. When the level rises above the lowest rim point the excess leaves over
the rim as a weir, Q = C_w (2/3) sqrt(2 g) b H^1.5 (C_w 0.6, b the wetted rim length, estimate), handed to the
caller as ballistic paint (painting.drips) so a knocked tray makes a real puddle.
"""

import math
import numpy as np

from . import params as P
from .film import _quat_rot


class Reservoir:
    def __init__(self, name, body, paints, points, weights, rim, *, solids=None, vehicle=0.0, wall_fn=None):
        self.name, self.body, self.paints = name, int(body), list(paints)
        self.points = np.asarray(points, float)
        self.weights = np.asarray(weights, float)
        self.capacity = float(self.weights.sum())
        self.rim = np.asarray(rim, float)                # rim polyline points (container frame)
        self.solids = np.zeros(len(self.paints)) if solids is None else np.asarray(solids, float).copy()
        self.vehicle = float(vehicle)
        self.wall_fn = wall_fn
        self._key, self._table = None, None
        self.spilled = np.zeros(len(self.paints) + 1)
        self.version = 0

    @property
    def volume(self):
        return float(self.solids.sum() + self.vehicle)

    def mass(self):
        """kg: solids at each paint's solids density (as supplied, rho = phi0 rho_s + (1 - phi0) rho_w) + vehicle."""
        return float(sum(s*p.rho_solids for s, p in zip(self.solids, self.paints)) + self.vehicle*P.WATER_RHO)

    def pose(self, body_q):
        q = body_q[self.body]
        return np.asarray(q[:3], float), _quat_rot(q[3:7])

    def _level_table(self, R):
        up = R.T @ np.array([0., 0., 1.])
        key = tuple(np.round(up, 3))                    # ~0.06 deg resolution; rebuild beyond 0.5 deg change
        if self._key is not None and np.dot(np.asarray(self._key), up) > math.cos(math.radians(0.5)):
            return self._table
        hgt = self.points @ up
        order = np.argsort(hgt)
        self._table = (hgt[order], np.cumsum(self.weights[order]), up, (self.rim @ up).min(), order)
        self._key = key
        return self._table

    def level(self, body_q):
        """Free-surface height (container frame, along 'up') and the up vector in the container frame."""
        p, R = self.pose(body_q)
        hgt, cum, up, rim_min, _ = self._level_table(R)
        V = self.volume
        if V <= 0:
            return hgt[0] - 1e-6, up, rim_min
        k = int(np.searchsorted(cum, V))
        lvl = hgt[min(k, len(hgt) - 1)] if k < len(hgt) else hgt[-1] + (V - cum[-1])/max(1e-6, self._top_area())
        return float(lvl), up, rim_min

    def _top_area(self):
        return 0.05

    def immersed(self, world_pts, body_q):
        """Boolean per point: inside the cavity's horizontal footprint and below the free surface. Returns (mask, depth)."""
        p, R = self.pose(body_q)
        loc = (np.asarray(world_pts, float) - p) @ R
        lvl, up, _ = self.level(body_q)
        inside = self.wall_fn(loc) if self.wall_fn is not None else np.ones(len(loc), bool)
        depth = lvl - loc @ up
        return inside & (depth > 0) & (self.volume > 0), depth

    def centroid_local(self, body_q):
        p, R = self.pose(body_q)
        hgt, cum, up, _, order = self._level_table(R)
        V = self.volume
        if V <= 0:
            return np.zeros(3)
        k = max(1, int(np.searchsorted(cum, V)))
        idx = order[:k]
        w = self.weights[idx]
        return (self.points[idx]*w[:, None]).sum(axis=0)/w.sum()

    def take(self, volume):
        """Remove up to `volume` m^3 with the pool's composition; returns (solids per paint, vehicle) removed."""
        V = self.volume
        if V <= 0 or volume <= 0:
            return np.zeros(len(self.paints)), 0.0
        f = min(1.0, volume/V)
        s, w = self.solids*f, self.vehicle*f
        self.solids = self.solids - s
        self.vehicle -= w
        self.version += 1
        return s, w

    def add(self, solids, vehicle):
        solids = np.asarray(solids, float)
        if np.any(solids < -1e-18) or vehicle < -1e-18:
            raise FloatingPointError(f"{self.name}: negative addition")
        self.solids = self.solids + solids
        self.vehicle += float(vehicle)
        self.version += 1

    def spill_rate(self, body_q):
        """Volume flow over the rim (m^3/s) and the world point and outward direction it leaves from."""
        lvl, up, rim_min = self.level(body_q)
        H = lvl - rim_min
        if H <= 0 or self.volume <= 0:
            return 0.0, None, None
        p, R = self.pose(body_q)
        hr = self.rim @ up
        wet = self.rim[hr < lvl]
        b = max(0.005, float(np.linalg.norm(np.diff(wet, axis=0), axis=1).sum()) if len(wet) > 1 else 0.01)
        Q = 0.6*(2/3)*math.sqrt(2*P.GRAVITY)*b*H**1.5
        lowest = self.rim[int(np.argmin(hr))]
        centre = self.points.mean(axis=0)
        out = lowest - centre
        out -= (out @ up)*up
        out /= max(1e-9, np.linalg.norm(out))
        return Q, p + R @ lowest, R @ out

    def snapshot(self):
        return dict(solids=self.solids.copy(), vehicle=self.vehicle, spilled=self.spilled.copy())

    def restore(self, s):
        self.solids, self.vehicle, self.spilled = s["solids"].copy(), s["vehicle"], s["spilled"].copy()
        self.version += 1


# ------------------------------------------------------------------ cavity samplers
def box_cavity(x0, x1, y0, y1, floor_fn, z1, step=0.004):
    xs = np.arange(x0 + step/2, x1, step); ys = np.arange(y0 + step/2, y1, step)
    pts, w = [], []
    for x in xs:
        zf = floor_fn(x)
        zs = np.arange(zf + step/2, z1, step)
        if len(zs) == 0:
            continue
        X, Yg, Zg = np.meshgrid([x], ys, zs, indexing="ij")
        pts.append(np.stack([X.ravel(), Yg.ravel(), Zg.ravel()], 1))
    pts = np.concatenate(pts)
    return pts, np.full(len(pts), step**3)


def cyl_cavity(r0, r1, z0, z1, step=0.004):
    zs = np.arange(z0 + step/2, z1, step)
    pts = []
    for z in zs:
        r = r0 + (r1 - r0)*(z - z0)/(z1 - z0)
        g = np.arange(-r + step/2, r, step)
        X, Yg = np.meshgrid(g, g, indexing="ij")
        keep = X**2 + Yg**2 <= r*r
        pts.append(np.stack([X[keep], Yg[keep], np.full(keep.sum(), z)], 1))
    pts = np.concatenate(pts)
    return pts, np.full(len(pts), step**3)


def ring(r, z, n=48):
    a = np.linspace(0, 2*math.pi, n, endpoint=False)
    return np.stack([r*np.cos(a), r*np.sin(a), np.full(n, z)], 1)
