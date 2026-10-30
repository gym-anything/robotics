"""Internal fields of the drillable concrete wall: the near- and far-face reinforcement mats (#4 bars at
300 mm each way, seeded offset) and the coarse aggregate (crushed-stone grains 4.75-19 mm, Fuller grading, 38 % by
volume), generated lazily per 50 mm cell from a seeded hash so that only the cells a bit enters are packed.

Frame: the wall body's frame with x along the wall, z up, and y into the wall from its face (depth = y)."""

import hashlib
import math
import numpy as np

from . import params as P

CELL = 0.05


class ConcreteFields:
    def __init__(self, seed, x_range, z_range, thickness=P.WALL_THICK_M, rebar_offset=(0., 0.)):
        self.seed = int(seed)
        self.thickness = float(thickness)
        s = P.REBAR_SPACING_M
        r = 0.5*P.REBAR_D_M
        ox, oz = rebar_offset
        # near mat: horizontal bars (along x) nearest the face, vertical bars behind them (depth = cover + r, + 2r)
        self.bars = []
        y_h = P.REBAR_COVER_M + r
        y_v = P.REBAR_COVER_M + 3*r
        for z in np.arange(z_range[0] + (oz % s), z_range[1], s):
            self.bars.append(dict(axis="x", y=y_h, c=float(z), r=r, mat="near"))
        for x in np.arange(x_range[0] + (ox % s), x_range[1], s):
            self.bars.append(dict(axis="z", y=y_v, c=float(x), r=r, mat="near"))
        y_far = self.thickness - P.FAR_MAT_COVER_M - r
        for z in np.arange(z_range[0] + ((oz + s/2) % s), z_range[1], s):
            self.bars.append(dict(axis="x", y=y_far, c=float(z), r=r, mat="far"))
        self._cells = {}

    # ---------------------------------------------------------------- reinforcement
    def bar_distance(self, p):
        """Signed distance (m) from point p (wall frame) to the nearest bar surface, and that bar."""
        best, which = 1e9, None
        for b in self.bars:
            if b["axis"] == "x":
                d = math.hypot(p[1] - b["y"], p[2] - b["c"]) - b["r"]
            else:
                d = math.hypot(p[1] - b["y"], p[0] - b["c"]) - b["r"]
            if d < best:
                best, which = d, b
        return best, which

    def _bar_arrays(self):
        """The bars as arrays (axis unit vectors, a point on each axis, radii), rebuilt when the list changes (a station
        may clear it after construction)."""
        key = (id(self.bars), len(self.bars))
        if getattr(self, "_bar_key", None) != key:
            u = np.array([(1., 0., 0.) if b["axis"] == "x" else (0., 0., 1.) for b in self.bars], float).reshape(-1, 3)
            c = np.array([(0., b["y"], b["c"]) if b["axis"] == "x" else (b["c"], b["y"], 0.) for b in self.bars],
                         float).reshape(-1, 3)
            r = np.array([b["r"] for b in self.bars], float)
            self._bar_np, self._bar_key = (u, c, r), key
        return self._bar_np

    def face_cut(self, pts, axis):
        """Steel the bit's face has cut. pts: (N, 3) points on the face (wall frame), axis: the hole axis (into the
        wall). For each point, the length of the line through it along the axis that lies inside a bar and behind the
        point (between where that line enters the bar and the face, at most the bar's chord), and whether the point
        itself is in steel. Returns (cut (N,) m, inside (N,) bool). The hole removed everything behind its face, so the
        largest cut over the face is the depth the hole has cut into the bar."""
        pts = np.asarray(pts, float).reshape(-1, 3)
        n = len(pts)
        u, c, r = self._bar_arrays()
        if not len(r) or not n:
            return np.zeros(n), np.zeros(n, bool)
        a = np.asarray(axis, float)/np.linalg.norm(axis)
        w = pts[:, None, :] - c[None, :, :]                                  # (N, B, 3)
        w_perp = w - np.einsum("nbk,bk->nb", w, u)[..., None]*u[None]
        a_perp = a[None, :] - (u @ a)[:, None]*u                              # (B, 3)
        A = np.einsum("bk,bk->b", a_perp, a_perp)                            # (B,)
        Bq = 2.*np.einsum("nbk,bk->nb", w_perp, a_perp)
        C = np.einsum("nbk,nbk->nb", w_perp, w_perp) - r[None, :]**2
        inside = (C < 0.).any(axis=1)
        disc = Bq*Bq - 4.*A[None, :]*C
        ok = (disc > 0.) & (A[None, :] > 1e-12)
        sq = np.sqrt(np.where(ok, disc, 0.))
        A2 = np.where(A > 1e-12, 2.*A, 1.)[None, :]
        t1, t2 = (-Bq - sq)/A2, (-Bq + sq)/A2                                # entry and exit along +axis
        cut = np.where(ok, np.clip(np.minimum(t2, 0.) - t1, 0., None), 0.)
        par = (A[None, :] <= 1e-12) & (C < 0.)                               # drilling along inside a bar
        cut = np.where(par, 1.0, cut)
        return cut.max(axis=1), inside

    # ---------------------------------------------------------------- aggregate
    def _cell(self, key):
        if key in self._cells:
            return self._cells[key]
        h = int.from_bytes(hashlib.sha256(f"agg/{self.seed}/{key}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(h)
        lo = np.asarray(key, float)*CELL
        grains = []
        target = P.AGGREGATE_FRACTION*CELL**3
        vol, tries = 0., 0
        dmin, dmax = 0.00475, P.AGGREGATE_DMAX_M
        while vol < target and tries < 4000:
            tries += 1
            # Fuller grading (n = 0.5): passing fraction (D/Dmax)^0.5 -> D = Dmax u^2, restricted to the coarse range
            u = rng.uniform((dmin/dmax)**0.5, 1.)
            dg = dmax*u*u
            c = lo + rng.uniform(0., CELL, 3)
            if any(np.linalg.norm(c - g[:3]) < 0.5*(dg + g[3]) for g in grains):
                continue
            grains.append(np.array([*c, dg]))
            vol += math.pi*dg**3/6.
        self._cells[key] = grains
        return grains

    def in_aggregate(self, p):
        key = tuple(int(math.floor(v/CELL)) for v in p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for g in self._cell((key[0] + dx, key[1] + dy, key[2] + dz)):
                        if np.linalg.norm(np.asarray(p) - g[:3]) < 0.5*g[3]:
                            return True
        return False

    def face_fractions(self, centre, axis, radius, n_ring=6):
        """Volume fractions of mortar / aggregate / steel under a bit face of `radius` at `centre` (wall frame),
        sampled at the centre and n_ring points at 0.7 R in the plane normal to `axis`."""
        a = np.asarray(axis, float)/np.linalg.norm(axis)
        t1 = np.cross(a, [0., 0., 1.]) if abs(a[2]) < .9 else np.cross(a, [1., 0., 0.])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(a, t1)
        pts = [np.asarray(centre, float)] + [np.asarray(centre) + 0.7*radius*(math.cos(k*2*math.pi/n_ring)*t1 +
                                                                        math.sin(k*2*math.pi/n_ring)*t2) for k in range(n_ring)]
        out = dict(mortar=0., aggregate=0., steel=0.)
        w = 1./len(pts)
        for p in pts:
            if self.bar_distance(p)[0] < 0.:
                out["steel"] += w
            elif self.in_aggregate(p):
                out["aggregate"] += w
            else:
                out["mortar"] += w
        return out
