"""Water surfaces: the body-attached cells that carry films, built from the same geometry as the colliders.

A *chart* is one connected surface of one body (or of the static room, body -1):
  * revolved: a cross-section polyline in the body's (r, z) half-plane, traversed with the material on its right
    (clockwise for a solid of revolution), revolved about the body z axis. Each profile segment is cut into rings of
    about `cell` arc length; each ring is cut into n_theta(r) ~ 2 pi r / cell cells (>= 6), so cells near the axis do not
    become slivers. Neighbouring rings with different n_theta share non-conformal edges (their theta intervals overlap).
    The cell normal is e_s x e_theta, pointing away from the material.
  * rect: a planar rectangle (origin, u, v, size) cut into nu x nv cells; normal u x v.
  * ellipsoid: an ellipsoid surface (centre, radii) as latitude rings x longitude cells; outward normal.
Every interior edge is listed once (c0, c1, length, n0, n1, kind, lip radius, midpoint; body frame): n0 is the unit
normal of the edge in c0's tangent plane pointing to c1, n1 the same direction continued in c1's tangent plane (equal on
flat charts, different across a profile turn, so a film rounding a rim keeps its speed instead of losing it). Chart
boundaries are typed: 'open' (water can leave: a drip line or free edge), 'wall' (no flux), 'lip' interior edges
across a sharp convex turn (Duez separation can eject there). Each cell has two triangles for ray hits (face -> cell).
A texture map (H x W -> cell) lets the renderer draw per-cell state on a UV-mapped mesh.
"""

import math
import numpy as np

TWO_PI = 2.0*math.pi


class Chart:
    def __init__(self, name, body, kind, role, material):
        self.name, self.body, self.kind, self.role, self.material = name, int(body), kind, role, material
        self.first = 0          # global index of the first cell (set by SurfaceSet.add)
        self.count = 0


class SurfaceSet:
    """All charts of a scene, assembled into flat arrays (numpy; uploaded by water.film)."""

    def __init__(self):
        self.charts = []
        self.pos, self.nrm, self.e1, self.e2, self.area, self.size = [], [], [], [], [], []
        self.chart_of, self.body_of = [], []
        self.edges = []          # (c0, c1, length, n0(3), n1(3), kind, lip_radius, mid(3))
        self.bedges = []         # (c, length, n(3), kind, mid(3))
        self.tri_v, self.tri_f, self.tri_cell = {}, {}, {}        # per body: vertices (list), faces, face -> cell
        self.tex = {}            # chart name -> (H x W) cell index map, and its mesh (v, f, uv)
        self.meshes = {}         # chart name -> dict(v, f, uv) for rendering
        self.n = 0

    # ------------------------------------------------------------------ helpers
    def _cells(self, chart, pos, nrm, e1, area, size):
        pos = np.asarray(pos, float); nrm = np.asarray(nrm, float); e1 = np.asarray(e1, float)
        e1 = e1 - (e1*nrm).sum(1, keepdims=True)*nrm
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(nrm, e1)
        k = len(pos)
        chart.first, chart.count = self.n, k
        self.pos.append(pos); self.nrm.append(nrm); self.e1.append(e1); self.e2.append(e2)
        self.area.append(np.asarray(area, float)); self.size.append(np.asarray(size, float))
        self.chart_of.append(np.full(k, len(self.charts), np.int32)); self.body_of.append(np.full(k, chart.body, np.int32))
        self.charts.append(chart)
        self.n += k
        return chart.first

    def _tris(self, body, quads, cells):
        """quads: (k, 4, 3) corner points (counter-clockwise seen from the normal side); cells: (k,) global cell ids."""
        V = self.tri_v.setdefault(body, [])
        F = self.tri_f.setdefault(body, [])
        C = self.tri_cell.setdefault(body, [])
        base = sum(len(x) for x in V)
        quads = np.asarray(quads, float)
        V.append(quads.reshape(-1, 3))
        idx = base + np.arange(len(quads))[:, None]*4
        F.append(np.concatenate([idx + np.array([0, 1, 2]), idx + np.array([0, 2, 3])], axis=0))
        C.append(np.concatenate([cells, cells]))

    # ------------------------------------------------------------------ rect
    def add_rect(self, name, body, origin, u, v, size, cell, role="target", material="steel", open_edges="",
                 wall_edges=None):
        """Planar rectangle; normal u x v. open_edges: subset of 'u0 u1 v0 v1' where water leaves (others: walls)."""
        o, u, v = np.asarray(origin, float), np.asarray(u, float), np.asarray(v, float)
        u, v = u/np.linalg.norm(u), v/np.linalg.norm(v)
        n = np.cross(u, v)
        Lu, Lv = float(size[0]), float(size[1])
        nu, nv = max(1, int(round(Lu/cell))), max(1, int(round(Lv/cell)))
        du, dv = Lu/nu, Lv/nv
        ch = Chart(name, body, "rect", role, material)
        ch.nu, ch.nv, ch.du, ch.dv, ch.origin, ch.u, ch.v, ch.normal, ch.size_m = nu, nv, du, dv, o, u, v, n, (Lu, Lv)
        ii, jj = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        pos = o + np.outer((ii + .5)*du, u) + np.outer((jj + .5)*dv, v)
        k = len(pos)
        first = self._cells(ch, pos, np.tile(n, (k, 1)), np.tile(u, (k, 1)), np.full(k, du*dv), np.full(k, min(du, dv)))
        gid = lambda i, j: first + i*nv + j
        for i in range(nu):
            for j in range(nv):
                if i + 1 < nu:
                    self.edges.append((gid(i, j), gid(i + 1, j), dv, u, u, 0, 0.0, o + (i + 1)*du*u + (j + .5)*dv*v))
                if j + 1 < nv:
                    self.edges.append((gid(i, j), gid(i, j + 1), du, v, v, 0, 0.0, o + (i + .5)*du*u + (j + 1)*dv*v))
        for side, (rng, nn, L) in dict(u0=(((0, j) for j in range(nv)), -u, dv), u1=(((nu - 1, j) for j in range(nv)), u, dv),
                                        v0=(((i, 0) for i in range(nu)), -v, du), v1=(((i, nv - 1) for i in range(nu)), v, du)).items():
            kind = 2 if side in open_edges.split() else 3
            for i, j in rng:
                mid = o + (i + .5)*du*u + (j + .5)*dv*v + 0.5*(du if side[0] == "u" else dv)*nn
                self.bedges.append((gid(i, j), L, nn, kind, mid))
        corners = np.stack([o + np.outer(ii*du, u) + np.outer(jj*dv, v), o + np.outer((ii + 1)*du, u) + np.outer(jj*dv, v),
                            o + np.outer((ii + 1)*du, u) + np.outer((jj + 1)*dv, v), o + np.outer(ii*du, u) + np.outer((jj + 1)*dv, v)], 1)
        self._tris(body, corners, first + np.arange(k))
        tex = (first + np.arange(k)).reshape(nu, nv).T[::-1]          # image rows: v from top, columns: u
        ch.tex = np.ascontiguousarray(tex)
        uv = np.stack([np.stack([ii/nu, jj/nv], 1), np.stack([(ii + 1)/nu, jj/nv], 1), np.stack([(ii + 1)/nu, (jj + 1)/nv], 1),
                       np.stack([ii/nu, (jj + 1)/nv], 1)], 1)
        self.meshes[name] = dict(quads=corners, uv=uv, body=body)
        return ch

    # ------------------------------------------------------------------ revolved
    def add_revolved(self, name, body, profile, cell, role="target", material="glaze", lips=(), open_end=False,
                     local=None, min_theta=6, max_theta=256):
        """profile: [(r, z), ...] traversed with the material on its right; first and last points on the axis (r = 0)
        close the surface, otherwise the ends are open (water leaves) or walls. lips: profile vertex indices that are
        sharp convex turns (rims) with their edge radius: [(index, radius_m)]. local: optional (p, R) placing the
        revolved frame in the body frame."""
        prof = np.asarray(profile, float)
        p0, R0 = (np.zeros(3), np.eye(3)) if local is None else (np.asarray(local[0], float), np.asarray(local[1], float))
        ch = Chart(name, body, "revolved", role, material)
        rings = []           # (s0, s1, r0, z0, r1, z1, segment index)
        for k in range(len(prof) - 1):
            (ra, za), (rb, zb) = prof[k], prof[k + 1]
            L = math.hypot(rb - ra, zb - za)
            if L < 1e-9:
                continue
            m = max(1, int(round(L/cell)))
            for i in range(m):
                t0, t1 = i/m, (i + 1)/m
                rings.append((ra + (rb - ra)*t0, za + (zb - za)*t0, ra + (rb - ra)*t1, za + (zb - za)*t1, k))
        lipset = dict(lips)
        nth = []
        for (r0, z0, r1, z1, k) in rings:
            rm = max(0.5*(r0 + r1), 1e-6)
            nth.append(int(np.clip(round(TWO_PI*rm/cell), min_theta, max_theta)))
        # cells
        pos, nrm, e1, area, size, quads, uvq = [], [], [], [], [], [], []
        ring_first = []
        H = len(rings)
        W = max(nth)
        for kr, ((r0, z0, r1, z1, kseg), n) in enumerate(zip(rings, nth)):
            ring_first.append(len(pos))
            ds = math.hypot(r1 - r0, z1 - z0)
            es = np.array([r1 - r0, z1 - z0])/ds                  # (dr, dz) along the profile
            dth = TWO_PI/n
            rm, zm = 0.5*(r0 + r1), 0.5*(z0 + z1)
            for m in range(n):
                th = (m + .5)*dth
                c, s = math.cos(th), math.sin(th)
                rhat, that = np.array([c, s, 0.]), np.array([-s, c, 0.])
                e_s = es[0]*rhat + np.array([0., 0., es[1]])
                nn = np.cross(e_s, that)
                pos.append(rm*rhat + np.array([0., 0., zm])); nrm.append(nn); e1.append(e_s)
                area.append(max(dth*0.5*(r0 + r1)*ds, 1e-10))
                size.append(max(min(ds, dth*max(rm, 1e-4)), 2e-4))
                pts = []
                for (r_, z_, a) in ((r0, z0, m*dth), (r1, z1, m*dth), (r1, z1, (m + 1)*dth), (r0, z0, (m + 1)*dth)):
                    pts.append(np.array([r_*math.cos(a), r_*math.sin(a), z_]))
                # counter-clockwise seen from the normal n = e_s x e_th: (s0,th0) (s1,th0) (s1,th1) (s0,th1)
                quads.append([pts[0], pts[1], pts[2], pts[3]])
                uvq.append([(m/n, kr/H), ((m + 1)/n, kr/H), ((m + 1)/n, (kr + 1)/H), (m/n, (kr + 1)/H)])
        pos, nrm, e1 = np.array(pos), np.array(nrm), np.array(e1)
        quads = np.array(quads)
        ch.cell_seg = np.concatenate([np.full(n, r[4], np.int32) for r, n in zip(rings, nth)])   # profile segment per cell
        pos = pos @ R0.T + p0; nrm = nrm @ R0.T; e1 = e1 @ R0.T
        quads = quads @ R0.T + p0
        first = self._cells(ch, pos, nrm, e1, area, size)
        rf = [first + x for x in ring_first]
        # edges around each ring
        for kr, ((r0, z0, r1, z1, kseg), n) in enumerate(zip(rings, nth)):
            ds = math.hypot(r1 - r0, z1 - z0)
            dth = TWO_PI/n
            if n < 2:
                continue
            rm, zm = 0.5*(r0 + r1), 0.5*(z0 + z1)
            for m in range(n):
                a = (m + 1)*dth
                # the face's own normal, e_theta at the face angle (finish() projects it into each cell's tangent
                # plane): with each cell's centre tangent instead, the Green-Gauss sum of a constant depth over a wedge
                # cell near the axis is not zero and a puddle at rest in a plate's well runs at 0.4 m/s
                tf = np.array([-math.sin(a), math.cos(a), 0.]) @ R0.T
                mid = np.array([rm*math.cos(a), rm*math.sin(a), zm]) @ R0.T + p0
                self.edges.append((rf[kr] + m, rf[kr] + (m + 1) % n, ds, tf, tf, 0, 0.0, mid))
        # edges between consecutive rings (theta-interval overlaps)
        for kr in range(H - 1):
            (ra0, za0, ra1, za1, ka), (rb0, zb0, rb1, zb1, kb) = rings[kr], rings[kr + 1]
            na, nb = nth[kr], nth[kr + 1]
            rbnd = ra1
            # profile tangent at the shared boundary, averaged between the two rings (direction from ring kr to kr + 1)
            ta = np.array([ra1 - ra0, za1 - za0]); ta /= np.linalg.norm(ta)
            tb = np.array([rb1 - rb0, zb1 - zb0]); tb /= np.linalg.norm(tb)
            tav = ta + tb
            if np.linalg.norm(tav) < 1e-6:
                tav = ta
            tav /= np.linalg.norm(tav)
            lip_r = lipset.get(kb, 0.0) if kb != ka else 0.0          # vertex kb starts segment kb
            kind = 1 if lip_r > 0 else 0
            ia = ib = 0
            a0 = b0 = 0.0
            # merge theta breakpoints
            ba = [m*TWO_PI/na for m in range(na + 1)]
            bb = [m*TWO_PI/nb for m in range(nb + 1)]
            pts = sorted(set([round(x, 12) for x in ba + bb]))
            for t0, t1 in zip(pts[:-1], pts[1:]):
                if t1 - t0 < 1e-9:
                    continue
                tm = 0.5*(t0 + t1)
                ma, mb = min(int(tm/(TWO_PI/na)), na - 1), min(int(tm/(TWO_PI/nb)), nb - 1)
                L = max(rbnd, 1e-6)*(t1 - t0)
                c, s = math.cos(tm), math.sin(tm)
                n0 = (ta[0]*np.array([c, s, 0.]) + np.array([0., 0., ta[1]])) @ R0.T
                n1 = (tb[0]*np.array([c, s, 0.]) + np.array([0., 0., tb[1]])) @ R0.T
                mid = np.array([rbnd*c, rbnd*s, za1]) @ R0.T + p0
                self.edges.append((rf[kr] + ma, rf[kr + 1] + mb, L, n0, n1, kind, lip_r, mid))
        # ends: on the axis (closed) or open boundaries
        for end, kr in (("start", 0), ("end", H - 1)):
            r_end = rings[kr][0] if end == "start" else rings[kr][2]
            if r_end < 1e-6:
                continue
            (r0, z0, r1, z1, k) = rings[kr]
            t = np.array([r1 - r0, z1 - z0]); t /= np.linalg.norm(t)
            if end == "start":
                t = -t
            n = nth[kr]
            z_end = z0 if end == "start" else z1
            for m in range(n):
                tm = (m + .5)*TWO_PI/n
                c, s = math.cos(tm), math.sin(tm)
                nn = (t[0]*np.array([c, s, 0.]) + np.array([0., 0., t[1]])) @ R0.T
                mid = np.array([r_end*c, r_end*s, z_end]) @ R0.T + p0
                self.bedges.append((rf[kr] + m, r_end*TWO_PI/n, nn, 2 if open_end else 3, mid))
        self._tris(body, quads, first + np.arange(len(pos)))
        # texture: H rows (rings, first ring at the image bottom) x W columns (theta), each ring stretched over W
        tex = np.zeros((H, W), np.int32)
        for kr, n in enumerate(nth):
            cols = np.minimum((np.arange(W) + .5)*n/W, n - 1).astype(int)
            tex[H - 1 - kr] = rf[kr] + cols
        ch.tex = tex
        ch.rings, ch.nth = rings, nth
        self.meshes[name] = dict(quads=quads, uv=np.array(uvq), body=body)
        return ch

    # ------------------------------------------------------------------ ellipsoid (produce)
    def add_ellipsoid(self, name, body, centre, radii, cell, role="target", material="skin"):
        """Ellipsoid of radii (a, a, c) about the body z axis, built as a revolved profile of its meridian."""
        a, b_, c = radii
        if abs(a - b_) > 1e-9:
            raise ValueError("ellipsoid charts need radii (a, a, c)")
        n = max(8, int(round(math.pi*0.5*(a + c)/cell)))
        th = np.linspace(-0.5*math.pi, 0.5*math.pi, n + 1)
        # from the top pole down the outside to the bottom pole: material on the right (clockwise)
        prof = [(a*math.cos(t), c*math.sin(t)) for t in th[::-1]]
        prof[0], prof[-1] = (0.0, prof[0][1]), (0.0, prof[-1][1])
        return self.add_revolved(name, body, prof, cell, role=role, material=material,
                                 local=(np.asarray(centre, float), np.eye(3)))

    # ------------------------------------------------------------------ assembly
    def finish(self):
        cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape)
        self.pos = cat(self.pos, (0, 3)); self.nrm = cat(self.nrm, (0, 3))
        self.e1 = cat(self.e1, (0, 3)); self.e2 = cat(self.e2, (0, 3))
        self.area = cat(self.area, (0,)); self.size = cat(self.size, (0,))
        self.chart_of = cat(self.chart_of, (0,)).astype(np.int32); self.body_of = cat(self.body_of, (0,)).astype(np.int32)
        E = self.edges
        self.e_c0 = np.array([e[0] for e in E], np.int32); self.e_c1 = np.array([e[1] for e in E], np.int32)
        self.e_len = np.array([e[2] for e in E], float)
        self.e_n0 = np.array([e[3] for e in E], float).reshape(-1, 3); self.e_n1 = np.array([e[4] for e in E], float).reshape(-1, 3)
        self.e_kind = np.array([e[5] for e in E], np.int32); self.e_lip = np.array([e[6] for e in E], float)
        self.e_mid = np.array([e[7] for e in E], float).reshape(-1, 3)
        if len(E):
            # edge normals in each cell's own tangent plane (projected, unit)
            for nn, cc in ((self.e_n0, self.e_c0), (self.e_n1, self.e_c1)):
                nc = self.nrm[cc]
                nn -= (nn*nc).sum(1, keepdims=True)*nc
                nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        # centre-to-centre distance across each edge (through the edge midpoint, so it follows a profile turn)
        self.e_dist = (np.linalg.norm(self.pos[self.e_c0] - self.e_mid, axis=1) + np.linalg.norm(self.pos[self.e_c1] - self.e_mid, axis=1)
                       if len(E) else np.zeros(0))
        B = self.bedges
        self.b_c = np.array([e[0] for e in B], np.int32); self.b_len = np.array([e[1] for e in B], float)
        self.b_n = np.array([e[2] for e in B], float).reshape(-1, 3); self.b_kind = np.array([e[3] for e in B], np.int32)
        self.b_mid = np.array([e[4] for e in B], float).reshape(-1, 3)
        mats = sorted({c.material for c in self.charts})
        self.materials = mats
        self.mat_of = np.array([mats.index(self.charts[k].material) for k in self.chart_of], np.int32) if self.n else \
            np.zeros(0, np.int32)
        self.role_of = [self.charts[k].role for k in self.chart_of]
        for b in list(self.tri_v):
            self.tri_v[b] = np.concatenate(self.tri_v[b]); self.tri_f[b] = np.concatenate(self.tri_f[b]).astype(np.int32)
            self.tri_cell[b] = np.concatenate(self.tri_cell[b]).astype(np.int32)
        # cell -> edges adjacency (CSR) for the impact spreading kernel
        deg = np.zeros(self.n, np.int64)
        np.add.at(deg, self.e_c0, 1); np.add.at(deg, self.e_c1, 1)
        self.adj_start = np.concatenate([[0], np.cumsum(deg)]).astype(np.int32)
        fill = self.adj_start[:-1].copy()
        self.adj_edge = np.zeros(int(deg.sum()), np.int32)
        for e, (a, b) in enumerate(zip(self.e_c0, self.e_c1)):
            self.adj_edge[fill[a]] = e; fill[a] += 1
            self.adj_edge[fill[b]] = e; fill[b] += 1
        self.chart_by_name = {c.name: c for c in self.charts}
        return self

    def check(self):
        """Geometry invariants: positive areas, unit normals, every interior edge joins two cells of one body."""
        assert np.all(self.area > 0) and np.all(np.isfinite(self.pos))
        assert np.allclose(np.linalg.norm(self.nrm, axis=1), 1, atol=1e-9)
        assert np.all(self.body_of[self.e_c0] == self.body_of[self.e_c1])
        assert np.allclose(np.linalg.norm(self.e_n0, axis=1), 1, atol=1e-6)
        assert np.allclose(np.linalg.norm(self.e_n1, axis=1), 1, atol=1e-6)
        # each edge normal lies in its cell's tangent plane
        assert np.abs((self.e_n0*self.nrm[self.e_c0]).sum(1)).max() < 1e-6
        assert np.abs((self.e_n1*self.nrm[self.e_c1]).sum(1)).max() < 1e-6
        return dict(cells=int(self.n), edges=int(len(self.e_c0)), boundary=int(len(self.b_c)), charts=len(self.charts),
                    area_m2=float(self.area.sum()))
