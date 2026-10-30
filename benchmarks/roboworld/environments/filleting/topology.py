"""Host-side topology change: the swept edge surface recorded by the knife-edge law is inserted into
the tetrahedral mesh.

The world uses mode 'surface' (Cut.mode): a mesh face is cut if and only if the segment joining its two tets'
centroids crosses the recorded swept surface (one global rule, no per-tet plane decisions), except where a tet's
centroid lies on that surface (within a quarter edge length): there the two tets' sides of the surface decide, each
tet keeping one side once a face of it is cut; the node stars on the cut are split into fans, crumbs
are re-attached by signed distance, and thin layers pressed against a support by the edge are severed through. The split variant described below (and mode 'assign') are kept for comparison; the
paragraphs "Split rule" onward describe that variant, not the world's.

Crack record. Every `record_every` ticks the rest-space anchor position of every cutting edge sample is
snapshotted; for adjacent anchored samples k, k+1 the quad (A_k(t0), A_k+1(t0), A_k+1(t1), A_k(t1)) is
a piece of the swept surface. Its normal is t x n = the blade normal, so all pieces of one cut share an
orientation. Pieces are binned into the tets containing their corners (rest space).

Split rule. A tet is split when the recorded surface covers at least `coverage` of its cross-section by
the fitted plane AND no active crack tip (anchored sample) lies inside it: the edge has passed through.
Vertices are classified against a per-vertex averaged plane (consistent between neighbouring tets);
a vertex within 20 % of the local edge length of the plane lies ON the cut and is duplicated rather than
creating a sliver. Cut edges receive a pair of new nodes (one per side), stored in a persistent edge map
so a neighbour split later reuses them (conforming). Each side's convex cell is tetrahedralized.
Mass is exact: every new tet carries its own rest volume.
"""

import numpy as np
from scipy.spatial import Delaunay, cKDTree


def tet_volume(P):
    return np.linalg.det(np.stack([P[..., 1, :] - P[..., 0, :], P[..., 2, :] - P[..., 0, :], P[..., 3, :] - P[..., 0, :]], -1))/6.


def plane_section_area(P, n, p0):
    """Area of the intersection of the plane (n, p0) with the tet with vertices P (4x3)."""
    d = (P - p0) @ n
    pts = []
    for a in range(4):
        if abs(d[a]) < 1e-12:
            pts.append(P[a])
        for b in range(a + 1, 4):
            if d[a]*d[b] < 0:
                f = d[a]/(d[a] - d[b])
                pts.append(P[a] + f*(P[b] - P[a]))
    if len(pts) < 3:
        return 0.
    pts = np.array(pts)
    c = pts.mean(0)
    u = pts[0] - c
    u /= max(np.linalg.norm(u), 1e-15)
    w = np.cross(n, u)
    ang = np.arctan2((pts - c) @ w, (pts - c) @ u)
    pts = pts[np.argsort(ang)]
    area = 0.
    for i in range(1, len(pts) - 1):
        area += .5*np.linalg.norm(np.cross(pts[i] - pts[0], pts[i + 1] - pts[0]))
    return area


class CrackRecorder:
    """Accumulates the swept surface per tet (rest space)."""

    def __init__(self, tissue, record_every=5, coverage=.5, bridge_m=2.5e-3):
        self.ts = tissue
        self.record_every = int(record_every)
        self.coverage = coverage
        self.bridge_m = bridge_m
        self.use_closure = False
        self.sever_J = 1.
        self.sever_gap = .6e-3          # m: edge this close to a support crushes the remaining layer through
        self.supports = []              # shape indices of rigid supports (board, bench)
        self.sever_into = None          # m below the support's top the window reaches (None: through the support)
        self.blades = []                # shape indices of the blade segments (kerf_faces diagnostic)
        self.kerf_behind_m = .5e-3      # m: kerf_faces counts faces this far behind the edge
        self.snap_frac = .3             # on-cut tolerance / snap radius as a fraction of the local edge length
        self.mode = "surface"           # 'surface': faces whose centroid segment crosses the swept surface are cut
        self.seal_m = .25e-3            # swept-surface gap tolerance for the crossing test
        self.on_sheet_frac = .25        # a tet whose centroid lies within this fraction of h0 of the sheet is 'on' it
        self.tet_side = {}              # tet -> (side +-1, sheet normal) locked when a face decided by sides is cut
        self.side_cut = 0               # faces cut by the side rule (reported)
        self.tie_twins = False          # predictive mode only (twins released by the passing blade)
        self.prev = None            # (a_on, a_rest) at the last snapshot
        self.acc = {}               # tet -> [sum area*n (3), sum area*centroid (3), sum area]
        self.surface = []           # list of rest-space triangles (for audit / rendering fallback)
        self.edge_map = {}          # (a, b) sorted node pair -> {a: node on a's side, b: node on b's side}
        self.twin = {}              # duplicated node -> its partner on the other side of the cut
        self.tet_plane = {}         # tet -> (n, p) of the cut that created it (inherited by children)
        self.crumbs_merged = 0      # fragments below the minimum volume re-attached (reported)
        self.min_fragment_m3 = 30e-9
        self.events = []            # topology event log
        self.tick = 0
        self._tree = None
        self.orig = tissue.htets[:tissue.nt].copy()      # original connectivity (assign mode: ids are permanent)
        self._nbr = None

    def _locator(self):
        ts = self.ts
        ids = np.nonzero(ts.hactive[:ts.nt])[0]
        cen = ts.hX[ts.htets[ids]].mean(1)
        self._ids = ids
        self._tree = cKDTree(cen)
        t = ts.htets[ids]
        P = ts.hX[t]
        self._T = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], -1)
        self._Tinv = np.linalg.inv(self._T)
        self._P0 = P[:, 0]
        self._rmax = np.linalg.norm(P - cen[:, None], axis=2).max()

    def locate(self, pts, k=8):
        """Rest-space containing tet for each point (-1 if none). Nearest-centroid candidates."""
        if self._tree is None:
            self._locator()
        pts = np.atleast_2d(pts)
        _, cand = self._tree.query(pts, k=min(k, len(self._ids)))
        out = np.full(len(pts), -1)
        bary = np.zeros((len(pts), 4))
        for j in range(cand.shape[1]):
            c = cand[:, j]
            l = np.einsum("nij,nj->ni", self._Tinv[c], pts - self._P0[c])
            b0 = 1 - l.sum(1)
            ok = (out < 0) & (b0 >= -1e-9) & (l >= -1e-9).all(1)
            out[ok] = self._ids[c[ok]]
            bary[ok] = np.column_stack([b0[ok], l[ok]])
        return out, bary

    def invalidate(self):
        self._tree = None

    def snapshot(self):
        import copy
        return {k: copy.deepcopy(v) for k, v in self.__dict__.items()
                if k not in ("ts", "_tree", "_ctree", "_T", "_Tinv", "_P0", "_ids", "_sheet_tree")}

    def restore(self, snap):
        import copy
        for k, v in snap.items():
            self.__dict__[k] = copy.deepcopy(v)
        self._tree = None
        self._sheet_tree = None
        if hasattr(self, "_cen0"):
            self._ctree = cKDTree(self._cen0)

    def record(self):
        """Call once per rigid tick after the tissue step. Each sample keeps its last recorded rest
        position; a sample that loses and regains its anchor within `bridge_m` (relocation across a
        freshly split element, a brief load dip) is connected to its previous point, so the recorded
        surface has no holes that would leave crossed tets unsplit."""
        self.tick += 1
        if self.ts.edge is None or self.tick % self.record_every:
            return
        e = self.ts.edge
        on = e["a_on"].numpy().astype(bool)
        A = e["a_rest"].numpy().astype(np.float64)
        epoch = e["a_epoch"].numpy()
        shape = e["shape"].numpy()
        n = len(A)
        if self.prev is None or len(self.prev[0]) != n:
            self.prev = (np.zeros(n, bool), np.zeros((n, 3)), epoch.copy())
        pvalid, P, pep = self.prev
        # a continuously held anchor is always joined to its previous point, however far it slipped (a worn
        # blade's crack runs several mm in one record interval after a deep indentation; a distance cap leaves
        # holes in the top layer: E4); across a re-anchoring only points within bridge_m join
        near = pvalid & on & ((epoch == pep) | (np.linalg.norm(A - P, axis=1) < self.bridge_m))
        adv = np.linalg.norm(A - P, axis=1)
        for k in range(n - 1):
            if near[k] and near[k + 1] and shape[k] == shape[k + 1] and (adv[k] > 1e-6 or adv[k + 1] > 1e-6):
                q = (P[k], P[k + 1], A[k + 1], A[k])
                for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                    self._add_triangle(np.array(tri))
        # each sample stands for ell of edge centred on it: the swept strip of an anchored run extends half a
        # sample spacing (plus the seal tolerance) beyond its end samples, out to a free surface such as the
        # end face of a block (without it the faces within 0.5 mm of an end face stayed uncut: development probe)
        ell = e["ell"].numpy()
        for k in range(n):
            if not (near[k] and (adv[k] > 1e-6)):
                continue
            for nb_, sgn in ((k - 1, -1.), (k + 1, 1.)):
                inner = k - int(sgn)
                if 0 <= nb_ < n and near[nb_] and shape[nb_] == shape[k]:
                    continue                      # not a run end on this side
                if not (0 <= inner < n and near[inner] and shape[inner] == shape[k]):
                    continue                      # a lone sample: no direction
                tr = A[k] - A[inner]
                ln = np.linalg.norm(tr)
                if ln < 1e-9:
                    continue
                ext = tr/ln*(.5*float(ell[k]) + self.seal_m)
                # wind the strip like the swept quads (lower sample index first) so every recorded triangle's
                # normal is the blade normal t x n (the crumb rule reads sides from it)
                q = (P[k] + ext, P[k], A[k], A[k] + ext) if sgn < 0 else (P[k], P[k] + ext, A[k] + ext, A[k])
                for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                    self._add_triangle(np.array(tri), "end")
        # severing against a support: tissue under an anchored edge sample crushed below J = 0.3 (edge
        # pressing it onto the board) is cut through to the far surface: the recorded surface is extended
        # from the anchor along the rest-space cutting direction until it leaves the material
        if self.sever_J > 0 and on.any():
            self._sever(on, A, shape)
        P = np.where(on[:, None], A, P)
        pvalid = pvalid | on
        self.prev = (pvalid, P, epoch.copy())

    def _sever(self, on, A, shape):
        """An anchored edge sample whose edge point is within `sever_gap` of a rigid support (board)
        along the cutting direction is pressing the remaining tissue onto that support: the thin crushed
        layer is cut through. The recorded surface is extended from the anchor along the rest-space
        cutting direction until it leaves the material."""
        ts = self.ts
        e = ts.edge
        if not self.supports:
            return
        pos = e["pos"].numpy().astype(np.float64)
        nn = e["nn"].numpy().astype(np.float64)
        tet = e["a_tet"].numpy()
        x = ts.x.numpy()
        p0, q0 = ts._pose_now[0], ts._pose_now[1]
        from scipy.spatial.transform import Rotation as Rot
        crushed = np.zeros(len(A), bool)
        dirs = np.zeros((len(A), 3))
        for k in np.nonzero(on)[0]:
            near = False
            for si in self.supports:
                R = Rot.from_quat(q0[si]).as_matrix()
                h = np.asarray(ts.shapes[si]["half"])
                # distance from the edge point to the support's top face along n
                pl = R.T @ (pos[k] - p0[si])
                if np.all(np.abs(pl[:2]) <= h[:2]):
                    gap = pl[2] - h[2]
                    # an edge pressed INTO the support (a robot's blade sinks 1-3 mm into the board under 40 N in
                    # MuJoCo's soft contact) is past the remaining layer too: the window reaches through the support
                    # (a window that stops 1 mm below the top leaves the last layer of a robot press cut)
                    into = 2*h[2] if self.sever_into is None else self.sever_into
                    if -into < gap < self.sever_gap and np.dot(R[:, 2], nn[k]) < -.5:
                        near = True
            if not near:
                continue
            t = int(tet[k])
            if t < 0 or not ts.hactive[t]:
                continue
            r = ts.htets[t]
            Dm = np.stack([ts.hX[r[i]] - ts.hX[r[0]] for i in (1, 2, 3)], 1)
            Ds = np.stack([x[r[i]] - x[r[0]] for i in (1, 2, 3)], 1)
            try:
                d = np.linalg.solve(Ds @ np.linalg.inv(Dm), nn[k])
            except np.linalg.LinAlgError:
                continue
            if np.linalg.norm(d) > 0:
                crushed[k] = True
                dirs[k] = d/np.linalg.norm(d)
        if not crushed.any():
            return
        h0 = self.h0()
        step = .15e-3
        ends = A.copy()
        for k in np.nonzero(crushed)[0]:
            q = A[k].copy()
            for _ in range(int(3*h0/step)):
                nxt = q + step*dirs[k]
                t, _ = self.locate(nxt[None])
                if t[0] < 0:
                    break
                q = nxt
            ends[k] = q
        for k in range(len(A) - 1):
            if crushed[k] and crushed[k + 1] and shape[k] == shape[k + 1]:
                quad = (A[k], A[k + 1], ends[k + 1], ends[k])
                for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                    self._add_triangle(np.array(tri), "sever")
        self.severed = getattr(self, "severed", 0) + int(crushed.sum())

    def _add_triangle(self, tri, kind="sweep"):
        nrm = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        area = .5*np.linalg.norm(nrm)
        if area < 1e-12:
            return
        n = nrm/(2*area)
        c = tri.mean(0)
        self.surface.append(tri)
        if not hasattr(self, "surface_kind"):
            self.surface_kind = []
        self.surface_kind.append(kind)
        pts = np.vstack([tri, c[None], .5*(tri + np.roll(tri, 1, 0))])
        tets, _ = self.locate(pts)
        for t in set(int(x) for x in tets if x >= 0):
            a = self.acc.get(t)
            if a is None:
                self.acc[t] = [area*n, area*c, area]
            else:
                if np.dot(a[0], n) < 0:
                    n = -n
                a[0] += area*n
                a[1] += area*c
                a[2] += area

    # ------------------------------------------------------------------ splitting
    def ready(self):
        """Tets whose recorded surface covers their cross-section and contain no live crack tip."""
        ts = self.ts
        if not self.acc:
            return {}
        e = ts.edge
        tips = set()
        if e is not None:
            on = e["a_on"].numpy().astype(bool)
            if on.any():
                tt, _ = self.locate(e["a_rest"].numpy()[on].astype(np.float64))
                tips = set(int(t) for t in tt if t >= 0)
                tips |= set(int(t) for t in e["a_tet"].numpy()[on] if t >= 0)
        out = {}
        for t, (sn, sc, sa) in self.acc.items():
            if not ts.hactive[t] or t in tips:
                continue
            ln = np.linalg.norm(sn)
            if ln < 1e-15:
                continue
            n = sn/ln
            p0 = sc/sa
            old = self.tet_plane.get(t)
            if self.mode != "assign" and old is not None and abs(np.dot(old[0], n)) > np.cos(np.radians(20)) \
                    and abs(np.dot(old[0], p0 - old[1])) < .3*self.h0():
                continue          # already a face of this crack (split mode)
            S = plane_section_area(ts.hX[ts.htets[t]], n, p0)
            if S > 0 and sa >= self.coverage*S:
                out[t] = (n, p0)
        return out

    def split_booked(self):
        """split(), with the energy the operation changes booked to the tissue ledger's 'topo' term (energy removed:
        kinetic + gravitational + elastic before minus after). A crack insertion moves node copies onto the blade's
        flanks and re-lumps and re-scales masses, all between substeps: without this term those jumps would show up
        as an unexplained ledger residual."""
        ts = self.ts
        e0 = ts.energies()
        nf0, n0 = len(getattr(self, "cut_faces", ())), ts.n
        r = self.split()
        if len(getattr(self, "cut_faces", ())) != nf0 or ts.n != n0:
            e1 = ts.energies()
            ts.book("topo", sum(e0[k] - e1[k] for k in ("ke", "pe", "psi")))
        return r

    def split(self, planes=None):
        """Split the ready tets. Returns the number of tets split."""
        ts = self.ts
        if self.mode == "surface" and planes is None:
            return self._surface_split()
        planes = self.ready() if planes is None else planes
        if not planes:
            return 0
        X = ts.hX
        planes = dict(planes)
        S = list(planes)
        # closure: every active tet that shares a cut edge (or an on-cut vertex with vertices on both sides)
        # with a split tet is split in the same batch, with the plane of that neighbour. Without it a
        # neighbour kept the whole edge while the split tet used the two new nodes (a hanging node), tied
        # the two sides together across the cut, stretched and inverted (development probe dbg7/dbg8).
        if self.use_closure:
            self._closure(planes, S)
        # orient normals consistently between neighbouring split tets (flood by shared vertices)
        ref = {}
        for t in S:
            n, p = planes[t]
            for v in ts.htets[t]:
                if v in ref and np.dot(ref[v][0], n) < 0:
                    n = -n
                    break
            planes[t] = (n, p)
            for v in ts.htets[t]:
                ref.setdefault(v, (n, p))
        # per-vertex averaged plane
        vn, vp, vc = {}, {}, {}
        for t in S:
            n, p = planes[t]
            for v in ts.htets[t]:
                if v in vn and np.dot(vn[v], n) < 0:
                    n = -n
                vn[v] = vn.get(v, 0) + n
                vp[v] = vp.get(v, 0) + p
                vc[v] = vc.get(v, 0) + 1
        dist = {}
        for v in vn:
            n = vn[v]/max(np.linalg.norm(vn[v]), 1e-15)
            dist[v] = float(np.dot(X[v] - vp[v]/vc[v], n))
            vn[v] = n
        # per-VERTEX tolerance (20 % of the mean length of its edges in split tets) and class, so that
        # every tet sees the same classification of a shared vertex (a per-tet tolerance let a vertex
        # be '+' in one tet and 'on the cut' in its neighbour, which then bridged the cut)
        elen = {}
        for t in S:
            vv = ts.htets[t]
            for a in range(4):
                for b in range(a + 1, 4):
                    L_ = float(np.linalg.norm(X[vv[a]] - X[vv[b]]))
                    for q in (vv[a], vv[b]):
                        e_ = elen.setdefault(int(q), [0., 0])
                        e_[0] += L_; e_[1] += 1
        h0 = self.h0()
        vtol = {v: self.snap_frac*h0 for v in dist}      # global: a local tolerance shrank with the mesh
        vcls = {v: (1 if dist[v] > vtol[v] else (-1 if dist[v] < -vtol[v] else 0)) for v in dist}
        # snap-then-split: move on-cut vertices exactly onto the cut plane (rest AND current, the current
        # displacement mapped with the local deformation gradient so no stress jump is introduced), so the
        # section is planar, the side cells are convex, and coning from their centroid gives no slivers.
        # A move that would degrade an incident tet (min altitude < 15 % of the vertex's mean edge) is refused
        # and the vertex keeps its side.
        x_cur = ts.x.numpy()
        act_all = np.nonzero(ts.hactive[:ts.nt])[0]
        rows_all = ts.htets[act_all]
        self.snapped = 0
        refused = set()
        for vtx_id, c in list(vcls.items()):
            if c != 0 or abs(dist[vtx_id]) < 1e-12 or int(vtx_id) in self.twin:
                continue
            inc = act_all[(rows_all == vtx_id).any(1)]
            n_, dmove = vn[vtx_id], -dist[vtx_id]*vn[vtx_id]
            # local deformation gradient (volume-weighted mean over incident tets)
            Fs, ws = np.zeros((3, 3)), 0.
            for t in inc:
                r_ = ts.htets[t]
                Dm = np.stack([X[r_[i]] - X[r_[0]] for i in (1, 2, 3)], 1)
                Ds = np.stack([x_cur[r_[i]] - x_cur[r_[0]] for i in (1, 2, 3)], 1)
                w_ = abs(np.linalg.det(Dm))
                if w_ > 0:
                    Fs += w_*(Ds @ np.linalg.inv(Dm)); ws += w_
            F = Fs/ws if ws > 0 else np.eye(3)
            old_X = X[vtx_id].copy()
            X[vtx_id] = old_X + dmove
            from filleting.mesher import min_altitude, signed_volume
            ok = (signed_volume(X, ts.htets[inc]) > 0).all() and (min_altitude(X, ts.htets[inc]) > .12*h0).all()
            if ok:
                x_cur[vtx_id] = x_cur[vtx_id] + F @ dmove
                dist[vtx_id] = 0.
                self.snapped += 1
            else:
                X[vtx_id] = old_X
                refused.add(int(vtx_id))      # stays on the cut (duplicated in place); its tets are not split
        self.refused_snaps = len(refused)
        ts.x.assign(x_cur)
        # tolerance per tet: 20 % of mean edge length
        new_tets, new_mat, new_sheet, retired, graze_minus, new_plane = [], [], [], [], [], []
        zero_minus = {}
        added_nodes = []

        def new_node(src_a, src_b, f):
            i = ts.n
            if i >= ts.Nc:
                raise RuntimeError("tissue node capacity exhausted")
            ts.n += 1
            X[i] = X[src_a] + f*(X[src_b] - X[src_a])
            ts.hkind[i] = 0
            added_nodes.append((i, src_a, src_b, f))
            return i

        if self.mode == "assign":
            return self._fan_split(planes)
        split_count = 0
        self.new_twins = []
        centroid_nodes = []
        side_hint = {}
        for t in S:
            vtx = ts.htets[t].copy()
            P = X[vtx]
            d = np.array([dist[v] for v in vtx])
            s = np.array([vcls[v] for v in vtx])
            if any(int(v_) in refused for v_ in vtx) and (s > 0).any() and (s < 0).any():
                # a vertex near the cut could not be snapped: keep this tet whole, on its centroid's side
                c_side = float(np.dot(X[vtx].mean(0) - vp[vtx[0]]/vc[vtx[0]], vn[vtx[0]]))
                if c_side < 0:
                    graze_minus.append(t)
                self.tet_plane[t] = planes[t]
                continue
            if not ((s > 0).any() and (s < 0).any()):
                if (s < 0).any():
                    graze_minus.append(t)      # lies on the minus side; its on-cut vertices re-point below
                self.tet_plane[t] = planes[t]
                continue
            split_count += 1
            retired.append(t)
            for a in range(4):
                if s[a] == 0 and int(vtx[a]) not in zero_minus:
                    zero_minus[int(vtx[a])] = zm_ = new_node(int(vtx[a]), int(vtx[a]), 0.)
                    self.twin[zm_] = int(vtx[a]); self.twin.setdefault(int(vtx[a]), zm_)
                    self.new_twins.append((int(vtx[a]), zm_))
            for a in range(4):
                for b in range(a + 1, 4):
                    if s[a]*s[b] < 0:
                        va, vb = int(vtx[a]), int(vtx[b])
                        key = (min(va, vb), max(va, vb))
                        if key not in self.edge_map:
                            f = d[a]/(d[a] - d[b])
                            na_, nb_ = new_node(va, vb, f), new_node(va, vb, f)
                            self.edge_map[key] = {va: na_, vb: nb_}
                            self.twin[na_] = nb_; self.twin[nb_] = na_
                            self.new_twins.append((na_, nb_))
            # side cells by clipping the tet's faces; quads split through their lowest-index vertex
            # (face-local rule, so neighbouring tets agree), then coned from the cell's lowest node
            def side_node(a, side):
                va = int(vtx[a])
                if s[a] == 0 and side < 0:
                    return zero_minus[va]
                return va

            def edge_node(a, b, side):
                va, vb = int(vtx[a]), int(vtx[b])
                m = self.edge_map[(min(va, vb), max(va, vb))]
                return m[va] if s[a] == side else m[vb]

            Vt = abs(tet_volume(P))
            vsum = 0.
            for a in range(4):
                side_hint[int(vtx[a])] = int(s[a]) if s[a] != 0 else 1
                if s[a] == 0:
                    side_hint[zero_minus[int(vtx[a])]] = -1
            for a in range(4):
                for b in range(a + 1, 4):
                    if s[a]*s[b] < 0:
                        va, vb = int(vtx[a]), int(vtx[b])
                        m_ = self.edge_map[(min(va, vb), max(va, vb))]
                        side_hint[m_[va]] = int(s[a]); side_hint[m_[vb]] = int(s[b])
            for side in (1, -1):
                faces = []
                for (a, b, c) in ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)):
                    poly = []
                    for u, w in ((a, b), (b, c), (c, a)):
                        if s[u] == side or s[u] == 0:
                            poly.append(side_node(u, side))
                        if s[u]*s[w] < 0:
                            poly.append(edge_node(u, w, side))
                    if len(poly) >= 3:
                        faces.append(poly)
                # section polygon: edge nodes + zero vertices of this side, ordered around the centroid
                sec = [side_node(a, side) for a in range(4) if s[a] == 0]
                for a in range(4):
                    for b in range(a + 1, 4):
                        if s[a]*s[b] < 0:
                            sec.append(edge_node(a, b, side))
                if len(sec) >= 3:
                    Q = X[sec]
                    c0 = Q.mean(0)
                    nrm = vn[int(vtx[0])]
                    u = Q[0] - c0
                    u = u - np.dot(u, nrm)*nrm
                    u /= max(np.linalg.norm(u), 1e-15)
                    w = np.cross(nrm, u)
                    order = np.argsort(np.arctan2((Q - c0) @ w, (Q - c0) @ u))
                    faces.append([sec[i] for i in order])
                tris = []
                for poly in faces:
                    if len(poly) == 3:
                        tris.append(poly)
                    else:          # quad (4); pentagons do not occur in a tet clipped by one plane
                        k0 = int(np.argmin(poly))
                        q = poly[k0:] + poly[:k0]
                        for i in range(1, len(q) - 1):
                            tris.append([q[0], q[i], q[i + 1]])
                cell = sorted(set(v for f in tris for v in f))
                if len(cell) < 4:
                    continue
                if len(cell) == 4:          # the cell is itself a tet
                    apex = None
                    tris_use = [cell[1:]]
                    apex = cell[0]
                else:
                    cpos = X[cell].mean(0)
                    Minv = np.linalg.inv(np.stack([P[1] - P[0], P[2] - P[0], P[3] - P[0]], 1))
                    lb = Minv @ (cpos - P[0])
                    apex = new_node(int(vtx[0]), int(vtx[0]), 0.)
                    X[apex] = cpos
                    centroid_nodes.append((apex, [int(q) for q in vtx], np.array([1 - lb.sum(), *lb])))
                    tris_use = tris
                for tri in tris_use:
                    if apex in tri:
                        continue
                    ctet = np.array([apex, *tri])
                    vol = tet_volume(X[ctet])
                    if abs(vol) < 1e-3*Vt:
                        self.dropped_volume = getattr(self, "dropped_volume", 0.) + abs(vol)
                        continue
                    if vol < 0:
                        ctet = ctet[[0, 1, 3, 2]]
                    vsum += abs(vol)
                    new_tets.append(ctet)
                    new_mat.append(ts.htmat[t])
                    new_sheet.append(ts.hsheet[t])
                    new_plane.append(planes[t])
            self.volume_error = max(getattr(self, "volume_error", 0.), abs(vsum - Vt)/Vt)
        # on-cut vertices of grazing minus-side tets also need a minus copy
        for t in graze_minus:
            for v in ts.htets[t]:
                if vcls.get(v, 1) == 0 and int(v) not in zero_minus:
                    zero_minus[int(v)] = zm_ = new_node(int(v), int(v), 0.)
                    self.twin[zm_] = int(v); self.twin.setdefault(int(v), zm_)
                    self.new_twins.append((int(v), zm_))
        # re-point every active, unsplit tet that lies on the minus side of an on-cut vertex
        if zero_minus:
            retired_set = set(int(r) for r in retired)
            act = np.nonzero(ts.hactive[:ts.nt])[0]
            for z, zm in zero_minus.items():
                n, p, tz = vn[z], vp[z]/vc[z], vtol[z]
                inc = act[(ts.htets[act] == z).any(1)]
                for t in inc:
                    if int(t) in retired_set:
                        continue
                    others = [v for v in ts.htets[t] if v != z]
                    sg = []
                    for v in others:
                        if v in vcls:
                            sg.append(vcls[v])
                        else:
                            dv = float(np.dot(X[v] - p, n))
                            sg.append(1 if dv > tz else (-1 if dv < -tz else 0))
                    sg = np.array(sg)
                    if (sg <= 0).all() and (sg < 0).any():
                        row = ts.htets[t]
                        row[row == z] = zm
        # commit: node states
        x = ts.x.numpy(); v = ts.v.numpy()
        for i, a, b, f in added_nodes:
            x[i] = x[a] + f*(x[b] - x[a])
            v[i] = v[a] + f*(v[b] - v[a])
        for i, parent, bc in centroid_nodes:        # affine image of the cell centroid in the parent
            x[i] = bc @ x[parent]
            v[i] = bc @ v[parent]
        if side_hint:
            hint = ts.hside
            for i_, sgn in side_hint.items():
                hint[i_] = sgn
            ts.side.assign(hint)
        for t in retired:
            ts.hactive[t] = 0
        if new_tets:
            k = len(new_tets)
            if ts.nt + k > ts.Tc:
                raise RuntimeError("tissue tet capacity exhausted")
            ts.htets[ts.nt:ts.nt + k] = np.array(new_tets)
            ts.htmat[ts.nt:ts.nt + k] = new_mat
            ts.hsheet[ts.nt:ts.nt + k] = new_sheet
            ts.hactive[ts.nt:ts.nt + k] = 1
            for j, pl in enumerate(new_plane):
                self.tet_plane[ts.nt + j] = pl
            ts.nt += k
        for t in list(retired) + list(graze_minus) + [t for t in S if t not in set(retired)]:
            self.acc.pop(t, None)          # resolved (split, or found to lie on one side of the cut)
        ts.x.assign(x); ts.v.assign(v)
        # (crumb merging by twin substitution was tried and rejected: twins share the rest position but not
        # the current one once the cut opens, so re-pointing stretched the crumb across the gap and
        # blew up; small fragments are kept as the free debris they are)
        self._commit_geometry()
        if self.tie_twins:
            ts.add_twins(self.new_twins)
        self.events.append(dict(tick=self.tick, split=split_count, new_tets=len(new_tets), new_nodes=len(added_nodes)))
        return split_count

    # ------------------------------------------------------------------ face-based node splitting
    def _adjacency(self):
        """Face adjacency and vertex stars of the ORIGINAL mesh. In assign mode tets are never created or
        retired, so tet ids are permanent and the original connectivity names every face for good."""
        if getattr(self, "_nbr", None) is not None:
            return
        ts = self.ts
        rows = ts.htets[:ts.nt].copy() if not hasattr(self, "orig") else self.orig
        self.orig = rows
        nt = len(rows)
        faces = np.concatenate([rows[:, [1, 2, 3]], rows[:, [0, 2, 3]], rows[:, [0, 1, 3]], rows[:, [0, 1, 2]]])
        key = np.sort(faces, 1)
        tid = np.tile(np.arange(nt), 4)
        fj = np.repeat(np.arange(4), nt)
        order = np.lexsort(key.T[::-1])
        ks = key[order]
        same = np.nonzero((ks[1:] == ks[:-1]).all(1))[0]
        nbr = -np.ones((nt, 4), np.int64)
        a, b = order[same], order[same + 1]
        nbr[tid[a], fj[a]] = tid[b]
        nbr[tid[b], fj[b]] = tid[a]
        self._nbr = nbr
        n0 = int(rows.max()) + 1
        flat = rows.ravel()
        o = np.argsort(flat, kind="stable")
        self._star_t = np.repeat(np.arange(nt), 4)[o]
        self._star_s = np.searchsorted(flat[o], np.arange(n0 + 1))
        self.cut_faces = set()

    def _pair_key(self, a, b):
        return (a, b) if a < b else (b, a)

    def _small_fragments(self, new, vol):
        """The fragments under min_fragment_m3 that the faces `new` would cut off, as {tet: fragment id} over their tets
        only: a flood fill from each tet beside a new face through faces neither cut nor new, stopped as soon as it
        holds min_fragment_m3. The same fragments as labelling the whole mesh (_fragments) and keeping those under the
        volume that touch a new face; that labelling, up to four times per batch, cost ~0.1 s on the 150k-tet
        tenderloin, and the tear check, proposing faces the crumb rule then refused every 10 ms, kept the torn loin's
        actions at 220-245 x wall/sim from physics alone."""
        ts = self.ts
        nbr = self._nbr
        act = ts.hactive
        cutf = self.cut_faces
        vmin = self.min_fragment_m3
        seeds = []
        for a, b in new:
            seeds.append(a); seeds.append(b)
        lab, big, cid = {}, set(), 0
        for s0 in seeds:
            s0 = int(s0)
            if s0 in lab or s0 in big or not act[s0]:
                continue
            seen = {s0}
            stack = [s0]
            v = float(vol[s0])
            large = v >= vmin
            while stack and not large:
                t = stack.pop()
                for j in range(4):
                    u = int(nbr[t, j])
                    if u < 0 or u in seen or not act[u]:
                        continue
                    key = (t, u) if t < u else (u, t)
                    if key in cutf or key in new:
                        continue
                    seen.add(u)
                    stack.append(u)
                    v += float(vol[u])
                    if v >= vmin:
                        large = True
                        break
            if large:
                big |= seen
            else:
                for t in seen:
                    lab[t] = cid
                cid += 1
        return lab

    def _fragments(self, extra=()):
        """Fragment label per active tet: tets joined through un-cut faces (after node splitting this equals
        node connectivity). Returns (labels over all original tets, -1 for inactive)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        ts = self.ts
        nt = len(self.orig)
        nbr = self._nbr
        t = np.repeat(np.arange(nt), 4)
        u = nbr.ravel()
        m = (u > t) & (ts.hactive[:nt][t] > 0)
        m &= ts.hactive[:nt][np.maximum(u, 0)] > 0
        t, u = t[m], u[m]
        cuts = set(self.cut_faces) | set(extra)
        if cuts:
            ck = np.array([a*nt + b for a, b in cuts], np.int64)
            keep = ~np.isin(t.astype(np.int64)*nt + u, ck)
            t, u = t[keep], u[keep]
        g = coo_matrix((np.ones(len(t)), (t, u)), shape=(nt, nt))
        _, lab = connected_components(g, directed=False)
        lab = lab.copy()
        lab[ts.hactive[:nt] == 0] = -1
        return lab

    def _fan_split(self, planes):
        """Crack insertion by node splitting along existing faces.

        Crack faces: the face between a ready tet a (edge passed, surface recorded) and a face neighbour b is
        cut when their centroids lie on opposite sides of the local crack plane (a's recorded plane, averaged
        with b's when b is ready too). Whole tets stay whole, so mass is exact and no sliver can form; the
        crack follows element faces within h/2 of the swept surface (smaller where vertices were snapped).
        Node splitting: each vertex of a cut face is split into one node per group of its incident tets
        that stay connected through un-cut faces (the standard fan rule), so the two sides separate exactly
        across the crack and nowhere else; the crack front, where the fan is still connected, stays whole.
        Crumbs: a fragment below min_fragment_m3 created by this batch is re-attached to the neighbour it
        shares the most new crack faces with (those faces are not cut)."""
        ts = self.ts
        self._adjacency()
        X = ts.hX
        nbr = self._nbr
        orig = self.orig
        nt = len(orig)
        cen = X[ts.htets[:nt]].mean(1)
        new = {}
        for a, (n, p) in planes.items():
            if a >= nt or not ts.hactive[a]:
                continue
            for j in range(4):
                b = int(nbr[a, j])
                if b < 0 or not ts.hactive[b]:
                    continue
                key = self._pair_key(int(a), b)
                if key in self.cut_faces or key in new:
                    continue
                nn_, pp = np.asarray(n, float), np.asarray(p, float)
                if b in planes:
                    nb, pb = planes[b]
                    nb = np.asarray(nb, float)
                    if np.dot(nb, nn_) < 0:
                        nb = -nb
                    nn_ = nn_ + nb
                    nn_ = nn_/max(np.linalg.norm(nn_), 1e-15)
                    pp = .5*(pp + np.asarray(pb, float))
                da, db = float((cen[a] - pp) @ nn_), float((cen[b] - pp) @ nn_)
                if da*db < 0:
                    new[key] = (nn_, pp)
        for t in planes:
            self.tet_plane[int(t)] = planes[t]
            self.acc.pop(int(t), None)
        return self._cut_faces(new, len(planes))

    def _surface_split(self):
        """Crack faces from the recorded swept surface directly: the face between face-neighbours a and b is
        cut when the segment joining their rest centroids crosses a recorded surface triangle (tolerance
        `seal_m`, which closes the sub-millimetre gaps between adjacent swept quads). This is one global,
        consistent rule (a per-tet plane test left holes where neighbouring planes disagreed by a fraction of
        a millimetre: development probe, 95 uncut faces, block never separated). A face whose tets host a
        live crack tip (anchored edge sample) waits until the edge has left them."""
        ts = self.ts
        self._adjacency()
        nt = len(self.orig)
        X = ts.hX
        if not hasattr(self, "_surf_ptr"):
            self._surf_ptr = 0
            self.pending = {}
            self._cen0 = X[ts.htets[:nt]].mean(1)
            self._ctree = cKDTree(self._cen0)
        cen = self._cen0
        nbr = self._nbr
        found = dict(self.pending)
        tris = self.surface[self._surf_ptr:]
        self._surf_ptr = len(self.surface)
        if tris:
            T = np.array(tris)
            tc = T.mean(1)
            rad = np.linalg.norm(T - tc[:, None], axis=2).max(1) + 1.5*self.h0()
            lists = self._ctree.query_ball_point(tc, rad)
            ti, ta = [], []
            for i, L in enumerate(lists):
                ti.extend([i]*len(L)); ta.extend(L)
            ti = np.array(ti, np.int64); ta = np.array(ta, np.int64)
            ti = np.repeat(ti, 4); tb = nbr[np.repeat(ta, 4), np.tile(np.arange(4), len(ta))]
            ta = np.repeat(ta, 4)
            m = tb > ta
            ti, ta, tb = ti[m], ta[m], tb[m]
            if len(ti):
                hit, nrm = _seg_tri(cen[ta], cen[tb], T[ti], self.seal_m)
                classic = {}
                for i_, a, b, nv in zip(ti[hit], ta[hit], tb[hit], nrm[hit]):
                    key = (int(a), int(b))
                    if key not in self.cut_faces and key not in classic:
                        classic[key] = (nv, T[i_].mean(0))
                # faces with a tet lying ON the sheet are decided by the tets' sides of it, not by the centroid
                # segment: a segment between two centroids that both lie on the sheet runs along it, so no single
                # recorded sliver registers the crossing where the sheet weaves between them (2.5 mm coupon: a row
                # of lattice tets centred on the cut plane stayed joined across it)
                pairs = {(int(a), int(b)) for a, b in zip(ta, tb)}
                decided = self._side_rule(pairs, cen)
                for key, val in classic.items():
                    if key not in decided and key not in found:
                        found[key] = val
                for key, val in decided.items():
                    if val is not None and key not in self.cut_faces and key not in found:
                        found[key] = val
        tips = self._tips()
        new, self.pending = {}, {}
        for key, val in found.items():
            if not (ts.hactive[key[0]] and ts.hactive[key[1]]):
                continue
            if key[0] in tips or key[1] in tips:
                self.pending[key] = val
            else:
                new[key] = val
        for key, val in new.items():
            if len(val) > 2:                     # decided by sides: those sides are now fixed for both tets
                nv, sides = val[0], val[2]["sides"]
                for t, sd in zip(key, sides):
                    self.tet_side[int(t)] = (sd, np.asarray(nv, float))
                self.side_cut += 1
        return self._cut_faces(new, len(new))

    def kerf_faces(self, blades=None, behind=None, report=False):
        """Diagnostic (owner-side; not a world rule): intact faces that cross a blade's body in current space, i.e.
        the segment joining two face-neighbours' current centroids crosses a blade segment's mid-plane inside its
        outline, at least `behind` from the edge toward the spine. Material across a hole in the swept record would
        straddle the blade like this. These faces are not cut as a rule: on the lattice a cut follows tet faces up to
        h/2 from the blade's plane, so tets of ONE piece lie on both sides of the 1 mm blade; such a rule proposed
        11,535 faces in one coupon press cut and the crumb rule refused 8,835 of them. Returns {face: (rest normal, rest
        point)}; with report=True also the faces' current points and their height above the edge."""
        ts = self.ts
        self._adjacency()
        blades = self.blades if blades is None else blades
        behind = self.kerf_behind_m if behind is None else behind
        nt = len(self.orig)
        act = np.nonzero(ts.hactive[:nt])[0]
        if not len(act) or not len(blades):
            return ({}, []) if report else {}
        from scipy.spatial.transform import Rotation as Rot
        x = ts.x.numpy()[:ts.n].astype(np.float64)
        P0, Q0 = ts.s_p0.numpy().astype(np.float64), ts.s_q0.numpy().astype(np.float64)
        h0 = self.h0()
        # coarse filter: centroids inside the box around all blade segments
        half = [np.asarray(ts.shapes[s]["half"], float) for s in blades]
        Rs = [Rot.from_quat(Q0[s]).as_matrix() for s in blades]
        corners = []
        for s, R, h in zip(blades, Rs, half):
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for z in (0., 2*h[2]):
                        corners.append(P0[s] + R @ np.array([sx*h[0], sy*h[1], z]))
        corners = np.array(corners)
        lo, hi = corners.min(0) - 1.5*h0, corners.max(0) + 1.5*h0
        tet = ts.htets[act]
        cen = x[tet].mean(1)
        m = np.all((cen >= lo) & (cen <= hi), 1)
        if not m.any():
            return ({}, []) if report else {}
        cand, C = act[m], cen[m]
        idx = np.full(nt, -1, np.int64)
        idx[cand] = np.arange(len(cand))
        nb = self._nbr[cand]
        A_, B_ = [], []
        for j in range(4):
            b = nb[:, j]
            ok = (b > cand) & (b >= 0)
            ok[ok] &= idx[b[ok]] >= 0
            A_.append(np.nonzero(ok)[0]); B_.append(idx[b[ok]])
        ia, ib = np.concatenate(A_), np.concatenate(B_)
        if not len(ia):
            return ({}, []) if report else {}
        out, rep = {}, []
        cen0 = self._cen0 if hasattr(self, "_cen0") else ts.hX[ts.htets[:nt]].mean(1)
        for s, R, h in zip(blades, Rs, half):
            La, Lb = (C[ia] - P0[s]) @ R, (C[ib] - P0[s]) @ R
            ya, yb = La[:, 1], Lb[:, 1]
            cr = ya*yb < 0
            if not cr.any():
                continue
            f = ya[cr]/(ya[cr] - yb[cr])
            pc = La[cr] + f[:, None]*(Lb[cr] - La[cr])
            inside = (np.abs(pc[:, 0]) <= h[0]) & (pc[:, 2] >= behind) & (pc[:, 2] <= 2*h[2])
            sel = np.nonzero(cr)[0][inside]
            for i_, z_ in zip(sel, pc[inside, 2]):
                a, b = int(cand[ia[i_]]), int(cand[ib[i_]])
                key = (a, b) if a < b else (b, a)
                if key in self.cut_faces or key in out:
                    continue
                # rest-space plane for the node fans: the bisector between the two rest centroids, oriented from the
                # tet on the blade's -y side to the one on its +y side
                neg, pos = (a, b) if ya[i_] < 0 else (b, a)
                d = cen0[pos] - cen0[neg]
                out[key] = (d/max(np.linalg.norm(d), 1e-15), .5*(cen0[a] + cen0[b]))
                if report:
                    rep.append((.5*(C[ia[i_]] + C[ib[i_]]), float(z_), int(s)))
        return (out, rep) if report else out

    def _sheet_nearest(self, P):
        """For points P (rest space): the nearest recorded surface triangle's unit normal (recorder winding: the
        blade normal), the signed offset along it and the in-plane distance of the point's projection outside
        that triangle (0 inside)."""
        T = np.asarray(self.surface)
        if getattr(self, "_sheet_tree", None) is None or self._sheet_n > len(T) or len(T) - self._sheet_n > 2000:
            self._sheet_tree, self._sheet_n = cKDTree(T.mean(1)), len(T)
        _, cand = self._sheet_tree.query(P, k=min(16, self._sheet_n))
        cand = np.asarray(cand).reshape(len(P), -1)
        if len(T) > self._sheet_n:                   # triangles recorded since the last rebuild
            _, near = cKDTree(T[self._sheet_n:].mean(1)).query(P, k=min(16, len(T) - self._sheet_n))
            cand = np.concatenate([cand, self._sheet_n + np.asarray(near).reshape(len(P), -1)], 1)
        n_pts, m = cand.shape
        Tc = T[cand.ravel()]
        Pr = np.repeat(P, m, 0)
        e1, e2 = Tc[:, 1] - Tc[:, 0], Tc[:, 2] - Tc[:, 0]
        nrm = np.cross(e1, e2)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-18)[:, None]
        off = np.einsum("ij,ij->i", Pr - Tc[:, 0], nrm)
        c = Pr - off[:, None]*nrm
        v2 = c - Tc[:, 0]
        d00 = np.einsum("ij,ij->i", e1, e1); d01 = np.einsum("ij,ij->i", e1, e2); d11 = np.einsum("ij,ij->i", e2, e2)
        d20 = np.einsum("ij,ij->i", v2, e1); d21 = np.einsum("ij,ij->i", v2, e2)
        den = np.where(np.abs(d00*d11 - d01*d01) > 0, d00*d11 - d01*d01, 1.)
        b1 = (d11*d20 - d01*d21)/den
        b2 = (d00*d21 - d01*d20)/den
        inside = (b1 >= 0) & (b2 >= 0) & (1 - b1 - b2 >= 0)
        out = np.full(len(Pr), np.inf)
        for a_, b_ in ((0, 1), (1, 2), (2, 0)):
            A, B = Tc[:, a_], Tc[:, b_]
            AB = B - A
            t = np.clip(np.einsum("ij,ij->i", c - A, AB)/np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-30), 0, 1)
            out = np.minimum(out, np.linalg.norm(c - (A + t[:, None]*AB), axis=1))
        out[inside] = 0.
        dist = np.hypot(off, out).reshape(n_pts, m)
        j = np.argmin(dist, 1)
        pick = np.arange(n_pts)*m + j
        return nrm[pick], off[pick], out[pick], c[pick]

    def _side_rule(self, pairs, cen):
        """Faces among `pairs` with at least one tet on the recorded sheet (centroid within on_sheet_frac*h0 of it)
        and both tets over it (projection within seal_m of a triangle): {face: None (same side: not a crack face)
        or (sheet normal, foot point, {'sides', 'sep'})}. A tet's side is the sign of its offset along the nearest
        triangle's normal (0 counts as +), or the side it was locked to when an earlier face of it was cut against
        a parallel sheet. 'sep' is the bisector plane between the two centroids, oriented - to +, for the crumb
        rule (the sheet's own plane cannot separate two centroids that both lie on it)."""
        if not pairs or not self.surface:
            return {}
        tets = np.array(sorted({t for ab in pairs for t in ab}), np.int64)
        nrm, off, outside, foot = self._sheet_nearest(cen[tets])
        band = self.on_sheet_frac*self.h0()
        info = {}
        for i, t in enumerate(tets):
            n = nrm[i]
            sd = 1 if off[i] >= 0 else -1
            lock = self.tet_side.get(int(t))
            if lock is not None and abs(float(lock[1] @ n)) > .9:
                sd = lock[0]*(1 if float(lock[1] @ n) > 0 else -1)
            info[int(t)] = (abs(off[i]) < band, outside[i] <= self.seal_m, sd, n, foot[i])
        decided = {}
        for a, b in pairs:
            ia, ib = info[a], info[b]
            if not (ia[0] or ib[0]) or not (ia[1] and ib[1]):
                continue
            n = ia[3] if ia[0] else ib[3]                  # the sheet's normal at the on-sheet tet
            sa = ia[2]*(1 if float(ia[3] @ n) >= 0 else -1)
            sb = ib[2]*(1 if float(ib[3] @ n) >= 0 else -1)
            if sa == sb:
                decided[(a, b)] = None
                continue
            neg, pos = (a, b) if sa < 0 else (b, a)
            d = cen[pos] - cen[neg]
            decided[(a, b)] = (n, .5*(ia[4] + ib[4]), dict(sides=(sa, sb), sep=(d/max(np.linalg.norm(d), 1e-15),
                                                                                .5*(cen[a] + cen[b]))))
        return decided

    def _tips(self):
        ts = self.ts
        e = ts.edge
        tips = set()
        if e is not None:
            on = e["a_on"].numpy().astype(bool)
            if on.any():
                tips |= set(int(t) for t in e["a_tet"].numpy()[on] if t >= 0)
        return tips

    def _cut_faces(self, new, region):
        """Apply a batch of crack faces {(a, b): (normal, point)}: crumb control, then node splitting."""
        ts = self.ts
        X = ts.hX
        nbr = self._nbr
        orig = self.orig
        nt = len(orig)
        cen = X[ts.htets[:nt]].mean(1)
        if not new:
            self.events.append(dict(tick=self.tick, region=region, cut_faces=0))
            return 0
        # crumbs created by this batch
        vol = np.abs(ts.hV0[:nt])
        for _ in range(4):
            lab = self._small_fragments(new, vol)
            small = sorted(set(lab.values()))
            if not small:
                break
            for c in small:
                # re-attach to ONE side of the crack. (Voting by neighbouring fragment un-cut faces on both sides
                # when, mid-cut, both sides were still one fragment, and voting by direction did the same for a
                # crumb lying on the surface: both bridged finished cuts in development probes.) The signed distance to the recorded surface decides sides: the crumb joins the side its own
                # centroid lies on, and only faces to neighbours on THAT side (each judged against the plane of
                # the triangle that cut that face) are un-cut; faces to the far side stay cut, so a crumb can
                # never bridge the cut (direction-based voting could: an on-plane crumb had both its faces un-cut)
                faces_c = []
                for key, val in new.items():
                    a, b = key
                    if (lab.get(a) == c) == (lab.get(b) == c):
                        continue
                    inner, outer = (a, b) if lab.get(a) == c else (b, a)
                    # a face cut by sides is judged against the plane between its two centroids (the sheet's own
                    # plane passes through both)
                    nv, pt = val[2]["sep"] if len(val) > 2 else val[:2]
                    nv = np.asarray(nv, float); pt = np.asarray(pt, float)
                    faces_c.append((key, float((cen[inner] - pt) @ nv), float((cen[outer] - pt) @ nv)))
                if not faces_c:
                    continue
                sd_c = np.mean([f[1] for f in faces_c])
                if abs(sd_c) < 1e-12:
                    sd_c = -np.mean([f[2] for f in faces_c])      # on the surface: the side with fewer neighbours
                side_c = 1. if sd_c > 0 else -1.
                popped = 0
                for key, _, sd_o in faces_c:
                    if sd_o*side_c > 0:
                        new.pop(key, None)
                        popped += 1
                if not popped:
                    # the crumb has no neighbour on its own side to rejoin: a thin layer between this batch's sheet and
                    # an older sheet that is already split (A4: repeated saw passes left 225 such crumbs loose, some
                    # thrown off at 1.5 m/s), or a sliver against the board under the edge. It joins ONE side: the one
                    # it shares most of this batch's faces with; its faces to the other side stay cut. (Un-cutting
                    # all its new faces would let a board sliver touching both halves tie them together: the 2.5 mm
                    # coupon's press cut left 638 faces uncut 0.6-1.4 mm above the board that way.)
                    pos = [key for key, _, sd_o in faces_c if sd_o > 0]
                    neg = [key for key, _, sd_o in faces_c if sd_o <= 0]
                    for key in (pos if len(pos) >= len(neg) else neg):
                        new.pop(key, None)
                    self.crumbs_rejoined_new_sheet = getattr(self, "crumbs_rejoined_new_sheet", 0) + 1
                self.crumbs_merged += 1
        self.cut_faces |= set(new)
        # vertex planes (for side hints): mean of the planes of the new crack faces around each vertex
        vpl = {}
        for (a, b), val in new.items():
            n, p = np.asarray(val[0], float), np.asarray(val[1], float)
            fv_ = set(orig[a].tolist()) & set(orig[b].tolist())
            for o in fv_:
                e_ = vpl.get(o)
                if e_ is None:
                    vpl[o] = [n.copy(), p.copy(), 1]
                else:
                    if np.dot(e_[0], n) < 0:
                        n = -n
                    e_[0] += n; e_[1] += p; e_[2] += 1
        x = ts.x.numpy(); v = ts.v.numpy()
        dup = 0
        placed = []
        for o, (n, psum, cnt) in vpl.items():
            n = n/max(np.linalg.norm(n), 1e-15)
            p = psum/cnt
            star = [int(t) for t in self._star_t[self._star_s[o]:self._star_s[o + 1]] if ts.hactive[t]]
            if len(star) < 2:
                continue
            idx = {t: i for i, t in enumerate(star)}
            par = list(range(len(star)))

            def find(i):
                while par[i] != i:
                    par[i] = par[par[i]]
                    i = par[i]
                return i
            for t in star:
                for j in range(4):
                    u = int(nbr[t, j])
                    if u in idx and self._pair_key(t, u) not in self.cut_faces:
                        ra, rb = find(idx[t]), find(idx[u])
                        if ra != rb:
                            par[ra] = rb
            groups = {}
            for t in star:
                pos = int(np.nonzero(orig[t] == o)[0][0])
                node = int(ts.htets[t][pos])
                groups.setdefault(node, {}).setdefault(find(idx[t]), []).append((t, pos))
            for node, comps in groups.items():
                if len(comps) < 2:
                    continue
                keys = sorted(comps, key=lambda k: -sum(vol[t] for t, _ in comps[k]))
                for kk in keys:
                    tl = comps[kk]
                    sgn = 1 if float((cen[[t for t, _ in tl]].mean(0) - p) @ n) >= 0 else -1
                    if kk == keys[0]:
                        tgt = node
                    else:
                        tgt = ts.n
                        if tgt >= ts.Nc:
                            raise RuntimeError("tissue node capacity exhausted")
                        ts.n += 1
                        X[tgt] = X[node]; x[tgt] = x[node]; v[tgt] = v[node]
                        ts.hkind[tgt] = ts.hkind[node]
                        self.twin[tgt] = node; self.twin.setdefault(node, tgt)
                        for t, pos in tl:
                            ts.htets[t][pos] = tgt
                        dup += 1
                    placed.append((tgt, sgn))
        # side hints only for nodes that were inside a blade and were put on their own flank: every other
        # split node keeps nearest-flank contact. (A node of one fragment may lie up to h/2 across the ideal
        # plane; with a hint it was pushed ACROSS the blade when it met the slab from its far side, a force
        # jump from 0 to k*2w that injected ~2.7 mJ per event when the knife was lifted: development probe.)
        moved = ts.place_nodes_on_flanks(x, placed)
        self.flank_moved = getattr(self, "flank_moved", 0) + len(moved)
        for node, sgn in moved:
            ts.hside[node] = sgn
        ts.side.assign(ts.hside)
        ts.x.assign(x); ts.v.assign(v)
        self._commit_geometry()
        self.events.append(dict(tick=self.tick, region=region, cut_faces=len(new), duplicated=dup,
                                snapped=getattr(self, "snapped", 0), refused=getattr(self, "refused_snaps", 0)))
        return region if dup else 0

    def _assign_split(self, planes, S, vn, vp, vc, dist, vcls, refused):
        """Node-splitting crack insertion. Every tet of the crossed region S is assigned whole to the side of
        its centroid w.r.t. the local cut plane (after snapping, most crack faces lie on the plane); each
        vertex of S whose S-tets fall on BOTH sides is duplicated: all incident tets on the minus side
        (S-tets and neighbours, by centroid) switch to the copy. No element is created, so mass is exact and
        no sliver can form; where a vertex could not be snapped the crack steps around a tet (<= h/2)."""
        ts = self.ts
        X = ts.hX
        x = ts.x.numpy(); v = ts.v.numpy()
        act = np.nonzero(ts.hactive[:ts.nt])[0]
        rows = ts.htets[act]
        order = np.argsort(rows.ravel(), kind="stable")
        flat_t = np.repeat(act, 4)[order]
        flat_v = rows.ravel()[order]
        starts = np.searchsorted(flat_v, np.arange(ts.n + 1))
        Sset = set(int(t) for t in S)

        def side_of(t, vtx_id):
            n, p = vn[vtx_id], vp[vtx_id]/vc[vtx_id]
            return 1 if np.dot(X[ts.htets[t]].mean(0) - p, n) >= 0 else -1
        dup = 0
        hints = {}
        batch = {}
        for vtx_id in list(vn.keys()):
            vtx_id = int(vtx_id)
            inc = [int(t) for t in flat_t[starts[vtx_id]:starts[vtx_id + 1]]]
            s_in = [side_of(t, vtx_id) for t in inc if t in Sset]
            if not (1 in s_in and -1 in s_in):
                continue
            # duplicate: minus-side tets take the copy
            new = ts.n
            if new >= ts.Nc:
                raise RuntimeError("tissue node capacity exhausted")
            ts.n += 1
            X[new] = X[vtx_id]; x[new] = x[vtx_id]; v[new] = v[vtx_id]
            ts.hkind[new] = ts.hkind[vtx_id]
            for t in inc:
                if side_of(t, vtx_id) < 0:
                    r = ts.htets[t]
                    r[r == vtx_id] = new
            hints[vtx_id] = 1; hints[new] = -1
            self.twin[new] = vtx_id; self.twin.setdefault(vtx_id, new)
            batch[new] = vtx_id; batch[vtx_id] = new
            dup += 1
        # crumbs: whole-tet assignment can isolate a few tets (all their vertices duplicated). While the
        # duplicated nodes still coincide (same call, no dynamics yet) re-attach every fragment smaller than
        # min_fragment_m3 by undoing this batch's duplications for its tets (development probe: a 3-tet crumb
        # latched onto the edge law and rode the blade down)
        if batch:
            act2 = np.nonzero(ts.hactive[:ts.nt])[0]
            rows2 = ts.htets[act2]
            lab = components(rows2, ts.n)
            cl = lab[rows2[:, 0]]
            vol = np.bincount(cl, weights=np.abs(ts.hV0[act2]), minlength=ts.n)
            small = np.nonzero((vol > 0) & (vol < self.min_fragment_m3))[0]
            for c_ in small:
                for t in act2[cl == c_]:
                    r = ts.htets[t]
                    for j in range(4):
                        if int(r[j]) in batch:
                            r[j] = batch[int(r[j])]
                self.crumbs_merged += 1
        pairs = [(o, c) for c, o in batch.items() if hints.get(c) == -1 and o in hints]
        self.flank_moved = getattr(self, "flank_moved", 0) + ts.place_on_flanks(x, pairs)
        for t in S:
            self.tet_plane[int(t)] = planes[t]      # kept in acc: vertex decisions are revisited next batch
        for i_, sgn in hints.items():
            ts.hside[i_] = sgn
        ts.side.assign(ts.hside)
        ts.x.assign(x); ts.v.assign(v)
        self._commit_geometry()
        self.events.append(dict(tick=self.tick, region=len(S), duplicated=dup, snapped=self.snapped,
                                refused=self.refused_snaps))
        return len(S) if dup else 0

    def h0(self):
        """Mean edge length of the original (uncut) mesh."""
        if not hasattr(self, "_h0"):
            ts = self.ts
            act = np.nonzero(ts.hactive[:ts.nt])[0][:4000]
            P = ts.hX[ts.htets[act]]
            self._h0 = float(np.mean([np.linalg.norm(P[:, a] - P[:, b], axis=1).mean()
                                      for a in range(4) for b in range(a + 1, 4)]))
        return self._h0

    def predict(self, lookahead=1.0):
        """Predictive split: tets that contain an anchored edge sample, or the point `lookahead` element
        lengths ahead of it along the cutting direction, are split along the local blade plane (mapped
        to rest space with the host tet's deformation gradient). Twin ties keep the material continuous
        until the blade passes. Returns the number of tets split."""
        ts = self.ts
        e = ts.edge
        if e is None:
            return 0
        on = e["a_on"].numpy().astype(bool)
        if not on.any():
            return 0
        A = e["a_rest"].numpy().astype(np.float64)[on]
        tt = e["t"].numpy().astype(np.float64)[on]
        nn = e["nn"].numpy().astype(np.float64)[on]
        host = e["a_tet"].numpy()[on]
        x = ts.x.numpy()
        planes = {}
        hlen = getattr(self, "_h", None)
        if hlen is None:
            act = np.nonzero(ts.hactive[:ts.nt])[0][:2000]
            P = ts.hX[ts.htets[act]]
            self._h = hlen = float(np.mean(np.linalg.norm(P[:, 0] - P[:, 1], axis=1)))
        for k in range(len(A)):
            t0 = int(host[k])
            if t0 < 0 or not ts.hactive[t0]:
                continue
            row = ts.htets[t0]
            Dm = np.stack([ts.hX[row[i]] - ts.hX[row[0]] for i in (1, 2, 3)], 1)
            Ds = np.stack([x[row[i]] - x[row[0]] for i in (1, 2, 3)], 1)
            try:
                F = Ds @ np.linalg.inv(Dm)
                Finv = np.linalg.inv(F)
            except np.linalg.LinAlgError:
                continue
            b = np.cross(tt[k], nn[k])                     # blade normal (world)
            nr = F.T @ b                                   # rest-space plane normal
            if np.linalg.norm(nr) < 1e-12:
                continue
            nr /= np.linalg.norm(nr)
            dr = Finv @ nn[k]                              # cutting direction in rest space
            dr /= max(np.linalg.norm(dr), 1e-12)
            pts = np.stack([A[k], A[k] + lookahead*hlen*dr, A[k] + .5*lookahead*hlen*dr])
            if not (np.isfinite(pts).all() and np.isfinite(nr).all()):
                continue
            tets_, _ = self.locate(pts)
            for t in tets_:
                t = int(t)
                if t >= 0 and ts.hactive[t] and t not in planes:
                    old = self.tet_plane.get(t)
                    if old is not None and abs(np.dot(old[0], nr)) > np.cos(np.radians(20)) \
                            and abs(np.dot(old[0], A[k] - old[1])) < .3*hlen:
                        continue          # this crack surface already passes through the tet
                    d = (ts.hX[ts.htets[t]] - A[k]) @ nr
                    if d.max() > .05*hlen and d.min() < -.05*hlen:
                        planes[t] = (nr, A[k].copy())
        if not planes:
            return 0
        self.tie_twins = True
        return self.split(planes)

    def _closure(self, planes, S, rounds=4):
        ts = self.ts
        X = ts.hX
        act = np.nonzero(ts.hactive[:ts.nt])[0]
        rows = ts.htets[act]
        # node -> incident active tets
        order = np.argsort(rows.ravel(), kind="stable")
        flat_t = np.repeat(act, 4)[order]
        flat_v = rows.ravel()[order]
        starts = np.searchsorted(flat_v, np.arange(ts.n + 1))
        inS = set(S)
        for _ in range(rounds):
            # signed distance of every vertex of S-tets to the plane of an S-tet containing it
            vplane = {}
            for t in S:
                for v in ts.htets[t]:
                    vplane.setdefault(int(v), planes[t])
            added = []
            for v, (n, p) in list(vplane.items()):
                for t in flat_t[starts[v]:starts[v + 1]]:
                    t = int(t)
                    if t in inS or not ts.hactive[t]:
                        continue
                    d = (X[ts.htets[t]] - p) @ n
                    h_ = max(np.linalg.norm(X[ts.htets[t]][:, None] - X[ts.htets[t]][None], axis=2).max(), 1e-9)
                    tol = .2*h_*.5
                    if (d > tol).any() and (d < -tol).any():
                        planes[t] = (n, p)
                        inS.add(t)
                        added.append(t)
            if not added:
                break
            S.extend(added)
        self.closure_added = len(S)

    def _merge_crumbs(self):
        """Re-attach fragments below min_fragment_m3 to the side they were cut from by replacing each of
        their duplicated nodes with its twin (mass and volume unchanged). A 3.5 mm mesh cannot represent
        sub-0.5 mm3 crumbs meaningfully; they only jitter inside the blade gap."""
        ts = self.ts
        for _ in range(3):
            act = np.nonzero(ts.hactive[:ts.nt])[0]
            rows = ts.htets[act]
            lab = components(rows, ts.n)
            cl = lab[rows[:, 0]]
            vols = np.bincount(cl, weights=np.abs(ts.hV0[act]) if len(ts.hV0) >= ts.nt else None, minlength=ts.n)
            # rest volumes of new tets are not computed yet: use geometry directly
            P = ts.hX[rows]
            vt = np.abs(tet_volume(P))
            vols = np.bincount(cl, weights=vt, minlength=ts.n)
            small = np.nonzero((vols > 0) & (vols < self.min_fragment_m3))[0]
            if not len(small):
                return
            changed = False
            for c in small:
                sel = act[cl == c]
                for t in sel:
                    row = ts.htets[t]
                    for j in range(4):
                        tw = self.twin.get(int(row[j]))
                        if tw is not None:
                            row[j] = tw
                            changed = True
                self.crumbs_merged += 1
            if not changed:
                return

    def _commit_geometry(self):
        ts = self.ts
        nt = ts.nt
        inv, vol = ts.rest_geometry(np.arange(nt))
        ts.hV0[:nt] = vol
        ts.hDminv[:nt] = inv
        ts.tets.assign(ts.htets); ts.tactive.assign(ts.hactive); ts.tmat.assign(ts.htmat)
        ts.Dm_inv.assign(ts.hDminv.astype(np.float32)); ts.V0.assign(ts.hV0.astype(np.float32))
        ts.sheet.assign(ts.hsheet); ts.X.assign(ts.hX.astype(np.float32)); ts.kind.assign(ts.hkind)
        ts.refresh_membrane()                  # skin triangles follow their tets' (split) nodes
        ts.recompute_mass()
        self.invalidate()
        ts.rebuild_bvh()
        self.update_anchorable()
        if hasattr(ts, "after_topology"):
            ts.after_topology(self)

    def update_anchorable(self):
        """Tets of fragments smaller than min_fragment_m3 (debris) cannot host an edge anchor."""
        ts = self.ts
        act = np.nonzero(ts.hactive[:ts.nt])[0]
        rows = ts.htets[act]
        lab = components(rows, ts.n)
        cl = lab[rows[:, 0]]
        vol = np.bincount(cl, weights=np.abs(ts.hV0[act]), minlength=ts.n)
        flag = np.zeros(ts.Tc, np.int32)
        flag[act] = (vol[cl] >= self.min_fragment_m3).astype(np.int32)
        ts.tanch.assign(flag)


def _seg_tri(p, q, T, tol):
    """Segment p->q against triangles T (m,3,3), rest space. A hit needs the segment to cross the triangle's
    plane strictly between its ends and the crossing point to lie within `tol` of the triangle. Returns
    (hit mask, triangle unit normals in vertex order)."""
    e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    n = np.cross(e1, e2)
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 1e-18
    n = n/np.maximum(ln, 1e-18)[:, None]
    dp = np.einsum("ij,ij->i", p - T[:, 0], n)
    dq = np.einsum("ij,ij->i", q - T[:, 0], n)
    cross = ok & (dp*dq < 0)
    s = dp/np.where(np.abs(dp - dq) > 0, dp - dq, 1.)
    c = p + s[:, None]*(q - p)
    # distance from c (in the plane) to the triangle: barycentrics, then edge distances when outside
    v2 = c - T[:, 0]
    d00 = np.einsum("ij,ij->i", e1, e1); d01 = np.einsum("ij,ij->i", e1, e2); d11 = np.einsum("ij,ij->i", e2, e2)
    d20 = np.einsum("ij,ij->i", v2, e1); d21 = np.einsum("ij,ij->i", v2, e2)
    den = np.where(np.abs(d00*d11 - d01*d01) > 0, d00*d11 - d01*d01, 1.)
    b1 = (d11*d20 - d01*d21)/den
    b2 = (d00*d21 - d01*d20)/den
    b0 = 1 - b1 - b2
    inside = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
    dist = np.full(len(p), np.inf)
    for a_, b_ in ((0, 1), (1, 2), (2, 0)):
        A, B = T[:, a_], T[:, b_]
        AB = B - A
        t = np.clip(np.einsum("ij,ij->i", c - A, AB)/np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-30), 0, 1)
        dist = np.minimum(dist, np.linalg.norm(c - (A + t[:, None]*AB), axis=1))
    hit = cross & (inside | (dist <= tol))
    return hit, n


def boundary_faces(tets):
    f = np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 3, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 1]]])
    key = np.sort(f, 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return f[cnt[inv] == 1]


def components(tets, n_nodes):
    """Connected components of the active tet mesh (union-find on shared nodes). Returns per-node label."""
    parent = np.arange(n_nodes)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for row in tets:
        r0 = find(row[0])
        for v in row[1:]:
            r = find(v)
            if r != r0:
                parent[r] = r0
    return np.array([find(i) for i in range(n_nodes)])
