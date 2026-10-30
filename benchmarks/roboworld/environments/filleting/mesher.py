"""Tetrahedral meshing of an implicit body (signed distance, negative inside) on a body-centred
cubic (BCC) lattice by isosurface stuffing (Labelle & Shewchuk 2007).

Why BCC + projection: no tetrahedral mesher is installed in the runtime, and
a BCC lattice gives well-shaped tetrahedra (edges h and h*sqrt(3)/2). Keeping tets whose centroid is
inside and projecting boundary nodes onto the surface removes the voxel staircase that the applecut
precedent shows in its renders. Each projection is accepted only if no incident tet inverts or
degenerates (quality floor below); otherwise it is halved.
"""

import numpy as np


def bcc_tets(lo, hi, h):
    """BCC lattice in the box [lo, hi]: corner nodes + cell centres; 4 tets per interior cell face."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = np.maximum(np.ceil((hi - lo)/h).astype(int), 1)
    nx, ny, nz = n
    # corner nodes (nx+1)(ny+1)(nz+1), centres nx*ny*nz
    gx, gy, gz = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    corners = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    cx, cy, cz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    centres = np.stack([cx, cy, cz], -1).reshape(-1, 3)
    nc = len(corners)

    def cid(i, j, k):
        return (i*(ny + 1) + j)*(nz + 1) + k

    def mid(i, j, k):
        return nc + (i*ny + j)*nz + k

    X = np.concatenate([lo + corners*h, lo + (centres + .5)*h])
    tets = []
    # faces between cell (i,j,k) and its +axis neighbour; also boundary faces are skipped (no neighbour)
    I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    I, J, K = I.ravel(), J.ravel(), K.ravel()
    for axis in range(3):
        sel = [I, J, K][axis] < [nx, ny, nz][axis] - 1
        i, j, k = I[sel], J[sel], K[sel]
        c1 = mid(i, j, k)
        c2 = mid(i + (axis == 0), j + (axis == 1), k + (axis == 2))
        # face corners in cyclic order in the plane at index+1 along axis
        a = [i + 1, j, k] if axis == 0 else ([i, j + 1, k] if axis == 1 else [i, j, k + 1])
        if axis == 0:
            quad = [cid(a[0], a[1], a[2]), cid(a[0], a[1] + 1, a[2]), cid(a[0], a[1] + 1, a[2] + 1), cid(a[0], a[1], a[2] + 1)]
        elif axis == 1:
            quad = [cid(a[0], a[1], a[2]), cid(a[0], a[1], a[2] + 1), cid(a[0] + 1, a[1], a[2] + 1), cid(a[0] + 1, a[1], a[2])]
        else:
            quad = [cid(a[0], a[1], a[2]), cid(a[0] + 1, a[1], a[2]), cid(a[0] + 1, a[1] + 1, a[2]), cid(a[0], a[1] + 1, a[2])]
        for q in range(4):
            tets.append(np.stack([c1, c2, quad[q], quad[(q + 1) % 4]], 1))
    tets = np.concatenate(tets)
    return X, tets


def orient(X, tets):
    """Make every tet positively oriented."""
    d = np.einsum("ij,ij->i", np.cross(X[tets[:, 1]] - X[tets[:, 0]], X[tets[:, 2]] - X[tets[:, 0]]),
                  X[tets[:, 3]] - X[tets[:, 0]])
    t = tets.copy()
    neg = d < 0
    t[neg, 2], t[neg, 3] = tets[neg, 3], tets[neg, 2]
    return t


def signed_volume(X, tets):
    return np.einsum("ij,ij->i", np.cross(X[tets[:, 1]] - X[tets[:, 0]], X[tets[:, 2]] - X[tets[:, 0]]),
                     X[tets[:, 3]] - X[tets[:, 0]])/6.


def boundary_faces(tets):
    """Faces used by exactly one tet, oriented outward (consistent with positive tets)."""
    f = np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 3, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 1]]])
    key = np.sort(f, 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return f[cnt[inv] == 1]


def min_altitude(X, tets):
    V = np.abs(signed_volume(X, tets))
    hs = []
    for a, b, c in ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)):
        A = .5*np.linalg.norm(np.cross(X[tets[:, b]] - X[tets[:, a]], X[tets[:, c]] - X[tets[:, a]]), axis=1)
        hs.append(3*V/np.maximum(A, 1e-30))
    return np.min(hs, 0)


ALPHA_LONG, ALPHA_SHORT = .24999, .41189          # Labelle & Shewchuk 2007, section 3 (warp thresholds)


def _lattice_edges(tets, X, h):
    e = np.concatenate([tets[:, [a, b]] for a in range(4) for b in range(a + 1, 4)])
    e = np.unique(np.sort(e, 1), axis=0)
    length = np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1)
    return e, length > .95*h                      # long edges have length h, short h*sqrt(3)/2


def mesh_sdf(sdf, lo, hi, h, *, refine=3, **_legacy):
    """Tetrahedralize {sdf < 0} with BCC spacing h by isosurface stuffing (Labelle & Shewchuk 2007):
    lattice vertices closer to the surface than alpha x (edge length) along a lattice edge are WARPED onto
    the surface; every other lattice edge crossing the surface gets a cut point on the surface; each BCC tet
    that is partly inside is clipped to its inside part and tetrahedralized by coning from its lowest-id
    vertex, with every polygon fanned from its own lowest-id vertex (so shared lattice faces are split the
    same way in both neighbours: the mesh is conforming). Warping keeps every cut point at least
    0.25 h from a lattice vertex, so no boundary sliver can form.

    The projection mesher (kept below as `mesh_sdf_projected`) accepts boundary tets down to a 0.12 h
    altitude; ~7 % of a box coupon's tets are such flat surface slivers, and a knife edge pressing on them
    inverts them on the 2.5 mm mesh, which adds spurious viscous dissipation.
    Returns (X, tets), positively oriented."""
    lo, hi = np.asarray(lo, float) - h, np.asarray(hi, float) + h
    XL, TL = bcc_tets(lo, hi, h)
    f = sdf(XL).astype(float)
    E, is_long = _lattice_edges(TL, XL, h)
    fa, fb = f[E[:, 0]], f[E[:, 1]]
    crossing = (fa < 0) & (fb > 0) | (fa > 0) & (fb < 0)
    Ec = E[crossing]
    a, b = XL[Ec[:, 0]], XL[Ec[:, 1]]
    ta = np.zeros(len(Ec))
    lo_t, hi_t = np.zeros(len(Ec)), np.ones(len(Ec))
    fa_c, fb_c = fa[crossing], fb[crossing]
    t = fa_c/(fa_c - fb_c)
    for _ in range(refine):                         # safeguarded secant on the true sdf along the edge
        ft = sdf(a + t[:, None]*(b - a))
        same_a = np.sign(ft) == np.sign(fa_c)
        lo_t = np.where(same_a, t, lo_t); hi_t = np.where(same_a, hi_t, t)
        fa_c = np.where(same_a, ft, fa_c); fb_c = np.where(same_a, fb_c, ft)
        tn = lo_t + fa_c/(fa_c - fb_c)*(hi_t - lo_t)
        t = np.clip(tn, lo_t, hi_t)
    cut = a + t[:, None]*(b - a)
    L = np.linalg.norm(b - a, axis=1)
    alpha = np.where(is_long[crossing], ALPHA_LONG, ALPHA_SHORT)
    # warp: each violated lattice vertex moves to its nearest violating cut point
    Xw = XL.copy()
    fw = f.copy()
    best = {}
    for k in np.nonzero(t < alpha)[0]:
        v, d = int(Ec[k, 0]), t[k]*L[k]
        if v not in best or d < best[v][0]:
            best[v] = (d, k)
    for k in np.nonzero(1 - t < alpha)[0]:
        v, d = int(Ec[k, 1]), (1 - t[k])*L[k]
        if v not in best or d < best[v][0]:
            best[v] = (d, k)
    for v, (_, k) in best.items():
        Xw[v] = cut[k]
        fw[v] = 0.
    fw[np.abs(fw) < 1e-12*h] = 0.
    # cut points survive only on edges whose endpoints are still strictly on opposite sides
    fa, fb = fw[E[:, 0]], fw[E[:, 1]]
    crossing2 = (fa < 0) & (fb > 0) | (fa > 0) & (fb < 0)
    edge_id = {}
    pts = [Xw]
    nL = len(XL)
    cross_rows = np.nonzero(crossing)[0]
    row_to_cut = dict(zip(cross_rows.tolist(), range(len(cross_rows))))
    extra = []
    for r in np.nonzero(crossing2)[0]:
        k = row_to_cut.get(int(r))
        if k is None:
            continue
        edge_id[(int(E[r, 0]), int(E[r, 1]))] = nL + len(extra)
        extra.append(cut[k])
    P = np.vstack([Xw] + ([np.array(extra)] if extra else []))
    sg = np.sign(fw[TL])
    inside_any = (sg < 0).any(1)
    whole = inside_any & ~(sg > 0).any(1)
    out = [TL[whole]]
    # all-zero tets: keep if their centroid is inside
    zero4 = (sg == 0).all(1)
    if zero4.any():
        zc = Xw[TL[zero4]].mean(1)
        out.append(TL[zero4][sdf(zc) < 0])
    faces_of = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    new = []
    for row, s_ in zip(TL[inside_any & (sg > 0).any(1)], sg[inside_any & (sg > 0).any(1)]):
        polys = []
        cap = []
        for fi in faces_of:
            poly = []
            for q in range(3):
                i0, i1 = row[fi[q]], row[fi[(q + 1) % 3]]
                s0, s1 = s_[fi[q]], s_[fi[(q + 1) % 3]]
                if s0 <= 0:
                    poly.append(int(i0))
                if s0*s1 < 0:
                    poly.append(edge_id[(int(min(i0, i1)), int(max(i0, i1)))])
            if len(poly) >= 3:
                polys.append(poly)
        for q in range(4):
            if s_[q] == 0:
                cap.append(int(row[q]))
        for q0 in range(4):
            for q1 in range(q0 + 1, 4):
                if s_[q0]*s_[q1] < 0:
                    cap.append(edge_id[(int(min(row[q0], row[q1])), int(max(row[q0], row[q1])))])
        if len(cap) >= 3:
            C = P[cap]
            c0 = C.mean(0)
            # cap normal: gradient of the linear interpolant of the signs' sdf over the tet
            Tm = np.stack([Xw[row[1]] - Xw[row[0]], Xw[row[2]] - Xw[row[0]], Xw[row[3]] - Xw[row[0]]])
            df = np.array([fw[row[1]] - fw[row[0]], fw[row[2]] - fw[row[0]], fw[row[3]] - fw[row[0]]])
            try:
                g = np.linalg.solve(Tm, df)
            except np.linalg.LinAlgError:
                g = np.array([0, 0, 1.])
            g = g/max(np.linalg.norm(g), 1e-30)
            u = C[0] - c0
            u = u - np.dot(u, g)*g
            u = u/max(np.linalg.norm(u), 1e-30)
            w = np.cross(g, u)
            ang = np.arctan2((C - c0) @ w, (C - c0) @ u)
            polys.append([cap[i] for i in np.argsort(ang)])
        verts = set(v for pl in polys for v in pl)
        apex = min(verts)
        for pl in polys:
            if apex in pl:
                continue
            m = int(np.argmin(pl))
            pl = pl[m:] + pl[:m]
            for q in range(1, len(pl) - 1):
                new.append((apex, pl[0], pl[q], pl[q + 1]))
    if new:
        out.append(np.array(new, np.int64))
    tets = np.concatenate(out).astype(np.int64)
    vol = signed_volume(P, tets)
    tets = tets[np.abs(vol) > 1e-9*h**3]
    used, inv = np.unique(tets, return_inverse=True)
    X = P[used]
    tets = inv.reshape(-1, 4).astype(np.int64)
    return X, orient(X, tets)


def mesh_sdf_projected(sdf, lo, hi, h, *, quality=.12, grad_eps=None, iters=4):
    """REJECTED (kept for the record): BCC tets with centroid inside, boundary nodes projected onto the
    surface with a quality floor. Left flat boundary slivers (see mesh_sdf)."""
    X, tets = bcc_tets(lo, hi, h)
    cen = X[tets].mean(1)
    keep = sdf(cen) < 0
    tets = tets[keep]
    used, inv = np.unique(tets, return_inverse=True)
    X = X[used]
    tets = inv.reshape(-1, 4).astype(np.int64)
    tets = orient(X, tets)
    # project boundary nodes onto the surface
    eps = grad_eps or 1e-3*h
    bf = boundary_faces(tets)
    bnodes = np.unique(bf)
    node_tets = [[] for _ in range(len(X))]
    for t, row in enumerate(tets):
        for n in row:
            node_tets[n].append(t)
    for _ in range(iters):
        P = X[bnodes]
        d = sdf(P)
        g = np.stack([(sdf(P + e) - sdf(P - e))/(2*eps) for e in np.eye(3)*eps], 1)
        gn = np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        step = -(d[:, None])*g/gn**2
        step_len = np.linalg.norm(step, axis=1)
        step[step_len > .7*h] *= (.7*h/step_len[step_len > .7*h])[:, None]
        for idx, node in enumerate(bnodes):
            s = step[idx]
            if not np.any(s):
                continue
            tl = node_tets[node]
            old = X[node].copy()
            for scale in (1., .5, .25):
                X[node] = old + scale*s
                if (signed_volume(X, tets[tl]) > 0).all() and (min_altitude(X, tets[tl]) > quality*h).all():
                    break
            else:
                X[node] = old
    return X, tets


def region_labels(X, tets, classifiers, default=0):
    """Assign a material/region id per tet by evaluating classifiers on tet centroids in order
    (later classifiers override earlier ones). classifiers: list of (id, fn(points)->bool)."""
    c = X[tets].mean(1)
    lab = np.full(len(tets), default, np.int32)
    for rid, fn in classifiers:
        lab[fn(c)] = rid
    return lab
