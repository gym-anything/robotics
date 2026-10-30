"""Render surface of the mixture's material points (visual only; physics never reads it).

The reconstruction of wine_pouring/surface.py, adapted to a material-point mixture of up to ~600 000 points 1.25 mm
apart whose surface carries per-vertex appearance:

  * Zhu & Bridson (2005) implicit surface phi(x) = |x - xbar(x)| - r, with xbar the kernel-weighted mean of the point
    centres within R of x (R = 2 spacings, r = 0.55 spacing), after the centres have been smoothed toward their
    neighbourhood mean (Yu & Turk 2013, lambda = 0.9), extracted with Warp's GPU marching cubes;
  * the field's cell is one point spacing (1.25 mm), not wine's half spacing: a mixing bowl's contents span
    0.3 x 0.3 x 0.06 m, 26 million cells at 0.625 mm. `cell` coarsens it further (R then grows to 1.6 cells);
  * everything after the extraction stays on the GPU: two Taubin passes (lambda 0.5, mu -0.53) over the triangles'
    umbrella operator, area-weighted vertex normals, and the per-vertex attributes (colour, gloss, translucency: the
    kernel-weighted mean of the attributes of the points within 1.6 spacings of each vertex). wine's NumPy versions
    of these (np.add.at over the edge list) took seconds on a 300 000-triangle surface.

reconstruct() returns one welded triangle mesh in world coordinates; TIMING holds the stage times of the last call.
"""

import time
import numpy as np
import warp as wp

MAX_CELLS = 24_000_000
TIMING = {}


@wp.kernel
def k_smooth_centres(grid: wp.uint64, x: wp.array(dtype=wp.vec3), R: float, lam: float, out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    xi = x[i]
    wsum = float(0.0)
    acc = wp.vec3(0.0)
    for q in wp.hash_grid_query(grid, xi, R):
        d = wp.length(xi - x[q])/R
        if d < 1.0:
            w = 1.0 - d*d*d
            wsum += w
            acc += w*x[q]
    if wsum > 0.0:
        out[i] = xi*(1.0 - lam) + (acc/wsum)*lam
    else:
        out[i] = xi


@wp.kernel
def k_apart(grid: wp.uint64, x: wp.array(dtype=wp.vec3), R: float, spacing: float, most: float, powder: wp.array(dtype=float),
            fill: float, rad: wp.array(dtype=float)):
    """How large a point's sphere is drawn, against a point of a liquid at rest: by how far apart its three nearest
    neighbours stand (their mean distance over the rest spacing), no less than one and no more than `most`; a point
    of dry powder is at least `fill` times as large."""
    i = wp.tid()
    xi = x[i]
    d0 = float(1.0e9)
    d1 = float(1.0e9)
    d2 = float(1.0e9)
    for q in wp.hash_grid_query(grid, xi, R):
        if q != i:
            d = wp.length(xi - x[q])
            if d < d0:
                d2 = d1
                d1 = d0
                d0 = d
            elif d < d1:
                d2 = d1
                d1 = d
            elif d < d2:
                d2 = d
    r = most
    if d2 < 1.0e8:
        r = wp.clamp((d0 + d1 + d2)/(3.0*spacing), 1.0, most)
    rad[i] = wp.max(r, 1.0 + (fill - 1.0)*powder[i])


@wp.kernel
def k_zhu_bridson(grid: wp.uint64, x: wp.array(dtype=wp.vec3), origin: wp.vec3, cell: float, R: float, r: float,
                  rad: wp.array(dtype=float), phi: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    p = origin + wp.vec3(float(i), float(j), float(k))*cell
    wsum = float(0.0)
    acc = wp.vec3(0.0)
    racc = float(0.0)
    for q in wp.hash_grid_query(grid, p, R):
        d = wp.length(p - x[q])/R
        if d < 1.0:
            a = 1.0 - d*d
            w = a*a*a
            wsum += w
            acc += w*x[q]
            racc += w*rad[q]
    if wsum > 1.0e-6:
        phi[i, j, k] = wp.length(p - acc/wsum) - r*racc/wsum
    else:
        phi[i, j, k] = R


@wp.kernel
def k_extreme(phi: wp.array3d(dtype=float), sign: float, out: wp.array3d(dtype=float)):
    """The smallest (sign 1) or largest (sign -1) value of the field among a cell and its 26 neighbours."""
    i, j, k = wp.tid()
    m = sign*phi[i, j, k]
    for a in range(-1, 2):
        for b in range(-1, 2):
            for c in range(-1, 2):
                ii = i + a
                jj = j + b
                kk = k + c
                if ii >= 0 and jj >= 0 and kk >= 0 and ii < phi.shape[0] and jj < phi.shape[1] and kk < phi.shape[2]:
                    m = wp.min(m, sign*phi[ii, jj, kk])
    out[i, j, k] = sign*m


@wp.kernel
def k_to_world(v: wp.array(dtype=wp.vec3), cell: float, lo: wp.vec3, out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = v[i]*cell + lo


@wp.kernel
def k_umbrella(tri: wp.array(dtype=int), v: wp.array(dtype=wp.vec3), acc: wp.array(dtype=wp.vec3),
               deg: wp.array(dtype=float)):
    f = wp.tid()
    a = tri[3*f]
    b = tri[3*f + 1]
    c = tri[3*f + 2]
    wp.atomic_add(acc, a, v[b] + v[c])
    wp.atomic_add(acc, b, v[c] + v[a])
    wp.atomic_add(acc, c, v[a] + v[b])
    wp.atomic_add(deg, a, 2.0)
    wp.atomic_add(deg, b, 2.0)
    wp.atomic_add(deg, c, 2.0)


@wp.kernel
def k_relax(v: wp.array(dtype=wp.vec3), acc: wp.array(dtype=wp.vec3), deg: wp.array(dtype=float), f: float):
    i = wp.tid()
    if deg[i] > 0.0:
        v[i] = v[i] + f*(acc[i]/deg[i] - v[i])


@wp.kernel
def k_relax_w(v: wp.array(dtype=wp.vec3), acc: wp.array(dtype=wp.vec3), deg: wp.array(dtype=float), f: float,
              wgt: wp.array(dtype=float)):
    i = wp.tid()
    if deg[i] > 0.0:
        v[i] = v[i] + f*wgt[i]*(acc[i]/deg[i] - v[i])


@wp.kernel
def k_gather1(grid: wp.uint64, x: wp.array(dtype=wp.vec3), a: wp.array(dtype=float), v: wp.array(dtype=wp.vec3), R: float,
              out: wp.array(dtype=float)):
    """One attribute of a surface vertex: the kernel-weighted mean over the points within R (0 where there are none)."""
    i = wp.tid()
    p = v[i]
    wsum = float(0.0)
    s = float(0.0)
    for q in wp.hash_grid_query(grid, p, R):
        u = wp.length(p - x[q])/R
        if u < 1.0:
            b = 1.0 - u*u
            w = b*b*b
            wsum += w
            s += w*a[q]
    if wsum > 1.0e-6:
        out[i] = s/wsum
    else:
        out[i] = 0.0


@wp.kernel
def k_face_normals(tri: wp.array(dtype=int), v: wp.array(dtype=wp.vec3), n: wp.array(dtype=wp.vec3),
                   vol: wp.array(dtype=float)):
    f = wp.tid()
    a = tri[3*f]
    b = tri[3*f + 1]
    c = tri[3*f + 2]
    fn = wp.cross(v[b] - v[a], v[c] - v[a])
    wp.atomic_add(n, a, fn)
    wp.atomic_add(n, b, fn)
    wp.atomic_add(n, c, fn)
    wp.atomic_add(vol, 0, wp.dot(v[a], wp.cross(v[b], v[c]))/6.0)


@wp.kernel
def k_normalize(n: wp.array(dtype=wp.vec3), sign: float):
    i = wp.tid()
    ln = wp.length(n[i])
    if ln > 1.0e-20:
        n[i] = n[i]*(sign/ln)
    else:
        n[i] = wp.vec3(0.0, 0.0, 1.0)


@wp.kernel
def k_gather(grid: wp.uint64, x: wp.array(dtype=wp.vec3), a0: wp.array(dtype=wp.vec4), a1: wp.array(dtype=wp.vec4),
             v: wp.array(dtype=wp.vec3), R: float, Rmax: float, o0: wp.array(dtype=wp.vec4), o1: wp.array(dtype=wp.vec4)):
    """Attributes of a surface vertex: the kernel-weighted mean over the points within R; where a smoothed vertex has
    none that close, the attributes of the nearest point within Rmax."""
    i = wp.tid()
    p = v[i]
    wsum = float(0.0)
    s0 = wp.vec4(0.0)
    s1 = wp.vec4(0.0)
    best = float(1.0e9)
    bq = int(-1)
    for q in wp.hash_grid_query(grid, p, Rmax):
        d = wp.length(p - x[q])
        if d < best:
            best = d
            bq = q
        u = d/R
        if u < 1.0:
            a = 1.0 - u*u
            w = a*a*a
            wsum += w
            s0 += w*a0[q]
            s1 += w*a1[q]
    if wsum > 1.0e-6:
        o0[i] = s0/wsum
        o1[i] = s1/wsum
    elif bq >= 0:
        o0[i] = a0[bq]
        o1[i] = a1[bq]


def _empty(n_attr):
    return dict(v=np.zeros((0, 3), np.float32), t=np.zeros((0, 3), np.int32), n=np.zeros((0, 3), np.float32),
                a=np.zeros((0, n_attr), np.float32))


POWDER_RADIUS = 1.45      # a loose powder point's sphere against a liquid point's
POWDER_FAIRING = 10       # more smoothing passes over the surface where it is powder
APART_MOST = 1.6          # the largest a point's sphere is drawn for standing apart from its neighbours
CLOSE_PITS = True         # holes of the drawn surface up to two cells across are closed (reconstruct)


def reconstruct(points, spacing, device="cuda:0", cell=None, attrs=None, smooth=0.9, taubin=2, powder=None):
    """Surface of the material around `points` ((n, 3), world): dict(v (m, 3) float32, t (k, 3) int32 with outward
    winding, n (m, 3) unit normals, a (m, A) per-vertex attributes). attrs: (n, A <= 8) float32 per point.
    spacing: the points' rest spacing; cell: the field's cell (default one spacing).

    powder: (n,) 0 to 1 per point, how far it is loose dry powder. A stirred powder's points stand apart and in
    clusters; drawn with a liquid's rule (a sphere of 0.55 of the rest spacing about each smoothed point) the heap is
    a pile of blobs with dark gaps between them, which reads as curd, not as flour. A powder point is drawn with a sphere POWDER_RADIUS times as large, so that neighbours close up, and
    the surface over powder is faired POWDER_FAIRING more times. This is the DRAWING of the heap: the points, their
    volumes and what the evaluator reads are untouched. The drawn heap is a little fuller than the points' own
    volume, as stirred flour is fluffier than poured flour.
    Any point, wet or dry, whose three nearest neighbours stand farther from it than the rest spacing is drawn with
    a sphere larger in that proportion (k_apart, at most APART_MOST): a clump the whisk has pulled loose is then a
    clump and not a sponge full of dark holes."""
    t0 = time.perf_counter()
    pts = np.ascontiguousarray(points, np.float32).reshape(-1, 3)
    n_attr = 0 if attrs is None else int(np.shape(attrs)[1])
    TIMING.clear()
    if len(pts) == 0:
        return _empty(n_attr)
    cell = float(cell or spacing)
    R = max(2.0*spacing, 1.6*cell)
    r = 0.55*spacing
    lo, hi = pts.min(0) - 2*R, pts.max(0) + 2*R
    ext = hi - lo
    cell = max(cell, float((np.prod(ext)/MAX_CELLS)**(1/3)))
    n = np.maximum(np.ceil(ext/cell).astype(int) + 1, 3)
    with wp.ScopedDevice(device):
        xa = wp.array(pts, dtype=wp.vec3)
        grid = wp.HashGrid(128, 128, 128)
        grid.build(xa, 2.0*spacing)
        if smooth and len(pts) > 1:
            xs = wp.zeros_like(xa)
            wp.launch(k_smooth_centres, len(pts), inputs=[grid.id, xa, 2.0*spacing, float(smooth)], outputs=[xs])
            xa = xs
            grid.build(xa, R)
        elif R != 2.0*spacing:
            grid.build(xa, R)
        wp.synchronize()
        TIMING["smooth_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        phi = wp.zeros((int(n[0]), int(n[1]), int(n[2])), dtype=float)
        pw = None
        if powder is not None and float(np.max(powder)) > 0.02:
            pw = wp.array(np.clip(np.asarray(powder, np.float32).reshape(-1), 0.0, 1.0), dtype=float)
        rad = wp.zeros(len(pts), dtype=float)
        wp.launch(k_apart, len(pts), inputs=[grid.id, xa, float(R), float(spacing), float(APART_MOST),
                                             pw if pw is not None else wp.zeros(len(pts), dtype=float),
                                             float(POWDER_RADIUS)], outputs=[rad])
        wp.launch(k_zhu_bridson, dim=phi.shape, inputs=[grid.id, xa, wp.vec3(*map(float, lo)), float(cell), float(R),
                                                       float(r), rad], outputs=[phi])
        if CLOSE_PITS:
            # pits and holes up to two cells across are closed: the material is grown by a cell and shrunk by a cell.
            # A stirred mixture's points stand in clusters with gaps of a spacing or two between them, and the field
            # above has a hole wherever a cell's nearest points all lie to one side of it: drawn as it comes, a clump
            # pulled loose by the whisk is a sponge of dark holes.
            tmp = wp.zeros_like(phi)
            wp.launch(k_extreme, dim=phi.shape, inputs=[phi, 1.0], outputs=[tmp])
            wp.launch(k_extreme, dim=phi.shape, inputs=[tmp, -1.0], outputs=[phi])
        mc = wp.MarchingCubes(int(n[0]), int(n[1]), int(n[2]), device=device)
        mc.surface(phi, 0.0)
        tri = mc.indices
        nv, nt = int(mc.verts.shape[0]), int(tri.shape[0])//3
        wp.synchronize()
        TIMING["field_s"] = time.perf_counter() - t1
        TIMING["cells"] = int(np.prod(n))
        TIMING["cell_m"] = float(cell)
        if nv == 0 or nt == 0:
            return _empty(n_attr)
        t2 = time.perf_counter()
        v = wp.zeros(nv, dtype=wp.vec3)
        wp.launch(k_to_world, nv, inputs=[mc.verts, float(cell), wp.vec3(*map(float, lo))], outputs=[v])
        acc = wp.zeros(nv, dtype=wp.vec3)
        deg = wp.zeros(nv, dtype=float)
        for _ in range(int(taubin)):
            for f in (0.5, -0.53):
                acc.zero_()
                deg.zero_()
                wp.launch(k_umbrella, nt, inputs=[tri, v], outputs=[acc, deg])
                wp.launch(k_relax, nv, inputs=[v, acc, deg, float(f)])
        if pw is not None:
            vw_ = wp.zeros(nv, dtype=float)
            wp.launch(k_gather1, nv, inputs=[grid.id, xa, pw, v, float(R)], outputs=[vw_])
            for _ in range(int(POWDER_FAIRING)):
                for f in (0.5, -0.53):
                    acc.zero_()
                    deg.zero_()
                    wp.launch(k_umbrella, nt, inputs=[tri, v], outputs=[acc, deg])
                    wp.launch(k_relax_w, nv, inputs=[v, acc, deg, float(f), vw_])
        nrm = wp.zeros(nv, dtype=wp.vec3)
        vol = wp.zeros(1, dtype=float)
        wp.launch(k_face_normals, nt, inputs=[tri, v], outputs=[nrm, vol])
        outward = float(vol.numpy()[0]) >= 0.0               # the enclosed volume of an outward-wound surface is positive
        wp.launch(k_normalize, nv, inputs=[nrm, 1.0 if outward else -1.0])
        out = dict()
        if n_attr:
            a = np.zeros((len(pts), 8), np.float32)
            a[:, :n_attr] = np.asarray(attrs, np.float32)
            a0, a1 = wp.array(a[:, :4], dtype=wp.vec4), wp.array(a[:, 4:], dtype=wp.vec4)
            o0, o1 = wp.zeros(nv, dtype=wp.vec4), wp.zeros(nv, dtype=wp.vec4)
            wp.launch(k_gather, nv, inputs=[grid.id, xa, a0, a1, v, float(1.6*spacing), float(R)], outputs=[o0, o1])
            out["a"] = np.concatenate([o0.numpy(), o1.numpy()], axis=1)[:, :n_attr].astype(np.float32)
        else:
            out["a"] = np.zeros((nv, 0), np.float32)
        out["v"] = v.numpy().astype(np.float32)
        t = tri.numpy().reshape(-1, 3).astype(np.int32)
        out["t"] = np.ascontiguousarray(t if outward else t[:, ::-1])
        out["n"] = nrm.numpy().astype(np.float32)
        TIMING["mesh_s"] = time.perf_counter() - t2
    TIMING["total_s"] = time.perf_counter() - t0
    TIMING["points"] = int(len(pts))
    TIMING["triangles"] = int(len(out["t"]))
    return out


if __name__ == "__main__":
    # speed check on a synthetic bowl filling: python -m ingredient_mixing.surface [points]
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600_000
    sp = 0.00125
    side = int(round((n/0.52)**(1/3)))
    g = (np.stack(np.meshgrid(*[np.arange(side)]*3, indexing="ij"), -1).reshape(-1, 3) - side/2 + 0.5)*sp
    g = g[np.linalg.norm(g*[1, 1, 2.2], axis=1) < 0.5*side*sp]
    g = g + np.random.default_rng(0).uniform(-0.1, 0.1, g.shape)*sp
    col = np.tile(np.array([[0.8, 0.7, 0.5, 0.5, 0.0]], np.float32), (len(g), 1))
    for k in range(4):
        m = reconstruct(g, sp, attrs=col)
        print(len(g), "points", {k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in TIMING.items()})
