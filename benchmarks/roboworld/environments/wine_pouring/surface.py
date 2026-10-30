"""Render surface of the liquid particles (visual only; physics never reads it).

Zhu & Bridson (2005) implicit surface: phi(x) = |x - xbar(x)| - r, with xbar the kernel-weighted mean of the particle
centres within R of x; isolated particles become spheres of radius r and a flat free surface sits r above its top layer
of centres. r = 0.55 dx (a particle's volume dx^3 is a sphere of radius 0.62 dx), R = 2 dx. Extracted with Warp's GPU
marching cubes on a grid of at most MAX_CELLS cells over the liquid's bounding box (1 mm cells for a glass or a pour;
coarser only when spilled liquid spreads the box over metres).

Before the field, particle centres are smoothed toward their neighbourhood mean (Yu & Turk 2013, lambda = 0.9 over
2 dx): unsmoothed, a falling stream rendered as a string of 2 mm beads and a glass's free surface as cobbles. After
extraction, two Taubin passes (lambda 0.5, mu -0.53) remove the marching-cubes terracing without shrinking the volume,
and the mesh carries area-weighted vertex normals so RTX shades it smoothly.
"""

import math
import numpy as np
import warp as wp

MAX_CELLS = 24_000_000


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
    out[i] = xi*(1.0 - lam) + (acc/wsum)*lam


@wp.kernel
def k_zhu_bridson(grid: wp.uint64, x: wp.array(dtype=wp.vec3), origin: wp.vec3, cell: float, nx: int, ny: int, nz: int,
                  R: float, r: float, phi: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    p = origin + wp.vec3(float(i), float(j), float(k))*cell
    wsum = float(0.0)
    acc = wp.vec3(0.0)
    for q in wp.hash_grid_query(grid, p, R):
        d = wp.length(p - x[q])/R
        if d < 1.0:
            a = 1.0 - d*d
            w = a*a*a
            wsum += w
            acc += w*x[q]
    if wsum > 1.0e-6:
        phi[i, j, k] = wp.length(p - acc/wsum) - r
    else:
        phi[i, j, k] = R


def clusters(pts, size=0.05):
    """Groups of particles whose 5 cm cells touch (26-neighbourhood): a few droplets thrown across the room must not
    stretch one marching-cubes grid over metres (its cells grew to 7 cm and the glass's wine vanished from a frame)."""
    keys = np.floor(pts/size).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    index = {tuple(k): i for i, k in enumerate(uniq)}
    label = -np.ones(len(uniq), int)
    nl = 0
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    for i in range(len(uniq)):
        if label[i] >= 0:
            continue
        stack = [i]
        label[i] = nl
        while stack:
            j = stack.pop()
            kj = uniq[j]
            for o in offs:
                n = index.get((kj[0] + o[0], kj[1] + o[1], kj[2] + o[2]))
                if n is not None and label[n] < 0:
                    label[n] = nl
                    stack.append(n)
        nl += 1
    return [np.nonzero(label[inv] == c)[0] for c in range(nl)]


def smooth_centres(pts, dx, device="cuda:0", lam=0.9):
    if len(pts) < 2:
        return pts
    with wp.ScopedDevice(device):
        xa = wp.array(pts, dtype=wp.vec3)
        grid = wp.HashGrid(128, 128, 128)
        grid.build(xa, 2.0*dx)
        out = wp.zeros_like(xa)
        wp.launch(k_smooth_centres, len(pts), inputs=[grid.id, xa, 2.0*dx, lam], outputs=[out])
        return out.numpy()


def taubin(v, t, passes=2, lam=0.5, mu=-0.53):
    """Taubin lambda|mu smoothing over the mesh's edge graph (umbrella operator)."""
    if len(t) == 0:
        return v
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    deg = np.bincount(e[:, 0], minlength=len(v)).astype(np.float32)[:, None]
    deg[deg == 0] = 1.
    v = v.astype(np.float32).copy()
    for _ in range(passes):
        for f in (lam, mu):
            acc = np.zeros_like(v)
            np.add.at(acc, e[:, 0], v[e[:, 1]])
            v += f*(acc/deg - v)
    return v


def vertex_normals(v, t):
    fn = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, t[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return (n/np.maximum(ln, 1e-12)).astype(np.float32)


def reconstruct(points, dx, device="cuda:0", cell=None, with_normals=False):
    """Triangle mesh (vertices (N, 3) float32, triangles (T, 3) int32, outward winding[, vertex normals]) of the liquid
    around `points`, one marching-cubes grid per spatial cluster."""
    pts = np.asarray(points, np.float32).reshape(-1, 3)
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
    if len(pts) == 0:
        return empty + ((np.zeros((0, 3), np.float32),) if with_normals else ())
    pts = smooth_centres(pts, dx, device)
    vs, ts, off = [], [], 0
    for idx in clusters(pts):
        v, t = _reconstruct_one(pts[idx], dx, device, cell)
        if len(t):
            vs.append(v); ts.append(t + off); off += len(v)
    if not vs:
        return empty + ((np.zeros((0, 3), np.float32),) if with_normals else ())
    v, t = np.concatenate(vs), np.ascontiguousarray(np.concatenate(ts), np.int32)
    v = taubin(v, t)
    if with_normals:
        return v, t, vertex_normals(v, t)
    return v, t


def _reconstruct_one(points, dx, device="cuda:0", cell=None):
    pts = np.asarray(points, np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    R, r = 2.0*dx, 0.55*dx
    lo, hi = pts.min(0) - 2*R, pts.max(0) + 2*R
    ext = hi - lo
    cell = cell or 0.5*dx
    cell = max(cell, float((np.prod(ext)/MAX_CELLS)**(1/3)))
    n = np.maximum(np.ceil(ext/cell).astype(int) + 1, 3)
    with wp.ScopedDevice(device):
        xa = wp.array(pts, dtype=wp.vec3)
        grid = wp.HashGrid(128, 128, 128)
        grid.build(xa, R)
        phi = wp.zeros((int(n[0]), int(n[1]), int(n[2])), dtype=float)
        wp.launch(k_zhu_bridson, dim=(int(n[0]), int(n[1]), int(n[2])),
                  inputs=[grid.id, xa, wp.vec3(*lo), float(cell), int(n[0]), int(n[1]), int(n[2]), float(R), float(r)],
                  outputs=[phi])
        mc = wp.MarchingCubes(int(n[0]), int(n[1]), int(n[2]), device=device)
        mc.surface(phi, 0.0)
        v = mc.verts.numpy()
        t = mc.indices.numpy().reshape(-1, 3)
    if len(t) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    v = (v*cell + lo).astype(np.float32)
    # outward winding: the enclosed volume must be positive
    vol = np.einsum("ij,ij->i", v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])).sum()/6
    if vol < 0:
        t = t[:, ::-1]
    return v, np.ascontiguousarray(t, np.int32)
