"""Cut-panel meshing in material coordinates.

A ply is a polygon in its own cut-panel frame (u across the seam, v along it, metres; the warp runs along
+v unless a grain angle is given). The mesh is graded: edges of FINE_H inside a band around the declared
fine curves (seam lines, the seam edge, chalk lines) and up to COARSE_H elsewhere, so every place the needle
can reasonably enter, the feed dog grips and the foot presses is resolved at 2.5 mm, while the bulk of the panel
stays cheap.

Construction: boundary points at the local spacing, interior points from hexagonal lattices accepted
greedily from fine to coarse with a minimum-distance rule, Delaunay triangulation, triangles kept when their
centroid is inside the polygon. Quality is reported, not assumed (min angle, aspect).
"""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from .params import FINE_H                  # the seam band's edge length (params numerics.fine_edge)
COARSE_H = 7.5e-3


@dataclass
class PanelSpec:
    name: str
    outline: np.ndarray                 # (n, 2) polygon, counter-clockwise, metres
    fine_curves: list = field(default_factory=list)   # list of (m, 2) polylines that get the fine band
    band: float = 22e-3                 # half-width of the fine band around each fine curve (m)
    fine_h: float = FINE_H
    coarse_h: float = COARSE_H
    grain_deg: float = 0.0              # warp direction measured from +v toward +u


def _polyline_distance(points, curves):
    """Distance from each point to the nearest of the given polylines (segments)."""
    d = np.full(len(points), np.inf)
    for c in curves:
        c = np.asarray(c, float)
        for a, b in zip(c[:-1], c[1:]):
            ab = b - a
            L2 = float(ab @ ab)
            t = np.clip(((points - a) @ ab)/max(L2, 1e-18), 0.0, 1.0)
            q = a + t[:, None]*ab
            d = np.minimum(d, np.linalg.norm(points - q, axis=1))
    return d


def spacing(spec, points):
    """Target edge length at each point: fine inside the band, then a linear ramp (growth over 3 coarse edges)."""
    if not spec.fine_curves:
        return np.full(len(points), spec.coarse_h)
    d = _polyline_distance(points, spec.fine_curves)
    ramp = np.clip((d - spec.band)/(3*spec.coarse_h), 0.0, 1.0)
    return spec.fine_h + ramp*(spec.coarse_h - spec.fine_h)


def _inside(poly, pts):
    from matplotlib.path import Path
    return Path(poly).contains_points(pts, radius=-1e-9)


def _boundary_points(spec):
    poly = np.asarray(spec.outline, float)
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        L = float(np.linalg.norm(b - a))
        if L < 1e-12:
            continue
        # sample at the local spacing (evaluated densely, then walked)
        s = 0.0
        out.append(a)
        while True:
            p = a + (b - a)*(s/L)
            h = float(spacing(spec, p[None])[0])
            s += h
            if s >= L - 0.5*h:
                break
            out.append(a + (b - a)*(s/L))
    return np.array(out)


def _hex_lattice(lo, hi, h):
    dy = h*math.sqrt(3)/2
    ys = np.arange(lo[1], hi[1] + dy, dy)
    pts = []
    for k, y in enumerate(ys):
        off = 0.5*h if k % 2 else 0.0
        xs = np.arange(lo[0] + off, hi[0] + h, h)
        pts.append(np.column_stack([xs, np.full(len(xs), y)]))
    return np.vstack(pts)


def mesh_panel(spec):
    """Returns dict(X (n,2) material coords, tri (t,3) counter-clockwise, quality)."""
    poly = np.asarray(spec.outline, float)
    lo, hi = poly.min(0), poly.max(0)
    bpts = _boundary_points(spec)
    accepted = [p for p in bpts]
    tree_pts = np.array(accepted)
    # candidate interior points from lattices of decreasing fineness; fine first so the band is regular
    levels = sorted({spec.fine_h, 0.5*(spec.fine_h + spec.coarse_h), spec.coarse_h})
    for h in levels:
        cand = _hex_lattice(lo, hi, h)
        cand = cand[_inside(poly, cand)]
        hloc = spacing(spec, cand)
        cand = cand[hloc <= h*1.30 + 1e-12]           # each level fills only where its spacing applies
        hloc = spacing(spec, cand)
        tree = cKDTree(tree_pts)
        keep = []
        new_pts = []
        for p, hp in zip(cand, hloc):
            if tree.query_ball_point(p, 0.80*hp):
                continue
            if new_pts:
                d2 = np.min(np.sum((np.asarray(new_pts) - p)**2, axis=1)) if len(new_pts) < 64 else None
                if d2 is not None and d2 < (0.80*hp)**2:
                    continue
            keep.append(p)
            new_pts.append(p)
            if len(new_pts) >= 64:
                tree_pts = np.vstack([tree_pts, np.asarray(new_pts)])
                tree = cKDTree(tree_pts)
                new_pts = []
        if new_pts:
            tree_pts = np.vstack([tree_pts, np.asarray(new_pts)])
    X = tree_pts
    tri = Delaunay(X).simplices
    cen = X[tri].mean(axis=1)
    tri = tri[_inside(poly, cen)]
    # orientation: counter-clockwise in (u, v)
    a, b, c = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    cross = (b[:, 0] - a[:, 0])*(c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1])*(c[:, 0] - a[:, 0])
    flip = cross < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    area = 0.5*np.abs(cross)
    tri = tri[area > 1e-12]
    # drop vertices not used by any triangle (Delaunay keeps hull points outside concave outlines)
    used = np.unique(tri)
    remap = -np.ones(len(X), int)
    remap[used] = np.arange(len(used))
    X, tri = X[used], remap[tri]
    return dict(X=X, tri=tri.astype(np.int32), quality=quality(X, tri))


def quality(X, tri):
    P = X[tri]
    angs = []
    for i in range(3):
        a, b, c = P[:, i], P[:, (i + 1) % 3], P[:, (i + 2) % 3]
        u, v = b - a, c - a
        cosv = np.sum(u*v, axis=1)/np.maximum(np.linalg.norm(u, axis=1)*np.linalg.norm(v, axis=1), 1e-18)
        angs.append(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
    angs = np.stack(angs, 1)
    e = np.stack([np.linalg.norm(P[:, (i + 1) % 3] - P[:, i], axis=1) for i in range(3)], 1)
    return dict(vertices=int(len(X)), triangles=int(len(tri)), min_angle_deg=float(angs.min()),
                p1_min_angle_deg=float(np.percentile(angs.min(1), 1)), edge_min_mm=float(e.min()*1e3),
                edge_max_mm=float(e.max()*1e3), edge_median_mm=float(np.median(e)*1e3))


def hinges(tri):
    """Interior edges with their two opposite vertices: (v0, v1, a, b) with v0-v1 the shared edge,
    a the third vertex of the triangle where v0->v1 is counter-clockwise, b of the other."""
    edges = {}
    for t, (i, j, k) in enumerate(tri):
        for a, b, c in ((i, j, k), (j, k, i), (k, i, j)):
            key = (min(a, b), max(a, b))
            edges.setdefault(key, []).append((a, b, c))
    out = []
    boundary = []
    for key, lst in edges.items():
        if len(lst) == 2:
            (a0, b0, c0), (a1, b1, c1) = lst
            out.append((a0, b0, c0, c1))
        elif len(lst) == 1:
            boundary.append(key)
    return np.array(out, np.int32).reshape(-1, 4), np.array(boundary, np.int32).reshape(-1, 2)


def color_sets(elems):
    """Greedy colouring so no two elements of one colour share a vertex (Gauss-Seidel in parallel)."""
    elems = np.asarray(elems)
    nv = int(elems.max()) + 1 if len(elems) else 0
    vert_colors = [set() for _ in range(nv)]
    colors = np.zeros(len(elems), np.int32)
    order = np.argsort(-np.bincount(elems.ravel(), minlength=nv)[elems].sum(1), kind="stable")
    for e in order:
        used = set()
        for v in elems[e]:
            used |= vert_colors[v]
        c = 0
        while c in used:
            c += 1
        colors[e] = c
        for v in elems[e]:
            vert_colors[v].add(c)
    groups = [np.nonzero(colors == c)[0].astype(np.int32) for c in range(int(colors.max()) + 1 if len(elems) else 0)]
    return colors, groups


def rect_panel(name, width, length, seam_offset=None, extra_curves=(), band=22e-3, grain_deg=0.0,
               fine_h=FINE_H, coarse_h=COARSE_H):
    """Rectangle [0, width] x [0, length]; the seam edge is u = width (right edge as laid at the machine)."""
    outline = np.array([[0, 0], [width, 0], [width, length], [0, length]], float)
    curves = [np.array([[width, 0], [width, length]])]
    if seam_offset is not None:
        curves.append(np.array([[width - seam_offset, 0], [width - seam_offset, length]]))
    curves += [np.asarray(c, float) for c in extra_curves]
    return PanelSpec(name, outline, curves, band, fine_h, coarse_h, grain_deg)


def seam_edge_path(width, length, radius, offset=0.0, n_arc=48):
    """The edge a curve seam follows (task curve, A6), or the line `offset` inside it: a polyline from the far
    end of the right edge (u = width, v = length) down the right edge, around the near-right corner rounded with
    `radius` (centre (width - radius, radius)), and along the near edge (v = 0) to the left edge (u = 0)."""
    W, L, R, s = float(width), float(length), float(radius), float(offset)
    if R <= 0.0:
        return np.array([[W - s, L], [W - s, s], [0.0, s]])
    c = np.array([W - R, R])
    a = np.linspace(0.0, -0.5*math.pi, n_arc)
    arc = c[None] + (R - s)*np.column_stack([np.cos(a), np.sin(a)])
    return np.vstack([[[W - s, L]], arc, [[0.0, s]]])


def rounded_panel(name, width, length, radius, seam_offset=None, band=22e-3, grain_deg=0.0, fine_h=FINE_H,
                  coarse_h=COARSE_H):
    """Rectangle [0, width] x [0, length] whose near-right corner (u = width, v = 0) is rounded with `radius`: the
    seam edge runs down the right edge, around the corner and along the near edge (task curve). The fine band covers
    that edge and the seam line `seam_offset` inside it."""
    edge = seam_edge_path(width, length, radius)
    outline = np.vstack([edge[::-1][:-1], [[width, length], [0.0, length]]])     # ccw from (0, 0)
    curves = [edge]
    if seam_offset is not None:
        curves.append(seam_edge_path(width, length, radius, seam_offset))
    return PanelSpec(name, outline, curves, band, fine_h, coarse_h, grain_deg)
