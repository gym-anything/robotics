"""Weld cross-sections measured from the physical state, the way a welding inspector reads a fillet gauge and a
macro-etched section (AWS D1.1 Fig 5.4 / Table 6.1, ISO 5817).

Fusion, penetration, overlap, undercut and the heat-affected zone are read from the thermal solver's cell maps after
the fact: material (air / parent steel / weld metal), body (base plate, bracket), peak liquid fraction (the fusion
boundary of an etched section) and peak temperature. Nothing is read from a commanded path.

The weld's outer shape -- legs, throat, size, convexity -- is what a fillet gauge reads, and the cells cannot resolve
it: a cell is weld metal when its centre lies under the pool's free surface, so a leg read on the 0.25 mm cells comes
out 0 to 0.25 mm off the toe depending on the face's slope there (true legs of 6.375-6.45 mm read 6.25 mm). That surface
exists in the world as a continuous curve: the pool model's free-surface profile of each slice (pool.SliceState: toes A
and B on the members, height h(u) over the chord between them; a frozen slice keeps the profile it froze with), from
which the cells were set. `profile_section` measures the shape on that surface, checked cell by cell against the cells
it produced: weld cells the profile does not explain (metal stacked or spilled onto the slice after, or frozen beyond a
surface that later drew back) are kept whole, cells the profile claims but that are not metal are removed, so nothing
the cells hold escapes the reading. A section with no profile (droplets stacked where no pool spanned the root) is read
on the cells alone (`geometry`: "cells").

Tee, near side (weld frame): the bracket's face is y = 0 (bracket metal at y < 0), the base plate's top z = 0 (metal
below), the fillet in the quadrant y > 0, z > 0, its root at (y, z) = (0, 0).
"""

import math

import numpy as np

from . import pool as PL
from . import thermal as TH

FUSED = .5                  # peak liquid fraction at or above which a cell was melted (the etched fusion boundary)
from . import params as PR
HAZ_C = PR.v(PR.A1_TEMP)    # A1: peak temperature above which steel transforms (heat-affected zone, diagnostic)
SUBCELL = 5                 # the face is rasterised at 1/SUBCELL of a cell (0.05 mm on the 0.25 mm band) for throat,
                            # size and convexity: 0.05 mm is the pool's own toe sampling (pool.TOE_SAMPLE)
ON_SURFACE = 1e-7           # m: a cell centre this close to the profile is on it (either state agrees with the profile)
GEOMETRY_KEYS = ("leg_vertical_m", "leg_horizontal_m", "throat_m", "size_m", "convexity_m", "concavity_m",
                 "face_width_m")


def _edges(g, j0, j1, k0, k1):
    return g.ye[j0:j1 + 1], g.ze[k0:k1 + 1]


def section_maps(proc, i0=None, i1=None):
    """Cell maps of the fillet region for x slices [i0, i1): dict of [n, bj, bk] arrays + the y/z edges."""
    s, g, pool = proc.s, proc.g, proc.pool
    i0 = 0 if i0 is None else i0
    i1 = g.nx if i1 is None else i1
    j0, j1, k0, k1 = pool.j0, pool.j1, pool.k0, pool.k1
    out = {}
    for name in ("peakf", "peakT", "fl", "T"):
        out[name] = s.slice_block(name, i0, i1)[:, j0:j1, k0:k1]
    m = s.mat.reshape(g.nx, g.ny, g.nz)[i0:i1, j0:j1, k0:k1]
    out["mat"] = m
    out["body"] = s.body.reshape(g.nx, g.ny, g.nz)[i0:i1, j0:j1, k0:k1]
    out["orig"] = proc.original_metal.reshape(g.nx, g.ny, g.nz)[i0:i1, j0:j1, k0:k1]
    out["ye"], out["ze"] = _edges(g, j0, j1, k0, k1)
    out["xc"] = g.xc[i0:i1]
    out["dx"] = g.dx[i0:i1]
    return out


def _near_polyline(pts, Y, Z, tol):
    """Points (Y, Z) within tol of the polyline pts [M, 2]."""
    out = np.zeros(np.shape(Y), bool)
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        L2 = float(d @ d)
        if L2 <= 0:
            continue
        f = np.clip(((Y - a[0])*d[0] + (Z - a[1])*d[1])/L2, 0., 1.)
        out |= (Y - a[0] - f*d[0])**2 + (Z - a[1] - f*d[1])**2 < tol*tol
    return out


def _prof(prof, name):
    return prof.get(name) if isinstance(prof, dict) else getattr(prof, name, None)


def _subdivide(e, j0, j1, sub):
    """Fine edges over coarse cells [j0, j1) of edges e, `sub` per cell, and each fine cell's coarse index."""
    fe = np.concatenate([np.linspace(e[j], e[j + 1], sub + 1)[:-1] for j in range(j0, j1)] + [e[j1:j1 + 1]])
    return fe, np.repeat(np.arange(j0, j1), sub)


def profile_section(prof, mat, orig, ye, ze, sub=SUBCELL):
    """The fillet's outer shape (legs, throat, size, convexity) measured on the pool's free surface of this slice
    (`prof`: pool.SliceState or a dict with A, B, u, h, chord_t, chord_n, in the weld frame's y-z plane), checked
    against the slice's cells (2-D maps [bj, bk] with y/z cell edges, as tee_section). None if the slice has no
    profile or its toes are not on the tee's members (then the cells alone are read).

    Each cell of the fillet quadrant is compared with what the profile set: weld metal iff its centre lies under the
    surface (pool.Pool._apply). Weld cells the profile does not explain ("extra": stacked or spilled onto the slice
    later, frozen outside a surface that drew back) are kept whole; cells the profile claims that are not metal
    ("missing") are removed; beyond the toes, where the profile says nothing, weld cells are kept whole. The section
    is then rasterised at 1/sub of a cell with the profile as the surface between the toes, and:
      legs   -- the toes A, B where the free surface meets each member (the pool's 0.05 mm toe sampling), unless
                whole extra cells carry weld metal up a member beyond the toe (their edge is then the leg, as on the
                cells) or the metal along the member stops short of the toe (a removed cell: the raster's run);
      size   -- the legs of the largest isosceles right triangle inscribed in the section (AWS A3.0 / D1.1): the
                smaller leg, or the face's nearest approach to the root along the triangle's hypotenuse y + z = s;
      throat -- the shortest distance from the root to the face (the actual throat);
      convexity / concavity -- the face's largest distance outside / inside the toe-to-toe line (D1.1 Fig 5.4)."""
    A, B, u, h = (_prof(prof, k) for k in ("A", "B", "u", "h"))
    t, n = _prof(prof, "chord_t"), _prof(prof, "chord_n")
    if any(v is None for v in (A, B, u, h, t, n)):
        return None
    A, B, u, h, t, n = (np.asarray(v, float) for v in (A, B, u, h, t, n))
    kind = _prof(prof, "kind") or "h"
    rr = _prof(prof, "r")
    dep = _prof(prof, "depth") or 0.
    pts = _prof(prof, "pts")
    pts = np.asarray(pts, float) if pts is not None else A[None, :] + u[:, None]*t[None, :] + h[:, None]*n[None, :]
    eps = 1e-6

    def member(p):
        if abs(p[0]) < eps and p[1] > -eps:
            return "bracket"
        if abs(p[1]) < eps and p[0] > -eps:
            return "base"
        return None
    if member(A) is None or member(B) is None:
        return None
    Lc = float(u[-1])
    yc, zc = .5*(ye[1:] + ye[:-1]), .5*(ze[1:] + ze[:-1])
    jb, kb = int(np.searchsorted(yc, 0.)), int(np.searchsorted(zc, 0.))
    Y, Z = np.meshgrid(yc, zc, indexing="ij")

    def surface(Yq, Zq):
        """(described, under, near): where the profile describes the section (its span; for an overhanging profile
        also everything on the chord's outer side), the points under its surface, and the points on it (within
        ON_SURFACE of the polyline). Both surface forms (pool.under_surface)."""
        und, span_, ww_ = PL.under_surface(kind, A, t, n, Lc, Yq, Zq, u=u, h=h, r=rr, depth=dep)
        described = span_ | ((ww_ >= 0) if kind == "polar" else False)
        return described, und, _near_polyline(pts, Yq, Zq, ON_SURFACE)
    span, under_, near_ = surface(Y, Z)
    q = (Y > 0) & (Z > 0)
    weld = mat == TH.WELD
    pred = span & under_ & ~orig & q
    act = weld & ~orig & q
    near = span & near_
    extra = act & ~pred & ~near
    missing = pred & ~act & ~near
    # the raster window: the quadrant out to the farthest weld metal, toe or profile point, plus two cells
    wy = max(float(np.max(pts[:, 0])), float(A[0]), float(B[0]), float(Y[act].max()) if act.any() else 0.)
    wz = max(float(np.max(pts[:, 1])), float(A[1]), float(B[1]), float(Z[act].max()) if act.any() else 0.)
    J0, K0 = max(jb - 1, 0), max(kb - 1, 0)
    J1 = min(int(np.searchsorted(ye, wy)) + 2, len(yc))
    K1 = min(int(np.searchsorted(ze, wz)) + 2, len(zc))
    fy, pj = _subdivide(ye, J0, J1, sub)
    fz, pk = _subdivide(ze, K0, K1, sub)
    fyc, fzc = .5*(fy[1:] + fy[:-1]), .5*(fz[1:] + fz[:-1])
    Yf, Zf = np.meshgrid(fyc, fzc, indexing="ij")
    ix = np.ix_(pj, pk)
    orig_f, mat_c = orig[ix], mat[ix]
    span_f, under_f, _ = surface(Yf, Zf)
    qf = (Yf > 0) & (Zf > 0)
    # the profile describes the surface between its toes; beyond them (outside its span) the cells are the only
    # description, so weld cells there enter whole, as do the extra cells and refilled parent metal
    weld_f = (span_f & under_f & ~orig_f & qf) | (~span_f & (mat_c == TH.WELD)) | extra[ix] \
        | ((mat_c == TH.WELD) & orig_f)
    weld_f &= ~missing[ix]
    metal_f = weld_f | (orig_f & (mat_c != TH.AIR))
    # legs: the raster's contiguous run of weld metal along each member from the root, set against the toes
    jbf, kbf = int(np.searchsorted(fyc, 0.)), int(np.searchsorted(fzc, 0.))
    col, row = weld_f[jbf, kbf:], weld_f[jbf:, kbf]
    run_v = float(fz[kbf + (int(np.argmin(col)) if not col.all() else len(col))]) if col[0] else 0.
    run_h = float(fy[jbf + (int(np.argmin(row)) if not row.all() else len(row))]) if row[0] else 0.
    toe_v = max([p[1] for p in (A, B) if member(p) == "bracket"], default=0.)
    toe_h = max([p[0] for p in (A, B) if member(p) == "base"], default=0.)

    def cell_run(line, edges, k0):
        """The cells' own contiguous run of weld metal along a member from the root (whole cells, as tee_section)."""
        return float(edges[k0 + (int(np.argmin(line)) if not line.all() else len(line))]) if line[0] else 0.

    def leg(run, run_c, toe, extra_line, edges_hi, cell):
        if run <= 0. and run_c <= 0.:
            return 0., "cells"
        beyond = extra_line & (edges_hi > toe + 1e-9)     # whole extra cells along the member past the toe
        if beyond.any() and run_c >= float(edges_hi[beyond].min()) - 1e-9:
            return run_c, "cells"                           # weld metal the cells hold beyond the toe (cold lap, stack)
        if run < toe - cell - 1e-9:                         # the metal along the member stops short of the toe
            return run, "cells"
        return toe, "profile"
    leg_v, src_v = leg(run_v, cell_run(weld[jb, kb:], ze, kb), toe_v, extra[jb, :], ze[1:], float(ze[kb + 1] - ze[kb]))
    leg_h, src_h = leg(run_h, cell_run(weld[jb:, kb], ye, jb), toe_h, extra[:, kb], ye[1:], float(ye[jb + 1] - ye[jb]))
    r = dict(leg_vertical_m=leg_v, leg_horizontal_m=leg_h, leg_vertical_from=src_v, leg_horizontal_from=src_h,
             profile_extra_cells=int(extra.sum()), profile_missing_cells=int(missing.sum()))
    air_f = ~metal_f & qf
    nb = np.zeros_like(weld_f)
    nb[1:, :] |= weld_f[:-1, :]
    nb[:-1, :] |= weld_f[1:, :]
    nb[:, 1:] |= weld_f[:, :-1]
    nb[:, :-1] |= weld_f[:, 1:]
    face = air_f & nb & (Yf <= leg_h + 2e-3) & (Zf <= leg_v + 2e-3)
    dpx = float(min(np.diff(fy).min(), np.diff(fz).min()))
    half = .5*dpx
    if not (leg_v > 0 and leg_h > 0 and face.any()):
        r.update(throat_m=0., size_m=0., convexity_m=0., concavity_m=0., face_width_m=0.)
        return r
    P0, P1 = np.array([0., leg_v]), np.array([leg_h, 0.])
    tl = (P1 - P0)/np.linalg.norm(P1 - P0)
    nl = np.array([tl[1], -tl[0]])
    if nl @ np.array([1., 1.]) < 0:
        nl = -nl
    wl = float(np.linalg.norm(P1 - P0))
    # on the raster (pixel centres; a pixel is air when its centre lies outside the face): throat, the inscribed
    # triangle y + z < s that holds no air, and the face's distance from the toe-to-toe line, each to half a pixel
    sd = (Yf - P0[0])*nl[0] + (Zf - P0[1])*nl[1]
    ud = (Yf - P0[0])*tl[0] + (Zf - P0[1])*tl[1]
    on = face & (ud > 0) & (ud < wl)
    s_air = float((Yf + Zf)[air_f].min()) - half if air_f.any() else np.inf
    ras = dict(throat_m=float(max(0., (np.sqrt(Yf**2 + Zf**2) - half)[face].min())),
               size_m=float(max(0., min(leg_v, leg_h, s_air))),
               convexity_m=float(max(0., (sd - half)[on].max())) if on.any() else 0.,
               concavity_m=float(max(0., -(sd - half)[on].min())) if on.any() else 0.)
    # on the profile itself (a dense polyline of the surface, exact): the same four
    f40 = np.linspace(0., 1., 41)[:-1]
    P = (pts[:-1, None, :] + f40[None, :, None]*np.diff(pts, axis=0)[:, None, :]).reshape(-1, 2)
    P = np.concatenate([P, pts[-1:]], axis=0)
    P = P[(P[:, 0] >= 0) & (P[:, 1] >= 0)]
    sp = (P[:, 0] - P0[0])*nl[0] + (P[:, 1] - P0[1])*nl[1]
    up = (P[:, 0] - P0[0])*tl[0] + (P[:, 1] - P0[1])*tl[1]
    onp = (up > 0) & (up < wl)
    pol = dict(throat_m=float(np.hypot(P[:, 0], P[:, 1]).min()),
               size_m=float(max(0., min(leg_v, leg_h, float((P[:, 0] + P[:, 1]).min())))),
               convexity_m=float(max(0., sp[onp].max())) if onp.any() else 0.,
               concavity_m=float(max(0., -sp[onp].min())) if onp.any() else 0.)
    if not missing.any() and not extra.any() and src_v == src_h == "profile":
        r.update(pol)                                   # the cells are exactly the profile's: read the surface
    elif not missing.any():
        # whole cells added on top of the profile: they can only add metal, so the throat and size are the profile's
        # unless the raster finds the added cells raising them by more than its own half-pixel resolution
        r.update(throat_m=max(pol["throat_m"], ras["throat_m"] - half), size_m=max(pol["size_m"], ras["size_m"] - half),
                 convexity_m=ras["convexity_m"], concavity_m=ras["concavity_m"])
    else:
        r.update(ras)                                   # cells missing under the profile: the raster of both
    r["face_width_m"] = wl
    return r


def tee_section(mat, orig, peakf, peakT, ye, ze, profile=None):
    """One near-side tee section (2-D maps [bj, bk] with y/z cell edges). Lengths in m. `profile`: the slice's
    free-surface profile (pool.SliceState) if it has one: the shape (GEOMETRY_KEYS) is then measured on it
    (profile_section) and the cells' own reading is kept as <key>_cells; `geometry` says which was used."""
    yc, zc = .5*(ye[1:] + ye[:-1]), .5*(ze[1:] + ze[:-1])
    dy, dz = np.diff(ye), np.diff(ze)
    jb = int(np.searchsorted(yc, 0.))          # first column on the weld side of the bracket face (y > 0)
    kb = int(np.searchsorted(zc, 0.))          # first row above the base plate top (z > 0)
    weld = mat == TH.WELD
    metal = mat != TH.AIR
    fused = orig & (peakf >= FUSED) & metal
    r = {}
    area = float(np.sum(weld*(dy[:, None]*dz[None, :])))
    r["weld_area_m2"] = area
    # legs: contiguous weld metal along the bracket face (up) and along the base plate (out) from the root
    col = weld[jb, kb:]
    nv = int(np.argmin(col)) if not col.all() else len(col)
    leg_v = float(ze[kb + nv] - 0.) if nv > 0 else 0.
    row = weld[jb:, kb]
    nh = int(np.argmin(row)) if not row.all() else len(row)
    leg_h = float(ye[jb + nh] - 0.) if nh > 0 else 0.
    r["leg_vertical_m"], r["leg_horizontal_m"] = leg_v, leg_h
    # face: air cells in the fillet quadrant touching weld metal; throat = shortest root-to-face distance
    q = (yc[:, None] > 0) & (zc[None, :] > 0)
    air = ~metal & q
    nb = np.zeros_like(weld)
    nb[1:, :] |= weld[:-1, :]
    nb[:-1, :] |= weld[1:, :]
    nb[:, 1:] |= weld[:, :-1]
    nb[:, :-1] |= weld[:, 1:]
    face = air & nb
    Y, Z = np.meshgrid(yc, zc, indexing="ij")
    half = .5*np.sqrt(dy[:, None]**2 + dz[None, :]**2)*0. + .5*np.minimum(dy[:, None], dz[None, :])
    if face.any() and leg_v > 0 and leg_h > 0:
        d = np.sqrt(Y**2 + Z**2) - half
        # only face cells between the toes (inside the triangle's bounding box widened by 2 mm) belong to this fillet
        inside = face & (Y <= leg_h + 2e-3) & (Z <= leg_v + 2e-3)
        r["throat_m"] = float(d[inside].min()) if inside.any() else 0.
        # convexity: largest distance of the face outward from the toe-to-toe line (D1.1 Fig 5.4)
        A, B = np.array([0., leg_v]), np.array([leg_h, 0.])
        t = (B - A)/np.linalg.norm(B - A)
        n = np.array([t[1], -t[0]])
        if n @ np.array([1., 1.]) < 0:
            n = -n
        s_ = (Y - A[0])*n[0] + (Z - A[1])*n[1] - half
        u_ = (Y - A[0])*t[0] + (Z - A[1])*t[1]
        span = inside & (u_ > 0) & (u_ < np.linalg.norm(B - A))
        r["convexity_m"] = float(max(0., s_[span].max())) if span.any() else 0.
        r["concavity_m"] = float(max(0., -s_[span].min())) if span.any() else 0.
        r["face_width_m"] = float(np.linalg.norm(B - A))
    else:
        r["throat_m"] = r["convexity_m"] = r["concavity_m"] = r["face_width_m"] = 0.
    # the fillet's size: the largest isosceles right triangle inscribed in the section (D1.1: legs, or the throat
    # for a concave face)
    r["size_m"] = float(min(leg_v, leg_h, r["throat_m"]/math.sqrt(.5))) if leg_v > 0 and leg_h > 0 else 0.
    # undercut: parent metal that melted and left as a groove (now air), measured from the original surface
    gone = orig & ~metal
    base_g = gone & (zc[None, :] < 0)
    brk_g = gone & (yc[:, None] < 0) & (zc[None, :] >= 0)
    r["undercut_base_m"] = float((-(ze[:-1])[None, :]*base_g).max()) if base_g.any() else 0.
    r["undercut_bracket_m"] = float((-(ye[:-1])[:, None]*brk_g).max()) if brk_g.any() else 0.
    if brk_g.any():
        r["undercut_bracket_z_m"] = float(Z[brk_g].max())
    if base_g.any():
        r["undercut_base_y_m"] = float(Y[base_g].max())
    # overlap (cold lap): weld metal beyond the fusion boundary over parent metal that never fused, lying on it or
    # standing over it (D1.1 Fig 5.4: weld metal protruding beyond the toe). Per column of the base plate, the plate's
    # top cell unfused and weld metal anywhere above it; per row of the bracket face, its face cell unfused and weld
    # metal anywhere in the row. (Counting only the cell lying on the unfused metal would read a face that overhangs
    # its toe, with an air wedge under it, as no overlap.)
    base_open = orig[jb:, kb - 1] & (peakf[jb:, kb - 1] < FUSED) & metal[jb:, kb - 1]
    r["overlap_base_m"] = float(np.sum((base_open & weld[jb:, kb:].any(1))*dy[jb:]))
    face_open = orig[jb - 1, kb:] & (peakf[jb - 1, kb:] < FUSED) & metal[jb - 1, kb:]
    r["overlap_bracket_m"] = float(np.sum((face_open & weld[jb:, kb:].any(0))*dz[kb:]))
    # fusion (the etched macro section): the root corner melted, and penetration into each member
    corner = [(jb, kb - 1), (jb - 1, kb)]           # base cell under the root, bracket cell beside it
    r["root_fused"] = bool(all(peakf[j, k] >= FUSED or (orig[j, k] and mat[j, k] == TH.WELD) for j, k in corner))
    fb = fused & (zc[None, :] < 0) & (yc[:, None] > -1e-9)
    r["penetration_base_m"] = float((-(ze[:-1])[None, :]*fb).max()) if fb.any() else 0.
    fk = fused & (yc[:, None] < 0) & (zc[None, :] > -1e-9)
    r["penetration_bracket_m"] = float((-(ye[:-1])[:, None]*fk).max()) if fk.any() else 0.
    # refilled parent metal (melted, displaced and replaced by weld metal) also lies inside the fusion boundary
    refill = orig & (mat == TH.WELD)
    if refill.any():
        rb = refill & (zc[None, :] < 0)
        rk = refill & (yc[:, None] < 0)
        if rb.any():
            r["penetration_base_m"] = max(r["penetration_base_m"], float((-(ze[:-1])[None, :]*rb).max()))
        if rk.any():
            r["penetration_bracket_m"] = max(r["penetration_bracket_m"], float((-(ye[:-1])[:, None]*rk).max()))
    r["fused_parent_area_m2"] = float(np.sum((fused | refill)*(dy[:, None]*dz[None, :])))
    haz = orig & metal & (peakT >= HAZ_C) & (peakf < FUSED)
    r["haz_base_depth_m"] = float((-(ze[:-1])[None, :]*(haz & (zc[None, :] < 0))).max()) if haz.any() else 0.
    r["geometry"] = "cells"
    if profile is not None and r["weld_area_m2"] > 0:
        p = profile_section(profile, mat, orig, ye, ze)
        if p is not None:
            for k in GEOMETRY_KEYS:
                r[k[:-2] + "_cells_m"] = r[k]
            r.update(p)
            r["geometry"] = "profile"
    return r


def weld_report(proc, x_range=None):
    """Per-slice sections along the weld (tee) and summary statistics. x_range: (x0, x1) m of slices to read."""
    g = proc.g
    if x_range is None:
        x_range = (g.xe[0], g.xe[-1])
    i0 = int(np.searchsorted(g.xc, x_range[0]))
    i1 = int(np.searchsorted(g.xc, x_range[1]))
    M = section_maps(proc, i0, i1)
    rows = []
    for n in range(i1 - i0):
        st = proc.pool.slices[i0 + n]
        r = tee_section(M["mat"][n], M["orig"][n], M["peakf"][n], M["peakT"][n], M["ye"], M["ze"],
                        profile=st if st.h is not None else None)
        r["x_m"] = float(M["xc"][n])
        r["dx_m"] = float(M["dx"][n])
        r["porosity"] = float(st.porosity)
        r["x_air_max"] = float(st.x_air_max)
        r["liquid"] = bool(np.any(M["fl"][n] >= FUSED))
        r["overflow_toe_ext"] = list(st.toe_ext)
        r["freeze_t"] = st.freeze_t                    # s: when the slice last froze (None: never, or liquid now)
        rows.append(r)
    return rows


def floating_weld(proc):
    """Weld metal with nothing to stand on: the weld cells' face-connected components (3-D, the whole grid) that have no
    face on steel (the plate or the bracket). Weld metal exists only on the metal it was laid on; a component without a
    face on steel hangs in the air (the evaluator would read such pieces as overlap). Returns dict(pieces, volume_m3,
    largest: the biggest piece's cells, volume, liquid cells and x / y / z extent in m)."""
    from scipy import ndimage
    g = proc.g
    mat = proc.s.mat.reshape(g.nx, g.ny, g.nz)
    weld = mat == TH.WELD
    lab, n = ndimage.label(weld)
    if not n:
        return dict(pieces=0, volume_m3=0., largest=None)
    on_steel = ndimage.binary_dilation(mat == TH.STEEL)
    held = np.unique(lab[weld & on_steel])
    loose = np.setdiff1d(np.arange(1, n + 1), held)
    if not len(loose):
        return dict(pieces=0, volume_m3=0., largest=None)
    vol = g.dx[:, None, None]*g.dy[None, :, None]*g.dz[None, None, :]
    vols = ndimage.sum(vol, lab, loose)
    big = int(loose[int(np.argmax(vols))])
    I, J, K = np.nonzero(lab == big)
    fl = proc.s.gather("fl", g.index(I, J, K))
    largest = dict(cells=int(len(I)), volume_m3=float(np.max(vols)), liquid_cells=int(np.sum(fl >= FUSED)),
                   x_m=[float(g.xc[I].min()), float(g.xc[I].max())], y_m=[float(g.yc[J].min()), float(g.yc[J].max())],
                   z_m=[float(g.zc[K].min()), float(g.zc[K].max())])
    return dict(pieces=int(len(loose)), volume_m3=float(np.sum(vols)), largest=largest)


def summarize(rows, nominal_leg=6.35e-3):
    """Weld-length statistics from section rows: the continuous fused length, leg sizes along it, profile and
    discontinuities (the inputs of the task criteria; thresholds live in task.py)."""
    if not rows:
        return {}
    welded = [r for r in rows if r["weld_area_m2"] > 0 and r["root_fused"]]
    x = np.array([r["x_m"] for r in rows])
    ok = np.array([r["weld_area_m2"] > 0 and r["root_fused"] for r in rows])
    # longest contiguous run of welded slices
    best, cur, start, bs = 0, 0, 0, 0
    for k, v in enumerate(ok):
        if v:
            if cur == 0:
                start = k
            cur += 1
            if cur > best:
                best, bs = cur, start
        else:
            cur = 0
    run = rows[bs:bs + best]
    out = dict(sections=len(rows), welded_sections=len(welded))
    if not run:
        return out
    dxs = np.array([r["dx_m"] for r in run])
    L = float(dxs.sum())
    legs = np.array([min(r["leg_vertical_m"], r["leg_horizontal_m"]) for r in run])
    size = np.array([r["size_m"] for r in run])
    out.update(
        continuous_from_m=float(run[0]["x_m"] - run[0]["dx_m"]/2), continuous_to_m=float(run[-1]["x_m"] + run[-1]["dx_m"]/2),
        continuous_length_m=L,
        leg_vertical_median_m=float(np.median([r["leg_vertical_m"] for r in run])),
        leg_horizontal_median_m=float(np.median([r["leg_horizontal_m"] for r in run])),
        size_median_m=float(np.median(size)), size_min_m=float(size.min()),
        frac_length_size_ge_nominal=float(np.sum(dxs*(size >= nominal_leg - 1e-9))/L),
        throat_median_m=float(np.median([r["throat_m"] for r in run])),
        leg_difference_max_m=float(max(abs(r["leg_vertical_m"] - r["leg_horizontal_m"]) for r in run)),
        convexity_max_m=float(max(r["convexity_m"] for r in run)),
        concavity_max_m=float(max(r["concavity_m"] for r in run)),
        undercut_max_m=float(max(max(r["undercut_base_m"], r["undercut_bracket_m"]) for r in run)),
        undercut_over_1mm_length_m=float(sum(r["dx_m"] for r in run if max(r["undercut_base_m"], r["undercut_bracket_m"]) > 1e-3)),
        overlap_max_m=float(max(max(r["overlap_base_m"], r["overlap_bracket_m"]) for r in run)),
        penetration_base_min_m=float(min(r["penetration_base_m"] for r in run)),
        penetration_bracket_min_m=float(min(r["penetration_bracket_m"] for r in run)),
        porosity_max=float(max(r["porosity"] for r in run)),
        weld_area_median_m2=float(np.median([r["weld_area_m2"] for r in run])),
        weld_metal_volume_m3=float(sum(r["weld_area_m2"]*r["dx_m"] for r in rows)),
        still_liquid_sections=int(sum(r["liquid"] for r in rows)),
    )
    return out
