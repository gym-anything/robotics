"""Gift wrap around the two-piece box: crease pattern, isometric fold map and taped start state.

Technique (Hallmark how-to): the sheet goes around the box's girth with a 40 mm overlap seam under the box;
at each end the side flaps fold in first, which tucks a 45-degree triangle under each corner of the top and
bottom flaps; the top flap folds down, the bottom flap folds up over it and is taped. One tape strip holds the
seam, one holds each end (Scotch Magic 810, 19 mm).

Material coordinates (u, s): u across the box length (x), s around the girth starting under the box at
y = -20 mm and running bottom -> back (+y) -> top -> front (-y) -> bottom. The map is isometric per facet.
Layers counted outward from the box surface:
faces 0 (seam outer strip 1 where it overlaps the inner strip); end faces in folding order: side flaps 0,
the top flap's tucked corner triangles 1, top flap 2, the bottom flap's tucked triangles 3, bottom flap 4
(5 where the seam's outer strip overlaps).
"""

import math
import numpy as np
from . import ipcworld as W
from . import params as Pm
from . import giftbox as GB

SEAM_OVERLAP = .040          # m [G, Hallmark "a couple of inches" -> 40 mm, A]
END_FLAP = .042              # m, overhang beyond each end [A: > half the box height so the end flaps overlap]
TAPE_W = .019                # m, Scotch Magic 810 3/4 in [V]
DEFAULT_STOCK = "commercial_81gsm"   # params.WRAP_STOCKS key of the authored wrap (Pm.WRAP)


def cuboid(box=None):
    """The wrapped outline: lid footprint and the closed box height (base outer bottom at z = 0). `box`: a
    giftbox.BoxSpec (default: the authored box)."""
    box = box or GB.box_spec()
    Lx = 2*(box.hx + box.clearance + box.wall)
    b = box.hy + box.clearance + box.wall
    Hc = box.h + box.wall + 5e-5
    return Lx, b, Hc


def net_rules(Lx, b, Hc, e=END_FLAP, ov=SEAM_OVERLAP):
    """The folding rules the net, its fold map and its tapes need, as {rule: bool}:
    end_flaps_overlap   the top and bottom end flaps overlap (2 e > Hc: "more than half the box height", Hallmark)
    end_tape_on_face    the end tape centred on the bottom flap's edge (z = e) stays on the end face (e + TAPE_W/2 <= Hc)
    side_flaps_fit      the side flaps' 45-degree triangles leave a top/bottom flap between them (e < b)
    seam_under_box      the girth seam overlap lies under the box (ov/2 < b - e, inside the bottom face's middle)"""
    return dict(end_flaps_overlap=2*e > Hc, end_tape_on_face=e + TAPE_W/2 <= Hc, side_flaps_fit=e < b,
                seam_under_box=ov/2 < b - e)


def _lines(segments):
    out = [0.]
    for length, n in segments:
        base = out[-1]
        out.extend(base + length*(np.arange(1, n + 1)/n))
    return np.asarray(out)


def wrap_mesh(cell=.006, box=None):
    Lx, b, Hc = cuboid(box)
    e, ov = END_FLAP, SEAM_OVERLAP
    bad = [k for k, ok in net_rules(Lx, b, Hc, e, ov).items() if not ok]
    if bad:
        raise ValueError(f"wrap net for box {Lx:.4f} x {2*b:.4f} x {Hc:.4f} m breaks the folding rules {bad} (net_rules)")
    ne = int(round(e/cell))
    c = e/ne
    nL = int(round(Lx/c))
    ulines = _lines([(e, ne), (Lx, nL), (e, ne)])
    y0 = -ov/2                                         # inner strip starts here (seam centred under the box)
    seg = dict(s0=0., s1=b - y0, s2=b - y0 + Hc, s3=b - y0 + Hc + 2*b, s4=b - y0 + 2*Hc + 2*b,
               s5=2*b - 2*y0 + 2*Hc + 2*b)
    nH = int(math.ceil(Hc/c))
    sl = _lines([((b - e) - y0, max(2, int(round(((b - e) - y0)/c)))), (e, ne),          # bottom inner
                 (Hc, nH),                                                                 # back
                 (e, ne), (2*(b - e), max(2, int(round(2*(b - e)/c)))), (e, ne),         # top
                 (Hc, nH),                                                                 # front
                 (e, ne), ((b - e) + ov/2, max(2, int(round(((b - e) + ov/2)/c))))])     # bottom outer
    assert abs(sl[-1] - seg["s5"]) < 1e-9
    U, S = np.meshgrid(ulines, sl, indexing="ij")
    X2 = np.stack([U.ravel(), S.ravel()], 1)
    nu, ns = len(ulines), len(sl)
    idx = np.arange(nu*ns).reshape(nu, ns)
    geo = dict(Lx=Lx, b=b, Hc=Hc, e=e, y0=y0, seg=seg)

    def on_diag(p):
        # a grid point on a strip boundary belongs to both strips: test either side
        for ds in (-1e-9, 1e-9):
            d, face, y = _end_coords((p[0], p[1] + ds), geo)
            if d is not None and face in ("top", "bottom") and abs(abs(y) - (b - d)) < 1e-8:
                return True
        return False
    T = []
    for i in range(nu - 1):
        for j in range(ns - 1):
            A, B, C, D = idx[i, j], idx[i+1, j], idx[i+1, j+1], idx[i, j+1]
            if on_diag(X2[A]) and on_diag(X2[C]):
                T += [(A, B, C), (A, C, D)]
            elif on_diag(X2[B]) and on_diag(X2[D]):
                T += [(A, B, D), (B, C, D)]
            elif (i + j) % 2 == 0:
                T += [(A, B, C), (A, C, D)]
            else:
                T += [(A, B, D), (B, C, D)]
    return X2, np.asarray(T), geo


def _face(s, geo):
    sg, b, y0, Hc = geo["seg"], geo["b"], geo["y0"], geo["Hc"]
    if s <= sg["s1"]:
        return "bottom_in", y0 + s
    if s <= sg["s2"]:
        return "back", s - sg["s1"]
    if s <= sg["s3"]:
        return "top", b - (s - sg["s2"])
    if s <= sg["s4"]:
        return "front", Hc - (s - sg["s3"])
    return "bottom_out", -b + (s - sg["s4"])


def _end_coords(p, geo):
    """(d, face, y) for a point in an end region (d = distance beyond the end edge), else (None, ...)."""
    u, s = p
    e, Lx = geo["e"], geo["Lx"]
    face, w = _face(s, geo)
    if u < e + 1e-12:
        d = max(e - u, 0.)
    elif u > e + Lx - 1e-12:
        d = max(u - e - Lx, 0.)
    else:
        return None, face, w
    return d, ("bottom" if face.startswith("bottom") else face), w


def _map(p, cen, geo):
    """Flat-state point: (region, a, b_, depth) with region 'body:<face>' (a = x, b_ = face coordinate) or
    'end:<side>' (a = y_e, b_ = z_e). Facet chosen by the triangle centroid `cen`."""
    u, s = p
    e, Lx, b, Hc, y0 = geo["e"], geo["Lx"], geo["b"], geo["Hc"], geo["y0"]
    face_c, wc = _face(cen[1], geo)
    face, w = face_c, None
    # coordinate of p along the centroid's face
    sg = geo["seg"]
    if face_c == "bottom_in":
        w = y0 + s
    elif face_c == "back":
        w = s - sg["s1"]
    elif face_c == "top":
        w = b - (s - sg["s2"])
    elif face_c == "front":
        w = Hc - (s - sg["s3"])
    else:
        w = -b + (s - sg["s4"])
    if e - 1e-12 <= cen[0] <= e + Lx + 1e-12:
        x = u - e - Lx/2
        depth = 0.
        if face_c == "bottom_out" and w > y0 - 1e-9:
            depth = 1.
        return "body:" + face_c, x, w, depth
    side = "left" if cen[0] < e else "right"
    d = (e - u) if side == "left" else (u - e - Lx)
    if face_c in ("back", "front"):
        ye = b - d if face_c == "back" else -b + d
        return f"end:{side}", ye, w, 0.
    _, _, wc_ = None, None, None
    dc = (e - cen[0]) if side == "left" else (cen[0] - e - Lx)
    yc = wc
    if face_c == "top":
        if yc > b - dc:
            return f"end:{side}", b - d, Hc - (b - w), 1.
        if yc < -b + dc:
            return f"end:{side}", -b + d, Hc - (w + b), 1.
        return f"end:{side}", w, Hc - d, 2.
    # bottom strips (folded last: their tucked triangles lie over the top flap)
    if yc > b - dc:
        return f"end:{side}", b - d, b - w, 3.
    if yc < -b + dc:
        return f"end:{side}", -b + d, w + b, 3.
    extra = 1. if (face_c == "bottom_out" and w > y0 - 1e-9) else 0.
    return f"end:{side}", w, d, 4. + extra


START_GAP = 5e-6             # m, every start layer sits this far outside its barrier range (zero initial force)


def build(base_origin, *, cell=.008, md="girth", seed=0, box=None, stock=DEFAULT_STOCK):
    """Wrap sheet in its folded, taped start state around the box whose base outer bottom centre is at
    `base_origin` (box axes = world axes). Layers are one caliper apart (contact_caliper rest distances), the
    first one at the sheet-on-rigid rest distance from the box. `box`: giftbox.BoxSpec (default the authored box);
    `stock`: a params.WRAP_STOCKS key. Returns (sheet, geo)."""
    X2, T, geo = wrap_mesh(cell, box)
    Lx, b, Hc = geo["Lx"], geo["b"], geo["Hc"]
    mat, crease_law = Pm.WRAP_STOCKS[stock]
    cal = W.contact_caliper(mat)
    # layer pitch: at a fold the vertices take the mean depth of their facets, and where four layers meet the body
    # within one cell the tilted facets come within ~0.41 pitch of each other; 2.8 dmin keeps every pair beyond
    # the incompressible core distance (66 um > 54 um)
    pitch = max(cal["rest_sheet_sheet"], 2.8*cal["dmin"]) + 2*START_GAP
    off = lambda depth: cal["rest_sheet_rigid"] + START_GAP + depth*pitch
    o = np.asarray(base_origin, float)
    flat = {}
    region = np.empty(len(T), object)
    depth_t = np.zeros(len(T))
    pts = np.full((len(X2), 3), np.nan)
    acc_d, cnt = np.zeros(len(X2)), np.zeros(len(X2))
    for t, tri in enumerate(T):
        cen = X2[tri].mean(0)
        for v in tri:
            reg, a, bb, dep = _map(X2[v], cen, geo)
            key = (v, reg)
            # in-plane position (independent of depth) must agree for every facet sharing a vertex
            if reg.startswith("body:"):
                f = reg[5:]
                if f.startswith("bottom"):
                    pin = (a, bb, 0.)
                elif f == "back":
                    pin = (a, b, bb)
                elif f == "top":
                    pin = (a, bb, Hc)
                else:
                    pin = (a, -b, bb)
            else:
                sx = -1. if reg.endswith("left") else 1.
                pin = (sx*Lx/2, a, bb)
            if v in flat and np.abs(np.asarray(flat[v]) - pin).max() > 1e-7:
                raise AssertionError(f"inconsistent wrap map at vertex {v}: {flat[v]} vs {pin}")
            flat[v] = pin
        region[t] = _map(X2[tri[0]], cen, geo)[0]
        depth_t[t] = max(_map(X2[v], cen, geo)[3] for v in tri)
        for v in tri:
            acc_d[v] += depth_t[t]; cnt[v] += 1
    depth_v = acc_d/np.maximum(cnt, 1)
    # outward offset by depth: body faces along their normals, end faces along +-x (edges: both)
    for v, (px, py, pz) in flat.items():
        dv = off(depth_v[v])
        q = np.array([px, py, pz])
        n = np.zeros(3)
        if abs(abs(px) - Lx/2) < 1e-9:
            n[0] = np.sign(px)
        if abs(abs(py) - b) < 1e-9:
            n[1] = np.sign(py)
        if abs(pz) < 1e-9:
            n[2] = -1.
        if abs(pz - Hc) < 1e-9:
            n[2] = 1.
        pts[v] = q + n*dv
    rng = np.random.default_rng(seed)
    x0 = pts + o + rng.uniform(-2e-6, 2e-6, pts.shape)
    sheet = W.Sheet("wrap", X2, T, x0, [mat], np.zeros(len(T), int), md_angle=(math.pi/2 if md == "girth" else 0.),
                    fmat="wrap")
    # creases: box edges (girth segment lines inside the body span, end-face edges) and the end diagonals
    creases = {}
    E = W.edges_of(T)
    sg = geo["seg"]
    s_lines = [sg["s1"], sg["s2"], sg["s3"], sg["s4"]]
    for i, j in E:
        a, c_ = X2[i], X2[j]
        on_s = any(abs(a[1] - L) < 1e-9 and abs(c_[1] - L) < 1e-9 for L in s_lines)
        on_u = any(abs(a[0] - L) < 1e-9 and abs(c_[0] - L) < 1e-9 for L in (geo["e"], geo["e"] + Lx))
        in_body_u = (geo["e"] - 1e-9 <= a[0] <= geo["e"] + Lx + 1e-9) and (geo["e"] - 1e-9 <= c_[0] <= geo["e"] + Lx + 1e-9)
        diag = abs(a[0] - c_[0]) > 1e-9 and abs(a[1] - c_[1]) > 1e-9
        if (on_s and in_body_u) or on_u or diag:
            creases[(int(i), int(j))] = crease_law
    sheet.creases = creases
    sheet.render = dict(kind="wrap", region=region, depth=depth_t)
    geo.update(base_origin=o, off=off, pitch=pitch, stock=stock)
    return sheet, geo


def tape_sheets(wrap_x, wrap_T, geo, *, cell=.0025):
    """Three Magic-tape strips laid on the outermost layer: one across the seam edge under the box (along x),
    one across each end's bottom-flap edge (along y). Each tape vertex sits on the paper surface it covers,
    at the bond rest offset, so every bond starts unloaded."""
    Lx, b, Hc, e, y0 = geo["Lx"], geo["b"], geo["Hc"], geo["e"], geo["y0"]
    o = geo["base_origin"]
    d0 = W.contact_caliper(Pm.WRAP_STOCKS[geo.get("stock", DEFAULT_STOCK)][0])["rest_sheet_sheet"] + START_GAP
    specs = []
    # seam: centred on the outer strip's edge (y = -y0) under the box, 60 mm along x
    specs.append(("tape_seam", np.array([0., -y0, 0.]), np.array([1., 0, 0]), np.array([0, 1., 0]), .060, np.array([0, 0, -1.])))
    for sx, nm in ((-1., "tape_left"), (1., "tape_right")):
        specs.append((nm, np.array([sx*Lx/2, 0., e]), np.array([0, 1., 0]), np.array([0, 0, 1.]), .050, np.array([sx, 0, 0.])))
    tapes = []
    T = wrap_T
    for nm, c, along, across, L, nrm in specs:
        nu, nv = int(round(L/cell)) + 1, int(round(TAPE_W/cell)) + 1
        uu, vv = np.linspace(-L/2, L/2, nu), np.linspace(-TAPE_W/2, TAPE_W/2, nv)
        UU, VV = np.meshgrid(uu, vv, indexing="ij")
        X2 = np.stack([UU.ravel(), VV.ravel()], 1)
        idx = np.arange(nu*nv).reshape(nu, nv)
        TT = []
        for i in range(nu - 1):
            for j in range(nv - 1):
                a, bb, cc, d = idx[i, j], idx[i+1, j], idx[i+1, j+1], idx[i, j+1]
                TT += [(a, bb, cc), (a, cc, d)] if (i + j) % 2 == 0 else [(a, bb, d), (bb, cc, d)]
        planar = o + c + X2[:, :1]*along + X2[:, 1:]*across
        # project each tape vertex onto the outermost wrap surface along -nrm
        x = np.zeros((len(planar), 3))
        A, B, C = wrap_x[T[:, 0]], wrap_x[T[:, 1]], wrap_x[T[:, 2]]
        e1, e2 = B - A, C - A
        dirv = -nrm
        pv = np.cross(dirv, e2)
        det = (e1*pv).sum(1)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1./np.where(ok, det, 1.), 0.)
        for k, p in enumerate(planar):
            start = p + nrm*.05
            tv = start - A
            uu_ = (tv*pv).sum(1)*inv
            qv = np.cross(tv, e1)
            vv_ = (qv @ dirv)*inv
            tt = (e2*qv).sum(1)*inv
            hit = ok & (uu_ >= -1e-9) & (vv_ >= -1e-9) & (uu_ + vv_ <= 1 + 1e-9) & (tt > 0)
            if not hit.any():
                raise AssertionError(f"{nm}: tape vertex {k} has no paper beneath")
            tmin = tt[hit].min()                   # first surface met from outside = outermost layer
            x[k] = start + dirv*tmin + nrm*d0
        # tape drapes over a flap-edge step instead of following it vertically: each vertex rises to the
        # highest neighbouring surface within 1.5 wrap cells (bonds form only where it touches the paper)
        hgt = (x - planar) @ nrm
        from scipy.spatial import cKDTree
        kd = cKDTree(X2)
        hd = np.array([hgt[kd.query_ball_point(X2[k], 9e-3)].max() for k in range(len(X2))])
        x = planar + hd[:, None]*nrm
        sh = W.Sheet(nm, X2, np.asarray(TT), x, [Pm.MAGIC_TAPE], np.zeros(len(TT), int), fmat="tape")
        sh.render = dict(kind="tape")
        tapes.append(sh)
    return tapes


def wrapped_gift(table_z, xy=(0., 0.), *, md="girth", seed=0, cell=.008, box=None, stock=DEFAULT_STOCK):
    """Box + lid + folded wrap + tapes resting on a surface at height table_z: the whole gift is lifted so the
    lowest paper surface (the folded ends' bottom edges) sits at the sheet-on-rigid rest distance."""
    base, lid, geom = GB.make_box(np.array([xy[0], xy[1], table_z + .01]), box=box)
    o = base.p - base.R @ base.render["com_offset"]
    sheet, geo = build(o, md=md, seed=seed, cell=cell, box=box, stock=stock)
    tapes = tape_sheets(sheet.x0, sheet.tris, geo)
    zmin = min(sheet.x0[:, 2].min(), *(t.x0[:, 2].min() for t in tapes))
    dz = (table_z + W.contact_caliper(Pm.WRAP_STOCKS[stock][0])["rest_sheet_rigid"] + START_GAP) - zmin
    for b in (base, lid):
        b.p = b.p + np.array([0, 0, dz])
    sheet.x0 = sheet.x0 + np.array([0, 0, dz])
    for t in tapes:
        t.x0 = t.x0 + np.array([0, 0, dz])
    geo["base_origin"] = geo["base_origin"] + np.array([0, 0, dz])
    return base, lid, geom, sheet, tapes, geo


# ----------------------------------------------------------------------------- relaxed, taped start state
RELAX_S = .20          # s, rest-morph blend from the folded geometry to the true paper (tapes bonded, normal friction)
HOLD_S = .08           # s, hold after the blend (the morph has ended)
STORE_S = 86400.       # s, the present was wrapped a day before it is opened [A]: paper stress relaxation in storage
STORE_SETTLE_S = .20   # s, the wrapped present re-equilibrates after the relaxation
RELAY_CYCLES = 12      # the tape's adhesive relaxes too: the bonds are laid again, unloaded, where the tape now lies, and
RELAY_S = .10          # the present settles RELAY_S, RELAY_CYCLES times (a loaded pressure-sensitive bond creeps) [A]
GIRTH_SLACK = 1.0e-3   # m, how much longer than its path around the box the paper is taped [E]: a wrapper lays the seam
                       # by hand, to about a millimetre, and cannot tape under tension. The tension at rest is zero
                       # for any value >= 0; the value sets how far the paper stands off the box


def girth_allowance(geo, cal):
    """(fraction, path part in m): how much longer the paper's girth is than the bare box's, as a fraction of the box
    girth 4 b + 2 Hc. The paper's middle surface lies one sheet-on-rigid rest distance off each face, so its path
    around the four long edges is 8 x that distance longer than the box (0.50 mm for the authored wrap), plus
    GIRTH_SLACK."""
    path = 8.*(cal["rest_sheet_rigid"] + START_GAP)
    return (path + GIRTH_SLACK)/(4.*geo["b"] + 2.*geo["Hc"]), path


def _cache_key(table_z, xy, md, seed, cell, box=None, stock=DEFAULT_STOCK):
    import hashlib
    from . import ipcworld as Wm, giftbox as GBm
    from . import tape_bonds as TBm
    from . import startcache as SC
    src = Pm.CONTACT_WORLD_VERSION + repr(Pm.TAPE_QUAD) + SC.code(TBm) + SC.code(GBm) \
        + SC.code(build) + SC.code(wrap_mesh) \
        + SC.code(relaxed_gift)
    mat, crease_law = Pm.WRAP_STOCKS[stock]
    from . import bagnet as Bm
    src += SC.code(Bm.age_creases) + SC.code(Bm.membrane_stretch) + SC.code(Bm.apply_membrane_set)
    src += repr((round(table_z, 9), tuple(map(float, xy)), md, seed, cell, RELAX_S, HOLD_S, mat, crease_law,
                 Pm.TAPE_ON_WRAP, Pm.MAGIC_TAPE, box or GB.box_spec(), END_FLAP, SEAM_OVERLAP, STORE_S, STORE_SETTLE_S,
                 Bm.PAPER_RELAX_P1, RELAY_CYCLES, RELAY_S, GIRTH_SLACK))
    src += SC.code(girth_allowance)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def relaxed_gift(table_z, xy=(0., 0.), *, md="girth", seed=0, cell=.008, cache=None, log=print, box=None,
                 stock=DEFAULT_STOCK):
    """The wrapped gift as a person leaves it: the folded wrap (build) and its three tapes, bonded, relaxed on the
    box while the wrap's rest shape blends from the folded geometry to the true flat paper (the 6.6 % local
    stretch of the analytic fold map flows out by sliding against normal friction; the tapes hold the folds as
    the wrapper's fingers and tape do). An untaped, frictionless relaxation lets the end folds fall open, so it
    is not used. Storage follows (STORE_S, a day): held in that shape, the wrap's in-plane stress and its panels'
    bending relax by paper's stress relaxation and its creases age by their law (bagnet.age_creases, the law the tote's
    shipping state uses); the present then re-equilibrates (STORE_SETTLE_S), and the tape's adhesive relaxes: the
    bonds are laid again, unloaded, where the tape lies, RELAY_CYCLES times. Without storage the paper keeps all the
    stress the fold map leaves, and each end tape holds 15-18 N in the task's start state. The
    sheet returned carries that history: `start_hinges` {(i, j): (theta_p, theta_r)} and `start_membrane` (n_tri, 2, 2),
    which the scene applies to the task world. Cached in `cache` (default .cache/gift_bags/start_states) by a hash of
    the generator and of the present's variation (`box`: giftbox.BoxSpec, default the authored box; `stock`: a
    params.WRAP_STOCKS key)."""
    from pathlib import Path
    from . import ipcworld as Wm, scene as S
    from . import startcache as SC
    key = _cache_key(table_z, xy, md, seed, cell, box, stock)
    path = SC.locate(f"wrapped_gift_{key}.npz", cache)
    base, lid, geom = GB.make_box(np.array([xy[0], xy[1], table_z + .01]), box=box)
    o = base.p - base.R @ base.render["com_offset"]
    sheet, geo = build(o, md=md, seed=seed, cell=cell, box=box, stock=stock)
    tapes = tape_sheets(sheet.x0, sheet.tris, geo)
    zmin = min(sheet.x0[:, 2].min(), *(t.x0[:, 2].min() for t in tapes))
    dz = (table_z + Wm.contact_caliper(Pm.WRAP_STOCKS[stock][0])["rest_sheet_rigid"] + START_GAP) - zmin
    for b in (base, lid):
        b.p = b.p + np.array([0, 0, dz])
    sheet.x0 = sheet.x0 + np.array([0, 0, dz])
    for t in tapes:
        t.x0 = t.x0 + np.array([0, 0, dz])
    geo["base_origin"] = geo["base_origin"] + np.array([0, 0, dz])
    if path.is_file():
        d = np.load(path)
        sheet.x0 = d["wrap_x"]
        for k, t in enumerate(tapes):
            t.x0 = d[f"tape{k}_x"]
        base.p, base.R, lid.p, lid.R = d["base_p"], d["base_R"], d["lid_p"], d["lid_R"]
        hinges = {(int(i), int(j)): (float(tp), float(tr)) for i, j, tp, tr in d["hinges"]}
        membrane = d["membrane"]
        info = dict(cached=str(path), **{k: float(d[k]) for k in ("max_strain_before", "max_strain_after", "relax_wall_s",
                                                                 "unconverged", "bond_intact_fraction", "store_s",
                                                                 "max_strain_stored", "membrane_set_max",
                                                                 "bond_force_before_storage_n", "bond_force_after_storage_n",
                                                                 "bond_force_after_adhesive_n", "storage_move_um",
                                                                 "girth_allowance_mm", "girth_path_allowance_mm",
                                                                 "girth_slack_mm", "girth_tension_n_per_m_max",
                                                                 "girth_tension_n_per_m_mean", "standoff_mm_max")})
    else:
        table = dict(name="table_top", center=(xy[0], xy[1], table_z - .02), half=(.30, .25, .02), mu=Pm.MU["box_table"], fmat="oak")
        w = Wm.IPCWorld([sheet] + tapes, [], [S.furniture_body(table), base, lid])
        from . import tape_bonds as TB
        bonds = TB.bond_tapes(w, w.sheet_slices[1:1 + len(tapes)], w.sheet_slices[0], Pm.TAPE_ON_WRAP,
                              d0=w.caliper["rest_sheet_sheet"])
        w.check_initial()
        # the paper as cut: longer around the girth than the fold map's net by girth_allowance (the true rest shape;
        # the blend below takes the folded geometry to it)
        from . import bagnet as B
        n_tri = len(sheet.tris)
        a_g, a_path = girth_allowance(geo, Wm.contact_caliper(Pm.WRAP_STOCKS[stock][0]))
        U_cut = np.tile(np.diag([1. + a_g, 1.] if md == "girth" else [1., 1. + a_g]), (n_tri, 1, 1))
        B.apply_membrane_set(w, U_cut)
        strain0 = _max_strain(w)
        Wm.prepare_rest_morph(w)
        import time as _t
        t0 = _t.perf_counter()
        dt = 4e-3
        n_relax, n_hold = int(round(RELAX_S/dt)), int(round(HOLD_S/dt))
        for k in range(n_relax + n_hold):
            if k < n_relax:
                Wm.set_rest_blend(w, (k + 1)/n_relax)
            elif w._morph is not None:
                Wm.end_rest_morph(w)
            w.step(dt)
            if k % 10 == 0 and log:
                log(f"  relax t={w.time:.3f} iters={w.stats['last_iters']} unconverged={w.stats['unconverged']} "
                    f"KE={w.kinetic_energy():.2e} bonds intact {np.mean([b.intact_fraction for b in bonds]):.3f}")
        if w._morph is not None:
            Wm.end_rest_morph(w)
        strain1 = _max_strain(w)

        def bond_force():                      # largest net bond force on one tape (what a tape holds), N
            x = w.x_free()
            if not all(hasattr(b, "_delta") for b in bonds):          # per-vertex bonds (TAPE_QUAD = 0): not reported
                return float("nan")
            return float(max(np.linalg.norm((((1. - b.D)*b.law.k*b.area)[:, None]*b._delta(x)).sum(0)) for b in bonds))
        f_before = bond_force()
        # storage: a day in this shape. The wrap's in-plane stress relaxes (rest shape moves a fraction of the way to
        # the held stretch), its panels' bending relaxes and its creases age (age_creases); the tapes' own film is
        # left elastic
        f_relax = min(1., B.PAPER_RELAX_P1*np.log1p(STORE_S))
        Up = (np.eye(2) + f_relax*(B.membrane_stretch(w) - np.eye(2)))[:n_tri]
        B.apply_membrane_set(w, Up)
        sl0 = w.sheet_slices[0]
        other = np.array([not (sl0.start <= a < sl0.stop and sl0.start <= b_ < sl0.stop) for a, b_, _, _ in w.hinge])
        tp0, tr0 = w.theta_p.copy(), w.theta_r.copy()
        B.age_creases(w, STORE_S)
        w.theta_p[other], w.theta_r[other] = tp0[other], tr0[other]
        x_s = w.x_free().copy()
        for k in range(int(round(STORE_SETTLE_S/dt))):
            w.step(dt)
            if k % 10 == 0 and log:
                log(f"  stored, settling t={w.time:.3f} iters={w.stats['last_iters']} unconverged={w.stats['unconverged']} "
                    f"KE={w.kinetic_energy():.2e} largest tape bond force {bond_force():.2f} N")
        f_stored = bond_force()
        # the adhesive: the generator's bonds hold every tape point within 0.6 mm at the nominal offset, which pulls
        # the tape down over flap edges and squeezes the layers under it (tens of newtons). A pressure-sensitive bond
        # under a steady load creeps until the load is gone. Each cycle lays the bonds again where the tape now lies
        # (as the task's start state does: bonded where it touches, unloaded) and lets the present settle.
        relay = []
        if Pm.TAPE_QUAD:
            for c in range(RELAY_CYCLES):
                old = list(w.bonds)
                w.bonds = []
                for attr in ("extra_terms", "pre_terms", "post_terms"):
                    setattr(w, attr, [f for f in getattr(w, attr, []) if getattr(f, "__self__", None) not in old])
                bonds = TB.bond_tapes(w, w.sheet_slices[1:1 + len(tapes)], w.sheet_slices[0], Pm.TAPE_ON_WRAP,
                                      d0=w.caliper["rest_sheet_sheet"], as_laid=True)
                for b in bonds:
                    b.begin(w, dt)
                for _ in range(int(round(RELAY_S/dt))):
                    w.step(dt)
                relay.append(bond_force())
                if log:
                    log(f"  adhesive cycle {c}: largest tape bond force {relay[-1]:.2f} N, bonded "
                        f"{[round(b.bonded_fraction, 3) for b in bonds]}, unconverged={w.stats['unconverged']}")
        wall = _t.perf_counter() - t0
        f_after, strain2 = bond_force(), _max_strain(w)
        move = float(1e6*np.linalg.norm(w.x_free() - x_s, axis=1).max())
        hinges = B.hinge_state(w)
        membrane = np.matmul(Up, U_cut)        # the task world's rest shape: the paper as cut, then the storage set
        # girth tension left in the sleeve (middle 70 % of the box length), N/m per face: what clamps the box
        gt = girth_tension(w, sheet, geo, md)
        intact = float(np.mean([b.intact_fraction for b in bonds]))
        sheet.x0 = w.x_free()[w.sheet_slices[0]].copy()
        for k, t in enumerate(tapes):
            t.x0 = w.x_free()[w.sheet_slices[1 + k]].copy()
        (bp, bR), (lp, lR) = w.body_pose(1), w.body_pose(2)
        base.p, base.R, lid.p, lid.R = bp, bR, lp, lR
        extra = dict(store_s=STORE_S, max_strain_stored=strain2, membrane_set_max=float(np.abs(Up - np.eye(2)).max()),
                     bond_force_before_storage_n=f_before, bond_force_after_storage_n=f_stored,
                     bond_force_after_adhesive_n=f_after, storage_move_um=move,
                     girth_allowance_mm=1e3*a_g*(4.*geo["b"] + 2.*geo["Hc"]), girth_path_allowance_mm=1e3*a_path,
                     girth_slack_mm=1e3*GIRTH_SLACK, girth_tension_n_per_m_max=max(gt["tension_n_per_m"].values()),
                     girth_tension_n_per_m_mean=float(np.mean(list(gt["tension_n_per_m"].values()))),
                     standoff_mm_max=gt["standoff_mm_max"])
        np.savez(path, wrap_x=sheet.x0, **{f"tape{k}_x": t.x0 for k, t in enumerate(tapes)}, base_p=bp, base_R=bR,
                 lid_p=lp, lid_R=lR, max_strain_before=strain0, max_strain_after=strain1, relax_wall_s=wall,
                 unconverged=w.stats["unconverged"], bond_intact_fraction=intact, membrane=membrane,
                 hinges=np.array([(i, j, tp, tr) for (i, j), (tp, tr) in hinges.items()]), **extra)
        info = dict(cached=None, built=str(path), max_strain_before=strain0, max_strain_after=strain1, relax_wall_s=wall,
                    unconverged=w.stats["unconverged"], bond_intact_fraction=intact, **extra)
    sheet.crease_rest = {}
    sheet.start_hinges, sheet.start_membrane = hinges, membrane      # the stored present's history (scene hook)
    geo["base_origin"] = base.p - base.R @ base.render["com_offset"]
    geo["start_state"] = info
    return base, lid, geom, sheet, tapes, geo


def girth_tension(w, sheet, geo, md="girth"):
    """Membrane force around the girth in the wrap's sleeve at the world's current state, per box face, over the
    middle 70 % of the box length (N/m, mean over the face's triangles; the paper's stiff direction runs around the
    girth for md = "girth"), and how far the paper stands off the box there at most (mm, beyond its rest distance).
    This is what clamps the box in its paper: at each long edge the paper presses with the resultant of the two
    faces' tensions."""
    _, _, S, _ = w.membrane_state()
    n_tri = len(sheet.tris)
    X2, T = np.asarray(sheet.X2), np.asarray(sheet.tris)
    cen = X2[T].mean(1)
    e, Lx, sg = geo["e"], geo["Lx"], geo["seg"]
    mid = (cen[:, 0] > e + .15*Lx) & (cen[:, 0] < e + .85*Lx)
    comp = 0 if md == "girth" else 1
    out = {}
    for name, lo, hi in (("bottom_in", sg["s0"], sg["s1"]), ("back", sg["s1"], sg["s2"]), ("top", sg["s2"], sg["s3"]),
                         ("front", sg["s3"], sg["s4"]), ("bottom_out", sg["s4"], sg["s5"])):
        m = mid & (cen[:, 1] > lo) & (cen[:, 1] < hi)
        out[name] = float((S[:n_tri][m, comp]*w.area[:n_tri][m]).sum()/max(w.area[:n_tri][m].sum(), 1e-30))
    x = w.x_free()[w.sheet_slices[0]]
    o = np.asarray(geo["base_origin"], float)
    b, Hc = geo["b"], geo["Hc"]
    vm = (X2[:, 0] > e + .15*Lx) & (X2[:, 0] < e + .85*Lx)
    p = x[vm] - o
    out_d = np.maximum.reduce([np.abs(p[:, 1]) - b, p[:, 2] - Hc, -p[:, 2]])       # distance outside the box (L-inf)
    return dict(tension_n_per_m=out, standoff_mm_max=1e3*float(out_d.max()))


def _max_strain(w):
    """Largest edge stretch of the wrap against its true rest lengths."""
    x = w.x_free()[w.sheet_slices[0]]
    sh = w.sheets[0]
    E = W.edges_of(np.asarray(sh.tris))
    X2 = np.asarray(sh.X2)
    Lr = np.linalg.norm(X2[E[:, 0]] - X2[E[:, 1]], axis=1)
    Lx = np.linalg.norm(x[E[:, 0]] - x[E[:, 1]], axis=1)
    return float(np.max(Lx/Lr - 1.))
