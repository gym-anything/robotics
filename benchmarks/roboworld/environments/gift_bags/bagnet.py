"""Factory-flat laminated Euro-tote gift bag: crease pattern, rest shape and flat-folded start state.

Rest shape: the open bag (front, back, two side gussets, block bottom; open top), panels planar.
Crease lines (scored and folded in manufacture): the four vertical corners, the gusset centre lines, the
bottom perimeter, the bottom centre line, the four 45-degree bottom diagonals and the four 45-degree gusset
diagonals. Turn-top band (40 mm) carries the 900 g/m2 board on the faces; the bottom is a glued double layer.

Flat state (how the bag ships, US8486500B2): gussets V-folded inward along their centre lines, the block
bottom flattened with its end triangles folded in. This is the bottom-extended ("pinch") flat state: the
bottom forms a trapezoidal double-layer flap beyond the bag face; real SOS/Euro totes ship with that flap
folded up once more against one face, an extra nested 180-degree fold of four layers that the 7 mm mesh
cannot represent without initial strain (REPORT limitations). The map is isometric facet by facet (rest edge
lengths are reproduced to < 1e-6 m).

Layer depths (front face on top): front 3, gusset front halves 2, gusset back halves 1, back 0; the bottom
flap: front half 3, its end triangles 2, back-half end triangles 1, back half 0.
"""

import numpy as np
from . import ipcworld as W
from . import params as Pm

IN = .0254
W_BAG, G_BAG, H_BAG = 8*IN, 4*IN, 10*IN      # 203 x 102 x 254 mm Euro tote [V, PackagingSource]
TURN_TOP = .040                                # m [A]
HANDLE_X = .040                                # m, hole offset from the centre line [E]; the hole is the mesh cell
HANDLE_DROP = .025                             # m, hole below the top edge [E]   containing this point (handles.py)


FOLD_ROW = 8e-4        # m, first mesh row at the bottom perimeter folds (graded mesh) [A]
FOLD_GROWTH = 1.6      # row-to-row growth away from the folds [A]


def graded_half(G, nG, first=FOLD_ROW, growth=FOLD_GROWTH):
    """Cell sizes over [0, G/2] starting at `first` beside a fold and growing by `growth` up to the nominal cell
    G/nG, the rest filled with equal cells. The same sequence is used along every bottom fold (x, y and z), so the
    45-degree bottom and gusset diagonals still run through mesh nodes."""
    s, half = G/nG, G/2
    steps, h = [], first
    while h < s and sum(steps) + h < half - s:
        steps.append(h); h *= growth
    rest = half - sum(steps)
    n = max(1, int(round(rest/s)))
    return np.asarray(steps + [rest/n]*n)


def bag_mesh(cells_per_gusset=14, graded=True, holes=False):
    """Square-celled grid over the five panels (2:1:2.5 bag proportions), welded into one open box. graded: rows
    refined toward the bottom perimeter folds (first row FOLD_ROW), so the shipping state's four layers can wrap
    around each other there; elsewhere the nominal G/cells_per_gusset cell. holes: punch the four rope-handle holes
    (one turn-top cell each, at (+-HANDLE_X, H - HANDLE_DROP) of each face). Returns X, T, kinds, s, holes
    {'front'|'back': [corner vertex ids of each hole, ordered so their normal points out of the bag]}."""
    Wd, G, H = W_BAG, G_BAG, H_BAG
    nG = cells_per_gusset
    s = G/nG
    nW, nH = int(round(Wd/s)), int(round(H/s))
    assert abs(nW*s - Wd) < 1e-9 and abs(nH*s - H) < 1e-9 and nG % 2 == 0
    g = graded_half(G, nG) if graded else np.full(nG//2, s)
    nh = len(g)                                               # cells in a graded half-gusset
    cum = lambda steps: np.concatenate([[0.], np.cumsum(steps)])
    ys = cum(np.r_[g, g[::-1]])                               # across a gusset / the bottom: graded at both edges
    xs = cum(np.r_[g, np.full(nW - nG, s), g[::-1]])          # across the faces / bottom: graded at both corners
    zs = cum(np.r_[g, np.full(nH - nG//2, s)])                # up the faces / gussets: graded at the bottom fold
    ys, xs, zs = ys*G/ys[-1], xs*Wd/xs[-1], zs*H/zs[-1]       # exact ends
    nGc, nWc = len(ys) - 1, len(xs) - 1
    pts, tris, kinds = [], [], []

    def panel(origin, u, v, us, vs, kind, diag_rule):
        """u, v unit directions; us, vs node distances along them."""
        base = len(pts)
        nu, nv = len(us) - 1, len(vs) - 1
        for i in range(nu + 1):
            for j in range(nv + 1):
                pts.append(origin + u*us[i] + v*vs[j])
        for i in range(nu):
            for j in range(nv):
                a = base + i*(nv + 1) + j
                b, c, d = a + nv + 1, a + nv + 2, a + 1
                if diag_rule(i, j):          # split along a -> c
                    tris.extend([(a, b, c), (a, c, d)])
                else:                        # split along b -> d
                    tris.extend([(a, b, d), (b, c, d)])
                kinds.extend([kind, kind])
    O = np.array([-Wd/2, -G/2, 0.])
    ex, ey, ez = np.eye(3)
    alt = lambda i, j: (i + j) % 2 == 0
    # gusset diagonals from the bottom corners (y = +-G/2, z = 0) to (y = 0, z = G/2):
    # gusset grid (i along y from -G/2, j along z): front-half diagonal i == j, back-half diagonal i + j == nGc - 1
    def gusset_rule(i, j):
        if i < nh and i == j:
            return True              # a(i,j) -> c(i+1,j+1)
        if i >= nh and i + j == nGc - 1:
            return False             # b(i+1,j) -> d(i,j+1)
        return alt(i, j)
    # bottom grid (i along x from -W/2, j along y from -G/2): diagonals from the corners to (+-(W/2 - G/2), 0)
    def bottom_rule(i, j):
        if i < nh and j < nh and i == j:
            return True
        if i < nh and j >= nh and i + j == nGc - 1:
            return False
        ii = nWc - 1 - i
        if ii < nh and j < nh and ii == j:
            return False
        if ii < nh and j >= nh and ii + j == nGc - 1:
            return True
        return alt(i, j)
    panel(O, ex, ez, xs, zs, "front", alt)
    panel(O + G*ey, ex, ez, xs, zs, "back", alt)
    panel(O, ey, ez, ys, zs, "left", gusset_rule)
    panel(O + Wd*ex, ey, ez, ys, zs, "right", gusset_rule)
    panel(O, ex, ey, xs, ys, "bottom", bottom_rule)
    P = np.asarray(pts)
    key = np.round(P/1e-7).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    X = P[first]
    T = inv.ravel()[np.asarray(tris)]
    # consistent outward orientation (normal away from the bag interior centre)
    cen = np.array([0., 0., H/2])
    for t in range(len(T)):
        a, b, c = X[T[t]]
        n = np.cross(b - a, c - a)
        mid = (a + b + c)/3
        out = mid - cen
        if T[t] is not None and kinds[t] == "bottom":
            out = np.array([0., 0., -1.])
        if n @ out < 0:
            T[t] = T[t][[0, 2, 1]]
    kinds = np.asarray(kinds)
    hole_ids = {}
    if holes:
        keep = np.ones(len(T), bool)
        cen = X[T].mean(1)
        node = lambda p: int(np.argmin(np.linalg.norm(X - np.asarray(p), axis=1)))
        for face, y in (("front", -G/2), ("back", G/2)):
            hole_ids[face] = []
            for sx in (-1, 1):
                tx, tz = sx*HANDLE_X, H - HANDLE_DROP
                i = int(np.searchsorted(xs - Wd/2, tx)) - 1
                j = int(np.searchsorted(zs, tz)) - 1
                x0, x1, z0, z1 = xs[i] - Wd/2, xs[i + 1] - Wd/2, zs[j], zs[j + 1]
                cell = (kinds == face) & (cen[:, 0] > x0) & (cen[:, 0] < x1) & (cen[:, 2] > z0) & (cen[:, 2] < z1)
                assert cell.sum() == 2, (face, sx, int(cell.sum()))
                keep &= ~cell
                c = [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
                if face == "back":
                    c = [c[0], c[3], c[2], c[1]]
                hole_ids[face].append([node(p) for p in c])
        T, kinds = T[keep], kinds[keep]
    return X, T, kinds, s, hole_ids


def crease_edges(X, T, s):
    """Edges on the manufactured fold lines."""
    Wd, G, H = W_BAG, G_BAG, H_BAG
    e = W.edges_of(T)
    A, B = X[e[:, 0]], X[e[:, 1]]
    tol = 1e-7

    def on(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        corner = (np.abs(np.abs(x) - Wd/2) < tol) & (np.abs(np.abs(y) - G/2) < tol)
        gcent = (np.abs(np.abs(x) - Wd/2) < tol) & (np.abs(y) < tol)
        bot = np.abs(z) < tol
        perim = bot & ((np.abs(np.abs(x) - Wd/2) < tol) | (np.abs(np.abs(y) - G/2) < tol))
        bcent = bot & (np.abs(y) < tol)
        bdiag = bot & (np.abs((Wd/2 - np.abs(x)) - (G/2 - np.abs(y))) < tol) & (np.abs(x) >= Wd/2 - G/2 - tol)
        gside = np.abs(np.abs(x) - Wd/2) < tol
        gdiag = gside & (np.abs(z - (G/2 - np.abs(y))) < tol) & (z <= G/2 + tol)
        return corner, gcent, perim, bcent, bdiag, gdiag
    ca, cb = on(A), on(B)
    names = ("corner", "gusset_centre", "bottom_perimeter", "bottom_centre", "bottom_diagonal", "gusset_diagonal")
    out = {}
    for k, nm in enumerate(names):
        m = ca[k] & cb[k]
        if nm in ("bottom_diagonal", "gusset_diagonal"):
            d = np.abs(A - B)
            m &= (np.abs(d[:, 0] - d[:, 1]) < tol) if nm == "bottom_diagonal" else (np.abs(d[:, 1] - d[:, 2]) < tol)
        for i, j in e[m]:
            out[(int(i), int(j))] = nm
    return out


NOSE_C = 3.         # fold nose: each layer's apex lies NOSE_C x its half-depth in the stack beyond the fold line [A]
NOSE_BLEND = .012   # m, beyond this distance from the fold line every layer lies exactly where the flat stack has it [A]
NOSE_PUFF = .5      # the layers start this fraction farther apart at the fold than in the flat stack [A]: where two joined
                    # layers' noses merge (gusset centre folds, the fold's two ends) their fans overlap in plan with
                    # 0.67-0.71 of the layer spacing between them, 96-105 um at one pitch against the contact world's
                    # 116 um minimum (measured); the relaxation closes the extra spacing under gravity and the press


def fold_section(s, ell, pitch, c=NOSE_C, blend=NOSE_BLEND, row=None, puff=NOSE_PUFF):
    """Cross-section of the shipping fold (the bottom flap folded 180 degrees up over the front face), per vertex.

    s: signed distance from the fold line in the bottom-extended flat map (>= 0 bag body, < 0 flap); ell: layer
    coordinate in the flat stack (0 back face ... 3 front face; fractional on the folds that join layers). Each layer
    turns the fold on its own nose, a V whose apex (the fold-line vertices) lies c rho beyond the fold line at the
    stack's mid depth, rho = (3.5 - ell) pitch being the layer's distance from that depth, and whose two legs end on the
    fold line at the layer's depth below (body) and above (flap: its layers come to lie reversed on the front face).
    The legs of all layers are parallel, c/sqrt(1 + c^2) of a pitch apart (0.95 at c = 3; the contact world needs
    > 0.77), which a coarser nose on this mesh cannot give: with one fold-line vertex per layer and the first mesh row
    0.8 mm away, noses of radius 0.07-0.5 mm drawn by arc length put neighbouring layers' chords 40-70 um apart.
    The leg carries the first mesh row (`row`, default FOLD_ROW) of material whatever its
    length (0.24-1.66 mm), and from there to `blend` the layer is stretched evenly (7 %) so that farther from the fold
    every layer lies exactly where the flat stack has it. The map is therefore not isometric within `blend` of the
    fold; shipping_state() blends the rest shape from this geometry to the true paper and lets the layers find their
    own radii. Returns (y, h): position along the bag from the fold line (negative beyond it) and height above the
    lowest layer's level."""
    row = FOLD_ROW if row is None else row
    s = np.asarray(s, float)
    u = np.abs(s)
    # layer spacing: one pitch in the flat stack, NOSE_PUFF more at the fold, easing back to one pitch at `blend`
    g = 1. + puff*np.clip((blend - u)/(blend - row), 0., 1.)
    rho = (3.5 - np.asarray(ell, float))*pitch*g
    f = np.clip(u/row, 0., 1.)                                  # along the leg: 0 apex, 1 its end on the fold line
    y = np.where(u <= row, -c*rho*(1. - f), np.where(u < blend, (u - row)*blend/(blend - row), u))
    return y, 3.5*pitch*g + np.where(s < 0, 1., -1.)*rho*f


def facet_map(X, T, kinds, flap_up=False):
    """Flat-state in-plane coordinates (Xf along width, Zf along height) and layer per triangle; returns
    per-vertex (Xf, Zf) (consistent across folds) and per-triangle layer. flap_up: the shipping state, the
    bottom-extended flap folded up once more over the front face about the bottom perimeter (Zf -> -Zf), its four
    layers reversed on top of the front face (layer L -> 7 - L)."""
    Wd, G = W_BAG, G_BAG
    flat = np.full((len(X), 2), np.nan)
    layer = np.zeros(len(T), int)

    def mp(v, kind, cen):
        x, y, z = v
        cx, cy, cz = cen
        if kind == "front":
            return (x, z), 3
        if kind == "back":
            return (x, z), 0
        if kind in ("left", "right"):
            sgn = -1. if kind == "left" else 1.
            if cy < 0:
                return (sgn*(Wd/2 - (y + G/2)), z), 2
            return (sgn*(Wd/2 - (G/2 - y)), z), 1
        # bottom
        if cy < 0:
            d = y + G/2
            if cx < -Wd/2 + (cy + G/2):
                return (-Wd/2 + d, -(x + Wd/2)), 2
            if cx > Wd/2 - (cy + G/2):
                return (Wd/2 - d, -(Wd/2 - x)), 2
            return (x, -d), 3
        d = G/2 - y
        if cx < -Wd/2 + (G/2 - cy):
            return (-Wd/2 + d, -(x + Wd/2)), 1
        if cx > Wd/2 - (G/2 - cy):
            return (Wd/2 - d, -(Wd/2 - x)), 1
        return (x, -d), 0
    for t, tri in enumerate(T):
        cen = X[tri].mean(0)
        for v in tri:
            (fx, fz), L = mp(X[v], kinds[t], cen)
            if np.isnan(flat[v, 0]):
                flat[v] = (fx, fz)
            elif abs(flat[v, 0] - fx) > 1e-7 or abs(flat[v, 1] - fz) > 1e-7:
                raise AssertionError(f"inconsistent fold map at vertex {v}: {flat[v]} vs {(fx, fz)}")
        layer[t] = mp(X[tri[0]], kinds[t], cen)[1]
    if flap_up:
        bot = np.flatnonzero(kinds == "bottom")
        bv = np.unique(T[bot])
        flat[bv, 1] = -flat[bv, 1]                     # fold-line vertices have Zf = 0 and stay shared
        layer[bot] = 7 - layer[bot]
    return flat, layer


def build(origin, *, cells_per_gusset=10, mouth_dir=(0., 1., 0.), seed=0, start_gap=5e-6, flap_up=False, graded=True,
          holes=True):
    """Sheet for the flat bag lying back-face-down with its mouth toward `mouth_dir` (horizontal); origin is
    the centre of the bag's bottom fold line on the support surface (z of the support). Layers are one caliper
    apart and the lowest sits at the sheet-on-rigid rest distance (contact_caliper), each `start_gap` outside
    its barrier range."""
    cal = W.contact_caliper(Pm.TOTE)
    pitch = cal["rest_sheet_sheet"] + start_gap
    base = cal["rest_sheet_rigid"] + start_gap
    X, T, kinds, s, hole_ids = bag_mesh(cells_per_gusset, graded=graded, holes=holes)
    creases = crease_edges(X, T, s)
    flat, layer = facet_map(X, T, kinds)             # the bottom-extended flat map and its layers (0 back ... 3 front)
    # per-vertex depth: mean layer of the incident triangles
    acc, cnt = np.zeros(len(X)), np.zeros(len(X))
    for t, tri in enumerate(T):
        acc[tri] += layer[t]; cnt[tri] += 1
    depth = acc/cnt
    along, height = flat[:, 1].copy(), pitch*depth + base
    flat_ext = flat.copy()
    if flap_up:
        # the shipping state: the flap folded up over the front face, every layer on its own nose (fold_section).
        # A vertex on a fold that joins two layers takes the OUTER layer's coordinate (the smaller one), not their mean:
        # at the two ends of the fold line the flap's diagonal creases join layers one mesh row from the fold, and a
        # join at the mean depth tilts the outer layer's leg to within 96 um of the inner layer's apex (measured).
        ell = np.full(len(X), np.inf)
        for t, tri in enumerate(T):
            ell[tri] = np.minimum(ell[tri], layer[t])
        along, h = fold_section(flat[:, 1], ell, pitch)
        height = base + h
        flat = np.c_[flat[:, 0], along]
        layer = np.where(kinds == "bottom", 7 - layer, layer)
    m = np.asarray(mouth_dir, float); m /= np.linalg.norm(m)
    wdir = np.cross(m, [0., 0., 1.])
    rng = np.random.default_rng(seed)
    x0 = (np.asarray(origin, float) + flat[:, :1]*wdir + along[:, None]*m
          + height[:, None]*np.array([0., 0., 1.]) + rng.uniform(-1e-6, 1e-6, (len(X), 3)))
    # materials: 0 laminate, 1 turn-top band (front/back), 2 glued bottom, 3 gusset turn-top (double laminate)
    cz = X[T].mean(1)[:, 2]
    tm = np.zeros(len(T), int)
    top = cz > H_BAG - TURN_TOP
    tm[top & np.isin(kinds, ["front", "back"])] = 1
    tm[kinds == "bottom"] = 2
    tm[top & np.isin(kinds, ["left", "right"])] = 2
    md = np.where((kinds == "bottom")[:, None], [[1., 0., 0.]], [[0., 0., 1.]])
    law = Pm.TOTE_CREASE
    sheet = W.Sheet("bag", X, T, x0, [Pm.TOTE, Pm.TURN_TOP, Pm.BOTTOM], tm, md_dir=md, fmat="laminate",
                    creases={k: law for k in creases})
    sheet.render = dict(kind="tote", kinds=kinds, flat=flat, flat_extended=flat_ext, layer=layer, flap_up=flap_up, holes=hole_ids)
    sheet.crease_names = creases
    return sheet


def handle_points(sheet, which, n=40):
    """Hole positions (vertex ids nearest the two holes of a face) for the rope handle."""
    X = sheet.X2
    y = -G_BAG/2 if which == "front" else G_BAG/2
    ids = []
    for sx in (-1, 1):
        target = np.array([sx*HANDLE_X, y, H_BAG - HANDLE_DROP])
        ids.append(int(np.argmin(np.linalg.norm(X - target, axis=1))))
    return ids


# ----------------------------------------------------------------------------- relaxed shipping state
RELAX_S = .20          # s, rest-morph blend from the folded geometry to the true paper
HOLD_S = .10           # s, hold after the blend


def relaxed_flat_bag(sheet, support_bodies, *, cache=None, log=print):
    """The shipping state as it lies on the stack: the flap-up fold map (build(flap_up=True)) relaxed on its support
    while the bag's rest shape blends from the folded geometry to the true paper, so the ~6 % local stretch at the
    four-layer fold flows out by bending and sliding (the same procedure as the wrapped present, wrapmap.relaxed_gift).
    The crease plastic rest angles are then those of the relaxed folded state (the scores were folded flat for
    shipping). Cached in `cache` (default .cache/gift_bags/start_states) by a hash of the generator."""
    import hashlib, time as _t
    from pathlib import Path
    from . import startcache as SC
    cache = Path(cache) if cache else Path(__file__).resolve().parents[1]/".cache"/"gift_bags"/"start_states"
    cache.mkdir(parents=True, exist_ok=True)
    src = "".join(SC.code(m) for m in (W, Pm)) + SC.code(build) + SC.code(facet_map) \
        + SC.code(relaxed_flat_bag) + repr((np.round(np.asarray(sheet.x0), 9).tobytes(), RELAX_S, HOLD_S))
    key = hashlib.sha256(src.encode() if isinstance(src, str) else src).hexdigest()[:16]
    path = cache/f"flat_tote_flap_up_{key}.npz"
    if path.is_file():
        d = np.load(path)
        sheet.x0 = d["x"]
        info = dict(cached=str(path), **{k: float(d[k]) for k in ("max_strain_before", "max_strain_after", "relax_wall_s",
                                                                  "unconverged")})
    else:
        w = W.IPCWorld([sheet], [], list(support_bodies))
        w.check_initial()
        strain0 = _edge_strain(w)
        W.prepare_rest_morph(w)
        t0 = _t.perf_counter()
        dt = 4e-3
        n_relax, n_hold = int(round(RELAX_S/dt)), int(round(HOLD_S/dt))
        for k in range(n_relax + n_hold):
            if k < n_relax:
                W.set_rest_blend(w, (k + 1)/n_relax)
            elif w._morph is not None:
                W.end_rest_morph(w)
            w.step(dt)
            if k % 10 == 0 and log:
                log(f"  relax tote t={w.time:.3f} iters={w.stats['last_iters']} unconverged={w.stats['unconverged']} "
                    f"KE={w.kinetic_energy():.2e}")
        if w._morph is not None:
            W.end_rest_morph(w)
        wall = _t.perf_counter() - t0
        strain1 = _edge_strain(w)
        sheet.x0 = w.x_free()[w.sheet_slices[0]].copy()
        np.savez(path, x=sheet.x0, max_strain_before=strain0, max_strain_after=strain1, relax_wall_s=wall,
                 unconverged=w.stats["unconverged"])
        info = dict(cached=None, built=str(path), max_strain_before=strain0, max_strain_after=strain1, relax_wall_s=wall,
                    unconverged=w.stats["unconverged"])
    sheet.crease_rest = {}
    return sheet, info


def _edge_strain(w):
    """Largest edge stretch of the tote against its true rest lengths."""
    sl = w.sheet_slices[0]
    x = w.x_free()[sl]
    X = np.asarray(w.sheets[0].X2, float)
    E = W.edges_of(np.asarray(w.sheets[0].tris))
    return float(np.abs(np.linalg.norm(x[E[:, 0]] - x[E[:, 1]], axis=1)/np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1) - 1).max())


# ----------------------------------------------------------------------------- shipping state, made by folding
SHIP_FOLD_S = 1.5             # s, the bottom flap is folded up over the front face [A, a packer's or a machine's fold]
SHIP_HOLD_S = 1.0             # s, held folded flat before ageing
SHIP_STORE_S = 7*24*3600.     # s, stored folded flat (carton, stack) before sale [A, one week; sensitivity in REPORT]
SHIP_RELEASE_S = .3           # s, the fold fixture fades out
SHIP_SETTLE_S = 1.0           # s, the bag then lies free
FOLD_FREE = 4.2e-3            # m, flap vertices within this of the fold line are left to the paper (graded rows)
SHIP_PRESS_PA = 650.          # Pa, stacking pressure while stored folded: ~50 totes above in a shipping carton
                              # (50 x 69 g x g over 203 x 254 mm) [A]
PAPER_RELAX_P1 = .05          # uncreased paper's bending/tensile stress relaxation per e-fold of time: Nagasawa et al.
                              # 2019 measured p1 0.046-0.057 at a held crease and found it equal to the tensile one [M]


def age_creases(w, t, n=60):
    """Age every hinge of world `w` for `t` seconds at its current angle (the geometry is held: the bag lies pressed in
    its carton). Creases: the crease law (ipcworld.CreaseLaw, ipcworld.crease_return) over a geometric time grid, as in
    its fit. Uncreased panels: their elastic bending relaxes logarithmically like paper's stress
    in tension, a fraction min(1, PAPER_RELAX_P1 ln(1 + t/1 s)) of the elastic angle (0.67 for a week). The contact
    world's panel hinges are rate-independent and do not creep during a simulation (seconds; 3.5 % per second of
    hold by this law, neglected), so storage creep is applied here."""
    th, _ = W.hinge_angles(w.x_free(), w.hinge)
    pan = ~w.is_crease
    d = W.wrap(th - w.theta_p)
    f = np.full(len(th), min(1., PAPER_RELAX_P1*np.log1p(t)))
    if t >= 3600.:
        # A panel hinge held AT its fully plastic moment (|elastic angle| at theta_yield: the paper there is bent far past
        # yield, i.e. it is part of a fold's bent zone) is in the crease regime, not the elastic one: it keeps flowing
        # and after a long hold its moment is gone, as the crease law has it (a crease held a week keeps 179.9 of 180
        # degrees; with the tote law's flow, omega1 sinh(m/m1), at a panel hinge's stiffness the moment left after a week
        # is < 0.1 degree of elastic angle). The logarithmic relaxation is for bending that stayed elastic. Without this
        # a fold that the mesh spreads over several rows keeps 1/3 of every row's yield angle and springs back by tens
        # of degrees (measured).
        f[np.abs(d) >= .95*w.theta_yield] = 1.
    w.theta_p[pan] = W.wrap(w.theta_p[pan] + f[pan]*d[pan])
    ts = np.geomspace(1e-3, t, n)
    for h in np.flatnonzero(w.is_crease):
        law = w.crease_law[h]
        d = float(W.wrap(th[h] - w.theta_p[h]))
        tp, tr, prev = float(w.theta_p[h]), float(w.theta_r[h]), 0.
        for tk in ts:
            s1, s2 = W.crease_return(law, d, tr, tk - prev)
            d -= s1 + s2; tp += s1 + s2; tr += s2; prev = tk
        w.theta_p[h], w.theta_r[h] = float(W.wrap(tp)), tr


def membrane_stretch(w):
    """Per-triangle in-plane elastic right stretch U (n, 2, 2) of the current state against the rest shape. For a
    crumpled triangle (ipcworld.slack_response) it leaves out the hidden length: the stretch that carries force."""
    _, r, _, _ = w.membrane_state()
    Cm = np.zeros((len(r), 2, 2))
    Cm[:, 0, 0], Cm[:, 1, 1] = 1. + 2.*r[:, 0], 1. + 2.*r[:, 1]
    Cm[:, 0, 1] = Cm[:, 1, 0] = r[:, 2]
    lam, V = np.linalg.eigh(Cm)
    return np.matmul(V*np.sqrt(np.maximum(lam, 1e-12))[:, None, :], V.transpose(0, 2, 1))


def apply_membrane_set(w, Up, tri0=0):
    """Give triangles tri0 .. tri0 + len(Up) the permanent in-plane set Up (rest shape Dm <- Up Dm): the membrane's
    counterpart of a hinge's plastic angle. A start state carries it (scene hook "membrane_set")."""
    sl = slice(tri0, tri0 + len(Up))
    Dm = np.linalg.inv(w.Dm_inv)
    Dm[sl] = np.matmul(Up, Dm[sl])
    w.Dm_inv = np.linalg.inv(Dm)
    b = np.zeros((len(w.tris), 3, 2))
    b[:, 1, :] = w.Dm_inv[:, 0, :]
    b[:, 2, :] = w.Dm_inv[:, 1, :]
    b[:, 0, :] = -(b[:, 1, :] + b[:, 2, :])
    w.bmat = b


def relax_membrane(w, t):
    """Storage relaxation of the in-plane stress held for `t` seconds at the current geometry: paper's stress at a
    held strain relaxes by a fraction min(1, PAPER_RELAX_P1 ln(1 + t/1 s)) (0.67 for a week), the law age_creases()
    applies to panel bending; it is measured in tension, so it applies to the membrane first of all. The rest shape of
    every triangle moves that fraction of the way to its current stretch. Returns the set applied (n, 2, 2)."""
    f = min(1., PAPER_RELAX_P1*np.log1p(t))
    Up = np.eye(2) + f*(membrane_stretch(w) - np.eye(2))
    apply_membrane_set(w, Up)
    return Up


def crumple_membrane(w, n_tri=None, on=True):
    """Switch on (or off) the in-plane crumpling of fresh paper while the tote is made (`n_tri`: the sheet's triangles).

    Where the bottom fold crosses the tucked gussets, layers that are joined turn the fold on different radii, so an
    inner layer has more length than its path: pi t per layer between neighbours, 3 pi t = 1.4 mm between the
    innermost and the outermost of four (t = 0.145 mm). And an inner layer's first mesh row, a straight chord, is
    longer than the path its paper takes inside the next layer's nose (an arc and a flat). Real paper takes such
    surplus as wrinkles and curvature in the inner layer; it does not stretch the outer layer, which would break at a
    few percent. The 0.8 mm mesh cannot draw either, so a triangle compressed in its plane beyond the stock's yield
    strain (ShellMat.yield_strain, 0.5 % [E: paper crushes in compression at about half its tensile strength; the force
    there, E_t x 0.005 = 4.5 / 2.2 kN/m MD / CD, is below the tensile strength 7.9 / 3.9 kN/m]; ipcworld.
    CRUMPLE_RESIDUAL of it once the triangle has begun to wrinkle) hides the excess as slack (ipcworld.slack_response,
    IPCWorld._slack_post): its true length is kept, and the patch is stress-free anywhere between the crushed and the
    true length. There is NO yield in tension: paper pulled beyond its strength tears, and a shipping state with a tear
    is refused.
    Fresh paper crumples in shipping_state() only: in a task the contact world has no in-plane plasticity for paper
    without slack; slack made here opens freely in the task, does not come back by itself, and forms again only under
    load (IPCWorld._slack_post)."""
    if not on:
        w.crumple_virgin = None
        return
    v = np.zeros(len(w.tris), bool)
    v[:len(w.tris) if n_tri is None else n_tri] = True
    w.crumple_virgin = v


def _principal_elastic(w, n_tri=None):
    """(largest, smallest) principal elastic membrane strain per triangle (crumpled triangles: without the hidden part)."""
    _, r, _, _ = w.membrane_state()
    r = r[:n_tri]
    mean, rad = .5*(r[:, 0] + r[:, 1]), np.hypot(.5*(r[:, 0] - r[:, 1]), .5*r[:, 2])
    return mean + rad, mean - rad


def apply_slack(w, slack, tri0=0):
    """Give triangles tri0 .. tri0 + len(slack) a start state's slack (rows c1, c2, phi): scene hook "slack"."""
    slack = np.asarray(slack, float)
    w.set_slack(slack[:, :2], slack[:, 2], tri0=tri0)


def hinge_state(w, sheet_index=0):
    """{(i, j) local edge: (theta_p, theta_r)} of every hinge of one sheet (start states carry their plastic history)."""
    sl = w.sheet_slices[sheet_index]
    out = {}
    for h, (a, b, _, _) in enumerate(w.hinge):
        if sl.start <= a < sl.stop and sl.start <= b < sl.stop:
            i, j = sorted((int(a) - sl.start, int(b) - sl.start))
            out[(i, j)] = (float(w.theta_p[h]), float(w.theta_r[h]))
    return out


def apply_hinge_state(w, state, sheet_index=0):
    """Inverse of hinge_state on a freshly built world (scene ipc_setup hook)."""
    sl = w.sheet_slices[sheet_index]
    n = 0
    for h, (a, b, _, _) in enumerate(w.hinge):
        key = tuple(sorted((int(a) - sl.start, int(b) - sl.start)))
        if key in state:
            w.theta_p[h], w.theta_r[h] = state[key]
            n += 1
    if n != len(state):
        raise ValueError(f"hinge state does not match this sheet ({n} of {len(state)} hinges found)")


SHIP_MORPH_S = .5             # s, the rest shape blends from the authored fold to the true paper [A]
SHIP_MORPH_SETTLE_S = .2      # s, the fold then settles on the true paper before anything can tear [A]: the blend is a
                              # numerical continuation, and its last steps leave strains the wrinkling has yet to take
SHIP_NIP_PA = 2.0e5           # Pa, the press that sets the fold after folding (press rolls / platen) [E, order of a
                              # roll nip; chosen from what the fold needs: an inner layer's surplus length arches 4 mm
                              # over ~30 mm, and an arch carrying the laminate's yield force (E_t x 0.005 = 4.5 kN/m)
                              # stands p = N 8a/L^2 = 160 kPa, so only a press above that crushes it flat]
SHIP_NIP_S = .6               # s, the press: a third up, a third held, a third down to the storage pressure [A]
SHIP_COUNTER_S = 3600.        # s, the tote then lies free on the counter's stack before the task starts [A, an hour]


def shipping_state_key(sheet):
    """Cache key of the shipping state: the contact world's physics version, the generator and its physical inputs."""
    import hashlib
    from . import startcache as SC
    phys = repr((Pm.TOTE, Pm.TURN_TOP, Pm.BOTTOM, Pm.TOTE_CREASE, W.contact_caliper(Pm.TOTE),
                 sorted((tuple(sorted(k)), v) for k, v in Pm.FRICTION.items() if "laminate" in k)))
    src = Pm.CONTACT_WORLD_VERSION + phys + "".join(SC.code(f) for f in (
        build, bag_mesh, graded_half, facet_map, crease_edges, fold_section, shipping_state, shipping_state_key,
        age_creases, relax_membrane, apply_membrane_set, membrane_stretch, crumple_membrane)) + repr((
            np.round(np.asarray(sheet.x0), 9).tobytes(), SHIP_MORPH_S, SHIP_HOLD_S, SHIP_STORE_S, SHIP_RELEASE_S,
            SHIP_SETTLE_S, FOLD_ROW, FOLD_GROWTH, SHIP_PRESS_PA, PAPER_RELAX_P1, NOSE_C, NOSE_BLEND, NOSE_PUFF,
            SHIP_NIP_PA, SHIP_NIP_S, SHIP_COUNTER_S, SHIP_MORPH_SETTLE_S, W.SLACK_MIN_STRETCH, W.CRUMPLE_RESIDUAL))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def shipping_state(origin, *, cells_per_gusset=10, seed=0, cache=None, log=print, dump=None, resume=None):
    """The tote as it ships: flat, its bottom flap folded up over the front face, stored a week. (Gussets tucked in
    and the bottom folded against a panel is how flat-bottom gusseted bags are packed; US8486500B2 describes it for a
    woven polypropylene bag. No source was found that shows it for a laminated rope-handle tote [E].)

    No fixture folds it. The folded geometry is authored (build(flap_up=True): every layer on its own nose,
    fold_section) and relaxed on a flat table under the storage pressure SHIP_PRESS_PA while the rest shape blends from
    that geometry to the true paper (SHIP_MORPH_S; the procedure that makes the wrapped present, wrapmap.relaxed_gift),
    then pressed as press rolls set a fold (up to SHIP_NIP_PA and back, SHIP_NIP_S) and held SHIP_HOLD_S at the storage
    pressure: the layers slide and find their own fold radii under contact. Storage follows at that held geometry:
      creases   every crease starts with the memory of the bag as it was formed (the open shape's angle, no
                recoverable set) and is aged SHIP_STORE_S at its folded angle by the crease law (age_creases);
      panels    their elastic bending relaxes by paper's stress relaxation (age_creases);
      membrane  the in-plane stress relaxes by the same law (relax_membrane); the set is part of the start state.
    While it is pressed and held the paper wrinkles in-plane where the fold leaves a layer with surplus length
    (crumple_membrane): the hidden length is stored as slack and is part of the start state (sheet.start_slack).
    The pressure then fades (SHIP_RELEASE_S), the bag lies free (SHIP_SETTLE_S), and an hour on the counter follows
    (SHIP_COUNTER_S: the creases' recovery after release, integrated quasi-statically).
    Returns (sheet with x0 = that state and start_slack = (n_tri, 3) rows c1, c2, phi, hinge state {local edge:
    (theta_p, theta_r)}, membrane set (n_tri, 2, 2: the storage relaxation's small change of rest shape),
    info). Cached in `cache` (default .cache/gift_bags/start_states). `dump`: also save the contact world's state as
    the rest shape reaches the paper (dump + ".morph") and at the end of the hold; `resume`: start from such a
    ".morph" state instead of repeating the blend (the blend is the same computation every time; info records it)."""
    import json, time as _t
    from pathlib import Path
    target_origin = np.asarray(origin, float)
    sheet = build(np.zeros(3), cells_per_gusset=cells_per_gusset, seed=seed, flap_up=True, graded=True, holes=True)
    n_tri = len(sheet.tris)
    from . import startcache as SC
    path = SC.locate(f"flat_tote_shipping_state_{shipping_state_key(sheet)}.npz", cache)
    if path.is_file():
        d = np.load(path, allow_pickle=False)
        sheet.x0 = d["x"] + target_origin
        state = {(int(i), int(j)): (float(tp), float(tr)) for i, j, tp, tr in d["hinges"]}
        sheet.start_slack = d["slack"]
        return sheet, state, d["membrane"], dict(cached=str(path), **json.loads(str(d["info"])))
    n0 = len(sheet.x0)
    flat = sheet.render["flat_extended"]
    X = np.asarray(sheet.X2, float)
    V, F = W.box_mesh((.4, .4, .02))
    table = W.Body("fold_table", V, F, kind="static", p=np.array([0., 0., -.02]), R=np.eye(3), mu=.4, fmat="counter")
    w = W.IPCWorld([sheet], [], [table], mu_pairs=Pm.FRICTION, friction_strict=True)
    E = W.edges_of(np.asarray(sheet.tris))
    L0 = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)

    def strain():
        xs = w.x_free()
        return np.linalg.norm(xs[E[:, 0]] - xs[E[:, 1]], axis=1)/L0 - 1
    flap = np.flatnonzero(flat[:, 1] < -FOLD_ROW)
    area_v = np.zeros(w.n_free)
    np.add.at(area_v, np.asarray(w.tris).ravel(), np.repeat(w.area/3., 3))
    press = np.zeros((w.n_free, 3))
    press[:, 2] = -SHIP_PRESS_PA*area_v                  # storage pressure: a downward load per vertex [A]
    dt = 4e-3
    n_m, n_h = int(round(SHIP_MORPH_S/dt)), int(round((SHIP_NIP_S + SHIP_HOLD_S)/dt))
    n_r, n_s = int(round(SHIP_RELEASE_S/dt)), int(round(SHIP_SETTLE_S/dt))
    n_nip = int(round(SHIP_NIP_S/dt))
    info = dict(authored_max_edge_strain=float(np.abs(strain()).max()))
    W.prepare_rest_morph(w)
    rows, t0 = [], _t.perf_counter()

    def row(i):
        xs = w.x_free()
        e = strain()
        r = dict(t=round(w.time, 3), iters=w.stats["last_iters"], unconverged=w.stats["unconverged"],
                 max_strain=float(np.abs(e).max()), p99_strain=float(np.percentile(np.abs(e), 99)),
                 thickness_mm=1e3*float(np.ptp(xs[:n0, 2])), flap_top_mm=1e3*float(xs[flap, 2].max()),
                 splits=len(w.tear_events))
        rows.append(r)
        if log:
            log(f"  shipping state {r}")
    # the paper wrinkles where the fold leaves a layer with surplus length: while the rest shape blends (so that no layer
    # is first stretched against a neighbour that has yet to give), while the fold settles, is pressed and is held
    crumple_membrane(w, n_tri)
    n_x = int(round(SHIP_MORPH_SETTLE_S/dt))
    if resume is not None:
        import pickle
        W.end_rest_morph(w, keep_suspended=True)
        with open(resume, "rb") as f_:
            w.set_state(pickle.load(f_))
        # a saved blend state can carry crack-band damage of the blend itself: not part of the paper
        w.damage[:] = 0.
        w.kappa_hist[:] = 0.
        w.suspend_tearing = True
        w.fixture_force = press
        info["resumed_from_blend_state"] = str(resume)
        row(-1)
    for i in range(n_m if resume is not None else 0, n_m + n_x + n_h):
        if i < n_m:
            lam = (i + 1)/n_m
            W.set_rest_blend(w, lam)
            w.fixture_force = press*lam
        elif i < n_m + n_x:
            if w._morph is not None:
                W.end_rest_morph(w, keep_suspended=True)
                if dump is not None:                          # DIAGNOSTIC: the state as the rest shape reaches the paper
                    import pickle
                    with open(str(dump) + ".morph", "wb") as f_:
                        pickle.dump(w.get_state(), f_)
        else:
            if w.suspend_tearing:                             # from here on the paper can tear
                w.suspend_tearing = False
                pr_ = _principal_elastic(w, n_tri)
                info.update(settled=dict(row(-1) or rows[-1]), settled_tensile_strain_max=float(pr_[0].max()),
                            settled_compressive_strain_min=float(pr_[1].min()),
                            settled_crumpled_triangles=int((w.slack_c[:n_tri] < 0.).any(1).sum()))
            # the press that sets the fold: up to SHIP_NIP_PA and back down to the storage pressure
            j = i - n_m - n_x
            u = 3.*(j + 1)/n_nip if j < n_nip else 3.
            prof = min(u, 1.) if u <= 2. else max(3. - u, 0.)
            w.fixture_force = press*(1. + (SHIP_NIP_PA/SHIP_PRESS_PA - 1.)*prof)
        w.step(dt)
        if i % 25 == 0:
            row(i)
    row(-1)
    crumple_membrane(w, on=False)                         # from here on only paper that already has slack wrinkles
    e = strain()
    th, _ = W.hinge_angles(w.x_free(), w.hinge)
    cr = np.asarray(w.is_crease, bool)
    info.update(held=dict(rows[-1]), held_strain_over_half_percent=int((np.abs(e) > .005).sum()),
                held_max_elastic_stretch=float(np.abs(np.linalg.eigvalsh(membrane_stretch(w)[:n_tri]) - 1.).max()),
                held_crease_elastic_deg_max=float(np.degrees(np.abs(W.wrap(th - w.theta_p))[cr].max())),
                held_panel_elastic_deg_max=float(np.degrees(np.abs(W.wrap(th - w.theta_p))[~cr].max())))
    if dump is not None:                                  # DIAGNOSTIC: the held state before storage
        import pickle
        with open(dump, "wb") as f_:
            pickle.dump(w.get_state(), f_)
    # storage at the held geometry
    th_open, _ = W.hinge_angles(np.r_[X, np.zeros((w.n_free - len(X), 3))], w.hinge)
    w.theta_p[cr] = th_open[cr]                          # as formed: the open bag, creases unloaded there
    w.theta_r[cr] = 0.
    age_creases(w, SHIP_STORE_S)
    tc = np.asarray(flat)[np.asarray(sheet.tris)].mean(1)
    cr_t = (w.slack_c[:n_tri] < 0.).any(1)
    e_h, r_h, S_h, k_h = w.membrane_state()
    # hidden length per triangle along its slack axes (m): (1 - crushed stretch) x the triangle's size sqrt(2 A)
    hid = (1. - np.sqrt(1. + 2.*w.slack_c[:n_tri]))*np.sqrt(2.*w.area[:n_tri])[:, None]
    pr = .5*(r_h[:n_tri, 0] + r_h[:n_tri, 1]) + np.hypot(.5*(r_h[:n_tri, 0] - r_h[:n_tri, 1]), .5*r_h[:n_tri, 2])
    Xt = np.array([min(w.mats[m].X_md/w.mats[m].Et_md, w.mats[m].X_cd/w.mats[m].Et_cd) if w.mats[m].X_md > 0 else np.inf
                   for m in w.tri_mat[:n_tri]])
    info.update(crumpled_triangles=int(cr_t.sum()), crumple_updates=int(w.stats.get("crumple_updates", 0)),
                crumple_stretch_min=float(np.sqrt(1. + 2.*w.slack_c[:n_tri].min())) if cr_t.any() else 1.,
                crumple_hidden_mm_max=1e3*float(hid.max()), crumple_at_floor=int((w.slack_c[:n_tri] <= .5*(W.SLACK_MIN_STRETCH**2 - 1.) + 1e-9).any(1).sum()),
                crumpled_at_fold_ends=int((cr_t & (np.abs(tc[:, 0]) > W_BAG/2 - .012) & (np.abs(tc[:, 1]) < .012)).sum()),
                crumpled_within_12mm_of_fold_line=int((cr_t & (np.abs(tc[:, 1]) < .012)).sum()),
                held_tensile_strain_max=float(pr.max()), held_tensile_strain_over_strength=int((pr > Xt).sum()))
    Up = relax_membrane(w, SHIP_STORE_S)[:n_tri]
    for i in range(n_r + n_s):
        f = max(0., 1. - i/n_r)
        w.fixture_force = press*f if f > 0 else None
        w.step(dt)
        if i % 25 == 0:
            row(i)
    row(-1)
    released = dict(rows[-1])
    # on the counter: free for SHIP_COUNTER_S. A released fold keeps opening for a long time as its recoverable set
    # flows back (the crease law; Farain 2020: logarithmic for days) and bent panels creep under their own weight. The
    # hour is integrated in growing time steps at the current geometry, the bag re-equilibrated after each, so the task
    # does not start in the fast first seconds of that recovery.
    prev = 0.
    for tk in np.geomspace(1., SHIP_COUNTER_S, 10):
        age_creases(w, tk - prev)
        prev = tk
        for _ in range(25):
            w.step(dt)
        row(-1)
    z_a = float(w.x_free()[flap, 2].max())
    for _ in range(25):
        w.step(dt)
    row(-1)
    info.update(released=released, counter_s=SHIP_COUNTER_S,
                flap_top_rate_mm_per_s_at_end=1e3*(float(w.x_free()[flap, 2].max()) - z_a)/(25*dt))
    xs = w.x_free()[w.sheet_slices[0]].copy()
    state = hinge_state(w)
    e = strain()
    th, _ = W.hinge_angles(w.x_free(), w.hinge)
    fold = [h for h, (a, b, _, _) in enumerate(w.hinge) if cr[h] and a < n0 and b < n0
            and abs(flat[a, 1]) < 1e-9 and abs(flat[b, 1]) < 1e-9]
    info.update(wall_s=_t.perf_counter() - t0, sim_s=w.time, unconverged=w.stats["unconverged"], steps=w.stats["steps"],
                splits=len(w.tear_events), vertices=int(len(xs)), mesh_vertices=int(n0),
                max_edge_strain=float(np.abs(e).max()), p99_edge_strain=float(np.percentile(np.abs(e), 99)),
                flat_thickness_mm=1e3*float(np.ptp(xs[:n0, 2])), flap_top_mm=rows[-1]["flap_top_mm"],
                flap_top_held_mm=info["held"]["flap_top_mm"], store_s=SHIP_STORE_S,
                membrane_set_max=float(np.abs(Up - np.eye(2)).max()),
                fold_line_angle_deg_median=float(np.degrees(np.median(np.abs(th[fold])))) if fold else None,
                fold_line_set_deg_median=float(np.degrees(np.median(np.abs(w.theta_p[fold])))) if fold else None,
                fold_line_recoverable_deg_median=float(np.degrees(np.median(np.abs(w.theta_r[fold])))) if fold else None,
                log=rows)
    hinges = np.array([(i, j, tp, tr) for (i, j), (tp, tr) in state.items()])
    slack = np.c_[w.slack_c, w.slack_phi][:n_tri]
    e_f, r_f, _, k_f = w.membrane_state()
    info.update(slack_strain_min=float(slack[:, :2].min()), hidden_strain_at_rest_min=float(k_f[:n_tri].min()),
                elastic_strain_at_rest_max=float(np.abs(r_f[:n_tri]).max()))
    np.savez(path, x=xs, hinges=hinges, membrane=Up, slack=slack, info=json.dumps(info))
    sheet.start_slack = slack
    sheet.x0 = xs + target_origin
    info["built"] = str(path)
    return sheet, state, Up, info


def shipping_state_status(cells_per_gusset=10, cache=None):
    """Whether the shipping state for the current code is cached and sound, without building anything:
    dict(key, path, cached, sound, reason). Unsound = the cached state has vertex splits (the tote tore while it was
    made); the counter scene refuses such a state and the catalogue reports the counter tasks unavailable."""
    import json
    from pathlib import Path
    sheet = build(np.zeros(3), cells_per_gusset=cells_per_gusset, seed=0, flap_up=True, graded=True, holes=True)
    from . import startcache as SC
    key = shipping_state_key(sheet)
    path = SC.locate(f"flat_tote_shipping_state_{key}.npz", cache)
    out = dict(key=key, path=str(path), cached=path.is_file(), where=SC.where(path), sound=None, reason=None)
    if out["cached"]:
        d = np.load(path, allow_pickle=False)
        info = json.loads(str(d["info"]))
        torn = int(info.get("splits", 0)) > 0 or len(d["x"]) != len(sheet.x0)
        out.update(sound=not torn, splits=int(info.get("splits", 0)), vertices=int(len(d["x"])), mesh_vertices=int(len(sheet.x0)),
                   flat_thickness_mm=info.get("flat_thickness_mm"), flap_top_mm=info.get("flap_top_mm"))
        if torn:
            out["reason"] = (f"the cached shipping state of the tote is torn ({out['splits']} vertex splits, {out['vertices']} "
                             f"vertices against the mesh's {out['mesh_vertices']}): it tore while it was made")
    return out


def shipping_key(sheet):
    """Cache key of the shipping state: the contact world's physics version, the generator, and the physical inputs the
    fold uses (tote stocks, crease law, laminate friction pairs, caliper), not the whole parameter file (unrelated
    ledger lines must not rebuild it)."""
    import hashlib
    from . import startcache as SC
    phys = repr((Pm.TOTE, Pm.TURN_TOP, Pm.BOTTOM, Pm.TOTE_CREASE, W.contact_caliper(Pm.TOTE),
                 sorted((tuple(sorted(k)), v) for k, v in Pm.FRICTION.items() if "laminate" in k)))
    src = Pm.CONTACT_WORLD_VERSION + phys + "".join(SC.code(f) for f in (
        build, bag_mesh, graded_half, facet_map, crease_edges, shipping_fold, shipping_key, age_creases)) + repr((
        np.round(np.asarray(sheet.x0), 9).tobytes(), SHIP_FOLD_S, SHIP_HOLD_S, SHIP_STORE_S, SHIP_RELEASE_S,
        SHIP_SETTLE_S, FOLD_FREE, FOLD_ROW, FOLD_GROWTH, SHIP_PRESS_PA, PAPER_RELAX_P1))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def shipping_fold(origin, *, cells_per_gusset=10, seed=0, cache=None, log=print, video_recorder=None):
    """Not used by the scene, which uses shipping_state above. Folded this way the tote tears at the two ends of the fold
    line (12 vertex splits, edge strain 12-25 % there) and the flap springs back to ~64 degrees when the fixture lets go.
    Cause: the fixture drives every flap vertex to one rigid 180-degree rotation about a single axis, but each stacked
    layer turns the fold at its own radius and must end about pi r short of that target (up to 1.6 mm for the outer
    layer), so the layers are stretched against each other: the stored energy is in-plane strain over the whole bag.

    The bottom-extended flat bag (build(flap_up=False), graded mesh)
    lies on a flat table; ASSISTED fixture springs (2 kN/m, 0.5 N cap per vertex) on the flap beyond FOLD_FREE of the
    bottom fold turn it 180 degrees about the fold line up over the front face (SHIP_FOLD_S) and hold it; every crease
    then ages for SHIP_STORE_S at that held geometry; the springs fade out and the bag lies free (SHIP_SETTLE_S).
    Contact keeps the four layers apart; the fold forms in the graded rows. `origin`: as build(); the flat bag's body
    spans the fold line to the mouth, so the folded bag covers W x H from it.
    Returns (sheet with x0 = the folded state, hinge state {local edge: (theta_p, theta_r)}, info). Cached in `cache`
    (default .cache/gift_bags/start_states) by a hash of the generator and the world code."""
    import hashlib, inspect, time as _t, json
    from pathlib import Path
    from scipy.spatial.transform import Rotation
    target_origin = np.asarray(origin, float)
    origin = np.zeros(3)                                      # folded at a canonical origin, then translated
    sheet = build(origin, cells_per_gusset=cells_per_gusset, seed=seed, flap_up=False, graded=True, holes=True)
    cache = Path(cache) if cache else Path(__file__).resolve().parents[1]/".cache"/"gift_bags"/"start_states"
    cache.mkdir(parents=True, exist_ok=True)
    key = shipping_key(sheet)
    path = cache/f"flat_tote_shipping_fold_{key}.npz"
    if path.is_file():
        d = np.load(path, allow_pickle=False)
        sheet.x0 = d["x"]
        state = {(int(i), int(j)): (float(tp), float(tr)) for i, j, tp, tr in d["hinges"]}
        info = dict(cached=str(path), **json.loads(str(d["info"])))
        sheet.x0 = sheet.x0 + target_origin
        sheet.render["flap_up"] = True
        return sheet, state, info
    flat = sheet.render["flat"]
    cal = W.contact_caliper(Pm.TOTE)
    pitch = cal["rest_sheet_sheet"]
    # the table: a static slab whose top is the support plane at origin z
    o = np.asarray(origin, float)
    V, F = W.box_mesh((.4, .4, .02))
    table = W.Body("fold_table", V, F, kind="static", p=o + np.array([0., 0., -.02]), R=np.eye(3), mu=.4, fmat="counter")
    w = W.IPCWorld([sheet], [], [table], mu_pairs=Pm.FRICTION, friction_strict=True)
    m = np.array([0., 1., 0.])                                  # build()'s default mouth direction
    wdir = np.cross(m, [0., 0., 1.])
    base = cal["rest_sheet_rigid"] + 5e-6
    axis_pt = o + np.array([0., 0., base + 3.5*(pitch + 5e-6)])
    flap = np.flatnonzero(flat[:, 1] < -FOLD_FREE)
    x0 = w.x_free()[flap].copy()
    k, cap = 2e3, .5
    dt = 4e-3
    t0 = _t.perf_counter()
    n_fold, n_hold = int(round(SHIP_FOLD_S/dt)), int(round(SHIP_HOLD_S/dt))
    n_rel, n_set = int(round(SHIP_RELEASE_S/dt)), int(round(SHIP_SETTLE_S/dt))

    def target(phi):
        return axis_pt + Rotation.from_rotvec(-phi*wdir).apply(x0 - axis_pt)
    log_rows = []
    # stacking pressure in storage: a downward load per vertex = pressure x its lumped area (every layer; layers in
    # contact pass it on to the table) [A]
    area_v = np.zeros(w.n_free)
    np.add.at(area_v, np.asarray(w.tris).ravel(), np.repeat(w.area/3., 3))
    press = np.zeros((w.n_free, 3))
    press[:, 2] = -SHIP_PRESS_PA*area_v
    n_ramp = int(round(.3/dt))
    for i in range(n_fold + n_hold + n_rel + n_set):
        if i < n_fold:
            tau = (i + 1)/n_fold
            w.fixture_springs = [(flap, target(np.pi*tau**3*(10 - 15*tau + 6*tau**2)), k, cap)]
        elif i < n_fold + n_hold:
            w.fixture_springs = [(flap, target(np.pi), k, cap)]
            w.fixture_force = press*min(1., (i - n_fold + 1)/n_ramp)
        elif i == n_fold + n_hold:
            age_creases(w, SHIP_STORE_S)
            w.fixture_springs = [(flap, target(np.pi), k, cap)]
        elif i < n_fold + n_hold + n_rel:
            f = 1. - (i - n_fold - n_hold)/n_rel
            w.fixture_springs = [(flap, target(np.pi), k*f, cap*f)]
            w.fixture_force = press*f
        else:
            w.fixture_springs = []
            w.fixture_force = None
        w.step(dt)
        if video_recorder is not None:
            video_recorder()
        if i % 25 == 0:
            xs = w.x_free()
            row = dict(t=round(w.time, 3), iters=w.stats["last_iters"], unconverged=w.stats["unconverged"],
                       flap_dev_mm=1e3*float(np.linalg.norm(xs[flap] - target(np.pi), axis=1).mean()) if i >= n_fold else None,
                       zmax_mm=1e3*float(xs[:, 2].max() - o[2]),
                       zmax_at_flat_mm=(1e3*flat[int(np.argmax(xs[:len(flat), 2]))]).round(1).tolist())
            log_rows.append(row)
            if log:
                log(f"  shipping fold {row}")
    xs = w.x_free()[w.sheet_slices[0]].copy()
    state = hinge_state(w)
    E = W.edges_of(np.asarray(sheet.tris))
    X = np.asarray(sheet.X2, float)
    strain = np.linalg.norm(xs[E[:, 0]] - xs[E[:, 1]], axis=1)/np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1) - 1
    perim = [h for h, (a, b, _, _) in enumerate(w.hinge) if w.is_crease[h] and
             sheet.crease_names.get(tuple(sorted((int(a), int(b))))) == "bottom_perimeter"]
    th, _ = W.hinge_angles(w.x_free(), w.hinge)
    info = dict(wall_s=_t.perf_counter() - t0, sim_s=w.time, unconverged=w.stats["unconverged"], steps=w.stats["steps"],
                max_edge_strain=float(np.abs(strain).max()), p99_edge_strain=float(np.percentile(np.abs(strain), 99)),
                flat_thickness_mm=1e3*float(np.ptp(xs[:, 2])), store_s=SHIP_STORE_S,
                perimeter_angle_deg_median=float(np.degrees(np.median(np.abs(th[perim])))) if perim else None,
                perimeter_set_deg_median=float(np.degrees(np.median(np.abs(w.theta_p[perim])))) if perim else None,
                perimeter_recoverable_deg_median=float(np.degrees(np.median(np.abs(w.theta_r[perim])))) if perim else None,
                log=log_rows)
    hinges = np.array([(i, j, tp, tr) for (i, j), (tp, tr) in state.items()])
    np.savez(path, x=xs, hinges=hinges, info=json.dumps(info))
    sheet.x0 = xs + target_origin
    sheet.render["flap_up"] = True
    info["built"] = str(path)
    return sheet, state, info
