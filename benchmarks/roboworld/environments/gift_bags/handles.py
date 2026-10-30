"""Rope handles of the laminated Euro tote: a 5 mm braided PP rope loop per face, through two holes punched in the
turn-top band, knotted inside. Anchoring is contact, not attachment: each rope end is welded to its
knot (the knot is the rope's own end, tied), and the knot is a rigid lump that cannot pass the hole, so a pull on
the handle reaches the bag as the knot's bearing on the inside of the band around the hole.

Holes: one mesh cell of the turn-top band each (10.2 mm square, bagnet.bag_mesh(holes=True); a punched hole is
~5 mm [V]; the cell is the mesh's resolution [A]). Knot: 14 x 14 x 6 mm (an overhand knot in 5 mm rope,
pressed flat in the stack [E]: 14 x 14 x 5 mm); it cannot pass a 10.2 mm square hole in any orientation (its smallest
section, 14 x 5 mm, would need its 14 mm side to fit a 14.4 - 5 = 9.4 mm diagonal opening; checked by
`knot_cannot_pass`).

The start state is assembled the way a flat tote with handles is: the folded shipping state (bagnet.shipping_state)
is lifted, the knots are placed inside between the faces under the holes, the ropes run from the knots out through
the holes and lie flat as loops (front loop on the front face, back loop under the bag), and the bag's rest shape is
blended from that authored geometry to the true paper (ipcworld.prepare_rest_morph) while everything settles under
gravity on the stack: the faces close over the knots (the lump a real flat tote has at its handles) and the bag
drapes over its back loop. Cached by a hash of the inputs.
"""
import numpy as np
from . import ipcworld as W
from . import params as Pm

# 5 mm braided PP rope: 12.5 g/m (D: PP 0.91 g/cm3 x pi r^2 x 0.7 packing; supplier coils of 6 mm PP
# rope weigh 11.0-11.4 g/m [V, 2.42-2.5 kg per 220 m]); EA 15 kN (E, break-load class over ~23 % stretch);
# EI 2e-5 N m^2 (E, a soft braid that droops)
# Contact compliance: the rope's surface can be squeezed by up to 0.5 mm across its radius; stiffness set so that a
# line load of 300 N/m (about 5 N pinched over 15 mm) compresses it 0.25 mm [E: an estimate from pinching such a rope,
# not a measurement]. With the paper's range (29 um on a 2.5 mm radius) a rope sliding over paper leaves the barrier's
# range within a step and the solver does not converge.
ROPE = W.RodMat("5 mm braided PP rope", radius=2.5e-3, lin_rho=12.5e-3, EA=15e3, EI=2e-5, color=(.95, .93, .88),
                contact_range=5e-4, contact_kappa=5.92e10)
LOOP_LEN = .400            # m, rope between the two holes of one handle (400 mm loop) [V]
SEG = .006                 # m, rope segment length [A]
KNOT_HALF = (.007, .007, .0025)  # m: 14 x 14 x 5 mm, an overhand knot of 5 mm rope pressed flat (no thinner than the
                                 # rope) [E]; inside the turn-top it narrows the mouth by ~5.3 mm per face
KNOT_MASS = .0012          # kg: ~0.1 m of rope in the knot (12.5 g/m) [D]
WELD_K = 2e5               # N/m, rope end to its knot (the knot is the rope; stiff) [A]
GAP = 4e-4                 # m, authored clearance between assembled parts (> every contact distance)
ASSEMBLE_S = .30           # s, rest-shape blend from the authored geometry to the paper
SETTLE_S = 3.0             # s, settle after the blend


def knot_cannot_pass(hole_side, half=KNOT_HALF):
    """True if no orientation of the knot box fits through a square hole of side hole_side (slab test on its two
    smallest sections: a w x l rectangle passes a square of side a only if l <= a sqrt(2) - w)."""
    a, b, c = sorted(2*np.asarray(half))
    passes = lambda w, l: (l <= hole_side and w <= hole_side) or l <= hole_side*np.sqrt(2) - w
    return not (passes(a, b) or passes(a, c))


def weld_term(pairs, k=WELD_K):
    """Extra energy for IPCWorld.extra_terms: 1/2 k |x_v - (A r + p)_b|^2 per (free vertex v, body b, local point r)."""
    def term(w, q, xc_prev, dt, want):
        E = 0.
        g = np.zeros(w.n_dof)
        coo = ([], [], [])
        for v, b, r in pairs:
            xb, J = w._point(b, np.asarray(r, float), q)
            d = q[3*v:3*v + 3] - xb
            E += .5*k*float(d @ d)
            g[3*v:3*v + 3] += k*d
            cols, vals = J
            for c in range(3):
                np.add.at(g, cols[c], -k*d[c]*vals[c])
                if want:
                    idx = np.r_[3*v + c, cols[c]]
                    val = np.r_[1., -vals[c]]
                    rr, cc = np.meshgrid(idx, idx, indexing="ij")
                    coo[0].append(rr.ravel()); coo[1].append(cc.ravel()); coo[2].append((k*np.outer(val, val)).ravel())
        return E, g, coo
    return term


def knot_body(name, p):
    hx, hy, hz = KNOT_HALF
    V, F = W.box_mesh(KNOT_HALF, n=(2, 2, 1))
    S = KNOT_MASS/3.*np.diag([hx*hx, hy*hy, hz*hz])
    return W.Body(name, V, F, mass=KNOT_MASS, S=S, kind="dynamic", p=np.asarray(p, float), R=np.eye(3), mu=.3,
                  fmat="knot", render=dict(material="rope_knot", half=KNOT_HALF))


def _polyline(points, seg=SEG):
    """Resample a polyline at ~seg spacing (keeps its ends)."""
    P = np.asarray(points, float)
    L = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    n = max(2, int(np.ceil(L[-1]/seg)) + 1)
    s = np.linspace(0., L[-1], n)
    return np.column_stack([np.interp(s, L, P[:, k]) for k in range(3)])


def hole_sites(sheet, x):
    """{'front'|'back': [(centre, outward normal), (centre, normal)]} of the four holes in the bag's current state
    (bag_mesh(holes=True) stores the corner vertices of each hole in sheet.render['holes'])."""
    out = {}
    for face, holes in sheet.render["holes"].items():
        sites = []
        for ids in holes:
            P = x[np.asarray(ids)]
            c = P.mean(0)
            n = np.cross(P[1] - P[0], P[2] - P[0]); n /= np.linalg.norm(n)
            sites.append((c, n))
        out[face] = sites
    return out


def author(sheet, support_z):
    """Authored (not yet physical) geometry of the bag with its handles, from the folded shipping state lying on a
    support at height support_z: (bag x0 lifted and bumped, [front rope x, back rope x], knot centres
    [front -x, front +x, back -x, back +x], weld spec [(rope index, rope vertex index, knot index, local point)])."""
    from scipy.spatial import cKDTree
    x = np.asarray(sheet.x0, float).copy()
    t = W.contact_caliper(Pm.TOTE)["t"]                       # contact caliper of the tote stock
    r, kz = ROPE.radius, KNOT_HALF[2]
    lift = 2*r + 3*GAP
    x[:, 2] += lift                                            # the back loop runs under the bag
    sites = hole_sites(sheet, x)
    kinds = sheet.render["kinds"]
    T = np.asarray(sheet.tris)
    front_v = np.unique(T[kinds == "front"])
    # knots: back knot on the back face's inside, front knot above it; the front face is raised over them
    knots, bump_h = [], []
    for (cf, nf), (cb, nb) in zip(sites["front"], sites["back"]):
        zb = cb[2] + t/2 + GAP + kz
        zf = zb + 2*kz + GAP
        knots.append(((cf[0], cf[1], zf), (cb[0], cb[1], zb)))
        bump_h.append(zf + kz + GAP + t/2 - cf[2])
    H = max(bump_h)
    R0, R1 = .011, .045
    c2 = np.array([[k[0][0], k[0][1]] for k in knots])
    d = np.min(np.linalg.norm(x[front_v, None, :2] - c2[None], axis=2), axis=1)
    b = np.where(d <= R0, 1., np.where(d >= R0 + R1, 0., np.cos(.5*np.pi*(d - R0)/R1)**2))
    x[front_v, 2] += H*b
    # rope paths: legs toward the bag's bottom (-y), a half circle joining them
    tree = cKDTree(x[:, :2])

    def above(p, rad=.008):
        ids = tree.query_ball_point(p[:2], rad)
        return (x[ids, 2].max() if ids else support_z) + t/2 + r + 2*GAP
    ropes, welds = [], []
    sep = abs(c2[1, 0] - c2[0, 0])/2
    for face in ("front", "back"):
        ki = [0, 1] if face == "front" else [2, 3]
        ends = [np.array(knots[0][0 if face == "front" else 1]), np.array(knots[1][0 if face == "front" else 1])]
        sgn = 1. if face == "front" else -1.
        start = [e + sgn*np.array([0., 0., kz + r + GAP]) for e in ends]
        if face == "front":
            out_z = lambda p: above(p)
        else:
            out_z = lambda p: support_z + r + GAP
        v_len = abs(out_z(ends[0]) - start[0][2])
        leg = (LOOP_LEN - 2*v_len - np.pi*sep)/2
        assert leg > .02, leg
        y0 = ends[0][1]
        pts = [start[0]]
        for s in np.linspace(0., leg, 30):
            p = np.array([ends[0][0], y0 - s, 0.]); p[2] = out_z(p); pts.append(p)
        cx, cy = (ends[0][0] + ends[1][0])/2, y0 - leg
        for a in np.linspace(np.pi, 2*np.pi, 40)[1:-1]:
            p = np.array([cx + sep*np.cos(a), cy + sep*np.sin(a), 0.]); p[2] = out_z(p); pts.append(p)
        for s in np.linspace(leg, 0., 30):
            p = np.array([ends[1][0], y0 - s, 0.]); p[2] = out_z(p); pts.append(p)
        pts.append(start[1])
        # The rope passes through its hole: each end node is at its knot's centre, inside the bag, and the first
        # segment crosses the paper through the hole to the first node outside.
        R = _polyline([ends[0]] + pts + [ends[1]])
        ropes.append(R)
        loc = np.zeros(3)
        welds += [(len(ropes) - 1, 0, ki[0], loc), (len(ropes) - 1, len(R) - 1, ki[1], loc)]
    centres = [knots[0][0], knots[1][0], knots[0][1], knots[1][1]]
    return x, ropes, centres, welds


def assembled(sheet, hinges, supports, *, membrane=None, cache=None, log=print, status_only=False):
    """The flat tote with its handles, settled (see the module docstring). supports: the bodies the bag lies on (the
    top one's top face must be the bag's support plane). Dynamic supports (the stack's lower totes) settle with the
    bag, closing their authored clearances, and are left at their settled poses (b.p, b.R updated in place): the
    start state is the whole settled stack, and a scene that kept the authored poses would start with the bag and
    ropes inside the contact minimum of a plate that has not dropped yet (83 um over four plates, which the contact
    world refuses). Returns (sheet with x0, [Rod, Rod], [knot Body x4], welds, hinge state after assembly, info);
    welds as in author() with knot indices into the returned knot list. Cached by a hash. status_only: build nothing,
    return dict(key, path, where) of this assembly's start-state file (startcache.where)."""
    import hashlib, json, time as _t
    from pathlib import Path
    from . import bagnet as B
    from . import startcache as SC
    support_z = float(np.asarray(sheet.x0)[:, 2].min()) - W.contact_caliper(Pm.TOTE)["rest_sheet_rigid"]
    hx = np.array(sorted((k, v) for k, v in hinges.items()), dtype=object)
    src = Pm.CONTACT_WORLD_VERSION + SC.code(author) + SC.code(assembled) + SC.code(weld_term) \
        + repr((ROPE, LOOP_LEN, SEG, KNOT_HALF, KNOT_MASS, WELD_K, GAP, ASSEMBLE_S, SETTLE_S, Pm.TOTE, Pm.TURN_TOP,
                Pm.BOTTOM, Pm.TOTE_CREASE, sorted((tuple(sorted(k)), v) for k, v in Pm.FRICTION.items())))
    src += repr([(b.name, np.round(b.p, 6).tolist(), np.round(np.ptp(np.asarray(b.verts), axis=0), 6).tolist()) for b in supports])
    slack = getattr(sheet, "start_slack", None)   # the shipping state's crumpled triangles (bagnet.crumple_membrane)
    key = hashlib.sha256(src.encode() + np.round(np.asarray(sheet.x0), 9).tobytes() + repr(hx.tolist()).encode()
                         + (np.round(np.asarray(membrane), 9).tobytes() if membrane is not None else b"")
                         + (np.round(np.asarray(slack), 9).tobytes() if slack is not None else b"")).hexdigest()[:16]
    path = SC.locate(f"flat_tote_handles_{key}.npz", cache)
    if status_only:
        return dict(key=key, path=str(path), where=SC.where(path))
    x, ropes, centres, welds = author(sheet, support_z)
    knots = [knot_body(f"knot_{f}_{s}", c) for (f, s), c in zip((("front", 0), ("front", 1), ("back", 0), ("back", 1)), centres)]
    # the two nodes at each end belong to the knot there (the knot is the rope): no rope-knot contact for them
    rods = [W.Rod(f"handle_{f}", R, ROPE, fmat="rope",
                  attached=((0, f"knot_{f}_0"), (1, f"knot_{f}_0"), (len(R) - 1, f"knot_{f}_1"), (len(R) - 2, f"knot_{f}_1")))
            for f, R in zip(("front", "back"), ropes)]
    if path.is_file():
        d = np.load(path, allow_pickle=False)
        sheet.x0 = d["x"]
        for k, rd in enumerate(rods):
            rd.x0 = d[f"rope{k}"]
        for k, kb in enumerate(knots):
            kb.p, kb.R = d["knot_p"][k], d["knot_R"][k]
        for k, sb in enumerate(supports):
            sb.p, sb.R = d["support_p"][k], d["support_R"][k]
        state = {(int(i), int(j)): (float(tp), float(tr)) for i, j, tp, tr in d["hinges"]}
        return sheet, rods, knots, welds, state, dict(cached=str(path), **json.loads(str(d["info"])))
    sheet.x0 = x
    w = W.IPCWorld([sheet], rods, list(supports) + knots, mu_pairs=Pm.FRICTION, friction_strict=True)
    B.apply_hinge_state(w, hinges)
    if membrane is not None:                 # the shipping state's in-plane set (bagnet.relax_membrane), before the morph
        B.apply_membrane_set(w, membrane)
    if slack is not None:
        B.apply_slack(w, slack)
    nb = len(supports)
    pairs = [(w.rod_slices[ri].start + vi, nb + ki, loc) for ri, vi, ki, loc in welds]
    w.extra_terms = list(getattr(w, "extra_terms", [])) + [weld_term(pairs)]
    W.prepare_rest_morph(w)
    dt = 4e-3
    n_a, n_s = int(round(ASSEMBLE_S/dt)), int(round(SETTLE_S/dt))
    t0 = _t.perf_counter()
    rows = []
    for i in range(n_a + n_s):
        if i < n_a:
            W.set_rest_blend(w, (i + 1)/n_a)
        elif w._morph is not None:
            W.end_rest_morph(w)
        w.step(dt)
        if i % 25 == 0:
            row = dict(t=round(w.time, 3), iters=w.stats["last_iters"], unconverged=w.stats["unconverged"],
                       ke=float(w.kinetic_energy()))
            rows.append(row)
            if log:
                log(f"  handles {row}")
    if w._morph is not None:
        W.end_rest_morph(w)
    xs = w.x_free()
    if w.sheet_slices[0].stop - w.sheet_slices[0].start != len(x):
        raise RuntimeError("the tote tore while its handles were assembled: the start state would not match the scene")
    sheet.x0 = xs[w.sheet_slices[0]].copy()
    for k, rd in enumerate(rods):
        rd.x0 = xs[w.rod_slices[k]].copy()
    kp = np.array([w.body_pose(nb + k)[0] for k in range(4)])
    kR = np.array([w.body_pose(nb + k)[1] for k in range(4)])
    for k, kb in enumerate(knots):
        kb.p, kb.R = kp[k], kR[k]
    sp = np.array([w.body_pose(k)[0] for k in range(nb)])
    sR = np.array([w.body_pose(k)[1] for k in range(nb)])
    drop = [float(1e6*(np.asarray(sb.p, float)[2] - sp[k][2])) for k, sb in enumerate(supports)]
    for k, sb in enumerate(supports):
        sb.p, sb.R = sp[k], sR[k]
    gap = [float(np.linalg.norm(xs[v] - w._point(b, np.asarray(r), w.q)[0])) for v, b, r in pairs]
    V3 = w.v[:3*w.n_free].reshape(-1, 3)
    rest = dict(tote_speed_um_s_max=1e6*float(np.linalg.norm(V3[w.sheet_slices[0]], axis=1).max()),
                rope_speed_um_s_max=1e6*float(max(np.linalg.norm(V3[sl_], axis=1).max() for sl_ in w.rod_slices)),
                knot_speed_um_s_max=1e6*float(max(np.linalg.norm(w.v[w.body_dof[nb + k]:w.body_dof[nb + k] + 3]) for k in range(4))),
                knot_turn_rate_deg_s_max=float(np.degrees(max(np.linalg.norm(w.v[w.body_dof[nb + k] + 3:w.body_dof[nb + k] + 12])
                                                              for k in range(4)))))
    info = dict(wall_s=_t.perf_counter() - t0, sim_s=w.time, unconverged=w.stats["unconverged"], at_rest=rest,
                weld_gap_max_mm=1e3*max(gap), bag_thickness_at_handles_mm=1e3*float(np.ptp(sheet.x0[:, 2])),
                knot_z_mm=(1e3*(kp[:, 2] - support_z)).round(2).tolist(),
                support_settle_um={sb.name: round(dz, 1) for sb, dz in zip(supports, drop)}, log=rows)
    state = B.hinge_state(w)
    np.savez(path, x=sheet.x0, rope0=rods[0].x0, rope1=rods[1].x0, knot_p=kp, knot_R=kR, support_p=sp, support_R=sR,
             info=json.dumps(info),
             hinges=np.array([(i, j, tp, tr) for (i, j), (tp, tr) in state.items()]))
    info["built"] = str(path)
    return sheet, rods, knots, welds, state, info


def install(ipc, welds, knot_names, rod_offset=0):
    """Scene ipc_setup hook: the rope-end welds on a built world (rods in scene order from rod_offset, knots by name)."""
    names = [b.name for b in ipc.bodies]
    pairs = [(ipc.rod_slices[rod_offset + ri].start + vi, names.index(knot_names[ki]), loc) for ri, vi, ki, loc in welds]
    ipc.extra_terms = list(getattr(ipc, "extra_terms", [])) + [weld_term(pairs)]
    ipc.handle_welds = pairs
