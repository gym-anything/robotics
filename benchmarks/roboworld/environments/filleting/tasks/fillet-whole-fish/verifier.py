"""Outcome of fillet-whole-fish: both fillets cut off in one piece each with no rib bones, laid skin-down in the
perforated fillet tray, the frame and head in the empty pan, the knife held or put down safely.

verify(e) evaluates an episode and returns its gar.evaluation.v1 result; e is the episode's evaluator (task.Evaluator:
the released checks, released_checks; no task-set contract selects this task). Its score is 1 on success, else
PROGRESS_CAP x its progress. prime(e) records the progress measure's start when the environment is built; env.py also
reads RECEPTACLES and STOP_AT_TIME_LIMIT."""
import numpy as np

from filleting import layout as LY, params as PR
from filleting.task import REF, TIME_LIMIT_S, check, near_boxes

PROGRESS_CAP = .95               # an attempt that is not a success scores below every success (evaluate)
SUNK_MAX = .25                   # share of a fillet's nodes inside the other fillet's footprint at the same height
A3_W = dict(upper=.4, lower=.4, frame=.2, cut=.6, place=.4, sever=.6, quality=.4)    # authored progress weights
CORE_M = 5.25e-3                 # m (1.5 mesh lengths): the meat core starts this far from frame or head tissue
# product receptacles (layout names): a piece descending into one is not thrown tissue (env.py)
RECEPTACLES = ("FILLET_TRAY", "FRAME_PAN")
# the episode ends at TIME_LIMIT_S: env.act runs no tick past it (an action crossing it is cut at the limit)
STOP_AT_TIME_LIMIT = True


def verify(e):
    return e.evaluate_checks(released_checks, progress_cap=PROGRESS_CAP)


def released_checks(e):
    """A3 fillet-whole-fish. Success: every check, each once.

    Per side (upper_/lower_: rest-material labels, so they follow the flesh however the fish is turned). The two
    fillets are two DIFFERENT removed pieces, each holding the most of its own side's muscle (_assign_fillets); a
    removed piece holds no rigid bone (a piece holding the head or a vertebra by the majority of its intact ties is
    frame or head, so a fillet that took a vertebra is not a fillet):
      one_piece   >= 97 % of the side's removed muscle in its fillet;
      yield       the fillet's mass with skin, less any of the other side's muscle, >= 0.33 x gutted (half the
                  published 66 % skilled lower bound);
      no_rib_pieces_over_2mm, no_backbone_or_fin_support_fragments (median fin supports, supraneurals);
      workmanship_score >= 85 (workmanship_A3: NOAA points per 25.4 mm square; compression recorded while the
                  knife's edge was within EDGE_REACH_M of a tet is the cut, not a bruise; a knife score is an
                  incision left in the fillet, while meat left on the bone is priced by yield alone);
      in_tray     resting in the perforated fillet tray: every node inside its walls, the lowest below its rim, no
                  node within TOUCH_M of a robot hand box or of the blade, net motion under 1 mm over 1 s (no
                  centring), and not sunk into the other fillet (more than SUNK_MAX of its nodes inside the
                  other's footprint at the same height: the tissue solver has no tissue-tissue contact, so a
                  fillet laid on the other passes through it; flat seed-0 fillets side by side pass with edges
                  overlapping up to about 40 % of a fillet's width); in A3 the tray stands on
                  the holding-arm side (BENCH in its setup.py), where both fillets fit side by side;
      skin_down   mean outward skin normal within 45 deg of down.
    frame_and_head_in_pan: every rigid bone (head, vertebrae) inside the empty pan's rim and below it
      (layout.FRAME_PAN), every piece holding one resting as above (head on the frame or cut off: both pass).
    knife_not_dropped: held by either hand (or in the right hand's grasp), or put down: blade and handle over the
      bench above its top, outside the tray, the pan and the ice pan, not on the fish or a product (in the block or
      on the board passes; on the floor, on the fish, in a receptacle fails).
    no_blade_contact_with_holding_hand; debris (pieces under 1 g within A2's allowance for each of the two sides:
      1 % of the tissue's mass, at most 20 pieces); no_tissue_thrown (no free piece - none a hand touches -
      faster than 1 m/s outside the tray and the pan, nothing on the floor); time <= 360 s (env.act runs no tick
      past it: STOP_AT_TIME_LIMIT). The rest tests compare only nodes present in both motion snapshots, so the
      evaluation works right after a cut or tear splits nodes.

    Progress (the score when not a success, evaluate): the delivered fraction of the stated outcome, a function of
    the state alone; no method, order or intermediate step is credited.
      P = 0.4 c_upper + 0.4 c_lower + 0.2 c_frame;   c_s = 0.6 cut_s + 0.4 place_s (0 if its fillet is on the floor)
      cut_s = 0.6 sever_s + 0.4 y_s q_s;   place_s = y_s r_s k_s (1/2 + o_s/2);   c_frame = b
      sever_s = free_s + attached_s (1 - R_s/R0_s): free_s / attached_s are the shares of the side's muscle in its
        fillet / in bone-holding pieces (muscle in other loose pieces is lost, so mincing earns nothing); R_s is the
        fewest intact connections (uncut tet faces, interface bonds, flesh-bone ties; tears count as cut near the
        knife's cut) still joining the side's meat core (CORE_M from frame or head, no ties)
        to any rigid bone, a minimum cut (_remaining_connection), so cuts that free nothing (a trim, a chop
        across) leave it as it was; R0_s the same before any cut. A minimum cut reads the narrowest remaining
        connection, so the same opened area reads differently by where it was opened: at half the separating
        surface opened, four stroke orders read 0.25 to 0.61. Tears are counted only near
        the knife's cut, so peeling a fillet off by pulling earns nothing until it comes free. The
        quality term exists only for a removed fillet, so sever_s steps by about 0.1 when a fillet comes free.
      y_s = min(1, yield / threshold); q_s = min(1, one-piece share / 0.97) x 1/(1 + bone pieces in the fillet) x
        min(1, max(0, workmanship) / 85); r_s = the fraction of the fillet's nodes inside the tray's walls if it rests
        there, else 0; k_s = 1 if no more than SUNK_MAX of it is sunk into the other fillet (the in-tray test), else
        1 - sunk_s; o_s = clip((90 deg - skin angle) / 45 deg, 0, 1).
      b = sum over rigid bones of w_j [inside and below the pan's rim] [its piece released and still] gate_p g_pan,
        with w = 1/2 for the head and 1/2 shared by the vertebrae (the head body is 87 % of the rigid bones' mass); a
        bone with no intact tie belongs to the piece holding a bone it is still jointed to, and a chain free of all
        tissue is released if no hand box touches it.
        gate_p = clip(1 - (fillet meat on that piece - t) / (m - t), 0, 1), m half the fish's fillet meat and
        t = 0.3 m (about the meat a passing pair may leave on the bone): a frame still carrying a fillet earns
        nothing, a head cut off first earns its share. g_pan, the same expression of the fillet meat lying inside
        the pan's rim on any piece, withholds the bones' credit while the fish or a fillet lies in the pan (a whole
        fish dropped in, its head torn off by the impact, earns nothing).
    The weights (A3_W) are authored: two fillets and one disposal; cutting before placing."""
    env, w = e.env, e.env.world
    ts = w.tissue
    W = A3_W
    top = LY.TABLE["top_z"]
    Y = .33*PR.v(PR.TROUT)["gutted_kg"]
    lab, frags = e.fragments()
    attach = e.bone_attachment(lab)
    nk = w.assembly["n_knife_bodies"]
    holders = {max(a, key=a.get) for a in attach.values() if a}
    cands = {c: f for c, f in frags.items() if c not in holders and f["mass"] >= 1e-3}
    assign = _assign_fillets(cands)
    pieces = e.rod_pieces(lab)
    x = ts.x.numpy()
    T = LY.fillet_tray(w.config.task)
    tray = (tuple(T["center"]), (T["size"][0]/2 - T["wall"], T["size"][1]/2 - T["wall"]), top + T["wall"],
            top + T["size"][2])
    blade_tets = _blade_compressed(e)
    nb = w.cut._nbr[blade_tets].ravel()
    knife_zone = blade_tets.copy()
    knife_zone[nb[nb >= 0]] = True              # tears within two faces of the knife's cut count as severed
    fnodes = {s: e.fragment_nodes(cands[assign[s]]) for s in ("upper", "lower") if assign[s] is not None}
    sunk = dict(upper=0., lower=0.)
    if len(fnodes) == 2:
        sunk["upper"], sunk["lower"] = _interpenetration(fnodes["upper"], fnodes["lower"], x)
    checks, metrics, prog = {}, {}, {}
    for s, k, ko in (("upper", REF["fillet_upper"], REF["fillet_lower"]),
                     ("lower", REF["fillet_lower"], REF["fillet_upper"])):
        total = max(sum(f["by_ref"][k] for f in frags.values()), 1e-12)
        removed = sum(f["by_ref"][k] for f in cands.values())
        attached = sum(frags[c]["by_ref"][k] for c in holders)
        R, R0 = _remaining_connection(e, s, knife_zone=knife_zone), _remaining_connection(e, s, intact=True)
        fid = assign[s]
        f = cands.get(fid)
        p = dict(fillet_fragment=fid, side_muscle_g=total*1e3, removed_g=removed*1e3, attached_g=attached*1e3,
                 remaining_connections=R, remaining_connections_at_start=R0)
        free = quality = place = 0.
        lost = False
        if f is None:
            note = "no piece of this side's muscle has been removed from the bones yet"
            checks.update([check(f"{s}_one_piece", 0., .97, "fraction", False, note),
                           check(f"{s}_yield", 0., Y, "kg", False, note),
                           check(f"{s}_no_rib_pieces_over_2mm", None, 0, "pieces", False, note),
                           check(f"{s}_no_backbone_or_fin_support_fragments", None, 0, "pieces", False, note),
                           check(f"{s}_workmanship_score", None, 85., "points", False, note),
                           check(f"{s}_in_tray", 0., 1., "fraction", False, note),
                           check(f"{s}_skin_down", None, 45., "deg", False, note)])
        else:
            one = f["by_ref"][k]/max(removed, 1e-12)
            ym = f["mass"] - f["by_ref"][ko] + e._skin_mass(f)
            ribs = [q for q in pieces if q["fragment"] == fid and q["kind"] == "rib" and q["length"] > 2e-3]
            fins = [q for q in pieces if q["fragment"] == fid and q["kind"] in ("fin_support", "supraneural")
                    and not q["name"].startswith("pelvic_girdle")]
            ws, wm = workmanship_A3(e, f, lab, k)
            nodes = fnodes[s]
            inside, rest, rm = _rests_in(e, nodes, tray, x)
            ok_tray = rest and inside >= .999 and sunk[s] <= SUNK_MAX
            down = e.skin_down(f)
            lost = bool((x[nodes, 2] < top - .02).any())
            why = [n for n, b in (("above the rim", rm["lowest_above_floor_mm"] >= 1e3*(tray[3] - tray[2])),
                                  ("touched by a hand", rm["held_by_hand"]), ("on the knife", rm["on_knife"]),
                                  ("moving", not rm["still"]),
                                  ("sunk into the other fillet", sunk[s] > SUNK_MAX)) if b]
            checks.update([
                check(f"{s}_one_piece", one, .97, "fraction of the side's removed muscle in its fillet", one >= .97),
                check(f"{s}_yield", ym, Y, "kg (fillet incl. skin, less the other side's muscle)", ym >= Y,
                      "0.33 x gutted = half of the published 66 % skilled lower bound"),
                check(f"{s}_no_rib_pieces_over_2mm", len(ribs), 0, "pieces", not ribs),
                check(f"{s}_no_backbone_or_fin_support_fragments", len(fins), 0, "pieces", not fins,
                      "a piece holding a vertebra is frame, not a fillet"),
                check(f"{s}_workmanship_score", ws, 85., "points (NOAA per square on state)", ws >= 85.),
                check(f"{s}_in_tray", inside if rest else 0., 1., "fraction of fillet nodes inside the tray's "
                      "walls, resting (below the rim, released, off the knife, still, not sunk into the other "
                      "fillet)", ok_tray, "" if ok_tray else "not resting in the tray: " + ", ".join(why)),
                check(f"{s}_skin_down", down if down is not None else 180., 45., "deg from down",
                      down is not None and down <= 45.)])
            free = f["by_ref"][k]/total
            y = min(1., ym/Y)
            q_ = min(1., one/.97)/(1. + len(ribs) + len(fins))*min(1., max(0., ws)/85.)
            quality = y*q_
            o = 0. if down is None else float(np.clip((90. - down)/45., 0., 1.))
            # placement credit: released and still below the rim (a knife lying on it fails knife_not_dropped and
            # the in-tray check, not the placement), less the share sunk into the other fillet beyond what the
            # in-tray test accepts
            laid = rm["lowest_above_floor_mm"] < 1e3*(tray[3] - tray[2]) and not rm["held_by_hand"] and rm["still"]
            place = y*(inside if laid else 0.)*(1. if sunk[s] <= SUNK_MAX else 1. - sunk[s])*(.5 + .5*o)
            p.update(fillet_g=f["mass"]*1e3, yield_g=ym*1e3, y=y, one_piece=one, bone_pieces=len(ribs) + len(fins),
                     workmanship=ws, workmanship_detail=wm, q=q_, tray=rm, sunk_share=sunk[s], skin_deg=down, o=o,
                     on_floor=lost)
        sever = min(1., max(0., free + attached/total*(1. - R/R0))) if R0 > 0 else free
        cut_ = W["sever"]*sever + W["quality"]*quality
        c_s = 0. if lost else W["cut"]*cut_ + W["place"]*place
        p.update(free=free, attached=attached/total, sever=sever, quality=quality, cut=cut_, place=place, credit=c_s)
        prog[s] = p
    # frame and head: every rigid bone inside and below the pan's rim, every piece holding one resting there
    st = ts.body_states()
    bones = st["p"][nk:]
    (pcx, pcy), (phx, phy), _, prim = pan = LY.frame_pan_inner()
    in_pan = (np.abs(bones[:, 0] - pcx) <= phx) & (np.abs(bones[:, 1] - pcy) <= phy) & (bones[:, 2] < prim)
    bone_still = np.linalg.norm(st["v"][nk:], axis=1) < .02
    holder_of = [max(attach[nk + j], key=attach[nk + j].get) if attach.get(nk + j) else None
                 for j in range(len(bones))]
    # a bare bone (no intact tie) goes with the piece holding a bone it is still jointed to (intervertebral joints):
    # a hand holding the frame holds its bare tail vertebrae
    ja, jb, j_on = ts.joints["a"].numpy(), ts.joints["b"].numpy(), ts.joints["on"].numpy()
    bone_joints = [(a_ - nk, b_ - nk) for a_, b_, o_ in zip(ja, jb, j_on) if o_ and a_ >= nk and b_ >= nk]
    for _ in range(len(bones)):
        todo = [(i, k) for i, k in bone_joints + [(k, i) for i, k in bone_joints]
                if holder_of[i] is None and holder_of[k] is not None]
        if not todo:
            break
        for i, k in todo:
            holder_of[i] = holder_of[k]
    meat = REF["fillet_upper"], REF["fillet_lower"]
    on = {c: f["by_ref"][meat[0]] + f["by_ref"][meat[1]] for c, f in frags.items()}
    m_side = max(.5*sum(on.values()), 1e-9)
    t_ = .3*m_side          # seed 0: ideal yields 115.7 + 107.6 g against 2 x 95.5 g leave about 32 g

    def meat_gate(m):
        return float(np.clip(1. - max(0., m - t_)/(m_side - t_), 0., 1.))

    def in_pan_share(c):
        q = x[e.fragment_nodes(frags[c])]
        return float(((np.abs(q[:, 0] - pcx) <= phx) & (np.abs(q[:, 1] - pcy) <= phy) & (q[:, 2] < prim)).mean())
    # the bones' credit is withheld while fillet meat lies in the pan, on any piece: a whole fish dropped in (its
    # head torn off by the impact) earns nothing
    m_pan = sum(m*in_pan_share(c) for c, m in on.items() if m > 1e-4)
    g_pan = meat_gate(m_pan)
    piece_rest, piece_gate, pans = {}, {}, {}
    for c in holders:
        _, r_, rm_ = _rests_in(e, e.fragment_nodes(frags[c]), pan, x)
        piece_rest[c], pans[str(c)] = r_, rm_
        # each bone-holding piece is credited by the fillet meat on it: a frame still carrying a fillet earns
        # nothing, a head cut off first earns its share, lifting one piece leaves another's
        piece_gate[c] = meat_gate(on[c])*g_pan
    # bones free of all tissue (a loose chain) are released if no robot hand box is within TOUCH_M of the capsules
    # of any bone jointed to them
    held_bare = np.zeros(len(bones), bool)
    sb, boxes = getattr(ts, "h_s_body", None), getattr(env, "_robot_boxes", [])
    if sb is not None and len(boxes):
        from filleting.workpiece import _R
        p0, q0 = ts.s_p0.numpy(), ts.s_q0.numpy()
        for j in (j for j, h in enumerate(holder_of) if h is None):
            for s in np.nonzero(sb == nk + j)[0]:
                hl, r = ts.shapes[s]["half"][:2]
                axis = p0[s] + np.outer(np.linspace(-1., 1., 5)*hl, _R(q0[s])[:, 0])
                held_bare[j] |= bool(near_boxes(w, boxes, axis, r + e.TOUCH_M).any())
        for _ in range(len(bones)):
            todo = [i for i, k in bone_joints + [(k, i) for i, k in bone_joints] if held_bare[k] and not held_bare[i]]
            if not todo:
                break
            held_bare[todo] = True
    rest_j = np.array([piece_rest.get(h, True) for h in holder_of]) & bone_still & ~held_bare
    gate_j = np.array([piece_gate.get(h, g_pan) for h in holder_of])
    w_j = np.full(len(bones), .5/max(len(bones) - 1, 1)); w_j[0] = .5   # head and frame half each (head body: 87 %
    b = float((w_j*in_pan*rest_j*gate_j).sum())                          # of the rigid bones' mass, seed 0)
    ok_frame = bool(in_pan.all() and rest_j.all())
    checks.update([check("frame_and_head_in_pan", float(in_pan.mean()), 1., "fraction of the rigid bones (head, "
                         "vertebrae) inside the empty pan's rim; the pieces holding them resting there", ok_frame)])
    prog["frame"] = dict(bones_in_pan=int(in_pan.sum()), bones=int(len(in_pan)), resting=bool(rest_j.all()),
                         bare_bones_held=int(held_bare.sum()), fillet_meat_in_pan_g=m_pan*1e3, g_pan=g_pan,
                         pieces=pans, gates={str(c): v for c, v in piece_gate.items()}, b=b, credit=b)
    # common: knife, holding hand, debris, thrown tissue, time. The knife is judged against the products (the two
    # fillets and the bone-holding pieces), not trimmings
    prods = [c for c in (assign["upper"], assign["lower"]) if c is not None] + sorted(holders)
    ok, where = e._knife_put_down(prods, frags, x)
    checks.update([check("knife_not_dropped", float(ok), 1., "held, or put down clear of the products and the "
                         "receptacles", ok, where)])
    n = env.blade_left_hand_contacts
    checks.update([check("no_blade_contact_with_holding_hand", n, 0, "ticks with contact", n == 0)])
    total_m = sum(f["mass"] for f in frags.values())
    deb = [f["mass"] for f in frags.values() if f["mass"] < e.DEBRIS_G*1e-3]
    budget, count = 2*e.DEBRIS_BUDGET*total_m, 2*e.DEBRIS_COUNT
    checks.update([check("debris", sum(deb)*1e3, budget*1e3, f"g in pieces under {e.DEBRIS_G:g} g (at most "
                         f"{count} pieces; A2's allowance for each of the two sides)",
                         sum(deb) <= budget and len(deb) <= count, f"{len(deb)} pieces")])
    ps, pat, pn = (tuple(getattr(env, "piece_speed_max_outside", (0., None, None))) + (None,))[:3]
    kd = ts.kind.numpy()[:ts.n]
    fallen = int((x[:ts.n, 2][(kd == 0) | (kd == 2)] < top - .02).sum())
    checks.update([check("no_tissue_thrown", ps, e.THROWN_MPS, f"m/s fastest free piece outside the tray and "
                         f"the pan ({pn} nodes, at {pat} s; a piece a hand touches is carried); {fallen} nodes below "
                         f"the bench top", ps <= e.THROWN_MPS and fallen == 0)])
    checks["time"] = check("time", env.sim_time, TIME_LIMIT_S["A3"], "s", env.sim_time <= TIME_LIMIT_S["A3"])[1]
    P = W["upper"]*prog["upper"]["credit"] + W["lower"]*prog["lower"]["credit"] + W["frame"]*prog["frame"]["credit"]
    metrics.update(progress=dict(value=float(P), weights=dict(W), **prog), fragments=len(frags),
                   debris_pieces=len(deb), debris_g=sum(deb)*1e3, tissue_nodes_below_bench=fallen,
                   piece_speed_max_all_mps=float(getattr(env, "piece_speed_max", (0.,))[0]))
    return checks, metrics


def _assign_fillets(cands):
    """{"upper": fragment or None, "lower": ...}: two DIFFERENT removed pieces chosen to hold the most of their own
    side's muscle together (a piece holding none of a side's muscle is never that side's fillet; a butterflied
    pair is one piece and serves one side)."""
    U, L = REF["fillet_upper"], REF["fillet_lower"]

    def top2(k):
        return sorted((c for c in cands if cands[c]["by_ref"][k] > 0), key=lambda c: -cands[c]["by_ref"][k])[:2]
    best, best_v = (None, None), -1.
    for u in top2(U) + [None]:
        for l_ in top2(L) + [None]:
            if u is not None and u == l_:
                continue
            v = (cands[u]["by_ref"][U] if u is not None else 0.) + (cands[l_]["by_ref"][L] if l_ is not None else 0.)
            if v > best_v:
                best, best_v = (u, l_), v
    return dict(upper=best[0], lower=best[1])


def _rests_in(e, nodes, rec, x):
    """A piece (its nodes) in a receptacle rec = (centre xy, inner half extents xy at the rim, floor top z, rim z):
    (fraction of its nodes inside the walls in plan, whether it rests there, details). Resting: its lowest node
    below the rim, no node within TOUCH_M of a robot hand box (released) or of the blade (off the knife), net motion
    under 1 mm over 1 s. A held piece is not put down; a released piece at rest below the rim inside the walls is
    supported by the receptacle or by what lies in it."""
    p = x[nodes]
    (cx, cy), (hx, hy), floor, rim = rec
    inside = float(((np.abs(p[:, 0] - cx) <= hx) & (np.abs(p[:, 1] - cy) <= hy)).mean())
    low = float(p[:, 2].min())
    held = bool(e._robot_touch(p).any())
    on_knife = bool(e._blade_touch(p, e.TOUCH_M).any())
    st = _settled_common(e, nodes)
    still = st is not None and st < 1e-3
    rest = low < rim and not held and not on_knife and still
    return inside, bool(rest), dict(inside=inside, lowest_above_floor_mm=(low - floor)*1e3, held_by_hand=held,
                                    on_knife=on_knife, still=bool(still),
                                    net_motion_1s_mm=None if st is None else st*1e3)


def _settled_common(e, nodes, window_s=1.):
    """settled() over the nodes that exist in both motion snapshots: cutting and tearing append split nodes after a
    snapshot is taken, so settled() alone would index past its end. A split node coincides with its twin when
    created."""
    h = e.env.motion_history
    if len(h) < 2:
        return None
    t_now, x_now = h[-1]
    old = [(t, x_) for t, x_ in h if t_now - t >= window_s - 1e-6]
    if not old:
        return None
    x0 = old[-1][1]
    nodes = np.asarray(nodes)
    nodes = nodes[nodes < min(len(x_now), len(x0))]
    return float(np.linalg.norm(x_now[nodes] - x0[nodes], axis=1).max()) if len(nodes) else None


def _interpenetration(na, nb, x, cell=3e-3, tol=2e-3):
    """(share of piece a's nodes, share of piece b's) lying in 3 mm plan cells that the other piece occupies over an
    overlapping height range deeper than `tol`: one fillet sunk into the other. The tissue solver has no
    tissue-to-tissue contact, so a fillet laid on the other passes through it to the tray floor (in one check
    65.6 % of its nodes inside the lower piece). Fillets lying side by side and touching share only an
    edge line of cells."""
    out = []
    cells = []
    for nd in (na, nb):
        pt = x[nd]
        key = np.floor(pt[:, 0]/cell).astype(np.int64)*1000003 + np.floor(pt[:, 1]/cell).astype(np.int64)
        u, inv = np.unique(key, return_inverse=True)
        lo = np.full(len(u), np.inf); hi = np.full(len(u), -np.inf)
        np.minimum.at(lo, inv, pt[:, 2]); np.maximum.at(hi, inv, pt[:, 2])
        cells.append((u, inv, lo, hi))
    (ua, ia, loa, hia), (ub, ib, lob, hib) = cells
    _, ca, cb = np.intersect1d(ua, ub, return_indices=True)
    deep = np.minimum(hia[ca], hib[cb]) - np.maximum(loa[ca], lob[cb]) > tol
    for u, inv, c_ in ((ua, ia, ca), (ub, ib, cb)):
        bad = np.zeros(len(u), bool); bad[c_[deep]] = True
        out.append(float(bad[inv].mean()))
    return tuple(out)


def prime(e):
    """The progress measure's starting connection counts (R0, intact topology), from the state the environment
    starts in, so they do not depend on when the first evaluation happens."""
    for s in ("upper", "lower"):
        _remaining_connection(e, s, intact=True)


def _blade_compressed(e):
    """Per original tet: where the knife's edge compresses the tissue it cuts (workmanship_A3): within one face of a
    tet that owns a knife-made cut face (neither an interface split at build nor torn) or holds a node of an
    interface bond the blade cut."""
    w = e.env.world
    ts = w.tissue
    nt = len(w.cut.orig)
    near = np.zeros(nt, bool)
    pre = w.assembly.get("interface_faces", set())
    torn = set(getattr(w.cut, "torn", {}))
    kf = [f for f in w.cut.cut_faces if f not in pre and f not in torn]
    if kf:
        near[np.asarray(kf, np.int64).ravel()] = True
    cutb = e.env.bond_cut_by_blade()
    if cutb.any():
        bn = np.unique(np.concatenate([ts.bonds["a"].numpy()[cutb], ts.bonds["b"].numpy()[cutb]]))
        act = np.nonzero(ts.hactive[:nt])[0]
        near[act[np.isin(ts.htets[act], bn).any(1)]] = True
    nb = w.cut._nbr[near].ravel()
    out = near.copy()
    out[nb[nb >= 0]] = True
    return out


def workmanship_A3(e, frag, lab, key):
    """NOAA workmanship on state as the standard grades it, per 25.4 mm square of the fillet's flesh side (rest
    coordinates projected on the fish's side plane): a square touched by a defect costs 1/2 point for a knife score
    (an incision left in the fillet: a knife-made cut face with both its tets in the fillet), 1/2 for a tear (a torn
    interface bond the blade did not cut) and 2 for a bruise (a tet compressed below J = 0.6 at any time), each at
    most once per square; normalised per 454 g of fillet. The knife's own cutting compression is not a bruise: the
    edge law crushes the tissue it cuts below J = 0.5 (tissue.py), so the bruise reads env.tet_jmin_far, each tet's
    worst J recorded only while the knife's edge was at least EDGE_REACH_M away. In the environment's E2 coupon cuts
    on the trout's 3.5 mm mesh (press and slice), no tet went below 0.6 in that record; a crush by a pad, the board
    or the blade's flat is recorded. Handling: every recorded robot grasp that lifted a fillet crushed it below 0.6
    (inverted tets under the fingers), and the position-mode closes that crushed nothing did not lift it; the
    threshold is kept, since those are real crushes in this world, and the grasp finding is reported as an
    environment issue. Bones have their own checks. (A2 uses workmanship(), which deducts per tet and per bond.)"""
    w = e.env.world
    ts = w.tissue
    asm = w.assembly
    tets = frag["tets"]
    rest_l = asm["to_local"](ts.hX[ts.htets[tets]].mean(1))
    cell = 25.4e-3
    gi = np.floor(rest_l[:, :2]/cell).astype(int)
    cells = {tuple(g) for g in gi}
    knife, tear, bruise = set(), set(), set()
    # a knife score: a knife-made cut face with both its tets in this fillet, an incision that did not separate
    # anything. Meat left on the bone is priced by yield alone
    own = set(tets.tolist())
    pre = asm.get("interface_faces", set())
    torn_f = set(getattr(w.cut, "torn", {}))
    inc = [t for f in w.cut.cut_faces if f[0] in own and f[1] in own and f not in pre and f not in torn_f for t in f]
    if inc:
        knife = {tuple(g) for g in gi[np.isin(tets, np.asarray(inc, np.int64))]}
    if getattr(ts, "n_bonds", 0):
        dmg = ts.bonds["damage"].numpy()
        a, b = ts.bonds["a"].numpy(), ts.bonds["b"].numpy()
        x = ts.x.numpy()
        nodes = np.unique(ts.htets[tets])
        torn = (dmg > .999) & (np.isin(a, nodes) | np.isin(b, nodes)) & \
            (np.linalg.norm(x[a] - x[b], axis=1) > 1e-3) & ~e.env.bond_cut_by_blade()
        if torn.any():
            pl = asm["to_local"](ts.hX[a[torn]])
            tear = {g for g in map(tuple, np.floor(pl[:, :2]/cell).astype(int)) if g in cells}
    jmin = getattr(e.env, "tet_jmin_far", None)
    jall = e.env.tet_jmin_history
    nb_counted = nb_edge = 0
    if jmin is not None:
        counted = jmin[tets] < .6
        nb_counted = int(counted.sum())
        nb_edge = int(((jall[tets] < .6) & ~counted).sum()) if jall is not None else 0
        bruise = {tuple(g) for g in gi[counted]}
    ded = .5*len(knife) + .5*len(tear) + 2.*len(bruise)
    score = 100. - ded*(.454/max(frag["mass"], 1e-6))
    return float(score), dict(cells=len(cells), knife_score_squares=len(knife), tear_squares=len(tear),
                              bruise_squares=len(bruise), deductions=float(ded), bruised_tets_counted=nb_counted,
                              compressed_tets_at_the_edge=nb_edge)


def _mincut_graph(e):
    """Static data for _remaining_connection, built once: the original tet face graph and each tet's side (its own
    label for muscle; for everything else - frame, head, belly edge - the side of the nearest muscle tet, so one
    side's number neither routes through nor changes with the other side's half of the fish)."""
    if getattr(e, "_mc", None) is not None:
        return e._mc
    from scipy.spatial import cKDTree
    w = e.env.world
    ts = w.tissue
    nt = len(w.cut.orig)
    ref = np.asarray(w.assembly["ref_tet"][:nt])
    U, L = REF["fillet_upper"], REF["fillet_lower"]
    side = np.where(ref == L, 1, 0)
    mus, oth = np.nonzero((ref == U) | (ref == L))[0], np.nonzero((ref != U) & (ref != L))[0]
    cen = ts.hX[w.cut.orig].mean(1)
    side[oth] = side[mus][cKDTree(cen[mus]).query(cen[oth])[1]]
    t = np.repeat(np.arange(nt), 4)
    u = w.cut._nbr[:nt].ravel()
    m = u > t
    t, u = t[m], u[m]
    keys = t.astype(np.int64)*nt + u
    pre = w.assembly.get("interface_faces", set())
    prek = np.fromiter((a*nt + b for a, b in pre), np.int64, len(pre))
    # the source of each side's minimum cut is its meat core: muscle more than CORE_M from frame or head tissue and
    # holding no flesh-bone tie, so a cut that leaves the near-bone layer on the frame (which the task allows; the
    # yield term prices it) severs the side as fully as one along the label boundary (459 of
    # the 789 starting connections are ties straight from fillet-labelled meat)
    fh = np.nonzero((ref == REF["frame"]) | (ref == REF["head"]))[0]
    dist = cKDTree(cen[fh]).query(cen[mus])[0] if len(fh) else np.full(len(mus), np.inf)
    tied = np.isin(w.cut.orig, np.unique(ts.bties["node"].numpy())).any(1) if ts.n_bties else np.zeros(nt, bool)
    core = np.zeros(nt, bool)
    core[mus[dist > CORE_M]] = True
    core &= ~tied
    e._mc = dict(nt=nt, ref=ref, side=side, t=t, u=u, keys=keys, iface=np.isin(keys, prek), R0={}, core=core)
    return e._mc


def _remaining_connection(ev, side, intact=False, knife_zone=None):
    """The fewest intact connections - uncut tet faces, interface bonds not fully damaged, flesh-bone ties not
    fully damaged, one each - still joining the side's muscle (rest label fillet_<side>) to any rigid bone (head or
    vertebra): a maximum flow / minimum cut over the tets of that side (_mincut_graph). It never rises as tissue is
    cut, is unchanged by cuts that free nothing, and is ~0 once the side's meat core is off the bone. Its source is
    the side's meat core (_mincut_graph, CORE_M). A torn face, a torn (not blade-cut) bond or a fully damaged tie
    counts as severed only inside `knife_zone` (per tet; None: nowhere): tearing that runs from the knife's cut is
    part of cutting, tearing elsewhere is not. A creator check dropped the uncut trout 20 mm into the frame pan:
    ~465 faces torn, 250 ties fully damaged, no knife; counting tears that had opened by 1 mm
    credited that drop 0.05 of progress, so peeling is not credited until the fillet comes free.
    intact=True: the same before any cut (only the interfaces split at build are cut faces, every bond and tie
    intact), cached."""
    g = _mincut_graph(ev)
    if intact and side in g["R0"]:
        return g["R0"][side]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import maximum_flow
    w = ev.env.world
    ts = w.tissue
    nt = g["nt"]
    s_ = 0 if side == "upper" else 1
    act = ts.hactive[:nt] > 0
    alive = (g["side"] == s_) & (act | intact)
    t, u = g["t"], g["u"]
    keep = alive[t] & alive[u]
    if intact:
        keep &= ~g["iface"]
    zone = np.zeros(nt, bool) if knife_zone is None else np.asarray(knife_zone, bool)[:nt]
    if not intact and w.cut.cut_faces:
        torn = set(getattr(w.cut, "torn", {}))
        sev = [f for f in w.cut.cut_faces if f not in torn or zone[f[0]] or zone[f[1]]]
        ck = np.fromiter((a*nt + b for a, b in sev), np.int64, len(sev))
        keep &= ~np.isin(g["keys"], ck)
    S, K = nt, nt + 1
    rows, cols = [t[keep], u[keep]], [u[keep], t[keep]]
    ai = np.nonzero(act | intact)[0]
    node_tet = np.full(ts.n, -1, np.int64)
    node_tet[ts.htets[ai].ravel()] = np.repeat(ai, 4)
    if getattr(ts, "n_bonds", 0):
        ta, tb = node_tet[ts.bonds["a"].numpy()], node_tet[ts.bonds["b"].numpy()]
        e = (ta >= 0) & (tb >= 0) & (ta != tb)
        if not intact:
            broken = ts.bonds["damage"].numpy() >= 1.
            cut_b = ev.env.bond_cut_by_blade()
            e &= ~broken | (~cut_b & ~zone[np.maximum(ta, 0)] & ~zone[np.maximum(tb, 0)])
        e[e] &= alive[ta[e]] & alive[tb[e]]
        rows += [ta[e], tb[e]]
        cols += [tb[e], ta[e]]
    if ts.n_bties:
        tt = node_tet[ts.bties["node"].numpy()]
        ok = (ts.bties["body"].numpy() >= w.assembly["n_knife_bodies"]) & (tt >= 0)
        if not intact:
            ok &= (ts.bties["damage"].numpy() < 1.) | ~zone[np.maximum(tt, 0)]
        tt = tt[ok]
        tt = tt[alive[tt]]
        rows.append(tt)
        cols.append(np.full(len(tt), K))
    src = np.nonzero(alive & g["core"] & (g["ref"] == (REF["fillet_upper"], REF["fillet_lower"])[s_]))[0]
    if not len(src):
        return 0
    rows.append(np.full(len(src), S))
    cols.append(src)
    r, c = np.concatenate(rows).astype(np.int64), np.concatenate(cols).astype(np.int64)
    cap = np.ones(len(r), np.int64)
    cap[-len(src):] = 10**6
    G = coo_matrix((cap, (r, c)), shape=(nt + 2, nt + 2)).tocsr()
    G.sum_duplicates()
    G.data = np.minimum(G.data, 2**31 - 1).astype(np.int32)
    G.setdiag(0)
    G.eliminate_zeros()
    R = int(maximum_flow(G, S, K).flow_value)
    if intact:
        g["R0"][side] = R
    return R
