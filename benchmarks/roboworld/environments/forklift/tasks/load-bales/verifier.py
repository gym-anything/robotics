"""Outcome of load-bales: exactly two bales in the target trailer, each upright, resting on dunnage strips and against
the front wall, not overlapping, the whole truck back behind the door face and the trailer not moved.

verify(world, start) gets the world at the evaluation (bodies, event latches, signals) and the resolved start, and
returns (criteria, details): the shared site rules (evaluator.shared) and the task's own criteria, each evaluator.C,
which evaluator.evaluate scores, and the details with each bale's state and the task's reward (load-bales-reward-v1,
BALE_REWARD in evaluator.py), which evaluator.evaluate scores instead of the fraction met."""
import math

import numpy as np

from forklift import params as P
from forklift.evaluator import (BALE_REWARD, C, IMPACT_SPEED, LIMIT_KMH, SET_DOWN_SPEED, STRIKE_N, _floor_events,
                               _strike_events, _trailer_frame, blades, body_frame, episode_load, footprint,
                               lean_limit, onset_normal, shared, yaw_of)


def box_corners(world, name):
    """The 8 corners (world) of a box load from its pose and its built size."""
    info = world.scene.loads[name]
    o, R = body_frame(world, info["body"])
    lx, ly, lz = info["size"]
    return np.array([o + R @ np.array([sx * lx / 2, sy * ly / 2, sz * lz / 2])
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]), o, R


# load-bales: the placement is judged from the bales' and the dunnage strips' current poses, on oriented plan
# footprints.
BALE_NOSE_GAP = (-0.005, 0.075)    # m: front face to the nose wall; 5 mm into its plane allowed (a resting contact sits a


#                                    fraction of a millimetre inside it)
BALE_UPRIGHT_DEG = 5.0             # its 48 in axis within 5 deg of the trailer's vertical


BALE_REST_TOL = 0.03               # m: its bottom to the top of the strips it rests on


BALE_OVERLAP_TOL = 0.005           # m: penetration allowed between the two bales' footprints (contact)


def _hull(pts):
    """Convex hull of plan points, counter-clockwise (monotone chain); fewer than 3 distinct points are returned as they are."""
    pts = sorted(set(map(tuple, np.round(np.asarray(pts, float).reshape(-1, 2), 9))))
    if len(pts) < 3:
        return np.array(pts, float).reshape(-1, 2)

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]
    return np.array(half(pts) + half(pts[::-1]), float)


def _clip(poly, by):
    """The part of convex polygon poly inside convex counter-clockwise polygon by (Sutherland-Hodgman)."""
    out = list(np.asarray(poly, float))
    for i in range(len(by)):
        a, b = by[i], by[(i + 1) % len(by)]
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        inp, out = out, []
        for j, p in enumerate(inp):
            q = inp[(j + 1) % len(inp)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if sp * sq < 0:
                out.append(p + (q - p) * sp / (sp - sq))
        if not out:
            break
    return np.array(out, float).reshape(-1, 2)


def _in_convex(p, poly):
    return len(poly) >= 3 and all((poly[(i + 1) % len(poly)][0] - poly[i][0]) * (p[1] - poly[i][1]) -
                                  (poly[(i + 1) % len(poly)][1] - poly[i][1]) * (p[0] - poly[i][0]) >= -1e-9
                                  for i in range(len(poly)))


def _plan_gap(A, B):
    """Signed plan distance between convex polygons A and B: their clearance (> 0) or minus their penetration depth."""
    sep = []
    for Q in (A, B):
        for i in range(len(Q)):
            e = Q[(i + 1) % len(Q)] - Q[i]
            n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            sep.append(max((B @ n).min() - (A @ n).max(), (A @ n).min() - (B @ n).max()))
    if max(sep) <= 0:
        return float(max(sep))

    def seg(p, a, b):
        t = max(0.0, min(1.0, float((p - a) @ (b - a) / ((b - a) @ (b - a)))))
        return float(np.linalg.norm(p - a - t * (b - a)))
    return min(seg(p, Q[i], Q[(i + 1) % len(Q)]) for S, Q in ((A, B), (B, A)) for p in S for i in range(len(Q)))


def _dunnage_tops(world, door):
    """Every dunnage strip's top face in the target trailer's frame (the nose strips and the row's: a strip carried in
    is a strip): name, height and plan rectangle."""
    tr, to, TR = _trailer_frame(world, door)
    out = []
    for n in sorted(world.scene.loads):
        if "dunnage" in n:
            loc = (box_corners(world, n)[0] - to) @ TR
            top = loc[np.argsort(loc[:, 2])[4:]]
            out.append(dict(name=n, top=float(top[:, 2].mean()), rect=_hull(top[:, :2])))
    return out


BOX_SIGNS = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])     # box_corners' order


def _floor_z(world, start, x, y):
    """The dock floor's height under a world point: 0, or on a deployed leveler's deck, rising from its rear hinge to the
    docked trailer's bed at the door face."""
    Ld, Wd = P.LEVELER["length"], P.LEVELER["width"]
    for dn, t in (start.get("trailers") or {}).items():
        d = world.scene.zones["doors"][dn]
        if abs(x - d["x"]) <= Wd / 2 and d["face_y"] - Ld <= y <= d["face_y"]:
            return float(t["bed_z"]) * (y - (d["face_y"] - Ld)) / Ld
    return 0.0


def _bale_state(world, start, name, strips, floored):
    """One bale against the work order, in the target trailer's frame (x from the rear sill, y left, z up). floored: the
    bales with a latched floor contact."""
    door = start["target"]["door"]
    tr, to, TR = _trailer_frame(world, door)
    corners, o, R = box_corners(world, name)
    loc = (corners - to) @ TR
    c = (o - to) @ TR
    tilt = math.degrees(math.acos(min(1.0, abs(float(TR[:, 2] @ R[:, 2])))))      # either way up
    low4 = loc[np.argsort(loc[:, 2])[:4]]
    base = _hull(low4[:, :2])                                                      # the face it stands on
    # the strips it bears on: under its bottom face, the face's plane within BALE_REST_TOL of the strip's top at the
    # closest point of their overlap (read at its lowest corner, a bale tilted past ~6 deg on the strips' edges hangs
    # more than 30 mm below their tops and reads as not resting on them)
    nb = TR.T @ R[:, 2]
    nb = nb if nb[2] > 0 else -nb
    plane = lambda q: low4[0, 2] - (nb[0] * (q[:, 0] - low4[0, 0]) + nb[1] * (q[:, 1] - low4[0, 1])) / nb[2]
    under = [(s["name"], s["top"], _clip(base, s["rect"])) for s in strips]
    under = [(n, a) for n, top, a in under if len(a) >= 3 and abs(float(plane(a).min()) - top) < BALE_REST_TOL]
    # resting on strips: its centre of mass over the support they give it (the hull of its contact patches on them)
    on_strips = bool(under) and _in_convex(c[:2], _hull(np.vstack([a for _, a in under])))
    nose_gap = float(tr["L"] - loc[:, 0].max())
    # against the front wall: the vertical face turned toward the nose (either end or side of the bale), at both of its
    # vertical edges, each read at its nearer corner. The nearest single corner would let a bale turned 25 deg, touching
    # the wall at one corner, pass.
    ax = TR.T @ R[:, :2]
    k, s = max(((k, s) for k in (0, 1) for s in (-1, 1)), key=lambda ks: ks[1] * ax[0, ks[0]])
    edges = [float(tr["L"] - loc[(BOX_SIGNS[:, k] == s) & (BOX_SIGNS[:, 1 - k] == e), 0].max()) for e in (-1, 1)]
    inside = bool(np.all((loc[:, 0] > 0) & (loc[:, 0] < tr["L"] + 0.005) & (np.abs(loc[:, 1]) < tr["W"] / 2)))
    at_wall = all(BALE_NOSE_GAP[0] <= g <= BALE_NOSE_GAP[1] for g in edges)
    placed = inside and on_strips and tilt <= BALE_UPRIGHT_DEG and at_wall
    # resting on a floor now: a latched floor contact and, not borne by strips, its lowest corner within the rest tolerance of the floor under it (a trailer's, in that trailer's frame, or the dock's)
    low = corners[int(np.argmin(corners[:, 2]))]
    on_floor = False
    if name in floored and not on_strips:
        for tn in world.scene.trailers:
            t_, o_, R_ = _trailer_frame(world, tn)
            cc = (o - o_) @ R_
            if 0.0 < cc[0] < t_["L"] and abs(cc[1]) < t_["W"] / 2:
                on_floor = float(((corners - o_) @ R_)[:, 2].min()) < BALE_REST_TOL
                break
        else:
            on_floor = float(low[2]) - _floor_z(world, start, float(low[0]), float(low[1])) < BALE_REST_TOL
    return dict(name=name, tilt=tilt, nose_gap=nose_gap, front_edge_gaps=edges, inside=inside, on_strips=on_strips,
                strips=[n for n, _ in under], placed=bool(placed), on_floor=bool(on_floor),
                in_trailer=bool(0.0 < c[0] < tr["L"] and abs(c[1]) < tr["W"] / 2),
                centre_trailer=[float(c[0]), float(c[1])], xy=[float(o[0]), float(o[1])], plan=_hull(loc[:, :2]))


def _nose_wall_rule(door):
    """load-bales' rule for a bale's contact with the target trailer's nose wall, the wall the work order sets the bales
    against. The shared rule reads the bale's total speed at first contact, so a bale lowered onto its strips while
    touching the wall would read as an impact, and it counts a resting contact's accumulated impulse, which any contact
    that stays passes. Here: not a strike if the bale approached the wall slower than
    IMPACT_SPEED (onset_normal, its speed toward the wall) and never pressed on it harder than its own weight; otherwise a
    strike only above STRIKE_N. The same for a dunnage strip there; a strip's weight bound is
    STRIKE_N. Side walls, other structures and other tasks keep the shared rule."""
    def rule(world, mover, cls, e):
        parts = e.get("parts") or []
        if (cls != "trailer_wall" or not mover.startswith(("load:bale_", "load:nose_dunnage_", "load:dunnage_"))
                or len(parts) < 2 or str(parts[1]) != f"{door}:nose"):
            return None
        if onset_normal(e) < IMPACT_SPEED and e["peak_N"] <= lean_limit(world, episode_load(mover, e)):
            return False
        return e["peak_N"] > STRIKE_N
    return rule


def _truck_past_face(world, door):
    """How far the truck's farthest point (footprint corners, counterweight to fork tips, and both blade tips) stands
    past the door's face toward the trailer (m; < 0: the whole truck behind it, on the dock or its leveler deck)."""
    p, R = world.body_pose("chassis")
    ys = [y for _, y in footprint(float(p[0]), float(p[1]), yaw_of(R))] + [float(b["tip"][1]) for b in blades(world)[0]]
    return float(max(ys) - world.scene.zones["doors"][door]["face_y"])


def verify(world, start):
    door = start["target"]["door"]
    crit, det = shared(world, start, dest_floor=lambda n, xy: not n.startswith("bale"), strike_rule=_nose_wall_rule(door))
    ns = next(c for c in crit if c["name"] == "no_strike")
    ns["threshold"] += (f"; load-bales: a bale's or a dunnage strip's contact with the trailer's nose wall counts only if it "
                        f"approached the wall at >= {IMPACT_SPEED:.2f} m/s or pressed on it harder than its own weight or "
                        f"{STRIKE_N:.0f} N, whichever is more, and then only above {STRIKE_N:.0f} N (not by the impulse a "
                        f"resting contact accumulates)")
    ns["source"] += "; load-bales nose-wall rule: authored"
    strips = _dunnage_tops(world, door)
    floored = {n for n, _, _ in det["dropped"]}
    bales = [_bale_state(world, start, n, strips, floored) for n in sorted(n for n in world.scene.loads if n.startswith("bale_"))]
    at_nose = [b for b in bales if b["placed"]]
    loaded = [b["name"] for b in bales if b["in_trailer"]]
    gaps = [_plan_gap(a["plan"], b["plan"]) for i, a in enumerate(at_nose) for b in at_nose[i + 1:]]
    clear = max(gaps) if gaps else None
    ok = clear is not None and clear >= -BALE_OVERLAP_TOL and len(loaded) == 2
    crit.append(C("two_bales_at_nose", dict(at_nose=[b["name"] for b in at_nose], bales_in_trailer=loaded, clearance_m=clear),
                  ok, "-", "exactly two bales in the trailer, each upright (<= 5 deg) and resting on dunnage strips (its "
                  "bottom within 30 mm of their tops, its centre over the strips it bears on), its front face <= 75 mm "
                  "from the front wall at both front edges (5 mm into it allowed); the two not overlapping (<= 5 mm, "
                  "oriented footprints); any two bales, either side, at any spacing",
                  "authored"))
    past = _truck_past_face(world, door)
    crit.append(C("truck_back_on_dock", past, past < 0.0, "m", "the whole truck, counterweight to fork tips, behind the "
                  "door face (measured: its farthest point past the face)", "the work order ('back out onto the dock')"))
    # the trailer held where it was spotted while the laden truck drove in and out twice, as in load-trailer
    mv = world.events.trailer_max_move.get(door, 0.0)
    crit.append(C("trailer_movement", mv, mv <= 0.025, "m", "<= 25 mm from its spotted position", "authored (as load-trailer)"))
    reward = _load_bales_reward(world, start, crit, det, bales)
    for b in bales:
        b.pop("plan")
        b["progress"] = reward["bales"][b["name"]]
    return crit, dict(det, bales=bales, reward=reward)


BALE_WORKMANSHIP = {
    "no_strike": "per contact event 1 / (1 + 0.5 log10 s) of its worst episode, s = (approach speed / 0.10 m/s)^2 if it "
                 "approached at >= 0.10 m/s, else max(1, peak / 2 kN); episodes of one mover on one structure class "
                 "< 0.25 s apart are one event; a bale's trailer-floor arrival is graded under no_dropped_load instead",
    "no_dropped_load": "per floor-contact event of a bale (latched entries < 0.3 s apart are one): 0.8 + 0.1 / (1 + duration "
                       "/ 1 s) if the bale came off the floor again, 0.8 if it rests on a floor or lies over, each times "
                       "g(v) = 1 for an approach speed v <= 0.10 m/s, else max(0.3, 1 / (1 + log10 (v / 0.10 m/s)))",
    "no_wheel_lift": "1 / (1 + (t - 0.3 s) / 1 s)",
    "roll_under_6deg": "1 / (1 + (roll - 6 deg) / 6 deg); 0 on overturn",
    "speed_limit": "1 / (1 + t_over / 10 s + 2 (v_max / limit - 1))",
    "operator_seated": "1 / (1 + (t_open - 1 s) / 1 s)",
    "head_clear_of_guard": "limit / peak force",
    "trailer_movement": "1 / (1 + (move - 25 mm) / 25 mm)",
}


def _to_nose_row(world, door, xy):
    """Plan distance (m) from a world point to the target trailer's nose row (a bale centre's place against its front
    wall, anywhere across): straight inside that trailer; from outside it, to its rear opening and on; from inside the
    other trailer, out through that one's rear sill first."""
    tr, to, TR = _trailer_frame(world, door)
    lx, ly = P.BALE["size"][:2]
    xg, yg = tr["L"] - lx / 2, (tr["W"] - ly) / 2

    def local(name, p):
        _, o_, R_ = _trailer_frame(world, name)
        return (R_.T @ (np.array([p[0], p[1], o_[2]]) - o_))[:2]

    def inside(name, r):
        t_ = world.scene.trailers[name]
        return 0.0 <= r[0] <= t_["L"] and abs(r[1]) <= t_["W"] / 2
    p, d = np.asarray(xy[:2], float), 0.0
    for name in world.scene.trailers:
        if name != door and inside(name, local(name, p)):
            d, p = d + float(np.linalg.norm(local(name, p))), _trailer_frame(world, name)[1][:2]
    r = local(door, p)
    if not inside(door, r):
        q = np.array([0.0, float(np.clip(r[1], -tr["W"] / 2, tr["W"] / 2))])
        d, r = d + float(np.linalg.norm(r - q)), q
    return d + float(np.hypot(abs(xg - r[0]), max(0.0, abs(r[1]) - yg)))


def _load_bales_reward(world, start, crit, det, bales):
    RW = BALE_REWARD
    floor_events, folded = _floor_events(world, det, RW["floor_merge_s"])
    starts = {ld["name"]: ld["centre"] for ld in start["loads"]}
    door = start["target"]["door"]
    p = {}
    for b in bales:
        lean = 1.0 - max(0.0, b["tilt"] - BALE_UPRIGHT_DEG) / (RW["carry_tilt_deg"] - BALE_UPRIGHT_DEG)
        if b["on_floor"] or lean <= 0.0:
            p[b["name"]] = 0.0
        elif b["placed"]:
            p[b["name"]] = 1.0
        elif b["inside"] and b["on_strips"]:
            short = max(0.0, max(b["front_edge_gaps"]) - BALE_NOSE_GAP[1]) / RW["short_decay_m"]
            p[b["name"]] = (1.0 - (1.0 - RW["carry_cap"]) * min(1.0, short)) * min(1.0, lean)
        else:
            d0 = _to_nose_row(world, door, starts[b["name"]]) - RW["carry_deadband_m"]
            p[b["name"]] = RW["carry_cap"] * max(0.0, (d0 - _to_nose_row(world, door, b["xy"])) / d0)
    placement = float(np.mean(sorted(p.values())[-2:]))
    by = {c["name"]: c for c in crit}
    gate = min(1.0, max(0.0, (placement - RW["exit_gate_from"]) / (1.0 - RW["exit_gate_from"])))
    out_ = float(by["truck_back_on_dock"]["passed"])
    progress = (1.0 - RW["exit_share"]) * placement + RW["exit_share"] * out_ * gate
    events = _strike_events([s for i, s in enumerate(det["strikes"]) if i not in folded], RW["strike_merge_s"])
    lying = {b["name"] for b in bales if b["on_floor"] or b["tilt"] > RW["carry_tilt_deg"]}
    speed = lambda v: 1.0 if v <= SET_DOWN_SPEED else max(RW["speed_floor"], 1.0 / (1.0 + math.log10(v / SET_DOWN_SPEED)))
    brush = lambda e: RW["strand_factor"] + (RW["brush_factor"] - RW["strand_factor"]) / (
        1.0 + (e["t1"] - e["t0"]) / RW["brush_decay_s"])

    def floor_factor(name, evs):
        f = float(np.prod([speed(e["v"]) for e in evs]))
        return f * (RW["strand_factor"] if name in lying else float(np.prod([brush(e) for e in evs])))
    ev = world.events
    m = lambda name: by[name]["measured"]
    grade = {
        "no_strike": lambda: float(np.prod([1.0 / (1.0 + 0.5 * math.log10(e["severity"])) for e in events])),
        "no_dropped_load": lambda: float(np.prod([floor_factor(n, evs) for n, evs in floor_events.items()])),
        "no_wheel_lift": lambda: 1.0 / (1.0 + max(0.0, m("no_wheel_lift") - 0.3)),
        "roll_under_6deg": lambda: 0.0 if ev.overturn is not None else 1.0 / (1.0 + max(0.0, ev.max_roll - 6.0) / 6.0),
        "speed_limit": lambda: 1.0 / (1.0 + ev.over_limit_s / 10.0 + 2.0 * max(0.0, m("speed_limit") / LIMIT_KMH - 1.0)),
        "operator_seated": lambda: 1.0 / (1.0 + max(0.0, m("operator_seated") - 1.0)),
        "head_clear_of_guard": lambda: min(1.0, P.HEAD_CONTACT_LIMIT / max(m("head_clear_of_guard"), 1e-9)),
        "trailer_movement": lambda: 1.0 / (1.0 + max(0.0, m("trailer_movement") - 0.025) / 0.025),
    }
    factors = {k: float(f()) for k, f in grade.items() if not by[k]["passed"]}
    return dict(contract=RW["contract"], rule="1 if success, 0 if the world faulted, else 0.9 x progress x workmanship",
                progress=progress, placement=placement, truck_out=out_, exit_gate=gate,
                workmanship=float(np.prod(list(factors.values()) or [1.0])), factors=factors, bales=p,
                strike_events=[dict(e, t_end=None if math.isinf(e["t_end"]) else e["t_end"]) for e in events],
                floor_events={n: [dict(e) for e in evs] for n, evs in floor_events.items()},
                fail_scale=RW["fail_scale"], workmanship_rules=BALE_WORKMANSHIP)
