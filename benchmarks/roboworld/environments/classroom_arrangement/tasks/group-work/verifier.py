"""Outcome of group-work: three pods of four from rows 1 and 2, a chair ready at each desk outside its pod, a 0.915 m
route from the door to the back, and the rest of the room as found.

The checks name pod places, not furniture: slot_<r><c> is the place desk_<r><c>'s target puts a desk (its dots, its
facing) and seat_<r><c> a chair at the desk in that place. by_place matches the room's identical desks and chairs to
the places. criteria(s) gives the checks (layout.setup puts them in the spec as "check"); verify(e, motion, window)
evaluates an episode (task.Evaluator.evaluate_criteria, which calls by_place for the released selection)."""
import math

import numpy as np

from classroom_arrangement import furniture as F, layout as L, params as P
from classroom_arrangement.task import E, _ang, edge_span, to_local

# The distance scale: each pod desk's move from its row onto its pod dots, 0.471 m (no credit beyond it).
REACH = (L.ROWS_Y[1] - L.ROWS_Y[0])/2 - P.DESK_TOP[1]/2


def criteria(s):
    check = {}
    for r in (1, 2):
        for c in range(1, 7):
            check[f"slot_{r}{c}"] = ("slot", f"desk_{r}{c}")
            check[f"seat_{r}{c}"] = ("seat", f"slot_{r}{c}")
    for pair in range(3):
        check[f"pod_{pair}"] = [f"slot_{r}{c}" for r in (1, 2) for c in (2*pair + 1, 2*pair + 2)]
    check["rest_of_room"] = "as_found"
    check["route"] = [(2.65, 0.6), (0., 8.5)]
    return check


def verify(e, motion, window):
    return e.evaluate_criteria(motion, window, by_place=by_place)


def by_place(e, motion, window):
    """group-work. Each place that needs a desk (12 pod slots, plus the 12 places of rows 3-4) and each place that
    needs a chair (12 pod seats, plus where every other chair stood) is matched one-to-one to whichever identical
    desk or chair stands there: the assignment with the most credit (scipy linear_sum_assignment), desks first, then
    chairs at the desks matched to the slots. Success needs every criterion. The score is 1 on success, else
    fail_scale x max(0, progress - deduction) x penalties (constants in params.EVAL_GROUP_WORK), where
      progress  = 0.60 x mean desk-place credit + 0.15 x mean front-edge-meeting credit (6) + 0.25 x mean chair
                  credit. A desk's position credit rises linearly as its worst glide approaches the place's dot,
                  from 0 at REACH (0.471 m: an untouched row desk, or a desk on the dots facing the wrong way) to 1
                  within 30 mm. At the six board-side places, whose facing differs from every desk's start facing,
                  the place credit is 0.3 x a turn credit (the desk faces the place's way: 0 at 90 deg or more off,
                  1 within 3 deg, wherever the desk stands, upright) + 0.7 x (0.7 x the position credit + 0.3 x
                  the turn credit x an approach credit falling linearly from 1 at 30 mm to 0 at 2.5 m between the
                  desk's centre and the place's); elsewhere it is the position credit. A meeting's credit is the
                  smaller of its two desks' position credits times a
                  fall-off over REACH of how far the edges are outside the pod's bounds (closest gap -2..+10 mm,
                  widest gap, sideways offset). A chair's is its desk's position credit times a fall-off over half
                  the chair's depth outside the seat window (and over 75 deg beyond the 15 deg facing), 0 unless
                  that desk is the nearest one the chair stands behind and faces (not a chair still pulled out at
                  another desk in between). Near credit needs no rest window (it reads the poses); criteria do.
      deduction = 0.05 (one desk place's share of progress) per whole place of the rest of the room not as
                  found, each place charged from 0 at its tolerance to 1 at 0.30 m / 45 deg (tote, wastebasket:
                  0.20 m to 0.50 m; knocked over or empty: 1), the same at any stage of the work;
      penalties = route: x1 at 0.915 m falling linearly to x0.5 at 0.815 m, the widest route width that still
                  connects the door to the back (only measured when 0.915 m fails); x0.5 for anything toppled or in
                  the door swing:
    the untouched room scores 0, and a constraint already met at the start earns nothing."""
    from scipy.optimize import linear_sum_assignment
    spec, G = e.spec, P.EVAL_GROUP_WORK
    chk, items, targets = spec["check"], spec["items"], spec["targets"]
    slots = [n for n, c in chk.items() if isinstance(c, tuple) and c[0] == "slot"]
    seats = [n for n, c in chk.items() if isinstance(c, tuple) and c[0] == "seat"]
    kind = {n: it["kind"] for n, it in items.items()}
    desks = sorted(n for n in items if kind[n] == "desk")
    chairs = sorted(n for n in items if kind[n] == "chair")
    homes = [n for n in items if n not in targets]             # every place the pods do not use, as found
    desk_homes = [n for n in homes if kind[n] == "desk"]
    chair_homes = [n for n in homes if kind[n] == "chair"]
    loose = [n for n in homes if kind[n] not in ("desk", "chair")]
    pose, tilt, up, rest = {}, {}, {}, {}
    for n in desks + chairs + loose:
        pose[n], tilt[n] = e.pose(n)
        up[n] = e.upright(n)[0]
        rest[n] = motion.get(n, math.inf) <= E["rest_motion"] and window
    standing = {n: tilt[n] <= G["toppled_deg"] for n in pose}
    tol = E["glide_tol"]

    def as_found(n, home):
        """1 when the item stands within the place's tolerance, upright; falling linearly to 0 at as_found_full."""
        (x, y, yaw), (x0, y0, yaw0) = pose[n], items[home]["pose"]
        d, dyaw = math.hypot(x - x0, y - y0), math.degrees(_ang(yaw - yaw0))
        (d0, a0) = G["as_found"] if kind[home] == "chair" else G["as_found_furniture"]
        (d1, a1) = G["as_found_full"]
        return up[n]*max(0., 1. - max((d - d0)/(d1 - d0), (dyaw - a0)/(a1 - a0), 0.))

    # desks -> 12 slots + 12 desk homes
    g = np.stack([e.glides_xyz(n)[:, :2] for n in desks])                         # (24, 4, 2)
    dots = np.array([spec["dots"][chk[s][1]] for s in slots])                       # (12, 4, 2)
    err = np.linalg.norm(g[:, None] - dots[None], axis=3).max(axis=2)               # worst glide, (24, 12)
    dyaw = np.array([[math.degrees(_ang(pose[n][2] - targets[chk[s][1]][2])) for s in slots] for n in desks])
    up_ = np.array([standing[n] for n in desks], float)[:, None]
    near = np.clip((REACH - err)/(REACH - tol), 0., 1.)*up_                         # position credit
    # board-side places: their facing differs from the start facing of the desk whose target they are (all desks
    # start facing the board), so a desk facing their way earns turn credit wherever it stands
    turned = np.array([_ang(targets[chk[s][1]][2] - items[chk[s][1]]["pose"][2]) > math.radians(E["yaw_tol_deg"])
                       for s in slots])
    turn = np.clip((G["turn_zero_deg"] - dyaw)/(G["turn_zero_deg"] - E["yaw_tol_deg"]), 0., 1.)*up_
    # a desk so turned also earns approach credit as its centre comes within approach_radius of the place
    centre = np.array([pose[n][:2] for n in desks])
    place = np.array([targets[chk[s][1]][:2] for s in slots])
    dist = np.linalg.norm(centre[:, None] - place[None], axis=2)
    approach = turn*np.clip((G["approach_radius"] - dist)/(G["approach_radius"] - tol), 0., 1.)
    a, b = G["turn_share"], G["approach_share"]
    credit = np.where(turned[None, :], a*turn + (1. - a)*((1. - b)*near + b*approach), near)
    V = np.hstack([credit, [[as_found(n, h) for h in desk_homes] for n in desks]])
    rows, cols = linear_sum_assignment(V, maximize=True)
    at_place = {c: r for r, c in zip(rows, cols)}
    criteria, slot_desk, slot_credit, slot_pos, out = {}, {}, {}, {}, {}
    for k, s in enumerate(slots):
        i = at_place.get(k)
        c = float(V[i, k]) if i is not None else 0.
        n = desks[i] if c > 0 else None
        slot_desk[s], slot_credit[s], slot_pos[s] = n, c, float(near[i, k]) if n else 0.
        if n is None:
            j = int(np.argmin(err[:, k]))
            criteria[s] = dict(check="slot", ok=False, desk=None, credit=0., closest_desk=desks[j],
                               closest_worst_glide_mm=round(float(err[j, k])*1e3, 1))
            continue
        ok = err[i, k] <= tol and dyaw[i, k] <= E["yaw_tol_deg"] and up[n] and rest[n]
        criteria[s] = dict(check="slot", ok=bool(ok), desk=n, worst_glide_mm=round(float(err[i, k])*1e3, 1),
                           yaw_error_deg=round(float(dyaw[i, k]), 2), tilt_deg=round(tilt[n], 2), upright=up[n],
                           at_rest=bool(rest[n]), credit=round(c, 4), position_credit=round(slot_pos[s], 4),
                           turn_credit=round(float(turn[i, k]), 4) if turned[k] else None)
    for k, h in enumerate(desk_homes):
        i = at_place.get(len(slots) + k)
        charge = 1. - (float(V[i, len(slots) + k]) if i is not None else 0.)
        if charge > 0:
            out[f"{h}'s place"] = round(charge, 4)

    # chairs -> 12 seats (at the desks now in the slots) + every other chair's place
    lo = F.DESK_EDGE_Y - F.CHAIR_SEAT_FRONT_Y + G["seat_front"][0]                 # chair centre behind the desk
    hi = F.DESK_EDGE_Y - F.CHAIR_SEAT_FRONT_Y + G["seat_front"][1]
    nearest = {}                     # the nearest desk the chair stands behind (student side) and faces
    for j in chairs:
        rel = []
        for d in desks:
            lx, ly = to_local(pose[d], *pose[j][:2])
            if ly > 0 and _ang(pose[j][2] - pose[d][2]) < math.pi/2:
                rel.append((math.hypot(lx, ly), d))
        nearest[j] = min(rel)[1] if rel else None
    fit = {}
    for j in chairs:
        for s in seats:
            d = slot_desk[chk[s][1]]
            if d is None:
                fit[j, s] = (0., None)
                continue
            lx, ly = to_local(pose[d], *pose[j][:2])
            face = math.degrees(_ang(pose[j][2] - pose[d][2]))
            off = math.hypot(max(lo - ly, ly - hi, 0.), max(abs(lx) - E["tuck_lateral"], 0.))
            over = max(face - E["facing_deg"], 0.)
            w = (max(0., 1. - off/G["chair_reach"])*max(0., 1. - over/(90. - E["facing_deg"]))*standing[j]
                 *(nearest[j] == d)*slot_pos[chk[s][1]])
            fit[j, s] = (w, dict(desk=d, behind_m=round(ly, 3), lateral_m=round(lx, 3), yaw_error_deg=round(face, 2),
                                 seat_front_beyond_edge_m=round(ly - (F.DESK_EDGE_Y - F.CHAIR_SEAT_FRONT_Y), 3),
                                 nearest_desk=nearest[j], window=off == 0. and over == 0.))
    W = np.hstack([[[fit[j, s][0] for s in seats] for j in chairs],
                   [[as_found(j, h) for h in chair_homes] for j in chairs]])
    rows, cols = linear_sum_assignment(W, maximize=True)
    at_place = {c: r for r, c in zip(rows, cols)}
    seat_credit = {}
    for k, s in enumerate(seats):
        i = at_place.get(k)
        w, m = fit[chairs[i], s] if i is not None else (0., None)
        seat_credit[s] = w
        if m is None or (w == 0. and not m["window"]):
            criteria[s] = dict(check="seat", ok=False, chair=None, desk=slot_desk[chk[s][1]], credit=0.)
            continue
        j = chairs[i]
        ok = m["window"] and m["nearest_desk"] == m["desk"] and up[j] and rest[j]
        criteria[s] = dict(check="seat", ok=bool(ok), chair=j, **m, tilt_deg=round(tilt[j], 2), upright=up[j],
                           at_rest=bool(rest[j]), credit=round(w, 4))
    for k, h in enumerate(chair_homes):
        i = at_place.get(len(seats) + k)
        charge = 1. - (float(W[i, len(seats) + k]) if i is not None else 0.)
        if charge > 0:
            out[f"{h}'s place"] = round(charge, 4)

    # pods: each board-side slot and the slot behind it meet front edge to front edge: the closest approach of the
    # edges -2..+10 mm (contact anywhere), the widest no more than pod_widest, sideways within pod_lateral
    meet = []
    for p, places in ((p, v) for p, v in chk.items() if p.startswith("pod")):
        pairs, ok = {}, True
        for f, b in ((places[0], places[2]), (places[1], places[3])):
            df, db = slot_desk[f], slot_desk[b]
            span = edge_span(pose[df], pose[db]) if df and db else None
            if span is None:
                gap = wide = lat = None
                good, c = False, 0.
            else:
                (gap, wide), lat = span, abs(pose[db][0] - pose[df][0])
                good = -0.002 <= gap <= E["pod_gap"] and wide <= G["pod_widest"] and lat <= G["pod_lateral"]
                excess = (max(0., gap - E["pod_gap"], -0.002 - gap) + max(0., wide - G["pod_widest"])
                          + max(0., lat - G["pod_lateral"]))
                c = min(slot_pos[f], slot_pos[b])*max(0., 1. - excess/REACH)
            meet.append(c)
            ok &= good
            pairs[f"{f}|{b}"] = dict(desks=[df, db], gap_mm=None if gap is None else round(gap*1e3, 1),
                                     widest_mm=None if wide is None else round(wide*1e3, 1),
                                     lateral_mm=None if lat is None else round(lat*1e3, 1), credit=round(c, 4))
        criteria[p] = dict(check="pod", ok=bool(ok), pairs=pairs)

    # the rest of the room as found; the route; nothing toppled or in the door swing
    for n in loose:
        if kind[n] in ("tote", "basket"):
            (x, y, _), (x0, y0, _) = pose[n], items[n]["pose"]
            d0, d1 = G["loose_as_found"][0], G["loose_full"]
            d = math.hypot(x - x0, y - y0)
            value = (tilt[n] <= G["loose_as_found"][1])*max(0., 1. - max((d - d0)/(d1 - d0), 0.))
        else:
            value = as_found(n, n)
        if value < 1:
            out[n] = round(1. - value, 4)
    criteria["rest_of_room"] = dict(check="as_found", ok=not out, out_of_place=sorted(out), charges=out,
                                    places=len(desk_homes) + len(chair_homes) + len(loose))
    criteria["route"] = dict(check="route", ok=e.route(*chk["route"]))
    if not criteria["route"]["ok"]:
        criteria["route"]["widest_route_m"] = round(e.route_width(*chk["route"], G["route_floor"]), 4)
    criteria["no_toppled_or_blocking"] = e.safety()

    w = G["weights"]
    progress = (w["desks"]*float(np.mean(list(slot_credit.values()))) + w["abutment"]*float(np.mean(meet))
                + w["chairs"]*float(np.mean(list(seat_credit.values()))))
    pen = G["penalty"]
    narrow = 0. if criteria["route"]["ok"] else \
        min(1., (P.AISLE_MIN - criteria["route"]["widest_route_m"])/(P.AISLE_MIN - G["route_floor"]))
    penalty = ((1. - (1. - pen["route"])*narrow)
               *(pen["safety"] if not criteria["no_toppled_or_blocking"]["ok"] else 1.))
    deduction = G["place_charge"]*sum(out.values())
    succeeded = all(c["ok"] for c in criteria.values())
    score = 1. if succeeded else round(G["fail_scale"]*max(0., progress - deduction)*penalty, 4)
    return {"succeeded": succeeded, "score": score, "progress": round(progress, 4),
            "reward": dict(rule="1 if success else fail_scale x max(0, progress - rest_of_room_deduction) x penalty",
                           fail_scale=G["fail_scale"], rest_of_room_deduction=round(deduction, 4),
                           weights=dict(w), desk_places=round(float(np.mean(list(slot_credit.values()))), 4),
                           front_edge_meetings=round(float(np.mean(meet)), 4),
                           chairs=round(float(np.mean(list(seat_credit.values()))), 4), penalty=round(penalty, 4),
                           penalty_parts=dict(route=round(1. - (1. - pen["route"])*narrow, 4),
                                              safety=pen["safety"] if not criteria["no_toppled_or_blocking"]["ok"] else 1.)),
            "criteria": criteria, "window_complete": bool(window),
            "note": "world state only (poses, glide heights, contacts, 1 s rest window); desks and chairs matched to "
                    "places, not by name"}
