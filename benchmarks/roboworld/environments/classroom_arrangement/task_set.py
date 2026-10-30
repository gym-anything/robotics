"""State-only, opt-in task grading. No simulator imports or changes to group-work.

The place assignment, rest-of-room costs and failure factor follow the group-work
evaluator. Progress measures reduction in geometric errors, relative to the
actual post-settle start. A feature already satisfied then has no reward
weight. Assignment and success still check it.
"""

import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import furniture as F, params as P
from .catalogue import CONTRACT
from .task import E, _ang, to_local

G = P.EVAL_GROUP_WORK


def clip(x):
    return min(1., max(0., float(x)))


def reduction(error, initial, tolerance=0.):
    """None means a constraint met at the start: preserve it, but do not reward it."""
    if initial <= tolerance + 1e-9:
        return None
    return clip((initial - error)/(initial - tolerance))


def mean_active(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def weighted_active(parts):
    active = [(w, p) for w, p in parts if p is not None]
    return sum(w*p for w, p in active)/sum(w for w, _ in active) if active else 0.


def outside(x, lo, hi):
    return max(lo - x, x - hi, 0.)


def prepare_start(e):
    """Called once after the environment's existing settle, before policy actions."""
    e._task_set_start = {
        "poses": {n: e.pose(n)[0] for n in e.spec["items"]},
        "glides": {n: e.glides_xyz(n)[:, :2].copy() for n in e._glide_geoms},
    }
    if "route" in e.spec["check"]:
        ends = e.spec["check"]["route"]
        e._task_set_start["route"] = P.AISLE_MIN if e.route(*ends) else e.route_width(*ends, 0., steps=12)


def chair_error(pose, reference, kind, rng=None, side=None):
    """Distance outside the allowed seat window plus facing error in metre equivalents.

    A full 75-degree facing error is worth half a chair depth. Ready seats use
    the actual seat front and table frame, not an arbitrary chair centre dot.
    """
    x, y, yaw = pose
    lx, ly = to_local(reference, x, y)
    face = math.degrees(_ang(yaw - reference[2]))
    if kind == "ready":
        front = (x - F.CHAIR_SEAT_FRONT_Y*math.sin(yaw), y + F.CHAIR_SEAT_FRONT_Y*math.cos(yaw))
        fx, fy = to_local(reference, *front)
        hx, hy = P.TABLE_TOP[0]/2, P.TABLE_TOP[1]/2
        if side == "room":
            distance, lateral, extent, want = -fy - hy, fx, hx - 0.2, math.pi
        elif side == "left_end":
            distance, lateral, extent, want = -fx - hx, fy, hy - 0.15, math.pi/2
        else:
            distance, lateral, extent, want = fx - hx, fy, hy - 0.15, -math.pi/2
        face = math.degrees(_ang(yaw - reference[2] - want))
        off = math.hypot(outside(distance, *E["ready_range"]), max(0., abs(lateral) - extent))
        limit = E["facing_deg"]
    else:
        lateral = lx
        limit = E["tuck_yaw_deg"] if kind == "tucked" else E["facing_deg"]
        if kind == "tucked":
            # A chair must be on the student side, not through the desk's closed front.
            lo, hi = 0., F.DESK_EDGE_Y - F.CHAIR_SEAT_FRONT_Y - E["tuck_depth"]
            distance, extent = ly, E["tuck_lateral"]
        elif kind == "behind":
            lo, hi = rng
            distance, extent = -ly, 0.40
            # Teacher faces the windows, not away from the desk (legacy accepted both).
            face = math.degrees(_ang(yaw - reference[2] - math.pi))
        else:
            lo, hi = E["use_range"]
            distance, extent = ly, E["tuck_lateral"]
        off = math.hypot(outside(distance, lo, hi), max(0., abs(lateral) - extent))
    over = max(0., face - limit)
    error = off + G["chair_reach"]*over/(90. - limit)
    return error, dict(distance_m=float(distance), lateral_m=float(lateral), facing_error_deg=face,
                       position_error_m=off, side=side)


def on_top(e, chair, desk):
    """Chair geometry and actual contact with the desk top; hovering never succeeds."""
    w = e.env.world
    ok, details = e.up_on(chair, desk)
    p, R = w.body_pose(chair)
    dp, dR = w.body_pose(desk)
    seat = next(part for part in w.scene.parts[chair] if part.role == "seat")
    # Lowest seat-box point in the desk's vertical axis, relative to its top.
    local = dR.T @ (p + R @ np.asarray(seat.pos) - dp)
    axes = dR.T @ R
    bottom = local[2] - float(np.abs(axes[2]) @ np.asarray(seat.size))
    gap = bottom - P.DESK_HEIGHT
    shapes = w.scene.items[desk]["shapes"]
    parts = [part for part in w.scene.parts[desk] if part.collide]
    top_geoms = {w._shape_geom[s] for s, part in zip(shapes, parts) if part.role == "top"}
    chair_geoms = {w._shape_geom[s] for s in w.scene.items[chair]["shapes"]}
    contact = any((a in top_geoms and b in chair_geoms) or (b in top_geoms and a in chair_geoms)
                  for a, b in (w.mjd.contact[i].geom for i in range(w.mjd.ncon)))
    supported = contact and abs(gap) <= 0.01
    tilt = e.pose(chair)[1]
    body = w.scene.bodies[chair]
    com = p + R @ np.asarray(w.model.body_com.numpy()[body], float)
    dpose = e.pose(desk)[0]
    cx, cy = to_local(dpose, com[0], com[1])
    inside_m = min(P.DESK_TOP[0]/2 - abs(cx), P.DESK_TOP[1]/2 - abs(cy))
    dside = np.array([-math.sin(dpose[2]), math.cos(dpose[2])])
    cside = R[:2, 1]/max(1e-9, float(np.linalg.norm(R[:2, 1])))
    side_deg = math.degrees(math.acos(float(np.clip(cside @ dside, -1., 1.))))
    # Smooth partial credit as the lifted seat approaches the top; the floor gives
    # zero. Contact is a success predicate, not a discontinuous half-point bonus.
    lift = clip((bottom - P.CHAIR_SEAT_H)/(P.DESK_HEIGHT - P.CHAIR_SEAT_H))
    proximity = clip(1. - max(0., gap)/0.10)
    inside = clip(inside_m/E["inside_edge"])
    inversion = clip((tilt - 90.)/(E["inverted_deg"] - 90.))
    side = clip((90. - side_deg)/(90. - E["up_side_deg"]))
    value = lift*proximity*inside*(0.5 + 0.5*min(inversion, side))
    details.update(supported_by_top=bool(supported), seat_bottom_gap_m=float(gap),
                   com_inside_edge_m=inside_m, backrest_side_error_deg=side_deg)
    return bool(ok and supported), value, details


class CatalogueEvaluation:
    def __init__(self, evaluator, motion, window):
        self.e, self.spec = evaluator, evaluator.spec
        self.motion, self.window = motion, bool(window)
        self.items, self.checks = self.spec["items"], self.spec["check"]
        self.start = evaluator._task_set_start
        self.poses = {n: evaluator.pose(n)[0] for n in self.items}
        self.tilts = {n: evaluator.pose(n)[1] for n in self.items}
        self.up = {n: evaluator.upright(n)[0] for n in self.items}
        self.criteria, self.charges, self.at = {}, {}, {}
        self.progress = {"desks": [], "chairs": [], "tote": [], "basket": [], "route": []}
        self.desk_quality = {}

    def resting(self, n):
        return self.window and self.motion.get(n, math.inf) <= E["rest_motion"]

    def home(self, n, h):
        p, p0 = self.poses[n], self.start["poses"][h]
        d, yaw = math.dist(p[:2], p0[:2]), math.degrees(_ang(p[2] - p0[2]))
        kind = self.items[h]["kind"]
        if kind in ("tote", "basket"):
            value = (self.tilts[n] <= 10)*clip(1. - max(0., d - 0.20)/(G["loose_full"] - 0.20))
        else:
            d0, a0 = G["as_found"] if kind == "chair" else G["as_found_furniture"]
            if self.checks.get(h) == "unmoved":
                d0, a0 = 0.05, 5.
            d1, a1 = G["as_found_full"]
            value = self.up[n]*clip(1. - max(0., (d - d0)/(d1 - d0), (yaw - a0)/(a1 - a0)))
        return value, dict(check="as_found", ok=bool(value >= 1. - 1e-12 and self.resting(n)),
                           item=n, distance_m=d, yaw_error_deg=yaw, at_rest=self.resting(n))

    def desk(self, n, h, candidates):
        target, dots = self.spec["targets"][h], np.asarray(self.spec["dots"][h])
        err = float(np.linalg.norm(self.e.glides_xyz(n)[:, :2] - dots, axis=1).max())
        yaw = math.degrees(_ang(self.poses[n][2] - target[2]))
        d0 = min(float(np.linalg.norm(self.start["glides"][j] - dots, axis=1).max()) for j in candidates)
        a0 = min(math.degrees(_ang(self.start["poses"][j][2] - target[2])) for j in candidates)
        value = mean_active([reduction(err, d0, E["glide_tol"]), reduction(yaw, a0, E["yaw_tol_deg"])])
        standing = self.tilts[n] <= G["toppled_deg"]
        if value is not None:
            value *= standing
        excess = max(0., err - E["glide_tol"])/G["chair_reach"] + max(0., yaw - E["yaw_tol_deg"])/90.
        quality = float(standing)/(1. + excess)
        ok = err <= E["glide_tol"] + 1e-12 and yaw <= E["yaw_tol_deg"] + 1e-12 and self.up[n] and self.resting(n)
        return quality, value, dict(check="on_marks", ok=bool(ok), item=n, worst_glide_mm=err*1000,
                                    yaw_error_deg=yaw, at_rest=self.resting(n))

    def chair(self, n, h, candidates):
        chk = self.checks[h]
        kind, ref = chk[:2]
        desk = self.at.get(ref, ref)
        gate = self.desk_quality.get(ref, 1.)
        if kind == "up_on":
            ok, value, detail = on_top(self.e, n, desk)
            return value*gate, value*gate, dict(check=kind, ok=bool(ok and self.resting(n)), item=n, desk=desk,
                                               at_rest=self.resting(n), **detail)
        side = None
        if kind == "ready":
            side = {"tchair_1": "room", "tchair_2": "room", "tchair_3": "left_end", "tchair_4": "right_end"}[h]
        rng = chk[2] if kind == "behind" else None
        reference = self.poses[desk]
        initial_reference = self.spec["targets"].get(ref, self.start["poses"][ref])
        initial = min(chair_error(self.start["poses"][j], initial_reference, kind, rng, side)[0] for j in candidates)
        err, detail = chair_error(self.poses[n], reference, kind, rng, side)
        standing = self.tilts[n] <= G["toppled_deg"]
        # Shape toward the fixed destination. Measuring this against a desk that
        # itself moves would pay for desk motion a second time and could give the
        # untouched jittered row credit. Success uses the actual placed desk.
        progress_error = chair_error(self.poses[n], initial_reference, kind, rng, side)[0]
        value = reduction(progress_error, initial)
        if value is not None:
            value *= standing*gate
        quality = float(standing)/(1. + err/G["chair_reach"])
        ok = err <= 1e-12 and self.up[n] and self.resting(n)
        return quality, value, dict(check=kind, ok=bool(ok), item=n, desk=desk, at_rest=self.resting(n), **detail)

    def assign(self, kind):
        names = sorted(n for n, it in self.items.items() if it["kind"] == kind)
        if not names:
            return
        # Every object of a kind fills exactly one target or one preserved place.
        # A perfect geometric solution has quality 1 in every column. No chair
        # identity or particular arm/action appears in the objective.
        entries = {}
        for n in names:
            for h in names:
                chk = self.checks.get(h)
                if chk == "on_marks":
                    q, p, detail = self.desk(n, h, names)
                elif isinstance(chk, tuple) and kind == "chair":
                    q, p, detail = self.chair(n, h, names)
                else:
                    q, detail = self.home(n, h)
                    p = None
                entries[n, h] = q, p, detail
        rows, cols = linear_sum_assignment([[entries[n, h][0] for h in names] for n in names], maximize=True)
        for i, j in zip(rows, cols):
            n, h = names[i], names[j]
            q, p, detail = entries[n, h]
            self.at[h] = n
            self.criteria[h] = detail
            if detail["check"] == "as_found":
                if q < 1.:
                    self.charges[h] = 1. - q
            else:
                group = "chairs" if kind == "chair" else "desks"
                self.progress[group].append(p)
                detail["progress"] = p
                if group == "desks":
                    self.desk_quality[h] = q
                    # Dots already met at reset (chairs-up) are preservation constraints.
                    if p is None and q < 1.:
                        self.charges[h] = 1. - q

    def loose(self, n):
        chk = self.checks.get(n)
        if chk is None:
            q, detail = self.home(n, n)
            self.criteria[n] = detail
            if q < 1.:
                self.charges[n] = 1. - q
            return
        def error(pose):
            if chk[0] == "near":
                return max(0., math.dist(pose[:2], chk[1]) - chk[2])
            xr, yr = chk[1]
            return math.hypot(outside(pose[0], *xr), outside(pose[1], *yr))
        err, initial = error(self.poses[n]), error(self.start["poses"][n])
        value = reduction(err, initial)
        if value is not None:
            value *= self.tilts[n] <= 10.
        # Zone/near are floor destinations; floating or buried props cannot pass.
        w = self.e.env.world
        z = float(w.body_pose(n)[0][2])
        floor = self.e._touches_floor(n)
        ok = err <= 1e-12 and self.tilts[n] <= 10. and floor and abs(z) < 0.03 and self.resting(n)
        self.criteria[n] = dict(check=chk[0], ok=bool(ok), distance_outside_m=err, at_rest=self.resting(n),
                                on_floor=bool(floor), progress=value)
        self.progress[n].append(value)

    def safety(self):
        # The chairs-up exception follows placed chairs, not names. Only chairs
        # actually supported on one of the target desks are exempt from tipping.
        allowed = {c["item"] for c in self.criteria.values() if c["check"] == "up_on" and c.get("supported_by_top")}
        bad = []
        for n, it in self.items.items():
            if it["kind"] in ("desk", "chair", "table", "teacher_desk") and self.tilts[n] > 30 and n not in allowed:
                bad.append(f"{n} tipped")
            if self.e.in_door_zone(n):
                bad.append(f"{n} in the door swing")
        return dict(check="safety", ok=not bad, problems=bad)

    def evaluate(self):
        for kind in ("desk", "teacher_desk", "table", "chair"):
            self.assign(kind)
        for n in ("tote", "basket"):
            self.loose(n)
        if "exam" in self.checks:
            chairs = [self.at[n] for n in self.checks["exam"]]
            spacing = min(math.dist(self.poses[a][:2], self.poses[b][:2]) for i, a in enumerate(chairs) for b in chairs[i+1:])
            self.criteria["exam"] = dict(check="exam_spacing", ok=spacing >= E["exam_pitch"] - 1e-9,
                                         min_chair_spacing_m=spacing)
        if self.spec["task"] == "seat-tutoring-group":
            sides = [self.criteria[f"tchair_{i}"]["side"] for i in range(1, 5) if self.criteria[f"tchair_{i}"]["ok"]]
            self.criteria["seating_plan"] = dict(check="seating_plan", ok=sorted(sides) == ["left_end", "right_end", "room", "room"])
        route_penalty = 1.
        if "route" in self.checks:
            ends = self.checks["route"]
            ok = self.e.route(*ends)
            width = P.AISLE_MIN if ok else self.e.route_width(*ends, 0., steps=12)
            self.criteria["route"] = dict(check="route", ok=ok, widest_route_m=width)
            self.progress["route"].append(reduction(P.AISLE_MIN - width, P.AISLE_MIN - self.start["route"]))
            narrow = clip((P.AISLE_MIN - width)/(P.AISLE_MIN - G["route_floor"]))
            route_penalty = 1. - (1. - G["penalty"]["route"])*narrow
        self.criteria["rest_of_room"] = dict(check="as_found", ok=not self.charges, charges=self.charges)
        self.criteria["no_toppled_or_blocking"] = self.safety()
        weights = {"desks": 0.6, "chairs": 0.4}
        if self.spec["task"] == "move-teacher-desk":
            weights = {"desks": 0.75, "chairs": 0.25}
        elif self.spec["task"] == "clear-floor":
            weights = {"tote": 0.25, "basket": 0.25, "chairs": 0.25, "route": 0.25}
        parts = {k: mean_active(v) for k, v in self.progress.items()}
        raw = weighted_active([(w, parts[k]) for k, w in weights.items()])
        deduction = G["place_charge"]*sum(self.charges.values())
        safety_penalty = 1. if self.criteria["no_toppled_or_blocking"]["ok"] else G["penalty"]["safety"]
        progress = clip(raw - deduction)*route_penalty*safety_penalty
        success = all(c["ok"] for c in self.criteria.values())
        return dict(succeeded=success, score=1. if success else G["fail_scale"]*progress, progress=progress,
                    contract=CONTRACT, window_complete=self.window, criteria=self.criteria,
                    reward=dict(contract=CONTRACT, rule="1 on success else 0.9 x progress", fail_scale=G["fail_scale"],
                                raw_progress=raw, parts=parts, rest_of_room_deduction=deduction,
                                route_penalty=route_penalty, safety_penalty=safety_penalty),
                    note="Simulator state only; one-to-one places; progress relative to post-settle start")
