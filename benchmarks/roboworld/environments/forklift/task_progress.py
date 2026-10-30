"""Opt-in task reward adapter. Reads the settled world; never writes physics or start state.

Success predicates are evaluator.py's, byte-for-byte unchanged. Continuous quantities below
replace its check-count score only for catalogue.CONTRACT.
"""

import copy
import math

import numpy as np

from . import evaluator as E
from . import params as P
from .catalogue import CONTRACT, SET_TASKS
from .task import TIME_LIMIT_S

FAIL_SCALE = E.BALE_REWARD["fail_scale"]
RULE = "1 if success, 0 if faulted or entered on red, else 0.9 x progress x workmanship"


def clip(x):
    return min(1.0, max(0.0, float(x)))


def ramp(x, full, zero):
    return clip((zero - x) / (zero - full))


def band(x, lo, hi, zero_lo, zero_hi):
    return min(ramp(-x, -lo, -zero_lo), ramp(x, hi, zero_hi))


def gained(value, initial):
    """No credit for the settled initial quality, even if it already meets some conditions."""
    return clip((value - initial) / max(1.0 - initial, 1e-12))


def approach(distance, initial):
    return clip(1.0 - distance / max(initial, 1e-12))


def _rect_quality(corners, rect, tolerance=0.3):
    outside = max(max(rect["x"][0] - c[0], c[0] - rect["x"][1],
                      rect["y"][0] - c[1], c[1] - rect["y"][1]) for c in corners)
    return ramp(outside, 0.0, tolerance)


def unsecured(start):
    if start["task"] != "load-trailer":
        return False
    tr = start["trailers"][start["target"]["door"]]
    return not tr.get("restraint") and not tr.get("chocks")


def _front(world):
    p, R = world.body_pose("chassis")
    return float(p[1] + E.REACH_LOADED * math.sin(E.yaw_of(R)))


def measurements(world, start, result):
    """Task quantities from poses, physical control signals and the existing event latches."""
    task = start["task"]
    by = {c["name"]: c["measured"] for c in result["criteria"]}
    ev, zones = world.events, world.scene.zones
    if task == "start-and-park":
        tilt = by["forks_tilted_to_floor"]
        quality = np.mean((clip(ev.rest_s / 2), clip(by["parking_brake_set"] / 0.5),
                           clip(1 - abs(by["direction_neutral"])), float(not by["engine_off"]),
                           float(by["key_off"] == "off"), ramp(by["forks_lowered"], 0.020, 0.30),
                           ramp(tilt["blade_pitch_deg"], 0, 5) * ramp(max(tilt["tip_heights_m"]), 0.005, 0.15)))
        return dict(distance=max(0.0, by["in_parking_bay"]["outside_by_m"]), quality=float(quality))
    if task == "aisle-cruise":
        way = start.get("cruise", "east")
        traj = ev.traj
        sections = []
        for a, b in E.CRUISE_SECTIONS[way]:
            # In-band distance, capped by the unique span traversed: a tiny in-band move or
            # repeated shuttling inside one part of a section cannot earn the whole section.
            xs = [r for r in traj if a <= r[1] <= b]
            good = sum(abs(q[1] - p[1]) for p, q in zip(xs, xs[1:]) if 6.5 <= abs(q[4]) * 3.6 <= 8.0)
            span = max((r[1] for r in xs), default=a) - min((r[1] for r in xs), default=a)
            sections.append(min(clip(good / (0.8 * (b - a))), clip(span / (b - a))))
        crossings = []
        for name in ("intersection_I1", "intersection_I2"):
            c = by[name]
            crossings.append(0.0 if c is None or c["since_horn_s"] is None else
                             ramp(c["speed_kmh"], 5, 9) * ramp(c["since_horn_s"], 3, 8))
        error = abs(by["at_far_line"])
        return dict(distance=error, sections=sections, crossings=crossings,
                    centre=ramp(by["aisle_centreline"], 0.6, 1.5),
                    stop=ramp(error, 0.2, 2.0) * clip(ev.rest_s))
    bl, axis = E.blades(world)
    if task == "drive-to-pallet":
        _, o, R = E.pallet_corners(world, "target")
        tip = (bl[0]["tip"] + bl[1]["tip"]) / 2
        goals = [o + sign * R[:, 0] * (P.GMA_PALLET["L"] / 2 + 0.25) for sign in (-1, 1)]
        d = min(float(np.linalg.norm(tip[:2] - g[:2])) for g in goals)
        quality = np.mean((ramp(abs(by["tips_short_of_pallet"] - 0.25), 0.05, 0.6),
                           ramp(abs(by["blades_level"]), 1, 5),
                           min(band(z, 0.02, 0.06, 0, 0.2) for z in by["blade_height"]),
                           ramp(-min(by["forks_in_openings"]), -0.04, 0.05),
                           ramp(abs(by["centred_on_pallet"]), 0.05, 0.3), ramp(by["square_to_pallet"], 3, 15)))
        return dict(distance=d, quality=float(quality), pallet=o.tolist())
    corners, o, R = E.pallet_corners(world, "carried")
    if task in ("carry-to-floor-bay", "ramp-down-loaded"):
        yard = task == "ramp-down-loaded"
        zone = zones[E.DROP_ZONE[start.get("drop_zone", "east")]] if yard else zones["bays"][start["bay"]]
        goal = np.array([np.mean(zone["x"]), np.mean(zone["y"])])
        level = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))
        orientation = ramp(level, 5, 25) if yard else ramp(by["square_to_lines"], 5, 25)
        placement = (_rect_quality(corners, zone) * orientation *
                     ramp(max(abs(z) for z in by["resting_on_floor"]), 0.01, 0.15))
        return dict(distance=float(np.linalg.norm(o[:2] - goal)), placement=placement,
                    exit=clip(by["backed_out"] / 0.5) * ramp(by["forks_lowered"], 0.15, 0.6) * clip(ev.rest_s))
    if task == "put-away-rack":
        loc = zones["rack_locations"][start["rack_location"]]
        sign = 1 if loc["face"] == "w" else -1
        goal = (loc["aisle_x"] + sign * (P.GMA_PALLET["L"] / 2 - 0.076), loc["y"])
        # Geometry, not a count of beam contacts: target-height and footprint proximity
        # grade placement continuously; the unchanged success rule still needs both beams.
        d = float(np.linalg.norm(o[:2] - goal))
        height_error = max(abs(z - loc["z"]) for z in by["at_beam_level"])
        placement = (ramp(d, 0.03, 0.4) * ramp(height_error, 0.03, 0.3) *
                     band(by["front_overhang"], 0.051, 0.101, -0.05, 0.25) *
                     ramp(-by["side_clearance"], -0.05, 0) * ramp(by["level"], 2, 6))
        withdraw = by["forks_withdrawn_lowered"]
        return dict(distance=d, height_error=abs(float(o[2]) - loc["z"]), placement=placement,
                    exit=clip(withdraw["tip_to_pallet_m"]) * ramp(withdraw["tip_height_m"], 0.30, 1.0))
    if unsecured(start):
        lane = zones["staging"]["lane" + start["target"]["door"][4:]]
        gap = min(float(np.linalg.norm(b["tip"][:2] - o[:2])) for b in bl)
        quality = np.mean((clip(ev.rest_s / 2), clip(world.signals.get("park", 0) / 0.5),
                           clip(1 - abs(world.signals.get("direction", 0))), ramp(gap, P.GMA_PALLET["L"], 2.0)))
        return dict(distance=max(0.0, lane["y"][1] - _front(world)), quality=float(quality))
    tr, to, TR = E._trailer_frame(world, start["target"]["door"])
    local = (corners - to) @ TR
    ac, _, _ = E.pallet_corners(world, start["target"]["behind"])
    ahead_x = float(((ac - to) @ TR)[:, 0].min())
    sign = 1 if start["target"].get("side", "L") == "L" else -1
    goal = (ahead_x - P.GMA_PALLET["L"] / 2 - 0.025,
            sign * (tr["W"] / 2 - P.GMA_PALLET["W"] / 2 - E.WALL_GAP_MAX / 2))
    local_centre = TR.T @ (o - to)
    placement = (_rect_quality(local, dict(x=(0, tr["L"]), y=(-tr["W"] / 2, tr["W"] / 2))) *
                 ramp(max(abs(z) for z in local[:, 2]), 0.015, 0.25) *
                 band(by["gap_to_pallets_ahead"], -0.005, 0.05, -0.15, 0.4) *
                 band(by["gap_to_side_wall"], 0, E.WALL_GAP_MAX, -0.15, 0.5) * ramp(by["level"], 2, 6))
    p, Rc = world.body_pose("chassis")
    fp = E.footprint(float(p[0]), float(p[1]), E.yaw_of(Rc))
    ymin, ymax = min(v[1] for v in fp), max(v[1] for v in fp)
    face = zones["doors"][start["target"]["door"]]["face_y"]
    return dict(distance=float(np.linalg.norm(local_centre[:2] - goal)), placement=placement,
                exit=clip((face - ymin) / max(ymax - ymin, 1e-9)))


def progress(task, now, initial):
    a = approach(now["distance"], initial["distance"])
    if task == "start-and-park" or (task == "load-trailer" and "quality" in now):
        return a * (0.5 + 0.5 * now["quality"])
    if task == "aisle-cruise":
        return float(np.mean([*now["sections"], *now["crossings"], a * now["centre"], now["stop"]]))
    if task == "drive-to-pallet":
        moved = float(np.linalg.norm(np.array(now["pallet"]) - initial["pallet"]))
        return a * (0.4 + 0.6 * now["quality"]) * ramp(moved, 0.02, 0.3)
    s = gained(now["placement"], initial["placement"])
    e = s * now["exit"]  # current placement, no required history or order of operations
    if task == "put-away-rack":
        h = approach(now["height_error"], initial["height_error"])
        return 0.35 * a + 0.15 * h + 0.3 * s + 0.2 * e
    if task == "load-trailer":
        return 0.45 * a + 0.35 * s + 0.2 * e
    return 0.5 * a + 0.3 * s + 0.2 * e


def workmanship(world, start, result):
    """The load-bales damage grading, applied to the task's loads and site rules."""
    by = {c["name"]: c for c in result["criteria"]}
    det = result["details"]
    if "dropped" not in det:  # legacy floor-bay evaluator returns {}, though shared() measured its latches
        zone = world.scene.zones["bays"][start["bay"]]
        def destination(name, xy):
            return name == "carried" and E.inside_rect(xy, (zone["x"][0] - .05, zone["x"][1] + .05),
                                                       (zone["y"][0] - .05, zone["y"][1] + .05))
        _, det = E.shared(world, start, dest_floor=destination)
    rw, ev = E.BALE_REWARD, world.events
    floor_events, folded = E._floor_events(world, det, rw["floor_merge_s"])
    strikes = E._strike_events([s for i, s in enumerate(det["strikes"]) if i not in folded], rw["strike_merge_s"])
    floors = []
    for name, events in floor_events.items():
        corners, _, R = E.pallet_corners(world, name)
        low = min(float(c[2] - world.scene.ground_height(*c[:2])) for c in corners)
        for door in world.scene.trailers if hasattr(world.scene, "trailers") else ():
            tr, to, TR = E._trailer_frame(world, door)
            local = (corners - to) @ TR
            if 0 < local[:, 0].mean() < tr["L"] and abs(local[:, 1].mean()) < tr["W"] / 2:
                low = min(low, float(local[:, 2].min()))
        stranded = low <= .03 or R[2, 2] < math.cos(math.radians(20))
        speed = math.prod(1.0 if e["v"] <= E.SET_DOWN_SPEED else max(rw["speed_floor"],
                         1 / (1 + math.log10(e["v"] / E.SET_DOWN_SPEED))) for e in events)
        brushes = math.prod(rw["strand_factor"] + (rw["brush_factor"] - rw["strand_factor"]) /
                            (1 + (e["t1"] - e["t0"]) / rw["brush_decay_s"]) for e in events)
        floors.append(speed * (rw["strand_factor"] if stranded else brushes))
    m = lambda name: by[name]["measured"]
    grades = {
        "no_strike": lambda: math.prod(1 / (1 + .5 * math.log10(e["severity"])) for e in strikes),
        "no_dropped_load": lambda: math.prod(floors),
        "no_wheel_lift": lambda: 1 / (1 + max(0, m("no_wheel_lift") - .3)),
        "roll_under_6deg": lambda: 0.0 if ev.overturn is not None else 1 / (1 + max(0, ev.max_roll - 6) / 6),
        "speed_limit": lambda: 1 / (1 + ev.over_limit_s / 10 + 2 * max(0, m("speed_limit") / E.LIMIT_KMH - 1)),
        "operator_seated": lambda: 1 / (1 + max(0, m("operator_seated") - 1)),
        "head_clear_of_guard": lambda: min(1, P.HEAD_CONTACT_LIMIT / max(m("head_clear_of_guard"), 1e-9)),
        "trailer_movement": lambda: 1 / (1 + max(0, m("trailer_movement") - .025) / .025),
        "stored_loads_undisturbed": lambda: ramp(m("stored_loads_undisturbed")["max_m"], .03, .15),
        "slow_on_ramp": lambda: 1 / (1 + 2 * max(0, m("slow_on_ramp") / 6 - 1)),
    }
    factors = {k: float(f()) for k, f in grades.items() if k in by and not by[k]["passed"]}
    return math.prod(factors.values()), factors


class TaskProgress:
    """One settled start's baseline and terminal score, owned/checkpointed by the GAR backend."""

    def __init__(self, world, start):
        if start["task"] not in SET_TASKS[:-1]:
            raise ValueError(f"{CONTRACT} does not replace {start['task']}")
        self.start = copy.deepcopy(start)
        self.initial = measurements(world, start, E.evaluate(world, start))
        self.terminal = None

    def reset(self):
        self.terminal = None

    def on_tick(self, world):
        if self.terminal is not None:
            return
        if world.fault is not None or world.sim_time >= TIME_LIMIT_S[self.start["task"]] or self.entered_on_red(world, history=False):
            self.evaluate(world)

    def entered_on_red(self, world, *, history=True):
        if not unsecured(self.start):
            return False
        edge = world.scene.zones["doors"][self.start["target"]["door"]]["face_y"] - P.LEVELER["length"]
        return _front(world) >= edge or (history and any(y + E.REACH_LOADED * math.sin(yaw) >= edge
                                                       for _, _, y, yaw, _ in world.events.traj))

    def evaluate(self, world):
        if self.terminal is not None:
            return copy.deepcopy(self.terminal)
        fault = world.fault
        red = False if fault is not None else self.entered_on_red(world)
        if fault is not None:
            out = dict(schema="gar.evaluation.v1", task=self.start["task"], success=False, criteria=[], details={},
                       fault=fault, sim_time=world.sim_time)
            p, w, factors = 0.0, 0.0, {}
        else:
            out = E.evaluate(world, self.start)
            now = measurements(world, self.start, out)
            p = 0.0 if red else clip(progress(self.start["task"], now, self.initial))
            w, factors = workmanship(world, self.start, out)
        if red:
            out["success"] = False
        done = fault is not None or red or world.sim_time >= TIME_LIMIT_S[self.start["task"]]
        out.update(task_contract=CONTRACT, progress=p, score_rule=RULE,
                   score=1.0 if out["success"] else FAIL_SCALE * p * w, episode_done=done,
                   end_reason="fault" if fault is not None else "entered_on_red" if red else "time_limit" if done else None)
        out["details"]["reward"] = dict(contract=CONTRACT, rule=RULE, fail_scale=FAIL_SCALE, progress=p,
                                          workmanship=w, factors=factors)
        if done:
            self.terminal = copy.deepcopy(out)
        return out
