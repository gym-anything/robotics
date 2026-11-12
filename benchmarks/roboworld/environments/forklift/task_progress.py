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
from .task import TIME_LIMIT_S, task_verifier

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


def measurements(world, start, result):
    """Task quantities from poses, physical control signals and the existing event latches: the task's verifier.py
    (tasks/<task>/) measures them, measurements(world, start, by) with by the evaluation's measured value of each
    criterion."""
    by = {c["name"]: c["measured"] for c in result["criteria"]}
    return task_verifier(start["task"]).measurements(world, start, by)


def set_down(world, by, zone, yard):
    """The measures of a pallet carried to a zone and set down there, as carry-to-floor-bay (a floor bay) and
    ramp-down-loaded (a yard drop zone, yard=True: judged level instead of square to painted lines) take them."""
    ev = world.events
    corners, o, R = E.pallet_corners(world, "carried")
    goal = np.array([np.mean(zone["x"]), np.mean(zone["y"])])
    level = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))
    orientation = ramp(level, 5, 25) if yard else ramp(by["square_to_lines"], 5, 25)
    placement = (_rect_quality(corners, zone) * orientation *
                 ramp(max(abs(z) for z in by["resting_on_floor"]), 0.01, 0.15))
    return dict(distance=float(np.linalg.norm(o[:2] - goal)), placement=placement,
                exit=clip(by["backed_out"] / 0.5) * ramp(by["forks_lowered"], 0.15, 0.6) * clip(ev.rest_s))


def progress(task, now, initial):
    """The task's progress from the settled start's measurements to now's: its verifier.py's progress(now, initial,
    approach), with approach() above."""
    return task_verifier(task).progress(now, initial, approach)


def set_down_progress(now, initial, approach):
    """carry-to-floor-bay's and ramp-down-loaded's progress: the approach, the placement gained and the exit."""
    a = approach(now["distance"], initial["distance"])
    s = gained(now["placement"], initial["placement"])
    e = s * now["exit"]  # current placement, no required history or order of operations
    return 0.5 * a + 0.3 * s + 0.2 * e


def workmanship(world, start, result):
    """The load-bales damage grading, applied to the task's loads and site rules."""
    by = {c["name"]: c for c in result["criteria"]}
    det = result["details"]
    if "dropped" not in det:  # legacy floor-bay evaluator returns {}, though shared() measured its latches
        det = task_verifier(start["task"]).site_details(world, start)
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
        """Whether the truck went past an unsecured trailer's leveler: the task's verifier.py decides (load-trailer's
        entered_on_red); no other task has a red signal."""
        own = getattr(task_verifier(self.start["task"]), "entered_on_red", None)
        return own is not None and own(world, self.start, history=history)

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
