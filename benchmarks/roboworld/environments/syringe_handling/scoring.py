"""Continuous rewards for the seven tasks other than vial-to-pad; no stepping or commanded-action access.

Success still comes from task.evaluate's existing physical predicates. Explicit contract only:
1 on success, else 0.9 * progress. A task's own progress is progress(env) in its verifier.py (tasks/<task>/); the parts
several tasks share are here.
"""
import math

import numpy as np

from . import params as P, hydraulics as H, task as T
from .config import EIGHT_TASK_CONTRACT, task_verifier

FAIL_SCALE = 0.9  # vial-to-pad's partial weights also sum to 0.9
COMMON = ("at_rest", "on_the_counter", "no_protective_stop", "spills", "needle_hygiene",
          "no_resheathing_by_hand", "needles_not_bent", "syringes_whole")


def clip(x):
    return max(0.0, min(1.0, float(x)))


def grade(x, full, zero):
    """One up to full, falling continuously to zero at zero (zero > full)."""
    return clip((zero - x)/(zero - full))


def improved(error, initial, tolerance):
    """Net improvement from handover; an already satisfied start earns no work credit."""
    return clip((initial - error)/(initial - tolerance)) if initial > tolerance else 0.0


def near_box(points, box, band=0.05):
    """Mean graded distance: 1 inside the cavity, linearly 0 at 50 mm outside.

    Samples use distances, never inside/outside booleans (which would give steps).
    Unlike a route to a slot this accepts every physically achieved placement.
    """
    pts = np.asarray(points, dtype=float)
    delta = np.maximum(np.maximum(np.asarray(box["lo"]) - pts, pts - np.asarray(box["hi"])), 0.0)
    return float(np.clip(1.0 - np.linalg.norm(delta, axis=-1)/band, 0.0, 1.0).mean())


def sharps_progress(env, names):
    w, fl, sc = env.world, env.world.fluid, env.world.scene
    amounts = []
    for name in names:
        p, R = w.body_pose(sc.bodies[name])
        if name in fl.syringes:
            end = p + R[:, 0]*fl.syringes[name].s["barrel_len"]
        else:
            # Actual bent tube's tip, not a nominal rigid needle length.
            end = fl.needles[name].tip_world(w)[0]
        pts = p + np.linspace(0., 1., 17)[:, None]*(end - p)
        amounts.append(near_box(pts, sc.sharps_cavity))
    contained = sum(amounts)/len(amounts)
    flap = grade(max(0.0, math.degrees(float(w.jq[sc.flap_joint.q]))), 5.0, 90.0)
    others = sum(near_box([w.bq[b, :3]], sc.sharps_cavity, band=0.02)
                 for n, b in sc.bodies.items() if n not in names and n != "sharps_flap" and b not in sc._robot_set)
    assemblies = sum(n in fl.syringes for n in names)
    progress = clip(contained*(0.5 + 0.5*flap) - others/max(assemblies, 1))
    return progress, dict(contained=contained, flap=flap, foreign_units=others/max(assemblies, 1))


def syringe_progress(env, graded):
    """set-dose's and draw-dose's progress: graded(env, name, d, s, v, air, tol), the task's own (its verifier.py), from
    the dose syringe's name and its device s, the dose d, the dose v and the air (mL) the syringe holds and the dose's
    tolerance tol."""
    spec, fl = env.spec, env.world.fluid
    name, d = spec["dose_syringe"], spec["dose_ml"]
    s = fl.syringes[name]
    v, air = max(s.chamber.liquid - s.chamber.dead, 0.0)/H.ML, s.chamber.gas0/H.ML
    tol = T.dose_tolerance_ml(d, s.s["nominal_ml"])
    return graded(env, name, d, s, v, air, tol)


def lock_progress(env, work=None, sp=None, timed=None):
    """The iv-lock tasks' progress, with the task's own part from its verifier.py: either work(env, F, K, parts), the
    task's own work (flush-lock) from F, the share of the flush in, and K, the lock clamped with nothing on it; or a
    dose of sp first (slow-push, fast-bolus), where timed(env, sp, D, parts) is how much of D, the share of the dose
    that reached the vein, counts by its rate and time. Times the line pressure's and the air's grades."""
    w, spec = env.world, env.spec
    fl = w.fluid
    led, lk = fl.ledger, fl.locks["lock"]
    saline = sum(float(m[P.SPECIES.index("saline")]) for _, m, _ in led.arrivals.get("lock:in", []))/H.ML
    F = clip(saline/(0.95*spec["flush_ml"]))
    limit = T.BLOCKED_LIMIT_PA if lk.blocked else P.LINE_LIMIT_PA
    pressure = grade(lk.peak_gauge, limit, 2*limit)
    air = grade(led.volume("vein", "air")/H.ML, T.AIR_LIMIT_ML, 0.5)
    r = lk.reading(w)
    clamped = clip(r["clamp_mm"]/(0.5e3*P.EXTENSION["clamp_travel"]))
    detached = 1.0 if lk.female.mate is None else 1.0 - clip(lk.female.mate.depth(w)/P.LUER["d_seat"])
    K = clamped*detached
    parts = dict(flush=F, pressure=pressure, air=air, closed_detached=K)
    if work is not None:
        progress = work(env, F, K, parts)
    else:
        D = clip(led.volume("vein", sp)/H.ML/(0.95*spec["dose_ml"]))
        delivered = timed(env, sp, D, parts)
        parts["dose"] = D
        progress = delivered*(1.0 + F + F*K)/3.0
    return progress*pressure*air, parts


def reward(env, criteria, success):
    """1 on success, else 0.9 x the progress (the task's verifier.py: progress(env) -> (raw progress, parts)) x the
    housekeeping factor."""
    task = env.spec["task"]
    task_progress = getattr(task_verifier(task), "progress", None)
    if task_progress is not None:
        raw, parts = task_progress(env)
    else:
        raise ValueError(f"No {EIGHT_TASK_CONTRACT} reward for {task!r}; vial-to-pad uses its own evaluator")
    failed = [k for k in COMMON if not criteria[k]["ok"]]
    house = max(T.HOUSEKEEPING_FLOOR, 1.0 - T.HOUSEKEEPING_STEP*len(failed))
    progress = clip(raw)*house
    return dict(contract=EIGHT_TASK_CONTRACT, score=1.0 if success else FAIL_SCALE*progress, progress=progress,
                score_parts=dict(parts, raw_progress=clip(raw), housekeeping=house, housekeeping_failed=failed,
                                 failure_factor=FAIL_SCALE, score_rule="1 if success else 0.9 x progress"))
