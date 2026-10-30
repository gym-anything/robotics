"""Continuous rewards for the seven tasks other than vial-to-pad; no stepping or commanded-action access.

Success still comes from task.evaluate's existing physical predicates. Explicit contract only:
1 on success, else 0.9 * progress.
"""
import math

import numpy as np

from . import params as P, hydraulics as H, task as T
from .config import EIGHT_TASK_CONTRACT

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


def syringe_progress(env):
    spec, fl = env.spec, env.world.fluid
    name, d = spec["dose_syringe"], spec["dose_ml"]
    s = fl.syringes[name]
    v, air = max(s.chamber.liquid - s.chamber.dead, 0.0)/H.ML, s.chamber.gas0/H.ML
    tol = T.dose_tolerance_ml(d, s.s["nominal_ml"])
    if spec["task"] == "set-dose":
        initial = env.start["syringe_dose_ml"][name]
        dose = improved(abs(v - d), abs(initial - d), tol)
        cleared = improved(air, env.start["syringe_air_ml"][name], T.AIR_LIMIT_ML)
        # Air credit vanishes continuously before the syringe is empty; never reward discarding the dose.
        retained = clip((v - 0.5*d)/(0.5*d))
        return 0.5*(dose + cleared*retained), dict(dose=dose, air_cleared=cleared, retained=retained)
    dose = clip(min(1.0, v/(d - tol)) - max(0.0, v - d - tol)/d)
    clear = grade(air, T.AIR_LIMIT_ML, max(0.5, 0.2*d))
    vial = fl.vials["vial"]
    r = vial.reading(env.world)
    moved = float(np.linalg.norm(env.world.bq[vial.body, :2] - np.asarray(env.start["vial_xy"]["vial"])))
    # Continuous physical deviations, not a fraction of the vial's Boolean checks.
    pose = min(grade(1.0 - r["upright_cos"], 0.01, 0.2), grade(moved, 0.03, 0.15),
               grade(env.world.bq[vial.body, 2] - P.COUNTER_Z, 0.004, 0.10))
    pressure = grade(abs(vial.vessel.gas_pressure() - P.P_ATM), T.VIAL_PRESSURE_PA, 100e3)
    loss = env.start["vial_ml"]["vial"] - r["liquid_ml"] - s.chamber.liquid/H.ML
    contents = grade(loss, 0.1, d + 0.1)
    kept = pose*pressure*contents
    return dose*(0.6 + 0.2*clear + 0.2*kept), dict(dose=dose, air_clear=clear, vial_pose=pose,
                                                    vial_pressure=pressure, vial_contents=contents)


def injection_progress(env):
    fl, spec = env.world.fluid, env.spec
    led, d, layer = fl.ledger, spec["dose_ml"], spec["layer"]
    recs = [r for nd in fl.needles.values() for r in nd.records if r["target"] == "pad"]
    angle = 45.0 if layer == "fat" else T.NEEDLE_ANGLE_DEG

    def quality(t):
        active = [r for r in recs if r["t"] - 1e-6 <= t <= (math.inf if r["t_out"] is None else r["t_out"] + 1e-6)]
        return min((min(grade(r["offset_mm"], T.RING_MM, 2*T.RING_MM),
                        grade(r["angle_deg"], angle, 2*angle)) for r in active), default=0.0)

    want = f"pad:{layer}"
    rows = [(float(m[:H.AIR].sum())/H.ML, quality(t)) for t, m, _ in led.arrivals.get(want, [])]
    came = sum(v for v, q in rows)
    site = sum(v*q for v, q in rows)/came if came > 0 else 0.0
    pad = {k: led.volume(k)/H.ML for k in led.places if k.startswith("pad:")}
    over = max(0.0, sum(pad.values()) - 1.1*d)/d
    back = max(0.0, pad.get("pad:surface", 0.0) - 0.05*d)/d
    volume = clip(min(1.0, pad.get(want, 0.0)/(0.9*d)) - max(over, back))
    rate = grade(T._tissue_window(led, 2.0), 0.3, T.RATE_ZERO_ML)
    given = volume*site*rate
    name = spec["dose_syringe"]
    cleared, parts = sharps_progress(env, [name, f"{name}_needle"])
    # Disposal is work only in proportion to a delivered dose; no fixed >0.1 step.
    used = clip(given/0.1)
    return 0.8*given + 0.2*used*cleared, dict(volume=volume, site=site, rate=rate, given=given,
                                            used=used, cleared=cleared, sharps=parts)


def lock_progress(env):
    w, spec = env.world, env.spec
    fl, task = w.fluid, spec["task"]
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
    if task == "flush-lock":
        work = clip(lk.peak_gauge/1e3) if lk.blocked else F
        parts["tried" if lk.blocked else "delivered"] = work
        progress = work*(0.5 + 0.5*K)
    else:
        sp = "medication" if task == "slow-push" else "contrast"
        D = clip(led.volume("vein", sp)/H.ML/(0.95*spec["dose_ml"]))
        if task == "slow-push":
            full = 1.5*15*spec["dose_ml"]/spec["dose_s"]
            R = grade(led.rate_window("vein", sp, 15.0)/H.ML, full, 2*full)
            span = led.span("vein", sp, 0.05, 0.95)
            timing = clip((span[1] - span[0])/(0.8*0.9*spec["dose_s"])) if span else 0.0
            delivered = D*R*timing
            parts.update(rate=R, time=timing)
        else:
            span = led.span("lock:in", sp, 0.05, 0.95)
            timing = grade(span[1] - span[0], 0.9*spec["bolus_s"], 2*spec["bolus_s"]) if span else 0.0
            delivered = D*timing
            parts["time"] = timing
        parts["dose"] = D
        progress = delivered*(1.0 + F + F*K)/3.0
    return progress*pressure*air, parts


def reward(env, criteria, success):
    task = env.spec["task"]
    if task == "sharps-away":
        names = list(env.spec["syringes"])
        raw, parts = sharps_progress(env, names + [f"{n}_needle" for n in names])
    elif task in ("set-dose", "draw-dose"):
        raw, parts = syringe_progress(env)
    elif task == "inject":
        raw, parts = injection_progress(env)
    elif task in ("flush-lock", "slow-push", "fast-bolus"):
        raw, parts = lock_progress(env)
    else:
        raise ValueError(f"No {EIGHT_TASK_CONTRACT} reward for {task!r}; vial-to-pad uses its own evaluator")
    failed = [k for k in COMMON if not criteria[k]["ok"]]
    house = max(T.HOUSEKEEPING_FLOOR, 1.0 - T.HOUSEKEEPING_STEP*len(failed))
    progress = clip(raw)*house
    return dict(contract=EIGHT_TASK_CONTRACT, score=1.0 if success else FAIL_SCALE*progress, progress=progress,
                score_parts=dict(parts, raw_progress=clip(raw), housekeeping=house, housekeeping_failed=failed,
                                 failure_factor=FAIL_SCALE, score_rule="1 if success else 0.9 x progress"))
