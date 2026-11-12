"""Outcome of inject: 90 to 110 % of the dose in the muscle of the injection pad (subcut: in the fat under the skin), at
most 5 % back out of the track, at most 0.3 mL in any 2 s, no air, every puncture into the printed ring within 15
degrees of square (subcut: 45) and withdrawn the way it went in; then the syringe with its needle in the sharps
container.

criteria(c) adds the task's own criteria to the ones every task has (c: task.Checks); verify(env) evaluates an episode
(env: syringe_handling.env.SyringeEnv, or a stand-in with its world, spec, start record and history) and returns the
evaluation dict (task.evaluate_criteria). progress(env) returns the task's progress under syringe-tasks-v1 and -v2
(scoring.reward reads it) and its parts: the dose given (its volume, site and rate), then the syringe's disposal in
proportion to it."""
import math

from syringe_handling import hydraulics as H, task as T
from syringe_handling.scoring import clip, grade, sharps_progress
from syringe_handling.task import evaluate_criteria


def criteria(c):
    spec = c.spec
    c.injection(spec["dose_ml"], spec["layer"])
    c.sharps([spec["dose_syringe"], f"{spec['dose_syringe']}_needle"])


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


progress = injection_progress


def verify(env):
    return evaluate_criteria(env, criteria)
