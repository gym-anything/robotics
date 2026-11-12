"""Outcome of drain-wash-compartment: the wash compartment drained to at most 0.5 L and every compartment's drain
closed, the rinse and sanitizer compartments kept as they were, the floor dry; no water running and nothing in the
hands.

The task exists only under the task-set contracts (task_set/evaluator.py, task_set_v2_evaluator.py). Its hooks there:
task_set_measure(e, data) adds the drain angles and the water each compartment has drained to the task-set measures
(data, m["task_set"]); task_set_criteria(e, m, gate) adds its gates to the evaluator's (e the evaluator, m its measures,
gate(name, ok, detail) records one); task_set_progress(ctx) gives the progress (task_set/scoring.py; ctx holds the
measures and terms that function reads for every task). verify(e, settle_s) gives the episode's verdict, the dict of
task.Evaluator.evaluate_settled (the settle branch of at most settle_s seconds, None: the default).
"""
import math

from surface_rinsing.task_set.scoring import clip


def task_set_measure(e, data):
    ws, w = e.env.water, e.env.world
    data["waste_rad"] = {n: float(w.joint_value(f["joint"])[0]) for n, f in ws.wastes.items()}
    data["drained_L"] = {n: float(ws.pools.pools[ws.pools.index[n]].drained)*1e3
                         for n in ("wash", "rinse", "sanitize")}


def task_set_criteria(e, m, gate):
    gate("drained", m["pools"]["wash"]["volume_L"] <= 0.5, m["pools"]["wash"])
    q = m["task_set"]["waste_rad"]
    gate("waste_closed", all(abs(q[n]) <= math.radians(5) for n in ("wash", "rinse", "sanitize")), q)
    changes = {n: dict(volume_change_L=m["pools"][n]["volume_L"] - e.start["pools"][n]["volume_L"],
                       drained_L=m["task_set"]["drained_L"][n] - e.start["task_set"]["drained_L"][n])
               for n in ("rinse", "sanitize")}
    gate("others_kept", all(abs(v["volume_change_L"]) <= 1.0 and v["drained_L"] <= 1.0 for v in changes.values()), changes)
    gate("floor_dry", m["contamination"]["floor_water_mL"] <= 50.0, m["contamination"])


def task_set_progress(ctx):
    start, m, passed, released, floor = ctx.start, ctx.m, ctx.passed, ctx.released, ctx.floor
    v0 = start["pools"]["wash"]["volume_L"]
    d = clip((v0 - m["pools"]["wash"]["volume_L"])/(v0 - 0.5))
    k = passed("waste_closed") * released
    # Cumulative physical drain accounts prevent a wrong-bowl drain/refill
    # from erasing collateral. Volume loss also catches scooping/overflow.
    lost = sum(max(0.0, start["pools"][n]["volume_L"] - m["pools"][n]["volume_L"],
                   m["task_set"]["drained_L"][n] - start["task_set"]["drained_L"][n])
               for n in ("rinse", "sanitize"))
    units = lost/v0 + floor
    value = (d + d*k)/2.0 - units
    terms = dict(drained=d, closed_released=k, other_water_lost_L=lost)
    return value, terms, units


def verify(e, settle_s=None):
    return e.evaluate_settled(settle_s)
