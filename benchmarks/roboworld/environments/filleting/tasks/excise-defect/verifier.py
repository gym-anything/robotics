"""Outcome of excise-defect: the dark (melanin) spot cut out of the salmon side, at most twice its mass of good flesh
taken with it and no skin gone from the side.

verify(e) evaluates an episode and returns its gar.evaluation.v1 result; e is the episode's evaluator: task.Evaluator
for the released checks (released_checks), or the task-set evaluator of filleting-tasks-v1 or -v2
(task_set_evaluator.py, task_set_v2_evaluator.py), whose cutting grading (_cut_task) reads PRODUCT, released_checks,
product and progress below."""
import numpy as np

from filleting.task import NOTHING_MADE, TIME_LIMIT_S, check
from filleting.task_set_progress import clip, ramp

PRODUCT = "separation"


def verify(e):
    if e.contract is None:
        return e.evaluate_checks(released_checks)
    return e.evaluate_graded()


def released_checks(e):
    """Cut away the melanin spot: >= 99 % of the spot removed from the side, good flesh removed <= 2 x the spot's
    mass, no skin removed from the side (no skin hole)."""
    env = e.env
    lab, frags, main = e._parts()
    rs = e._region_id("melanin_spot")
    spot = sum(f["by_reg"][rs] for f in frags.values())
    spot_off = sum(f["by_reg"][rs] for c, f in frags.items() if c != main)
    good_off = sum(f["mass"] - f["by_reg"][rs] for c, f in frags.items() if c != main)
    skin_all = sum(f["skin_m2"] for f in frags.values())
    skin_off = skin_all - frags[main]["skin_m2"] if main is not None else 0.
    checks = dict([check("spot_removed", spot_off/max(spot, 1e-12), .99, "fraction of the spot's mass", spot_off >= .99*spot),
                   check("good_flesh_removed", good_off*1e3, 2*spot*1e3, "g (2 x the spot)",
                         spot_off > 0 and good_off <= 2*spot, "" if spot_off > 0 else NOTHING_MADE),
                   check("skin_hole_mm2", skin_off*1e6, 0., "mm2 of skin gone from the side",
                         spot_off > 0 and skin_off*1e6 < 1., "" if spot_off > 0 else NOTHING_MADE),
                   check("time", env.sim_time, TIME_LIMIT_S["B3"], "s", env.sim_time <= TIME_LIMIT_S["B3"])])
    dm = e._common_cut(checks, frags)
    return checks, dict(**dm, spot_kg=spot, spot_removed_kg=spot_off, good_removed_kg=good_off)


def product(e, ref, reg):
    """The separation's terminals (source, target): the spot's tets, and the other tets more than two mesh lengths
    from them."""
    source = np.flatnonzero(reg == e._region_id("melanin_spot"))
    candidates = np.flatnonzero(reg != e._region_id("melanin_spot"))
    return source, e._targets(source, candidates)


def progress(e, checks, metrics, edges):
    """(raw progress, remaining connections): the spot's separation, then the good flesh and skin taken with it."""
    total = metrics["spot_kg"]
    sever, remaining = e._sever(clip(metrics["spot_removed_kg"]/max(total, 1e-12)), edges)
    raw = sever*ramp(metrics["good_removed_kg"]/max(total, 1e-12), 2., 6.)*ramp(checks["skin_hole_mm2"]["value"], 1., 400.)
    if total <= 0:
        raw = 0.
    checks["spot_exists"] = check("spot_exists", total, 0., "kg; strictly positive", total > 0)[1]
    return raw, remaining
