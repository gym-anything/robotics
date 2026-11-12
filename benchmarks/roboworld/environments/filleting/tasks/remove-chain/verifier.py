"""Outcome of remove-chain: the chain (side muscle) cut off the pork tenderloin along its seam, at most 3 % of the main
muscle lost and no cut running more than 25 mm into the main piece.

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
    """Remove the chain along its seam: >= 95 % of the chain's mass separated from the main muscle, main-muscle
    loss <= 3 %, no crack running into the main piece further than 25 mm (the 'tail split', approximated by the
    longest un-separating cut inside the main piece)."""
    env = e.env
    lab, frags, main_c = e._parts()
    rm, rc = e._region_id("main"), e._region_id("chain")
    # the main piece: the fragment holding most main-muscle mass
    main_c = max(frags, key=lambda c: frags[c]["by_reg"][rm]) if frags else None
    chain = sum(f["by_reg"][rc] for f in frags.values())
    chain_off = sum(f["by_reg"][rc] for c, f in frags.items() if c != main_c)
    main_all = sum(f["by_reg"][rm] for f in frags.values())
    main_off = main_all - frags[main_c]["by_reg"][rm] if main_c is not None else 0.
    inner = e._cut_faces_between(lab, main_c, inside=True) if main_c is not None else np.zeros((0, 3))
    split = float(np.ptp(inner[:, 0])) if len(inner) else 0.
    checks = dict([check("chain_separated", chain_off/max(chain, 1e-12), .95, "fraction of the chain's mass",
                         chain_off >= .95*chain),
                   check("main_loss", main_off/max(main_all, 1e-12), .03, "fraction of the main muscle",
                         chain_off > 0 and main_off <= .03*main_all, "" if chain_off > 0 else NOTHING_MADE),
                   check("tail_split_mm", split*1e3, 25., "mm", split <= .025),
                   check("time", env.sim_time, TIME_LIMIT_S["C1"], "s", env.sim_time <= TIME_LIMIT_S["C1"])])
    dm = e._common_cut(checks, frags, [c for c, f in frags.items() if c != main_c and f["mass"] >= 1e-3])
    return checks, dict(**dm, chain_kg=chain, chain_separated_kg=chain_off, main_lost_kg=main_off)


def product(e, ref, reg):
    """The separation's terminals (source, target): the chain's tets, and the main muscle's tets more than two mesh
    lengths from them."""
    source = np.flatnonzero(reg == e._region_id("chain"))
    candidates = np.flatnonzero(reg == e._region_id("main"))
    return source, e._targets(source, candidates)


def progress(e, checks, metrics, edges):
    """(raw progress, remaining connections): the chain's separation, then the main muscle lost and the tail split."""
    sever, remaining = e._sever(clip(metrics["chain_separated_kg"]/max(metrics["chain_kg"], 1e-12)), edges)
    raw = sever*ramp(checks["main_loss"]["value"], .03, .10)*ramp(checks["tail_split_mm"]["value"], 25., 60.)
    if metrics["chain_kg"] <= 0:
        raw = 0.
    checks["chain_exists"] = check("chain_exists", metrics["chain_kg"], 0., "kg; strictly positive", metrics["chain_kg"] > 0)[1]
    return raw, remaining
