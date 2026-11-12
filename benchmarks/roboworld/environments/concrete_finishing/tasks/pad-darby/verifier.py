"""Outcome of pad-darby: the pad darbied level (the 1.2 m straightedge and the forms' plane), the stone
down and the surface left open, with no water worked in; the forms undisturbed, the tools back on the
cart and the arms clear of the pad.

verify(ev) is called by task.Evaluator.evaluate with the episode's evaluator and returns its evaluation (a dict:
task, success, score, criteria, ...): ev.judge grades criteria(ev), the task's criteria ({name: criterion}), by the
episode's contract: the share of the criteria passed (task.Evaluator.judge), or under a concrete-tasks contract also
progress(ev, crit, q, gates, add, scalar, residue), which adds the task's graded quantities and defect factors
(task_set_progress.Evaluator._measure). LIMIT_S is the advisory simulated-time limit reported with the evaluation.
pad_criteria(ev, r, core, on) gives what task.Evaluator._pad adds before the forms, the tools and the arms.
"""
import numpy as np

from concrete_finishing.task import E, _c
from concrete_finishing.task_set_progress import falloff

LIMIT_S = 25*60              # s: three times an expert's robot time plus the waits the material needs (task.LIMITS_S)


def pad_criteria(ev, r, core, on):
    out = {}
    out.update(ev._flatness(r))
    share = float((r.E[core] < E["agg_max"]).mean())
    out["stone_down"] = _c(share, share >= E["agg_share"], E["agg_share"],
                           "share of the area with coarse aggregate exposure below 2 % (CIP 14 'embeds large aggregate')")
    out["not_sealed"] = _c(float(r.S[on].max()), float(r.S[on].max()) < E["seal_max"], E["seal_max"],
                           "largest sealing: a darby leaves the surface open (CIP 14)")
    out["no_water_worked"] = ev._no_water_worked(r)
    return out


def criteria(ev):
    return ev._pad(ev.env.world.regions["pad"], ev.task, pad_criteria)


def progress(ev, crit, q, gates, add, scalar, residue):
    w = ev.env.world
    r = w.regions["pad"]
    on, core = r.paste_mask, ev._core(r)
    ev._pad_tools(gates)
    ev._pad_level(r, add, scalar, gates)
    # Exposure itself gives feedback before a cell reaches the 2% limit.
    raw = np.clip((0.30 - r.E[core])/(0.30 - E["agg_max"]), 0, 1)
    share = crit["stone_down"]["value"]
    add("stone_down", 0.5*share + 0.5*float(raw.mean()), E["agg_share"])
    gates["open_surface"] = falloff(float(r.S[on].max()), E["seal_max"], 0.6)


def verify(ev):
    return ev.judge(criteria, progress)
