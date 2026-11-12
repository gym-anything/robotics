"""Outcome of tool-sealant: the sealant tooled concave with at least 6 mm left, wetted against both
joint walls, with no void, no smear beyond the joint and every stroke made before the skin formed.

verify(ev) is called by task.Evaluator.evaluate with the episode's evaluator and returns its evaluation (a dict:
task, success, score, criteria, ...): ev.judge grades criteria(ev), the task's criteria ({name: criterion}), by the
episode's contract: the share of the criteria passed (task.Evaluator.judge), or under a concrete-tasks contract also
progress(ev, crit, q, gates, add, scalar, residue), which adds the task's graded quantities and defect factors
(task_set_progress.Evaluator._measure). LIMIT_S is the advisory simulated-time limit reported with the evaluation.
prepare(ev), called when the evaluator is built, reads the task's geometry from the start state.
"""
import numpy as np

from concrete_finishing import params as P
from concrete_finishing.task import E, SMEAR_T, _c, _with_off
from concrete_finishing.task_set_progress import band, clip

LIMIT_S = 30*60              # s: three times an expert's robot time plus the waits the material needs (task.LIMITS_S)


def prepare(ev):
    w = ev.env.world
    if "joint" in w.regions:
        r = w.regions["joint"]
        X, Y, Z = ev._joint_xyz
        ev.sj_seg = r.paste_mask & (Z >= P.CJ_SEGMENT[0]) & (Z <= P.CJ_SEGMENT[1])
        ev.sj_across = Y - P.CJ_Y


def _sealant(ev, r):
    seg = ev.sj_seg
    a = ev.sj_across
    centre = seg & (np.abs(a) < 0.002)
    h, t = r.h, r.t
    lo, hi = E["sealant_centre_mm"]
    ok = (-h[centre] >= lo*1e-3) & (-h[centre] <= hi*1e-3) & (t[centre] >= E["sealant_min_mm"]*1e-3)
    conc = float(ok.mean())
    rows_l = (r.contact_l > 0) & seg & r.edge_l
    rows_r = (r.contact_r > 0) & seg & r.edge_r
    both = float(min((rows_l.sum(axis=0) > 0)[seg.any(axis=0)].mean(), (rows_r.sum(axis=0) > 0)[seg.any(axis=0)].mean()))
    void = int((seg & (t < 1e-3)).sum())
    X, Y, Z = ev._joint_xyz
    inseg = (Z >= P.CJ_SEGMENT[0]) & (Z <= P.CJ_SEGMENT[1])
    smear = float((~r.paste_mask & inseg & (np.abs(a) > P.CJ_W/2 + 0.001) & (t > SMEAR_T)).sum())*r.cell**2*1e4 \
        / (P.CJ_SEGMENT[1] - P.CJ_SEGMENT[0])
    worked = seg & (r.worked > 0)
    work_age = np.maximum(r.seal_age, r.seal_t-r.t0)
    before = bool(worked.any()) and float(work_age[worked].max()) < r.material.skin and not r.skin_torn[seg].any()
    return {"concave_and_thick": _c(conc, conc >= 0.95, 0.95, "share of the centre line 0.5-2.5 mm below the wall faces "
                                                               "with at least 6 mm of sealant (Sikaflex-1A minimum)"),
            "wall_contact": _c(both, both >= E["contact_share"], E["contact_share"], "share of the length wetted against "
                                                                                    "both joint walls ('full contact')"),
            "no_void": _c(void, void == 0, 0, "joint cells with less than 1 mm of sealant", "cells"),
            "smear": _with_off(_c(smear, smear < E["smear_cm2_m"], E["smear_cm2_m"], "sealant beyond 1 mm of the joint "
                                  "edges per metre", "cm2/m"), r, "on the wall beyond the joint's modelled strip"),
            "tooled_before_skin": _c(bool(before), before, True, "every tooled cell worked before the skin formed")}


def criteria(ev):
    return _sealant(ev, ev.env.world.regions["joint"])


def progress(ev, crit, q, gates, add, scalar, residue):
    w = ev.env.world
    r = w.regions["joint"]
    seg, a = ev.sj_seg, ev.sj_across
    centre = seg & (np.abs(a) < 0.002)
    lo, hi = E["sealant_centre_mm"]
    grade = np.minimum(band(-r.h[centre]*1e3, lo, hi, 6.0),
                       np.clip(r.t[centre]/(E["sealant_min_mm"]*1e-3), 0, 1))
    add("concave_and_thick", 0.5*crit["concave_and_thick"]["value"] + 0.5*float(grade.mean()), 0.95)
    rows = seg.any(axis=0)
    left = np.max(np.where(seg & r.edge_l, np.clip(r.contact_l, 0, 1), 0), axis=0)[rows]
    right = np.max(np.where(seg & r.edge_r, np.clip(r.contact_r, 0, 1), 0), axis=0)[rows]
    add("wall_contact", min(float(left.mean()), float(right.mean())), E["contact_share"])
    void = float(np.maximum(1e-3 - r.t[seg], 0).mean())
    add("no_void", -void, 0, keep=clip(1.0 - void/(0.02*1e-3)))
    _, _, z = ev._joint_xyz
    mask = ~r.paste_mask & (z >= P.CJ_SEGMENT[0]) & (z <= P.CJ_SEGMENT[1]) & (np.abs(a) > P.CJ_W/2 + 0.001)
    residue("smear", r, mask, 6.0, length=P.CJ_SEGMENT[1] - P.CJ_SEGMENT[0])
    worked = seg & (r.worked > 0)
    age = np.maximum(r.seal_age, r.seal_t - r.t0)
    bad = (r.skin_torn > 0) | (worked & (age >= r.material.skin))
    # Unworked cells are neutral, not a point for waiting. Damage reduces
    # progress even when a constructed bad cell is no longer marked worked.
    gates["skin_sound"] = 1.0 - float(bad[seg].mean())


def verify(ev):
    return ev.judge(criteria, progress)
