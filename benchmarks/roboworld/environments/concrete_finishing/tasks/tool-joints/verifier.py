"""Outcome of tool-joints: the joints tooled concave and compacted at thumbprint hardness, with no smear
on the block faces and no void.

verify(ev) is called by task.Evaluator.evaluate with the episode's evaluator and returns its evaluation (a dict:
task, success, score, criteria, ...): ev.judge grades criteria(ev), the task's criteria ({name: criterion}), by the
episode's contract: the share of the criteria passed (task.Evaluator.judge), or under a concrete-tasks contract also
progress(ev, crit, q, gates, add, scalar, residue), which adds the task's graded quantities and defect factors
(task_set_progress.Evaluator._measure). LIMIT_S is the advisory simulated-time limit reported with the evaluation.
prepare(ev), called when the evaluator is built, reads the task's geometry from the start state.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

from concrete_finishing import params as P
from concrete_finishing.task import E, COMPACT_MIN, SMEAR_T, _c, _with_off
from concrete_finishing.task_set_progress import band, clip

LIMIT_S = 60*60              # s: three times an expert's robot time plus the waits the material needs (task.LIMITS_S)


def prepare(ev):
    w = ev.env.world
    if "infill" in w.regions:
        r = w.regions["infill"]
        ev.joint_e = distance_transform_edt(r.paste_mask)*r.cell          # distance in from the block faces


def _joints(ev, r):
    on = r.paste_mask
    e = ev.joint_e
    centre = on & (e >= 0.75*P.JOINT_W/2)
    h = r.h
    depth = -h[centre]
    lo, hi = E["concave_mm"]
    conc = float(((depth >= lo*1e-3) & (depth <= hi*1e-3)).mean())
    comp = float((r.compact[centre] >= COMPACT_MIN).mean())
    tw = r.tau_w[centre]
    ripe = float(((tw >= E["mortar_window"][0]) & (tw <= E["mortar_window"][1])).mean())
    length = float(centre.sum())*r.cell/max(int(round(0.75*P.JOINT_W/r.cell)) - 1, 1)
    length = max(length, 1e-3)
    smear = float(((~on) & (r.t > SMEAR_T)).sum())*r.cell**2*1e4/length
    void = int((on & (h < -(hi*1e-3 + 1.5e-3))).sum())
    return {"concave": _c(conc, conc >= E["joint_share"], E["joint_share"], "share of the joint centres 1.4-2.8 mm deep "
                                                                           "(a 1/2 in concave jointer in a 3/8 in joint)"),
            "compacted": _c(comp, comp >= E["joint_share"], E["joint_share"], "share of joint centres with normalized surface consolidation >=0.5; "
                                                                             "confined plastic-work model, inspired by BIA tooling practice, not measured porosity"),
            "tooled_thumbprint_hard": _c(ripe, ripe >= 0.90, 0.90, "share of the joint centres last tooled at 5-20 kPa "
                                                                   "(thumbprint hard, BIA TN 7B)"),
            "smear": _with_off(_c(smear, smear < E["smear_cm2_m"], E["smear_cm2_m"], "mortar thicker than 0.2 mm on the "
                                  "block faces per metre of joint", "cm2/m"), r, "on the wall beyond the infill"),
            "no_void": _c(void, void == 0, 0, "joint cells deeper than 4.3 mm below the face", "cells")}


def criteria(ev):
    return _joints(ev, ev.env.world.regions["infill"])


def progress(ev, crit, q, gates, add, scalar, residue):
    w = ev.env.world
    r = w.regions["infill"]
    on = r.paste_mask
    centre = on & (ev.joint_e >= 0.75*P.JOINT_W/2)
    depth = -r.h[centre]*1e3
    lo, hi = E["concave_mm"]
    graded = float(band(depth, lo, hi, hi).mean())
    add("concave", 0.5*crit["concave"]["value"] + 0.5*graded, E["joint_share"], group="tooling")
    graded = float(np.clip(r.compact[centre]/COMPACT_MIN, 0, 1).mean())
    add("compacted", 0.5*crit["compacted"]["value"] + 0.5*graded, E["joint_share"], group="tooling")
    lo, hi = E["mortar_window"]
    tw = r.tau_w[centre]
    graded = float(np.minimum(np.clip(tw/lo, 0, 1), np.clip((2*hi - tw)/hi, 0, 1)).mean())
    add("tooled_thumbprint_hard", 0.5*crit["tooled_thumbprint_hard"]["value"] + 0.5*graded, 0.90, group="tooling")
    deficit = float(np.maximum(-r.h[on] - (E["concave_mm"][1]*1e-3 + 1.5e-3), 0).mean())
    add("no_void", -deficit, 0, keep=clip(1.0 - deficit/0.003))
    length = max(float(centre.sum())*r.cell/max(int(round(0.75*P.JOINT_W/r.cell)) - 1, 1), 1e-3)
    residue("smear", r, ~on, 6.0, length=length)


def verify(ev):
    return ev.judge(criteria, progress)
