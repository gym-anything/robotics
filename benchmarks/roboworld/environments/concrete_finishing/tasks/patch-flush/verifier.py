"""Outcome of patch-flush: the patch flush with the slab (no lip, its mean height level), filled against
the saw cut, no feather edge on the slab, a float texture, and finished within 30 min of working.

verify(ev) is called by task.Evaluator.evaluate with the episode's evaluator and returns its evaluation (a dict:
task, success, score, criteria, ...): ev.judge grades criteria(ev), the task's criteria ({name: criterion}), by the
episode's contract: the share of the criteria passed (task.Evaluator.judge), or under a concrete-tasks contract also
progress(ev, crit, q, gates, add, scalar, residue), which adds the task's graded quantities and defect factors
(task_set_progress.Evaluator._measure). LIMIT_S is the advisory simulated-time limit reported with the evaluation.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

from concrete_finishing.task import E, SMEAR_T, _c, _with_off
from concrete_finishing.task_set_progress import band

LIMIT_S = 25*60              # s: three times an expert's robot time plus the waits the material needs (task.LIMITS_S)


def _patch(ev, r):
    rec = r.paste_mask
    h, t = r.h, r.t
    inner = rec & ~np.roll(rec, 1, 0) | rec & ~np.roll(rec, -1, 0) | rec & ~np.roll(rec, 1, 1) | rec & ~np.roll(rec, -1, 1)
    outer = ~rec & (np.roll(rec, 1, 0) | np.roll(rec, -1, 0) | np.roll(rec, 1, 1) | np.roll(rec, -1, 1))
    # the lip: at every inner perimeter cell, the step to the slab beside it
    lip = float(np.abs(h[inner]).max())
    mean = float(h[rec].mean())
    filled = float((h[inner] >= -5e-4).mean())
    dist = distance_transform_edt(~rec)*r.cell
    smear = int(((dist > 0.010) & (t > SMEAR_T) & ~rec).sum())
    worked = rec & (r.worked > 0)
    tex = float(((r.a[rec] >= E["float_tex"][0]) & (r.a[rec] <= E["float_tex"][1]) & worked[rec]).mean())
    late = float(r.tau_w[worked].max()) if worked.any() else None
    return {"no_lip": _c(lip*1e3, lip*1e3 <= E["lip_mm"], E["lip_mm"], "largest step between the patch and the slab "
                                                                   "at the saw cut (ADA 1/4 in is a hard limit)", "mm"),
            "no_hard_lip": _c(lip*1e3, lip*1e3 <= E["lip_hard_mm"], E["lip_hard_mm"], "ADA 1/4 in", "mm"),
            "flush_mean": _c(mean*1e3, abs(mean)*1e3 <= E["patch_mean_mm"], E["patch_mean_mm"],
                             "mean height of the patch against the slab plane", "mm"),
            "against_the_cut": _c(filled, filled >= E["fill_share"], E["fill_share"],
                                  "share of the perimeter filled to within 0.5 mm of the slab top (Sika: force material "
                                  "against the edge)"),
            "no_feather_edge": _with_off(_c(smear*r.cell**2*1e4, smear == 0, 0.0, "mortar thicker than 0.2 mm on the slab "
                                            "beyond 10 mm of the cut (Sika: do not feather edge)", "cm2"), r,
                                         "on the slab more than 0.2 m from the cut"),
            "float_texture": _c(tex, tex >= E["tex_share"], E["tex_share"], "share of the patch floated to a 0.15-0.6 mm "
                                                                           "texture"),
            "finished_in_time": _c(None if late is None else late/1e3, late is not None and late <= 30e3, 30.0,
                                   "the stiffest the mortar was where it was last worked (Sika: finish within 30 min)", "kPa")}


def criteria(ev):
    return _patch(ev, ev.env.world.regions["patch"])


def progress(ev, crit, q, gates, add, scalar, residue):
    w = ev.env.world
    r = w.regions["patch"]
    on = r.paste_mask
    inner = on & (~np.roll(on, 1, 0) | ~np.roll(on, -1, 0)
                  | ~np.roll(on, 1, 1) | ~np.roll(on, -1, 1))
    scalar("no_lip", E["lip_hard_mm"])
    scalar("flush_mean", 2*E["patch_mean_mm"], absolute=True)
    fill = np.clip(1.0 - np.maximum(-5e-4 - r.h[inner], 0)/0.01, 0, 1)
    add("against_the_cut", 0.5*crit["against_the_cut"]["value"] + 0.5*float(fill.mean()), E["fill_share"])
    timely = (r.worked > 0) & (r.tau_w <= 30e3)
    lo, hi = E["float_tex"]
    tex = band(r.a, lo, hi, hi)*timely
    share = float(((r.a >= lo) & (r.a <= hi) & timely)[on].mean())
    add("float_texture", 0.5*share + 0.5*float(tex[on].mean()), E["tex_share"])
    # An arbitrarily late bad stroke still prevents success; it cannot earn
    # valid texture credit simply because another cell was worked in time.
    gates["timely"] = 1.0 - float(((r.worked > 0) & (r.tau_w > 30e3))[on].mean())
    mask = ~on & (distance_transform_edt(~on)*r.cell > 0.010)
    residue("no_feather_edge", r, mask, 20.0)


def verify(ev):
    return ev.judge(criteria, progress)
