"""Outcome of pad-trowel: the pad steel-troweled smooth and hard without blisters, delamination risk or
trowel burns, still level with its edge radius kept, no water worked in and no rise of the surface's
water-cement ratio; the forms undisturbed, the tools back on the cart and the arms clear of the pad.

verify(ev) is called by task.Evaluator.evaluate with the episode's evaluator and returns its evaluation (a dict:
task, success, score, criteria, ...): ev.judge grades criteria(ev), the task's criteria ({name: criterion}), by the
episode's contract: the share of the criteria passed (task.Evaluator.judge), or under a concrete-tasks contract also
progress(ev, crit, q, gates, add, scalar, residue), which adds the task's graded quantities and defect factors
(task_set_progress.Evaluator._measure). LIMIT_S is the advisory simulated-time limit reported with the evaluation.
pad_criteria(ev, r, core, on) gives what task.Evaluator._pad adds before the forms, the tools and the arms.
"""
import numpy as np

from concrete_finishing import params as P
from concrete_finishing.task import E, DELAM_Q, blister_lift, _c
from concrete_finishing.task_set_progress import clip

LIMIT_S = 90*60              # s: three times an expert's robot time plus the waits the material needs (task.LIMITS_S)


def pad_criteria(ev, r, core, on):
    out = {}
    share = float(((r.D[core] >= E["dens_min"]) & (r.a[core] <= E["trowel_tex"])).mean())
    out["smooth_hard"] = _c(share, share >= E["tex_share"], E["tex_share"],
                            "share with densification >= 0.8 and texture <= 0.05 mm (CIP 14 'smooth', dense surface)")
    bl = int((r.bl[on] > 3e-4).sum())
    out["no_blister"] = _c(bl, bl == 0, 0, "cells raised by a blister (CIP 13)", "cells")
    dl = float((r.q[on] > DELAM_Q).mean())
    out["delamination_risk"] = _c(dl, dl < E["delam_max"], E["delam_max"],
                                  "share of the area sealed while bleed water still rose (CIP 20)")
    burn = float(((r.late[on] > 0) & (r.D[on] >= 0.98)).mean())
    out["no_trowel_burn"] = _c(burn, burn < E["burn_max"], E["burn_max"],
                               "share troweled to densification >= 0.98 after initial set (CIP 14 'trowel burns')")
    out.update(ev._flatness(r))
    e = ev._pad_edges(r, 0.90)
    out["radius_profile"] = e["radius_profile"]
    out["no_water_worked"] = ev._no_water_worked(r)
    out["surface_wc"] = ev._surface_wc(r)
    return out


def criteria(ev):
    return ev._pad(ev.env.world.regions["pad"], ev.task, pad_criteria)


def progress(ev, crit, q, gates, add, scalar, residue):
    w = ev.env.world
    r = w.regions["pad"]
    on, core = r.paste_mask, ev._core(r)
    ev._pad_tools(gates)
    ev._pad_level(r, add, scalar, gates)
    # Align early-stop water damage and all-shelf tool return with
    # pad-finish, without changing that task or any material law.
    qf = ev._committed_water(r)
    blister = blister_lift(qf, r.bl, r.cell) > 3e-4
    dl = float((qf[on] > DELAM_Q).mean())
    crit["delamination_risk"] = _c(dl, dl < E["delam_max"], E["delam_max"],
        "share trapping more than 0.01 mm bleed water by the end of bleeding")
    nb = int(blister[on].sum())
    crit["no_blister"] = _c(nb, nb == 0, 0, "cells blistered now or by committed bleed water", "cells")
    gates["water_free"] = 1.0 - float(((qf > DELAM_Q) | blister)[on].mean())
    gates["surface_free"] = 1.0 - float(((r.wet_worked > 0) | (r.dwc >= P.DWC_DUST)
        | ((r.late > 0) & (r.D >= 0.98)))[on].mean())
    # water_worked is already part of surface_free: don't count it twice.
    del gates["water_worked"]
    tex = np.clip((E["float_tex"][0] - r.a)/(E["float_tex"][0] - E["trowel_tex"]), 0, 1)
    dens = np.clip((r.D - ev._D0)/np.maximum(E["dens_min"] - ev._D0, 1e-6), 0, 1)
    raw = float(np.minimum(tex, dens)[core].mean())
    add("smooth_hard", 0.5*crit["smooth_hard"]["value"] + 0.5*raw, E["tex_share"])
    radius = crit["radius_profile"]["value"]
    add("radius_profile", radius, 0.90, keep=clip((radius - 0.45)/0.45))


def verify(ev):
    return ev.judge(criteria, progress)
