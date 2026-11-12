"""Outcome of pad-edge: the edges tooled to the edger's 3/8 in radius with its lip run along the forms, no
gouge, the field beyond the edger's track untouched and no edge worked under bleed water; the forms
undisturbed, the tools back on the cart and the arms clear of the pad.

verify(ev) is called by task.Evaluator.evaluate with the episode's evaluator and returns its evaluation (a dict:
task, success, score, criteria, ...): ev.judge grades criteria(ev), the task's criteria ({name: criterion}), by the
episode's contract: the share of the criteria passed (task.Evaluator.judge), or under a concrete-tasks contract also
progress(ev, crit, q, gates, add, scalar, residue), which adds the task's graded quantities and defect factors
(task_set_progress.Evaluator._measure). LIMIT_S is the advisory simulated-time limit reported with the evaluation.
pad_criteria(ev, r, core, on) gives what task.Evaluator._pad adds before the forms, the tools and the arms.
"""
import math

import numpy as np

from concrete_finishing import layout as L, params as P
from concrete_finishing.task import E, COMPACT_MIN, LIP_W
from concrete_finishing.task_set_progress import Quantity, clip, falloff

LIMIT_S = 110*60             # s: three times an expert's robot time plus the waits the material needs (task.LIMITS_S)


def pad_criteria(ev, r, core, on):
    out = {}
    out.update(ev._pad_edges(r, E["edge_share"], with_field=True))
    return out


def criteria(ev):
    return ev._pad(ev.env.world.regions["pad"], ev.task, pad_criteria)


def progress(ev, crit, q, gates, add, scalar, residue):
    w = ev.env.world
    r = w.regions["pad"]
    on = r.paste_mask
    ev._pad_tools(gates)
    for side, value in _edge_credit(ev, r).items():
        add(f"edge_{side}", value, 1, criterion="radius_profile", group="edge")
        # Both arc and lip must pass before an edge term can be complete.
        v = q[f"edge_{side}"]
        q[f"edge_{side}"] = Quantity(v.value, v.target,
            v.passed and crit["lip_passed"]["passed"], v.keep, v.group, v.weight)
    gates["gouge"] = falloff(crit["no_gouge"]["value"], E["gouge_mm"], 6.0)
    gates["field"] = falloff(crit["field_untouched"]["value"], E["field_change_mm"], 5.0)
    edge = on & (ev.pad_d < 0.080)
    gates["edge_dry"] = 1.0 - float((r.wet_worked[edge] > 0).mean())


def _edge_credit(ev, r):
    """Per-side graded arc/lip credit against the legacy form-top reference.

    Uses the same lip exclusion, allowed shifts and 1.5 mm tolerance as _edges.
    Radius errors shrink continuously before an entire line passes.
    """
    x, y, _ = ev._pad_xyz
    nx = int(((x[:, 0] > L.PX0) & (x[:, 0] < L.PX1)).sum())
    out = {s: [] for s in "snwe"}
    rad, tol = P.TOOLS["edger"]["radius"], E["radius_tol_mm"]*1e-3
    for i, (h, d, comp, _, _) in enumerate(ev._perimeter(r)):
        side = "sn"[i % 2] if i < 2*nx else "we"[(i - 2*nx) % 2]
        if not len(h):
            continue
        error = math.inf
        for shift in (0.0, 0.5*r.cell, r.cell):
            ds = d - shift
            arc = P.FORM[1] - (rad - np.sqrt(np.maximum(rad*rad - (rad - ds)**2, 0)))
            sel = (ds >= LIP_W) & (ds < rad)
            if sel.any():
                error = min(error, float(np.abs(h[sel] - arc[sel]).max()))
        radius = clip(1.0 - max(error - tol, 0)/rad)
        lip_cells = comp[d < 0.0025]
        lip = clip(float(lip_cells.max())/COMPACT_MIN) if len(lip_cells) else 0.0
        out[side].append(0.5*radius + 0.5*radius*lip)
    return {s: float(np.mean(values)) if values else 0.0 for s, values in out.items()}


def verify(ev):
    return ev.judge(criteria, progress)
