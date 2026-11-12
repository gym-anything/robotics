"""Outcome of patch-drywall: the anchor holes filled, the compound knifed flush with the wall off the
mesh and over it, the mesh covered, and no stray compound.

verify(ev) is called by task.Evaluator.evaluate with the episode's evaluator and returns its evaluation (a dict:
task, success, score, criteria, ...): ev.judge grades criteria(ev), the task's criteria ({name: criterion}), by the
episode's contract: the share of the criteria passed (task.Evaluator.judge), or under a concrete-tasks contract also
progress(ev, crit, q, gates, add, scalar, residue), which adds the task's graded quantities and defect factors
(task_set_progress.Evaluator._measure). LIMIT_S is the advisory simulated-time limit reported with the evaluation.
prepare(ev), called when the evaluator is built, reads the task's geometry from the start state.
"""
import numpy as np

from concrete_finishing import params as P
from concrete_finishing.task import E, OFF_TOL, _c, _with_off
from concrete_finishing.task_set_progress import clip, falloff

LIMIT_S = 15*60              # s: three times an expert's robot time plus the waits the material needs (task.LIMITS_S)


def prepare(ev):
    w = ev.env.world
    if "wallboard" in w.regions:
        r = w.regions["wallboard"]
        X, Y, Z = ev._wallboard_xyz
        near = np.hypot(X - P.DW_PATCH[0], Z - P.DW_PATCH[1]) < 0.15
        for hx, hz in P.DW_HOLES:
            near |= np.hypot(X - hx, Z - hz) < 0.15
        ev.dw_near = near


def _drywall(ev, r):
    h, t = r.h, r.t
    holes = r.holes
    filled = float(h[holes].min())
    comp = t > 5e-5
    mesh = r.mesh
    off = comp & ~mesh
    flush = float(np.abs(h[off]).max()) if off.any() else 0.0
    over = float(np.abs(h[comp & mesh] - P.MESH_T).max()) if (comp & mesh).any() else 0.0
    far = int((comp & ~ev.dw_near).sum())
    # Filling openings is not covering fibres. Allow 10 micrometres for
    # field/interpolation resolution around the 0.5 mm contact envelope.
    bare = float((h[mesh] < r.support_height()[mesh] - 1e-5).mean()) if mesh.any() else 0.0
    return {"holes_filled": _c(filled*1e3, filled >= -0.3e-3, -0.3, "lowest point over the anchor holes against the wall "
                                                                     "plane", "mm"),
            "flush": _c(flush*1e3, flush <= E["flush_mm"]*1e-3, E["flush_mm"], "farthest compound surface from the wall "
                                                                              "plane off the mesh (paint does not hide more)", "mm"),
            "over_mesh": _c(over*1e3, over <= E["flush_mm"]*1e-3, E["flush_mm"], "compound over the mesh against its own "
                                                                                "plane", "mm"),
            "mesh_covered": _c(bare, bare < 0.02, 0.02, "share below the fibre-top contact envelope by more than 0.01 mm"),
            "no_stray_compound": _with_off(_c(far*r.cell**2*1e4, far == 0, 0.0, "compound beyond 150 mm of the holes and "
                                              "the patch", "cm2"), r, "on the wall beyond the modelled board")}


def criteria(ev):
    return _drywall(ev, ev.env.world.regions["wallboard"])


def progress(ev, crit, q, gates, add, scalar, residue):
    w = ev.env.world
    r = w.regions["wallboard"]
    x, _, z = ev._wallboard_xyz
    for i, (hx, hz) in enumerate(P.DW_HOLES):
        hole = r.holes & ((x - hx)**2 + (z - hz)**2 < (P.DW_HOLE_D/2)**2)
        value = float(r.h[hole].min())*1e3
        add(f"hole_{i}", value, -0.3, criterion="holes_filled", group="holes")
    comp = r.t > 5e-5
    for name, mask, plane in (("flush", comp & ~r.mesh, 0), ("over_mesh", comp & r.mesh, P.MESH_T)):
        scalar(name, 2*E["flush_mm"], group=name, weight=0.5)
        errors = np.abs(r.h[mask] - plane)*1e3
        graded = float(np.clip(1 - np.maximum(errors - E["flush_mm"], 0)/3.0, 0, 1).mean()) if len(errors) else 1.0
        add(name + "_area", graded, 1, criterion=name, group=name, weight=0.5)
    deficit = np.maximum(r.support_height()[r.mesh] - 1e-5 - r.h[r.mesh], 0)
    graded = float(np.clip(1 - deficit/P.MESH_T, 0, 1).mean())
    add("mesh_covered", 0.5*(1 - crit["mesh_covered"]["value"]) + 0.5*graded, 0.98,
        keep=clip(1 - float((deficit > 0).mean())))
    # Drywall stray compound uses its 0.05 mm visibility threshold, not SMEAR_T.
    excess = float(np.maximum(r.t[~ev.dw_near] - 5e-5, 0).sum())*r.cell**2
    excess += max(float(r.off_m3) - OFF_TOL, 0)
    area_equivalent = excess/5e-5*1e4
    add("no_stray_compound", -area_equivalent, 0, keep=falloff(area_equivalent, 0, 10.0))


def verify(ev):
    return ev.judge(criteria, progress)
