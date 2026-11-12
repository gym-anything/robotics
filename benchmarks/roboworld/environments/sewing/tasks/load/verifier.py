"""Outcome of load: both plies under the lowered foot (more than 1 N on them), each crossed once by the needle's axis,
10 +- 1.6 mm from its right edge and 3-8 mm from its far end, their cut edges registered within 1.5 mm where they lie
stacked, and no robot reflex.

verify(e) evaluates an episode; e is its evaluator. Under the released contract (sewing-v1) that is task.Evaluator,
which grades checks(e) with PRIMARY, the checks without which the score is 0 (the plies not lying under the needle).
Under sewing-tasks-v1 and -v2 it is task_set_scoring.TaskEvaluator or task_set_v2_scoring.TaskEvaluator, which grade
outcome(e): the same checks and the evaluator's continuous load_progress; HANDLING marks a task without a seam (the
sewing-tasks-v2 failure factor). It returns the evaluation (task, success, score, primary_outcome, checks, info,
task_contract)."""
from sewing.task import TOL_LINE, _check

PRIMARY = ("ply0_single_layer", "ply1_single_layer")
HANDLING = True


def verify(e):
    return e.verdict(checks, primary=PRIMARY, outcome=outcome)


def outcome(e):
    return checks(e), e.load_progress()


def checks(e):
    env = e.env
    w = env.world
    W, L = w.config.ply_width, w.config.ply_length
    c = {}
    c["foot_down"] = _check(w.machine.presser == "down" and w.machine.foot_N > 1.0, round(w.machine.foot_N, 2))
    for k in (0, 1):
        hits = e.under_needle(k)
        c[f"ply{k}_single_layer"] = _check(len(hits) == 1, len(hits))
        if hits:
            u, v = hits[0]
            c[f"ply{k}_seam_allowance"] = _check(abs((W - u) - 0.010) <= TOL_LINE, round((W - u)*1e3, 2),
                                                 "10 +- 1.6 mm")
            c[f"ply{k}_from_end"] = _check(3e-3 <= L - v <= 8e-3, round((L - v)*1e3, 2), "3-8 mm")
    off, n_edge = e.ply_registration()
    c["edges_aligned"] = _check(n_edge > 0 and off <= 1.5e-3,
                                dict(misregistration_mm=round(off*1e3, 2), edge_points=n_edge),
                                "<= 1.5 mm in material coordinates, where the plies lie stacked")
    c["no_reflex"] = _check(not w.reflexes, len(w.reflexes))
    return c
