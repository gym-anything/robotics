"""Outcome of load-and-seam: the plies laid from the table under the foot and joined by a plain seam 10 mm from their
right edges along their whole length, three or more reverse stitches at both ends; the needle intact, no thread
broken, the threads trimmed and the work let go.

verify(e) evaluates an episode; e is its evaluator: task_set_scoring.TaskEvaluator under sewing-tasks-v1 or
task_set_v2_scoring.TaskEvaluator under sewing-tasks-v2, which grade the seam along their straight full-length line
with both ends backtacked and pass its progress through blend(e, progress), which keeps credit for loading the plies
(the evaluator's load_progress) while the seam is short. It returns the evaluation (task, success, score,
primary_outcome, checks, info, task_contract)."""


def verify(e):
    return e.verdict(blend=blend)


def blend(e, progress):
    # Smoothly retain loading credit as a real joined span forms. A
    # first stitch never unlocks a flat 0.2 bonus; sequence is immaterial.
    c = e._info["progress"]["coverage"]
    return max(.2*e.load_progress(), progress + .2*(1 - progress)*min(1., c/.05))
