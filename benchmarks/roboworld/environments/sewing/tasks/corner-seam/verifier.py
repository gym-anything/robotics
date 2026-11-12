"""Outcome of corner-seam (defined by sewing-tasks-v2 only): the pivot's L seam through both plies, 10 mm from the right
edge, a square corner in both plies, then 10 mm from the near edge to the left edge, with three or more reverse
stitches at both ends; the needle intact, no thread broken, the threads trimmed and the work let go.

verify(e) evaluates an episode; e is its evaluator, task_set_v2_scoring.TaskEvaluator, whose seam grading reads
seam_path, the target line (the pivot's, tasks/pivot/verifier.py), and CORNERS, which asks for the corner in both
plies; both ends are backtacked as on every seam it grades. It returns the evaluation (task, success, score,
primary_outcome, checks, info, task_contract)."""
from sewing.task import task_verifier

CORNERS = True


def verify(e):
    return e.verdict()


def seam_path(width, length, allowance=.010):
    return task_verifier("pivot").seam_path(width, length, allowance)
