"""Outcome of plain-seam: the two plies joined by one seam 10 mm from their right edges, from within 6 mm of one end
to within 6 mm of the other, 2.5 mm stitches at 9-11 an inch, three or more reverse stitches at both ends, no tucks,
the plies' ends meeting within 3 mm; the needle intact, no thread broken, the threads trimmed and the work let go.

verify(e) evaluates an episode; e is its evaluator. Under the released contract (sewing-v1) that is task.Evaluator,
which grades checks(e) with PRIMARY, the check without which the score is 0 (three stitches through both plies); under
sewing-tasks-v1 and -v2 it is task_set_scoring.TaskEvaluator or task_set_v2_scoring.TaskEvaluator, which grade the
seam along their straight full-length line with both ends backtacked (nothing of this task's own). It returns the
evaluation (task, success, score, primary_outcome, checks, info, task_contract)."""

PRIMARY = ("stitches_formed",)


def verify(e):
    return e.verdict(checks, primary=PRIMARY)


def checks(e):
    c = e.seam_checks(0.010, 2.5e-3, backtack=True)
    c.update(e.common())
    return c
