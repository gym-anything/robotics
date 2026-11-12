"""Outcome of lined-measure: the lined glass filled at least to its etched 175 mL line and at most 10 mL over
(task.LINED_WINDOW_ML), upright on its spot, nothing spilt, no drips running down the bottle, the bottle upright back on
its coaster and never touched against the glass, the arms still and released and the wine at rest.

verify(e) receives the episode's evaluator (task.Evaluator, or task_contracts.Evaluator under the task-set contracts)
and returns its evaluation (Evaluator.evaluate_checks) with the checks the five fill tasks share (task.fill_checks)
and, under wine-tasks-v1 and -v2, their continuous progress (task_contracts.fill_progress); the volume window is the
goal's from setup.py."""
from wine_pouring.task import fill_checks
from wine_pouring.task_contracts import fill_progress


def verify(e):
    return e.evaluate_checks(fill_checks, progress=fill_progress)
