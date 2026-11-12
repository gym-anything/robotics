"""Outcome of position-torch: the torch at the start of the near-side seam ready to weld, the wire tip on the root
at a 45 deg work angle, a 5-10 deg push angle and 19 mm contact tip to work, held still, with no arc and the gun's
hardware touching nothing.

verify(e, settle=False) receives the episode's task.Evaluator (e.env the WeldingEnv; settle, the on-demand
evaluation's flag, changes nothing here) and returns the evaluation: success, score, checks, progress and reward. Its
checks and progress are the evaluator's own (Evaluator._eval_position_torch, _progress_position_torch), which
hold-and-present shares. WELD: the task is not judged on the weld's sections."""

WELD = False


def verify(e, settle=False):
    return e.evaluate_checks(e._eval_position_torch, e._progress_position_torch)
