"""Outcome of lift-lid: the lid off the box's base and put down on the table, the base neither lifted nor tipped, the
photo frame still inside the base, and no robot contact through the hands-off window.

verify(e) gets the episode's evaluator (task.Evaluator, or task_set_evaluator / task_set_v2_evaluator.Evaluator for a
selection that names a contract) and returns its outcome: dict(success, score, checks, metrics), plus contract under
the task-set evaluators. It grades through e.evaluate_task: the released evaluation (legacy selections) adds released's
checks; the gift-bags-tasks-v1 and -v2 contracts take measure's checks and metrics and score progress."""


def released(e, checks, metrics, rows):
    e._eval_lid(checks, metrics)


def measure(e, c, m, rows):
    e._eval_lid(c, m)
    e._lid_geometry(m)


def progress(e, S, m):
    frame = 1. - .5*S.ramp(m["frame_outside_distance_mm"], 0., 50.)
    return S.lid_off(m, e.start_metrics, frame)


def verify(e):
    return e.evaluate_task(released, progress_of=progress)
