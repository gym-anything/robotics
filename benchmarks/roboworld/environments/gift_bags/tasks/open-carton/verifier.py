"""Outcome of open-carton: the seam tape peeled so the top flaps' seam is open, all four top flaps folded open and
staying open through the hands-off window, the carton upright where it stood, its contents inside and in place, and
no robot contact through the window.

verify(e) gets the episode's evaluator (task.Evaluator, or task_set_evaluator / task_set_v2_evaluator.Evaluator for a
selection that names a contract) and returns its outcome: dict(success, score, checks, metrics), plus contract under
the task-set evaluators. It grades through e.evaluate_task: the released evaluation (legacy selections) adds released's
checks; the gift-bags-tasks-v1 and -v2 contracts take measure's checks and metrics and score progress."""


def released(e, checks, metrics, rows):
    e._eval_carton(checks, metrics, rows)


def measure(e, c, m, rows):
    e._eval_carton(c, m, rows)
    e._carton_geometry(c, m)


def progress(e, S, m):
    from gift_bags.carton import FLAPS
    return S.carton(m, e.start_metrics, FLAPS, taped=True)


def verify(e):
    return e.evaluate_task(released, progress_of=progress)
