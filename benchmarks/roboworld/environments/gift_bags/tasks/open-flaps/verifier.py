"""Outcome of open-flaps: all four top flaps of the untaped carton folded open and staying open through the hands-off
window, the carton upright where it stood, its contents inside and in place, and no robot contact through the window.

verify(e) gets the episode's evaluator (task_set_evaluator or task_set_v2_evaluator.Evaluator: open-flaps is a task of
the task-set contracts, on open-carton's world) and returns its outcome: dict(success, score, checks, metrics,
contract), plus task under gift-bags-tasks-v2. It grades through e.evaluate_task with measure's checks and metrics
(open-carton's, without the seam: there is no tape) and progress."""


def measure(e, c, m, rows):
    e._eval_carton(c, m, rows)
    c.pop("seam_open")
    m["seam_tape_present"] = False
    e._carton_geometry(c, m)


def progress(e, S, m):
    from gift_bags.carton import FLAPS
    return S.carton(m, e.start_metrics, FLAPS, taped=False)


def verify(e):
    return e.evaluate_task(progress_of=progress)
