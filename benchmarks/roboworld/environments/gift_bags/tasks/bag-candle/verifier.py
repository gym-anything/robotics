"""Outcome of bag-candle: the customer's boxed candle inside the open tote, upright and resting on the board in it, the
tote still open, squared and standing on the counter mat, not torn, and no robot contact with the job's items through
the hands-off window.

verify(e) gets the episode's evaluator (task_set_evaluator or task_set_v2_evaluator.Evaluator: bag-candle is a task of
the task-set contracts, on bag-purchase's world) and returns its outcome: dict(success, score, checks, metrics,
contract), plus task under gift-bags-tasks-v2. It grades through e.evaluate_task with measure's checks and metrics
(bag-purchase's) and progress."""


def measure(e, c, m, rows):
    from gift_bags import task as T
    # Internal task is bag-purchase: reuse every geometry correction of its evaluator, including signed normals.
    e._eval_bag(c, m, rows)
    e._eval_tear(c, m, 0, "bag", T.TEAR_TOL_MM)
    e._bag_progress(c, m)


def progress(e, S, m):
    return S.bag(m, e.start_metrics, purchase=True)


def verify(e):
    return e.evaluate_task(progress_of=progress)
