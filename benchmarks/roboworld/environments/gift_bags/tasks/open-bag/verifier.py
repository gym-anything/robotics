"""Outcome of open-bag: the top tote taken from the stack, its mouth open, standing on the counter mat with its block
bottom squared, a bottom board inside it flat on the bottom, the tote not torn, and no robot contact with the job's
items through the hands-off window; the released evaluation also needs the stack under it undisturbed.

verify(e) gets the episode's evaluator (task.Evaluator, or task_set_evaluator / task_set_v2_evaluator.Evaluator for a
selection that names a contract) and returns its outcome: dict(success, score, checks, metrics), plus contract under
the task-set evaluators. It grades through e.evaluate_task: the released evaluation (legacy selections, on open-bag's
own world) adds released's checks; the gift-bags-tasks-v1 and -v2 contracts, on bag-purchase's world, take measure's
checks and metrics (bag-purchase's, without its candle checks) and score progress."""


def released(e, checks, metrics, rows):
    from gift_bags import task as T
    e._eval_bag(checks, metrics, rows)
    e._eval_tear(checks, metrics, 0, "bag", T.TEAR_TOL_MM)


def measure(e, c, m, rows):
    from gift_bags import task as T
    # Internal task is bag-purchase: reuse every geometry correction of its evaluator, including signed normals.
    e._eval_bag(c, m, rows)
    e._eval_tear(c, m, 0, "bag", T.TEAR_TOL_MM)
    e._bag_progress(c, m)
    for k in ("candle_in_bag", "candle_upright", "candle_on_board"):
        c.pop(k)


def progress(e, S, m):
    return S.bag(m, e.start_metrics, purchase=False)


def verify(e):
    return e.evaluate_task(released, progress_of=progress)
