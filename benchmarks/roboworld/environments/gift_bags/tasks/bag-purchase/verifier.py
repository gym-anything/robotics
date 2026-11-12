"""Outcome of bag-purchase: the top tote taken from the stack, open and squared, standing mouth up on the counter mat,
a bottom board inside it flat on the bottom, the customer's boxed candle inside, upright, resting on that board with its
print up, the tote not torn, and no robot contact with the job's items through the hands-off window.

verify(e) gets the episode's evaluator (task.Evaluator; task_set_evaluator.Evaluator under bag-purchase-v1, which hands
it the released evaluation) and returns its outcome: dict(success, score, checks, metrics), plus contract under the
task-set evaluator. It grades through e.evaluate_task with released's checks; score makes the score 1 on success, else
FAILED_SCORE_FACTOR x the continuous progress of task.Evaluator._bag_progress, and reports the mean of the checks."""

FAILED_SCORE_FACTOR = .9       # score: 1 on success, else this x progress (label_application's convention) [A]


def released(e, checks, metrics, rows):
    from gift_bags import task as T
    e._eval_bag(checks, metrics, rows)
    e._eval_tear(checks, metrics, 0, "bag", T.TEAR_TOL_MM)


def score(e, success, score, checks, metrics):
    # 1 on success, else FAILED_SCORE_FACTOR x progress: every failure below every success, every success 1.
    # The mean of the checks is also reported.
    metrics["mean_of_checks"] = score
    progress = e._bag_progress(checks, metrics)
    score = 1. if success else FAILED_SCORE_FACTOR*progress
    return score


def verify(e):
    return e.evaluate_task(released, score_of=score)
