"""Outcome of checkout-present: the sale holds every item, each item's own barcode read once (presented from the
basket to the scanner in its cradle) and no inner unit's, every item in the bagging tote and the tote upright on the
counter.

verify(e, q, rec, reads, checks, metrics) judges an episode: e is the task.Evaluator, q the bodies' poses (n x 7:
position, quaternion xyzw), rec the host app's record, reads the firmware's reads, and checks and metrics what every
task checks and measures (Evaluator.evaluate). It adds the task's own checks and returns the evaluation: success, score,
progress, contract, checks and metrics. The checks are Evaluator._checkout's, which the three checkout tasks share."""

CONTRACT = "checkout-present-v1"    # the scoring contract its evaluation reports (evaluation["contract"])


def verify(e, q, rec, reads, checks, metrics):
    returned = e._returned(checks, metrics)
    return e._checkout(q, rec, reads, checks, metrics, returned)
