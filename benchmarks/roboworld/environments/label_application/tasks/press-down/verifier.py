"""Outcome of press-down: the shipping label the co-worker laid on the carton's top pressed down where it lies: still
entirely on the flap half it was laid on, on the top, clear of the seam tape and of the top's edge bevel, wetted out to
the standard over its area and along its edges, without trapped air or a crease in the barcode, untorn, bonded only to
the carton and settled; the carton resting on the bench and nothing touching the label or the carton; the barcode
decodes when the optical reader runs.

verify(e, optical) gets the episode's evaluator (task.Evaluator) and the optical reader's verdict (dict with "match", or
None) and returns the evaluation (task.Evaluator.evaluate_task: success, score, checks, metrics, progress, reward;
progress is the work done on what the laid label left to do, the START_CREDIT_TASKS rule). The evaluator reads this
task's item (TARGET), placement rules (placement) and progress zone (zone: the carton top on the flap half the label was
laid on) from here; the zone holds every placement rule, so there is no registration factor."""
import numpy as np

TARGET = "carton"


def placement(e, on_target):
    """The laid label must end where the carrier rules allow, on the flap half it was laid on ("entirely on that
    flap half")."""
    base = e._place_carrier_top(on_target)
    side = np.sign(e.world.scene.task_geometry["laid_label"]["centre"][1])
    q = e._item_local("carton")
    checks = {k: v for k, v in base["checks"].items() if k != "one_flap_half"}
    checks["on_the_flap_half_it_was_laid_on"] = bool((side*q[:, 1] > 0).all())
    return dict(checks=checks, metrics=base["metrics"])


def zone(e, on_target):
    # the flap half the label was laid on ("where it lies")
    return e.carton_top_zone(lambda q: float(np.sign(e.world.scene.task_geometry["laid_label"]["centre"][1])))


def verify(e, optical=None):
    return e.evaluate_task(optical)
