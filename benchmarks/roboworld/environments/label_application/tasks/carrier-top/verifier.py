"""Outcome of carrier-top: the printed shipping label taken from the printer and applied to the carton's top, entirely
on one flap half, clear of the seam tape and of the top's edge bevel; wetted out to the standard over its area and along
its edges, without trapped air or a crease in the barcode, untorn, bonded only to the carton and settled; the carton
resting on the bench and nothing touching the label or the carton; the barcode decodes when the optical reader runs.

verify(e, optical) gets the episode's evaluator (task.Evaluator) and the optical reader's verdict (dict with "match", or
None) and returns the evaluation (task.Evaluator.evaluate_task: success, score, checks, metrics, progress, reward). The
evaluator reads this task's item (TARGET), placement rules (placement: task.Evaluator._place_carrier_top, which
press-down, the repair rungs and cover-old-label build on) and progress zone (zone: the carton top on the flap half
holding the label) from here; the zone holds every placement rule, so there is no registration factor."""

TARGET = "carton"


def placement(e, on_target):
    return e._place_carrier_top(on_target)


def zone(e, on_target):
    # the flap half holding the label (flap_half): either half is accepted
    return e.carton_top_zone(lambda q: e.flap_half(q, on_target))


def verify(e, optical=None):
    return e.evaluate_task(optical)
