"""Outcome of finish-securing: every lifted area of the partly applied label secured: every area that started lifted
wetted out to the standard over its area and along its own label edge, as are every edge region and every 25 mm edge
window of the label and the label as a whole; the label on the carton's top, entirely on one flap half, clear of the
seam tape and of the top's edge bevel, without trapped air, without a barcode crease at any time in the episode, untorn,
bonded only to the carton and settled; the carton resting on the bench and nothing touching the label or the carton; the
barcode decodes when the optical reader runs.

verify(e, optical) gets the episode's evaluator (task.Evaluator) and the optical reader's verdict (dict with "match", or
None) and returns the evaluation (task.Evaluator.evaluate_task: success, score, checks, metrics, progress, reward; the
repair rungs' checks and progress are task.Evaluator's REPAIR_TASKS rules). The evaluator reads this task's item
(TARGET), placement rules (placement: task.Evaluator._place_repair_rung, carrier-top's rules) and progress zone (zone:
the carton top on the flap half holding the label) from here."""

TARGET = "carton"


def placement(e, on_target):
    return e._place_repair_rung(on_target)


def zone(e, on_target):
    # the flap half holding the label (flap_half): either half is accepted
    return e.carton_top_zone(lambda q: e.flap_half(q, on_target))


def verify(e, optical=None):
    return e.evaluate_task(optical)
