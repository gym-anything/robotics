"""Outcome of checkout-bulky: checkout-present's checks (Evaluator._checkout) over the basket's items, and the case of
water at the counter's end scanned in place with the handheld scanner, never lifted (its rise under 10 mm), the
scanner back in its cradle.

verify(e, q, rec, reads, checks, metrics) judges an episode: e is the task.Evaluator, q the bodies' poses (n x 7:
position, quaternion xyzw), rec the host app's record, reads the firmware's reads, and checks and metrics what every
task checks and measures (Evaluator.evaluate). It adds the task's own checks and returns the evaluation: success, score,
progress, contract, checks and metrics."""
from barcode_scanning.scoring import clip01

CONTRACT = "checkout-bulky-v1"    # the scoring contract its evaluation reports (evaluation["contract"])
LIFT_FREE_M = .010      # checkout-bulky: the case's highest rise with full case credit (the case_not_lifted limit) ...
LIFT_ZERO_M = .050      # ... falling linearly to no case credit at this rise


def lift_credit(rise_m):
    """checkout-bulky: 1 up to LIFT_FREE_M of the case's highest rise in the episode, falling linearly to 0 at
    LIFT_ZERO_M (read from the history, so a lift cannot be undone by setting the case back)."""
    return clip01((LIFT_ZERO_M - float(rise_m))/(LIFT_ZERO_M - LIFT_FREE_M))


def water_case(e, rung, checks, metrics, credits):
    """The case's checks for Evaluator._checkout: its highest rise in the episode (env.history) under LIFT_FREE_M, and
    its credit, lift_credit of that rise once the case was rung up (else 0), appended to the items' credits."""
    w = e.world
    hist = getattr(e.env, "history", {}) or {}
    if "water_case" in w.scene.items:
        rise = float((hist.get("max_rise_m") or {}).get(int(w.scene.items["water_case"]["body"]), 0.))
        metrics["water_case_max_rise_mm"] = round(rise*1e3, 2)
        checks["case_not_lifted"] = rise < LIFT_FREE_M
        credits.append(lift_credit(rise) if "water_case" in rung else 0.)


def verify(e, q, rec, reads, checks, metrics):
    returned = e._returned(checks, metrics)
    return e._checkout(q, rec, reads, checks, metrics, returned, case=water_case)
