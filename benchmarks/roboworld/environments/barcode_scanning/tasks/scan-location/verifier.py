"""Outcome of scan-location: the counted slot's location label read correctly, its code the location the host app
holds, no other slot's location label read, and the scanner seated in its cradle again.

verify(e, q, rec, reads, checks, metrics) judges an episode: e is the task.Evaluator, q the bodies' poses (n x 7:
position, quaternion xyzw), rec the host app's record, reads the firmware's reads, and checks and metrics what every
task checks and measures (Evaluator.evaluate). It adds the task's own checks and returns the evaluation: success, score,
progress, contract, checks and metrics."""
from barcode_scanning.scoring import DISTURBED_COST, RETURN_SHARE, clip01

CONTRACT = "scan-location-v1"    # the scoring contract its evaluation reports (evaluation["contract"])
WRONG_LOCATION_COST = .25   # scan-location: each read of another slot's location label


def location_progress(hit, returned, wrong_reads, on_floor):
    """scan-location: hit x (0.7 + 0.3 x returned), less WRONG_LOCATION_COST per read of another slot's location label
    and DISTURBED_COST per free body on the floor; hit = the target's label read correctly and its code still the
    recorded location. Aiming earns nothing; a second read of the right label costs nothing."""
    p = bool(hit)*(1 - RETURN_SHARE + RETURN_SHARE*bool(returned))
    return clip01(p - WRONG_LOCATION_COST*wrong_reads - DISTURBED_COST*on_floor)


def verify(e, q, rec, reads, checks, metrics):
    returned = e._returned(checks, metrics)
    att = e._attributions(reads)
    target, on_floor = e.meta.get("target_slot"), metrics["on_floor"]
    checks["location_recorded"] = rec["location"] == target
    # a read of another slot's location label, or one that made another slot's code the location (a
    # second read of the right label changes nothing in the record and is not a workmanship error)
    label = f"loc_{target}"
    wrong = [a.get("label") for r, a in zip(reads, att) if r.get("sets_location") not in (None, target)
             or (a.get("kind") == "location" and a.get("label") != label)]
    metrics["wrong_location_reads"] = wrong
    checks["no_wrong_location_read"] = not wrong
    hit = checks["location_recorded"] and any(a.get("label") == label and not a.get("misread") for a in att)
    metrics["location_hit"] = bool(hit)
    return e._done(checks, metrics, location_progress(hit, returned, len(wrong), len(on_floor)))
