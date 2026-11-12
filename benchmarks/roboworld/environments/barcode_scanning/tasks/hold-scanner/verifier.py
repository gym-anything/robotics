"""Outcome of hold-scanner: the scanner taken out of its cradle, held steady (under 20 mm/s and 0.25 rad/s, a finger or
hand pressing it) for 5 s with nothing read, and seated in its cradle again.

verify(e, q, rec, reads, checks, metrics) judges an episode: e is the task.Evaluator, q the bodies' poses (n x 7:
position, quaternion xyzw), rec the host app's record, reads the firmware's reads, and checks and metrics what every
task checks and measures (Evaluator.evaluate). It adds the task's own checks and returns the evaluation: success, score,
progress, contract, checks and metrics."""
from barcode_scanning.scoring import DISTURBED_COST, RETURN_SHARE, clip01

CONTRACT = "hold-scanner-v1"    # the scoring contract its evaluation reports (evaluation["contract"])
HOLD_S = 5.             # hold-scanner: the steady hold the instruction asks for
READ_FACTOR = .5        # hold-scanner: progress factor once anything was read (the task forbids reads)


def hold_progress(steady_s, returned, reads, on_floor):
    """hold-scanner: min(1, longest steady hold / HOLD_S) x (0.7 + 0.3 x returned) x (READ_FACTOR once anything was
    read), less DISTURBED_COST per free body on the floor. Putting the scanner back earns nothing without a hold."""
    hold = min(1., max(0., float(steady_s))/HOLD_S)
    p = hold*(1 - RETURN_SHARE + RETURN_SHARE*bool(returned))*(READ_FACTOR if reads else 1.)
    return clip01(p - DISTURBED_COST*on_floor)


def verify(e, q, rec, reads, checks, metrics):
    returned = e._returned(checks, metrics)
    hist, on_floor = getattr(e.env, "history", {}) or {}, metrics["on_floor"]
    steady = float(hist.get("max_steady_out_s", 0.))
    checks["no_reads"] = len(reads) == 0
    metrics["steady_hold_s"] = round(steady, 3)
    checks["held_steady_5s"] = steady >= HOLD_S
    return e._done(checks, metrics, hold_progress(steady, returned, len(reads), len(on_floor)))
