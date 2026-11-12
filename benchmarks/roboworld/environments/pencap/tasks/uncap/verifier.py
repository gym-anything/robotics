"""Outcome of uncap: the writing tip out of the cap (at least 10 mm from its mouth), the pen and the cap lying on the
bench or on the notebook and released by both hands, and both still within 2 mm over the trailing 0.5 s.

verify(e, env) gives one sample of the episode, every physics tick, as a dict (task_id, contract, success, score,
progress, ...): the evaluator's grade (task_scoring.TaskEvaluator, or task_scoring_v2's under pencap-tasks-v2), which
measures the scene and calls predicates(...) (the success predicates now and the parts the rest test tracks) and
progress(...) (released contract) or progress_v2(...) (pencap-tasks-v2), the continuous progress, with the measurements
as keyword arguments; with NOTEBOOK_ON_BENCH a part lying on the notebook lies on the bench."""
from pencap.task_scoring import clip
from pencap.task_scoring_v2 import uncap_terms

NOTEBOOK_ON_BENCH = True


def verify(e, env):
    return e.grade(env)


def beyond(value, initial):
    """Share of the way from the start's value to 1 (no credit for the start)."""
    return clip((value-initial)/(1.-initial)) if initial < 1.-1e-9 else 0.


def predicates(*, snap, pen_down, cap_down, released, **_):
    valid = (not snap["engaged"] and snap["tip_to_mouth_m"] >= .010 and pen_down and cap_down
             and released["pen"] and released["cap"])
    tracked = ("pen", "cap")
    return valid, tracked


def progress(*, c, smooth, initial, pen_quality, cap_quality, release, still, gain, **_):
    # Withdrawal through either open end or sideways, independent of which part moves.
    start = initial["cap"]
    withdrawn = max(clip((start["depth"]-smooth["depth"])/(start["depth"]+.010)),
                    clip((smooth["radial"]-start["radial"])/(c.bore_radius+.010)),
                    clip((smooth["depth"]-start["depth"])/(c.cap_length+.010-start["depth"])))
    # Normalize to the actual stored (float32 in Newton) seated start.
    # Its micrometre-scale rounding must not award idle uncapping credit.
    unsnapped = gain(smooth["axial"]*smooth["coverage"], start["axial"]*start["coverage"])
    down = .5*(pen_quality*release["pen"]+cap_quality*release["cap"])*still
    progress = .4*unsnapped+.4*withdrawn+.2*withdrawn*down
    return progress


def progress_v2(*, c, snap, smooth, initial, pen_quality, cap_quality, release, still, **_):
    # Independent of which part moves or which hand pulls.
    # Start-relative as well: the authored seated start gives (0, 0), so this only
    # matters if a start were ever already partly off (it then earns nothing).
    unsnapped, withdrawn = (beyond(x, x0) for x, x0 in zip(uncap_terms(c, snap, smooth), initial["uncap"]))
    down = .5*(pen_quality*release["pen"]+cap_quality*release["cap"])*still
    progress = .4*unsnapped+.4*withdrawn+.2*withdrawn*down
    return progress
