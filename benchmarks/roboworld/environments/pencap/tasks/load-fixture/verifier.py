"""Outcome of load-fixture: the yellow pen standing tail down in the capping fixture's bore (at least 40 mm in, within
10 degrees of vertical), released, and still within 2 mm over the trailing 0.5 s.

verify(e, env) gives one sample of the episode, every physics tick, as a dict (task_id, contract, success, score,
progress, ...): the evaluator's grade (task_scoring.TaskEvaluator, or task_scoring_v2's under pencap-tasks-v2), which
measures the scene and calls predicates(...) (the success predicates now and the parts the rest test tracks) and
progress(...) (released contract) or progress_v2(...) (pencap-tasks-v2), the continuous progress, with the measurements
as keyword arguments."""
from pencap.task_scoring import clip
from pencap.task_scoring_v2 import standing


def verify(e, env):
    return e.grade(env)


def predicates(*, fixture, released, **_):
    valid, tracked = fixture["inside"] and released["pen"], ("pen",)
    return valid, tracked


def progress(*, tail_distance, fixture, initial, release, still, gain, **_):
    approach = gain(tail_distance, initial["tail_distance"])
    upright = clip((fixture["upright"]-initial["upright"])/max(1.-initial["upright"], 1e-12))
    progress = (.25*approach+.25*upright+.5*fixture["insertion"])*(.9+.1*release["pen"]*still)
    return progress


def progress_v2(*, pen, tail_distance, fixture, initial, release, still, gain, **_):
    approach = gain(tail_distance, initial["tail_distance"])
    # A lying pen settles about 1 degree tail-down (probe: 0.0039 idle credit
    # with the old clip(-axis_z) term); only standing it up counts.
    upright = clip((standing(pen["axis"])-initial["upright"])/max(1.-initial["upright"], 1e-12))
    progress = (.25*approach+.25*upright+.5*fixture["insertion"])*(.9+.1*release["pen"]*still)
    return progress
