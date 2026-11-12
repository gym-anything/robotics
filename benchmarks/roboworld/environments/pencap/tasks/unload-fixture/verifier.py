"""Outcome of unload-fixture: the cap still seated, the pen out of the capping fixture, the pen and the cap lying on the
bench and released by both hands, and both still within 2 mm over the trailing 0.5 s.

verify(e, env) gives one sample of the episode, every physics tick, as a dict (task_id, contract, success, score,
progress, ...): the evaluator's grade (task_scoring.TaskEvaluator, or task_scoring_v2's under pencap-tasks-v2), which
measures the scene and calls predicates(...) (the success predicates now and the parts the rest test tracks) and
progress(...) (the continuous progress, the same code under both contracts) with the measurements as keyword
arguments."""


def verify(e, env):
    return e.grade(env)


def predicates(*, snap, fixture, pen_down, cap_down, released, **_):
    valid = snap["seated"] and fixture["out"] and pen_down and cap_down and released["pen"] and released["cap"]
    tracked = ("pen", "cap")
    return valid, tracked


def progress(*, pen, smooth, fixture, initial, pen_quality, cap_quality, release, still, gain, **_):
    out = gain(fixture["remaining"], initial["remaining"])
    down = min(pen_quality, cap_quality)*min(release["pen"], release["cap"])*still
    progress = out*(.4+.3*(1.-abs(pen["axis"][2]))+.3*down)*(.5+.5*smooth["quality"])
    return progress
