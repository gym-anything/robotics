"""Outcome of cap-in-fixture: seat-cap's success test (newton15/task.py Evaluator: every nub snapped past the bead crest
and both parts still within 10 mm over the trailing 0.5 s) with the pen standing in the capping fixture (at least
FIXTURE_DEPTH_M in) over the whole 0.5 s window, and nothing fallen off the bench.

verify(e, env) gives one sample of the episode, every physics tick, as a dict (task_id, contract, success, score,
progress, ...): the evaluator's grade (task_scoring.TaskEvaluator, or task_scoring_v2's under pencap-tasks-v2), which
measures the scene and calls predicates(...) (the success predicates now and the parts the rest window tracks),
success(e, env, ...) (the success verdict) and progress(...) (the continuous progress, the same code under both
contracts) with the measurements as keyword arguments; the fixture test requires FIXTURE_DEPTH_M of insertion."""

FIXTURE_DEPTH_M = .030


def verify(e, env):
    return e.grade(env)


def predicates(*, snap, fixture, **_):
    valid, tracked = snap["seated"], ("pen", "cap")
    valid = valid and fixture["inside"]
    return valid, tracked


def success(e, env, *, intact, **_):
    # Same snap and 10 mm stillness test as seat-cap.
    legacy = e.legacy.sample(env)
    success = legacy["success"] and intact
    success = success and all(v for _, _, v in e.recent)
    return success


def progress(*, smooth, fixture, **_):
    progress = smooth["quality"]*(.8+.2*fixture["insertion"])
    return progress
