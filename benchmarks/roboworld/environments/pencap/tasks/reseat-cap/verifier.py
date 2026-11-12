"""Outcome of reseat-cap: seat-cap's success test (newton15/task.py Evaluator: every nub snapped past the bead crest and
both parts still within 10 mm over the trailing 0.5 s), with nothing fallen off the bench.

verify(e, env) gives one sample of the episode, every physics tick, as a dict (task_id, contract, success, score,
progress, ...): the evaluator's grade (task_scoring.TaskEvaluator, or task_scoring_v2's under pencap-tasks-v2), which
measures the scene and calls predicates(...) (the predicates reported now, the cap seated, and the parts the rest window
tracks), success(e, env, ...) (the success verdict) and progress(...) (the continuous progress, the same code under both
contracts) with the measurements as keyword arguments."""
from pencap.task_scoring import clip


def verify(e, env):
    return e.grade(env)


def predicates(*, snap, **_):
    valid, tracked = snap["seated"], ("pen", "cap")
    return valid, tracked


def success(e, env, *, intact, **_):
    # Same snap and 10 mm stillness test as seat-cap.
    legacy = e.legacy.sample(env)
    success = legacy["success"] and intact
    return success


def progress(*, smooth, initial, **_):
    progress = clip((smooth["quality"]-initial["cap"]["quality"])/(1.-initial["cap"]["quality"]))
    return progress
