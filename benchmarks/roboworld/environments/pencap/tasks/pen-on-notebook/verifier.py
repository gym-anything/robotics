"""Outcome of pen-on-notebook: the yellow pen lying on the notebook, released by both hands, and still within 2 mm over
the trailing 0.5 s.

verify(e, env) gives one sample of the episode, every physics tick, as a dict (task_id, contract, success, score,
progress, ...): the evaluator's grade (task_scoring.TaskEvaluator, or task_scoring_v2's under pencap-tasks-v2), which
measures the scene and calls predicates and progress with the measurements as keyword arguments. They are the
pen-on-notebook and pass-pen outcome, task_scoring.notebook_predicates (the success predicates now and the parts the
rest test tracks) and notebook_progress (the continuous progress)."""
from pencap.task_scoring import notebook_predicates, notebook_progress

predicates, progress = notebook_predicates, notebook_progress


def verify(e, env):
    return e.grade(env)
