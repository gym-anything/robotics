"""Outcome of access: the upward M4 seated through the guard's opening by the yellow 0.60 N m driver until it clicks and
holding its load after the bit is withdrawn; the eight-task contracts also ask for the rail level within 3 mrad and
hands off the frame.

verify(env, metrics) gets the TightenEnv and its per-screw metrics (TightenEnv._evaluate) and returns the evaluation
dict: TightenEnv._evaluate_released under tighten-legacy-v1, task_evaluation.evaluate under tighten-tasks-v1 and -v2.
Those graders read required below."""
from tighten_codex.task import uses_eight
from tighten_codex.task_evaluation import evaluate


def required(config):
    return ("frame_top",)


def verify(env, metrics):
    if uses_eight(env.config):
        return evaluate(env, metrics)
    return env._evaluate_released(metrics)
