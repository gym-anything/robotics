"""Outcome of support: both M4 screws seated by the yellow 0.60 N m driver until it clicks and holding their load after
the bit is withdrawn, with the rail returned to its reference: level and square within 3 mrad and within 0.5 mm of its
slot reference; the eight-task contracts judge the rail hands-off as well and grade its slot offset to zero at 1.5 mm.

verify(env, metrics) gets the TightenEnv and its per-screw metrics (TightenEnv._evaluate) and returns the evaluation
dict: TightenEnv._evaluate_released under tighten-legacy-v1, task_evaluation.evaluate under tighten-tasks-v1 and -v2.
Those graders read the criteria below."""
from tighten_codex.task import uses_eight
from tighten_codex.task_evaluation import evaluate

# The rail's slot offset: in reference within the band, graded down to no credit at the zero (m).
RAIL_SLOT_M = (.0005, .0015)


def required(config):
    return ("frame_front", "frame_top")


def verify(env, metrics):
    if uses_eight(env.config):
        return evaluate(env, metrics)
    return env._evaluate_released(metrics)
