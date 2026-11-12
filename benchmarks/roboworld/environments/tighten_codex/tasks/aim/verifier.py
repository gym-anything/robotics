"""Outcome of aim: the W4F turned about its pivot until its spot lands on the reference flag and locked there, both M3
screws seated by the green 0.35 N m driver until it clicks and holding their load after the bit is withdrawn, the M3 bit
moment never above 0.4 N m and the sensor detecting the flag; the eight-task contracts also ask for hands off the sensor
and the flag at its home.

verify(env, metrics) gets the TightenEnv and its per-screw metrics (TightenEnv._evaluate) and returns the evaluation
dict: TightenEnv._evaluate_released under tighten-legacy-v1, task_evaluation.evaluate under tighten-tasks-v1 and -v2.
Those graders read the criteria below; the rail is not part of this job."""
from tighten_codex.presets import SCREWS
from tighten_codex.task import uses_eight
from tighten_codex.task_evaluation import evaluate

DETECTION = True
RAIL_LEVEL = False
HANDS_OFF_GROUP = "sensor"


def required(config):
    return SCREWS[:2]


def verify(env, metrics):
    if uses_eight(env.config):
        return evaluate(env, metrics)
    return env._evaluate_released(metrics)
