"""Outcome of front-m4: the forward M4 re-seated by the yellow 0.60 N m driver until it clicks and holding its load
hands-off, the other three screws still holding theirs, the rail level, the reference flag at its home with the sensor
detecting it (the spot centre on the flag), and hands off the frame. Each screw's credit is graded by that whole job.

verify(env, metrics) gets the TightenEnv and its per-screw metrics (TightenEnv._evaluate) and returns the evaluation
dict: task_evaluation.evaluate under tighten-tasks-v1 and -v2, the contracts that select front-m4. Those graders read
the criteria below."""
from tighten_codex.presets import loose_screw
from tighten_codex.task_evaluation import evaluate

DETECTION = True
SPOT_CENTRE_DETECTION = True
PRESERVE_OTHERS = True
WHOLE_JOB_GRADE = True


def required(config):
    return (loose_screw(config.task, config.seed),)


def verify(env, metrics):
    return evaluate(env, metrics)
