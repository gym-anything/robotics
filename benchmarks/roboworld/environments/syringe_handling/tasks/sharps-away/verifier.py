"""Outcome of sharps-away: every syringe with its needle inside the sharps container through the flap, the flap closed
and nothing else put in; no sheath pushed back onto a needle by hand and no bare needle touching anything on the way.

criteria(c) adds the task's own criteria to the ones every task has (c: task.Checks); verify(env) evaluates an episode
(env: syringe_handling.env.SyringeEnv, or a stand-in with its world, spec, start record and history) and returns the
evaluation dict (task.evaluate_criteria). progress(env) returns the task's progress under syringe-tasks-v1 and -v2
(scoring.reward reads it) and its parts: how far each syringe and needle lies inside the container and the flap closed,
less anything else in it."""
from syringe_handling.scoring import sharps_progress
from syringe_handling.task import evaluate_criteria


def criteria(c):
    spec = c.spec
    names = list(spec["syringes"]) + [f"{n}_needle" for n in spec["syringes"]]
    c.sharps(names)


def progress(env):
    names = list(env.spec["syringes"])
    return sharps_progress(env, names + [f"{n}_needle" for n in names])


def verify(env):
    return evaluate_criteria(env, criteria)
