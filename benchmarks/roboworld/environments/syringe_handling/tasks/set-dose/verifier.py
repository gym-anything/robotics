"""Outcome of set-dose: the syringe left holding the 2.5 mL dose within the ISO 7886-1 tolerance and no air (at most
0.05 mL, with at least half the dose in it), and the rules every task has kept (nothing spilt outside the kidney basin,
no sheath put back on by hand, the bare needle touching nothing it may not).

criteria(c) adds the task's own criteria to the ones every task has (c: task.Checks); verify(env) evaluates an episode
(env: syringe_handling.env.SyringeEnv, or a stand-in with its world, spec, start record and history) and returns the
evaluation dict (task.evaluate_criteria). progress(env) returns the task's progress under syringe-tasks-v1 and -v2
(scoring.reward reads it) and its parts: how much nearer the dose and how much less the air than at the start, the air's
credit fading as the syringe empties below half the dose."""
from syringe_handling import task as T
from syringe_handling.scoring import clip, improved, syringe_progress
from syringe_handling.task import evaluate_criteria


def criteria(c):
    spec = c.spec
    c.syringe_dose(spec["dose_syringe"], spec["dose_ml"])


def progress(env):
    return syringe_progress(env, graded)


def graded(env, name, d, s, v, air, tol):
    """set-dose's part of scoring.syringe_progress."""
    initial = env.start["syringe_dose_ml"][name]
    dose = improved(abs(v - d), abs(initial - d), tol)
    cleared = improved(air, env.start["syringe_air_ml"][name], T.AIR_LIMIT_ML)
    # Air credit vanishes continuously before the syringe is empty; never reward discarding the dose.
    retained = clip((v - 0.5*d)/(0.5*d))
    return 0.5*(dose + cleared*retained), dict(dose=dose, air_cleared=cleared, retained=retained)


def verify(env):
    return evaluate_criteria(env, criteria)
