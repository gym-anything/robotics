"""Outcome of draw-dose: the syringe holds the 2.5 mL dose within the ISO 7886-1 tolerance and no air (at most 0.05 mL),
and the vial stands upright within 30 mm of where it was, nothing in it or on it, its headspace within 20 kPa of
ambient, having lost the dose and at most 0.1 mL more.

criteria(c) adds the task's own criteria to the ones every task has (c: task.Checks); verify(env) evaluates an episode
(env: syringe_handling.env.SyringeEnv, or a stand-in with its world, spec, start record and history) and returns the
evaluation dict (task.evaluate_criteria). progress(env) returns the task's progress under syringe-tasks-v1 and -v2
(scoring.reward reads it) and its parts: the dose drawn, the air cleared and the vial's pose, pressure and contents
kept."""
import numpy as np

from syringe_handling import hydraulics as H, params as P, task as T
from syringe_handling.scoring import clip, grade, syringe_progress
from syringe_handling.task import ML, evaluate_criteria


def criteria(c):
    spec, fl = c.spec, c.fluid
    c.syringe_dose(spec["dose_syringe"], spec["dose_ml"])
    c.vial_after(fl.syringes[spec["dose_syringe"]].chamber.liquid/ML)


def progress(env):
    return syringe_progress(env, graded)


def graded(env, name, d, s, v, air, tol):
    """draw-dose's part of scoring.syringe_progress."""
    fl = env.world.fluid
    dose = clip(min(1.0, v/(d - tol)) - max(0.0, v - d - tol)/d)
    clear = grade(air, T.AIR_LIMIT_ML, max(0.5, 0.2*d))
    vial = fl.vials["vial"]
    r = vial.reading(env.world)
    moved = float(np.linalg.norm(env.world.bq[vial.body, :2] - np.asarray(env.start["vial_xy"]["vial"])))
    # Continuous physical deviations, not a fraction of the vial's Boolean checks.
    pose = min(grade(1.0 - r["upright_cos"], 0.01, 0.2), grade(moved, 0.03, 0.15),
               grade(env.world.bq[vial.body, 2] - P.COUNTER_Z, 0.004, 0.10))
    pressure = grade(abs(vial.vessel.gas_pressure() - P.P_ATM), T.VIAL_PRESSURE_PA, 100e3)
    loss = env.start["vial_ml"]["vial"] - r["liquid_ml"] - s.chamber.liquid/H.ML
    contents = grade(loss, 0.1, d + 0.1)
    kept = pose*pressure*contents
    return dose*(0.6 + 0.2*clear + 0.2*kept), dict(dose=dose, air_clear=clear, vial_pose=pose,
                                                    vial_pressure=pressure, vial_contents=contents)


def verify(env):
    return evaluate_criteria(env, criteria)
