"""Outcome of slow-push: at least 95 % of the dose of medication into the vein, at most 1.5 times the mean rate in any
15 s and 5 % to 95 % of it over at least 0.72 times the instructed time, then at least 95 % of the 5 mL saline flush
into the line; the lock left clamped with nothing attached, its connector not pulled, the line pressure within its
limit, no air into the vein.

criteria(c) adds the task's own criteria to the ones every task has (c: task.Checks); verify(env) evaluates an episode
(env: syringe_handling.env.SyringeEnv, or a stand-in with its world, spec, start record and history) and returns the
evaluation dict (task.evaluate_criteria). progress(env) returns the task's progress under syringe-tasks-v1 and -v2
(scoring.reward reads it) and its parts: the dose in, graded by its rate and time (timed below), with the flush and the
clean-up, times the pressure's and the air's grades (scoring.lock_progress)."""
from syringe_handling import hydraulics as H
from syringe_handling.scoring import clip, grade, lock_progress
from syringe_handling.task import ML, _c, evaluate_criteria

SPECIES = "medication"           # what the dose syringe gives (task.Checks.push, scoring.lock_progress)


def criteria(c):
    C, spec, led = c.C, c.spec, c.ledger
    sp = SPECIES
    c.push(sp)
    mean15 = 15.0*spec["dose_ml"]/spec["dose_s"]
    worst = led.rate_window("vein", sp, 15.0)/ML
    C["delivery_rate"] = _c(worst <= 1.5*mean15, round(worst, 4), f"<= {1.5*mean15:.3f} mL of {sp} into the vein in any 15 s")
    span = led.span("vein", sp, 0.05, 0.95)
    took = (span[1] - span[0]) if span else 0.0
    need = 0.8*0.9*spec["dose_s"]
    C["delivery_time"] = _c(took >= need, round(took, 2), f"5 % to 95 % of the dose in not less than {need:.0f} s")


def progress(env):
    return lock_progress(env, sp=SPECIES, timed=timed)


def timed(env, sp, D, parts):
    """slow-push's share of the dose in scoring.lock_progress, by its rate and time."""
    led, spec = env.world.fluid.ledger, env.spec
    full = 1.5*15*spec["dose_ml"]/spec["dose_s"]
    R = grade(led.rate_window("vein", sp, 15.0)/H.ML, full, 2*full)
    span = led.span("vein", sp, 0.05, 0.95)
    timing = clip((span[1] - span[0])/(0.8*0.9*spec["dose_s"])) if span else 0.0
    delivered = D*R*timing
    parts.update(rate=R, time=timing)
    return delivered


def verify(env):
    return evaluate_criteria(env, criteria)
