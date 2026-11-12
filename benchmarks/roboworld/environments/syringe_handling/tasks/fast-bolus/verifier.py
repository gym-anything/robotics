"""Outcome of fast-bolus: at least 95 % of the 10 mL of contrast into the vein, 5 % to 95 % of it out of the syringe
within 7.2 s (0.9 times the 8 s instructed), then at least 95 % of the 5 mL saline flush into the line; the lock left
clamped with nothing attached, its connector not pulled, the line pressure within its limit, no air into the vein.

criteria(c) adds the task's own criteria to the ones every task has (c: task.Checks); verify(env) evaluates an episode
(env: syringe_handling.env.SyringeEnv, or a stand-in with its world, spec, start record and history) and returns the
evaluation dict (task.evaluate_criteria). progress(env) returns the task's progress under syringe-tasks-v1 and -v2
(scoring.reward reads it) and its parts: the dose in, graded by how fast it went (timed below), with the flush and the
clean-up, times the pressure's and the air's grades (scoring.lock_progress)."""
from syringe_handling.scoring import grade, lock_progress
from syringe_handling.task import _c, evaluate_criteria

SPECIES = "contrast"           # what the dose syringe gives (task.Checks.push, scoring.lock_progress)


def criteria(c):
    C, spec, led = c.C, c.spec, c.ledger
    sp = SPECIES
    c.push(sp)
    span = led.span("lock:in", sp, 0.05, 0.95)
    took = (span[1] - span[0]) if span else float("inf")
    C["bolus_time"] = _c(took <= 0.9*spec["bolus_s"], None if not span else round(took, 2),
                         f"5 % to 95 % of the dose out of the syringe into the line within {0.9*spec['bolus_s']:.1f} s")


def progress(env):
    return lock_progress(env, sp=SPECIES, timed=timed)


def timed(env, sp, D, parts):
    """fast-bolus's share of the dose in scoring.lock_progress, by its rate and time."""
    led, spec = env.world.fluid.ledger, env.spec
    span = led.span("lock:in", sp, 0.05, 0.95)
    timing = grade(span[1] - span[0], 0.9*spec["bolus_s"], 2*spec["bolus_s"]) if span else 0.0
    delivered = D*timing
    parts["time"] = timing
    return delivered


def verify(env):
    return evaluate_criteria(env, criteria)
