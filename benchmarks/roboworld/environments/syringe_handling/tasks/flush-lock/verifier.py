"""Outcome of flush-lock: at least 95 % of the 5 mL of saline from the syringe into the line (blocked: the flush tried,
pressure raised at the connector, and not forced past 100 kPa); the lock left clamped with nothing attached, its
connector not pulled more than 10 mm, the line pressure within its limit, no air into the vein.

criteria(c) adds the task's own criteria to the ones every task has (c: task.Checks); verify(env) evaluates an episode
(env: syringe_handling.env.SyringeEnv, or a stand-in with its world, spec, start record and history) and returns the
evaluation dict (task.evaluate_criteria). progress(env) returns the task's progress under syringe-tasks-v1 and -v2
(scoring.reward reads it) and its parts: the flush in (blocked: the pressure tried), then the lock clamped and the
syringe off, times the pressure's and the air's grades (scoring.lock_progress with flushed below)."""
from syringe_handling import params as P
from syringe_handling.scoring import clip, lock_progress
from syringe_handling.task import BLOCKED_LIMIT_PA, ML, _c, evaluate_criteria


def criteria(c):
    C, spec, fl, led = c.C, c.spec, c.fluid, c.ledger
    lk = fl.locks["lock"]
    c.lock_after()
    if lk.blocked:
        # the line takes nothing: the result is that it was tried (pressure raised at the connector) and not forced
        C["line_not_forced"] = _c(1e3 < lk.peak_gauge <= BLOCKED_LIMIT_PA, round(lk.peak_gauge/1e3, 1),
                                  f"the flush was tried and the pressure kept under {BLOCKED_LIMIT_PA/1e3:g} kPa")
    else:
        C["flush_delivered"] = _c(sum(float(r[1][P.SPECIES.index("saline")]) for r in led.arrivals.get("lock:in", []))/ML >= 0.95*spec["flush_ml"],
                                  dict(from_syringe_ml=round(sum(float(r[1][P.SPECIES.index("saline")]) for r in led.arrivals.get("lock:in", []))/ML, 3),
                                       total_saline_at_vein_ml=round(led.volume("vein", "saline")/ML, 3)),
                                  f">= 95 % of {spec['flush_ml']:g} mL of saline from the syringe into the line (initial priming saline excluded)")


def progress(env):
    return lock_progress(env, work=flushed)


def flushed(env, F, K, parts):
    """flush-lock's work in scoring.lock_progress."""
    lk = env.world.fluid.locks["lock"]
    work = clip(lk.peak_gauge/1e3) if lk.blocked else F
    parts["tried" if lk.blocked else "delivered"] = work
    return work*(0.5 + 0.5*K)


def verify(env):
    return evaluate_criteria(env, criteria)
