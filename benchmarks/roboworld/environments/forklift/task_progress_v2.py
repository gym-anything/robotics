"""forklift-tasks-v2's reward adapter: forklift/task_progress.py (forklift-tasks-v1, unchanged) with the fix round's two
changes. Reads the settled world; never writes physics or start state.

Success predicates are evaluator.py's, byte-for-byte unchanged; measurements and workmanship are task_progress.py's.
- approach(): a start already at the goal (initial distance <= 0) has nothing to close, so it earns no approach credit,
  as gained() gives none for an initial quality already full. progress() is task_progress.progress() with this approach.
- TaskProgress keeps only the settled start's baseline: no terminal latch, no tick hook. Every evaluation reads the
  current world. The time limit is evaluator.py's unchanged time_limit criterion (sim_time <= TIME_LIMIT_S, simulated
  time from the settled start): success needs the end state at an evaluation at or before the limit; past it the same
  state scores 0.9 x progress x workmanship, never 1. Each evaluation reports time_limit_s and within_time_limit.
  Entering on red reads the trajectory the event latches sample every 10 ms, so backing out does not undo it; a fault
  scores 0. Commands are never cut or refused (forklift/task_set_v2_backend.py runs the base's command execution).
"""

import copy

from . import evaluator as E
from .task import TIME_LIMIT_S, task_verifier
from .task_progress import FAIL_SCALE, RULE, clip, measurements, workmanship
from .task_set_v2 import CONTRACT, TASKS


def approach(distance, initial):
    """Share of the settled start's distance closed. A start already at the goal (initial <= 0) has nothing to close,
    so it earns no approach credit, as gained() gives none for an initial quality already full."""
    if initial <= 0:
        return 0.0
    return clip(1.0 - distance / max(initial, 1e-12))


def progress(task, now, initial):
    """task_progress.progress() with this approach: the task's verifier.py's progress(now, initial, approach)."""
    return task_verifier(task).progress(now, initial, approach)


class TaskProgress:
    """One settled start's baseline, owned by the GAR backend; no other state, so checkpoints and resets need nothing.

    Every evaluation reads the current world. The time limit is evaluator.py's unchanged time_limit criterion
    (sim_time <= TIME_LIMIT_S, simulated time from the settled start): success needs the end state at an evaluation at
    or before the limit. Commands are never cut or refused (the backend's command execution is the base's)."""

    def __init__(self, world, start):
        if start["task"] not in TASKS[:-1]:
            raise ValueError(f"{CONTRACT} does not replace {start['task']}")
        self.start = copy.deepcopy(start)
        self.initial = measurements(world, start, E.evaluate(world, start))

    def reset(self):
        """Nothing to clear: the released backend's reset calls this; the adapter keeps no state beyond the baseline."""

    def entered_on_red(self, world, *, history=True):
        """Whether the truck went past an unsecured trailer's leveler: the task's verifier.py decides (load-trailer's
        entered_on_red); no other task has a red signal."""
        own = getattr(task_verifier(self.start["task"]), "entered_on_red", None)
        return own is not None and own(world, self.start, history=history)

    def evaluate(self, world):
        fault = world.fault
        red = False if fault is not None else self.entered_on_red(world)
        if fault is not None:
            out = dict(schema="gar.evaluation.v1", task=self.start["task"], success=False, criteria=[], details={},
                       fault=fault, sim_time=world.sim_time)
            p, w, factors = 0.0, 0.0, {}
        else:
            out = E.evaluate(world, self.start)
            now = measurements(world, self.start, out)
            p = 0.0 if red else clip(progress(self.start["task"], now, self.initial))
            w, factors = workmanship(world, self.start, out)
        if red:
            out["success"] = False
        limit = TIME_LIMIT_S[self.start["task"]]
        out.update(task_contract=CONTRACT, progress=p, score_rule=RULE,
                   score=1.0 if out["success"] else FAIL_SCALE * p * w, time_limit_s=limit,
                   within_time_limit=world.sim_time <= limit)
        out["details"]["reward"] = dict(contract=CONTRACT, rule=RULE, fail_scale=FAIL_SCALE, progress=p,
                                          workmanship=w, factors=factors)
        return out
