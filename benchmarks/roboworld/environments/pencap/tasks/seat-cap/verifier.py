"""Outcome of seat-cap: the blue cap snapped fully onto the yellow pen's writing tip (every nub past the bead crest) and
both parts still within 10 mm over the trailing 0.5 s, held or set down.

verify(e, env) gives one sample of the episode, every physics tick: the released evaluator's (newton15/task.py
Evaluator, e.legacy of task_scoring.SeatCapEvaluator e) result unchanged, with the task id and contract added."""
from pencap.tasks import CONTRACTS, DEFAULT_TASK


def verify(e, env):
    return dict(e.legacy.sample(env), task_id="pencap/seat-cap", contract=CONTRACTS[DEFAULT_TASK])
