"""Outcome of return-key: the key back on its stand on the near (-Y) side and let go of by the robot, the other key on
its stand.

verify(**readings) receives the task evaluator's readings of the current state as keywords (TaskProgress.update in
task_set/scoring.py for lockkey-tasks-v1 and scoring_v2.py for lockkey-tasks-v2: seated, held, key_back, other_back,
released, unlocked, opened, insert, turn, swing, back, out, stow, chain) and returns (success, progress). Progress is
how near and upright the key is on its stand. KEY_ALWAYS_NEEDED and NEEDS_UNLOCK tell the evaluator when the progress
drops to zero: the near key off the bench at any time; a release of the latch does not end this task.
TILT_FROM_START: lockkey-tasks-v2 measures the key's tilt toward its stand from the tilt it starts at, lying flat on
the mat (scoring_v2.TaskProgress)."""

KEY_ALWAYS_NEEDED = True
NEEDS_UNLOCK = False
TILT_FROM_START = True


def verify(*, key_back, other_back, back, **_):
    return key_back and other_back, back
