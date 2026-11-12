"""Outcome of insert-key: the near key seated in the padlock's keyway now and let go of by the robot, the other key on
its stand.

verify(**readings) receives the task evaluator's readings of the current state as keywords (TaskProgress.update in
task_set/scoring.py for lockkey-tasks-v1 and scoring_v2.py for lockkey-tasks-v2: seated, held, key_back, other_back,
released, unlocked, opened, insert, turn, swing, back, out, stow, chain) and returns (success, progress). Progress is
the key's insertion. KEY_ALWAYS_NEEDED and NEEDS_UNLOCK tell the evaluator when the progress drops to zero: the near
key off the bench at any time; a release of the latch does not end this task."""

KEY_ALWAYS_NEEDED = True
NEEDS_UNLOCK = False


def verify(*, seated, held, other_back, insert, **_):
    # Read the seat NOW; opening and subsequently removing the key is not insertion.
    return seated and not held and other_back, insert
