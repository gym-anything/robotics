"""Outcome of remove-key: the key out of the open padlock and back on its stand on the near (-Y) side, let go of by the
robot, the shackle still swung open and the other key on its stand.

verify(**readings) receives the task evaluator's readings of the current state as keywords (TaskProgress.update in
task_set/scoring.py for lockkey-tasks-v1 and scoring_v2.py for lockkey-tasks-v2: seated, held, key_back, other_back,
released, unlocked, opened, insert, turn, swing, back, out, stow, chain) and returns (success, progress). Progress is
the key's withdrawal and return, scaled by the shackle's swing, while the padlock is unlocked with its key.
KEY_ALWAYS_NEEDED and NEEDS_UNLOCK tell the evaluator when the progress drops to zero: the near key off the bench at
any time, a latch released without the near key seated. WITHDRAWAL_FROM_START: the withdrawal is measured from where
the key starts, seated, not from the seat line (both contracts)."""

KEY_ALWAYS_NEEDED = True
NEEDS_UNLOCK = True
WITHDRAWAL_FROM_START = True


def verify(*, key_back, opened, other_back, unlocked, stow, swing, **_):
    return key_back and opened and other_back, stow*swing if unlocked else 0.
