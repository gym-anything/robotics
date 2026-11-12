"""Outcome of unlock-and-stow: the padlock unlocked with the near key and the shackle swung open, then the key back on
its stand and let go of by the robot; the other key on its stand.

verify(**readings) receives the task evaluator's readings of the current state as keywords (TaskProgress.update in
task_set/scoring.py for lockkey-tasks-v1 and scoring_v2.py for lockkey-tasks-v2: seated, held, key_back, other_back,
released, unlocked, opened, insert, turn, swing, back, out, stow, chain) and returns (success, progress). Progress is
the unlock chain, plus the key's withdrawal and return once unlocked. KEY_ALWAYS_NEEDED and NEEDS_UNLOCK tell the
evaluator when the progress drops to zero: the near key off the bench at any time, a latch released without it
seated."""

KEY_ALWAYS_NEEDED = True
NEEDS_UNLOCK = True


def verify(*, unlocked, opened, key_back, other_back, chain, stow, **_):
    # Returning the key before unlocking earns nothing.
    success = unlocked and opened and key_back and other_back
    progress = .6*chain + (.4*stow if unlocked else 0.)
    return success, progress
