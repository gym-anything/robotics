"""Outcome of swing-open: the shackle swung open, its short leg clear of its bore, after the padlock was unlocked with
its key, and the other key on its stand.

verify(**readings) receives the task evaluator's readings of the current state as keywords (TaskProgress.update in
task_set/scoring.py for lockkey-tasks-v1 and scoring_v2.py for lockkey-tasks-v2: seated, held, key_back, other_back,
released, unlocked, opened, insert, turn, swing, back, out, stow, chain) and returns (success, progress). Progress is
the shackle's swing once the padlock is unlocked with its key. KEY_ALWAYS_NEEDED and NEEDS_UNLOCK tell the evaluator
when the progress drops to zero: the near key off the bench before the unlock, a latch released without the near key
seated."""

KEY_ALWAYS_NEEDED = False
NEEDS_UNLOCK = True


def verify(*, opened, other_back, unlocked, swing, **_):
    return opened and other_back, swing if unlocked else 0.
