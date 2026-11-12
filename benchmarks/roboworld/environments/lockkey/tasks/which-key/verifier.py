"""Outcome of which-key: the padlock unlocked with the key that opens it (the latch released with that key seated) and
the shackle swung open; the other key on its stand.

verify(**readings) receives the task evaluator's readings of the current state as keywords (TaskProgress.update in
task_set/scoring.py for lockkey-tasks-v1 and scoring_v2.py for lockkey-tasks-v2: seated, held, key_back, other_back,
released, unlocked, opened, insert, turn, swing, back, out, stow, chain) and returns (success, progress). Progress is
the whole unlock chain. KEY_ALWAYS_NEEDED and NEEDS_UNLOCK tell the evaluator when the progress drops to zero: the
opening key off the bench before the unlock, a latch released without it seated."""

KEY_ALWAYS_NEEDED = False
NEEDS_UNLOCK = True


def verify(*, opened, other_back, chain, **_):
    return opened and other_back, chain
