"""Start of turn-to-open: the near key already in the padlock, recorded mid-cycle before the turn
(task_set/start_states.json): the key seated in the keyway, the plug not yet turned, the padlock still locked; the
other key on its stand.

setup(config, contract=None) receives the backend's Config and the selection's contract and returns the task's
environment at its start (task_set/adapter.py's task_env: lockkey-tasks-v2's TaskLockKeyEnvV2 under that contract,
the released TaskLockKeyEnv without one). Both place the key and the mechanism from the recorded snapshot (MID_CYCLE:
presets.apply_start, presets_v2.apply_start); lockkey-tasks-v2 then settles it with the latch still engaged (SETTLED,
settle_v2.settle)."""

MID_CYCLE = True
SETTLED = True


def setup(config, contract=None):
    from lockkey.task_set.adapter import task_env
    return task_env(config, "turn-to-open", contract)
