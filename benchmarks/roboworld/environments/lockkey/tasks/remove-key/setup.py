"""Start of remove-key: the padlock open with its key still in it, recorded mid-cycle (task_set/start_states.json): the
near key seated and turned in the plug, the latch released and the shackle swung open; the other key on its stand.

setup(config, contract=None) receives the backend's Config and the selection's contract and returns the task's
environment at its start (task_set/adapter.py's task_env: lockkey-tasks-v2's TaskLockKeyEnvV2 under that contract,
the released TaskLockKeyEnv without one). Both place the key and the mechanism from the recorded snapshot (MID_CYCLE:
presets.apply_start, presets_v2.apply_start); lockkey-tasks-v2 then settles it with the latch released (SETTLED,
LATCH_RELEASED) and checks that the shackle is still open (check_settled, settle_v2.settle)."""
from lockkey.newton15.evaluation import OPEN_LATERAL

MID_CYCLE = True
SETTLED = True
LATCH_RELEASED = True


def check_settled(env, key, lateral, problems):
    if not lateral > OPEN_LATERAL:
        problems.append(f"shackle no longer open ({lateral*1e3:.2f} mm)")


def setup(config, contract=None):
    from lockkey.task_set.adapter import task_env
    return task_env(config, "remove-key", contract)
