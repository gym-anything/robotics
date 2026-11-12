"""Start of swing-open: the padlock just unlocked with its key, recorded mid-cycle (task_set/start_states.json): the
near key seated and turned in the plug, the latch released and the shackle risen over its bore; the other key on its
stand.

setup(config, contract=None) receives the backend's Config and the selection's contract and returns the task's
environment at its start (task_set/adapter.py's task_env: lockkey-tasks-v2's TaskLockKeyEnvV2 under that contract,
the released TaskLockKeyEnv without one). Both place the key and the mechanism from the recorded snapshot (MID_CYCLE:
presets.apply_start, presets_v2.apply_start); lockkey-tasks-v2 then settles it with the latch released (SETTLED,
LATCH_RELEASED) and checks that the shackle is still over its bore (check_settled, settle_v2.settle)."""
from lockkey.newton15.evaluation import OPEN_LATERAL

MID_CYCLE = True
SETTLED = True
LATCH_RELEASED = True


def check_settled(env, key, lateral, problems):
    if not lateral < OPEN_LATERAL/2:
        problems.append(f"shackle not over its bore ({lateral*1e3:.2f} mm)")


def setup(config, contract=None):
    from lockkey.task_set.adapter import task_env
    return task_env(config, "swing-open", contract)
