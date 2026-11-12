"""Start of unlock-and-stow: the bench as the native scene builds it, nothing moved: both keys on their stands, the
padlock locked with its shackle closed.

setup(config, contract=None) receives the backend's Config and the selection's contract and returns the task's
environment at its start (task_set/adapter.py's task_env: lockkey-tasks-v2's TaskLockKeyEnvV2 under that contract,
the released TaskLockKeyEnv without one). Nothing is placed (presets.apply_start records the catalogue start)."""


def setup(config, contract=None):
    from lockkey.task_set.adapter import task_env
    return task_env(config, "unlock-and-stow", contract)
