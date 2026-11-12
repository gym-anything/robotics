"""Start of which-key: the two keys swapped onto each other's stands, the key that opens the padlock on the far (+Y)
stand and the other on the near (-Y) one; the padlock locked with its shackle closed.

setup(config, contract=None) receives the backend's Config and the selection's contract and returns the task's
environment at its start (task_set/adapter.py's task_env: lockkey-tasks-v2's TaskLockKeyEnvV2 under that contract,
the released TaskLockKeyEnv without one). placements(table_height) gives both keys' poses for either contract
(presets.placements, presets_v2.placements); stands(moved) makes those poses the keys' stands, where the task's
evaluator expects each key back (presets.apply_start, presets_v2.apply_start)."""
import numpy as np


def placements(table_height=.75):
    return {name: np.array([.16, y, table_height+.055, 0., 0., 0., 1.])
            for name, y in (("key", .16), ("wrong_key", -.16))}


def stands(moved):
    return {name: pose.copy() for name, pose in moved.items()}


def setup(config, contract=None):
    from lockkey.task_set.adapter import task_env
    return task_env(config, "which-key", contract)
