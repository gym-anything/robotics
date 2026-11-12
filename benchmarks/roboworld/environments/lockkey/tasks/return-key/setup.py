"""Start of return-key: the padlock's key lying flat on the work mat, its stand on the near (-Y) side empty; the other
key on its stand and the padlock locked.

setup(config, contract=None) receives the backend's Config and the selection's contract and returns the task's
environment at its start (task_set/adapter.py's task_env: lockkey-tasks-v2's TaskLockKeyEnvV2 under that contract,
the released TaskLockKeyEnv without one). placements(table_height) is the key's pose on the mat for lockkey-tasks-v1
(presets.placements), placements_v2(table_height) for lockkey-tasks-v2 (presets_v2.placements); lockkey-tasks-v2 then
settles it (SETTLED) and checks that the key stayed on its spot on the mat (check_settled, settle_v2.settle)."""
import numpy as np
from scipy.spatial.transform import Rotation

SETTLED = True
# lockkey-tasks-v1 (placements; MAT_TOP for both):
# Entire flat key, including its 11 mm-radius bow, stays inside the work mat.
# At x=.24 the bow would extend beyond x=.27.
MAT_KEY_XY_V1 = (.21, -.20)
MAT_TOP = .0016
FLAT_BOW_HALF_THICKNESS = .0025
# lockkey-tasks-v2 (placements_v2, check_settled):
# Entire flat key, including its 11 mm-radius bow, stays inside the work mat. At x=.21 (v1) the settle
# showed the bow's contact with the thin mat box turning into deep spurious contacts near the mat's x=.27
# edge: the key was pushed +x and slid off the mat within 0.5 s. At x=.15 (bow 71 mm from that edge, blade
# tip 97 mm from the chest) it stays within 0.5 mm of its place over 3 s.
MAT_KEY_XY = (.15, -.20)
# Resting pose of the flat key, from its actual collision shapes (the 2.0 mm blade prisms and the 5.0 mm
# bow/neck hull; centre of mass between them): lying flat (+90 deg about x) the key rests on its bow only and
# tips onto its blade tip. Placed level, the tip fell 1.5 mm, struck the mat at about 5 rad/s and launched
# the key 0.36 m across the bench within 0.2 s. Pitched 2.735 deg tip-down about world y with the lowest
# points 1.214 mm below the origin, the neck edge and the blade tip both rest on the mat. The 0.5 s settle
# (settle_v2.py) checks it.
FLAT_KEY_PITCH_DEG = 2.735
FLAT_KEY_ORIGIN_ABOVE_MAT = .001214
MAT_SPOT_M = .005


def placements(table_height=.75):
    position = [*MAT_KEY_XY_V1, table_height+MAT_TOP+FLAT_BOW_HALF_THICKNESS]
    return {"key": np.r_[position, Rotation.from_euler("x", 90., degrees=True).as_quat()]}


def placements_v2(table_height=.75):
    position = [*MAT_KEY_XY, table_height+MAT_TOP+FLAT_KEY_ORIGIN_ABOVE_MAT]
    rest = Rotation.from_euler("y", -FLAT_KEY_PITCH_DEG, degrees=True)*Rotation.from_euler("x", 90., degrees=True)
    return {"key": np.r_[position, rest.as_quat()]}


def check_settled(env, key, lateral, problems):
    on_mat = (abs(key[0]-MAT_KEY_XY[0]) < MAT_SPOT_M and abs(key[1]-MAT_KEY_XY[1]) < MAT_SPOT_M
              and key[2] < env.config.table_height+.01)
    if not on_mat:
        problems.append(f"key left its place on the mat: {np.round(key[:3], 4).tolist()}")


def setup(config, contract=None):
    from lockkey.task_set.adapter import task_env
    return task_env(config, "return-key", contract)
