"""Start of reseat-cap: the yellow pen lying on the bench with the blue cap only partly on its writing tip, its mouth
RESEAT_DEPTH from the tip instead of seated.

setup(c, poses) receives the scene config (newton15 Config) and the default poses of the pen, the cap and the notebook
(task_presets.default_poses, position and quaternion xyzw), moves the task's parts in place and returns the changes to
that scene: dict(moved=("cap",), props=())."""
from pencap.task_presets import cap_on_pen

RESEAT_DEPTH = .034


def setup(c, poses):
    poses["cap"] = cap_on_pen(poses["pen"], RESEAT_DEPTH)
    return dict(moved=("cap",), props=())
