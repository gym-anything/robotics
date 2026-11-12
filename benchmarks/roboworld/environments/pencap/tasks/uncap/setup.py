"""Start of uncap: the yellow pen lying on the bench with the blue cap seated on its writing tip.

setup(c, poses) receives the scene config (newton15 Config) and the default poses of the pen, the cap and the notebook
(task_presets.default_poses, position and quaternion xyzw), moves the task's parts in place and returns the changes to
that scene: dict(moved=("cap",), props=())."""
from pencap.task_presets import cap_on_pen


def setup(c, poses):
    poses["cap"] = cap_on_pen(poses["pen"], c.seat_u)
    return dict(moved=("cap",), props=())
