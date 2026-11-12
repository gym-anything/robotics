"""Start of pass-pen: the default scene with the yellow pen moved to the right arm's side of the bench, its writing tip
at PASS_TIP_XY, beyond the left arm's reach screen (task_presets.reach_report).

setup(c, poses) receives the scene config (newton15 Config) and the default poses of the pen, the cap and the notebook
(task_presets.default_poses, position and quaternion xyzw), moves the task's parts in place and returns the changes to
that scene: dict(moved=("pen",), props=())."""

PASS_TIP_XY = (-.52, -.20)


def setup(c, poses):
    poses["pen"][:2] = PASS_TIP_XY
    return dict(moved=("pen",), props=())
