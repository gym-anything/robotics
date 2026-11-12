"""Start of pen-on-notebook: the default scene, the yellow pen and the blue cap lying apart on the bench and the
notebook to the right.

setup(c, poses) receives the scene config (newton15 Config) and the default poses of the pen, the cap and the notebook
(task_presets.default_poses, position and quaternion xyzw), moves the task's parts in place and returns the changes to
that scene: dict(moved=(), props=()), no part moved and no prop added."""


def setup(c, poses):
    return dict(moved=(), props=())
