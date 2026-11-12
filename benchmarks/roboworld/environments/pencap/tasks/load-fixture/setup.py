"""Start of load-fixture: the default scene, the yellow pen lying on the bench, with the capping fixture added.

setup(c, poses) receives the scene config (newton15 Config) and the default poses of the pen, the cap and the notebook
(task_presets.default_poses, position and quaternion xyzw), moves the task's parts in place and returns the changes to
that scene: dict(moved=(), props=("capping_fixture",)), the fixture added (task_presets.fixture_boxes)."""


def setup(c, poses):
    return dict(moved=(), props=("capping_fixture",))
