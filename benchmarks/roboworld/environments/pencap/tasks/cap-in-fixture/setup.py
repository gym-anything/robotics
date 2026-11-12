"""Start of cap-in-fixture: the yellow pen standing tail down in the capping fixture, the blue cap lying on the bench.

setup(c, poses) receives the scene config (newton15 Config) and the default poses of the pen, the cap and the notebook
(task_presets.default_poses, position and quaternion xyzw), moves the task's parts in place and returns the changes to
that scene: dict(moved=("pen",), props=("capping_fixture",)), the fixture added (task_presets.fixture_boxes)."""
from pencap.task_presets import pen_in_fixture


def setup(c, poses):
    poses["pen"] = pen_in_fixture(c)
    return dict(moved=("pen",), props=("capping_fixture",))
