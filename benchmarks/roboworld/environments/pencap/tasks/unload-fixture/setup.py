"""Start of unload-fixture: the capped yellow pen standing tail down in the capping fixture, the cap seated on its
writing tip.

setup(c, poses) receives the scene config (newton15 Config) and the default poses of the pen, the cap and the notebook
(task_presets.default_poses, position and quaternion xyzw), moves the task's parts in place and returns the changes to
that scene: dict(moved=("pen", "cap"), props=("capping_fixture",)), the fixture added (task_presets.fixture_boxes)."""
from pencap.task_presets import cap_on_pen, pen_in_fixture


def setup(c, poses):
    poses["pen"] = pen_in_fixture(c)
    poses["cap"] = cap_on_pen(poses["pen"], c.seat_u)
    return dict(moved=("pen", "cap"), props=("capping_fixture",))
