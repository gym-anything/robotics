"""Start of drain-burette: the waste beaker standing on the bench at (0.10, -0.27) m, and the burette charged as the
variation has it.

arrange(s) places the beaker on task_presets.Start (task_presets.preset); setup(config, **kwargs) receives
task_session.make_session's arguments and returns the task's session (task_session.catalogue_session)."""
from titrate.newton15.task_session import catalogue_session


def arrange(s):
    config, poses = s.config, s.poses
    poses['rinse_beaker'] = (.10, -.27, config.bench_height+.0002)


def setup(config=None, **kwargs):
    return catalogue_session('drain-burette', config, **kwargs)
