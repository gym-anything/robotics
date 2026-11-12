"""Start of decant-acid: the hydrochloric acid stock bottle standing on the bench at (-0.05, -0.28) m and the clean
beaker beside it at (0.12, -0.28) m.

arrange(s) places both on task_presets.Start (task_presets.preset); setup(config, **kwargs) receives
task_session.make_session's arguments and returns the task's session (task_session.catalogue_session)."""
from titrate.newton15.task_session import catalogue_session


def arrange(s):
    config, poses = s.config, s.poses
    poses['acid_bottle'] = (-.05, -.28, config.bench_height+.0002)
    poses['rinse_beaker'] = (.12, -.28, config.bench_height+.0002)


def setup(config=None, **kwargs):
    return catalogue_session('decant-acid', config, **kwargs)
