"""Start of start-stirring: the selected variation's bench as it is, the stirrer off; nothing is moved, so the task has
no arrange(s) for task_presets.preset.

setup(config, **kwargs) receives task_session.make_session's arguments and returns the task's session
(task_session.catalogue_session)."""
from titrate.newton15.task_session import catalogue_session


def setup(config=None, **kwargs):
    return catalogue_session('start-stirring', config, **kwargs)
