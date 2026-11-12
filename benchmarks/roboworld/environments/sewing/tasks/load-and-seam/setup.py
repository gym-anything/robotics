"""Start of load-and-seam: the load start (BASE): the two 200 x 420 mm plies stacked and aligned on the table left of
the machine, the foot down on the empty plate, with load's seeded stack offsets. Its world, mesh and cached start
state are load's.

BASE is the task whose start this one reuses (task.TASK_SETUP copies its settings; task_set.BASE_TASK). setup(env,
config) is called by the task adapter (task_set_env.TaskEnvironment, and its sewing-tasks-v2 subclass) with the
episode's Config before the world is built; it returns the Config the world and its cached start state are built from:
the episode's with BASE as its task."""
from dataclasses import replace

BASE = "load"


def setup(env, config):
    return replace(config, task=BASE)
