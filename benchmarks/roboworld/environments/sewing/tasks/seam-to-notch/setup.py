"""Start of seam-to-notch: the plain-seam start (BASE): the two 200 x 420 mm plies stacked and aligned under the
lowered foot, far ends 8 mm past the needle, right edges 10 mm right of it, with plain-seam's seeded offsets. Its world,
mesh and cached start state are plain-seam's.

BASE is the task whose start this one reuses (task.TASK_SETUP copies its settings; task_set.BASE_TASK). setup(env,
config) is called by the task adapter (task_set_env.TaskEnvironment, and its sewing-tasks-v2 subclass) with the
episode's Config before the world is built; it returns the Config the world and its cached start state are built from:
the episode's with BASE as its task."""
from dataclasses import replace

BASE = "plain-seam"


def setup(env, config):
    return replace(config, task=BASE)
