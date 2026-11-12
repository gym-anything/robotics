"""Start of corner-seam (defined by sewing-tasks-v2 only): the pivot start (BASE): two 200 x 200 mm plies stacked and
aligned under the lowered foot, with the pivot's fine band along the near edge and its seeded offsets. Its world, mesh
and cached start state are the pivot's.

BASE is the task whose start this one reuses (task_set_v2.EXTRA_TASKS and BASE_TASK). setup(env, config) is called by
the sewing-tasks-v2 task adapter (task_set_v2_env.TaskEnvironment) with the episode's Config before the world is
built; it returns the Config the world and its cached start state are built from: the episode's with BASE as its
task."""
from dataclasses import replace

BASE = "pivot"


def setup(env, config):
    return replace(config, task=BASE)
