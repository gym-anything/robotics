"""Start of load: the two 200 x 420 mm muslin plies stacked and aligned on the table left of the machine (the tray
start), the foot down on the empty plate, the machine idle with the needle up; a seed other than 0 moves the stack up
to 20 mm each way and turns it up to 5 deg (variants.stack_offset).

PLIES is the task's start and ply and machine settings (task.TASK_SETUP; variants.resolve lays them and the seed's
stack offset into the world's Config). setup(env, config) is called by the task adapter (task_set_env.TaskEnvironment,
and its sewing-tasks-v2 subclass) with the episode's Config before the world is built; it returns the Config the world
and its cached start state are built from, here a copy of the episode's own."""
from dataclasses import replace

PLIES = dict(start="tray", ply_width=0.200, ply_length=0.420, stitch_length=2.5e-3)


def setup(env, config):
    return replace(config)
