"""Start of plain-seam: the two 200 x 420 mm muslin plies stacked and aligned under the lowered foot, their far ends
8 mm past the needle and their right edges 10 mm right of it (the under-foot start), the part nearest the operator
draped over the table's front edge; a seed other than 0 offsets the stack up to 1 mm across the seam, 2 mm along it
and 0.5 deg, and the top ply up to 0.5 mm across the seam (variants.stack_offset).

PLIES is the task's start and ply and machine settings (task.TASK_SETUP; variants.resolve lays them and the seed's
offsets into the world's Config). setup(env, config) is called by the task adapter (task_set_env.TaskEnvironment, and
its sewing-tasks-v2 subclass) with the episode's Config before the world is built; it returns the Config the world and
its cached start state are built from, here a copy of the episode's own."""
from dataclasses import replace

PLIES = dict(start="under_foot", ply_width=0.200, ply_length=0.420, stitch_length=2.5e-3)


def setup(env, config):
    return replace(config)
