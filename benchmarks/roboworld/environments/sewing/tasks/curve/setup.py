"""Start of curve: two 200 x 300 mm muslin plies with their near right corner rounded to a 100 mm radius, stacked and
aligned under the lowered foot, their far ends 8 mm past the needle and their right edges 10 mm right of it (the
under-foot start). A seed other than 0 draws the radius in 80-120 mm (variants.resolve), offsets the stack up to 1 mm
across the seam, 2 mm along it and 0.5 deg, and the top ply up to 0.5 mm across the seam (variants.stack_offset).

PLIES is the task's start and ply and machine settings (task.TASK_SETUP; variants.resolve lays them and the seed's
draws into the world's Config). setup(env, config) is called by the task adapter (task_set_env.TaskEnvironment) with
the episode's Config before the world is built; it returns the Config the world and its cached start state are built
from, here a copy of the episode's own."""
from dataclasses import replace

PLIES = dict(start="under_foot", ply_width=0.200, ply_length=0.300, stitch_length=2.5e-3, corner_radius=0.100)


def setup(env, config):
    return replace(config)
