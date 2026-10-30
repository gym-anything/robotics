"""Preset / task / seed resolution for the sewing environment.

Seeds vary where the plies start, within what an operator leaves when laying cut parts down: for tasks that start
under the foot, the stack's offset (+-1 mm across the seam, +-2 mm along it, +-0.5 deg) and the top ply's misalignment
across the seam (0-0.5 mm); for the load task, the stack's place on the table's left (+-20 mm, +-5 deg) [authored
ranges]; for the curve task, the corner's radius (80-120 mm). Seed 0 is the authored layout. Seeds never
change a physical parameter or a threshold.
"""

import numpy as np

from .task import PRESETS, TASKS, TASK_SETUP

SEEDS = range(0, 10)


def stack_offset(task, seed):
    """((dx m, dy m, yaw deg), misalign m) for a seed (deterministic)."""
    if seed == 0:
        return (0., 0., 0.), 0.
    rng = np.random.default_rng(2000 + int(seed))
    if TASK_SETUP[task]["start"] == "tray":
        return (float(rng.uniform(-.02, .02)), float(rng.uniform(-.02, .02)), float(rng.uniform(-5., 5.))), 0.
    return ((float(rng.uniform(-1e-3, 1e-3)), float(rng.uniform(-2e-3, 2e-3)), float(rng.uniform(-.5, .5))),
            float(rng.uniform(0., 0.5e-3)))


def resolve(preset=None, task=None, seed=0):
    preset = preset or "sample-room"
    if preset not in PRESETS:
        raise KeyError(preset)
    task = task or "plain-seam"
    if task not in TASKS:
        code = {v: k for k, v in TASKS.items()}
        if task in code:
            task = code[task]
        else:
            raise ValueError(f"Unknown task {task!r}; tasks: {', '.join(TASKS)}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise ValueError(f"seed must be an integer in [{SEEDS.start}, {SEEDS.stop - 1}]")
    offset, misalign = stack_offset(task, seed)
    v = dict(preset=preset, task=task, seed=seed, ply_offset=offset, misalign=misalign, **TASK_SETUP[task])
    if "corner_radius" in v and seed != 0:
        # the curve's radius, 80-120 mm (seed 0 is the authored 100 mm)
        v["corner_radius"] = float(np.random.default_rng(3000 + int(seed)).uniform(0.080, 0.120))
    return v


def check_robot(robot, task):
    """Every task can be attempted with any hosted rig (one hand can guide a seam); nothing to refuse."""
    return None


def task_resolve(selection):
    """GAR task catalogue hook (gym_anything_robotics/tasks.py): selection -> task id, instruction, backend options."""
    from .task import INSTRUCTIONS
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0))
    return dict(task_id=f"sewing/v1/{v['preset']}/{v['task']}/seed-{v['seed']}",
                description=INSTRUCTIONS[v["task"]],
                backend_options=dict(preset=v["preset"], task=v["task"], seed=v["seed"]),
                selection=dict(preset=v["preset"], task=v["task"], seed=v["seed"]), split=None)
