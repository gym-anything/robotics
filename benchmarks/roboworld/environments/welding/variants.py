"""Preset / task / seed / variation resolution for the welding environment.

Two ways to choose the cell's initial state, never both:
  seed 0-9        legacy layouts: where the coupon sits on the fixture table within the play a fitter leaves when
                  clamping a part without locating pins (+-3 mm in x and y, +-2 deg in yaw) [authored range]; seed 0 is
                  the authored layout; split null.
  variation_id    the frozen catalogue welding.variations (welding-steel-cell-v1, 10,000 entries with train /
                  validation / test splits: coupon pose, the wire stick-out left in the gun, the draft); or a split and a
                  selection_seed to pick one. Variation 0 is the authored layout (= seed 0).
The weld process is in the part's own frame, so the coupon pose changes only where the robot must find the joint.
"""

import math

import numpy as np

from .task import PRESETS, TASKS, NEEDS_HAND

SEEDS = range(0, 10)
PRESET_SLUG = {"steel-cell": "steel-cell"}


def coupon_offset(seed):
    """(dx m, dy m, yaw deg) of the coupon for a seed (deterministic)."""
    if seed == 0:
        return 0., 0., 0.
    rng = np.random.default_rng(1000 + int(seed))
    return float(rng.uniform(-3e-3, 3e-3)), float(rng.uniform(-3e-3, 3e-3)), float(rng.uniform(-2., 2.))


def resolve(preset=None, task=None, seed=0):
    preset = preset or "steel-cell"
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; presets: {', '.join(PRESETS)}")
    task = task or "fillet-2f"
    if task not in TASKS:
        code = {v: k for k, v in TASKS.items()}
        if task in code:
            task = code[task]
        else:
            raise ValueError(f"Unknown task {task!r}; tasks: {', '.join(TASKS)}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise ValueError(f"seed must be an integer in [{SEEDS.start}, {SEEDS.stop - 1}]")
    return dict(preset=preset, task=task, seed=seed, coupon_offset=coupon_offset(seed))


TORCH_ONLY = ("ur10e", "single-fr3")          # deployments without a handling hand (welding/robot_setup.py)
# deployments whose handling hand has never been shown holding the bracket: the G1's left Dex3. No scripted G1 program
# has pressed or gripped the bracket's top edge -- the tack demonstration's reach put the left arm onto the table top
# and the robot fell, another program found no press pose for the arm with its waist where it stood, and a staged reach
# that kept the robot standing pressed with the folded thumb, not the palm, while its torso leaned on the table and the
# torch did not reach the first tack
NO_HOLD = ("g1",)
# a task's two-arm configuration not shown on a deployment: hold-and-present holds the bracket while the torch stands at
# the seam's start (x 0-10 mm). On the ur10e-fr3 the tack demonstration does it (the hand at x = 120 mm, the torch
# brought to the first tack's start at the near end). On the two-Franka rigs that order stopped both arms on an arm-arm
# contact (welding/demos/task_demos.py DEMO_MIRRORED), and their demonstrated order holds the near end (x = 30 mm) and
# tacks the far end, so the torch at the near end with the hand holding is not shown there
NOT_SHOWN = {"hold-and-present": ("dual-fr3", "dual-panda")}


def check_robot(robot, task):
    """Tasks that need the handling arm refuse the torch-only rigs, and the G1 (no shown hold), before any physics
    import; a task whose two-arm configuration is not shown on a deployment (NOT_SHOWN) refuses it."""
    if robot in NOT_SHOWN.get(task, ()):
        raise ValueError(f"task {task!r} holds the bracket while the torch is at the seam's start; on robot {robot!r} "
                         f"the two arms met in that configuration (its demonstrated order holds the near end and works "
                         f"at the far end), so the task is not offered on it; use the default 'ur10e-fr3'")
    if task in NEEDS_HAND and robot in TORCH_ONLY:
        raise ValueError(f"task {task!r} needs the FR3's hand (or another deployment's handling hand) to hold the "
                         f"bracket; robot {robot!r} (torch arm only) cannot do it; use the default 'ur10e-fr3', "
                         f"'dual-fr3' or 'dual-panda'")
    if task in NEEDS_HAND and robot in NO_HOLD:
        raise ValueError(f"task {task!r} needs a hand that holds the bracket (to place it, or while the torch tacks "
                         f"it); robot {robot!r}: "
                         f"its left Dex3 has not been shown holding the bracket (no scripted program has pressed or "
                         f"gripped its top edge without the robot falling, leaning on the table or losing the torch's "
                         f"reach), so the task is not offered on it; use the default 'ur10e-fr3', 'dual-fr3' or "
                         f"'dual-panda'")


def check_launch(robot, backend_options):
    """Launch-time check (robots.catalog ENVIRONMENTS['welding']['launch_check'], called by GAR's LaunchSpec before
    any session is created): the robot (alias, rig id, setup object or path; None = the default) against the task the
    options select (a name or its code; fillet-2f by default), then every backend option the way the backend's
    constructor checks it (gym_anything_robotics/backends/newton_welding.py check_options: option names, cameras,
    renderer options, config fields, seed / variation; no world, no GPU). Raises ValueError."""
    from .robot_setup import validate_setup, ALIASES
    options = backend_options or {}
    alias = "ur10e-fr3" if robot is None else robot if isinstance(robot, str) and robot in ALIASES \
        else validate_setup(robot)
    task = options.get("task") or "fillet-2f"
    if task not in TASKS:
        task = {v: k for k, v in TASKS.items()}.get(task, task)
    check_robot(alias, task)
    from gym_anything_robotics.backends.newton_welding import check_options
    try:
        check_options(alias, **options)
    except (RuntimeError, TypeError) as exc:           # UnsupportedConfigurationError is a RuntimeError
        raise ValueError(str(exc)) from exc


CATALOGUE_SELECTORS = ("variation_id", "split", "selection_seed")


def task_resolve(selection):
    """GAR task catalogue hook (gym_anything_robotics/tasks.py): selection -> task id, instruction, backend options.
    Legacy seeds (split null) or a catalogue variation (its frozen split); a seed and a variation are exclusive."""
    from .task import INSTRUCTIONS
    catalogue = {k: selection[k] for k in CATALOGUE_SELECTORS if k in selection}
    if catalogue:
        if "seed" in selection:
            raise ValueError("seed (legacy layouts 0-9) and the variation catalogue (variation_id / split / "
                             "selection_seed) are exclusive; variation 0 is seed 0's authored layout")
        from .variations import select, VERSION
        v = resolve(selection.get("preset"), selection.get("task"), 0)
        item = select(**catalogue)
        i = item["variation_id"]
        return dict(task_id=f"welding/v1/{PRESET_SLUG[v['preset']]}/{v['task']}/variation-{i:04d}",
                    description=INSTRUCTIONS[v["task"]],
                    backend_options=dict(preset=v["preset"], task=v["task"], variation_id=i),
                    selection=dict(preset=v["preset"], task=v["task"], variation_id=i, split=item["split"],
                                   catalogue=VERSION, sha256=item["sha256"]), split=item["split"])
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0))
    return dict(task_id=f"welding/v1/{PRESET_SLUG[v['preset']]}/{v['task']}/seed-{v['seed']}",
                description=INSTRUCTIONS[v["task"]],
                backend_options=dict(preset=v["preset"], task=v["task"], seed=v["seed"]),
                selection=dict(preset=v["preset"], task=v["task"], seed=v["seed"]), split=None)
