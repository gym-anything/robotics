"""Seeded task layouts: which glassware stands where on the bar, what it holds, and what the task asks.

A layout is a pure function of (preset, task, seed). Positions are on the bar top (z = params.BAR_TOP_Z_M); the robot
pedestal is at y = -0.18 m, guests beyond y = 0.58 m. Delivery spots follow the Toni-style lit pick-up positions
(params.DELIVERY_SPOTS). Seeds vary the spot, fill levels, wine, and +-15 mm / +-20 deg placement jitter.
"""

import functools
import math
import numpy as np
from gym_anything_robotics import task_folders

from . import params as P

PACKAGE = "wine_pouring"
BAR = P.BAR_TOP_Z_M
COASTER_BASE = 0.0015            # coaster sheet thickness under the bottle (geometry.coaster base)
WINES = ("red_wine", "white_wine", "rose_wine")


def _rng(cfg):
    import hashlib
    h = hashlib.sha256(f"{cfg.preset}/{cfg.task}/{cfg.seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _jit(rng, xy, d=0.015):
    return (float(xy[0] + rng.uniform(-d, d)), float(xy[1] + rng.uniform(-d, d)))


def _bottle(rng, xy, volume, liquid="red_wine", color=None, name="bottle"):
    color = color or ("glass_flint" if liquid != "red_wine" else "glass_green")
    return [dict(name="coaster", kind="coaster", xy=xy, role="coaster"),
            dict(name=name, kind="bottle_leonor", xy=xy, z=BAR + COASTER_BASE + 2e-4, yaw=float(rng.uniform(0, 2*math.pi)),
                 color=color, liquid=liquid, volume=float(volume), role="source")]


@functools.lru_cache(maxsize=None)
def task_setup(task):
    """The benchmark task's setup.py (tasks/<task>/), or None for a task outside the benchmark."""
    return task_folders.setup(PACKAGE, task) if task_folders.exists(PACKAGE, task) else None


@functools.lru_cache(maxsize=None)
def task_verifier(task):
    """The benchmark task's verifier.py (tasks/<task>/), or None for a task outside the benchmark."""
    return task_folders.verifier(PACKAGE, task) if task_folders.exists(PACKAGE, task) else None


class Start:
    """What a benchmark task's setup.py builds its start on (task_layout calls its setup(s)).

    cfg        the selection (preset, task, seed)
    rng        the generator seeded by (preset, task, seed), after the preset's own draws
    items      the list the task appends its glassware to (dict(name, kind, xy, role, ...))
    spots      the delivery spots on the bar (params.DELIVERY_SPOTS)
    bottle_xy  service preset: the bottle's place on the bar, drawn for every service task whether it has a bottle or
               not; None in the other presets

    setup(s) returns the task's goal (layout["goal"]): what the instruction names and the evaluator judges."""

    def __init__(self, cfg, rng, items, spots, bottle_xy=None):
        self.cfg, self.rng, self.items, self.spots, self.bottle_xy = cfg, rng, items, spots, bottle_xy


def task_layout(cfg):
    rng = _rng(cfg)
    t = cfg.task
    items, goal = [], {}
    spots = P.DELIVERY_SPOTS
    folder = task_setup(t)
    if cfg.preset == "service":
        bottle_xy = _jit(rng, (0.02, 0.15))
        if folder is not None:              # every service task is a benchmark task: tasks/<task>/setup.py
            goal = folder.setup(Start(cfg, rng, items, spots, bottle_xy))
    elif cfg.preset == "tasting":
        wine = WINES[int(rng.integers(0, 3))]
        xs = (-0.16, 0.0, 0.16)
        if t == "tasting-flight":
            items += _bottle(rng, _jit(rng, (0.30, 0.14)), float(rng.uniform(600e-6, 750e-6)), liquid=wine)
            for i, x in enumerate(xs):
                items.append(dict(name=f"iso_{i + 1}", kind="glass_iso", xy=_jit(rng, (x, 0.40), 0.01), role="target"))
            goal = dict(targets=[f"iso_{i + 1}" for i in range(3)], volume=50e-6, tol=8e-6)
        elif t == "decant-carafe":
            items += _bottle(rng, _jit(rng, (0.25, 0.14)), 750e-6, liquid="red_wine")
            items.append(dict(name="carafe", kind="carafe_1l", xy=_jit(rng, (-0.10, 0.20)), role="target"))
            goal = dict(target="carafe", leave_max=30e-6)
        elif folder is not None:            # dump-and-pour: tasks/dump-and-pour/setup.py
            goal = folder.setup(Start(cfg, rng, items, spots))
    elif cfg.preset == "lined":
        if folder is not None:              # lined-measure: tasks/lined-measure/setup.py
            goal = folder.setup(Start(cfg, rng, items, spots))
    if not items:
        raise ValueError(f"No layout for {cfg.preset}/{t}")
    tgt = next((i for i in items if i["role"] in ("target", "carried")), items[0])
    return dict(items=items, goal=goal, focus=(tgt["xy"][0], tgt["xy"][1], BAR + 0.10))
