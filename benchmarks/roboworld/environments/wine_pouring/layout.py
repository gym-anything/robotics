"""Seeded task layouts: which glassware stands where on the bar, what it holds, and what the task asks.

A layout is a pure function of (preset, task, seed). Positions are on the bar top (z = params.BAR_TOP_Z_M); the robot
pedestal is at y = -0.18 m, guests beyond y = 0.58 m. Delivery spots follow the Toni-style lit pick-up positions
(params.DELIVERY_SPOTS). Seeds vary the spot, fill levels, wine, and +-15 mm / +-20 deg placement jitter.
"""

import math
import numpy as np

from . import params as P

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


def task_layout(cfg):
    rng = _rng(cfg)
    t = cfg.task
    items, goal = [], {}
    spots = P.DELIVERY_SPOTS
    if cfg.preset == "service":
        bottle_xy = _jit(rng, (0.02, 0.15))
        if t == "pour-glass":
            k = int(rng.integers(1, 5))
            fill = float(rng.uniform(450e-6, 750e-6))
            items += _bottle(rng, bottle_xy, fill)
            items.append(dict(name="glass", kind="glass_pure", xy=_jit(rng, spots[k]), role="target", spot=k))
            goal = dict(target="glass", volume=150e-6, tol=15e-6, spot=k)
        elif t == "taste-pour":
            k = int(rng.integers(1, 5))
            items += _bottle(rng, bottle_xy, float(rng.uniform(600e-6, 750e-6)))
            items.append(dict(name="glass", kind="glass_pure", xy=_jit(rng, spots[k]), role="target", spot=k))
            goal = dict(target="glass", volume=37.5e-6, tol=7.5e-6, spot=k)
        elif t == "split-bottle":
            k = int(rng.integers(1, 4))
            fill = float(rng.uniform(250e-6, 450e-6))
            # black glass, so the wine left cannot be seen (a green bottle shows its level from the side). Appearance only: the same LEONOR geometry, mass and contact as every bottle.
            # LEONOR geometry in black glass: Saverglass sells a black glass (Onyx), but catalogues LEONOR itself only in
            # antique green and extra-white flint
            items += _bottle(rng, bottle_xy, fill, color="glass_black")
            items.append(dict(name="glass_a", kind="glass_pure", xy=_jit(rng, spots[k]), role="target", spot=k))
            items.append(dict(name="glass_b", kind="glass_pure", xy=_jit(rng, spots[k + 1]), role="target", spot=k + 1))
            goal = dict(targets=["glass_a", "glass_b"], share_tol=0.10, leave_max=15e-6)
        elif t == "water-tumbler":
            k = int(rng.integers(1, 5))
            items.append(dict(name="carafe", kind="carafe_1l", xy=_jit(rng, (-0.10, 0.14)), liquid="water",
                              volume=float(rng.uniform(500e-6, 800e-6)), role="source"))
            items.append(dict(name="tumbler", kind="tumbler", xy=_jit(rng, spots[k]), role="target", spot=k))
            goal = dict(target="tumbler", volume=250e-6, tol=25e-6, spot=k)
        elif t == "carry-glass":
            k = int(rng.integers(1, 5))
            items.append(dict(name="glass", kind="glass_pure", xy=_jit(rng, (0.10, 0.14)), liquid="red_wine",
                              volume=150e-6, role="carried"))
            goal = dict(target="glass", deliver_spot=k, keep=150e-6)
        elif t == "refill-glass":
            k = int(rng.integers(1, 5))
            items += _bottle(rng, bottle_xy, float(rng.uniform(450e-6, 700e-6)))
            items.append(dict(name="glass", kind="glass_pure", xy=_jit(rng, spots[k]), role="target", spot=k,
                              liquid="red_wine", volume=float(rng.uniform(40e-6, 90e-6))))
            goal = dict(target="glass", volume=150e-6, tol=15e-6, spot=k)
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
        elif t == "dump-and-pour":
            items += _bottle(rng, _jit(rng, (0.30, 0.14)), float(rng.uniform(600e-6, 750e-6)), liquid="white_wine")
            items.append(dict(name="iso", kind="glass_iso", xy=_jit(rng, (0.0, 0.38), 0.01), role="target",
                              liquid="red_wine", volume=float(rng.uniform(20e-6, 60e-6))))
            items.append(dict(name="dump", kind="dump_bucket", xy=_jit(rng, (-0.32, 0.16)), role="dump"))
            goal = dict(target="iso", fresh="white_wine", old="red_wine", volume=50e-6, tol=8e-6, old_max=2e-6,
                        dump="dump")
    elif cfg.preset == "lined":
        k = int(rng.integers(1, 5))
        items += _bottle(rng, _jit(rng, (0.02, 0.15)), float(rng.uniform(500e-6, 750e-6)))
        items.append(dict(name="glass", kind="glass_lined", xy=_jit(rng, spots[k]), role="target", spot=k))
        goal = dict(target="glass", line="line_175", tol=5e-6, spot=k)
    if not items:
        raise ValueError(f"No layout for {cfg.preset}/{t}")
    tgt = next((i for i in items if i["role"] in ("target", "carried")), items[0])
    return dict(items=items, goal=goal, focus=(tgt["xy"][0], tgt["xy"][1], BAR + 0.10))
