"""Visible pigment per texel, shared by the board verifiers: the bridge's (tasks/bridge/verifier.py) and the task
set's (task_set/verifier.py).

In each texel the pigment with the largest optical density (mean density x |absorption row| of the board colour law,
ink.py) is the colour a viewer sees; its weight rises linearly from 0 at density VISIBLE_DENSITY to 1 at DRAWN_DENSITY.
A verdict that is not a success scores FAIL_SCORE_SCALE x progress.
"""

import numpy as np

from .config import COLORS

VISIBLE_DENSITY, DRAWN_DENSITY = 0.02, 0.15
FAIL_SCORE_SCALE = 0.9

_ABSORPTION = -np.log(np.maximum(np.asarray(list(COLORS.values())), 0.012))  # ink.py rgb()
_STRENGTH = np.linalg.norm(_ABSORPTION, axis=1)


def _tiles(ink):
    """Visible colour and weight per allocated tile."""
    out = {}
    for (tx, ty), (pigment, _) in ink.tiles.items():
        h, w = min(128, ink.height - ty * 128), min(128, ink.width - tx * 128)
        density = pigment[:h, :w].mean(axis=2)
        dominant = np.argmax(density * _STRENGTH, axis=2)
        value = np.take_along_axis(density, dominant[..., None], axis=2)[..., 0]
        weight = np.clip((value - VISIBLE_DENSITY) / (DRAWN_DENSITY - VISIBLE_DENSITY), 0, 1).astype(np.float32)
        if weight.any():
            out[tx, ty] = (dominant.astype(np.int8), weight)
    return out


def _crop(tiles, crop):
    u0, v0, u1, v1 = crop
    colour = np.full((v1 - v0, u1 - u0), -1, np.int8)
    weight = np.zeros((v1 - v0, u1 - u0), np.float32)
    for (tx, ty), (dominant, value) in tiles.items():
        ya, yb = max(v0, ty * 128), min(v1, ty * 128 + value.shape[0])
        xa, xb = max(u0, tx * 128), min(u1, tx * 128 + value.shape[1])
        if ya < yb and xa < xb:
            colour[ya - v0:yb - v0, xa - u0:xb - u0] = dominant[ya - ty * 128:yb - ty * 128, xa - tx * 128:xb - tx * 128]
            weight[ya - v0:yb - v0, xa - u0:xb - u0] = value[ya - ty * 128:yb - ty * 128, xa - tx * 128:xb - tx * 128]
    return colour, weight
