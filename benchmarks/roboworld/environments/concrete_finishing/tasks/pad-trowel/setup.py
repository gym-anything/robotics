"""Start of pad-trowel: the pad struck off, darbied, edged and floated, 165 min old at 21 C, ready for
the steel trowel.

setup(seed, temp_c, air, bleed, skin_s) is called by layout.setup with the seed's site conditions
(variants.site) and returns the task's wet-material regions in their start state ({name: paste.Region}).
STATION is the task's robot start, cart, tools on the cart and props (layout.station).
"""
import math

import numpy as np

from concrete_finishing import layout as L, params as P
from concrete_finishing.layout import PX0, PX1, PY0, PY1

STATION = dict(robot=(3.20, 1.85, math.pi/2), cart=(4.35, 1.95, 0.0),
               tools=[("trowel", -0.12, 0.09, 0.0), ("edger", 0.25, 0.10, 0.0), ("penetrometer", 0.25, -0.19, 0.0)])
AGE_21C = 165*60.0     # s: the start's age at 21 C (layout.AGES_21C), scaled by the seed's temperature (masco_factor)


def worked(reg, inside, rng, X, Y, stage):
    """The work done before the start (layout.pad_region): the darby's, the edger's and the floats'."""
    L._darbied(reg, inside, rng)
    stage["darbied"] = True
    L._edged(reg, inside, X, Y)
    stage["edged"] = True
    _floated(reg, inside, rng, X, Y)
    stage["floated"] = True


def _floated(reg, inside, rng, X, Y):
    """The floats' work: the surface where it was, less the edger's track step (floated out), with a float's sandy
    texture, the paste layer brought up to its full depth and the first densification."""
    h = reg.b + reg.tb + reg.ts
    d = np.minimum.reduce([X - PX0, PX1 - X, Y - PY0, PY1 - Y])
    h = np.where(inside & (d >= 0) & (d < 0.0762), h + P.EDGER_TRACK, h)         # the track's step, floated out
    reg.a[:] = np.where(inside, 2.5e-4, 0.0)
    reg.kind[:] = np.where(inside, 3, reg.kind)
    reg.ts[:] = np.where(inside, P.T_SURF_MAX, 0.0)
    reg.D[:] = np.where(inside, 0.10, 0.0)
    reg.tb[:] = np.where(inside, np.maximum(h - reg.b - reg.ts, 0.0), reg.tb)


def setup(seed, temp_c, air, bleed, skin_s):
    regions = {}
    regions["pad"] = L.pad_region("pad-trowel", seed, temp_c, air, bleed, AGE_21C, worked)
    return regions
