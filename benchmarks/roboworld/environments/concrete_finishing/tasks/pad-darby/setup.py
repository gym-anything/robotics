"""Start of pad-darby: the pad just struck off, 5 min old at 21 C: screed ridges, a low corner where the
board bridged, a high band where it rode on a stone, the stone up; leftover concrete in the tub beside it.

setup(seed, temp_c, air, bleed, skin_s) is called by layout.setup with the seed's site conditions
(variants.site) and returns the task's wet-material regions in their start state ({name: paste.Region}).
STATION is the task's robot start, cart, tools on the cart and props (layout.station).
"""
import math

from concrete_finishing import layout as L
from concrete_finishing.materials import masco_factor

STATION = dict(robot=(3.20, 1.85, math.pi/2), cart=(4.35, 1.95, 0.0),
               # the darby along the cart's near (south) edge, where both hands reach it from a clear stand
               # (at the far edge its handle and knob were 1.0 m ahead of the nearest clear stand)
               tools=[("darby", 0.0, -0.12, math.pi), ("mag_float", -0.18, 0.10, 0.0), ("margin_trowel", 0.25, 0.10, 0.0)],
               tub=True)
AGE_21C = 5*60.0       # s: the start's age at 21 C (layout.AGES_21C), scaled by the seed's temperature (masco_factor)


def setup(seed, temp_c, air, bleed, skin_s):
    regions = {}
    regions["pad"] = L.pad_region("pad-darby", seed, temp_c, air, bleed, AGE_21C)
    regions["tub"] = L.tub_region(AGE_21C*masco_factor(temp_c), temp_c, seed, bleed)
    return regions
