"""Start of pad-finish: the pad just struck off, 5 min old at 21 C (as pad-darby's), with the leftover
concrete in the tub; the whole finish is the task: darby, edger, floats and steel trowel.

setup(seed, temp_c, air, bleed, skin_s) is called by layout.setup with the seed's site conditions
(variants.site) and returns the task's wet-material regions in their start state ({name: paste.Region}).
STATION is the task's robot start, cart, tools on the cart and props (layout.station).
"""
import math

from concrete_finishing import layout as L
from concrete_finishing.materials import masco_factor

STATION = dict(robot=(3.20, 1.85, math.pi/2), cart=(4.35, 1.95, 0.0),
               tools=[("darby", 0.0, -0.14, math.pi), ("mag_float", -0.20, 0.05, 0.0), ("trowel", 0.215, 0.05, 0.0),
                      ("edger", -0.30, 0.17, 0.0), ("margin_trowel", 0.30, 0.17, 0.0)], tub=True)
AGE_21C = 5*60.0       # s: the start's age at 21 C (layout.AGES_21C), scaled by the seed's temperature (masco_factor)


def setup(seed, temp_c, air, bleed, skin_s):
    regions = {}
    regions["pad"] = L.pad_region("pad-finish", seed, temp_c, air, bleed, AGE_21C)
    regions["tub"] = L.tub_region(AGE_21C*masco_factor(temp_c), temp_c, seed, bleed)
    return regions
