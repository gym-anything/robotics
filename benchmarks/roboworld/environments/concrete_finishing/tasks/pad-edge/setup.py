"""Start of pad-edge: the pad struck off and darbied, an hour old at 21 C, its edges not yet tooled.

setup(seed, temp_c, air, bleed, skin_s) is called by layout.setup with the seed's site conditions
(variants.site) and returns the task's wet-material regions in their start state ({name: paste.Region}).
STATION is the task's robot start, cart, tools on the cart and props (layout.station).
"""
import math

from concrete_finishing import layout as L

STATION = dict(robot=(3.20, 1.85, math.pi/2), cart=(4.35, 1.95, 0.0),
               tools=[("edger", -0.20, 0.08, 0.0), ("margin_trowel", 0.12, 0.10, 0.0), ("penetrometer", 0.25, -0.19, 0.0)])
AGE_21C = 60*60.0      # s: the start's age at 21 C (layout.AGES_21C), scaled by the seed's temperature (masco_factor)


def worked(reg, inside, rng, X, Y, stage):
    """The work done before the start (layout.pad_region): the darby's (layout._darbied)."""
    L._darbied(reg, inside, rng)
    stage["darbied"] = True


def setup(seed, temp_c, air, bleed, skin_s):
    regions = {}
    regions["pad"] = L.pad_region("pad-edge", seed, temp_c, air, bleed, AGE_21C, worked)
    return regions
