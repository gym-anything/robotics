"""Start of scrub-rinse-pan: a 10 in aluminium fry pan with dried egg on its inside soaks in the wash compartment,
handle toward the front, and the soapy scrub sponge lies on the back ledge.

setup(s) adds them to layout.Start (layout.setup: the pot sink's three compartments filled, drains shut); the sponge is
layout.ledge_sponge, which rinse-sponge's start shares. The seed moves the pan up to 2 cm, turns it and varies the egg
load and how long it has soaked. DRESS_WALL_SHELF tells render._dressing to leave the wall shelf bare: it is this task's
destination."""
import math

from surface_rinsing.layout import BOWL_Y0, BOWL_Z0, BOWLS, ledge_sponge, yaw_quat

# the wall shelf is this task's destination: an accepted pan (centre over the clean board, handle any way) can
# reach the shelf's far end, and the drawn-only jugs render._dressing sets on it have no colliders
DRESS_WALL_SHELF = False


def setup(s):
    items, rng, seed, jit = s.items, s.rng, s.seed, s.jit
    ledge_sponge(s)
    wx = 0.5*sum(BOWLS["wash"])
    # its 200 mm handle points at the bowl's front: the centre stands 0.36 m back from the front wall so that
    # the handle's tip (0.322 m out) clears it (at 0.275 m the handle would stand 5 cm through the wall, a
    # pierce that makes no contact and that the contact-based overlap check does not see)
    items["pan"] = dict(kind="pan", pose=(wx + jit(0.02), BOWL_Y0 + 0.36 + jit(0.02), BOWL_Z0 + 0.0005,
                                          *yaw_quat(math.radians(-90) + jit(0.3))),
                        soil=dict(kind="egg", load=1.0 + (jit(0.3) if seed else 0.0),
                                  soaked_s=float(rng.uniform(20.0, 240.0)) if seed else 120.0), wet=None)
