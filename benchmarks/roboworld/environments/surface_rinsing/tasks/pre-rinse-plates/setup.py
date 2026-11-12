"""Start of pre-rinse-plates: six plates soiled with dried tomato sauce stand in the peg rack on the pre-rinse sink.

setup(s) adds the rack and the plates to layout.Start (layout.setup: the pre-rinse sink's drain open, no fills); the
seed varies the sauce load by up to 30 %."""
import math

from surface_rinsing.layout import PRS_X0, PRS_X1, PRS_Y0, PRS_Y1, TABLE_Z, yaw_quat


def setup(s):
    items, rng, seed = s.items, s.rng, s.seed
    rx = (PRS_X0 + PRS_X1)/2
    ry = (PRS_Y0 + PRS_Y1)/2
    items["peg_rack"] = dict(kind="peg_rack", pose=(rx, ry, TABLE_Z + 0.012 + 0.0005, *yaw_quat(math.pi/2)))
    load = 1.0 + (rng.uniform(-0.3, 0.3) if seed else 0.0)
    for k in range(6):
        items[f"plate_{k}"] = dict(kind="plate", in_rack=("peg_rack", k), soil=dict(kind="tomato", load=load), wet=None)
