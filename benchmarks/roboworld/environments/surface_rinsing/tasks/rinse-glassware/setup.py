"""Start of rinse-glassware: six washed glasses (four rocks glasses, two mixing glasses) stand on the soiled drainboard,
wet with wash solution and suds, and the glass rack waits on the clean drainboard.

setup(s) adds the free items to layout.Start (layout.setup: the pot sink's three compartments filled, drains shut); the
seed moves each glass up to 2 cm, turns it and varies its foam patches."""
import math

from surface_rinsing.layout import BOWL_Y0, BOWL_Y1, DRAINBOARDS, board_z, on_board


def setup(s):
    items, rng, seed, jit = s.items, s.rng, s.seed, s.jit
    c_wash, T_wash, dz, cx, zdb = s.c_wash, s.T_wash, s.dz, s.cx, s.zdb
    spots = [(cx - 0.16, BOWL_Y0 + 0.13), (cx, BOWL_Y0 + 0.13), (cx + 0.16, BOWL_Y0 + 0.13), (cx - 0.16, BOWL_Y0 + 0.30),
             (cx + 0.02, BOWL_Y0 + 0.32), (cx + 0.18, BOWL_Y0 + 0.33)]
    kinds = ["rocks_glass"]*4 + ["mixing_glass"]*2
    for k, (kind, (x, y)) in enumerate(zip(kinds, spots)):
        x, y = x + jit(0.02), y + jit(0.02)
        items[f"glass_{k}"] = dict(kind=kind, pose=(x, y, zdb(x) + dz, *on_board("soiled", jit(math.pi))),
                                   wet=dict(h=25e-6, surf=c_wash, T=T_wash, foam_patches=2 + int(rng.integers(0, 3))
                                            if seed else 2))
    cx2 = sum(DRAINBOARDS["clean"])/2
    items["glass_rack"] = dict(kind="glass_rack", pose=(cx2, (BOWL_Y0 + BOWL_Y1)/2, board_z("clean", cx2) + dz,
                                                        *on_board("clean", 0.0)))
