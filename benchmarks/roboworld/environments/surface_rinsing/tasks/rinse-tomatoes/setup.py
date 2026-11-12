"""Start of rinse-tomatoes: the colander on the prep-sink drainboard holds fifteen dusty cherry tomatoes; a clean steel
bowl and an HDPE cutting board are on the work table.

setup(s) adds them to layout.Start (layout.setup: the prep sink's drain open, no fills); the seed moves and turns the
colander and varies each tomato's dust load by up to 30 %."""
import math

from surface_rinsing.layout import DB_DROP, V_RIM, VBOWL_Y0, VDB_Y0, VSINK_X0, WORK, X, yaw_quat


def setup(s):
    items, seed, jit = s.items, s.seed, s.jit
    cx, cy = (VSINK_X0 + X)/2 - 0.05, (VDB_Y0 + VBOWL_Y0)/2
    items["colander"] = dict(kind="colander", pose=(cx + jit(0.02), cy + jit(0.02), V_RIM + DB_DROP*0.5 + 0.0005,
                                                     *yaw_quat(jit(math.pi))))
    for k in range(15):
        items[f"tomato_{k}"] = dict(kind="tomato", in_colander=k, soil=dict(kind="dust", load=1.0 + (jit(0.3) if seed else 0)))
    w = WORK
    items["clean_bowl"] = dict(kind="steel_bowl", pose=(0.5*(w["x0"] + w["x1"]), w["y1"] - 0.30, w["z"] + 0.0005, *yaw_quat(0.0)))
    items["cutting_board"] = dict(kind="cutting_board", pose=(0.5*(w["x0"] + w["x1"]), w["y0"] + 0.36, w["z"] + 0.0005,
                                                              *yaw_quat(math.pi/2)))
