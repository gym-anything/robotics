"""Room, stations, presets and tasks. World frame: x east along the north wall, y north, z up; floor z = 0.

A 7.2 x 5.4 m dish room and back prep area of a full-service restaurant (3.05 m ceiling). Four stations, each a minor
setup change of the same room:
  pot-sink    P: Advance Tabco FS-3-1824-24RL three-compartment sink on the north wall, pre-rinse unit with a swing
                 faucet and a spray valve on a spring balancer, lever wastes; glass rack on the clean drainboard
  pre-rinse   R: soiled dish table on the west wall with its 20 x 20 x 5 in pre-rinse sink, a second pre-rinse unit and
                 the door-type dish machine beyond
  prep-sink   V: single-compartment prep sink with a gooseneck faucet on the east wall, drainboard, work table with an
                 HDPE cutting board and a clean bowl
  can-wash    C: can-wash bay in the south-east corner, floor sloped 1/8 in per ft to a 6 in floor drain, open hose reel
                 with a squeeze valve, 32 gal BRUTE can, dunnage shelf (clean storage)
Every fixture is a static collider; presets only instantiate their own station's free items and water charts.
`setup(task, seed)` returns the episode spec (robot start, items, fills, soil, seeds).
"""

import math
import numpy as np

from classroom_arrangement.furniture import Part
from . import params as P

INCH = P.INCH
ROOM = dict(x=7.2, y=5.4, z=3.05, wall_t=0.10)
X, Y, Z, WT = ROOM["x"], ROOM["y"], ROOM["z"], ROOM["wall_t"]

# ------------------------------------------------------------------ pot sink (north wall)
PS = P.POT_SINK
BW, BD, BH = PS["bowl"]                      # 18 x 24 x 14 in
RIM = PS["rim_z"]
SINK_Y0 = Y - 30*INCH                        # front of the sink (30 in deep overall)
BOWL_Y0, BOWL_Y1 = SINK_Y0 + 0.038, SINK_Y0 + 0.038 + BD
BOWL_Z0 = RIM - BH
SINK_X0 = 2.0
DB = PS["drainboard"]
BOWLS = {"wash": (SINK_X0 + DB, SINK_X0 + DB + BW), "rinse": (SINK_X0 + DB + BW, SINK_X0 + DB + 2*BW),
         "sanitize": (SINK_X0 + DB + 2*BW, SINK_X0 + DB + 3*BW)}
DRAINBOARDS = {"soiled": (SINK_X0, SINK_X0 + DB), "clean": (SINK_X0 + DB + 3*BW, SINK_X0 + 2*DB + 3*BW)}
SINK_X1 = DRAINBOARDS["clean"][1]
DB_DROP = 0.006                              # drainboards pitch 6 mm down toward the bowls over 24 in (1 %)
RISER_X = BOWLS["wash"][1]                   # pre-rinse unit over the wash/rinse partition
RISER_Y = Y - 0.045
SPLASH_TOP = RIM + PS["splash_h"]

# ------------------------------------------------------------------ dish table (west wall)
DT = P.DISH_TABLE
TABLE_X1 = DT["depth"]
TABLE_Y0, TABLE_Y1 = 1.55, 1.55 + 72*INCH
TABLE_Z = DT["height"]
PRS_W, PRS_D, PRS_H = DT["sink"]
PRS_X0, PRS_X1 = 0.12, 0.12 + PRS_W
PRS_Y1 = TABLE_Y1 - 0.06
PRS_Y0 = PRS_Y1 - PRS_D
MACHINE = dict(x0=0.0, x1=0.70, y0=TABLE_Y1, y1=TABLE_Y1 + 0.66, z1=1.70)
R_RISER = (0.035, 0.5*(PRS_Y0 + PRS_Y1))
# extra height of both pre-rinse units' overhead arms for the longer free hose (links beyond the first 6, 7 cm each):
# the arm tops are 1.9-2.0 m above the floor, as a wall-mounted unit's (base on the backsplash + ~35 in)
PRSV_RISE = 0.07*(P.PRSV["hose_links"] - 6)

# ------------------------------------------------------------------ prep sink and work table (east wall)
VS = P.PREP_SINK
VB_W, VB_D, VB_H = VS["bowl"]
V_RIM = VS["rim_z"]
VSINK_X0 = X - 30*INCH
VBOWL_X0, VBOWL_X1 = VSINK_X0 + 0.038, VSINK_X0 + 0.038 + VB_D
VBOWL_Y0, VBOWL_Y1 = 3.80, 3.80 + VB_W
VDB_Y0 = VBOWL_Y0 - VS["drainboard"]
WORK = dict(x0=X - 30*INCH, x1=X, y0=1.95, y1=1.95 + 48*INCH, z=VS["table_h"])
V_FAUCET = (X - 0.05, 0.5*(VBOWL_Y0 + VBOWL_Y1))

# ------------------------------------------------------------------ can-wash bay (south-east)
CB = P.CAN_BAY
BAY = dict(x0=X - CB["size"], x1=X, y0=0.0, y1=1.8)
DRAIN = (0.5*(BAY["x0"] + BAY["x1"]), 0.5*(BAY["y0"] + BAY["y1"]))
REEL = dict(x=6.70, y=0.06, z=1.50)
# wall holder the hose valve docks in, head down: an octagonal sleeve whose floor is a ring (hole_r = the hole's apothem)
HOLSTER = dict(x=6.28, y=0.10, z0=0.95, r=0.042, depth=0.09, hole_r=0.020)
SHELF = dict(x0=4.35, x1=5.05, y0=0.05, y1=0.55, levels=(0.25, 0.80, 1.35))


def _box(name, lo, hi, material, station="room", role=""):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Part(name, "box", tuple((hi - lo)/2), tuple((hi + lo)/2), material=material, role=role or station)


def _cyl(name, c, r, hh, material, quat=(0., 0., 0., 1.), station="room", collide=True):
    return Part(name, "cylinder", (r, hh), tuple(c), quat, material=material, role=station, collide=collide)


Y_AXIS_Q = (0.7071067811865476, 0., 0., 0.7071067811865476)      # local z -> world -y (rotation +90 deg about x)
X_AXIS_Q = (0., 0.7071067811865476, 0., 0.7071067811865476)      # local z -> world +x


def _tilt_q(axis, deg):
    from scipy.spatial.transform import Rotation
    return tuple(Rotation.from_rotvec(np.radians(deg)*np.asarray(axis, float)).as_quat())


def static_parts():
    """All installed parts of the room: (Part) with role = station tag. Boxes, cylinders and capsules only."""
    s = []
    # walls, ceiling (floor: the ground plane), back door frame
    s.append(_box("wall_n", (-WT, Y, 0), (X + WT, Y + WT, Z), "tile"))
    s.append(_box("wall_s_w", (-WT, -WT, 0), (3.0, 0, Z), "tile"))
    s.append(_box("wall_s_e", (4.0, -WT, 0), (X + WT, 0, Z), "tile"))
    s.append(_box("wall_s_head", (3.0, -WT, 2.13), (4.0, 0, Z), "tile"))
    s.append(_box("back_door", (3.05, -0.05, 0.0), (3.95, -0.01, 2.10), "door"))
    s.append(_box("wall_w", (-WT, 0, 0), (0, Y, Z), "tile"))
    s.append(_box("wall_e", (X, 0, 0), (X + WT, Y, Z), "tile"))
    s.append(_box("ceiling", (-WT, -WT, Z), (X + WT, Y + WT, Z + 0.1), "ceiling"))
    s += pot_sink_parts()
    s += dish_table_parts()
    s += prep_parts()
    s += bay_parts()
    return s


def pot_sink_parts():
    s, t = [], 0.002
    st = "pot-sink"
    for name, (x0, x1) in BOWLS.items():
        s.append(_box(f"{name}_floor", (x0, BOWL_Y0, BOWL_Z0 - t), (x1, BOWL_Y1, BOWL_Z0), "steel", st, f"pool:{name}"))
        s.append(_box(f"{name}_front", (x0, BOWL_Y0 - t, BOWL_Z0), (x1, BOWL_Y0, RIM), "steel", st, f"pool:{name}"))
        s.append(_box(f"{name}_back", (x0, BOWL_Y1, BOWL_Z0), (x1, BOWL_Y1 + t, RIM), "steel", st, f"pool:{name}"))
        # strainer basket rim at the bowl centre (drawn flush with the floor; drain law in water.pools)
    for k, x in enumerate([BOWLS["wash"][0], BOWLS["wash"][1], BOWLS["rinse"][1], BOWLS["sanitize"][1]]):
        left = [n for n, (x0, x1) in BOWLS.items() if abs(x1 - x) < 1e-9]
        right = [n for n, (x0, x1) in BOWLS.items() if abs(x0 - x) < 1e-9]
        role = f"pool:{(right or left)[0]}"
        s.append(_box(f"partition_{k}", (x - t, BOWL_Y0, BOWL_Z0), (x + t, BOWL_Y1, RIM), "steel", st, role))
        s.append(Part(f"partition_roll_{k}", "capsule", (0.006, BD/2 - 0.006), (x, (BOWL_Y0 + BOWL_Y1)/2, RIM),
                      (0.7071067811865476, 0., 0., 0.7071067811865476), material="steel", role=st))
    # front apron and its rolled rim, back ledge, backsplash
    s.append(_box("apron", (SINK_X0, SINK_Y0, RIM - 0.10), (SINK_X1, SINK_Y0 + t, RIM), "steel", st))
    s.append(_box("apron_bowls", (BOWLS["wash"][0], SINK_Y0, RIM - 0.10), (BOWLS["sanitize"][1], BOWL_Y0 - t, RIM - 0.10 + t),
                  "steel", st))
    s.append(Part("front_roll", "capsule", (0.012, (SINK_X1 - SINK_X0)/2 - 0.012), ((SINK_X0 + SINK_X1)/2, SINK_Y0 + 0.012, RIM),
                  (0., 0.7071067811865476, 0., 0.7071067811865476), material="steel", role=st))
    s.append(_box("front_ledge", (BOWLS["wash"][0], SINK_Y0 + 0.012, RIM - t), (BOWLS["sanitize"][1], BOWL_Y0, RIM), "steel", st))
    s.append(_box("back_ledge", (SINK_X0, BOWL_Y1, RIM - t), (SINK_X1, Y - 0.02, RIM), "steel", st))
    s.append(_box("backsplash", (SINK_X0, Y - 0.02, RIM), (SINK_X1, Y, SPLASH_TOP), "steel", st))
    s.append(_box("backsplash_cap", (SINK_X0, Y - 0.05, SPLASH_TOP - 0.01), (SINK_X1, Y, SPLASH_TOP), "steel", st))
    # drainboards: 2 mm tops pitched toward the bowls, outer side splashes and the front roll
    for name, (x0, x1) in DRAINBOARDS.items():
        toward = 1.0 if name == "soiled" else -1.0           # bowls lie to +x of the soiled board, -x of the clean one
        ang = math.degrees(math.atan2(DB_DROP, x1 - x0))*toward
        c = ((x0 + x1)/2, (SINK_Y0 + 0.024 + Y - 0.02)/2, RIM + DB_DROP/2 - t/2)
        s.append(Part(f"drainboard_{name}", "box", ((x1 - x0)/2, (Y - 0.02 - SINK_Y0 - 0.024)/2, t/2), c,
                      _tilt_q((0., 1., 0.), ang), material="steel", role=st))
        xo = x0 if name == "soiled" else x1
        s.append(_box(f"side_splash_{name}", (xo - 0.02 if name == "soiled" else xo, SINK_Y0, RIM),
                      (xo if name == "soiled" else xo + 0.02, Y, RIM + 0.10), "steel", st))
    # legs (1 5/8 in tubes) and cross rails
    for x in (SINK_X0 + 0.05, BOWLS["wash"][0] + 0.03, BOWLS["sanitize"][1] - 0.03, SINK_X1 - 0.05):
        for y in (SINK_Y0 + 0.08, Y - 0.08):
            zt = BOWL_Z0 - 0.002 if BOWLS["wash"][0] < x < BOWLS["sanitize"][1] else RIM - 0.002
            s.append(_cyl(f"leg_{x:.2f}_{y:.2f}", (x, y, zt/2), 0.0206, zt/2, "steel", station=st))
    # pre-rinse unit: mixing valve body, riser, spring arm (static); handles, spout, spray valve are articulated
    s.append(_box("prerinse_valve_body", (RISER_X - 0.11, Y - 0.07, SPLASH_TOP - 0.03), (RISER_X + 0.11, Y - 0.02,
                                                                                         SPLASH_TOP + 0.03), "chrome", st))
    s.append(_cyl("prerinse_riser", (RISER_X, RISER_Y - 0.03, SPLASH_TOP + 0.25 + 0.5*PRSV_RISE), 0.0127, 0.25 + 0.5*PRSV_RISE,
                  "chrome", station=st))
    s.append(Part("prerinse_arm", "capsule", (0.008, 0.11), (RISER_X, RISER_Y - 0.15, SPLASH_TOP + 0.50 + PRSV_RISE),
                  (0.7071067811865476, 0., 0., 0.7071067811865476), material="chrome", role=st))
    s.append(_cyl("wall_bracket", (RISER_X, Y - 0.015, SPLASH_TOP + 0.40), 0.03, 0.015, "chrome", Y_AXIS_Q, station=st))
    # chemical dispensers on the tile above the sanitize bowl (static colliders)
    for k, x in enumerate((BOWLS["sanitize"][0] + 0.12, BOWLS["sanitize"][0] + 0.32)):
        s.append(_box(f"dispenser_{k}", (x - 0.07, Y - 0.11, 1.45), (x + 0.07, Y, 1.75), "plastic_white", st))
    s.append(_box("sink_sign", (BOWLS["wash"][0] + 0.05, Y - 0.006, 1.52), (BOWLS["sanitize"][1] - 0.05, Y, 1.66), "sign", st))
    # wall shelf above the clean drainboard (pans drain on it inverted)
    x0, x1 = DRAINBOARDS["clean"]
    s.append(_box("wall_shelf", (x0, Y - 0.30, 1.50), (x1 + 0.3, Y, 1.52), "steel", st))
    return s


def dish_table_parts():
    s, t, st = [], 0.002, "pre-rinse"
    # table top with the pre-rinse sink cut out (four pieces), rolled front rim, backsplash on the wall
    s.append(_box("table_top_s", (0.0, TABLE_Y0, TABLE_Z - t), (TABLE_X1, PRS_Y0, TABLE_Z), "steel", st))
    s.append(_box("table_top_n", (0.0, PRS_Y1, TABLE_Z - t), (TABLE_X1, TABLE_Y1, TABLE_Z), "steel", st))
    s.append(_box("table_top_w", (0.0, PRS_Y0, TABLE_Z - t), (PRS_X0, PRS_Y1, TABLE_Z), "steel", st))
    s.append(_box("table_top_e", (PRS_X1, PRS_Y0, TABLE_Z - t), (TABLE_X1, PRS_Y1, TABLE_Z), "steel", st))
    s.append(Part("table_roll", "capsule", (0.015, (TABLE_Y1 - TABLE_Y0)/2 - 0.015), (TABLE_X1 - 0.015, (TABLE_Y0 + TABLE_Y1)/2,
                  TABLE_Z + 0.008), (0.7071067811865476, 0., 0., 0.7071067811865476), material="steel", role=st))
    s.append(_box("table_splash", (0.0, TABLE_Y0, TABLE_Z), (0.02, TABLE_Y1, TABLE_Z + DT["splash_h"]), "steel", st))
    # pre-rinse sink: floor, walls (pool walls)
    z0 = TABLE_Z - PRS_H
    s.append(_box("prs_floor", (PRS_X0, PRS_Y0, z0 - t), (PRS_X1, PRS_Y1, z0), "steel", st, "pool:prerinse"))
    s.append(_box("prs_w", (PRS_X0 - t, PRS_Y0, z0), (PRS_X0, PRS_Y1, TABLE_Z), "steel", st, "pool:prerinse"))
    s.append(_box("prs_e", (PRS_X1, PRS_Y0, z0), (PRS_X1 + t, PRS_Y1, TABLE_Z), "steel", st, "pool:prerinse"))
    s.append(_box("prs_s", (PRS_X0, PRS_Y0 - t, z0), (PRS_X1, PRS_Y0, TABLE_Z), "steel", st, "pool:prerinse"))
    s.append(_box("prs_n", (PRS_X0, PRS_Y1, z0), (PRS_X1, PRS_Y1 + t, TABLE_Z), "steel", st, "pool:prerinse"))
    # rails across the sink that carry the peg rack
    for k, y in enumerate((PRS_Y0 + 0.03, PRS_Y1 - 0.03)):
        s.append(Part(f"prs_rail_{k}", "capsule", (0.006, PRS_W/2), ((PRS_X0 + PRS_X1)/2, y, TABLE_Z + 0.006),
                      (0., 0.7071067811865476, 0., 0.7071067811865476), material="steel", role=st))
    for x in (0.08, TABLE_X1 - 0.06):
        for y in (TABLE_Y0 + 0.06, TABLE_Y1 - 0.06):
            s.append(_cyl(f"tleg_{x:.2f}_{y:.2f}", (x, y, (TABLE_Z - 0.03)/2), 0.0206, (TABLE_Z - 0.03)/2, "steel", station=st))
    m = MACHINE
    s.append(_box("dish_machine", (m["x0"], m["y0"], 0.0), (m["x1"], m["y1"], m["z1"]), "steel_brushed", st))
    s.append(_box("clean_table", (0.0, m["y1"], TABLE_Z - 0.04), (TABLE_X1, m["y1"] + 0.6, TABLE_Z), "steel", st))
    rx, ry = R_RISER
    s.append(_box("r_valve_body", (0.0, ry - 0.11, TABLE_Z + DT["splash_h"] + 0.05), (0.05, ry + 0.11, TABLE_Z + DT["splash_h"] + 0.11),
                  "chrome", st))
    s.append(_cyl("r_riser", (rx, ry, TABLE_Z + DT["splash_h"] + 0.40 + 0.5*PRSV_RISE), 0.0127, 0.30 + 0.5*PRSV_RISE, "chrome",
                  station=st))
    s.append(Part("r_arm", "capsule", (0.008, 0.13), (rx + 0.15, ry, TABLE_Z + DT["splash_h"] + 0.70 + PRSV_RISE),
                  (0., 0.7071067811865476, 0., 0.7071067811865476), material="chrome", role=st))
    return s


def prep_parts():
    s, t, st = [], 0.002, "prep-sink"
    z0 = V_RIM - VB_H
    s.append(_box("v_floor", (VBOWL_X0, VBOWL_Y0, z0 - t), (VBOWL_X1, VBOWL_Y1, z0), "steel", st, "pool:prep"))
    s.append(_box("v_w", (VBOWL_X0 - t, VBOWL_Y0, z0), (VBOWL_X0, VBOWL_Y1, V_RIM), "steel", st, "pool:prep"))
    s.append(_box("v_e", (VBOWL_X1, VBOWL_Y0, z0), (VBOWL_X1 + t, VBOWL_Y1, V_RIM), "steel", st, "pool:prep"))
    s.append(_box("v_s", (VBOWL_X0, VBOWL_Y0 - t, z0), (VBOWL_X1, VBOWL_Y0, V_RIM), "steel", st, "pool:prep"))
    s.append(_box("v_n", (VBOWL_X0, VBOWL_Y1, z0), (VBOWL_X1, VBOWL_Y1 + t, V_RIM), "steel", st, "pool:prep"))
    s.append(_box("v_ledge", (VBOWL_X1, VDB_Y0, V_RIM - t), (X, VBOWL_Y1 + 0.05, V_RIM), "steel", st))
    s.append(_box("v_apron", (VSINK_X0, VDB_Y0, V_RIM - 0.10), (VSINK_X0 + t, VBOWL_Y1 + 0.05, V_RIM), "steel", st))
    s.append(_box("v_side_n", (VSINK_X0, VBOWL_Y1, V_RIM), (X, VBOWL_Y1 + 0.05, V_RIM + 0.10), "steel", st))
    s.append(_box("v_splash", (X - 0.02, VDB_Y0, V_RIM), (X, VBOWL_Y1 + 0.05, V_RIM + 0.25), "steel", st))
    ang = math.degrees(math.atan2(DB_DROP, VS["drainboard"]))
    s.append(Part("v_drainboard", "box", ((X - 0.02 - VSINK_X0)/2, VS["drainboard"]/2, t/2),
                  ((VSINK_X0 + X - 0.02)/2, (VDB_Y0 + VBOWL_Y0)/2, V_RIM + DB_DROP/2 - t/2),
                  _tilt_q((1., 0., 0.), -ang), material="steel", role=st))
    s.append(_box("v_side_s", (VSINK_X0, VDB_Y0 - 0.02, V_RIM), (X, VDB_Y0, V_RIM + 0.10), "steel", st))
    s.append(Part("v_front_roll", "capsule", (0.012, (VBOWL_Y1 + 0.05 - VDB_Y0)/2 - 0.012),
                  (VSINK_X0 + 0.012, (VDB_Y0 + VBOWL_Y1 + 0.05)/2, V_RIM), (0.7071067811865476, 0., 0., 0.7071067811865476),
                  material="steel", role=st))
    for x in (VSINK_X0 + 0.06, X - 0.06):
        for y in (VDB_Y0 + 0.05, VBOWL_Y1):
            zt = (z0 - 0.002) if y > VBOWL_Y0 else V_RIM - 0.002
            s.append(_cyl(f"vleg_{x:.2f}_{y:.2f}", (x, y, zt/2), 0.0206, zt/2, "steel", station=st))
    fx, fy = V_FAUCET
    s.append(_box("v_faucet_body", (X - 0.06, fy - 0.11, V_RIM + 0.18), (X - 0.02, fy + 0.11, V_RIM + 0.24), "chrome", st))
    # gooseneck spout: riser from the faucet body, an arm out over the bowl, a turned-down nozzle ending at the outlet
    # (scene.fixtures["gooseneck"]["outlet"])
    xo, zo = VBOWL_X0 + 0.5*VB_D + 0.02, V_RIM + 0.36
    xb, r_sp = X - 0.04, 0.011
    z_top = zo + 0.085
    s.append(_cyl("v_spout_riser", (xb, fy, 0.5*(V_RIM + 0.24 + z_top)), r_sp, 0.5*(z_top - V_RIM - 0.24), "chrome", station=st))
    arm_len = xb - xo
    s.append(Part("v_spout_arm", "capsule", (r_sp, 0.5*arm_len), (0.5*(xb + xo), fy, z_top), _tilt_q((0., 1., 0.), 90.0),
                  material="chrome", role=st))
    s.append(_cyl("v_spout_nozzle", (xo, fy, 0.5*(zo + z_top)), r_sp + 0.001, 0.5*(z_top - zo), "chrome", station=st))
    w = WORK
    s.append(_box("work_top", (w["x0"], w["y0"], w["z"] - 0.04), (w["x1"], w["y1"], w["z"]), "steel", st))
    s.append(_box("work_shelf", (w["x0"] + 0.03, w["y0"] + 0.03, 0.25), (w["x1"] - 0.03, w["y1"] - 0.03, 0.27), "steel", st))
    for x in (w["x0"] + 0.04, w["x1"] - 0.04):
        for y in (w["y0"] + 0.04, w["y1"] - 0.04):
            s.append(_cyl(f"wleg_{x:.2f}_{y:.2f}", (x, y, (w["z"] - 0.04)/2), 0.0206, (w["z"] - 0.04)/2, "steel", station=st))
    return s


def bay_parts():
    s, st = [], "can-wash"
    # floor drain strainer (flush), hose reel drum and bracket, valve hook, dunnage shelf, can-drying hooks
    # the strainer is flush with the floor (the floor plane is its collider; drawn only)
    s.append(_cyl("drain_strainer", (DRAIN[0], DRAIN[1], 0.0015), CB["drain_r"], 0.0015, "chrome", station=st, collide=False))
    r = REEL
    s.append(_cyl("reel_drum", (r["x"], r["y"] + 0.17, r["z"]), 0.20, 0.10, "reel_red", Y_AXIS_Q, station=st))
    s.append(_box("reel_bracket", (r["x"] - 0.06, 0.0, r["z"] - 0.30), (r["x"] + 0.06, 0.06, r["z"] + 0.12), "reel_red", st))
    # hose-valve holster: an octagonal stainless sleeve on a wall bracket; the valve docks in it head down, its 66 mm
    # bumper on a ring floor with a 40 mm hole, through which a docked valve's spray falls to the bay floor. (First
    # build: a closed cup floor, an up-facing face that kept whatever struck it: a valve squeezed while docked lost
    # 3.4 kg of water into the cup in 20 s, where a real 0.5 L cup would fill and overflow; P11 chaos.)
    h = HOLSTER
    ro, ri = h["r"] + 0.003, h["hole_r"]
    for k in range(8):
        a = 2*math.pi*(k + 0.5)/8
        q = (0., 0., math.sin(a/2), math.cos(a/2))
        c = (h["x"] + (h["r"] + 0.0015)*math.cos(a), h["y"] + (h["r"] + 0.0015)*math.sin(a), h["z0"] + h["depth"]/2)
        s.append(Part(f"holster_{k}", "box", (0.0015, h["r"]*math.tan(math.pi/8) + 0.0015, h["depth"]/2), c, q,
                      material="steel", role=st))
        rc = 0.5*(ro + ri)
        s.append(Part(f"holster_floor_{k}", "box", (0.5*(ro - ri), ro*math.tan(math.pi/8), 0.002),
                      (h["x"] + rc*math.cos(a), h["y"] + rc*math.sin(a), h["z0"] - 0.002), q, material="steel", role=st))
    s.append(_box("holster_bracket", (h["x"] - 0.02, 0.0, h["z0"] - 0.004), (h["x"] + 0.02, h["y"] - h["r"], h["z0"] + 0.06), "steel", st))
    sh = SHELF
    for k, z in enumerate(sh["levels"]):
        s.append(_box(f"shelf_{k}", (sh["x0"], sh["y0"], z - 0.02), (sh["x1"], sh["y1"], z), "shelf_wire", st))
    for x in (sh["x0"] + 0.015, sh["x1"] - 0.015):
        for y in (sh["y0"] + 0.015, sh["y1"] - 0.015):
            s.append(_cyl(f"shelf_post_{x:.2f}_{y:.2f}", (x, y, 0.85), 0.0127, 0.85, "chrome", station=st))
    for k, x in enumerate((6.05, 6.75)):
        s.append(_box(f"can_hook_{k}", (x - 0.02, 0.0, 1.30), (x + 0.02, 0.16, 1.33), "chrome", st))
    return s


# ------------------------------------------------------------------ water surfaces of the stations (static charts)
def static_charts(station):
    """[(name, origin, u, v, size (Lu, Lv), cell, material, open_edges, role)] of the installed surfaces that carry
    films at a station: the drainboards and floor strip (pot sink), the table top and floor (pre-rinse), the drainboard,
    work table and floor (prep sink), the sloped bay floor (can wash)."""
    out = []
    if station == "pot-sink":
        y0, y1 = SINK_Y0 + 0.024, Y - 0.02
        for name, (x0, x1) in DRAINBOARDS.items():
            # the board's own pitch: the soiled board falls toward +x (the wash bowl), the clean one toward -x
            Lx = x1 - x0
            dz = -DB_DROP if name == "soiled" else DB_DROP
            u = np.array([Lx, 0.0, dz])/math.hypot(Lx, dz)
            z0 = RIM + (DB_DROP if name == "soiled" else 0.0)
            out.append((f"drainboard_{name}", (x0, y0, z0), tuple(u), (0, 1, 0), (math.hypot(Lx, dz), y1 - y0), 0.005,
                        "steel", "u1" if name == "soiled" else "u0", "clean" if name == "clean" else "soiled", {}))
        out.append(("floor_front", (SINK_X0 - 0.2, SINK_Y0 - 1.2, 0.0), (1, 0, 0), (0, 1, 0), (SINK_X1 - SINK_X0 + 0.4, 1.2),
                    0.010, "tile", "u0 u1 v0", "floor", {}))
    elif station == "pre-rinse":
        out.append(("table_s", (0.02, TABLE_Y0, TABLE_Z), (1, 0, 0), (0, 1, 0), (TABLE_X1 - 0.05, PRS_Y0 - TABLE_Y0), 0.005,
                    "steel", "v1", "soiled", {}))
        out.append(("table_n", (0.02, PRS_Y1, TABLE_Z), (1, 0, 0), (0, 1, 0), (TABLE_X1 - 0.05, TABLE_Y1 - PRS_Y1), 0.005,
                    "steel", "v0", "soiled", {}))
        m = MACHINE
        out.append(("clean_table", (0.0, m["y1"], TABLE_Z), (1, 0, 0), (0, 1, 0), (TABLE_X1, 0.6), 0.005, "steel", "",
                    "clean", {}))
        out.append(("floor_front", (TABLE_X1, TABLE_Y0 - 0.2, 0.0), (1, 0, 0), (0, 1, 0), (1.2, TABLE_Y1 - TABLE_Y0 + 1.0), 0.010,
                    "tile", "u1 v0 v1", "floor", {}))
    elif station == "prep-sink":
        Ly = VBOWL_Y0 - VDB_Y0
        v = np.array([0.0, Ly, -DB_DROP])/math.hypot(Ly, DB_DROP)
        out.append(("v_drainboard", (VSINK_X0 + 0.024, VDB_Y0, V_RIM + DB_DROP), (1, 0, 0), tuple(v),
                    (X - 0.02 - VSINK_X0 - 0.024, math.hypot(Ly, DB_DROP)), 0.005, "steel", "v1", "soiled", {}))
        out.append(("floor_front", (VSINK_X0 - 1.2, VDB_Y0 - 0.2, 0.0), (1, 0, 0), (0, 1, 0), (1.2, VBOWL_Y1 - VDB_Y0 + 0.4),
                    0.010, "tile", "u0 v0 v1", "floor", {}))
    elif station == "can-wash":
        b = BAY
        out.append(("bay_floor", (b["x0"], b["y0"], 0.0), (1, 0, 0), (0, 1, 0), (b["x1"] - b["x0"], b["y1"] - b["y0"]),
                    CB["cell"], "tile", "u0 v1", "floor", dict(drain=(DRAIN[0], DRAIN[1], CB["drain_r"]), slope=CB["slope"])))
        out.append(("floor_west", (b["x0"] - 1.0, b["y0"], 0.0), (1, 0, 0), (0, 1, 0), (1.0, b["y1"] - b["y0"]), 0.010, "tile",
                    "u0 v1", "floor", {}))
    return out


def static_pools(station):
    """Static pools (box cavities): name -> dict(lo, hi, rim (z of the lowest rim), drain_d, drain_xy)."""
    t = 0.002
    if station == "pot-sink":
        return {n: dict(lo=(x0 + t, BOWL_Y0, BOWL_Z0), hi=(x1 - t, BOWL_Y1, RIM), drain_d=PS["drain_d"],
                        drain_xy=((x0 + x1)/2, (BOWL_Y0 + BOWL_Y1)/2)) for n, (x0, x1) in BOWLS.items()}
    if station == "pre-rinse":
        return {"prerinse": dict(lo=(PRS_X0, PRS_Y0, TABLE_Z - PRS_H), hi=(PRS_X1, PRS_Y1, TABLE_Z), drain_d=0.035,
                                 drain_xy=((PRS_X0 + PRS_X1)/2, (PRS_Y0 + PRS_Y1)/2))}
    if station == "prep-sink":
        return {"prep": dict(lo=(VBOWL_X0, VBOWL_Y0, V_RIM - VB_H), hi=(VBOWL_X1, VBOWL_Y1, V_RIM), drain_d=0.035,
                             drain_xy=((VBOWL_X0 + VBOWL_X1)/2, (VBOWL_Y0 + VBOWL_Y1)/2))}
    return {}


# ------------------------------------------------------------------ presets and tasks
STATIONS = ("pot-sink", "pre-rinse", "prep-sink", "can-wash")
# The Duo's shoulders (fr3 link1) sit 0.53 m ahead of base_link at 0.71 m with the Spine down, below every rim here: the
# base starts with the shoulders 0.07-0.11 m short of the fixture front (at 4.17 m they would be inside the pot sink's
# bowls and the stowed arms would trip the velocity reflex at tick 1)
ROBOT_START = {"pot-sink": (3.296, 4.00, math.pi/2), "pre-rinse": (1.38, 0.5*(PRS_Y0 + PRS_Y1), math.pi),
               "prep-sink": (5.82, 3.80, 0.0), "can-wash": (5.15, 1.10, 0.0)}

# Spine height at the start: the robot waits at its station raised so that the head camera (on the Spine carriage,
# pitched down) overlooks the work. With the Spine down the camera sat below every rim and its first view was the
# fixture's apron, 10 cm away (first stills); a policy would have had to raise the Spine before it could see anything.
ROBOT_SPINE = {"pot-sink": 0.45, "pre-rinse": 0.45, "prep-sink": 0.42, "can-wash": 0.25}

TASKS = {
    "rinse-glassware": dict(preset="pot-sink", tier="medium", instruction=(
        "Six washed glasses (four 12 oz rocks glasses and two 16 oz mixing glasses) stand on the soiled (left) drainboard, "
        "still wet with detergent solution and suds. Rinse each one free of detergent inside and out under the swing "
        "faucet or in the rinse compartment, immerse it in the sanitizer compartment (right bowl) for at least 30 s, and "
        "set it mouth-down in the glass rack on the clean drainboard to air-dry. Do not chip or drop any glass; leave the "
        "faucet and handles closed and nothing in the hands.")),
    "rinse-dishware": dict(preset="pot-sink", tier="medium", instruction=(
        "Four washed 9 in plates lie on the soiled drainboard with a soup bowl standing in each of the two back plates, "
        "all wet with detergent solution. Rinse every face of each (including the back and foot ring), sanitize each for "
        "at least 30 s, stand the plates in the peg rack and set the bowls upside down on the clean drainboard. Leave the "
        "faucet closed and nothing in the hands.")),
    "scrub-rinse-pan": dict(preset="pot-sink", tier="hard", instruction=(
        "The 10 in aluminium fry pan soaking in the wash compartment (left bowl) has dried egg stuck to its inside; a "
        "scrub sponge is on the back ledge. Clean the egg off the pan, rinse it free of detergent, immerse it in the "
        "sanitizer compartment (right bowl) for at least 30 s, and set it upside down on the wall shelf over the clean "
        "drainboard to drain. Finish with the sponge on the back ledge, no water running and nothing in the hands.")),
    "rinse-sponge": dict(preset="pot-sink", tier="entry", instruction=(
        "The scrub sponge on the back ledge is full of wash solution. Rinse it under the faucet until it releases no more "
        "suds, wring it out and set it back on the ledge. Leave the faucet closed and nothing in the hands.")),
    "pre-rinse-plates": dict(preset="pre-rinse", tier="medium-hard", instruction=(
        "Six plates soiled with dried tomato sauce stand in the peg rack on the pre-rinse sink. Mix the water to about "
        "49 C (110-120 F), pre-rinse every plate with the pre-rinse spray valve until no sauce is visible on its face or "
        "rim, without spraying onto the clean dish table, then shut the water and hang the spray valve back up.")),
    "rinse-tomatoes": dict(preset="prep-sink", tier="medium", instruction=(
        "The colander on the prep-sink drainboard holds fifteen dusty cherry tomatoes. Rinse them under running water "
        "until none carries visible soil, without losing any, turning them so every side is rinsed, then set the "
        "colander in the clean bowl on the work table to drain and close the faucet.")),
    "rinse-garbage-can": dict(preset="can-wash", tier="hard", instruction=(
        "The 32 gal garbage can in the can-wash bay has been scrubbed with detergent. Rinse its inside and outside free "
        "of soap and loosened debris with the hose, empty the rinse water into the floor drain, and leave the can "
        "upright with no water in it. Dock the hose valve back in its wall holster and do not spray the dunnage shelf.")),
    "flush-floor": dict(preset="can-wash", tier="medium", instruction=(
        "Soapy water and debris cover part of the can-wash bay floor. Flush them into the floor drain with the hose "
        "without spraying the dunnage shelf, then dock the hose valve back in its wall holster.")),
}
PRESETS = {s: tuple(t for t, v in TASKS.items() if v["preset"] == s) for s in STATIONS}


def yaw_quat(yaw):
    return (0., 0., math.sin(yaw/2), math.cos(yaw/2))


def board_tilt(name):
    """Rotation angle about +y of a pot-sink drainboard (pitched 6 mm down toward the bowls)."""
    x0, x1 = DRAINBOARDS[name]
    return math.atan2(DB_DROP, x1 - x0)*(1.0 if name == "soiled" else -1.0)


def on_board(name, yaw):
    """Quaternion (xyzw) of an item standing on a drainboard: the board's pitch, then the item's yaw."""
    from scipy.spatial.transform import Rotation
    return tuple((Rotation.from_euler("y", board_tilt(name))*Rotation.from_euler("z", yaw)).as_quat())


def board_z(name, x):
    x0, x1 = DRAINBOARDS[name]
    f = (x1 - x)/(x1 - x0) if name == "soiled" else (x - x0)/(x1 - x0)
    return RIM + DB_DROP*f


def setup(task, seed=0):
    """Episode spec of a task/seed: robot start, free items with poses and initial wetting/soil, pool fills."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose {sorted(TASKS)}")
    station = TASKS[task]["preset"]
    rng = np.random.default_rng(10007*seed + sum(map(ord, task)))
    jit = (lambda s: float(rng.uniform(-s, s))) if seed else (lambda s: 0.0)
    T_wash = 43.0 + (rng.uniform(0, 6) if seed else 0.0)
    supply = dict(p_static=P.SUPPLY["p_static"]*(rng.uniform(40, 80)/60.0 if seed else 1.0),
                  T_hot=P.SUPPLY["T_hot"], T_cold=P.SUPPLY["T_cold"] + (rng.uniform(-10, 10) if seed else 0.0))
    c_wash = P.SURFACTANT["c_wash"]*(rng.uniform(0.67, 1.67) if seed else 1.0)
    items, fills, drains = {}, {}, {}
    if station == "pot-sink":
        dz = 0.0005
        fills = {"wash": dict(depth=PS["fill_depth"], T=T_wash, surf=c_wash, foam=0.02),
                 "rinse": dict(depth=PS["fill_depth"], T=T_wash, surf=0.0, foam=0.0),
                 "sanitize": dict(depth=PS["fill_depth"], T=24.0, surf=0.0, foam=0.0, chlorine_ppm=100.0)}
        drains = {n: False for n in BOWLS}
        x0, x1 = DRAINBOARDS["soiled"]
        cx = (x0 + x1)/2
        zdb = lambda x: board_z("soiled", x)
        if task == "rinse-glassware":
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
        elif task in ("rinse-sponge", "scrub-rinse-pan"):
            # the soapy sponge lies on the back ledge behind the rinse bowl (where a dishwasher leaves it)
            rx = 0.5*sum(BOWLS["rinse"]) + jit(0.03)
            items["sponge"] = dict(kind="sponge", pose=(rx, 0.5*(BOWL_Y1 + Y - 0.02), RIM + 0.0005, *yaw_quat(jit(0.12))),
                                   pores=dict(fill=0.75, surf=c_wash, T=T_wash))
            if task == "scrub-rinse-pan":
                wx = 0.5*sum(BOWLS["wash"])
                # its 200 mm handle points at the bowl's front: the centre stands 0.36 m back from the front wall so that
                # the handle's tip (0.322 m out) clears it (at 0.275 m the handle would stand 5 cm through the wall, a
                # pierce that makes no contact and that the contact-based overlap check does not see)
                items["pan"] = dict(kind="pan", pose=(wx + jit(0.02), BOWL_Y0 + 0.36 + jit(0.02), BOWL_Z0 + 0.0005,
                                                      *yaw_quat(math.radians(-90) + jit(0.3))),
                                    soil=dict(kind="egg", load=1.0 + (jit(0.3) if seed else 0.0),
                                              soaked_s=float(rng.uniform(20.0, 240.0)) if seed else 120.0), wet=None)
        else:
            for k in range(4):
                x, y = cx + (-0.13 if k % 2 == 0 else 0.13) + jit(0.01), BOWL_Y0 + 0.16 + 0.25*(k//2) + jit(0.01)
                items[f"plate_{k}"] = dict(kind="plate", pose=(x, y, zdb(x) + dz, *on_board("soiled", jit(math.pi))),
                                           wet=dict(h=25e-6, surf=c_wash, T=T_wash, foam_patches=1))
            # the 24 x 24 in drainboard holds four 9 in plates and nothing else: the washer has set each bowl in the
            # well of a back plate (flat foot on the flat well), as on any crowded drainboard
            zw = P.ITEMS["plate"]["foot_h"] + P.ITEMS["plate"]["t"]
            for k in range(2):
                x, y = items[f"plate_{k + 2}"]["pose"][:2]
                items[f"bowl_{k}"] = dict(kind="bowl", pose=(x, y, zdb(x) + zw + 2*dz, *on_board("soiled", 0.0)),
                                          wet=dict(h=25e-6, surf=c_wash, T=T_wash, foam_patches=1))
            cx2 = sum(DRAINBOARDS["clean"])/2
            items["peg_rack"] = dict(kind="peg_rack", pose=(cx2, (BOWL_Y0 + BOWL_Y1)/2, board_z("clean", cx2) + dz,
                                                            *on_board("clean", 0.0)))
    elif station == "pre-rinse":
        fills = {}
        drains = {"prerinse": True}
        rx = (PRS_X0 + PRS_X1)/2
        ry = (PRS_Y0 + PRS_Y1)/2
        items["peg_rack"] = dict(kind="peg_rack", pose=(rx, ry, TABLE_Z + 0.012 + 0.0005, *yaw_quat(math.pi/2)))
        load = 1.0 + (rng.uniform(-0.3, 0.3) if seed else 0.0)
        for k in range(6):
            items[f"plate_{k}"] = dict(kind="plate", in_rack=("peg_rack", k), soil=dict(kind="tomato", load=load), wet=None)
    elif station == "prep-sink":
        fills = {}
        drains = {"prep": True}
        cx, cy = (VSINK_X0 + X)/2 - 0.05, (VDB_Y0 + VBOWL_Y0)/2
        items["colander"] = dict(kind="colander", pose=(cx + jit(0.02), cy + jit(0.02), V_RIM + DB_DROP*0.5 + 0.0005,
                                                         *yaw_quat(jit(math.pi))))
        for k in range(15):
            items[f"tomato_{k}"] = dict(kind="tomato", in_colander=k, soil=dict(kind="dust", load=1.0 + (jit(0.3) if seed else 0)))
        w = WORK
        items["clean_bowl"] = dict(kind="steel_bowl", pose=(0.5*(w["x0"] + w["x1"]), w["y1"] - 0.30, w["z"] + 0.0005, *yaw_quat(0.0)))
        items["cutting_board"] = dict(kind="cutting_board", pose=(0.5*(w["x0"] + w["x1"]), w["y0"] + 0.36, w["z"] + 0.0005,
                                                                  *yaw_quat(math.pi/2)))
    elif station == "can-wash":
        fills = {}
        items["can"] = dict(kind="can", pose=(6.10 + jit(0.05), 1.10 + jit(0.05), 0.0005, *yaw_quat(jit(math.pi))),
                            wet=dict(h=30e-6, surf=c_wash, T=20.0, foam_patches=3 if task == "rinse-garbage-can" else 0),
                            soil=dict(kind="debris", load=1.0) if task == "rinse-garbage-can" else None)
        if task == "flush-floor":
            items["can"]["pose"] = (6.75, 1.55, 0.0005, *yaw_quat(0.0))
            items["can"]["wet"] = None
    tools = {"pot-sink": ["faucet", "prsv", "mixing_valve", "lever_wastes"], "pre-rinse": ["prsv_r", "mixing_valve_r"],
             "prep-sink": ["v_faucet"], "can-wash": ["reel"]}[station]
    floor_soil = None
    if task == "flush-floor":
        floor_soil = dict(region=(BAY["x0"] + 0.3, BAY["x0"] + 1.3, 0.35, 1.45), surf=c_wash, h=0.6e-3, grit=0.02, load=1.0)
    return dict(task=task, preset=station, station=station, seed=int(seed), robot=ROBOT_START[station],
                spine=ROBOT_SPINE[station], items=items, glides="nylon",
                fills=fills, drains=drains, tools=tools, supply=supply, c_wash=c_wash, T_wash=T_wash, floor_soil=floor_soil,
                instruction=TASKS[task]["instruction"])
