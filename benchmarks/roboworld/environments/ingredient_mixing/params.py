"""Parameter ledger: every constant the ingredient-mixing simulation reads, with value, unit, label and basis.

Labels: read (a source opened for this environment and the number seen in it), search (a search summary only),
graph (read off a published figure by eye), derived (computed here from labelled values), estimate (engineering
estimate with its reasoning), open (no digital basis found; chosen and disclosed), authored (a design choice of this
environment), repo (qualified elsewhere in this repository), literature (classical result), measured (measured on this
machine), engine (a numerical setting of this simulator, not a property of a real thing).
`python -m ingredient_mixing.params` prints the table.
"""

import math

LEDGER = []


def _p(name, value, unit, label, basis):
    LEDGER.append(dict(name=name, value=value, unit=unit, label=label, basis=basis))
    return value


# ------------------------------------------------------------------ numerics
GRAVITY = _p("GRAVITY", 9.81, "m/s2", "literature", "gravity")
DT = _p("DT", 0.001, "s", "authored", "the FR3's 1 kHz control rate, the rigid tick of the repository's Franka environments")
CONTACT_TIMECONST = _p("CONTACT_TIMECONST", 0.004, "s", "repo", "rigid contact of every non-pad geom (barcode_scanning E17)")
CONTACT_SOLIMP = _p("CONTACT_SOLIMP", (0.95, 0.99, 0.001), "-", "repo", "MuJoCo contact impedance, the repository's setting")
SLIDE_REFERENCE = _p("SLIDE_REFERENCE", 1.0, "-", "derived",
                     "a friction row of a contact without a finger pad asks for no more deceleration than this times "
                     "what Coulomb friction can give the contact, mu (g + the normal row's reference) "
                     "(world._contact_rule). MuJoCo's own rule asks for b v_t whatever the friction, and a sliding body "
                     "is thrown off the surface on most ticks (E0). 0: MuJoCo's own rule")
STOP_SOLIMP = _p("STOP_SOLIMP", (0.99, 0.999, 0.5, 2.0), "-", "repo", "joint stops (barcode_scanning)")
SOLVER_ITERATIONS = _p("SOLVER_ITERATIONS", 50, "-", "repo", "barcode_scanning / wine_pouring / drilling")
SOLVER_IMPRATIO = _p("SOLVER_IMPRATIO", 100.0, "-", "repo", "drilling: friction rows stiff against the normal rows")
NOSLIP_ITERATIONS = _p("NOSLIP_ITERATIONS", 5, "-", "repo", "MuJoCo no-slip passes (wine_pouring, drilling)")
NOSLIP_TOLERANCE = _p("NOSLIP_TOLERANCE", 1e-6, "-", "repo", "as above")
SOLVER_LS_ITERATIONS = _p("SOLVER_LS_ITERATIONS", 200, "-", "repo",
                          "line-search passes of MuJoCo's main solve (drilling's setting; wine_pouring has 20). Measured "
                          "here on a whisk and a spatula let fall level onto the counter at 0.3 and 0.6 m/s: with 20 or "
                          "200 passes they are equally fast after landing (0.62 to 0.92 m/s) and neither comes off the "
                          "counter again; 1.5 s later the whisk still moves at 0.02 to 0.19 mm/s with 20 passes and at "
                          "under 0.001 with 200 (E0)")
SOLVER_LS_TOLERANCE = _p("SOLVER_LS_TOLERANCE", 1e-6, "-", "repo", "as above (drilling)")
PAD_TIMECONST = _p("PAD_TIMECONST", 0.005, "s", "repo", "Franka fingertip rubber pad (label_application, anchored)")
PAD_TIMECONST_REF_MASS = 0.0291
PAD_SLIP_RATE_DT = _p("PAD_SLIP_RATE_DT", 1.0, "1/tick", "engine",
                      "rate at which a pad's friction rows remove a slip, as a fraction per tick (as in drilling). "
                      "MuJoCo gives a contact's "
                      "friction rows the damping rate of its soft normal contact, 2/(dmax x time constant) = 421 /s "
                      "for the pad's 5 ms, so a held tool slid in the hand at (hand acceleration)/421 whatever its "
                      "mass and the grip force: 2.0-2.5 mm along the pads in 5 s of 60 mm circles at 0.8 s a turn, "
                      "25 mm at the whisk's tip in a minute's whisking with or without the mixture. The pads "
                      "carry MuJoCo's direct form: the normal stiffness of PAD_TIMECONST unchanged, the rate 1/dt")
FINGER_GEAR_SOLREF = _p("FINGER_GEAR_SOLREF", (2*DT, 1.), "s, -", "engine",
                        "the tie between a Franka Hand's two fingers, with FINGER_GEAR_SOLIMP (as in drilling). The "
                        "hand is one mechanism: Franka's description declares the second finger joint a mimic of the "
                        "first and the hand's interface has one width and one force. The robot import carried the "
                        "mimic into MuJoCo as a joint equality at MuJoCo's default softness (20 ms, impedance "
                        "0.9-0.95): a held tool and both fingers shifted 0.9 mm along the closing axis when the hand "
                        "was waved (one finger opening as the other closed)")
FINGER_GEAR_SOLIMP = _p("FINGER_GEAR_SOLIMP", (.9999, .9999, .001, .5, 2.), "-", "engine", "see FINGER_GEAR_SOLREF")
PAD_STICK = _p("PAD_STICK", dict(shear_ratio=1.0/3.0, reach_m=0.0002, lever_m=0.0042), "-, m, m", "estimate",
               "a rubber pad that grips holds a place, not only a speed. MuJoCo's friction rows remove relative "
               "velocity and carry no memory of where the surfaces stuck, so what the time stepping leaves over "
               "each tick (a free body moves along a tangent while a joint moves along its arc: a_c dt^2/2, drilling "
               "ledger PAD_SLIP_RATE_DT) adds up for as long as the hand moves: 1.0-1.3 mm in 5 s of brisk circles "
               "after the two settings above, whatever the tool's mass. Each pad contact has an anchor (the "
               "held body's pose in the finger's frame when they met); the friction rows of its contacts are given "
               "the offset from it as their position error, with the pad's normal stiffness times shear_ratio "
               "(G/E = 1/(2(1+nu)), nu = 0.5 for rubber: derived), so the pad springs back as rubber in shear does. "
               "The anchor follows once the offset exceeds reach_m, the pad's elastic reach before it slides "
               "(estimate: mu N / (G A / t) for a 2 mm pad of 3 cm2 at G about 1 MPa under 35 N is 0.19 mm; no "
               "datasheet value found: open), and a twist about the contact normal is held the same way over the "
               "pad's torsional lever (PAD_TORSIONAL_M). The Coulomb limit stays the solver's")
PAD_TORSIONAL_M = _p("PAD_TORSIONAL_M", 0.0042, "m", "repo",
                     "torsional friction of a pad contact about its normal: mu L/4, the twisting resistance of a line "
                     "contact as long as the pad (wine_pouring). Right where MuJoCo makes the contact in "
                     "one point (a round handle lying along the fingers). Where it makes two (a bar crossing the "
                     "pads, a knob's ridge) their tangential friction resists twisting as well, and the hand holds "
                     "up to three times a real pad's twist: a hanging 1 kg mixer was held against 1.03 N m about the "
                     "pinch where two 21 mm pads at 35 N hold 0.31 (open). Taking the "
                     "torsional rows away is not the repair: without them a held round handle spun 40-55 degrees "
                     "about its own axis in 5 s of brisk circles in the air (tried and reverted)")
CONTACT_GAP_M = _p("CONTACT_GAP_M", 0.0, "m", "repo", "box shapes (drilling e1b: a 2 mm gap gave false deep box-box contacts)")
MESH_CONTACT_GAP_M = _p("MESH_CONTACT_GAP_M", 0.002, "m", "repo", "hulls, capsules, cylinders (drilling)")
MAX_OBJECT_SPEED_M_S = _p("MAX_OBJECT_SPEED_M_S", 20.0, "m/s", "repo", "fault threshold (barcode_scanning)")
MAX_OBJECT_SPIN_RAD_S = _p("MAX_OBJECT_SPIN_RAD_S", 1000.0, "rad/s", "repo", "fault threshold; beaters exempt")

# ------------------------------------------------------------------ robot (FR3/Panda + Franka Hand)
FINGER_EFFORT_N = _p("FINGER_EFFORT_N", 35.0, "N", "repo", "Franka Hand 70 N continuous grasp, half per finger (drilling ledger)")
GRIP_FORCE_BAND_M = _p("GRIP_FORCE_BAND_M", 0.0005, "m", "repo", "hand-over between holding a force and a position (drilling)")
FRANKA_HAND_FINGER_SPEED = _p("FRANKA_HAND_FINGER_SPEED", 0.05, "m/s", "repo", "Franka Hand datasheet, per finger")
FR3_PAYLOAD_KG = _p("FR3_PAYLOAD_KG", 3.0, "kg", "repo", "FR3 datasheet rated payload at the flange (drilling ledger)")
FRANKA_HAND_KG = _p("FRANKA_HAND_KG", 0.73, "kg", "repo", "Franka Hand mass (drilling ledger); tool payload 2.27 kg")
LOAD_COM_MAX_M = _p("LOAD_COM_MAX_M", 0.5, "m", "authored", "farthest centre of mass a configured load may have from the TCP")
REFLEX_FILTER_S = _p("REFLEX_FILTER_S", 0.02, "s", "repo", "filtered external torque estimate (wine_pouring)")
REFLEX_RECOVERY_S = _p("REFLEX_RECOVERY_S", 1.0, "s", "repo", "targets refused until recovery (barcode_scanning)")
REFLEX_TORQUE_NM = _p("REFLEX_TORQUE_NM", (10.0, 20.0), "N m", "repo",
                      "libfranka reference collision thresholds: 10 N m holding, 20 N m while the arm accelerates "
                      "(wine_pouring). Stirring a stiff paste is contact the task wants, far below them: E9")
ROBOT_VELOCITY_FRACTION = _p("ROBOT_VELOCITY_FRACTION", 0.95, "-", "repo", "trajectory generator plans below the reflex (drilling)")
FR3_MAX_JOINT_ACC = _p("FR3_MAX_JOINT_ACC", (10.0,)*7, "rad/s2", "repo", "FR3 limit table (drilling ledger)")
FR3_MAX_JOINT_JERK = _p("FR3_MAX_JOINT_JERK", (5000.0,)*7, "rad/s3", "repo", "FR3 limit table (drilling ledger)")
PANDA_MAX_JOINT_ACC = _p("PANDA_MAX_JOINT_ACC", (15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0), "rad/s2", "repo", "FER limit table")
PANDA_MAX_JOINT_JERK = _p("PANDA_MAX_JOINT_JERK", (7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0), "rad/s3",
                          "repo", "FER limit table")
FRANKA_MAX_TORQUE_RATE = _p("FRANKA_MAX_TORQUE_RATE", 1000.0, "N m/s", "repo", "FR3/FER desired torque rate")
WRENCH_MAX_N = _p("WRENCH_MAX_N", 60.0, "N", "authored", "largest cartesian_wrench force accepted; mixing loads are below 20 N (E9)")
WRENCH_MAX_NM = _p("WRENCH_MAX_NM", 10.0, "N m", "authored", "largest cartesian_wrench moment accepted")
READY_POSE = (0.0, -0.8, 0.0, -2.6, 0.0, 1.9, 0.785, 0.04, 0.04)   # hands 0.44 m above the counter, 0.13 m over its front edge
# per arm: the same hand position with the elbow turned outward (away from the other arm). From the symmetric pose both
# elbows swung inward when the hands worked over one bowl and the forearms met (35-75 N, both reflexes)
READY_POSE_ARM = dict(right=(-0.55, -0.72, 0.62, -2.50, 0.42, 1.92, 0.50, 0.04, 0.04),
                      left=(0.55, -0.72, -0.62, -2.50, -0.42, 1.92, 1.07, 0.04, 0.04))
HAND_TCP_OFFSET = 0.1034
IK_MAX_EVALUATIONS = 300
IK_ROTATION_WEIGHT = 0.12
IK_SEED_WEIGHT = 1e-5

# ------------------------------------------------------------------ room, counter, station
# One bay of a teaching kitchen, 5.4 m along the wall with the counter run and 4.2 m deep: the robot's station at the
# counter, a prep table with stools behind it, dry-store shelving, a hand-wash basin, a window and a door.
ROOM = dict(x0=-2.4, x1=3.0, y0=-3.565, y1=0.635, z1=2.7)
# Room furniture: drawn by dressing.py, with static collision proxies of the drawn outlines from scene.furniture (boxes:
# nothing here moves). x and y ranges in the world frame, heights from the floor.
WINDOW = dict(y=(-2.15, -0.55), z=(0.95, 2.20), reveal=0.12)            # in the right wall (x = ROOM x1)
DOOR = dict(x=(1.75, 2.67), z=2.04)                                    # in the front wall (y0)
BOARD = dict(x=(-1.55, 0.25), z=(0.95, 2.00))                          # the lesson board on the front wall
WORK_TABLE = dict(x=(-1.65, -0.15), y=(-3.25, -2.55), z=0.90, top=0.040, shelf_z=0.22, leg=0.040)
STOOLS = ((-1.25, -2.20), (-0.55, -2.20))                              # stool centres (x, y)
STOOL = dict(seat_r=0.25, seat_z=0.75)                                 # the pinned bar stool's outline: 0.48 m across, 0.75 m high
SHELVING = dict(x=(-2.39, -1.89), y=(-3.37, -2.27), z=2.14)            # dry-store rack against the left wall (the pinned rack: 1.10 x 0.50 x 2.14 m)
BASIN = dict(x=(1.95, 2.45), y=(0.255, 0.635), z=(0.74, 0.86), tap_z=0.99)   # wall-hung hand-wash basin on the back wall
DISPENSER = dict(x=(2.07, 2.33), y=(0.535, 0.635), z=(1.30, 1.62))     # paper towels above the basin
BIN = dict(xy=(2.72, 0.33), r=0.16, h=0.62)                            # pedal bin in the corner beside the basin
APRON_HOOKS = dict(x=(0.40, 1.45), z=1.68)                             # a rail with aprons on the front wall
NOTICE = dict(y=(-1.75, -0.75), z=(1.05, 1.80))                        # cork notice board on the left wall (x = ROOM x0)
FIRST_AID = dict(y=-0.35, z=1.45, half=(0.06, 0.15, 0.15))             # first-aid cabinet on the left wall
STATION_CAMERA = dict(off_wall=0.055, z=1.60, fov=84., half=(0.030, 0.045, 0.024))   # the policy's camera on its wall bracket
# What stands on the counter beside the work: the packets and jars the recipe was measured from. Free bodies (scene.props):
# footprint half sizes (x, y) and height in metres, mass in kilograms, round: a body of revolution (a twelve-sided prism
# for contact). Masses are the packs' printed contents with an estimate for the container (label: estimate): a 1 kg
# paper bag of flour; 600 g of sugar in a 350 g glass jar; a 1 litre carton of milk; six 60 g eggs in a pulp tray;
# 500 ml of oil in glass; an empty half-litre glass jug; a 432 g cake mix, a 96 g pudding mix and a 400 g starch carton;
# a 400 g jar of mayonnaise; a 340 g squeeze bottle of ketchup; a folded cotton tea towel.
PROP = dict(
    flour_bag=dict(half=(0.055, 0.036), h=0.200, mass=1.02), sugar_jar=dict(half=(0.050, 0.050), h=0.160, mass=0.95, round=True),
    baking_tin=dict(half=(0.034, 0.034), h=0.090, mass=0.28, round=True), salt=dict(half=(0.024, 0.024), h=0.095, mass=0.30, round=True),
    milk_carton=dict(half=(0.036, 0.036), h=0.240, mass=1.06), egg_carton=dict(half=(0.078, 0.053), h=0.075, mass=0.42),
    oil_bottle=dict(half=(0.033, 0.033), h=0.265, mass=0.83, round=True), jug=dict(half=(0.062, 0.050), h=0.115, mass=0.45),
    scale=dict(half=(0.100, 0.075), h=0.030, mass=0.45), cake_box=dict(half=(0.068, 0.022), h=0.190, mass=0.47),
    pudding_box=dict(half=(0.043, 0.016), h=0.070, mass=0.11), mayo_jar=dict(half=(0.042, 0.042), h=0.125, mass=0.62, round=True),
    ketchup=dict(half=(0.031, 0.031), h=0.205, mass=0.40, round=True), relish_jar=dict(half=(0.036, 0.036), h=0.095, mass=0.48, round=True),
    starch_box=dict(half=(0.046, 0.028), h=0.150, mass=0.43), tub=dict(half=(0.085, 0.060), h=0.068, mass=0.52),
    towel=dict(half=(0.110, 0.075), h=0.022, mass=0.09, mu=0.6))
# per task: (kind, x from the station's centre, y from the counter's front edge, yaw in degrees). Two zones clear of the
# arms' ways to the vessel, the crock and the mixer: at the back left (x -0.95 .. -0.52) and, in the bake preset, at the
# back right (x 0.85 .. 1.40); on the hob everything stands to the left, away from the heat.
# (Placed clear of the towel that hangs from its rail to 76 mm above the counter between x -0.80 and -0.54, its front
# 50 mm off the wall: a prop standing inside the towel's slab is thrown over or pushed out when a scene is built.)
PROPS = {
    "dry-whisk": (("flour_bag", -0.86, 0.52, 8), ("sugar_jar", -0.71, 0.53, 0), ("baking_tin", -0.60, 0.50, 0), ("salt", -0.50, 0.57, 0),
                  ("scale", 1.05, 0.47, 4), ("towel", 1.30, 0.40, -12)),
    "muffin-fold": (("flour_bag", -0.87, 0.53, 6), ("sugar_jar", -0.73, 0.515, 0), ("baking_tin", -0.62, 0.50, 0),
                    ("milk_carton", 0.90, 0.54, 12), ("egg_carton", 1.07, 0.50, -8), ("oil_bottle", 1.24, 0.55, 0), ("jug", 1.32, 0.40, 30)),
    "pancake-whisk": (("flour_bag", -0.87, 0.53, -5), ("sugar_jar", -0.73, 0.515, 0), ("baking_tin", -0.61, 0.51, 0),
                      ("milk_carton", 0.90, 0.54, 10), ("egg_carton", 1.08, 0.50, 5), ("oil_bottle", 1.25, 0.55, 0), ("jug", 1.30, 0.40, 20)),
    "pudding-whisk": (("pudding_box", -0.62, 0.50, 20), ("milk_carton", -0.80, 0.54, -8), ("jug", 1.00, 0.48, 25), ("towel", 1.28, 0.42, 10)),
    "cake-mixer": (("cake_box", -0.80, 0.535, 4), ("egg_carton", 0.97, 0.50, -6), ("oil_bottle", 1.15, 0.55, 0), ("jug", 1.30, 0.44, 35)),
    "sauce-blend": (("mayo_jar", -0.82, 0.52, 0), ("ketchup", -0.68, 0.54, 0), ("relish_jar", -0.57, 0.50, 0), ("towel", 1.10, 0.42, 6)),
    "stir-sauce": (("milk_carton", -1.48, 0.54, 10), ("starch_box", -1.33, 0.55, -6), ("sugar_jar", -1.18, 0.53, 0), ("jug", -1.05, 0.42, 25)),
    "reheat-porridge": (("tub", -1.40, 0.50, 8), ("milk_carton", -1.20, 0.55, -10), ("towel", -1.02, 0.40, 14)),
}
COUNTER_Z = _p("COUNTER_Z", 0.914, "m", "search", "36 in finished counter height (NKBA figures on trade pages)")
COUNTER_DEPTH = _p("COUNTER_DEPTH", 0.635, "m", "search", "24 in carcass and 1 in overhang")
COUNTER_LENGTH = _p("COUNTER_LENGTH", 2.4, "m", "authored", "one bay of a teaching kitchen's counter run")
COUNTER_TOP_THICK = _p("COUNTER_TOP_THICK", 0.030, "m", "authored", "30 mm quartz top")
BACKSPLASH_Z = _p("BACKSPLASH_Z", 1.372, "m", "search", "18 in clear under the wall cabinets")
# Room fittings within reach of an arm or of a tool held at full stretch (1.26 m from an arm base): drawn by
# dressing.py and given static collision proxies of their drawn outline by scene.room() from these numbers.
SINK = dict(x=(-1.65, -0.95), bowl_inset=(0.13, 0.15), bowl_y=(0.10, 0.48), bowl_depth=0.180)
TOWEL_RAIL = dict(x=(-0.90, -0.40), posts=(-0.88, -0.42), off_wall=0.040, z=1.30, rod_r=0.006, towel_x=(-0.80, -0.54),
                  towel_drop=(0.20, 0.31))
SHELF_ITEMS = dict(canisters=((-0.80, 0.060, 0.200), (-0.665, 0.052, 0.160), (-0.55, 0.045, 0.125)),   # x, radius, height
                   canister_off_wall=0.115, jars=((0.98, 0.050, 0.180), (1.10, 0.046, 0.150)), jar_off_wall=0.11,
                   prep_bowls=(1.27, 0.085, 0.050), prep_off_wall=0.12)
SOCKETS = dict(x=(-0.15, 0.55), z=1.12, size=(0.146, 0.007, 0.086))
RECIPE_CARD = dict(x=-0.30, z_bottom=1.20, size=(0.148, 0.210))        # clipped flat to the tiles (no stand)
PLANT_XY = (2.15, -1.80)                                               # on the floor in the room's corner, out of reach
# The plant's pot as stacked upright cylinders (z from, z to, radius), measured on the drawn model's pot mesh (pinned
# potted_plant_02: rings of its outer wall at 0.128 m radius at the foot, 0.237 at the belly 0.20 m up, 0.205 at the
# neck, 0.232 at the rim 0.336 m up; the soil mounds to 0.341): each band takes the widest ring within it, so the
# proxy stands out of the drawn pot by up to 35 mm (the foot) and nowhere inside it. The leaves have no collider.
PLANT_POT = ((0.000, 0.046, 0.163), (0.046, 0.077, 0.187), (0.077, 0.112, 0.210), (0.112, 0.145, 0.226), (0.145, 0.262, 0.238),
             (0.262, 0.311, 0.217), (0.311, 0.341, 0.233))
# The fridge left of the sink unit, as dressing.py draws it: the cabinet on four feet, two doors in front of it (the
# gasket between), a bar handle on each door. scene.furniture gives each its box (a handle: the box round its bar and
# its two standoffs, so the 32 mm between bar and door is closed; a foot: the square round its 36 mm disc).
FRIDGE = dict(x=(-2.37, -1.67), y=(0.03, 0.63), z=(0.02, 1.80), door_y=(-0.032, 0.023), doors=((0.03, 0.725), (0.735, 1.80)),
              handle=dict(from_right=0.05, stand=0.040, r=0.008, over=0.012, z=((0.36, 0.66), (0.80, 1.25))),
              foot=dict(r=0.018, h=0.022, inset=0.05))
SHELF_BRACKET_X = (-0.875, -0.125, 0.625, 1.375)                       # wall plate, arm under the board, diagonal brace
SHELF_STRIP = dict(off_wall=0.175, size=(0.018, 0.008))                # the light strip's profile under the shelf
SINK_TAP = dict(behind_bowl=0.075, column=(0.024, 0.28), spout=(0.17, 0.243), spout_r=0.0125)
SINK_BOTTLE = dict(from_right=0.075, y=0.55, half=(0.05, 0.05), height=0.28)
ARM_BASE_Z = _p("ARM_BASE_Z", 0.95, "m", "measured", "cart deck height; 38-43 of 45 tool-tip poses reached")
ARM_SPACING = _p("ARM_SPACING", 0.56, "m", "measured", "the drilling cart's spacing")
ARM_SETBACK = _p("ARM_SETBACK", 0.20, "m", "measured", "arm bases in front of the counter edge")
ARM_YAW = math.pi/2
STATION_X = dict(bake=0.0, hob=0.75)       # cart centre along the counter per preset
VESSEL_Y = _p("VESSEL_Y", 0.30, "m", "authored", "vessel centre from the counter's front edge")
HOB_ZONE = (0.75, 0.32)

# ------------------------------------------------------------------ vessels
STEEL = 7900.0
BOWL5 = dict(rim_r=_p("BOWL5_RIM_R", 0.149, "m", "read", "Vollrath 69050 5-qt bowl: 11 3/4 in across"),
             depth=_p("BOWL5_DEPTH", 0.114, "m", "read", "Vollrath 69050: 4 1/2 in high"),
             base_r=_p("BOWL5_BASE_R", 0.045, "m", "authored", "flat foot of 90 mm"),
             wall=_p("BOWL5_WALL", 0.0006, "m", "search", "22-gauge stainless, 0.6 mm"),
             mass=_p("BOWL5_MASS", 0.55, "kg", "search", "Winco MXBH-500 class heavy-duty 5-qt bowl"))
BOWL2 = dict(rim_r=_p("BOWL2_RIM_R", 0.100, "m", "search", "200 mm glass mixing bowl, 1.5 L class"),
             depth=_p("BOWL2_DEPTH", 0.081, "m", "search", "as above"),
             base_r=_p("BOWL2_BASE_R", 0.040, "m", "authored", "flat foot"),
             wall=_p("BOWL2_WALL", 0.004, "m", "estimate", "pressed glass bowl wall"),
             mass=_p("BOWL2_MASS", 0.60, "kg", "open", "no weight found for the bowl; glass volume x 2500 kg/m3 of this shape"))
PAN = dict(radius=_p("PAN_RADIUS", 0.076, "m", "search", "All-Clad D3 1.5-qt saucepan, 152 mm inside the rim"),
           depth=_p("PAN_DEPTH", 0.089, "m", "search", "as above"),
           wall=_p("PAN_WALL", 0.0026, "m", "search", "tri-ply, 2.6 mm"),
           core=_p("PAN_AL_CORE", 0.0017, "m", "search", "aluminium core of the tri-ply"),
           mass=_p("PAN_MASS", 0.91, "kg", "search", "as above, without lid"),
           handle_len=_p("PAN_HANDLE_LEN", 0.190, "m", "search", "stay-cool hollow handle"),
           handle_r=0.011, corner=0.012)
PITCHER = dict(r_bottom=0.050, r_top=_p("PITCHER_R_TOP", 0.0575, "m", "estimate", "115 mm glass pitcher of 1.2 L class"),
               depth=_p("PITCHER_DEPTH", 0.170, "m", "derived", "height that gives 1.5 L brimful with these radii"),
               wall=0.004, mass=_p("PITCHER_MASS", 0.60, "kg", "estimate", "glass pitcher of this size"))
POT = dict(radius=_p("POT_RADIUS", 0.090, "m", "estimate", "ceramic utensil crock with a three-way divider, 180 mm across inside"),
           depth=_p("POT_DEPTH", 0.165, "m", "estimate", "as above"), wall=0.006,
           mass=_p("POT_MASS", 1.6, "kg", "estimate", "stoneware crock of this size"))
MAT = dict(size=(0.30, 0.30, 0.002), mass=_p("MAT_MASS", 0.20, "kg", "estimate", "300 mm silicone baking mat"))

# ------------------------------------------------------------------ friction (Coulomb, pairwise by material)
MU = dict(default=_p("MU_DEFAULT", 0.4, "-", "estimate", "unlisted pairs"),
          pad=_p("MU_PAD", 0.8, "-", "repo", "Franka fingertip rubber (catalog profile; barcode_scanning)"),
          steel_quartz=_p("MU_STEEL_QUARTZ", 0.25, "-", "estimate", "polished steel on polished stone, dry; open"),
          silicone=_p("MU_SILICONE", 0.9, "-", "estimate", "silicone mat against steel and stone"),
          steel_steel=_p("MU_STEEL_STEEL", 0.2, "-", "estimate", "whisk wire or spoon on a steel bowl or pan, dry"),
          wood_steel=_p("MU_WOOD_STEEL", 0.3, "-", "estimate", "wooden spoon on steel"),
          glass=_p("MU_GLASS", 0.3, "-", "estimate", "glass on stone, steel on glass"))

# ------------------------------------------------------------------ hand tools
WHISK = dict(length=_p("WHISK_LENGTH", 0.292, "m", "read", "OXO 11 1/2 in balloon whisk"),
             handle_len=_p("WHISK_HANDLE_LEN", 0.117, "m", "read", "as above"),
             handle_r=_p("WHISK_HANDLE_R", 0.014, "m", "estimate", "soft handle 28-30 mm thick"),
             cage_len=_p("WHISK_CAGE_LEN", 0.175, "m", "read", "as above"),
             cage_r=_p("WHISK_CAGE_R", 0.032, "m", "read", "64 mm cage"),
             wire=_p("WHISK_WIRE", 0.0014, "m", "search", "1.4 mm stainless wire; not confirmed on a datasheet (open)"),
             loops=_p("WHISK_LOOPS", 5, "-", "search", "ten wires"),
             mass=_p("WHISK_MASS", 0.10, "kg", "read", "as above"),
             tip_stiffness=_p("WHISK_TIP_STIFFNESS", 440.0, "N/m", "derived",
                              "ten legs of 44 N/m each (1.4 mm wire, 170 mm cantilever, E 193 GPa) acting "
                              "together as one sprung cage; the cage's two bending axes at the ferrule"))
SPATULA = dict(length=_p("SPATULA_LENGTH", 0.279, "m", "search", "GIR 11 in spatula"),
               handle_r=0.008, blade=(0.060, 0.100), blade_t=_p("SPATULA_BLADE_T", 0.005, "m", "estimate", "8 to 2 mm taper, mean"),
               mass=_p("SPATULA_MASS", 0.09, "kg", "estimate", "silicone over a nylon core"),
               tip_stiffness=_p("SPATULA_TIP_STIFFNESS", 250.0, "N/m", "estimate",
                                "blade tip under a normal load: a 60 x 50 x 4 mm silicone-over-nylon cantilever; open"))
SPOON = dict(length=_p("SPOON_LENGTH", 0.305, "m", "read", "Choice 12 in wooden spoon"),
             handle_r=_p("SPOON_HANDLE_R", 0.00475, "m", "read", "9.5 mm handle"), bowl=(0.051, 0.079),
             bowl_t=_p("SPOON_BOWL_T", 0.012, "m", "estimate",
                       "the bowl from its back to its lip: the hollow and 5 mm of wood under it (two cells of the "
                       "mixture's 2.5 mm grid, the least it resolves: the mixture then meets the boxes the rigid world "
                       "does). The seller gives no thickness"),
             bowl_depth=_p("SPOON_BOWL_DEPTH", 0.007, "m", "estimate",
                           "depth of the hollow. The seller lists the bowl's 2 x 3 1/8 in outline and no depth (none "
                           "found). 7 mm inside the 5 mm lip holds 16 ml level, about a "
                           "tablespoon (14.8 ml), taken here as what a cooking spoon of this size lifts: unsourced, open"),
             bowl_rim=_p("SPOON_BOWL_RIM", 0.005, "m", "estimate",
                         "width of the lip round the hollow: two grid cells, as the floor (a real lip is 3-4 mm)"),
             mass=_p("SPOON_MASS", 0.032, "kg", "read", "as above"))
STIRRER = dict(length=_p("STIRRER_LENGTH", 0.300, "m", "authored", "flat-edged wooden spatula for a pan"),
               handle_r=0.006, blade=(0.040, 0.090), blade_t=0.006,
               mass=_p("STIRRER_MASS", 0.05, "kg", "estimate", "beech, this size"))
THERMOMETER = dict(length=_p("THERMOMETER_LENGTH", 0.235, "m", "estimate", "pen-type digital probe thermometer: a 110 mm body and a 125 mm stem"),
                   handle_len=0.110, handle_r=0.008, stem_r=0.00175,
                   mass=_p("THERMOMETER_MASS", 0.03, "kg", "estimate", "pocket digital thermometer with its cell"),
                   tau_s=_p("THERMOMETER_TAU", 1.0, "s", "search",
                            "time constant of its tip in food: an instant-read thermometer of the ThermoPop class gives "
                            "its reading in 2-4 s, about three time constants (ThermoWorks' figures, seen in search "
                            "results only; the Thermapen ONE states 0.2 s)"),
                   tau_air_s=15.0, sense_r=0.006)
ROD = dict(length=0.300, r=0.003, end_r=0.012, mass=_p("ROD_MASS", 0.03, "kg", "estimate", "6 mm steel bar spoon"))
GRIP_SEAT = _p("GRIP_SEAT", 0.037, "m", "repo", "handle butt against the palm, pads this far from it (drilling)")
GRIP_SLEEVE = _p("GRIP_SLEEVE", dict(whisk=0.012, spatula=0.009, spoon=0.007, stirrer=0.007, rod=0.004), "m", "authored",
                 "the `sleeves` variant only: half width of a square-section non-slip sleeve on each tool's handle "
                 "(a parallel hand's flat pads then meet a flat face). The tools as "
                 "listed have the round handles they are sold with (handle_r of each tool): the pads meet them "
                 "along a line, and a tool turning about that line under a stirring load is this hand's result")

# ------------------------------------------------------------------ hand mixer (KitchenAid 5-speed class)
MIXER = dict(mass=_p("MIXER_MASS", 1.0, "kg", "read", "KitchenAid's specification sheet for the KHM5AP (5 Speed Ultra Power), opened: "
                        "net weight 2.3 lb."),
             body=(0.188, 0.088, 0.110),
             beater_r=_p("MIXER_BEATER_R", 0.0222, "m", "read",
                         "'Beater Head Diameter: 1.75 in' (44.5 mm) for the KHM512 in America's Test Kitchen's review of "
                         "hand mixers (opened)"),
             beater_len=_p("MIXER_BEATER_LEN", 0.070, "m", "estimate", "beater cage length; open"),
             beater_gap=_p("MIXER_BEATER_GAP", 0.040, "m", "estimate", "axis spacing, cages intermeshing by 5 mm"),
             shaft_len=0.075, strip=_p("MIXER_STRIP", 0.004, "m", "estimate", "flat strip of the Turbo Beater; open"),
             strips=4,
             speeds_rpm=_p("MIXER_SPEEDS", (0.0, 288.0, 493.0, 698.0, 902.0, 1107.0), "rpm", "read",
                           "the five settings' beater speeds. The lowest and the highest are measured: 'Lowest Speed: "
                           "288 rpm', 'Highest Speed: 1,107 rpm' for the KitchenAid 5-Speed Ultra Power Hand Mixer "
                           "(model KHM512) in America's Test Kitchen's review of hand mixers (the review's table of "
                           "what was tested; opened; whether running free or in a batter is not said). The three settings between are "
                           "not published by the maker or the review: even steps (derived)"),
             stall_nm=_p("MIXER_STALL", 1.4, "N m", "estimate",
                         "the torque at which the beaters stop. The maker's data sheet of the 5-speed mixer sold in "
                         "Europe gives 'Wattage 85 W, Motor DC (Direct Current)' (KitchenAid 5KHM5110, opened) "
                         "and no torque; 85 W at 580 rpm is 1.4 N m, an estimate"),
             detent_pitch=_p("MIXER_DETENT_PITCH", 0.005, "m", "open", "slide travel per setting"),
             detent_force=_p("MIXER_DETENT_FORCE", 3.0, "N", "estimate",
                             "force to leave a detent of the speed switch. The mixer's own switch is not published; a "
                             "slide switch of the kind small appliances use is rated 'Operating force: 250 +/- 100 g' "
                             "over a 3 mm stroke, 1.5 to 3.4 N (Multicomp slide switch, top actuating, data sheet of "
                             "17/09/12, opened): the 3 N here lies inside that"),
             handle_r=0.014)

# ------------------------------------------------------------------ hob
HOB = dict(levels_w=_p("HOB_LEVELS", (0.0, 100.0, 180.0, 260.0, 340.0, 420.0, 500.0, 580.0, 660.0, 740.0, 820.0, 900.0, 1000.0,
                                      1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0), "W", "read",
                       "Duxtop BT-200DZ / 9600LS / 9610LS user manual, power mode: 'Output: 100-1800 watts', 'Power "
                       "Levels: 0.5-10', twenty levels in half steps with these wattages (0.5 is 100 W, 5.0 is 820 W, "
                       "6.0 is 1000 W, 10 is 1800 W), and off. The knob has one position per level (the unit itself "
                       "has +/- keys: the rotary knob is this environment's, authored)"),
           ring_r=_p("HOB_RING_R", 0.045, "m", "read", "the 'circle of maximum heat' of the portable-hob review"),
           ring_w=0.030, detent_deg=15.0,
           detent_nm=_p("HOB_DETENT_NM", 0.035, "N m", "read",
                        "torque to leave a knob detent: the top of the range a maker of range and oven controls gives for "
                        "its detented rotary controls, 'For all detents versions of more than 13 detents the detent "
                        "torque will be 0.5 to 3.5 Ncm' (Piher PTC-15 data sheet, applications 'ovens, ranges, "
                        "dishwashers ...'; opened). A sturdier push-to-turn hob control of the "
                        "same maker was seen at 9 to 15 N cm in search summaries only"),
           knob_r=0.022, knob_h=0.028,
           loss_w_m2k=_p("HOB_LOSS", 10.0, "W/m2/K", "estimate", "natural convection and radiation of a pan wall and a food surface"),
           film=_p("HOB_FILM", 0.00015, "m", "estimate",
                   "food film on the pan bottom that dries out before it burns on. No measurement of it was found. The "
                   "outcome it sets, with the boiling law below, is how long a hot sauce may be left alone, "
                   "checked against one cook's figure, milk's 'grace period "
                   "of 15 seconds at 180 degrees Fahrenheit' (Pastry Chef Online as quoted by Tasting Table; a kitchen "
                   "claim, not a measurement). The value was not changed to meet it"),
           burnt_k=_p("HOB_BURNT_K", 0.37, "W/m/K", "read",
                      "thermal conductivity of the burnt-on film, which lies between the plate and the food as a "
                      "layer of the film's thickness. From the fouling layer that 15 % non-fat dry milk left in a "
                      "heated stainless tube (140 C, 60 min): the estimated conductivity fell 'from ... 0.63 to "
                      "0.37' W/(m K) as the layer grew; its thickness was 0.56 +/- 0.07 mm and its thermal resistance "
                      "1.53e-3 +/- 2.0e-4 (m2 K)/W, and 0.56e-3 / 1.53e-3 = 0.37 (Cantarero Rivera, Mishra, Ozadali "
                      "and Benyathiar, Sensors 21(4):1271, 2021, 'Real-Time Detection of Fouling-Layer with a "
                      "Non-Intrusive Continuous Sensor (NICS) during Thermal Processing in Food Manufacturing'; full "
                      "text opened through Europe PMC, PMC7916806). The same paper's heavy-whipping-cream layer "
                      "(0.58 mm, 5.95e-3 (m2 K)/W) gives 0.10 W/(m K): a fatty deposit, not used; with it the plate "
                      "under a burnt patch would run about four times as far above the food. A deposit grown under "
                      "liquid at 140 C stands in for a film dried and scorched on a pan: transferred"),
           latent=_p("LATENT", 2.26e6, "J/kg", "literature", "water"),
           efficiency=_p("HOB_EFFICIENCY", 0.84, "-", "read",
                         "share of the level's watts that reaches the pan. Induction cooking tops transfer energy with "
                         "'approximately 85% efficiency' (ENERGY STAR, 2021 Residential Induction Cooking Tops; opened); "
                         "the maker's 83 % was seen on resellers' pages only. An outcome to hold it against: two "
                         "induction cookers heating water in a pot from 21 to 93 C put 74.9 to 77.6 % of the "
                         "electricity into the water (Sweeney and others, ACEEE Summer Study 2014, table 2; opened). "
                         "That is after the pot's own heating and its losses, which this world models itself: the "
                         "hob records write the same share (`share_of_the_electricity_in_the_food`)"),
           boil_b=_p("HOB_BOIL_B", 137.0, "W/m2/K3", "read",
                     "nucleate boiling on the pan bottom, q = B (T_w - T_sat)^3: Rohsenow's correlation, "
                     "q = mu h_fg sqrt(g (rho_l - rho_v) / sigma) (c_p dT / (C_sf h_fg Pr^n))^3, with water's "
                     "properties at 1 atm (mu 279e-6 Pa s, h_fg 2257 kJ/kg, c_p 4.217 kJ/kg/K, Pr 1.76, sigma 58.9 "
                     "mN/m, specific volumes 1.044e-3 and 1.679 m3/kg: A. A. Al-Tabbakh, Worked Examples in Boiling "
                     "and Condensation, Al-Mustansiriyah University, example 1) and the surface constant of water on "
                     "mechanically polished stainless steel, n = 1: the table has 0.0132 (the same author's lecture "
                     "2, Boiling Heat Transfer Coefficients Calculations, 2021, table 1: 0.0133 chemically etched, "
                     "0.0132 mechanically polished, 0.008 ground and polished) and a second course's worked examples "
                     "use 0.013 (Majid, Boiling-2). B is 136.9 with 0.013 and 130.8 with 0.0132; the worked example's "
                     "own answer is reproduced (10.75 C excess at 159 kW/m2 with 0.0133). At this hob's 46 kW/m2 the two "
                     "constants put the plate 7.0 and 7.1 C over boiling. Measured on water, used for a sauce: "
                     "transferred"),
           simmer_c=_p("HOB_SIMMER_C", 96.0, "C", "read",
                       "mean temperature of the food below which no vapour made on the pan's bottom leaves (the "
                       "bubbles collapse in the food above them); from there to 100 C the share that leaves rises to "
                       "all of it. From a measured heating curve: 5 litres of water in an open stainless pot on an "
                       "electric range, heated from room temperature at about 1.4 kW, with a film of oil on the water "
                       "so that nothing evaporates from its surface: 'the run done with the oil film shows a constant "
                       "slope until boiling is reached' (A. Berick, Heat Losses In A Cook Pot at Constant Temperature, "
                       "Aprovecho Research Center, figure 7; opened). On the figure the oiled "
                       "pot's curve is straight to within a few degrees of its plateau: the slope, read off the "
                       "plot's pixels, holds to about 96 C and is gone by the plateau. A second source points the same "
                       "way for single bubbles: steam bubbles rising in water at 1 bar reached the surface at 5 K "
                       "subcooling and not at 9.3 K or more (Isikan, PhD thesis, University of Strathclyde, 1986: a "
                       "pool 40 mm deep, bubbles from an orifice; read in a summary only). "
                       "The bend of an open pan's heating curve "
                       "from about 85 C on is the surface's evaporation (heat.EVAPORATION_*), not this"),
           scrape_j_m2=_p("HOB_SCRAPE", 10.0, "J/m2", "read",
                          "work to scrape a burnt-on film off the pan per area: the upper end of the apparent "
                          "adhesive strength of baked whey-protein deposits on stainless steel, 0-10 J/m2 by "
                          "micromanipulation (Liu, Christian, Zhang and Fryer, Int. Dairy J. 16, 2006; abstract "
                          "opened). A starch-milk film was not measured there: transferred, open"),
           cutout_c=_p("HOB_CUTOUT_C", 238.0, "C", "read",
                       "Duxtop BT-200DZ / 9600LS / 9610LS user manual, error E2: 'Cooking surface temperature exceeds "
                       "460 degrees F (238 C) limit and the unit shuts down automatically'"),
           sensor_tau=_p("HOB_SENSOR_TAU", 8.0, "s", "estimate",
                         "the sensor is 'located underneath the glass' (same manual): the lag of 4 mm of glass-ceramic, "
                         "0.4 L2/alpha with alpha about 7.7e-7 m2/s (k 1.6 W/m/K, 2600 kg/m3, 800 J/kg/K, from memory of "
                         "the Ceran data sheet)"))
STEEL_K, AL_K = 16.0, 237.0

# ------------------------------------------------------------------ recipes (grams)
RECIPES = dict(
    dry_mix=dict(flour=240.0, sugar=25.0, baking_powder=12.0, salt=3.0),
    muffin_wet=dict(milk=227.0, egg=50.0, oil=62.0),
    pancake_wet=dict(milk=283.0, egg=100.0, oil=43.0), pancake_dry=dict(flour=180.0, sugar=25.0, baking_powder=10.0, salt=3.0),
    pudding=dict(milk=454.0, mix=96.0),
    cake=dict(mix=432.0, water=237.0, oil=99.0, egg=150.0),
    sauce=dict(mayonnaise=120.0, ketchup=60.0, relish=30.0),
    sauce_honey=dict(honey=80.0, mayonnaise=100.0, mustard=30.0),
    pet_meal=dict(canned=300.0, supplement=5.0),
    stir_sauce=dict(milk=500.0, starch=30.0, sugar=50.0),
    porridge=dict(porridge=600.0),
    lemonade=dict(water_ml=700.0, lemon_ml=150.0, syrup_ml=150.0))
EGG_WATER = _p("EGG_WATER", 0.76, "-", "literature", "whole egg: 76 % water, 10 % fat, 13 % protein")
EGG_FAT = 0.10


if __name__ == "__main__":
    for e in LEDGER:
        print(f"{e['name']:24s} {str(e['value']):40s} {e['unit']:8s} {e['label']:10s} {e['basis']}")
    from .ledger_mixture import entries          # the food's constants (materials.py, task.py, heat.py, continuum.py)
    print()
    for e in entries():
        print(f"{e['name']:28s} {str(e['value'])[:60]:62s} {e['unit']:22s} {e['label']:11s} {e['basis']}")
