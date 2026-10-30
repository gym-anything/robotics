"""Parameter ledger: every physical constant the drilling simulation reads, with value, unit, label and basis.

Labels: read (primary source opened), derived (computed here from labelled values), estimate (engineering estimate;
swept in the sensitivity probes), open (no digital basis; chosen, disclosed and swept), authored (a design choice of
this environment), repo (a value qualified elsewhere in this repository), literature (classical result), measured
(measured on this machine), engine (a setting or a reading of this simulator, not a property of a real tool).
Sources are in the report's reference list. `python -m drilling.params` prints the table (the report's appendix
"Parameter ledger" is generated from it).
"""

import math

LEDGER = []


def _p(name, value, unit, label, basis):
    LEDGER.append(dict(name=name, value=value, unit=unit, label=label, basis=basis))
    return value


# ------------------------------------------------------------------ numerics
GRAVITY = _p("GRAVITY", 9.80665, "m/s2", "literature", "standard gravity")
DT = _p("DT", 0.001, "s", "authored", "the FR3's 1 kHz control rate (barcode_scanning, wine_pouring); E16 compares 0.5 ms")
SOLVER_ITERATIONS = _p("SOLVER_ITERATIONS", 50, "-", "repo", "barcode_scanning / wine_pouring Config")
SOLVER_IMPRATIO = _p("SOLVER_IMPRATIO", 100.0, "-", "engine",
                     "how much harder the main solve's friction is than its normal contact (MuJoCo impratio, elliptic "
                     "cones). MuJoCo's friction carries a load only with a slip "
                     "acceleration, (1 - d)/d x inverse inertia / impratio x the load, which the no-slip pass then "
                     "removes with the normal forces held fixed. With 3 the main solve had the clamped timber stack "
                     "accelerating before that pass (top board 5.8 m/s2 and 79 rad/s2, the straps 1,200 rad/s2: same "
                     "state solved with and without the pass), so a light part lying on the stack got normal forces "
                     "for a support that then did not move, and hopped: a carriage bolt on the clamped boards slid "
                     "13 mm a minute, an M8 bolt on the bar clamped in the vise crept 1.6 mm a minute. With 100 the "
                     "top board's residual is 0.03 m/s2 and 0.7 rad/s2 (together with "
                     "HOLD_DOWN_BEARING_ROLLS). Chosen inside a measured plateau, not at an edge: the same six "
                     "set-down cases rest at 30, 50, 100 and 300; at 10 nothing changes and at 1000 the boards creep "
                     "0.23 mm in 6 s. A numerical setting with no material meaning: real shear and normal contact "
                     "stiffness are of one order, and what the larger ratio removes is slip under a held load")
NOSLIP_ITERATIONS = _p("NOSLIP_ITERATIONS", 100, "-", "repo", "barcode_scanning")
NOSLIP_TOLERANCE = _p("NOSLIP_TOLERANCE", 1e-10, "-", "repo", "barcode_scanning: default tolerance left grip creep")
SOLVER_LS_ITERATIONS = _p("SOLVER_LS_ITERATIONS", 200, "-", "engine",
                          "line-search iterations of MuJoCo's Newton solve. With a coarse search (20 iterations, "
                          "tolerance 0.01) the main solve returned after one iteration with contact forces its own "
                          "accelerations did not contain, and the no-slip pass turned them into motion: a hammered "
                          "anchor sliding at 1.4 m/s along its hole got 192, 96 and 108 N from three wall boxes it was "
                          "0.00-0.08 mm into, where the converged solve gives 6 N, and was thrown 4.6 mm sideways in one "
                          "tick; its washer meeting the wall got 7.3 kN where the converged solve gives 1.6 kN (same "
                          "state solved 22 ways: the same answer comes from the CG and PGS solvers, from impratio 1 or "
                          "2 and from the pyramidal cone). The contact model is the same (elliptic cone, no-slip "
                          "pass); only the solve is finished")
SOLVER_LS_TOLERANCE = _p("SOLVER_LS_TOLERANCE", 1e-6, "-", "engine",
                         "line-search tolerance of the Newton solve: see SOLVER_LS_ITERATIONS")
SOLVER_CCD_TOLERANCE = _p("SOLVER_CCD_TOLERANCE", 1e-6, "m", "engine",
                          "tolerance of MuJoCo's convex collision routine, which places the contacts of every cylinder "
                          "with a box, a cylinder or a hull: the engine's default, kept. Its contact points move by "
                          "about this much from tick to tick, and a round shank lying across two box rails rocked on "
                          "them at 27 Hz (four loose anchors in the wall kit, 0.116 mm/s against the rest test's "
                          "0.1 mm/s). Tightening it was tried and NOT adopted: at 1e-9 m with 200 iterations the anchors "
                          "lay at 0.0003 mm/s, but a hammered anchor then seated with 12 kN of tension it had not "
                          "been given (E9 clean), the hole brush rolled 26 mm in its 30 s of rest, and the loose vise "
                          "moved at 0.21 mm/s (rest test): flat faces meeting through this routine have no "
                          "unique contact point, and a finer tolerance moves it elsewhere. The rests were changed "
                          "instead (stations.py: rounded bearing points, whose contact with a cylinder is analytic)")
SOLVER_CCD_ITERATIONS = _p("SOLVER_CCD_ITERATIONS", 35, "-", "engine",
                           "iterations allowed to the convex collision routine (the engine's default): see "
                           "SOLVER_CCD_TOLERANCE")
CONTACT_TIMECONST = _p("CONTACT_TIMECONST", 0.004, "s", "repo", "rigid contact of every non-pad geom (barcode_scanning E17)")
CONTACT_SOLIMP = _p("CONTACT_SOLIMP", (0.95, 0.99, 0.001), "-", "repo",
                    "MuJoCo contact impedance (d0, dmax, width) of every geom without its own entry: the engine's "
                    "numerical softness, not a material property (barcode_scanning's setting)")
STOP_SOLIMP = _p("STOP_SOLIMP", (0.99, 0.999, 0.5, 2.0), "-", "repo",
                 "hard joint stops (barcode_scanning: d0, dmax, midpoint, power; width per joint)")
SOLID_SOLIMP = _p("SOLID_SOLIMP", (0.97, 0.99, 0.0005), "-", "engine",
                  "metal workpiece faces: with STIFF_SOLIMP a loose 6 mm bar under a 100 N punch push rocked at 15 rad/s "
                  "on the bench and was thrown off at 22 m/s; 0.97/0.99 holds it (0.04 rad/s) "
                  "and a 13 g shank or a 0.35 kg block still sinks under 0.01 mm at rest")
STIFF_SOLIMP = _p("STIFF_SOLIMP", (0.99, 0.999, 0.0005), "-", "engine",
                  "hardened steel clamping faces (vise jaws): with the default a clamped bar sank 0.7 mm into the jaw "
                  "under 3.8 kN. A numerical stiffness, chosen as the hardest the 1 ms step holds")
WALL_SOLIMP = _p("WALL_SOLIMP", (0.99, 0.999, 0.0005), "-", "engine",
                 "metal hole walls carrying a fastener. TUNED on E10, not grounded in a material property: of three "
                 "values tried with the chamfer cone, d0 = 0.999 threw a bolt jammed between "
                 "a wall and a chamfer at 5.7 m/s (a solver jump, 20 N to 0 N in one tick); 0.995 and 0.99 jam 1.1 and "
                 "1.2 mm calmly and pass 0.9 mm. The independent observable it is compared with is geometric: two 9 mm "
                 "holes pass an 8 mm bolt up to 1.0 mm out of line, and E10's pass/jam boundary sits at 1.0-1.1 mm; "
                 "the 0-0.1 mm band is the contact's softness")
PAD_TIMECONST = _p("PAD_TIMECONST", 0.005, "s", "repo", "Franka fingertip rubber pad (label_application, anchored)")
PAD_TIMECONST_REF_MASS = 0.0291                 # kg, FR3 finger link that time constant was anchored on
PAD_SLIP_RATE_DT = _p("PAD_SLIP_RATE_DT", 1.0, "1/tick", "engine",
                      "rate at which a pad's friction rows remove a slip, as a fraction per tick. MuJoCo gives the "
                      "friction rows of a contact the damping rate of its soft normal contact, b = 2/(dmax x time "
                      "constant) = 421 /s for the pad's 5 ms, and neither solve nor the no-slip pass knows that a "
                      "rotating hand's contact points accelerate inwards (the Jacobian's rate of change is not in "
                      "the constraint). A part pinched in a swinging hand therefore slid outwards by a_c dt (1/b - "
                      "dt/2) per tick, a_c the centripetal acceleration, whatever its mass and the grip force: the "
                      "centre punch 0.442 mm on a 0.4 rad swing of joint 1 and as much again on the way back, 0.118 "
                      "and 0.039 mm on swings of 1 and 3 s. The pads carry MuJoCo's direct form: the "
                      "normal stiffness of PAD_TIMECONST unchanged, the rate 1/dt, the largest MuJoCo's own "
                      "stability rule allows (a time constant of two ticks). The same swing then moves the punch "
                      "0.118 mm: the remainder a_c dt^2/2 is the time stepping's (a free body moves along a "
                      "tangent for a tick, a joint along its arc) and halves with the tick. The pad's normal "
                      "contact is over-damped by this (ratio 2.4)")
FINGER_GEAR_SOLREF = _p("FINGER_GEAR_SOLREF", (2*DT, 1.), "s, -", "engine",
                        "the tie between a Franka Hand's two fingers, with FINGER_GEAR_SOLIMP. The hand is one "
                        "mechanism: Franka's description declares the second finger joint a mimic of the first "
                        "(franka_description hand.xacro; fr3_franka_hand.urdf in Newton's assets, line 358) and the "
                        "hand's interface has one width and one force. The robot import carried that mimic into "
                        "MuJoCo as a joint equality at MuJoCo's default softness (time constant 20 ms, impedance "
                        "0.9-0.95): at that softness a pinched part sits on a spring of 2.7 N/mm along the "
                        "closing axis. Measured: 5 N on the pinched centre punch moved it and both fingers 1.83 mm "
                        "the same way (6.54/6.54 mm became 8.35/4.73) and back when removed; a hammer carried "
                        "through a swing moved 0.6 mm with both fingers. The tie here is as hard as the drill press's rack and the vise's screw. "
                        "The force a finger meets alone is then the whole hand's (2 x the per-finger figure), as "
                        "with one motor")
FINGER_GEAR_SOLIMP = _p("FINGER_GEAR_SOLIMP", (.9999, .9999, .001, .5, 2.), "-", "engine", "see FINGER_GEAR_SOLREF")
DRIVEN_DWELL = _p("DRIVEN_DWELL", dict(seconds=0.02, band_m=5e-5), "s, m", "authored",
                  "how the world records where an anchor was driven to (the evaluator's embedment): the deepest its "
                  "tip has stayed within band_m for seconds, not the deepest it has been. In the blow that seats the "
                  "washer the stud goes 2.2-2.4 mm past its place for a few milliseconds and springs back (the "
                  "washer's contact gives 1.2-1.5 mm at a 6 kN peak, E20's seating table): read as the deepest "
                  "point, four correctly driven and set anchors of clean-and-set were scored at 97.05-97.21 mm, "
                  "outside 95 +- 2, while they stood at 94.6-94.9 mm")
BOLT_END_FEET = _p("BOLT_END_FEET", dict(circle_r_m=0.0045, foot_r_m=0.001, proud_m=0.0003), "m", "engine",
                   "the carriage bolt's flat end is carried by three points: spheres of foot_r_m on a circle of "
                   "circle_r_m (inside the chamfered face), proud_m proud of it. A bolt dropped through the bored "
                   "boards stands on its end in the backer's hole, its head 18 mm above the top board. Flat on the "
                   "hole's flat floor it stood on contact points that MuJoCo's convex collision routine places anew "
                   "every tick and never came to rest: 0.24 mm/s at seed 3 of timber-bolt after the task (under "
                   "1 um of travel), 0.56-0.62 mm/s in E22's seated case (7 um). A tighter collision tolerance made it "
                   "worse (1.29 mm/s). A flat face stands on three asperities, and three spheres on a flat floor are "
                   "three contacts computed in closed form: 0.000 and 0.037 mm/s after the task at seeds 3 and 0 "
                   "(single runs on the real task; the batch's figures are in E12 and E22). "
                   "Tried and rejected: one crown in the middle of the face. It rested where the bolt stood on the "
                   "bare backer (E22) "
                   "and rocked in the dimple the auger's point leaves (1.0-1.2 mm/s decaying, 0.18-0.30 mm of "
                   "travel, all six seeds of timber-bolt): a bolt on one point is a pendulum in its hole")
PAD_TORSIONAL_M = _p("PAD_TORSIONAL_M", 0.0042, "m", "repo", "wine_pouring: mu L/4 line-contact torsion")
CONTACT_GAP_M = _p("CONTACT_GAP_M", 0.0, "m", "engine",
                   "box shapes of the world (not the hole proxies' boxes, which meet only round items): MuJoCo 3.11's "
                   "box-box narrowphase with a 2 mm gap returned 0.47 m and 1.73 m deep "
                   "contacts for a bar resting flat on the bench and threw it off at 37 m/s; gap 0 gives the face "
                   "contacts and the same rest (E1b)")
CONTACT_GAP_ROUND_M = _p("CONTACT_GAP_ROUND_M", 0.002, "m", "engine",
                         "cylinders, capsules, spheres, convex hulls and meshes: the gap of the component experiments. "
                         "With CONTACT_GAP_M = 0 on these shapes too, the drill press's "
                         "feed-shaft hub, flush against the head, then got a 136 mm deep contact from MuJoCo 3.11's convex "
                         "narrowphase: the shaft spun at 90 rad/s and drove the quill 132 mm down before any action")
MESH_CONTACT_GAP_M = CONTACT_GAP_ROUND_M
MAX_OBJECT_SPEED_M_S = _p("MAX_OBJECT_SPEED_M_S", 20.0, "m/s", "repo", "fault threshold (barcode_scanning)")
MAX_OBJECT_SPIN_RAD_S = _p("MAX_OBJECT_SPIN_RAD_S", 1000.0, "rad/s", "repo", "fault threshold; spindles exempt")
CONTACT_FORCE_TOL_N = 0.5

# ------------------------------------------------------------------ robot (FR3/Panda + Franka Hand)
FINGER_EFFORT_N = _p("FINGER_EFFORT_N", 35.0, "N", "read",
                     "Franka Hand product manual 1.2 section 5.1: grasping (continuous) force adjustable 30-70 N, a "
                     "gripper-level figure (its travel speed is the one quoted per finger); Franka's own simulation of "
                     "the hand applies half the commanded grasp force to each finger (franka_ros franka_gripper_sim.cpp)")
GRIP_FORCE_BAND_M = _p("GRIP_FORCE_BAND_M", 0.0005, "m", "authored",
                       "force mode of the hand: a finger stopped at least this far short of its commanded position "
                       "presses with the whole commanded force, and with a proportional part of it nearer (the "
                       "hand-over between holding a force and holding a position; world._servo). Franka's grasp call "
                       "holds its force on whatever stops the fingers and reports success within a 5 mm width "
                       "tolerance by default (libfranka Gripper::grasp, epsilon_inner / epsilon_outer); 0.5 mm is "
                       "chosen here so that every part of the tasks (thinnest pinched: the 4.9 mm bracket leg) "
                       "gets the commanded force")
FRANKA_HAND_FINGER_SPEED = _p("FRANKA_HAND_FINGER_SPEED", 0.05, "m/s", "read", "Franka Hand datasheet 50 mm/s per finger")
FR3_PAYLOAD_KG = _p("FR3_PAYLOAD_KG", 3.0, "kg", "read",
                    "FR3 datasheet v2.6: 'Rated payload 3 kg', measured at the flange. The datasheet does not say that "
                    "the hand counts against it; that it does is this environment's reading of a flange rating (the hand "
                    "is bolted to the flange), not a sentence of Franka's")
FRANKA_HAND_KG = _p("FRANKA_HAND_KG", 0.73, "kg", "read",
                    "Franka Hand product manual: mass about 0.73 kg. With the reading above an arm's rated payload beyond "
                    "its hand is 2.27 kg. The world does not refuse a heavier load (the real arm does not either: it "
                    "leaves its specification); it logs one payload_over_rating event per episode and the evaluator "
                    "fails the episode once an arm has carried more than that for PAYLOAD_OVER_S "
                    "(world._payload_torques, task.common)")
REFLEX_FILTER_S = _p("REFLEX_FILTER_S", 0.02, "s", "repo", "filtered external torque estimate (wine_pouring)")
REFLEX_RECOVERY_S = _p("REFLEX_RECOVERY_S", 1.0, "s", "repo", "targets refused until recovery (barcode_scanning)")
# Franka's joint-space limits (docs source control-parameters-joint-fr3.csv / -fer.csv, frankarobotics/docs; the same
# values as libfranka's rate_limiting.h): necessary conditions on the commanded joint trajectory
ROBOT_VELOCITY_FRACTION = _p("ROBOT_VELOCITY_FRACTION", 0.95, "-", "authored",
                            "the arm's trajectory generator plans to this fraction of each joint's velocity limit. "
                            "Planned at the limit itself, a full-speed move ended in the robot's own velocity reflex: "
                            "the joint servo overshoots the generated velocity by 5 % at the end of its acceleration "
                            "(2.752 rad/s against 2.62 in E14), and the reflex stands at 1.05 x the limit. Franka's FR3 "
                            "velocity limits also depend on joint position "
                            "(https://frankarobotics.github.io/docs/robot_specifications.html); the constants used "
                            "here are their published maxima")
PAYLOAD_OVER_S = _p("PAYLOAD_OVER_S", 0.5, "s", "authored",
                    "an arm counts as having carried more than its rated payload once its carried load has been above "
                    "the rating for this long in total (the carried-load estimate switches with the fingers' contacts "
                    "during a hand-over; evaluator constraint, task.py)")
FR3_MAX_JOINT_ACC = _p("FR3_MAX_JOINT_ACC", (10.0,)*7, "rad/s2", "read", "FR3 limit table: 10 rad/s2 on every joint")
FR3_MAX_JOINT_JERK = _p("FR3_MAX_JOINT_JERK", (5000.0,)*7, "rad/s3", "read", "FR3 limit table: 5000 rad/s3 on every joint")
PANDA_MAX_JOINT_ACC = _p("PANDA_MAX_JOINT_ACC", (15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0), "rad/s2", "read",
                         "Franka Emika Robot (FER, Panda) limit table")
PANDA_MAX_JOINT_JERK = _p("PANDA_MAX_JOINT_JERK", (7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0), "rad/s3",
                          "read", "Franka Emika Robot (FER, Panda) limit table")
FRANKA_MAX_TORQUE_RATE = _p("FRANKA_MAX_TORQUE_RATE", 1000.0, "N m/s", "read",
                            "FR3 and FER limit tables: desired joint torque rate on every joint")
READY_POSE = (0.0, -0.55, 0.0, -2.2, 0.0, 1.7, 0.785, 0.04, 0.04)   # authored ready pose (arm joints, fingers)
# inverse kinematics (drilling/kinematics.py): the shared solver's own problem, restated because that module
# evaluates it with an analytic Jacobian; the values are the shared solver's
IK_MAX_EVALUATIONS = _p("IK_MAX_EVALUATIONS", 300, "-", "repo",
                        "cap on the least-squares function evaluations of one inverse-kinematics solve, as "
                        "drilling/kinematics.py calls the shared solver (max_evaluations=300); a count, so "
                        "which targets are accepted does not depend on the machine's load")
IK_ROTATION_WEIGHT = _p("IK_ROTATION_WEIGHT", 0.12, "m/rad", "repo",
                        "weight of the rotation-vector error against the position error in the solve's residual "
                        "(robots/controllers/kinematics.py SerialKinematics.solve residual)")
IK_SEED_WEIGHT = _p("IK_SEED_WEIGHT", 1e-5, "m/rad", "repo",
                    "pull towards the seed joints in the same residual: picks one of a 7-joint arm's solutions")

# ------------------------------------------------------------------ room and stations (authored)
ROOM = dict(x0=-4.0, x1=4.0, y0=-3.0, y1=3.0, z1=3.0, h=3.0)
WALL_FACE_Y = _p("WALL_FACE_Y", 3.0, "m", "authored", "north wall interior face")
WALL_THICK_M = _p("WALL_THICK_M", 0.20, "m", "authored", "cast-in-place basement wall")
CART_DECK_Z = _p("CART_DECK_Z", 0.80, "m", "authored", "robot cart deck height")
ARM_SPACING_M = _p("ARM_SPACING_M", 0.56, "m", "authored", "arm bases at the cart's front corners")
BENCH_TOP_Z = _p("BENCH_TOP_Z", 0.90, "m", "authored", "steel workbench top (commercial 36 in)")
BENCH_SIZE = (1.83, 0.76)

# ------------------------------------------------------------------ cordless drill/driver (DeWalt DCD800 class)
DCD800_MANUAL = ("https://service.blackanddecker.co.za/i/DEWALT/GLOBALBOM/B1/DCD800N/1/Instruction_Manual/EN/"
                 "NA408973_DCD800_DCD805_T1_Asia.pdf")
DRILL_BARE_KG = _p("DRILL_BARE_KG", 1.28, "kg", "read", f"DCD800 'Weight (without battery pack) 1.28 kg' ({DCD800_MANUAL})")
BATTERY_2AH_KG = _p("BATTERY_2AH_KG", 0.40, "kg", "read", f"DCB183/DCB203 2.0 Ah 0.40 kg, same manual ({DCD800_MANUAL})")
DRILL_MASS_KG = _p("DRILL_MASS_KG", DRILL_BARE_KG + BATTERY_2AH_KG, "kg", "derived",
                   "DCD800 bare + 2.0 Ah battery: the built drill without its bit (tools.CordlessDrill; printed by E1a)")
DRILL_LENGTH_M = _p("DRILL_LENGTH_M", 0.162, "m", "read",
                    "DCD800 161.8 mm; NOT built: the colliders follow the Drill_01 visual model, 183 mm from the battery's "
                    "back to the chuck's front (tools.CordlessDrill), open")
DRILL_GEAR_RPM = _p("DRILL_GEAR_RPM", (650.0, 2000.0), "rpm", "read", "DCD800 no-load speeds gear 1 / gear 2")
DRILL_SOFT_TORQUE_NM = _p("DRILL_SOFT_TORQUE_NM", 27.0, "N m", "read",
                          f"DCD800 'Max. torque (hard/soft) 90/27 Nm' ({DCD800_MANUAL}): the soft-joint figure, the "
                          "torque the drill sustains into a yielding joint; the hard-joint 90 N m is a peak that "
                          "includes the rotor's kinetic energy")
DRILL_STALL_NM = _p("DRILL_STALL_NM", tuple(round(DRILL_SOFT_TORQUE_NM*DRILL_GEAR_RPM[0]/n, 2) for n in DRILL_GEAR_RPM),
                    "N m", "derived",
                    "sustained stall torque at the chuck per gear: the manual's soft-joint torque in gear 1 (27 N m) "
                    "and the same motor through gear 2's ratio (27 x 650/2000 = 8.78 N m). It decides whether "
                    "a jammed bit in gear 2 can trip an FR3 wrist's 11.4 N m reflex by itself: 8.78 N m cannot")
DRILL_POWER_W = _p("DRILL_POWER_W", DRILL_STALL_NM[0]*DRILL_GEAR_RPM[0]*2*math.pi/60/4, "W", "derived",
                   "peak mechanical output of the linear motor line M = M_s (1 - w/w0): M_s w0/4 = 459 W. For "
                   "comparison only (nothing reads it): DeWalt rates the DCD800 'up to 40% more UWO vs. the DCD791B' "
                   "(https://www.dewalt.com/product/dcd800b/20v-max-xr-brushless-cordless-1-2-in-drill-driver-tool-only), "
                   "whose output is listed as 460 W (RS 136-2870, search-level)")
DRILL_CLUTCH_NM = _p("DRILL_CLUTCH_NM", (1.7, 8.5), "N m", "read", "Milwaukee 2903 clutch 15-75 in-lb over 16 settings, "
                     "applied to this class (transferred)")
DRILL_SOFT_START_S = _p("DRILL_SOFT_START_S", 0.2, "s", "estimate", "electronic soft start ramp")
DRILL_BRAKE_S = _p("DRILL_BRAKE_S", 0.15, "s", "open", "brake on release; no stop time published")
TRIGGER_TRAVEL_M = _p("TRIGGER_TRAVEL_M", 0.008, "m", "estimate", "variable-speed switch travel (no datasheet)")
TRIGGER_DEADBAND_M = _p("TRIGGER_DEADBAND_M", 0.001, "m", "estimate", "switch dead band")
TRIGGER_PRELOAD_N = _p("TRIGGER_PRELOAD_N", 4.0, "N", "estimate", "return spring at rest (5-15 N range)")
TRIGGER_FULL_N = _p("TRIGGER_FULL_N", 10.0, "N", "estimate", "return spring at full travel")
TRIGGER_MASS_KG = 0.006
TRIGGER_ARMATURE_KG = _p("TRIGGER_ARMATURE_KG", 0.1, "kg", "estimate",
                         "reflected inertia of the trigger mechanism (switch slider, contacts, spring) on the trigger "
                         "axis; sized by the drop sweep E14: a 6 g trigger between a 3 kg housing and its stop was thrown "
                         "at 27.9 m/s (fault) and 14 mm past its 8 mm stop by 1 m drops; 0.1 kg: none of 32. Costs a "
                         "finger 0.16 N at a 1.6 m/s^2 squeeze")
SELECTOR_ARMATURE_KG = _p("SELECTOR_ARMATURE_KG", 0.1, "kg", "estimate",
                          "reflected inertia of the drill's direction and gear sliders (detent, linkage); E14: a gear "
                          "slider 9 mm past its stop in a 1 m drop at 0.002 kg")
ROTOR_INERTIA_CHUCK = _p("ROTOR_INERTIA_CHUCK", 2e-4, "kg m2", "estimate", "rotor reflected through gear 1 to the chuck")

# ------------------------------------------------------------------ rotary hammer (DeWalt DCH273 class, SDS-plus)
DCH273_MANUAL = "https://service.blackanddecker.com.au/i/DEWALT/GLOBALBOM/XE/DCH273/2/Instruction_Manual/ZH/DCH273_T2_ASIA.pdf"
HAMMER_BARE_KG = _p("HAMMER_BARE_KG", 2.5, "kg", "read",
                    f"DCH273 technical data 'Weight (without battery pack) 2.5 kg' ({DCH273_MANUAL}); the built tool "
                    "without its bit (housing, chuck, trigger, mode knob and button, depth rod) is scaled to it")
BATTERY_5AH_KG = _p("BATTERY_5AH_KG", 0.62, "kg", "read", f"DCB184 5.0 Ah 'Weight 0,62 kg', same manual ({DCH273_MANUAL})")
HAMMER_MASS_KG = _p("HAMMER_MASS_KG", HAMMER_BARE_KG + BATTERY_5AH_KG, "kg", "derived",
                    "DCH273 bare + DCB184 5 Ah, the built hammer without its bit (tools.RotaryHammer; printed by E1a)")
SDS_BIT_KG_PER_M = _p("SDS_BIT_KG_PER_M", 0.5*(0.091 + 0.081)/0.215, "kg/m at 10 mm", "derived",
                      "KIT (Uhl et al. 2022, https://publikationen.bibliothek.kit.edu/1000151192/149581508) weighed "
                      "10 mm SDS-plus bits with 150 mm working length: Bosch SDS plus-7X 91 g, Hilti TE-CX 81 g; per "
                      "metre of a 215 mm overall bit (Bosch 10/150/215 designation, inferred), scaled by (d/10 mm)^2: "
                      "the 13.3 x 160 mm bit weighs 0.113 kg")
MODE_BUTTON_N = _p("MODE_BUTTON_N", 8.0, "N", "estimate",
                   "mode selector release button force; DCH273 manual: 'Before attempting to rotate the mode selector "
                   "depress the mode selector button' and 'There are no operable positions in between' "
                   f"({DCH273_MANUAL}); force not published, the push-button value (PRESS_BUTTON_N)")
MODE_BUTTON_TRAVEL_M = _p("MODE_BUTTON_TRAVEL_M", 0.002, "m", "estimate", "mode selector button travel")
MODE_UNLOCK_FRACTION = _p("MODE_UNLOCK_FRACTION", 0.75, "-", "authored",
                          "the lock pin leaves its notch only in the last quarter of the button's stroke: the button "
                          "stands 1 mm proud of the knob's pointer, so a flat body lying on the knob cannot unlock it")
MODE_LOCK_PLAY_RAD = _p("MODE_LOCK_PLAY_RAD", math.radians(2.0), "rad", "estimate",
                        "angular play of the selector's lock pin in its notch; outside it the gears are not engaged "
                        "(no rotation, no blows): 'no operable positions in between'. Without the lock the knob was "
                        "turned from hammer-drill to rotary by 0.28-0.37 N m of contact from a robot arm link against "
                        "a 0.48 N m/rad detent spring")
HAMMER_BLOW_J = _p("HAMMER_BLOW_J", 2.1, "J", "read", "DCH273 EPTA 05/2009 single impact energy")
HAMMER_BPM = _p("HAMMER_BPM", 4600.0, "1/min", "read", "DCH273 0-4600 bpm")
HAMMER_RPM = _p("HAMMER_RPM", 1100.0, "rpm", "read", "DCH273 0-1100 rpm")
HAMMER_ENERGY_EXP = _p("HAMMER_ENERGY_EXP", 2.4, "-", "derived", "KIT 2026 rig: 0.59 J @24.6 Hz, 1.65 @36.9, 2.95 @50 Hz")
HAMMER_PUSH_N = _p("HAMMER_PUSH_N", 5.0*GRAVITY, "N", "read",
                   "DCH273 manual: 'Apply only a gentle pressure to the tool (approx. 5 kg). Excessive force does not "
                   f"speed up drilling but decreases tool performance' ({DCH273_MANUAL}): the push that keeps every blow")
IDLE_WINDOW_M = _p("IDLE_WINDOW_M", 0.002, "m", "estimate", "bit shank must be pushed back within 2 mm of the stop")
SDS_PLAY_M = _p("SDS_PLAY_M", 0.012, "m", "estimate", "SDS-plus bit axial play in the chuck")
SDS_SPRING_N = _p("SDS_SPRING_N", 5.0, "N", "estimate", "light bit return spring: its force with the bit fully out")
SDS_SPRING_K_N_M = _p("SDS_SPRING_K_N_M", 400.0, "N/m", "estimate",
                      "rate of the bit return spring: 9.8 N with the bit pushed fully back (tools.RotaryHammer)")
HAMMER_BOTTOM_N = _p("HAMMER_BOTTOM_N", 5.0, "N", "engine",
                     "read from this simulation, not from a tool: mean force the housing passes to the bit through the chuck's stop in steady drilling at the "
                     "manual's push: 49.0 N push - 9.7 N return spring - 34.2 N mean recoil = 5.1 N, the same for "
                     "striker masses of 0.045 to 0.075 kg (E8 diagnostic: per scheduled blow, the bit's slide, the window and "
                     "whether the blow landed). Used only to derive STRIKER_MASS_KG")
STRIKER_MASS_KG = _p("STRIKER_MASS_KG",
                     ((HAMMER_PUSH_N - (SDS_SPRING_N + SDS_SPRING_K_N_M*SDS_PLAY_M) - HAMMER_BOTTOM_N)/(HAMMER_BPM/60.))**2 /
                     (2*HAMMER_BLOW_J), "kg", "derived",
                     "steady drilling returns each blow's momentum sqrt(2 m_s E_b) to the housing; the push that keeps "
                     "every blow must cover that mean recoil f_b sqrt(2 m_s E_b) AND the bit return spring at the working "
                     "position (9.8 N) AND what the bottomed housing passes to the bit (HAMMER_BOTTOM_N). The manual's "
                     "49 N then leaves 34.2 N for the recoil: m_s = 0.047 kg at 9.4 m/s. It is the LARGEST striker "
                     "momentum with which the simulated hammer keeps its blows at the maker's push (a calibration to "
                     "that one statement; E8's check at that push is therefore not an independent test). With 0.097 kg "
                     "(the whole 49 N taken as recoil) the hammer landed 63 % of its blows at 49 N and "
                     "all of them only from 75 N, the delivered recoil capped at 34 N whatever the striker (97 of 153 "
                     "blows at 0.097 kg, 119 at 0.075, 138 at 0.055, 153 at 0.045). No "
                     "SDS-plus striker mass found in KIT's hammer-drill papers (Uhl 2022, Zimprich 2026)")
HAMMER_POWER_W = _p("HAMMER_POWER_W", 400.0, "W", "read",
                    f"DCH273 technical data 'Max power output 400 W' ({DCH273_MANUAL})")
HAMMER_STALL_NM = _p("HAMMER_STALL_NM", 4.*HAMMER_POWER_W/(HAMMER_RPM*2.*math.pi/60.), "N m", "derived",
                     "rotation stall torque at the chuck, from the linear motor line M = M_s (1 - w/w0) whose peak power "
                     "M_s w0/4 is the manual's maximum output: 13.9 N m at 1100 rpm (an upper bound, since the hammer "
                     "mechanism takes part of that output)")
HAMMER_CLUTCH_FRACTION = _p("HAMMER_CLUTCH_FRACTION", 0.9, "x stall", "estimate",
                            "the electronic overload clutch: 'In case of jamming of a drill bit, the drive to the drill "
                            "spindle is interrupted ... the motor is repeatedly switched on and off for a few seconds "
                            "... Release and depress the trigger to re-engage drive' (DCH273 manual, Overload Clutch). "
                            "How it detects a jam and at what level is not published; here: the motor at or above "
                            "this share of its stall torque (the spindle slowed to a tenth of its commanded speed)")
HAMMER_CLUTCH_S = _p("HAMMER_CLUTCH_S", 0.05, "s", "estimate",
                     "time the overload must last before the electronic clutch interrupts the drive (not published)")
HAMMER_CLUTCH_NM = _p("HAMMER_CLUTCH_NM", HAMMER_CLUTCH_FRACTION*HAMMER_STALL_NM, "N m", "derived",
                      "motor torque at which the electronic clutch counts an overload at full trigger (12.5 N m). The "
                      "mechanical slip clutches of Milwaukee's corded 1-1/8 in SDS hammers "
                      "(Product Support Bulletin #403, https://documents.milwaukeetool.com/TIY403.pdf: 21.7-48.8 N m) "
                      "slip at a level this 400 W tool's motor cannot reach")

# ------------------------------------------------------------------ drill press (Grizzly G7943 class)
PRESS_TRAVEL_M = _p("PRESS_TRAVEL_M", 0.0826, "m", "read", "G7943 quill travel 3-1/4 in")
PRESS_PINION_R = _p("PRESS_PINION_R", 0.0202, "m", "open",
                    "TRANSFERRED from another press: the JET JDP-17 advertises 127 mm of quill travel per handle "
                    "revolution (20.2 mm pitch radius). No figure for the Grizzly G7943 was found (its product "
                    "page refused the fetch, 2026-10-07). If the G7943 "
                    "covers its 82.6 mm in one turn the radius is 13.1 mm and the handle's leverage over the quill is "
                    "16 instead of 10.4: every handle force in this environment's press runs would be 35 % lower "
                    "and every handle angle 54 % larger. Not changed without a figure for this press")
PRESS_HANDLE_R = _p("PRESS_HANDLE_R", 0.21, "m", "measured",
                    "knob distance from the hub on the drill_press_01 model that draws the press")
PRESS_QUILL_KG = _p("PRESS_QUILL_KG", 2.5, "kg", "estimate",
                    "quill, spindle and chuck (without the bit); tools.DrillPress builds exactly this (E1a prints it)")
PRESS_CHUCK_KG = _p("PRESS_CHUCK_KG", 1.1, "kg", "estimate",
                    "keyed 5/8 in chuck on its arbor, the turning part of PRESS_QUILL_KG")
PRESS_SPRING_PRE_NM = _p("PRESS_SPRING_PRE_NM", 1.5, "N m", "estimate", "return spring preload at the feed shaft (open)")
PRESS_SPRING_RATE = _p("PRESS_SPRING_RATE", 1.0, "N m/rad", "estimate", "return spring rate (open)")
PRESS_MOTOR_W = _p("PRESS_MOTOR_W", 560.0, "W", "read", "3/4 hp motor")
PRESS_BREAKDOWN = _p("PRESS_BREAKDOWN", 2.25, "x rated", "literature", "induction motor breakdown 2-2.5 x rated")

# ------------------------------------------------------------------ vise and punch
VISE_PITCH_M = _p("VISE_PITCH_M", 0.0025, "m", "estimate", "drill-press vise screw pitch")
VISE_THREAD_MU = _p("VISE_THREAD_MU", 0.15, "-", "literature",
                    "Shigley Table 8-5 (threaded pairs; reproduced at https://amesweb.info/Screws/screw-nut-collar-"
                    "friction-table.aspx): steel screw in a steel or cast-iron nut 0.15-0.25 dry, 0.11-0.17 with machine "
                    "oil; a vise screw is oiled: 0.15")
VISE_COLLAR_MU = _p("VISE_COLLAR_MU", 0.12, "-", "literature",
                    "thrust-collar friction, Shigley Table 8-6 (same reproduction): soft steel on cast iron 0.12 running "
                    "(0.17 starting); the screw's shoulder bears on the cast-iron body end without a thrust bearing")
VISE_COLLAR_R_M = _p("VISE_COLLAR_R_M", 0.5*(0.012 + 0.0065 + 0.0025/4), "m", "derived",
                     "mean radius of the collar face: from the screw's major radius (VISE_SCREW_RM + pitch/4) to the "
                     "12 mm screw hub (tools.Vise); with the collar the vise gives 396 N of clamp per N m at the crank "
                     "(724 N per N m without it; a Kurt D688 with a roller thrust bearing: 1,129 lb at 10 ft lb = "
                     "370 N per N m, 7,968 lb at 80 ft lb = 327 N per N m, https://ctemag.com/node/17211)")

# ------------------------------------------------------------------ hold-down strap clamps (panel-holes, timber-bolt)
TECO_PLAIN_CLAMP = ("TE-CO plain clamp 30402 (https://catalog.te-co.com/viewitems/machined-clamps/plain-clamps): total "
                    "length 4.000 in, width 1.000 in, height 0.625 in, slot length 1.438 in, for 5/16 and 3/8 in studs; "
                    "1018 steel, case hardened, black oxide, tapered nose")
CARRLANE_STRAP = ("Carr Lane, Standard Strap Clamp Forces (https://www.carrlane.com/engineering-resources/"
                  "technical-design-information/clamp-strap): 3/8-16 stud, recommended torque 16 ft lb, clamping force "
                  "1300 lb, tensile force in stud 2600 lb; 'based on a clean, dry clamping stud that is torqued to "
                  "approximately 33% of its 100,000 psi yield strength (2:1 lever ratio)'")
HOLD_DOWN_STRAP_M = _p("HOLD_DOWN_STRAP_M", (4.0*0.0254, 1.0*0.0254, 0.625*0.0254), "m", "read",
                       f"strap length, width, height: {TECO_PLAIN_CLAMP}")
HOLD_DOWN_SLOT_LEN_M = _p("HOLD_DOWN_SLOT_LEN_M", 1.438*0.0254, "m", "read", f"stud slot length: {TECO_PLAIN_CLAMP}")
HOLD_DOWN_SLOT_W_M = _p("HOLD_DOWN_SLOT_W_M", 0.4375*0.0254, "m", "estimate",
                        "stud slot width 7/16 in, a clearance slot for the 3/8 in stud (not in TE-CO's table); the slot "
                        "is taken centred along the strap (not in the table either)")
HOLD_DOWN_STUD_D_M = _p("HOLD_DOWN_STUD_D_M", 0.375*0.0254, "m", "read", f"3/8-16 clamping stud: {CARRLANE_STRAP}")
HOLD_DOWN_STUD_AT_M2 = _p("HOLD_DOWN_STUD_AT_M2", 0.0775*0.0254**2, "m2", "literature",
                          "tensile stress area of a 3/8-16 UNC thread, 0.0775 in2 (ASME B1.1; handbook value)")
HOLD_DOWN_NUT_M = _p("HOLD_DOWN_NUT_M", (0.875*0.0254, 0.500*0.0254, 0.125*0.0254, 0.688*0.0254), "m", "read",
                     "spherical flange nut: flange diameter, height, flange thickness, width across flats: TE-CO spherical "
                     "flange nut 41803, 3/8-16, 12L14 case hardened, black oxide, 0.875 x 0.500 in, 0.688 in across flats "
                     "(https://catalog.te-co.com/viewitems/nuts-washers/spherical-flange-nuts); the 0.125 in flange "
                     "thickness is that of TE-CO's plain flange nut 41603 of the same envelope "
                     "(https://catalog.te-co.com/viewitems/nuts-washers/flange-nuts)")
HOLD_DOWN_WASHER_M = _p("HOLD_DOWN_WASHER_M", (0.875*0.0254, 0.156*0.0254, 1.250*0.0254), "m", "read",
                        "female spherical washer under the nut: outside diameter, thickness, seat radius: TE-CO 42742 "
                        "(inside diameter 0.438 in, 12L14 case hardened, black oxide; 'compensates for slight misalignment "
                        "between workpiece and strap clamp', https://catalog.te-co.com/viewitems/spherical/"
                        "spherical-washers-steel-female)")
HOLD_DOWN_SEAT_LIMIT_RAD = _p("HOLD_DOWN_SEAT_LIMIT_RAD", math.radians(3.0), "rad", "estimate",
                              "largest tilt the spherical seat takes up (spherical washer pairs are made for about 3 "
                              "degrees, DIN 6319)")
HOLD_DOWN_REC_TORQUE_NM = _p("HOLD_DOWN_REC_TORQUE_NM", 16*1.3558179, "N m", "read",
                             f"the maker's recommended nut torque, 16 ft lb: {CARRLANE_STRAP}")
HOLD_DOWN_REC_TENSION_N = _p("HOLD_DOWN_REC_TENSION_N", 2600*4.4482216, "N", "read",
                             f"stud tension at that torque, 2600 lb: {CARRLANE_STRAP}")
HOLD_DOWN_NUT_FACTOR = _p("HOLD_DOWN_NUT_FACTOR", HOLD_DOWN_REC_TORQUE_NM/(HOLD_DOWN_REC_TENSION_N*HOLD_DOWN_STUD_D_M), "-",
                          "derived", "torque factor K = T/(F d) of Carr Lane's own table row (0.197: a clean, dry stud)")
HOLD_DOWN_TORQUE_NM = _p("HOLD_DOWN_TORQUE_NM", 0.5*HOLD_DOWN_REC_TORQUE_NM, "N m", "authored",
                         "start state: each clamp nut was tightened to 8 ft lb, half the maker's recommended torque (about "
                         "55 N on a 200 mm wrench). Chosen below the recommended value because of the work: at 16 ft lb the "
                         "1 in wide toe, reaching 22 mm over the board, would bear on Douglas fir with 10 MPa on average, at "
                         "8 ft lb with 5.0 MPa, under the 800 lbf/in2 (5.5 MPa) compression strength perpendicular to grain "
                         "of coast Douglas-fir at 12 % moisture (Wood Handbook Table 5-3b, as reproduced at "
                         "https://engineersedge.com/civil_engineering/wood_engineering_properties_2_13744.htm). The torque "
                         "itself is this environment's choice")
HOLD_DOWN_PRELOAD_N = _p("HOLD_DOWN_PRELOAD_N", HOLD_DOWN_TORQUE_NM/(HOLD_DOWN_NUT_FACTOR*HOLD_DOWN_STUD_D_M), "N", "derived",
                         "stud tension of a tightened clamp, F = T/(K d) = 5783 N (1300 lb); the strap shares it between "
                         "the work and the heel block by its lever arms (HOLD_DOWN_HEEL_M: about 2.8 kN on the work per "
                         "clamp; E21 measures the share)")
HOLD_DOWN_E_PA = _p("HOLD_DOWN_E_PA", 200e9, "Pa", "literature", "Young's modulus of carbon steel (stud and strap)")
HOLD_DOWN_TOE_OVERLAP_M = _p("HOLD_DOWN_TOE_OVERLAP_M", 0.022, "m", "authored",
                             "how far the strap's toe reaches over the work's end (start-state placement). The strap lies "
                             "nose-down by a fraction of a degree, so it bears on the work with its toe's tip")
HOLD_DOWN_STUD_OFFSET_M = _p("HOLD_DOWN_STUD_OFFSET_M", 0.030, "m", "authored",
                             "stud axis beyond the clamped part's end: as close to the work as the plywood backer's 15-20 mm "
                             "overhang lets it stand (5 mm clear of the backer's edge in the timber stack)")
HOLD_DOWN_HEEL_M = _p("HOLD_DOWN_HEEL_M", (0.025, 0.038, 0.008), "m", "authored",
                      "heel block: length along the strap, width, and how much of it the strap's heel end covers. Its "
                      "height is the stack's (a steel block of the work's height). The strap bears 49 mm in front of the "
                      "stud (toe) and 45.6 mm behind it (heel): the lever puts 45.6/94.6 = 0.48 of the stud tension on "
                      "the work (Carr Lane's table assumes 0.5)")
HOLD_DOWN_BEARING_M = _p("HOLD_DOWN_BEARING_M", (0.003, 0.0005), "m", "engine",
                         "the strap's two bearing lines (toe on the work, heel on the block) are rounded: two contact "
                         "spheres of 3 mm radius each, 0.5 mm proud of the strap's flat underside (the flat boxes stand "
                         "back by that much). A sphere on a face has that face's normal whatever the strap's tilt. With "
                         "flat box faces as the bearing surfaces the strap, tilted 1.2 degrees by this engine's soft "
                         "stack, touched with its edges, MuJoCo's box-box contact normal jumped between the two bodies' "
                         "face normals")
LOOSE_SOLMIX = _p("LOOSE_SOLMIX", 1e3, "-", "engine",
                  "weight of a small loose part's own contact law where it meets a hardened face (MuJoCo solmix, at "
                  "equal contact priority; hardened faces weigh 1): the softer of two bodies decides how a contact "
                  "gives, as two springs in series do. The parts: the heads of the M8 and carriage bolts and the "
                  "centre punch's body (scene.loose_shapes). With the hardened face's law for "
                  "everything on it, a part of 18-137 g lay on a contact 0.02-1 micrometre deep, which one "
                  "tick's error lifts it off: on the clamped boards with the stack itself quiet (other bodies below "
                  "0.012 mm/s) a carriage bolt hopped at 12 mm/s in bursts in 5 of 6 placements, and with its own "
                  "law in none of 5. Friction is the larger of the two "
                  "coefficients and the rolling resistance is the part's, as MuJoCo mixes equal priorities")
HOLD_DOWN_SOLIMP = _p("HOLD_DOWN_SOLIMP", (0.999, 0.9999, 0.0005), "-", "engine",
                      "contact impedance of the hold-down clamps (heel block, washer, strap toe, tail, rails, bearing "
                      "spheres) and of the wooden parts of a clamped stack. "
                      "MuJoCo's friction holds a load only with a slip acceleration proportional to 1 - impedance "
                      "(SOLVER_IMPRATIO), and a strap clamp keeps hundreds of newtons of friction locked in under its "
                      "washer, toe and heel: with STIFF_SOLIMP, (0.99, 0.999, 0.0005), before the no-slip pass the "
                      "main solve had the top board of the timber stack accelerating at 0.55 m/s2 and 13.9 rad/s2 with "
                      "nothing on it, and a carriage bolt laid on it hopped at 11 mm/s. With this value the board's "
                      "residual is 0.002 m/s2 and 0.06 rad/s2 and the bolt lies at 0.02 mm/s. The stack "
                      "shortens 0.04 mm under the two clamps, where a real one shortens about 0.14 mm and "
                      "STIFF_SOLIMP gives 0.36 mm (HOLD_DOWN_STACK_STIFF). A numerical stiffness with no material meaning")
HOLD_DOWN_BEARING_ROLLS = _p("HOLD_DOWN_BEARING_ROLLS", False, "-", "engine",
                             "whether the load-bearing faces of a strap clamp (toe, heel block, tail, the four bearing "
                             "spheres) and the wooden parts of a clamped stack carry rolling and twisting resistance "
                             "(MuJoCo condim 6, ROLLING_B_M). Hardened faces carry it so that a round "
                             "part stops rolling on a metal face. On these faces nothing rolls, and the rows do "
                             "harm: the main solve leaves them slack and the no-slip pass makes them rigid, so after "
                             "that pass the clamp's balance leaned on moments the main solve had not seen (the straps "
                             "465 rad/s2 and the top board 21 rad/s2 short of balance at impratio 100, 7 and 0.7 "
                             "without the rows). The bearing faces also outrank the work (contact priority 2), so a "
                             "toe on a steel sheet has none either. The washer on the strap's rails keeps its rows: "
                             "without them its seat crept 1.2 micro-rad a second and the clamp twitched every 0.32 s "
                             "at 0.04 mm/s (measured). Metal work faces keep theirs for the round parts "
                             "that lie on them (centre punch, anchors)")
HOLD_DOWN_STACK_STIFF = _p("HOLD_DOWN_STACK_STIFF", True, "-", "engine",
                           "the wooden parts of a clamped stack (plywood backer, boards) take the stiff contact of "
                           "clamping faces (STIFF_SOLIMP, 2-tick time constant, contact priority 1: scene.stiff_shapes) "
                           "instead of the default contact (clamps.stiffen_stack). Under the two clamps' 5.6 kN the "
                           "stack's faces stand in each other by, summed over the stack: timber stack 2.26 mm with the "
                           "default contact (0.77 + 0.81 + 0.68), 0.36 mm with this one (0.12 + 0.14 + 0.10); panel stack "
                           "0.96 mm (0.30 + 0.66) and 0.32 mm (0.24 between the steel sheet, which keeps SOLID_SOLIMP, "
                           "and the backer, + 0.08). A real stack shortens under the toes by about 0.14 mm (timber) and "
                           "0.08 mm (panel): hand estimate from WOOD_ET, 2.78 kN per toe spreading at 45 degrees from a "
                           "25 x 6 mm bearing strip (0.10-0.18 and 0.06-0.10 mm for a 12-3 mm strip). Chosen as the nearer "
                           "of the two; still 2.6 and 4 times the real figure. At rest for 30 s both settings pass E0's "
                           "rule (this one: panel 0.0001 mm, timber 0.010 mm from hand-over, at most 0.013 mm/s; default: "
                           "0.0001 and 0.0025 mm), and panel-holes and timber-bolt pass end to end with either. With the "
                           "default contact the timber stack also leaned 0.05 mm in 2 s under half its sideways "
                           "capacity, and the stud tensions drifted 0.6 % in 30 s; with this one 0.001 mm and 0.01 %. "
                           "Side effect: these faces' friction rules or ties their contacts (MuJoCo priority): "
                           "the sheet on its backer 0.45 (0.3 otherwise), a clamp toe on a board 0.45 (0.4 otherwise). "
                           "Measured in E21 with either setting, E0's rest check and E12")
HOLD_DOWN_NUT_ARMATURE_KG = _p("HOLD_DOWN_NUT_ARMATURE_KG", 10.0, "kg", "engine",
                               "inertia on the nut's slide along the stud (a relative coordinate against the bench: it adds "
                               "nothing to any moving part). The stud and strap are 3-5e7 N/m stiff; with critical damping "
                               "on this inertia the 1 ms step integrates that spring stably (h^2 k < 4 m + 2 h c)")
HOLD_DOWN_TIGHTEN = _p("HOLD_DOWN_TIGHTEN", dict(gain_per_s=200.0, max_speed_m_s=0.005, tol=0.002, hold_ticks=100,
                                                 limit_s=2.5), "-", "authored",
                       "start-state preparation (before hand-over only): the nut runs down the stud at up to 5 mm/s until "
                       "the stud tension stays within 0.2 % of HOLD_DOWN_PRELOAD_N for 0.1 s (at most 2.5 s); from then on "
                       "the nut's place on the stud is fixed and only the stud's and strap's elasticity holds the load")

PUNCH_MASS_KG = _p("PUNCH_MASS_KG", 0.070, "kg", "read",
                   "mass of the Rennsteig 430 130 automatic centre punch: retailers list 65 g and 78 g "
                   "(rapidonline.com 57-7065, toolteam.com, farnell 4133602; searched 2026-10-08) and another listing 67 g; 70 g taken. "
                   "A real one is a tube round its spring and hammer, not solid steel bars (which would weigh 131 g). "
                   "The barrel and cap carry the density that gives this total (tools.CentrePunch)")
PUNCH_BODY_M = _p("PUNCH_BODY_M", (0.125, 0.014, 0.004), "m", "read",
                  "automatic centre punch: overall length, shaft diameter, tip diameter of the Rennsteig 430 130 "
                  "('Automatikkoerner 125 mm Schaftquerschnitt 14 mm 60-130 N', listings: "
                  "https://www.testbericht.de/produkte/rennsteig-werkzeuge-automatischer-koerner-430-130 ; search-level)")
PUNCH_RELEASE_RANGE_N = _p("PUNCH_RELEASE_RANGE_N", (60.0, 130.0), "N", "read",
                           "adjustable release force of the Rennsteig 430 130 (same listings; also "
                           "https://toolteam.com/en-GB/rennsteig-automatic-centre-punch-o-4-mm-60-130-n-430-130-t-25385-"
                           "4049002002380). The cap's setting is a start state: each task layout draws it from this "
                           "range with its seed (layout.py)")
PUNCH_RELEASE_N = _p("PUNCH_RELEASE_N", 95.0, "N", "authored",
                     "release force of a punch built without a seeded setting (component probes): the middle of "
                     "PUNCH_RELEASE_RANGE_N; in a task the force comes from the tool's range and the seed")
PUNCH_ENERGY_J = _p("PUNCH_ENERGY_J", 0.5*PUNCH_RELEASE_N*0.003, "J", "derived",
                    "strike energy at the default setting: the main spring's energy at the release, 1/2 PUNCH_RELEASE_N x "
                    "PUNCH_RELEASE_TRAVEL_M (3 mm); a punch with a seeded setting stores 1/2 F x of its own "
                    "(tools.CentrePunch spends exactly this in each strike; the dimple takes the share the part's support "
                    "does not absorb). The maker's energy and hammer mass are not published (open: 0.08-0.25 J swept)")
PUNCH_TIP_HALF_ANGLE = _p("PUNCH_TIP_HALF_ANGLE", math.radians(45), "rad", "read", "60-90 deg tip; 90 deg used")
PUNCH_TIP_R_M = _p("PUNCH_TIP_R_M", 0.00105, "m", "engine",
                   "radius of the point's rounded contact tip, the rod's own (a capsule). A real 90-degree point has a "
                   "tip radius of ~0.1-0.3 mm (estimate), but a 0.3 mm tip sphere and a convex hull of the sharp cone were "
                   "both pushed sideways once the soft contact let them sink under the 100 N push (the bar thrown at "
                   "14.7 m/s, strikes without a dimple; sinking vs radius inferred). On a flat 2.1 mm disc the punch "
                   "balanced and buzzed for ever; a centre of mass 70 mm above a 1.05 mm rounded tip is unstable")
PUNCH_POINT_DAMPING = _p("PUNCH_POINT_DAMPING", 1.0, "x critical", "estimate",
                         "damping of the point's spring for the punch body bouncing on it (0.7 of critical for the "
                         "point's 0.03 kg armature alone is only 0.26 for the 0.22 kg body: pushed on a mark, the body chattered "
                         "on the point at 0.2-0.4 m/s and the punch fired at a 50 N push); the internal hammer "
                         "and point slide in oiled bores, no measured value")
PUNCH_POINT_FRICTION_N = _p("PUNCH_POINT_FRICTION_N", 2.0, "N", "estimate",
                            "dry friction of the point and the internal hammer sliding in their bores (2 % of the "
                            "release force, below what shifts the release): it damps the cocked point's spring, which "
                            "otherwise kept a resting punch buzzing at 2-7 cm/s on its point")

# ------------------------------------------------------------------ cutting laws: metal (Machinery's Handbook form)
MH_A = _p("MH_A", 1.085, "-", "read", "torque factor A of the c/d = 0.18 row of MH Table 7 (Chisel Edge Factors), the "
           "row MH prescribes for drills of standard design; Machinery's Handbook 31st ed. (2020) pp. 1151-1153, read in the archive.org copy https://archive.org/details/machinerys-handbook-31st-v-1.0")
MH_B = _p("MH_B", 1.355, "-", "read", "thrust factor B, MH Table 7 row c/d = 0.18 (with A, J and CHISEL_FRACTION)")
MH_J = _p("MH_J", 0.030, "-", "read", "chisel-edge thrust factor J, MH Table 7 row c/d = 0.18 (the c/d = 0.20 row has "
           "A 1.105, B 1.380, J 0.040); 9 mm threshold Kd d^2 J = 231 N")
KD_STEEL = _p("KD_STEEL", 95.0, "MPa", "derived", "fitted to Rubenstein's AISI 1020 data (Strenkowski 2004 Table 1)")
KD_ALUMINIUM = _p("KD_ALUMINIUM", 48.3, "MPa", "read", "MH Table 6 (Work Material Factor Kd, sharp drill): 'Most "
                   "Aluminum Alloys 7,000' (psi) = 48.3 MPa")
CHISEL_FRACTION = _p("CHISEL_FRACTION", 0.18, "x d", "read", "MH Table 7: 'For drills of standard design, use c/d = "
                      "0.18' (w/d 0.155), the row A, B and J are taken from")
POINT_ANGLE = _p("POINT_ANGLE", math.radians(118.0), "rad", "read", "DIN 338 jobber point")
RUB_MU = _p("RUB_MU", 0.4, "-", "literature", "chisel rubbing friction below the threshold")
WEAR_PER_MM3 = _p("WEAR_PER_MM3", 5e-5, "1/mm3", "derived", "+25 % thrust after ~30 holes 6x6 mm (Subramanian via Audy)")
STEEL_YIELD_MPA = _p("STEEL_YIELD_MPA", 250.0, "MPa", "literature", "S235/1018 class")
STEEL_HARDNESS_PA = _p("STEEL_HARDNESS_PA", 1.3e9, "Pa", "literature", "Brinell ~130 HB")
ALU_YIELD_MPA = _p("ALU_YIELD_MPA", 276.0, "MPa", "literature", "6061-T6")
ALU_HARDNESS_PA = _p("ALU_HARDNESS_PA", 0.95e9, "Pa", "literature", "6061-T6 ~95 HB")
COOK_COEF = _p("COOK_COEF", 0.4, "-", "read",
               "Cook (1973) mean tool-chip interface temperature rise dT = 0.4 (u/rho C) (V t0/K)^(1/3), u specific "
               "energy, V cutting speed, t0 uncut chip thickness, rho C and K of the work; as reproduced in Groover, "
               "Fundamentals of Modern Manufacturing (lecture copy read: uomustansiriyah.edu.iq lecture 19)")
STEEL_RHO_C = _p("STEEL_RHO_C", 3.82e-3, "J/mm3 K", "literature", "AISI 1018: 7870 kg/m3 x 486 J/kg K (MatWeb)")
STEEL_DIFFUSIVITY = _p("STEEL_DIFFUSIVITY", 1.36e-5, "m2/s", "derived", "AISI 1018 k 51.9 W/m K / rho c (MatWeb)")
ALU_RHO_C = _p("ALU_RHO_C", 2.42e-3, "J/mm3 K", "literature", "6061-T6: 2700 kg/m3 x 896 J/kg K (MatWeb)")
ALU_DIFFUSIVITY = _p("ALU_DIFFUSIVITY", 6.9e-5, "m2/s", "derived", "6061-T6 k 167 W/m K / rho c (MatWeb)")

# ------------------------------------------------------------------ cutting laws: wood (MDF-shaped, Szwajka 2019)
WOOD_T0 = _p("WOOD_T0", 13.0, "N per 10 mm", "read", "MDF 10 mm drill thrust intercept (Szwajka 2019 uncoated)")
WOOD_TF = _p("WOOD_TF", 66.7, "N/(mm/rev) per 10 mm", "read", "MDF thrust slope")
WOOD_M0 = _p("WOOD_M0", 0.093, "N m per (10 mm)^2", "read", "MDF torque intercept")
WOOD_MF = _p("WOOD_MF", 0.747, "N m/(mm/rev) per (10 mm)^2", "read", "MDF torque slope")
KAPPA_WOOD = _p("KAPPA_WOOD", 1.0, "-", "estimate",
                 "factor on the intercepts of the wood law for Douglas fir. The law's form and intercepts are Szwajka's "
                 "MDF fit (742 kg/m3, 10 mm carbide twist drill, f 0.10-0.20 mm/rev, speed terms dropped), "
                 "TRANSFERRED to solid softwood: no measurement of the thrust a brad point needs before it cuts at all "
                 "was found (open)")
WOOD_SOLID_DATUM = _p("WOOD_SOLID_DATUM", dict(d_mm=25.4, f_mm_rev=1.02, thrust_n=150*4.448222, torque_nm=34*0.1129848),
                      "-", "literature",
                      "the one solid-wood drilling measurement found: McMillin & Woodson 1972, Forest Prod. J. 22(11):55, "
                      "southern pine SG 0.53, 25.4 mm spur machine bit, 1.02 mm/rev, across the grain, at 10.7 % "
                      "moisture: 150 lb and 34 in-lb, i.e. 667 N and 3.84 N m (Table 1, read in the paper's scan, "
                      "https://research.fs.usda.gov/download/treesearch/8240.pdf, 2026-10-08). The paper's averages over 0 to 80 % "
                      "moisture (138 lb, 33 in-lb: 618 N and 3.73 N m) are not the air-dry row. The same table: thrust peaks "
                      "at 4.4 % (186 lb) and falls to 100-120 lb from 19 % up")
WOOD_SOLID_TF = _p("WOOD_SOLID_TF", (WOOD_SOLID_DATUM["thrust_n"]/(WOOD_SOLID_DATUM["d_mm"]/10.) - 13.0)
                   / WOOD_SOLID_DATUM["f_mm_rev"]/66.7, "-", "derived",
                   "factor on the wood law's thrust slope for solid softwood, such that the law reproduces "
                   "WOOD_SOLID_DATUM's thrust with the MDF intercept kept: 3.67 (3.39 with the all-moisture "
                   "average). With it a 14.3 mm bit under 40 N feeds 0.061 mm/rev where it feeds 0.22 with the MDF "
                   "slope")
WOOD_SOLID_MF = _p("WOOD_SOLID_MF", (WOOD_SOLID_DATUM["torque_nm"]/(WOOD_SOLID_DATUM["d_mm"]/10.)**2 - 0.093)
                   / WOOD_SOLID_DATUM["f_mm_rev"]/0.747, "-", "derived",
                   "factor on the torque slope, from the same datum's torque: 0.66 (one point fixes one number per "
                   "law; the intercepts stay transferred)")
WOOD_GC = _p("WOOD_GC", 273.0, "J/m2", "read", "Scots pine cutting toughness (Chuchala 2021) as stand-in")
WOOD_ET = _p("WOOD_ET", 0.74e9, "Pa", "derived", "Douglas-fir E_T = 0.050 x 1.1 x 13.4 GPa (Wood Handbook)")
WOOD_NU = _p("WOOD_NU", 0.4, "-", "read", "Poisson ratio for the tear-out criterion")
FD_SLOPE = _p("FD_SLOPE", 0.006, "1/N", "derived", "Cimperman particleboard: Fd 1.00 @60 N, 1.08 @82 N, 1.34 @117 N")
FD_MAX = _p("FD_MAX", 1.8, "-", "read", "spade-bit exit Fd 1.55-1.82 (MDF)")
WOOD_DENSITY = _p("WOOD_DENSITY", 538.0, "kg/m3", "derived", "Douglas-fir Coast SG 0.48 at 12 % MC")
BRAD_SPUR_D_FRAC = _p("BRAD_SPUR_D_FRAC", 0.20, "x d", "estimate",
                      "centre spur diameter of a brad-point bit as a fraction of the bit diameter (about 3 mm on the "
                      "9/16 in bit); no maker's drawing found")
BRAD_SPUR_PIERCE_M = _p("BRAD_SPUR_PIERCE_M", 0.0002, "m", "estimate",
                        "depth at which a brad point's sharp centre spur has pierced the wood and seats the bit (a "
                        "twist drill seats when its chisel edge is in, 0.87 mm for 14.3 mm)")
BRAD_SPUR_LEN_M = _p("BRAD_SPUR_LEN_M", 0.003, "m", "estimate",
                     "length the centre spur leads the lips by, embedded ahead of them in the wood")
WOOD_EMBED_K = _p("WOOD_EMBED_K", 0.082, "N/mm2 per kg/m3", "literature",
                  "embedment strength of timber for a dowel, f_h = 0.082 (1 - 0.01 d) rho (EN 1995-1-1 eq. 8.32, d in "
                  "mm): the lateral load an embedded spur carries before it ploughs through the wood")
WOOD_SCORCH_C = _p("WOOD_SCORCH_C", (200.0, 300.0), "deg C", "read", "Wood Handbook: pyrolysis 200-300, char front 300")
TIMBER_AXIS_TOL_DEG = _p("TIMBER_AXIS_TOL_DEG", math.degrees(math.atan((9/16 - 1/2)*0.0254/0.0762)), "deg", "derived",
                         "a through-bolt hole square to the faces: a 1/2 in bolt in its 9/16 in hole through the 3 in "
                         "(76.2 mm) two-board stack can lean 1.19 deg in the hole, so a hole within 1.19 deg of the face "
                         "normal lets the bolt stand square to both faces (its head and washer bear flat). AWC NDS 2018 "
                         "12.1.3.2: holes 1/32 to 1/16 in larger than the bolt, 'accurately aligned', bolts not forcibly "
                         "driven (https://awc.org/wp-content/uploads/2026/08/AWC_NDS2018-withCommentary_20240204_AWC"
                         "Website_Chapter-12-Dowel-Type-Fasteners.pdf); no published angular tolerance was found")

# ------------------------------------------------------------------ concrete (percussive)
TE6S_MANUAL = "https://staff.washington.edu/nelsonr/manuals/Hilti%20TE%206-S.pdf"
CONCRETE_EV_BY_D = _p("CONCRETE_EV_BY_D", ((8.0, 0.328), (12.0, 0.204), (16.0, 0.187)), "mm, J/mm3", "derived",
                      "energy of hammer blows per volume of concrete removed, by bit diameter: from a maker's rated "
                      "drilling performance in medium-hard concrete, 28, 45 and 49 cm3/min at 8, 12 and 16 mm with "
                      f"1.8 J blows at 5,100 per minute (Hilti TE 6-S operating instructions, technical data, p. 7 of "
                      f"{TE6S_MANUAL}, read in the PDF 2026-10-08): blow power 153 W over 467, 750 and 817 mm3/s. "
                      "Interpolated linearly between the three diameters and held constant outside them "
                      "(materials.concrete_ev): 0.198 J/mm3 at the 13.3 mm anchor bit. The energy per volume falls "
                      "as the bit grows. Caveats: a brochure figure of another tool of nearly the modelled hammer's "
                      "blow power (161 W), concrete grade and push not stated. One value for every diameter, "
                      "CONCRETE_EV, drills the 13.3 mm hole at about half the maker's rate")
CONCRETE_EV = _p("CONCRETE_EV", 0.34, "J/mm3", "derived",
                 "KIT's measured rates with a 10 mm bit: interpolated 0.32 (C20/25) - 0.39 (C50/60). Kept as the second, "
                 "independent source and not used by the world: at 10 mm it is 28 % above CONCRETE_EV_BY_D "
                 "(0.266 J/mm3)")
AGGREGATE_EV_FACTOR = _p("AGGREGATE_EV_FACTOR", 1.3, "x", "estimate", "crushed-stone grains (open, swept 1-1.6)")
STEEL_EV = _p("STEEL_EV", 30.0, "J/mm3", "estimate", "carbide on rebar: effectively no penetration")
CARBIDE_CHIP_J = _p("CARBIDE_CHIP_J", 50.0, "J", "open", "blow energy on steel before the carbide head chips")
CONCRETE_ROT_MU = _p("CONCRETE_ROT_MU", 0.45, "-", "estimate", "rotation friction at the bit face")
JAM_RATE_NM_PER_DEG = _p("JAM_RATE_NM_PER_DEG", 2.0, "N m/deg", "estimate", "torque rise of an edge caught on a bar")
CONCRETE_DENSITY = _p("CONCRETE_DENSITY", 2350.0, "kg/m3", "literature", "plain normal-weight concrete")
AGGREGATE_FRACTION = _p("AGGREGATE_FRACTION", 0.38, "-", "estimate", "coarse aggregate volume fraction")
AGGREGATE_DMAX_M = _p("AGGREGATE_DMAX_M", 0.019, "m", "literature", "3/4 in maximum aggregate")
REBAR_D_M = _p("REBAR_D_M", 0.0127, "m", "read", "#4 bar")
REBAR_SPACING_M = _p("REBAR_SPACING_M", 0.30, "m", "authored", "near-face mat each way (typical 200-300 mm)")
REBAR_COVER_M = _p("REBAR_COVER_M", 0.025, "m", "authored",
                   "near (room) face: 1 in, above ACI 318-19 Table 20.5.1.3.1's 3/4 in minimum for walls not exposed to "
                   "weather or ground, No. 11 bar and smaller")
FAR_MAT_COVER_M = _p("FAR_MAT_COVER_M", 0.0381, "m", "read",
                     "ACI 318-19 Table 20.5.1.3.1, concrete exposed to weather or in contact with ground: No. 5 bar and "
                     "smaller 1-1/2 in (No. 6 through No. 18: 2 in), as reproduced at "
                     "https://help.idecad.com/ideCAD/shear-wall-reinforcement-detailing (search-level). The wall's bars "
                     "are No. 4")
REBAR_CLEAR_MARGIN_M = _p("REBAR_CLEAR_MARGIN_M", 0.002, "m", "authored",
                         "gap kept between the bit's edge and a near-mat bar when a wall mark is laid out after the scan, "
                         "on top of the bit and bar radii, the 3 mm placement and the 3 deg axis tolerances "
                         "(layout.rebar_clearance_m: 20.6 mm from a bar's axis on the bare wall, 22.6 mm through a ledger)")
DUST_BULKING = _p("DUST_BULKING", 1.6, "x", "estimate", "loose drilling dust volume / solid volume")
DUST_RESIDUAL_FRACTION = _p("DUST_RESIDUAL_FRACTION", 0.29, "-", "open",
                            "share of produced dust the turning flutes hold back (in transit; it settles when the bit "
                            "stops or withdraws). Set so a 108 mm hole drilled without intermediate withdrawals keeps "
                            "~25 mm of dust and ~10 mm with them (open estimates). No measured residual "
                            "found. Only bound: Hilti's 2017 KB-TZ letter says dust left "
                            "in an uncleaned hole 'can reduce the available anchor embedment depth, thus over-drilling of "
                            "the hole may be required', i.e. it can exceed h0 - h_nom = 13 mm on the 1/2 in anchor; the "
                            "E9 column after one withdrawal is 86 x this fraction mm (20.2 mm), so >= 0.15 is consistent "
                            "with that statement. Effect measured (E9): uncleaned 20.2 mm, 18.4 mm after "
                            "compaction under the anchor, anchor stops at 93.7 of 95 mm; cleaned 2.4 mm, 94.7 mm")
VACUUM_PASS_REMOVES = _p("VACUUM_PASS_REMOVES", 0.85, "-", "open",
                         "loose dust removed per nozzle pass to the bottom, only while the vacuum runs "
                         "(holes.vacuum_running); no per-pass removal data found. Effect (E9 clean): 4 passes and 2 strokes take "
                         "a 20.2 mm column to 2.4 mm")
BRUSH_STROKE_LOOSENS = _p("BRUSH_STROKE_LOOSENS", 0.60, "-", "open",
                          "packed dust loosened per brush stroke; no per-stroke data found (the KB-TZ2 procedure has no "
                          "brush: ESR-4266 4.2.3)")
DUST_COMPACT_RATIO = _p("DUST_COMPACT_RATIO", 0.91, "-", "literature",
                        "loose / compacted bulk density of a granular material: 'usually between 0.87 and 0.96' (BS EN "
                        "1097-3 loose vs rodded, aggregate lecture notes, Mustansiriyah Univ., https://uomustansiriyah.edu.iq/"
                        "media/lectures/5/5_2018_12_14!08_37_07_PM.pdf); mid value, transferred to drilling dust: the "
                        "dust column under a driven anchor shortens to this fraction (holes._insertion_update)")
DUST_COMPACT_PA = _p("DUST_COMPACT_PA", 1.0e4, "Pa", "estimate",
                     "pressure on the dust column above which it is pressed to its compacted density (rodding in BS EN "
                     "1097-3 is a light tamping); on a 13.3 mm hole 1.4 N, so a part resting on the dust under its own "
                     "weight barely compacts it and any push or blow does")
SPALL_DIAMETER_FACTOR = _p("SPALL_DIAMETER_FACTOR", 1.4, "x d", "estimate", "entry crater (1.8 d with > 40 N lateral)")

# ------------------------------------------------------------------ bits, heat, chips
HSS_E = _p("HSS_E", 219e9, "Pa", "read", "Bohler S600")
HSS_DENSITY = _p("HSS_DENSITY", 8100.0, "kg/m3", "read", "Erasteel E M2")
HSS_CP = _p("HSS_CP", 433.0, "J/kg K", "read", "Bohler S600")
HSS_BEND_STRENGTH = _p("HSS_BEND_STRENGTH", 4.7e9, "Pa", "read", "Erasteel E M2 4-point bend")
SDS_BODY_BEND_STRENGTH = _p("SDS_BODY_BEND_STRENGTH", 1.48e9, "Pa", "derived",
                            "outer-fibre stress at which an SDS bit's fluted steel body yields (holes.sds_bending_limit): "
                            "kwb HB2X SDS-plus bits have a 40Cr body hardened to HRC 45-52 "
                            "(https://www.kwb.eu/en/detail/index/sArticle/3368982, search-level); HRC 45 converts to "
                            "~1.48 GPa tensile strength (ASTM A370 / ISO 18265 conversion tables, lower end). The "
                            "HSS figure (4.7 GPa, 4-point bend of M2) does not apply to a tool-steel SDS body")
HSS_SHEAR_ULT = _p("HSS_SHEAR_ULT", 2.0e9, "Pa", "estimate", "0.35-0.5 of the bend strength")
FLUTE_I_FACTOR = _p("FLUTE_I_FACTOR", 0.4, "-", "derived", "fluted second moment / solid (0.3-0.5)")
STEEL_E_PA = _p("STEEL_E_PA", 200e9, "Pa", "literature", "Young's modulus of tool steel (a bit's bending stiffness 3 E I / L^3)")
BEND_TAU_MAX_S = _p("BEND_TAU_MAX_S", 0.05, "s", "engine",
                    "upper bound of the time constant with which a bit's bending moment follows the wall forces on it "
                    "(holes._bending_now): where neither the tool nor the workpiece can move (a press spindle over a "
                    "clamped bar) there is no mass behind an impact and the time constant sqrt(m_eff / k_bit) has no "
                    "meaning; a sustained load then registers in full within 0.15 s")
HEAT_TO_TIP = _p("HEAT_TO_TIP", (0.15, 0.40), "-", "estimate", "fraction of cutting power into the bit: metal, dry wood; "
                 "for metal also the share by which a hot tip body raises the corners' interface temperature "
                 "(Blok partition, holes._heat)")
TIP_H_W_M2K = _p("TIP_H_W_M2K", 20.0, "W/m2 K", "estimate", "still air; x4 with cutting oil")
HSS_SOFTEN_C = _p("HSS_SOFTEN_C", 450.0, "deg C", "estimate", "wear acceleration onset inside Movchan's reversible band")
HSS_BURN_C = _p("HSS_BURN_C", 550.0, "deg C", "read", "irreversible loss above ~820 K; Bohler tempers at 550-570")
CHIP_DOUBLE_DEPTH_D = _p("CHIP_DOUBLE_DEPTH_D", 3.5, "x d", "estimate", "torque doubles at 3.5 d packed depth (peck rule 3 d)")
BIT_HOLE_CLEARANCE_M = _p("BIT_HOLE_CLEARANCE_M", 3e-5, "m", "literature",
                          "radial running clearance of a drill in its own hole: drilled holes are 0.02-0.24 mm oversize (OSG)")
FEED_MAX_D = _p("FEED_MAX_D", 0.2, "x d mm/rev", "estimate",
                "largest chip per revolution a twist drill can cut (thin-sheet grabs reach ~1 mm/rev on 6 mm, OSG)")
FEED_GRAB_MAX_D = _p("FEED_GRAB_MAX_D", math.pi/math.tan(math.radians(30.)), "mm/rev per mm of d", "derived",
                     "after the exit cap collapses the lips catch the remaining ring like a thread: the chip per "
                     "revolution is bounded by the flute's helix lead pi d/tan(30 deg), not by the lips' chip capacity "
                     "(lunge and grab; with the 0.2 d cap a 6 mm drill through 1.5 mm sheet peaked at "
                     "1.0 N m and never grabbed: E6)")
CHIP_ONSET_D = _p("CHIP_ONSET_D", 2.0, "x d", "estimate",
                  "packing torque starts beyond 2 d of unretracted depth (chips 'cluster beyond about 3 d', Mellinger)")
LIP_ERROR_M = _p("LIP_ERROR_M", 3e-5, "m", "estimate", "lip-height error of a hand-ground jobber")
ASYM_GAIN = _p("ASYM_GAIN", 5.0, "-", "open", "lateral force from lip asymmetry (2-10)")
TIP_MU = _p("TIP_MU", 0.4, "-", "literature", "spinning tip lateral friction coefficient")

# ------------------------------------------------------------------ anchors and fasteners
KBTZ2 = _p("KBTZ2", dict(d=0.0127, h_nom=0.095, h0=0.108, torque=68.0), "m, m, m, N m", "read",
           "Hilti KWIK Bolt TZ2 1/2 in carbon steel: stud diameter 1/2 in; nominal embedment 3-3/4 in; hole depth "
           "4-1/4 in; installation torque 50 ft lb (ICC-ES ESR-4266 Table 1 and the Hilti technical supplement's "
           "setting table)")
ANCHOR_CLIP_N = _p("ANCHOR_CLIP_N", 500.0, "N", "open",
                   "KB-TZ2 expansion clip drag on insertion in the nominal 13.3 mm hole (the split clip springing against "
                   "the wall, CLIP_WALL_MU x its ring force); it sets the clip's ring stiffness (parts.WedgeAnchor), so "
                   "the drag falls in a larger hole and is zero beyond the clip's free diameter. No published value "
                   "in newtons was found. 150 N is a drag a steady push overcomes (160 N seats an anchor in 0.4 s), "
                   "where the approval document requires hammering ('The anchor must be hammered into the predrilled hole "
                   "until hnom is achieved', ICC-ES ESR-4266, 4.2.3), a setting-tool patent says the clip 'is slightly "
                   "larger than the pre-drilled hole and requires that a significant force be applied' (US 8,424,180) "
                   "and an industrial arm of 13 kg payload pushed a 12 mm wedge anchor 7 mm in before it stuck "
                   "(arXiv 2403.19948, IV-B; no force given). 500 N is chosen between two bounds, neither a "
                   "measurement: above what one FR3 pushes steadily (136-158 N) and what a hand presses on a stud's "
                   "end, and such that an installer's 2 lb hammer (KIT_HAMMER) drives the anchor in a number of "
                   "blows a person would accept (E20, E12). A "
                   "split steel ring of the clip's size pressed 0.4 mm smaller gives 200 to 2,600 N for a band 0.8 to "
                   "1.5 mm thick (piston-ring formula, mu 0.6): the order, not a value. It is the model's only "
                   "resistance to driving the anchor and does not depend on speed. In a hole worn or drilled "
                   "oversize the drag is smaller in proportion and an anchor can again be pushed in (E9's hole, 13.61 mm: "
                   "113 N): such a hole also lets the anchor spin or pull up (E20)")
# KB-TZ2 setting (parts.WedgeAnchor). ESR-4266 3.1: 'a tapered mandrel ... enclosed by a
# three-section expansion element ... restrained by the mandrel taper and by a collar. When torque is applied to the
# nut ..., the mandrel is drawn into the expansion element, which is in turn expanded against the wall of the hole'
ESR4266 = "ICC-ES ESR-4266 (rev. June 2026), https://icc-es.org/wp-content/uploads/report-directory/ESR-4266.pdf"
KBTZ2_PITCH_M = _p("KBTZ2_PITCH_M", 0.0254/13, "m", "read", "1/2-13 UNC thread: 13 threads per inch, 1.954 mm per turn "
                   "(ASME B1.1; the KB-TZ2 1/2 in stud carries a 1/2-13 thread and an ASTM A563 Grade A hex nut, "
                   f"{ESR4266} section 3.1)")
KBTZ2_D2_M = _p("KBTZ2_D2_M", 0.0254*(0.5 - 0.649519/13), "m", "derived",
                "basic pitch diameter of 1/2-13 UNC, d - 0.649519 P = 0.4500 in = 11.43 mm (ASME B1.1 thread form)")
KBTZ2_THREAD_LEN_M = _p("KBTZ2_THREAD_LEN_M", {"G": 2.375*0.0254, "L": 4.75*0.0254}, "m", "read",
                        f"thread length from the top: 1/2 x 4-1/2 (G) 2-3/8 in, 1/2 x 7 (L) 4-3/4 in (Hilti KB-TZ2 Technical Supplement, Dec 2020, Table 47); "
                        "the nut cannot run past it")
NUT_BEARING_R_M = _p("NUT_BEARING_R_M", 0.25*(0.01905 + 0.0135), "m", "derived",
                     "mean radius of the nut's bearing annulus on the washer: from the washer's 13.5 mm hole (0.53 in, "
                     "estimate) to the nut's 3/4 in across flats (ASME B18.2.2), (D_o + D_i)/4")
NUT_FACTOR_K = _p("NUT_FACTOR_K", 0.20, "-", "estimate",
                  "torque factor K (T = K F d): the generic handbook value for zinc-plated threads, Shigley's Mechanical "
                  "Engineering Design Table 8-15 (non-plated black 0.30, zinc-plated 0.20, lubricated 0.18), used as "
                  "an estimate for the KB-TZ2, whose own factor is not published: its "
                  "carbon-steel stud is zinc-nickel plated (ESR-4266 3.1) and 'nuts and bolts are finished with a "
                  "proprietary coating. Only Hilti KB-TZ2 nuts can be used with KB-TZ2 bolts' (Hilti KB-TZ2 Technical "
                  "Supplement, Dec 2020, p. 3, https://images.thdstatic.com/catalog/pdfImages/a3/a3a0c225-ae97-40b5-"
                  "92b3-745632435551.pdf). It decides the preload: the thread and bearing friction below are solved "
                  "from it, so the stud tension at 68 N m is 68/(K d) = 26.8 kN by arithmetic (63 % of the neck's "
                  "42.2 kN yield), unconfirmed by any measurement on this anchor")
THREAD_MU = _p("THREAD_MU", (NUT_FACTOR_K*0.0127 - (0.0254/13)/(2*math.pi))/(0.577*0.0254*(0.5 - 0.649519/13) +
                                                                            0.25*(0.01905 + 0.0135)), "-", "derived",
               "thread (mu_G) and nut-bearing (mu_K) friction of the zinc-plated nut on the stud and washer, taken equal: "
               "the value for which the ISO 16047 / VDI 2230 torque split T = F (P/2 pi + 0.577 mu_G d2 + mu_K r_b) gives "
               "NUT_FACTOR_K on the 1/2-13 thread; the solved preload at 68 N m is then T/(K d) = 26.8 kN")
BEARING_MU = _p("BEARING_MU", THREAD_MU, "-", "derived", "nut-on-washer bearing friction, equal to THREAD_MU (see there)")
CONE_ANGLE_DEG = _p("CONE_ANGLE_DEG", 3.5, "deg", "literature",
                    "taper of the expansion cone (mandrel) against the axis: 'The optimum angle is 3.5 deg', range 2.5-4.5, "
                    "for a 1/2 in expansion stud with a 1.0 in wedge tip and a 0.5 in sleeve (US Patent 4,052,925, US "
                    "Expansion Bolt Co., 1977, https://patents.google.com/patent/US4052925); the KB-TZ2's own taper is "
                    "not published (estimate for it)")
CONE_MU = _p("CONE_MU", 0.16, "-", "estimate",
             "static friction of the expansion element on the cone (mandrel): the value ASSUMED as an input by the "
             "finite-element study of a torque-controlled sleeve anchor (a different anchor) of Noh, Kim and Joo, Trans. "
             "Korean Nuclear Society, Spring 2014 (https://www.kns.org/files/pre_paper/31/601노상훈.pdf); not a "
             "measurement, and not of a KB-TZ2. The study shows that follow-up expansion needs this to stay below "
             "the clip-on-concrete friction. Here it enters only the friction that keeps the stud from turning with "
             "the nut (parts.WedgeAnchor)")
CLIP_WALL_MU = _p("CLIP_WALL_MU", 0.6, "-", "estimate",
                  "static friction of the expansion element on the concrete hole wall: an assumed input of the same "
                  "finite-element study (Noh, Kim and Joo 2014, as CONE_MU), not a measurement. Here it only converts "
                  "the clip's drag ANCHOR_CLIP_N into its ring force (which sets the stud's spin friction before the "
                  "cone grips)")
CONE_R_M = _p("CONE_R_M", 0.25*(0.0090 + 0.0127), "m", "derived",
              "mean radius of the cone under the clip: from the 9.0 mm neck (ESR-4266 effective tensile stress area "
              "63.6 mm2 for the 1/2 in carbon-steel anchor, d = sqrt(4 A/pi)) to the 12.7 mm body")
WEDGE_BETA_N_M = _p("WEDGE_BETA_N_M", 158585*4.448222/0.0254, "N/m", "read",
                    f"axial stiffness of the installed 1/2 in KB-TZ2 in the service load range, uncracked concrete, "
                    f"beta_uncr = 158,585 lb/in = 27.8 kN/mm ({ESR4266}, design table, 1/2 in column; cracked 113,515 "
                    "lb/in). Used as the stiffness of the cone being drawn into the clip (follow-up expansion), i.e. the "
                    "stud rise per newton of stud tension while setting: no published stud rise during setting was found, "
                    "so the setting displacement F/beta (about 1 mm at 26.8 kN) is derived, "
                    "not checked against an independent measurement")
WEDGE_STEEL_N = _p("WEDGE_STEEL_N", 11240*4.448222, "N", "read",
                   f"nominal steel strength in tension of the 1/2 in carbon-steel KB-TZ2, N_sa = 11,240 lb = 50.0 kN "
                   f"({ESR4266}, Table 4, read 2026-10-07: f_uta 114,000 psi on A_se,N 0.099 in2 = 63.6 mm2; yield "
                   "f_y 96,300 psi, 42.2 kN on that area). The capacity of the tie that models the expansion "
                   "(parts.WedgeAnchor): at this stud tension the neck breaks and the anchor has failed for good. With "
                   "NUT_FACTOR_K it corresponds to 127 N m at the nut, 1.87 x the installation torque. The same table "
                   "gives no pullout strength for the 1/2 in anchor (N/A: it does not govern) and k_uncr = 24 at "
                   "h_ef 3-1/4 in for concrete breakout (not modelled: see the class docstring)")
KIT_HAMMER = _p("KIT_HAMMER", dict(head_kg=0.907, head_len_m=0.10, handle_len_m=0.34, handle_r_m=0.013, head_x_m=0.179),
                "kg, m", "read",
                "the kit's hand hammer for driving wedge anchors: a 32 oz (2 lb, 907 g) ball-peen, 14-1/2 in (368 mm) "
                "overall (Estwing E3-32BP as listed by distributors, searched 2026-10-08: "
                "https://www.drapertools.com/product/23578/estwing-e3-32bp-ball-pein-hammer-32oz-907g/). Built as "
                "a steel cylinder of that mass 100 mm long (face 38 mm) on a hickory handle; head_x is the head's "
                "centre from the handle's middle. The head collides as that plain cylinder (it is drawn with its "
                "ball and its eye section): dropped, it can roll on its side for more than 10 s and can stand on "
                "the end where its ball is drawn (E14's sweep). Against the clip's 500 N drag (ANCHOR_CLIP_N) a 16 oz "
                "hammer (0.45 kg head) would need three times the blows")
HAMMER_SWING_M_S = _p("HAMMER_SWING_M_S", 5.0, "m/s", "authored",
                      "speed the phantom hand gives the kit hammer before a blow on an anchor (E12, E20, E9): a firm "
                      "one-hand swing, 13 J with the 1.04 kg hammer. An operator's choice, not a property of the "
                      "world; no measurement of installers' hammer speeds was sought")
CLIP_FREE_D_M = _p("CLIP_FREE_D_M", 0.0137, "m", "estimate",
                   "outside diameter of the unexpanded clip, free: above the largest hole a standard 1/2 in carbide bit "
                   "drills (ANSI B212.15 tip 0.520-0.530 in, 13.21-13.46 mm), so the clip drags in every standard hole. "
                   "A hole larger than this leaves the clip loose: the anchor cannot be set (spins or pulls up). Not "
                   "published; its effect is measured in E20 (oversize holes)")
CLIP_LEN_M = _p("CLIP_LEN_M", 0.020, "m", "estimate", "length of the expansion clip along the stud (2-22 mm from the tip)")
CLIP_TRAVEL_M = _p("CLIP_TRAVEL_M", 0.0127, "m", "estimate",
                   "travel of the cone into the clip before the clip runs off the taper: the 1.0 in wedge tip less the "
                   "0.5 in sleeve of US 4,052,925 (at 3.5 deg, 0.78 mm of radial expansion)")
CLIP_PLAY_M = _p("CLIP_PLAY_M", 0.0005, "m", "estimate",
                 "axial play of the clip between the collar it is driven by and the start of the taper (ESR-4266 3.1: "
                 "'restrained by the mandrel taper and by a collar'); the stud rises this far before the cone meets the "
                 "clip. Not published")
ANCHOR_SET_ON_N = _p("ANCHOR_SET_ON_N", 20.0, "N", "authored",
                     "thread load, and washer load on the fixture, above which the nut is taken to bear "
                     "(parts.WedgeAnchor: only then can the stud's tension draw the cone into the clip; the evaluator "
                     "counts the stud's turning as spin only while the nut bears)")
NUT_SLIDE_ARMATURE_KG = _p("NUT_SLIDE_ARMATURE_KG", 10.0, "kg", "engine",
                           "inertia on the nut's slide along the stud (a relative coordinate; parts.WedgeAnchor.bind): "
                           "it keeps the nut on its thread position through a hammer blow (0.002-0.007 mm of slip; "
                           "without it 0.03-0.35 mm for a few ticks at every blow, with no fault and no stud tension: "
                           "E20 seating_blows and real_hammer on SOLVER_LS_ITERATIONS 200)")
NUT_TURN_ARMATURE = _p("NUT_TURN_ARMATURE", 1e-4, "kg m2", "engine",
                       "inertia on the nut's turn relative to the stud (parts.WedgeAnchor.bind), 100 x the nut's own "
                       "and about that of the anvil and socket that drive it (8e-5 kg m2). Without it the nut's own "
                       "friction stops the bare nut within a tick whenever the socket's flats lose contact, and in a "
                       "run-down at 2300 rpm the socket then stands up to 11.3 deg into the nut's hex for one tick "
                       "(E20 dusty; 4.6 deg with it; module at 2300 rpm). Cost: it reacts on the stud, "
                       "which turns 2-6 deg in its hole during a setting in a nominal hole, 10-12 deg in E9's "
                       "oversize hole and 15 deg in a 13.6 mm hole (0.25 deg without; the evaluator's limit is 45 deg). The "
                       "wrench's blows do not need it (E20 sound without it at 1500 rpm: 0 corners "
                       "passed, hex at most 5.9 deg, the set nut 0.06 deg and 12 N back in 2 s)")
CLIP_END_BAND_M = _p("CLIP_END_BAND_M", 2e-5, "m", "engine",
                     "the clip counts as against its collar, or the cone as against the clip, within this distance of "
                     "that end of the clip's play (parts.WedgeAnchor.pre_step): with 0.1 um the 5 % velocity reversal of "
                     "the friction row that stops the stud took the stud off the end every other tick and the clip was "
                     "dragged out 0.3 mm at half its drag; 20 um is 4 % of the play")
CLIP_WRONG_WAY_N = _p("CLIP_WRONG_WAY_N", 1.0, "N", "engine",
                      "the clip's drag is taken off for a tick when the last tick's drag held the stud with more than "
                      "this force in the direction that end of the play cannot hold (parts.WedgeAnchor.pre_step); 0.7 % "
                      "of the drag, above the solver's residual on a resting stud")
NUT_AF_M = _p("NUT_AF_M", 0.01905, "m", "literature",
              "1/2-13 hex nut, width across flats 3/4 in (ASME B18.2.2 hex nuts; from memory)")
NUT_H_M = _p("NUT_H_M", 0.0111, "m", "literature", "1/2-13 hex nut, thickness 7/16 in (ASME B18.2.2; from memory)")
WASHER_T_M = _p("WASHER_T_M", 0.0032, "m", "estimate",
                "thickness of the KB-TZ2's washer. The carbon-steel anchor's 'washers meet the requirements of ASTM "
                "F844' (Hilti KB-TZ2 Technical Supplement, Dec 2020, p. 3; for the stainless anchors the same page "
                "names 'ANSI B18.22.1, Type A, plain'). Neither standard was opened and the supplement does not give "
                "the size: a 1/2 in Type A wide plain washer is 0.086-0.132 in thick (from memory)")
WASHER_R_M = _p("WASHER_R_M", 0.0172, "m", "estimate",
                "outer radius of the KB-TZ2's washer (an ASTM F844 washer, see WASHER_T_M; size not given by Hilti: a "
                "1/2 in Type A wide plain washer is 1.375 in across, 17.5 mm radius; from memory)")
NUT_CHAMFER_H_M = _p("NUT_CHAMFER_H_M", 0.00085, "m", "derived",
                     "depth of the hex nut's top chamfer at its corners (parts.anchor: over this depth the nut's top is a hex "
                     "with its corners on the across-flats circle). ASME B18.2.2 (as summarized by engineersedge.com and "
                     "torqbolt.com, read 2026-10-06): chamfer circle between the width across flats F and 0.95 F, chamfer "
                     "length at the corners 5-15 % of the thread diameter (0.64-1.9 mm for 1/2 in); with a 30 degree chamfer "
                     "from the circle F to the corners G: (21.99 - 19.05)/2 x tan 30 = 0.85 mm")
NUT_PREVAILING_NM = _p("NUT_PREVAILING_NM", 0.02, "N m", "estimate",
                       "torque that turns the free (unloaded) nut on the stud and the washer under it, so that a loose "
                       "anchor's nut turns only under an applied torque; below the clip's spin hold in a hole that fits it "
                       "(CONE_MU x ring force x CONE_R_M, 0.22 N m in the nominal hole), above it when the clip is loose")
NUT_HEX_SOLIMP = _p("NUT_HEX_SOLIMP", (0.9999, 0.9999, 0.0005), "-", "engine",
                    "contact impedance of the nut's hex and of the socket's hex walls (parts.WedgeAnchor.bind, "
                    "tools.ImpactWrench.bind): MuJoCo's stiffest. With the hardened faces' STIFF_SOLIMP the socket stood "
                    "2 degrees into the nut's flats at 57 N m (a contact stiffness of about 1,600 N m/rad about the "
                    "axis, softer than the wrench's own jaw spring of 5,400): each blow lost T^2/2k in it and the "
                    "wrench stalled at 59 N m after 188 blows (E20 sound)")
NUT_TURNING_RAD_S = _p("NUT_TURNING_RAD_S", 0.2, "rad/s", "authored",
                      "nut speed on the stud above which the nut counts as being turned on (parts.WedgeAnchor records "
                      "the friction torque it turns against as the installation torque only then): 2 % of the "
                      "speed a blow of the wrench turns it at")
ANCHOR_SPIN_MAX_DEG = _p("ANCHOR_SPIN_MAX_DEG", 45.0, "deg", "authored",
                         "the evaluator's limit on the stud turning in its hole while its nut is loaded: an anchor that "
                         "turns with the nut has spun (the installer's sign of a bad hole); no published threshold")
T_INST_TOL = _p("T_INST_TOL", 0.10, "-", "authored",
                "the evaluator accepts an installation torque from 0.9 T_inst (ISO 6789 torque-wrench accuracy is 4-6 %, "
                "plus margin); above 1.7 T_inst is over-torque: the setting torque of ACI 355.2's splitting test, "
                "'greater than the smaller of 1.7 T_inst and T_inst + 100 ft-lb' (test 9.3, as quoted by R. Morgan, "
                "STRUCTURE magazine, Apr 2017, https://www.structuremag.org/article/how-installation-torque-can-"
                "affect-expansion-anchors/; 1.7 T_inst is the smaller for this anchor). For the 1/2 in KB-TZ2 with "
                "NUT_FACTOR_K that is 115.6 N m and 45.5 kN, between the neck's yield (42.2 kN) and its strength "
                "(WEDGE_STEEL_N, 50.0 kN, 127 N m), where the model's anchor fails")
# the setting tool (tools.ImpactWrench): Hilti SIW 6AT-A22 with the SI-AT-A22 Adaptive Torque module, which ESR-4266
# 4.2.3 permits for the 1/2 in KB-TZ2 in place of a torque wrench, except where Hilti's installation instructions
# exclude it: "AT-System not applicable for grout-filled CMU and wood fastening applications" (2274242-03.2025). The
# ledger-anchor task uses it through a timber ledger: a disclosed deviation (layout.wall_layout)
SIW_SPECS = ("Hilti SIW 6AT-A22 technical data: 1.04 kg (tool), 1/2 in square drive, 3 gears 1000/1500/2300 rpm, max torque "
             "90/135/213 N m, up to 3,500 impacts/min, 228 x 154 x 94 mm (hire listing https://www.speedyservices.com/hire/"
             "power-tools/02_0288-h-Hilti-SIW-6AT-A22--Impact-Wrench-22v-104kg, Hilti product-page search snippet)")
IMPACT_TOOL_KG = _p("IMPACT_TOOL_KG", 1.04, "kg", "read", SIW_SPECS)
IMPACT_BATTERY_KG = _p("IMPACT_BATTERY_KG", 0.83, "kg", "read",
                       "Hilti B 22/5.2 Li-ion pack (22 V, 5.2 Ah), weight 0.830 kg: a distributor's data table for the "
                       "Hilti item (https://www.oglaend-system.com/product-variants/battery-pack-b22-5-2-li-ion-"
                       "article39899-52088.html?pid=42260, read 2026-10-08; Hilti's own pages refuse automated "
                       "reading)")
AT_MODULE_KG = _p("AT_MODULE_KG", 0.26, "kg", "read",
                  "the SI-AT-A22 Adaptive Torque module between tool and battery: 0.26 kg (EPTA Procedure 01/2003, "
                  "without battery; Hilti's product pages for the module as returned by a search, 2026-10-08: "
                  "https://www.hilti.co.nz/r4838537 and regional copies; the page itself could not be opened). With these two the wrench weighs "
                  "1.04 + 0.83 + 0.26 + 0.20 (socket) = 2.33 kg, more than one FR3 carries beyond its hand (2.27 kg)")
SOCKET_KG = _p("SOCKET_KG", 0.20, "kg", "estimate", "3/4 in deep impact socket on a 1/2 in drive (63 mm long, 28 mm OD) with "
               "the anvil: the turning mass the blows accelerate")
SOCKET_CLEARANCE_M = _p("SOCKET_CLEARANCE_M", 0.00015, "m", "estimate",
                        "clearance of the socket's hex on each flat of the 3/4 in nut (a 19.35 mm opening on 19.05 mm)")
SOCKET_LEAD_IN_M = _p("SOCKET_LEAD_IN_M", 0.001, "m", "estimate",
                      "the 45 degree chamfer at the mouth of the socket's hex (tools.ImpactWrench): the socket finds a nut "
                      "up to this far off its axis and takes the nut's corners in before the hexes are in line; not "
                      "published for Hilti's sockets (impact sockets carry a lead-in of about a millimetre)")
IMPACT_MAX_RPM = _p("IMPACT_MAX_RPM", 2300.0, "rpm", "read", SIW_SPECS)
IMPACT_MAX_IPM = _p("IMPACT_MAX_IPM", 3500.0, "1/min", "read", SIW_SPECS)
IMPACT_MAX_TORQUE_NM = _p("IMPACT_MAX_TORQUE_NM", 213.0, "N m", "read", SIW_SPECS)
IMPACT_CAM_NM = _p("IMPACT_CAM_NM", 5.0, "N m", "estimate",
                   "anvil torque at which the hammer's spring lets the cam ride off and blows begin (below it the anvil "
                   "turns with the motor, as a nut runs down its thread); not published")
IMPACT_ROTOR_INERTIA = _p("IMPACT_ROTOR_INERTIA", 5e-5, "kg m2", "estimate",
                          "motor rotor and gear train reflected to the anvil while the hammer is engaged (running)")
IMPACT_SPINUP_S = _p("IMPACT_SPINUP_S", 0.10, "s", "estimate",
                     "time the motor takes from the trigger's pull to its set speed (the electronic soft start of a "
                     "brushless tool; tools.ImpactWrench ramps the anvil's speed linearly); not published for the SIW 6AT-A22")
IMPACT_GEAR_RPM = _p("IMPACT_GEAR_RPM", (1000.0, 1500.0, 2300.0), "rpm", "read",
                     "no-load speeds of the wrench's three gears; their rated torques 90/135/213 N m are 0.090, 0.090 and "
                     f"0.093 N m per rpm, proportional to speed ({SIW_SPECS}). The plain wrench (no Adaptive Torque module) "
                     "runs at its gear's speed")
IMPACT_HAMMER_INERTIA = _p("IMPACT_HAMMER_INERTIA", 0.30*0.022**2, "kg m2", "estimate",
                           "moment of inertia of the wrench's hammer about the anvil's axis: a 0.30 kg steel ring with a "
                           "22 mm radius of gyration (it has to fit the 60 mm head of a 1.04 kg tool); not published. It "
                           "sets the energy of a blow, I w^2/2 = 4.2 J at 2300 rpm (245 W at 3,500 blows/min), and with "
                           "IMPACT_JAW_STIFFNESS how many blows a setting takes (the count is proportional to 1/I)")
IMPACT_JAW_STIFFNESS = _p("IMPACT_JAW_STIFFNESS",
                          (IMPACT_MAX_TORQUE_NM/(IMPACT_MAX_RPM*2*math.pi/60))**2/IMPACT_HAMMER_INERTIA, "N m/rad", "derived",
                          "one-way torsion spring between the hammer and the anvil (jaws, anvil, socket): the value for "
                          "which a blow at the largest speed peaks at the maker's largest torque against a nut that does "
                          "not turn, w sqrt(k I) = IMPACT_MAX_TORQUE_NM at IMPACT_MAX_RPM (5,384 N m/rad with the hammer "
                          "above; only the product k I is calibrated). Same order as the torsional stiffness of a 1/2 in "
                          "square drive 25 mm long, G J/L = 80 GPa x 0.1406 x (12.7 mm)^4 / 25 mm = 11,700 N m/rad, in "
                          "series with the jaws and the socket")
IMPACT_DILATION = _p("IMPACT_DILATION", 16.0, "-", "engine",
                     "time dilation of a blow (tools.ImpactWrench, declared device): the hammer's inertia x 16^2 at 1/16 "
                     "of its speed on the same spring, and the anvil's and socket's inertia x 16^2 for the duration of a "
                     "blow. Blow energy and peak torque are unchanged; a blow lasts 5-50 ms instead of 0.3-3 ms and the "
                     "hammer moves 0.9 deg per 1 ms tick at 2300 rpm (13.8 deg undilated, at which the socket passes "
                     "the nut's corners; dilated, the anvil turned at most 21 rad/s, 1.2 deg a tick, while impacting, E20). "
                     "The explicit spring needs 16^2 x the anvil's 3.06e-5 kg m2 >= k dt^2 = 5.4e-3 (factor "
                     ">= 13.3): with 10 the peak against a held nut came out at 86 N m for 139. Checked against a "
                     "2 microsecond undilated solution of the same equations (hammer, spring, a nut whose friction "
                     "torque rises 21.9 N m/rad): 65 blows and 1.74 s to 68 N m at 1500 rpm for 63 and 1.66 s; held "
                     "nut 138.9 N m for 138.9; gear 3 unmoderated 154 N m after 3 s for 166")
AT_FIRST_RPM = _p("AT_FIRST_RPM", 2300.0, "rpm", "authored",
                  "motor speed the Adaptive Torque module sets for the scanned 1/2 in KB-TZ2 (the patent's 'first "
                  "rotational speed'; US 10,589,407 B2, https://patents.google.com/patent/US10589407B2: 'rotational "
                  "speeds assigned to different tightening torques are stored', Hilti's table is not published): the "
                  "wrench's full speed, where a blow carries 4.2 J and peaks at 213 N m against a held nut, so the "
                  "module's stop on the sensed torque does the limiting. Chosen on the one timing the patent gives, "
                  "'approximately 2 seconds' for an impact setting: at 1500 rpm (1.8 J a blow) the setting took 3.2 s "
                  "and 111 blows on a concrete face and ran into the 5 s time-out through a wood ledger (183 blows, "
                  "67.3 N m; E20). The speed whose blows peak at 68 N m itself (734 rpm) is no program "
                  "at all: the torque then approaches 68 N m only asymptotically (reduced model: 0.9 of it after "
                  "about 390 blows, 21 s)")
AT_TIMEOUT_S = _p("AT_TIMEOUT_S", 5.0, "s", "authored",
                  "impacting time after which the module stops without green (red); the patent puts a whole impact setting "
                  "at about 2 s. The module's stop itself has no parameter of its own: it stops (green) when the torque "
                  "a blow passed to the socket reaches the anchor's installation torque (the threshold is authored: the "
                  "patent's 'predetermined threshold value' is not published)")
KBTZ2_SUPPLEMENT = ("Hilti KWIK Bolt TZ2 Concrete and Masonry Technical Supplement, Dec 2020, Table 47 "
                    "(https://images.thdstatic.com/catalog/pdfImages/a3/a3a0c225-ae97-40b5-92b3-745632435551.pdf)")
KBTZ2_WALL_LENGTH_M = _p("KBTZ2_WALL_LENGTH_M", 4.5*0.0254, "m", "read",
                         f"KB-TZ2 1/2 x 4-1/2 (letter G): fixture 0 to 1/4 in at h_nom 3-3/4 in ({KBTZ2_SUPPLEMENT}); "
                         "clean-and-set seats its washer on the bare wall")
KBTZ2_LEDGER_LENGTH_M = _p("KBTZ2_LEDGER_LENGTH_M", 7.0*0.0254, "m", "read",
                           f"KB-TZ2 1/2 x 7 (letter L), the shortest that fastens a 1-1/2 in ledger at h_nom 3-3/4 in: "
                           f"fixture 0 to 2-3/4 in (1/2 x 5-1/2 takes only 1-1/4 in) ({KBTZ2_SUPPLEMENT}). Its washer "
                           "seats h_nom + the ledger thickness from the tip")
KBTZ2_CLEANING = ("ICC-ES ESR-4266 (rev. June 2026) section 4.2.3: 'If dust and debris is removed from the drilled hole "
                  "with ... compressed air or a manual pump, hnom is achieved at the specified value of h0 ... The anchor "
                  "must be hammered into the predrilled hole until hnom is achieved' (https://icc-es.org/wp-content/"
                  "uploads/report-directory/ESR-4266.pdf); Figure 7 path 1a (carbide bit) has no cleaning step; Hilti: "
                  "'The KB-TZ2 does not require hole cleaning when using this drilling method, but it can be done if "
                  "desired' (https://www.hilti.com/engineering/question/drill-bit-for-kb-tz2-anchors/uknp9g)")
ABORTED_HOLE_DISTANCE_FACTOR = _p("ABORTED_HOLE_DISTANCE_FACTOR", 2.0, "x aborted depth", "read",
                                  "a new hole at least twice the depth of the aborted hole away from it, or closer if "
                                  "the aborted hole is filled with high-strength mortar (filling is the alternative, not "
                                  "a requirement): ETAG 001 Part 1 section 7.3 (https://umwelt-online.de/recht/bau/bauprod/"
                                  "etag001_1b.htm); ETA-06/0168 Annex B1 'In case of aborted hole: new drilling at a "
                                  "minimum distance away of twice the depth of aborted hole or smaller distance if the "
                                  "aborted hole is filled with high strength mortar and if under shear or oblique tension "
                                  "load it is not in the direction of the load application' (https://www.dibt.de/"
                                  "pdf_storage/2025/ETA-06!0168(8.06.01-128!25)e.pdf). Measured from the aborted hole; "
                                  "ESR-4266 (US) has no aborted-hole text")
RELOCATE_MAX_M = _p("RELOCATE_MAX_M", 0.15, "m", "authored",
                    "farthest a relocated anchor may move along its strut channel from the aborted hole without the "
                    "layout being redrawn (half the 300 mm anchor spacing); excludes the line's other anchor hole")
LEDGER_THICK_M = _p("LEDGER_THICK_M", 0.038, "m", "literature", "2x lumber dressed 1-1/2 in (38 mm)")
LEDGER_DEPTH_M = _p("LEDGER_DEPTH_M", 0.184, "m", "literature", "2x8 dressed 7-1/4 in (184 mm)")
IRC_LEDGER = ("IRC Table R507.9.1.3(2) 'Placement of lag screws and bolts in deck ledgers' (ledger: top edge 2 in, bottom "
              "edge 3/4 in, ends 2 in, row spacing 1-5/8 in; staggered top to bottom), reproduced at https://gocodebook.com/"
              "us/california/california-residential-code/floors/slabs-decks-and-steel-framing/exterior-decks-attachment")
LEDGER_EDGE_TOP_M = _p("LEDGER_EDGE_TOP_M", 2.0*0.0254, "m", "read", IRC_LEDGER)
LEDGER_EDGE_BOTTOM_M = _p("LEDGER_EDGE_BOTTOM_M", 0.75*0.0254, "m", "read", IRC_LEDGER)
LEDGER_ENDS_M = _p("LEDGER_ENDS_M", 2.0*0.0254, "m", "read", IRC_LEDGER)
LEDGER_ROW_SPACING_M = _p("LEDGER_ROW_SPACING_M", 1.625*0.0254, "m", "read", IRC_LEDGER)
BOSCH_SDS = "Bosch Bulldog SDS-plus bits (https://www.boschtools.com/us/en/brocas-para-martillo-perforador-bulldog-sds-plus%C2%AE-43549-ocs-ac/)"
SDS_BIT_WORKING_M = _p("SDS_BIT_WORKING_M", 6*0.0254, "m", "read",
                       f"Bosch HC2083 1/2 in: working length 6 in, total 8 in ({BOSCH_SDS}); the 4 in working-length bit "
                       "(HC2081) cannot drill the 108 mm anchor hole")
SDS_BIT_STICKOUT_M = _p("SDS_BIT_STICKOUT_M", 0.16, "m", "derived",
                        "the modelled bit ahead of the chuck sleeve: its tip reaches stickout - 10 mm (SDS slide in the "
                        "idle-stroke window) = 150 mm, the HC2083's 152 mm working length within 2 mm")
SDS_LONG_BIT_WORKING_M = _p("SDS_LONG_BIT_WORKING_M", 10*0.0254, "m", "read",
                            f"Bosch HC2084 1/2 in: working length 10 in, total 12 in ({BOSCH_SDS}): through a 38 mm ledger "
                            "to 108 mm needs 146 mm; the 6 in bit leaves 6 mm, and in this model (tip reach 150 mm) the "
                            "full-diameter depth through the ledger would stop at 108.9 mm")
SDS_LONG_BIT_STICKOUT_M = _p("SDS_LONG_BIT_STICKOUT_M", SDS_BIT_STICKOUT_M + SDS_LONG_BIT_WORKING_M - SDS_BIT_WORKING_M,
                             "m", "derived", "HC2084 stickout: the HC2083's plus the 4 in of extra working length")
ROLLING_B_M = _p("ROLLING_B_M", 1e-4, "m", "literature",
                 "rolling-resistance lever arm (MuJoCo condim-6 rolling friction, torque = b N) of the round rims of "
                 "loose steel parts (punch barrel and cap, anchor washer, carriage-bolt dome): b = 0.1 mm, 'hardened "
                 "steel ball bearings on steel' (Crr 0.0010-0.0015) in the rolling-resistance table of "
                 "https://en.wikipedia.org/wiki/Rolling_resistance (secondary source). Without it (condim 3) round parts "
                 "rolled 0.7-1.8 m on the steel plate and were still moving after 20 s. The metal "
                 "faces (world.py, contact priority 1) and the hole module's face pads (priority 2) carry the same lever "
                 "arm, since the higher-priority geom's contact parameters rule. Machinery's Handbook 31st ed. p. 174 "
                 "gives 0.02 in (0.51 mm) for iron on iron and calls published values approximate: 0.1-0.51 mm brackets "
                 "it. What kept round parts rocking on the plate was their shank's face support switching on and off "
                 "(holes.face_near_segment), not this value")
MU = dict(default=0.5, pad=0.8, steel_steel=0.2, steel_wood=0.4, steel_concrete=0.45, wood_wood=0.35, rubber=0.9,
          bit_wall=0.1, floor=0.6, steel=0.3, aluminium=0.3, douglas_fir=0.45, plywood=0.45, concrete=0.6)             # bit margins on the hole wall (lubricated by chips; estimate)
_p("MU", MU, "-", "estimate",
   "friction coefficients (MuJoCo: one coefficient, static = sliding). Machinery's Handbook 31st ed. p. 173, Bowden and "
   "Tabor's tables: steel on steel 0.8 static and 0.4 sliding clean, 0.16 static and 0.03 sliding oiled; iron on iron "
   "1.0 clean, 0.15-0.20 lubricated; aluminium on aluminium 1.35 clean, 0.30 lubricated; wood on metal 0.2-0.6 clean; "
   "wood on wood 0.25-0.5 clean. Shop steel is lightly oiled: steel 0.3 and steel_steel 0.2 lie BETWEEN the oiled "
   "values (up to 0.16) and the clean sliding value (0.4), inside neither range, chosen, not measured (open). "
   "steel_concrete 0.45 is below a published 0.57-0.70 for steel "
   "on concrete (search summary; not re-read) and is kept as a chosen value for a dusty face "
   "(open). The Franka pad 0.8 is the repository's qualified value (barcode_scanning)")

# ------------------------------------------------------------------ hole-module constants
CONCRETE_ROT_EV_J_MM3 = _p("CONCRETE_ROT_EV_J_MM3", 5.0, "J/mm3", "estimate",
                           "rotation-only scraping of concrete: why the hammer mode must be on")
REAM_ZONE_D = _p("REAM_ZONE_D", 1.0, "x d", "estimate",
                 "axial length, in bit diameters, of the hole wall a side-loaded spinning bit wears (holes._ream): the "
                 "wall the bit's friction work at CONCRETE_ROT_EV_J_MM3 removes is spread over it as a crescent. With "
                 "it a 0.2 deg misfit growing at 0.9 mm/s of drilling is worn away at a ~40 N side load; no measured "
                 "value found (open in substance)")
KAPPA_SDS_WOOD = _p("KAPPA_SDS_WOOD", 2.0, "-", "estimate", "SDS carbide head cutting timber rotary-only (scraping)")
SDS_IN_METAL_FACTOR = _p("SDS_IN_METAL_FACTOR", 1.5, "x W", "estimate", "SDS head in steel (negative rake)")
HAMMER_INDEX_DEG = _p("HAMMER_INDEX_DEG", 30.0, "deg", "estimate",
                      "rotation between blows needed for a fresh strike (full trigger indexes ~86 deg per blow)")
HAMMER_NO_INDEX_EFF = _p("HAMMER_NO_INDEX_EFF", 0.2, "-", "estimate",
                         "removal efficiency of a blow on the spot the last blow crushed (hammer-only mode)")
DUST_CUSHION_M = _p("DUST_CUSHION_M", 0.003, "m", "estimate", "dust column under the bit that halves a blow ~ ln2 x this")
WITHDRAW_CARRIES = _p("WITHDRAW_CARRIES", 0.5, "-", "open", "share of loose dust a withdrawal of the bit carries out")
NOZZLE_BOTTOM_TOL_M = _p("NOZZLE_BOTTOM_TOL_M", 0.005, "m", "authored",
                         "a vacuum pass counts if the nozzle reaches within 5 mm of the floor; a "
                         "threshold of this environment, no published value")
BRUSH_DEPTH_FRACTION = _p("BRUSH_DEPTH_FRACTION", 0.9, "-", "authored", "a brush stroke reaches >= 90 % depth")
SPALL_DIAMETER_FACTOR_LATERAL = _p("SPALL_DIAMETER_FACTOR_LATERAL", 1.8, "x d", "estimate",
                                   "entry crater when the bit is pushed sideways during the first blows")
SPALL_LATERAL_N = _p("SPALL_LATERAL_N", 40.0, "N", "estimate", "lateral load that enlarges the entry spall")
SPALL_CRATER_SLOPE_DEG = _p("SPALL_CRATER_SLOPE_DEG", 35.0, "deg", "literature",
                            "slope of the entry spall crater's wall from the face: the concrete breakout cone of the "
                            "CCD method (projected radius 1.5 h_ef, about 35 deg; ACI 318 R17.6.2.1, EN 1992-4), used "
                            "as the crater shape. A 1.4 d crater on the 13.3 mm bit is then 1.86 mm deep; the hole's "
                            "wall proxies start below it (holes._place_hole)")
BURR_GROSS_M = _p("BURR_GROSS_M", 0.0016, "m", "read",
                  "exit burr that must be removed before bolting: RCSC Specification for Structural Joints (2004) "
                  "section 3.4, 'Burrs that extend 1/16 in. or less above the surface are permitted to remain on the "
                  "faying surfaces of snug-tightened joints ... Burrs that extend over 1/16 in. ... shall be removed' "
                  "(https://www.boltcouncil.org/files/2004RCSCSpecification.pdf). Under BURR_CAP_FRACTION a 9 mm drill in S235 leaves 0.19 mm "
                  "eased, 0.42 mm at 1.1 kN, 0.49 mm at 1.5 kN: this limit is not reached by any push the press allows")
BURR_MIN_M = _p("BURR_MIN_M", 5e-5, "m", "estimate", "uniform minimum exit burr of a backed or eased exit")
BURR_CAP_FRACTION = _p("BURR_CAP_FRACTION", 0.25, "x cap thickness", "derived",
                       "exit burr height / remaining thickness at cap collapse sqrt(2T/(pi sigma_y)): geometric mean of "
                       "nine St37 + conventional HSS twist-drill burr heights (Alrabii 2009, J. Eng. Univ. Baghdad 15(4) "
                       "4219, Figs 5-6: 0.12-1.15 mm for 5-12.5 mm drills; 9 mm at 0.03 mm/rev 0.24 mm) against the "
                       "cap thickness this law gives at their thrust; ratios 0.10-0.77 (geometric s.d. x1.8). The cap is "
                       "the drill cap pushed out; the burr is the rim left behind (Kim & Dornfeld's uniform burr with cap)")
WOOD_SCORCH_K_PER_J = _p("WOOD_SCORCH_K_PER_J", 1.5, "K/J", "estimate",
                         "lumped wall-segment temperature rise per joule of rubbing (0.5 mm segment, k 0.12 W/m K)")
WOOD_RUB_RECOVER_J_S = _p("WOOD_RUB_RECOVER_J_S", 5.0, "J/s", "estimate", "rubbing-heat loss of a wall segment")
OIL_H_FACTOR = _p("OIL_H_FACTOR", 4.0, "x", "estimate", "cutting oil raises tip heat transfer")
HSS_K = _p("HSS_K", 21.8, "W/m K", "read", "Bohler S600 conductivity")
BURN_WEAR_STEP = _p("BURN_WEAR_STEP", 0.3, "W", "estimate", "burnt corners: irreversible wear step above 550 C")
SDS_WEAR_PER_MM3 = _p("SDS_WEAR_PER_MM3", 8.9e-8, "W/mm3", "derived",
                      "KIT: ROP -15 % after 21 m of 10 mm holes, i.e. W +0.7 %/m = 0.007 per 78,540 mm3 removed")

# ------------------------------------------------------------------ stations and mechanisms
REFLEX_EFFORT_FRACTION = _p("REFLEX_EFFORT_FRACTION", 0.95, "-", "authored",
                            "drilling contact setting: a joint stops above 95 % of its effort limit")
RECOIL_DUTY = _p("RECOIL_DUTY", 0.30, "x blow period", "derived",
                 "width of the half-sine recoil on the housing per blow, as a share of the blow period: the half-sine "
                 "with the same impulse and peak as the air-spring (upper-chamber gauge) pressure pulse of an "
                 "electro-pneumatic hammer, digitised from Wang 2014 Fig. 4.9 (Concordia MASc, Bosch 11313 EVS model, "
                 "1830 bpm, 75 N push; https://spectrum.library.concordia.ca/978749/1/Wang_MSc_F2014.pdf): equivalent "
                 "half-sine 9.8-10.1 ms of the 32.8 ms period (0.30), FWHM 5.2 ms (0.16); scaled with the crank angle "
                 "(an assumption) to the hammer's blow rate")
RECOIL_PULSE_S = _p("RECOIL_PULSE_S", RECOIL_DUTY*60./HAMMER_BPM, "s", "derived",
                    "recoil half-sine width at the full blow rate (RECOIL_DUTY x 13.0 ms = 3.9 ms; at a partial trigger "
                    "the pulse widens with the blow period)")
PRESS_STAND_M = _p("PRESS_STAND_M", 0.76, "m", "authored",
                    "steel stand under the benchtop press; puts the table at 0.964 m")
PRESS_BUTTON_N = _p("PRESS_BUTTON_N", 8.0, "N", "estimate", "push-button actuation force")
PRESS_BUTTON_TRAVEL_M = _p("PRESS_BUTTON_TRAVEL_M", 0.003, "m", "estimate", "push-button travel")
PRESS_BUTTON_SWITCH_FRACTION = _p("PRESS_BUTTON_SWITCH_FRACTION", 0.5, "-", "derived",
                                  "the switch point is where the button spring (PRESS_BUTTON_N over the travel, half "
                                  "preloaded) pushes back with PRESS_BUTTON_N; at 0.7 of the travel a steady push needed "
                                  "9.6 N, not the 8 N actuation force")
RIDGID_RV2400HF = ("RIDGID RV2400HF 14 gal wet/dry vac with certified HEPA filter, owner's manual SP6775 "
                   "(https://images.thdstatic.com/catalog/pdfImages/24/24e4f496-2f24-43d8-88fd-97a2df60ab0a.pdf): p. 10 "
                   "'turn the unit on by pushing the switch from position \"O\" to position \"I\"'; p. 16 parts 830564 "
                   "'Actuator, Switch', 73180 'Switch, Toggle'")
VACUUM_TANK_D_M = _p("VACUUM_TANK_D_M", 0.46, "m", "estimate",
                     f"tank diameter of a 14 gal (53 l) drum vac; the manual gives no dimensions ({RIDGID_RV2400HF})")
VACUUM_TANK_H_M = _p("VACUUM_TANK_H_M", 0.45, "m", "estimate", "tank height on its caster base (14 gal class)")
VACUUM_HEAD_H_M = _p("VACUUM_HEAD_H_M", 0.12, "m", "estimate", "power head height above the tank (14 gal class)")
VACUUM_ROCKER_SIZE_M = _p("VACUUM_ROCKER_SIZE_M", (0.032, 0.025, 0.008), "m", "read",
                          "on/off rocker 1.25 x 1 in (RIDGID 83797 rocker switch for the WD1660 vac, retailer listing "
                          "https://empirerigging.com/ridgid-83797, search-level); its 8 mm actuator thickness estimated. "
                          "That is another RIDGID model's switch: the modelled RV2400HF's own parts list names a toggle "
                          "switch with an actuator (vacuum.py), whose size was not found, so the modelled switch is a "
                          "rocker of the WD1660's size (cosmetic)")
VACUUM_ROCKER_THROW_RAD = _p("VACUUM_ROCKER_THROW_RAD", math.radians(10.), "rad", "estimate",
                             "rocker tilt either side of its middle (typical rocker switch)")
VACUUM_ROCKER_PRESS_N = _p("VACUUM_ROCKER_PRESS_N", 8.0, "N", "estimate",
                           "push on the middle of a rocker half that turns it over; the push-button value "
                           "(PRESS_BUTTON_N), no published figure for this switch. Measured on the built switch: "
                           "a vertical push on the raised half's top face (6.5 mm from the "
                           "pivot) turns it over between 9.0 and 10.5 N; 4 N does not")
VACUUM_ROCKER_DETENT_NM = _p("VACUUM_ROCKER_DETENT_NM", VACUUM_ROCKER_PRESS_N*0.25*VACUUM_ROCKER_SIZE_M[0], "N m",
                             "derived", "detent torque at the rocker's middle: VACUUM_ROCKER_PRESS_N at a quarter of its "
                             "length from the pivot (the middle of a half)")
VACUUM_ROCKER_ARMATURE = _p("VACUUM_ROCKER_ARMATURE", 1e-4, "kg m2", "estimate",
                            "reflected inertia of the toggle mechanism under the rocker (keeps omega dt < 1 at 1 ms)")
VACUUM_ROCKER_DAMPING = _p("VACUUM_ROCKER_DAMPING", 0.002, "N m s/rad", "estimate", "rocker joint damping")
PRESS_ROTOR_INERTIA = _p("PRESS_ROTOR_INERTIA", 0.005, "kg m2", "estimate",
                         "3/4 hp rotor (~0.002 kg m2) reflected through a 1.7:1 belt")
PRESS_STOP_PITCH_M = _p("PRESS_STOP_PITCH_M", 0.0015, "m", "estimate", "depth-stop rod M12 x 1.5")
PRESS_COAST_S = _p("PRESS_COAST_S", 1.0, "s", "estimate", "spindle coast-down time constant after switching off")
VISE_JAW_MU = _p("VISE_JAW_MU", 0.25, "-", "estimate", "ground steel jaw plates on steel")
VISE_CRANK_R = _p("VISE_CRANK_R", 0.08, "m", "estimate", "crank knob radius")
VISE_DRY_FRICTION_NM = _p("VISE_DRY_FRICTION_NM", 0.05, "N m", "estimate", "unloaded screw drag")
VISE_SCREW_RM = _p("VISE_SCREW_RM", 0.0065, "m", "estimate", "mean thread radius of the vise screw")
PUNCH_RELEASE_TRAVEL_M = _p("PUNCH_RELEASE_TRAVEL_M", 0.003, "m", "estimate", "point travel to the release force")
PUNCH_POINT_ARMATURE = _p("PUNCH_POINT_ARMATURE", 0.03, "kg", "estimate",
                          "internal hammer and spring mass reflected to the point (keeps omega dt < 1)")
PUNCH_RETURN_N = _p("PUNCH_RETURN_N", 5.0, "N", "estimate",
                    "light spring that pushes the point back out after the release, at full point travel (half of it "
                    "preloaded); once the internal hammer has struck, the main spring's stored energy (1/2 k x^2 = "
                    "PUNCH_ENERGY_J at the release) has gone into the strike and no longer pushes the point")
PUNCH_RECOCK_M = _p("PUNCH_RECOCK_M", 5e-5, "m", "authored",
                    "the trip pin re-seats (cocked again) once the point is back out within 0.05 mm, where the main "
                    "spring's force (1.7 N) is below the return spring's (2.5 N): re-cocking adds no energy")
HARDNESS_PA = dict(steel=1.3e9, aluminium=0.95e9, douglas_fir=3.0e7, plywood=3.0e7, concrete=0.)
PHANTOM_LATERAL_K = _p("PHANTOM_LATERAL_K", 24000.0, "N/m", "read",
                       "KIT hand-arm model lateral stiffness 24 N/mm (compliant holder of the walking bench E5)")
VISE_PRECLAMP_NM = _p("VISE_PRECLAMP_NM", 5.0, "N m", "authored",
                      "crank torque of the start-state clamp (a firm hand on an 80 mm crank: 62 N)")
WRENCH_MAX_N = _p("WRENCH_MAX_N", 150.0, "N", "authored",
                  "largest cartesian_wrench force accepted (the drilling contact setting's hand-force range)")
WRENCH_MAX_NM = _p("WRENCH_MAX_NM", 25.0, "N m", "authored", "largest cartesian_wrench moment accepted")
HAND_TCP_OFFSET = 0.1034                       # m, Franka Hand flange to TCP (catalog; barcode/wine kinematics)
DENSITY = dict(steel=7850., aluminium=2700., douglas_fir=WOOD_DENSITY, plywood=600., concrete=CONCRETE_DENSITY)

# ------------------------------------------------------------------ task constants
SDS_BIT_D_MM = _p("SDS_BIT_D_MM", 13.3, "mm", "read",
                  "1/2 in SDS-plus carbide tip: ANSI B212.15 allows 0.520 to 0.530 in, the middle is 13.3 mm")
PRESS_BITS = _p("PRESS_BITS", {"press-drill": (9.0, 0.081, 1000.), "press-blind": (6.8, 0.069, 2400.),
                               "match-bolt": (9.0, 0.081, 1000.)}, "mm, m, rev/min", "authored",
                "the bit in the press's chuck, its stick-out and the spindle speed the belt is set to, per press task: a "
                "9 mm clearance drill for an M8 bolt (ISO 273, medium) and a 6.8 mm tapping drill for M8 (ISO 2306). "
                "The speeds are round numbers chosen here (28 m/min at the 9 mm drill's rim, 51 m/min at the 6.8 mm "
                "drill's), not steps read from the G7943's belt chart")
WALL_MARK_TOL_M = _p("WALL_MARK_TOL_M", 0.003, "m", "authored",
                     "the evaluator passes a wall hole whose entry is within 3 mm of its mark (a 1/2 in anchor in a "
                     "9/16 in fixture hole has 1.6 mm of clearance all round)")
WALL_AXIS_TOL_DEG = _p("WALL_AXIS_TOL_DEG", 3.0, "deg", "authored",
                       "the evaluator passes a wall hole whose axis is within 3 degrees of the wall's normal")
MAX_LAYOUT_SHIFT_M = _p("MAX_LAYOUT_SHIFT_M", 0.15, "m", "authored",
                        "rebar-relocate: the largest shift of a mark the task allows, half the 300 mm bar spacing")


if __name__ == "__main__":
    for e in LEDGER:
        print(f"| {e['name']} | {e['value']} | {e['unit']} | {e['label']} | {e['basis']} |")
