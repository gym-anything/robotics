"""Parameter ledger: every physical constant the simulation reads, with its value, unit, label and basis.

Labels: datasheet (manufacturer document), retailer (published product listing), published (paper/standard),
asset (official robot description), derived (computed here from other entries), estimate (judgement, with the
range E24 varies it over), authored (a design choice of this environment), repo (a value qualified
elsewhere in this repository).
`python -m classroom_arrangement.params` prints the table.
"""

import math

LEDGER = []


def _p(name, value, unit, label, basis):
    LEDGER.append(dict(name=name, value=value, unit=unit, label=label, basis=basis))
    return value


# ------------------------------------------------------------------ numerics
GRAVITY = _p("GRAVITY", 9.81, "m/s2", "authored", "standard gravity rounded as in the other environments")
DT = _p("DT", 0.0005, "s", "derived", "sliding-glide chatter halves from 1 ms to 0.5 ms; E3 requalifies")
SOLVER_ITERATIONS = _p("SOLVER_ITERATIONS", 50, "-", "repo", "barcode_scanning Config")
SOLVER_IMPRATIO = _p("SOLVER_IMPRATIO", 3.0, "-", "repo", "barcode: impacts converged at 3")
NOSLIP_ITERATIONS = _p("NOSLIP_ITERATIONS", 100, "-", "repo", "barcode: converged no-slip pass")
NOSLIP_TOLERANCE = _p("NOSLIP_TOLERANCE", 1e-10, "-", "repo", "barcode")
CONTACT_TIMECONST = _p("CONTACT_TIMECONST", 0.004, "s", "repo", "rigid contact (whiteboard/rigid_contact.py, barcode)")
CONTACT_SOLIMP = _p("CONTACT_SOLIMP", (0.95, 0.99, 0.001), "-, -, m", "repo", "rigid contact impedance (barcode)")
PAD_TIMECONST = _p("PAD_TIMECONST", 0.005, "s", "repo", "Franka fingertip rubber pad (label_application, anchored)")
STOP_SOLIMP = (0.99, 0.999, 0.5, 2.0)           # hard joint stops (barcode params; width set per joint)
CONTACT_GAP_M = _p("CONTACT_GAP_M", 0.0, "m", "repo", "boxes/planes: no early narrowphase (barcode)")
MESH_CONTACT_GAP_M = _p("MESH_CONTACT_GAP_M", 0.002, "m", "repo", "hulls/meshes: 2 mm envelope (barcode)")
CONTACT_CAPACITY = 12288
MAX_OBJECT_SPEED_M_S = _p("MAX_OBJECT_SPEED", 20.0, "m/s", "authored", "nothing in a classroom moves faster: a fault")
MAX_OBJECT_SPIN_RAD_S = _p("MAX_OBJECT_SPIN", 1000.0, "rad/s", "authored", "fault guard")
SETTLE = dict(min_s=1.0, max_s=5.0, speed_m_s=0.002, window_s=0.1)   # reset settle (as in barcode)

# ------------------------------------------------------------------ room 114
ROOM_X = _p("ROOM_X", 7.40, "m", "authored", "whiteboard classroom width; 71 m2 interior (Texas 700 ft2 minimum, 19 TAC 61.1040)")
ROOM_Y = _p("ROOM_Y", 9.60, "m", "authored", "whiteboard classroom with the back wall moved 8.2 -> 9.6 m")
ROOM_Z = _p("ROOM_Z", 3.25, "m", "authored", "whiteboard classroom ceiling")
DOOR_X = _p("DOOR_X", 2.65, "m", "authored", "door centre on the front wall (whiteboard)")
DOOR_LEAF = _p("DOOR_LEAF", 0.93, "m", "authored", "36 in leaf (whiteboard); swing zone kept clear")
AISLE_MIN = _p("AISLE_MIN", 0.915, "m", "published", "36 in clear width of an accessible route, 2010 ADA Standards 403.5.1 "
              "(literature; text not fetched)")

# ------------------------------------------------------------------ student desk, Virco 785
DESK_TOP = _p("DESK_TOP", (0.6096, 0.4572, 0.025), "m", "datasheet", "24 x 18 in hard plastic top [Virco 785 product page]")
DESK_HEIGHT = _p("DESK_HEIGHT", 0.7366, "m", "authored", "29 in setting within the 785's 23-31 in range [Virco 785 product page]")
# Mass basis (one rule for every item): the listed weight on the Ezell Company product page where it
# lists the item (desk, chair, table), else SchoolOutlet's spec table (teacher's desk). Listed weights may include
# packaging; E24's -15 % case covers that.
DESK_MASS = _p("DESK_MASS", 10.9, "kg", "retailer", "24 lb, Ezell listing (other resellers 28.5-30.2 lb: Schools In, Furniture Tycoon); "
               "E24 varies x0.85 / x1.15")
DESK_PARTS = _p("DESK_PARTS", dict(top=3.3, book_box=2.8, rails=1.6, legs=3.0, glides=0.2), "kg", "estimate",
                "material x section, scaled to DESK_MASS")
DESK_BOX = _p("DESK_BOX", (0.559, 0.381, 0.121), "m", "datasheet", "book box 22 x 15 x 4.75 in [Virco 785 product page]")
DESK_LEGS_FLOOR = _p("DESK_LEGS_FLOOR", (0.275, 0.19), "m", "authored", "glide half-spacing (whiteboard desk; no dimensioned "
                     "source, Virco's Revit family was not converted): E24 runs +-0.02 m")
DESK_LEGS_TOP = _p("DESK_LEGS_TOP", (0.249, 0.17), "m", "authored", "slight splay of the frame (whiteboard)")
DESK_LEG_R = _p("DESK_LEG_R", 0.0127, "m", "datasheet", "1 in OD steel tube legs [Virco 785 product page]")
DESK_GLIDE = _p("DESK_GLIDE", (0.0143, 0.012), "m", "retailer", "1-1/8 in glide diameter [Schools In listing, search summary]; 12 mm tall "
                "(estimate)")

# ------------------------------------------------------------------ student chair, Virco 9018
CHAIR_MASS = _p("CHAIR_MASS", 5.26, "kg", "retailer", "11.6 lb, Ezell listing (COE Distributing 11.6 lb; "
                "SchoolOutlet 10.417 lb, VastMarket 11 lb item); E24 varies x0.85 / x1.15")
CHAIR_PARTS = _p("CHAIR_PARTS", dict(seat=0.9, back=0.7, legs=1.5, rails=0.6, back_supports=0.8, glides=0.2), "kg",
                 "estimate", "injection-moulded polyethylene shell 1.6 kg [Virco 9000 product page], chrome frame, scaled to CHAIR_MASS")
CHAIR_SEAT_H = _p("CHAIR_SEAT_H", 0.457, "m", "datasheet", "18 in seat height [Virco 9000 product page]")
CHAIR_SEAT_HALF_W = _p("CHAIR_SEAT_HALF_W", 0.1905, "m", "retailer", "seat 15 in wide [VastMarket listing]")
CHAIR_OVERALL = _p("CHAIR_OVERALL", (0.476, 0.533, 0.775), "m", "datasheet", "W x D x H [Virco 9000 product page]")
CHAIR_GLIDES = _p("CHAIR_GLIDES", (0.218, 0.217), "m", "authored", "glide half-spacing (whiteboard chair, inside the V5 "
                  "footprint; no dimensioned source): E24 runs +-0.02 m")
CHAIR_GLIDE = _p("CHAIR_GLIDE", (0.016, 0.012), "m", "estimate", "1-1/4 in nylon swivel glide")
CHAIR_LEG_R = _p("CHAIR_LEG_R", 0.0111, "m", "datasheet", "7/8 in chrome tube frame [Virco 9000 product page]")

# ------------------------------------------------------------------ other furniture
TABLE_TOP = _p("TABLE_TOP", (1.524, 0.762, 0.0286), "m", "datasheet", "Virco 4000 Series 30 x 60 in, 1-1/8 in top [Virco product page]")
TABLE_MASS = _p("TABLE_MASS", 32.3, "kg", "retailer", "71.3 lb, Ezell listing; parts: 720 kg/m3 particleboard "
                "top 23.9 kg, telescoping legs, aprons, anti-warp bar")
TEACHER_TOP = _p("TEACHER_TOP", (1.219, 0.762, 0.0286), "m", "datasheet", "Virco 543 48 x 30 in, 1-1/8 in top [Virco product page, SchoolOutlet listing]")
TEACHER_HEIGHT = _p("TEACHER_HEIGHT", 0.752, "m", "retailer", "29-5/8 in, SchoolOutlet Virco 543 spec table")
TEACHER_MASS = _p("TEACHER_MASS", 62.6, "kg", "retailer", "138 lb, SchoolOutlet Virco 543 spec table (1 carton); "
                  "parts: 720 kg/m3 top 19.1 kg, steel pedestal with three drawers 35.7 kg, modesty panel, legs")
TOTE = _p("TOTE", (0.52, 0.37, 0.20, 7.6), "m, kg", "estimate", "plastic tote with four textbooks (1.6 kg each)")
BASKET = _p("BASKET", (0.15, 0.38, 1.0), "m, kg", "estimate", "plastic wastebasket, 0.30 m dia, 0.38 m tall")
BOOK = _p("BOOK", (0.26, 0.205, 0.040, 1.6), "m, kg", "estimate", "hardcover textbook (desk contents variation)")

# ------------------------------------------------------------------ friction
GLIDE_MU = _p("GLIDE_MU", dict(nylon=(0.235, 0.195), felt=(0.35, 0.29), steel=(0.12, 0.115)), "static, kinetic",
              "published", "Substratum GS-1 sled on vinyl sheet [Substratum Group 2018] (bar chart read +-0.01); a furniture-scale pull "
              "measured mu_s 0.45 for a metal-base desk on vinyl [Yeow et al. 2018]: E24 runs nylon/steel up to 0.45")
STRIBECK_V = _p("STRIBECK_V", 0.002, "m/s", "estimate", "static -> kinetic transition speed; E24 runs 1 and 5 mm/s")
MU_FLOOR = _p("MU_FLOOR", 0.30, "-", "estimate", "VCT against steel tube / plastic (tipped furniture, tote, basket)")
MU_PLASTIC = _p("MU_PLASTIC", 0.30, "-", "estimate", "hard plastic on hard plastic (desk top-top, shell-top); x0.7-1.3")
MU_STEEL = _p("MU_STEEL", 0.35, "-", "estimate", "steel leg on steel leg; x0.7-1.3")
MU_TOTE = _p("MU_TOTE", 0.35, "-", "estimate", "polypropylene tote / basket on VCT")
MU_BOOK = _p("MU_BOOK", 0.30, "-", "estimate", "book cover on painted steel book box; spill at atan(0.30) = 16.7 deg")
MU_ROBOT = _p("MU_ROBOT", 0.40, "-", "estimate", "robot covers (painted metal / plastic) on furniture")
MU_WHEEL = _p("MU_WHEEL", 0.60, "-", "estimate", "Vulkollan 95A tread on VCT, range 0.4-0.8")
MU_PAD = _p("MU_PAD", 0.80, "-", "repo", "Franka fingertip pad (barcode MU pad_housing); applied to every pad contact")

# ------------------------------------------------------------------ robot
WHEEL_R = _p("WHEEL_R", 0.05, "m", "asset", "ArgoDrive wheel radius (franka_description tmrv0_2, AD-100.12 datasheet)")
DRIVES = _p("DRIVES", dict(front=(0.30, -0.20), rear=(-0.30, 0.20)), "m", "asset", "ArgoDrive positions in base_link (URDF)")
WHEEL_TORQUE = _p("WHEEL_TORQUE", (16.0, 28.4), "N m", "datasheet", "AD-100.12 max acceleration / motor braking torque "
                  "[ArgoDrive data sheet]; a stopped wheel is held by its servo up to the motor braking torque. The drive's spring-applied "
                  "holding brake (slip torque 35.5 N m +-10 %, same datasheet) is not modelled: a parked base resists a "
                  "shove ~25 % less than the real one")
REFLEX_STOPS_BASE = _p("REFLEX_STOPS_BASE", True, "-", "authored", "an arm protective stop also stops the base and refuses base "
                       "commands for its 1 s hold: an integration rule (an operator program that does not stop the drive "
                       "lets the base push through reflex-frozen arms); no Franka source; launch option reflex_stops_base")
STEER_RATE = _p("STEER_RATE", math.pi, "rad/s", "datasheet", "AD-100.12 steering 180 deg/s [ArgoDrive data sheet]")
BASE_LIMITS = _p("BASE_LIMITS", dict(v=1.0, w=1.0, a=0.8, alpha=1.5), "m/s, rad/s, m/s2, rad/s2", "authored",
                 "software limits within the Mobile FR3 Duo manual's 1.5 m/s and the ArgoDrive's 0.8 m/s2")
SPINE = _p("SPINE", dict(stroke=0.827, speed=0.3, force=670.2, accel=1.0), "m, m/s, N, m/s2", "datasheet + derived",
           "stroke 827 mm [Mobile FR3 Duo manual] (franka_description URDF range 0-0.85 m cut to it); 0.3 m/s [manual]; 2 pi 2.37 N m / 0.020 m x 0.9; "
           "accel 1.0 m/s2 authored (motion profile)")
ARM_ACCEL = _p("ARM_ACCEL", 10.0, "rad/s2", "datasheet", "FR3 joint acceleration limit, all joints (Franka FCI robot specifications)")
ARM_SPEED_FRACTION = _p("ARM_SPEED_FRACTION", 0.90, "-", "authored", "joint reference cruise speed / FR3 speed limit "
                        "(libfranka-style margin; with acceleration feed-forward the servo tracks within a few %; E25)")
ARM_ACCEL_FEEDFORWARD = _p("ARM_ACCEL_FEEDFORWARD", True, "-", "authored", "FR3 joint servo adds M(q) a_ref (the real "
                           "controller uses its dynamic model); without it a stowed -> ready target overshoots to 1.28 x "
                           "the joint speed limit and trips the velocity reflex (E25)")
ROBOT_MASS = _p("ROBOT_MASS", 216.7, "kg", "derived", "URDF + reconciled TMR 130 kg / Spine 45 kg (datasheet total 225 kg, Mobile FR3 Duo manual)")
_FR_J = (20., 20., 18., 18., 16., 14., 12.)
_FR_F = (20., 20., 20., 25., 25., 25.)
REFLEX_PROFILES = _p("REFLEX_PROFILES", dict(
    # stock franka_ros franka_control_node.yaml collision_config: upper torque thresholds (both phases) and upper
    # Cartesian force/torque thresholds [N, N, N, N m, N m, N m] (both phases)
    franka_ros=dict(nominal=_FR_J, acceleration=_FR_J, force_nominal=_FR_F, force_acceleration=_FR_F),
    # libfranka examples_common.cpp setDefaultBehavior: torques 10 (constant velocity) / 20 (acceleration) N m,
    # forces 10 / 20 N
    libfranka_default=dict(nominal=(10.,)*7, acceleration=(20.,)*7, force_nominal=(10.,)*6, force_acceleration=(20.,)*6),
    # franka_ros scaled x5: the practice of boschresearch/franka_utils `set_collision_behaviour` (scales the default
    # thresholds by a ratio; its README example is 5x "when interacting with stiff objects and expecting larger than
    # default contact forces"). The environment's default: moving furniture needs hand forces of 21-100 N, which the
    # stock 20 N Cartesian threshold stops (E9 stock-threshold control). Joint thresholds above the FR3 torque limits
    # (87 / 12 N m) no longer bind; the Cartesian 100 N / 125 N m thresholds do.
    contact_x5=dict(nominal=tuple(5*v for v in _FR_J), acceleration=tuple(5*v for v in _FR_J),
                    force_nominal=tuple(5*v for v in _FR_F), force_acceleration=tuple(5*v for v in _FR_F)),
    ), "N m per joint; N, N m at the hand", "datasheet + practice",
    "franka_ros franka_control_node.yaml collision_config; libfranka setDefaultBehavior; boschresearch/franka_utils x5")
REFLEX_PROFILE = "contact_x5"
REFLEX_FILTER_S = _p("REFLEX_FILTER_S", 0.010, "s", "estimate",
                     "first-order filter on the contact-derived external torque and hand wrench, standing in for the "
                     "FR3's own filtered estimates (tau_ext_hat_filtered, K_F_ext_hat_K); single-tick contact impulses "
                     "are not what the robot's estimator reports")
REFLEX_RECOVERY_S = _p("REFLEX_RECOVERY_S", 1.0, "s", "repo", "post-reflex hold before commands are accepted (barcode)")
HAND = _p("HAND", dict(speed=0.05, cont=70.0, max_total=140.0, open=0.04), "m/s, N, N, m", "datasheet",
          "Franka Hand: 50 mm/s per finger, 70 N continuous grasp, 140 N max, 80 mm stroke [Franka Hand manual]")
ARM_GRAVCOMP = _p("ARM_GRAVCOMP", True, "-", "authored", "the FR3 controller compensates its own links' gravity and "
                  "Coriolis terms: MuJoCo's bias force on the FR3 joints added to the servo, inside the joint torque "
                  "limits; the base still carries the arms (MuJoCo body gravcomp would lift the arms off the base)")

# ------------------------------------------------------------------ cameras
POLICY_RES = _p("POLICY_RES", (640, 360), "px", "authored", "per policy camera")
HEAD_HFOV = _p("HEAD_HFOV", 90.0, "deg", "datasheet", "ZED Mini class head camera, 90 x 60 deg (D1 kit)")
WRIST_HFOV = _p("WRIST_HFOV", 87.0, "deg", "datasheet", "RealSense D405 class wrist RGB (D1 kit), as barcode")
BASE_CAMERA_HFOV = _p("BASE_CAMERA_HFOV", 90.0, "deg", "datasheet", "four RealSense D455 on the TMR (front, rear, left, "
                      "right; manual R04245 5.6): RGB 90 x 65 deg, at the URDF's *_mounting_point frames, optical axis "
                      "level and outward")
LIDAR = _p("LIDAR", dict(fov_deg=275.0, step_deg=0.5, range_m=(0.05, 10.0), mounts=((0.3275, 0.2175, 0.19065, 45.0),
                                                                                    (-0.3275, -0.2175, 0.19065, 225.0))),
           "deg, m", "datasheet / authored", "two Sick nanoScan3 (manual R04245 5.6) at the URDF's lidar_*_mounting_point "
           "frames: 275 deg scanning angle (datasheet); 0.5 deg steps (authored, the scanner resolves 0.17 deg); ranges "
           "capped at 10 m (the room; the scanner measures to 40 m); scan centre pointing diagonally outward from each "
           "corner (authored from the frames' placement); ray-cast against the collision geometry, the robot's chassis "
           "excluded; the safety fields are not configured on delivery (manual), so the scans are observations only")
IMU_FILTER_S = _p("IMU_FILTER_S", 0.010, "s", "authored", "first-order low-pass of the base IMU's specific force "
                  "(Olive Robotics IMU on the TMR, manual R04245 5.6; at imu_mounting_point)")

# ------------------------------------------------------------------ evaluation (all authored)
EVAL = _p("EVAL", dict(window_s=1.0, glide_tol=0.030, yaw_tol_deg=3.0, pod_gap=0.010, pod_lateral=0.030,
                       tuck_depth=0.30, tuck_lateral=0.08, tuck_yaw_deg=10.0, use_range=(0.35, 0.55), ready_range=(0.20, 0.35),
                       facing_deg=15.0, exam_pitch=1.25, upright_deg=3.0, rest_motion=0.002, inverted_deg=175.0,
                       inside_edge=0.05, glide_load_frac=0.05, up_side_deg=30.0), "m, deg", "authored",
           "floor dots hidden under a glide up to 24 mm; JCQ 11.10 exam spacing 1.25 m (a UK rule; no U.S. "
           "seat pitch found); tuck_depth 0.30 m: the seat front at least 0.30 m under the top, i.e. within ~0.12 m of the "
           "fully tucked end stop (0.418 m, backrest 20 mm from the edge) so the walkway stays clear (0.10 m "
           "would pass a chair 0.30 m out); up_side_deg: an inverted chair's backrest within 30 deg of "
           "the student side")
EVAL_GROUP_WORK = _p("EVAL_GROUP_WORK", dict(
    seat_front=(-0.15, 0.35), as_found=(0.10, 10.0), as_found_furniture=(0.06, 5.0), as_found_full=(0.30, 45.0),
    loose_as_found=(0.20, 10.0), loose_full=0.50,
    pod_lateral=round(2*EVAL["glide_tol"], 4),
    pod_widest=round(EVAL["pod_gap"] + DESK_TOP[0]*math.sin(math.radians(EVAL["yaw_tol_deg"])), 4),
    toppled_deg=30.0, turn_share=0.3, turn_zero_deg=90.0, approach_share=0.3, approach_radius=2.5,
    chair_reach=round(CHAIR_OVERALL[1]/2, 4), route_floor=0.815,
    weights=dict(desks=0.60, abutment=0.15, chairs=0.25), fail_scale=0.9, place_charge=0.05,
    penalty=dict(route=0.5, safety=0.5)),
                     "m, deg, -", "authored + derived",
                     "group-work: seat_front: a chair at a pod desk is pulled out with its seat front "
                     "from 0.15 m under the top edge (half of tuck_depth, so not tucked; make-pod's start chairs, 'pulled "
                     "out by the students (in use)', are 0.12 m under) to 0.35 m beyond it (seat-tutoring-group's "
                     "ready-to-sit limit); as_found: each chair place the pods do not use still holds a chair within "
                     "0.10 m and 10 deg of where one stood, upright (inside the catalogue's own 'tucked'); "
                     "as_found_furniture: each desk, teacher's desk and table place within 0.06 m and 5 deg (push-in-"
                     "chairs' 'do not move the desks' is 0.05 m / 5 deg, plus 1 cm; at 0.10 m / 10 deg 36.7 % of "
                     "straighten-row's 'drifted off their places' desks counted as 'as it is', at 0.06 m / 5 deg 3.9 %; "
                     "the route check only catches a route closed everywhere, not a narrowed "
                     "aisle); the tote and wastebasket within 0.20 m (clear-floor's 'in its place') and tilted <= 10 deg; "
                     "as_found_full / loose_full (authored): the displacement at which a place's charge is complete; "
                     "pod_lateral (derived, 2 x glide_tol): facing desks may stand as far apart sideways as their two dot "
                     "tolerances allow; pod_widest (derived, pod_gap + desk width x sin yaw_tol): the closest-gap limit "
                     "plus the opening of one desk at the slot's full 3 deg against a square partner (a pod touching at "
                     "one corner with that twist opens 31.9 mm; at the 10 mm closest gap the same twist opens 41.9 mm); "
                     "reward: progress = weighted mean credit of the 12 desk places "
                     "(board-side places: turn_share for facing their way, from 0 at turn_zero_deg off to 1 within "
                     "yaw_tol, wherever the desk stands; of the rest, approach_share for a desk so turned coming within "
                     "approach_radius of the place, centre distance, which covers the front band where desks are turned"
                     "), 6 front-edge meetings and 12 chairs (fall-off over chair_reach, half the "
                     "chair's depth); score 1 on success, else fail_scale x max(0, progress - place_charge x the charges "
                     "of the rest of the room) x penalties: place_charge is one desk place's share of progress (0.60/12), "
                     "so putting a displaced item back is worth what placing a desk is, at any stage; "
                     "route x1 at 0.915 m falling to x0.5 at route_floor (ADA 2010 403.5.1 exception: 32 in / 815 mm, the "
                     "narrowest any accessible route may be), x0.5 for anything toppled or in the door swing")


def table():
    rows = ["| Parameter | Value | Unit | Label | Basis |", "|---|---|---|---|---|"]
    for e in LEDGER:
        v = e["value"]
        if isinstance(v, float):
            v = f"{v:g}"
        rows.append(f"| `{e['name']}` | {v} | {e['unit']} | {e['label']} | {e['basis']} |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(table())
