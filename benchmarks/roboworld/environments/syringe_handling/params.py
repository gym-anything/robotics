"""Parameter ledger: every constant the syringe-handling simulation reads, with value, unit, label and basis.

Labels: standard (read in the text of a standard or pharmacopoeia chapter), datasheet (read in a maker's or
regulator's document), read (a paper or textbook whose text was opened), stream (found by a research stream, mostly
search summaries; not opened by the author), derived (computed here from labelled values), estimate (engineering
estimate with its reasoning), open (no digital basis found; chosen and disclosed), authored (a design choice of this
environment), repo (qualified elsewhere in this repository), literature (classical result), measured (measured on this
machine), engine (a numerical setting of this simulator, not a property of a real thing). `python -m
syringe_handling.params` prints the table.

The numerics and robot entries down to IK_SEED_WEIGHT are the ingredient_mixing ledger's (same arms, hands, cart and
solver settings).
"""

import math

LEDGER = []


def _p(name, value, unit, label, basis):
    LEDGER.append(dict(name=name, value=value, unit=unit, label=label, basis=basis))
    return value


# ------------------------------------------------------------------ numerics
GRAVITY = _p("GRAVITY", 9.81, "m/s2", "literature", "gravity")
DT = _p("DT", 0.0005, "s", "numerical", "0.5 ms coupled rigid/fluid integration; the elastic multi-body needle at 540 rad/s is unstable at 1 ms; spin/timestep checks bound this setting, not an FR3 hardware rate")
CONTACT_TIMECONST = _p("CONTACT_TIMECONST", 0.004, "s", "repo", "rigid contact of every non-pad geom (barcode_scanning E17)")
CONTACT_SOLIMP = _p("CONTACT_SOLIMP", (0.95, 0.99, 0.001), "-", "repo", "MuJoCo contact impedance, the repository's setting")
SLIDE_REFERENCE = _p("SLIDE_REFERENCE", 1.0, "-", "derived",
                     "a friction row of a contact without a finger pad asks for no more deceleration than this times "
                     "what Coulomb friction can give the contact, mu (g + the normal row's reference) "
                     "(world._contact_rule). MuJoCo's own rule asks for b v_t whatever the friction, and a sliding body "
                     "is thrown off the surface on most ticks. 0: MuJoCo's own rule")
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
                          "under 0.001 with 200")
SOLVER_LS_TOLERANCE = _p("SOLVER_LS_TOLERANCE", 1e-6, "-", "repo", "as above (drilling)")
SOLVER_TOLERANCE = _p("SOLVER_TOLERANCE", 1e-12, "-", "engine",
                      "early-exit tolerance of MuJoCo's main solve. At the runtime's 1e-8 the solve stopped after 5 to 11 "
                      "passes on some ticks with the rows of the lightest bodies (a 0.7 g tip cap on its Luer joint, "
                      "preloaded with 27 N) not yet settled, the no-slip pass that follows amplified what was left, and "
                      "a capped syringe lying untouched on the tray had its cap kicked at 0.15 m/s, again at 0.17 and "
                      "then thrown at 16 m/s (fast-bolus walkthrough, 28.8 s; replayed from a checkpoint: the same tick "
                      "every time). Replays of that stretch, 1,500 ticks each: 1e-8 faults; 1e-10, 1e-12 and 1e-14 give "
                      "0.0000 m/s at the cap, with fewer passes a tick (mean 4.1 against 8.3, the warm start being a "
                      "settled solution) and no more time (1.20 against 1.29 ms a tick)")
PAD_TIMECONST = _p("PAD_TIMECONST", 0.005, "s", "repo", "Franka fingertip rubber pad (label_application, anchored)")
PAD_TIMECONST_REF_MASS = 0.0291
PAD_STIFFNESS = _p("PAD_STIFFNESS", 2.3e4, "N/m", "derived",
                   "normal stiffness of one fingertip pad on whatever it holds, however many points the collision "
                   "detector reports. It is the stiffness the repository's pad setting (PAD_TIMECONST on the finger's "
                   "0.0291 kg, impedance 0.95) gives one contact point against a heavy body: (1/0.005^2 /0.95^2) x 0.95 "
                   "x 0.0291 x 0.95/0.05 = 2.3e4 N/m. MuJoCo makes a contact row's force stiffness inversely "
                   "proportional to the two bodies' inverse weights and gives every reported point its own spring: on "
                   "the 4 g syringe one point had 3.5 N/mm, so the 15 N grip sank each pad 1.4 mm into the 10.2 mm "
                   "barrel on a three-point tick and 4.1-4.9 mm on a one-point tick, where the pads stood 1 mm from "
                   "the axis and pinched the plunger rod through the wall (set-dose rollout: plunger drawn 1.2 mm, "
                   "0.075 mL of air; world._pad_normal). Cross-check: a rigid 5 mm cylinder pressed 17.6 mm long into "
                   "rubber of E about 2.5 MPa (Hertz line contact) indents 0.3-0.6 mm at 15 N, 25-50 N/mm (estimate)")
PAD_DAMPING_RATIO = _p("PAD_DAMPING_RATIO", 1.0, "-", "engine",
                       "critical normal contact damping; spring and damper integrated implicitly in the constraint "
                       "row. PAD_STIFFNESS is no longer capped as a function of the part's inverse mass")
PAD_SLIP_RATE_DT = _p("PAD_SLIP_RATE_DT", 1.0, "1/tick", "engine",
                      "rate at which a pad's friction rows remove a slip, as a fraction per tick (drilling's setting). "
                      "MuJoCo gives a contact's "
                      "friction rows the damping rate of its soft normal contact, 2/(dmax x time constant) = 421 /s "
                      "for the pad's 5 ms, so a held tool slides in the hand at (hand acceleration)/421 whatever its "
                      "mass and the grip force: 2.0-2.5 mm along the pads in 5 s of 60 mm circles at 0.8 s a turn, "
                      "25 mm at the whisk's tip in a minute's whisking with or without the mixture. The pads "
                      "carry MuJoCo's direct form: the normal stiffness of PAD_TIMECONST unchanged, the rate 1/dt")
FINGER_GEAR_SOLREF = _p("FINGER_GEAR_SOLREF", (2*DT, 1.), "s, -", "engine",
                        "the tie between a Franka Hand's two fingers, with FINGER_GEAR_SOLIMP (drilling's setting). The "
                        "hand is one mechanism: Franka's description declares the second finger joint a mimic of the "
                        "first and the hand's interface has one width and one force. As a joint equality at MuJoCo's "
                        "default softness (20 ms, impedance 0.9-0.95) the mimic lets a held tool and both fingers shift "
                        "0.9 mm along the closing axis when the hand is waved (one finger opening as the other closes)")
FINGER_GEAR_SOLIMP = _p("FINGER_GEAR_SOLIMP", (.9999, .9999, .001, .5, 2.), "-", "engine", "see FINGER_GEAR_SOLREF")
PAD_STICK = _p("PAD_STICK", dict(shear_ratio=1.0/3.0, reach_m=0.0002, lever_m=0.0042), "-, m, m", "estimate",
               "a rubber pad that grips holds a place, not only a speed. MuJoCo's friction rows remove relative "
               "velocity and carry no memory of where the surfaces stuck, so what the time stepping leaves over "
               "each tick (a free body moves along a tangent while a joint moves along its arc: a_c dt^2/2, drilling "
               "ledger PAD_SLIP_RATE_DT) adds up for as long as the hand moves: 1.0-1.3 mm in 5 s of brisk circles "
               "after the two repairs above, whatever the tool's mass. Each pad contact now has an anchor (the "
               "held body's pose in the finger's frame when they met); the friction rows of its contacts are given "
               "the offset from it as their position error, with the pad's normal stiffness times shear_ratio "
               "(G/E = 1/(2(1+nu)), nu = 0.5 for rubber: derived), so the pad springs back as rubber in shear does. "
               "The anchor follows once the offset exceeds reach_m, the pad's elastic reach before it slides "
               "(estimate: mu N / (G A / t) for a 2 mm pad of 3 cm2 at G about 1 MPa under 35 N is 0.19 mm; no "
               "datasheet value found: open), and a twist about the contact normal is held the same way over the "
               "pad's torsional lever (PAD_TORSIONAL_M). The Coulomb limit stays the solver's")
PAD_TORSIONAL_M = _p("PAD_TORSIONAL_M", 0.0042, "m", "repo",
                     "torsional friction of a pad contact about its normal: mu L/4, the twisting resistance of a line "
                     "contact as long as the pad (as in wine_pouring). Right where MuJoCo makes the contact in "
                     "one point (a round handle lying along the fingers). Where it makes two (a bar crossing the "
                     "pads, a knob's ridge) their tangential friction resists twisting as well, and the hand holds "
                     "up to three times a real pad's twist: a hanging 1 kg mixer was held against 1.03 N m about the "
                     "pinch where two 21 mm pads at 35 N hold 0.31 (demos/robot_mixer rehearsal; open). Taking the "
                     "torsional rows away is not the repair: without them a held round handle spun 40-55 degrees "
                     "about its own axis in 5 s of brisk circles in the air (tried in this round, reverted)")
CONTACT_GAP_M = _p("CONTACT_GAP_M", 0.0, "m", "repo", "box shapes (drilling e1b: a 2 mm gap gave false deep box-box contacts)")
MESH_CONTACT_GAP_M = _p("MESH_CONTACT_GAP_M", 0.002, "m", "repo", "hulls, capsules, cylinders (drilling)")
MAX_OBJECT_SPEED_M_S = _p("MAX_OBJECT_SPEED_M_S", 20.0, "m/s", "repo", "fault threshold (barcode)")
MAX_OBJECT_SPIN_RAD_S = _p("MAX_OBJECT_SPIN_RAD_S", 1000.0, "rad/s", "repo", "fault threshold; beaters exempt")

# ------------------------------------------------------------------ robot (FR3/Panda + Franka Hand)
FINGER_EFFORT_N = _p("FINGER_EFFORT_N", 35.0, "N", "repo", "Franka Hand 70 N continuous grasp, half per finger (drilling ledger)")
GRIP_FORCE_BAND_M = _p("GRIP_FORCE_BAND_M", 0.0005, "m", "repo", "hand-over between holding a force and a position (drilling)")
FRANKA_HAND_FINGER_SPEED = _p("FRANKA_HAND_FINGER_SPEED", 0.05, "m/s", "repo", "Franka Hand datasheet, per finger")
FR3_PAYLOAD_KG = _p("FR3_PAYLOAD_KG", 3.0, "kg", "repo", "FR3 datasheet rated payload at the flange (drilling ledger)")
FRANKA_HAND_KG = _p("FRANKA_HAND_KG", 0.73, "kg", "repo", "Franka Hand mass (drilling ledger); tool payload 2.27 kg")
LOAD_COM_MAX_M = _p("LOAD_COM_MAX_M", 0.5, "m", "authored", "farthest centre of mass a configured load may have from the TCP")
REFLEX_FILTER_S = _p("REFLEX_FILTER_S", 0.02, "s", "repo", "filtered external torque estimate (wine_pouring)")
REFLEX_RECOVERY_S = _p("REFLEX_RECOVERY_S", 1.0, "s", "repo", "targets refused until recovery (barcode)")
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
# elbows swing inward when the hands work over one bowl and the forearms meet (35-75 N, both reflexes)
READY_POSE_ARM = dict(right=(-0.55, -0.72, 0.62, -2.50, 0.42, 1.92, 0.50, 0.04, 0.04),
                      left=(0.55, -0.72, -0.62, -2.50, -0.42, 1.92, 1.07, 0.04, 0.04))
HAND_TCP_OFFSET = 0.1034
IK_MAX_EVALUATIONS = 300
IK_ROTATION_WEIGHT = 0.12
IK_SEED_WEIGHT = 1e-5


# ------------------------------------------------------------------ room and station
COUNTER_Z = _p("COUNTER_Z", 0.914, "m", "repo", "36 in counter (ingredient_mixing ledger)")
COUNTER_DEPTH = _p("COUNTER_DEPTH", 0.635, "m", "repo", "24 in carcass and 1 in overhang (ingredient_mixing ledger)")
COUNTER_LENGTH = _p("COUNTER_LENGTH", 2.4, "m", "authored", "one bay's casework run")
COUNTER_TOP_THICK = _p("COUNTER_TOP_THICK", 0.030, "m", "authored", "solid-surface top")
COUNTER_X0 = -1.15                    # the run spans x = -1.15 .. 1.25 (the basin at its left end)
ARM_BASE_Z = _p("ARM_BASE_Z", 0.95, "m", "repo", "cart deck height (ingredient_mixing: measured reach study)")
ARM_SPACING = _p("ARM_SPACING", 0.56, "m", "repo", "arm bases apart (ingredient_mixing, drilling)")
ARM_SETBACK = _p("ARM_SETBACK", 0.20, "m", "repo", "arm bases in front of the counter edge (ingredient_mixing)")
ARM_YAW = math.pi/2
ROOM = dict(x0=-1.90, x1=1.70, y0=-2.35, y1=0.655, z1=2.70)          # the bay: 3.6 m x 3.0 m x 2.7 m (authored)
STATION_CAMERA = dict(z=1.70, off_wall=0.10, fov=62.0, half=(0.03, 0.025, 0.02))                # on the services trunking, looking down and out
# a skills laboratory's ceiling recording camera behind the operator (fov: horizontal, deg): the counter, both arms, the
# instruction poster and the wall clock in one view, which no other policy camera has (slow-push's instruction times the
# push by the wall clock); the pose chosen from renders of a recorded slow-push, the clock 120 px across at 1280 x 720
BAY_CAMERA = _p("BAY_CAMERA", dict(eye=(0.40, -1.50, 2.45), target=(0.25, 0.55, 1.54), fov=72.0), "m, deg", "authored",
                "ceiling-mounted recording camera of a clinical skills laboratory; pose chosen from renders")

# ------------------------------------------------------------------ friction (Coulomb, pairwise by material)
MU = dict(default=_p("MU_DEFAULT", 0.4, "-", "estimate", "unlisted pairs"),
          pad=_p("MU_PAD", 0.8, "-", "repo", "Franka fingertip rubber (catalog profile; barcode_scanning)"),
          pp=_p("MU_PP", 0.3, "-", "estimate", "polypropylene on polypropylene or laminate (syringe, sheath, tray, basin)"),
          glass=_p("MU_GLASS", 0.4, "-", "estimate", "glass vial on laminate or polypropylene"),
          steel=_p("MU_STEEL", 0.2, "-", "estimate", "needle tube on plastic or laminate"),
          skin=_p("MU_SKIN", 0.9, "-", "estimate", "silicone skin of a trainer against plastic"),
          laminate=_p("MU_LAMINATE", 0.35, "-", "estimate", "counter top against plastic props"))

# ------------------------------------------------------------------ liquids
P_ATM = _p("P_ATM", 101325.0, "Pa", "literature", "standard atmosphere")
SIGMA = _p("SIGMA", 0.072, "N/m", "literature", "water-air surface tension, 20 C")
LIQUIDS = dict(
    saline=dict(rho=1005.0, mu=1.0e-3, rgb=(0.86, 0.93, 0.97), tint=0.05),
    medication=dict(rho=1000.0, mu=1.0e-3, rgb=(0.20, 0.45, 0.95), tint=0.55),
    contrast=dict(rho=1349.0, mu=11.6e-3, rgb=(0.90, 0.62, 0.15), tint=0.55),
    blood=dict(rho=1000.0, mu=1.0e-3, rgb=(0.72, 0.05, 0.06), tint=0.9),
)
_p("LIQUID_SALINE", (1005.0, 1.0e-3), "kg/m3, Pa s", "literature", "0.9 % saline at 20 C: water within a few per cent")
_p("LIQUID_MEDICATION", (1000.0, 1.0e-3), "kg/m3, Pa s", "authored", "simulated medication: water with a pale blue tint so cameras can see it")
_p("LIQUID_CONTRAST", (1349.0, 11.6e-3), "kg/m3, Pa s", "datasheet",
   "iohexol 300 mg I/mL at 20 C: 11.6 mPa s (Omnipaque data sheet, Medsafe NZ); specific gravity 1.349 (US label, stream)")
_p("LIQUID_BLOOD", (1000.0, 1.0e-3), "kg/m3, Pa s", "estimate", "trainer's simulated blood taken as water")
AIR = dict(rho=1.2, mu=1.8e-5)
DRAG = _p("DRAG", "Schiller-Naumann", "-", "literature",
          "drag of a drop in flight as of a rigid sphere: Cd = 24/Re (1 + 0.15 Re^0.687) below Re 1000, 0.44 above (Schiller and "
          "Naumann 1933, as given in Clift, Grace and Weber 1978); air at 20 C, 1.2 kg/m3 and 1.8e-5 Pa s")
SPECIES = ("saline", "medication", "contrast", "blood")

# ------------------------------------------------------------------ syringes
# Frame of a syringe: +x from the nozzle toward the thumb rest; origin on the axis at the chamber's front wall (where
# the stopper's front rib stands when the plunger is home). Lengths in metres.
SYRINGES = dict(
    s3=dict(label="3 mL Luer-lock", nominal_ml=3.0, bore=8.585e-3, od=10.2e-3, barrel_len=0.064, stroke_max=0.058,
            flange=(0.022, 0.013, 0.0018), rod_w=0.0065, thumb_d=0.0135, thumb_t=0.0015, proj=0.009, stopper_len=0.006,
            div_ml=0.1, mass_barrel=1.9e-3, mass_plunger=1.5e-3, dead_ml=0.05, compliance=3e-13,
            seal_break=2.0, seal_glide=0.7),
    s10=dict(label="10 mL Luer-lock", nominal_ml=10.0, bore=14.43e-3, od=16.2e-3, barrel_len=0.080, stroke_max=0.070,
             flange=(0.028, 0.019, 0.0020), rod_w=0.010, thumb_d=0.019, thumb_t=0.0020, proj=0.0125, stopper_len=0.008,
             div_ml=0.2, mass_barrel=5.2e-3, mass_plunger=3.3e-3, dead_ml=0.07, compliance=1e-12,
             seal_break=3.3, seal_glide=1.2),
    s1=dict(label="1 mL Luer-lock", nominal_ml=1.0, bore=4.699e-3, od=6.6e-3, barrel_len=0.070, stroke_max=0.062,
            flange=(0.017, 0.009, 0.0015), rod_w=0.0035, thumb_d=0.009, thumb_t=0.0012, proj=0.008, stopper_len=0.004,
            div_ml=0.01, mass_barrel=1.0e-3, mass_plunger=0.7e-3, dead_ml=0.04, compliance=1e-13,
            seal_break=1.0, seal_glide=0.15),
)
SYRINGES["flush"] = dict(SYRINGES["s10"], label="prefilled 10 mL saline flush", seal_first=12.0, seal_glide=1.5)
_p("SYRINGE_BORES", (4.699e-3, 8.585e-3, 14.43e-3), "m", "stream",
   "BD 1 / 3 / 10 mL barrel inside diameters (two syringe-pump makers' tables)")
_p("SYRINGE_OUTSIDE", "SYRINGES[*] od, barrel_len, flange, rod_w, thumb_d, proj", "m", "estimate",
   "anchored on Prasetyono & Adhistana 2019's measured Terumo syringes (read) and ISO 7886-1's plunger projections "
   "of 8 / 9 / 12.5 mm (standard); no maker's drawing was found")
_p("SYRINGE_MASS", "SYRINGES[*] mass_barrel, mass_plunger", "kg", "estimate", "polypropylene volume of the estimated shapes")
_p("SYRINGE_DEAD_SPACE", (0.04, 0.05, 0.07), "mL", "estimate", "under ISO 7886-1 Table 1's maxima 0.07 / 0.07 / 0.10 mL (standard)")
_p("SYRINGE_COMPLIANCE", (1e-13, 3e-13, 1e-12), "m3/Pa", "estimate",
   "barrel and stopper; the barrel's hoop strain alone gives 2e-13 for 10 mL (derived)")
_p("SEAL_BREAK_N", (1.0, 2.0, 3.3), "N", "read", "break-loose force of fresh 1 / 3 / 10 mL syringes at 100 mm/min (Prasetyono & Adhistana 2019)")
_p("SEAL_GLIDE_N", (0.15, 0.7, 1.2), "N", "estimate",
   "between the residual seal shares (0.07-0.6 N) and the paper's wide-needle gliding forces (0.13-1.9 N)")
_p("SEAL_FIRST_FLUSH_N", 12.0, "N", "estimate",
   "first movement of a stored prefilled syringe: 9 N read for a 1 mL-class one (Yoshino et al. 2014), scaled to the 10 mL barrel")
SEAL_RECOVERY_S = _p("SEAL_RECOVERY_S", 20.0, "s", "estimate", "rest time over which break-loose returns; ISO 7886-1 Annex G waits 30 s")
SEAL_STRIBECK_M_S = _p("SEAL_STRIBECK_M_S", 0.002, "m/s", "estimate", "speed over which stick gives way to glide (open; E3 varies it)")
SEAL_VISCOUS = _p("SEAL_VISCOUS", 20.0, "N s/m", "estimate", "viscous part of the seal's glide (0.03 N at 100 mm/min)")
PLUNGER_RETENTION_N = _p("PLUNGER_RETENTION_N", 15.0, "N", "estimate", "pull at which the stopper passes the barrel's retaining ring")
PLUNGER_BASE_ARMATURE = _p("PLUNGER_BASE_ARMATURE", 0.0, "kg", "engine",
                           "no artificial plunger inertia: hydraulic/seal velocity loads are in the implicit "
                           "bilateral constraint-force solve")
FRICTION_ROW = _p("FRICTION_ROW", (0.9999, 0.9999, 0.001, 0.5, 2.0), "-", "engine",
                  "impedance of every dry-friction row of a fit, with time constant 2 dt")

# Luer fittings (ISO 594-1/-2, read)
LUER = dict(tip_d=0.00396, mouth_d=0.00429, cone_len=0.0075, collar_od=0.0096, nozzle_len=0.0105, nozzle_proj=0.0021,
            lugs_d=0.00783, lead=0.005, d_thread=0.0032, d_seat=0.0055, bore_d=0.0020)
_p("LUER_GEOMETRY", "LUER", "m", "standard", "ISO 594-1:1986 and 594-2:1991 (as IS 3234): 6 % cone, lugs 7.83 mm, 5 mm lead, "
   "thread start 3.2 mm from the tip; seat at 5.5 mm (derived: mouth 4.29 mm less tip 3.96 mm over the 6 % taper)")
LUER_CAPTURE = _p("LUER_CAPTURE", dict(radius=0.000165, depth=0.0025, angle_deg=5.0, lead_in=-0.0002), "m, m, deg, m", "authored",
                  "capture requires at least 0.2 mm actual insertion and clearance within the 4.29 mm mouth / 3.96 mm tip; "
                  "guides it (a weld between the nozzle's carrier and the fitting)")
LUER_FRICTION = _p("LUER_FRICTION", dict(tau0=0.003, mu=0.3, r_seat=0.0020, r_thread=0.0039, k_seat=5.0e4, slide_n=0.3),
                   "N m, -, m, m, N/m, N", "estimate",
                   "the turn rubs on the cone (mean radius 2.0 mm) and on the thread's flanks (lugs at 3.9 mm, standard) with a "
                   "polypropylene pair's 0.3: 0.05 N m seats with 18 N, a joint made with 27.5 N resists 0.030 N m of unscrewing "
                   "(ISO 594-2 asks 0.02) and is self-locking under a pull (0.3 x 3.9 mm > 5 mm / 2 pi); the seat's stiffness is "
                   "a stability choice (0.55 mm of taper travel for 27.5 N)")
LUER_DAMPING = _p("LUER_DAMPING", (20.0, 0.002), "N s/m, N m s/rad", "engine",
                  "viscous loss of the fit's slide and turn: without it a fitting turned by a constant torque spins up over "
                  "the 166 degrees to the seat and arrives with the whole work of the torque (0.02 N m gave 68 N of seat force "
                  "and an overridden thread on the bench, where the static balance is 8 N); at a hand's 1 rad/s it adds 0.002 N m")
THREAD_CATCH_DEG = _p("THREAD_CATCH_DEG", 10.0, "deg", "authored",
                      "turn against the thread's start after which the lugs are under the thread and the joint holds both ways")
LUER_OVERRIDE_N = _p("LUER_OVERRIDE_N", 85.0, "N", "derived",
                     "seat force above which the thread overrides: ISO 594-2 asks that 0.15 N m with 27.5 N of push does not "
                     "override, which in this joint's balance is 78 N of seat force (bench E6)")
CARRIER_MASS = _p("CARRIER_MASS", 2e-4, "kg", "engine", "mass of a fit's carrier body; its joints carry CARRIER_ARMATURE")
CARRIER_ARMATURE = _p("CARRIER_ARMATURE", (0.05, 2e-7), "kg, kg m2", "engine", "slide and turn of a fit's carrier (drilling)")
FIT_SOLREF = (2*DT, 1.0)
FIT_SOLIMP = (0.9999, 0.9999, 0.001, 0.5, 2.0)

# needles
NEEDLES = dict(
    g18b=dict(label="18 G x 38 mm blunt fill", od=1.27e-3, bore=0.838e-3, length=0.0381, colour=(0.93, 0.45, 0.62), blunt=True),
    g21=dict(label="21 G x 38 mm", od=0.819e-3, bore=0.514e-3, length=0.0381, colour=(0.10, 0.50, 0.22), blunt=False),
    g23=dict(label="23 G x 25 mm", od=0.642e-3, bore=0.337e-3, length=0.0254, colour=(0.10, 0.25, 0.75), blunt=False),
    g25=dict(label="25 G x 25 mm", od=0.515e-3, bore=0.260e-3, length=0.0254, colour=(0.95, 0.50, 0.10), blunt=False),
    g27=dict(label="27 G x 13 mm", od=0.413e-3, bore=0.210e-3, length=0.0127, colour=(0.55, 0.55, 0.55), blunt=False),
)
_p("NEEDLE_TUBES", "NEEDLES[*] od, bore", "m", "datasheet", "Hamilton needle gauge chart (nominal); hub colours ISO 6009 (standard)")
NEEDLE_BORE_VARIANT = _p("NEEDLE_BORE_VARIANT", 0.9, "-", "authored",
                         "the 'tight' selector: nominal bore less 10 %, inside ISO 7864's stylet sizes (standard)")
NEEDLE_HUB = dict(length=0.015, od=0.0062, mass=0.25e-3)
NEEDLE_HUB_FLOW = _p("NEEDLE_HUB_FLOW", dict(embedded=0.001, bore=0.0015, embedded_range=(0.0, 0.0075)), "m", "estimate",
                     "1 mm of cannula behind the exposed-length datum, bounded 0-7.5 mm; the remaining hub passage "
                     "is 1.5 mm bore. This is an explicit uncertain short adhesive bond/passage model, not the old "
                     "unsupported 7.5 mm of needle bore. Published-force comparison reports the full 0-7.5 mm bound")
NEEDLE_SHEATH = dict(od=0.0065, wall=0.0007, overlap=0.010, mass=0.35e-3, pull_n=6.0, cap_tol=0.001)
_p("NEEDLE_SHEATH_PULL_N", 6.0, "N", "estimate", "press fit of a sheath on its hub; ISO 7864's hub-tube bond of 22-69 N is far above it")
E_STEEL = _p("E_STEEL", 193e9, "Pa", "repo", "austenitic stainless (control_cable)")
NEEDLE_YIELD = _p("NEEDLE_YIELD", 1.0e9, "Pa", "estimate", "hard-drawn 304 tube (full-hard strip is specified from 965 MPa)")
NEEDLE_BEND_FLAG_DEG = _p("NEEDLE_BEND_FLAG_DEG", 2.0, "deg", "authored", "permanent set above which a needle counts as bent")
NEEDLE_BLUNT_N = _p("NEEDLE_BLUNT_N", 0.5, "N", "estimate", "tip load on a hard surface that hooks a point")
K_ENTRY = _p("K_ENTRY", 1.2, "-", "fit", "kinetic entry and exit loss fitted against a published force table, not a measured coefficient")

# what a needle enters: puncture force at 0.5 mm outside diameter (scaled by diameter), static and
# sliding friction per metre inserted, cutting force, surface stiffness before puncture
LAYERS = dict(
    stopper=dict(thick=0.0035, puncture=3.1, fs=300.0, fk=200.0, cut=0.0, k=2500.0),
    skin=dict(thick=0.002, puncture=1.2, fs=30.0, fk=20.0, cut=0.0, k=400.0, perm=5e-12),
    fat=dict(thick=0.008, puncture=0.0, fs=10.0, fk=7.0, cut=0.1, k=0.0, perm=3e-11),
    muscle=dict(thick=0.030, puncture=0.5, fs=20.0, fk=15.0, cut=0.3, k=0.0, perm=1e-11),
)
_p("LAYER_LAWS", "LAYERS", "N, N/m, N", "estimate",
   "bounded by USP <381> (10 N for a 21 G needle through a closure, standard) and pen needles in ISO 7864's foil "
   "(0.36-0.73 N peak, 0.04-0.05 N sliding; Leonardi et al. 2019, read); three-part model after Okamura et al. (survey read)")
TRACK_LATERAL = _p("TRACK_LATERAL", dict(pad=(300.0, 0.003), stopper=(5000.0, 0.001), hard=5000.0, damping=0.5), "N/m, m, N/m, N s/m", "estimate",
                   "sideways give of the place a needle went in: (stiffness, reach) for skin over fat and for a stopper "
                   "in its crimp, the stiffness beyond the reach, and a damper. Skin with a needle through it moves a "
                   "millimetre under a few tenths of a newton (the estimate; no measurement found: open); a stopper is "
                   "rubber in a metal crimp. With the entry point pinned rigidly, the needle 9 mm in and 17 mm of it "
                   "free, the 0.1 mm a gripping hand moves when the other hand presses the plunger is 2 N at the skin "
                   "and the needle is flagged bent")
TIP_GRIP = _p("TIP_GRIP", 1.0, "-", "estimate", "sideways hold of a point in the dimple it presses into skin or rubber, as a share of its "
              "pressing force (a friction coefficient of 1; before puncture only)")
BLUNT_FILL_FACTOR = _p("BLUNT_FILL_FACTOR", 1.5, "-", "estimate", "a blunt fill needle's puncture force over a sharp one's of its size")
TISSUE = dict(bulk=1.0e5, region_r=0.010, track_leak=0.1)
_p("TISSUE", (1e-11, 3e-11, 1e5, 0.1), "m2, m2, Pa, -", "estimate",
   "synthetic layered trainer with a lumped poroelastic depot. 100 kPa is the injection-pressure modulus scale "
   "in Thomsen et al. 2014's subcutaneous fit, not a drained-compression modulus or validated muscle property. "
   "Layer labels describe trainer targets. Permeabilities 1e-11 and 3e-11 m2 lie in that digital order-of-magnitude "
   "range; the 10 mm storage radius and 0.1 track-return fraction remain authored uncertain parameters. "
   "No waiting-time pass/fail is calibrated by these values. A sensitivity run bounds modulus, permeability, "
   "storage radius and track leakage together; no patient-specific retained-dose claim is supported")
TRACK_PIVOT_K = _p("TRACK_PIVOT_K", 5.0e4, "N/m per m", "estimate", "lateral stiffness of a track per inserted length")

# vial (ISO 8362-1 10R, read)
VIAL = dict(od=0.024, height=0.045, flange_d=0.020, bore=0.0126, wall=0.001, overflow_ml=13.5, mass=0.0130,
            plug_ml=0.9, target_d=0.007, neck_h=0.0085, cap_h=0.0045)
_p("VIAL_10R", "VIAL", "m, mL, kg", "standard", "ISO 8362-1:2003 Table 1 (10R: 24 x 45 mm, flange 20, bore 12.6, 13.5 mL, 10.2 g glass); "
   "closure mass, 0.9 mL stopper plug and 7 mm target are estimates")
ADAPTER = dict(height=0.032, od=0.024, mass=0.004)

# intravenous lock
CONNECTOR = dict(length=0.033, od=0.0085, mass=0.0025, spring0=2.0, spring_k=1200.0, open_from=0.0035, open_full=0.0055,
                 r_open=3.2e9, prime_ml=0.1, mount_k=800.0, mount_kr=0.05, mount_c=3.0, mount_cr=2e-4)
_p("CONNECTOR", "CONNECTOR", "-", "estimate",
   "needle-free connector: spring 2 N + 1.2 N/mm (under ISO 594-2's 20 N for assembly), open over 3.5-5.5 mm of nozzle "
   "depth; open resistance from 183 mL/min under 1 m of water (stream); held to the skin by tape: 800 N/m, 0.05 N m/rad")
EXTENSION = dict(length=0.15, bore=0.0029, od=0.0041, compliance=1e-12, clamp_travel=0.008, clamp_n=4.0)
_p("EXTENSION_SET", (0.15, 0.0029), "m", "read", "1 mL in the extension tubing (Doyle & McCutcheon 7.5): 150 mm of 2.9 mm bore")
CANNULAS = dict(g24=dict(bore=0.47e-3, length=0.019), g22=dict(bore=0.60e-3, length=0.025),
                g20=dict(bore=0.80e-3, length=0.032), g18=dict(bore=0.95e-3, length=0.032))
_p("CANNULAS", "CANNULAS", "m", "estimate", "bores matched to published gravity flows within 1.5 (flows stream)")
VEIN_GAUGE_PA = _p("VEIN_GAUGE_PA", 2900.0, "Pa", "authored", "reservoir of simulated blood 0.30 m above the vein")
LINE_LIMIT_PA = _p("LINE_LIMIT_PA", 300e3, "Pa", "standard", "leak-test pressure of syringes (ISO 7886-1) and Luer joints (ISO 594)")

# open liquid
DROP_FACTOR = _p("DROP_FACTOR", 0.6, "-", "literature", "a hanging drop falls at about 0.6 of Tate's volume (Harkins and Brown)")
JET_WEBER = _p("JET_WEBER", 4.0, "-", "literature", "dripping gives way to jetting near this Weber number")
JET_PARCEL_ML = 0.002
JET_CONE_DEG = _p("JET_CONE_DEG", 5.0, "deg", "estimate", "spread of a jet's parcels")
BOND_RISE = _p("BOND_RISE", 0.842, "-", "literature", "a long bubble rises in a vertical tube above this Bond number (Bretherton 1961)")
BUBBLE_DRIFT = _p("BUBBLE_DRIFT", 0.35, "-", "literature", "drift speed of a bubble in a tube, in units of sqrt(g D)")
BUBBLE_TILT_DEG = _p("BUBBLE_TILT_DEG", 20.0, "deg", "estimate", "tip elevation above which gas at the nozzle end leaves first")

# props
TRAY = dict(size=(0.300, 0.200, 0.022), wall=0.003, rib_h=0.002, rib_pitch=0.015, mass=0.180)
_p("TRAY", "TRAY", "m, kg", "authored", "polypropylene procedure tray, catalogue-typical; ribs 2 mm high every 15 mm along its length")
BASIN = dict(size=(0.250, 0.120, 0.045), wall=0.0025, mass=0.060)
SHARPS = dict(size=(0.270, 0.100, 0.270), wall=0.003, slot=(0.180, 0.045), mass=0.420, flap_mass=0.012, flap_closing_nm=4.0e-4)
_p("SHARPS", "SHARPS", "m, kg, N m", "estimate",
   "counter-standing sharps container with a horizontal-drop slot; the flap's counterweight holds it level with 0.4 mN m "
   "(a 3 mL syringe laid on it tips it; the lightest has to be pushed)")
PAD = dict(size=(0.130, 0.130), board=0.005, ring_d=0.020, mass=0.60)
PAD_UPRIGHT_Z = 1.12
LAYOUT = dict(tray=(-0.03, 0.16), work=(0.12, 0.36), basin=(-0.33, 0.18), vial=(-0.22, 0.36), wipes=(-0.52, 0.42),
              sharps=(0.50, 0.44), spares=(-0.20, 0.52), stand=(1.40, -0.42))
PROPS = dict(wipes_box=(0.0, 0.0, (0.0525, 0.0275, 0.025), 0.30, "wipes_box"),
             gauze_stack=(0.0, -0.11, (0.026, 0.026, 0.008), 0.020, "gauze"))
TRAINING_ARM = dict(board=0.02, hand=0.18, forearm=0.29, hand_thick=0.035, wrist_r=0.028, elbow_r=0.042, connector_tilt_deg=25.0)
_p("TRAINING_ARM", "TRAINING_ARM", "m, deg", "authored",
   "adult forearm and hand of an intravenous training arm, 0.47 m on the bench (distributor figures, stream); section sizes authored")

# the bay's fixed fittings: collision boxes of what is drawn
TRUNKING = dict(z=1.50, height=0.15, depth=0.07)
WALL_FITTINGS = dict(                      # name: (centre x, width, height, depth from the wall, bottom z)
    gloves=(-0.62, 0.40, 0.27, 0.11, 1.74),
    towels=(-1.38, 0.28, 0.36, 0.20, 1.28),
    handrub=(0.98, 0.09, 0.22, 0.10, 1.12),
    clock=(0.62, 0.30, 0.30, 0.045, 1.95),
    poster=(0.02, 0.42, 0.594, 0.004, 1.74),
)
SINK = dict(x=(-1.13, -0.75), y=(0.08, 0.56), rim=0.018, tap_z=0.24)
BED = dict(x=(-0.45, 1.65), y=(-2.30, -1.35), z=0.56)


if __name__ == "__main__":
    for e in LEDGER:
        print(f"{e['name']:28s} {str(e['value'])[:60]:60s} {e['unit']:14s} {e['label']:10s} {e['basis'][:110]}")
