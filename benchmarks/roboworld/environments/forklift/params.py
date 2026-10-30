"""Parameter ledger of the forklift environment.

Every physical constant a model reads lives here with its unit, provenance label and source. Labels: read
(primary source opened and the number found in it), read* (via a fetch summariser, the page's URL given), search (not
confirmed against an opened copy: a search snippet, or a page seen only through a fetch summariser whose figure could not
be re-read; the URL where known), measured (this machine), derived (computed; arithmetic in the comment), fitted (set so the model reproduces a read observable: the probe that reproduces that observable
checks the fit and is not a validation), estimate (stated reason; a class value recalled from textbooks without a source
opened is an estimate and says so), authored (a design choice), repo (a value established and tested by an earlier
environment of this repository, cited by name), engine (the physics engine's own default, kept), open (no basis;
swept in the probes). A label "a/b" (read/estimate, ...) marks an entry whose fields have different provenance; its
source says which is which.
`ledger()` returns the table.
"""

import math

LEDGER = {}


def P(name, value, unit, label, source):
    LEDGER[name] = dict(value=value, unit=unit, label=label, source=source)
    return value


IN, FT, LB = 0.0254, 0.3048, 0.45359237
G = P("G", 9.81, "m/s^2", "authored", "standard gravity (repository convention)")

# ------------------------------------------------------------------------------------------------ clock and solver
DT = P("DT", 0.0005, "s", "authored", "0.5 ms: a floating humanoid needs it (barcode_scanning)")
CONTACT_TIMECONST = P("CONTACT_TIMECONST", 0.004, "s", "repo", "barcode_scanning: 4 ms rigid contacts")
CONTACT_SOLIMP = P("CONTACT_SOLIMP", (0.95, 0.99, 0.001), "-", "repo", "barcode_scanning")
SOLVER_ITERATIONS = P("SOLVER_ITERATIONS", 60, "-", "authored", "MuJoCo Newton iterations")
NOSLIP_ITERATIONS = P("NOSLIP_ITERATIONS", 10, "-", "repo", "barcode_scanning (held objects creep without it)")
IMPRATIO = P("IMPRATIO", 10.0, "-", "repo", "barcode_scanning")
LS_ITERATIONS = P("LS_ITERATIONS", 50, "-", "measured",
                  "MuJoCo line-search iterations per solver iteration (its default). At 20 the Newton solver's line search ran out on hard ticks, the solve stalled with its gradient at 60 (scaled) and returned unbalanced forces: 13 kN m from a shoulder joint limit that needed 61 N m threw the arm at 134 rad/s (E17)")
SOLVER_STALL_GRADIENT = P("SOLVER_STALL_GRADIENT", 1.0, "-", "authored",
                          "fault when the constraint solver ends a tick with its scaled gradient above this (converged ticks end below 1e-4; the stalled ones were at 60): forces from a stalled solve are not physics")
INTEGRATOR = P("INTEGRATOR", "implicitfast", "-", "repo", "MuJoCo integrator")
SOLVER_ISLANDS = P("SOLVER_ISLANDS", True, "-", "engine", "MuJoCo's per-island constraint solve (its default)")
JOINT_LIMIT_TIMECONST = P("JOINT_LIMIT_TIMECONST", 0.001, "s", "repo", "joint limits: MuJoCo solref time constant, 2 x dt (barcode_scanning)")
CCD_TOLERANCE = P("CCD_TOLERANCE", 1.0e-8, "m", "measured",
                  "MuJoCo convex-collision tolerance. At the default 1e-6 the narrow phase reported a pallet resting on 25 contact points (0.3-0.5 um deep) as separated for single ticks (369-396 of 6,000 in the chocked dock): it dropped, re-landed and shook the trailer and the leveler; 0 of 6,000 at 1e-8 and at 1e-9. At 1e-9 one E17 run ended with MuJoCo's 'EPA: out of memory' and a thrown horn button, so 1e-8")
CCD_ITERATIONS = P("CCD_ITERATIONS", 35, "-", "measured", "MuJoCo's default; at 100 three of six E17 runs ended in a segmentation fault inside mj_step (the convex-collision buffers), none of six at 35")
MAX_LOAD_RISE = P("MAX_LOAD_RISE", 6.0, "m/s", "derived", "fault for a load (or the truck) moving upward faster than this: the fastest legal upward motion is a rebound of a load dropped from the highest point it can reach (10.7 m/s at impact) with a restitution of at most 0.5 (wood, cardboard; estimate): 5.4 m/s. Cartons thrown 7 m up at 11 m/s by a box-box contact stayed under the 15 m/s speed fault, which a load near the floor may legally approach after a 5.8 m fall")
MAX_BODY_SPEED = P("MAX_BODY_SPEED", 54.0, "m/s", "derived",
                   "fault for the robot's links: 1.1 x the fastest an H1-2 hand can move with every joint of the chain at its motor's no-load speed: torso 23 rad/s x 0.95 m (21.85) + shoulder pitch and roll 9 rad/s x 0.75 m each (13.50) + elbow 20 rad/s x 0.42 m (8.40) + wrist 31.4 rad/s x 0.17 m (5.34) = 49.1 m/s; x 1.1 = 54. The G1's chain is shorter and slower, so the same guard covers it")
ROBOT_GAIN_BOUNDS = P("ROBOT_GAIN_BOUNDS", {"leg": (1500.0, 60.0), "torso": (1000.0, 40.0), "arm": (500.0, 25.0), "wrist": (200.0, 10.0),
                                            "hand": (20.0, 1.0)}, "N m/rad, N m s/rad", "authored",
                      "upper bounds on the stiffness and damping a policy may command per joint class (keep commands physical)")
ROBOT_ARMATURE = P("ROBOT_ARMATURE", {"leg": 0.10, "torso": 0.10, "arm": 0.03, "wrist": 0.01, "hand": 5.0e-3}, "kg m^2", "estimate",
                   "reflected rotor inertia per joint class where the robot's controller profile gives none. Fingers (Inspire RH56: micro "
                   "lead-screw servos through linkages): rotor inertia x (rotor rad per finger rad)^2 spans 0.003-0.03 kg m^2 per finger "
                   "(rotor 2-10e-9 kg m^2, reduction 1,200-1,800, neither published: open); 0.01 per finger, the range's geometric middle, "
                   "split equally between the driver joint and its linkage follower (5e-3 each). At 1e-4 a wrist impact "
                   "whipped the fingers to 30 m/s (E17). Leg, torso, arm, wrist: estimates of geared actuators")
JOINT_SPEED_FLOOR = P("JOINT_SPEED_FLOOR", 100.0, "rad/s", "authored",
                      "the joint-speed fault never triggers below this. A back-driven joint is moved by contact, not its motor, so it may legally exceed its motor's no-load speed (an Inspire finger's is 0.5 rad/s in the URDF) whenever the hand knocks a control. 100 rad/s is a 40 mm finger link whose tip moves 4 m/s relative to the palm; the solver failure this guard exists for threw finger tips at 25-34 m/s (E17), 600-850 rad/s on such a link. It is not tied to each motor: a floor between knocks and failures, authored")
JOINT_SPEED_FAULT = P("JOINT_SPEED_FAULT", 5.0, "-", "authored",
                      "fault when a robot joint turns faster than this many times its motor's no-load speed (a struck limb can exceed the motor's speed; the solver failures of E17 gave 15 x)")
MAX_LOAD_SPEED = P("MAX_LOAD_SPEED", 15.0, "m/s", "derived", "fault for the truck and every load: the fastest legal motion here is a free fall from the highest point a load can reach (4.44 m lift + 1.4 m load = 5.84 m: 10.7 m/s); the truck tops out at 5.2 m/s")
MAX_BODY_SPIN = P("MAX_BODY_SPIN", 1000.0, "rad/s", "repo", "barcode_scanning")

# ------------------------------------------------------------------------------------------------ truck (Hyster H2.5A LPG class)
WHEELBASE = P("WHEELBASE", 1.649, "m", "measured", "Forklift C asset wheel centres (H2.5A sheet 1.650 m, read)")
FRONT_WHEEL_CENTRE = P("FRONT_WHEEL_CENTRE", (-0.003, 0.499, 0.325), "m", "measured", "asset left front wheel mesh centre")
REAR_WHEEL_CENTRE = P("REAR_WHEEL_CENTRE", (-1.649, 0.5685, 0.2565), "m", "measured", "asset left rear wheel mesh centre")
FRONT_WHEEL_RADIUS = P("FRONT_WHEEL_RADIUS", 0.323, "m", "measured", "asset 7.00-12 tyre OD 0.646 m (Trelleborg 25.6 in = 0.650 m, read)")
REAR_WHEEL_RADIUS = P("REAR_WHEEL_RADIUS", 0.2565, "m", "measured", "asset 6.00-9 tyre OD 0.513 m")
FRONT_TYRE_WIDTH = P("FRONT_TYRE_WIDTH", 0.173, "m", "read", "Trelleborg Elite XP 7.00-12 section 6.8 in")
REAR_TYRE_WIDTH = P("REAR_TYRE_WIDTH", 0.142, "m", "read", "Trelleborg 6.00-9 SE section width")
KINGPIN = P("KINGPIN", (-1.6519, 0.4061, 0.256), "m", "measured",
            "steer-axle kingpin axis (vertical): the centre of the cylindrical kingpin boss (r 32.8 mm) of the asset's rear "
            "knuckle mesh, fitted to its top and bottom rings (E21); "
            "it coincides within 0.002 mm with the asset's own authored rotator joint pivot. z: the knuckle's mid-height")
AXLE_PIVOT = P("AXLE_PIVOT", (-1.62, 0.0, 0.30), "m", "estimate", "centre-pin bracket in the asset (components at x -1.64..-1.54, z 0.28..0.41)")
AXLE_OSC_LIMIT = P("AXLE_OSC_LIMIT", math.radians(3.5), "rad", "open", "no source for the steer-axle oscillation stop; swept 2-5 deg")
STEER_INNER_MAX = P("STEER_INNER_MAX", math.radians(80.0), "rad", "search", "Hangcha XC-series CPD20-35 maintenance "
                                                                           "manual: inner 80 deg, outer 55.7 deg (https"
                                                                           "://www.manualslib.com/manual/4086789/Hc-Xc-"
                                                                           "Series.html, read through a fetch "
                                                                           "summariser, not re-opened: search)")
MAST_TILT_PIVOT = P("MAST_TILT_PIVOT", (0.17, 0.0, 0.33), "m", "estimate", "outer upright foot on the drive-axle housing (asset mast x 0.14-0.39 m)")
MAST_TILT_FWD = P("MAST_TILT_FWD", math.radians(6.0), "rad", "read", "Hyster H2.5A: tilt forward/back 6/6 deg")
MAST_TILT_BACK = P("MAST_TILT_BACK", math.radians(6.0), "rad", "read", "Hyster H2.5A")
MAST_STAGE_STROKE = P("MAST_STAGE_STROKE", 1.69, "m", "derived", "per-stage stroke of a 3-stage full-free-lift mast of the asset's 2.17 m lowered height: Hyster H2.5A table (p. 9, read) for 2,170 mm lowered: extended 5,549 mm without load backrest, TOF 4,950 mm, so (5,549 - 2,170) / 2 = 1,690 mm")
CARRIAGE_LOWER = P("CARRIAGE_LOWER", -0.095, "m", "authored", "carriage below the asset pose so the blades reach 9 mm above the floor")
CARRIAGE_FREE_LIFT = P("CARRIAGE_FREE_LIFT", 0.96, "m", "derived", "backrest top 1.203 m to inner-stage top 2.164 m in the asset")
SIDESHIFT_TRAVEL = P("SIDESHIFT_TRAVEL", 0.10, "m", "authored", "integral side-shifter +/-100 mm (class II common option)")

SERVICE_WEIGHT = P("SERVICE_WEIGHT", 3990.0, "kg", "read", "Hyster H2.5A LPG service weight")
AXLE_LOAD_UNLADEN = P("AXLE_LOAD_UNLADEN", (1728.0, 2262.0), "kg", "read", "Hyster H2.5A front/rear")
AXLE_LOAD_LADEN = P("AXLE_LOAD_LADEN", (5698.0, 792.0), "kg", "read", "Hyster H2.5A front/rear with 2500 kg at 500 mm")
RATED_CAPACITY = P("RATED_CAPACITY", 2500.0, "kg", "read", "Hyster H2.5A at 500 mm load centre (2-stage limited-free-lift mast, no side-shift). Check, held out: the data plate below gives this truck 2,320 kg at 500 mm; Hyster rates its H2.5A with a 3-stage full-free-lift mast of the same 2,170 mm lowered height and side-shift + fork positioner at 2,310 kg at 500 mm (p. 9, read)")
LOAD_CENTRE = P("LOAD_CENTRE", 0.500, "m", "read", "Hyster H2.5A")
# The data plate of the simulated truck. Forward stability is a moment about the front axle, and the simulated truck has
# the H2.5A's masses and wheelbase, so it carries the H2.5A's rated load moment. Its fork face (the asset's) stands
# further from the front axle than the H2.5A's, so the same moment is a smaller load at a given load centre.
RATED_LEVER = P("RATED_LEVER", (2262.0 - 792.0) * 1.649 / 2500.0, "m", "derived",
                "the H2.5A's rated load CG ahead of its front axle, from the sheet's rear axle loads: (2262 - 792) kg x wheelbase / 2500 kg = 0.970 m")
LOAD_DISTANCE = P("LOAD_DISTANCE", 0.545, "m", "measured",
                  "front axle centre (x -0.003) to the fork shanks' front face above the blades (x 0.542) on the built model's collision "
                  "hulls, forks level at 0.10 m lift. The hulls' rear face is at 0.483 m. The H2.5A's is RATED_LEVER - 0.5 = 0.470 m")


def capacity(load_centre):
    """Rated capacity (kg, load and pallet) with the load's centre `load_centre` metres ahead of the fork face: the
    H2.5A's rated moment over this truck's lever. 0.5 m: 2,320 kg; 24 in: 2,099 kg; a 48 in pallet carried lengthwise
    against the fork face (0.6125 m): 2,094 kg; 0.425 m: 2,500 kg."""
    return float(RATED_CAPACITY) * float(RATED_LEVER) / (float(LOAD_DISTANCE) + float(load_centre))


CG_X_UNLADEN = P("CG_X_UNLADEN", -1.649 * 2262.0 / 3990.0, "m", "derived", "truck CG behind the front axle from axle loads (0.935 m)")
CG_Z_UNLADEN = P("CG_Z_UNLADEN", 0.68, "m", "estimate",
                 "unladen truck CG height, estimated from the components (counterweight, engine, frame, mast). "
                 "No published value was found; its plausible band is 0.55-0.80 m, over which the derived "
                 "unladen roll threshold falls from 0.54 g to 0.33 g. No probe sweeps "
                 "the band (only truck.component_masses reads it), so the stability results hold at "
                 "0.68 m only")

# component masses (estimates within the service weight; see truck.py for the chassis remainder)
M_WHEEL_FRONT = P("M_WHEEL_FRONT", 70.0, "kg", "estimate", "Maxam 7.00-12 SE tyre 48.5 kg (search) + rim and hub ~22 kg")
M_WHEEL_REAR = P("M_WHEEL_REAR", 42.0, "kg", "estimate", "6.00-9 SE tyre ~32 kg + rim ~10 kg")
M_KNUCKLE = P("M_KNUCKLE", 18.0, "kg", "estimate", "cast steering knuckle")
M_AXLE = P("M_AXLE", 110.0, "kg", "estimate", "steer axle beam with straight-through cylinder")
M_MAST_OUTER = P("M_MAST_OUTER", 230.0, "kg", "estimate", "triplex outer upright with lift cylinders")
M_MAST_INT = P("M_MAST_INT", 160.0, "kg", "estimate", "triplex intermediate upright")
M_MAST_INNER = P("M_MAST_INNER", 130.0, "kg", "estimate", "triplex inner upright with free-lift cylinder")
M_CARRIAGE = P("M_CARRIAGE", 120.0, "kg", "estimate", "class II carriage")
M_SIDESHIFT = P("M_SIDESHIFT", 70.0, "kg", "estimate", "side-shift plate 45 kg + load backrest 25 kg")
M_FORK = P("M_FORK", 42.0, "kg", "read", "Bolzoni class 2A 100x40x1000 mm fork")
M_SEAT = P("M_SEAT", 25.0, "kg", "estimate", "suspension seat upper (cushion, backrest, armrests)")

# ------------------------------------------------------------------------------------------------ tyres
# Vertical law anchored on a measured super-elastic forklift tyre: Oh, Danilov, Fischer, Guenthner, Bruns, "Untersuchung
# und Modellierung der Schwingungsuebertragung von Flurfoerderzeugreifen", Logistics Journal Proceedings 2013
# (doi 10.2195/lj_Proc_oh_de_201310_01), Continental 18x7-8 SE on a hydropulse rig: mean force 1 / 5 / 10 / 15 / 20 kN at
# 2.7 / 8 / 12.5 / 16.5 / 19.8 mm (Fig. 2), which F = 7.1e6 p^1.5 fits within 3 % at every point (derived); damping
# 4.4 / 6.7 / 8.5 / 9.2 kN s/m at 2 Hz for 2 / 5 / 10 / 15 kN mean force, falling about as 1/f (Fig. 4).
TYRE_K_SE18X7 = P("TYRE_K_SE18X7", 7.1e6, "N/m^1.5", "derived", "fit to the measured 18x7-8 SE load-deflection curve (Oh et al. 2013, Fig. 2, read)")
TYRE_K = P("TYRE_K", TYRE_K_SE18X7 * (0.173 / 0.178) * math.sqrt(0.323 / 0.2285), "N/m^1.5", "derived",
           "front 7.00-12: the 18x7-8 fit scaled by section width (0.173 / 0.178 m) and sqrt(radius) (0.323 / 0.2285 m), a line contact's patch length; 8.2e6")
TYRE_K_REAR = P("TYRE_K_REAR", TYRE_K_SE18X7 * (0.142 / 0.178) * math.sqrt(0.2565 / 0.2285), "N/m^1.5", "derived",
                "rear 6.00-9 (0.142 m section, Trelleborg, read): same scaling; 6.0e6")
TYRE_EXP = P("TYRE_EXP", 1.5, "-", "read", "Traigo contact law exponent m = 1.5; it also fits the 18x7-8 measurement")
TYRE_MAXWELL_C = P("TYRE_MAXWELL_C", 14.5e3, "N s/m", "derived",
                   "rubber damping of a wheel at 10 kN as a Maxwell element (dashpot C in series with spring k_s) fitted to the measured 18x7-8 SE damping, 13.2 kN s/m at 1 Hz and 3.8 at 5 Hz (Oh et al. 2013 Fig. 4, read); it gives 10.0 at 2 Hz (measured 8.5) and 1.2 at 10 Hz (measured 2.2)")
TYRE_MAXWELL_K = P("TYRE_MAXWELL_K", 2.72e5, "N/m", "derived", "the Maxwell element's series spring at 10 kN (same fit; time constant C / k_s = 53 ms)")
TYRE_DAMP_EXP = P("TYRE_DAMP_EXP", 0.28, "-", "derived", "the damping grows with wheel load as (F / 10 kN)^0.28 (log fit to Fig. 4 at 1 Hz: 8.4 / 11.5 / 13.2 / 15.0 kN s/m at 2 / 5 / 10 / 15 kN)")
SUPREM_FRONT = P("SUPREM_FRONT", dict(kF1=80209.0, ka=9.16, kF2=4.996e-4, kM=11.91, kr=1.007), "N, deg, deg/N, 1/m, -",
                 "derived", "18x7-8 manufacturer-1 set (Logistics Research 10:1 Table 2, read) scaled by rated load to 7.00-12")
SUPREM_REAR = P("SUPREM_REAR", dict(kF1=56970.0, ka=9.16, kF2=7.034e-4, kM=11.91, kr=1.007), "N, deg, deg/N, 1/m, -",
                "derived", "same, scaled to a 6.00-9 steer wheel rated ~2400 kg (estimate)")
SUPREM_KD, SUPREM_KV = P("SUPREM_KD", 0.28, "s", "read", "time constant T = kd (v km/h)^-kv"), P("SUPREM_KV", 0.39, "-", "read", "18x7-8 M1")
KAPPA_STAR = P("KAPPA_STAR", 0.22, "-", "derived",
               "longitudinal slip scale of F_x = mu_x F_z tanh(slip / KAPPA_STAR), from the braking-slip curve of a super-elastic "
               "forklift tyre at 10 kN and 10 km/h on the Helmut-Schmidt-University drum (Pross & Bruns, Logistics Journal: "
               "Proceedings 2019, doi 10.2195/lj_Proc_pross_de_201912_01, Fig. 7, read; the tyre size is not stated in the text), "
               "digitised (E21): the tanh fit with its ceiling at the "
               "measured peak 0.94 gives 0.219, the initial slope 4.1 per unit slip gives 0.94 / 4.1 = 0.229 (mean 0.224); the "
               "peak lies at slip 0.36 (a tanh cannot peak: past 0.36 the law holds the peak where the tyre falls to 0.85). "
               "One tyre at one load")
SIGMA_MIN = P("SIGMA_MIN", 0.05, "m", "estimate", "relaxation length floor at walking speed (T held at its 0.5 km/h value); also the floor of the longitudinal "
              "relaxation length sigma_x (the patch half-length, tyres.py)")
MU = P("MU", {k: round(0.711 * math.exp(10.0e3 / 50917.0) * r, 3) for k, r in dict(concrete_dry=1.0, concrete_dusty=0.50 / 0.67,
                                                         concrete_wet=0.46 / 0.67, asphalt_dry=1.0, asphalt_wet=0.50 / 0.67,
                                                         steel=0.42 / 0.67, grating=0.54 / 0.67, oak=0.46 / 0.67).items()},
       "-", "derived/estimate",
       "SUPREM's base coefficient mu_B of each surface: the tyre law's lateral peak is mu_B exp(-F_z/k_F1) (Logistics Research "
       "10:1, read), its longitudinal peak MU_LONG_RATIO times that. Dry concrete is the measured drum level of the tyre whose "
       "SUPREM set the law uses (18x7-8, manufacturer 1, k_F1 50,917 N): its quasi-static lateral coefficient is 0.711 at "
       "10 kN (SUPREM Fig. 4, digitised in E21), so mu_B = 0.711 exp(10/50.9) = 0.865 (derived). "
       "The drum is coated with polyurethane and 0.4-0.7 mm quartz sand 'to reproduce the tarmac friction characteristic' "
       "(SUPREM, read), and Wong gives tyres the same peak on dry asphalt and dry concrete (0.80-0.90, read): dry sealed "
       "concrete is taken as the drum (estimate: no measurement of a super-elastic tyre on a warehouse floor was found). "
       "Cross-checks on the same drum (Pross & Bruns 2019, digitised): another SE tyre at 10 kN, 0.766 lateral and 0.94 "
       "braking, against this law's 0.757 and 0.93 for the front tyre at 10 kN. Load law: the measured coefficient falls "
       "with load on a 28 kN scale over 1-10 kN (SUPREM Fig. 4) and 19 kN over 4-10 kN (Pross Fig. 8), but flattens above "
       "10 kN: Pross's dynamic peaks give a 36 kN scale from 10 to 18 kN, and with their 10 kN dynamic-over-quasi-static "
       "ratio (1.14) their 18 kN quasi-static value is about 0.61, which SUPREM's k_F1 (0.865 exp(-18/50.9) = 0.607) "
       "reproduces; below 10 kN the law under-predicts by up to 13 % (0.85 against 0.97 at 1 kN). The laden front tyre's "
       "28 kN is 17.8 kN on an 18x7-8 under the rating scaling of SUPREM_FRONT, inside the measured range. E21b brackets "
       "the level and the load law. The other surfaces keep their ratios to dry concrete (estimates: wet 0.69, dusty 0.75, steel plate 0.63, grating 0.81, oak 0.69)")
MU_SLIDE_RATIO = P("MU_SLIDE_RATIO", 0.85 / 0.94, "-", "derived",
                   "sliding over peak friction of a super-elastic tyre: the locked-wheel 0.85 over the peak 0.94 of the same "
                   "braking-slip curve (Pross & Bruns 2019 Fig. 7, digitised; their fitted "
                   "model curve gives 0.845 / 0.92)")
MU_LONG_RATIO = P("MU_LONG_RATIO", 0.94 / 0.766, "-", "derived",
                  "longitudinal over lateral peak friction of a super-elastic tyre on the same surface: braking peak 0.94 "
                  "(Pross & Bruns 2019 Fig. 7, 10 kN, 10 km/h) over lateral peak 7.66 kN / 10 kN = 0.766 (the same paper's Fig. 8, "
                  "12 km/h, 1 deg/s; experiment marks digitised; one tyre model, the HSU "
                  "drum; their model curves give 0.92 / 0.78 = 1.18). The tyre law reads MU and SUPREM's exp(-F_z/k_F1) as the "
                  "lateral coefficient and caps the resultant on Kamm's ellipse. One load point: the longitudinal coefficient's "
                  "own load dependence is not measured (open; the law keeps the lateral exp(-F_z/k_F1))")
C_RR = P("C_RR", dict(concrete_dry=0.020, concrete_dusty=0.022, concrete_wet=0.020, asphalt_dry=0.020, asphalt_wet=0.020,
                      steel=0.020, grating=0.025, oak=0.025),
         "-", "search", "solid forklift tyre 0.021 (Mahidol prototype); 0.006-0.020 pneumatic trucks/tractors (Wong, read)")

# ------------------------------------------------------------------------------------------------ drivetrain (Yanmar 4TN88G LPG, powershift)
ENGINE_TORQUE_CURVE = P("ENGINE_TORQUE_CURVE", ((700, 120.0), (1000, 150.0), (1690, 174.0), (2200, 168.0), (2500, 160.4), (2700, 140.0)),
                        "rpm, N m", "estimate", "174 N m at 1690 and 42 kW at 2500 rpm (read); shape between and beyond estimated")
ENGINE_IDLE = P("ENGINE_IDLE", 750.0, "rpm", "estimate", "LPG industrial engine idle")
ENGINE_MAX = P("ENGINE_MAX", 2600.0, "rpm", "estimate", "governed maximum")
ENGINE_J = P("ENGINE_J", 0.30, "kg m^2", "estimate", "flywheel and converter impeller")
ENGINE_FRICTION = P("ENGINE_FRICTION", 12.0, "N m", "estimate", "motored friction at idle")
GOVERNOR_GAIN = P("GOVERNOR_GAIN", 0.02, "1/rpm", "authored", "throttle opening per rpm of speed error")
TC_STALL_RATIO = P("TC_STALL_RATIO", 2.94, "-", "fitted", "E3a fit with DRIVE_RATIO to the sheet's drawbar pull and top speed at the tyres' effective rolling radius; industrial converters 'up to 3.0' (search)")
TC_COUPLING_SR = P("TC_COUPLING_SR", 0.85, "-", "estimate", "converter coupling point: speed ratio at which the torque ratio reaches 1 (a textbook class value for industrial converters, recalled, not opened)")
TC_STALL_RPM = P("TC_STALL_RPM", 1900.0, "rpm", "estimate", "converter stall speed at full throttle")
DRIVE_RATIO = P("DRIVE_RATIO", 16.43, "-", "fitted", "E3a fit: drawbar pull 17145 N at 1.6 km/h laden and 18.7 km/h unladen top speed (read), solved together at the effective rolling radius r - delta/3 (12.7 from the drawbar pull alone gives 22.7 km/h)")
DRIVE_EFFICIENCY = P("DRIVE_EFFICIENCY", 0.88, "-", "estimate", "gears and converter mechanical")
CLUTCH_TORQUE = P("CLUTCH_TORQUE", 600.0, "N m", "open", "powershift clutch capacity at the turbine (2x the converter's stall torque)")
CLUTCH_RAMP = P("CLUTCH_RAMP", 0.35, "s", "estimate", "modulated engagement time")
PEDAL_REFERENCE = P("PEDAL_REFERENCE", dict(height=0.131, brake_play=0.010, inching_play=0.050), "m", "read",
                    "TCM FG20T3/FG25T3 (2.0-3.5 t) service manual SEF-0F7BE, Fig. 4.3 p. 64 (https://www.ages.nl/wp-content/"
                    "uploads/2016/01/servicemanuelFG25T3.pdf, opened 2026-10-17): the brake and inching pedal pads 131 mm above the floor plate (one height drawn for "
                    "both), 'Brake pedal play: 10 mm', 'Inching pedal play: 50 mm (until the master cylinder moves)'")
TCM_PEDAL_STROKE = P("TCM_PEDAL_STROKE", 0.236, "m", "measured (drawing)",
                     "TCM's left (inching) and right brake pedals' stroke to the floor plate, measured on Fig. 4.3 of "
                     "PEDAL_REFERENCE's manual at the drawing's own scale: the "
                     "131 mm dimension's extension lines (34.43 pt, 3.805 mm/pt) as the scale; the pivot at the centre of the "
                     "concentric shaft circles 103 mm below the floor plate (its lever to the master-cylinder clevis gives a "
                     "pedal ratio of 5.96 against the table's 6.3); the pad's solid arcs and the dimension's reference point on "
                     "the pad, 234 mm from the pivot; pressed forward, a pad point meets the floor plate after 57.9 deg, 236 mm "
                     "of arc at the reference point. A floor-pivot pedal: the asset's pedals hang from the dash, so TCM's "
                     "millimetres transfer as fractions of this stroke (the stroke a real master cylinder or stop allows may be "
                     "shorter: at 150 mm the fractions below would be 1.6 x)")
PEDAL_INCH_RAISE = P("PEDAL_INCH_RAISE", math.radians(8.699), "rad", "derived",
                     "the inching pedal's rest turned back about its pivot from the asset's pose until its pad's lowest "
                     "collider point stands at the brake pad's (0.7695 m): the asset draws the inching pad 25.5 mm lower than "
                     "the brake pad, where TCM draws both at the same 131 mm (PEDAL_REFERENCE). Both pad centres are then "
                     "126.6-126.8 mm above the floor plate, against TCM's 131 mm to the pad top. "
                     "Without it the inching pad meets the toe board after 33 mm, "
                     "less than TCM's 50 mm of inching play before the master cylinder moves")
PEDAL_FLOOR_STOPS = P("PEDAL_FLOOR_STOPS", dict(brake=math.radians(17.62), inch=math.radians(18.02)), "rad", "measured",
                      "each hanging pedal's full stroke from its rest: the angle at which its pad's collider first meets the "
                      "toe board plate (geometry TOEBOARD, measured from the chassis mesh), the floor being the stop "
                      "(collision_shapes, 0.01 deg steps; the drawn meshes meet it at the same angle within 0.1 deg): brake "
                      "64.0 mm of pad stroke, inching 66.7 mm from its raised rest")
INCH_PAD_ARM = P("INCH_PAD_ARM", 0.2121, "m", "measured", "inching pedal pad (the mesh's far end) from the pedal pivot, on the asset")
INCHING_SLIP_FRACTION = P("INCHING_SLIP_FRACTION",
                          (PEDAL_REFERENCE["inching_play"] / TCM_PEDAL_STROKE * PEDAL_FLOOR_STOPS["inch"]
                           - PEDAL_REFERENCE["brake_play"] / TCM_PEDAL_STROKE * PEDAL_FLOOR_STOPS["brake"])
                          / PEDAL_FLOOR_STOPS["inch"], "-", "derived",
                          "fraction of the inching pedal's travel over which it modulates the clutch, where its lug meets "
                          "the brake pedal (INCH_BRAKE_LUG). TCM's design as fractions of its 236 mm stroke "
                          "(TCM_PEDAL_STROKE): the master cylinder starts moving after 50 mm of the left pedal's travel "
                          "(0.212) and after 10 mm of the right's (0.042, PEDAL_REFERENCE); on the asset the master cylinder "
                          "starts at 0.212 of the inching travel (3.82 deg), the lug engaging the brake pedal 0.75 deg (its "
                          "transferred play) earlier: 3.07 deg, 0.170 of the travel, 11.4 mm at the pad. Transferring TCM's "
                          "absolute millimetres instead gives 0.600 and leaves no brake: the "
                          "asset's 66.7 mm stroke is a fifth of TCM's")
INCH_BRAKE_LUG = P("INCH_BRAKE_LUG", dict(k=2000.0, c=5.0), "N m/rad, N m s/rad", "authored",
                   "contact of the inching pedal's lug on the brake pedal's arm (controls.py): stiffness and damping of the "
                   "pedals' own end stops (truck.py). The lug meets the brake pedal at INCHING_SLIP_FRACTION (0.170) of the "
                   "inching travel, so past it the inching pedal works the same master cylinder as the brake pedal, as TCM's "
                   "left brake pedal does after its play (PEDAL_REFERENCE, TCM_PEDAL_STROKE), and brakes with the force on "
                   "its pad: 250 N on the inching pad gives 4.2 MPa and 1,008 N m per drum, "
                   "450 N gives 8.5 MPa and 2,138 N m (0.89 of the brake pedal's 2,413 N m at 450 N: the inching spring and "
                   "the lug's arm ratio take the rest), its pad then 1.7 mm off the toe board; the brake pedal pushed to its "
                   "floor (1,800 N) gives 39.2 MPa and 10,228 N m. The phantom's full inching press (its PD capped at 250 N) "
                   "gives 3.2 MPa and 727 N m. Komatsu's US4441596A (1984, read* through a fetch summariser of "
                   "https://patents.google.com/patent/US4441596A/en) couples the pedals the same way, the master cylinder "
                   "building pressure past an initial angle")
# Service brake (drivetrain.brake_pressure, controls.py): the pedal pushes a master cylinder through its lever ratio; the
# line pressure follows the pad force (pressure x area = pushrod force), and the pedal's travel follows from the oil the
# brakes take (BRAKE_LINE's pressure-volume curve); each front drum's torque is BRAKE_FACTOR x drum radius x wheel-cylinder
# force above the shoe return springs.
BRAKE_PAD_ARM = P("BRAKE_PAD_ARM", 0.208, "m", "measured", "brake pedal pad (the phantom foot's handle point, the mesh's far end) from the pedal pivot, on the asset")
BRAKE_HARDWARE = P("BRAKE_HARDWARE", dict(pedal_ratio=6.3, mc_bore=0.01905, wc_bore=0.02858, drum_d=0.310,
                                          play=PEDAL_REFERENCE["brake_play"] / TCM_PEDAL_STROKE * BRAKE_PAD_ARM
                                          * PEDAL_FLOOR_STOPS["brake"]),
                   "-, m, m, m, m", "read/derived",
                   "TCM FG20T3/FG25T3 service manual (PEDAL_REFERENCE's source), brake table p. 61, column FG20/FHG20/FG25/"
                   "FHG25: pedal ratio 6.3, master cylinder bore 19.05 mm, duo-servo drums, wheel cylinder bore 28.58 mm, drum "
                   "inside diameter 310 mm, linings 324 x 60 x 7 mm. play (derived): TCM's 10 mm of brake pedal play (p. 64) as "
                   "its fraction of TCM's stroke (0.042, TCM_PEDAL_STROKE) of the asset's 64.0 mm: 2.7 mm. "
                   "With BRAKE_FACTOR derived from "
                   "the standard below, the bores set the pedal's travel per bar and the factor, not the torque per newton")
BRAKE_LINE = P("BRAKE_LINE", dict(takeup=4 * math.pi * 0.02858 ** 2 / 4 * 0.425e-3, p_shoe=0.4e6, travel_at_standard=0.70, floor_force=1800.0,
                                  tube_d=3.34e-3, tube_L=2.0, fluid_nu=6e-6, fluid_rho=1050.0),
               "m^3, Pa, -, N, m, m, m^2/s, kg/m^3", "read/estimate/read*",
               "pressure-volume curve of the brake circuit: takeup, the oil that moves the shoes onto the drums (4 wheel-cylinder "
               "pistons of BRAKE_HARDWARE's 28.58 mm bore x 0.425 mm of shoe clearance, the middle of the 0.4-0.45 mm the "
               "automatic adjuster holds on 2.0-2.5 t trucks, read: TCM manual p. 68, PEDAL_REFERENCE's source: 1.09 cm^3), against the shoe return springs, which hold the line at p_shoe "
               "(0.4 MPa, 18 N at the pad, estimate) before any torque; beyond it the circuit (linings, drums, hoses, master "
               "cylinder) stiffens with pressure, V - takeup = a ln(1 + (p - p_shoe) / p_c) (BRAKE_PV), so the pedal keeps "
               "building pressure as it is pushed harder instead of bottoming. Two anchors set a and p_c: the standard's 450 N "
               "leaves the pedal at 70 % of its travel (authored reserve; no pressure-volume curve of a forklift brake was "
               "found), and the pedal reaches its floor stop only at 1,800 N, the force ISO 6292's component-strength clause "
               "has a fully depressed pedal withstand (GB/T 18849-2023 section 4.6, read* through the fetch summariser of "
               "https://www.chinesestandard.net/PDF/English.aspx/GBT18849-2002, 2026-10-14). The fluid's flow to the wheels damps the pedal: 3/16 in steel brake tube (0.028 in "
               "wall: 3.34 mm bore), 2 m to the drums, glycol brake fluid ~6 mm^2/s at 20-40 C (DOT 3/4 class), laminar "
               "(Hagen-Poiseuille), 0.37 N m s/rad at the pedal; without it the pedal rang undamped on the fluid (estimates)")


def _brake_pv():
    """(a, p_c) of the brake circuit's pressure-volume curve from BRAKE_LINE's two anchors (bisection on p_c): the pushrod
    volume at 70 % of the pedal's travel holds the 450 N pressure and the volume at the floor stop the 1,800 N pressure
    (each pressure from the pad force less the return spring at that travel, through the pedal ratio)."""
    hw, ln = BRAKE_HARDWARE, BRAKE_LINE
    a_mc = math.pi * hw["mc_bore"] ** 2 / 4
    travel = PEDAL_FLOOR_STOPS["brake"]                          # PEDAL_BRAKE's travel (defined below; the same)

    def anchor(frac, force):
        x = frac * travel
        v = a_mc * max(0.0, x * BRAKE_PAD_ARM - hw["play"]) / hw["pedal_ratio"] - ln["takeup"]
        spring = (PEDAL_BRAKE_SPRING["preload"] + PEDAL_BRAKE_SPRING["k"] * x) / BRAKE_PAD_ARM
        return v, (force - spring) * hw["pedal_ratio"] / a_mc - ln["p_shoe"]
    v1, d1 = anchor(ln["travel_at_standard"], BRAKE_STANDARD["pedal_force"])
    v2, d2 = anchor(1.0, ln["floor_force"])
    lo, hi = 1e4, 1e9                                           # p_c (Pa): the volume ratio grows with p_c (to d2/d1)
    for _ in range(200):
        pc = math.sqrt(lo * hi)
        r = math.log1p(d2 / pc) / math.log1p(d1 / pc)
        lo, hi = (lo, pc) if r > v2 / v1 else (pc, hi)
    pc = math.sqrt(lo * hi)
    return dict(a=v1 / math.log1p(d1 / pc), p_c=pc)


BRAKE_STANDARD = P("BRAKE_STANDARD", dict(drag=0.25, pedal_force=450.0, drag_1969=0.30), "-, N, -", "read/read*",
                   "service-brake performance: drawbar drag 25 % of the truck's weight plus rated capacity (read: OSHA "
                   "standard interpretation of 1990-06-21 to R. Sauger, "
                   "https://www.osha.gov/laws-regs/standardinterpretations/1990-06-21-0, opened 2026-10-16). "
                   "ANSI B56.1-1969 asked 'brake "
                   "drawbar drag for fork lift trucks be 30% of gross vehicle weight plus rated capacity load for "
                   "speeds of 10 miles per hour or more'; 'The 30% brake drawbar drag requirements worked well for the "
                   "larger trucks but made the smaller trucks unstable. Therefore, the brake drawbar drag requirement "
                   "was lowered to 25% maximum', and OSHA accepts 'a 25% brake drawbar drag' (the letter's words; it "
                   "calls the lowered figure a maximum, which the model reads as the requirement); this truck reaches "
                   "18.7 km/h = 11.6 mph. Pedal force: 'an operating force of no more than 450 N' for foot-operated "
                   "service brakes (GB/T 18849-2023, the identical adoption of ISO 6292:2020, section 4.5), read* "
                   "through a fetch summariser of the English translation's preview "
                   "https://www.chinesestandard.net/PDF/English.aspx/GBT18849-2002 (2026-10-14; the same summary gave "
                   "the pedal's strength clause, 1,200 N, and 1,800 N fully depressed, section 4.6); "
                   "neither ISO's nor GB/T's own text was accessible")
# the brake pedal's own return spring (the master cylinder's spring and the pedal's), before the line pressure acts
PEDAL_BRAKE_SPRING = dict(preload=2.0, k=(28.0 * BRAKE_PAD_ARM - 2.0) / PEDAL_FLOOR_STOPS["brake"])   # 10 N, 28 N
BRAKE_PV = P("BRAKE_PV", _brake_pv(), "m^3, Pa", "derived",
             "the brake circuit's pressure-volume curve V - takeup = a ln(1 + (p - p_shoe) / p_c), solved from BRAKE_LINE's "
             "two anchors (params._brake_pv)")
BRAKE_LADEN_AXLE_HEIGHT = P("BRAKE_LADEN_AXLE_HEIGHT", 0.300, "m", "measured",
                            "front wheel centre above the floor with the rated load (E8 start state): the lever "
                            "of the braked wheel's ground force in a drawbar-drag test")


def _brake_factor():
    """The smallest duo-servo brake factor that meets BRAKE_STANDARD: at 450 N on the pad, both front drums together hold
    a drawbar drag of 25 % of (SERVICE_WEIGHT + RATED_CAPACITY) g at the laden axle height. The pad force less the return
    spring (at the travel the pressure gives, fixed point) times the pedal ratio over the master cylinder's area is the
    line pressure."""
    hw, ln = BRAKE_HARDWARE, BRAKE_LINE
    a_mc, a_wc = math.pi * hw["mc_bore"] ** 2 / 4, math.pi * hw["wc_bore"] ** 2 / 4
    torque = BRAKE_STANDARD["drag"] * (SERVICE_WEIGHT + RATED_CAPACITY) * G * BRAKE_LADEN_AXLE_HEIGHT / 2
    x = 0.0
    for _ in range(50):
        spring = (PEDAL_BRAKE_SPRING["preload"] + PEDAL_BRAKE_SPRING["k"] * x) / BRAKE_PAD_ARM      # N at the pad
        p = (BRAKE_STANDARD["pedal_force"] - spring) * hw["pedal_ratio"] / a_mc
        v = ln["takeup"] + BRAKE_PV["a"] * math.log1p(max(p - ln["p_shoe"], 0.0) / BRAKE_PV["p_c"])
        x = (hw["play"] + hw["pedal_ratio"] * v / a_mc) / BRAKE_PAD_ARM
    return torque / (hw["drum_d"] / 2 * a_wc * (p - ln["p_shoe"]))


BRAKE_FACTOR = P("BRAKE_FACTOR", _brake_factor(), "-", "derived",
                 "drum torque over (wheel-cylinder force x drum radius) per wheel: the smallest value that meets BRAKE_STANDARD "
                 "with BRAKE_HARDWARE and BRAKE_LINE (params._brake_factor; 2.65). E8's laden 450 N stop therefore meets "
                 "the standard by construction (plateau 0.25 g): a check of this derivation, not of the world. A duo-servo "
                 "drum multiplies its actuating force several times; this truck's actual factor and its margin over the "
                 "standard are not published (open). ASME's note that 30 % made small trucks unstable grounds a value near "
                 "the minimum. Bracket (E8, laden, forks 0.15 m up and 3 deg back, 450 N in 0.15 s): "
                 "margin 1.0, plateau 0.252 g, the steer wheels up 22.5 mm and a wheel off the floor 0.57 s, the load "
                 "slides 1 mm; margin 1.2 (the 1969 edition's 30 %), 0.30 g, 122 mm, 1.89 s, 35 mm; tyre damping x 1.5 "
                 "(pitch damping up) changes 0.57 to 0.53 s")
PARK_BRAKE_TORQUE = P("PARK_BRAKE_TORQUE", 3000.0, "N m", "open", "per front wheel: holds the laden truck on 15 % with margin (6490 x 9.81 x 0.148 x 0.31 / 2 = 1460 N m)")

# ------------------------------------------------------------------------------------------------ steering
ORBITROL_DISPLACEMENT = P("ORBITROL_DISPLACEMENT", 80e-6, "m^3/rev", "search", "Hangcha load-sensing steering unit 80 "
                                                                              "ml/rev (Hangcha XC-series CPD20-35 "
                                                                              "maintenance manual, https://www.manualsl"
                                                                              "ib.com/manual/4086789/Hc-Xc-Series.html,"
                                                                              " read through a fetch summariser: "
                                                                              "search)")
STEER_TURNS_LOCK = P("STEER_TURNS_LOCK", 4.8, "rev", "read", "Crown FC 5200: 4.8 turns lock to lock")
STEER_TORQUE_SPRING = P("STEER_TORQUE_SPRING", dict(preload=0.8, full=3.0), "N m", "read",
                        "neutral setting springs of the (non-reaction) steering unit: Danfoss OSPB/OSPC/OSPD Technical "
                        "Information p. 12, 'Standard: 0.8 - 3 N.m in normal steering situations' (catalogue units have the "
                        "standard springs). Preloaded: the spool stays "
                        "centred until the wheel torque reaches 0.8 N m and the torque rises to 3.0 N m at the end of the "
                        "spool's travel (steering.py); p. 14: 'With non-reaction steering units there is no corresponding "
                        "movement of the steering wheel when the driver is not steering'. A linear "
                        "spring from zero would let the spinner knob's own weight (0.26-0.35 N m about the column) turn a released "
                        "wheel")
STEER_SPOOL_DEFLECTION = P("STEER_SPOOL_DEFLECTION", math.radians(9.0), "rad", "estimate", "orbitrol spool/sleeve play: +/-8-10 deg recalled as the class range (not opened), 9 deg its middle")
STEER_PRIORITY_FLOW = P("STEER_PRIORITY_FLOW", 14e-3 / 60, "m^3/s", "estimate", "priority flow 14 l/min: 175 rpm of an 80 cc unit (Danfoss 100-150 rpm design)")
STEER_RELIEF = P("STEER_RELIEF", 12e6, "Pa", "search", "Hangcha steering safety valve 12 MPa (Hangcha XC-series CPD20-35"
                                                      " maintenance manual, "
                                                      "https://www.manualslib.com/manual/4086789/Hc-Xc-Series.html, "
                                                      "read through a fetch summariser: "
                                                      "search)")
STEER_LEAK = P("STEER_LEAK", 3.5 / 60.0 * ORBITROL_DISPLACEMENT / STEER_RELIEF, "m^3/(s Pa)", "derived",
               "laminar clearance leakage Q = c dp of the steering unit and cylinder, from the slip a vehicle maker "
               "allows at the travel limit: 'For example, a typical specification for wheel slip would be somewhere in "
               "the range of about two to about five revolutions per minute, as a maximum, permissible amount of slip' "
               "(US 6,769,249 B2, Eaton, low-slip steering, https://patents.google.com/patent/US6769249B2/en; opened "
               "2026-10-16). Against "
               "the stops the cylinder stands at the relief pressure and the wheel turns only as fast as the unit "
               "leaks: c = n D / (60 p_relief); 3.5 rev/min (the range's middle) gives 3.9e-13; 2 and 5 rev/min give "
               "2.2e-13 and 5.6e-13 (E21 runs the 5 rev/min limit). Linear in dp: laminar flow through the spool-sleeve"
               " clearance")
STEER_MANUAL_MAX = P("STEER_MANUAL_MAX", 120.0, "N m", "read", "Danfoss: maximum torque an average operator can exert (pump off)")
STEER_KNUCKLE_STIFFNESS = P("STEER_KNUCKLE_STIFFNESS", 6.0e4, "N m/rad", "authored", "oil-column stiffness capped for the 0.5 ms explicit step (omega dt <= 0.1)")
STEER_LINKAGE = P("STEER_LINKAGE", dict(timeconst=0.002), "s", "estimate",
                  "the steering linkage that ties the right knuckle to the left (the Ackermann joint equality, truck.build): "
                  "steel links from the cylinder's rod ends to the steering arms, whose pins sit 72.9 mm behind the kingpins "
                  "(measured, E21). A 20 mm steel link 0.15 m long (estimate; no drawing) is "
                  "EA/L = 4.4e8 N/m, 2.3e6 N m/rad at that arm, ~1e6 N m/rad between the knuckles for two links in series: "
                  "0.002 deg under the rear tyres' +/-37 N m rolling-resistance moments. MuJoCo's equality time constant "
                  "0.002 s (4 ticks; MuJoCo wants at least 2) is the stiffest stable setting near that; its yield is measured "
                  "in E21 (knuckle offsets). The default 0.02 s yielded 0.036 deg under the same moments")
STEER_WHEEL_J = P("STEER_WHEEL_J", 0.05, "kg m^2", "estimate", "steering wheel with hub")

# ------------------------------------------------------------------------------------------------ hydraulics
PUMP_DISPLACEMENT = P("PUMP_DISPLACEMENT", 32.7e-6, "m^3/rev", "fitted", "to the sheet's 0.60 m/s laden lift at 2500 rpm, eta_v 0.92 (E5's lift speed checks the fit)")
PUMP_VOL_EFF = P("PUMP_VOL_EFF", 0.92, "-", "estimate", "gear pump")
PUMP_HM_EFF = P("PUMP_HM_EFF", 0.88, "-", "estimate", "gear pump hydromechanical efficiency (0.85-0.92 for external gear pumps, class value)")
PUMP_STANDBY_PRESSURE = P("PUMP_STANDBY_PRESSURE", 0.5e6, "Pa", "estimate",
                          "open-centre circulation pressure with no function selected (valve and line losses at pump flow; 0.3-1 MPa class)")
RELIEF_PRESSURE = P("RELIEF_PRESSURE", 18.5e6, "Pa", "search", "Hangcha 2.0-3.0 t main relief 18.5 MPa (Hangcha "
                                                              "XC-series CPD20-35 maintenance manual, https://www.manua"
                                                              "lslib.com/manual/4086789/Hc-Xc-Series.html, read through"
                                                              " a fetch summariser: "
                                                              "search)")
LIFT_BORE_MAIN = P("LIFT_BORE_MAIN", 0.050, "m", "derived", "two main lift cylinders, 15 MPa at rated load")
LIFT_BORE_FREE = P("LIFT_BORE_FREE", 0.073, "m", "derived", "free-lift cylinder needing 15 % less pressure than the main stage")
TILT_BORE = P("TILT_BORE", 0.063, "m", "estimate", "two double-acting tilt cylinders")
TILT_ARM = P("TILT_ARM", 0.75, "m", "estimate", "tilt cylinder lever about the mast pivot")
SHIFT_BORE = P("SHIFT_BORE", 0.040, "m", "estimate", "side-shift cylinder")
BULK_MODULUS = P("BULK_MODULUS", 0.8e9, "Pa", "search", "0.7-1.0 GPa with entrained air and hoses")
LOWERING_SPEED_MAX = P("LOWERING_SPEED_MAX", 0.58, "m/s", "fitted",
                       "the lowering regulator's fork-speed cap, set to the sheet's laden lowering speed (Hyster H2.5A row 5.3: "
                       "0.58 / 0.50 m/s laden / unladen, read); the truck's own regulator setting is not published. Fitted, not "
                       "read: the read number is the observable the cap is set to reproduce, so E5's laden lowering checks this "
                       "fit and is not a validation. The value is the sheet's and was never tuned")
LIFT_SPEED_SHEET = P("LIFT_SPEED_SHEET", (0.60, 0.65), "m/s", "read", "Hyster H2.5A lift laden/unladen (check)")
LIFT_LEAK = P("LIFT_LEAK", 4.2e-15, "m^3/(s Pa)", "fitted", "rated load at 2.0 m creeps ~35 mm in 10 min (mid of the 20-50 mm estimate band; ISO 3691-1 <=100 mm, search): fork drop rate = 2 LEAK (p_main/A_main + p_free/A_free) with the measured 14.92 / 13.01 MPa (both stages leak and the forks move 2:1 on each)")
MAST_FRICTION = P("MAST_FRICTION", 0.06, "-", "estimate",
                  "mast friction as a fraction of the force a lift column transmits: cylinder seals ~3 % plus rollers under the load moment and chain sheaves ~3 % (hydraulic-cylinder mechanical efficiency 0.95-0.97, rolling contacts; open 3-10 %)")
VALVE_CD = P("VALVE_CD", 0.62, "-", "estimate", "discharge coefficient of a sharp-edged orifice (the textbook 0.6-0.62, recalled, not opened)")
LIFT_SPOOL_METERING = P("LIFT_SPOOL_METERING", 1.0, "-", "open",
                        "exponent of the lift spool's lowering area against its stroke beyond the dead band (area = "
                        "VALVE_AREA_MAX x stroke^n): 1, a straight-sided notch; forklift "
                        "lift spools meter through notches cut in a land (search: CN111032564A, US4716929A), whose shape and "
                        "so whose curve for this truck is not published. With the rated load the regulator's flow is reached "
                        "at 36 % of the stroke at n = 1 (E5b). Bracket (E5b), run with a restrictor regulator: the smooth rated-load "
                        "cycle's steer axle empties for 0.30-0.35 / 0.09 / 0 s (n = 1 / 1.5 / 2; at 2 its minimum is 172 N), "
                        "but from rest a 2 s lever opening empties it for 0.29 s at n = 2 as at n = 1 (0.31 s). With "
                        "the load-compensated regulator (LOWERING_REGULATOR) the exponent only shapes the lever's "
                        "speed curve")
VALVE_AREA_MAX = P("VALVE_AREA_MAX", 29.4e-6, "m^2", "estimate",
                   "lowering orifice of the lift spool at full lever stroke: fitted to the sheet's "
                   "unladen 0.50 m/s through a restrictor regulator's metering orifice, kept, not refit. With the "
                   "load-compensated regulator (LOWERING_REGULATOR) the sheet's two lowering speeds fit dp_set and the "
                   "compensator's opening, and the rated load's lowering flow follows the lever's opening whatever this "
                   "area is; it is not separately anchored")
OIL_DENSITY = P("OIL_DENSITY", 870.0, "kg/m^3", "estimate", "ISO VG 32 mineral oil at 15 C, the usual 0.86-0.88 of datasheets (recalled, not opened)")
def _lowering_fit(unladen_p=2.4e6):
    """(dp_set, open_ratio) of the load-compensated lowering regulator from the sheet's two lowering speeds (Hyster
    H2.5A row 5.3, read): laden at full lever the compensator holds dp_set across the spool's full-stroke orifice and
    passes 0.58 m/s; unladen the main stage's pressure (2.4 MPa, measured in E5) is too low to reach dp_set, the compensator
    stays open and the spool and the open compensator in series pass 0.50 m/s (line losses, ~2 %, neglected: E5b checks)."""
    a_main = 2 * math.pi * LIFT_BORE_MAIN ** 2 / 4
    k_sp = VALVE_CD * VALVE_AREA_MAX * math.sqrt(2.0 / OIL_DENSITY)
    q_l, q_u = LOWERING_SPEED_SHEET[0] * a_main / 2, LOWERING_SPEED_SHEET[1] * a_main / 2
    k_c = (unladen_p / q_u ** 2 - 1.0 / k_sp ** 2) ** -0.5
    return (q_l / k_sp) ** 2, k_c / k_sp


LOWERING_SPEED_SHEET = P("LOWERING_SPEED_SHEET", (0.58, 0.50), "m/s", "read",
                         "Hyster H2.5A lowering speed laden / unladen (sheet row 5.3); the load-compensated regulator's fit")
_DP_SET, _OPEN_RATIO = _lowering_fit()
LOWERING_REGULATOR = P("LOWERING_REGULATOR", dict(sense="spool", dp_set=_DP_SET, open_ratio=_OPEN_RATIO, droop=0.0, tau=0.05),
                       "-, Pa, -, -, s", "fitted/open",
                       "pressure compensator of the lowering circuit, sense 'spool' (load-compensated lowering): it holds "
                       "the drop across the lift spool's own lowering orifice at dp_set, so the lowering flow follows the "
                       "lever whatever the load, as a forklift lowering circuit with a pressure-compensating flow-rate valve "
                       "does ('the flow rate control valve 35 functions as a pressure compensation valve to reduce changes in "
                       "the speed of the lowering operation speed caused by the load', the lever's operation amount setting "
                       "the speed: Toyota Industries EP2962981A1, 2016, read* through a fetch summariser of "
                       "https://patents.google.com/patent/EP2962981A1/en, 2026-10-16). Whether the H2.5A lowers this way "
                       "or through a restrictor regulator that caps a spool-metered flow is not published (open). dp_set "
                       "(1.70 MPa) and open_ratio (the open compensator over the spool's full-stroke orifice, 1.05) are "
                       "fitted to the sheet's laden 0.58 and unladen 0.50 m/s (params._lowering_fit; E5's and E5b's lowering "
                       "speeds check the fit). droop, the setting's rise as the compensator closes (open, 0). tau, the "
                       "compensator spool's first-order lag (open: search snippets only, 34-44 ms step responses for "
                       "pressure-compensated flow controls of other designs and 50-200 ms for cartridges; bracketed 0-0.15 s in E5b). "
                       "Sensing a separate metering orifice instead (sense 'metering', dp_set 0.7 MPa, "
                       "open_ratio 2: a restrictor that caps the flow but lets the spool meter below the cap), the "
                       "rated load reaches the cap at 36 % of the lever, accelerating to it within about 0.4 s "
                       "of a 2 s lever opening, and the smooth rated-load cycle empties the steer axle for 0.22-0.35 s at "
                       "every tau of 0-0.15 s, an ideal regulator included (E5b bracket). With the load-compensated regulator the same cycle keeps the steer axle loaded at every "
                       "tau of the bracket: lowest load while lowering 2,571 / 2,571 / 462 N at tau 0 / 0.05 / 0.15 s, and "
                       "1,990 N with the pitch damping lowered to zeta 0.017 (tyre damping x 0.7); the abrupt cycle's longest "
                       "steer-axle lift 0.57 s (0.96 s with the restrictor); a 60 % lever meters 0.24 m/s (E5b)")

# ------------------------------------------------------------------------------------------------ controls (forces and travels open)
# Return springs set so the force at the pad or knob at full travel is the class value each note states, over the handle
# arms measured on the asset (pads 0.161 / 0.208 / 0.212 m, lever knobs 0.346 m from their pivots). The service brake pedal's force
# is mostly its master cylinder's line pressure (BRAKE_LINE above; its spring here only returns it).
# Forces are estimates of the class: unboosted hydraulic brakes of 2-3 t trucks reach full braking at ~250-350 N; ISO 6292
# and ISO 3691-1 cap control forces (their tables were not accessible).
PEDAL_ACCEL = P("PEDAL_ACCEL", dict(travel=math.radians(18), k=14.2, preload=1.2), "rad, N m/rad, N m", "estimate",
                "35 N at the pad at full travel (25-40 N class), 7.5 N to start it")
PEDAL_BRAKE = P("PEDAL_BRAKE", dict(travel=PEDAL_FLOOR_STOPS["brake"], k=PEDAL_BRAKE_SPRING["k"],
                                    preload=PEDAL_BRAKE_SPRING["preload"]),
                "rad, N m/rad, N m", "measured/estimate",
                "travel: to the toe board (PEDAL_FLOOR_STOPS, measured). The pedal's return spring alone: 10 N at the pad to "
                "start it, 28 N at its floor stop (estimates); the brake's line pressure adds its reaction through the master "
                "cylinder (BRAKE_LINE), 450 N at 70 % of the travel")
PEDAL_INCH = P("PEDAL_INCH", dict(travel=PEDAL_FLOOR_STOPS["inch"], k=(60.0 * INCH_PAD_ARM - 3.0) / PEDAL_FLOOR_STOPS["inch"],
                                  preload=3.0, rest_raise=PEDAL_INCH_RAISE), "rad, N m/rad, N m, rad", "measured/estimate",
               "travel: from its raised rest (PEDAL_INCH_RAISE) to the toe board (PEDAL_FLOOR_STOPS, measured); spring 60 N at "
               "the pad at full travel, 14 N to start it (estimates); past INCHING_SLIP_FRACTION (0.170) of the travel its lug pushes the brake pedal")
LEVER = P("LEVER", dict(travel=math.radians(16), k=16.3, preload=0.6), "rad, N m/rad, N m", "estimate",
          "15 N at the knob at full travel (10-20 N class: spool centring spring through the linkage), 1.7 N to break it out of centre")
DIR_LEVER = P("DIR_LEVER", dict(travel=math.radians(14), detent=0.35), "rad, N m", "estimate", "~10 N detent at the stalk tip")
PARK_LEVER = P("PARK_LEVER", dict(travel=math.radians(30), overcentre=math.radians(18), k=16.4, preload=1.72,
                                  slack=math.radians(5)), "rad, rad, N m/rad, N m, rad", "authored",
               "over-centre hand lever, pull back to apply: ~60 N at the grip to pull it over centre, ~45 N to push it off "
               "(authored forces); k and preload solve those two forces at the grip's surface centroid, measured 0.115 m from the pin (area-weighted over the grip's mesh; the mean of its vertices, 0.129 m, is biased by the densely meshed end cap; the grip spans 0.063-0.173 m from the pin) "
               "(geometry.PARK_PIN): preload + k*overcentre = 60*0.115, preload + k*(travel-overcentre) = 45*0.115. The "
               "brake clamps as the cable stretches: none within the slack, rising in proportion to the pull to full at the "
               "toggle's centre and held there past it")
KEY = P("KEY", dict(on=math.radians(40), start=math.radians(75), k=0.4), "rad, N m/rad", "estimate", "OFF/ON/START, spring return from START (0.3 N m)")
HEAD_CONTACT_LIMIT = P("HEAD_CONTACT_LIMIT", 130.0, "N", "read",
                       "the most force the seated operator's head (the robot's chest/head link) may meet the overhead guard with: "
                       "ISO/TS 15066 Table A.2's maximum permissible force for the skull and forehead (130 N), read from its "
                       "reproduction in arXiv:2311.13814 Table 1 (the standard itself not opened). "
                       "With the guard at the sheet's height the passive H1-2's head met the rear slat at 3.1 kN in a hard stop")
SEAT_SUSPENSION = P("SEAT_SUSPENSION", dict(travel=0.080, k=11000.0, c=900.0), "m, N/m, N s/m", "read/estimate",
                    "travel 80 mm: Hyster J2.2-3.5XN technical guide, 'a new full suspension seat with 80 mm suspension travel' "
                    "(read; https://v2.zeppelin.sk/userfiles/dokumenty/J2%202-3%205XN-ENG.pdf); "
                    "spring and damper set for the operator's weight "
                    "(estimate). Damping checked against measured seats: the model's ratio 0.49 gives a "
                    "resonance peak of 1.48 at 1.65 Hz; NIOSH (Mayton et al., six seat suspensions, resonances near 1.5 Hz, an OEM "
                    "peak of 1.3 at 2.25 Hz; https://stacks.cdc.gov/view/cdc/9065/cdc_9065_DS1.pdf) "
                    "puts it at the well-damped end of measured seats")
SEAT_FOAM_SET = P("SEAT_FOAM_SET", 0.030, "m", "estimate", "seat cushion foam compression under a seated adult (25-40 mm); the rigid cushion collider sits at its loaded thickness")
SEAT_RAIL = P("SEAT_RAIL", 0.075, "m", "authored", "seat fore-aft setting: fully forward on its 150 mm rails (Toyota 8-series range, read), as the tall H1-2 needs (E13)")
SEAT_RIDE = P("SEAT_RIDE", 0.0, "m", "authored", "weight setting of the suspension set for the operator: the loaded seat rides at mid-travel")
GUARD_RAISE = P("GUARD_RAISE", 0.049, "m", "derived",
                "the overhead guard raised to the H2.5A's sheet height: h6 = 2,160 mm (Hyster H2.0-3.5A sheet, row 4.7,"
                " read, p. 4) minus the asset's guard slat tops, 2.111 m "
                "(measured) = 0.049 m; vertices above GUARD_SPLIT move up, the posts stretch. "
                "Head room at this height: at rest the seated H1-2's head top is 2.074 m, 28.5 mm under "
                "the rear slat (guard_slat_c) underside; the cushion at its "
                "bottom stop is 1.036 m (40.8 in) from that slat, over the 39 in seat-to-guard minimum (OAR "
                "437-002-0227, search). Without the lean below, in a hard stop the head reaches the slat: 3.7-3.8 mm into its "
                "collider for about 2.5 s and a 3.1 kN contact peak in the E13 stop, and a 17 N touch after a 0.16 g autopilot stop. "
                "The seated posture, not the guard, resolves this: "
                "the H1-2 starts leaning 5 deg forward (operator.START_LEAN, inside ISO 11226's 0-20 deg trunk band), "
                "which puts its head under the guard's open bay (at 0 deg the same stop struck the slat at 2,811 N), "
                "and every phantom force is reacted on the chassis (on the seat a hard push lifted it "
                "35 mm); E13 and the evaluator judge head contact against HEAD_CONTACT_LIMIT. Hyster's low-profile seat option was not used: no lowering figure is published and this "
                "cab has 34 mm under the cushion (measured)")
GUARD_SPLIT = P("GUARD_SPLIT", 1.75, "m", "measured", "chassis vertices above this height belong to the guard (posts, slats, lamps, mirror); nothing else of the chassis reaches it")
SEAT_LOAD_SHARE = P("SEAT_LOAD_SHARE", 0.75, "-", "measured", "share of the seated H1-2's weight on the cushion in the start posture (498 N of 661 N, E13 rest); sets the seat's weight dial")
SEAT_SWITCH_N = P("SEAT_SWITCH_N", 180.0, "N", "open", "operator-presence switch closing force")

# ------------------------------------------------------------------------------------------------ belt
BELT = P("BELT", dict(k=8.0e4, c=400.0, lock_g=0.45), "N/m, N s/m, g", "estimate/read",
         "k, c: polyester webbing a few % strain at belt loads (recalled, not opened: estimate). lock_g: vehicle "
         "sensitivity of the emergency-locking retractor, UN R16 6.2.5.3.1.1 (read): a type 4 retractor must have "
         "locked 'cuando la desaceleración del vehículo alcance 0,45 g' (consolidated text, OJ L 304, 20.11.2015, "
         "Spanish edition); that is the least sensitive retractor R16 allows, a bound and not a measured "
         "threshold of this truck's belt. Modelled as the pendulum sensor it is: a trip angle atan(0.45) = 24.2 deg of "
         "the apparent gravity in the seat frame, inside R16's tilt clauses (6.2.5.3.1.3: no lock at 12 deg or less; "
         "6.2.5.3.1.4: lock at 27 deg; same copy) and FMVSS 209 S4.3(j)(2) (read, 49 CFR 571.209: "
         "lock before 25 mm payout at 0.7 g; "
         "vehicle-sensitive: no lock at 15 deg or less, lock above 45 deg)")
BELT_RETRACT_N = P("BELT_RETRACT_N", 7.0, "N", "read",
                   "lap-belt retractor's retraction force: 'no será inferior a 0,7 daN' (UN R16 6.2.5.3.4; read in the "
                   "consolidated text, OJ L 304, 20.11.2015, Spanish edition); "
                   "FMVSS 209 S4.3(j)(2): 'not less than 3"
                   " N when attached to pelvic restraint' (read, 49 CFR 571.209, "
                   "https://www.law.cornell.edu/cfr/text/49/571.209). The R16 minimum")
BELT_WEBBING_LOCK_G = P("BELT_WEBBING_LOCK_G", 3.0, "g", "read",
                        "webbing sensitivity of the seat's retractor (forklift suspension seats carry a 'duo-sensitive retractable belt': Grammer MSG 20 sheet, read). UN R16 6.2.5.3.2 (consolidated text, OJ L 304 of 2015-11-20, Spanish edition): a multi-sensitivity emergency-locking retractor with strap sensitivity shall lock when the strap acceleration in the direction of unreeling is not less than 3,0 g")
BELT_SENSOR_TAU = P("BELT_SENSOR_TAU", 0.010, "s", "estimate",
                    "response time of the webbing sensor (an inertia disc on the spool), applied to the payout speed and to its rate: R16's "
                    "locking test applies its strap acceleration at 55-150 g/s (7.6.2.2, read), reaching the 3 g lock threshold in 20-55 ms")
BELT_ANCHOR_SEAT = P("BELT_ANCHOR_SEAT", ((-0.12, 0.27, 0.06), (-0.12, -0.19, 0.06)), "m", "authored",
                     "lap-belt anchors on the seat frame beside the hips (left, right), offsets from the seat body's origin")

# ------------------------------------------------------------------------------------------------ loads and contact
MU_FORK_PALLET = P("MU_FORK_PALLET", 0.35, "-", "estimate", "steel on wood mu_s 0.45 / mu_k 0.25 (patent US10988360B2, read); MuJoCo uses one mu")
MU_PALLET_FLOOR = P("MU_PALLET_FLOOR", 0.50, "-", "estimate", "Douglas fir on smooth concrete 0.58 (search)")
MU_CARTON = P("MU_CARTON", 0.30, "-", "search", "box friction 'about 0.3' (patent US5339957)")
GMA_PALLET = P("GMA_PALLET", dict(L=48 * IN, W=40 * IN, H=4.76 * IN, deck=0.630 * IN, stringer_w=1.375 * IN, mass=20.0),
               "m, kg", "read/estimate", "NWPCA PDS sheet of the 48x40 GMA pallet (read): "
                                "48.00 x 40.00 in, height 4.76 in, deck boards 0.630 "
                                "in thick (lead boards 5.500, inner 3.500 in wide), stringers 1.380 in wide; the "
                                "ledger's stringer_w 1.375 in is the 1-3/8 in nominal, 0.13 mm under the sheet. Mass: "
                                "15-22 kg (search) -> 20 kg (estimate)")

# ------------------------------------------------------------------------------------------------ facility
DOCK_HEIGHT = P("DOCK_HEIGHT", 48 * IN, "m", "read", "Blue Giant: 'The most common dock height is 48\" - 52\" (1219 to "
                                                     "1321mm)' (read; https://www.bluegiant.com/architects/planning-and"
                                                     "-design/dock-height/); 48 in for a "
                                                     "53 ft trailer is SafeRack's "
                                                     "(https://saferack.com/posts/loading-dock-heights-by-truck-type/, "
                                                     "not re-opened)")
LEVELER = P("LEVELER", dict(width=78 * IN, length=8 * FT, lip=16 * IN, capacity=35000 * LB), "m, kg", "search/authored",
            "Blue Giant leveler widths 6'0\"/6'6\"/7'0\", lengths 6-12 ft, lips 16-24 in "
            "(https://www.bluegiant.com/architects/planning-and-design/leveler-width-and-length/) and Pentalift's "
            "capacity range "
            "(https://www.pentalift.com/solutions/dock-leveler-capacity-understanding-loading-dock-capacity/), not "
            "re-opened: search; the 78 in x 8 ft deck, 16 in lip and 35,000 lb "
            "chosen within them (authored)")
TRAILER_INSIDE = P("TRAILER_INSIDE", dict(L=52.5 * FT, W=101 * IN, H=110.25 * IN, floor=1.375 * IN), "m", "read/estimate", "Waba"
                                                                                                                  "sh D"
                                                                                                                  "uraP"
                                                                                                                  "late"
                                                                                                                  " HD "
                                                                                                                  "dry "
                                                                                                                  "van "
                                                                                                                  "shee"
                                                                                                                  "t (r"
                                                                                                                  "ead)"
                                                                                                                  ": 'I"
                                                                                                                  "nter"
                                                                                                                  "ior "
                                                                                                                  "widt"
                                                                                                                  "h si"
                                                                                                                  "dewa"
                                                                                                                  "ll "
                                                                                                                  "to s"
                                                                                                                  "idew"
                                                                                                                  "all:"
                                                                                                                  " 101"
                                                                                                                  "\"', "
                                                                                                                  "'Fro"
                                                                                                                  "nt i"
                                                                                                                  "nter"
                                                                                                                  "ior "
                                                                                                                  "heig"
                                                                                                                  "ht: "
                                                                                                                  "110-"
                                                                                                                  "1/4\""
                                                                                                                  "', '"
                                                                                                                  "1-3/"
                                                                                                                  "8\" t"
                                                                                                                  "hick"
                                                                                                                  " lam"
                                                                                                                  "inat"
                                                                                                                  "ed "
                                                                                                                  "oak "
                                                                                                                  "floo"
                                                                                                                  "rboa"
                                                                                                                  "rds'"
                                                                                                                  ". "
                                                                                                                  "The "
                                                                                                                  "insi"
                                                                                                                  "de l"
                                                                                                                  "engt"
                                                                                                                  "h "
                                                                                                                  "52.5"
                                                                                                                  " ft "
                                                                                                                  "is "
                                                                                                                  "not "
                                                                                                                  "on "
                                                                                                                  "the "
                                                                                                                  "shee"
                                                                                                                  "t "
                                                                                                                  "(it "
                                                                                                                  "give"
                                                                                                                  "s "
                                                                                                                  "the "
                                                                                                                  "53 "
                                                                                                                  "ft o"
                                                                                                                  "vera"
                                                                                                                  "ll l"
                                                                                                                  "engt"
                                                                                                                  "h): "
                                                                                                                  "esti"
                                                                                                                  "mate")
TRAILER_SAG = P("TRAILER_SAG", (3.5 * IN, 5.0 * IN), "m", "read",
                "Hendrickson L816 table, total vertical change at the rear sill while a forklift loads an air-ride trailer WITHOUT a dump valve or deck-height lock: trailer drop 2.5-3.5 in + tyre deflection 1.0-1.5 in. The simulated trailer is that type. (A dump-valve trailer instead lowers 2.5-3.5 in before loading and then drops only by its tyres, 1.0-1.5 in; a spring trailer 2.75-3.25 in in all)")
STORED_LOAD_MASS = P("STORED_LOAD_MASS", (500.0, 1000.0), "kg", "authored",
                     "wrapped unit loads stored in the racks, pallet excluded: drawn uniformly per location; the put-away task's carried "
                     "loads are 600-1,000 kg, light SKUs go lower")
RACK = P("RACK", dict(frame_depth=42 * IN, beam=96 * IN, levels=(0.0, 1.60, 3.20), upright=3 * IN, beam_h=5 * IN, beam_w=0.05,
                      frame_gap=12 * IN), "m",
         "read/authored", "Steel King rack design guide (read; "
                          "https://www.cisco-eagle.com/uploads/SK-General-Rack/RackDesignGuide.pdf): "
                          "upright frame depth = pallet depth"
                          " - 6 in (42 in for 48 in pallets), beam = 2 x load width + 12 in, rounded up to the next "
                          "beam length (96 in); 3 in uprights (not located in that guide: estimate); 5 in step beams "
                          "(estimate within Steel King's range); beam heights authored. A double row's frames stand on "
                          "12 in row spacers so that the 48 in pallets' 3 in rear overhangs leave a nominal 6 in "
                          "longitudinal flue between the loads (the flue NFPA 13 asks of double-row racks; recalled, "
                          "not opened: estimate)")
RACK_LIFT_OFF = P("RACK_LIFT_OFF", 6 * IN, "m", "read",
                  "vertical lift-off clearance between the top of a stored unit load and the underside of the beam above "
                  "(facility.STORED_LOAD_H sizes the stored loads to it). Interlake Mecalux, Pallet Rack Selection Guide p. 2, "
                  "Upright Frame Selection: upright height = 'Overall height of load + (4\" - 6\") vertical loading clearance + "
                  "height of beam x total amount of levels' (PDF opened, res.cloudinary.com/stauffer-glove-safety/image/upload/"
                  "Documents/Spec_Sheets/Interlake Mecalux - Selective Pallet Rack Guide.pdf); Cisco-Eagle, 'Pallet Rack "
                  "Clearances: Lateral, Vertical and Building Structures' (S. Stone, 2021-05-11): 'the minimum space between the "
                  "top of the tallest load and the bottom of the next beam is 6\"' (read*, www.cisco-eagle.com/blog/2021/05/11/"
                  "pallet-rack-clearances-lateral-vertical-and-building-structures). 6 in: the top of the maker's range and the "
                  "distributor's minimum")
AISLE_RACK = P("AISLE_RACK", 4.50, "m", "derived/authored",
               "working aisle between the stored loads' aisle faces (the obstacles the truck turns between: the beam faces stand 3 in further back, facility.STORED_OVERHANG), VDI 2198 Ast = Wa + x + l6 + a measured as the Hyster H2.0-3.5A technical guide draws it (p. 3: from the stack face to the obstacle behind the counterweight, a/2 at the counterweight) for this truck with a 48 in GMA pallet carried lengthways: Wa 2.365 m (measured in E4: the chassis outline's farthest point from the turning centre; the sheet's 2.263-2.357 m) + x 0.545 (LOAD_DISTANCE) + l6 1.219 + a 0.200 (derived from the same guide: Ast - Wa - x - l6 = 200 mm for the H2.0A, H2.5A Primematch and H2.5A Corematch, pp. 4, 5, 9) = 4.329 m. The aisle is 4.50 m (authored, 0.17 m over the formula, the layout every start state and demo was built on), so the counterweight keeps 271 mm. The sheet's 1000x1200 crossways figure, 4.0 m, lets the truck strike the rack turning")
POSTED_SPEED = P("POSTED_SPEED", 8.0 / 3.6, "m/s", "read", "posted limits: 'Always observe posted speed limits (usually"
                                                           " 5 mph) at your workplace' (OSHAcademy course 622, read; ht"
                                                           "tps://www.oshacademy.com/courses/training/622-forklift-comp"
                                                           "etent-person/622-3-3.php); 8 km/h = "
                                                           "4.97 mph")
TRAILER = P("TRAILER", dict(mass=6800.0, tandem_from_rear=2.6, gear_from_rear=12.6, wall=0.025, subframe=0.22,
                            track=1.83, outside_W=102 * IN), "kg, m", "estimate",
             "empty 53 ft dry van ~15,000 lb (search); slider fully back; landing gear ~3.4 m behind the nose; 72 in tandem track")
TRAILER_K = P("TRAILER_K", 4.11e5, "N/m", "fitted",
              "per side of the tandem: the laden truck's 5,698 kg front axle at the sill sags it 4.25 in (mid of the read 3.5-5.0 in band, TRAILER_SAG) "
              "with the landing gear 12.6 m and the tandem 2.6 m from the sill: k = (12.6/10.0)^2 W / (2 x 0.108 m)")
TRAILER_ARM_SLOPE = P("TRAILER_ARM_SLOPE", 0.15, "-", "open",
                      "fore-aft travel of a tandem axle relative to the trailer body per metre of air-spring compression beyond its "
                      "static set (trailing-arm air suspension: the arm rotates about its front hanger, the axle swings rearward; "
                      "slope = tan of the arm's angle at ride height, 8.5 deg here; no geometry found for this trailer). The "
                      "mechanism is the documented cause of dock walk (Heavy Duty Trucking, 'Preventing dock walk': the air "
                      "bags compress, the arms rotate, the braked or chocked wheels push the body forward); order-of-magnitude "
                      "check against Reyco Granning's demonstration in the same article (4 in after three forklift trips with "
                      "heavy paper rolls) in E11 walk")
TRAILER_C = P("TRAILER_C", 2.0e4, "N s/m", "estimate", "per side: air spring and shock absorbers, ~0.3 of critical on the rear half mass")
GEAR_K = P("GEAR_K", 4.0e6, "N/m", "estimate", "landing-gear leg and sand shoe, per leg")
TRAILER_STAND = P("TRAILER_STAND", dict(k=4.0e6, from_nose=0.55, y=0.85), "N/m, m", "estimate",
                  "two screw stabilizing jacks under the uncoupled trailer's nose, cranked up to touch its underside with no preload "
                  "(OSHA 1910.178(m)(10); Bluff MobilePRO manual p. 3: do not use the ramp unless the support jacks under the trailer are "
                  "extended). One-sided supports; stiffness as a landing-gear leg (GEAR_K); position authored")
MU_TYRE_SPRINGBRAKE = P("MU_TYRE_SPRINGBRAKE", 0.70, "-", "estimate", "braked truck-trailer tyres on dry concrete/asphalt (Wong 0.513-0.825 range, read)")
MU_SAND_SHOE = P("MU_SAND_SHOE", 0.40, "-", "estimate", "steel sand shoe on concrete")
STICK_K = P("STICK_K", 1.5e6, "N/m", "estimate", "tangential compliance of a braked tandem tyre group / landing-gear leg (presliding)")
CHOCK = P("CHOCK", dict(k=2.0e6, capacity=60e3), "N/m, N", "estimate",
          "steel wheel chock: stops the trailer moving away from the dock until the tyres climb it (~60 kN per side, estimate)")
RESTRAINT = P("RESTRAINT", dict(k=5.0e6, capacity=142e3), "N/m, N", "search",
              "rotating-hook vehicle restraint, 32,000 lb holding force (Rite-Hite class, search)")
LEVELER_MASS = P("LEVELER_MASS", dict(deck=650.0, lip=90.0), "kg", "estimate", "6 ft 6 in x 8 ft hydraulic leveler deck and 16 in lip plates")
BALE = P("BALE", dict(size=(60 * IN, 30 * IN, 48 * IN), mass=(450.0, 590.0), mu=0.40), "m, kg, -", "search", "OCC bale 60x30x48 in, 1,000-1,300 lb (search); friction estimate")
DUNNAGE = P("DUNNAGE", dict(size=(4 * IN, 4 * IN, 48 * IN), mass=4.0), "m, kg", "authored", "two 4x4 in timber strips under each bale")
YARD_RAMP = P("YARD_RAMP", dict(incline=30 * FT, level=6 * FT, width=84 * IN, rise=48 * IN), "m", "read/authored", "Bluff "
                                                                                                          "MobilePRO+ "
                                                                                                          "16,000 lb: "
                                                                                                          "'WIDTH 84\" "
                                                                                                          "(78\" "
                                                                                                          "useable) "
                                                                                                          "LENGTH 36' "
                                                                                                          "(30' "
                                                                                                          "Incline; 6' "
                                                                                                          "Level-Off)' "
                                                                                                          "(dealer "
                                                                                                          "listing, "
                                                                                                          "read); "
                                                                                                          "rise 48 in "
                                                                                                          "authored "
                                                                                                          "inside the "
                                                                                                          "listed 36-60"
                                                                                                          " in service "
                                                                                                          "range")
YARD_RAMP_LIP = P("YARD_RAMP_LIP", dict(length=16 * IN, travel=3 * IN, pieces=4, piece_gap=0.006, plate=5 / 8 * IN, hinge_gap=0.010,
                                        hinge_friction=5.0), "m, -, N m", "read/estimate",
                  "Bluff MobilePRO: multi-piece hinged lip, 16 in plates (dealer listing), 'adjustment up to 3 in up and 3 in down' "
                  "(manual p. 7), lowered onto the trailer floor and reaching at least 6 in into it (manual p. 19), folded upright for "
                  "storage (p. 21). Read: length, travel. Estimates: four pieces (the plan drawing's hinge groups), 5/8 in plate as the "
                  "dock leveler's lip, 6 mm between pieces, the hinge pin 10 mm beyond the level-off's edge, 5 N m pin friction")
STEEL_DENSITY = P("STEEL_DENSITY", 7850.0, "kg/m^3", "estimate", "the customary structural-steel density (EN 1991-1-1 Annex A gives 77-78.5 kN/m^3; recalled, not opened)")


def ledger():
    return [dict(symbol=k, **v) for k, v in LEDGER.items()]


# ------------------------------------------------------------------------------------------------ module constants
# (constants that shape the physics; each module reads NAME = P.NAME)
MU_TRUCK = P("MU_TRUCK", 0.6, "-", "estimate",
             "native contact friction of the truck's own parts (painted steel, the vinyl cushion and backrest, pedal pads, levers); "
             "MuJoCo uses the larger mu of a pair, so the robot's finger and sole pads (higher) set their own grip")
MU_ROBOT_SHELL = P("MU_ROBOT_SHELL", 1.0, "-", "open",
                   "contact friction of the robot's links other than the hands (shell, soles): Newton's default shape mu, never set "
                   "for the robot; no measurement of the robots' shell or sole materials on painted steel, the floor plate or the seat cover "
                   "was found. MuJoCo takes the larger mu of a pair, so it, not MU_TRUCK, sets the robot's grip on the cushion and backrest "
                   "(E13: the thighs' tangential/normal force ratio on the cushion reached 0.93-0.94 in a stop)")
MU_ROBOT_PAD = P("MU_ROBOT_PAD", dict(slide=1.0, torsion=0.005), "-, m", "estimate",
                 "finger and palm pads of the hands (Inspire RH56 silicone tips, Dex3-1 tactile pads): sliding mu and the torsional "
                 "(spin) friction length that lets a pad hold the steering knob")
MU_STRUCTURE = P("MU_STRUCTURE", 0.50, "-", "estimate",
                 "native (non-tyre) contact friction of floors, ramps, walls, trailer and leveler plates (racks: MU_RACK_STEEL): at "
                 "the wood-on-concrete value (MU_PALLET_FLOOR) so a pallet slides on concrete at its own mu (CONTACT_PRIORITY)")
MU_COIL = P("MU_COIL", 0.40, "-", "estimate",
            "the steel coil's surface against whatever it touches outside its own pallet (the load backrest, other loads; the "
            "forks, a rack, a floor if it leaves the pallet), its own mu at priority 3 (CONTACT_PRIORITY); no source. The coil "
            "and its timber chocks are one rigid body with the pallet (loads.add_pallet), so there is no coil-on-chock "
            "contact")
MU_CONVEYOR_ROLLER = P("MU_CONVEYOR_ROLLER", 0.45, "-", "estimate",
                       "steel roller under a pallet's bottom boards (steel on wood), over the pallet at priority 2 "
                       "(CONTACT_PRIORITY): the static coefficient of MU_FORK_PALLET's steel-on-wood source (US10988360B2 mu_s "
                       "0.45, read), a driven roller carrying the boards mostly without slip")
MU_RACK_STEEL = P("MU_RACK_STEEL", MU_FORK_PALLET, "-", "estimate",
                  "pallet bottom boards on the rack's step beams, and anything against its uprights: steel on wood, the pair and "
                  "source of MU_FORK_PALLET (US10988360B2 mu_s 0.45 / mu_k 0.25, read; MuJoCo uses one mu), over pallets at "
                  "priority 2 (CONTACT_PRIORITY)")
CONTACT_PRIORITY = P("CONTACT_PRIORITY", dict(carton=3, wrapped=3, bale=3, coil=3, fork=2, rack=2, roller=2, wood=1, dunnage=1,
                                              other=0), "-", "authored",
                     "MuJoCo contact priority per material. MuJoCo 3.11 gives a contact the friction of its higher-priority "
                     "geom and the larger of the two at equal priority (measured). Ranked so that "
                     "each pair the ledger names takes its entry: cartons and wrapped loads (MU_CARTON), bales (BALE mu) and "
                     "the coil (MU_COIL) their own mu against anything; the steel that carries pallets, fork blades and shanks "
                     "(MU_FORK_PALLET), rack beams and uprights (MU_RACK_STEEL) and conveyor rollers (MU_CONVEYOR_ROLLER), its "
                     "steel-on-wood mu over pallets and dunnage; pallet and dunnage wood MU_PALLET_FLOOR over floors and decks "
                     "(MU_STRUCTURE). A pair the ledger does not name takes the mu of its higher-ranked side, or the larger "
                     "at equal rank: truck, robot and structures (rank 0) against each other the larger (robot on the seat "
                     "MU_ROBOT_SHELL); a tyre or the truck against a rack or roller the rack's or roller's mu; pallet wood "
                     "against the truck's body or the conveyor guides MU_PALLET_FLOOR (steel on wood has no single-geom "
                     "rank that leaves the robot's grip on the truck alone)")
CARTON_MASS = P("CARTON_MASS", (12.0, 20.0), "kg", "authored", "a loose-stack carton (0.40 x 0.50 x 0.25 m), drawn uniformly")
WHEEL_ARMATURE = P("WHEEL_ARMATURE", dict(front=4.0, axle=1.0), "kg m^2", "estimate",
                   "front wheel joint: drivetrain inertia reflected to each drive wheel (differential, half-shafts, planetary hubs, brake "
                   "drum; turbine side carried by drivetrain.TURBINE_J), about the wheel's own 4.8 kg m^2; steer-axle trunnion: beam and "
                   "cylinder about the centre pin")
SPOOL_DEADBAND = P("SPOOL_DEADBAND", 0.12, "-", "estimate", "fraction of lever travel before a spool opens (open-centre valve overlap)")
SEAT_SWITCH_FILTER = P("SEAT_SWITCH_FILTER", 0.05, "s", "estimate", "seat-switch debounce")
LEVER_DAMPING = P("LEVER_DAMPING", 1.5, "N m s/rad", "estimate", "hydraulic lever: spool drag in oil referred to the lever")
DIR_DAMPING = P("DIR_DAMPING", 0.2, "N m s/rad", "estimate", "direction-lever detent friction, about 0.7 of critical")
KEY_DAMPING = P("KEY_DAMPING", 0.004, "N m s/rad", "estimate", "key cylinder, about 0.7 of critical")
LEVELER_LIP_PLATE = P("LEVELER_LIP_PLATE", dict(thickness=5 / 8 * IN, extended_stop_deg=-3.0), "m, deg", "estimate",
                      "dock leveler lip: 5/8 in plate; its extended stop 3 deg below in-line with the deck")
TURBINE_J = P("TURBINE_J", 0.12, "kg m^2", "estimate", "converter turbine and transmission input shafts")
STARTER = P("STARTER", dict(torque=60.0, fire_rpm=400.0, stall_rpm=350.0), "N m, rpm", "estimate",
            "1.4 kW starter through the ring gear; the engine fires at 400 rpm and a running engine stalls below 350 rpm")
SERVICE_BRAKE_STICK_K = P("SERVICE_BRAKE_STICK_K", 8.0e5, "N m/rad", "authored",
                          "stick spring of the brake element while it holds (stiff against the tyre's own compliance)")
WHEEL_J_MIN = P("WHEEL_J_MIN", 9.0, "kg m^2", "derived", "lightest wheel-side inertia: a drive wheel (4.8) plus its armature (WHEEL_ARMATURE front 4.0), rounded up")
HYD_DEAD_LEN = P("HYD_DEAD_LEN", 0.15, "m", "estimate", "oil column of hoses and dead volume per circuit, as cylinder stroke")
HYD_LINES = P("HYD_LINES", dict(main=dict(d=0.019, L=3.0, K=4.0), free=dict(d=0.019, L=5.0, K=5.0)), "m, m, -", "estimate",
                "lift lines from each lift cylinder to the valve: bore d, length L and the fittings' minor-loss sum K (elbows, tees, the cylinder port, the load-check valve). Bore 3/4 in: the 75 l/min pump flow (E5) at 4.4 m/s, inside the 2-5 m/s design band of pressure lines (a 1/2 in line would run at 9.9 m/s); main lines 3 m from the mast foot under the floor plate to the valve, the free-lift line 5 m (up the outer upright over its hose sheave); K 0.5-1 per fitting. The drop is Darcy friction plus K rho v^2/2 (hydraulics._line_table): 0.07 MPa in a main line at the 68 l/min of the laden lowering")
OIL_VISCOSITY = P("OIL_VISCOSITY", 21.5e-6, "m^2/s", "derived",
                  "ISO VG 32 at a 50 C working temperature: 32 mm^2/s at 40 C (the grade's definition, ISO 3448) and 5.4 mm^2/s at 100 C (a VI-100 mineral oil, literature), interpolated by the ASTM D341 Walther law; the working temperature (40-60 C in a forklift tank) is an estimate")
HYD_ZETA = P("HYD_ZETA", 0.15, "-", "estimate", "tilt and side-shift oil-column damping ratio (line losses)")
PITCH_DAMPING_REF = P("PITCH_DAMPING_REF", dict(zeta=0.031, f_hz=(2.96, 3.51), payload_t=(1.23, 0.90)), "-, Hz, t", "read",
                      "the only measured forklift pitch damping found: N. Haritos, T. Nguyen, E. F. Gad, J. L. Wilson, 'Floor vibrations induced "
                      "by forklifts', AEES 2013 conference (http://aees.org.au/wp-content/uploads/2015/06/55-Haritos-Nick-"
                      "Floor.pdf, opened 2026-10-14): a 2.0 t Viper reach "
                      "truck's pitch mode excited by stopping the forks' travel, 'fo = 3.51 Hz, zeta = 3.1%' with 0.90 t and "
                      "'fo = 2.96 Hz, zeta = 3.1%' with 1.23 t (Fig. 4). A different truck type and frequency; a Toyota SAS 25 "
                      "counterbalance truck in the same paper rang at 3.69 and 2.35 Hz with 0.60 and 1.44 t, its damping not "
                      "given. Not a model parameter: the truck's pitch damping comes out of its tyres (TYRE_MAXWELL_C, "
                      "measured), the reference the probes compare it with (E5b: 0.83 Hz, zeta 0.024-0.031 by window at 4.44 m with the rated "
                      "load; E5b and E8 bracket it by scaling the tyre damping: x 0.7 / 1.0 / 1.5 give zeta 0.017 / 0.024 / "
                      "0.037 at height, around this 0.031)")
LIFT_CUSHION = P("LIFT_CUSHION", dict(length=0.050, min_open=0.01), "m, -", "estimate",
                 "end-of-stroke cushions of the lift cylinders: the last 50 mm of stroke throttled to 1 % of the line's opening "
                 "(cubic taper)")
CONVEYOR = P("CONVEYOR", dict(roller_d=2.5 * IN, pitch=0.10, roller_mass=6.5, speed=0.20, roller_torque=25.0), "m, kg, m/s, N m",
             "estimate", "pallet roller conveyor: 2.5 in rollers at 100 mm pitch, 6.5 kg each, 0.20 m/s surface speed (pallet conveyors "
             "30-60 fpm), 25 N m per roller from the chain drive")
STEER_STOP = P("STEER_STOP", dict(k=3000.0, c=5.0), "N m/rad, N m s/rad", "estimate",
               "steering wheel against the metering unit's stop beyond the spool travel (rigid feel)")
TYRE_BRISTLE = P("TYRE_BRISTLE", dict(zeta=0.7, fade_v=0.5, turn_slip_angle=0.05), "-, m/s, rad", "authored/estimate",
                 "bristle damping ratio (critically damped stick, authored); the damper fades above 0.5 m/s "
                 "(authored); torsional deflection of the patch at its pivoting torque 0.05 rad (estimate)")
FACILITY = P("FACILITY", dict(wall_h=9.0, upright_h=16 * FT, dock_face_y=20.0, door_pitch=12 * FT, door=(9 * FT, 10 * FT), dock_bumper=4.5 * IN,
                              trailer_rear_frame=0.10, ramp_bumper=4 * IN, dressing_reach=6.0), "m", "authored/read/search",
             "building eave 9 m (authored); 16 ft rack frames (rack catalogues, not located: search); 9 x 10"
             " ft dock doors on 12 ft centres (authored); 4.5 in moulded dock bumpers (catalogue, not located: "
             "search); the trailer's rear frame 0.10 m behind its floor (estimate); the yard ramp's 4 in bumpers "
             "(read: Bluff MobilePRO manual p. 7, '4\"x10\" loading dock bumpers'); "
             "colliders under every drawn prop below 6 m, the "
             "mast's reach at full lift (derived)")


# ------------------------------------------------------------------------------------------------ literals found inside functions
# (numbers that shape the world from inside function bodies. Each module reads the entry. These entries record the
# values, they do not tune them.)
KEY_DETENT = P("KEY_DETENT", 0.03, "N m", "estimate",
               "peak torque of the key cylinder's OFF/ON detent, a sine over 0..ON (controls.Controls.torques; no source)")
KEY_SWITCH = P("KEY_SWITCH", dict(on=0.5, start=0.6), "-", "authored",
               "ignition switch contacts: ON closes past half the ON angle, START past 60 % of the way from ON to START")
HORN_SPRING = P("HORN_SPRING", dict(preload=3.0, k=1500.0, c=2.0, switch=0.002), "N, N/m, N s/m, m", "estimate",
                "horn button return spring: 3 N preload, 1.5 N/mm, 2 N s/m damping; the switch closes at 2 mm of its travel "
                "(no source; a membrane horn switch class)")
PARK_LEVER_DAMPING = P("PARK_LEVER_DAMPING", 0.5, "N m s/rad", "estimate",
                       "parking-brake lever: cable and pivot drag as viscous damping (no source)")
ENGINE_FRICTION_SHAPE = P("ENGINE_FRICTION_SHAPE", dict(at_rest=0.6, per_idle=0.4, w_reg=5.0), "-, -, rad/s", "estimate",
                          "motored friction T = ENGINE_FRICTION (at_rest + per_idle w / w_idle) tanh(w / w_reg): 60 % of the "
                          "idle value at rest, rising linearly with speed (estimate); w_reg regularises the sign at standstill")
GOVERNOR_BIAS = P("GOVERNOR_BIAS", 0.15, "-", "authored", "throttle opening of the speed governor at zero speed error")
CLUTCH_RELEASE = P("CLUTCH_RELEASE", 0.1, "s", "estimate", "release time of the engaged powershift pack on a direction change or neutral")
BRAKE_STICK_ZETA = P("BRAKE_STICK_ZETA", 0.7, "-", "authored",
                     "damping ratio of the brake element's stick spring (SERVICE_BRAKE_STICK_K) on the lightest wheel inertia")
HYD_COLUMN_LEN = P("HYD_COLUMN_LEN", dict(tilt=0.06, shift=0.10), "m", "estimate",
                   "oil column of the tilt and side-shift cylinders beyond the dead length (HYD_DEAD_LEN), as stroke: about "
                   "half of each cylinder's stroke")
PORT_RELIEF_RATIO = P("PORT_RELIEF_RATIO", 1.05, "-", "estimate",
                      "tilt and side-shift port reliefs set 5 % above the main relief (RELIEF_PRESSURE)")
STEER_CYL_DAMPING = P("STEER_CYL_DAMPING", dict(zeta=0.2, inertia=30.0), "-, kg m^2", "estimate",
                      "steer cylinder line-loss damping: 0.2 of critical on a knuckle-reflected inertia of 30 kg m^2 (both "
                      "estimates)")
TYRE_PIVOT_RADIUS = P("TYRE_PIVOT_RADIUS", 0.38, "-", "estimate",
                      "pivoting (spin) torque of a contact patch M = mu F r_t with r_t = 0.38 hypot(a, w / 2), a the patch "
                      "half-length, w the width: the factor has no recorded derivation (a uniform-pressure square patch gives "
                      "0.54, a long thin one 0.50)")
TYRE_DAMP_REF_LOAD = P("TYRE_DAMP_REF_LOAD", 1.0e4, "N", "derived",
                       "reference wheel load of the Maxwell damping law (TYRE_MAXWELL_C / K fitted at 10 kN; TYRE_DAMP_EXP)")
LEVELER_HINGE_FRICTION = P("LEVELER_HINGE_FRICTION", dict(deck=200.0, lip=20.0), "N m", "estimate",
                           "Coulomb friction of the leveler's rear deck hinge and its lip hinge pin (no source)")
LEVELER_TRAVEL = P("LEVELER_TRAVEL", dict(deck_deg=10.0, lip_pendant_deg=-95.0, lip_stowed_up_deg=15.0), "deg", "estimate",
                   "deployed deck +/-10 deg about its rear hinge (no source; a 12 in working range above and below the dock, recalled "
                   "as typical and not opened, would be 7.2 deg over the 8 ft deck); the lip hangs to -95 deg (pendant); a "
                   "stowed lip may lift 15 deg")
GEAR_C = P("GEAR_C", 2.0e4, "N s/m", "estimate", "landing-gear leg and stabilizing jack damping, per leg (as TRAILER_C)")
STICK_DAMPING = P("STICK_DAMPING", dict(zeta=0.7, mass=2000.0), "-, kg", "authored",
                  "presliding stick of the trailer's supports damped at 0.7 of critical on a 2,000 kg share of the trailer")
GEAR_TRACK_RATIO = P("GEAR_TRACK_RATIO", 0.9, "-", "estimate", "landing-gear legs 0.9 of the tandem track apart (1.65 m)")
TRAILER_BODY = P("TRAILER_BODY", dict(cg_x_fraction=0.52, cg_z=0.9, inertia_height=2.6), "-, m, m", "estimate",
                 "empty van trailer: CG 52 % of the length from the rear sill, 0.9 m above the floor plane; box inertia over "
                 "a 2.6 m height")
CONTROL_MASSES = P("CONTROL_MASSES", dict(steering_wheel=2.5, knob=0.10, pedal_inch=0.9, pedal_brake=1.0, pedal_accel=0.7,
                                          lever=0.35, dir_lever=0.15, park_lever=0.31, key=0.02, horn=0.05), "kg", "estimate",
                   "masses of the truck's control bodies (truck.component_masses; the horn button truck.build), taken from "
                   "the service weight's chassis remainder. park_lever: the handle only: steel flat bar 23.5 cm3 "
                   "(its hull) x 7,850 kg/m3 = 0.18 kg + moulded grip 107 cm3 (hull) x ~1,200 kg/m3 = 0.13 kg (estimate); the "
                   "hidden linkage's inertia is the joint's armature. The others: estimates, no source")
TRUCK_INERTIA = P("TRUCK_INERTIA", dict(chassis_box=(1.9, 1.10, 0.95), sideshift_box=(0.9, 1.1, 1.0), fork_com=(0.80, 0.0, 0.20),
                                        plate_com=(0.50, 0.0, 0.60), tyre_inner_ratio=0.55, horn_inertia=1e-5),
                  "m, m, m, m, -, kg m^2", "estimate",
                  "inertia build-up: the chassis remainder and the side-shifter as uniform boxes of these sizes; fork and plate "
                  "mass centres in the truck frame; the solid tyre a ring of inner radius 0.55 r; the horn button's inertia")
JOINT_ARMATURE = P("JOINT_ARMATURE", dict(knuckle=0.5, wheel_rear=0.3, mast_tilt=20.0, mast_stage=5.0, sideshift=2.0,
                                          steering_wheel_fraction=0.2, knob=2e-5, horn=0.002, pedal=0.01, lever=0.004,
                                          dir_lever=0.001, park_lever=0.01, key=1e-5, seat=1.0),
                   "kg m^2 (revolute), kg (prismatic)", "estimate",
                   "MuJoCo armature (reflected inertia) of the truck's joints other than the wheels (WHEEL_ARMATURE): the "
                   "cylinders' moving oil and rods on the mast joints, bearings and linkages on the controls; steering wheel: "
                   "this fraction of STEER_WHEEL_J. No source; the light controls' values also keep them stable at 0.5 ms")
CONTROL_JOINT_FRICTION = P("CONTROL_JOINT_FRICTION", dict(steering_wheel=0.05, knob=0.005), "N m", "estimate",
                           "Coulomb friction of the steering column bearing and the spinner knob's bearing")
CONVEYOR_ROLLER_JOINT = P("CONVEYOR_ROLLER_JOINT", dict(spin_factor=0.8, armature=0.01, bearing_friction=0.05, speed_gain=2.0),
                          "-, kg m^2, N m, N m s/rad", "estimate",
                          "a roller's spin inertia m r^2 x 0.8 (thick-walled tube), its joint armature and bearing friction, "
                          "and the drive's speed gain toward the surface speed (capped at CONVEYOR roller_torque)")
MAST_TILT_GYRATION = P("MAST_TILT_GYRATION", 1.2, "m", "estimate",
                       "radius of gyration of the mast and carriage about the tilt pivot, for the tilt column's effective "
                       "inertia (World: hydraulic damping)")
JOINT_LIMIT_SOLIMP = P("JOINT_LIMIT_SOLIMP", (0.99, 0.999, 1e-3, 0.5, 2.0), "-", "repo",
                       "joint-limit impedance with JOINT_LIMIT_TIMECONST (barcode_scanning: the default 0.9-0.95 "
                       "lets a stop overtravel ~1 mm under a gripper's squeeze)")
TYRE_FAULT_LOAD = P("TYRE_FAULT_LOAD", 2.0e5, "N", "authored",
                    "fault when one tyre carries more than this: a tyre deep inside a surface (7 x the laden front wheel's "
                    "static 28 kN)")
OPERATOR_MASS = P("OPERATOR_MASS", 67.0, "kg", "authored",
                  "the operator's mass where no robot sits: the seat's weight setting (Controls) and the ballast strapped to "
                  "the seat in robot-less probes (Scene.add_truck). The H1-2's mass rounded: 67.4 kg summed from the built "
                  "model's bodies (measured)")
ROBOT_LINKAGE_TICKS = P("ROBOT_LINKAGE_TICKS", 8, "ticks", "authored",
                        "time constant of the hand linkages' joint equalities, in ticks (4 ms at 0.5 ms; E17)")
COIL_LOAD = P("COIL_LOAD", dict(length=0.9, diameter=0.62), "m", "authored", "the feed-conveyor task's steel coil on its pallet")
CARTON_SIZE = P("CARTON_SIZE", (0.40, 0.50, 0.25), "m", "authored", "a loose-stack carton (CARTON_MASS)")
GMA_BOARDS = P("GMA_BOARDS", dict(lead=5.5 * IN, inner=3.5 * IN), "m", "read",
               "GMA pallet deck boards: 5.5 in lead boards, 3.5 in inner boards (NWPCA GMA sheet, GMA_PALLET)")
DRIVE_KIT_MOUNTS = P("DRIVE_KIT_MOUNTS", dict(frame=(-0.62, 0.02, 0.70), pedal_unit_offset=(0.14, 0.0, 0.06),
                                              lever_bank=(-0.40, -0.32, 1.30), camera=(-0.86, 0.04, 1.93),
                                              camera_pitch_deg=15.0, stator_behind_rim=0.075),
                     "m, deg", "authored",
                     "where the driving-robot kit (--robot drive-kit; operator.kit_fit) is bolted into this truck, truck "
                     "frame: its floor frame (with the IMU) on the footwell floor plate, the pedal actuators' unit on it, "
                     "the lever actuators' bank on the cowl right of the levers, the steering robot's stator 75 mm behind "
                     "the rim's plane on the column axis, and a D435i-class camera on a mast hanging from the overhead "
                     "guard's slat above the seat (the slat's collider, read at build) at an operator's eye point, looking "
                     "forward 15 deg down. The actuators' own axes are the "
                     "controls' pivots (asset geometry); the kit's construction is its catalogue model's "
                     "(robots/models/drive_kit/robot.json)")
DRIVE_KIT_BALLAST = P("DRIVE_KIT_BALLAST", dict(mass=OPERATOR_MASS, density=7850.0, footprint=(0.30, 0.30), y=0.04),
                      "kg, kg/m^3, m", "authored/estimate",
                      "steel plates strapped to the seat cushion with the driving-robot kit, so the operator-presence "
                      "switch (SEAT_SWITCH_N) closes on a driverless truck as on test vehicles carrying a driver's ballast: "
                      "OPERATOR_MASS (authored), steel density 7850 kg/m^3 (estimate, recalled), a 0.30 x 0.30 m stack "
                      "centred on the cushion (authored)")
