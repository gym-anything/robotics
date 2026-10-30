"""Parameter ledger: every physical constant the soil, tools and yard read, with value, unit, label and basis.

Labels (classroom_arrangement convention): datasheet (manufacturer document), retailer (published listing), published
(paper / standard / handbook table), derived (computed here from other entries), estimate (judgement, with a range),
authored (a design choice), calibrated (set from a virtual-laboratory bench of this package), repo (a value qualified
elsewhere in this repository).
The robot (Mobile FR3 Duo stack, DT, reflexes, gains) is classroom_arrangement.params, unchanged.
`python -m soil_excavation.params` prints the table.
"""

import math

LEDGER = []


def _p(name, value, unit, label, basis):
    LEDGER.append(dict(name=name, value=value, unit=unit, label=label, basis=basis))
    return value


GRAVITY = 9.81

# ------------------------------------------------------------------ soil numerics (bench B1)
SOIL_TICKS = _p("SOIL_TICKS", 4, "robot ticks", "derived", "soil step 2 ms = 4 x 0.5 ms: a tool at the FR3's 2 m/s "
                "Cartesian limit moves 4 mm = 0.2 voxel per soil step; the coupling lag equals one soil step")
SOIL_DT = SOIL_TICKS*0.0005
VOXEL = _p("VOXEL", dict(yard=0.02, fine=0.01), "m", "derived", "2 cm yard blocks (19.6 ms per loaded step for "
           "512 k particles), 1 cm for the auger bed and the physics benches")
PPC_AXIS = _p("PPC_AXIS", 2, "-", "repo", "2 x 2 x 2 = 8 particles per cell (Newton MPM examples)")
MPM_ITERATIONS = _p("MPM_ITERATIONS", 1000, "-", "numerical qualification",
                    "Nonlinear iteration cap, paired with 1e-6 residual tolerance. Increasing the cap alone at 1e-4 "
                    "does not stop drift; tightening the tolerance gives late loose-column rates of 0.050 mm/s at 200 "
                    "iterations and 0.024 mm/s at 1000, against 0.821 mm/s at 1e-4. "
                    "Long-horizon and loaded-world qualification remain separate; the cap is not a soil property.")
MPM_TOLERANCE = _p("MPM_TOLERANCE", 1.0e-6, "-", "numerical qualification",
                   "Newton stress residual tolerance. A 1e-4 tolerance leaves continuing particle and inclusion drift "
                   "even at a 1000-iteration cap. Tighter convergence costs more and must be included in runtime measurements.")
CRITICAL_FRACTION = _p("CRITICAL_FRACTION", 0.78, "-", "derived",
                       "packing fraction below which the yield surface collapses (Newton critical_fraction): disturbed "
                       "common earth bulks 20-50 %, typ. 25 %, loam 23 % [Caterpillar Performance Handbook swell tables]; this yard uses amended loam, not the 43-56 % organic topsoil category; 1/0.78 = "
                       "28 % swell of bank material packed at fraction 1.0 (B6 checks the loose density of real spoil)")
SOLVER_CRITICAL_FRACTION = _p("SOLVER_CRITICAL_FRACTION", 0.65, "-", "calibrated",
                              "Newton's critical_fraction (below it a node's yield surface collapses: free flow). Loose "
                              "spoil sits at CRITICAL_FRACTION (0.78, its 28 % swell); with the solver's threshold also "
                              "at 0.78 every dilation of loose soil unjammed it and spoil heaps slumped to 11-15 deg "
                              "(B2); 0.65 leaves loose soil jammed until it dilates past 54 % swell")
COLLIDER_BASIS = _p("COLLIDER_BASIS", "S2", "-", "calibrated", "Newton 1.5's default quadratic serendipity collider basis: "
                    "its contact activation band is half a voxel / degree = 5 mm at 2 cm voxels. With trilinear Q1 (1 cm "
                    "band) a 2.5 mm spade edge acted 2 cm thick: insertion 193 / 334 N at 10 / 20 cm in topsoil and "
                    "822 / 1328 N in subsoil against the edge-bearing estimates 93 / 146 and 243 / 299 N; S2 gave 98 / "
                    "188 N (B4). S2 costs 2.5-3x Q1 per soil step")
GRID_CAPACITY = _p("GRID_CAPACITY", 2.0, "x material cells", "authored", "rebuildable sparse grid capacity over the cells "
                   "the particles fill at bank packing: the one-cell halo, loose spoil on the tarp, heave and airborne "
                   "clods (a planting hole moves ~3 % of the bed block; overflow faults the episode). 3.0 holds "
                   "6.9 GB of the shared GPU for the bed block (D1 runs)")
MAX_SOIL_SPEED = _p("MAX_SOIL_SPEED", 15.0, "m/s", "authored", "fault guard: no clod in a garden moves faster")
SOIL_SETTLE_S = _p("SOIL_SETTLE_S", 0.30, "s", "authored", "gravity settle of the lattice before tick 0 (elastic "
                   "compression gamma z / E < 0.3 %); B11 measures the rest drift after it")

# ------------------------------------------------------------------ soil targets
# rho: bank (moist) density; c, phi: Mohr-Coulomb compression-meridian targets; pt: tensile cut-off; E, nu: elastic;
# Y: compaction cap at bank density (pressure that recompacts loose spoil to the bank density); swell: bank -> loose
# volume (realised through CRITICAL_FRACTION and, for sand/gravel, a reduced bank packing); dil: dilatancy (0..1);
# color: moist sRGB for the renderer (render only)
MATERIALS = _p("MATERIALS", dict(
    topsoil=dict(water="loam", rho=1300., c=3.0e3, phi=32., pt=0.5e3, E=5.0e6, nu=0.30, Y=1.2e5, swell=0.28, dil=0.0,
                 color=(78, 58, 40), basis="friable amended loam: USDA NRCS bulk density < 1.40 ideal; c 2-5 kPa, phi "
                 "30-35 deg (upland sandy loam 3.3 kPa / 33.7 deg [Wong-type terramechanics tables]); E loam 5-20 MPa [Bowles-type tables]; cap: recompaction of loose "
                 "loam to ~1.3 Mg/m3 at ~100-150 kPa (estimate); tension 0.1-0.2 c (estimate)"),
    turf=dict(water="loam", rho=1150., c=9.0e3, phi=32., pt=3.0e3, E=3.0e6, nu=0.30, Y=1.2e5, swell=0.28, dil=0.0,
              color=(64, 72, 38), basis="40 mm root mat: topsoil + root cohesion c_r 6 kPa (Wu-Waldron 2.3-15 kPa [root reinforcement studies]); "
              "tension raised by roots (estimate)"),
    subsoil=dict(water="clay", rho=1700., c=20.0e3, phi=26., pt=5.0e3, E=2.0e7, nu=0.35, Y=4.0e5, swell=0.28, dil=0.1,
                 color=(150, 108, 62), basis="firm clay loam: undrained 'firm' 40-75 kPa in BS 5930 terms, drained c' "
                 "15-30 kPa, phi' 22-30 deg (estimate); E stiff clay 15-50 MPa [Bowles-type tables]; adhesion [Birch et al. 2016]"),
    yard=dict(water="clay", rho=1650., c=12.0e3, phi=28., pt=2.5e3, E=1.5e7, nu=0.30, Y=3.0e5, swell=0.28, dil=0.1,
              color=(112, 86, 60), basis="compacted silt/clay-loam yard soil (fill, foot traffic): bulk density near "
              "the 1.65-1.80 root-restriction band [USDA NRCS]; c, phi estimate"),
    hardpan=dict(water="clay", rho=1800., c=45.0e3, phi=28., pt=10.0e3, E=4.0e7, nu=0.30, Y=1.0e6, swell=0.35, dil=0.2,
                 color=(146, 120, 92), basis="dry compacted clay path: cone index >= 2-3 MPa (root growth stops ~2 MPa "
                 "[Penn State Extension]); c 30-60 kPa (estimate); dry clay is dilatant and brittle (estimate)"),
    sand=dict(water="sand", rho=1600., c=0.5e3, phi=34., pt=0.0, E=1.0e7, nu=0.30, Y=2.0e5, swell=0.12, dil=0.1,
              color=(176, 150, 112), basis="moist washed concrete sand (bedding): compact sand 33.3 deg / 0.69 kPa "
              "[Wong-type terramechanics tables]; swell 5-15 % [Caterpillar swell tables]; E loose-medium sand 10-25 MPa [Bowles-type tables]"),
    gravel=dict(water="granular", rho=1550., c=0.0, phi=40., pt=0.0, E=3.0e7, nu=0.25, Y=3.0e5, swell=0.12, dil=0.2,
                color=(128, 124, 118), basis="3/4 in crushed stone, loose: repose 38-42 deg (estimate)"),
    mulch=dict(water="granular", rho=350., c=0.2e3, phi=40., pt=0.2e3, E=1.0e6, nu=0.30, Y=3.0e4, swell=0.28, dil=0.0,
               color=(92, 60, 38), basis="shredded hardwood bark mulch, 50 mm layer (estimate)"),
    ), "kg/m3, Pa, deg, Pa, Pa, -, Pa, -, -", "published / estimate", "per-material basis")

MOISTURE = _p("MOISTURE", dict(dry=dict(ca=0.3, mut=0.9), moist=dict(ca=1.0, mut=1.0), wet=dict(ca=2.0, mut=0.8)), "x",
              "estimate", "episode moisture state scales the soil-tool interface: adhesion and interface friction "
              "follow Birch et al.'s soil-metal water-content trends (adhesion peaks wet, friction falls). The "
              "soil's own cohesion follows MOISTURE_COHESION per soil class")
MOISTURE_COHESION = _p("MOISTURE_COHESION", dict(
    clay=dict(dry=1.8, moist=1.0, wet=0.5), loam=dict(dry=0.6, moist=1.0, wet=0.6), sand=dict(dry=0.0, moist=1.0, wet=0.3),
    granular=dict(dry=1.0, moist=1.0, wet=1.0)), "x", "estimate",
    "cohesion and tension cut-off relative to the moist value, per soil class (MATERIALS 'water'). Clay soils (subsoil, "
    "yard soil, hardpan) gain strength as they dry: shrinkage and suction harden them (a dry compacted clay path is "
    "the hardpan's own basis, cone index >= 2-3 MPa [Penn State Extension]; unsaturated soil mechanics, Fredlund & Rahardjo 1993) and "
    "soften when wet. A friable loam's apparent cohesion is suction, largest moist and lost both when dry (crumbly) and "
    "when wet; a sand's is suction only (dry sand c = 0)")

# ------------------------------------------------------------------ yield-law mapping
# Newton's projected yield: q = sqrt(3 J2) <= s h + mu (A p + B p_t) for the Drucker-Prager branch; the
# (A, B) are fit parameters, unused by the exact compression_meridian default (B15).
YIELD_LAW = _p("YIELD_LAW", dict(A=1.87, B=1.04, mode="compression_meridian"), "-", "derived",
               "Exact ascending-branch mapping from the installed Newton stress projector: sqrt(J2) <= s + mu(p+pt). "
               "Compression-meridian MC match, checked by B15. A/B are retained only for historical optional modes; "
               "they are not used by the default. Constitutive identity is not excavation calibration.")
KAPPA_DS = _p("KAPPA_DS", 0.546, "-", "calibrated / unresolved",
              "Historical scale mu = tan(phi) / KAPPA_DS. Derived from an earlier B1 outer-cone run, then retained "
              "for heap tuning; the final B1 friction angle missed its target. Not a verified material constant.")
KAPPA_DS_C = _p("KAPPA_DS_C", 0.922, "-", "calibrated / unresolved",
                "Historical B1 intercept scale: c_ds = KAPPA_DS_C (s + B mu p_t). B1 on the final mapping gave "
                "6.9 kPa against 3 kPa. Calibration across stress paths remains open.")
E_CAP = _p("E_CAP", 4.0e7, "Pa", "derived",
           "Young's modulus used by the solver = min(E, E_CAP): no material is capped (the largest estimated modulus "
           "in MATERIALS, hardpan, is 40 MPa). Without the in-interval damper and the under-relaxation a held blade "
           "chatters on the explicit 2 ms coupling at 5 MPa (C1); with both, the robot-held spade pushed into the bed is as "
           "steady at 20 MPa as at 1 MPa (velocity noise <= 0.008 m/s, soil force changing <= 0.65 N per step). A 1 MPa "
           "cap would make the hardpan rubbery (unconfined strength at ~15 % elastic strain) and every soil too soft "
           "under a post")
CUT_CLOSURE = _p("CUT_CLOSURE", dict(band_voxels=0.5, v_min=0.01), "voxels, m/s", "closure",
                 "a moving blade plate, pick or adze remolds the soil it passes through to the loose "
                 "state (J_p >= J_p loose: residual strength retained, friction kept) within half a voxel across its thickness only, advancing edge-first; "
                 "the 2 cm grid cannot keep a 2-3 mm cut open, so without it cut faces re-bond (B5, D2-P v1-v4). Not applied to the "
                 "auger: with it the turning flight loosened its whole swept cylinder and the hole filled back (D9 v6)")
COUPLING_RELAXATION = _p("COUPLING_RELAXATION", 0.35, "-", "calibrated",
                         "under-relaxation of the soil wrench applied to kinematically coupled bodies (running average per "
                         "2 ms soil step): removes the period-2 oscillation of the explicit coupling (diag D2-P v5)")
ROBOT_GUARD = _p("ROBOT_GUARD", dict(below_grade_m=0.25, speed_m_s=4.0), "m, m/s", "authored",
                 "fault guard on the robot's own bodies (the classroom guard covers objects only): no body of the TMR "
                 "base (hull, drive units, casters; arms excluded, a hand may reach into a dug hole) goes 0.25 m below "
                 "grade on a yard whose ground is everywhere at grade or above, and no FR3 link (1.7-2 m/s TCP limit) "
                 "or base (~1 m/s) moves at 4 m/s")
COUPLING_DAMPING = _p("COUPLING_DAMPING", dict(v_ref=0.02, w_ref=0.10, cap=1.0), "m/s, rad/s, -", "calibrated",
                      "a kinematically coupled body's lagged soil wrench is supplemented, inside each 2 ms soil interval, "
                      "by a damper on its deviation from the velocity the soil step assumed: D = min(|F_soil| / v_ref, "
                      "cap m / dt_mujoco) (rotation likewise with w_ref and the inertia). It vanishes in steady motion and "
                      "in air. In C1 a 0.6 kg blade on a 5 kN/m grip chatters at +-434 N std without it and holds +-6.9 N "
                      "with it")
LOOSE_STRENGTH = _p("LOOSE_STRENGTH", 0.15, "x bank", "estimate", "cohesion, tension cut-off and cap of loosened "
                    "cohesive soil relative to the bank (remoulded loam keeps a little suction cohesion; the cap keeps the "
                    "friction cone of loose spoil up to ~0.15 Y, i.e. 18 kPa for topsoil, piles to ~1 m: B2)")
HARDENING_RATE = _p("HARDENING_RATE", 1.0, "-", "authored", "J_p follows the plastic volume ratio in compaction")
SOFTENING_RATE = _p("SOFTENING_RATE", 1.0, "-", "authored", "J_p follows the plastic volume ratio in dilation: a "
                    "loosened cell loses its cohesion as it bulks toward the loose density")

# ------------------------------------------------------------------ tool-soil interfaces
# collider Coulomb friction mu_t = tan(delta) and shear adhesion c_a (Pa), moist state.
# Newton expects a normal offset c_a / mu_t; materials.contact_normal_offset performs the conversion.
INTERFACE = _p("INTERFACE", dict(
    stainless=(0.40, 750.), steel=(0.47, 1500.), aluminium=(0.42, 1000.), wood=(0.50, 500.), pvc=(0.35, 300.),
    rootball=(0.60, 1000.), stone=(0.55, 200.), robot=(0.50, 500.), concrete=(0.60, 0.), tarp=(0.40, 200.),
    plastic=(0.35, 200.), wheel=(0.60, 0.), caster=(0.10, 0.)), "mu, Pa", "estimate",
    "smooth steel in moist loam delta 20-30 deg; adhesion means 1.74 / 2.35 / 3.35 kPa (sandy loam / clay loam / clay) "
    "at high water contents and 0.22-7.62 kPa range [Birch et al. 2016]; polished stainless 'for minimal soil adhesion' [Spear & Jackson product page]; "
    "Vulkollan drive tread uses the robot model's 0.6 traction estimate; a free rolling caster uses 0.1 resistance; "
    "tires have no cohesive adhesion")
FAR_FIELD = _p("FAR_FIELD", "rough", "-", "authored", "block side/bottom faces: mu = tan(phi), adhesion = c of the "
               "adjacent soil, standing in for undisturbed ground")

# ------------------------------------------------------------------ tools
SPADE = _p("SPADE", dict(head=(0.180, 0.290, 0.0025), length=0.985, mass=1.90, shaft_r=0.0175, tread=0.012, edge_sagitta=0.012),
           "m, kg", "datasheet", "Spear & Jackson Traditional Stainless digging spade: head 290 x 180 mm, mirror-polished "
           "stainless, treaded, hardwood D-handle; Neverbend 1160SP 985 mm overall, 1.9 kg [Spear & Jackson product pages]")
TRENCH_SHOVEL = _p("TRENCH_SHOVEL", dict(head=(0.108, 0.292, 0.0020), length=1.49, mass=1.60, shaft_r=0.016,
                                         v_deg=20.0, lift_deg=18.0),
                   "m, kg", "datasheet", "Ames Razor-Back 47171: 11.5 x 4.25 in V blade, 48 in handle, 58.75 in overall, "
                   "3.5 lb [Ames listings]; V and lift angles estimated from product photos")
TROWEL = _p("TROWEL", dict(head=(0.078, 0.150, 0.0025), length=0.307, depth=0.030, mass=0.165, grip_r=0.016),
            "m, kg", "datasheet", "Fiskars Ergo trowel: 307 mm long, 78 mm wide, 30 mm deep, cast aluminium, 0.165 kg [Fiskars listings]")
MATTOCK = _p("MATTOCK", dict(handle=0.381, head_mass=0.68, mass=0.90, pick=0.150, adze=(0.075, 0.090)),
             "m, kg", "datasheet + estimate", "garden mattock: 1.5 lb forged pick + adze head, 15 in fiberglass handle "
             "[INTERTOOL listing]; 0.9 kg all-in (estimate: a comparable 15 in pick lists 1.49 lb)")
DIGGER = _p("DIGGER", dict(blade=(0.140, 0.203), spread=0.140, handles=1.22, pin_above_tip=0.22, mass=3.2,
                           pin_friction=0.3), "m, kg, N m", "datasheet + estimate",
            "Ames/True Temper clamshell post-hole digger: 8 x 5.5 in blades, 5.5 in spread, 48 in handles, 58.5 in "
            "overall, 6.75-7.2 lb [Ames/True Temper catalogue]; pin height above the tips and pin friction estimated (no published value)")
DRILL = _p("DRILL", dict(mass=2.10, side_handle=0.15, gears_rpm=(450., 1300., 2000.), torque_hard=95., torque_soft=66.,
                         power_w=820., trigger_travel=0.008, trigger_spring=(30.0, 1500.)),
           "kg, rpm, N m, W, m, (N, N/m)", "datasheet + estimate",
           "DeWalt DCD996 18/20 V XR: 0-450 / 0-1300 / 0-2000 rpm, 95 N m hard / 66 N m soft, 820 W, 2.1 kg with "
           "battery, 360 deg side handle [DeWalt data]; trigger travel and spring estimated: 30 N preload, 42 N at full travel. "
           "Known limitation: in the start hold the drill hangs from the side handle and its weight rests "
           "~37 N on the open right hand's finger at the trigger, which keeps it pressed; at 45 N preload the trigger "
           "stayed out but the gripper's squeeze no longer reached it")
DRILL_CLUTCH = _p("DRILL_CLUTCH", {k: round(1.0 + 0.9*(k - 1), 2) for k in range(1, 12)}, "N m at the spindle",
                  "estimate", "DCD996 electronic clutch, 11 settings + drill mode (no cap); slip torques are not published, "
                  "linear 1-10 N m assumed (typical driver clutches span ~1-10 N m); None = drill mode")
AUGER = _p("AUGER", dict(diameter=0.0762, length=0.305, flight_len=0.28, pitch=0.057, shaft_r=0.0079,
                         flight_t=0.0034, mass=0.71), "m, kg", "datasheet + estimate",
           "Power Planter 312: 3 x 12 in, 11 in digging depth, 5/8 in solid shaft, 10-gauge flighting, 3/8 in hex, "
           "0.71 kg [Power Planter listings]; pitch 0.75 D estimated from product photos")
TAMPER = _p("TAMPER", dict(section=(0.038, 0.089), length=1.20, rho=450.), "m, kg/m3", "published + estimate",
            "2x4 SPF actual 1.5 x 3.5 in; SPF ~450 kg/m3 (estimate) -> 1.83 kg")
ROOTBALL = _p("ROOTBALL", dict(diameter=0.165, height=0.160, mass=2.0), "m, kg", "published + estimate",
              "#1 nursery container (ANSI Z60.1 2.49-4.12 L; 6-7 in pots); wet peat-bark mix 0.5-0.65 kg/L")
POST = _p("POST", dict(section=0.0889, length=1.829, rho=370.), "m, kg/m3", "published + estimate",
          "4x4 (3.5 in actual) western red cedar, 6 ft; ~370 kg/m3 air-dry (estimate) -> 5.35 kg")
PIPE = _p("PIPE", dict(od=0.0483, depth=0.30), "m", "published + estimate",
          "1-1/2 in Schedule 40 PVC irrigation supply line, OD 48.3 mm (ASTM D1785); 0.30 m cover (irrigation practice, estimate)")
PAIL = _p("PAIL", dict(r=0.12, height=0.25, wall=0.003, fill=0.15), "m", "retailer + estimate",
          "2-gallon HDPE pail ~9.5 in across, 10 in tall (estimate from listings); bedding sand 15 cm deep (~8 kg)")
STONE_RHO = _p("STONE_RHO", 2650., "kg/m3", "published", "quartz/granite fieldstone")
TARP = _p("TARP", dict(size=(1.83, 2.44), mass=0.40), "m, kg", "retailer", "6 x 8 ft poly tarp")

POST_TEST = _p("POST_TEST", dict(force=150.0, height=1.0, ramp_s=0.5, hold_s=0.5, creep_s=2.0, creep_window_s=0.25,
                                 unload_s=0.25, rest_s=0.25, release_s=0.25, max_ground_deflection=0.0127,
                                 plumb_deg=1.0, plumb_zero_deg=5.0),
               "N, m, s, s, s, s, s, s, s, m, deg, deg", "published + authored",
               "set-post verdict (task.Evaluator.post_load_test), on a restored copy of the state with the unburied tools "
               "parked and the braces released. Load (authored estimate): a person leaning on a post, 150 N at 1 m above "
               "grade, ramped over 0.5 s and held 0.5 s, toward +x, -x, +y and -y. Reading: the peak horizontal motion of "
               "the post at grade during the push, and that motion projected 2 s further at the creep rate of the last "
               "0.25 s at full load -- the rest of a few-second lean; it is conservative, since creep slows (press40: "
               "projected 9.4 mm against 9.13 mm measured after 2 s more). A 2 mm/s 'settled' rule would be satisfied "
               "by a steady creep. Limit (published premise): 12.7 mm "
               "-- IBC 1806.3.4 lets isolated poles 'not adversely affected by a 1/2-inch (12.7 mm) motion at the ground "
               "surface due to short-term lateral loads' use doubled lateral bearing values; the code's tolerable motion "
               "is applied to the authored lean load and measured where the code measures it. It "
               "is not a code design check: by IBC Eq. 18-1 a 0.55 m embedment in class-5 soil is allowed only ~57 N at "
               "1 m. Plumb (authored; a carpenter's level reads 1 deg = 17 mm/m): the released post within 1 deg after "
               "0.25 s unloaded, and again after each push is ramped off over 0.25 s and rested 0.25 s (a lean left by "
               "the push; after-unload recovery is complete within ~0.05 s); credit "
               "falls to 0 at 5 deg. The after-push lean is read over the same 3 s lean as the support reading "
               "(the lean left after the 1 s push scaled by the projected over the end-of-push lean; the share left "
               "after unloading was 0.86 after 1 s and 0.88 after 3 s on press40), unloaded only where the projected "
               "lean passes 1 deg; a direction that gives way is read at its lean when it gave way, so plumb and its "
               "credit never ignore a failing direction. For uniform fills this 1 deg "
               "reading, not the 12.7 mm, decides: press40 holds 10.9 mm projected but is left 1.25 deg out of plumb. "
               "Its basis is the task text, 'stay plumb within 1 degree' after the "
               "push, and a carpenter's level. Support credit: the share of the notional 3 s lean's load impulse "
               "carried before the post at grade reached the limit (1 = never); a fill that gives way before full load "
               "earns at most ~9 % of it, by design: the support half pays for holding a person's lean. Spread: "
               "same-state repeats within 1.4 %; two builds of the same construction differed by up to ~2.5 % per "
               "direction (native) and ~11 % (press40), so a state within ~10 % of a line can read either way on "
               "another build")
POST_FINISH = _p("POST_FINISH", dict(low=-0.030, crown=0.050, share=0.90, yard_low=-0.030, yard_share=0.95,
                                     lip_inset=0.015, lip_margin=0.03, post_margin=0.010, in_hole_depth=0.030,
                                     clear_margin=0.30, clear_height=0.02, contact_n=0.5, placement_full=0.88,
                                     lawn_allowance_m3=0.001),
                 "m, m, -, m, -, m, m, m, m, m, m, N, -, m3", "authored",
                 "set-post finish, read from 1 cm column tops (task.Ledger.heightfield). Backfill to grade: >= 90 % of "
                 "the dug footprint's columns (inside the hand-dug lip outline at grade less 15 mm, the post section + "
                 "10 mm excluded) within -30/+50 mm of grade: 'back up to grade' with a slight drainage crown; a mean "
                 "would let pits cancel heaps. The outermost 15 mm inside the lip and the first 30 "
                 "mm beyond it are read by neither surface criterion: 1 cm columns with 2 cm voxels straddle the ragged "
                 "wall there. Surrounding grade: >= 95 % of the block's columns 30 mm or more "
                 "beyond the lip no lower than -30 mm. Fill from the tarp: <= 1 L (about 1.6 kg) of native yard soil "
                 "moved more than 50 mm from beyond the lip, an allowance for scuffs; the particle origin record shows "
                 "lawn soil skimmed for fill, which the surface tolerance would let through; the same "
                 "allowance is applied before placement subtracts yard soil. Only yard soil now in the hole counts "
                 "(lawn damage elsewhere is surrounding_grade's). Post free: no robot or held-tool "
                 "contact on the post above 0.5 N; no robot part below grade + 20 mm on the soil block within 0.30 m "
                 "beyond the lip, about the depth of the post's lateral pivot (0.31 m), i.e. the soil the lean loads"
                 "; held tools are not checked for clearance, since the test parks them; no tool part more than 30 mm "
                 "below grade in the footprint with soil over it (a tool in a low spot of the surface is not buried). "
                 "Placement: tarp soil in the cavity up to "
                 "grade counts in full at 88 % of the cavity's bank volume, the volume a settled bank-state fill to "
                 "grade holds in this world (30.8 of 34.86 L)")
BRACE = _p("BRACE", dict(k=2.0e4, c=150.0, height=1.0, reach=0.9), "N/m, N s/m, m, m", "estimate",
           "two 1x2 diagonal braces nailed to the post 1 m up and to stakes 0.9 m out: axial stiffness dominated by the "
           "nailed joints (estimate)")

# ------------------------------------------------------------------ strike / fragile contact
STRIKE = _p("STRIKE", dict(force_n=30.0, speed_m_s=0.2), "N, m/s", "authored",
            "a tool contact on the marked lateral with normal force > 30 N or approach speed > 0.2 m/s counts as a "
            "strike to report (a light trowel touch locates the line; Nebraska 811 hand-dig rules)")


def table():
    rows = ["| Parameter | Value | Unit | Label | Basis |", "|---|---|---|---|---|"]
    for e in LEDGER:
        v = e["value"]
        if isinstance(v, float):
            v = f"{v:g}"
        if isinstance(v, dict):
            v = "; ".join(f"{k}: {('{' + ', '.join(f'{a}={b}' for a, b in x.items() if a not in ('basis', 'color')) + '}') if isinstance(x, dict) else x}"
                          for k, x in v.items())
        rows.append(f"| `{e['name']}` | {v} | {e['unit']} | {e['label']} | {e['basis']} |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(table())
