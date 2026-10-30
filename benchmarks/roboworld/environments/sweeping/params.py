"""Parameter ledger of the sweeping environment, in one place.

Labels: datasheet (a maker's or standard's published number), measured (by a probe of this package, named), repo
(taken from a qualified package of this repository), derived (arithmetic from other rows), estimate (a number chosen
inside a stated range because no published value was found), authored (a design choice of this environment), prior
(set from an expectation of the outcome that was not measured; the row names the expectation), engine (a numerical
setting, not a property of the world).
"""

import math
from classroom_arrangement import params as CP

# ------------------------------------------------------------------ engine
DT = CP.DT                         # 0.5 ms rigid tick (repo: classroom_arrangement)
GRAVITY = CP.GRAVITY
SUBSTEPS = 2                       # bristle and debris substeps per tick (0.25 ms): engine
ROOM = (-3.70, 0.0, 3.70, 9.60)    # floor extent x0, y0, x1, y1 (m): repo (room 114 shell)
ROOM_HEIGHT = 3.25
CELL = 0.10                        # collider cell size of the debris solver (m): engine

# ------------------------------------------------------------------ filaments
# Filament moduli from one brush maker's fibre table (Carolina Brush, "Fiber Property Comparisons", archived page,
# read 2026-10-11): "Polyester" 445 kpsi = 3.07 GPa, "Polypropylene" 740 kpsi = 5.10 GPa.
# The maker's polyester is PBT; the angle broom's fibre is PET, for which no brush maker's figure was found: 3.0 GPa is
# kept and is a stand-in. Bracketed, the mug stroke goes from nothing to everything carried over it.
PET = dict(E=3.0e9, rho=1380.0)
PP = dict(E=5.1e9, rho=905.0)
BRISTLE_MU = 0.30                  # filament tip on waxed vinyl, kinetic (estimate: steel wire on concrete 0.27)
BRISTLE_ZETA = 0.08                # damping ratio of a filament's first bending mode (estimate, 5 to 10 %)
# Damping ratio of the bristle chain's shortest wave (two segments, about 600 Hz), from material damping that grows
# with frequency. Prior: polymer loss factors are of the order of 0.1, so 0.05 of critical (general materials
# knowledge, not measured or looked up in this build). Measured here: without it that wave rang undamped whenever a
# bank was pressed on the floor; with 0.05, 0.2 or 0.5 it does not, and the bank's force is the same to 0.5 %.
BRISTLE_ZETA_TOP = 0.05

# ------------------------------------------------------------------ tools
ANGLE_BROOM = dict(
    block=(0.279, 0.038, 0.050), handle_d=0.019, handle_len=1.17, length=1.346, mass=0.50,
    bristle_len=(0.115, 0.140), filament_d=0.40e-3, material="PET", filaments=8000, clusters=800, rows=5, nodes=11,
    source="Libman Commercial 201 (sweep 279 mm, 1.346 m overall, 127 mm fibre); mass estimated under the case weight bound")
PUSH_BROOM = dict(
    block=(0.457, 0.064, 0.038), handle_d=0.024, handle_len=1.52, handle_angle_deg=55.0, mass=1.10,
    bristle_len=(0.076, 0.076), filament_d=0.50e-3, material="PP", filaments=12600, clusters=1260, rows=6, nodes=7,
    source="Rubbermaid FG9B0600 (457 mm block, 76 mm trim); mass estimated")
LOBBY_BROOM = dict(
    # THE HEIGHT is the maker's: 37.00 in (0.94 m) from the cap's top to the bristle tips, which with the cap, a
    # 40 mm block and 140 mm of bristle leaves 0.734 m of handle (measured on the built tool: 0.940 m)
    block=(0.190, 0.032, 0.040), handle_d=0.022, handle_len=0.734, length=0.94, mass=0.32,
    # fill: the angle broom's tuft density over this head's 190 mm (5,400 filaments of 0.40 mm polypropylene). A
    # first estimate (4,000 of 0.35 mm) gave a bank a sixth as stiff as the angle broom's, which rode over every pile
    bristle_len=(0.125, 0.140), filament_d=0.40e-3, material="PP", filaments=5400, clusters=540, rows=5, nodes=11,
    source="Rubbermaid FG637400, the maker's page: height 37.00 in (0.94 m), weight 0.70 lb (0.32 kg), polypropylene, staple set, "
           "vinyl-coated metal handle. The head's 190 mm (7.5 in) is from resellers' listings; the maker's page gives the "
           "product's length as 6.00 in. Block height, bristle length and the filaments are estimates")
COUNTER_BRUSH = dict(
    block=(0.203, 0.030, 0.022), handle_len=0.115, length=0.318, mass=0.20,
    # 600 clusters of 5 filaments: with 300 of 10 the clusters of a row stood 2.6 mm apart and were 1.2 mm thick,
    # and 42 % of a lane's eraser crumbs went through the 1.4 mm gaps (the three brooms' gaps are
    # under 0.1 mm). Now 0.43 mm, less than the finest piece (grit of 0.6 mm)
    bristle_len=(0.064, 0.064), filament_d=0.30e-3, material="PP", filaments=3000, clusters=600, rows=4, nodes=7,
    source="Rubbermaid FG634200 (318 mm, 203 mm face, 64 mm trim). Weight DERIVED: the maker's page gives no weight of "
           "one brush; it gives 3.00 lb as the product's weight and 3.10 lb (1.40 kg) for the case of six, so one brush "
           "weighs 0.23 kg at the most; 0.20 kg taken")
LOBBY_PAN = dict(
    # THE WEIGHT is the maker's: 1.86 lb (0.84 kg). The handle keeps its 0.21 kg (a vinyl-coated steel tube); the pan
    # has the rest
    pan=(0.286, 0.287, 0.127), outer_w=0.325, lip=0.0005, wall=0.003, handle_len=0.76, handle_d=0.022, mass=0.84,
    pan_mass=0.63, handle_mass=0.21, wheel_r=0.020, wheel_w=0.012, wheel_mass=0.008, wheel_mu=0.5, mu=0.30,
    source="Rubbermaid FG253100, the maker's page: weight 1.86 lb (0.84 kg), pan 12-3/4 x 11-1/4 x 5 in (324 x 286 x 127 mm), "
           "36 in tall with its handle, injection-moulded plastic, vinyl-coated metal handle. Mouth 286 mm and handle 0.76 m "
           "from resellers' listings; edge thickness not published, 0.5 mm assumed; the split of the weight between pan and "
           "handle is an estimate")
HAND_PAN = dict(
    pan=(0.311, 0.210, 0.060), lip=0.0005, wall=0.003, handle_len=0.114, handle_d=0.028, mass=0.23, mu=0.30,
    source="Rubbermaid FG200500, the maker's page: 12.25 x 8.25 in (311 x 210 mm), 0.50 lb (0.23 kg), an open scoop "
           "with a flat handle and a hang hole; wall height and edge thickness not published (60 mm, 0.5 mm assumed)")
# Weights: the maker's own pages (rubbermaidcommercial.com), read 2026-10-11.
BARREL = dict(d=0.50, h=0.58, wall=0.004, mass=2.64,
              source="Rubbermaid BRUTE 2620, 20 gal, grey (FG262000GRAY): 22.9 in high, 22.5 in across its handles; the maker's "
                     "product weight 5.82 lb = 2.64 kg (its colours run from 2.49 to 2.65 kg)")
PAIL = dict(d=0.30, h=0.29, wall=0.003, mass=0.95,
            source="14 qt round utility pail (Rubbermaid FG261400: 11.25 in high, 13.13 in across): the maker's page gives a "
                   "product weight of 10.70 lb, which cannot be right for a 13 litre pail and disagrees with its own case "
                   "(12.57 lb for six): 2.10 lb = 0.95 kg from the case, packaging included (derived; one seller lists 2.094 lb)")
WASTEBASKET = dict(size=(0.37, 0.26, 0.38), wall=0.003, mass=0.74,
                   source="28 qt rectangular wastebasket (Rubbermaid FG295600): 14.5 x 10.5 x 15.0 in, the maker's product "
                          "weight 1.64 lb = 0.74 kg")
TOOL_MU = 0.35                     # tool plastics and painted steel on vinyl and on each other (estimate)
HANDLE_MU = 0.45                   # painted steel handle on a finger pad uses the pad's 0.8 (pad priority); elsewhere this
# The round bottoms of the barrel and the pail on the floor: resistance to rolling and to spinning at the contact, as
# fractions of the bottom's radius (the torque each can carry is that length times the normal force: 4.5 and 7.5 mm
# for the pail, 7.5 and 12.5 mm for the barrel). They stand for everything that damps the wobble of a knocked
# container (rim flex, micro-slip, small impacts); no measurement was found. Prior: a knocked plastic pail or barrel
# stands still within two or three seconds (everyday observation, not measured). With zero a knocked pail rolled on
# its rim for ever.
BOTTOM_ROLLING = 0.03
BOTTOM_TORSIONAL = 0.05
# The cap a broom hangs by, rolling on its prongs: 0.5 mm (rolling and spinning). Prior: a broom on a hook, pushed
# a hand's width aside, hangs still again within about a quarter of a minute (everyday observation, not measured).
# Measured here: without it a knocked broom swings at 1.23 degrees for ever; with it, at rest 20 s after a knock that
# swings it 12 degrees.
CAP_ROLLING = 0.5e-3

# ------------------------------------------------------------------ debris
# name: (friction on floor, friction on other debris and on tools, friction on bristles, restitution, rolling resistance)
DEBRIS_CLASSES = {
    # grit on grit 0.50, an estimate (quartz on quartz is quoted between 0.45 and 0.6). It cannot be calibrated on
    # the heap's slope: three pours of 1,200 grains at the same friction lie between 24.7 and 33.9 degrees, and the
    # means at 0.5, 0.65 and 0.8 are 30, 34 and 33 degrees. Dry sand's 30 to 35 degrees is met within
    # that scatter.
    "grit":   (0.50, 0.50, 0.40, 0.30, 0.30),
    "rice":   (0.35, 0.35, 0.30, 0.40, 0.05),
    "cereal": (0.40, 0.40, 0.35, 0.30, 0.05),
    "bead":   (0.30, 0.30, 0.30, 0.60, 0.02),
    "crumb":  (0.50, 0.50, 0.40, 0.20, 0.30),
    "wad":    (0.40, 0.40, 0.40, 0.10, 0.20),
    "scrap":  (0.40, 0.40, 0.40, 0.10, 0.30),
    "shard":  (0.50, 0.45, 0.40, 0.30, 0.30),
    "pencil": (0.35, 0.35, 0.30, 0.30, 0.04),
}
# Paper yields where a rigid shape presses on it. A wad is a lumpy rigid ball for the floor, for bristles and for
# other pieces. Against a rigid shape (a finger pad, a tool's block, a wall, a wheel) each of its spheres gives like
# a spring of this stiffness, half critically damped, up to half the sphere's radius, and is rigid beyond that; the
# push is along the shape's normal at the wad's centre (a crumpled ball flattens against what presses on it), it
# resists turning with the radius of its contact patch, and friction holds its centre against the shape. Prior:
# a crumpled sheet gives about a centimetre under a pinch of a few newtons (everyday observation, not measured).
# Without it a hand could not hold a wad.
SOFT = {"wad": 400.0}              # N/m per sphere in contact
KINETIC_RATIO = 0.8                # kinetic over static friction of debris (estimate)
GRIT_DENSITY = 2650.0              # quartz (literature)
RICE = dict(length=6.61e-3, width=2.12e-3, thick=1.88e-3, mass=19.9e-6)     # long-grain (search)
CEREAL = dict(d=12.0e-3, tube=3.2e-3, mass=0.11e-3)                          # oat ring; ring size open
BEAD = dict(d=9.0e-3, length=6.0e-3, mass=0.29e-3)                           # pony bead (search)
CRUMB_DENSITY = 1100.0
WAD = dict(d=0.060, mass=4.5e-3)                                             # crumpled letter sheet of 75 g/m2
SCRAP = dict(size=(0.020, 0.040), thick=0.8e-3, areal=0.075)                 # thicker than paper: stated limitation
MUG = dict(d=0.082, h=0.096, wall=4.5e-3, mass=0.36, density=2300.0)         # 11 oz stoneware (search, estimate)
PENCIL = dict(length=0.190, d=7.0e-3, mass=5.0e-3)

# ------------------------------------------------------------------ fine dust (reduced-order film, sweeping/dust.py)
DUST = dict(
    cell=0.004, pan_cell=0.002,    # grids: the room's floor, a pan's floor (m): engine
    bulk_density=1100.0,           # kg/m3: IEC 62885-2 mineral test dust, the maker's page
    repose_deg=38.0,               # estimate for a fine mineral dust (fine sand: 30 to 34)
    pickup=0.755,                  # share of what a bank CAN move in its path that one pass takes with it: derived from one
                                   # study's first sweep of building sand (73.3 % cleared; Tabia et al.) and the share it
                                   # never clears (below): 0.733/(1 - 0.029).
    leak_length=4.5,               # m of path over which the ridge a bank pushes loses 63 % back onto the floor: prior
                                   # (a fifth per metre; no measurement found)
    residue_share=0.029,           # share of what lay in a place that a dry broom never moves from it: derived from the
                                   # same study's building sand after ten sweeps, 0.4 to 8.5 % of the area uncleared over
                                   # its nine settings (99.58, 98.83, 97.07 % cleared at 70 degrees; 98.65, 96.94, 91.46 at
                                   # 80; 96.76, 97.14, 95.17 at 90), median 2.9 %.
    lodging=0.2,                   # fraction of a lifted tip's load that stays in the bristles: prior, bounded by
                                   # the same study's count of sweeps to 95 %
    tau_lodged=2.0,                # s, lodged dust trickling out under a sliding tip: prior, same bound
    a_tap=30.0,                    # m/s2, a tip stopped harder than this lets go of its lodged dust: prior
    hold=0.5,                      # s before what a tip laid down can be picked up again (the bank has passed): engine
    contact_gap=1.0e-3)            # a tip this close to a surface is on it (m): engine

# Fine dust laid at the start (sweeping/layout.py). AUTHORED amounts inside one published distribution: the dust found
# on the smooth floors (tile, vinyl, wood) of 51 classrooms of 17 French schools, vacuumed from 2 to 5 m2 and sieved
# under 355 um, was 0.13 +- 0.16 g/m2, median 0.07 (Canha et al., Atmosphere 6(11):1676, 2015, Table 3; opened
# 2026-10-11).
# It is a stock found on a floor, not a daily fall.
DUST_LOADS = dict(pile_g=1.0,        # g in a disc of 0.40 m round the swept pile: what 8 m2 of floor at the mean gives up
                  open_g_m2=0.13,    # closing time, the open floor: the mean
                  door_g_m2=0.45,    # closing time, by the door (tracked in): the mean plus two spreads
                  dusty_g_m2=0.29)   # variant `dusty`, over the task's zone: the mean plus one spread

# ------------------------------------------------------------------ debris solver
# Rest detector: 25 um in 0.1 s. At 50 um a poured grit heap stands at 37.5 to 41.6 degrees instead of 34.2 (slow
# settling between 0.25 and 0.5 mm/s is cut). At 25 um a swept pile sleeps in 3 to 4 s from its unsettled start, the
# heap at 34.2 degrees.
REST_DISP = 2.5e-5                 # rest detector: no net motion above this (m) ...
REST_TIME = 0.1                    # ... for this long (s) puts a piece to sleep
SLIP_DISP = 1.0e-5                 # pre-sliding distance of a friction anchor (m): micro-slip scale, authored
DEPENETRATION_SPEED = 1.0          # m/s, for overlaps that exist at the start of a substep (engine)
MAX_DEBRIS_SPEED = 20.0            # m/s: faster is a fault (unified contract, invariant 10). While no bristle bank is
                                   # at work the fault comes sooner: 8 m/s (a fall of 3.3 m) plus twice the speed of
                                   # the fastest rigid shape (solver.V_FALL, Sweeper.push)

# ------------------------------------------------------------------ room and fixtures
COVE_BASE = (0.102, 0.010)         # height, thickness of the rubber wall base (search: 4 in cove base)
TOE_SPACE = (0.100, 0.075)         # height, depth under the casework doors (authored from usual casework)
CASEWORK = dict(x=(-2.40, 2.40), depth=0.61, top=0.914)
CUBBIES = dict(y=(6.30, 8.60), depth=0.35, plinth=0.100, height=1.22)
MAT = dict(centre=(2.65, 0.80), size=(0.90, 1.50), border=0.009, pile=0.006)
SQUARE = dict(centre=(2.30, 2.85), side=0.45)
SQUARE_ALCOVE = dict(centre=(2.75, 8.10), side=0.45)
OPEN_ZONE = (1.50, 3.60, 3.50, 5.80)
RAIL = dict(x=3.70, y=(1.30, 2.70), z=1.42)
TABLES = {"A1": (-1.95, 2.90), "A2": (-1.95, 5.10), "A3": (-1.95, 7.30), "B1": (0.55, 2.90), "B2": (0.55, 5.10), "B3": (0.55, 7.30)}
TEACHER_DESK = (-2.35, 1.05)
SAND_TABLE = dict(centre=(-3.15, 8.25), size=(0.58, 0.94), height=0.62)
MU_FLOOR = CP.MU_FLOOR

# ------------------------------------------------------------------ robot (repo: classroom_arrangement ledger, catalog)
ROBOT_START = (2.30, 2.00, 0.0)            # x, y, yaw: in the right-hand lane facing the rail (+x)
DOCK_START = (-1.95, 6.22, math.pi/2)      # docked at the near long side of craft table A3 (T6). At 6.15 the base
                                           # stands in the two far chairs of table A2 and throws them over at the
                                           # reset; they are also pushed in for this task
                                           # (layout.setup). Further forward the arms' first links meet the table.
GRIP_FORCE_TOOL = 40.0                     # N per finger on a tool handle (FR3 hand: 70 N continuous)
# The fingertip pad (classroom_arrangement: the Franka Hand finger's tip box, MuJoCo time constant 5 ms) is rubber over
# the finger's hard body. Thickness of that rubber: repo, label_application's pad (rubber 2 MPa over 2 mm,
# materials.py; labelsim PAD_COMPRESSION_DEPTH), an estimate there. Behind it the scene puts the hard body as a box
# (scene.pad_cores) and the floor fault judges that box (world._floor_fault).
PAD_RUBBER = 0.002
# Impedance of the hard body's contacts: MuJoCo's hardest (its clamp, mjMAXIMP), engine. MuJoCo sizes a contact's
# softness on the mass it assumes for the body (efc_diagA: 85 g for a finger), while 0.95 to 4.3 kg of arm stands behind
# a fingertip pressed on the floor (J M^-1 J', measured). At 0.9999 the hard body is at least
# as rigid against that arm (MuJoCo's regulariser over the true inverse mass, (1-d)/d x m_arm/m_assumed: 0.001 to 0.005)
# as the world's own rigid contacts (0.99: 0.01) are against a free body whose mass MuJoCo assumes rightly.
PAD_CORE_SOLIMP = (0.9999, 0.9999, 0.001)


def ledger():
    """Rows (parameter, value, label, basis) of the ledger."""
    rows = [
        ("Rigid tick; bristle and debris substep", "0.5 ms; 0.25 ms", "engine", "classroom_arrangement"),
        ("Room", "7.40 x 9.60 x 3.25 m, waxed vinyl tile, friction 0.5 on glides' own schedule", "repo", "room 114 shell"),
        ("PET filament", "E 3.0 GPa, 1380 kg/m3", "stand-in",
         "the same table's Polyester row (445 kpsi = 3.07 GPa), which is PBT; no brush maker's figure for PET was found"),
        ("Polypropylene filament", f"E {PP['E']/1e9:g} GPa, 905 kg/m3", "datasheet",
         "a brush maker's fibre table: 740 kpsi (Carolina Brush)"),
        ("Filament tip on vinyl", f"{BRISTLE_MU} kinetic", "estimate", "steel wire on concrete 0.27"),
        ("Filament damping", f"{BRISTLE_ZETA:.0%} of critical", "estimate", "none found"),
        ("Filament damping of the chain's shortest wave", f"{BRISTLE_ZETA_TOP:.0%} of critical", "prior",
         "material damping growing with frequency; forces the same at 5, 20 and 50 %"),
    ]
    for name, t in (("Angle broom", ANGLE_BROOM), ("Push broom", PUSH_BROOM), ("Lobby broom", LOBBY_BROOM),
                    ("Counter brush", COUNTER_BRUSH)):
        rows.append((name, f"block {t['block'][0]*1e3:.0f} x {t['block'][1]*1e3:.0f} x {t['block'][2]*1e3:.0f} mm, "
                           f"bristles {t['bristle_len'][0]*1e3:.0f} to {t['bristle_len'][1]*1e3:.0f} mm, "
                           f"{t['filaments']} filaments of {t['filament_d']*1e3:.2f} mm {t['material']} in {t['clusters']} "
                           f"clusters of {t['nodes']} nodes, {t['mass']:.2f} kg"
                           + (f", {t['length']} m tall" if t is LOBBY_BROOM else ""),
                     "datasheet, derived, estimate" if t is COUNTER_BRUSH else "datasheet, estimate", t["source"]))
    rows += [
        ("Lobby dustpan", f"mouth 286 mm, pan 287 mm deep, 127 mm high, feather edge 0.5 mm, handle 0.76 m on a pivot, {LOBBY_PAN['mass']} kg",
         "datasheet, estimate", LOBBY_PAN["source"]),
        ("Hand dustpan", f"{HAND_PAN['pan'][0]*1e3:.0f} x {HAND_PAN['pan'][1]*1e3:.0f} mm open scoop, feather edge 0.5 mm, {HAND_PAN['mass']} kg",
         "datasheet, estimate", HAND_PAN["source"]),
        ("Barrel", f"{BARREL['d']} x {BARREL['h']} m, {BARREL['mass']} kg", "datasheet", BARREL["source"]),
        ("Pail; wastebasket", f"{PAIL['d']} x {PAIL['h']} m, {PAIL['mass']} kg; 0.37 x 0.26 x 0.38 m, {WASTEBASKET['mass']} kg", "derived; datasheet",
         PAIL["source"] + "; " + WASTEBASKET["source"]),
    ]
    for name, (mf, mu, mb, e, mr) in DEBRIS_CLASSES.items():
        rows.append((f"Debris: {name}", f"friction {mf} on vinyl, {mu} on debris and tools, {mb} on bristles; "
                                        f"restitution {e}; rolling resistance {mr}", "estimate", "varied in a probe"))
    rows += [
        ("Round container bottoms on the floor: resistance to rolling and to spinning",
         f"{BOTTOM_ROLLING:g} and {BOTTOM_TORSIONAL:g} of the bottom's radius (torque = that length x normal force)", "prior",
         "a knocked pail or barrel stands still within two or three seconds (everyday observation, not measured)"),
        ("Paper wad pressed by a rigid shape: stiffness of each sphere in contact, depth to which it yields", f"{SOFT['wad']:.0f} N/m, half the sphere's radius", "prior",
         "a crumpled sheet gives about a centimetre under a pinch of a few newtons (everyday observation, not measured)"),
        ("Rolling and spinning resistance of a broom's cap on its prongs", f"{CAP_ROLLING*1e3:g} mm", "prior",
         "a hanging broom pushed aside hangs still again within about a quarter of a minute (everyday observation, not measured)"),
        ("Kinetic over static friction of debris", f"{KINETIC_RATIO}", "estimate", ""),
        ("Fine dust: share of what lies in its path that one pass of a bank takes", f"{DUST['pickup']}", "derived",
         "first sweep of building sand in Tabia et al. (73.3 % of the area cleared)"),
        ("Fine dust: length over which the ridge a bank pushes trickles back", f"{DUST['leak_length']} m", "prior",
         "a fifth lost per metre; no measurement found; bracketed by runs at 1 m, 4.5 m and no loss"),
        ("Fine dust laid at the start: round a swept pile; open floor and by the door at closing time; variant dusty",
         f"{DUST_LOADS['pile_g']} g; {DUST_LOADS['open_g_m2']} and {DUST_LOADS['door_g_m2']} g/m2; {DUST_LOADS['dusty_g_m2']} g/m2", "authored",
         "dust found on the smooth floors of 51 French classrooms: 0.13 +- 0.16 g/m2, median 0.07 (Atmosphere 6(11):1676, 2015, Table 3): "
         "the mean on the open floor, the mean plus one spread for `dusty`, plus two by the door; a stock found on a floor, not a daily fall"),
        ("Fine dust: share of what lay in a place that a dry broom never moves from it", f"{100*DUST['residue_share']:g} %", "derived",
         "building sand after ten sweeps in the same study: 0.4 to 8.5 % of the area uncleared over its nine settings, median 2.9 %; "
         "the study measures cleared area and names no load"),
        ("Fine dust: lodging fraction, trickle time, tap threshold", f"{DUST['lodging']}, {DUST['tau_lodged']} s, {DUST['a_tap']} m/s2", "prior",
         "the study shows dust caught in bristles laid down again and gives no rate; bounded by its sweeps to 95 %"),
        ("Fine dust: bulk density, repose angle", f"{DUST['bulk_density']:.0f} kg/m3, {DUST['repose_deg']:.0f} deg", "datasheet, estimate",
         "IEC 62885-2 mineral test dust (density); the angle sets the line at a pan's lip and when a pan's film slides"),
        ("Fine dust: grids", f"{DUST['cell']*1e3:.0f} mm on the floor, {DUST['pan_cell']*1e3:.0f} mm in a pan", "engine", ""),
        ("Rest detector", f"no net motion above {REST_DISP*1e6:.0f} um for {REST_TIME} s", "engine",
         "50 um steepens a grit heap by 3 to 7 degrees"),
        ("Friction anchor pre-sliding distance", f"{SLIP_DISP*1e6:.0f} um", "authored", "micro-slip scale"),
        ("Rolling couple at a contact", "at most 1.4 mu_roll N r", "derived", "a lone sphere then decelerates at mu_roll g"),
        ("Depenetration speed; speed fault", f"{DEPENETRATION_SPEED} m/s; {MAX_DEBRIS_SPEED} m/s, and 8 m/s plus twice the fastest "
                                             "rigid shape while no bristle bank is at work", "engine", ""),
        ("A wheel passes over a piece on the floor whose top is lower than this share of the wheel's radius", "0.162", "prior",
         "4.9 mm under a caster, 8.1 mm under a drive wheel; not measured on a wheel; solver.ROLL_OVER"),
        ("A wheel pushes a taller piece along the floor where its rim faces along the floor by at least", "0.30", "engine",
         "it never presses a piece down and is never lifted; solver.WHEEL_FLAT"),
        ("Stiffness of a held piece under a rigid body, as a share of the body's effective mass over the tick squared", "0.1", "engine",
         "critically damped, shared among the pieces the body presses on; solver.Sweeper.kappa"),
        ("How deep a sphere may end a substep in the room's floor and in a rigid shape", "0.1 mm; 0.1 mm", "engine",
         "20 um was tried for the floor and took the floor's friction from a stack; solver.FLOOR_TOL, SHAPE_TOL"),
        ("A free rigid body is a fault of the world if it is faster than; if it is further outside the room than", "12 m/s; 0.30 m", "engine",
         "a fall from the ceiling gives 8 m/s; world.BODY_SPEED_FAULT, ROOM_MARGIN"),
        ("Fingertip pad: rubber over a hard body; the hard body's contact impedance",
         f"{PAD_RUBBER*1e3:g} mm; {PAD_CORE_SOLIMP[1]}", "repo; engine",
         "label_application's pad, rubber 2 MPa over 2 mm (an estimate there); MuJoCo's hardest impedance, because 0.95 to 4.3 kg of arm "
         "stands behind a fingertip on the floor and MuJoCo sizes the contact on the finger's 85 g; "
         "scene.pad_cores, world._floor_fault"),
        ("The evaluator's thresholds for swept mass", "85 to 98 % by task", "authored",
         "the one measurement found: a cleaning laboratory swept 197 to 298 g of soil from 11 m2 of vinyl tile with an angle broom into a "
         "dustpan and recovered 86.3, 90.8 and 90.5 % (cleanersolutions.org report 7396, 2012; opened 2026-10-11); "
         "that soil was finer than rice and cereal"),
    ]
    return rows
