"""Parameter ledger of concrete_finishing: every constant the simulation reads, with value, unit, label and basis.

Labels: opened (read on the source's page or in its PDF), derived (arithmetic on
other entries), estimate (judgement, with the reason), authored (a choice of this environment), repo (a value
qualified elsewhere in this repository), measured (a probe of this environment). The robot's own constants are
classroom_arrangement.params (repo), reused unchanged.

`python -m concrete_finishing.params` prints the table.
"""

import math

LEDGER = []


def _p(name, value, unit, label, basis):
    LEDGER.append(dict(name=name, value=value, unit=unit, label=label, basis=basis))
    return value


# ------------------------------------------------------------------ room (mechanical room B-012)
ROOM = _p("ROOM", (6.10, 4.88, 3.05), "m", "estimate", "a small pump room, 20 x 16 ft, deck at 10 ft")
WALL_T = _p("WALL_T", 0.194, "m", "opened", "8 in CMU, 7-5/8 in actual (standard nominal block; BIA TN 10 3/8 in joints)")
PARTITION_T = _p("PARTITION_T", 0.124, "m", "estimate", "3-5/8 in steel studs with one layer of 5/8 in board each side")
DOOR = _p("DOOR", (4.60, 5.51, 2.13), "m", "estimate", "hollow metal door 36 x 84 in in the south partition (x from, to, height)")
MU_FLOOR = _p("MU_FLOOR", 0.50, "-", "estimate", "steel-troweled sealed slab against painted steel and wood")

# ------------------------------------------------------------------ the new pump pad (station P)
PAD_CENTRE = _p("PAD_CENTRE", (3.20, 2.90), "m", "authored", "room layout")
PAD_SIZE = _p("PAD_SIZE", (1.219, 0.914), "m", "authored", "48 x 36 in inside the forms")
PAD_T = _p("PAD_T", 0.089, "m", "estimate", "a '4 in' pad formed with 2x4s on edge is their dressed 3-1/2 in")
FORM = _p("FORM", (0.038, 0.089), "m", "opened", "2x4 dressed 1-1/2 x 3-1/2 in (standard lumber size)")
KICKER = _p("KICKER", (0.150, 0.038, 0.089, 0.60), "m", "estimate", "2x4 blocks fastened to the floor every 0.6 m outside the forms")
FORM_DISTURB_N = _p("FORM_DISTURB_N", 150.0, "N", "estimate", "lateral load on a braced form that would move it")

# ------------------------------------------------------------------ the floor patch (station F)
PATCH_CENTRE = _p("PATCH_CENTRE", (4.60, 1.25), "m", "authored", "where the old pad was broken out")
PATCH_SIZE = _p("PATCH_SIZE", (0.400, 0.250), "m", "authored", "room layout")
PATCH_DEPTH = _p("PATCH_DEPTH", (0.025, 0.038), "m", "opened", "SikaQuick Pro 1000: at least 1/4 in, neat to 2 in; chipped 25-38 mm (estimate)")
PATCH_MARGIN = _p("PATCH_MARGIN", 0.20, "m", "derived", "slab modelled around the cut: the 406 mm wood float laid across the 250 mm patch overhangs the cut by 78 mm a side and spills 20 mm beyond its ends; a 50 mm margin put both spill ends off the modelled slab (R6)")
PATCH_SAWCUT = _p("PATCH_SAWCUT", 0.013, "m", "estimate", "square saw-cut edge 1/2 in deep (ICRI practice; not opened)")

# ------------------------------------------------------------------ the CMU infill (station W1) and the control joint (W2)
BLOCK = _p("BLOCK", (0.397, 0.194, 0.194), "m", "opened", "8 x 8 x 16 in CMU: 15-5/8 x 7-5/8 x 7-5/8 in actual")
JOINT_W = _p("JOINT_W", 0.0095, "m", "opened", "3/8 in mortar joint (BIA TN 10)")
INFILL_X = _p("INFILL_X", (3.89, 5.11), "m", "authored", "old duct opening, three blocks a course")
INFILL_Z0 = _p("INFILL_Z0", 0.813, "m", "authored", "sill at four courses (4 x 203 mm)")
INFILL_COURSES = _p("INFILL_COURSES", 4, "-", "authored", "813 mm opening")
JOINT_BED_DEPTH = _p("JOINT_BED_DEPTH", 0.020, "m", "estimate", "mortar joint depth modelled behind the block face (face shells 32 mm)")
CJ_Y = _p("CJ_Y", 2.00, "m", "authored", "control joint in the east wall")
CJ_W = _p("CJ_W", 0.013, "m", "opened", "1/2 in sealant joint (Sikaflex-1A: 2:1 width to depth)")
CJ_BACKER = _p("CJ_BACKER", 0.0085, "m", "derived", "backer rod face 8.5 mm behind the wall face: 6.5 mm of sealant at the tooled centre + 2 mm of concavity")
CJ_SEGMENT = _p("CJ_SEGMENT", (0.45, 2.00), "m", "authored", "task segment between two strips of painter's tape")

# ------------------------------------------------------------------ the gypsum partition (station D)
DW_PATCH = _p("DW_PATCH", (1.80, 1.25, 0.038), "m", "authored", "centre x, z of the mesh-patched 1-1/2 in hole")
DW_HOLES = _p("DW_HOLES", ((1.52, 1.42), (1.60, 1.42), (2.02, 1.08), (2.10, 1.08)), "m", "authored",
              "anchor holes of two removed conduit straps (x, z)")
DW_HOLE_D = _p("DW_HOLE_D", 0.006, "m", "estimate", "1/4 in plastic anchor holes")
BOARD_T = _p("BOARD_T", 0.0159, "m", "opened", "5/8 in gypsum board (standard)")
MESH_T = _p("MESH_T", 0.0005, "m", "estimate", "self-adhesive fibreglass mesh patch")
MESH_SOLID = _p("MESH_SOLID", 0.30, "-", "estimate", "homogenized fibre area/volume fraction; USG J780A requires compound forced through open mesh; not a measured weave")

# ------------------------------------------------------------------ concrete (pad, tub)
RHO_CONCRETE = _p("RHO_CONCRETE", 2350.0, "kg/m3", "estimate", "normal-weight concrete")
SLUMP = _p("SLUMP", 0.100, "m", "opened", "CIP 13: 3-5 in (75-125 mm) to avoid blisters")
TAU_B0 = _p("TAU_B0", RHO_CONCRETE/347.0*(300.0 - 100.0) + 212.0, "Pa", "derived", "NISTIR 6094 eq. 19 at slump 100 mm (opened)")
A_THIX = _p("A_THIX", 0.25, "Pa/s", "estimate", "inside Roussel's ordinary range: < 0.1 non-thixotropic, > 0.5 highly (opened)")
N_DEEP = _p("N_DEEP", 9.0, "-", "estimate", "deep circular punch bearing factor (Skempton; snippet/memory); 6 would move set times 8 %")
PR_IS = _p("PR_IS", 3.45e6, "Pa", "opened", "ASTM C403 initial set, 500 psi (Humboldt H-4134 manual)")
PR_FS = _p("PR_FS", 27.6e6, "Pa", "opened", "ASTM C403 final set, 4000 psi (Humboldt H-4137 manual)")
T_IS = _p("T_IS", 4.0*3600, "s", "estimate", "initial set 4 h at 70 F (Humboldt: first test at 3-4 h)")
T_FS = _p("T_FS", 6.0*3600, "s", "opened", "final set about 6 h at 70 F (MASCO table)")
SET_M = _p("SET_M", math.log(PR_FS/PR_IS)/math.log(T_FS/T_IS), "-", "derived", "power of time between the two set anchors (5.13)")
TAU_IS = _p("TAU_IS", PR_IS/N_DEEP, "Pa", "derived", "yield stress at initial set (383 kPa)")
MASCO = _p("MASCO", {30: 19.0, 40: 14 + 40/60, 50: 10 + 20/60, 60: 8.0, 70: 6.0, 80: 4.0, 90: 2 + 40/60, 100: 1 + 40/60},
           "F: h", "opened", "approximate final set against temperature (MASCO forming catalogue 2016)")
R_SURF = _p("R_SURF", (0.15, 0.30), "-", "estimate", "surface layer / bulk strength, floated -> densified")
T_SURF_MAX = _p("T_SURF_MAX", 0.003, "m", "estimate", "paste layer after floating (CIP 20 delamination 3-6 mm, CIP 1 top 1/4 in)")
T_SURF_STRUCK = _p("T_SURF_STRUCK", 0.0008, "m", "estimate", "paste layer of a struck-off surface")
MU_BULK = _p("MU_BULK", 50.0, "Pa s", "estimate", "plastic viscosity of slump concrete; only slow flow reads it")
MU_SURF = _p("MU_SURF", 5.0, "Pa s", "estimate", "plastic viscosity of the paste layer")
CEMENT = _p("CEMENT", 350.0, "kg/m3", "estimate", "cement content, for the water worked into the top 6 mm")
WC = _p("WC", 0.50, "-", "estimate", "water-cement ratio of the pad's mix")
DWC_DUST = _p("DWC_DUST", 0.05, "-", "estimate", "rise of the top 6 mm's w/c counted as a dusting risk (CIP 1 mechanism)")
AGG_EXPOSED = _p("AGG_EXPOSED", 0.30, "-", "estimate", "share of a struck-off surface with coarse aggregate showing")
EMBED_PASS = _p("EMBED_PASS", 8.3, "1/pass at 1 kPa", "estimate",
                "e-folds of aggregate exposure per pass of a bearing sole at 1 kPa (linear in pressure, 0.1-20 kPa): two "
                "passes of the 1.3 kg darby lying flat on its own weight (0.164 kPa) take the struck surface's 30 % below "
                "2 % (CIP 14: the darby 'embeds large aggregate'; two passes an estimate)")
TEX_PASS = _p("TEX_PASS", 2.2, "1/pass", "estimate",
              "e-folds of the surface texture toward the tool's own per pass of a bearing sole where the surface yields "
              "fully (what a 40 mm sliding length gives an 89 mm float bearing over its whole width)")
DENS_ENERGY = _p("DENS_ENERGY", 112.0, "J/m2", "estimate",
                 "e-fold surface densification work; equivalent to 56 kPa plastic work density over 2 mm; test 56-224 J/m2, not a CIP/ASTM measurement")

# ------------------------------------------------------------------ bleeding and air
BLEED_MEAN = _p("BLEED_MEAN", 0.15, "kg/m2/h", "estimate", "M6b: 0.15 gives CIP 14's order (sheen gone after the boot holds); FHWA paving eq. 0.09, general use 0.5-1.5 (opened)")
BLEED_END_TAU = _p("BLEED_END_TAU", 1.0e4, "Pa", "estimate", "bleeding ends when the bulk reaches 10 kPa (about 110 min at 21 C)")
SHEEN_W = _p("SHEEN_W", 2.0e-5, "m", "estimate", "a film thicker than 0.02 mm reads as sheen (render and evaluator)")
AIR = _p("AIR", dict(still=(20.0, 50.0, 0.2), door=(20.0, 50.0, 1.0), fan=(20.0, 40.0, 2.5)), "C, %, m/s", "estimate",
         "room air at the slab: still, door open, a pedestal fan left running")
BLISTER_Q = _p("BLISTER_Q", 2.0e-4, "m", "estimate", "trapped bleed water that raises a blister (CIP 13 sizes 12-100 mm, skin 3 mm)")

# ------------------------------------------------------------------ other pastes
REPAIR = _p("REPAIR", dict(rho=2051.0, tau0=2000.0, t_is=50*60.0, mu=20.0, bleed=0.05), "kg/m3, Pa, s, Pa s, kg/m2/h",
            "scenario blend / estimate", "generic repair mortar: Quikrete density 2051 kg/m3 combined with SikaQuick 35-85 min set and 30 min finishing; not a product-specific material")
MORTAR = _p("MORTAR", dict(rho=2000.0, tau0=1000.0, tau_ref=8000.0, t_ref=35*60.0, m=3.0, mu=10.0), "kg/m3, Pa, Pa, s, -, Pa s",
            "estimate", "Type N mortar stiffening by suction: thumbprint hard 5-20 kPa about 30-50 min after laying")
THUMBPRINT = _p("THUMBPRINT", (5.0e3, 2.0e4), "Pa", "estimate", "a thumb (20 N on 2 cm2) prints without pick-up (BIA TN 7B rule)")
MORTAR_COMPACT = _p("MORTAR_COMPACT", 1.0, "-", "model definition",
                    "normalized mobilized surface fabric, NOT a lost volume or air fraction; compact=1 is saturated surface consolidation, task requires >=0.5")
MORTAR_LAYER = _p("MORTAR_LAYER", 0.002, "m", "estimate",
                  "surface layer rearranged by tooling; order of mortar sand grains; sensitivity 1-3 mm, not whole 20 mm joint depth")
MORTAR_STRAIN = _p("MORTAR_STRAIN", 6.0, "-", "estimate",
                   "characteristic accumulated shear strain for surface fabric rearrangement; W*=tau_bulk*layer*strain; sensitivity 3-12")
MORTAR_HARDEN = _p("MORTAR_HARDEN", 0.30, "-", "estimate",
                   "maximum surface yield increase from mobilized fabric; bounded sensitivity 0.15-0.45; not a compressive-strength or permeability prediction")
SEALANT = _p("SEALANT", dict(rho=1400.0, tau0=600.0, mu=50.0, skin=2*3600.0), "kg/m3, Pa, Pa s, s", "estimate / opened",
             "non-sag bound 180 Pa for 13 mm (derived); skin 2 h (Tremco Dymonic 100)")
COMPOUND = _p("COMPOUND", dict(rho=1000.0, tau0=700.0, mu=6.0), "kg/m3, Pa, Pa s", "opened / estimate",
              "3M US 11,267,758: 693-753 Pa (Ex. 31, 20), 5.8 Pa s at 1.5 kPa; density estimate")

# ------------------------------------------------------------------ the tool-paste law
NC = _p("NC", 2.0 + math.pi, "-", "opened", "Prandtl's strip factor; Green's combined-load factor (Femmam et al. 2018 Table 1)")
TACK = _p("TACK", (700.0, 3900.0), "Pa", "derived / estimate", "graph force anchors 0.85-4.9 N divided by pi*(0.020 m)^2, rounded; cellulose-modified cement-lime mortar, initial 3 mm thickness, 10-500 micrometre/s separation (Le Thanh Hai & Phan Van Tien 2014); cross-material extrapolation estimated")
TACK_GAP = _p("TACK_GAP", 0.003, "m", "estimate", "linear detachment distance, not measured: Le Thanh Hai & Phan Van Tien's 3 mm is INITIAL specimen thickness, not detachment length; cross-material multipliers and robot lift-rate extrapolation are estimates")
M_INTERFACE = _p("M_INTERFACE", (0.05, 0.30), "-", "estimate", "interface shear / surface yield stress, wet film -> stiff (open)")
PLOUGH = _p("PLOUGH", 3.0, "-", "estimate", "work per volume ploughed by a sole's edge, in units of the yield stress")
PAD_RUBBER_E = _p("PAD_RUBBER_E", 3.6e6, "Pa", "repo / estimate",
                  "a fingertip pad's rubber (Shore A 60, E 3.6 MPa: sewing.params; syringe_handling 2.5 MPa): a rubber edge "
                  "pressed against paste whose plough stress PLOUGH tau exceeds about its modulus deforms over it instead "
                  "of cutting it (otherwise fingertips shave 7 h concrete)")
K_CONTACT = _p("K_CONTACT", 0.15, "-", "authored", "paste contact stiffness = this x tool mass / dt^2 (explicit stability; M2 scheme B)")
C_CONTACT = _p("C_CONTACT", 0.30, "-", "authored", "paste contact damping = this x tool mass / dt")
ROLL_LIFT = _p("ROLL_LIFT", dict(float=0.003, trowel=0.001, knife=0.001), "m", "estimate", "how far a bevelled or turned edge lifts the roll")
DROP_STICK = _p("DROP_STICK", 6.0, "-", "estimate", "a load stays on a lifted blade while rho g h < tau / this")

# ------------------------------------------------------------------ tools (Marshalltown pages)
TOOLS = _p("TOOLS", dict(
    float=dict(part="145D", blade=(0.406, 0.0794, 0.006), mass=0.50, handle_h=0.060, handle=(0.230, 0.016)),
    wood_float=dict(part="144D", blade=(0.406, 0.0889, 0.019), mass=0.28, handle_h=0.055, handle=(0.200, 0.016)),
    trowel=dict(part="PB64D", blade=(0.356, 0.1016, 0.0015), mass=0.45, handle_h=0.065, handle=(0.125, 0.016)),
    darby=dict(part="4626D", blade=(0.914, 0.085, 0.008), mass=1.30, handle_h=0.055, handle=(0.18, 0.016)),
    edger=dict(part="136D", blade=(0.152, 0.0762, 0.002), mass=0.25, handle_h=0.050, handle=(0.110, 0.016),
               radius=0.0095, lip=0.0127),
    margin=dict(part="52D", blade=(0.127, 0.0508, 0.0015), mass=0.12, handle=(0.120, 0.014)),
    jointer=dict(part="BJCC12", rod=(0.0127, 0.150), mass=0.12, handle=(0.110, 0.014)),
    sealant_tool=dict(part="5/8 in tooling spatula (Albion classic widths)", blade=(0.0159, 0.100, 0.003), mass=0.03,
                      handle=(0.080, 0.008), tip_r=0.011),
    joint_knife=dict(part="JK886D", blade=(0.152, 0.070, 0.0006), mass=0.16, handle=(0.110, 0.015)),
    putty_knife=dict(part="PK878D", blade=(0.051, 0.070, 0.0006), mass=0.08, handle=(0.100, 0.013)),
    penetrometer=dict(part="H-4134", shaft=(0.0032, 0.043), mass=0.227, body=(0.135, 0.0095)),
), "m, kg", "opened / estimate", "sizes from the maker's pages; masses = shipping weight less 10-25 % (estimate); the "
   "penetrometer from the Humboldt H-4134 manual (opened: needle 1/20 sq in, body 3/4 in x 7 in, 8 oz); the sealant "
   "tool's 5/8 in width is the Albion classic spatula size next above the 1/2 in joint (opened: chemical-concepts.com, "
   "sizes 3/8 to 1-1/2 in), so it rides both joint edges; its rounded nose, 11 mm in radius across the blade, is an "
   "estimate that stands 2.1 mm into a 13 mm joint between the edges ('tooled flush to slightly concave', NIH ORF "
   "News to Use, March 2021, opened)")
CARRY_CAP = _p("CARRY_CAP", dict(sealant_tool=1.5e-5, putty_knife=2.5e-5, joint_knife=1.0e-4, margin_trowel=1.5e-4), "m3",
               "estimate", "paste a loaded blade holds (the sealant tool: a gob about 30 mm across riding ahead of its nose; "
                           "a knife's or margin trowel's face); what a sole cuts beyond it runs off over the joint's edges or "
                           "the blade's ends. The tool-sealant bead displaces about 45 ml per metre when tooled (B7), so a "
                           "finisher wipes the tool every 0.3 m")
RAG = _p("RAG", (0.25, 0.25, 0.012), "m", "estimate", "a folded shop towel on the cart top")
RAG_WIPE = _p("RAG_WIPE", 0.05, "m", "estimate", "rubbing a loaded blade on the rag over this distance leaves 1/e of its load on it")
EDGER_TRACK = _p("EDGER_TRACK", 0.0012, "m", "estimate", "the edger's flat pressed this far below the field leaves a step at its track's inner edge (floated out after)")
FLEX_K = _p("FLEX_K", 2800.0, "N/m", "estimate", "6 in flex joint knife at its edge: 0.5 mm spring steel, 70 mm free")
SKIN_FACTOR = _p("SKIN_FACTOR", 3.0, "-", "estimate", "minimal post-skin surface resistance multiplier, not polymer calibration; tooling tears the skin and cannot establish new wall wetting")

# ------------------------------------------------------------------ carts and props
CART = _p("CART", (0.86, 0.46, 0.84, 15.0), "m, kg", "estimate", "three-shelf utility cart (length, width, top height, mass)")
TUB = _p("TUB", (0.90, 0.60, 0.20), "m", "estimate", "mixing tub with leftover concrete")
HAWK = _p("HAWK", 0.330, "m", "opened", "Marshalltown 1D aluminium hawk 13 x 13 in")
MORTAR_BOARD = _p("MORTAR_BOARD", (0.749, 0.737), "m", "opened", "Marshalltown mortar board 29 x 29-1/2 in")

# ------------------------------------------------------------------ evaluator (authored criteria)
EVAL = _p("EVAL", dict(
    gap_mm=3.0, straightedge_m=1.2, plane_mm=3.0, band_m=0.03, agg_max=0.02, agg_share=0.98, seal_max=0.2,
    radius_tol_mm=1.5, edge_share=0.95, gouge_mm=2.0, field_change_mm=1.0, float_tex=(1.5e-4, 6.0e-4), tex_share=0.95,
    track_step_mm=1.0, dens_min=0.8, trowel_tex=5.0e-5, delam_max=0.01, burn_max=0.05,
    lip_mm=3.0, lip_hard_mm=6.4, patch_mean_mm=1.5, fill_share=0.98, smear_mm=0.2, mortar_window=(5.0e3, 2.0e4),
    concave_mm=(1.4, 2.8), joint_share=0.95, smear_cm2_m=2.0, sealant_centre_mm=(0.5, 2.5), sealant_min_mm=6.0,
    contact_share=0.98, flush_mm=0.3), "-", "authored", "authored criteria")

# ------------------------------------------------------------------ numerics
SLOW_DT = _p("SLOW_DT", 0.05, "s", "authored", "material clock step while the robot works (strength, bleeding, slow flow)")
WAIT_DT = _p("WAIT_DT", 1.0, "s", "authored", "material clock step of a wait command")
WAIT_MAX = _p("WAIT_MAX", 3600.0, "s", "authored", "longest wait command")
REST_SPEED = _p("REST_SPEED", 1.0e-3, "m/s", "authored", "a wait needs every free body slower than this")


def table():
    return [dict(r) for r in LEDGER]


if __name__ == "__main__":
    for r in LEDGER:
        v = r["value"]
        print(f"{r['name']:16s} {str(v)[:70]:70s} {r['unit']:14s} {r['label']:18s} {r['basis'][:90]}")
