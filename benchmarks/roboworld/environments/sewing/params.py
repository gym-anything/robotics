"""Parameter ledger. Every physical value the world reads is defined here with its unit, provenance label and
source (labels: read, search, literature, repo, measured, derived, fitted (chosen so a model
output matches a target, not predicted), estimate, authored, open).
PARAMETERS.md is generated from this module (`python -m sewing.params`), and tests/test_params.py fails when the
generated table is stale.
"""

import math

IN = 0.0254
GF = 9.80665e-3          # N per gram-force

# ---------------------------------------------------------------------------------------------------- ledger
LEDGER = {}


def P(key, value, unit, label, source, note=""):
    LEDGER[key] = dict(value=value, unit=unit, label=label, source=source, note=note)
    return value


# ------------------------------------------------------------------------------------------- machine (Juki A)
MAX_SPM = P("machine.max_spm", 5000.0, "stitch/min", "read",
            "Juki DDL-9000C-SMS manual: 5,000 sti/min for stitch length <= 4 mm")
MAX_SPM_LONG = P("machine.max_spm_long", 4000.0, "stitch/min", "read", "Juki DDL-9000C: 4,000 sti/min above 4 mm")
STITCH_MAX = P("machine.stitch_max", 5.0e-3, "m", "read", "Juki DDL-9000C: max stitch length 5 mm")
STROKE = P("machine.needle_bar_stroke", 30.7e-3, "m", "read", "Juki DDL-9000C/DDL-8700 medium: needle bar stroke 30.7 mm")
CRANK_RATIO = P("machine.rod_over_crank", 3.5, "-", "estimate", "slider-crank connecting rod / crank radius (open)")
POINT_BDC = P("machine.needle_point_at_bdc", -12.0e-3, "m", "derived",
              "Pfaff adjustment manual (pickup 1.8 mm past BDC, eye 0.8 mm below hook point) and Sofronova 2025 "
              "(plate to hook 5-8 mm): point 12 mm below the plate at BDC")
PICKUP_RISE = P("machine.loop_pickup_rise", 2.0e-3, "m", "read", "loop pickup 1.8 mm (Pfaff) to 2.0-2.4 mm past BDC")
HOOK_CLEARANCE = P("machine.hook_clearance", 0.07e-3, "m", "read", "Juki DDL-8700 manual: needle-hook 0.04-0.1 mm")
DOG_HEIGHT = P("machine.feed_dog_height", 0.80e-3, "m", "read", "Juki DDL-9000C: feed dog 0.60-1.20 mm, standard 0.80")
DOG_BELOW = P("machine.feed_dog_below", 0.50e-3, "m", "read", "Juki: >= 0.5 mm clearance below the throat plate")
DOG_L = P("machine.feed_dog_length", 10.0e-3, "m", "estimate", "4-row feed dog footprint along the feed (measure)")
DOG_W = P("machine.feed_dog_row_width", 3.5e-3, "m", "estimate", "teeth rows each side of the needle (measure)")
DOG_TOOTH_GRIP = P("machine.dog_tooth_grip", 0.0, "Pa", "open",
                   "the teeth's engagement: a feed dog row holds the cloth its tooth tips are in by mu N plus this times "
                   "the cloth's area on it, whatever the load (the tips are in the weave before the load arrives). 0: "
                   "smooth blocks with the effective friction mu_dog")
DOG_GAP = P("machine.feed_dog_needle_gap", 1.0e-3, "m", "estimate", "half gap between the rows around the needle")
PRESSER_F = P("machine.presser_force", 40.0, "N", "read",
              "Juki DDL-9000C: panel 100 (factory) = approx. 40 N (S type); range 10-60 N")
PRESSER_MASS = P("machine.presser_mass", 0.12, "kg", "estimate", "presser bar + foot (open)")
FOOT_LIFT_HAND = P("machine.foot_lift_hand", 5.5e-3, "m", "read", "Juki DDL-9000C: 5.5 mm by hand",
                   "not consumed: the hand-lifter lever is not built; the control box's auto lifter is")
FOOT_LIFT_AUTO1 = P("machine.foot_lift_auto1", 5.0e-3, "m", "read", "Juki DDL-9000C auto lifter 1st step 5.0 mm")
FOOT_LIFT_AUTO2 = P("machine.foot_lift_auto2", 8.5e-3, "m", "read", "Juki DDL-9000C auto lifter 2nd step 8.5 mm")
FOOT_LIFT_SPEED = P("machine.foot_lift_speed", 0.2, "m/s", "estimate",
                    "auto-lifter travel: 8.5 mm in ~40 ms (solenoid; open). The lifter is a stop the foot rests on")
FOOT_LEN = P("machine.foot_sole_length", 26.0e-3, "m", "estimate", "standard hinged foot sole (measure from drawing)")
FOOT_W_RIGHT = P("machine.foot_toe_right", 4.5e-3, "m", "estimate", "right toe width")
FOOT_W_LEFT = P("machine.foot_toe_left", 6.0e-3, "m", "estimate", "left toe width")
FOOT_SLOT = P("machine.foot_needle_slot", 2.0e-3, "m", "estimate", "needle slot between the toes")
FOOT_BACK = P("machine.foot_behind_needle", 11.0e-3, "m", "estimate", "sole length behind the needle (+y side)")
PLATE_GAP_EDGE = P("machine.plate_hole_edge", 0.7e-3, "m", "read",
                   "Pfaff manual: needle centre 0.7 mm from the needle-hole edge in the sewing direction")
NEEDLE_TENSION_REF = P("machine.needle_tension_ref", 0.59, "N", "read",
                       "Juki DDL-9000C: active tension 60 = 0.59 N with spun #60 (reference)")
BOBBIN_TENSION = P("machine.bobbin_tension", 0.25, "N", "read",
                   "longarm bobbin 18-22 g (Linda's/Towa); apparel 20-40 gf [literature]")
TRIM_TAIL = P("machine.trim_tail", 3.0e-3, "m", "read", "Juki DDL-9000C: remaining thread about 3 mm")
SERVO_TAU_MAX = P("machine.servo_peak_torque", 2.5, "N m", "estimate",
                  "400 W servo: 0.76 N m continuous at 5,000 rpm (derived); peak 2-3 N m (open)",
                  "not consumed: the shaft follows its speed command; a 6 N puncture on the 15.35 mm crank needs "
                  "~0.09 N m, 4 % of this, so the shaft does not sag")
SHAFT_INERTIA = P("machine.shaft_inertia", 1.0e-3, "kg m^2", "estimate", "reflected main-shaft inertia (open)",
                  "not consumed: see machine.servo_peak_torque")
SPEED_RAMP = P("machine.speed_ramp", 25000.0, "stitch/min per s", "estimate",
               "soft start: 0 -> 5,000 sti/min in 0.2 s (open; Juki has a soft-start function)")

# ------------------------------------------------------------------------------------------------- needle
NEEDLE_E = P("needle.youngs_modulus", 205e9, "Pa", "literature", "carbon steel")
NEEDLE_SIGMA_Y = P("needle.yield_stress", 1.5e9, "Pa", "estimate", "hardened needle steel (open)")
NEEDLE_ALPHA = P("needle.stiffness_factor", 4.6, "-", "derived",
                 "Tanaka, Yasumori, Kamata 1997 (Nm 65: 0.38 N -> 52 um) against the free cantilever at L = 15 mm "
                 "(needle bar 15 mm above the plate at BDC, Pfaff); swept 1-4.6 in E6")
NEEDLE_L_BDC = P("needle.clamp_to_plate_bdc", 15.0e-3, "m", "read", "Pfaff manual: needle bar 15 mm above the plate at BDC")
PEN_F_QS = P("needle.penetration_quasistatic", 1.56, "N", "derived",
             "fit to Chmielowiec 1993 (1.7 N at 200 rpm, 5.45 N at 5,500 rpm, inertia removed)")
PEN_C_V = P("needle.penetration_speed_coeff", 0.44, "N s/m", "derived", "same fit")
PEN_POINT_LEN = P("needle.point_length", 1.5e-3, "m", "estimate", "depth over which penetration force rises")
PEN_SHAFT_FRAC = P("needle.shaft_resistance_fraction", 0.6, "-", "open", "0.4-1.0 (Chmielowiec: shank peak above eye)")
PEN_UP_FRAC = P("needle.withdrawal_fraction", 0.3, "-", "open", "0.2-0.5 of F_qs drags the ply up on withdrawal")
PIN_STIFF = P("needle.hole_pin_stiffness", 2.0e4, "N/m", "estimate", "local hole stiffness in series with needle bending")
MISS_LOOP = P("needle.missed_loop_deflection", 0.30e-3, "m", "estimate", "about one thread diameter of loop protrusion")
TIMING_DEFL = P("needle.feed_direction_deflection", 0.50e-3, "m", "estimate", "loop timing tolerance")
FLAG_LIFT = P("needle.flagging_lift", 1.0e-3, "m", "estimate", "half the 1.8-2.4 mm loop-forming rise")
EDGE_LIG = P("needle.edge_ligament", 1.0e-3, "m", "estimate", "about three yarn spacings")

# ------------------------------------------------------------------------------------------------- thread
THREAD_BREAK = P("thread.break_tex30", 14.9, "N", "read", "Coats Epic 100 (Tex 30): 1,490 cN")
THREAD_EA = P("thread.EA", 64.0, "N", "derived", "Rengasamy & Wesley: Tex 30 50.7 cN/tex at 23.7 %")
THREAD_SEW_LOSS = P("thread.sewing_strength_loss", 0.40, "-", "abstract", "Midha et al. 2013: ~40 % loss through eye/hook")
THREAD_LOOP = P("thread.loop_factor", 0.68, "-", "read", "Rengasamy & Wesley: loop strength 0.68 x straight")
THREAD_DIAM = P("thread.diameter", 0.40e-3, "m", "read", "spun polyester 29.7 tex: 0.40 mm optical diameter (rendered "
                "only)")
THREAD_CAPSTAN_MU = P("thread.capstan_mu", 0.24, "-", "read/open", "thread-metal 0.18 (Rengasamy); thread-fabric 0.18-0.30")
NEEDLE_PATH_LOSS = P("thread.needle_path_loss", NEEDLE_TENSION_REF/BOBBIN_TENSION, "-", "derived",
                     "the needle thread's set tension reaches the interlock divided by this (friction of its path "
                     "through the eye and the hole): the default dial pair sews balanced stitches (FED-STD-751a, "
                     "machine defaults), and an interlock inside the stack is in balance when the two threads pull it "
                     "equally, so the factor is the default ratio 0.59/0.25 = 2.36 (e^(mu theta) with theta ~205 deg "
                     "at mu 0.24); the seam's needle-thread elements carry T_set,n / 2.36")
TENSION_SPEED = P("thread.dynamic_factor_slope", 1.68, "-", "derived",
                  "Chmielowiec: needle-thread peaks 47 -> 126 cN from 200 -> 5,500 rpm; g(n) = 1 + 1.68 (n-200)/5300")
BIND_MAX_GAP = P("thread.bind_slack", 0.05e-3, "m", "estimate", "interlock: plies held to the stack thickness + 0.05 mm")


def thread_set_tension(static, spm, fabric_factor=1.0):
    """Stitch set tension (N): static dial tension times the dynamic factor g(n)."""
    g = 1.0 + TENSION_SPEED*max(0.0, spm - 200.0)/5300.0
    return static*(1.0 + (g - 1.0)*fabric_factor)


# ------------------------------------------------------------------------------------------------- fabrics
def _toe(emt, lt=0.6, toe_load_frac=0.1, load=490.3):
    """Bilinear tension law (K0 toe, K1 beyond e_t) passing through the KES point (EMT, 490 N/m) with the KES
    linearity LT: r = e_t/EMT = 1 + f - LT for a toe carrying the fraction f of the KES load."""
    r = 1.0 + toe_load_frac - lt
    et = r*emt
    nt = toe_load_frac*load
    K0 = nt/et
    K1 = (load - nt)/(emt - et)
    return K0, K1, et


_PAL_BW_BF = 0.0609/0.0319     # warp/weft bending ratio of Pal et al.'s cotton shirting (KES B), used to split muslin's mean
_mus_K0w, _mus_K1w, _mus_etw = _toe(0.066, 0.597)

# Bending hysteresis (yarn-crossing friction): KES-FB2 2HB is the width of the moment-curvature loop, HB = 2HB / 2 the
# friction moment per unit width (N m/m = N). 1 gf cm/cm = 9.807e-3 N.
KES_2HB_PER_B = P("fabric.kes_2hb_over_b_woven", 124.15, "1/m", "derived",
                  "Ramaiah et al. KES, woven fabrics C and D: 2HB/B = 0.0277/0.0270 and 0.1820/0.1249 (gf cm/cm per "
                  "gf cm2/cm) = 1.03 and 1.46 /cm, mean 1.24 /cm; used for the fabrics without a measured 2HB")
BEND_STICK_CURV = P("fabric.bend_stick_curvature", 25.0, "1/m", "estimate",
                    "curvature span of the bending-friction micro-slip: the Iwan elements slip at 1/8, 3/8, 5/8, 7/8 "
                    "of it after a reversal (0.25 /cm). Read off the rounded reversal of the KES-FB2 loop in Ramaiah "
                    "et al. Fig. 11 (completes over ~0.3-0.6 /cm; image resolution limits the reading); any value "
                    "below 0.75 /cm reproduces 2HB at +-1 /cm")
_mus_HB = 0.5*0.0277*9.807e-3
# Tensile and shear hysteresis (yarn-crossing friction in the plane). KES-FB1 RT is the fraction of the
# tensile work (to 500 gf/cm) that the unloading returns; 2HG and 2HG5 the widths of the shear-force loop at 0.5 and 5
# deg (gf/cm).
TENSILE_RT = P("fabric.tensile_RT", 0.4994, "-", "read",
               "KES tensile resilience of the plain cotton (Ramaiah et al. fabric C, RT 49.94 %): friction elements "
               "carry phi = (1 - RT)/(1 + RT) of the elastic tension once they slip (Coulomb friction under the "
               "crossings' pressure, which grows with the tension), so loading follows the measured law and unloading "
               "returns RT of the work")
TENS_STICK_STRAIN = P("fabric.tensile_stick_strain", 3.0e-3, "-", "estimate",
                      "strain span of the tensile-friction micro-slip after a reversal (the Iwan elements slip at 1/8, "
                      "3/8, 5/8, 7/8 of it): the steep start of a KES-FB1 unloading curve, 0.6 % for a full reversal "
                      "(not measured here; open)")
SHEAR_2HG = P("fabric.shear_2HG", 1.24*GF*100.0, "N/m", "read",
              "KES shear hysteresis at 0.5 deg of the plain cotton (Ramaiah et al. fabric C, 2HG 1.24 gf/cm)")
SHEAR_2HG5 = P("fabric.shear_2HG5", 1.79*GF*100.0, "N/m", "read",
               "KES shear hysteresis at 5 deg (Ramaiah et al. fabric C, 2HG5 1.79 gf/cm): the loop of one +-8 deg test "
               "is wider at 5 deg than at 0.5 deg (the crossings jam as the net shears)")
SHEAR_HG1 = P("fabric.shear_HG_slope", 0.5*(SHEAR_2HG5 - SHEAR_2HG)/math.radians(4.5), "N/m/rad", "derived",
              "the shear friction's slip force per unit width HG0 + HG1 |gamma| through HG = 2HG/2 at 0.5 deg and "
              "2HG5/2 at 5 deg (linear between the two KES readings; KES measures them under 10 gf/cm)")
SHEAR_HG0 = P("fabric.shear_HG0", 0.5*SHEAR_2HG - SHEAR_HG1*math.radians(0.5), "N/m", "derived",
              "the shear friction's slip force at zero shear (see fabric.shear_HG_slope)")
SHEAR_STICK_ANGLE = P("fabric.shear_stick_angle", math.radians(0.5), "rad", "estimate",
                      "angle span of the shear-friction micro-slip after a reversal (the Iwan elements slip at 1/8, "
                      "3/8, 5/8, 7/8 of it): a reversal rounded over ~1 deg of a +-8 deg KES-FB1 shear loop (not "
                      "measured here; open)")
# Shear locking: the shear law above is calibrated over KES-FB1's
# +-8 deg and its force (G sin 2 gamma / 2 for straight yarns) falls past 45 deg, while a woven fabric's rises once the
# yarns of a set touch (jamming); beside the foot and under a still hand the plies shear to 72-89 deg (E19). Past the
# locking angle the shear energy gains 1/2 k_lock A (gamma - gamma_lock)^2 on the shear angle gamma itself
# (atan2(|F a_w . F a_f|, |F a_w x F a_f|)): its stress k_lock (gamma - gamma_lock) keeps rising to the collapse at
# 90 deg. A form on sin gamma loses its stress toward 90 deg (345 N/m at most).
SHEAR_LOCK_DEG = P("fabric.shear_lock_angle", 50.0, "deg", "derived",
                   "the yarns of a set touch when their spacing p cos(gamma) reaches the yarn width w: gamma_lock = "
                   "arccos(w / p). For a plain cotton of 137.8 g/m2 at 23.6 x 23.2 threads/cm (Naujokaityte, "
                   "Strazdiene, Domskiene, Fibres & Textiles in Eastern Europe 16(5) 2008, Table 1; the muslin's own "
                   "thread count is not in Ramaiah's table): p 0.427 mm, a ~29 tex yarn (the areal density over the "
                   "thread count, crimp neglected) 0.20 mm across as a round packed yarn (fibre 1.52 g/cm3, packing "
                   "0.6), 0.26-0.30 mm flattened in the cloth (estimate) -> 45-62 deg; 50 taken. That fabric's own "
                   "bias-extension curve runs to 37.5 deg without locking (their Fig. 9)")
SHEAR_LOCK_K = P("fabric.shear_lock_stiffness", 2850.0, "N/m per rad", "prior",
                 "tangent shear stiffness past locking, read from the one cotton plain weave found tested past it: "
                 "Dridi, 'Essais de caracterisation des structures tissees', thesis, INSA Lyon 2010, picture frame, "
                 "Fig. 2.52-2.53 (100 % cotton plain weave, 24 x 18 threads/cm, 216 g/m2): the shear force per unit "
                 "length rises 2,500-3,200 N/m per rad between 40 and 57 deg and holds that slope to the test's end "
                 "(no steeper knee); the mean taken. A heavier cloth than the 125 g/m2 muslin and a pre-tensioned "
                 "frame (the thesis's second frame reads ~15 % softer), so an upper prior for the muslin; values "
                 "read off the plots, 5-15 % reading error")

# Compression (KES-FB3 to 50 gf/cm2): a fabric's `thickness` is its thickness Tm at that pressure; with the free
# thickness T0 and the linearity LC it fixes the exponential pressure law of cloth.compression_law.
KES_PM = P("fabric.kes_compression_pm", 4903.3, "Pa", "read", "KES-FB3 compression test maximum, 50 gf/cm2")
COMP_LC = P("fabric.compression_LC", 0.242, "-", "read",
            "KES linearity of compression of the plain cotton (Ramaiah et al. fabric C); with WC 0.181 gf cm/cm2 it "
            "gives Tm = T0 - 2 WC/(LC Pm) = 0.311 mm; the default curve shape for fabrics without KES compression data")
_mus_Tm = 0.610e-3 - 2*0.181/(0.242*50.0)*1e-2
COMP_RC = P("fabric.compression_RC", 0.378, "-", "read",
            "KES compression resilience of the plain cotton (Ramaiah et al. fabric C, RC 37.84 %): the fraction of the "
            "compression work the unloading returns; the compression set keeps (1 - RC) of a surface's largest "
            "compression")
SET_RECOVERY_TAU = P("fabric.compression_set_recovery", 1.0, "s", "estimate",
                     "time constant of an unloaded surface's recovery from its compression set (thickness recovery of a "
                     "flattened fabric: not measured here; open)")
SET_RECOVERY_PA = P("fabric.compression_set_unloaded", 49.03, "Pa", "read",
                    "a surface counts as unloaded below KES's 0.5 gf/cm2 reference pressure (T0 is quoted there)")
_sh_HB = 0.5*KES_2HB_PER_B*0.5*(0.0609 + 0.0319)*9.807e-5
_den_HB = 0.5*KES_2HB_PER_B*0.5*(2.5e-4 + 0.6e-4)
FABRICS = {
    # Unbleached cotton plain weave, the toile/mockup cloth: Ramaiah et al. KES of unfinished cotton plain
    # (125.2 g/m2, 0.61 mm, LT 0.597, WT 9.91, EMT ~6.6 %, B 0.0270 gf cm2/cm, G 0.52 gf/cm deg, MIU 0.191).
    "muslin_125": dict(
        label="unbleached cotton muslin, 125 g/m2",
        areal_density=0.1252, thickness=_mus_Tm, thickness_free=0.61e-3, comp_LC=COMP_LC, comp_RC=COMP_RC,
        K0w=_mus_K0w, K1w=_mus_K1w, etw=_mus_etw, K0f=_mus_K0w, K1f=_mus_K1w, etf=_mus_etw, kc=0.08,
        G=0.52*0.9807*180/math.pi, Bw=2*2.65e-6*_PAL_BW_BF/(1 + _PAL_BW_BF), Bf=2*2.65e-6/(1 + _PAL_BW_BF), HB=_mus_HB,
        bend_stick_curv=BEND_STICK_CURV,
        tensile_RT=TENSILE_RT, tens_stick_strain=TENS_STICK_STRAIN, HG0=SHEAR_HG0, HG1=SHEAR_HG1,
        shear_stick_angle=SHEAR_STICK_ANGLE, shear_lock_deg=SHEAR_LOCK_DEG, shear_lock_k=SHEAR_LOCK_K,
        air_perm=0.4, mu_steel=0.15, mu_cloth=0.65, mu_table=0.30, mu_pad=0.80, mu_dog=1.0,
        sources=dict(areal_density="read (Ramaiah)",
                     thickness="derived (Ramaiah KES: T0 0.610 mm, WC 0.181 gf cm/cm2, LC 0.242 -> Tm 0.311 mm)",
                     tension="derived from KES EMT/LT (Ramaiah) with a 10 % toe-load fraction (estimate)",
                     G="read (Ramaiah) converted",
                     B="read mean 2.65e-6 split by Pal's warp/weft ratio 0.0609/0.0319 = 1.91 (estimate): 3.48e-6 / "
                       "1.82e-6",
                     HB="read (Ramaiah fabric C: 2HB 0.0277 gf cm/cm)",
                     tensile_RT="read (Ramaiah fabric C: RT 49.94 %)",
                     HG="derived from KES 2HG 1.24 and 2HG5 1.79 gf/cm (Ramaiah fabric C)",
                     friction="read ranges (cotton-steel 0.10-0.21, cloth-cloth 0.59-0.82); table 0.30, dog 1.0 open; "
                              "pad 0.80 is the robot catalog's generic link value (repo); found since: smooth ABS on "
                              "cotton shirting 0.83 static / 0.80 kinetic (US 10458053 B2 table 4, ASTM D1894 sled, "
                              "vendor data), a forefinger on plain-weave cotton 1.05 +- 0.32 (Nagatsu et al., Sen'i "
                              "Gakkaishi 64(4) 2008); the Franka Hand's fingertip material is not documented")),
    # Cotton plain shirting (Pal et al. 2024): 126 g/m2, 0.811 mm, EMT 2.72 / 12.53 %, B 0.0609 / 0.0319,
    # G 1.03 / 0.97 gf/cm deg. LT not reported: 0.6 assumed.
    "shirting_126": dict(
        label="cotton plain shirting, 126 g/m2",
        areal_density=0.126, thickness=0.81e-3*_mus_Tm/0.610e-3, thickness_free=0.81e-3, comp_RC=COMP_RC,
        **dict(zip(("K0w", "K1w", "etw"), _toe(0.0272))), **dict(zip(("K0f", "K1f", "etf"), _toe(0.1253))), kc=0.08,
        G=1.0*0.9807*180/math.pi, Bw=0.0609*9.807e-5, Bf=0.0319*9.807e-5, HB=_sh_HB,
        bend_stick_curv=BEND_STICK_CURV, air_perm=0.15, mu_steel=0.15, mu_cloth=0.65, mu_table=0.30, mu_pad=0.80, mu_dog=1.0,
        sources=dict(tension="read EMT (Pal), LT 0.6 assumed", B="read (Pal)", G="read (Pal), mean",
                     thickness="derived: T0 (Pal) times the muslin's KES Tm/T0 (no compression data; estimate)",
                     HB="derived: mean B times the woven 2HB/B ratio (fabric.kes_2hb_over_b_woven)")),
    # 13 oz denim, 1/3 twill (Nayak et al. 2013 FAST: 439-506 g/m2, E100 2.0-3.8 %, B 2.1-3.1e-4 / 0.4-0.9e-4 N m,
    # G 434-671 N/m). FAST is a low-load test (98 N/m): its secant is used as the toe stiffness; K1 = 2 K0 (estimate)
    "denim_440": dict(
        label="13 oz cotton denim, 1/3 twill",
        areal_density=0.44, thickness=0.80e-3, thickness_free=1.02e-3, comp_RC=COMP_RC,
        K0w=98.1/0.024, K1w=2*98.1/0.024, etw=0.024, K0f=98.1/0.020, K1f=2*98.1/0.020, etf=0.020, kc=0.10,
        G=550.0, Bw=2.5e-4, Bf=0.6e-4, HB=_den_HB, bend_stick_curv=BEND_STICK_CURV,
        air_perm=0.03, mu_steel=0.19, mu_cloth=0.65, mu_table=0.32, mu_pad=0.85, mu_dog=1.0,
        sources=dict(all="read (Nayak FAST, Wang thesis); K1 and thickness (Tm at 50 gf/cm2) estimates",
                     HB="derived: mean B times the woven 2HB/B ratio (FAST measures no hysteresis)")),
    # Ruled paper sheet (preset C): the repository's datasheet-anchored stock (label_application/materials.py):
    # 75.5 g/m2 at 83.8 um, E ~5 GPa MD / 2.5 GPa CD -> E t 4.2e5 / 2.1e5 N/m, D = E t^3/12(1-nu^2)
    "paper_75": dict(
        label="ruled paper, 75 g/m2",
        areal_density=0.0755, thickness=0.084e-3, thickness_free=0.084e-3,
        K0w=4.19e5, K1w=4.19e5, etw=0.01, K0f=2.10e5, K1f=2.10e5, etf=0.01, kc=1.0,
        G=2.96e5*0.38, Bw=5e9*(83.8e-6)**3/(12*0.91), Bf=2.5e9*(83.8e-6)**3/(12*0.91), HB=0.0,
        air_perm=0.001, mu_steel=0.20, mu_cloth=0.40, mu_table=0.30, mu_pad=0.80, mu_dog=0.8,
        sources=dict(all="repo (label_application datasheet stock); shear from isotropic estimate; friction open")),
}

AIR = dict(
    perm=P("air.permeability", "per fabric (air_perm)", "m/s at 125 Pa", "prior",
           "typical ASTM D737 ranges (muslin 50-150 cfm, shirting 15-60, 13 oz denim 2-15; office paper Gurley "
           "10-30 s/100 mL), from general knowledge, not sourced in this repository (open: measure per fabric). "
           "Two searches found no measured muslin value; one plain cotton control fabric read 108.7 "
           "L/m2/s, 0.11 m/s (PMC10073890, table 5; its test pressure not checked here), against the 0.4 m/s used "
           "for muslin"),
    viscosity=P("air.viscosity", 1.81e-5, "Pa s", "read", "dry air at 20 C (standard tables)"),
    gap_length=P("air.gap_length", 20e-3, "m", "prior", "lateral extent of the squeezed gap under a lifting region"),
)

CONTACT = dict(
    presser_sole_mu=P("friction.cloth_steel", 0.15, "-", "read", "cotton-steel 0.10-0.21 (tribology.rs 2021); KES MIU"),
    ply_ply_mu=P("friction.cloth_cloth", 0.65, "-", "read", "woven fabric-fabric mu_k 0.59-0.78 (JTATM)"),
    dog_mu=P("friction.cloth_dog_effective", 1.0, "-", "open", "teeth below mesh scale: prior 0.8-1.5, checked in E2/E4"),
    table_mu=P("friction.cloth_laminate", 0.30, "-", "open", "estimate 0.25-0.35"),
    pad_mu=P("friction.cloth_pad", 0.80, "-", "repo", "fr3@1 fingertip pad value"),
)

# ------------------------------------------------------------------------------------------------ the cell
TABLE_TOP = P("cell.table_top", (1.200, 0.533, 0.030), "m", "read",
              "Juki DDL-9000C table drawing 1,200 mm; standard 21 in depth (Apparel Resources)")
TABLE_EDGE_RADIUS = P("cell.table_edge_radius", 0.006, "m", "estimate",
                      "sewing workstations have 'rounded edges' [read], the radius not given; laminate "
                      "tops' edge bands and post-formed edges span ~2-12 mm (recalled). The cloth sees it as a capsule "
                      "along the front edge; the rendered and rigid table keep a square edge")
TABLE_H = P("cell.table_height", 0.762, "m", "read", "about 30 in, adjustable (Apparel Resources)")
NEEDLE_FROM_FRONT = P("cell.needle_from_front", 0.203, "m", "read", "8 in typical (Apparel Resources)")
NEEDLE_FROM_LEFT = P("cell.needle_from_left", 0.457, "m", "read", "18 in typical (Apparel Resources)")
BED = P("cell.bed", (0.517, 0.178), "m", "read", "Juki DDL-9000C bed 517 x 178 mm")
THROAT = P("cell.needle_to_arm", 0.303, "m", "read", "Juki DDL-9000C needle-to-arm 303 mm")

# ------------------------------------------------------------------------------------------------ the robots
# The motion generator's joint acceleration by robot model: the arms' controllers rate-limit commanded joint targets to
# these (and to the joint speed limits of the robot catalog's models).
JOINT_ACCEL_FR3 = P("robot.joint_accel_fr3", (10.0,)*7, "rad/s^2", "read",
                    "libfranka main (Franka Robotics, the FR3 library; commit e88d08f6) include/franka/"
                    "rate_limiting.h kMaxJointAcceleration, joints 1-7, read 2026-10-11")
JOINT_ACCEL_PANDA = P("robot.joint_accel_panda", (15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0), "rad/s^2", "read",
                      "libfranka master and 0.9.2 (Franka Emika, the Panda library) include/franka/rate_limiting.h "
                      "kMaxJointAcceleration, joints 1-7, read 2026-10-11")
JOINT_IMPEDANCE_MAX = P("robot.joint_impedance_max", 14250.0, "N m/rad", "read",
                        "libfranka main include/franka/robot.h Robot::setJointImpedance: K_theta in [0, 14250] N m/rad "
                        "for each joint of the internal controller, read 2026-10-11")
JOINT_IMPEDANCE_EXAMPLE = P("robot.joint_impedance_example", dict(K=(600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0),
                                                                D=(50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0)),
                            "N m/rad, N m s/rad", "read",
                            "libfranka main examples/joint_impedance_control.cpp k_gains, d_gains (read 2026-10-11): "
                            "the damping a stiffness gets when none is given keeps these ratios, D = D_ex sqrt(K / K_ex)")
JOINT_ACCEL_G1 = P("robot.joint_accel_g1", 10.0, "rad/s^2", "authored",
                   "the G1 has no motion generator: its joint servos take position targets as sent (Unitree's low-level "
                   "interface). This ramp of the commanded targets stands in for the interpolation a G1 client does "
                   "itself; it is the environment's, not the robot's")

# ------------------------------------------------------------------------------------------------ numerics
DT = P("numerics.dt", 5e-4, "s", "authored", "2 kHz tick: >= 9 ticks of feed-dog contact at 5,000 spm")
SUBSTEPS = P("numerics.substeps", 20, "-", "authored",
             "substeps of the rejected XPBD/VBD paths kept behind method=; the default implicit method "
             "takes one solved step per tick")
CG_ITERS = P("numerics.cg_iterations", 4000, "-", "authored",
             "cap of the PCG iterations of one Newton step (block-Jacobi). The solve ends at CG_TOL_ACC; a solve that "
             "reaches the cap is counted (cloth.cg_stats). At 40 iterations a solve on the "
             "1.25 mm seam band left 20 % of the residual at rest and stalled a sliding coupon. At 400, "
             "2.7 % of a plain seam's solves stopped at the cap and the Newton loop made up for them with more "
             "iterations; at 4,000 none did, and the same 10 stitches took 9 % fewer CG iterations in all")
CG_TOL_ACC = P("numerics.cg_tol_acc", 3.0e-2, "m/s^2", "derived",
               "the linear solve ends when the r.m.s. over the solve's free particles of each one's unbalanced "
               "acceleration (its residual force over its own mass) is below this. With H >= M/dt^2 a residual a "
               "leaves a velocity error near dt a in the inertia-dominated modes: 1.5e-5 m/s, under a sixth of the "
               "Newton loop's 1e-4 m/s. Converged: 10 stitches of the plain seam at 3e-1, 1e-1, 3e-2, 1e-2 and 1e-3 "
               "end within 0.001 deg of turn of a 1e-4 reference, every vertex within 0.11 mm and every needle hole "
               "within 0.003 mm, except that the 1e-1 run took a discrete alternative (0.6 mm at a vertex, 0.05 mm "
               "at the holes, the stitch 0.2 % longer) that the tolerance does not order: 3e-1 did not")
CG_BLOCK = P("numerics.cg_block", 8, "-", "authored", "PCG iterations between two checks of the residual")
CG_FORCING = P("numerics.cg_forcing", 0.0, "-", "authored",
               "inexact Newton (Dembo, Eisenstat & Steihaug 1982), off: with it a solve may end at this fraction of the "
               "residual it started from, and once such a step is below the Newton tolerance the next is solved to "
               "CG_TOL_ACC. Tried at 0.1: half the CG work of a tick while sewing (465 against 957 iterations), but "
               "the 10 stitches' turn ended 0.025 deg from the fully solved runs, which agree within 0.001 deg at "
               "every tolerance from 3e-1 to 1e-4, and twice the ticks ended unconverged at the Newton cap: "
               "rejected")
NEWTON_START_ACCEL = P("numerics.newton_start_accel", 9.81, "m/s^2", "authored",
                       "the Newton iterate starts at the inertial position plus dt^2 times the vertex's acceleration "
                       "over the last tick, bounded by this (gravity's): resting cloth starts at rest")
NEWTON_MIN = P("numerics.newton_min", 1, "-", "authored",
               "Newton iterations every tick: one (the iterate starts from the last tick's "
               "acceleration); the loop goes on while the step exceeds the Newton tolerance")
NEWTON_MAX = P("numerics.newton_max", 20, "-", "authored",
               "cap of the convergence loop; a tick that ends at the cap with its last step above "
               "the tolerance is counted (cloth.newton_stats)")
CC_PEAK_RULE = True              # ply on ply: the shapes' friction load-change rule
STICK_START_SNAP = P("numerics.stick_start_snap", 0.0, "preslips", "authored",
                     "the stick-first Newton start (a contact slipping less than this per tick starts "
                     "the tick stuck) is off with the elastic-plastic friction: it restarted every contact sliding "
                     "slower than 50 mm/s from rest each tick, which the Newton loop then had to undo")
NEWTON_TOL_V = P("numerics.newton_tol_v", 1e-4, "m/s", "authored",
                 "velocity accuracy that ends the Newton loop (tangential friction steps use half the stick zone)")
CONTACT_KAPPA = P("numerics.contact_kappa", 1.0e5, "-", "authored",
                  "stiff contact penalty k = kappa m / dt^2: the C1 continuation of the compression law where its "
                  "stiffness reaches k (a fibre-packing stop far above the presser's pressure)")
CONTACT_ACT = P("numerics.contact_hessian_margin", 1.0e-6, "m", "authored",
                "contacts within this gap enter the Newton Hessian")
CC_REACH = P("numerics.cc_reach", 3.0, "ticks", "authored",
             "ply-on-ply contact reach: detection searches this many ticks of the fastest cloth vertex's tick-start "
             "travel beyond the query radius, and a pair apart at the tick start is kept when its closing speed covers "
             "the gap in this many ticks (two surfaces approaching, an acceleration within the tick); with the pairs "
             "set up at the tick start only, plies moving faster than ~1 m/s crossed each other")
CC_REACH_MAX = P("numerics.cc_reach_max", 5.0e-3, "m", "authored",
                 "the largest extra detection radius the reach adds")
FRICTION_EPS = P("friction.ipc_eps_v", 1.0e-6, "m/s", "authored",
                 "IPC smoothed-Coulomb regularisation (Li et al. 2020): a contact below its slip limit still creeps at "
                 "a speed that scales with eps_v (4 um/s at 70 % on the incline with 1e-5; the draped "
                 "stack's corner 0.02 mm/s with 1e-5 and 0.004 mm/s with 1e-6). A numerical "
                 "regularisation whose physical value is zero: the smallest tested")
FEED_STROKE_PER_SETTING = P("machine.feed_stroke_per_setting", 1.145, "-", "fitted",
                            "fitted to the dial: the stitch length is calibrated, not predicted. The dog's horizontal "
                            "stroke per mm of the stitch-length dial; a dial is graduated in the "
                            "stitch the machine sews on the fabric it is set for (the E2 criterion: the fed "
                            "length within ~3 % of the setting at 40 N). The reference stack (two muslin plies, 40 N, "
                            "600 sti/min) at the 2.5 mm setting fed 2.185 mm per stitch with the stroke equal to the dial "
                            "and 2.588 with it 1.186 times the dial; the secant gives 1.145, "
                            "which feeds 1.5, 2.5 and 4 mm at -1.3, -0.2 and -0.4 %. "
                            "Of the 1.145, 1.083 is the locus's geometry (the teeth are above the plate for 0.923 of "
                            "the stroke) and the rest the hand-off slip. On the solved step the "
                            "1.145 feeds 2.5, 4 and 5 mm within 1 % of the dial; with the stroke derived from the "
                            "locus alone no tooth friction reaches the dial, so the stroke stays "
                            "calibrated and the slip is the model's, not anchored to a measurement")
CASTING_MU = P("machine.casting_mu", 0.3, "-", "estimate",
               "cloth on the painted cast iron of the machine's pillar and the head's underside (open; the bed and the "
               "plate use the fabric's cotton-steel value)")
HAND_HOUSING_MU = P("coupling.hand_housing_mu", 0.4, "-", "estimate",
                    "cloth on the Franka Hand's housing and the Dex3-1 palm (open; the fingers use the fabric's mu_pad)")
HAND_COMPLIANT = True             # the hands' pads ride on a body sprung to the arm, solved with the cloth
HAND_STIFFNESS = P("coupling.hand_stiffness", 3.0e4, "N/m", "estimate",
                   "the hand's pads ride on a body sprung to the arm along the vertical (the direction a hand presses "
                   "the work) and solved with the cloth, so an impact loads a spring instead of driving a rigid pad "
                   "through the fabric. 3e4 N/m: the order of a collaborative arm's structural stiffness "
                   "at its tool (harmonic drives, joint torque sensors, the hand's fingers) [estimate; no measured FR3 "
                   "or Dex3 value was found; the crushing peak's sensitivity at 1e4 and 1e5 N/m is measured]")
HAND_DAMPING_RATIO = P("coupling.hand_damping_ratio", 0.3, "-", "estimate",
                       "structural damping of that compliance (zeta of the hand on its spring; rubber pads, joint "
                       "friction) [estimate]. It does not reach the G1's ~180 Hz ringing on the stack (E18): that mode is "
                       "the body's mass on the stack (HAND_MASS)")
HAND_MASS = P("coupling.hand_mass", 0.010, "kg", "estimate",
              "the mass beyond that compliance: the pads and fingertip shells. The hand's own mass is in the arm's "
              "model, which drives the spring. With 0.1 kg here the G1's palm body rang on the stack's compressed "
              "stiffness (~2e5 N/m) at ~180 Hz, the contact force swinging 0-11 N while the body moved 0.03 mm "
              "(E18); at 10 g that mode lies above 700 Hz, where the implicit step damps it (E18 records what each "
              "robot's hand holds at rest and while the machine sews) [estimate]")
FINGER_STIFFNESS = P("coupling.finger_stiffness", 2.0e5, "N/m", "estimate",
                     "each Franka Hand finger's pads ride on a body of the cloth solve that moves along the vertical "
                     "(the arm's compliance, HAND_STIFFNESS shared by the two fingers) and along the finger's closing "
                     "axis, where this spring holds it to the finger MuJoCo moves: the fingertip's own compliance, "
                     "through which a squeeze reaches the finger's drive. 2e5 N/m: a rubber pad 18 x 18 "
                     "mm in face and 9.5 mm thick (the Franka Hand manual's pad drawing) "
                     "of mid hardness (Shore A 60, E 3.6 MPa; shape factor 0.47) [estimate: the pad's rubber and the "
                     "finger's own flexing are not published]")
FINGER_PAD_MASS = P("coupling.finger_pad_mass", 0.005, "kg", "estimate",
                    "the mass beyond that compliance (the pad and the fingertip's skin); the finger's own 29 g is in "
                    "the arm's model, which drives the spring")
FINGER_RANGE_FAULT = P("coupling.finger_range_fault", 5.0e-3, "m", "authored",
                       "a finger found this far outside its joint range ends the episode as a world fault. The "
                       "range's own stop is soft in the robot's model (10,000 N/m: 3.5 mm at the 35 N effort limit), "
                       "so the fault is set just beyond what a legal force can do")
UNRESOLVED_STEP_FAULT = P("numerics.unresolved_step_fault", 1.0e4, "tolerances", "authored",
                          "a cloth step whose Newton loop ends at its cap with its last step above this many Newton "
                          "tolerances (NEWTON_TOL_V each: 1e4 is a velocity error of 1 m/s) ends the episode as a world "
                          "fault: the world could not resolve that motion. The worst such step in the qualification "
                          "runs was 2,744 (the scripted guided seam); the machine run unguided at "
                          "5,000 sti/min reached 5e5 before the cloth ran away")
FRICTION_MODEL = "elastoplastic"  # cloth contact friction: "elastoplastic" (preslip) or "ipc"
FRICTION_PRESLIP = P("friction.preslip", 1.0e-4, "m", "estimate",
                     "elastic-plastic contact friction: each cloth contact carries a tangential spring "
                     "mu N / preslip up to its Coulomb limit mu N and slides at mu N beyond it (Dahl's/Iwan's "
                     "pre-sliding displacement), stored from tick to tick; a load below the limit is held at a fixed "
                     "displacement, so a draped stack does not creep; on a contact with a shape a rising load keeps the "
                     "stored tangential force above its peak since the stick was set, below it the displacement is kept. "
                     "0.1 mm: the order "
                     "of the pre-sliding displacement of fibrous contacts (fibre bending at the asperities) [estimate]. "
                     "With the dog's stroke equal to the dial the drop feed delivered 87 % of its 2.5 mm setting (E2), "
                     "outside the E2 criterion's ~3 %, so the stroke is calibrated to the dial "
                     "(FEED_STROKE_PER_SETTING, derived at 0.1 mm). The preslip's sensitivity, stroke "
                     "equal to the dial: the reference feed is 2.198, 2.180 and 2.116 mm at 0.05, 0.1 and 0.2 mm. "
                     "Replaces IPC's velocity "
                     "regularisation, a numerical value whose stick zone (eps_v dt, 0.5 nm) the solve could not resolve")
REACTION_TAU = P("coupling.reaction_tau", 1e-3, "s", "authored",
                 "first-order filter on the cloth's reactions on the hands, applied in the next rigid step: a "
                 "numerical damper of that explicit one-tick coupling, physical value zero (a pad's compliance is a "
                 "spring, not a delay). 5 ms (label_application, filleting precedent) let a hand pressing the work "
                 "fall into a ~115 Hz chatter; 0 leaves a tick-to-tick alternation; 1 ms, two ticks, is the "
                 "smallest tested that holds a steady press (E18)")
FINE_BAND = P("numerics.fine_band", 50e-3, "m", "authored",
              "half-width of the fine (FINE_H) band around each seam line and seam edge of a ply; beyond it the mesh "
              "grades to COARSE_H over three coarse edges. At 22 mm, the work's own turn carries an unguided "
              "seam ~0.6 mm a stitch in from its edge, and once the stitch line reached the band's graded edge the "
              "sewn work buckled behind the foot and the needle struck the plate, the 5 mm stitch's failure at its "
              "23rd-30th stitch; at 50 mm the same seam sewed flat with the needle intact. "
              "A needle hole in a triangle coarser than 2 FINE_H is a world fault (machine.py)")
FINE_H = P("numerics.fine_edge", 1.25e-3, "m", "authored",
           "seam band edge length. At 2.5 mm the baste task's 6 mm allowance put "
           "the top ply's cut edge 0.2 mm outside the right toe's corner with one row of triangles straddling it; at the "
           "5 mm stitch the edge curled up behind the foot in every run and the work wadded and bent the needle; "
           "at 1.25 mm the same start sewed flat. The plain seam's turn under the foot "
           "is the same on both (5.5 against 5.8 deg at 1.0 s). Cost: 1.65 times the 2.5 mm mesh's per tick while "
           "sewing, 2 times at rest (side by side on the shared host)")
COARSE_H = P("numerics.coarse_edge", 7.5e-3, "m", "authored", "bulk edge length")


def solver_kwargs():
    """The cloth solver's constructor settings as the world builds it. The bench and the coupon probes build theirs
    from this too, so they run the world's friction law, not the solver's class defaults."""
    return dict(contact_kappa=CONTACT_KAPPA, friction_eps=FRICTION_EPS, friction_model=FRICTION_MODEL,
                preslip=FRICTION_PRESLIP, cg_iterations=CG_ITERS, cg_tol_acc=CG_TOL_ACC, cg_tol_rel=CG_FORCING,
                cg_block=CG_BLOCK,
                newton_iterations=NEWTON_MIN, newton_max=NEWTON_MAX)


def solver_attrs(cl):
    """The solver settings that are attributes, as the world sets them."""
    cl.contact_act = CONTACT_ACT
    cl.newton_tol_v = NEWTON_TOL_V
    cl.set_recovery_tau = SET_RECOVERY_TAU
    cl.set_recovery_pa = SET_RECOVERY_PA
    cl.start_accel_max = NEWTON_START_ACCEL
    cl.stick_snap_ep = STICK_START_SNAP
    cl.cc_peak_rule = bool(CC_PEAK_RULE)
    cl.cc_reach = float(CC_REACH)
    cl.cc_reach_max = float(CC_REACH_MAX)
    return cl


def ledger_markdown():
    lines = ["# Sewing parameter ledger", "", "Generated by `python -m sewing.params`; do not edit by hand.", "",
             "| Key | Value | Unit | Label | Source |", "|---|---|---|---|---|"]
    for k, v in LEDGER.items():
        val = v["value"]
        if isinstance(val, float):
            val = f"{val:.6g}"
        note = f" ({v['note']})" if v.get('note') else ""
        lines.append(f"| `{k}` | {val} | {v['unit']} | {v['label']} | {v['source']}{note} |")
    lines += ["", "## Fabric profiles", ""]
    for name, f in FABRICS.items():
        lines.append(f"### {name}: {f['label']}")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        for k, v in f.items():
            if k in ("sources", "label"):
                continue
            lines.append(f"| {k} | {v:.6g} |" if isinstance(v, float) else f"| {k} | {v} |")
        for k, v in f.get("sources", {}).items():
            lines.append(f"| source: {k} | {v} |")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).with_name("PARAMETERS.md")
    out.write_text(ledger_markdown())
    print(f"wrote {out} ({len(LEDGER)} parameters, {len(FABRICS)} fabrics)")
