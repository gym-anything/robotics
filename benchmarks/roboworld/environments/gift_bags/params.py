"""Parameter ledger. Every constant carries its provenance label (REPORT §5):

  V vendor/manufacturer datasheet or product page  S standard  M measured (peer-reviewed)
  T textbook/handbook  D derived here from labelled inputs  E engineering estimate  A authored choice
  C calibrated in this repository  F fitted here to published data
  U a seller's listing of the product in use (weaker than a manufacturer's data sheet)  W read from a product web page

Each ShellMat/CreaseLaw/CohesiveLaw is read by the physics code (ipcworld.py); nothing here is decorative.
"""

import math
from .ipcworld import ShellMat, CreaseLaw, RodMat, CohesiveLaw

IN = .0254
G = 9.81

# Version of the contact world's PHYSICS (gift_bags/ipcworld.py). The start-state caches (shipping state, handles,
# wrapped present) are keyed on it rather than on the file's source, so a performance-only change does not rebuild
# hour-long start states. tests/test_gift_bags.py pins ipcworld.py's SHA-256 to the version: any edit fails that test
# until it is either a new version (the physics changed) or its hash is added under the current version together with
# the evidence that results are unchanged.
CONTACT_WORLD_VERSION = "2026-10-07b"
CONTACT_WORLD_SHA = {"2026-10-07b": ["4420b4278f581f7d"]}

# ----------------------------------------------------------------------------- paper stocks
# Laminated Euro-tote stock: 157 g/m2 coated art paper [U, Interplas; bracketed by 120-190 g/m2 V] with
# 18 um matte BOPP thermal lamination (12 um BOPP + 6 um EVA) [V, Arrsys]. Paper caliper 127 um [D, HannoArt
# 150 g/m2 -> 121 um]. E_MD/E_CD 6.83/3.11 GPa, nu12 0.23, G12 2.17 GPa measured on 150 g/m2 cover paper
# [M, proxy stock: Daler-Rowney Canford, an uncoated card; Nayakanti, Tawfick & Hart 2018; the coated, laminated tote
# stock is expected to be stiffer, so the model is likely soft]; BOPP 2/4 GPa [E]. Membrane per width = sum E_i t_i.
_T_PAPER, _T_BOPP = 127e-6, 12e-6
TOTE = ShellMat(
    "tote laminate 157 g/m2 + BOPP", rho_A=.157 + .0109 + .0056, t=145e-6,
    Et_md=6.83e9*_T_PAPER + 2e9*_T_BOPP, Et_cd=3.11e9*_T_PAPER + 4e9*_T_BOPP, nu12=.23,
    Gt12=2.17e9*_T_PAPER + .8e9*_T_BOPP,
    D_md=6.83e9*_T_PAPER**3/(12*(1 - .23**2)), D_cd=3.11e9*_T_PAPER**3/(12*(1 - .23**2)),   # 1.23 / 0.56 mN m [D]
    yield_strain=.005, X_md=7.9e3, X_cd=3.9e3, Gamma=1.6, Gamma_I=1.0, tear_cd=1.1, color=(.62, .10, .14))
# Tear of the laminate (none of these is sourced: no vendor tear figure was found for 150-170 g/m2 coated art paper,
# BOPP lamination film or chipboard):
#   Gamma_III 1.6 J/m [E, bracket 1.1-2.0]: paper ply 2 x Elmendorf with tear index 3.5-5.0 mN m2/g x 157 g/m2 =
#     1.1-1.6 J/m (printing-paper range, an unverified BIS snippet), plus the 12 um BOPP ply ~0.4 J/m (2 x 15-16 N/mm x
#     12 um trapezoid tear force, US6063482A, an order-of-magnitude reading); how lamination couples the plies is not known.
#   Gamma_I 1.0 J/m [E, bracket 0.6-1.4]: paper ply 0.62 J/m [D, Makela J_Ic index of coated paper 3.98 J m/kg x 157 g/m2]
#     plus an unsourced BOPP term ~0.4 J/m. Essential-work values for LWC (19-26 J m/kg MD, 4-12 CD,
#     US6712936B2) would give 0.7-4 J/m: the in-plane value is uncertain by a factor of several.
#   tear_cd 1.1 [E, bracket 1.0-1.2: CD over MD Elmendorf of machine-made paper; Billerud sack kraft 1.00 V, Mondi 1.08-1.18 U].
# Turn-top band: laminate folded over 900 g/m2 chipboard [V, Onlineprinters CLASSIC], ~1.43 mm board
# [D, Eska density]; board E_b 3 / 1.6 GPa [E]. Bending dominated by the board.
_T_BOARD = 1.43e-3
TURN_TOP = ShellMat(
    "turn-top laminate + 900 g/m2 board", rho_A=2*TOTE.rho_A + .900, t=145e-6,
    Et_md=TOTE.Et_md*2 + 3e9*_T_BOARD, Et_cd=TOTE.Et_cd*2 + 1.6e9*_T_BOARD, nu12=.3, Gt12=.8e9*_T_BOARD,
    D_md=3e9*_T_BOARD**3/12/(1 - .09), D_cd=1.6e9*_T_BOARD**3/12/(1 - .09), yield_strain=.005,
    X_md=20e3, X_cd=12e3, Gamma=15., Gamma_I=6., tear_cd=1.1, color=TOTE.color)
# Turn-top tear [E]: 900 g/m2 board Gamma_III 9-23 J/m (tear index 5-13 mN m2/g; 13 = virgin sack kraft, Billerud V, as
# an upper bound) plus two laminate layers (2 x 1.6) -> 15 J/m; Gamma_I: board J_Ic index 3-6 J m/kg [E] x 0.9 kg/m2 plus
# 2 x 1.0 -> 6 J/m. A turn-top practically does not tear by hand; these only keep it finite.
# Glued block bottom: two laminate layers glued [E, construction]
BOTTOM = ShellMat(
    "glued bottom (2 x laminate)", rho_A=2*TOTE.rho_A, t=145e-6, Et_md=2*TOTE.Et_md, Et_cd=2*TOTE.Et_cd,
    nu12=.23, Gt12=2*TOTE.Gt12, D_md=8*TOTE.D_md, D_cd=8*TOTE.D_cd, yield_strain=.005,
    X_md=2*TOTE.X_md, X_cd=2*TOTE.X_cd, Gamma=2*TOTE.Gamma, Gamma_I=2*TOTE.Gamma_I, tear_cd=TOTE.tear_cd,
    color=TOTE.color)                                           # tear: two glued laminate layers [D from TOTE, E]

# Gift wrap: 50 lb basis (81 g/m2) white gloss [V, Uline S-8551; D 1 lb/3000 ft2 = 1.628 g/m2]. Caliper 85 um,
# E 5/2 GPa, tensile 4.0/2.0 kN/m [E, not sourced; bracketed by newsprint and 70 g/m2 medical paper 7.3/4.0 kN/m V].
# Tear energy per crack length: out-of-plane Gamma_III = 2 F_Elmendorf with Elmendorf 0.3-0.6 N [E] -> ~1 J/m;
# in-plane Gamma_I 0.30 J/m [D: J_Ic index 3.43-3.98 J m/kg (newsprint, MWC; Makela in Coffin et al.
# 2013) x 81 g/m2 = 0.28-0.32]; a crack along CD costs tear_cd = 1.1 x one along MD [E, bracket 1.0-1.2]. The tear
# itself is IPCWorld's vertex splitting (VCCT at crack tips, extrinsic cohesive bonds dissipating Gamma per length).
_T_WRAP = 85e-6
WRAP_TEAR_TARGET_N = .45          # mean trouser tear force a calibration would aim at [E, Elmendorf 0.3-0.6 N]
WRAP = ShellMat(
    "gift wrap 81 g/m2 gloss", rho_A=.081, t=_T_WRAP, Et_md=5e9*_T_WRAP, Et_cd=2e9*_T_WRAP, nu12=.29,
    Gt12=.387*math.sqrt(5e9*2e9)*_T_WRAP,                       # Baum rule [M]
    D_md=5e9*_T_WRAP**3/12/(1 - .29**2*.4), D_cd=2e9*_T_WRAP**3/12/(1 - .29**2*.4),
    yield_strain=.005, X_md=4.0e3, X_cd=2.0e3, Gamma=1.0, Gamma_I=.30, tear_cd=1.1, color=(.75, .08, .12))

# Set-up box covering paper 128 g/m2 [V]; caliper 120 um [E]. Only its contact caliper is used
# (the box is rigid): it sets the contact geometry of a world without free sheets (lift-lid).
BOX_COVER = ShellMat(
    "box cover paper 128 g/m2", rho_A=.128, t=120e-6, Et_md=5e9*120e-6, Et_cd=2.5e9*120e-6, nu12=.29,
    Gt12=.387*math.sqrt(5e9*2.5e9)*120e-6, D_md=5e9*(120e-6)**3/12, D_cd=2.5e9*(120e-6)**3/12, color=(.85, .85, .82))

# Brown kraft 80 g/m2 (packing paper, parcel wrap): tensile 7.4/4.4 kN/m [V/U, Mondi]; Elmendorf 1.04-1.20 N
# [V, Billerud/Mondi] -> Gamma_III 2.1-2.4 J/m [D]; caliper 110 um [D, density 0.73]; E 7/3 GPa [E]. Gamma_I 1.07 J/m
# [D: sack kraft J_Ic index 13.4 J m/kg x 80 g/m2]; tear_cd 1.0 [V, Billerud Performance Brown SE: tear
# 1040 mN MD and CD, ISO 1974].
_T_KRAFT = 110e-6
KRAFT = ShellMat(
    "brown kraft 80 g/m2", rho_A=.080, t=_T_KRAFT, Et_md=7e9*_T_KRAFT, Et_cd=3e9*_T_KRAFT, nu12=.29,
    Gt12=.387*math.sqrt(7e9*3e9)*_T_KRAFT, D_md=7e9*_T_KRAFT**3/12/.96, D_cd=3e9*_T_KRAFT**3/12/.96,
    yield_strain=.005, X_md=7.4e3, X_cd=4.4e3, Gamma=2.2, Gamma_I=1.07, tear_cd=1.0, color=(.55, .38, .22))

# Office tape (Scotch Magic 810): matte acetate, 56 um total; tensile 22.8 N/cm; 35 % elongation [V, 3M].
# Acetate E ~2.5 GPa [E]; areal mass ~65 g/m2 [E].
MAGIC_TAPE = ShellMat(
    "Scotch Magic 810 tape", rho_A=.065, t=56e-6, Et_md=2.5e9*40e-6, Et_cd=2.5e9*40e-6, nu12=.35,
    Gt12=.9e9*40e-6, D_md=2.5e9*(40e-6)**3/12, D_cd=2.5e9*(40e-6)**3/12, yield_strain=.02,
    X_md=2280., X_cd=1500., Gamma=0., color=(.92, .93, .9))
# Packing tape 3M 371: BOPP, 46 um; tensile 385 N/100 mm MD [V]; E 2/4 GPa [E]
BOPP_TAPE = ShellMat(
    "3M 371 box-sealing tape", rho_A=.05, t=46e-6, Et_md=2e9*30e-6, Et_cd=4e9*30e-6, nu12=.4,
    Gt12=.8e9*30e-6, D_md=2e9*(30e-6)**3/12, D_cd=4e9*(30e-6)**3/12, yield_strain=.03,
    X_md=3850., X_cd=7700., Gamma=0., color=(.7, .55, .32))

# ----------------------------------------------------------------------------- creases
# Creased coated board 0.30 mm, 228-237 g/m2 (Nagasawa, Kaneko & Adachi, JAMDSM 13(1) 2019) [M]: one-loop test,
# fold to 90 deg at 0.02-0.4 rps, hold 0-20 s, turn back to zero load, watch the release angle. Measured: M at
# 90 deg 0.215 N m/m, 0.175 after 1 s (0.2 rps); release angle 46.0 deg after no hold, 48.5 / 54.5 / 59.4 deg
# after 1 / 5 / 20 s; released folds recover ~2.2 deg per e-fold of time. Law: CreaseLaw (two mechanisms,
# permanent + recoverable with a back spring) fitted to 93 points of their Eqs. 5-7 and Figs. 12-14
# (RMS residual 1.2 measurement standard deviations, a single-mechanism law 5.6) [F].
# Not represented: the first-fold transient peak Mp1 0.244 (ply delamination on a fresh crease; the scene's
# creases are factory-folded) and the initial loading slope of a fresh crease (~2 N m/m/rad, Fig. 8; fit 0.86).
_NAG = dict(k=.864, m_a=.01456, omega0=4.27e-7, k2=.2707, m2=.008636, omega2=.003232)


# Paper-class creases (gift wrap, kraft): refitted on Farain 2020 printing paper (36 deg, the smallest fold angle
# reached at saturating press force, used here as the angle a pressed fold settles at, which the source does not
# state; opening ~5.7 per decade for days, no unit printed; the board law gives 49 deg and 0.7 deg/decade): factors on the board law's m_a, k2,
# m2, omega2, the solution nearest the board law [F]. The BOPP-laminated tote keeps the board law (Mylar-like: 50 deg
# measured, 49 deg of the law) [E].
_PAPER = dict(m_a=.9057, k2=.5043, m2=2.1133, omega2=.8056)


def crease_for(t, name, stock="board"):
    """Crease law for stock of caliper t: moments scale with t^2, angles and rates do not. Crease stiffness ~ B/h
    ~ E h^2 was measured on creased Mylar 150-500 um, with a rest angle that "depends weakly" on h (fitted 27, 43 and
    39 deg; Lechenault, Thiria & Adda-Bedia, PRL 112:244301, 2014, arXiv 1404.1243) [M, proxy material: Mylar, not
    paper], so thickness-independent angles are an approximation; Farain 2020 found the fold-opening rate independent of
    thickness for Mylar and paper [M]. stock "paper" applies the thin-paper factors (_PAPER) to the time behaviour."""
    s = (t/.30e-3)**2
    f = _PAPER if stock == "paper" else dict(m_a=1., k2=1., m2=1., omega2=1.)
    return CreaseLaw(name, k=_NAG["k"]*s, m_a=_NAG["m_a"]*s*f["m_a"], omega0=_NAG["omega0"], k2=_NAG["k2"]*s*f["k2"],
                     m2=_NAG["m2"]*s*f["m2"], omega2=_NAG["omega2"]*f["omega2"])


TOTE_CREASE = crease_for(145e-6, "tote crease (scaled Nagasawa)")
WRAP_CREASE = crease_for(_T_WRAP, "wrap fold (paper law)", stock="paper")
KRAFT_CREASE = crease_for(_T_KRAFT, "kraft fold (paper law)", stock="paper")
# Scored C-flute flap line: moment at 90 deg ~0.8 N m/m [E, no source; sensitivity 0.3-2]; angles and time
# behaviour transferred from the board data [E]
FLAP_CREASE = CreaseLaw("C-flute score line", k=_NAG["k"]*.8/.215, m_a=_NAG["m_a"]*.8/.215, omega0=_NAG["omega0"],
                        k2=_NAG["k2"]*.8/.215, m2=_NAG["m2"]*.8/.215, omega2=_NAG["omega2"])

# ----------------------------------------------------------------------------- rope
# 5 mm woven PP rope [V, Uline]; 12.5 g/m [D: 0.91 g/cm3 x pi r^2 x 0.7 packing]; EA ~15 kN [E]; EI 2e-5 [E]
PP_ROPE = RodMat("5 mm PP rope", radius=2.5e-3, lin_rho=.0125, EA=15e3, EI=2e-5, color=(.95, .93, .88))

# ----------------------------------------------------------------------------- adhesion
# Pressure-sensitive tape on paper: datasheet peel on steel (180 deg, D3330) gives Gamma_adh ~480 J/m2
# (Magic) / ~880 (371) [D]; on paper the surface delaminates first: Gamma_eff = min(Gamma_adh, Gamma_delam)
# with Gamma_delam ~150 J/m2 (gloss wrap) / 200 (kraft liner) [E, anchored to SBS ply bond >= 120 J/m2 V].
# Adhesive layer stiffness 1.5e10 Pa/m (label_application ledger [E]). Strength = the paper's z-direction tensile
# strength, since the paper surface is what lets go: 150 g/m2 kraftliner measures ~300-550 kPa (ISO 15754 / TAPPI
# T541; ABB L&W ZD Tensile Tester documentation) [V]; 0.40 MPa for coated wrap [E, within that range], 0.45 MPa
# for kraft liner [V range]. Steady 90-degree peel of a 19 mm strip: Gamma w = 2.85 N.
TAPE_ON_WRAP = CohesiveLaw("Magic tape on gift wrap (paper delamination)", k=1.5e10, Gamma=150., sigma=.40e6)
TAPE_QUAD = 4       # tape bonds per tape triangle: 0 = one per tape vertex (ipcworld.Bonds); m > 0 = the centroids of an
                    # m x m subdivision of every tape triangle (tape_bonds.QuadBonds).
                    # 90-degree peel at 10 mm/s: steady 3.44 N with m = 4 against 5.24 N per vertex
                    # (energy balance 2.85 N), largest spike 11.8 against 22.3 N; cost in the wrapped present at rest
                    # 0.684 against 0.556 s per contact step, same load [C]
TAPE_ON_KRAFT = CohesiveLaw("371 on kraft liner (liner delamination)", k=1.5e10, Gamma=200., sigma=.45e6)

# ----------------------------------------------------------------------------- friction
# paper on paper 0.2-0.9 across paper samples and measuring conditions together [M, de Silveira & Hutchings 1997: the
# method alone moves a value 15-30 %; no coated or laminated paper was measured there]; pair values [E]
MU = dict(paper_paper=.35, kraft_kraft=.5, paper_table=.4, finger_paper=.5, box_box=.35, box_table=.4,
          carton=.5, carton_table=.3)

# Friction by MATERIAL PAIR: the contact world uses exactly these values (IPCWorld(mu_pairs=FRICTION,
# friction_strict=True) refuses any pair not listed; tests/test_gift_bags.py asserts the built table equals this one).
#   laminate  tote shell (matte BOPP outside, paper inside) and the four lower totes of the stack
#   wrap      81 g/m2 gloss gift wrap            tape     Magic tape backing
#   greyboard paper-covered set-up gift box      chipboard bottom boards      carton  SBS folding carton (candle)
#   counter   laminate counter top               mat      rubber counter mat      oak     dining table
#   frame     photo frame (wood/glass)           finger   Franka Hand fingertip    link   robot link covers
#   phantom   ASSISTED fixture bodies (formers, paddles)
#   rope      5 mm braided PP rope handles        knot     the rope's knots inside the turn-top (same PP rope)
# Labels: paper on paper 0.2-0.9 across samples and conditions, de Silveira & Hutchings 1997 [M range; uncoated
# papers only]; the values chosen inside such ranges are [E]. The Franka Hand manual (220010 v1.2, 2022) does not state the fingertip material; the standard
# fingertips are hard ridged plastic, so finger on paper/board is taken as a hard-plastic-on-paper value 0.5 [E,
# range 0.3-0.6]; robot link covers 0.3 [E].
_MU_LIST = [
    (("laminate", "laminate"), .35), (("laminate", "counter"), .40), (("laminate", "mat"), .60),
    (("laminate", "chipboard"), .40), (("laminate", "carton"), .40), (("laminate", "greyboard"), .35),
    (("wrap", "wrap"), .35), (("wrap", "tape"), .40), (("tape", "tape"), .40), (("wrap", "greyboard"), .35),
    (("tape", "greyboard"), .40), (("wrap", "oak"), .40), (("tape", "oak"), .40),
    (("greyboard", "greyboard"), .35), (("greyboard", "oak"), .40), (("greyboard", "frame"), .35),
    (("frame", "oak"), .40), (("frame", "frame"), .40), (("frame", "wrap"), .40), (("frame", "tape"), .40),
    (("chipboard", "chipboard"), .40), (("chipboard", "counter"), .40), (("chipboard", "mat"), .60),
    (("chipboard", "carton"), .40), (("carton", "counter"), .40), (("carton", "mat"), .60), (("carton", "carton"), .50),
    (("counter", "mat"), .60), (("counter", "counter"), .40), (("mat", "mat"), .80), (("oak", "oak"), .40),
]
# PP rope [E]: PP on paper/board 0.30, PP on PP 0.25, on the rubber mat 0.50 (polymer-on-paper and polymer-on-polymer
# values inside the paper-friction range above; no measurement of braided PP rope on laminate found)
for _m in ("laminate", "chipboard", "carton", "counter", "greyboard", "wrap", "tape", "oak", "frame"):
    _MU_LIST += [(("rope", _m), .30), (("knot", _m), .30)]
_MU_LIST += [(("rope", "mat"), .50), (("knot", "mat"), .50), (("rope", "rope"), .25), (("rope", "knot"), .25),
             (("knot", "knot"), .25)]
for _m in ("laminate", "wrap", "tape", "greyboard", "chipboard", "carton", "counter", "mat", "oak", "frame", "rope", "knot"):
    _MU_LIST += [(("finger", _m), .50), (("link", _m), .30), (("phantom", _m), .30)]
_MU_LIST += [(("finger", "finger"), .50), (("finger", "link"), .30), (("link", "link"), .30),
             (("phantom", "phantom"), .30), (("phantom", "finger"), .30), (("phantom", "link"), .30)]
# Receiving preset (carton.py): corrugated  kraft-liner faces of the C-flute carton and its flaps; packtape  3M 371 BOPP
# backing; steel  stainless work table; glass  the jar; the tea cartons are SBS folding cartons ("carton" above).
# Corrugated on corrugated 0.50 and carton on stainless 0.30 are estimates [E]; tape
# backing on paper/board 0.30 [E]; the rest [E] inside the paper-on-paper range 0.2-0.9 [M range,
# de Silveira & Hutchings 1997] or polymer/glass/metal-on-paper values of the same order. (glass, glass) and
# (steel, steel) are listed only because the strict check asks for every class pair: one jar, one static table.
_MU_LIST += [(("corrugated", "corrugated"), .50), (("corrugated", "steel"), .30), (("corrugated", "packtape"), .30),
             (("corrugated", "carton"), .40), (("corrugated", "glass"), .35), (("packtape", "packtape"), .30),
             (("packtape", "steel"), .30), (("packtape", "carton"), .30), (("packtape", "glass"), .30),
             (("carton", "steel"), .30), (("carton", "glass"), .30), (("glass", "steel"), .20), (("glass", "glass"), .40),
             (("steel", "steel"), .40)]
for _m in ("corrugated", "packtape", "steel", "glass"):
    _MU_LIST += [(("finger", _m), .50), (("link", _m), .30), (("phantom", _m), .30)]
FRICTION = {frozenset(k): v for k, v in _MU_LIST}
# Dex3-1 finger surfaces of the Unitree G1 configuration (gift_bags/g1.py FINGER_LINKS: the 7 finger links of each hand;
# the palm and every other G1 link are "link"). Unitree's Dex3-1 page (unitree.com/Dex3-1, retrieved 2026-10-06) lists
# 3 x 4 tactile arrays at the fingertips (33 sensors) but no surface material or friction coefficient [V: not stated].
# The finger pads are taken as an elastomer skin [E]: on paper, board and tape 0.6 (range 0.4-1.0, rubber on paper);
# on the hard counter laminate, oak, stainless steel, glass and the photo frame 0.5 (range 0.4-0.8); on the rubber mat
# 0.8; on the PP rope and knots 0.4; elastomer on elastomer 0.8; on the robot's own link covers 0.4; on ASSISTED phantoms
# 0.3 like every robot class.
# Kept as a separate table merged into the world's strict friction ledger only when the G1 is present
# (world.World): bagnet.shipping_state keys its hour-long start-state cache on every FRICTION pair containing "laminate",
# so listing (dex3_finger, laminate) in FRICTION itself would discard that cache although no Dex3 takes part in the fold.
_MU_DEX3 = [(("dex3_finger", _m), .60) for _m in ("laminate", "wrap", "tape", "greyboard", "chipboard", "carton",
                                                     "corrugated", "packtape")]
_MU_DEX3 += [(("dex3_finger", _m), .50) for _m in ("counter", "oak", "steel", "glass", "frame")]
_MU_DEX3 += [(("dex3_finger", "mat"), .80), (("dex3_finger", "rope"), .40), (("dex3_finger", "knot"), .40),
             (("dex3_finger", "dex3_finger"), .80), (("dex3_finger", "link"), .40), (("dex3_finger", "phantom"), .30)]
FRICTION_DEX3 = {frozenset(k): v for k, v in _MU_DEX3}

# ----------------------------------------------------------------------------- air (lid film)
AIR_MU = 1.81e-5      # Pa s [T]
AIR_RHO = 1.204       # kg/m3 [T]
SLOT_LOSS_K = 1.5     # entrance + exit loss coefficient of the lid slot, laminar Re ~250 [T/E]

# ----------------------------------------------------------------------------- home preset variation (variants.HOME_AXES)
# Seeds of the home preset draw a box size, a lid fit and a wrap stock (seed 0: the authored present).
#
# Consumer roll wrap: Holmen GIFT Silk 53 g/m2, the lightest Silk grade of a mill's wrapping-paper base for gravure and
# flexo printed gift wrap (Holmen GIFT technical data sheet, target values, updated 2025-09-01: Silk 53-70 g/m2, bulk
# 1.1 cm3/g, burst 85 kPa at 53 g/m2; Lite 43-50 g/m2) [V]. Caliper 58.3 um [D: bulk x grammage]. Everything else by the
# rules WRAP uses, scaled by grammage where WRAP's value is an index times its 81 g/m2:
#   E 5/2 GPa, nu12, Baum G12 and the bending rule as WRAP [E, the same estimate; not sourced for this grade]
#   tensile 2.62/1.31 kN/m [D: WRAP's 4.0/2.0 kN/m (E) as a tensile index x 53 g/m2]
#   Gamma_III 0.65 J/m [D: WRAP's Elmendorf 0.3-0.6 N (E) as a tear index, 3.7-7.4 mN m2/g, x 53 g/m2 -> 0.20-0.39 N,
#     Gamma_III = 2 F; i.e. WRAP.Gamma x 53/81]; trouser target 0.29 N (WRAP_TEAR_TARGET_N x 53/81) [D]
#   Gamma_I 0.20 J/m [D: J_Ic index 3.43-3.98 J m/kg (as WRAP) x 53 g/m2 = 0.18-0.21]
#   tear_cd 1.1 [E, as WRAP]. Tape on it: TAPE_ON_WRAP (a coated wrapping-paper surface like WRAP's) [E].
# The Lite grades (43-50 g/m2, 47-55 um) are thinner than the Magic tape (56 um): the contact world takes its caliper from
# the thickest sheet stock (ipcworld.IPCWorld), so with them the tape, not the paper, would set the caliper; not offered.
_G_CONSUMER = .053
_T_WRAP_CONSUMER = 1.1e-6*53.                                   # m: 1.1 cm3/g x 53 g/m2 = 58.3 um [D]
WRAP_CONSUMER_TEAR_TARGET_N = WRAP_TEAR_TARGET_N*_G_CONSUMER/WRAP.rho_A   # 0.29 N [D]
WRAP_CONSUMER = ShellMat(
    "consumer gift wrap 53 g/m2 silk", rho_A=_G_CONSUMER, t=_T_WRAP_CONSUMER, Et_md=5e9*_T_WRAP_CONSUMER,
    Et_cd=2e9*_T_WRAP_CONSUMER, nu12=.29, Gt12=.387*math.sqrt(5e9*2e9)*_T_WRAP_CONSUMER,
    D_md=5e9*_T_WRAP_CONSUMER**3/12/(1 - .29**2*.4), D_cd=2e9*_T_WRAP_CONSUMER**3/12/(1 - .29**2*.4),
    yield_strain=.005, X_md=WRAP.X_md*_G_CONSUMER/WRAP.rho_A, X_cd=WRAP.X_cd*_G_CONSUMER/WRAP.rho_A,
    Gamma=WRAP.Gamma*_G_CONSUMER/WRAP.rho_A, Gamma_I=WRAP.Gamma_I*_G_CONSUMER/WRAP.rho_A, tear_cd=1.1,
    color=(.10, .30, .62))
WRAP_CONSUMER_CREASE = crease_for(_T_WRAP_CONSUMER, "consumer wrap fold (paper law)", stock="paper")
# wrap stock name -> (shell material, crease law); "commercial_81gsm" is the authored WRAP (Uline S-8551)
WRAP_STOCKS = {"commercial_81gsm": (WRAP, WRAP_CREASE), "consumer_53gsm": (WRAP_CONSUMER, WRAP_CONSUMER_CREASE)}

# Two-piece set-up gift boxes beside the authored one (giftbox.BOX_HALF / BOX_H / WALL: 180 x 130 x 55 mm outer, 1.5 mm
# greyboard [A]), each holding a photo frame that fits it. Catalogue sizes are the vendor's W x L x D,
# read as INSIDE dimensions [E: Uline's box convention; neither product page says]; the base's outer size adds one wall
# per side and the floor (giftbox.box_spec). Both lids keep the authored partial-telescope skirt giftbox.LID_H [A] and take
# the seeded clearance. Masses [D]: board and covering-paper areas x areal masses below.
#   uline_window_7.5x7.5x2  Uline S-25459R Window Gift Box 7 1/2 x 7 1/2 x 2 in: "pre-assembled, rigid chipboard", two
#                           piece, 6 x 6 in clear acetate window in the lid [V]. Board caliper not stated: 1.5 mm [E; Uline's
#                           3.9 kg per carton of 18 is 0.217 kg a box with the shipping carton's share, against 0.14 kg
#                           derived here at 1.5 mm and 0.09 kg at 0.8 mm]. Window film 0.25 mm acetate [E] in the lid's
#                           mass. Holds an IKEA FISKBO 10x15 cm frame.
#   uline_setup_8.5x11x2    Uline S-10708 Rigid Set-Up Box 8 1/2 x 11 x 2 in (letterhead): "fully assembled", two piece,
#                           .032 in board [V]. Holds an IKEA FISKBO 13x18 cm frame.
GREYBOARD_RHO = .975/1.5e-3        # kg/m3: 650 [D: 0.975 kg/m2 at 1.5 mm]; used for every box board [E]
BOX_COVER_RHO_A = .128             # kg/m2 covering paper [V]; on the Uline boxes [E]
ACETATE_RHO = 1300.                # kg/m3 cellulose acetate window film [T]
GIFT_BOXES = {
    "uline_window_7.5x7.5x2": dict(inside=(7.5*IN, 7.5*IN, 2*IN), wall=1.5e-3, window=(6*IN, 6*IN, .25e-3),
                                   frame="ikea_fiskbo_10x15"),
    "uline_setup_8.5x11x2": dict(inside=(11*IN, 8.5*IN, 2*IN), wall=.032*IN, window=None, frame="ikea_fiskbo_13x18"),
}
# Photo frames (outer size L x W x depth along the box's x, y, z; mass): IKEA FISKBO, fibreboard with paper foil and a
# polystyrene front: 10x15 cm picture -> frame 13 x 18 x 1 cm, package 0.11 kg (ikea.com/my 20297420; 0.12 kg on
# ikea.com/ch 00295653); 13x18 cm -> 16 x 21 x 1 cm, package 0.15 kg (ikea.com/my 80297422) [V; the package weight, frame
# plus its film, is taken as the frame's mass]. The authored frame stays scene.FRAME_HALF / FRAME_MASS [E].
PHOTO_FRAMES = {
    "ikea_fiskbo_10x15": dict(size=(.180, .130, .010), mass=.11),
    "ikea_fiskbo_13x18": dict(size=(.210, .160, .010), mass=.15),
}
# Lid clearance per side [A]: the values benched for the authored box (phantom lift, frame inside: base rise at 10 cm/s
# 1.22 mm at 0.5 mm, 0 at 0.8 and 1.0 mm; 0 for all at 1 cm/s) and for every box; vendors say "fractions of a
# millimeter" [V qualitative]. 0.3 mm (43.9 mm base rise at 10 cm/s) is not drawn.
LID_CLEARANCES = (.5e-3, .8e-3, 1.0e-3)
