"""Parameter ledger. Every physical constant the environment uses lives here,
with its value, unit and provenance label, so the runtime reads one source.

Provenance labels: read (primary source read), abstract, search, literature, repo, measured,
derived (arithmetic shown), estimate, authored, open (no digital basis). An open value's basis may state a proposed
sweep range ("sweep range, not run"): no sweep of these values has been run.
"""

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class P:
    value: object
    unit: str
    label: str
    basis: str
    # "model": an input of the world's physics or evaluators (read by its code);
    # "reference": recorded for reference only (a source a derived value or a refusal rests on, physics the world does
    # not model)
    use: str = "model"


def v(p):
    return p.value


# --------------------------------------------------------------------------------------------- fish tissue
FISH_DENSITY = P(1050., "kg/m3", "abstract+estimate", "fresh seafood 1042-1093 kg/m3 at 20 C (Rahman & Driscoll 1994); "
                 "fatty salmonids at the low end")
FISH_E = P(60e3, "Pa", "derived", "raw cod G' ~ 22 kPa -> E ~ 66 kPa (Blikra 2019 via arXiv 2112.08280); raw tuna "
           "large-strain secant 150-180 kPa (Jumilla-Lorenz & Pares 2025); checked in probe E1 against salmon "
           "flat-cylinder breaking force 6.6-20.9 N (Torgersen 2014)")
FISH_NU = P(.45, "-", "read(porcine)+estimate", "porcine nu_LT 0.47 (Takaza 2013); fish unmeasured; nearly incompressible")
FISH_TAU_V = P(1e-4, "s", "estimate", "stiffness-proportional (Kelvin-Voigt on the Green strain rate, eta = tau*mu, "
               "normalised by the stretch, 3 / max(3, tr C)) "
               "damping time. It damps kHz mesh-scale oscillation only and is chosen small enough not to set the "
               "explicit step; slow quasi-linear relaxation of muscle (porcine: ~50 % viscous part at 0.5 %/s, "
               "Van Loocke 2008) is NOT modelled")
TISSUE_VISCO = P(dict(g=.2, tau=.015), "-", "derived+estimate",
                 "structural damping of the flesh: one Maxwell branch on the deviatoric Green strain (relative modulus "
                 "g, relaxation time tau) beside the elastic law. Fish muscle's loss modulus is about one order of "
                 "magnitude below its storage modulus over 1-100 rad/s (cooked cod/lutefisk, arXiv 2112.08280; raw cod, "
                 "Blikra et al. 2019): tan delta ~0.1. One branch with g 0.2 has tan delta at most g/(2 sqrt(1+g)) = 0.09, "
                 "at 1/(tau sqrt(1+g)) = 61 rad/s, near the workpieces' lowest modes (~10 Hz); a broader spectrum (and "
                 "the slow relaxation, FISH_TAU_V) is not modelled. Used for pork too (estimate). Without it the "
                 "flesh has only the numerical damping (zeta ~0.004 at 13 Hz) and a disturbed fish rings for seconds")
FISH_R_PERP = P(300., "J/m2", "derived", "tuna razor shear 4.3-5.5 N across the connective sheets via V = R*w, "
                "w = 8.9-30 mm -> 140-620 J/m2 (upper bound: friction and deformation included)")
FISH_R_PAR = P(100., "J/m2", "derived", "tuna razor shear 1.8-2.0 N parallel to the sheets, same method -> 60-225 J/m2")
FISH_HEAD_E = P(1.0e6, "Pa", "estimate", "head (skull, gill covers, cheek) lumped as a stiff region; no source", use="reference")
FISH_HEAD_R = P(4000., "J/m2", "estimate", "head lumped toughness; the head is removed at the joint, not cut through", use="reference")
SHEET_SIGMA = P(10e3, "Pa", "open", "myoseptal sheet strength: no measurement found; sweep range 5-20 kPa, not run")
SHEET_G = P(3., "J/m2", "open", "myoseptal sheet fracture energy: no measurement; sweep range 1-10 J/m2, not run")
SHEET_BETA = P(1., "-", "estimate", "weight of the shear traction in the sheet failure law "
               "sqrt(<t_n>^2 + (beta t_s)^2); no measurement")
FISH_SIGMA_T = P(50e3, "Pa", "open", "fish muscle tensile strength off the sheets: no accessible measurement; checked "
                 "against the flat-cylinder puncture (Torgersen 2014: a 12.5 mm flat cylinder at 1 mm/s breaks the "
                 "salmon fillet surface at 6.6-20.9 N; E1)")

SKIN_THICKNESS_TROUT = P(3.0e-4, "m", "read(other species)+estimate", "striped bass 0.23-0.47 mm; grass carp 0.55 mm")
SKIN_THICKNESS_SALMON = P(5.0e-4, "m", "read(other species)+estimate", "as above; salmon skin thicker")
SKIN_E = P(20e6, "Pa", "search", "bass skin ~13 MPa at 0 % strain rising to 32-40 MPa at 20 %")
SKIN_STRENGTH = P(30e6, "Pa", "read", "striped bass stratum compactum 20-60 MPa, failure strain 0.2-0.5 (Szewciw 2017)")
SKIN_DENSITY = P(1100., "kg/m3", "estimate", "collagenous dermis")
SKIN_R = P(2000., "J/m2", "estimate", "within the dermis scissor-cut range 1-3 kJ/m2 (search excerpt, mammalian "
           "dermis); fish skin is thinner and weaker, so not above it. The 17 kJ/m2 of pork dermis (Comley & Fleck 2010) "
           "is a trouser TEAR, not a cut")
SKIN_FLESH_G = P(30., "J/m2", "open", "skin-flesh interface fracture energy; no peel test found. Not modelled: the skin "
                 "membrane shares the flesh's surface nodes (no interface; skinning, B4, is refused for it); kept for "
                 "the finite-difference check of the interface law", use="reference")
SKIN_FLESH_SIGMA = P(20e3, "Pa", "open", "skin-flesh interface strength; as SKIN_FLESH_G, not modelled",
                     use="reference")
FLESH_BONE_G = P(50., "J/m2", "open", "flesh-bone interface energy; no measurement; sweep range 20-200, not run")
FLESH_BONE_SIGMA = P(30e3, "Pa", "open", "flesh-bone interface strength; sweep range 10-80 kPa, not run")
JOINT_R = P(1500., "J/m2", "abstract+estimate", "intervertebral joint tissue; perimysium work of fracture 0.4-1.8 kJ/m2 "
            "(cooked beef, Purslow 1985)")

# --------------------------------------------------------------------------------------------- bones
BONE_E = P(3.0e9, "Pa", "read", "herring intermuscular bone 2.6-3.5 GPa (Fiedler 2019)")
BONE_UTS = P(110e6, "Pa", "read", "herring intermuscular bone 108-111 MPa (Fiedler 2019)")
BONE_DENSITY = P(2000., "kg/m3", "estimate", "mineralised fish bone")
BONE_CRUSH = P(150e6, "Pa", "read(other tissue)+estimate", "mammalian cortical compressive strength 154-205 MPa", use="reference")
BONE_SHEAR = P(55e6, "Pa", "estimate", "~half the tensile strength; a 1 mm rib needs ~40 N to be severed by an edge", use="reference")
PIN_PULL_SALMON = P((5.41, 1.71), "N", "read", "post-rigor salmon, 1 mm/s tension (Schroeder et al. 2019)", use="reference")
PIN_BREAK_SALMON = P((3.91, 2.56), "N", "read", "tensile grips 1 mm/s (Schroeder et al. 2019)", use="reference")
PIN_COUNT_SALMON = P(29, "-", "read", "micro-CT, 3-5 kg fish (Schroeder et al. 2019)")
PIN_LENGTH_SALMON = P((14e-3, 50e-3), "m", "read", "micro-CT range (Schroeder et al. 2019)", use="reference")
PIN_SHAFT_TAU = P(78e3, "Pa", "derived", "5.41 N / (pi * 0.55 mm * 40 mm)")
PIN_ROOT_SHARE = P(.5, "-", "read+derived", "cutting the root tendon halves extraction force (SINTEF 2009)")

# --------------------------------------------------------------------------------------------- pork
PORK_E = P(100e3, "Pa", "read+estimate", "raw porcine longissimus in tension (Takaza 2013, abstract read): 10 kPa along "
           "and 77 kPa across the fibres at stretch 1.1, i.e. secant moduli 10/0.1 = 100 kPa along and 770 kPa across "
           "(across: 'broadly linear'); the isotropic model takes the soft, along-fibre value and so ignores a 7.7x "
           "anisotropy that matters most for slicing across the fibres (C3)")
PORK_R = P(400., "J/m2", "estimate", "Atkins' illustrative beef value 400, bracketed by frozen beef 100-200 "
           "(Dobraszczyk 1987) and raw pork 840 +- 800 in notched tension (Aryeetey 2022; not a cutting toughness)")
PORK_DENSITY = P(1060., "kg/m3", "literature", "general meat density")
PORK_SIGMA_T = P(115e3, "Pa", "derived", "raw porcine longissimus across the fibres (Takaza 2013, abstract read): broadly "
                 "linear, 77 kPa at stretch 1.1, failure at stretch ~1.15, so 77 x 0.15/0.10 ~ 115 kPa at failure (along "
                 "the fibres failure comes at stretch ~1.65)")
FAT_E = P(100e3, "Pa", "open", "chilled back fat: ~1 kPa at 25-37 C (Comley & Fleck 2012) and 'much firmer' chilled "
          "(Wijarnprecha 2022); chilled value unmeasured, sweep range 10-500 kPa, not run")
FAT_R = P(3000., "J/m2", "read(warm)+open", "adipose trouser tear 4.1 kJ/m2 at body temperature (Comley & Fleck 2010)")
FAT_SIGMA_T = P(50e3, "Pa", "open", "chilled fat tensile strength: no measurement")
SALMON_BELLY = P(dict(E_factor=1., R_factor=1.), "-", "open",
                 "salmon belly flap strip (the Trim D strip, region belly_fat): fish muscle richer in lipid, 28.1 % of "
                 "wet weight against 9.6 % in the white muscle (Aursand et al. 1994, J Sci Food Agric 64:239, abstract), "
                 "adipocyte-dominated (Aquaculture 2007, abstract); no measurement of its stiffness or cutting toughness "
                 "at chill temperature was found, so it takes the fish muscle's values (factors on FISH_E and FISH_R); "
                 "sweep range 0.5-2 each, not run")
SILVER_THICKNESS = P(2e-4, "m", "open", "silver skin thickness; no measurement; 0.1-0.5 mm")
SILVER_E = P(50e6, "Pa", "open", "epimysium membrane modulus; no measurement")
SILVER_STRENGTH = P(10e6, "Pa", "open", "no measurement")
SILVER_R = P(2000., "J/m2", "open", "no measurement of epimysium cutting; taken as the collagenous dermis's scissor-cut "
             "range 1-3 kJ/m2 (SKIN_R)")
SILVER_BOND_G = P(150., "J/m2", "open", "hand-peeling is possible but hard; sweep range 50-500, not run")
CHAIN_SEAM_G = P(30., "J/m2", "open", "no measurement; sweep range 10-100, not run")
CHAIN_SEAM_SIGMA = P(10e3, "Pa", "open", "chain seam (loose fat and fascia) peak traction; no measurement; placeholder "
                     "below the skin-flesh value, sweep range 5-30 kPa, not run")
SILVER_BOND_SIGMA = P(50e3, "Pa", "open", "silver skin to muscle peak traction; no measurement; hand-peeling is possible "
                      "but hard: placeholder above the flesh-bone value, sweep range 20-100 kPa, not run")

# --------------------------------------------------------------------------------------------- tools, contact
FILLET_BLADE = P(dict(length=.173, height=.025, spine=1.2e-3, mass=.088, bevel_deg=18.), "m,kg", "read",
                 "Victorinox Fibrox 5.3703.18 (knivesandtools datasheet)")
BONING_BLADE = P(dict(length=.150, height=.020, spine=2.0e-3, mass=.102, bevel_deg=18.), "m,kg", "read",
                 "Victorinox Fibrox 5.6303.15 (knivesandtools datasheet)")
HANDLE = P(dict(length=.140, height=.032, width=.019), "m", "read+estimate",
           "Victorinox 5.3813.18 handle envelope 19 x 32 mm (manufacturer); 140 mm length (retailer)")
STEEL_E = P(200e9, "Pa", "literature", "stainless blade steel")
STEEL_G = P(80e9, "Pa", "literature", "stainless blade steel")
BSI_HONED = P(.22, "-", "read+estimate", "scalpel sharp BSI 0.216 (McCarthy 2007); transfer to tissue estimated")
BSI_WORN = P(.50, "-", "read+estimate", "scalpel blunted BSI 0.514 (McCarthy 2007)")
MU_BLADE_TISSUE = P(.2, "-", "open", "no wet-tissue measurement; fitted food mu 0.18-1.0 (Jamdagni & Jia)")
TAU0_BLADE_TISSUE = P(1e3, "Pa", "open", "wet adhesive shear; Atkins tau_f 10-20 kPa for sticky cheese/salami at -8 C", use="reference")
MU_BLADE_BONE = P(.25, "-", "estimate", "steel on wet bone")
MU_SKIN_HDPE = P(.15, "-", "estimate", "fish mucus 'ultra-low friction' (Wu et al. 2014, qualitative)")
MU_FLESH_HDPE = P(.3, "-", "estimate", "wet flesh on polymer")
MU_PAD_SKIN = P(.3, "-", "estimate", "rubber on slimy skin")
MU_PAD_FLESH = P(.5, "-", "estimate", "rubber on wet flesh")
MU_INSERT_HANDLE = P(.8, "-", "repo", "shared Franka finger pad friction")
# the FR3's protective stops (the repo's qualified robot interface: label_application, barcode_scanning)
REFLEX_TORQUE_NM = P((10., 20.), "N m", "read", "libfranka examples/examples_common.cpp setDefaultBehavior: upper "
                     "collision torque thresholds (nominal, acceleration) on every joint; as barcode_scanning/params.py")
REFLEX_TORQUE_CONTACT_NM = P((100., 100.), "N m", "read",
                             "libfranka examples/force_control.cpp and cartesian_impedance_control.cpp "
                             "setCollisionBehavior: 100 N m upper torque thresholds on every joint for contact tasks "
                             "(read 2026-09-29 from github.com/frankaemika/libfranka master); the 'contact' collision "
                             "profile a deployment may choose for cutting and pressing (World Config collision_profile)")
# ASSISTED phantom knife rig (owner-side only): a force-limited servo on the knife handle, as a hand holds a knife,
# instead of a pose the handle follows with unbounded force (a pose-following rig driven into the head's bone or the
# board squeezes flesh until the explicit solver goes non-finite)
RIG_FORCE_MAX = P(150., "N", "authored", "a firm push of a hand on a knife; above the largest cutting load the "
                  "assisted demonstrations need (55 N, B5 rock-and-draw), so it only caps rams into bone or "
                  "the board")
RIG_TORQUE_MAX = P(15., "N m", "authored", "a wrist holding a knife against a press on its blade (wrist strength of "
                   "the order of 10-20 N m: recalled, not a sourced measurement); 5 N m let a 33 N press 150 mm down "
                   "the blade pitch the knife 5.6 deg")
RIG_STIFFNESS = P((2e5, 600.), "N/m, N m/rad", "authored", "the servo's tracking stiffness (translation, rotation): "
                  "0.13 mm behind the path under a 25 N press cut, 0.24 deg under 2.5 N m; critically damped with the "
                  "knife handle's own mass and inertia (bodies.k_body_servo)")
REFLEX_RECOVERY_S = P(1.0, "s", "authored", "after a collision reflex the arm holds and refuses new targets this long "
                      "(the real FR3 stays stopped until an error recovery); as barcode_scanning/params.py")
JOINT_ACCEL_PANDA = P((15., 7.5, 10., 12.5, 15., 20., 20.), "rad/s^2", "read",
                      "libfranka include/franka/rate_limiting.h kMaxJointAcceleration (Panda), read 2026-09-29; the "
                      "arm's commanded joint targets are rate-limited to these and to the joint speed limits")
JOINT_ACCEL_FR3 = P((10.,)*7, "rad/s^2", "prior",
                    "Franka's FR3 control-parameter documentation lists 10 rad/s^2 for every joint (recalled; the page "
                    "could not be re-read on 2026-09-29, its URL returned 404); the Panda's read values are 7.5-20")
MU_TWEEZER_BONE = P(.4, "-", "estimate", "textured stainless tips on bone")

# --------------------------------------------------------------------------------------------- robot, workpieces
FRANKA_HAND = P(dict(continuous_n=70., stroke_m=.08, finger_speed=.05, pad=(.018, .018)), "-", "read",
                "Franka Hand manual v1.2 / datasheet", use="reference")
FR3 = P(dict(payload=3., reach=.855, tau=(87, 87, 87, 87, 12, 12, 12), k_trans=(10, 3000), k_rot=(1, 300)), "-",
        "read", "FR3 datasheet v2.4")
TROUT = P(dict(gutted_kg=.2895, length=.315, depth=.0736, width=.034), "kg,m", "measured(generator)+read",
          "31.5 cm gutted farmed trout digitised from FDA RFEIMG-0305, MRI steak cross-sections (Frontiers in Genetics "
          "2021); gutted mass from the generator's own volume x density. FishBase length-weight gives "
          "~268 g gutted at this length")
TROUT_FILLET_YIELD = P((.69, .03), "fraction of gutted", "read", "Chefs Resources seafood yields: trout skin-on 69 %")
SALMON_SIDE = P(dict(kg=.9, length=.32, width=.17, loin=.038), "kg,m", "derived+estimate",
                "front half of a 1.5-1.7 kg side (Mowi 2024 yields); thickness unsourced")
TENDERLOIN = P(dict(kg=.55, length=.33, diameter=.065), "kg,m", "search+estimate", "retail 450-680 g, 30-45 cm, 5-7.5 cm")

IMPS = P(dict(thickness_tol_le_25mm=5e-3, uniformity=5e-3, weight_tol_lt_170g=7e-3, lean_exposed_denuded=.90,
              flake_fat_max=3e-3, tail_split_max=25e-3, suspension_edge=13e-3), "m,kg", "read",
         "USDA IMPS Series 400 Fresh Pork (2014) sections 3.3, 3.4, 6.3, 6.4, item 415")
CODEX_BONE = P(dict(per_kg=1, length=10e-3, diameter=1e-3, ignore_length=5e-3), "-", "read", "Codex CXS 190", use="reference")


def ledger():
    """All parameters as a JSON-ready dict (name -> value, unit, label, basis)."""
    out = {}
    for k, val in globals().items():
        if isinstance(val, P):
            out[k] = asdict(val)
    return out


# why each "reference" entry is not a model input
REFERENCE_ONLY = {
    "FISH_HEAD_E": "the head is a rigid body",
    "FISH_HEAD_R": "the head is a rigid body, removed at its joint, not cut through",
    "BONE_CRUSH": "vertebrae are rigid and cannot be cut; thin bones break at BONE_UTS",
    "BONE_SHEAR": "no edge-on-bone shear failure is modelled; thin bones break at BONE_UTS",
    "PIN_PULL_SALMON": "the source of the derived PIN_SHAFT_TAU (a model input)",
    "PIN_BREAK_SALMON": "pin-boning (B1) is refused",
    "PIN_LENGTH_SALMON": "the salmon's pin bones are generated by anatomy/salmon.py",
    "TAU0_BLADE_TISSUE": "blade-tissue adhesion is not modelled (Coulomb friction, MU_BLADE_TISSUE)",
    "FRANKA_HAND": "the hand's limits come from the robot catalog's model (robots/models)",
    "CODEX_BONE": "the specification the evaluators' bone checks follow",
    "SKIN_FLESH_G": "the skin membrane shares the flesh's surface nodes: no skin-flesh interface (skinning, B4, is refused)",
    "SKIN_FLESH_SIGMA": "as SKIN_FLESH_G",
}
