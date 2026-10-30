"""Label stocks and adhesive surface classes with their provenance.

Every number carries a provenance tag:
  datasheet  read from a manufacturer/standards document
  derived    computed from datasheet values in this file
  estimate   engineering estimate with a stated reason
  calibrated fitted to a datasheet observable by a peel-test fixture
"""

from dataclasses import dataclass, field, replace
import math

G = 9.81


@dataclass(frozen=True)
class LabelStock:
    name: str
    E_md: float            # Pa
    E_cd: float            # Pa
    nu: float
    t_face: float          # m, facestock caliper
    t_adhesive: float      # m
    rho_s: float           # kg/m^2 including adhesive
    yield_strain: float    # bending yield strain for permanent set
    strength_md: float     # N/m tensile
    strength_cd: float     # N/m tensile
    drag: float = 0.9      # N s / m^3, normal air drag per area (estimate)
    sources: dict = field(default_factory=dict)

    @property
    def thickness(self):
        return self.t_face + self.t_adhesive

    @property
    def strength_shear(self):
        """In-plane shear strength (N/m), estimate: von Mises relation on the geometric-mean tensile
        strength, sqrt(X_MD X_CD)/sqrt(3) (no datasheet value; 2.06 kN/m paper, 8.8 kN/m PET)."""
        return math.sqrt(self.strength_md*self.strength_cd)/math.sqrt(3.)

    def D(self, E):
        return E*self.t_face**3/(12.*(1.-self.nu**2))

    @property
    def D_md(self):
        return self.D(self.E_md)

    @property
    def D_cd(self):
        return self.D(self.E_cd)


# Direct-thermal paper shipping/specimen label. Facestock 51 lb/ream (25x38 in)
# = 75.5 g/m^2 at 3.3 mil (83.8 um); MD tensile 29 lbf/in = 5.08 kN/m
# [LabelsOnTime DT spec]. Adhesive 17.5 g/m^2 at ~20 um [3M 7818 coat weight,
# used as the class value]. E is an estimate bounded below by strength/strain
# (60.6 MPa at 1.5-2 % gives a >= 3.4 GPa secant modulus).
PAPER_DT = LabelStock(
    name="direct-thermal paper, permanent acrylic",
    E_md=5.0e9, E_cd=2.5e9, nu=0.30,
    t_face=83.8e-6, t_adhesive=20e-6, rho_s=(75.5+17.5)*1e-3,
    yield_strain=0.005, strength_md=5.08e3, strength_cd=2.5e3,
    sources=dict(E="estimate (bounded by datasheet strength)", t_face="datasheet",
                 rho_s="derived", yield_strain="estimate", strength_md="datasheet",
                 strength_cd="estimate ~half of MD"))

# 3M 7818-class polyester label: facestock 0.08 mm, adhesive 0.02 mm
# [3M 7818 TDS]; PET modulus 4.1-4.3 GPa ASTM D882 [Mylar A]; PET density
# 1390 kg/m^3 (estimate) -> 111 g/m^2 + 17.5 g/m^2 adhesive.
PET_7818 = LabelStock(
    name="polyester film, acrylic 310-class",
    E_md=4.2e9, E_cd=4.2e9, nu=0.38,
    t_face=80e-6, t_adhesive=20e-6, rho_s=(1390*80e-6*1e3+17.5)*1e-3,
    yield_strain=0.03, strength_md=190e6*80e-6, strength_cd=190e6*80e-6,
    sources=dict(E="datasheet (Mylar A)", t_face="datasheet (3M 7818)", rho_s="derived",
                 yield_strain="estimate", strength_md="datasheet (Mylar A 190 MPa)"))

STOCKS = {"paper": PAPER_DT, "pet": PET_7818}


@dataclass(frozen=True)
class Surface:
    """What a label's ADHESIVE face does when it touches this surface."""
    name: str
    gamma90: float        # J/m^2 at 5 mm/s after 10 min dwell (90 deg peel reference)
    alpha_dwell: float    # fractional increase per decade of dwell (10 min -> 72 h fit)
    p_rough: float        # Pa, contact-formation pressure scale
    mu: float             # facestock-side Coulomb friction
    stiffness: float      # Pa/m normal contact stiffness per area
    source: str = ""
    v_ref: float = 5e-3   # m/s speed at which gamma90 was measured (ASTM D3330 / FINAT: 300 mm/min)
    shear_stiffness: float = float("inf")   # Pa/m tangential compliance of the surface itself (rubber pads)


# Adhesive-layer compliance: E_a/h_a with E_a ~ 0.3 MPa (below Dahlquist 0.1 MPa
# shear class, Creton 2003) over 20 um. Estimate.
ADHESIVE_NORMAL_STIFFNESS = 1.5e10   # Pa/m
ADHESIVE_SHEAR_STIFFNESS = 5.0e9     # Pa/m
W0 = 0.1                             # J/m^2 intrinsic work of adhesion (literature order)
RATE_EXPONENT = 0.6                  # Gent-Schultz exponent (estimate, 0.4-0.7 typical)
V_REF = 5e-3                         # m/s, ASTM D3330 / FINAT 300 mm/min
P_TACK = 1e3                         # Pa quick-stick wetting without applied pressure (estimate)
# Wet-out follows SUSTAINED pressure, so a single impact spike does not count as a full application. The
# pressure that drives the wet-out law is the local contact pressure passed through a first-order filter of
# time constant WETOUT_TIME. Anchor (derived, inferred): the roughness pressure p_r comes from the adhesive's
# modulus at 1 Hz (Dahlquist criterion, the modulus of a ~1 s contact), so the equilibrium law describes
# pressure held for about a second; tau = 1/(2 pi * 1 Hz) = 0.16 s reaches 99.8 % of it after 1 s. A 1 ms spike contributes 0.6 %
# of its pressure; a hand-roller contact of 30-60 ms about 20-30 %. Not a measured wet-out kinetics.
WETOUT_TIME = 1./(2.*math.pi)        # s
DWELL_REF = 600.0                    # s, 10 min datasheet dwell

# 3M 7818 (PET/acrylic 310) 90 deg peel 10 min (N/cm) -> gamma90 = F/b; the dwell slope comes from the
# same sheet's 90 deg 10 min -> 72 h ratio (the model's law is a 90 deg law). [3M 7818 TDS] 90 deg peel,
# N/cm, 10 min -> 72 h at room temperature: SS 3.8 -> 4.5, PC 4.0 -> 4.7, PP 1.8 -> 2.6, glass 3.7 -> 5.1,
# HDPE 1.8 -> 2.2, LDPE 1.3 -> 2.4 (pinned 2017 sheet, pp. 3-4, read 2026-09-25). The 180 deg column gives a
# different slope (e.g. PP 2.0 -> 2.0, slope 0, where the 90 deg table gives +44 %).
# [3M 7818 TDS] = "3M Durable Label Materials" data sheet, May 2017 revision
# (multimedia.3m.com/mws/media/1561216O/3m-durable-label-materials.pdf); the current 7818 sheet
# (media/2399682O) keeps the construction but publishes a shorter peel table, so the 2017 revision is
# the pinned source. Steel is the calibration point of the peel model; the other substrates' values
# are supplied to the model as inputs, so reproducing them (e.g. PP) is an implementation check of the
# peel law across Gamma, not a held-out prediction of substrate chemistry.
def _pet(ncm90, r72):
    return ncm90*100., (r72-1.)/math.log10(72*3600/600)


# ANGLE DEPENDENCE (bounded, not applied). The same 3M 7818 table gives the 10 min 180 deg peel next to
# the 90 deg one. Read through the elastic Rivlin-Kendall balance (G = F(1 - cos theta)/b) the two angles
# imply Gamma180 = 2 F180/b and Gamma90 = F90/b, whose ratio is 2.2-3.4 on the six substrates: the backing's
# bending and the adhesive's process zone dissipate more at high angle. The label solver's adhesion law is
# elastic in angle (Gamma90 at every angle), because the peel angle a 3 mm front node can read is too coarse
# to carry an angle law: a peel at exactly 90 deg reads a median 69 deg (10-90 % range 49-96 deg after
# smoothing over 3 elements), one at exactly 180 deg a median 119 deg (76-152 deg); a law applied to that
# reading raises the calibrated 90 deg point by 24 %.
# Consequence, bounded where it matters: the liner and the silicone exit plate are referenced to a 180 deg
# datasheet release, so the elastic 180 deg peel reproduces them; applications peel at low angle; the
# fingertip-pad bond (open proxy, 7818 LDPE 130 J/m^2 at 90 deg, 440 J/m^2 at 180 deg) carries the gap and
# is bounded up to 440 J/m^2 in the sensitivity study.
def _ratio(ncm180, ncm90):
    return 2.*ncm180/ncm90


ANGLE_RATIOS_7818 = dict(steel=_ratio(4.7, 3.8), pc=_ratio(5.1, 4.0), pp=_ratio(2.0, 1.8),
                         glass=_ratio(5.7, 3.7), hdpe=_ratio(2.6, 1.8), ldpe=_ratio(2.2, 1.3))


SS = _pet(3.8, 4.5/3.8)
PC = _pet(4.0, 4.7/4.0)
PP = _pet(1.8, 2.6/1.8)
GLASS = _pet(3.7, 5.1/3.7)
HDPE = _pet(1.8, 2.2/1.8)
LDPE = _pet(1.3, 2.4/1.3)

SURFACES = {
    # Corrugated kraft: no datasheet; paper-label adhesive (RP37 class, 90 deg
    # 8 N/25 mm = 320 J/m^2 on the FINAT reference) reduced for rough porous kraft.
    "kraft": Surface("corrugated kraft liner", 250., 0.10, 40e3, 0.45, 5e9,
                     "estimate from RP37 320 J/m^2 [search]"),
    "pp_tape": Surface("polypropylene packing tape", 180., 0.0, 2e3, 0.30, 1.5e10,
                       "PP value of 3M 7818 table [datasheet]"),
    "paper_label": Surface("paper label facestock", 250., 0.10, 20e3, 0.40, 5e9, "estimate"),
    "pet_tube": Surface("PET tube wall", 320., 0.07, 2e3, 0.35, 1.5e10,
                        "estimate: RP37 on smooth polyester"),
    "steel": Surface("stainless steel", SS[0], SS[1], 2e3, 0.30, 1.5e10, "3M 7818 SS [datasheet]"),
    "hdpe": Surface("HDPE", HDPE[0], HDPE[1], 2e3, 0.30, 1.5e10, "3M 7818 HDPE [datasheet]"),
    "pp": Surface("polypropylene", PP[0], PP[1], 2e3, 0.30, 1.5e10, "3M 7818 PP [datasheet]"),
    "glass": Surface("glass", GLASS[0], GLASS[1], 2e3, 0.20, 1.5e10, "3M 7818 glass [datasheet]"),
    "ldpe": Surface("LDPE", LDPE[0], LDPE[1], 2e3, 0.40, 1.5e10, "3M 7818 LDPE [datasheet]"),
    # Robot finger pads: Franka fingertip polymer unknown; LDPE-class adhesion is
    # an explicit proxy. Pad compliance 2 MPa / 2 mm (estimate).
    # Bonds to the pad act in series with the rubber: normal 1e9 Pa/m (2 MPa / 2 mm) and shear
    # G/h = (E/3)/h = 3.3e8 Pa/m for nearly incompressible rubber (same estimate).
    "finger_pad": Surface("robot finger pad (LDPE-class proxy)", LDPE[0], 0.0, 2e3, 0.8, 1e9,
                          "open: proxy", shear_stiffness=3.3e8),
    # Release liner: 3M 7818 release range 5-50 g/2 in at 180 deg -> 1.9-19 J/m^2 [datasheet]. Its only
    # datasheet value IS a 180 deg release, the geometry a peel-bar printer uses (Gamma referenced at 180 deg).
    # Peel-mode printers/dispensers use easy-release liners: the low end of the measured range.
    "liner": Surface("silicone release liner (dispensing grade)", 2.0, 0.0, 2e3, 0.25, 1.5e10,
                     "datasheet range 1.9-19 J/m^2 at 300 in/min (TLMI); low end for dispensing grade",
                     v_ref=.127),
    # The silicone-coated exit plate is the same release chemistry as the liner: it takes the liner's
    # datasheet-anchored reference (2 J/m^2 at 300 in/min).
    "peel_plate": Surface("dispenser exit plate (silicone-coated)", 2.0, 0.0, 2e3, 0.25, 1.5e10,
                          "liner datasheet (same silicone release chemistry)", v_ref=.127),
    "laminate": Surface("bench laminate", 200., 0.05, 2e3, 0.35, 1.5e10, "estimate"),
    # ASTM D3330 roller covering: ~6 mm of Shore A 80 rubber. Gent's Shore relation gives E = 9.4 MPa;
    # as a thin layer on a steel core its normal stiffness is E/h = 1.6e9 Pa/m (compressible estimate;
    # a bonded incompressible layer is stiffer). Adhesion to it: LDPE-class proxy (facestock faces it).
    "rubber_roller": Surface("roller rubber (Shore A 80, 6 mm layer)", 130., 0.0, 2e3, 0.9, 1.6e9,
                             "ASTM D3330 roller spec; E from Shore A 80 (Gent); layer estimate"),
    "pp_tool": Surface("PP squeegee", PP[0], 0.0, 2e3, 0.30, 5e9, "3M 7818 PP [datasheet]"),
    "nonstick": Surface("non-bonding enclosure", 0.0, 0.0, 2e3, 0.3, 1.5e10, "authored"),
}


def with_gamma(surface, gamma90):
    return replace(surface, gamma90=gamma90)
