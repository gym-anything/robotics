"""Parameter ledger. Every physical constant the world's physics reads lives here with its value, unit and provenance
label, so the physics reads one source. The evaluator's acceptance thresholds stay beside their checks in task.py with
their basis; render-only constants stay in render.py with theirs.

Provenance labels: read (primary source opened), secondary (the value as quoted in another
source that was opened; the primary was not), search (snippet only), literature (from knowledge, not fetched), repo,
measured, derived (arithmetic shown), estimate, authored, open (no digital basis: the value is disclosed and its
sensitivity is reported, never tuned per task).
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class P:
    value: object
    unit: str
    label: str
    basis: str
    # "model": read by the world's physics or evaluators;
    # "reference": recorded for reference only
    use: str = "model"


def v(p):
    return p.value


C2K = 273.15
SIGMA_SB = 5.670374419e-8

# ============================================================================================== steel (preset A)
STEEL_RHO = P(7850., "kg/m3", "read", "EN 1993-1-2:2005 s3.2.2 unit mass of steel")
STEEL_CP = P("EN1993", "J/kg K", "read", "EN 1993-1-2:2005 eq. 3.2a-d, 20-1200 C: 425+0.773t-1.69e-3t^2+2.22e-6t^3 (to "
             "600 C), 666+13002/(738-t) (600-735), 545+17820/(t-731) (735-900), 650 (900-1200); held at 650 above 1200 C "
             "to the solidus (estimate), liquid 808 J/kg K")
STEEL_CP_LIQ = P(808., "J/kg K", "read", "A36 liquid specific heat (Kim, Zhang & DebRoy 2003 / Zhang 2004 table)")
STEEL_K = P("EN1993", "W/m K", "read", "EN 1993-1-2:2005 eq. 3.3: 54-0.0333 t to 800 C, 27.3 to 1200 C; 31 W/m K from "
            "1200 C to the liquidus (Mills 2002 iron 29.3 at 900 C, 39.4 at 1500 C; estimate between)")
STEEL_SOLIDUS = P(1472., "C", "read", "A36 solidus 1745 K (Kim 2003)")
STEEL_LIQUIDUS = P(1512., "C", "read", "A36 liquidus 1785 K (Kim 2003)")
STEEL_LATENT = P(247e3, "J/kg", "read", "pure iron heat of fusion 247 J/g (Mills 2002, read via an unofficial copy)")
STEEL_GAMMA = P(1.2, "N/m", "read", "A36 liquid surface tension at the melting point, 1.2 N/m (Kim 2003 Table II; Zhang "
                  "2004 part II Table II, which pairs it with dgamma/dT = -0.35e-4 N/m K)")
STEEL_RHO_LIQ = P(6950., "kg/m3", "derived", "iron 7030-0.86(T-1538) kg/m3 (Mills) at ~1630 C")
F_MARANGONI = P(420./31., "-", "estimate", "[the value is read; its use for 0.9 mm wire at ~240 A is an assumption, and it "
                "is the main control on penetration: sensitivity in E3] conduction-only weld models represent pool "
                "convection by an enhanced liquid conductivity: the Penn State conduction-only GMAW fillet model used "
                "420 W/m K, ~20x the 21 W/m K they used for the solid, to match measured bead width and depth (Kim, "
                "Zhang & DebRoy 2003 Table II; quoted in Zhang, Kim & DebRoy 2004 part II p. 5222 and Fig. 4; read), "
                "calibrated together with an arc efficiency of 54 % into the surface Gaussian plus 10-12 % droplet "
                "heat and r_b = 5.0 mm, for 1.32 mm wire at 287-362 A. Here eta = 0.80 in all, ~1.2x the heat the 420 "
                "was matched with: the value is used outside its calibration. The convective model of the same group "
                "needed 88.2 W/m K (x4.2; Zhang 2004 part II Table II). E3: x0.5 / x2 change the bead's fused area by "
                "-2.7 % / -1.1 %. The liquid's 31 W/m K is multiplied by 420/31 so that it is the published 420; a "
                "smaller factor puts the surface at the boiling cap (x4 evaporates 70 % of the arc's heat)")
STEEL_T_BOIL = P(2860., "C", "literature", "iron boiling point; surface temperature cap, excess booked as evaporation")
STEEL_EMISSIVITY = P(.7, "-", "read", "EN 1993-1-2:2005 s2.2(2): emissivity of the steel surface 0.7 for carbon steel "
                     "(read 2026-10-10); oxidised 0.79 (Engineering ToolBox)")
H_AIR = P(12., "W/m2 K", "read", "still-air convection used in weld models (Koch & Enzinger 2012); EN 1991-1-2 4-9")
H_CONTACT_STEEL = P(1500., "W/m2 K", "estimate", "clamped steel-steel thermal contact conductance (literature range "
                    "1e3-1e4); plate on table, faying surfaces of unfused parts")
K_Y_STEEL = P(((20., 1.), (400., 1.), (500., .78), (600., .47), (700., .23), (800., .11), (900., .06), (1000., .04),
               (1100., .02), (1200., 0.)), "-", "read", "EN 1993-1-2:2005 Table 3.1 effective yield strength factor")
WELD_UTS = P(482e6, "Pa", "literature", "E70/ER70S-6 minimum tensile strength 70 ksi (AWS A5.18 classification)")
WELD_SHEAR_RATIO = P(.852, "-", "read", "ultimate shear strength of a fillet weld on its throat / the filler's tensile "
                     "strength, longitudinal loading: 0.852 (1 + 0.5 sin^1.5 theta) F_EXX from tests (AWS D1.1:2000 "
                     "Commentary C2.14.4, Lesik 1990 for E70); the codes' 0.6 F_EXX (AISC 360 J2.4) and 0.30 F_EXX (D1.1 "
                     "allowable) are design values with margins, not the load at which a weld breaks")
WELD_TRANSVERSE_FACTOR = P(1.0, "-", "read", "strength factor of this single-sided fillet across the weld (transverse "
                           "shear, tension and peel; bonds.py). The directional increase 1 + 0.5 sin^1.5 theta (1.5 "
                           "across: AISC 360-16 J2.4b, AWS D1.1, CSA S16) was derived from fillets welded on BOTH sides "
                           "of the loaded plate (Butler & Kulak 1971, Miazga & Kennedy 1989, Lesik & Kennedy 1990, Ng et "
                           "al. 2004). 40 single-sided fillets under transverse load (Thomas & Tousignant 2022, J. "
                           "Struct. Eng. 148(9) 04022118, postprint read; Table 6, mean test-to-predicted ratios against "
                           "CSA S16 with the sin theta factor, nominal 0.67 x 1.5 = 1.005 F_EXX on the throat) broke at "
                           "0.761 (x1.005 = 0.765 F_EXX) when nominally concentric, 0.463 / 0.370 (0.465 / 0.372 F_EXX) "
                           "with the root opened by bending (15 / 30 mm load offset, 'b' series) and 0.955 / 1.043 "
                           "(0.96 / 1.05 F_EXX) with the root closed by bending ('a' series), against 1.493 (1.50 F_EXX) "
                           "for double-sided specimens: 'when the sin theta factor is used, predictions of weld strength "
                           "are unsafe for all single-sided fillet welds'; CSA S16:19 13.13.2.2 allows the factor only for "
                           "'cases other than single-sided fillet welds connected to an element in tension' (quoted "
                           "there). Without it the rule's transverse strength 0.852 F_EXX lies 11 % above the concentric "
                           "single-sided tests (COV 0.12), about 1.8-2.3x the root-opening ones and 11-19 % under the "
                           "root-closing ones: the strength depends on which way the load bends the root, which this "
                           "single factor does not represent")
BOND_STIFFNESS = P((19600., 12918.), ("N m/rad", "N"), "estimate", "rotational stiffness of the bond's weld equality "
                   "and the capacity it belongs to: the cold tacks' 19 600 N m/rad at their 12 918 N (MuJoCo impedance "
                   "0.9995), inside the 2e3-1.5e5 N m/rad derived for a real tacked joint (throat bending over the "
                   "leg; the weld in tension against bearing at the root); the bond's stiffness scales with its "
                   "capacity (both grow with the fused throat area and the steel's strength as it cools). A fixed "
                   "stiffness would let a 0.7 N first sliver of fusion take the bracket's support from the contact and "
                   "crack")
BOND_SOFT_C0 = P(9.8, "N m/rad", "measured", "MuJoCo's soft-constraint law for the bond in this cell, k = C0/(d(1-d)) at "
                 "time constant 4 ms: static 7.5 N m peel at d = 0.99 / 0.998 / 0.9995 gave 981 / 4905 / 19 604 N m/rad "
                 "(fit within 1 %); the impedance for a wanted stiffness follows")
SHRINK_TRANSVERSE = P(.8e-3, "m", "read", "transverse shrinkage allowance per fillet weld ~0.8 mm (TWI Job Knowledge 33)", use="reference")
SHRINK_ANGLE_REF = P((1.15, 22.7e-6, .150), "deg, m2, m", "derived", "angular distortion of the single-sided fillet: "
                     "the bracket turns toward the weld about the root line as its weld metal shrinks on cooling ('in "
                     "a single side fillet weld, non-uniform contraction produces angular distortion of the upstanding "
                     "leg', TWI Job Knowledge 33, read). Anchor: measured free, tacked A36 T-joints with one fillet, "
                     "GMAW, ER70S-6 (La Porta Neto 2017, UFRGS MSc, Table 5.1, read; 300 mm long, flange 9.525 mm, web "
                     "6.35-12.7 mm, arc energy U I / v 910 / 1385 J/mm): the web-to-flange angle on the weld's side "
                     "closed by 0.83 / 1.05 deg at 910 J/mm and 1.33 / 1.44 / 1.34 deg at 1385 J/mm (the flange's own "
                     "angle theta1 0.95-1.07 / 1.34-1.49 deg). This cell's 1/4 in fillet at Lincoln's procedure (29 V, "
                     "230-241 A, 14 in/min: 1125-1179 J/mm; 21.7-22.7 mm2 deposit against their 17.5 / 26.3 mm2 at the "
                     "two levels) lies 0.45-0.59 of the way between: 1.14-1.19 deg for the web-to-flange angle, "
                     "1.19-1.24 for theta1; 1.15 deg for 22.7 mm2 over 150 mm. Their specimens were unrestrained; here "
                     "the plate's back wing is held by two 400 N toggle clamps and the weld-side wing is free to lift, "
                     "so the clamps do not restrain this angle (they give ~50 N m about the root against the ~700 N m "
                     "that would hold the weld-side wing flat, derived): the free value applies. Differences kept: "
                     "their beads were short circuit in Ar/25CO2 with 1.2 mm wire, welded flat on a 45 deg support; "
                     "individual specimens scatter +-0.1-0.3 deg. An estimate between Satoh & Terasaki's bead-on-plate "
                     "law (~0.6 deg at Q/h^2 = 9 J/mm3) and the AWS Welding Handbook fillet chart extrapolated in "
                     "deposit (~0.9 deg), both read via La Porta's reproductions, gives 0.75 deg, 35 % under these "
                     "measurements. The model scales the angle by the bonded sections' weld area (throat^2) times "
                     "length times the fraction of their cooling from the solidus to 20 C that has happened, against "
                     "22.7 mm2 x 150 mm (deposit-per-length logic); the plate is rigid in the model, so the whole "
                     "angle is the bracket's turn relative to it (in the cell the weld-side wing would lift instead: "
                     "the same web-to-plate angle at the weld). No law for tacks was found: the volume scaling gives "
                     "two 15 mm tacks ~0.1 deg; a rigid bracket held only by the tacks might turn up to ~0.5 deg (the "
                     "fillet's angle scaled by the tacks' throat^2 alone, 10.1 / 22.7 mm2); unquantified between")
A1_TEMP = P(727., "C", "literature", "eutectoid A1; HAZ display/diagnostic threshold")

# ============================================================================================== weld pool (pool.py)
POOL_TOE_SLOPE_MAX = P(88., "deg", "derived", "the face's angle to its chord at a toe up to which a slice holds its liquid "
                       "as a height over the chord; past it the slice holds the rest as a bulge that overhangs its toe "
                       "(pool._PolarSurface): on cold steel the liquid's contact "
                       "freezes as it touches and its base stays pinned while liquid keeps arriving (Tavakoli & "
                       "Kavehpour, ISCST 2012), so the face steepens past 90 deg. The edge advances onto more metal only "
                       "where the liquid would not freeze at the contact (_contact_freezes), and only once a slice is "
                       "this full. 88 deg is the steepest a height function over the chord represents reliably "
                       "[numerical]")
POOL_CAP_MODEL = P("solved", "-", "authored", "how much a slice's chord holds before its toes extend: 'solved', the "
                   "pressure at which the slice's own surface, with the liquid's weight and the "
                   "arc's pressure, meets a toe at POOL_TOE_SLOPE_MAX, bounded by the pressure at which its free span "
                   "would overhang; 'weightless' (kept as E3's comparison case), a circular arc at that slope, "
                   "2 gamma sin(theta)/L, which holds 2-6x too little under gravity. Decides the fillet's overlap "
                   "verdict and the bead's shape")
POOL_SOLID_SHRINK = P(.03, "-", "read", "volume shrinkage of C-Mn weld metal on solidification, ~3 % (TWI Job Knowledge "
                      "33; a further 7 % in cooling to room temperature is not modelled in the pool): the pool's liquid "
                      "holds this much more volume than the same metal solid; freezing metal is fed from the pool and the "
                      "last liquid freezes short, the crater. Mass is conserved")
POOL_LONG_SIGMA = P(1.0e-3, "m", "authored", "longitudinal sharing kernel of the pool's metal: shorter than the arc "
                    "pressure spot (sigma 1.78 mm at 240 A, ARC_PRESSURE_SIGMA), longer than one slice (0.5 mm); stands in "
                    "for surface tension along the weld. Sensitivity not measured")
POOL_BRIDGE_SLICES = P(3, "slices", "authored", "a liquid slice without fused toes takes the chord of a pool slice at "
                       "most this many slices (1.5 mm) away [numerical]")
POOL_TOE_SAMPLE = P(5e-5, "m", "authored", "arclength sampling of the joint's surface: toes move in these steps "
                    "[numerical]")
POOL_SPREAD_MAX = P(1e-3, "m", "estimate", "farthest an overfull pool's edge moves per 5 ms pool update per side "
                    "(0.2 m/s): liquid steel's capillary spreading is faster (~1 m/s, not sourced) and the freezing "
                    "contact line slows it (not quantified). Decides how fast overflow spreads as cold lap")
POOL_FAN_REACH = P(4e-3, "m", "authored", "an overhang's rays reach this far past a half-chord plus the centre's depth, "
                   "plus 4 mm (the polar surface's search length) [numerical: past the largest overhang seen, "
                   "2-3 mm]")
POOL_STACK_REACH = P(8e-3, "m", "authored", "cold-start droplets may be placed on cells within this distance along the "
                     "weld of where they land [numerical: the isotropic cold-start dome is far smaller]")

# ============================================================================================== arc (preset A)
WIRE_D = P(.889e-3, "m", "read", "0.035 in ER70S-6 (Lincoln SuperArc L-56/L-59)")
LESNEWICH_A = P(3.152e-4, "m/(s A)", "derived", "fit of the melt rate MR = a I + b L_s I^2 (Lesnewich's form) to six "
                "Lincoln L-56/L-59 / C4.200 rows (80/120/175 A short circuit at 12 mm, 195/230/275 A spray at 19 mm), "
                "with L_s = CTWD - 6.4 mm for spray (Lincoln's rule) and CTWD - DIP_ARC_LENGTH for short circuit "
                "(melt rates within -9..+12 % of the rows). No published coefficient for this wire was found")
LESNEWICH_B = P(1.859e-4, "1/(s A^2)", "derived", "same fit")
ARC_V0 = P(11.55, "V", "read", "anode + cathode fall of the free-flight arc u_a+c = 11.55 V (Golob & Tovornik 2010, "
           "EUROSIM, Table 1, read 2026-10-10), with R_arc 0.03 ohm and E 0.675 V/mm from the same table ('derived from "
           "experimental conditions'; their wire 1.02 mm2, ~1.14 mm; gas not stated). Not fitted: with these and "
           "WIRE_RESISTIVITY the Lincoln spray row (12.7 m/min, 29 V, 19 mm) comes out at ~241 A against the chart's "
           "230 A on a 6.5 mm arc (Lincoln's rule ~1/4 in) and the 600 ipm / 30 V row at ~264 A against 275 A (E2). "
           "Short-circuit transfer has no constant of its own (arc.py)")
ARC_RA = P(.03, "ohm", "read", "arc resistance R_arc (Golob & Tovornik 2010 Table 1). Thomsen 2005 (ACC, Table I, "
           "read) prints 0.022 ohm, taken from Moore, Naidu, Yender & "
           "Tyler 1997 (not opened), with U0 15.7 V, E_a 636 V/m and rho_r 0.2821 ohm/m for a 1.0 mm steel electrode; "
           "Thomsen: 'the values are only used for enabling numerical simulation'")
ARC_EA = P(675., "V/m", "read", "arc column field 0.675 V/mm (Golob & Tovornik 2010 Table 1); Thomsen 2005 / Moore "
           "et al. 1997 give 636 V/m (as ARC_RA)")
WIRE_RESISTIVITY = P(9.0e-7, "ohm m", "derived", "length-mean resistivity of the 0.035 in ER70S-6 stick-out at the "
                     "spray procedure. Measured resistivity of ER70S-6 wire (Lincoln SuperArc L-56, 0.045 in): 0.34 "
                     "uOhm m at 25 C, 1.14 at 600 C, 1.27 at 800 C, 1.44 at 1200 C, 1.51 extrapolated to 1395 C (Guest "
                     "2014, U. Alberta PhD, Fig. 3.10 p. 80, fast-heating curve, read from figure; Table 3.2 p. 79), "
                     "averaged over the stick-out's adiabatic Joule-heating profile v dH/dx = rho_e(T) j^2 with the "
                     "EN 1993-1-2 c_p: at 230 A, 12.7 m/min, "
                     "12.6 mm the wire's end reaches 1270 C to the solidus and the mean is 0.84-0.97 uOhm m; 0.55-0.85 "
                     "at the short-circuit rows (80-175 A). Golob & Tovornik 2010 use 1.0e-6")
CABLE_R = P(.006, "ohm", "estimate", "torch cable + contact tip + work lead resistance (~1.4 V at 230 A)")
CV_SLOPE = P(0., "V/A", "estimate", "inverter CV sources regulate terminal voltage: the chart volts are the loaded "
              "terminal volts; static slope taken as 0 (not found; 'comparatively flat', Lincoln C4.200)")
OCV = P(72., "V", "read", "maximum open-circuit voltage of the Miller Auto-Continuum 350, 72 VDC (spec sheet AU/10.0, "
        "Mar. 2022, p. 8, read 2026-10-10; Miller's web page and Continuum 350 listings print 75 VDC); Fronius TPS 320i "
        "73 V")
ARC_ETA = P(.80, "-", "read", "thermal efficiency factor k of EN 1011-1 for MIG/MAG, 0.8 (read via TWI and SSAB; TWI: "
            "'all efficiency factors relate to the efficiency of submerged arc welding', i.e. relative to SAW's 1.0); "
            "absolute calorimetry gives 0.74-0.84 for GMAW (DuPont & Marder 1995; Arevalo & Vilarinho 2012), the sweep; "
            "Haelsig et al. 2014 report > 15 % spread within one process")
DEPOSITION_EFF = P({"spray": .98, "short_circuit": .93, "globular": .89}, "-", "read",
                   "electrode efficiency (wire that ends up in the weld) by transfer mode, Lincoln C4.200 p. 16: axial "
                   "spray 98 %; short circuit in Ar/CO2 >= 93 %; globular in argon blends 88-90 % (85-88 % in CO2; p. 8 "
                   "says 87-93 %). The globular value also stands for the arc in air (no source for welding without "
                   "gas: estimate). GMAW solid wire 92-98 % (Meyer & Koltz 2023)")
TRANSITION_CURRENT = P(180., "A", "read", "globular-to-spray transition current of 0.035 in (0.9 mm) ER70S-6 in 90 % Ar / "
                       "10 % CO2: 175-185 A (Lincoln C4.200 Table 1, p. 9; 130-140 A in 98/2 Ar/O2)")
SPRAY_MIN_AR = P(.82, "-", "read", "least argon in the shielding gas for axial spray: '82% Argon + 18% CO2 -- the "
                 "effective limit' (Lincoln C4.200 p. 14; textbooks: 'at least 80 % argon')")
DIP_ARC_LENGTH = P(.889e-3, "m", "estimate", "a free-flight arc shorter than the droplet forming at the wire tip "
                   "(about one wire diameter) cannot stand: the drops bridge it and the transfer is short circuit "
                   "(Lancaster, The Physics of Welding: dip transfer when the arc is shorter than the drop) [estimate]")
DROPLET_T = P(2673. - C2K, "C", "read", "droplet temperature 2673 K (Kim, Zhang & DebRoy 2003)")
ARC_SIGMA = P(4e-3, "m", "open", "Gaussian sigma (r_b) of the arc surface flux. The Penn State conduction and convection "
              "fillet models used r_b = 5.0 mm with arc efficiency 54 % + droplet heat for 1.32 mm wire at 312-362 A "
              "(Zhang 2004 part II Table II, read); GTAW 1.5-3.6 mm (Tsai & Eagar, search). 4 mm for 0.9 mm wire at "
              "~240 A, an estimate between. Calibrated by Penn State together with eta 54 % + droplet heat (not 0.80). "
              "Sensitivity measured only on a bead on plate (E3: sigma 3 / 5 mm move the fused "
              "area by about +26 % / -24 %, penetration by one 0.25 mm cell); its effect on the fillet's legs, overlap "
              "and fusion checks is not measured")
ARC_FORCE = P((-.04017, 2.5533e-4), "N, N/A", "read", "arc force F = -0.04017 + 2.5533e-4 I (Kim 2003 fit to Lin & "
              "Eagar)")
ARC_PRESSURE_SIGMA = P((1.4875e-3, 1.233e-6), "m, m/A", "read", "arc pressure spread sigma = 1.4875 + 0.001233 I mm "
                       "(Kim 2003)")
POOL_TAU = P((6950.*(5e-3)**3/1.2)**.5, "s", "derived", "capillary-inertial response time of the pool surface, "
             "sqrt(rho L^3 / gamma) with L = 5 mm (pool half length) = 27 ms: the free surface cannot follow the arc "
             "force faster than this, so the arc pressure reaches the pool through a first-order lag of this time "
             "constant (without it the arc length / current / arc force / surface loop oscillates with period 2 "
             "pool updates, 220 <-> 274 A)")
SOURCE_I_MAX = P(400., "A", "read", "the power source's output ceiling: the Miller Auto-Continuum 350 outputs 20-400 A, "
                 "10-44 V (rated 350 A at 31.5 V, 100 % duty; 400 A up to 50 % duty; spec sheet AU/10.0, Mar. 2022, pp. "
                 "1 and 8, read 2026-10-10). Above it the model's source holds the current and its voltage falls below "
                 "the set point (what the source does at its ceiling is not in the spec sheet: the owner's manual was "
                 "not retrievable; an inverter's current regulation is inferred). Without a ceiling 25 m/min at 44 V "
                 "and 10 mm would run at 792 A")
STUB_SHORT_FRAC = P(.35, "-", "estimate", "dip transfer stubs when the wire is shorted more than this fraction of the "
                    "time: too little voltage for the wire feed, the wire reaches the plate unmelted and the feeder keeps "
                    "driving it (with a CV source the wire feed 'can be changed over a considerable range before "
                    "stubbing to the workpiece or burning-back into the contact tip occurs', Lincoln Electric, MIG/MAG "
                    "Welding Guide for GMAW, C4.200, 9/98 edition, printed p. 7 = PDF p. 12 of the copy at "
                    "referenciaict.uo.edu.cu, read 2026-10-11). Lincoln's short-circuit chart rows run at "
                    "shorted fractions 0.17-0.20 in this model; no measured threshold was found, so the "
                    "value is an estimate: a stable dip arc is shorted for well under half of its cycle")
WIRE_E = P(200e9, "Pa", "read", "Young's modulus of steel (EN 1993-1-1 3.2.6: 210 GPa; 200 GPa used for the wire): "
           "with the wire's second moment it bounds the push a stubbed stick-out can carry (Euler buckling, fixed in "
           "the contact tip, free on the plate: pi^2 E I / (4 L^2); 46 N on an 18 mm stick-out). A hot stick-out is "
           "softer, so this is an upper bound")
WIRE_PUSH_MAX = P(150., "N", "estimate", "most push the wire drive gives a stubbed wire before its rolls slip (no "
                  "drive-roll force was found); the push on the gun is the lesser of this and the stick-out's "
                  "buckling load (WIRE_E)")
STUB_PUSH_S = P(.1, "s", "estimate", "time the feeder keeps driving a stubbed wire before the source stops it (a "
                "robot source's stuck-wire detection; not found, estimate)")
DROPLET_CYL_DEPTH = P(1.5e-3, "m", "estimate", "depth of the spray droplet heat cylinder (Kumar & Bhaduri form "
                      "h_v - x_v + D_d evaluated for 0.9 mm wire at ~230 A: ~1-2 mm); short circuit: surface layer")
SHIELD_FLOW_SPRAY = P(19., "L/min", "read", "spray fillets 35-45 cfh (17-21 L/min) (Lincoln C4.200, 9/98 edition, "
                      "Table 9 p. 27, the 0.035 in 1/4 in fillet row: 500 in/min, 230 A, 29 V, 14 in/min; read "
                      "2026-10-11); 40 cfh = 18.9 L/min. AIR_DRAFT is read off Kobelco's chart at this flow",
                      use="reference")
NOZZLE_ID = P(15.9e-3, "m", "read", "5/8 in nozzle bore (Tregaskiss CA3)")
AIR_POROSITY = P((.01, .015), "-", "read", "1 % air entrainment -> distributed porosity, >1.5 % -> gross surface pores "
                 "(TWI Job Knowledge 042)")
DRAFT_LIMIT = P(2.2, "m/s", "read", "AWS D1.1 s5.12.1: wind at the weld <= 5 mph for GMAW. Not read by the model: the "
                "shielding's draft term is AIR_DRAFT (Kobelco's chart)", use="reference")
TRAVEL_ANGLE_LIMIT = P(22.5, "deg", "read", "travel angles beyond 20-25 deg: 'more spatter, less penetration' (Miller)")
TOUCH_SENSE_V = P(70., "V", "read", "Fronius TouchSensing ~70 V (<= 3 A) on the wire or nozzle")
SHIELD_CORE_D = P(1.5, "nozzle diameters", "estimate", "nozzle-to-work distance up to which the shielding jet's core "
                  "covers the pool (laminar exit 1.4-1.8 m/s at 17-21 L/min through 15.9 mm, derived); beyond it air "
                  "entrainment grows. Phenomenological; set so that the procedures' 19 mm CTWD gives no porosity")
AIR_PER_NOZZLE_D = P(.01, "1/nozzle diameter", "authored", "air fraction added per nozzle diameter of nozzle-to-work "
                     "distance beyond SHIELD_CORE_D (arc.air_fraction); no source ties stand-off to entrainment. "
                     "Sensitivity not measured")
AIR_PER_TRAVEL_DEG = P(.01/12.5, "1/deg", "authored", "air fraction added per degree of travel angle beyond "
                       "TRAVEL_ANGLE_LIMIT (1 % at 35 deg) (arc.air_fraction); no source ties torch angle to entrainment")


def _kobelco_draft_onset(flow_l_min):
    """Cross-draft speed (m/s) below which no air reaches the arc, by shielding-gas flow: Kobelco's Fig. 6 curves read
    with JIS grade 2 = 1 % air (AIR_DRAFT), linear between its 15 / 20 / 25 L/min curves (1.0 / 1.5 / 2.0 m/s)."""
    q = min(25., max(15., float(flow_l_min)))
    return round(1.0 + .1*(q - 15.), 6)


AIR_DRAFT = P((_kobelco_draft_onset(v(SHIELD_FLOW_SPRAY)), .01), "m/s, 1/(m/s)", "derived", "cross draft: no air "
              "below the first value, then the second per m/s (arc.air_fraction): at the cell's 19 L/min, 1 % air "
              "(porosity onset, AIR_POROSITY) at 2.4 m/s, 1.5 % (gross pores) at 2.9 m/s, 2 % at 3.4 m/s. Read from "
              "Kobelco, Fundamentals of MAG welding s10 Fig. 6 (CO2, 300 A, 30 cm/min bead on a 9 mm plate, 16 mm "
              "nozzle, 15 mm nozzle-to-work, wind across the weld; figure read 2026-10-11 to +-0.1 grade): JIS "
              "radiographic grade at 0 / 1 / 2 / 3 / 4 / 5 m/s is 1 / 1 / 2 / 4 / 4 / 4 at 15 L/min, 1 / 1 / 1 / 3 / 4 / "
              "4 at 20 L/min and 1 / 1 / 1 / 2 / 4 / 4 at 25 L/min. With grade 1 = under 1 % air, grade 2 = 1 % (TWI's "
              "distributed-porosity onset), grade 3 = 1.5 % (TWI's gross pores), grade 4 = 2 % or more (the mapping "
              "is an assumption), every curve is x_air = 0.01 (v - v0) with v0 = 1.0 / 1.5 / 2.0 m/s at 15 / 20 / 25 "
              "L/min, linear in flow; at 19 L/min v0 = 1.4 m/s (_kobelco_draft_onset, from SHIELD_FLOW_SPRAY). The "
              "cell's gas is 90Ar/10CO2 at ~240 A through a 15.9 mm nozzle 15.8 mm off the work: close to the chart's "
              "nozzle and stand-off, not its gas or current")
POROSITY_CURVE = P((.05, .15), "-", "authored", "frozen pore fraction (arc.porosity_fraction): 0 at or below "
                   "AIR_POROSITY's 1 % air, rising linearly to the first value at 2 % air, capped at the second (reached "
                   "at 4 % air). Only the 1 % and 1.5 % thresholds are read (TWI JK 042); the slope and cap are authored")
SPRAY_AIR_RULE = P("argon dilution", "-", "estimate", "air under the arc counts as a non-argon fraction against "
                   "SPRAY_MIN_AR, so spray stops at 8.9 % air in the 90/10 gas (arc.transfer_mode). The 82 % limit is "
                   "read for CO2 additions (Lincoln C4.200 p. 14); applying it to air is an estimate (no source found for "
                   "the air fraction at which spray stops). Porosity is a pool phenomenon from 1 % air, so a "
                   "spray arc can run in a shield too contaminated for sound metal: the defect is the pores", use="reference")

SCHEDULES = P({
    "spray_0p9_90-10": dict(wfs_m_min=12.7, voltage_v=29., chart_transfer="spray", gas="90Ar/10CO2", gas_ar=.90,
                            ctwd_m=.019, ref_amps=230., ref_travel_m_s=14*.0254/60,
                            use="1/4 in horizontal fillet (flagship A2)"),
    "sc_0p9_C25_175A": dict(wfs_m_min=6.4, voltage_v=20., chart_transfer="short_circuit", gas="75Ar/25CO2", gas_ar=.75,
                            ctwd_m=.012, ref_amps=175., ref_travel_m_s=None,
                            use="reference: short-circuit tacks need 12 mm, which the 5/8 in nozzle cannot reach in "
                                "this corner"),
    "sc_0p9_C25_vertical": dict(wfs_m_min=5.7, voltage_v=18., chart_transfer="short_circuit", gas="75Ar/25CO2",
                                gas_ar=.75, ctwd_m=.012, ref_amps=160., ref_travel_m_s=5.5*.0254/60,
                                use="vertical-up 1/4 in fillet (A4)"),
    "sc_0p9_C25_sheet": dict(wfs_m_min=3.8, voltage_v=17., chart_transfer="short_circuit", gas="75Ar/25CO2", gas_ar=.75,
                             ctwd_m=.0127, ref_amps=120., ref_travel_m_s=20*.0254/60, use="16 ga sheet (A5)"),
}, "-", "read", "Lincoln C4.200 / SuperArc L-56, L-59 typical operating procedures; "
   "chart_transfer is the chart's label for the row, not a source setting (the model's mode follows from the operating "
   "point: arc.CVSource.transfer_mode)")

# ============================================================================================== robots (world.py)
# the arms' protective stops and motion generator (world._protective_stops, world._motion_generator); the G1 has no
# stop reflex (its low-level joint interface has none) and its own constants stay in humanoid.py
FRANKA_FINGER_COUPLING_TAU = P(4e-3, "s", "estimate", "time constant of the MuJoCo joint equality coupling a Franka "
                               "Hand's two fingers: the real hand drives both from one motor through a rigid mechanism "
                               "(Franka Hand documentation, not opened), so the coupling is held as stiff as the cell's "
                               "4 ms contacts instead of MuJoCo's default 20 ms, under which the Panda's 14 g fingers "
                               "part by millimetres under the grip force (the arm's peak joint-3 reading while the "
                               "hand closes: 12.8 -> 8.6 N m, no reflex)")
UR_FORCE_LIMIT = P(150., "N", "secondary", "UR10e force limit, the default safety configuration (normal mode): 'The "
                   "default setting is 150 Newtons at the tool, but that can be adjusted down to 100 Newtons' "
                   "(Crossco, a UR integrator, 'Technical Guide: How to Change Default Force Setting with Universal "
                   "Robots', read 2026-10-11; UR's own manual was not opened for the value). Where UR applies it: 'the "
                   "forces exerted by the robot at the TCP/tool positions (Tool Force) and elbow joint (Elbow Force, "
                   "on e-Series only)' (PolyScope 5 manual, Force Limits, read 2026-10-11): here the net contact force "
                   "on the wrist links and the gun (tool) and on the upper arm and forearm (elbow), each group on its "
                   "own; contacts on the shoulder are not limited. UR states a 25 N tolerance (UR5e manual SW 5.22, "
                   "SF7); the model stops at the limit itself")
UR_BACK_OFF_TRACE = P(.05, "s", "authored", "after a force-limit stop a UR arm retraces this much of its own path: "
                      "'the robot will stop, and then back off to a position where the force limit is not exceeded' "
                      "(UR5e manual SW 5.22, SF7; translated from Spanish); how "
                      "far is not documented")
VELOCITY_STOP = P(1.05, "x rated speed", "authored", "an arm joint moving faster than this multiple of its rated speed "
                  "trips a protective stop (all arms). UR documents the joint-speed limit's tolerance as 1.15 deg/s "
                  "(SF4, UR5e manual SW 5.22), +1 % at 120 deg/s, and says the "
                  "robot slows to stay within it. The motion generator plans at PLAN_SPEED_FRACTION, so the PD servo's "
                  "measured 5.1 % tracking overshoot stays under this")
REFLEX_TORQUE = P((10., 20.), "N m", "read", "FR3 / Panda collision thresholds on every joint's external torque "
                  "(nominal, while accelerating): the upper torque thresholds libfranka's examples set, "
                  "setCollisionBehavior(lower_acc 20, upper_acc 20, lower_nominal 10, upper_nominal 10, ...) in "
                  "examples/examples_common.cpp setDefaultBehavior (github.com/frankaemika/libfranka master, read "
                  "2026-10-11; parameter order from include/franka/robot.h). These are the examples' choice, not a "
                  "factory default a robot program must use")
REFLEX_RECOVERY = P(1., "s", "authored", "a protective stop refuses the arm's commands this long, then unlocks: stands "
                    "for the robot program's own unlock -- UR's dashboard 'unlock protective stop' (immediately "
                    "allowed, except 5 s after too-frequent stops: UR Dashboard Server e-Series), libfranka's "
                    "automaticErrorRecovery()")
STOP_BACK_OFF = P((2., 1.1), "s, -", "authored", "window after the unlock in which the load the stopped arm held may "
                  "persist without a new stop, so that a command can back the arm off, and the margin above that load "
                  "that stops it again (the arm pushing further). A robot pinned against an obstacle above its "
                  "threshold otherwise re-trips every step and can never be moved; the "
                  "real controllers' behaviour under a load kept after the unlock is not in the sources read")
PLAN_SPEED_FRACTION = P(.95, "-", "authored", "the motion generator plans joint speeds this far under the rated speed: "
                        "the PD servo's tracking overshoot (measured 5.1 %: 2.2006 rad/s on a 2.094 rad/s shoulder) "
                        "otherwise trips VELOCITY_STOP on a legal full-speed move (E6)")
UR_ACCEL = P(4., "rad/s2", "authored", "UR10e motion-generator joint acceleration. URScript's movej default is 1.4 "
             "rad/s2 (literature, not opened); a robot program may set more")
FR3_ACCEL = P(10., "rad/s2", "literature", "FR3 motion-generator joint acceleration: Franka's FR3 control-parameter "
              "documentation lists 10 rad/s2 for every joint (recalled in filleting/params.py JOINT_ACCEL_FR3; its page "
              "returned 404 on 2026-09-29)")
PANDA_ACCEL = P((15., 7.5, 10., 12.5, 15., 20., 20.), "rad/s2", "read", "Panda joints 1-7 maximum joint acceleration, "
                "libfranka include/franka/rate_limiting.h kMaxJointAcceleration (github.com/frankaemika/libfranka "
                "master, read 2026-10-11)")

# ============================================================================================== copper brazing (B)
# Presets B (copper brazing) and C (electrical soldering) are NOT BUILT in this release: their parameters below are
# recorded from the research (use="reference") and read by nothing.
CU_RHO = P(8935., "kg/m3", "read", "copper (Wikipedia)", use="reference")
CU_CP = P(385., "J/kg K", "read", "copper 0.385-0.39 kJ/kg K (Engineering ToolBox)", use="reference")
CU_K = P(((0., 401.), (127., 392.), (327., 383.), (527., 371.), (727., 357.), (927., 342.)), "W/m K", "read",
         "pure copper (Engineering ToolBox); C12200's 339 at 20 C (literature) is the sensitivity", use="reference")
CU_MELT = P(1084.6, "C", "read", "copper melting point", use="reference")
CU_EMISSIVITY = P((.03, .78), "-", "read", "polished 0.023-0.052, heavily oxidised 0.78 (Engineering ToolBox)", use="reference")
BRASS_K = P(111., "W/m K", "read", "yellow brass (Engineering ToolBox)", use="reference")
TUBE_OD = P(.875*.0254, "m", "read", "7/8 in ACR / 3/4 nominal Type L OD (CDA Table 14.2)", use="reference")
TUBE_WALL = P(.045*.0254, "m", "read", "0.045 in wall (CDA Table 14.2e)", use="reference")
SOCKET_ID = P((.877*.0254, .881*.0254), "m", "read", "3/4 nominal fitting socket ID 0.877-0.881 in (CDA Table 14.9)", use="reference")
SOCKET_DEPTH = P(.75*.0254, "m", "read", "socket depth >= 0.75 in (CDA Table 14.9)", use="reference")
BCUP5_SOLIDUS = P(643., "C", "read", "BCuP-5 solidus 1190 F (CDA Table 14.12)", use="reference")
BCUP5_LIQUIDUS = P(802., "C", "read", "BCuP-5 liquidus 1475 F (CDA Table 14.12)", use="reference")
BCUP5_RHO = P(8380., "kg/m3", "read", "Sil-Fos 15 data sheet SG 8.38", use="reference")
BCUP5_GAMMA = P(1.0, "N/m", "open", "no Cu-P melt data found; between liquid tin 0.42-0.49 and copper ~1.3 N/m", use="reference")
BCUP5_MU = P(5e-3, "Pa s", "open", "no data; liquid Cu ~4 mPa s (literature)", use="reference")
BCUP5_THETA = P(20., "deg", "open", "contact angle on clean copper (self-fluxing); no data", use="reference")
ROD_SECTION = P((.125*.0254, .050*.0254), "m", "read", "1/8 x 0.050 in BCuP strip rod (CDA Table 14.11)", use="reference")
ACETYLENE_HHV = P(1470.*1055.06/.0283168, "J/m3", "read", "1470 BTU/ft3 gross heat content (Uniweld)", use="reference")
TIP_FLOW = P({1: 5., 2: 8.}, "SCFH", "read", "oxy-acetylene tip 1 (to 3/4 in copper) 5 SCFH, tip 2 (to 1 in) 8 SCFH "
             "(Uniweld chart)", use="reference")
FLAME_T_AD = P(3160., "C", "read", "neutral oxy-acetylene ~3160 C (Wikipedia MAPP page; ET lists 3480 adiabatic)", use="reference")
FLAME_ETA = P(.4, "-", "open", "flame-to-work transfer efficiency at the recommended inner-cone distance; no source; "
              "bounded by heating-time arithmetic (15-40 s) and automatic machines' ~36 s; swept 0.2-0.6", use="reference")
NITROGEN_TEST = P((2.76e6, 600., 35e3), "Pa, s, Pa", "authored", "standing nitrogen test 400 psig, 10 min, fail if "
                  "decay > 5 psi; below the annealed rating; verdict insensitive to the threshold", use="reference")

# ============================================================================================== soldering (C)
SN63_T = P(183., "C", "read", "Sn63Pb37 eutectic (NIST)", use="reference")
SN63_RHO = P(8400., "kg/m3", "read", "NIST Table 1.18", use="reference")
SN63_K = P(50.9, "W/m K", "read", "NIST Table 2.2.3 at 30 C", use="reference")
SN63_CP = P(188., "J/kg K", "derived", "rule of mixtures from NIST element values", use="reference")
SN63_LATENT = P(46.7e3, "J/kg", "derived", "rule of mixtures (Sn 59.6, Pb 24.7 J/g)", use="reference")
SN63_GAMMA = P(.417, "N/m", "read", "NIST Table 2.2.19, in air at liquidus + 50 C", use="reference")
SN63_THETA = P(16., "deg", "read", "on Cu with RMA flux (NIST Table 2.2.13)", use="reference")
SN63_RESISTIVITY = P(14.5e-8, "ohm m", "read", "NIST", use="reference")
CU_RESISTIVITY = P(1.69e-8, "ohm m", "read", "NIST Table 2.2.5", use="reference")
IRON_P_MAX = P(65., "W", "read", "Hakko FX-8801 65 W", use="reference")
IRON_SETPOINT = P(345., "C", "read", "Kester 44: 315-371 C for Sn63; MIL-DTL-24308G 360+-10 C; Hakko rule 183+50+100", use="reference")
TIP_HEAT_CAPACITY = P(2.5, "J/K", "estimate", "~6 g copper core + plating of a T18 tip; swept x0.5-2", use="reference")
TIP_R_HT = P(1.5, "K/W", "estimate", "heater-to-tip resistance from a ~30 K drop at ~20 W (JBC claim scaled); swept", use="reference")
H_TIP_DRY = P(5e3, "W/m2 K", "literature", "dry metal contact (Incropera Tables 3.1-3.2 range 1e3-1e4)", use="reference")
H_TIP_WET = P(2e5, "W/m2 K", "literature", "solder-filled interface 7e4-4e5 (Incropera)", use="reference")
PVC_DAMAGE_T = P(250., "C", "read", "PVC degradation onset near 250 C (Wikipedia)", use="reference")
CUP_BORE = P(1.07e-3, "m", "read", "MIL-DTL-24308G size 20 solder cup bore", use="reference")
CUP_LENGTH = P(2.5e-3, "m", "read", "cup length 2.36-2.79 mm", use="reference")
CUP_OD = P(1.5e-3, "m", "read", "cup outside diameter 1.45-1.55 mm", use="reference")
STRAND = P((19, .160e-3), "-, m", "literature", "22 AWG 19/34 stranding; strand diameter read (AWG table)", use="reference")
SOLDER_WIRE_D = P(.8e-3, "m", "read", "Kester 44 wire diameters 0.25-6.0 mm", use="reference")


def ledger():
    """Every entry as a dict (name, value, unit, label, basis, use), in file order."""
    out = []
    for name, val in globals().items():
        if isinstance(val, P):
            out.append(dict(name=name, **asdict(val)))
    return out
