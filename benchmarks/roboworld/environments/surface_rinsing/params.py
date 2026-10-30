"""Parameter ledger: every physical constant the rinsing simulation reads, with value, unit, label and basis.

Labels: datasheet, regulation, standard, literature-read, search, measured, repo, literature, derived, estimate,
authored, open. `python -m surface_rinsing.params` prints the table.

The robot, rigid contact, controller gains and reflexes are classroom_arrangement's qualified Mobile FR3 Duo stack with
painting's contact refinements, unchanged (classroom_arrangement/params.py as CP); only what is new for rinsing is
listed here.
"""

import math

from classroom_arrangement import params as CP

LEDGER = []


def _p(name, value, unit, label, basis):
    LEDGER.append(dict(name=name, value=value, unit=unit, label=label, basis=basis))
    return value


GRAVITY = CP.GRAVITY
DT = CP.DT
GPM = 3.785411784e-3/60.0          # m3/s per US gallon per minute
PSI = 6894.757                     # Pa
INCH = 0.0254

# ------------------------------------------------------------------ fluids
AIR = _p("AIR", dict(rho=1.20, mu=1.81e-5), "kg/m3, Pa s", "literature", "dry air at 20-23 C")
WATER = _p("WATER", dict(rho=998.0, sigma=0.072, sigma_cmc=0.030, cp=4186.0, mu_43=6.16e-4),
           "kg/m3, N/m, N/m, J/(kg K), Pa s", "literature",
           "water near 20 C; surface tension of pure water and of an anionic detergent solution above its CMC; mu(T) by "
           "the Vogel fit below (1.00 / 0.62 / 0.46 mPa s at 20 / 43 / 60 C)")
VOGEL = _p("VOGEL", dict(A=2.414e-5, B=247.8, C=140.0), "Pa s, K, K", "literature",
           "mu = A 10^(B / (T - C)), T in K (water, 0-100 C, within 2.5 %)")


def viscosity(T_C):
    v = VOGEL
    return v["A"]*10.0**(v["B"]/(T_C + 273.15 - v["C"]))


# ------------------------------------------------------------------ supply and outlets
SUPPLY = _p("SUPPLY", dict(p_static=60*PSI, T_hot=60.0, T_cold=15.0, lag_s=0.002), "Pa, C, C, s",
            "regulation / estimate", "60 psi: DOE/ASTM F2324 test pressure (codes cap static pressure at 80 psi); hot 140 F "
            "as DOE cites ASHRAE; cold 15 C estimate (seeds 5-25); first-order line lag 2 ms vs the 1.4 ms riser+hose "
            "response time rho L Q / (A dp) (derived)")
PRSV = _p("PRSV", dict(Q_rated=1.15*GPM, p_rated=60*PSI, n_orifice=9, cone_deg=7.0, ring_r=0.006, Cv=0.98,
                       body_len=0.20, face_r=0.027, mass=0.55, lever_arm=0.060, travel=0.010, crack_n=13.0, full_n=22.0,
                       crack_travel=0.001, lever_mass=0.020, hose_links=9),
            "SI", "search / literature-read / derived / estimate",
            "T&S B-0107: 1.15 gpm at 60 psi (retailer listings, search); jet momentum flux rho Q v = 2.1 N reproduces "
            "the DOE spray-force band (M2, 1.4 % from DOE's own force/flow line); 9 orifices (estimate, hole count "
            "open) giving 0.60 mm effective jets; cone 7 deg (estimate); lever 13 N to crack and 22 N full open over "
            "10 mm (patent 3-5 lbf, literature-read; travel estimate); body 0.55 kg (estimate); the unit's 44 in hose and "
            "overhead spring are a 0.63 m free hose (9 links of 7 cm) on a balancer slide with 0.6 m of travel (estimate)")
REEL = _p("REEL", dict(Q_rated=3.89*GPM, p_rated=60*PSI, hose_L=35*12*INCH, hose_D=0.375*INCH, n_orifice=9,
                       cone_deg=6.0, ring_r=0.007, Cv=0.98, free_hose=2.5, hose_lin_mass=0.22, hose_r=0.009),
            "SI", "search / derived / estimate",
            "T&S B-7132-01 open reel, 35 ft x 3/8 in hose, EB-0107 3.89 gpm valve (search); with the hose's Blasius loss "
            "the 60 psi supply gives 3.3 gpm and 5.1 N (M2); the last 2.5 m of hose is a capsule chain on the floor, "
            "the rest stays on the reel (estimate); 0.22 kg/m with water (estimate)")
FAUCET = _p("FAUCET", dict(Q_reg=2.2*GPM, d_out=0.016, Cd=0.6, A_max=1.6e-5, turn_deg=90.0, dead_deg=5.0,
                           friction_nm=0.25, blade_len=0.10, blade_mass=0.06),
            "SI", "regulation / estimate / literature",
            "2.2 gpm laminar stream regulator (federal maximum for lavatory/kitchen faucets; the regulator class an "
            "estimate); 16 mm outlet (estimate); wrist-blade quarter-turn cartridges with 5 deg dead band, Coulomb "
            "friction 0.25 N m (estimate; ADA 309.4 caps operating force at 5 lbf = 22.2 N, 0.9 N m at a 4 in blade); "
            "each cartridge's open area makes the regulator, not the valve, limit full flow (estimate)")
LINE = _p("LINE", dict(riser_L=0.46 + 1.12, riser_D=0.0095, faucet_L=0.6, faucet_D=0.0127), "m", "estimate",
          "pre-rinse riser 18 in + 44 in hose at 3/8 in ID; faucet body and spout 0.6 m at 1/2 in")

# ------------------------------------------------------------------ jets, drops, splatter
PACKETS = _p("PACKETS", dict(capacity=1 << 18, max_age_s=3.0, drop_mass=4.0e-6, drop_d_min=2.0e-4), "-, s, kg, m",
             "authored", "ring of 262144 packets; a packet older than 3 s is counted as lost (none should be); film "
             "edge and drip outflow is released as drops of >= 4 mg (2 mm)")
JET = _p("JET", dict(Cv=0.98, breakup_k=12.0), "-", "literature / literature-read",
         "orifice velocity coefficient 0.98; capillary breakup length L_b/d = 12 sqrt(We) (1 + 3 Oh) (Weber 1931 as quoted "
         "by Bhunia & Lienhard 1994)")
SPLATTER = _p("SPLATTER", dict(onset_a=130.0, onset_b=5e-7, plateau=((5550.0, 0.30), (7800.0, 0.37), (31000.0, 0.75)),
                               ring_d=3.0, speed=0.45, elev=(math.radians(15.0), math.radians(45.0)),
                               drop_we_splash=2100.0),
              "-", "literature-read / estimate",
              "Bhunia & Lienhard (1994): 5 % onset at l/d = 130/(1 + 5e-7 We^2), their correlation inside its range and "
              "the Fig. 8 plateaus outside (literature-read, read from page images); droplets leave from a ring of "
              "3 jet diameters at 0.27-0.63 of the jet's NORMAL speed and 15-45 deg, all round, plus the jet's tangential "
              "velocity (so oblique jets splatter forward; estimate); falling drops splash only above We Oh^-0.4 = 2100 "
              "(Cossali et al., literature)")
DUEZ = _p("DUEZ", dict(K=0.44), "-", "repo / literature-read",
          "inertial-capillary lip separation U_c^2 = K (1 + cos theta) (sigma/rho) (r_i + e/2) r_i / e^3, K 0.44 "
          "(Duez et al. 2010; wine_pouring's closure)")

# ------------------------------------------------------------------ films
FILM = _p("FILM", dict(cfl=0.4, beta=1.2, accel_filter_s=0.008, h_drip=1.0e-3, drip_gn=-2.0, h_wet=2.0e-6,
                       max_stride=8, re_turb=1000.0, wetting_deg=40.0),
          "-, -, s, m, m/s2, m, -, -, deg", "literature / estimate / authored",
          "CFL 0.4 per substep; momentum shape factor 6/5 (parabolic profile); body acceleration low-pass 8 ms; "
          "films facing down (g_n < -2 m/s2) shed drops above 1 mm (0.4 capillary length, estimate); "
          "a cell counts as wetted above 2 um; up to 8 ticks per film step while the CFL allows (authored); turbulent "
          "wall friction above Re_f 1000; substrates with theta_e >= 40 deg bead up (literature order)")
SUBSTRATES = _p("SUBSTRATES", dict(
    glass=dict(theta=20.0, gamma_m=1.0e-7, h_break=0.0, mu_pad=(0.8, 0.7, 0.2)),
    glaze=dict(theta=25.0, gamma_m=1.0e-7, h_break=0.0, mu_pad=(0.8, 0.7, 0.2)),
    steel=dict(theta=70.0, gamma_m=3.0e-7, h_break=6.0e-5, mu_pad=(0.8, 0.6, 0.2)),
    aluminum=dict(theta=75.0, gamma_m=3.0e-7, h_break=6.0e-5, mu_pad=(0.8, 0.6, 0.2)),
    polymer=dict(theta=95.0, gamma_m=5.0e-7, h_break=1.0e-4, mu_pad=(0.8, 0.6, 0.25)),
    skin=dict(theta=100.0, gamma_m=5.0e-7, h_break=1.0e-4, mu_pad=(0.8, 0.5, 0.3)),
    tile=dict(theta=60.0, gamma_m=3.0e-7, h_break=4.0e-5, mu_pad=(0.8, 0.6, 0.3)),
    concrete=dict(theta=40.0, gamma_m=3.0e-7, h_break=2.0e-5, mu_pad=(0.8, 0.6, 0.3))),
    "deg, kg/m2, m, -", "literature / estimate",
    "equilibrium contact angles (clean glass/glaze 10-30, stainless ~70, LLDPE ~95, tomato skin ~100 deg; literature "
    "orders); adsorbed-surfactant capacity Gamma_m 0.1/0.3/0.5 mg/m2 (estimate bracketing the measured 2-7 ug per "
    "dish); h_break: mean depth of the beads a non-wetting substrate retains on a VERTICAL wall, 0.02-"
    "0.10 mm (estimate: canopy interception storage is ~0.05 mm per unit leaf area, MIKE SHE user guide, search-read; "
    "a few grams cling to a rinsed plate); the engine holds films below h_break / sqrt(sin slope) immobile (a pinned "
    "drop: hysteresis against rho g h sin), at most 5 h_break near level: 0.5 mm on level polymer, 0.3 mm on level "
    "steel; glass and glaze sheet (0); finger-pad friction "
    "dry / wet clean / soapy: pad on glass 0.8 (repo) / 0.7 (estimate in 0.4-1.0, search) / 0.2 (fingertip on SDS "
    "solution, search); others estimates")

# ------------------------------------------------------------------ species
SURFACTANT = _p("SURFACTANT", dict(c_wash=0.75, cmc=0.42, K=20.0, k_d=0.05, gamma_ref=5.0e-8, K_sigma=50.0),
                "kg/m3, kg/m3, m3/kg, 1/s, kg/m2, m3/kg", "search / literature / estimate",
                "wash solution 3 g/L product at 25 % actives = 0.75 g/L (search-level dosing; seeds 0.5-1.25); CMC of LAS "
                "0.42 g/L (literature); Langmuir adsorption dGamma/dt = k_a c (Gamma_m - Gamma) - k_d Gamma with "
                "K = k_a/k_d 20 m3/kg and k_d 0.05 /s (estimates: a 20 s desorption tail); friction soap fraction "
                "phi_s = 1 - exp(-(Gamma + s)/Gamma_ref), Gamma_ref 0.05 mg/m2 (estimate); Szyszkowski-type sigma(c) "
                "with K_sigma 50 m3/kg (estimate)")
FOAM = _p("FOAM", dict(phi=0.95, R=3.0e-4, tau_y=5.0, half_life=300.0, k_dil=1.0), "-, m, Pa, s, 1/s",
          "literature / measured / estimate",
          "foam liquid fraction 5 %, bubble radius 0.3 mm; Mason et al. yield stress 0.51 (sigma/R)(phi - 0.635)^2 = 5 Pa "
          "(M2); half-life 300 s (estimate); collapses at 1 /s once the liquid around it is below half the CMC "
          "(estimate)")
SOIL = _p("SOIL", dict(rho_dry=1500.0, t_soak20=5.9, sigma_dry=2.0e5, sigma_wet=150.0, k_i=2.0, k_e=0.05,
                       k_front=2.0e-3, k_abr=6.0e-9, to_suspended=1.0, m_clean=1.0e-6, soak_doubling_C=15.0),
          "kg/m3, s, Pa, Pa, kg/(m2 s), kg/(m2 s Pa), s m/kg, 1/Pa, -, kg/m2, C", "open -> calibrated / literature-read",
          "dried tomato paste (ASTM F2324-03 recipe): solids 1.5 g/cm3; rehydrated strength sigma_wet 150 Pa (the yield "
          "stress of tomato paste is of order 100-300 Pa; literature order, not fetched; a 2 kPa estimate "
          "leaves everything between the jets' 1-cell tracks untouched); hydration time t_soak (20 C) and strength "
          "sigma_c(w) = sigma_dry (1 - w)^2 + sigma_wet: impingement removal k_i (p_imp/sigma_c - 1)+, shear erosion "
          "k_e (tau_w - sigma_c)+, Wilson front speed k' M (k' for model soils 1e-4..2e-3 s m/kg, Wilson et al. 2014, "
          "literature-read), Archard abrasion; t_soak calibrated on bench B6 toward the FSTC mean of 21 s per plate at "
          "1.15 gpm: the time to clean steps with the raster's revisit of the last dirty spot (18.0 s for t_soak <= 5.56 s, "
          "22.5 s for 5.86-6.75 s), so 21 s itself is not reachable; "
          "frozen at 5.9 s (22.5 s, inside the verified-valve 16-26 s); held out on the 0.64 and 1.48 gpm valves: "
          "1.48 gpm 16.0 s passes, 0.64 gpm is not clean at 45 s, a FAILED hold-out (film modes scale with flow); a 1 ug/cm2 = 1e-5 kg/m2 gauge is the visible-residue limit; "
          "below 1 mg/m2 a cell counts as clean for the front law (authored); hydration doubles per 15 C (estimate)")
DUST = _p("DUST", dict(sigma_dry=4.0, sigma_wet=0.5, t_soak20=2.0, load=0.5e-3), "Pa, Pa, s, kg/m2", "estimate",
          "field dust on produce skin and debris film in a used can: weakly adhered silt removed by film shear "
          "(0.5-4 Pa); 0.5 g/m2 initial load (estimate; 0.2-1 g/m2)")
EGG = _p("EGG", dict(sigma_dry=2.0e5, sigma_wet=300.0, t_soak20=40.0, load=0.08), "Pa, Pa, s, kg/m2", "estimate",
          "dried egg (a protein film that softens slowly and stays tougher than tomato paste when soaked): rehydrated "
          "strength 300 Pa, hydration time 40 s at 20 C (halving per 15 C: 13.8 s under the 43 C wash water), 80 g/m2 dried "
          "(about one egg's residue over the pan's 0.03 m2 bottom); the pan has already soaked 120 s when the episode "
          "starts (seeds: 20-240 s, layout.setup), so a short-soaked pan must be left to soak; estimates, exercised by "
          "demo A5 scrub")
VRL = _p("VRL", 1.0e-5, "kg/m2", "search", "visible-residue limit 1 ug/cm2 (Jenkins & Vanderwielen; range 0.4-10 by "
         "substance, Forsyth; Fourman & Mullen ~4 ug/cm2)")

# ------------------------------------------------------------------ pools, immersion, drains
POOL = _p("POOL", dict(Cw=0.6, Cd=0.6, Cd_body=1.1, Ca=0.5, k_m0=1.0e-5, k_m1=1.0e-3, lld_k=0.94, slosh_zeta=0.01,
                       samples=4000, h_lld_max=2.0e-4),
          "-", "literature / estimate",
          "weir and orifice discharge coefficients 0.6; immersed drag C_d 1.1 and added-mass 0.5 (estimates); "
          "pool-surface mass transfer k_m = k0 + k1 |v| (estimates); Landau-Levich-Derjaguin withdrawal film "
          "h = 0.94 l_c Ca^(2/3), capped at 0.2 mm (literature); slosh damping 1 % (estimate, SP-106 mode)")

# ------------------------------------------------------------------ sinks and fixtures
POT_SINK = _p("POT_SINK", dict(bowl=(18*INCH, 24*INCH, 14*INCH), rim_z=0.940, drainboard=24*INCH, wall_t=0.002,
                               drain_d=0.035, splash_h=0.20, fill_depth=10*INCH, roll_r=0.012),
              "m", "datasheet / estimate",
              "Advance Tabco FS-3-1824-24RL: three 18 x 24 x 14 in bowls, 24 in drainboards, 37 in rim (datasheet); 14 "
              "gauge (2 mm) walls; 1.5 in basket drain with a 35 mm open bore (estimate); bowls filled to 10 in "
              "(authored, Food Code practice)")
DISH_TABLE = _p("DISH_TABLE", dict(depth=30*INCH, height=34*INCH, sink=(20*INCH, 20*INCH, 5*INCH), splash_h=10.5*INCH),
                "m", "search", "Advance Tabco DTS-S70-72R soiled dish table with a 20 x 20 x 5 in pre-rinse sink")
PREP_SINK = _p("PREP_SINK", dict(bowl=(24*INCH, 24*INCH, 12*INCH), rim_z=0.914, drainboard=24*INCH, table_h=0.876),
               "m", "estimate", "single-compartment prep sink, 36 in rim; work table 34.5 in")
CAN_BAY = _p("CAN_BAY", dict(size=2.0, slope=1.0/96, drain_r=0.076, cell=0.010), "m, -, m, m",
             "literature / authored", "floor sloped 1/8 in per ft to a 6 in round floor-drain strainer")

# ------------------------------------------------------------------ items
ITEMS = _p("ITEMS", dict(
    rocks_glass=dict(H=0.095, r_top=0.0445, r_bot=0.035, wall=0.003, base=0.012, mass=0.32),
    mixing_glass=dict(H=0.150, r_top=0.0445, r_bot=0.032, wall=0.0035, base=0.012, mass=0.42),
    plate=dict(R=0.1143, well_r=0.0889, well_depth=0.012, foot_r=0.075, foot_h=0.005, t=0.0045, H=0.022, mass=0.47),
    bowl=dict(R=0.076, H=0.060, foot_r=0.040, wall=0.004, mass=0.35),
    colander=dict(R=0.140, r_bot=0.095, H=0.125, wall=0.0008, hole_d=0.004, pitch=0.009, mass=0.45),
    tomato=dict(r=0.0143, c=0.0135, mass=0.0113),
    can=dict(R=0.263, r_bot=0.224, H=0.692, wall=0.003, mass=4.5),
    sponge=dict(L=0.111, W=0.066, H=0.017, scour=0.005, mass=0.008, E=5.0e4, phi=0.9),
    pan=dict(R=0.127, r_bot=0.100, H=0.051, wall=0.003, mass=1.05)),
    "m, kg", "search / estimate",
    "Libbey 15587 12 oz rocks glass 95 mm tall, 89 mm rim, 0.32 kg (search); 16 oz mixing glass (estimate); 9 in "
    "vitrified china plate 0.47 kg (search); 6 in bowl (estimate); 5 qt stainless colander with 4 mm holes on 9 mm "
    "pitch (estimate); cherry tomato 28.6 x 27 mm, 11.3 g at 980 kg/m3 (estimate; floats); Rubbermaid BRUTE 2632 559 mm (over its lip) x 692 mm, 4.5 kg (search; body 526 mm at the top and 448 mm at the base so that the cavity holds the rated 32 gal, derived); "
    "Scotch-Brite heavy-duty scrub sponge 111 x 66 x 17 mm (search); Wear-Ever 10 in fry pan 1.05 kg (search)")
DENSITY = _p("DENSITY", dict(glass=2500.0, glaze=2400.0, steel=7900.0, aluminum=2700.0, lldpe=920.0, pp=900.0,
                             hdpe=950.0, tomato=980.0, brass=8500.0), "kg/m3", "literature",
             "soda-lime glass, vitrified china, stainless, aluminium, polyethylenes, polypropylene, ripe tomato flesh, "
             "chrome-plated brass")
FRICTION = _p("FRICTION", dict(glass_steel=0.35, china_steel=0.30, lldpe_tile=0.45, tomato_steel=0.40,
                               rack_steel=0.35, hose_floor=0.6, wet_factor=0.85),
              "-", "estimate", "dry static coefficients between ware and stainless, can and quarry tile; water halves "
              "nothing here, but wet contacts are 15 % lower (estimate)")
SPONGE = _p("SPONGE", dict(E_wet=5.0e4, A_eff=3.0e-3, damping=30.0, q_stop=0.0105, tau_cap=0.1, k_abr=1.2e-5,
                           scour_factor=4.0, sigma_ref=1.0e3, k_fill=2.0, k_mix0=0.2, k_mix1=5.0, conform=0.006,
                           end_r=0.040),
              "Pa, m2, N s/m, m, s, kg/(m2 Pa m), -, Pa, 1/s, 1/s, 1/m, m, m", "search / estimate",
              "wet cellulose foam modulus 50 kPa (range 20-200 kPa; a saturated non-reinforced cellulose sponge measured "
              "0.2 MPa, search) over an effective 30 cm2 of a pinch, spring F = E A (eps + 3 eps^3) (painting's nap form); "
              "slide damping 30 N s/m (the joint's implicit damping); full compression 10.5 mm of 12 mm (the stop); the contact patch is every charted cell within 6 mm of the pressed block and facing it (the soft cellulose wraps a corner; a rigid block's face cannot enter one); capillary uptake time 0.1 s; Archard-form abrasion "
              "k_abr p v sigma_ref/sigma_c with the scour face 4x the cellulose; immersed refill 2 /s and exchange "
              "0.2 + 5 |v| per s (all estimates); the firm core's ends are arcs of 40 mm radius in plan (a real "
              "sponge's corners are its softest part: a square-cornered core against the pan's 97 mm wall puts its corners "
              "3-4 mm into the wall's 4 mm convex sector hulls, which push them out sideways through the sectors' side "
              "planes, 1.4 kN each way, and lock sponge and pan together, A5 scrub)")
ROLLING = _p("ROLLING", dict(tomato=5.0e-4, tomato_torsion=1.0e-3), "m", "estimate",
             "MuJoCo rolling and torsional friction lengths of a cherry tomato: rolling-resistance coefficient ~0.035 of "
             "a soft fruit times its 14 mm radius (estimate; painting's 15 um lacquered-handle value let tomatoes roll "
             "down a 1 % pitch indefinitely)")
IMPACT_GATE = _p("IMPACT_GATE", 1.0, "m/s", "authored",
                 "provisional normal impact-speed gate for glass and china on steel or tile (open until "
                 "anchored on ASTM C368-type data)")
BREAKABLE = _p("BREAKABLE", ("rocks_glass", "mixing_glass", "plate", "bowl"), "item kinds", "authored",
               "the glass and china the impact gate applies to, by item kind; the stainless bowl and colander, the "
               "aluminium pan, the polyethylene can, the sponge and the tomatoes do not break at these speeds (matched "
               "by kind, not by item name, so the prep sink's stainless clean bowl is not held to the china limit)")

LEVER = _p("LEVER", dict(armature=1.0e-3), "kg m2", "numerical / measured",
           "rotor inertia added to a squeeze lever's hinge so that a 20 g lever (5e-5 kg m2 about its pivot) is "
           "well-conditioned against finger contacts: 0.28 kg effective at the pad. The hinge is critically damped "
           "against its return spring, c = 2 sqrt(k I) (0.12 N m s/rad; a poppet valve's seals and the water it displaces "
           "do not let a lever ring), response time sqrt(I/k) = 17 ms. With 0.05 kg m2 and 0.02 N m s/rad, a "
           "14 kg lever rings at 1.35 Hz (measured, damping ratio 0.02): a part-squeezed valve hunts between no "
           "flow and nearly full flow (A6's gentle pass), which a real lever does not do")
# ------------------------------------------------------------------ evaluator thresholds
HOLD = _p("HOLD", dict(noslip_iterations=100, noslip_tolerance=1.0e-10, slip_off=0.005, slip_on=0.002), "-, -, m/s, m/s",
          "repo / measured",
          "MuJoCo's post-solve no-slip pass at classroom_arrangement's qualified setting, on unless a "
          "contact slides faster than 5 mm/s (back on below 2 mm/s): with it on while a 76 g wet sponge was pressed and "
          "slid, it rocked, sank through the pan's floor hull and left through the sink (A5 scrub; 3 of 12 press-and-drag trials "
          "faulted with it, 0 without); raising impratio instead (30, 100) still faulted and let tomatoes "
          "roll away. "
          "Measured on a rocks glass pinched by its rim at 20 N per finger among the pot sink's ~350 constraint rows "
          "(P18): no pass, the glass creeps 0.05 mm/s; 30-100 iterations stop it (< 0.001 mm in 2 s) at 10 ms per "
          "tick; painting's 1000 iterations / 1e-15 cost 44 ms at 300 and ~190 ms at 1000 (88x and ~390x) for no "
          "measurable gain here. At rest without it glasses creep 0.4 um/s down the drainboard and tomatoes roll 0.1 mm/s "
          "on the colander's floor (P11), with it 0 and 0.2 um/s")
CRITERIA = _p("CRITERIA", dict(surf_per_item=10e-9, surf_gauge=1.0e-6, sanitize_s=30.0, drained_glass=2e-6,
                               drained_bowl=5e-6, drained_colander=10e-6, drained_can=100e-6, settle_s=30.0, settle_min_s=2.0,
                               sponge_suds_g_L=0.02, sponge_wrung=0.60,
                               floor_water=50e-6, place_tilt_deg=20.0, rack_lean_deg=30.0, park_m=0.15, contam_frac=1e-3),
              "kg, kg/m2, s, m3, ..., s, m3, deg, m, -", "search / regulation / authored",
              "residual surfactant <= 10 ug per item (measured residues after ordinary rinsing 2.1-7.2 ug per dish, "
              "search) and no 1 cm2 gauge above 1 mg/m2 (authored local cap); >= 30 s cumulative immersion in the "
              "sanitizer (Food Code 4-703.11, conservative); free-water limits inside an item, pool plus the film above what "
              "clings (authored: glass 2 mL, bowl and pan 5 mL, colander 10 mL, can 100 mL; <= 0.6 % of capacity); "
              "evaluation after a settle branch of 2 s and then until the world is quiet, at most 30 s (authored, painting's "
              "rule: Evaluator.evaluate); a rinsed sponge's pore water <= 0.02 g/L "
              "of surfactant (under 5 % of the CMC: no lather; estimate) and wrung to <= 60 % of its free capacity (authored: "
              "a 90 N hand squeeze; the robot's 70 N pinch leaves 65 %, so it must press the sponge on a surface); ware "
              "set down within 20 deg of mouth-down / upright (authored); a plate in a peg-rack slot within 30 deg of "
              "vertical (the 35 mm slots between 60 mm posts allow ~21 deg, measured in demo D1)")


def table():
    rows = ["| Parameter | Value | Unit | Label | Basis |", "|---|---|---|---|---|"]
    for e in LEDGER:
        v = e["value"]
        if isinstance(v, float):
            v = f"{v:g}"
        elif isinstance(v, dict):
            v = ", ".join(f"{k} {x:g}" if isinstance(x, float) else f"{k} {x}" for k, x in v.items())
        rows.append(f"| `{e['name']}` | {v} | {e['unit']} | {e['label']} | {e['basis']} |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(table())
