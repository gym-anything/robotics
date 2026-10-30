"""Parameter ledger: every physical constant the paint simulation reads, with value, unit, label and basis.

Labels: datasheet (manufacturer document), retailer (product listing), published (paper/standard), search (read only
as a search excerpt), derived (computed here from other entries), estimate (judgement), authored (a design choice),
repo (a value qualified elsewhere in this repository). `python -m painting.params` prints the table.

The robot, rigid contact, controller gains and reflexes are classroom_arrangement's qualified stack, unchanged
(classroom_arrangement/params.py, imported as CP); only what is new for painting is listed here.
"""

import math

from classroom_arrangement import params as CP

LEDGER = []


def _p(name, value, unit, label, basis):
    LEDGER.append(dict(name=name, value=value, unit=unit, label=label, basis=basis))
    return value


GRAVITY = CP.GRAVITY
DT = CP.DT

# ------------------------------------------------------------------ air and room climate
AIR = _p("AIR", dict(rho=1.20, mu=1.81e-5), "kg/m3, Pa s", "published", "dry air at 20-23 C")
CLIMATE = _p("CLIMATE", dict(T_C=23.0, RH=0.50), "C, -", "authored", "seeded 18-27 C and RH 30-70 % (variants)")
WATER_RHO = _p("WATER_RHO", 997.0, "kg/m3", "published", "water at 25 C")

# ------------------------------------------------------------------ paints
# Interior latex eg-shel, ProMar 200 Zero VOC class [Sherwin-Williams data sheet]: 350-400 ft2/gal at 4 mil wet / 1.7 mil dry, 42 % volume solids,
# 10.81 lb/gal. Rheology: thixotropic Cross/Bingham fit to the TA/AZoM waterborne profile and Hester and Squire 1997.
LATEX = _p("LATEX", dict(
    rho=1295.3, phi0=0.42, sigma=0.032,
    eta_inf=0.1796, eta0_min=2.258, eta0_max=256.6, gamma_c=0.03089, n=0.676, tau_y_max=2.0, t_r=20.0,
    phi_m=0.64, phi_we=0.570, phi_set=0.64, k_m=0.006, S_um=0.187, haze=0.06, gloss_dry=0.45, gloss_wet=0.18,
    sorptivity_scale=0.3), "SI (S_um per um of dry film)", "datasheet + derived + estimate",
    "rho, phi0 [ProMar 200 data sheet]; eta/tau_y fit [TA/AZoM profile, Hester and Squire 1997] (tau_y_max 2 Pa and t_r 20 s estimates); sigma 30-35 mN/m estimate; "
    "phi_m 0.64 random close packing, phi_we 0.570 = viscosity x10; k_m 0.006 m/s from Eastman wet edge 2-4 min "
    "[Eastman open-time paper]; S 0.187 /um is an optical estimate, NOT a product hiding measurement or a consequence of rated "
    "spread rate. Class sensitivity bracket 0.08-0.30 /um from ~2-8 S/mil for acrylic/TiO2 paints in Dow "
    "Reducing Titanium Dioxide in Paints, figures 3 and 7; tint-base factors remain estimates. "
    "haze: the wet film's binder-"
    "emulsion scattering as a fraction of the white base's S (wet latex reads lighter, the milky binder clears on drying "
    "[paint-store guidance]): polymer particles in water scatter ~17x less per volume than TiO2 in binder (Rayleigh contrast "
    "((m^2-1)/(m^2+2))^2, m 1.11 against 1.8) at ~2x its volume -> 0.06, so 100 um of wet clear emulsion reflects ~40 % "
    "over black -- milky, translucent (estimate; only the render reads it); gloss as render roughness (eg-shel "
    "15-20 GU [ProMar 200 data sheet] -> 0.45, wet 0.18 -- a wet film is glossier than its dry sheen; estimate); vehicle sorptivity = water x sqrt(1/10) for 5-50 mPa s (estimate)")
LATEX_OPTICAL_RANGE = _p("LATEX_OPTICAL_RANGE", (0.08, 0.30), "1/um", "estimate bounded by coating-class data",
    "Dow, Reducing Titanium Dioxide in Paints, figures 3/7 span roughly 2-8 S/mil in acrylic/TiO2 formulations (divide by 25.4); "
    "a sensitivity envelope, not a confidence interval or ProMar-specific drawdown. Nominal S is 0.187.")
ACRYLIC = _p("ACRYLIC", dict(
    rho=1350.0, phi0=0.45, sigma=0.035,
    eta_inf=0.30, eta0_min=1.0, eta0_max=8.0, gamma_c=1.0, n=0.4, tau_y_max=0.5, t_r=10.0,
    phi_m=0.64, phi_we=0.57, phi_set=0.64, k_m=0.006, S_um=0.06, haze=0.06, gloss_dry=0.30, gloss_wet=0.14,
    sorptivity_scale=0.3), "SI", "search + estimate",
    "artist fluid acrylic: 800-1,200 cP Brookfield [Golden Artist Colors charts] -> eta at low shear 1-8 Pa s, mild yield (estimates); semi-"
    "transparent artist pigments S 0.06 /um (estimate); drying and binder haze as latex (same water-borne binder; "
    "estimate)")
ENAMEL = _p("ENAMEL", dict(
    rho=1100.0, phi0=0.25, sigma=0.028,
    eta_inf=0.03, eta0_min=0.03, eta0_max=0.05, gamma_c=1.0, n=1.0, tau_y_max=0.0, t_r=10.0,
    phi_m=0.64, phi_we=0.45, phi_set=0.60, k_m=0.030, S_um=0.20, haze=0.0, gloss_dry=0.10, gloss_wet=0.05,
    sorptivity_scale=0.5, solvent=True), "SI", "datasheet + estimate",
    "aerosol enamel after propellant flash (Rust-Oleum Painter's Touch 2X class, per its data sheet: 1-2 mil DFT, touch 20 min; acetone, n-butyl "
    "acetate, naphtha, xylene): Newtonian 0.03 Pa s at landing, solids 0.25, k_m x5 for the volatile solvent (touch in "
    "20 min); its evaporation does not depend on the room's water humidity (solvent=True: the rate is taken at the "
    "reference 50 % RH, so the nominal climate is unchanged); all estimates except the data sheet values")
LACQUER = _p("LACQUER", dict(ENAMEL, phi0=0.20, S_um=0.0, rho=1000.0), "SI", "estimate", "clear lacquer aerosol as the "
             "enamel without pigment (S = K = 0)")

TINT_BASES = _p("TINT_BASES", dict(white=(0.70, 1.0), pastel=(0.40, 0.58), medium=(0.15, 0.33), deep=(0.0, 0.020)),
                "LRV (Y) minimum, scattering / white base", "search + calibrated + estimate",
                "tint bases by TiO2 share of the solids: white ~30 %, pastel 15-20 %, medium 8-12 % [tint base guides] -> scattering "
                "in proportion (17.5/30, 10/30; estimates). Deep/accent bases carry no TiO2 (deep-tone bases are the pastel "
                "formula with the TiO2 removed [CoatingsTech 2013]), so their scattering is the extenders' and colorants': the factor "
                "0.020 is calibrated on coats-to-hide for a deep colour (SW 6244 Naval): one 43 um coat does not hide and "
                "two do over a grey tinted primer, two do not and three do over white (SW / practitioners' guidance for "
                "Naval [Sherwin-Williams colour page]; PPG Accent Color Base: 'two coat coverage guaranteed' [PPG data sheet]) -- dE 1 at ~63 um over the "
                "P-Shade primer (1.5 coats) and ~110 um over white (2.5 coats). The base a colour is mixed in is chosen by its lightness (estimate)")

# colours: sRGB of an infinitely thick film; linearised and inverted to K/S per channel (painting/paint.py)
COLORS = _p("COLORS", dict(
    alabaster=(237, 234, 224), extra_white=(238, 239, 234), lullaby=(203, 212, 212), accessible_beige=(209, 199, 184),
    naval=(47, 61, 76), primer=(240, 240, 236), pshade_primer=(110, 112, 114), cadmium_red=(196, 32, 38), ultramarine=(38, 48, 140),
    gloss_red=(178, 28, 34), clear=(255, 255, 255), drywall_paper=(196, 192, 182), plywood=(214, 182, 140),
    watercolour_paper=(244, 241, 232), canvas_cloth=(214, 204, 182), pet_clear=(235, 238, 240), pet_on_plywood=(206, 180, 142), photo=(120, 140, 160),
    oak_floor=(150, 104, 66), casing_white=(244, 243, 238), ceiling_white=(243, 243, 240), cardboard=(168, 132, 92)),
    "sRGB 0-255", "search / authored",
    "Sherwin-Williams SW 7008 Alabaster, SW 7006 Extra White, SW 9136 Lullaby (sRGB 203/212/212), SW 7036 Accessible "
    "Beige, SW 6244 Naval [Sherwin-Williams colour listings] (search-level); a mid-grey P-Shade tinted primer under a deep colour (ProMar 200 TDS: "
    "'use the recommended P-Shade primer to optimize hide'; the shade for Naval is an estimate); artist and "
    "substrate colours authored")

# ------------------------------------------------------------------ film fields
FILM = _p("FILM", dict(every_ticks=40, max_substeps=512, cfl=0.4, h_min=1e-9, set_thin=0.2e-6, struct_after_shear=0.0,
                       tex_wl_roller=2.0e-3, tex_wl_brush=1.0e-3, tex_wl_spray=0.6e-3, lap_visible=4e-6),
          "ticks, -, -, m, m, -, m", "derived / estimate",
          "film transport every 20 ms with CFL substeps (explicit limits 0.35-5.6 s at 1-2 mm cells after "
          "application); a film thinner than 0.2 um sets at once (estimate); an applicator's passage leaves lambda = 0 "
          "(steady-state structure at 1e4/s); sub-grid texture wavelengths: roller stipple 2 mm, brush marks 1 mm, "
          "spray orange peel 0.6 mm (estimates within the 0.5-5 mm levelling range); a lap mark is "
          "counted where the set film under fresh paint holds >= 4 um of solids (~10 % of a coat; a thinner stroke "
          "fringe doubled is not a visible lap -- estimate; the no-merge physics applies at any thickness)")
CELL = _p("CELL", dict(wall=2.0e-3, trim=2.0e-3, ceiling=2.0e-3, floor=4.0e-3, tray=2.0e-3, paper=1.0e-3, crate=1.0e-3,
                       stencil=1.0e-3, box=4.0e-3, table=4.0e-3, print=1.0e-3), "m", "derived",
          "2 mm resolves roller stipple and 1 mm brush marks and stencil edges; P2 levelling needs >= 8 cells per wavelength; "
          "the E7 and P3 refine runs compare the task metrics at 1 mm and 2 mm")
SUBSTRATE = _p("SUBSTRATE", dict(
    primed_drywall=dict(sorptivity=0.1e-6, capacity=5e-6), spackle=dict(sorptivity=1.0e-6, capacity=30e-6),
    watercolour_paper=dict(sorptivity=2.97e-6, capacity=200e-6), canvas_cloth=dict(sorptivity=5e-6, capacity=400e-6),
    sealed=dict(sorptivity=0.0, capacity=0.0), plywood_sealed=dict(sorptivity=0.05e-6, capacity=3e-6),
    cardboard=dict(sorptivity=2e-6, capacity=150e-6), pet_on_plywood=dict(sorptivity=0.0, capacity=0.0)),
    "m/s^0.5, m", "datasheet + estimate",
    "watercolour paper: Cobb60 23 g/m2 (300 g/m2 cotton [Reflex Aquarell Da Capo data sheet]) -> S = 23e-6 m / sqrt(60 s); capacity 0.51 mm x 40 % "
    "pores (estimate); PVA-primed drywall 0.1 um/s^0.5 and spackle x10 (estimates); canvas drop cloth, "
    "cardboard: estimates")

# ------------------------------------------------------------------ roller
ROLLER = _p("ROLLER", dict(
    cover_len=0.2286, core_od=0.0424, nap=0.0095, n_around=72, n_along=46, E_nap=40e3, beta=3.0, c_nap=2.0e3,
    mu_dry=0.6, mu_wet=0.4, v_reg=0.02, porosity=0.9, sat=0.154, express=2.0, uptake=1.5, split=0.5, bearing=0.002,
    cover_mass=0.060, frame_mass=0.34, grip_r=0.019, grip_len=0.150, wire_r=0.00397, wire_len=0.170,
    hard_stop=0.85), "m, -, Pa, -, Pa s/m, -, m/s, -, -, -, 1/s, -, N m, kg", "datasheet + derived + estimate",
    "9 in x 3/8 in cover on a 1.5 in core, OD 61 mm [Wooster cover listings]; nap modulus 40 kPa (20-80, estimate; no compression "
    "curve found), densification beta 3 (estimate), nap damping 2e3 Pa s/m (its hysteresis is the rolling resistance, ~5-10 % "
    "of the normal force at 0.3-0.5 m/s; estimate); "
    "nap-wall friction 0.6 dry / 0.4 wet (estimates); regularisation 20 mm/s (stability: c dt / m_eff < 1); pore "
    "fraction 0.9 (estimate); uptake saturation 0.154 of the pore volume (BASF, PCI Magazine 2018: ~40 g in a 4 in cover's 222 mL nap); "
    "nip expression 2.0 x nap strain of the held liquid (calibrated in E9 so a loaded cover at 10 N lays ~100 um on a "
    "first pass, the ProMar 200 TDS 4 mil); uptake rate 1.5 /s (E8); film split 0.5 at equal speeds [Walker and Fetsko 1955, Coyle et al. 1986]; bearing friction 2 mN m "
    "(estimate); cover 60 g dry (estimate); Sherlock R017-9 frame 0.34 kg, 38 mm grip, 5/16 in shank [Wooster frame listings]; the "
    "core collider sits at 85 % nap compression (densified nap: hard stop)")
TRAY = _p("TRAY", dict(size=(0.368, 0.279, 0.089), well_len=0.12, well_depth=0.08, ramp_drop=0.07, wall=0.0025, mass=0.25,
                       ramp_ribs=0.002), "m, kg", "retailer + estimate",
          "Wooster R404 deep-well tray 14.5 x 11 x 3.5 in, 2 qt [PaintSupply listing]; well/ramp split, rib height and 0.25 kg PP "
          "mass are estimates")

# ------------------------------------------------------------------ brushes -- PRBM tufts
# pseudo-rigid-body model of a cantilever (Howell): pivot at (1 - gamma) L, rigid link gamma L, torsion spring
# K = gamma K_theta E I / L with gamma 0.85, K_theta 2.65 (tip-loaded cantilever; buckling at K_theta E I / L^2,
# 7 % above Euler's pi^2/4).
PRBM = _p("PRBM", dict(gamma=0.85, k_theta=2.65), "-", "published", "Howell, Compliant Mechanisms (2001): pseudo-rigid-"
          "body cantilever, characteristic radius 0.85 and stiffness coefficient 2.65 (literature)")
BRUSHES = _p("BRUSHES", dict(
    sash=dict(width=0.0635, thick=0.0159, length=0.0746, bevel_deg=15.0, tufts=(24, 6), filaments=6642, E=2.9e9,
              d_root=0.305e-3, d_tip=0.203e-3, handle_len=0.18, handle_r=0.011, ferrule=(0.068, 0.020, 0.030),
              mass=0.10, kc=100.0, mu=0.35, metering=0.075, chisel_deg=4.0, cohesion=10.0),
    stencil=dict(width=0.0127, thick=0.0127, length=0.026, bevel_deg=0.0, tufts=(6, 6), filaments=2539, E=2.8e9,
                 d_root=0.20e-3, d_tip=0.13e-3, handle_len=0.14, handle_r=0.008, ferrule=(0.016, 0.016, 0.020),
                 mass=0.015, kc=100.0, mu=0.45, metering=0.012, round=True),
    round8=dict(width=0.0050, thick=0.0050, length=0.0225, bevel_deg=0.0, tufts=(5, 5), filaments=1453, E=2.9e9,
                d_root=0.10e-3, d_tip=0.04e-3, handle_len=0.20, handle_r=0.0045, ferrule=(0.007, 0.007, 0.025),
                mass=0.010, kc=20.0, mu=0.30, metering=0.60, round=True, pointed=True, cohesion=5.0, ribbon=5.0),
    longreach=dict(width=0.0508, thick=0.0140, length=0.060, bevel_deg=15.0, tufts=(20, 5), filaments=4680, E=2.9e9,
                   d_root=0.28e-3, d_tip=0.18e-3, handle_len=0.27, handle_r=0.011, ferrule=(0.054, 0.018, 0.025),
                   mass=0.12, kc=100.0, mu=0.35, metering=0.075, dogleg_deg=35.0, chisel_deg=4.0, cohesion=10.0)),
    "m, Pa, kg, N/m", "datasheet + derived + estimate",
    "Purdy XL Glide 2.5 in angled sash: 63.5 x 15.9 mm, 74.6 mm out [Purdy product page], Tynex 612 wet modulus 2.9 GPa [DuPont filament brochure], "
    "long-reach dog-leg brush for behind radiators (ProDec PDLB50 class: 2 in head set at 35 deg to a wooden handle, "
    "355 mm overall, 60 mm bristles; Hamilton radiator brush 0.12 kg [retailer listings]; thickness and filament count by the "
    "sash brush's packing, estimates), "
    "12/8 mil taper (DuPont standard tapers), filament count from packing; hog stencil brush 0.5 in, 25-28 mm "
    "hair [Gordon Brush and fauxbrushes listings], hog 2.8 GPa wet [DuPont filament brochure]; round no. 8: 5 mm x 22.5 mm hair (e-David, Guelzow 2024); tip contact stiffness kc per "
    "tuft and wet bristle friction 0.35 (estimates); metering constant C of the law h_m = L_c sqrt(C eta U / w) (the "
    "brush's mean line load w, metered over each touching tip's own sliding with the brush): calibrated on the E7 "
    "brush-out to the 80-115 um brushed film build [Rhoplex SG-10M data sheet] (sash, with the chisel and wet-bristle cohesion; the "
    "long-reach brush borrows it) and on the E6 round-brush strokes: fluid acrylic laid "
    "~40-60 um wet by a loaded round (estimate; round); "
    "round brushes: wet-hair cohesion couples neighbouring tufts' bending (stiffness 5 x the tuft's own: tufts move as "
    "one bundle at a light touch, e-David's narrow light strokes, Guelzow 2024); its hairs converge to a point at the tip; tufts "
    "touch at the deepest of three points along their length (the belly lies down under a press), cannot share space "
    "on the surface (overlap penalty) and a pressed tuft flattens into a ribbon b (1 + 5 sin tilt). Cohesion and ribbon "
    "are calibrated on E6 to a #8's spread: 2 mm lines at a 1-2 mm press growing to 6 mm at 6 mm (a #8, 5 mm at the "
    "ferrule, spreads to ~6-8 mm pressed; e-David's linear width-depth law, Guelzow 2024). Above ~8 mm the bundle splays fully (9 mm at a 9 mm press)")
BRUSH_PAINT = _p("BRUSH_PAINT", dict(k_up=2.0, hold=0.75, residual=0.25, k_drip=0.5, c_tip=0.05, D_bundle=1.0e-4,
                                     h_max=150e-6),
                 "1/s, -, -, 1/s, N s/m, m2/s", "published + calibrated + estimate",
                 "paint in the bristle bundle: the full free length's pore volume (packing) wetted up to the dip "
                 "depth at k_up (Washburn in ~40 um gaps: seconds; estimate); drains by drops (k_drip, estimate) to "
                 "'hold' of the wetted pore volume after withdrawal (ASTM D5068 hang); a residual saturation of "
                 "0.25 of the wetted pores never leaves (capillary hold-up: 1 - 0.25/0.75 = 67 % of a drained load is "
                 "released at any dip depth, 65-70 % measured after a D5068 half dip [eGate filament test]); the bundle is one porous body: liquid redistributes between tufts toward "
                 "the ones being wiped with an effective diffusivity D_bundle (calibrated in E7 so a brush delivers "
                 "along its stroke; capillary Darcy flow alone, k ~ 2e-10 m2 at eta ~ 1 Pa s, is ~400x slower -- the "
                 "bending squeeze of a pressed bundle is the mechanism, not resolved per bristle); tip damping c_tip "
                 "(wet bristles; estimate, ~0.2 of critical); the metered wet film is capped at h_max = 150 um (the "
                 "80-115 um brush-out range [Rhoplex SG-10M data sheet] with margin; the law's light-load limit, estimate)")
PAIL = _p("PAIL", dict(r_top=0.089, r_bottom=0.080, height=0.140, wall=0.0015, mass=0.14), "m, kg", "retailer + estimate",
          "Bercom Handy Paint Pail 1 qt: 7 in x 5.5 in tall [Bercom listing] (taper estimate)")
CAN = _p("CAN", dict(r=0.0838, height=0.190, wall=0.0004, mass=0.35, fill=2.5e-3), "m, kg, m3", "estimate",
         "US 1-gal steel paint can 6-5/8 in x 7.5 in (common size, estimate); 2.5 L left in it")

# ------------------------------------------------------------------ aerosol and stencil
AEROSOL = _p("AEROSOL", dict(
    r=0.0330, height=0.196, can_mass=0.12, fill_mass=0.34, button_r=0.0075, button_h=0.012, stroke=0.0022,
    preload=10.0, k=7.5e3, x_open=(1.0e-3, 1.8e-3), mdot=2.0e-3, U0=25.0, half_angle_deg=12.0, d50=25e-6, sigma_g=1.7,
    parcels_per_s=16000, jet_K=7.0, jet_spread=0.113, jet_kappa=0.0075, stagnation=0.15, solvent_flash=0.6,
    nozzle_h=0.006, mist_age=3.0, kernel_r=1.5e-3, drift=0.25),
    "m, kg, N, N/m, kg/s, m/s, deg, m, -", "search + published + calibrated + estimate",
    "12 oz can 66 x 196 mm (estimate); valve stroke 1.5-2.5 mm and finger force 15-30 N, spring 8-25 N [aerosol valve guides] -> preload "
    "10 N + 7.5 N/mm (full open 23.5 N at 1.8 mm); discharge 2.0 g/s: the high-output 'broad-stream' class "
    "(15-35 g / 10 s, Kwasny US 5957341; conventional valves ~10 g / 10 s) that a '2X' can [Rust-Oleum data sheet] is (estimate "
    "inside the range; a 12 oz can empties in ~3 min); launch 25 m/s, 12 deg half cone (estimates); droplet median "
    "25 um [Goehler et al. 2014], sigma_g 1.7 (estimate); self-similar round jet of the spray's momentum flux J: U_c = 7 sqrt(J/rho_a)/x "
    "(B = 6.2 [Pope]), Gaussian e-folding radius 0.113 x (velocity half-width 0.094 x, Hussein et al. via Pope 5.1; "
    "with K = 7 it carries exactly J), the radial velocity from continuity (entrainment); the droplets' turbulent "
    "dispersion an isotropic random walk with D = kappa U_c x, kappa calibrated so the "
    "droplet flux spreads like the jet's passive scalar, half-width 0.11 x [Pope; Panchapakesan & Lumley 1993]; "
    "stagnation layer 0.15 H (estimate); overspray mist settles as dry dust to the first surface below with a lateral drift of sigma 0.25 x the fall height (room air; estimate); parcels still airborne after 3 s, or slow outside the plume, are overspray "
    "mist; 60 % of the liquid is propellant/solvent that flashes in flight (estimate, solids 0.25 after flash); a "
    "16,000 parcels/s share the emitted mass and land through a Gaussian reconstruction with radius 1.5-6 mm "
    "(numerical sampling estimate, not droplet size). Only locally contacting ideal-seal stencil cells block support; "
    "open sheet gaps allow under-edge coating. Fine fringe and can coverage remain calibration limitations.")
STENCIL = _p("STENCIL", dict(size=(0.40, 0.30), thick=0.0003, rho=1390.0, mu=0.35, seal_tolerance=20e-6), "m, kg/m3", "search + estimate",
             "12 mil (0.30 mm) clear plastic stencil [Cutting Edge Stencils product pages] cut from a 16 x 12 in blank (blanks run 8.5 x 11 to 12 x 18 in; "
             "a 30 x 20 cm sheet would leave 1 cm above and below the 18 cm motif, inside the spray's halo), PET "
             "density 1,390 kg/m3; rigid plate on a flat sealed panel. Ideal smooth-contact seal only within 20 um local "
             "clearance (authored numerical tolerance, not measured roughness/adhesion); resolved open gaps do not seal")

# ------------------------------------------------------------------ room
ROOM = _p("ROOM", dict(x=3.66, y=3.35, z=2.438, wall_t=0.10), "m", "authored",
          "US 11 x 12 ft bedroom, 8 ft ceiling")
WINDOW = _p("WINDOW", dict(x=0.55, w=0.914, sill=0.813, head=2.032, casing=0.057, casing_t=0.0175, stool_d=0.075,
                           stool_t=0.022, apron=0.083), "m", "search + authored",
            "36 x 48 in opening, 2-1/4 in colonial casing 11/16 in proud [trim size guides]; stool and apron dimensions authored")
BASEBOARD = _p("BASEBOARD", dict(h=0.083, t=0.0127), "m", "search", "3-1/4 in x 1/2 in [trim size guides]")
OUTLET = _p("OUTLET", dict(x=-0.90, z=0.356, w=0.070, h=0.114, t=0.005), "m", "search",
            "duplex plate 2.75 x 4.5 in, 14 in to centre [outlet height guides]")
RADIATOR = _p("RADIATOR", dict(len=0.64, depth=0.229, height=0.635, gap=0.064, sections=10, pipe_r=0.0167, pipe_x=0.22),
              "m", "authored", "4-column cast-iron radiator, 25 in tall, back 2.5 in off the wall, 10 sections at the "
              "2.5 in (63.5 mm) pitch of 4-column cast iron (typical, authored)")
TABLE = _p("TABLE", dict(size=(1.22, 0.61), height=0.74, top_t=0.025, mass=10.0, leg_r=0.0127), "m, kg", "authored",
           "Lifetime-class 4 ft folding table (estimate)")

# ------------------------------------------------------------------ evaluation
EVAL = _p("EVAL", dict(hide_dft=33e-6, hide_dE=1.0, hide_frac=0.98, run_h=250e-6, sag_move=1e-3, lap_area=1e-4, dE=1.0,
                       protected_area=0.5e-4, settle_max_s=1800.0, settle_fast=True, stencil_hide_dft=15e-6, stencil_frac=0.90,
                       bleed=0.5e-3, overrun=1.0e-3, spray_dft=(25e-6, 50e-6), gauge=5e-3, stroke_dft=5e-6),
          "m, -, m, m, m2, -, m2, s", "derived + authored",
          "hide: the dry stack within dE*ab 1.0 of the paint's own infinitely thick colour over the old colour under it "
          "(authored perceptual criterion; optical coefficients are estimates with the coating-class sensitivity "
          "range of LATEX_OPTICAL_RANGE, not a product-backed hiding-power measurement; the "
          "DFT is reported); runs: wet film > 250 um (multinotch sag range, ASTM D4400) before the settle (progress) "
          "and a front that moves > 1 mm during the settle (verify; behind-radiator: more than 50 um of dry paint carried into a cell by "
          "the film's flow, task.BEHIND_RADIATOR run_flow); laps <= 1 cm2; protected faces <= 0.5 cm2 (sink surfaces: "
          "4 mm voxels holding >= 1 um); wet target film touched by a robot link or a tool's non-applicator part <= "
          "0.5 cm2 (untouched); settle until set or 30 min of film time (settle_fast: charts whose solids moved < 0.1 nm "
          "per 2 s step for three steps are only dried from then on, P17); aerosol coat 25-50 um DFT [Rust-Oleum data sheet] read as a DFT gauge does: the mean over a 5 mm "
          "probe footprint (SSPC-PA 2 spot readings; probe size estimate); paper: a filled circle / swatch counts "
          "cells at >= 15 um dry, the S-stroke (a shape task) cells at >= 5 um dry (visibly painted; estimate)")


def table():
    rows = ["| Parameter | Value | Unit | Label | Basis |", "|---|---|---|---|---|"]
    for e in LEDGER:
        v = e["value"]
        if isinstance(v, float):
            v = f"{v:g}"
        rows.append(f"| `{e['name']}` | {v} | {e['unit']} | {e['label']} | {e['basis']} |")
    return "\n".join(rows)


def evaporation_rate(paint, climate=None):
    """Constant-rate water (vehicle) loss, m/s of liquid: j = k_m c_sat(T) (1 - RH) / rho_w; c_sat from
    the Magnus saturation pressure and the ideal gas law."""
    c = climate or CLIMATE
    T = c["T_C"]
    p_sat = 610.94*math.exp(17.625*T/(T + 243.04))
    c_sat = p_sat*0.018015/(8.314*(T + 273.15))
    # solvent-borne paints (aerosol enamel, lacquer): the room's water vapour does not slow their solvent's evaporation
    # (RH 30-70 % would change the spray coats' drying 2.3x); the calibration point stays at 50 %
    rh = 0.5 if paint.get("solvent") else c["RH"]
    return paint["k_m"]*c_sat*(1.0 - rh)/WATER_RHO


if __name__ == "__main__":
    print(table())
