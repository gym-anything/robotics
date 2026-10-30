"""Parameter ledger of the percussion environment: every physical constant with its provenance.

Labels: datasheet | standard | literature-read | search | literature | measured (script or probe named) | derived |
estimate | authored | calibrated (fitted to a stated physical observable).
Probes and demos READ these values; they never override them on a command line. `ledger()` returns the
machine-readable form.

World frame: x to the robot's right, y from the robot towards the client, z up; floor z = 0. SI units throughout.
"""

import math

GRAVITY = 9.80665                       # m/s^2 (literature)
RHO_AIR = 1.204                         # kg/m^3, 20 C (literature)
C_AIR = 343.2                           # m/s, 20 C (literature)
P_REF = 2e-5                            # Pa, reference sound pressure (standard, IEC 61672)

# ------------------------------------------------------------------ clocks
AUDIO_FS = 48000                        # Hz: every instrument mode, every vibro-contact solve, every microphone
RIGID_DT_S = 0.001                      # s, world tick = FR3/Panda FCI rate (1 kHz)
SAMPLES_PER_TICK = 48                   # AUDIO_FS * RIGID_DT_S
WINDOW_GUARD_M = 0.002                  # impact window opens when a striker-surface gap falls below this + |v_n| 1.5 ms
WINDOW_HOLD_S = 0.001                   # ... and closes after every pair has separated for this long
# (no resting-contact mode: stepping MuJoCo every 2-8 samples under resting contacts halves the cost of a pressed G1
# hand, but its substep-held coupling keeps the head humming at 46-57 dB SPL where the per-sample solve decays to
# silence)

# ------------------------------------------------------------------ xylophone bars (Rucz et al. 2021, literature-read)
BAR_RHO = 1116.0                        # kg/m^3, rosewood (Rucz Sec. III A, mean of 8 samples)
BAR_E_L = 24.0e9                        # Pa (Rucz)
BAR_G_RL = 0.147*BAR_E_L                # shear, vertical bending plane (Rucz)
BAR_G_LT = 0.097*BAR_E_L                # shear, horizontal plane / torsion (reproduces the torsion family)
BAR_E_T = 0.097*BAR_E_L                 # transverse modulus used for contact stiffness (derived)
BAR_WIDTH_M = 0.031                     # Rucz
BAR_THICK_M = 0.016                     # Rucz
BAR_STONE_R_M = 0.0625                  # m, 125 mm grindstone of the undercut (Rucz)
BAR_ETA_S = 9.4e-7                      # s, viscous loss coefficient zeta = eta*omega/2 (Rucz Table V; a literature
                                        # value Rucz took from his refs. 3, 4, 8, used with his direct acoustic coupling)
# A frequency-independent floor under it. Measured rosewood bars in situ (Yamaha YM-5100A marimba, C2-C7,
# resonators off: material, support and radiation together) have a constant loss factor eta = 2/(omega tau) =
# 0.0036-0.0068, mean 0.0045 +- 0.0009, tau ~ 1/f for fundamentals and higher harmonics (Hill 2011, [S56]); Rucz's
# viscous law alone gives 0.0016 at C4 (T60 5.3 s). Every bar mode gets zeta = max(BAR_LOSS_FACTOR/2, eta omega/2):
# the measured floor governs below ~760 Hz, Rucz's rising law above (where the two sources disagree, the larger)
BAR_LOSS_FACTOR = 0.0045
BAR_SHEAR_K = 5/6
# measured partials f1, f2, f3 (Hz) of the Sonor Orff soprano xylophone bars (Rucz Table II) and the per-bar
# orthotropic E0/E scale Rucz used (Table II); the 13 diatonic bars are played, the three chromatic ones are spares
BAR_PARTIALS = {"C4": (264, 1048, 2656), "D4": (296, 1176, 2788), "E4": (324, 1320, 3196), "F4": (349, 1400, 3376),
                "F#4": (369, 1480, 3376), "G4": (392, 1570, 3530), "A4": (440, 1762, 3876),
                "Bb4": (464, 1864, 4104), "B4": (496, 1976, 4216), "C5": (520, 2096, 4304), "D5": (584, 2300, 4368),
                "E5": (659, 2523, 4733), "F5": (705, 2755, 5176), "F#5": (742, 2816, 4984), "G5": (784, 2984, 5360),
                "A5": (880, 3168, 5308)}
BAR_E_SCALE = {"C4": 0.715, "D4": 0.723, "E4": 1.050, "F4": 0.997, "F#4": 1.048, "G4": 1.052, "A4": 1.120,
               "Bb4": 1.208, "B4": 1.220, "C5": 1.047, "D5": 1.002, "E5": 1.076, "F5": 1.032, "F#5": 0.997,
               "G5": 0.940, "A5": 0.884}
BAR_SEMITONE = {"C4": 0, "D4": 2, "E4": 4, "F4": 5, "F#4": 6, "G4": 7, "A4": 9, "Bb4": 10, "B4": 11, "C5": 12,
                "D5": 14, "E5": 16, "F5": 17, "F#5": 18, "G5": 19, "A5": 21}
XYLO_NOTES = ("C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5", "G5", "A5")


def bar_length(note):
    """Rucz: lengths run from 325 mm (C4) to 228 mm (A5); intermediate lengths geometric in pitch (estimate)."""
    return 0.325*(0.228/0.325)**(BAR_SEMITONE[note]/21)


BAR_MODE_FMAX_HZ = 20000.0              # every bending/torsion mode below 20 kHz (Rucz: "a few tens of modes")
BAR_PITCH_M = 0.037                     # centre-to-centre spacing of the bars: 31 mm + 6 mm gap (authored, Orff
                                        # instruments leave a finger's width; Studio 49 AX 1000 class)
BAR_HOLE_R_M = 0.0032                   # node holes for the pins (estimate: 6.4 mm drill for a 4 mm pin with sleeve)
PIN_R_M = 0.0022                        # pin + rubber sleeve radius (estimate)
FELT_THICK_M = 0.006                    # felt support strip (estimate; "the sound bars lay on a felt support", Rucz)
FELT_STRIP_W_M = 0.012

# resonators of the Sonor instrument (Rucz Table III: V cm^3, S cm^2, l_eff mm, measured bare Hz) and the measured
# detuning by the covering bars (Table IV, all bars in place, %); bars per resonator 2-3-2-2-2-2 (Rucz Sec. III)
RESONATORS = ((1830.4, 33.56, 55.8, 305.5, 6.1), (1560.0, 49.91, 66.1, 386.5, 14.9), (627.8, 34.72, 54.8, 559.0, 9.2),
              (531.0, 37.62, 55.9, 641.5, 12.8), (415.5, 40.52, 56.6, 740.5, 13.8), (259.9, 46.32, 55.9, 1007.5, 15.5))
RESONATOR_BARS = (("C4", "D4"), ("E4", "F4", "G4"), ("A4", "B4"), ("C5", "D5"), ("E5", "F5"), ("G5", "A5"))
# radiation damping of a Helmholtz opening in a baffle (the box top): zeta = k S / (4 pi l_eff) (literature: flanged
# piston radiation resistance rho c k^2 S^2/(2 pi) over 2 omega rho l_eff S), scaled by the factor that maps the formula
# onto Rucz's FE value for their revised F4 resonator (25 mm circular opening, 3 mm neck: formula 1.06e-2, Rucz Table V
# 8.8e-3) - calibrated, one factor for every resonator
RESONATOR_RAD_FACTOR = 8.8e-3/1.06e-2
RESONATOR_WALL_ZETA = 1.0e-3            # wall / viscous losses (estimate; Rucz: plastic walls vibrate, not measured)
RESONATOR_COUPLING_ETA = 0.5            # fraction of a bar's area over the opening whose volume displacement pumps the
                                        # neck (estimate: air escapes sideways through the 6 mm gaps)
RESONATOR_TOP_GAP_M = 0.012             # bar bottom to resonator top plate (felt + rail: estimate)

XYLO_BOX = dict(length=0.61, width_low=0.20, width_high=0.14, height=0.09, mass=4.4)   # length, height, mass: Studio
# 49 AX 1000 datasheet [S26]. The instrument is a hybrid: Rucz's measured Sonor bars (31 x 16 mm, 325 -> 228 mm) on
# a frame with the AX 1000's length, height and mass; the AX 1000 datasheet's widths (0.34 -> 0.23 m, for its 32 x 18
# mm bars) are NOT used - the frame must carry the measured bars at their nodes.
# box widths (estimate): the bars rest on felt over the box's top edges at their nodes (+-0.278 to +-0.296 L for the
# undercut bars: 0.18 m apart for C4, 0.13 m for A5) and overhang the box by 45-62 mm at both ends, as on Orff instruments - the overhang is where a
# hand (and the Franka Hand, fingers above and below) lifts a bar off its pins; the 6 mm gaps between bars admit no
# finger
                                        # (datasheet [S26]); the resonator boxes are inside this frame
# resonator bars (Studio 49 KB/AX alto: 32 x 18 mm rosewood bar over "individually tuned single resonators of
# maple" [S27]); modelled as the Rucz bar profile for the same note (31 x 16 mm) on its own box. Box outer size from the
# dealer's datasheet of the playing-ready instrument (length 32 cm for c1 down to 27 cm for h1, width 5 cm, height
# 11 cm, 0.40 kg [S27b]); lengths between linear in semitones (estimate); wall thickness derived from the listed mass
# (0.40 kg minus the bar, maple 650 kg/m^3 over the box's surface: 3.4-4.0 mm); the opening is a slot under the bar's
# centre, 25 mm wide (fits under the 31-32 mm bar: authored), whose length tunes the box to the bar's f1 with its
# physical neck: l_eff = wall + 2 delta, delta the end correction of a uniform-velocity rectangular piston of the slot's
# size in a baffle on each side (modal.slot_end_correction; an equal-area circle's 0.85 a is 22 % too large for the G4
# slot and forces a 55 %-long slot). The G4 slot is still 42 % of its box, which is 0.33 wavelengths long: past the
# lumped Helmholtz model's range.
RESONATOR_BAR_NOTES = ("C4", "G4")
RESONATOR_BAR_DATASHEET = dict(length_c1=0.32, length_h1=0.27, width=0.05, height=0.11, mass=0.40)   # [S27b]
RESONATOR_BAR_SLOT_W_M = 0.025
MAPLE_RHO = 650.0                       # kg/m^3 (Wood Handbook: hard maple 705, soft maple 545 at 12 % MC [S33])

# ------------------------------------------------------------------ drum (Remo Ambassador Coated 14", Wagner 2006)
DRUM_RADIUS_M = 0.1778                  # 14 in head (datasheet)
DRUM_FILM_M = 0.254e-3                  # 10 mil Mylar (Remo datasheet [S30])
DRUM_FILM_RHO = 1390.0                  # kg/m^3 (Mylar datasheets [S32])
DRUM_COATING = 0.15                     # coating mass fraction (estimate; the data cannot resolve 0-30 %)
DRUM_SIGMA = DRUM_FILM_RHO*DRUM_FILM_M*(1 + DRUM_COATING)       # 0.406 kg/m^2
DRUM_E_FILM = 4.0e9                     # Pa (Mylar 3.5-4.9 GPa datasheets): bending stiffness of the film
DRUM_NU_FILM = 0.38                     # estimate
DRUM_TENSIONS = {"low": 3080.0, "mid": 4630.0, "high": 5960.0}  # N/m: fitted to Wagner's measured (1,1) (variant C
                                        # + 15 % coating); (0,1) and (2,1) are then predicted within 2 %
DRUM_SHELL_DEPTH_M = 0.127              # 5 in shell, open bottom (Wagner's Ludwig 14 x 5, snares + snare head removed)
DRUM_MODE_FMAX_HZ = 5000.0              # membrane modes kept (residual compliance of the rest added in series)
DRUM_ZETA_INTERNAL = 1.0e-3             # internal + edge losses of the coated head (estimate:
                                        # a tensioned membrane stores energy in its pre-tension, so the film's material
                                        # loss tan d ~ 0.01 enters only at second order; T60 of a 2 kHz mode 0.55 s)
DRUM_MASS_KG = 2.9                      # 14 x 5 in steel-hoop wood shell drum with lugs (estimate: catalogue 2.5-3.5 kg)
STAND_MASS_KG = 3.2                     # snare stand, double-braced (estimate: catalogue 2.8-4 kg)
DRUM_HOOP_H_M = 0.008                   # hoop top above the head plane (estimate: triple-flanged hoop)
DRUM_HOOP_R_M = (0.1795, 0.1855)        # hoop inner/outer radius (estimate)

# ------------------------------------------------------------------ drumsticks (Vic Firth American Classic 5A)
STICK = dict(L=0.4064, D=0.0144, dn=0.0070, dt=0.0090, taper=0.120, tip=0.014, mass=0.050)   # datasheet [S29]; taper
                                        # and tip shape estimates
STICK_E = 14.9e9*1.10                   # Pa, shagbark hickory MOE 12 % MC +10 % shear correction (Wood Handbook [S33])
STICK_G_RATIO = 0.07                    # G/E estimate (Wood Handbook ranges 0.05-0.09)
STICK_FULCRUM_FRAC = 1/3                # matched-grip fulcrum ~1/3 from the butt (Wagner; Dahl)
STICK_MODES = 3                         # free-free bending modes kept (5A model 390/933/1631 Hz)
STICK_MODE_ZETA = 0.0275                # in-hand ringing ~0.1 s at 400 Hz (Wagner App. A) -> zeta (estimate)

# mallets (rubber-head Orff xylophone mallets, Studio 49 class) and felt beater; dimensions are estimates
MALLET = dict(L=0.28, handle_r=0.0045, head_r=0.0125, head_mass=0.010, handle_rho=700.0)
BEATER = dict(L=0.30, handle_r=0.005, head_r=0.020, head_mass=0.012, handle_rho=700.0)

# ------------------------------------------------------------------ contact laws
# F = K delta^alpha (1 + lambda d(delta)/dt) for delta > 0 (Hertz / Hunt-Crossley). The fourth entry is unused:
# the patch over which the struck structure's mode shapes are averaged is computed per pair from the contact's
# physical size (contact.patch_radius); kept so old tables stay readable.
PAD_TAN_DELTA = 0.15                    # small-amplitude loss factor of the fingertip rubbers (Franka pad, Dex3 skin):
                                        # natural rubber 0.05-0.15, silicone 0.10-0.30 at 10-100 Hz (rubber damping
                                        # reference tables [S45]); estimate within those ranges
PAD_MUTE_REF = dict(F_n=3.0, f_hz=300.0)   # the muting regime the pad damping is tied to: a 3 N touch on a 260-400 Hz
                                        # bar. Linearised Hunt-Crossley about the preload d0: k = alpha K d0^(alpha-1),
                                        # c = lambda F0, loss factor eta = omega c / k = omega lambda d0 / alpha, so
                                        # lambda = alpha tan(delta) / (omega d0) - see PAD_LAMBDA below
_K_PAD = 7.5e5                          # rubber pad Hertz constant: E 3 MPa, R 20 mm (estimate, labelled in the ledger)
_d0_pad = (PAD_MUTE_REF["F_n"]/_K_PAD)**(1/1.5)
PAD_LAMBDA = 1.5*PAD_TAN_DELTA/(2*math.pi*PAD_MUTE_REF["f_hz"]*_d0_pad)   # 0.47 s/m. Cross-check: Hunt-Crossley's
                                        # impact form lambda = 3(1 - e)/(2 v) with restitution e 0.5 at 1 m/s gives 0.75
                                        # s/m - the two anchors agree within 1.6x
CONTACT = {
    # striker material -> surface material -> (K, alpha, lambda s/m, unused)
    "rubber_mallet": {"rosewood": (5.0e7, 1.5, 0.0, 0.003), "mylar": (5.0e7, 1.5, 0.0, 0.005)},   # Rucz Table V
    "hickory_tip": {"mylar": (8.9e7, 1.5, 0.0, 0.004), "rosewood": (6.9e7, 1.5, 0.0, 0.003)},     # derived
    "hickory_shaft": {"mylar": (1.1e8, 1.5, 0.0, 0.005), "rosewood": (8.5e7, 1.5, 0.0, 0.004)},   # derived (R 7 mm)
    "felt": {"mylar": (5.6e7, 2.5, 0.3, 0.010), "rosewood": (5.6e7, 2.5, 0.3, 0.008)},             # estimate [S25]
    "rubber_pad": {"mylar": (_K_PAD, 1.5, PAD_LAMBDA, 0.007), "rosewood": (_K_PAD, 1.5, PAD_LAMBDA, 0.005)},
    "plastic": {"mylar": (1.5e8, 1.5, 0.0, 0.004), "rosewood": (1.5e8, 1.5, 0.0, 0.003)},          # Franka hand
                                        # housing, finger bodies, link 7 (PA/aluminium: stiff, estimate)
}
# contact patch (contact.patch_radius): the radius over which the struck structure's mode shapes are averaged and the
# cut-off of the membrane's residual (log-singular) point compliance is the physical size of the loaded area at the
# striker's reference force: on the head, where the film wraps a sphere of radius R under tension T (slope match:
# a_w = sqrt(F R / (2 pi T))), smoothed by the film's bending length l_b = sqrt(D/T) (D = E h^3/12(1-nu^2), 1.1-1.4
# mm); the striker's own indentation radius sqrt(R delta(F)) where it is softer (felt, rubber pads). Bars: smooth beam
# shapes, the Hertz radius (1-3 mm) - insensitive. F_ref per striker material: its mf reference stroke's peak force
# (bench strokes: stick 20-30 N, mallet ~90 N on a bar, ~25 N on the head); a resting or pressing contact uses the same
# patch (the patch enters the compliance only logarithmically: x2 in F gives x1.4 in a and ~8 % in compliance).
PATCH_REF_FORCE_N = {"hickory_tip": 25.0, "hickory_shaft": 25.0, "rubber_mallet": 25.0, "felt": 25.0,
                     "rubber_pad": 5.0, "plastic": 10.0}
# tangential friction of the vibro-contacts (world.py; regularised Coulomb on the solved normal force)
FRICTION = {
    # striker material -> surface material -> mu (kinetic)
    "hickory_tip": {"mylar": 0.40, "rosewood": 0.35},     # lacquered hickory on the coated PET head: PET film dynamic
    "hickory_shaft": {"mylar": 0.40, "rosewood": 0.35},   # COF 0.40 (DuPont Teijin Mylar datasheet, ASTM D1894 [S46b]);
                                                          # wood on wood 0.25-0.5 (handbook range) - estimates
    "rubber_mallet": {"mylar": 0.70, "rosewood": 0.70},   # rubber on film / finished wood (estimate, 0.5-1.0 range)
    "felt": {"mylar": 0.30, "rosewood": 0.30},            # wool felt (estimate)
    "rubber_pad": {"mylar": 0.80, "rosewood": 0.80},      # = PAD_MU (repository-qualified pad rubber value)
    "plastic": {"mylar": 0.30, "rosewood": 0.30},         # PA / aluminium on film or wood (estimate)
}
# static friction: a sticking contact is held by a tangential spring-damper to an anchor on the surface.
# Its stiffness is Mindlin's no-slip tangential stiffness of an elastic contact, k_t/k_n = 2(1-nu)/(2-nu) (Mindlin
# 1949; Johnson, Contact Mechanics, 7.2): 0.67 for rubber (nu 0.5) to 0.82 for wood or plastic (nu 0.3) - 0.7 here;
# k_n = dF/d(delta) of the pair's normal law at the current force. Critically damped, and capped at
# TANGENTIAL_K_DT2 / (dt^2 m_t^-1) so the explicit per-sample force stays stable with up to ~40 contacts on one body.
TANGENTIAL_STIFFNESS_RATIO = 0.7
TANGENTIAL_K_DT2 = 0.1
# rigid impacts on the instruments' non-vibrating parts (hoop, lugs, shell, stand, xylophone frame, resonator boxes):
# a short transient radiated as a point source (world._rigid_clicks). Radiated energy = RIGID_CLICK_ETA x the impact's
# normal kinetic energy 1/2 m v_n^2 (m: the lighter body's mass, robot links capped at 1 kg) - estimate (radiation
# fractions of impacts on small metal / wooden parts 1e-4 to 1e-2); spectra are authored damped partials.
RIGID_CLICK_ETA = 1.0e-3
RIGID_CLICK_MIN_SPEED_M_S = 0.05
RIGID_CLICK_SPECTRA = {"metal": ((1150.0, 0.030, 1.0), (2650.0, 0.022, 0.8), (4300.0, 0.015, 0.6)),   # (Hz, T60 s, amp)
                       "wood": ((720.0, 0.012, 1.0), (1650.0, 0.008, 0.7), (3100.0, 0.005, 0.4))}
# floor: carpet tiles - rolling resistance as MuJoCo rolling friction (torque = b N). Wheelchair coasting tests:
# carpet adds C_RR 0.0118 over linoleum for 0.305 m wheels (b ~ 3.6 mm; Hoffman et al. 2003 [S47]); loop pile 5.4 mm
# (Sauret et al. 2012 [S48]). A light hardwood stick compresses the pile far less than a loaded wheel: b = 1 mm
# (estimate within that order) stops a stick rolling at 0.67 m/s within ~0.25 m.
FLOOR_ROLLING_M = 1.0e-3
FLOOR_TORSIONAL_M = 5.0e-3

# ------------------------------------------------------------------ room and furniture
ROOM = dict(x0=-3.0, x1=3.0, y0=-1.6, y1=3.4, h=2.75)        # authored: a 6 x 5 m therapy room
WALL_THICK_M = 0.12                     # collider thickness of the room's walls (outside the visible wall planes)
# room furniture placed by dressing.py (pinned CC0 models) with a box collider each in scene.py, so a dropped tool, a
# thrown stick or a fallen robot meets the furniture and the walls instead of passing through them. bounds: the
# model's axis-aligned bounding box (model frame, before scale; measured from the pinned USD files with pxr) -
# collider = that box (a simplification: chairs and shelves are open underneath).
FURNITURE = (
    dict(name="client_chair", asset="SchoolChair_01", xy=(0.0, 1.90), yaw=math.pi, scale=1.0,
         bounds=((-0.283, -0.338, 0.0), (0.283, 0.338, 1.004))),
    dict(name="armchair", asset="modern_arm_chair_01", xy=(2.2, 2.1), yaw=math.radians(-130), scale=1.0,
         bounds=((-0.407, -0.411, 0.0), (0.414, 0.575, 1.023))),
    dict(name="shelves", asset="steel_frame_shelves_03", xy=(-2.75, 1.0), yaw=-math.pi/2, scale=1.0,
         bounds=((-1.044, -0.363, 0.0), (1.307, 0.358, 2.312))),
    dict(name="basket", asset="wicker_basket_02", xy=(-2.65, 2.0), yaw=0.0, scale=1.0,
         bounds=((-0.108, -0.109, 0.0), (0.267, 0.161, 0.217))),
    dict(name="plant", asset="potted_plant_02", xy=(2.55, 3.0), yaw=0.0, scale=1.0,
         bounds=((-0.397, -0.237, 0.0), (0.304, 0.42, 0.841))),
    dict(name="sofa", asset="sofa_02", xy=(-1.55, 2.90), yaw=math.pi, scale=1.0,
         bounds=((-0.903, -0.34, 0.0), (0.904, 0.478, 0.71))),
    dict(name="ottoman", asset="Ottoman_01", xy=(-0.55, 2.55), yaw=0.3, scale=0.8,
         bounds=((-0.44, -0.311, 0.0), (0.444, 0.31, 0.624))),
    dict(name="cabinet", asset="modern_wooden_cabinet", xy=(1.75, -1.32), yaw=0.0, scale=0.75,
         bounds=((-1.22, -0.25, 0.0), (1.22, 0.27, 0.68))),
    dict(name="crate", asset="plastic_crate_01", xy=(-2.55, 2.75), yaw=0.2, scale=1.0,
         bounds=((-0.149, -0.204, 0.0), (0.149, 0.204, 0.264))),
)
PIANO = dict(xy=(-2.35, -1.05), body_half=(0.49, 0.15, 0.05), body_z=0.78)   # authored digital piano on an X-stand
# octave-band absorption coefficients 125..4000 Hz from standard tables: painted 1/2 in gypsum board on studs (exactly
# the Owens Corning Noise Control Design Guide table row [S49]), carpet tiles on concrete, mineral-fibre acoustic
# ceiling tile, glass window (table values; the room's construction is authored)
ABSORPTION = dict(walls=(0.10, 0.08, 0.05, 0.03, 0.03, 0.03), floor=(0.02, 0.06, 0.14, 0.37, 0.60, 0.65),
                  ceiling=(0.30, 0.45, 0.65, 0.75, 0.80, 0.80), window=(0.35, 0.25, 0.18, 0.12, 0.07, 0.04))
OCTAVES_HZ = (125, 250, 500, 1000, 2000, 4000)
TABLE = dict(cx=0.0, cy=0.67, half=(0.60, 0.30), top_z=0.72, thick=0.025)   # authored adult activity table 1.2x0.6 m
CART = dict(cy=-0.20, half=(0.45, 0.28), top_z=0.80)                        # (legacy name) station footprint centre
# the robot station: a parked mobile dual-arm platform (castors braked) - a 0.66 x 0.48 m base, a column and a
# shoulder beam carrying both FR3 bases at 0.80 m, 0.50 m apart, with the sensor mast at its back (authored; dual-arm
# mobile manipulators of this class: e.g. the repository's mobile-fr3-duo model, classroom_arrangement)
STATION = dict(base_c=(0.0, -0.19, 0.21), base_half=(0.33, 0.24, 0.15), column_c=(0.0, -0.24, 0.57),
               column_half=(0.10, 0.09, 0.21), beam_c=(0.0, -0.15, 0.78), beam_half=(0.36, 0.075, 0.02),
               mast_xy=(0.0, -0.38))
ARM_BASE_XY = {"left_arm_mount": (-0.25, -0.15), "right_arm_mount": (0.25, -0.15)}
ARM_BASE_Z_M = 0.80
ARM_YAW = math.pi/2                     # bases face +y
CLIENT_CHAIR_XY = (0.0, 1.85)
CLIENT_EAR = (0.0, 1.85, 1.10)          # evaluator microphone: the client's ear (estimate: seated adult/child mean)
STATION_MIC = (0.0, -0.18, 1.36)        # the robot's measurement microphone on the cart mast (UMIK-1 class [S39])
MIC_MAX_SPL_DB = 133.0                  # UMIK-1 maximum SPL (datasheet)
ROOM_NOISE_DBA = 35.0                   # ANSI/ASA S12.60 background limit (standard [S40]); off by default

# ------------------------------------------------------------------ robot (datasheets via repo)
FRANKA_HAND_FINGER_SPEED = 0.05         # m/s per finger (datasheet; barcode/label precedent)
FRANKA_HAND_FORCE_N = (30.0, 70.0)      # continuous grasp force (datasheet)
FR3_JOINT_ACC = 10.0                    # rad/s^2, libfranka kMaxJointAcceleration (standard [S35, S36])
FR3_JOINT_JERK = 5000.0                 # rad/s^3, libfranka kMaxJointJerk
PANDA_JOINT_ACC = (15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0)   # rad/s^2, Panda datasheet limits (standard)
PANDA_JOINT_JERK = (7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0)
REFLEX_TORQUE_NM = (10.0, 20.0)         # libfranka setDefaultBehavior upper thresholds (nominal, accelerating)
REFLEX_RECOVERY_S = 1.0
REFLEX_FILTER_S = 0.02                  # filtered external-torque estimate (as in wine pouring)
PAD_SOLREF_TIMECONST = 0.005            # s, fingertip pad contact (label_application, anchored)
PAD_SOLREF_REF_MASS = 0.0291
PAD_MU = 0.8                            # pad rubber on lacquered wood / plastic (repo qualified value)
PAD_TORSIONAL_M = 0.0018                # pad torsional friction (m). mu L/4 = 0.0042 is the flat-pad value for a
                                        # bottle; a stick lies along the pad, and MuJoCo already resolves that line
                                        # contact into two points whose friction couple resists turning: with 0.0042 the
                                        # measured slip torque grew at 2x the line-contact estimate 2 mu N l/4, with
                                        # 0.0018 its slope matches it (grip bench) - calibrated against the estimate
STOP_SOLIMP = (0.99, 0.999, 0.5, 2.0)
CONTACT_GAP_M = 0.002
MAX_OBJECT_SPEED_M_S = 25.0             # guard: the fastest legal stick tip is ~11 m/s (G1)
MAX_OBJECT_SPIN_RAD_S = 3000.0             # guard (blow-up detector, not physics): a G1 hand at full servo speed spun a 21 g
                                           # mallet about its own axis to 1400 rad/s (friction at the 12.5 mm head: mu N r /
                                           # I_axial ~ 7e5 rad/s^2 for 2 ms, in a G1 stress test), which a 1000 rad/s guard
                                           # stopped as "unphysical"; the noslip divergence it exists for reached 28 km/s
READY_POSE = (0.0, -0.6, 0.0, -2.3, 0.0, 1.75, 0.785, 0.04, 0.04)   # authored: hands 0.35 m in front of the
                                        # bases, 0.25 m above the cart top, pointing down

# ------------------------------------------------------------------ evaluation
STRIKE_CLUSTER_S = {"drum": 0.010, "bar": 0.003}   # contacts closer than this belong to one strike (rim reflection)
RESTRIKE_WINDOW_S = (0.015, 0.120)      # a new cluster this soon after a strike is a double bounce
CONTACT_MAX_S = {"drum": 0.015, "bar": 0.003}       # longer = a buried / pressed stroke
LEVEL_BAND_DB = 3.0
REFERENCE_SPEEDS = {"bar": {"p": 0.5, "mf": 1.5, "f": 3.0}, "drum": {"p": 1.0, "mf": 3.0, "f": 6.0}}  # m/s
                                        # drum: Dahl's measured stroke speeds (literature); bar: estimates around Rucz's
                                        # 1 m/s mallet (Rucz used one speed; p/mf/f spread authored) - the level anchors
# finger-tap's mp reference (task.reference_level): a fingertip tap = the index fingertip's effective mass moving about
# the MCP joint, 21.5 g (Hajian & Howe 1997, J Biomech Eng 119:109 - measured, mean of subjects, flexion-extension)
# at 0.5 m/s, the middle of the peak fingertip speeds of light keystrokes, 0.3-0.7 m/s (Rempel et al. 1994, J Biomech
# 27:1101 - measured); mp within that range is a judgement (estimate).
FINGERTIP_TAP = dict(m_eff_kg=0.0215, v_m_s=0.5)
LEVEL_WINDOW_S = 0.25
SETTLE_STILL_S = 0.5
# take-sticks "put them back on the trap tables" (eight-task set): a stick is back when it rests on either
# trap table - its midpoint over the table's top, its axis within max_tilt_deg of horizontal, its midpoint at most
# max_above_top_m above the top (lying on it or in its saddles, 24 mm), no hand holding it, and still: within still_m
# of where it is now over the last still_s (the robot's own stillness window). Authored; a rule of within 30 mm of the
# starting butt position would fail a stick laid back 4 cm along the table
STICK_RETURN = dict(max_tilt_deg=10.0, max_above_top_m=0.05, still_m=0.005, still_s=SETTLE_STILL_S)
STRIKE_MIN_SPEED_M_S = 0.05
MELODY_EVAL = dict(strikers=("mallet_left_head", "mallet_right_head"), onset_s=0.05, tempo_rel=0.03, tone_min_db=-10.0,
                   neighbour_max=0.05, ring_max_damping_db=10.0, ring_beats=0.5, regroup_s=RESTRIKE_WINDOW_S[1],
                   onset_zero_s=0.10, tempo_zero_rel=0.20, fail_scale=0.9, tone_zero_db=-20.0, contact_zero_ms=6.0,
                   neighbour_zero=0.10, ring_zero_db=20.0, rhythm_span_beats=4.0)


def ledger():
    """Every scalar/tuple/dict constant of this module (run metadata)."""
    out = {}
    for k, v in globals().items():
        if k.isupper() and isinstance(v, (int, float, tuple, dict, bool, str)):
            out[k] = v
    return out


# Continuous progress for the five-task set (finger-tap, strike-and-stop, echo, melody, pentatonic-setup), following
# the melody contract (MELODY_EVAL): score = 1 if every check passes, else fail_scale x progress. Each quantity is
# graded linearly from 1 at the check's own threshold to 0 at its zero point, so a near miss is visible and any way of
# doing the job earns the same credit. The eight-task set adds take-sticks, accent-ostinato and steady-beat under the
# same rule.
TASK_PROGRESS = dict(
    fail_scale=0.9,
    tempo_zero_rel=0.20,          # mean interval off the stated tempo by 20 %: no tempo credit
    steadiness_zero_cv=0.20,      # interval spread (coefficient of variation) 20 %: no steadiness credit
    level_zero_db=12.0,           # a tap 12 dB away from the mezzo-piano reference: no level credit
    start_zero_beats=4.0,         # echo begun 4 beats after the stimulus ends (2 are allowed): no start credit
    ratio_zero_factor=3.0,        # echo interval ratio off by 3x its tolerance: no rhythm credit for that interval
    tilt_zero_deg=45.0,           # a removed bar in the tray tilted 45 deg (15 allowed): no "flat" credit
    off_pins_credit=0.25,         # a bar to remove that is off its pins but not in the tray
    beat_tempo_zero_rel=0.10,     # steady-beat: a section 10 % off its tempo (the size of the stated change, 1 %
                                  # allowed) earns no tempo credit, so a beat that ignores the change gets none there
    level_sd_zero_db=8.0,         # steady-beat: stroke levels spread 8 dB SD (2 allowed): no even-level credit
    accent_share=0.5,             # accent-ostinato: the accents' half of the progress (the other half: tempo,
                                  # steadiness and clean strokes), so an even beat without accents earns half
)
