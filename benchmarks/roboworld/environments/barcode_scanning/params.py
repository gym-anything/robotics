"""Parameter ledger: every physical constant the environment uses, with its provenance.

Provenance labels: datasheet | standard | search | derived | estimate | authored | identified
("identified" = fitted on the decode-range bench against a published observable).
Probes READ these values; they never override them on a command line.
`LEDGER` is the machine-readable form written into every evidence file's metadata.
"""

import math

IN = 0.0254

# ------------------------------------------------------------------ scanner body (Zebra DS2278 class)
SCANNER_MASS_KG = 0.214                 # datasheet
SCANNER_CORDED_MASS_KG = 0.1616         # datasheet: the corded DS2208, 1 cm shorter; the cordless unit's extra 52 g
                                        # (its battery) sits in the handle and base
SCANNER_CORDED_HEAD_FRACTION = 0.6      # estimate: the corded unit's head (imager, window, LEDs) against its handle
SCANNER_ENVELOPE_M = (0.175, 0.066, 0.090)     # H x W x D, datasheet
CRADLE_MASS_KG = 0.151                  # datasheet
CRADLE_ENVELOPE_M = (0.072, 0.094, 0.122)      # H x W x D, datasheet

# ------------------------------------------------------------------ imager
WIDTH_PX, HEIGHT_PX = 640, 480          # datasheet
HFOV_DEG, VFOV_DEG = 32.8, 24.8         # datasheet
WAVELENGTH_M = 645e-9                   # illumination LEDs, datasheet
AIM_WAVELENGTH_M = 624e-9               # aim line, datasheet
FRAME_RATE_HZ = 30.0                    # estimate: SE4710 default; DS2208 unpublished
PUPIL_SETBACK_M = 0.050                 # derived: at the datasheet's 13 mm UPC near limit a UPC-A with its quiet zones (113
                                        # modules x 0.33 mm = 37.3 mm) just fills the 32.8 deg field 63 mm from the pupil (37.1 mm);
                                        # the near limit is a field-of-view limit (PRG Table 3-3 note).
PUPIL_DIAMETER_M = 0.00041              # derived: decoder-independent fit: the least worst blur, in
                                        # module widths, over every published near and far limit of PRG Table 3-3
FOCUS_DISTANCE_M = 0.232                # derived (the same fit), from the entrance pupil: 182 mm before the window, against the
                                        # 194 mm two other Zebra engines publish
LED_OFFSET_M = 0.010                    # estimate: LEDs either side of the lens
AIM_OFFSET_M = 0.008                    # estimate: aim source above the lens
AIM_CROSSING_M = 0.150                  # estimate: aim/optical axes cross (SE4710: 195 mm)
LED_FWHM_H_DEG, LED_FWHM_V_DEG = 39.3, 28.3   # derived: the SE4710 engine's pattern (50.3 x 32.0 deg full widths at half
                                        # intensity) scaled by the ratio of the fields of view (DS2208 32.8 x 24.8 deg,
                                        # SE4710 42 x 28 deg): the DS2208's engine is not the SE4710 and its pattern is
                                        # unpublished; elliptical Gaussian IES profile (a hard 25 deg cone left the edges unlit)
AIM_FAN_DEG, AIM_THICK_DEG = 32.8, 0.3  # estimate: fan = HFOV ("aiming line smaller when closer", PRG)

# ------------------------------------------------------------------ sensor
SENSOR_WEIGHTS = (0.35, 0.45, 0.20)     # estimate: mono CMOS response to the R, G, B render bands
FULL_WELL_E = 6000.0                    # estimate
READ_NOISE_E = 8.0                      # estimate
PRNU = 0.01                             # estimate
AE_TARGET_DN = 110.0                    # estimate
AE_HIGHLIGHT_DN = 230.0                 # estimate: hands-free mode holds the 99.5th percentile of the frame's block means
                                        # at or below this (highlight protection; a clipped label loses its bars with it)
AE_BLOCK_PX = 8                         # estimate: AE statistics on 8 x 8-pixel block means (a subsampled histogram)
GAIN_MAX = 8.0                          # estimate
# path-traced samples per pixel of a scanner frame: enough that the render's Monte Carlo noise is a third of the sensor's at
# that frame's gain. Measured: 1.27 % relative noise on a white card at
# 128 samples, falling about as 1/samples; the sensor's relative noise at the exposure target is sqrt(255 g /
# (AE_TARGET_DN FULL_WELL_E)), 2.0 % at gain 1 and 5 % at the hands-free gains of 6-7: 247 samples at gain 1, 97 at 6.5.
RENDER_MC_REF = (128, 0.0127)           # measured: (samples, relative Monte Carlo noise)
RENDER_NOISE_FRACTION = 1/3             # derived: the convergence criterion
RENDER_SPP_RANGE = (64, 256)            # authored: floor and ceiling (cost per sample in the task scenes on the shared GPU;
                                        # a bench scene costs ~4 ms)
RENDER_IMAGER_MAX_BOUNCES = 3           # derived: path depth of the scanner's frames (1.7-2.5x cheaper per
                                        # sample than 6; frames with a read within 2.7 % of 6's radiance, empty dark frames up
                                        # to 8 % darker; the same decodes on all 17 recorded task frames compared)
T_MIN_S = 20e-6                         # estimate
T_EXPOSURE_START_S = 5e-4               # estimate
T_MAX_HANDHELD_S = 0.5*0.33e-3/0.13     # derived: half a 13 mil module of blur at the 13 cm/s motion tolerance (datasheet)
T_MAX_HANDSFREE_S = 0.5*0.33e-3/0.762   # derived: 76.2 cm/s swipe speed (datasheet)
SENSOR_K = 3.1e6                        # identified (render calibration): white paper under LED+200 lx at 150 mm -> DN 128 at 0.4 ms, gain 1

# ------------------------------------------------------------------ light levels (renderer units, calibrated on the decode-range bench)
LED_INTENSITY = 2.03e6                  # derived from an estimate (render calibration): LED:room sensor signal 10:1 on white paper at
                                        # 150 mm, 200 lx room; the 10:1 is a power-budget estimate (0.3 W to the LEDs at 20-30 %,
                                        # ratio 6-14), not a device measurement; rescaled x 2.19/83.4 when the hard cone became the IES
                                        # profile, and / 1.163 when the pattern was scaled to the field of view (the on-axis gain
                                        # per unit intensity, measured with the normalised IES)
AIM_INTENSITY = 2.0e4                   # authored (render calibration): aim-line peak ~7x the 200 lx room illuminance at 150 mm
                                        # (a visible line on white stock; no published luminance)
ROOM_LUX_STOCKROOM = 200.0              # search: 150-300 lx manned rack areas / packing
ROOM_LUX_COUNTER = 500.0                # search: till areas

# ------------------------------------------------------------------ firmware (DS2208 Product Reference Guide)
DECODE_SESSION_TIMEOUT_S = 9.9
SAME_SYMBOL_TIMEOUT_S = 0.5
DIFFERENT_SYMBOL_TIMEOUT_S = 0.1
HANDSFREE_SESSION_MIN_S, HANDSFREE_SESSION_MAX_S = 0.25, 2.5
AIM_IDLE_OFF_S = 2.0
DECODE_LATENCY_S = 0.030                # estimate
GOOD_READ_LED_S = 0.2                   # estimate
MOTION_WAKE_LINEAR_M_S, MOTION_WAKE_ANGULAR_RAD_S = 0.02, 0.2   # estimate (stand-in for motion sensing)
PRESENTATION_WAKE_SPEED_M_S, PRESENTATION_ZONE_M = 0.005, 0.40  # authored stand-in for image-change detection

# ------------------------------------------------------------------ trigger (estimates; switch bounds from Omron D2F)
TRIGGER_ARM_M = 0.025
TRIGGER_PRELOAD_N = 1.0
TRIGGER_RATE_N_PER_M = 450.0
TRIGGER_CLOSE_M = 0.0030
TRIGGER_HYSTERESIS_M = 0.00019          # D2F MD <= 0.12 mm x lever ratio 1.6 (estimate)
TRIGGER_STOP_M = 0.0036                 # >= 0.25 mm overtravel (D2F)
TRIGGER_MASS_KG = 0.003
TRIGGER_DAMPING_RATIO = 0.4             # estimate: plastic pivot + snap switch (undamped, the lever rang 0.4-3.5 mm)
SWITCH_DEBOUNCE_S = 0.005
STOP_SOLIMP = (0.99, 0.999, 0.5, 2.0)   # numerical: MuJoCo limit impedance (d0, dmax, mid, power) for the trigger and key stops
                                        # (MuJoCo's default 0.9-0.95 let a 35 N squeeze drive the 3 g trigger ~1 mm through its
                                        # plastic stop)

# ------------------------------------------------------------------ keypad (Cherry MX Red class)
KEY_OPERATE_M, KEY_TRAVEL_M = 0.0020, 0.0040
KEY_FORCE_OPERATE_N, KEY_FORCE_INITIAL_N = 0.45, 0.30
KEY_RELEASE_M = 0.0018                  # estimate
KEY_MASS_KG = 0.002                     # estimate: ABS keycap + slider (1u); larger keys scale with area
KEY_DAMPING_RATIO = 0.3                 # estimate: slider friction + leaf contact
KEY_PITCH_M = 0.01905

# ------------------------------------------------------------------ cradle seat sensor (authored stand-in)
SEAT_TOL_M, SEAT_TOL_DEG, SEAT_HOLD_S = 0.003, 8.0, 0.2

# ------------------------------------------------------------------ hardcover book
BOOK_BOARD_M = 0.003                    # estimate: 2.5-3 mm greyboard + cloth (hardcover)
BOOK_COVER_MASS_KG = 0.08               # derived: 0.156 x 0.235 x 0.003 m greyboard at ~700 kg/m3 + cloth (K12)
BOOK_HINGE_FRICTION_NM = 0.02           # estimate: an opened cover lies open, a closed one stays closed
PAPERBACK_COVER_M = 0.0003              # estimate: 10-12 pt (0.25-0.30 mm) coated cover board
PAPERBACK_COVER_KG_M2 = 0.28            # estimate: 250-300 g/m2 cover stock
PAPERBACK_HINGE_FRICTION_NM = 0.002     # estimate: a creased card flap; its own weight (3 mN m) swings it open past vertical


def book_cover(sku):
    """(thickness m, mass kg, hinge friction N m) of a library book's front cover by binding."""
    w, _, h = sku.dims
    if sku.size_text == "Paperback":
        return PAPERBACK_COVER_M, PAPERBACK_COVER_KG_M2*w*h, PAPERBACK_HINGE_FRICTION_NM
    return BOOK_BOARD_M, BOOK_COVER_MASS_KG*(w*h)/(0.156*0.235), BOOK_HINGE_FRICTION_NM

# ------------------------------------------------------------------ friction (Coulomb, estimates)
# One coefficient per SHAPE, named by the material the value was estimated for (the keys read like pairs, but a value is
# given to each shape of that material). MuJoCo applies the LARGER of the two shapes' coefficients at equal priority
# (the fingertip pads, priority 1, govern their contacts).
MU = dict(pad_housing=0.8, board_deck=0.40, can_deck=0.30, glass_deck=0.35, film_deck=0.35, item_tote=0.35,
          board_board=0.45, book_counter=0.30, scanner_cradle=0.30, robot_default=0.8)
BOX_AS_HULL_MIN_MASS_KG = 5.0          # authored: box SKUs at least this heavy (the 12.5 kg water case) collide as 8-vertex convex
                                        # hulls: MuJoCo's box-box routine threw the case at 31.7 m/s from a bogus 584 mm contact. All
                                        # free boxes as hulls was measured and rejected (resting creep 24x, held-scanner slip 3.5x;
                                        # scene.Scene.box)
TORSIONAL_FRICTION_M = 0.001            # estimate: spin resistance of a free body's contacts, (2/3) mu a for a contact patch of
                                        # radius a ~ 4 mm at mu 0.35-0.4 (MuJoCo condim 4; world.py); the fingertip pads too
                                        # (a ~ 1.9 mm at the pad's mu 0.8)

# ------------------------------------------------------------------ scene geometry
SHELF_BAY_M = (36*IN, 18*IN, 72*IN)     # Uline H-7115 class
SHELF_PITCH_M = 1.5*IN                  # Uline H-7115
DECK_THICK_M = 5/8*IN                   # particle board, Uline H-7115
CART_TOP_M = 0.75                       # authored from reach
ARM_Y_M = 0.25
BASE_TO_SHELF_M = 0.40
COUNTER_TOP_M = 0.915                   # ADA 2010 §904.4

READY_POSE = (0., -1.3, 0., -2.8, 0., 1.6, 0.785, 0.04, 0.04)   # robot ready pose (arm joints 1-7, fingers): authored from
                                                                 # forward kinematics

# ------------------------------------------------------------------ physics
RIGID_DT_S = 0.001                      # default step; the environment's steps are in config.py
                                        # (1 ms for the FR3/Panda configurations, 0.5 ms for the G1)
GRAVITY = 9.81                          # standard gravity

# ------------------------------------------------------------------ contact detection and the robot's protective stop
REFLEX_RECOVERY_S = 1.0                # authored: after a collision reflex the arm holds and refuses new targets this long (the
                                       # real FR3 stays stopped until an error recovery; libfranka does not publish its duration)
REFLEX_TORQUE_NM = (10., 20.)          # sourced: libfranka examples/examples_common.cpp setDefaultBehavior, upper torque
                                       # thresholds (nominal, acceleration) on every joint (fetched 2026-09-29)
CONTACT_GAP_M = 0.0                    # contact detection envelope of boxes and planes (E17: MuJoCo's box-box routine returns
                                       # bogus deep points for separated boxes inside a positive gap; at 1 kHz a 3 m/s link moves
                                       # 3 mm per step, less than the thinnest free collider it can meet, the 4 mm basket wall)
MESH_CONTACT_GAP_M = 0.002             # convex hulls (GJK/EPA): with no envelope resting hull contacts flicker and a shelf of
                                       # jars crept faster and faster (2 mm/s after 10 s, rest check)


def ledger():
    """Machine-readable snapshot of this module (written into evidence metadata)."""
    g = globals()
    return {k: g[k] for k in sorted(g) if k.isupper() and not k.startswith("_")}

# ------------------------------------------------------------------ numerical-failure guard (world.py)
# No object here can move faster than a dropped or swung one; above these bounds the state is a solver failure and the
# episode stops as a backend fault, like a non-finite state.
MAX_OBJECT_SPEED_M_S = 20.             # authored: 3x the fastest plausible (a 2 m drop reaches 6.3 m/s)
MAX_OBJECT_SPIN_RAD_S = 1000.          # authored: 8x the fastest plausible (a 16 mm lip balm rolling at 1 m/s spins at 128 rad/s)
MAX_FINGER_COUPLING_M = 0.010          # authored: a Franka Hand's two fingers ride one drive, so their joint positions cannot
                                       # part; the stiff coupling lets them part <= 2.7 mm under the hardest legal impacts
                                       # (158 FR3/Panda cases); unconverged solves parted them over 10 mm in 36 of those cases,
                                       # up to 232 mm. Robot links are not under the speed bound above (driven at its URDF joint
                                       # speeds a G1 fingertip reached 24 m/s)
