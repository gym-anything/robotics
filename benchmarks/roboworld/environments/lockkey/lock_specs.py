"""Real dimensions of a Kwikset KW1 5-pin residential cylinder, and the
shared geometry helpers the lock-and-key modules use.

Source labels: READ = primary document read first hand, WEB = web page,
BLOG = single informal source. Units are METRES throughout, because the lock
is a real object at real size and the sim carries true dimensions.

Frame convention:
  X = plug/keyway axis, key inserts along -X into the plug face at X=0.
  Z = up. Pins ride vertically along Z. Springs push down (-Z). Gravity -Z.
  The plug rotates about X. The shear line is the horizontal plane Z = 0
  through the plug centre, i.e. the cylindrical surface at plug radius.
"""
import math

IN = 0.0254  # inches -> metres

# ---- pins (READ, cross-verified lsamichigan + naswek pinning sheet) ----
PIN_DIA = 0.115 * IN          # 2.921 mm, all pins
PIN_R = PIN_DIA / 2.0
DRIVER_LEN = 0.180 * IN       # 4.572 mm, regular Kwikset top pin
# Kwikset key (bottom) pin length by depth 1..7 (index 0 unused)
KEY_PIN_LEN_IN = [None, 0.172, 0.195, 0.218, 0.241, 0.264, 0.287, 0.310]
KEY_PIN_LEN = [None if v is None else v * IN for v in KEY_PIN_LEN_IN]

# ---- plug / cylinder (READ) ----
PLUG_DIA = 0.500 * IN         # 12.70 mm (lockwiki measured 0.498)
PLUG_R = PLUG_DIA / 2.0
N_PINS = 5

# ---- bitting geometry (READ, lsamichigan KW1 sheet) ----
CUT_INCREMENT = 0.023 * IN    # 0.5842 mm between adjacent depths
FIRST_CUT = 0.247 * IN        # 6.2738 mm, shoulder to cut 1
CUT_SPACING = 0.150 * IN      # 3.81 mm cut-to-cut
CUT_ANGLE_DEG = 90.0          # included angle of the V cut
CUT_ROOT_FLAT = 0.084 * IN    # 2.1336 mm flat at the bottom of each cut
DEPTH_TOL = 0.003 * IN        # +/- 76 um key cut tolerance
SPACE_TOL = 0.004 * IN        # +/- 102 um key spacing tolerance
MACS = 4                      # max adjacent cut difference

# ---- key blade (WEB measured) ----
BLADE_THICK = 2.03e-3         # KW1 measured 0.079-0.081 in
BLADE_HEIGHT = 0.335 * IN     # 8.509 mm
ROOT_DEPTH_IN = [None, 0.329, 0.306, 0.283, 0.260, 0.237, 0.214, 0.191]
ROOT_DEPTH = [None if v is None else v * IN for v in ROOT_DEPTH_IN]
EDP_IN = 0.501                # key-pin length + root depth (checked both ends)

# ---- forces / torques (READ, ANSI/BHMA A156.5-2014) ----
MAX_INSERT_FORCE_N = 13.0     # 3 lbf, 5-6 pin, new cylinder
MAX_INSERT_FORCE_WORN_N = 26.0  # 6 lbf, post 40k-cycle
MAX_PLUG_TORQUE_G1 = 0.14     # 18 ozf-in, Grade 1, unloaded plug
CYCLE_CAM_LOAD = 0.34         # 3 in-lbf cam preload during cycling
ATTACK_TORQUE = 34.0          # plug must not pass 45 deg at 300 lbf-in
KEY_STRENGTH_TORQUE = 2.6     # 23 in-lb, key survives 5 s

# ---- materials (WEB) ----
# brass plug/shell/pins ~8.5 g/cc; nickel-silver better pins/keys ~8.7
RHO_BRASS = 8500.0            # kg/m^3
# friction: brass-brass 0.9 dry / 0.6 lubricated (WEB, valve handbook);
# working range with factory lube treated as a calibration target
MU_START = 0.35

# ---- spring (BLOG single source; real closure = calibrate to insert gate) ----
SPRING_FORCE_GUESS = 0.3      # N near operating compression (0.2-0.5 blog)

# ---- the deadbolt the plug drives (WEB, multi-vendor) ----
BOLT_THROW = 1.0 * IN         # 25.4 mm
BOLT_DIA = 0.875 * IN         # 7/8 in

# ---- manufacturing realism: per-chamber drill error (WEB order-of-mag) ----
# a zero-error lock has no binding order, which real locks never show; this
# is a disclosed realism parameter, observable = a stable binding order.
CHAMBER_ERR = 0.001 * IN      # ~25 um


def pin_mass(length):
    """mass of a solid brass pin cylinder of the given length (m)."""
    return RHO_BRASS * math.pi * PIN_R * PIN_R * length


def stack_total(depth):
    """key-pin + driver length for a chamber cut to `depth`, in metres.
    the EDP closure says key-pin(depth) + root_depth(depth) = 0.501 in, so
    every correct stack lifts its key-pin top to the same shear plane."""
    return KEY_PIN_LEN[depth] + DRIVER_LEN


def chamber_x(i):
    """centre X of chamber i (0-based) measured from the plug face, in metres.
    negative because chambers sit inside the plug along -X from the face."""
    return -(FIRST_CUT + i * CUT_SPACING)


if __name__ == "__main__":
    # self-check: print the real numbers and verify the EDP closure law
    print("PIN_DIA mm      ", round(PIN_DIA * 1e3, 4))
    print("PLUG_DIA mm     ", round(PLUG_DIA * 1e3, 4))
    print("DRIVER_LEN mm   ", round(DRIVER_LEN * 1e3, 4))
    print("CUT_INCREMENT mm", round(CUT_INCREMENT * 1e3, 4))
    print("BLADE_THICK mm  ", round(BLADE_THICK * 1e3, 4))
    print("chamber X mm    ", [round(chamber_x(i) * 1e3, 2) for i in range(5)])
    print("--- EDP closure (key-pin + root should = 0.501 in) ---")
    for d in range(1, 7):
        s = (KEY_PIN_LEN_IN[d] + ROOT_DEPTH_IN[d])
        print(f"  depth {d}: {KEY_PIN_LEN_IN[d]:.3f} + {ROOT_DEPTH_IN[d]:.3f}"
              f" = {s:.3f} in {'OK' if abs(s - EDP_IN) < 1e-6 else 'MISMATCH'}")
    print("--- pin masses (g) ---")
    print("  driver:", round(pin_mass(DRIVER_LEN) * 1e3, 4))
    for d in range(1, 7):
        print(f"  keypin depth {d}:", round(pin_mass(KEY_PIN_LEN[d]) * 1e3, 4))
