"""Tool and prop geometry (one parameter set for physics, paint charts and rendering).

Each item is a list of classroom_arrangement.furniture.Part (box: half extents; capsule/cylinder: radius, half height
along local z; collide=False parts carry mass and are drawn only). Body frames:
  tray      origin on the floor under the footprint centre, x along the length (well at -x), z up
  roller    frame body: origin where the wire meets the cage axle, axle along +y, handle down -z; cover body: its centre
  brush     origin at the ferrule's bristle face, +z along the bristles toward the tips, x across the width
  pail/can  origin at the bottom centre, z up; aerosol: same, the button is a separate body on a slide joint
  table/box/board/crate/stencil/palette: origin at the bottom centre (box: open front toward -y)
"""

import math
import zlib
import numpy as np
from scipy.spatial.transform import Rotation

from classroom_arrangement.furniture import Part, _seg
from . import params as P


def _q(rot):
    return tuple(float(x) for x in rot.as_quat())


def octagon_prism(name, r, half_len, centre, mass, material, mu=None, axis_quat=(0., 0., 0., 1.), n=6):
    """A round grip of radius r along local z as the union of n thin boxes 180/n deg apart, each a x b with
    b = a tan(90/n deg) and corners on the circle (a = r cos(90/n deg)): the outline stays between 0.966 r and r for
    n = 6. MuJoCo's cylinder-box pair yields a single contact point (no torsional resistance at a Franka pad), box-box
    yields a contact patch, as the slightly compliant real grip does. Drawn as a cylinder. (Two square boxes 45 deg
    apart would make an eight-pointed star whose points stand 1.31 r out -- 43 mm on the 66 mm aerosol can, which
    jams the hand at its open stop.)"""
    th = math.pi/(2*n)
    a = r*math.cos(th)
    b = a*math.tan(th)
    base = Rotation.from_quat(axis_quat)
    return [Part(f"{name}_{k}", "box", (a, b, half_len), centre, _q(base*Rotation.from_euler("z", k*math.pi/n)), mass=mass/n,
                 material=material, role="grip", mu=mu) for k in range(n)]


def _scale(parts, total):
    m = sum(p.mass for p in parts)
    for p in parts:
        p.mass *= total/m
    return parts


# ------------------------------------------------------------------ tray (Wooster R404 class)
def tray_geometry():
    T = P.TRAY
    L, W, H = T["size"]
    t = T["wall"]
    x_well = -L/2 + T["well_len"]
    z_bot = H - T["well_depth"]                   # well floor (top surface)
    z_top_ramp = z_bot + T["ramp_drop"]
    x_ramp_end = L/2 - 0.012
    ang = math.atan2(z_top_ramp - z_bot, x_ramp_end - x_well)
    return dict(L=L, W=W, H=H, t=t, x_well=x_well, z_bot=z_bot, z_ramp=z_top_ramp, x_ramp_end=x_ramp_end, ang=ang)


def tray():
    g = tray_geometry()
    L, W, H, t = g["L"], g["W"], g["H"], g["t"]
    parts = []
    xw, zb = g["x_well"], g["z_bot"]
    parts.append(Part("well_floor", "box", ((xw + L/2)/2, W/2 - t, t/2), ((-L/2 + xw)/2, 0., zb - t/2), mass=1., material="tray"))
    rl = math.hypot(g["x_ramp_end"] - xw, g["z_ramp"] - zb)
    c = np.array([(xw + g["x_ramp_end"])/2, 0., (zb + g["z_ramp"])/2])
    n = np.array([-math.sin(g["ang"]), 0., math.cos(g["ang"])])
    parts.append(Part("ramp", "box", (rl/2, W/2 - t, t/2), tuple(c - n*t/2), _q(Rotation.from_euler("y", -g["ang"])),
                      mass=1.4, material="tray", role="ramp"))
    parts.append(Part("lip", "box", ((L/2 - g["x_ramp_end"])/2, W/2 - t, t/2), ((g["x_ramp_end"] + L/2)/2, 0., g["z_ramp"] - t/2),
                      mass=0.1, material="tray"))
    for k, y in enumerate((-W/2 + t/2, W/2 - t/2)):
        parts.append(Part(f"side_{k}", "box", (L/2, t/2, (H - zb + t)/2), (0., y, (H + zb - t)/2), mass=1.0, material="tray"))
    parts.append(Part("front", "box", (t/2, W/2, (H - zb + t)/2), (-L/2 + t/2, 0., (H + zb - t)/2), mass=0.4, material="tray"))
    parts.append(Part("back", "box", (t/2, W/2, (H - g["z_ramp"] + t)/2), (L/2 - t/2, 0., (H + g["z_ramp"] - t)/2), mass=0.1,
                      material="tray"))
    # feet: the well floor stands on a skirt; the ramp's high end on two moulded legs
    for k, (x, y) in enumerate(((-L/2 + 0.02, -W/2 + 0.02), (-L/2 + 0.02, W/2 - 0.02))):
        parts.append(Part(f"foot_{k}", "box", (0.015, 0.015, (zb - t)/2), (x, y, (zb - t)/2), mass=0.05, material="tray"))
    for k, y in enumerate((-W/2 + 0.02, W/2 - 0.02)):
        parts.append(Part(f"leg_{k}", "box", (0.015, 0.015, (g["z_ramp"] - t)/2), (L/2 - 0.03, y, (g["z_ramp"] - t)/2),
                          mass=0.08, material="tray"))
    return _scale(parts, P.TRAY["mass"])


# ------------------------------------------------------------------ roller frame and cover (Sherlock R017-9 class)
def roller_frame():
    R = P.ROLLER
    parts = []
    c, q, hl = _seg((0, 0, 0), (0, 0, -R["wire_len"]))
    parts.append(Part("wire", "capsule", (R["wire_r"], hl), c, q, mass=0.10, material="steel"))
    c, q, hl = _seg((0, 0, 0), (0, R["cover_len"] + 0.016, 0))
    parts.append(Part("axle", "capsule", (R["wire_r"], hl), c, q, mass=0.10, material="steel", collide=False))
    gz = -R["wire_len"] - R["grip_len"]/2
    parts += octagon_prism("grip", R["grip_r"], R["grip_len"]/2, (0., 0., gz), 0.14, "grip", mu=0.8)
    parts.append(Part("ferrule_collar", "cylinder", (R["grip_r"]*0.75, 0.01), (0., 0., -R["wire_len"] + 0.01), mass=0.0,
                      material="steel", collide=False))
    return _scale(parts, R["frame_mass"])


def roller_cover_center():
    R = P.ROLLER
    return (0., 0.008 + R["cover_len"]/2, 0.)


def roller_cover():
    R = P.ROLLER
    r_out = R["core_od"]/2 + R["nap"]
    r_stop = r_out - R["hard_stop"]*R["nap"]
    q = _q(Rotation.from_euler("x", -math.pi/2))           # cylinder axis (local z) along +y
    return [Part("core", "cylinder", (r_stop, R["cover_len"]/2), (0., 0., 0.), q, mass=R["cover_mass"], material="nap",
                 role="cover", mu=R["mu_dry"])]


# ------------------------------------------------------------------ brushes
def brush(kind):
    B = P.BRUSHES[kind]
    fx, fy, fz = B["ferrule"]
    parts = [Part("ferrule", "box" if not B.get("round") else "cylinder",
                  (fx/2, fy/2, fz/2) if not B.get("round") else (fx/2, fz/2), (0., 0., -fz/2), mass=0.3, material="ferrule")]
    hl = B["handle_len"]
    if kind == "sash":
        parts.append(Part("handle", "box", (0.0125, 0.0095, hl/2), (0., 0., -fz - hl/2), mass=0.55, material="alder",
                          role="grip", mu=0.6))
    elif B.get("dogleg_deg"):
        c, q, h2 = _seg((0, 0, -fz), tuple(np.array([0, 0, -fz]) + hl*dogleg_dir(kind)))
        parts.append(Part("handle", "capsule", (B["handle_r"], h2), c, q, mass=0.55, material="alder", role="grip", mu=0.6))
    else:
        c, q, h2 = _seg((0, 0, -fz), (0, 0, -fz - hl))
        parts.append(Part("handle", "capsule", (B["handle_r"], h2), c, q, mass=0.55, material="alder", role="grip", mu=0.6))
    parts.append(Part("bristles", "box", (B["width"]/2, B["thick"]/2, B["length"]/2), (0., 0., B["length"]/2), mass=0.15,
                      material="bristle", collide=False))
    return _scale(parts, B["mass"])


def dogleg_dir(kind):
    """Handle direction (brush frame) of a dog-leg brush: back from the ferrule, bent by the head angle toward -y (the
    brush's thickness side), so the head meets a wall at that angle while the handle runs along it."""
    a = math.radians(P.BRUSHES[kind]["dogleg_deg"])
    return np.array([0., -math.sin(a), -math.cos(a)])


def tuft_layout(kind):
    """Tuft clamp points (brush frame, z = 0 face), rest directions, free lengths and filaments per tuft."""
    B = P.BRUSHES[kind]
    nx, ny = B["tufts"]
    W, Tk, L = B["width"], B["thick"], B["length"]
    xs = (np.arange(nx) + 0.5)/nx*W - W/2
    ys = (np.arange(ny) + 0.5)/ny*Tk - Tk/2
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    if B.get("round"):
        keep = (X/(W/2))**2 + (Y/(Tk/2))**2 <= 1.0 + 1e-9
        X, Y = X[keep], Y[keep]
    n = len(X)
    rng = np.random.default_rng(zlib.crc32(kind.encode()))      # stable seed (Python's str hash differs per process)
    # rest directions: across the width the bundle flares slightly out of the ferrule (2 deg at the edge, estimate);
    # across the thickness a trade sash brush is chiselled -- its bristles converge to a tip line a few mm thick
    # (chisel_deg at the outer rows: the 16 mm pack narrows to ~4 mm at the tips, estimate from the product's chisel
    # trim, Purdy product page); plus 0.5 deg waviness (1.5 deg for the untrimmed stencil brush). A 4 deg outward
    # flare in both directions would stand the outer rows' tips 10.5 mm off the axis, and they would buckle outward
    # under any press (E8)
    rx = X/(W/2); ry = Y/(Tk/2)
    ch = math.radians(B.get("chisel_deg", -4.0))
    wav = np.radians(0.5 if "chisel_deg" in B else 1.5)          # a trimmed chisel holds its tip line to ~1 mm
    tilt = np.stack([np.radians(2.0 if "chisel_deg" in B else 4.0)*rx, -ch*ry], 1) + rng.normal(0, wav, (n, 2))
    d = np.stack([np.sin(tilt[:, 0]), np.sin(tilt[:, 1]), np.ones(n)], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    # trim profile: angled sash 15 deg bevel across the width (long side at +x); round brushes taper to a point
    if B["bevel_deg"] > 0:
        Ls = L - (W/2 - X)*math.tan(math.radians(B["bevel_deg"]))
    elif B.get("pointed"):
        Ls = L*(1.0 - 0.35*np.sqrt(rx**2 + ry**2))
        # a wet pointed round holds a point: capillary bridges pull the hairs together, so every tuft converges on the
        # brush axis at the tip (to within ~10 % of its radius) instead of flaring. With parallel / flaring tufts a
        # #8 round touches paper with a 10 mm line even at a 1 mm press (E6): no pressure control
        conv = -0.9*np.stack([X, Y], 1)/Ls[:, None]
        d = np.stack([conv[:, 0], conv[:, 1], np.ones(n)], 1) + np.c_[rng.normal(0, np.radians(1.0), (n, 2)), np.zeros(n)]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
    else:
        Ls = np.full(n, L)
    return np.stack([X, Y, np.zeros(n)], 1), d, Ls, B["filaments"]/n


# ------------------------------------------------------------------ containers
def _staves(r0, r1, h, t, n, mass, material, z0=0.0):
    parts = []
    for k in range(n):
        a = 2*math.pi*k/n
        r = (r0 + r1)/2
        tilt = math.atan2(r1 - r0, h)
        w = 2*math.pi*max(r0, r1)/n*0.55 + 0.002
        rot = Rotation.from_euler("z", a)*Rotation.from_euler("y", tilt)
        parts.append(Part(f"stave_{k}", "box", (t/2, w, h/2), (r*math.cos(a), r*math.sin(a), z0 + h/2), _q(rot), mass=mass/n,
                          material=material))
    return parts


def pail():
    G = P.PAIL
    parts = [Part("bottom", "cylinder", (G["r_bottom"], 0.001), (0., 0., 0.001), mass=0.03, material="pail")]
    parts += _staves(G["r_bottom"] + G["wall"]/2, G["r_top"] + G["wall"]/2, G["height"], G["wall"], 20, 0.11, "pail")
    return _scale(parts, G["mass"])


def can():
    G = P.CAN
    parts = [Part("bottom", "cylinder", (G["r"], 0.002), (0., 0., 0.002), mass=0.08, material="can_steel")]
    parts += _staves(G["r"], G["r"], G["height"], 0.002, 16, 0.27, "can_steel")
    return _scale(parts, G["mass"])


def palette():
    return [Part("plate", "box", (0.125, 0.085, 0.006), (0., 0., 0.006), mass=0.15, material="palette")]


PALETTE_WELLS = ((-0.07, 0.035), (0.0, 0.035), (0.07, 0.035), (-0.07, -0.035), (0.0, -0.035), (0.07, -0.035))
PALETTE_WELL = dict(r=0.020, depth=0.008, top=0.012)


# ------------------------------------------------------------------ furniture and props
def table():
    T = P.TABLE
    sx, sy = T["size"]
    zt = T["height"]
    parts = [Part("top", "box", (sx/2, sy/2, T["top_t"]/2), (0., 0., zt - T["top_t"]/2), mass=6.0, material="table_top")]
    for k, (x, y) in enumerate(((-sx/2 + 0.08, -sy/2 + 0.06), (-sx/2 + 0.08, sy/2 - 0.06), (sx/2 - 0.08, -sy/2 + 0.06),
                                (sx/2 - 0.08, sy/2 - 0.06))):
        c, q, hl = _seg((x, y, 0.012), (x, y, zt - T["top_t"]))
        parts.append(Part(f"leg_{k}", "capsule", (T["leg_r"], hl), c, q, mass=1.0, material="steel_dark"))
    return _scale(parts, T["mass"])


def board():
    return [Part("board", "box", (0.30, 0.225, 0.006), (0., 0., 0.006), mass=2.4, material="mdf")]


def spray_box():
    sx, sy, sz, t = 0.60, 0.45, 0.45, 0.004
    parts = [Part("floor", "box", (sx/2, sy/2, t/2), (0., 0., t/2), mass=0.25, material="cardboard"),
             Part("back", "box", (sx/2, t/2, sz/2), (0., sy/2 - t/2, sz/2), mass=0.2, material="cardboard"),
             Part("left", "box", (t/2, sy/2, sz/2), (-sx/2 + t/2, 0., sz/2), mass=0.15, material="cardboard"),
             Part("right", "box", (t/2, sy/2, sz/2), (sx/2 - t/2, 0., sz/2), mass=0.15, material="cardboard")]
    # a shipping carton on its side with the top and front flaps cut away: the usual hobby spray booth (an open top
    # also lets the arm reach over the work)
    return parts


SPRAY_BOX = dict(size=(0.60, 0.45, 0.45), t=0.004)


def crate_side():
    return [Part("panel", "box", (0.225, 0.15, 0.006), (0., 0., 0.006), mass=1.1, material="plywood")]


def stencil():
    S = P.STENCIL
    sx, sy = S["size"]
    m = S["rho"]*sx*sy*S["thick"]
    return [Part("sheet", "box", (sx/2, sy/2, S["thick"]/2), (0., 0., S["thick"]/2), mass=m, material="pet", mu=S["mu"])]


def aerosol_can():
    A = P.AEROSOL
    body_h = A["height"] - 0.026
    parts = octagon_prism("can", A["r"], body_h/2, (0., 0., body_h/2), A["can_mass"]*0.85, "aerosol", mu=0.5)
    parts += [Part("shoulder", "cylinder", (A["r"]*0.72, 0.006), (0., 0., body_h + 0.006), mass=A["can_mass"]*0.1,
                  material="aerosol_metal"),
             Part("valve_cup", "cylinder", (0.0127, 0.004), (0., 0., body_h + 0.016), mass=A["can_mass"]*0.05,
                  material="aerosol_metal")]
    return parts


def aerosol_button_base():
    """Button rest position (can frame): its bottom sits 2.2 mm above the valve cup (the stroke)."""
    A = P.AEROSOL
    body_h = A["height"] - 0.026
    return (0., 0., body_h + 0.020 + A["stroke"] + A["button_h"]/2)


def aerosol_button():
    A = P.AEROSOL
    return [Part("button", "cylinder", (A["button_r"], A["button_h"]/2), (0., 0., 0.), mass=0.003, material="button",
                 role="button", mu=0.6)]
