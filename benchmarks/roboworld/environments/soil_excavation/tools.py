"""Tool and prop geometry and mass models (one parameter set for MuJoCo collision, the MPM colliders and rendering).

Each builder returns classroom_arrangement.furniture.Part lists in the tool's body frame; part masses are shares of the
catalogue mass (params), placed where the steel, aluminium and wood are, so centres of mass and inertias follow.
Part.role tags: "blade" (soil-engaging steel), "grip" (where a hand holds it), "no_mpm" (never a soil collider),
"mpm_only" (soil collider only: the auger flight, which MuJoCo sees as its swept cylinder), "visual" (drawn only).

Frames
  spade, trench shovel, hoe: origin at the middle of the cutting edge, +z up the shaft, +x out of the blade's front
      face (the face that leads when pushed into the ground), +y across the blade.
  trowel: origin at the blade tip, +z toward the handle, +x out of the scoop's hollow.
  mattock: origin at the handle's butt, +z up the handle to the head, pick toward +x, adze toward -x.
  digger halves: origin at the hinge pin (pin axis +y); the blade below (-z), the handle above; half "a" at +x.
  drill: origin on the spindle axis at the chuck face, spindle along -z (drilling down), pistol grip toward -x.
  auger (on the spindle): origin at the hex end in the chuck, flight toward -z.
  tamper (2x4), post (4x4): origin at the bottom end's centre, +z along the length.
  root ball: origin at the bottom centre, +z up.
"""

import math
import numpy as np
from classroom_arrangement.furniture import Part, _seg, _scale
from . import params as P

STEEL, STAINLESS, ALU, WOOD = 7850., 7900., 2700., 650.


def _rod(name, a, b, r, mass, material, role="", kind="cylinder", mu=None):
    c, q, h = _seg(a, b)
    if kind == "capsule":
        h = max(h - r, 1e-4)
    return Part(name, kind, (r, h), c, q, mass, material, role, True, mu)


def _rotq(axis, deg):
    from scipy.spatial.transform import Rotation
    return tuple(Rotation.from_rotvec(np.asarray(axis, float)*math.radians(deg)).as_quat())


def curved_plate(width, length, thick, sagitta, n=16):
    """A flat plate (x thickness, y width, z length from z = 0 up) whose cutting edge (z = 0) is a shallow arc of the
    given sagitta (the edge centre lowest): closed triangle mesh for the MPM collider. A real digging spade's edge is
    not a straight grid-aligned line; with one, the whole edge crosses a node plane at once and the insertion force comes
    in 250-360 N spikes between 2-20 N (B4 kinematic run)."""
    ys = np.linspace(-width/2, width/2, n + 1)
    R = (sagitta**2 + (width/2)**2)/(2*sagitta) if sagitta > 0 else 1e9
    zb = sagitta - (R - np.sqrt(np.maximum(R*R - ys**2, 0.))) if sagitta > 0 else np.zeros_like(ys)
    zb = zb - zb.min()
    verts = []
    for x in (-thick/2, thick/2):
        for y, z in zip(ys, zb):
            verts += [(x, y, z), (x, y, length)]
    verts = np.asarray(verts, np.float32)
    m = len(ys)
    f = []
    idx = lambda side, i, top: side*2*m + 2*i + top
    for i in range(m - 1):
        for side, flip in ((0, True), (1, False)):
            a, b, c, d = idx(side, i, 0), idx(side, i + 1, 0), idx(side, i + 1, 1), idx(side, i, 1)
            f += [(a, c, b), (a, d, c)] if flip else [(a, b, c), (a, c, d)]
        for top in (0, 1):                                   # bottom edge and top edge strips
            a, b, c, d = idx(0, i, top), idx(0, i + 1, top), idx(1, i + 1, top), idx(1, i, top)
            f += [(a, b, c), (a, c, d)] if top == 0 else [(a, c, b), (a, d, c)]
    for i, side_i in ((0, 0), (m - 1, m - 1)):              # end caps
        a, b, c, d = idx(0, i, 0), idx(0, i, 1), idx(1, i, 1), idx(1, i, 0)
        f += [(a, b, c), (a, c, d)] if i == 0 else [(a, c, b), (a, d, c)]
    import trimesh
    m = trimesh.Trimesh(verts, np.asarray(f, np.int32), process=False)
    trimesh.repair.fix_normals(m)                       # consistent outward winding (the collider's inside/outside)
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32)


def spade():
    S = P.SPADE
    w, L, t = S["head"]
    blade_L = L - 0.02                                   # the 290 mm head includes the tread fold
    r = S["shaft_r"]
    parts = [Part("blade", "box", (t/2, w/2, blade_L/2), (0., 0., blade_L/2), mass=STAINLESS*w*blade_L*t,
                  material="stainless", role="blade no_mpm cut"),
             Part("blade_edge", "mesh", (1., 1., 1.), (0., 0., 0.), mass=0.0, material="stainless", role="mpm_only blade",
                  collide=False),
             Part("tread_l", "box", (0.009, 0.045, 0.004), (-0.007, 0.045, blade_L), mass=0.04, material="stainless", role="blade"),
             Part("tread_r", "box", (0.009, 0.045, 0.004), (-0.007, -0.045, blade_L), mass=0.04, material="stainless", role="blade"),
             _rod("socket", (0., 0., blade_L - 0.01), (0., 0., blade_L + 0.13), 0.020, 0.30, "stainless"),
             _rod("shaft", (0., 0., blade_L + 0.13), (0., 0., 0.84), r, WOOD*math.pi*r*r*(0.84 - blade_L - 0.13), "ash",
                  role="grip"),
             _rod("d_left", (0., 0.012, 0.84), (0., 0.058, 0.95), 0.011, 0.06, "ash"),
             _rod("d_right", (0., -0.012, 0.84), (0., -0.058, 0.95), 0.011, 0.06, "ash"),
             _rod("d_grip", (0., -0.062, 0.955), (0., 0.062, 0.955), 0.016, 0.13, "ash", role="grip")]
    # the soil sees the blade with its shallow edge arc (S&J digging spade: a straight-looking edge with rounded corners;
    # 12 mm sagitta across 180 mm, estimate from product photos)
    parts[1].mesh = curved_plate(w, blade_L, t, P.SPADE["edge_sagitta"])
    return _scale(parts, S["mass"])


def trench_shovel():
    S = P.TRENCH_SHOVEL
    w, L, t = S["head"]
    v, lift = math.radians(S["v_deg"]), math.radians(S["lift_deg"])
    r = S["shaft_r"]
    # blade from the edge (origin) up to the socket along a direction leaning back by the lift angle
    axis = np.array([-math.sin(lift), 0., math.cos(lift)])
    top = axis*L
    parts = []
    for side, sgn in (("l", 1.), ("r", -1.)):
        # each half of the V: a plate w/2 wide, its inner edge on the centre ridge (behind the front, -x)
        n = np.array([math.cos(v)*math.cos(lift), -sgn*math.sin(v), math.cos(v)*math.sin(lift)])
        across = np.cross(axis, n); across /= np.linalg.norm(across)
        if across[1]*sgn < 0:
            across = -across
        n = np.cross(across, axis)                 # right-handed frame (x = y x z); the plate normal up to sign
        c = axis*L/2 + across*w/4
        R = np.stack([n, across, axis], 1)
        from scipy.spatial.transform import Rotation
        q = tuple(Rotation.from_matrix(R).as_quat())
        parts.append(Part(f"blade_{side}", "box", (t/2, w/4, L/2), tuple(map(float, c)), tuple(map(float, q)),
                          mass=STEEL*w/2*L*t, material="steel_red", role="blade cut"))     # python floats: Warp rejects float64
    parts.append(_rod("socket", top - axis*0.02, top + np.array([0., 0., 0.12]), 0.019, 0.25, "steel_red"))
    s0 = top + np.array([0., 0., 0.12])
    parts.append(_rod("shaft", s0, s0 + np.array([0., 0., S["length"] - s0[2] - 0.02]), r,
                      WOOD*math.pi*r*r*(S["length"] - s0[2]), "ash", role="grip"))
    parts.append(_rod("cap", (0., 0., S["length"] - 0.03), (0., 0., S["length"]), 0.018, 0.04, "rubber_black"))
    return _scale(parts, S["mass"])


def trowel():
    S = P.TROWEL
    w, bl, t = S["head"]
    d = S["depth"]
    ang = math.atan2(d, w/2 - 0.02)            # side walls rising to the 30 mm scoop depth
    parts = [Part("blade_mid", "box", (t/2, 0.020, bl/2), (0., 0., bl/2), mass=ALU*0.04*bl*t, material="aluminium",
                  role="blade cut")]
    side_w = math.hypot(w/2 - 0.02, d)
    for sgn, nm in ((1., "l"), (-1., "r")):
        q = _rotq((0., 0., 1.), 0.)
        from scipy.spatial.transform import Rotation
        R = Rotation.from_rotvec((0., 0., -sgn*ang))           # tilt the plate about the tool axis
        c = (0.5*side_w*math.sin(ang), sgn*(0.02 + 0.5*side_w*math.cos(ang)), bl/2)
        parts.append(Part(f"blade_{nm}", "box", (t/2, side_w/2, bl/2), c, tuple((R*Rotation.from_quat(q)).as_quat()),
                          mass=ALU*side_w*bl*t, material="aluminium", role="blade cut"))
    parts.append(_rod("neck", (0., 0., bl), (0., 0., bl + 0.03), 0.008, 0.012, "aluminium"))
    parts.append(_rod("handle", (0., 0., bl + 0.03), (0., 0., S["length"]), S["grip_r"], 0.07, "rubber_green", role="grip",
                      kind="capsule"))
    return _scale(parts, S["mass"])


def mattock():
    S = P.MATTOCK
    Lh = S["handle"]
    parts = [_rod("handle", (0., 0., 0.), (0., 0., Lh), 0.016, S["mass"] - S["head_mass"], "fiberglass_yellow", role="grip",
                  kind="capsule"),
             _rod("eye", (0., 0., Lh - 0.025), (0., 0., Lh + 0.015), 0.022, 0.16*S["head_mass"], "steel_black"),
             _rod("pick", (0.02, 0., Lh - 0.005), (S["pick"], 0., Lh - 0.035), 0.009, 0.36*S["head_mass"], "steel_black",
                  role="blade cut", kind="capsule"),
             Part("adze", "box", (0.003, S["adze"][0]/2, S["adze"][1]/2), (-0.02 - S["adze"][1]/2, 0., Lh - 0.03),
                  _rotq((0., 1., 0.), 90. - 12.), 0.48*S["head_mass"], "steel_black", "blade cut")]
    return parts


def digger_half(side):
    """One clamshell half: curved blade below the pin (3-plate arc of radius ~ blade width / 2), handle above."""
    S = P.DIGGER
    bw, bl = S["blade"]
    sgn = 1. if side == "a" else -1.
    pin = S["pin_above_tip"]
    rr = bw/2
    parts = []
    from scipy.spatial.transform import Rotation
    for k, a in enumerate((-50., 0., 50.)):
        # arc plates concave toward the opposite half (-sgn x); the side plates' edges sit 12 mm above the middle
        # plate's (the real blades' edges curve up toward their sides; one straight grid-aligned edge crosses the MPM
        # node planes all at once, as on the spade)
        ang = math.radians(a)
        cx = sgn*(0.012 + rr*(1 - math.cos(ang))*0.6)
        cy = rr*math.sin(ang)
        R = Rotation.from_rotvec((0., 0., ang*sgn))
        dz = 0.012 if a else 0.0
        parts.append(Part(f"blade_{k}", "box", (0.0015, rr*0.55, bl/2), (cx, cy, -pin + bl/2 + dz), tuple(R.as_quat()),
                          mass=STEEL*0.0015*2*rr*1.1*bl/3*2, material="steel_red", role="blade cut"))
    parts.append(Part("strap", "box", (0.004, 0.02, 0.03), (sgn*0.016, 0., -pin + bl + 0.02), mass=0.12, material="steel_red"))
    # handle: from just above the pin, splaying slightly outward (spread handles close the blades)
    a0 = np.array([sgn*0.02, 0., 0.04])
    a1 = np.array([sgn*0.07, 0., S["handles"] - pin + 0.12])
    parts.append(_rod("handle", a0, a1, 0.017, 0.85, "ash", role="grip"))
    parts.append(_rod("grip_cap", a1 - (a1 - a0)/np.linalg.norm(a1 - a0)*0.12, a1, 0.019, 0.08, "rubber_black", role="grip"))
    parts.append(Part("hinge_plate", "box", (0.006, 0.03, 0.04), (sgn*0.012, 0., 0.), mass=0.10, material="steel_red"))
    return _scale(parts, S["mass"]/2)


def drill_body():
    S = P.DRILL
    parts = [_rod("housing", (0.01, 0., 0.04), (-0.19, 0., 0.04), 0.036, 0.90, "dewalt_yellow"),
             Part("grip", "box", (0.022, 0.017, 0.055), (-0.14, 0., -0.03), _rotq((0., 1., 0.), -12.), 0.25, "rubber_black",
                  "grip"),
             Part("battery", "box", (0.055, 0.040, 0.030), (-0.135, 0., -0.115), mass=0.55, material="dewalt_black"),
             Part("collar", "box", (0.01, 0.03, 0.03), (0.0, 0., 0.04), mass=0.10, material="dewalt_black"),
             _rod("side_handle", (0.0, 0.03, 0.04), (0.0, 0.17, 0.04), 0.016, S["side_handle"], "rubber_black", role="grip")]
    # drill axis: the motor housing lies along x; the chuck faces +x. Rotate so the spindle points along -z.
    return _scale(parts, S["mass"] + S["side_handle"])


def drill_chuck():
    return [_rod("chuck", (0.0, 0., 0.04), (0.075, 0., 0.04), 0.022, 0.35, "steel_black")]


def drill_trigger():
    return [Part("trigger", "box", (0.006, 0.008, 0.016), (0., 0., 0.), mass=0.01, material="dewalt_black", role="trigger")]


def auger_flight_mesh(n_turn_seg=48):
    """Helical flight surface (inner radius = shaft, outer = D/2, pitch) as a thin closed sheet mesh for the MPM
    collider, in the auger frame (axis -z from the hex end)."""
    A = P.AUGER
    R0, R1, p, Lf, L = A["shaft_r"], A["diameter"]/2, A["pitch"], A["flight_len"], A["length"]
    turns = Lf/p
    n = int(turns*n_turn_seg)
    th = np.linspace(0, 2*np.pi*turns, n + 1)
    z = -L + (th/(2*np.pi))*p
    t = A["flight_t"]
    verts, faces = [], []
    for k, (a, zz) in enumerate(zip(th, z)):
        c, s = math.cos(a), math.sin(a)
        verts += [(R0*c, R0*s, zz - t/2), (R1*c, R1*s, zz - t/2), (R1*c, R1*s, zz + t/2), (R0*c, R0*s, zz + t/2)]
    for k in range(n):
        a, b = 4*k, 4*(k + 1)
        for i in range(4):
            i2 = (i + 1) % 4
            faces += [(a + i, b + i, b + i2), (a + i, b + i2, a + i2)]
    faces += [(0, 1, 2), (0, 2, 3), (4*n, 4*n + 2, 4*n + 1), (4*n, 4*n + 3, 4*n + 2)]
    return np.asarray(verts, np.float32), np.asarray(faces, np.int32)


def auger():
    A = P.AUGER
    L, Lf = A["length"], A["flight_len"]
    shaft_m = STEEL*math.pi*A["shaft_r"]**2*L
    parts = [_rod("shaft", (0., 0., 0.), (0., 0., -L), A["shaft_r"], shaft_m, "steel_black", role="no_mpm_core"),
             # MuJoCo: the flight's swept cylinder (stones, ground, robot); the soil sees the helix (mpm_only)
             _rod("flight_envelope", (0., 0., -L + Lf), (0., 0., -L), A["diameter"]/2, 0.0, "steel_black", role="no_mpm"),
             Part("flight", "mesh", (1., 1., 1.), (0., 0., 0.), mass=A["mass"] - shaft_m, material="steel_black",
                  role="mpm_only blade", collide=False),
             _rod("tip", (0., 0., -L), (0., 0., -L - 0.02), 0.006, 0.01, "steel_black", role="blade", kind="capsule")]
    parts[2].mesh = auger_flight_mesh()
    return parts


def tamper():
    T = P.TAMPER
    a, b = T["section"]
    L = T["length"]
    return [Part("stick", "box", (a/2, b/2, L/2), (0., 0., L/2), mass=T["rho"]*a*b*L, material="spf", role="grip blade")]


def post():
    S = P.POST
    a, L = S["section"], S["length"]
    return [Part("post", "box", (a/2, a/2, L/2), (0., 0., L/2), mass=S["rho"]*a*a*L, material="cedar", role="blade")]


def rootball():
    R = P.ROOTBALL
    r, hh = R["diameter"]/2, R["height"]/2
    return [Part("ball", "cylinder", (r, hh), (0., 0., hh), mass=R["mass"], material="rootball", role="blade"),
            Part("crown", "cylinder", (0.03, 0.02), (0., 0., 2*hh + 0.02), mass=0.05, material="bark", role="no_mpm"),
            Part("shrub", "cylinder", (0.16, 0.13), (0., 0., 2*hh + 0.17), mass=0.25, material="foliage", role="visual",
                 collide=False)]


def stone(radii, name="stone"):
    a, b, c = radii
    m = P.STONE_RHO*4/3*math.pi*a*b*c
    return [Part(name, "ellipsoid", (a, b, c), (0., 0., 0.), mass=m, material="stone", role="blade")]


def mass_properties(parts):
    """Mass, centre of mass and inertia (about the COM, body frame) of a part list (solid boxes, cylinders, capsules,
    ellipsoids; meshes as point masses at their vertices' centroid). Used for the MPM colliders' rewind and for
    checks; MuJoCo computes its own from the same shapes and densities."""
    from scipy.spatial.transform import Rotation
    mt = sum(p.mass for p in parts)
    if mt <= 0:
        return 0.0, np.zeros(3), np.eye(3)*1e-6
    com = sum(p.mass*np.asarray(p.pos, float) for p in parts)/mt
    I = np.zeros((3, 3))
    for p in parts:
        m = p.mass
        if m <= 0:
            continue
        s = p.size
        if p.kind == "box":
            a, b, c = 2*np.asarray(s)
            Il = m/12*np.diag([b*b + c*c, a*a + c*c, a*a + b*b])
        elif p.kind in ("cylinder", "capsule"):
            r, hh = s
            L = 2*hh
            Il = np.diag([m*(3*r*r + L*L)/12, m*(3*r*r + L*L)/12, m*r*r/2])
        elif p.kind == "ellipsoid":
            a, b, c = s
            Il = m/5*np.diag([b*b + c*c, a*a + c*c, a*a + b*b])
        else:
            Il = np.eye(3)*m*1e-4
        R = Rotation.from_quat(p.quat).as_matrix()
        d = np.asarray(p.pos, float) - com
        I += R @ Il @ R.T + m*(np.dot(d, d)*np.eye(3) - np.outer(d, d))
    return mt, com, I
