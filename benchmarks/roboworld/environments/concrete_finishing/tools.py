"""The hand tools as rigid bodies: collision parts (classroom_arrangement.furniture.Part), the working sole the paste
model reads, and where a hand takes each one.

Body frame of every tool: origin at the centre of its working sole (the bottom face of a blade, the bottom of a
jointer's rod, a knife blade's lower face), x along the blade's length (the edge of a knife), y across it, z out of
the sole toward the handle. Sizes are the maker's (Marshalltown pages, params.TOOLS); masses are shipping weights
less the package (estimate); the handle is round, 32 mm across (snippet: retailer listings of 145D), as sold: not
squared for the gripper.

Each tool also says how its sole behaves on paste: `carry` (the material its edge cuts stays on the blade when it is
lifted: knives and small trowels that load from a hawk or a board) and the factors by which sliding densifies and
seals the surface (a steel trowel 1, a magnesium float 0.3, a wood float 0.2; estimates).
"""

import math
from dataclasses import dataclass, field
import numpy as np
from classroom_arrangement.furniture import Part
from . import params as P
from .paste import Sole

MU_TOOL = 0.35          # painted / bare metal and wood on cart tops and forms (estimate)


def box(name, size, pos, mass, material, role="tool", quat=(0., 0., 0., 1.), collide=True, mu=MU_TOOL):
    return Part(name, "box", tuple(s/2 for s in size), tuple(pos), tuple(quat), mass, material, role, collide, mu)


def cap_x(name, r, length, pos, mass, material, role="handle", mu=MU_TOOL):
    """A capsule along the tool's x axis, total length `length`."""
    q = (0., math.sin(math.pi/4), 0., math.cos(math.pi/4))           # local z -> x
    return Part(name, "capsule", (r, max(length/2 - r, 1e-4)), tuple(pos), q, mass, material, role, True, mu)


def cap_y(name, r, length, pos, mass, material, role="handle", mu=MU_TOOL):
    q = (-math.sin(math.pi/4), 0., 0., math.cos(math.pi/4))          # local z -> y
    return Part(name, "capsule", (r, max(length/2 - r, 1e-4)), tuple(pos), q, mass, material, role, True, mu)


def cyl_z(name, r, length, pos, mass, material, role="handle", mu=MU_TOOL):
    return Part(name, "cylinder", (r, length/2), tuple(pos), (0., 0., 0., 1.), mass, material, role, True, mu)


@dataclass
class Tool:
    name: str
    kind: str
    parts: list
    sole: Sole
    sole_part: str                    # the part that rests on the support plate
    grasp: tuple                      # (position, handle axis) in the tool frame where the hand pinches
    mass: float
    carry: bool = False
    dens: float = 0.0                 # densification factor of sliding (steel trowel 1)
    seal: float = 0.0                 # sealing factor (pitched steel)
    texture: float = 2.5e-4           # relief the sole leaves on a surface it shapes (m)
    second_grasp: tuple = None        # a second hand's place (the darby's knob)
    no_plate: tuple = ()              # parts that never rest on a plate (the edger's lip)
    meta: dict = field(default_factory=dict)
    bear: Sole = None                 # the flat part a support plate carries, where the cutting sole is not flat
    carry_cap: float = None           # m3 a carrying blade holds; beyond it the cut runs off as a roll's excess does


def hand_float(name="mag_float"):
    s = P.TOOLS["float"]
    L, W, T = s["blade"]
    hh = s["handle_h"]
    m = s["mass"]
    parts = [box("blade", (L, W, T), (0, 0, T/2), 0.70*m, "magnesium", role="sole"),
             box("post_a", (0.016, 0.020, hh - T - 0.010), (-0.075, 0, T + (hh - T - 0.010)/2), 0.04*m, "aluminum"),
             box("post_b", (0.016, 0.020, hh - T - 0.010), (0.075, 0, T + (hh - T - 0.010)/2), 0.04*m, "aluminum"),
             cap_x("handle", s["handle"][1], s["handle"][0], (0, 0, hh), 0.22*m, "handle_soft")]
    return Tool(name, "float", parts, Sole("blade", L/2, W/2, lift=P.ROLL_LIFT["float"]), "blade",
                grasp=((0, 0, hh), (1, 0, 0)), mass=m, dens=0.3, seal=0.1, texture=2.5e-4, meta=dict(part=s["part"]))


def wood_float(name="wood_float"):
    s = P.TOOLS["wood_float"]
    L, W, T = s["blade"]
    hh = s["handle_h"]
    m = s["mass"]
    parts = [box("blade", (L, W, T), (0, 0, T/2), 0.70*m, "redwood", role="sole"),
             cap_x("handle", s["handle"][1], s["handle"][0], (0, 0, hh), 0.30*m, "handle_wood")]
    parts.insert(1, box("post", (0.12, 0.02, hh - T - 0.010), (0, 0, T + (hh - T - 0.010)/2), 0.0, "handle_wood", collide=False))
    return Tool(name, "float", parts, Sole("blade", L/2, W/2, lift=P.ROLL_LIFT["float"]), "blade",
                grasp=((0, 0, hh), (1, 0, 0)), mass=m, dens=0.2, seal=0.05, texture=5.0e-4, meta=dict(part=s["part"]))


def finishing_trowel(name="trowel"):
    s = P.TOOLS["trowel"]
    L, W, T = s["blade"]
    hh = s["handle_h"]
    m = s["mass"]
    Tc = 0.002                                                        # collision thickness of the 1.5 mm blade
    parts = [box("blade", (L, W, Tc), (0, 0, Tc/2), 0.55*m, "trowel_steel", role="sole"),
             box("mounting", (0.21, 0.022, 0.006), (0, 0, Tc + 0.003), 0.15*m, "aluminum"),
             box("shank_a", (0.012, 0.012, hh - 0.025), (-0.05, 0, (hh - 0.025)/2 + 0.008), 0.03*m, "aluminum"),
             box("shank_b", (0.012, 0.012, hh - 0.025), (0.05, 0, (hh - 0.025)/2 + 0.008), 0.03*m, "aluminum"),
             cap_x("handle", s["handle"][1], s["handle"][0], (0, 0, hh), 0.24*m, "handle_soft")]
    return Tool(name, "trowel", parts, Sole("blade", L/2, W/2, lift=P.ROLL_LIFT["trowel"]), "blade",
                grasp=((0, 0, hh), (1, 0, 0)), mass=m, dens=1.0, seal=1.0, texture=3.0e-5, meta=dict(part=s["part"]))


def darby(name="darby"):
    s = P.TOOLS["darby"]
    L, W, T = s["blade"]
    hh = s["handle_h"]
    m = s["mass"]
    parts = [box("blade", (L, W, T), (0, 0, T/2), 0.78*m, "magnesium", role="sole"),
             box("rib", (L*0.9, 0.010, 0.012), (0, 0, T + 0.006), 0.06*m, "magnesium"),
             box("post", (0.020, 0.020, hh - T - 0.012), (-0.10, 0, T + (hh - T - 0.012)/2), 0.02*m, "aluminum"),
             cap_x("handle", s["handle"][1], s["handle"][0], (-0.10, 0, hh), 0.08*m, "handle_soft"),
             # the knob 70 mm tall (estimate) and taken by its upper part: on a 40 mm knob the fingers that pinched it
             # reached into the concrete when the darby sank
             cyl_z("knob", 0.018, 0.070, (0.33, 0, T + 0.035), 0.06*m, "handle_soft")]
    return Tool(name, "float", parts, Sole("blade", L/2, W/2, lift=P.ROLL_LIFT["float"]), "blade",
                grasp=((-0.10, 0, hh), (1, 0, 0)), second_grasp=((0.33, 0, T + 0.058), (1, 0, 0)), mass=m,
                dens=0.2, seal=0.05, texture=4.0e-4, meta=dict(part=s["part"]))


def edger(name="edger"):
    s = P.TOOLS["edger"]
    L, W, T = s["blade"]
    hh = s["handle_h"]
    m = s["mass"]
    lip = s["lip"]
    parts = [box("blade", (L, W, T), (0, 0, T/2), 0.55*m, "edger_steel", role="sole"),
             box("lip", (L, 0.0015, lip), (0, W/2 + 0.00075, T - lip/2), 0.10*m, "edger_steel"),
             box("post", (0.020, 0.016, hh - 0.020), (0, 0, (hh - 0.020)/2 + T), 0.05*m, "aluminum"),
             cap_x("handle", s["handle"][1], s["handle"][0], (0, 0, hh), 0.30*m, "handle_soft")]
    return Tool(name, "edger", parts, Sole("edger", L/2, W/2 + 0.0015, radius=s["radius"], lip=lip, lip_side=1),
                "blade", grasp=((0, 0, hh), (1, 0, 0)), mass=m, dens=0.6, seal=0.2, texture=5.0e-5, no_plate=("lip",),
                meta=dict(part=s["part"], radius=s["radius"], lip=lip), bear=Sole("blade", L/2, W/2))


def margin_trowel(name="margin_trowel"):
    s = P.TOOLS["margin"]
    L, W, T = s["blade"]
    m = s["mass"]
    hl, hr = s["handle"]
    parts = [box("blade", (L, W, 0.002), (0, 0, 0.001), 0.55*m, "trowel_steel", role="sole"),
             box("tang", (0.040, 0.010, 0.004), (L/2 + 0.015, 0, 0.012), 0.10*m, "trowel_steel"),
             cap_x("handle", hr, hl, (L/2 + 0.035 + hl/2, 0, 0.026), 0.35*m, "handle_soft")]
    return Tool(name, "knife", parts, Sole("blade", L/2, W/2, lift=P.ROLL_LIFT["knife"]), "blade",
                grasp=((L/2 + 0.035 + hl/2, 0, 0.026), (1, 0, 0)), mass=m, carry=True, dens=0.5, seal=0.3,
                texture=1.0e-4, meta=dict(part=s["part"]), carry_cap=P.CARRY_CAP["margin_trowel"])


def jointer(name="jointer"):
    s = P.TOOLS["jointer"]
    d, L = s["rod"]
    r = d/2
    m = s["mass"]
    hl, hr = s["handle"]
    parts = [cap_x("rod", r, L, (0, 0, r), 0.50*m, "jointer_steel", role="sole"),
             box("bridge_a", (0.008, 0.008, 0.030), (-0.050, 0, 2*r + 0.015), 0.05*m, "jointer_steel"),
             box("bridge_b", (0.008, 0.008, 0.030), (0.050, 0, 2*r + 0.015), 0.05*m, "jointer_steel"),
             cap_x("handle", hr, hl, (0, 0, 2*r + 0.030 + hr), 0.40*m, "handle_wood")]
    return Tool(name, "rod", parts, Sole("rod", L/2 - r, r*0.98, radius=r), "rod",
                grasp=((0, 0, 2*r + 0.030 + hr), (1, 0, 0)), mass=m, dens=0.6, seal=0.0, texture=3.0e-5,
                meta=dict(part=s["part"], radius=r))


def sealant_tool(name="sealant_tool"):
    s = P.TOOLS["sealant_tool"]
    m = s["mass"]
    r = s["tip_r"]                       # the nose's radius across the blade: wider than the joint, it rides both edges
    hw = s["blade"][0]/2
    parts = [cap_x("tip", r, 0.040, (0, 0, r), 0.35*m, "nylon_white", role="sole"),
             cap_x("handle", s["handle"][1], s["handle"][0], (0.020 + s["handle"][0]/2, 0, r + 0.010), 0.65*m, "nylon_white")]
    return Tool(name, "rod", parts, Sole("tip", 0.020 - r, hw, radius=r), "tip",
                grasp=((0.020 + s["handle"][0]/2, 0, r + 0.010), (1, 0, 0)), mass=m, carry=True, dens=0.3,
                texture=2.0e-5, meta=dict(part=s["part"], radius=r), carry_cap=P.CARRY_CAP["sealant_tool"])


def knife(name, spec):
    s = P.TOOLS[spec]
    w, l, t = s["blade"]
    m = s["mass"]
    hl, hr = s["handle"]
    parts = [box("blade", (w, l, 0.001), (0, 0, 0.0005), 0.45*m, "blue_steel", role="sole"),
             box("ferrule", (0.030, 0.020, 0.012), (0, l/2 + 0.010, 0.006), 0.10*m, "aluminum"),
             cap_y("handle", hr, hl, (0, l/2 + 0.020 + hl/2, 0.010), 0.45*m, "handle_soft")]
    return Tool(name, "knife", parts, Sole("blade", w/2, l/2, lift=P.ROLL_LIFT["knife"]), "blade",
                grasp=((0, l/2 + 0.020 + hl/2, 0.010), (0, 1, 0)), mass=m, carry=True, dens=0.5, seal=0.3,
                texture=3.0e-5, meta=dict(part=s["part"]), carry_cap=P.CARRY_CAP[name])


def penetrometer(name="penetrometer"):
    s = P.TOOLS["penetrometer"]
    r, L = s["shaft"]
    m = s["mass"]
    bl, br = s["body"]
    parts = [cyl_z("needle", r, L, (0, 0, L/2), 0.10*m, "steel", role="sole"),
             cyl_z("body", br, bl, (0, 0, L + bl/2), 0.90*m, "penetrometer_body")]
    side = r*math.sqrt(math.pi)/2                                      # a square of the needle's area
    return Tool(name, "needle", parts, Sole("blade", side, side), "needle", grasp=((0, 0, L + 0.6*bl), (0, 0, 1)),
                mass=m, meta=dict(part=s["part"], area_m2=math.pi*r*r))


BUILDERS = dict(mag_float=hand_float, wood_float=wood_float, trowel=finishing_trowel, darby=darby, edger=edger,
                margin_trowel=margin_trowel, jointer=jointer, sealant_tool=sealant_tool,
                joint_knife=lambda: knife("joint_knife", "joint_knife"), putty_knife=lambda: knife("putty_knife", "putty_knife"),
                penetrometer=penetrometer)


def make(name):
    t = BUILDERS[name]()
    t.name = name
    return t


SIDE_RES = 0.001                 # m: grid of a side face's profile (the finest paste lattice is 1 mm)
_SIDES = {}


def _surface_points(p, step):
    """Points on a collision part's surface, in the tool frame."""
    R = np.asarray(_quat_matrix(p.quat))
    if p.kind == "box":
        h = np.asarray(p.size, float)
        pts = []
        for ax in range(3):
            a, b = [k for k in range(3) if k != ax]
            ga = np.linspace(-h[a], h[a], max(2, int(math.ceil(2*h[a]/step)) + 1))
            gb = np.linspace(-h[b], h[b], max(2, int(math.ceil(2*h[b]/step)) + 1))
            A, B = np.meshgrid(ga, gb, indexing="ij")
            for s in (-1.0, 1.0):
                q = np.zeros((A.size, 3))
                q[:, a], q[:, b], q[:, ax] = A.ravel(), B.ravel(), s*h[ax]
                pts.append(q)
        loc = np.concatenate(pts)
    else:
        r, hh = p.size
        cap = r if p.kind == "capsule" else 0.0
        zs = np.linspace(-hh - cap, hh + cap, max(2, int(math.ceil(2*(hh + cap)/step)) + 1))
        th = np.linspace(0.0, 2*math.pi, max(8, int(math.ceil(2*math.pi*r/step))), endpoint=False)
        Z, T = np.meshgrid(zs, th, indexing="ij")
        rr = np.sqrt(np.maximum(r*r - np.maximum(np.abs(Z) - hh, 0.0)**2, 0.0)) if cap else np.full(Z.shape, r)
        loc = np.stack([(rr*np.cos(T)).ravel(), (rr*np.sin(T)).ravel(), Z.ravel()], 1)
        if not cap:                                                   # a cylinder's flat ends
            rs = np.linspace(0.0, r, max(2, int(math.ceil(r/step)) + 1))
            Rr, Tt = np.meshgrid(rs, th, indexing="ij")
            disc = np.stack([(Rr*np.cos(Tt)).ravel(), (Rr*np.sin(Tt)).ravel()], 1)
            loc = np.concatenate([loc] + [np.column_stack([disc, np.full(len(disc), s*hh)]) for s in (-1.0, 1.0)])
    return loc @ R.T + np.asarray(p.pos, float)


def _quat_matrix(q):
    x, y, z, w = q
    return ((1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)),
            (2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)),
            (2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)))


def side_faces(t):
    """The tool's five other sides as paste soles, besides its working sole (which faces -z): its back (+z), its ends
    (+-x) and its long sides (+-y). Each is the face of the tool's bounding box on that side, in a frame whose z points
    from the face into the tool (the sole convention), with a profile: how far behind the face plane the tool's collision
    parts begin over each point of the face (0 where a part reaches the plane, nan where no part lies behind it). A float
    turned over rests on its handle, one on edge on its blade's edge (coupling: a tool's side contactor)."""
    if t.name in _SIDES:
        return _SIDES[t.name]
    pts = np.concatenate([_surface_points(p, 0.5*SIDE_RES) for p in t.parts if p.collide])
    eye = np.eye(3)
    out = []
    for name, d, xa in (("back", eye[2], eye[0]), ("end+x", eye[0], eye[1]), ("end-x", -eye[0], eye[1]),
                        ("side+y", eye[1], eye[0]), ("side-y", -eye[1], eye[0])):
        z = -d
        y = np.cross(z, xa)
        a, b, e = pts @ xa, pts @ y, pts @ d
        lo_a, hi_a, lo_b, hi_b, top = a.min(), a.max(), b.min(), b.max(), e.max()
        hx, hy = 0.5*(hi_a - lo_a), 0.5*(hi_b - lo_b)
        n, m = max(1, int(math.ceil(2*hx/SIDE_RES))), max(1, int(math.ceil(2*hy/SIDE_RES)))
        i = np.clip(((a - lo_a)/SIDE_RES).astype(int), 0, n - 1)
        j = np.clip(((b - lo_b)/SIDE_RES).astype(int), 0, m - 1)
        depth = np.full((n, m), np.inf)
        np.minimum.at(depth, (i, j), top - e)
        depth[~np.isfinite(depth)] = np.nan
        centre = (0.5*(lo_a + hi_a))*xa + (0.5*(lo_b + hi_b))*y + top*d
        out.append(dict(name=name, p_ls=centre, R_ls=np.column_stack([xa, y, z]),
                        sole=Sole("profile", hx, hy, grid=depth, res=SIDE_RES, lift=0.0)))
    _SIDES[t.name] = out
    return out


def summary(t):
    return dict(name=t.name, kind=t.kind, mass_kg=t.mass, sole=dict(kind=t.sole.kind, half_extents_m=[t.sole.hx, t.sole.hy]),
                grasp=t.grasp, part=t.meta.get("part"), carry=t.carry)
