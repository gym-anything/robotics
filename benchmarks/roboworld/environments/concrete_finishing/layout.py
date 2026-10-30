"""Mechanical room B-012 during a pump replacement: installed parts, stations, carts, tools, robot start, and the start
state of every task's wet material.

Frame: x east, y north, z up, origin on the floor at the south-west corner of the room's interior. Walls stand outside
the interior box [0, 6.10] x [0, 4.88]: CMU to the north, east and west, a gypsum partition with the door to the south.

Start states are what a crew leaves: a struck-off pad with screed ridges, a low corner where the board bridged and a
high band where it rode on a stone; a patch overfilled with a void left at one edge; block joints struck flush with
crumbs on the faces and one head joint short; a gunned sealant bead with gun stops and a thin stretch; a spackled
mesh patch with knife ridges and anchor holes still open. Ages are given for 21 C and scale with the seed's
temperature (MASCO factor) so that each start is the same stage of the work.
"""

import math
import numpy as np
from . import params as P
from .materials import Concrete, RepairMortar, MasonryMortar, Sealant, Compound, masco_factor
from .paste import Region

RX, RY, RZ = P.ROOM
TASKS = ("pad-darby", "pad-edge", "pad-float", "pad-trowel", "pad-finish", "patch-flush", "tool-joints", "tool-sealant",
         "patch-drywall")
PRESETS = {"b012": TASKS}
VARIANTS = ("default", "door", "fan", "hot", "cool", "fast-sealant")
AIRS = tuple(P.AIR)

TUB_CENTRE = (4.60, 4.24)      # the mixing tub with the leftover concrete, in the room's north-east corner beside the pad
WORKLIGHT = (1.10, 4.40)       # the work light's tripod, in the north-west corner, aimed across the pad

# pad interior
PX0 = P.PAD_CENTRE[0] - P.PAD_SIZE[0]/2
PX1 = P.PAD_CENTRE[0] + P.PAD_SIZE[0]/2
PY0 = P.PAD_CENTRE[1] - P.PAD_SIZE[1]/2
PY1 = P.PAD_CENTRE[1] + P.PAD_SIZE[1]/2
FW, FH = P.FORM


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw/2), math.cos(yaw/2))


# ------------------------------------------------------------------ installed parts (static boxes: name, centre, half extents, material)
def statics(variant="default"):
    t = P.WALL_T
    out = [
        ("wall_north", (RX/2, RY + t/2, RZ/2), (RX/2 + t, t/2, RZ/2), "cmu"),
        ("wall_west", (-t/2, RY/2, RZ/2), (t/2, RY/2, RZ/2), "cmu"),
        ("deck", (RX/2, RY/2, RZ + 0.10), (RX/2 + t, RY/2 + t, 0.10), "deck"),
    ]
    # south partition with the door opening (x from DOOR[0] to DOOR[1], 2.13 m high)
    pt = P.PARTITION_T
    d0, d1, dh = P.DOOR
    out += [("partition_w", (d0/2, -pt/2, RZ/2), (d0/2, pt/2, RZ/2), "gypsum"),
            ("partition_e", ((d1 + RX)/2, -pt/2, RZ/2), ((RX - d1)/2, pt/2, RZ/2), "gypsum"),
            ("partition_head", ((d0 + d1)/2, -pt/2, (dh + RZ)/2), ((d1 - d0)/2, pt/2, (RZ - dh)/2), "gypsum")]
    if variant != "door":           # the door is shut (its leaf closes the opening); open, it swings out into the corridor
        out.append(("door_leaf", ((d0 + d1)/2, -0.025, dh/2), ((d1 - d0)/2 - 0.004, 0.022, dh/2 - 0.005), "door"))
    # east wall with the control joint (a 13 mm gap at y = CJ_Y, the backer rod CJ_BACKER behind the face)
    cj, w = P.CJ_Y, P.CJ_W
    out += [("wall_east_s", (RX + t/2, (cj - w/2)/2, RZ/2), (t/2, (cj - w/2)/2, RZ/2), "cmu"),
            ("wall_east_n", (RX + t/2, (cj + w/2 + RY)/2, RZ/2), (t/2, (RY - cj - w/2)/2, RZ/2), "cmu"),
            ("backer_rod", (RX + P.CJ_BACKER + 0.005, cj, RZ/2), (0.005, w/2, RZ/2), "backer")]
    # north wall infill: blocks with recessed joints in the old opening (the wall box above has a hole cut by these
    # parts: the infill region is drawn over it; the rigid faces are the blocks)
    out += infill_blocks()
    # forms of the new pad: 2x4 on edge outside the pad's interior, and kicker blocks
    fy0, fy1 = PY0 - FW/2, PY1 + FW/2
    fx0, fx1 = PX0 - FW/2, PX1 + FW/2
    out += [("form_s", (P.PAD_CENTRE[0], fy0, FH/2), (P.PAD_SIZE[0]/2 + FW, FW/2, FH/2), "form_wood"),
            ("form_n", (P.PAD_CENTRE[0], fy1, FH/2), (P.PAD_SIZE[0]/2 + FW, FW/2, FH/2), "form_wood"),
            ("form_w", (fx0, P.PAD_CENTRE[1], FH/2), (FW/2, P.PAD_SIZE[1]/2, FH/2), "form_wood"),
            ("form_e", (fx1, P.PAD_CENTRE[1], FH/2), (FW/2, P.PAD_SIZE[1]/2, FH/2), "form_wood")]
    kl, kw, kh, ks = P.KICKER
    for k, x in enumerate(np.arange(PX0 + 0.15, PX1, ks)):
        out.append((f"kicker_s{k}", (x, PY0 - FW - kw/2, kh/2), (kl/2, kw/2, kh/2), "form_wood"))
        out.append((f"kicker_n{k}", (x, PY1 + FW + kw/2, kh/2), (kl/2, kw/2, kh/2), "form_wood"))
    for k, y in enumerate((PY0 + 0.25, PY1 - 0.25)):
        out.append((f"kicker_w{k}", (PX0 - FW - kw/2, y, kh/2), (kw/2, kl/2, kh/2), "form_wood"))
        out.append((f"kicker_e{k}", (PX1 + FW + kw/2, y, kh/2), (kw/2, kl/2, kh/2), "form_wood"))
    # the existing pumps on their hardened pads along the west wall
    for k, y in enumerate((1.30, 3.30)):
        out += [(f"old_pad{k}", (0.65, y, 0.05), (0.45, 0.30, 0.05), "old_concrete"),
                (f"pump_base{k}", (0.65, y, 0.115), (0.40, 0.20, 0.015), "pump_red"),
                (f"pump_motor{k}", (0.85, y, 0.28), (0.18, 0.13, 0.15), "pump_motor"),
                (f"pump_volute{k}", (0.40, y, 0.26), (0.11, 0.13, 0.13), "pump_red")]
    # the work light's tripod (its mast and three feet; the legs between them are drawn, not colliders)
    wx, wy = WORKLIGHT
    out.append(("worklight_mast", (wx, wy, 0.82), (0.02, 0.02, 0.82), "tripod"))
    for k in range(3):
        a = 2*math.pi*k/3 + 0.3
        out.append((f"worklight_foot{k}", (wx + 0.45*math.cos(a), wy + 0.45*math.sin(a), 0.03), (0.03, 0.03, 0.03), "tripod"))
    # the new pump, crated on its pallet by the door (clear of the floor patch: at (5.40, 0.70) the pallet's corner
    # touched the patch's east end and came 25 mm from its south side, and the float run onto the slab there hit it)
    out += [("crate_pallet", (5.45, 0.55, 0.07), (0.60, 0.40, 0.07), "pallet"),
            ("crate", (5.45, 0.55, 0.415), (0.55, 0.30, 0.275), "crate")]
    return out


def infill_blocks():
    """The infill's blocks as static boxes: three a course in running bond (half blocks at the ends of courses 2 and
    4), 3/8 in joints, faces flush with the old wall's face (y = RY); the joints are recessed JOINT_BED_DEPTH."""
    bl, bt, bh = P.BLOCK
    j = P.JOINT_W
    x0, x1 = P.INFILL_X
    out = []
    for c in range(P.INFILL_COURSES):
        z0 = P.INFILL_Z0 + j + c*(bh + j)
        if c % 2 == 0:
            xs = [(x0 + j + k*(bl + j), bl) for k in range(3)]
        else:
            half = (bl - j)/2
            xs = [(x0 + j, half)] + [(x0 + j + half + j + k*(bl + j), bl) for k in range(2)] + \
                 [(x0 + j + half + j + 2*(bl + j), x1 - j - (x0 + j + half + j + 2*(bl + j)))]
        for k, (xa, w) in enumerate(xs):
            out.append((f"block_c{c}_{k}", (xa + w/2, RY + bt/2, z0 + bh/2), (w/2, bt/2, bh/2), "cmu_new"))
    # the joint bed behind the joints (mortar lies on it)
    out.append(("infill_bed", ((x0 + x1)/2, RY + P.JOINT_BED_DEPTH + 0.01, P.INFILL_Z0 + P.INFILL_COURSES*(bh + j)/2 + j/2),
                ((x1 - x0)/2, 0.01, (P.INFILL_COURSES*(bh + j) + j)/2), "mortar_bed"))
    # the old wall around the opening (the north wall box stops short of it: wall_north is split in statics_north)
    return out


def north_wall_split():
    """The north wall without the opening: four boxes around the infill."""
    t = P.WALL_T
    x0, x1 = P.INFILL_X
    z0 = P.INFILL_Z0
    z1 = z0 + P.INFILL_COURSES*(P.BLOCK[2] + P.JOINT_W) + P.JOINT_W
    y = RY + t/2
    return [("wall_north_w", (x0/2 - t/2, y, RZ/2), (x0/2 + t/2, t/2, RZ/2), "cmu"),
            ("wall_north_e", ((x1 + RX + t)/2, y, RZ/2), ((RX + t - x1)/2, t/2, RZ/2), "cmu"),
            ("wall_north_low", ((x0 + x1)/2, y, z0/2), ((x1 - x0)/2, t/2, z0/2), "cmu"),
            ("wall_north_high", ((x0 + x1)/2, y, (z1 + RZ)/2), ((x1 - x0)/2, t/2, (RZ - z1)/2), "cmu")]


def all_statics(variant="default"):
    s = [x for x in statics(variant) if x[0] != "wall_north"]
    return s + north_wall_split()


# ------------------------------------------------------------------ carts and where each task puts things
CART_TOP = P.CART[2]
TASK_SETUP = {
    # robot start (x, y, yaw), cart centre (x, y, yaw), tools on the cart (name, x, y, yaw on the cart top in the
    # cart frame), extra regions. The robot starts with its arms in the FR3 ready pose, hands 0.9-1.0 m ahead of the
    # base origin: at least 1.2 m from the wall it faces (closer, the hands start inside the wall and the reset settle
    # stops on a protective stop); it drives in to work
    "pad-darby": dict(robot=(3.20, 1.85, math.pi/2), cart=(4.35, 1.95, 0.0),
                      # the darby along the cart's near (south) edge, where both hands reach it from a clear stand
                      # (at the far edge its handle and knob were 1.0 m ahead of the nearest clear stand)
                      tools=[("darby", 0.0, -0.12, math.pi), ("mag_float", -0.18, 0.10, 0.0), ("margin_trowel", 0.25, 0.10, 0.0)],
                      tub=True),
    "pad-edge": dict(robot=(3.20, 1.85, math.pi/2), cart=(4.35, 1.95, 0.0),
                     tools=[("edger", -0.20, 0.08, 0.0), ("margin_trowel", 0.12, 0.10, 0.0), ("penetrometer", 0.25, -0.19, 0.0)]),
    "pad-float": dict(robot=(3.20, 1.85, math.pi/2), cart=(4.35, 1.95, 0.0),
                      tools=[("mag_float", -0.10, 0.08, 0.0), ("edger", 0.25, 0.10, 0.0), ("wood_float", -0.10, -0.12, 0.0)]),
    "pad-trowel": dict(robot=(3.20, 1.85, math.pi/2), cart=(4.35, 1.95, 0.0),
                       tools=[("trowel", -0.12, 0.09, 0.0), ("edger", 0.25, 0.10, 0.0), ("penetrometer", 0.25, -0.19, 0.0)]),
    "pad-finish": dict(robot=(3.20, 1.85, math.pi/2), cart=(4.35, 1.95, 0.0),
                       tools=[("darby", 0.0, -0.14, math.pi), ("mag_float", -0.20, 0.05, 0.0), ("trowel", 0.215, 0.05, 0.0),
                              ("edger", -0.30, 0.17, 0.0), ("margin_trowel", 0.30, 0.17, 0.0)], tub=True),
    "patch-flush": dict(robot=(3.75, 1.25, 0.0), cart=(3.45, 0.45, 0.0),
                        tools=[("margin_trowel", 0.14, 0.12, 0.0), ("mag_float", -0.18, 0.12, 0.0), ("wood_float", -0.18, -0.10, 0.0)]),
    "tool-joints": dict(robot=(4.50, 3.63, math.pi/2), cart=(5.65, 3.55, math.pi/2),
                        tools=[("jointer", -0.12, 0.05, 0.0), ("margin_trowel", 0.12, 0.05, 0.0)], board=True),
    "tool-sealant": dict(robot=(4.85, 1.75, 0.0), cart=(4.45, 2.85, 0.0),
                         tools=[("sealant_tool", 0.0, 0.0, 0.0)], rag=(-0.24, 0.06)),
    "patch-drywall": dict(robot=(1.82, 1.25, -math.pi/2), cart=(2.90, 0.55, 0.0),
                          # the knives clear of the hawk (local x -0.415..-0.085): at (-0.10, 0.04) the joint knife lay
                          # over the hawk's edge and the hand that took it touched the hawk
                          tools=[("joint_knife", 0.08, -0.08, 0.0), ("putty_knife", 0.30, -0.08, 0.0)], hawk=True),
}


def cart_parts():
    """A three-shelf utility cart (static: casters locked) as boxes in its own frame: origin on the floor under its
    centre; top shelf at CART_TOP."""
    L, W, H, m = P.CART
    out = [("top", (0, 0, H - 0.015), (L/2, W/2, 0.015)), ("lip_n", (0, W/2 - 0.01, H + 0.012), (L/2, 0.01, 0.012)),
           ("lip_s", (0, -W/2 + 0.01, H + 0.012), (L/2, 0.01, 0.012)), ("shelf_mid", (0, 0, 0.48), (L/2, W/2, 0.015)),
           ("shelf_low", (0, 0, 0.14), (L/2, W/2, 0.015))]
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append((f"post_{sx}{sy}", (sx*(L/2 - 0.025), sy*(W/2 - 0.025), (H - 0.03)/2 + 0.05), (0.02, 0.02, (H - 0.03)/2)))
            out.append((f"caster_{sx}{sy}", (sx*(L/2 - 0.04), sy*(W/2 - 0.04), 0.05), (0.03, 0.03, 0.05)))
    return out


def cart_pose(task):
    return TASK_SETUP[task]["cart"]


def tool_poses(task):
    """World pose (position of the tool frame = sole centre, yaw) of each tool on the cart top."""
    cx, cy, cyaw = TASK_SETUP[task]["cart"]
    c, s = math.cos(cyaw), math.sin(cyaw)
    out = {}
    for name, x, y, yaw in TASK_SETUP[task]["tools"]:
        if name == "penetrometer":
            # a pocket penetrometer lies on its side (it cannot stand on its 6.4 mm needle), against the cart's lip
            from scipy.spatial.transform import Rotation
            q = (Rotation.from_euler("z", cyaw + yaw)*Rotation.from_euler("y", -math.pi/2)).as_quat()
            out[name] = ((cx + c*x - s*y, cy + s*x + c*y, CART_TOP + P.TOOLS["penetrometer"]["body"][1] + 0.0005), tuple(q))
            continue
        out[name] = ((cx + c*x - s*y, cy + s*x + c*y, CART_TOP + 0.0015), cyaw + yaw)
    return out


# ------------------------------------------------------------------ start states of the wet materials
AGES_21C = {"pad-darby": 5*60.0, "pad-edge": 60*60.0, "pad-float": 150*60.0, "pad-trowel": 165*60.0,
            "pad-finish": 5*60.0, "patch-flush": 6*60.0, "tool-sealant": 10*60.0, "patch-drywall": 4*60.0}


def _noise(rng, shape, cells):
    """Gaussian noise smoothed over about `cells` cells, scaled to unit standard deviation. Each pass is the 3x3
    binomial kernel (the cell itself weighted in: averaging only the four neighbours never damps the checkerboard mode,
    and leaves white noise with a 1-sigma jump between neighbouring cells)."""
    k = rng.normal(0, 1, shape)
    for _ in range(max(1, int(cells))**2):
        k = 0.5*k + 0.125*(np.roll(k, 1, 0) + np.roll(k, -1, 0) + np.roll(k, 1, 1) + np.roll(k, -1, 1))
    return k/max(float(k.std()), 1e-9)


def pad_region(task, seed, temp_c, air, bleed):
    """The pad's surface on a 2 mm grid from 40 mm outside the forms' inner faces (the form tops are reference
    surfaces: concrete on them is smear)."""
    rng = np.random.default_rng(1000 + seed)
    cell = 0.002
    o = (PX0 - FW - 0.002, PY0 - FW - 0.002, 0.0)
    nu = int(round((P.PAD_SIZE[0] + 2*FW + 0.004)/cell))
    nv = int(round((P.PAD_SIZE[1] + 2*FW + 0.004)/cell))
    mat = Concrete(temp_c, bleed_mean=bleed)
    reg = Region("pad", mat, o, (1, 0, 0), (0, 1, 0), cell, (nu, nv), temp_c=temp_c, air=air, texture_kind=1)
    X = o[0] + reg.u[:, None] + 0*reg.v[None, :]
    Y = o[1] + 0*reg.u[:, None] + reg.v[None, :]
    inside = (X > PX0) & (X < PX1) & (Y > PY0) & (Y < PY1)
    form = ~inside
    reg.paste_mask = inside
    reg.ref_mask = form
    reg.b[:] = np.where(form, FH, 0.0)
    s = masco_factor(temp_c)
    age = AGES_21C[task]*s
    reg.t0[:] = -age
    # struck-off surface: ridges +-2 mm on a 60 mm wavelength, wavy across the pad, a low corner where the board
    # bridged, a high band where it rode on a stone, 0.5 mm of correlated roughness
    u = X - PX0
    v = Y - PY0
    lo_c = (0.15 + 0.10*rng.random(), 0.15 + 0.10*rng.random()) if seed % 2 == 0 else (P.PAD_SIZE[0] - 0.2, 0.18)
    hi_c = (0.55 + 0.25*rng.random(), 0.45 + 0.25*rng.random())
    ridge = 0.002*np.sin(2*math.pi*u/0.060 + 0.8*np.sin(2*math.pi*v/0.35) + rng.random()*6.28)
    low = -0.005*np.exp(-((u - lo_c[0])**2 + (v - lo_c[1])**2)/(2*0.08**2))
    high = 0.004*np.exp(-((u - hi_c[0])**2 + (v - hi_c[1])**2)/(2*0.05**2))
    rough = 0.0005*_noise(rng, reg.b.shape, 2)
    struck = ridge + low + high + rough
    top = FH + struck
    ts = P.T_SURF_STRUCK
    reg.ts[:] = np.where(inside, ts, 0.0)
    reg.tb[:] = np.where(inside, np.maximum(top - ts, 0.0), 0.0)
    reg.E[:] = np.where(inside, np.clip(P.AGG_EXPOSED*(1 + 0.4*_noise(rng, reg.b.shape, 3)), 0, 1), 0.0)
    reg.a[:] = np.where(inside, 8.0e-4, 0.0)
    reg.th[:] = 0.0
    # smear on two form tops (concrete the board pushed over them): a band 1-3 mm thick, thinning to its ends
    for side in ("s", "w"):
        band = ((Y < PY0) & (Y > PY0 - FW) & (X > PX0 + 0.2) & (X < PX0 + 0.75)) if side == "s" else \
               ((X < PX0) & (X > PX0 - FW) & (Y > PY0 + 0.3) & (Y < PY0 + 0.6))
        along = (X - (PX0 + 0.475))/0.275 if side == "s" else (Y - (PY0 + 0.45))/0.15
        taper = np.clip(1.0 - np.abs(along), 0.0, 1.0)**0.5
        reg.tb[:] = np.where(band, (0.0015 + 0.0008*_noise(rng, reg.b.shape, 8).clip(-1.5, 1.5))*taper, reg.tb)
    np.maximum(reg.tb, 0.0, out=reg.tb)
    stage = dict(task=task)
    if task in ("pad-edge", "pad-float", "pad-trowel"):
        _darbied(reg, inside, rng)
        stage["darbied"] = True
    if task in ("pad-float", "pad-trowel"):
        _edged(reg, inside, X, Y)
        stage["edged"] = True
    if task == "pad-trowel":
        _floated(reg, inside, rng, X, Y)
        stage["floated"] = True
    # the bleed film and trapped water as the material clock leaves them at this age (no tool has sealed it yet)
    _bleed_history(reg, age)
    reg.t0[:] = -age
    reg.clock = 0.0
    reg.volume0 = reg.volume()
    reg.stage = stage
    return reg


def _darbied(reg, inside, rng):
    top = P.FORM[1] + 0.0003*_noise(rng, reg.b.shape, 8) + 0.0001*_noise(rng, reg.b.shape, 1)   # flat within about +-1.3 mm
    ts = 0.002
    reg.ts[:] = np.where(inside, ts, 0.0)
    reg.tb[:] = np.where(inside, np.maximum(top - ts, 0.0), reg.tb)
    reg.E[:] = np.where(inside, 0.01, 0.0)
    reg.a[:] = np.where(inside, 4.0e-4, 0.0)
    reg.kind[:] = np.where(inside, 2, reg.kind)
    reg.th[:] = 0.0


def _edged(reg, inside, X, Y):
    """The edger's work: a 3/8 in arris along all four forms, the lip's cut, and the edger's 3 in track pressed
    EDGER_TRACK below the field, a step at its inner edge that the floats take out."""
    r = P.TOOLS["edger"]["radius"]
    d = np.minimum.reduce([X - PX0, PX1 - X, Y - PY0, PY1 - Y])
    top = reg.b + reg.tb + reg.ts
    drop = np.where((d >= 0) & (d < r), r - np.sqrt(np.maximum(r*r - (r - d)**2, 0.0)), 0.0)
    track = (d >= 0) & (d < 0.0762)
    step = np.where(track, -P.EDGER_TRACK, 0.0)
    new_top = np.where(inside, top - drop + step, top)
    reg.tb[:] = np.where(inside, np.maximum(new_top - reg.b - reg.ts, 0.0), reg.tb)
    reg.compact[:] = np.where(inside & (d < 0.0025), 1.0, reg.compact)          # lip passed: the first cell off the form
    reg.a[:] = np.where(track & inside, 5.0e-5, reg.a)
    reg.kind[:] = np.where(track & inside, 4, reg.kind)


def _floated(reg, inside, rng, X, Y):
    """The floats' work: the surface where it was, less the edger's track step (floated out), with a float's sandy
    texture, the paste layer brought up to its full depth and the first densification."""
    h = reg.b + reg.tb + reg.ts
    d = np.minimum.reduce([X - PX0, PX1 - X, Y - PY0, PY1 - Y])
    h = np.where(inside & (d >= 0) & (d < 0.0762), h + P.EDGER_TRACK, h)         # the track's step, floated out
    reg.a[:] = np.where(inside, 2.5e-4, 0.0)
    reg.kind[:] = np.where(inside, 3, reg.kind)
    reg.ts[:] = np.where(inside, P.T_SURF_MAX, 0.0)
    reg.D[:] = np.where(inside, 0.10, 0.0)
    reg.tb[:] = np.where(inside, np.maximum(h - reg.b - reg.ts, 0.0), reg.tb)


def _bleed_history(reg, age):
    """Run the bleed / evaporation clock from strike-off to `age` (10 s steps) on the start surface."""
    m = reg.material
    if not getattr(m, "bleeds", False):
        return
    clock, dt = -age, 10.0
    reg.w[:] = 0.0
    while clock < -1e-9:
        step = min(dt, -clock)
        a = clock + step/2 + age
        r = float(m.bleed_rate(a))*reg.paste_mask
        reg.w += (step*r/1000.0).astype(np.float32)
        reg.w -= np.float32(step*reg.evap/3600.0/1000.0)
        np.maximum(reg.w, 0.0, out=reg.w)
        clock += step


def tub_region(age, temp_c, seed, bleed):
    """Leftover concrete in the mixing tub beside the pad (north-east): a heap of about 6 L in a 0.9 x 0.6 m tub."""
    rng = np.random.default_rng(2000 + seed)
    cx, cy = TUB_CENTRE
    cell = 0.004
    L, W, H = P.TUB
    o = (cx - L/2, cy - W/2, 0.012)
    nu, nv = int(round(L/cell)), int(round(W/cell))
    reg = Region("tub", Concrete(temp_c, bleed_mean=bleed), o, (1, 0, 0), (0, 1, 0), cell, (nu, nv), temp_c=temp_c,
                 air="still", texture_kind=1)
    u = reg.u[:, None] - L/2 + 0*reg.v[None, :]
    v = 0*reg.u[:, None] + reg.v[None, :] - W/2
    heap = np.clip(0.065 - np.sqrt((u/1.2)**2 + v**2)*0.25, 0.0, None)
    heap += np.where(heap > 0.003, 0.002*_noise(rng, heap.shape, 3), 0.0)
    reg.tb[:] = np.maximum(heap, 0.0)
    reg.ts[:] = 0.0
    reg.E[:] = 0.3
    reg.a[:] = 1.0e-3
    reg.t0[:] = -age
    reg.volume0 = reg.volume()
    return reg


def patch_region(seed, temp_c):
    """The floor patch and the slab around it on a 1 mm grid, PATCH_MARGIN beyond the cut. The start state is drawn on
    the 50 mm-margin grid it was authored on (its noise depends on the grid's shape) and set in the middle of the
    wider one: the same patch, cell for cell."""
    rng = np.random.default_rng(3000 + seed)
    cx, cy = P.PATCH_CENTRE
    L, W = P.PATCH_SIZE
    cell = 0.001
    m0, m = 0.05, P.PATCH_MARGIN
    k = int(round((m - m0)/cell))
    o = (cx - L/2 - m, cy - W/2 - m, 0.0)
    nu, nv = int(round((L + 2*m)/cell)), int(round((W + 2*m)/cell))
    reg = Region("patch", RepairMortar(temp_c), o, (1, 0, 0), (0, 1, 0), cell, (nu, nv), temp_c=temp_c, air="still",
                 texture_kind=1)
    X = o[0] + reg.u[:, None] + 0*reg.v[None, :]
    Y = o[1] + 0*reg.u[:, None] + reg.v[None, :]
    recess = (np.abs(X - cx) < L/2) & (np.abs(Y - cy) < W/2)
    shape0 = (int(round((L + 2*m0)/cell)), int(round((W + 2*m0)/cell)))
    inner = (slice(k, k + shape0[0]), slice(k, k + shape0[1]))

    def drawn(field):
        """A noise field drawn on the authored grid, zero (unused) outside it."""
        out = np.zeros((nu, nv))
        out[inner] = field
        return out

    d0, d1 = P.PATCH_DEPTH
    depth = d0 + (d1 - d0)*(0.5 + 0.35*drawn(_noise(rng, shape0, 12)).clip(-1.4, 1.4))
    reg.b[:] = np.where(recess, -depth, 0.0)
    reg.paste_mask = recess.copy()
    reg.ref_mask = ~recess
    over = 0.003 + 0.004*(0.5 + 0.5*drawn(_noise(rng, shape0, 10)).clip(-1, 1))
    top = np.where(recess, over, 0.0)
    # the void left along the west short edge: mortar not forced into the corner (top 6 mm below the slab, 18 mm wide)
    void = recess & (X < cx - L/2 + 0.018)
    top = np.where(void, -0.006 - 0.003*(X - (cx - L/2))/0.018, top)
    reg.tb[:] = np.where(recess, np.maximum(top - reg.b, 0.0), 0.0)
    # smear on the slab beside the north and east sides (2 mm, 25 mm wide)
    smear = (~recess) & (((Y > cy + W/2) & (Y < cy + W/2 + 0.025) & (np.abs(X - cx) < L/2 - 0.05)) |
                         ((X > cx + L/2) & (X < cx + L/2 + 0.025) & (np.abs(Y - cy) < W/2 - 0.03)))
    reg.tb[:] = np.where(smear, 0.002*(0.6 + 0.4*drawn(_noise(rng, shape0, 3)).clip(-1, 1)), reg.tb)
    np.maximum(reg.tb, 0.0, out=reg.tb)
    reg.a[:] = np.where(recess | smear, 6.0e-4, 0.0)
    reg.E[:] = 0.0
    age = AGES_21C["patch-flush"]*masco_factor(temp_c)
    reg.t0[:] = -age
    reg.volume0 = reg.volume()
    reg.stage = dict(task="patch-flush", recess=dict(centre=(cx, cy), size=(L, W)))
    return reg


def infill_region(seed, temp_c):
    """The block infill's face on a 1.5 mm grid; N points into the room (-y)."""
    rng = np.random.default_rng(4000 + seed)
    x0, x1 = P.INFILL_X
    j = P.JOINT_W
    bl, bt, bh = P.BLOCK
    z0 = P.INFILL_Z0
    z1 = z0 + P.INFILL_COURSES*(bh + j) + j
    cell = 0.0015
    m = 0.03
    o = (x0 - m, RY, z0 - m)
    nu, nv = int(round((x1 - x0 + 2*m)/cell)), int(round((z1 - z0 + 2*m)/cell))
    reg = Region("infill", MasonryMortar(temp_c), o, (1, 0, 0), (0, 0, 1), cell, (nu, nv), temp_c=temp_c, air="still",
                 texture_kind=5)
    X = o[0] + reg.u[:, None] + 0*reg.v[None, :]
    Z = o[2] + 0*reg.u[:, None] + reg.v[None, :]
    joint = np.zeros(reg.b.shape, bool)
    course_of = np.full(reg.b.shape, -1, int)
    inside = (X > x0) & (X < x1) & (Z > z0) & (Z < z1)
    for c in range(P.INFILL_COURSES):
        zb = z0 + c*(bh + j)
        bed = inside & (Z >= zb) & (Z < zb + j)
        joint |= bed
        course_of = np.where(bed, c, course_of)
        zc0, zc1 = zb + j, zb + j + bh
        row = inside & (Z >= zc0) & (Z < zc1)
        heads = [x0, x1 - j]
        if c % 2 == 0:
            heads += [x0 + j + bl + k*(bl + j) for k in range(2)]
        else:
            half = (bl - j)/2
            heads += [x0 + j + half, x0 + j + half + j + bl, x0 + j + half + j + 2*bl + j]
        for hx in heads:
            hj = row & (X >= hx) & (X < hx + j)
            joint |= hj
            course_of = np.where(hj, c, course_of)
    top_bed = inside & (Z >= z1 - j)
    joint |= top_bed
    course_of = np.where(top_bed, P.INFILL_COURSES - 1, course_of)
    reg.paste_mask = joint
    reg.ref_mask = ~joint
    reg.b[:] = np.where(joint, -P.JOINT_BED_DEPTH, 0.0)
    flush = 0.0004*_noise(rng, reg.b.shape, 2)
    reg.tb[:] = np.where(joint, P.JOINT_BED_DEPTH + flush, 0.0)
    # one head joint 3 mm short (a void), and crumbs of squeeze-out on the faces near the joints
    heads_mask = joint & (course_of == 1)
    cand = np.argwhere(heads_mask & (np.abs(X - (x0 + j + (bl - j)/2 + j + bl + j/2)) < j))
    if len(cand):
        void = heads_mask & (np.abs(X - (x0 + j + (bl - j)/2 + j + bl + j/2)) < j) & (Z > z0 + 1*(bh + j) + j + 0.06) & \
               (Z < z0 + 1*(bh + j) + j + 0.12)
        reg.tb[:] = np.where(void, P.JOINT_BED_DEPTH - 0.003, reg.tb)
    near = (~joint) & inside
    dist = np.zeros(reg.b.shape)
    crumbs = near & (rng.random(reg.b.shape) < 0.004)
    crumbs = np.maximum.reduce([np.roll(np.roll(crumbs, a, 0), b2, 1) for a in (-1, 0, 1) for b2 in (-1, 0, 1)])
    jn = np.maximum.reduce([np.roll(np.roll(joint, a, 0), b2, 1) for a in range(-6, 7, 3) for b2 in range(-6, 7, 3)])
    crumbs &= jn & ~joint
    reg.tb[:] = np.where(crumbs, 0.0015, reg.tb)
    reg.a[:] = np.where(joint, 5.0e-4, 0.0)
    ages = np.array([45.0, 37.0, 28.0, 20.0])*60*masco_factor(temp_c)
    age = np.where(course_of >= 0, ages[np.clip(course_of, 0, 3)], ages[0])
    reg.t0[:] = -age
    reg.course = course_of
    reg.volume0 = reg.volume()
    reg.stage = dict(task="tool-joints", ages_min=(ages/60).round(1).tolist())
    return reg


def joint_region(seed, temp_c, skin_s=None):
    """The control joint's sealant on a 1 mm grid: u runs along -y, v up, N into the room (-x)."""
    rng = np.random.default_rng(5000 + seed)
    cj, w = P.CJ_Y, P.CJ_W
    za, zb = P.CJ_SEGMENT
    cell = 0.001
    half_u = 0.030
    o = (RX, cj + half_u, za - 0.05)
    nu, nv = int(round(2*half_u/cell)), int(round((zb - za + 0.10)/cell))
    reg = Region("joint", Sealant(temp_c, skin_s), o, (0, -1, 0), (0, 0, 1), cell, (nu, nv), temp_c=temp_c, air="still",
                 texture_kind=6)
    Yw = cj + half_u - reg.u[:, None] + 0*reg.v[None, :]
    Z = o[2] + 0*reg.u[:, None] + reg.v[None, :]
    joint = np.abs(Yw - cj) < w/2
    reg.paste_mask = joint
    reg.ref_mask = ~joint
    reg.b[:] = np.where(joint, -P.CJ_BACKER, 0.0)
    across = (Yw - cj)/(w/2)
    bead = 0.0015 + 0.0012*(1 - across**2)                             # a convex bead 1.5-2.7 mm proud
    stops = np.zeros(reg.b.shape)
    for zc in (za + 0.35 + 0.1*rng.random(), za + 1.05 + 0.1*rng.random()):
        stops += 0.003*np.exp(-((Z - zc)/0.012)**2)
    top = bead + stops + 0.0003*_noise(rng, reg.b.shape, 2)
    # a thin stretch 40 mm long on the joint's north side: the bead sagged there and does not touch that wall
    thin = joint & (Z > za + 0.70) & (Z < za + 0.74) & (Yw > cj)
    top = np.where(thin, -0.004, top)
    reg.tb[:] = np.where(joint, np.maximum(top - reg.b, 0.0), 0.0)
    edge_l = joint & (np.abs(Yw - (cj - w/2)) < 0.0015)
    edge_r = joint & (np.abs(Yw - (cj + w/2)) < 0.0015)
    gaps = rng.random(reg.b.shape) < 0.25
    reg.contact_l[:] = np.where(edge_l & ~gaps, 1.0, 0.0)
    reg.contact_r[:] = np.where(edge_r & ~gaps & ~thin, 1.0, 0.0)
    reg.edge_l, reg.edge_r = edge_l, edge_r
    reg.a[:] = np.where(joint, 3.0e-4, 0.0)
    age = AGES_21C["tool-sealant"]*masco_factor(temp_c)
    reg.t0[:] = -age
    reg.volume0 = reg.volume()
    reg.stage = dict(task="tool-sealant", segment=(za, zb))
    return reg


def drywall_region(seed, temp_c):
    """The partition's face around the patch on a 1 mm grid: u runs along -x, v up, N into the room (+y)."""
    rng = np.random.default_rng(6000 + seed)
    cx, cz, hole_d = P.DW_PATCH
    xa, xb, za, zb = 1.40, 2.24, 0.96, 1.56
    cell = 0.001
    o = (xb, 0.0, za)
    nu, nv = int(round((xb - xa)/cell)), int(round((zb - za)/cell))
    reg = Region("wallboard", Compound(temp_c), o, (-1, 0, 0), (0, 0, 1), cell, (nu, nv), temp_c=temp_c, air="still",
                 texture_kind=7)
    X = xb - reg.u[:, None] + 0*reg.v[None, :]
    Z = za + 0*reg.u[:, None] + reg.v[None, :]
    reg.b[:] = 0.0
    holes = np.zeros(reg.b.shape, bool)
    for hx, hz in P.DW_HOLES:
        holes |= (X - hx)**2 + (Z - hz)**2 < (P.DW_HOLE_D/2)**2
    reg.b[:] = np.where(holes, -P.BOARD_T, 0.0)
    mesh = (np.abs(X - cx) < 0.05) & (np.abs(Z - cz) < 0.05)
    # Homogenized open weave: fibre solid volume occupies 30% of the 0.5 mm
    # envelope. The remaining volume accepts compound, unlike a solid plateau.
    reg.b[:] = np.where(mesh, P.MESH_T*P.MESH_SOLID, reg.b)
    reg.holes, reg.mesh = holes, mesh
    # compound over the mesh: a mound 1-3 mm proud with knife ridges, feathered out to 90 mm from the centre
    r = np.hypot((X - cx)/0.09, (Z - cz)/0.09)
    mound = np.clip(1 - r, 0, 1)**0.7*0.0022 + 0.0006*np.sin(2*math.pi*(X - cx)/0.035)*(r < 0.9)
    mound += 0.0003*_noise(rng, reg.b.shape, 2)
    reg.tb[:] = np.where(r < 1.0, np.maximum(mound - np.maximum(reg.b, 0.0), 0.0), 0.0)
    reg.paste_mask = np.ones(reg.b.shape, bool)
    reg.ref_mask = ~holes
    reg.a[:] = np.where(reg.tb > 0, 4.0e-4, 0.0)
    age = AGES_21C["patch-drywall"]*masco_factor(temp_c)
    reg.t0[:] = -age
    reg.volume0 = reg.volume()
    reg.stage = dict(task="patch-drywall", holes=len(P.DW_HOLES))
    return reg


def hawk_region(task, temp_c, seed):
    """Compound on the hawk (the hawk lies on the cart top, its face 15 mm above it)."""
    rng = np.random.default_rng(7000 + seed)
    cx, cy, cyaw = TASK_SETUP[task]["cart"]
    hx, hy = cx - 0.25, cy - 0.02
    cell = 0.002
    s = P.HAWK
    o = (hx - s/2, hy - s/2, CART_TOP + 0.015)
    n = int(round(s/cell))
    reg = Region("hawk", Compound(temp_c), o, (1, 0, 0), (0, 1, 0), cell, (n, n), temp_c=temp_c, air="still", texture_kind=7)
    u = reg.u[:, None] - s/2 + 0*reg.v[None, :]
    v = 0*reg.u[:, None] + reg.v[None, :] - s/2
    heap = np.clip(0.022 - np.hypot(u, v)*0.30, 0.0, None)
    heap += np.where(heap > 0.002, 0.0005*_noise(rng, reg.b.shape, 3), 0.0)
    reg.tb[:] = np.maximum(heap, 0.0)
    reg.a[:] = 5.0e-4
    reg.t0[:] = -240.0
    reg.volume0 = reg.volume()
    return reg


def board_region(task, temp_c, seed):
    """Mortar on the mortar board on its stand next to the infill (the board's face 0.75 m up)."""
    rng = np.random.default_rng(8000 + seed)
    cx, cy, _ = TASK_SETUP[task]["cart"]
    bx, by = cx, cy - 0.70
    cell = 0.003
    L, W = P.MORTAR_BOARD
    o = (bx - L/2, by - W/2, 0.75)
    nu, nv = int(round(L/cell)), int(round(W/cell))
    reg = Region("board", MasonryMortar(temp_c), o, (1, 0, 0), (0, 1, 0), cell, (nu, nv), temp_c=temp_c, air="still", texture_kind=5)
    u = reg.u[:, None] - L/2 + 0*reg.v[None, :]
    v = 0*reg.u[:, None] + reg.v[None, :] - W/2
    heap = np.clip(0.05 - np.hypot(u, v)*0.25, 0.0, None)
    heap += np.where(heap > 0.003, 0.001*_noise(rng, reg.b.shape, 3), 0.0)
    reg.tb[:] = np.maximum(heap, 0.0)
    reg.a[:] = 8.0e-4
    reg.t0[:] = -30*60.0
    reg.volume0 = reg.volume()
    return reg


# ------------------------------------------------------------------ props: static colliders that belong to a task's station
def cured_pad(task):
    """In a task that does not work the pad, the pad has been poured and finished on an earlier day: a hardened,
    steel-troweled slab inside its forms (a static part)."""
    if task.startswith("pad"):
        return []
    return [("pad_cured", (P.PAD_CENTRE[0], P.PAD_CENTRE[1], P.FORM[1]/2), (P.PAD_SIZE[0]/2, P.PAD_SIZE[1]/2, P.FORM[1]/2),
             "old_concrete")]


def props(task):
    """Static boxes (name, centre, half extents, material) of the station's props: the mixing tub with its leftover
    concrete (pad tasks), the hawk lying on a compound pail on the cart (drywall), the mortar board on its stand
    (joints), the rag folded on the cart top (sealant). The tub region's base is the tub floor's top (z 0.012); the hawk's face is 15 mm above the cart top
    only as it lies on the pail's lid; the board's face is at 0.75 m."""
    ts = TASK_SETUP[task]
    out = []
    if ts.get("tub"):
        L, W, H = P.TUB
        cx, cy = TUB_CENTRE
        t = 0.008
        out += [("tub_floor", (cx, cy, 0.006), (L/2 + t, W/2 + t, 0.006), "tub_black"),
                ("tub_wall_s", (cx, cy - W/2 - t/2, H/2), (L/2 + t, t/2, H/2), "tub_black"),
                ("tub_wall_n", (cx, cy + W/2 + t/2, H/2), (L/2 + t, t/2, H/2), "tub_black"),
                ("tub_wall_w", (cx - L/2 - t/2, cy, H/2), (t/2, W/2, H/2), "tub_black"),
                ("tub_wall_e", (cx + L/2 + t/2, cy, H/2), (t/2, W/2, H/2), "tub_black")]
    if ts.get("hawk"):
        cx, cy, _ = ts["cart"]
        hx, hy = cx - 0.25, cy - 0.02
        s = P.HAWK
        out += [("hawk_face", (hx, hy, CART_TOP + 0.015 - 0.00075), (s/2, s/2, 0.00075), "hawk_aluminum"),
                ("hawk_pail", (hx, hy, CART_TOP + 0.007), (0.13, 0.13, 0.007), "pail_white")]
    if ts.get("rag"):
        cx, cy, cyaw = ts["cart"]
        lx, ly = ts["rag"]
        c, s_ = math.cos(cyaw), math.sin(cyaw)
        L, W, T = P.RAG
        out.append(("rag", (cx + c*lx - s_*ly, cy + s_*lx + c*ly, CART_TOP + T/2), (L/2, W/2, T/2), "rag_red"))
    if ts.get("board"):
        cx, cy, _ = ts["cart"]
        bx, by = cx, cy - 0.70
        L, W = P.MORTAR_BOARD
        out += [("board_face", (bx, by, 0.75 - 0.0095), (L/2, W/2, 0.0095), "board_plastic")]
        for k, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            out.append((f"board_leg{k}", (bx + sx*(L/2 - 0.05), by + sy*(W/2 - 0.05), (0.75 - 0.019)/2), (0.02, 0.02, (0.75 - 0.019)/2), "steel_black"))
    return out


def cart_boxes(task):
    """The locked cart's parts as static boxes in the world frame."""
    cx, cy, cyaw = TASK_SETUP[task]["cart"]
    c, s = math.cos(cyaw), math.sin(cyaw)
    out = []
    for name, (x, y, z), (hx, hy, hz) in cart_parts():
        wx, wy = cx + c*x - s*y, cy + s*x + c*y
        out.append((f"cart_{name}", (wx, wy, z), (hx, hy, hz), cyaw, "cart_grey" if not name.startswith("caster") else "rubber_black"))
    return out


def setup(task, seed=0, *, temp_c=21.1, air="still", bleed=None, variant="default", skin_s=None):
    """Everything a world needs for a task: statics, cart, tools and their poses, robot start, regions."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; tasks: {', '.join(TASKS)}")
    if air not in P.AIR:
        raise ValueError(f"air must be one of {', '.join(P.AIR)}")
    if not (10.0 <= temp_c <= 32.0):
        raise ValueError("temp_c must be in [10, 32] C")
    bleed = P.BLEED_MEAN if bleed is None else float(bleed)
    if not (0.05 <= bleed <= 0.5):
        raise ValueError("bleed must be in [0.05, 0.5] kg/m2/h")
    ts = TASK_SETUP[task]
    regions = {}
    if task.startswith("pad"):
        regions["pad"] = pad_region(task, seed, temp_c, air, bleed)
        if ts.get("tub"):
            regions["tub"] = tub_region(AGES_21C[task]*masco_factor(temp_c), temp_c, seed, bleed)
    elif task == "patch-flush":
        regions["patch"] = patch_region(seed, temp_c)
    elif task == "tool-joints":
        regions["infill"] = infill_region(seed, temp_c)
        regions["board"] = board_region(task, temp_c, seed)
    elif task == "tool-sealant":
        regions["joint"] = joint_region(seed, temp_c, skin_s)
    elif task == "patch-drywall":
        regions["wallboard"] = drywall_region(seed, temp_c)
        regions["hawk"] = hawk_region(task, temp_c, seed)
    for reg in regions.values():
        if reg.material.name == "concrete":
            reg.reserve[:] = np.minimum(np.maximum(P.T_SURF_MAX-reg.ts, 0.), reg.tb)
    return dict(task=task, seed=int(seed), temp_c=float(temp_c), air=air, bleed=bleed, variant=variant, skin_s=skin_s,
                statics=all_statics(variant) + props(task) + cured_pad(task), cart=ts["cart"], cart_boxes=cart_boxes(task), tools=tool_poses(task), robot=ts["robot"], regions=regions,
                tub=bool(ts.get("tub")), hawk=bool(ts.get("hawk")), board=bool(ts.get("board")),
                glides="nylon", items={})
