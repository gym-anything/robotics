"""Tools and containers: rigid parts for the rigid world and bristle banks for the bristle solver.

Every tool is a list of `Part`s (classroom_arrangement.furniture.Part: collision primitives that carry the tool's
mass where its steel and plastic are) and, for a brush, a bristle bank in the tool's body frame. Body frames:

  brooms         origin at the centre of the block; x along the block's long side (the sweep width); the bristles
                 leave the block's lower face toward -z; the handle goes up from the block.
  counter brush  as a broom, the handle continuing the block along +x.
  dustpans       origin on the floor under the middle of the pan's floor; +y is where the mouth opens (the lip is
                 at +y); z up. The lobby pan's handle is a second body on a pivot across the top of the mouth.
  containers     origin on the floor under the centre; z up; open at the top.
"""

import math
import numpy as np
from classroom_arrangement.furniture import Part, tube
from . import params as P
from .solver import Bank

def cap_radius(spec):
    """Hanger cap at the end of a broom handle, which rests on the rail's two prongs: 7.5 mm proud of the handle all
    round. With one size for every handle (16 mm) the cap of the thickest handle sat so deep between
    its prongs that the contact was steeper than the friction angle: the push broom wedged itself there under its own
    weight (130 N on each prong for a broom of 10.8 N) and would have needed a pull of some 60 N to come off. With
    the cap in proportion every broom's contact is about 40 degrees from the vertical, well clear of the 26.6 of
    friction 0.5."""
    return spec["handle_d"]/2 + 0.0075


def _box(name, size, pos, mass, material, role="tool", mu=P.TOOL_MU, quat=(0., 0., 0., 1.)):
    return Part(name, "box", tuple(s/2 for s in size), tuple(pos), tuple(quat), mass, material, role, True, mu)


def _bank(name, spec, roots, dirs, length, per_row):
    mat = P.PET if spec["material"] == "PET" else P.PP
    return Bank(name, np.asarray(roots, float), np.asarray(dirs, float), np.asarray(length, float), spec["nodes"],
                spec["filaments"]/spec["clusters"], spec["filament_d"], mat["E"], mat["rho"], P.BRISTLE_MU,
                P.BRISTLE_ZETA, per_row)


def _rows(spec, face, splay_deg, lean_deg=0.0, inset=0.004):
    """Cluster roots on the block's lower face, row by row: (roots, dirs, lengths, per_row). The outer rows lean
    outward by splay_deg, all rows lean toward +y by lean_deg, and the free length runs linearly from bristle_len[0]
    at -x to bristle_len[1] at +x (an angled trim)."""
    bx, by, bz = spec["block"]
    rows = spec["rows"]
    per = spec["clusters"]//rows
    xs = np.linspace(-bx/2 + inset, bx/2 - inset, per)
    ys = np.linspace(-by/2 + inset + 0.002, by/2 - inset - 0.002, rows)
    roots, dirs, length = [], [], []
    for k, y in enumerate(ys):
        a = math.radians(lean_deg + splay_deg*(2*k/(rows - 1) - 1) if rows > 1 else lean_deg)
        for x in xs:
            fan = math.radians(4.0)*(2*x/bx)                 # the ends of the bank fan out a little along the block
            d = np.array([math.sin(fan), math.sin(a), -math.cos(a)*math.cos(fan)])
            roots.append((x, y, face))
            dirs.append(d/np.linalg.norm(d))
            length.append(spec["bristle_len"][0] + (spec["bristle_len"][1] - spec["bristle_len"][0])*(x + bx/2)/bx)
    return np.array(roots), np.array(dirs), np.array(length), per


def _bristle_mass(spec):
    mat = P.PET if spec["material"] == "PET" else P.PP
    return spec["filaments"]*mat["rho"]*math.pi*spec["filament_d"]**2/4*float(np.mean(spec["bristle_len"]))


def angle_broom(name="angle_broom", spec=P.ANGLE_BROOM, color="broom_red"):
    bx, by, bz = spec["block"]
    r = spec["handle_d"]/2
    top = bz/2 + spec["handle_len"]
    mb = _bristle_mass(spec)
    m_handle = 0.24 if spec is P.ANGLE_BROOM else 0.16      # 19 mm steel tube of 0.45 mm wall: 0.21 kg/m
    parts = [_box("block", (bx, by, bz), (0., 0., 0.), spec["mass"] - m_handle - 0.012, color),
             tube("handle", (0., 0., bz/2 - 0.01), (0., 0., top - r), r, m_handle, "handle_steel", "handle"),
             Part("cap", "capsule", (cap_radius(spec), 0.004), (0., 0., top + 0.004), (0., 0., 0., 1.), 0.012, "cap_black", "cap", True,
                  P.TOOL_MU)]
    parts[1].mu = P.HANDLE_MU
    roots, dirs, length, per = _rows(spec, -bz/2, splay_deg=7.0)
    # grasp frames: points on the handle axis, measured from the cap downward (two pinches 0.5 m apart).
    # They are where an operator's hands hold and nothing physical. The lobby broom's are 14 mm further from its cap
    # since its handle is 14 mm longer (the maker's height)
    down = 0.114 if spec is P.LOBBY_BROOM else 0.10
    grasps = dict(upper=(0., 0., top - down), lower=(0., 0., top - down - 0.50))
    return dict(name=name, kind="broom", parts=parts, bank=_bank(name, spec, roots, dirs, length, per), bristle_mass=mb,
                handle_axis=((0., 0., bz/2), (0., 0., top)), grasps=grasps, hang="cap", spec=spec)


def lobby_broom(name="lobby_broom"):
    return dict(angle_broom(name, P.LOBBY_BROOM, "broom_black"), kind="broom")


def push_broom(name="push_broom", spec=P.PUSH_BROOM):
    bx, by, bz = spec["block"]
    r = spec["handle_d"]/2
    a = math.radians(spec["handle_angle_deg"])               # between the handle and the bristles
    h = np.array([0., -math.sin(a), math.cos(a)])
    foot = np.array([0., 0., bz/2 - 0.008])
    end = foot + h*spec["handle_len"]
    mb = _bristle_mass(spec)
    m_handle = 0.42                                          # 24 mm lacquered wood, 1.52 m at 600 kg/m3
    parts = [_box("block", (bx, by, bz), (0., 0., 0.), spec["mass"] - m_handle - 0.012, "broom_block"),
             tube("handle", foot, end - h*r, r, m_handle, "handle_wood", "handle"),
             Part("cap", "capsule", (cap_radius(spec), 0.004), tuple(end + h*0.004), parts_quat(h), 0.012, "cap_black", "cap", True,
                  P.TOOL_MU)]
    parts[1].mu = P.HANDLE_MU
    roots, dirs, length, per = _rows(spec, -bz/2, splay_deg=5.0)
    grasps = dict(upper=tuple(foot + h*(spec["handle_len"] - 0.15)), lower=tuple(foot + h*(spec["handle_len"] - 0.70)))
    return dict(name=name, kind="broom", parts=parts, bank=_bank(name, spec, roots, dirs, length, per), bristle_mass=mb,
                handle_axis=(tuple(foot), tuple(end)), grasps=grasps, hang="block", spec=spec)


def parts_quat(z):
    """Quaternion (xyzw) turning local z onto the unit vector z."""
    z = np.asarray(z, float)/np.linalg.norm(z)
    axis = np.cross((0., 0., 1.), z)
    s = float(np.linalg.norm(axis))
    if s < 1e-12:
        return (0., 0., 0., 1.) if z[2] > 0 else (1., 0., 0., 0.)
    ang = math.atan2(s, float(z[2]))
    axis /= s
    return (*(axis*math.sin(ang/2)), math.cos(ang/2))


def counter_brush(name="counter_brush", spec=P.COUNTER_BRUSH):
    bx, by, bz = spec["block"]
    hl = spec["handle_len"]
    mb = _bristle_mass(spec)
    parts = [_box("block", (bx, by, bz), (0., 0., 0.), spec["mass"]*0.6, "brush_wood"),
             tube("handle", (bx/2 - 0.005, 0., 0.), (bx/2 + hl, 0., 0.), 0.012, spec["mass"]*0.4, "brush_wood", "handle")]
    parts[1].mu = P.HANDLE_MU
    roots, dirs, length, per = _rows(spec, -bz/2, splay_deg=6.0)
    return dict(name=name, kind="brush", parts=parts, bank=_bank(name, spec, roots, dirs, length, per), bristle_mass=mb,
                handle_axis=((bx/2, 0., 0.), (bx/2 + hl, 0., 0.)), grasps=dict(upper=(bx/2 + 0.06, 0., 0.)), hang=None, spec=spec)


def _pan_shell(w, depth, height, wall, lip, mass, material, mu, ramp=0.030, tray=0.006):
    """Open scoop: a floor slab `tray` thick lying on the ground, two sides, a back, a hood over the rear half, and
    the lip: a strip sloping from the slab's front edge down to the ground, ending in a feather edge `lip` thick
    (the step a bristle or a grain has to climb: a cluster of 0.7 mm radius rides over 0.5 mm, a grain finer than
    the edge stops at it). The loose pieces see the slab and the lip as the thin boxes they are, from both sides
    (solver.sd_box_from: a sphere leaves a box by the face it came in through).
    Frame: origin on the ground under the middle of the pan; mouth toward +y."""
    hw, hd = w/2, depth/2
    slope = math.atan2(tray - lip, ramp)
    area = dict(floor=w*(depth - ramp), sides=2*depth*height*0.8, back=w*height, hood=w*depth*0.45, lip=w*ramp)
    tot = sum(area.values())
    m = {k: mass*a/tot for k, a in area.items()}
    # about -x: the front (mouth) edge goes down.
    q_lip = (-math.sin(slope/2), 0., 0., math.cos(slope/2))
    parts = [
        _box("floor", (w, depth - ramp, tray), (0., -ramp/2, tray/2), m["floor"], material, "pan_floor", mu),
        _box("lip", (w, math.hypot(ramp, tray - lip), lip), (0., hd - ramp/2, (tray + lip)/2 - lip/2), m["lip"], material,
             "pan_lip", mu, q_lip),
        # The sides stop at 0.8 of the height: the lobby pan's yoke arms are straight rods from the pivots outside
        # the walls to the handle's foot over the pan, and pass over the walls' tops there. It leaves a slot under
        # the hood (22 mm on the lobby pan, 9 mm on the hand pan) that a real pan does not have.
        _box("side_l", (wall, depth, height*0.8), (-hw + wall/2, 0., tray + height*0.4), m["sides"]/2, material, "pan_wall", mu),
        _box("side_r", (wall, depth, height*0.8), (hw - wall/2, 0., tray + height*0.4), m["sides"]/2, material, "pan_wall", mu),
        _box("back", (w, wall, height), (0., -hd + wall/2, tray + height/2), m["back"], material, "pan_wall", mu),
        _box("hood", (w, depth*0.45, wall), (0., -hd + depth*0.225, tray + height - wall/2), m["hood"], material, "pan_wall", mu),
    ]
    return parts


def lobby_pan(name="lobby_pan", spec=P.LOBBY_PAN):
    w, depth, height = spec["pan"]
    tray = 0.003
    wr, ww, wm = spec["wheel_r"], spec["wheel_w"], spec["wheel_mass"]
    parts = _pan_shell(w, depth, height, spec["wall"], spec["lip"], spec["pan_mass"] - 2*wm, "pan_black", spec["mu"], ramp=0.025, tray=tray)
    # Two rear wheels, each its own body on a hinge along the pan's width, beside the side walls at the heel (the
    # product has them: "rear wheels" on the maker's page). The axle is 10 mm behind the back wall's outer face and the
    # wheel's lowest point 0.3 mm above the plane of the pan's underside: a pan standing flat stands on its floor, a pan
    # tipped back by more than 1.7 degrees (0.3 mm over 10 mm) stands on its wheels and rolls. With the axle ahead of
    # the back wall the heel's edge reached the floor first at every angle below 80 degrees.
    turn = (0., math.sin(math.pi/4), 0., math.cos(math.pi/4))           # the cylinder's axis (z) laid along x
    wheel = lambda k: Part(f"wheel_{k}", "cylinder", (wr, ww/2), (0., 0., 0.), turn, wm, "wheel_grey", "wheel", True, spec["wheel_mu"])
    wheels = [dict(parts=[wheel(k)], pos=(sx*(w/2 + 0.002 + ww/2), -depth/2 - 0.010, wr + 0.0003), axis=(1., 0., 0.))
              for k, sx in (("l", -1.), ("r", 1.))]
    # The handle is its own body (the scene joins them with a hinge): a rod standing over the rear of the pan on a
    # yoke whose two arms come forward and down outside the side walls to the pivots. Nothing crosses the mouth (a bar
    # across it caught the broom and the pan went up with it). The pivots are ahead of the pan's centre of mass, so that
    # a pan lifted by its handle swings mouth-up and keeps its load (pivots at the back made it swing mouth-down).
    pivot = (0., 0.08, tray + 0.80*height)
    r = spec["handle_d"]/2
    back, rise = -0.17, 0.20                      # the rod stands over the rear of the pan: the mouth is free for a broom
    ax = w/2 + 0.009
    handle = [tube("handle", (0., back, rise), (0., back, rise + spec["handle_len"] - r), r, spec["handle_mass"] - 0.05, "handle_steel",
                   "handle"),
              tube("yoke_l", (-ax, 0., 0.), (-0.03, back, rise), 0.006, 0.025, "pan_black", "yoke"),
              tube("yoke_r", (ax, 0., 0.), (0.03, back, rise), 0.006, 0.025, "pan_black", "yoke")]
    handle[0].mu = P.HANDLE_MU
    for p in handle[1:]:
        p.mu = spec["mu"]
    top = (0., back, rise + spec["handle_len"])
    return dict(name=name, kind="pan", parts=parts, handle_parts=handle, wheels=wheels, pivot=pivot, pivot_axis=(1., 0., 0.),
                # handle against its back stop at +8 deg; forward it turns 50 deg over the mouth and then carries the
                # pan with it, so that a handle swung forward past the horizontal turns the pan mouth-down
                pivot_limits=(math.radians(-50.0), math.radians(8.0)),
                handle_axis=((0., back, rise), top), grasps=dict(upper=(0., back, rise + spec["handle_len"] - 0.10)),
                bank=None, spec=spec, mouth=dict(width=w - 2*spec["wall"], depth=depth, y_lip=depth/2), tray=tray)


def hand_pan(name="hand_pan", spec=P.HAND_PAN):
    """Hand dustpan (Rubbermaid FG200500): an open moulded scoop, 311 mm across. A floor slab and a lip like the
    lobby pan's; a back wall of the full height; side walls that come down from the back wall's height to a third
    of it at the mouth (two boxes a side); a flat handle rising from the top of the back wall. No hood."""
    w, depth, height = spec["pan"]
    wall, lip, mu = spec["wall"], spec["lip"], spec["mu"]
    tray, ramp = 0.004, 0.022
    hw, hd = w/2, depth/2
    hr, hf = 0.85*height, 0.35*height                       # the side walls' height at the rear and toward the mouth
    front = depth/2 - ramp                                   # the front boxes stop where the lip begins
    area = dict(floor=w*(depth - ramp), lip=w*ramp, back=w*height, rear=depth*hr, front=2*front*hf)
    shell = spec["mass"] - 0.03
    m = {k: shell*a/sum(area.values()) for k, a in area.items()}
    slope = math.atan2(tray - lip, ramp)
    q_lip = (-math.sin(slope/2), 0., 0., math.cos(slope/2))
    mat = "pan_black"
    parts = [
        _box("floor", (w, depth - ramp, tray), (0., -ramp/2, tray/2), m["floor"], mat, "pan_floor", mu),
        _box("lip", (w, math.hypot(ramp, tray - lip), lip), (0., hd - ramp/2, (tray + lip)/2 - lip/2), m["lip"], mat, "pan_lip", mu, q_lip),
        _box("back", (w, wall, height), (0., -hd + wall/2, tray + height/2), m["back"], mat, "pan_wall", mu),
        _box("side_l", (wall, depth/2, hr), (-hw + wall/2, -depth/4, tray + hr/2), m["rear"]/2, mat, "pan_wall", mu),
        _box("side_r", (wall, depth/2, hr), (hw - wall/2, -depth/4, tray + hr/2), m["rear"]/2, mat, "pan_wall", mu),
        _box("side_lf", (wall, front, hf), (-hw + wall/2, front/2, tray + hf/2), m["front"]/2, mat, "pan_wall", mu),
        _box("side_rf", (wall, front, hf), (hw - wall/2, front/2, tray + hf/2), m["front"]/2, mat, "pan_wall", mu),
    ]
    hl = spec["handle_len"]
    p = tube("handle", (0., -depth/2, tray + height*0.80), (0., -depth/2 - hl, tray + height*1.05), spec["handle_d"]/2, 0.03, mat, "handle")
    p.mu = P.HANDLE_MU
    parts.append(p)
    return dict(name=name, kind="pan", parts=parts, bank=None, spec=spec, handle_axis=(p.pos, p.pos),
                grasps=dict(upper=(0., -depth/2 - hl*0.55, tray + height*0.94)),
                mouth=dict(width=w - 2*spec["wall"], depth=depth, y_lip=depth/2))


def staves(d, wall, at_least=12):
    """How many flat staves a round wall needs so that the bottom disc (radius d/2) reaches the staves' inner faces
    all the way round: their inner corners are at (d/2 - wall)/cos(pi/n), which must not exceed d/2."""
    r = d/2
    return max(at_least, math.ceil(math.pi/math.acos((r - wall)/r)))


def _ring(name, d, h, wall, mass, material, n=12, base=0.004):
    """Open-top round shell: n flat staves and a bottom. With too few staves the wall's inner corners stand outside
    the bottom disc: the barrel had 14 and the pail 12, which left a crescent hole up to 2.3 mm wide at every corner,
    above a 4 mm slot under the staves. Rice and grit emptied into the barrel fell through and lay on the room's
    floor, some of them outside the wall (found in the trace of a robot emptying the dustpan: 31 pieces seen inside
    the barrel ended outside it). `staves` gives the count that closes the bottom (18 and 16)."""
    r = d/2
    n = staves(d, wall, n)
    w = 2*r*math.tan(math.pi/n) + 0.004
    side = 0.85*mass/n
    parts = [Part("bottom", "cylinder", (r, base/2), (0., 0., base/2), (0., 0., 0., 1.), 0.15*mass, material, "container_bottom",
                  True, P.TOOL_MU)]
    for k in range(n):
        a = 2*math.pi*k/n
        q = (0., 0., math.sin(a/2), math.cos(a/2))
        parts.append(_box(f"stave_{k}", (wall, w, h - base), ((r - wall/2)*math.cos(a), (r - wall/2)*math.sin(a), base + (h - base)/2),
                          side, material, "container_wall", P.TOOL_MU, q))
    return parts


def barrel(name="barrel", spec=P.BARREL):
    n = staves(spec["d"], spec["wall"])
    # inner["r"] is the largest circle inside the wall; r_corner the wall's inner corners (a piece lying in a corner
    # of the wall is inside the barrel: the evaluator's circle used to leave such pieces out)
    return dict(name=name, kind="container", parts=_ring(name, spec["d"], spec["h"], spec["wall"], spec["mass"], "barrel_grey", n),
                inner=dict(shape="round", r=spec["d"]/2 - spec["wall"], r_corner=(spec["d"]/2 - spec["wall"])/math.cos(math.pi/n), h=spec["h"]), spec=spec)


def pail(name="pail", spec=P.PAIL):
    n = staves(spec["d"], spec["wall"])
    return dict(name=name, kind="container", parts=_ring(name, spec["d"], spec["h"], spec["wall"], spec["mass"], "pail_blue", n),
                inner=dict(shape="round", r=spec["d"]/2 - spec["wall"], r_corner=(spec["d"]/2 - spec["wall"])/math.cos(math.pi/n), h=spec["h"]), spec=spec)


def wastebasket(name="wastebasket", spec=P.WASTEBASKET):
    x, y, h = spec["size"]
    w = spec["wall"]
    m = spec["mass"]
    parts = [_box("bottom", (x, y, w), (0., 0., w/2), 0.15*m, "basket"),
             _box("wall_0", (x, w, h), (0., -y/2 + w/2, h/2), 0.25*m, "basket"), _box("wall_1", (x, w, h), (0., y/2 - w/2, h/2), 0.25*m, "basket"),
             _box("wall_2", (w, y, h), (-x/2 + w/2, 0., h/2), 0.175*m, "basket"), _box("wall_3", (w, y, h), (x/2 - w/2, 0., h/2), 0.175*m, "basket")]
    return dict(name=name, kind="container", parts=parts, inner=dict(shape="box", hx=x/2 - w, hy=y/2 - w, h=h), spec=spec)


BUILDERS = dict(angle_broom=angle_broom, push_broom=push_broom, lobby_broom=lobby_broom, counter_brush=counter_brush,
                lobby_pan=lobby_pan, hand_pan=hand_pan, barrel=barrel, pail=pail, wastebasket=wastebasket)


def summary(tool):
    parts = list(tool["parts"]) + list(tool.get("handle_parts", []))
    m = [p.mass for p in parts]
    c = [p.pos for p in parts]
    for wh in tool.get("wheels", ()):                            # each wheel is a body of its own, at wh["pos"] in the pan's frame
        parts += wh["parts"]
        m += [q.mass for q in wh["parts"]]
        c += [wh["pos"] for q in wh["parts"]]
    m, c = np.array(m), np.array(c)
    out = dict(mass=float(m.sum()), com=np.round((m[:, None]*c).sum(0)/m.sum(), 4).tolist(), parts=len(parts))
    if tool.get("bank") is not None:
        b = tool["bank"]
        out.update(clusters=len(b.roots), nodes=b.nodes, bristle_mass=float(tool["bristle_mass"]),
                   free_length=(float(b.length.min()), float(b.length.max())), cluster_radius=float(b.radius))
    return out
