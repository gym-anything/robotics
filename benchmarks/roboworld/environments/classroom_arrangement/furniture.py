"""Furniture geometry and mass model (one parameter set for physics and rendering).

Body frames: origin on the floor under the centre of the glide pattern, z up. Desk: +y toward its open (student)
side. Chair: +y toward its backrest (the sitter faces -y). Table and teacher's desk: long side along x.
Each item is a list of parts; a part's mass is its share of the item's catalogue mass (params *_PARTS, scaled), so
the body's centre of mass and inertia follow from where the steel, plastic and wood actually are.
Collision primitives: boxes, capsules (tubes) and cylinders (glides). `collide=False` parts are drawn only.
"""

from dataclasses import dataclass, field
import math
import numpy as np
from . import params as P


@dataclass
class Part:
    name: str
    kind: str                 # box | capsule | cylinder | disc (render-only)
    size: tuple               # box: half extents; capsule/cylinder: (radius, half_height)
    pos: tuple
    quat: tuple = (0., 0., 0., 1.)          # xyzw
    mass: float = 0.
    material: str = "steel"
    role: str = ""            # glide | top | leg | seat | back | ...
    collide: bool = True
    mu: float = None


def _seg(a, b):
    """Centre, quaternion (xyzw, local z along a->b) and half length of a segment."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    z = d/L
    axis = np.cross((0., 0., 1.), z)
    s = float(np.linalg.norm(axis))
    if s < 1e-12:
        q = (0., 0., 0., 1.) if z[2] > 0 else (1., 0., 0., 0.)
    else:
        ang = math.atan2(s, float(z[2]))
        axis /= s
        q = (*(axis*math.sin(ang/2)), math.cos(ang/2))
    return tuple((a + b)/2), tuple(map(float, q)), L/2


def tube(name, a, b, r, mass, material="chrome", role="frame", collide=True):
    c, q, h = _seg(a, b)
    return Part(name, "capsule", (r, max(h - r, 1e-4)), c, q, mass, material, role, collide)


def _scale(parts, total):
    m = sum(p.mass for p in parts)
    for p in parts:
        p.mass *= total/m
    return parts


def desk(glide="nylon", mass=None):
    """Virco 785 student desk: hard-plastic top, steel book box open toward +y, 1 in steel legs, glides."""
    m = P.DESK_PARTS
    tx, ty, tt = P.DESK_TOP
    top_z = P.DESK_HEIGHT
    bx, by, bz = P.DESK_BOX
    gx, gy = P.DESK_LEGS_FLOOR
    ux, uy = P.DESK_LEGS_TOP
    gr, gh = P.DESK_GLIDE
    parts = [Part("top", "box", (tx/2, ty/2, tt/2), (0., 0., top_z - tt/2), mass=m["top"], material="desk_top", role="top",
                  mu=P.MU_PLASTIC)]
    under = top_z - tt
    box_y = -0.02                         # book box centre: set back from the student edge
    wall = 0.0015
    box_parts = [("box_bottom", (bx/2, by/2, wall), (0., box_y, under - bz + wall)),
                 ("box_back", (bx/2, wall, bz/2 - wall), (0., box_y - by/2 + wall, under - bz/2)),
                 ("box_side_l", (wall, by/2, bz/2 - wall), (-bx/2 + wall, box_y, under - bz/2)),
                 ("box_side_r", (wall, by/2, bz/2 - wall), (bx/2 - wall, box_y, under - bz/2))]
    area = {n: (s[0]*s[1] if "bottom" in n else s[0]*s[2] if "back" in n else s[1]*s[2]) for n, s, _ in box_parts}
    for n, s, c in box_parts:
        parts.append(Part(n, "box", s, c, mass=m["book_box"]*area[n]/sum(area.values()), material="desk_steel",
                          role="book_box", mu=P.MU_STEEL))
    for k, sy in enumerate((-1, 1)):
        parts.append(Part(f"rail_{k}", "box", (ux, 0.012, 0.010), (0., sy*uy, under - 0.010), mass=m["rails"]/2,
                          material="desk_frame", role="rail", mu=P.MU_STEEL))
    for k, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        a, b = (sx*gx, sy*gy, gh + 0.004), (sx*ux, sy*uy, under - 0.012)
        p = tube(f"leg_{k}", a, b, P.DESK_LEG_R, m["legs"]/4, "desk_frame", "leg")
        p.mu = P.MU_STEEL
        parts.append(p)
        parts.append(Part(f"glide_{k}", "cylinder", (gr, gh/2), (sx*gx, sy*gy, gh/2), mass=m["glides"]/4,
                          material=f"glide_{glide}", role="glide", mu=P.GLIDE_MU[glide][0]))
    return _scale(parts, mass or P.DESK_MASS)


def chair(glide="nylon", mass=None):
    """Virco 9018 stack chair: one-piece polyethylene seat/back shell (seat 15 in wide) on a 7/8 in chrome frame."""
    m = P.CHAIR_PARTS
    gx, gy = P.CHAIR_GLIDES
    gr, gh = P.CHAIR_GLIDE
    seat_z = P.CHAIR_SEAT_H
    r = P.CHAIR_LEG_R
    parts = [Part("seat", "box", (P.CHAIR_SEAT_HALF_W, 0.205, 0.006), (0., -0.01, seat_z - 0.006), mass=m["seat"], material="chair_shell",
                  role="seat", mu=P.MU_PLASTIC)]
    # backrest: three thin panels following the shell's curve, leaning back 12 deg, top at 0.775 m
    lean = math.radians(12.)
    for k, (dx, yaw, w) in enumerate(((0., 0., 0.12), (-0.15, math.radians(20.), 0.075), (0.15, math.radians(-20.), 0.075))):
        q_lean = (math.sin(-lean/2), 0., 0., math.cos(lean/2))
        q_yaw = (0., 0., math.sin(yaw/2), math.cos(yaw/2))
        from scipy.spatial.transform import Rotation as R
        q = tuple((R.from_quat(q_yaw)*R.from_quat(q_lean)).as_quat())
        cy = 0.215 + (0.02 if k else 0.)
        parts.append(Part(f"back_{k}", "box", (w, 0.006, 0.120), (dx, cy, 0.655), q, mass=m["back"]/3,
                          material="chair_shell", role="back", mu=P.MU_PLASTIC))
    for k, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        top = (sx*0.155, sy*0.17, seat_z - 0.03)              # frame tops under the 15 in seat (inside its edges)
        p = tube(f"leg_{k}", (sx*gx, sy*gy, gh + 0.004), top, r, m["legs"]/4, "chrome", "leg")
        p.mu = P.MU_STEEL
        parts.append(p)
        parts.append(Part(f"glide_{k}", "cylinder", (gr, gh/2), (sx*gx, sy*gy, gh/2), mass=m["glides"]/4,
                          material=f"glide_{glide}", role="glide", mu=P.GLIDE_MU[glide][0]))
    for k, sx in enumerate((-1, 1)):
        p = tube(f"rail_{k}", (sx*0.155, -0.17, seat_z - 0.03), (sx*0.155, 0.17, seat_z - 0.03), r, m["rails"]/2, "chrome", "rail")
        p.mu = P.MU_STEEL
        parts.append(p)
        p = tube(f"back_support_{k}", (sx*0.155, 0.17, seat_z - 0.03), (sx*0.155, 0.245, 0.72), r, m["back_supports"]/2,
                 "chrome", "back_support")
        p.mu = P.MU_STEEL
        parts.append(p)
    return _scale(parts, mass or P.CHAIR_MASS)


def table(glide="nylon"):
    """Virco 4000 Series 30 x 60 in activity table (tutoring table)."""
    tx, ty, tt = P.TABLE_TOP
    h = P.DESK_HEIGHT
    parts = [Part("top", "box", (tx/2, ty/2, tt/2), (0., 0., h - tt/2), mass=24.0, material="table_top", role="top",
                  mu=P.MU_PLASTIC),
             Part("apron_0", "box", (tx/2 - 0.08, 0.012, 0.04), (0., -ty/2 + 0.07, h - tt - 0.04), mass=1.0,
                  material="desk_frame", role="rail", mu=P.MU_STEEL),
             Part("apron_1", "box", (tx/2 - 0.08, 0.012, 0.04), (0., ty/2 - 0.07, h - tt - 0.04), mass=1.0,
                  material="desk_frame", role="rail", mu=P.MU_STEEL),
             Part("antiwarp", "box", (tx/2 - 0.10, 0.02, 0.012), (0., 0., h - tt - 0.012), mass=1.0,
                  material="desk_frame", role="rail", mu=P.MU_STEEL)]
    gx, gy = tx/2 - 0.07, ty/2 - 0.07
    for k, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        parts.append(Part(f"leg_{k}", "box", (0.022, 0.022, (h - tt - 0.016)/2), (sx*gx, sy*gy, 0.016 + (h - tt - 0.016)/2),
                          mass=1.3, material="desk_frame", role="leg", mu=P.MU_STEEL))   # parts sum to TABLE_MASS
        parts.append(Part(f"glide_{k}", "cylinder", (0.016, 0.008), (sx*gx, sy*gy, 0.008), mass=0.05,
                          material=f"glide_{glide}", role="glide", mu=P.GLIDE_MU[glide][0]))
    return _scale(parts, P.TABLE_MASS)


def teacher_desk(glide="nylon"):
    """Virco 540 Series 48 x 30 in single-pedestal teacher's desk: steel pedestal (x > 0), modesty panel, two legs.
    The teacher sits on the -y side; the modesty panel faces the class (+y)."""
    tx, ty, tt = P.TEACHER_TOP
    h = P.TEACHER_HEIGHT
    pw = 0.40
    top = 720.*tx*ty*tt                                     # particleboard, as the table top (19.1 kg)
    parts = [Part("top", "box", (tx/2, ty/2, tt/2), (0., 0., h - tt/2), mass=top, material="teacher_top", role="top",
                  mu=P.MU_PLASTIC),
             Part("pedestal", "box", (pw/2, ty/2 - 0.02, (h - tt - 0.02)/2), (tx/2 - pw/2 - 0.01, 0., 0.02 + (h - tt - 0.02)/2),
                  mass=P.TEACHER_MASS - top - 5.0 - 2.4 - 0.4, material="teacher_steel", role="pedestal", mu=P.MU_STEEL),
             Part("modesty", "box", ((tx - pw)/2 - 0.03, 0.008, 0.20), (-pw/2, ty/2 - 0.03, h - tt - 0.22), mass=5.0,
                  material="teacher_steel", role="panel", mu=P.MU_STEEL)]
    for k, sy in enumerate((-1, 1)):
        parts.append(Part(f"leg_{k}", "box", (0.02, 0.02, (h - tt - 0.016)/2), (-tx/2 + 0.04, sy*(ty/2 - 0.05), 0.016 + (h - tt - 0.016)/2),
                          mass=1.2, material="teacher_steel", role="leg", mu=P.MU_STEEL))
    gx = [-tx/2 + 0.04, -tx/2 + 0.04, tx/2 - 0.04, tx/2 - 0.04]
    gy = [-(ty/2 - 0.05), ty/2 - 0.05, -(ty/2 - 0.05), ty/2 - 0.05]
    for k in range(4):
        parts.append(Part(f"glide_{k}", "cylinder", (0.016, 0.008), (gx[k], gy[k], 0.008), mass=0.1,
                          material=f"glide_{glide}", role="glide", mu=P.GLIDE_MU[glide][0]))
    return _scale(parts, P.TEACHER_MASS)


def tote():
    """Plastic book tote with its four textbooks (one rigid body: the books cannot fall out of a carried tote)."""
    x, y, z, mass = P.TOTE
    w = 0.004
    parts = [Part("bottom", "box", (x/2, y/2, w), (0., 0., w), mass=0.25, material="tote", role="tote", mu=P.MU_TOTE)]
    for k, (s, c) in enumerate((((x/2, w, z/2), (0., -y/2 + w, z/2)), ((x/2, w, z/2), (0., y/2 - w, z/2)),
                                ((w, y/2, z/2), (-x/2 + w, 0., z/2)), ((w, y/2, z/2), (x/2 - w, 0., z/2)))):
        parts.append(Part(f"wall_{k}", "box", s, c, mass=0.2, material="tote", role="tote", mu=P.MU_TOTE))
    bx, by, bt, bm = P.BOOK
    for k in range(4):
        parts.append(Part(f"book_{k}", "box", (bt/2 - 0.001, by/2, bx/2), (-0.09 + k*(bt + 0.004), 0., 2*w + bx/2 + 0.001),
                          mass=bm, material=f"book_{k % 4}", role="contents", mu=P.MU_BOOK))
    return _scale(parts, mass)


def basket():
    """Plastic wastebasket (a closed hull: nothing is thrown into it in these tasks)."""
    r, h, mass = P.BASKET
    return [Part("body", "cylinder", (r, h/2), (0., 0., h/2), mass=mass, material="basket", role="basket", mu=P.MU_TOTE)]


def book(k=0):
    bx, by, bt, bm = P.BOOK
    return [Part("book", "box", (bx/2, by/2, bt/2), (0., 0., bt/2), mass=bm, material=f"book_{k % 4}", role="contents",
                 mu=P.MU_BOOK)]


def summary(parts):
    """Mass, centre of mass and the glide positions of an item (point masses at part centres)."""
    m = np.array([p.mass for p in parts])
    c = np.array([p.pos for p in parts])
    return dict(mass=float(m.sum()), com=tuple(np.round((m[:, None]*c).sum(0)/m.sum(), 4)),
                glides=[p.pos[:2] for p in parts if p.role == "glide"])


# the 785's top front edge and the chair's backrest, for the tuck geometry
DESK_EDGE_Y = P.DESK_TOP[1]/2           # student-side edge of the top, desk frame
CHAIR_BACK_FRONT_Y = 0.223              # front face of the backrest at desk-top height (0.72 m), chair frame (back_0 panel)
CHAIR_SEAT_FRONT_Y = -0.215             # front edge of the seat, chair frame
TUCK_DY = DESK_EDGE_Y + 0.02 - CHAIR_BACK_FRONT_Y   # chair pushed in until its backrest is 20 mm from the top edge
USE_DY = 0.45                           # chair "in use": centre 0.45 m behind the desk centre
