"""Brush and debris solver (Warp): bristle clusters as bending chains, loose debris as rigid clusters of spheres.

It owns every bristle and every loose piece; the rigid world (MuJoCo through Newton) owns the robot, the furniture and
the tools' rigid parts. Each 0.5 ms tick of the rigid world the solver takes SUBSTEPS substeps with the tools' block
poses and the rigid shapes ("colliders") interpolated across the tick, and returns the forces it put on them.

Bristles (checked against a planar elastica)
  A cluster is a chain of nodes standing for n filaments bending side by side: bending energy
  (n EI / 2 l^3) sum |x_{j-1} - 2 x_j + x_{j+1}|^2 integrated explicitly, segments kept at length l by projection along
  the chain, node weight, damping relative to the block. The block's face is at the middle of the held segment (nodes 0
  and 1). An 11-node chain comes out a few per cent stiffer than the elastica. Floor and rigid shapes act on nodes
  through a penalty spring with regularised Coulomb friction, so the force a bank puts on the floor is read directly.

Debris
  A piece is a rigid body (position, orientation, linear and angular velocity, mass, principal inertia) carrying
  spheres at fixed local offsets. Contacts are position-based with small substeps: non-penetration, Coulomb friction
  with a static and a kinetic coefficient applied at the contact point (so a sphere pushed along a floor rolls),
  rolling resistance as an angular deceleration while in contact, restitution on fast normal impacts. A piece whose
  position has not changed by more than REST_DISP over REST_TIME goes to sleep and is woken by a contact that
  overlaps it by more than WAKE_PEN or by a moving support.

Nothing here is task logic: no piece is moved, removed or counted by a rule other than contact.
"""

from dataclasses import dataclass, field
import math

import numpy as np
import warp as wp

SUBSTEPS = 2                 # per rigid tick
BACK_NONE = 1.0e9
ROLL_OVER = 0.162              # a wheel rolls over a piece on the floor whose top is lower than this share of its radius
WHEEL_OVER = 100               # what a sphere remembers of a wheel it overlaps (sph_mf): passed over, or pushed
WHEEL_PUSH = 101
WHEEL_FLAT = 0.30              # a wheel pushes a piece only where its rim faces along the floor by at least this much
                               # (contact_world): (R - r)/(R + r) = 0.85, the rim met within 32 deg of its lowest point
SUNK = 0.34                    # a sphere deeper in the room's floor than this share of its radius is no strut (contact_world)
WAKE_REACH = 1.0e-3            # m: a moving shape this near a sleeping piece wakes it without touching it (contact_world)
NET_TICKS = 10                 # stepped ticks between two snapshots of the pieces' places (Sweeper.bodies)
SHAPE_TOL = 1.0e-4             # how deep a sphere may stay in a rigid shape after a substep (shape_deep)
FLOOR_TOL = 1.0e-4             # m: how deep in the room's floor a sphere may end a substep (floor_apply). Not less:
                               # what the floor carries through this pass it carries without friction, and with
                               # 20 um a four-sphere stack slid apart at a friction of 0.35 (statics:
                               # it stands above 0.318). At 0.1 mm the pass is a net under the ordinary floor contact.
BACK_IDS = 256.0               # a holder is recorded as level*BACK_IDS + body (see back_key)
V_MAX = 20.0                   # m/s: a piece faster than this is a fault in any case (unified contract)
V_FALL = 8.0                   # m/s: what a fall of 3.3 m gives; the limit while no bristle bank is at work and no shape moves
V_ARM = 0.5                    # m: a turning shape's speed is taken at this distance from its origin             # 'nothing holds this piece from that side' (an inverse mass, 1/kg)
ON_FLOOR = 1073741824        # carrier code of a piece that lies on the room's floor, or on pieces that do
KC = 48                      # colliders listed per floor cell (two hands side by side over a table leg: 30)
NB = 4                       # fixed-topology neighbours per cluster
BOX, CAPSULE, CYLINDER = 0, 1, 2
BOTH, RODS_ONLY, DEBRIS_ONLY = 0, 1, 2


# ------------------------------------------------------------------------------------------------ geometry helpers
@wp.func
def sgn(x: float) -> float:
    if x < 0.0:
        return -1.0
    return 1.0


@wp.func
def sd_collider(ctype: int, size: wp.vec3, pl: wp.vec3):
    """Signed distance and outward normal (collider frame) of a point to a box (half extents), a capsule along
    local z (radius, half height) or a cylinder along local z (radius, half height)."""
    d = float(0.0)
    n = wp.vec3(0.0, 0.0, 1.0)
    if ctype == 0:
        q = wp.vec3(wp.abs(pl[0]) - size[0], wp.abs(pl[1]) - size[1], wp.abs(pl[2]) - size[2])
        qm = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
        lo = wp.length(qm)
        if lo > 0.0:
            d = lo
            n = wp.vec3(qm[0]*sgn(pl[0]), qm[1]*sgn(pl[1]), qm[2]*sgn(pl[2]))/lo
        else:
            if q[0] > q[1] and q[0] > q[2]:
                d = q[0]
                n = wp.vec3(sgn(pl[0]), 0.0, 0.0)
            elif q[1] > q[2]:
                d = q[1]
                n = wp.vec3(0.0, sgn(pl[1]), 0.0)
            else:
                d = q[2]
                n = wp.vec3(0.0, 0.0, sgn(pl[2]))
    elif ctype == 2:
        # cylinder along local z (radius, half height)
        rr = wp.sqrt(pl[0]*pl[0] + pl[1]*pl[1])
        qr = rr - size[0]
        qz = wp.abs(pl[2]) - size[1]
        rad = wp.vec3(1.0, 0.0, 0.0)
        if rr > 1.0e-9:
            rad = wp.vec3(pl[0]/rr, pl[1]/rr, 0.0)
        if qr > 0.0 and qz > 0.0:
            d = wp.sqrt(qr*qr + qz*qz)
            n = (rad*qr + wp.vec3(0.0, 0.0, sgn(pl[2])*qz))/d
        elif qr > qz:
            d = qr
            n = rad
        else:
            d = qz
            n = wp.vec3(0.0, 0.0, sgn(pl[2]))
    else:
        z = wp.clamp(pl[2], -size[1], size[1])
        v = wp.vec3(pl[0], pl[1], pl[2] - z)
        ln = wp.length(v)
        d = ln - size[0]
        if ln > 1.0e-9:
            n = v/ln
        else:
            n = wp.vec3(1.0, 0.0, 0.0)
    return d, n


@wp.func
def sd_box_from(size: wp.vec3, pl: wp.vec3, po: wp.vec3):
    """Signed distance and outward normal of a point to a box (half extents), with the side decided by where the
    point came from: `po` is its place a substep ago, in the box's frame as it is now.

    The plain rule (nearest face) is wrong for a thin plate. A sphere centre that has got past the middle of a 3 mm
    wall is pushed out of the far side, and one that crosses a plate in a single substep is never seen at all. Here:
    a point inside the box leaves by the face it came in through; a point whose way from `po` went through the box
    and out of the other side is brought back to that face; a point that was already inside keeps the face nearest
    to where it was."""
    q = wp.vec3(wp.abs(pl[0]) - size[0], wp.abs(pl[1]) - size[1], wp.abs(pl[2]) - size[2])
    qo = wp.vec3(wp.abs(po[0]) - size[0], wp.abs(po[1]) - size[1], wp.abs(po[2]) - size[2])
    inside = int(0)
    if q[0] < 0.0 and q[1] < 0.0 and q[2] < 0.0:
        inside = 1
    old_in = int(0)
    if qo[0] < 0.0 and qo[1] < 0.0 and qo[2] < 0.0:
        old_in = 1
    # the way from po to pl against the three slabs: where it enters the box and where it leaves it
    t_in = float(-1.0e9)
    t_out = float(1.0e9)
    k_in = int(-1)
    hit = int(1)
    for k in range(3):
        dk = pl[k] - po[k]
        if wp.abs(dk) < 1.0e-12:
            if wp.abs(po[k]) > size[k]:
                hit = 0
        else:
            t1 = (-size[k] - po[k])/dk
            t2 = (size[k] - po[k])/dk
            lo_k = wp.min(t1, t2)
            hi_k = wp.max(t1, t2)
            if lo_k > t_in:
                t_in = lo_k
                k_in = k
            t_out = wp.min(t_out, hi_k)
    crossed = int(0)
    if hit != 0 and old_in == 0 and k_in >= 0 and t_in <= t_out and t_in >= 0.0 and t_in <= 1.0:
        crossed = 1
    if inside == 0 and crossed == 0:
        # outside, and it did not come through the box: the ordinary distance to the surface
        qm = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
        lo = wp.length(qm)
        return lo, wp.vec3(qm[0]*sgn(pl[0]), qm[1]*sgn(pl[1]), qm[2]*sgn(pl[2]))/wp.max(lo, 1.0e-12)
    k = k_in
    if crossed == 0:
        # it was inside a substep ago too (or the way in cannot be read): the face nearest to where it was
        k = 2
        if qo[0] > qo[1] and qo[0] > qo[2]:
            k = 0
        elif qo[1] > qo[2]:
            k = 1
    sg = sgn(po[k])
    n = wp.vec3(0.0, 0.0, 0.0)
    n[k] = sg
    return sg*pl[k] - size[k], n


@wp.func
def back_key(invm: float, body: int) -> float:
    """What holds a piece from one side, as one number that atomic_min can order: the heavier holder wins. The
    number carries the holder's inverse mass in eight steps to a factor of two (level 0: it is fixed) and which
    rigid body it is (0: the room). The body is needed because a body cannot squeeze a piece between two of its own
    parts: with the inverse mass alone, a heap that reached from one side of a pail to the other counted as held
    between the pail's walls, each wall was pushed outward with some 90 N, and the pail walked across the floor
    and tipped."""
    lvl = float(0.0)
    if invm > 0.0:
        lvl = wp.clamp(wp.round(8.0*wp.log2(invm/1.0e-4)) + 1.0, 1.0, 250.0)
    return lvl*BACK_IDS + float(body)


@wp.func
def back_invm(key: float) -> float:
    lvl = wp.floor(key/BACK_IDS)
    if lvl < 0.5:
        return 0.0
    return 1.0e-4*wp.pow(2.0, (lvl - 1.0)/8.0)


@wp.func
def back_body(key: float) -> int:
    return int(key - BACK_IDS*wp.floor(key/BACK_IDS) + 0.5)


@wp.func
def bin_of(v: wp.vec3) -> int:
    """Which of the six half-axes (+x, -x, +y, -y, +z, -z: 0 to 5) a direction lies along, or -1 if it is more than
    45 degrees from all of them."""
    ax = wp.abs(v[0])
    ay = wp.abs(v[1])
    az = wp.abs(v[2])
    if ax >= ay and ax >= az:
        if ax < 0.7:
            return -1
        if v[0] > 0.0:
            return 0
        return 1
    if ay >= az:
        if ay < 0.7:
            return -1
        if v[1] > 0.0:
            return 2
        return 3
    if az < 0.7:
        return -1
    if v[2] > 0.0:
        return 4
    return 5


@wp.func
def box_face(nl: wp.vec3) -> int:
    """The face of a box that an outward normal along one of its axes belongs to (0 to 5: +x, -x, +y, -y, +z, -z), or
    -1 for a normal that points off an edge or a corner."""
    if nl[0] > 0.999:
        return 0
    if nl[0] < -0.999:
        return 1
    if nl[1] > 0.999:
        return 2
    if nl[1] < -0.999:
        return 3
    if nl[2] > 0.999:
        return 4
    if nl[2] < -0.999:
        return 5
    return -1


@wp.func
def sd_box_kept(size: wp.vec3, pl: wp.vec3, face: int):
    """A sphere that came to a box through `face` stays that face's business for as long as it overlaps the box: its
    distance is measured to that face's plane and it is pushed out of that face, however deep it is and even when
    its centre has come out of the other side. Valid while the centre is over the face (inside its outline); the
    third value says whether it is."""
    k = int(2)
    if face < 2:
        k = 0
    elif face < 4:
        k = 1
    sg = float(1.0)
    if face - 2*k == 1:
        sg = -1.0
    o1 = (k + 1) % 3
    o2 = (k + 2) % 3
    n = wp.vec3(0.0, 0.0, 0.0)
    n[k] = sg
    if wp.abs(pl[o1]) > size[o1] or wp.abs(pl[o2]) > size[o2]:
        return 0.0, n, 0
    return sg*pl[k] - size[k], n, 1


@wp.func
def shape_hit(ctype: int, size: wp.vec3, cx: wp.vec3, cq: wp.quat, cv: wp.vec3, cw: wp.vec3, xs: wp.vec3, rs: float,
              xo: wp.vec3, dt: float, face: int):
    """Overlap of a sphere (centre xs, radius rs, centre a substep ago xo) with a rigid shape, the direction the
    shape pushes it in (world frame), and, for a box, the face to remember the sphere by (-1: none). `face` is the
    face remembered from the substep before, or -1.

    The memory keeps a piece on its own side of a thin wall whatever presses it in. Deciding the side from one
    substep back (sd_box_from) has a weak point: a sphere that stays inside a 3 mm wall for several substeps (a
    piece's wall contacts are averaged with what pushes it from behind) and gets past the middle of the wall is then
    pushed out of the far side, "the face nearest to where it was". The memory closes that weak point."""
    pl = wp.quat_rotate_inv(cq, xs - cx)
    d = float(0.0)
    nl = wp.vec3(0.0, 0.0, 1.0)
    fo = int(-1)
    if ctype == 0:
        kept = int(0)
        if face >= 0:
            d, nl, kept = sd_box_kept(size, pl, face)
            fo = face
        if kept == 0:
            # where the sphere was, carried along with the shape's own motion of this substep
            vs = cv + wp.cross(cw, xs - cx)
            d, nl = sd_box_from(size, pl, wp.quat_rotate_inv(cq, xo + vs*dt - cx))
            fo = box_face(nl)
    else:
        d, nl = sd_collider(ctype, size, pl)
    return rs - d, wp.quat_rotate(cq, nl), fo


@wp.func
def soft_normal(ctype: int, size: wp.vec3, cx: wp.vec3, cq: wp.quat, centre: wp.vec3, n: wp.vec3):
    """For a piece that yields (crumpled paper): the push is along the shape's normal at the piece's centre, and a
    flat face pushes along its own normal also where the ball overhangs the face's edge (the face that looks most
    toward the centre). A pad 17.5 mm wide then holds a wad it meets 15 mm off its middle; with the edge's slanted
    normal it held only within 8 mm."""
    plb = wp.quat_rotate_inv(cq, centre - cx)
    db, nlb = sd_collider(ctype, size, plb)
    if db <= 0.0:
        return n
    if ctype == 0:
        ax = wp.abs(nlb[0])
        ay = wp.abs(nlb[1])
        az = wp.abs(nlb[2])
        if ax >= ay and ax >= az:
            nlb = wp.vec3(wp.sign(nlb[0]), 0.0, 0.0)
        elif ay >= az:
            nlb = wp.vec3(0.0, wp.sign(nlb[1]), 0.0)
        else:
            nlb = wp.vec3(0.0, 0.0, wp.sign(nlb[2]))
    return wp.quat_rotate(cq, nlb)


@wp.func
def cell_of(p: wp.vec3, org: wp.vec3, inv_cs: float, nx: int, ny: int) -> int:
    ix = int(wp.floor((p[0] - org[0])*inv_cs))
    iy = int(wp.floor((p[1] - org[1])*inv_cs))
    if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
        return -1
    return iy*nx + ix


@wp.func
def gen_w(inv_m: float, inv_i: wp.vec3, q: wp.quat, r: wp.vec3, n: wp.vec3) -> float:
    """Generalised inverse mass of a rigid body for a positional impulse along n applied at lever r."""
    rl = wp.quat_rotate_inv(q, wp.cross(r, n))
    return inv_m + rl[0]*rl[0]*inv_i[0] + rl[1]*rl[1]*inv_i[1] + rl[2]*rl[2]*inv_i[2]


@wp.func
def ang_corr(inv_i: wp.vec3, q: wp.quat, r: wp.vec3, p: wp.vec3) -> wp.vec3:
    """Rotation vector (world) produced by a positional impulse p applied at lever r."""
    tl = wp.quat_rotate_inv(q, wp.cross(r, p))
    return wp.quat_rotate(q, wp.vec3(tl[0]*inv_i[0], tl[1]*inv_i[1], tl[2]*inv_i[2]))


@wp.func
def friction_imp(lt: float, wt: float, lam_n: float, mu_s: float, mu_k: float):
    """Tangential positional impulse for a relative tangential slip lt of the contact points (generalised inverse
    mass wt along the slip) under a normal impulse lam_n, and whether the contact sticks. Coulomb's law compares
    impulses, not displacements: a sphere's contact point moves 3.5 times as far per unit impulse along the floor
    as its centre does along the normal, so comparing displacements made rolling spheres slip at a third of their
    friction."""
    need = lt/wt
    if need <= mu_s*lam_n:
        return need, 1
    return mu_k*lam_n, 0


@wp.func
def wheel_passes(remembered: int, soft: float, top: float, radius: float, below: float) -> int:
    """Does a wheel pass over this piece (1) or push it (0)? Decided when a sphere of the piece first overlaps the
    wheel, for the whole piece (its highest point `top`), and kept for as long as that sphere overlaps the wheel."""
    if remembered == WHEEL_OVER:
        return 1
    if remembered == WHEEL_PUSH:
        return 0
    if soft == 0.0 and top < ROLL_OVER*radius and below < 0.5:
        return 1
    return 0


@wp.func
def pen_limited(pen: float, closing: float, dep: float) -> float:
    """Penetration to resolve this substep: all of what the approach of this substep produced (closing, the distance
    the two surfaces moved toward each other), but an overlap that was already there is pushed out no faster than
    the depenetration speed (dep = speed x substep). A piece released inside another, or squeezed by a tool, then
    separates at a bounded speed instead of being thrown."""
    old = pen - wp.max(closing, 0.0)
    if old <= 0.0:
        return pen
    return wp.min(pen, wp.max(closing, 0.0) + dep)


@wp.func
def small_rot(q: wp.quat, qref: wp.quat) -> wp.vec3:
    """Rotation vector (world frame, small angle) that takes the pose qref to q."""
    d = q*wp.quat_inverse(qref)
    if d[3] < 0.0:
        return wp.vec3(-2.0*d[0], -2.0*d[1], -2.0*d[2])
    return wp.vec3(2.0*d[0], 2.0*d[1], 2.0*d[2])


@wp.func
def ang_w(inv_i: wp.vec3, q: wp.quat, axis: wp.vec3) -> float:
    l = wp.quat_rotate_inv(q, axis)
    return wp.dot(wp.cw_mul(l, inv_i), l)


@wp.func
def ang_turn(inv_i: wp.vec3, q: wp.quat, imp: wp.vec3) -> wp.vec3:
    return wp.quat_rotate(q, wp.cw_mul(wp.quat_rotate_inv(q, imp), inv_i))


@wp.func
def roll_couple(th: wp.vec3, w_ang_a: float, w_ang_b: float, lam_n: float, mu_roll: float, r: float, gain: float):
    """Rolling resistance at a contact: a couple against the relative turning th of the two pieces, at most
    1.4 mu_roll N r (with that factor a lone sphere rolling on a plane decelerates at mu_roll g). It is proportional
    to the normal force of the contact, so a grain carrying a pile does not roll out from under it as a free ball
    would (Zhou et al. 1999: sphere piles need it to stand at a realistic slope). Returns the angular impulse
    (opposing th) and whether it stops the turning. `gain` is the share of the turning the couple takes back in one
    substep while it holds: 1 against the turning of the substep, 0.3 against a turn away from the anchor, where
    friction at the same contact already takes back the part of the turn that slips; with both at full gain a chip
    wedged between a shard and the floor flipped back and forth by 34 mrad every substep for ever (a position-level
    correction with velocity carried over is unstable above a total gain of 4/3)."""
    lr = wp.length(th)
    wsum = w_ang_a + w_ang_b
    if lr < 1.0e-12 or wsum <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0), 1
    need = lr/wsum
    bound = 1.4*mu_roll*lam_n*r
    if need <= bound:
        return th*(-gain/wsum), 1
    return th*(-bound/lr), 0


# ------------------------------------------------------------------------------------------------ bristles
@wp.kernel
def rod_step(x: wp.array2d(dtype=wp.vec3), v: wp.array2d(dtype=wp.vec3), xo: wp.array2d(dtype=wp.vec3),
             root: wp.array(dtype=wp.vec3), rdir: wp.array(dtype=wp.vec3), rod_tool: wp.array(dtype=int),
             rod_n: wp.array(dtype=int), rod_seg: wp.array(dtype=float), rod_kb: wp.array(dtype=float),
             rod_m: wp.array(dtype=float), rod_rad: wp.array(dtype=float), rod_mu: wp.array(dtype=float),
             rod_damp: wp.array(dtype=float), rod_kv: wp.array(dtype=float), rod_nb: wp.array2d(dtype=int),
             tool_x: wp.array(dtype=wp.vec3), tool_q: wp.array(dtype=wp.quat), tool_v: wp.array(dtype=wp.vec3),
             tool_w: wp.array(dtype=wp.vec3), tool_on: wp.array(dtype=int),
             coll_type: wp.array(dtype=int), coll_size: wp.array(dtype=wp.vec3), coll_x: wp.array(dtype=wp.vec3),
             coll_q: wp.array(dtype=wp.quat), coll_v: wp.array(dtype=wp.vec3), coll_w: wp.array(dtype=wp.vec3),
             coll_mu: wp.array(dtype=float), coll_tool: wp.array(dtype=int), coll_who: wp.array(dtype=int),
             cells: wp.array2d(dtype=int),
             org: wp.vec3, inv_cs: float, nx: int, ny: int,
             dt: float, gravity: float, k_pen: float, v_eps: float,
             tool_f: wp.array(dtype=wp.vec3), tool_tq: wp.array(dtype=wp.vec3),
             coll_f: wp.array(dtype=wp.vec3), coll_tq: wp.array(dtype=wp.vec3), rod_fn: wp.array(dtype=float),
             ranc: wp.array2d(dtype=wp.vec3), raj: wp.array2d(dtype=int)):
    r = wp.tid()
    t = rod_tool[r]
    n_nodes = rod_n[r]
    seg = rod_seg[r]
    tx = tool_x[t]
    tq = tool_q[t]
    d = wp.quat_rotate(tq, rdir[r])
    p0 = tx + wp.quat_rotate(tq, root[r]) - d*(0.5*seg)
    p1 = p0 + d*seg
    v0 = tool_v[t] + wp.cross(tool_w[t], p0 - tx)
    v1 = tool_v[t] + wp.cross(tool_w[t], p1 - tx)
    if tool_on[t] == 0:
        # a tool at rest in its place: the bank rides rigidly in its straight shape and costs nothing
        for i in range(n_nodes):
            x[r, i] = p0 + d*(float(i)*seg)
            xo[r, i] = x[r, i]
            v[r, i] = v0
            raj[r, i] = -2
        rod_fn[r] = 0.0
        return
    for i in range(n_nodes):
        xo[r, i] = x[r, i]
    x[r, 0] = p0
    x[r, 1] = p1
    v[r, 0] = v0
    v[r, 1] = v1
    m = rod_m[r]
    rad = rod_rad[r]
    kb = rod_kb[r]
    mu = rod_mu[r]
    k_c = k_pen*m/(dt*dt)
    c_c = 2.0*0.5*wp.sqrt(k_c*m)
    f_tool = wp.vec3(0.0, 0.0, 0.0)
    tq_tool = wp.vec3(0.0, 0.0, 0.0)
    fn_sum = float(0.0)
    for i in range(2, n_nodes):
        pi = xo[r, i]
        vi = v[r, i]
        acc = wp.vec3(0.0, 0.0, -gravity)
        cim = xo[r, i - 2] - 2.0*xo[r, i - 1] + pi
        ci = wp.vec3(0.0, 0.0, 0.0)
        cip = wp.vec3(0.0, 0.0, 0.0)
        if i <= n_nodes - 2:
            ci = xo[r, i - 1] - 2.0*pi + xo[r, i + 1]
        if i + 1 <= n_nodes - 2:
            cip = pi - 2.0*xo[r, i + 1] + xo[r, i + 2]
        acc = acc - kb*(cip - 2.0*ci + cim)
        # Material damping (Kelvin-Voigt): a force against the rate at which the curvature changes, the same stencil
        # on the velocities. It grows with frequency, so it leaves the first bending mode alone (whose damping is
        # rod_damp) and stops the chain's shortest wave. Without it that wave, two segments long, rang at 600 Hz for
        # as long as a bank was pressed on the floor (damping ratio 0.0006): clusters drawn as zig-zags, bends of
        # 160 degrees in a leading stroke (traced per substep on a bare bank).
        dim = v[r, i - 2] - 2.0*v[r, i - 1] + vi
        di = wp.vec3(0.0, 0.0, 0.0)
        dip = wp.vec3(0.0, 0.0, 0.0)
        if i <= n_nodes - 2:
            di = v[r, i - 1] - 2.0*vi + v[r, i + 1]
        if i + 1 <= n_nodes - 2:
            dip = vi - 2.0*v[r, i + 1] + v[r, i + 2]
        acc = acc - rod_kv[r]*(dip - 2.0*di + dim)
        fc = wp.vec3(0.0, 0.0, 0.0)
        # STATIC FRICTION OF A NODE. A node that lies on the floor or on a shape is held at the place where it came to
        # lie (its anchor: ranc, on the surface raj: -1 the floor, j a shape, in that shape's own frame) by a damped
        # spring as stiff as the contact itself, for as long as that spring needs no more than mu N. Beyond that it
        # slides against mu N and draws its anchor along. A viscous law (mu N v/v_eps below v_eps = 1 mm/s) has no
        # hold at all. A tip under a sideways load of a tenth of mu N creeps at 0.1 mm/s for ever: a broom let go
        # standing on its bristles sinks 22 mm and leans to 13 degrees in 50 s without ever resting or falling, and
        # with a broom lying dead still one node in twenty of its bank goes on moving at 5 to 40 mm/s and pushes the
        # grains under it about. One anchor a node: on the floor if it touches the floor, else on the first shape it
        # touches; any further shape it touches at the same time keeps the old law.
        held = int(-2)
        # floor
        pen = rad - pi[2]
        if pen > 0.0:
            f_n = wp.max(0.0, k_c*pen - c_c*vi[2])
            vt = wp.vec3(vi[0], vi[1], 0.0)
            fc = fc + wp.vec3(0.0, 0.0, f_n)
            if raj[r, i] != -1:
                ranc[r, i] = wp.vec3(pi[0], pi[1], 0.0)
                raj[r, i] = -1
            anc = ranc[r, i]
            u = wp.vec3(pi[0] - anc[0], pi[1] - anc[1], 0.0)
            ft = u*(-k_c) - vt*c_c
            lim = mu*f_n
            lf = wp.length(ft)
            if lf > lim:
                ft = ft*(lim/wp.max(lf, 1.0e-12))
                lu = wp.length(u)
                if lu > 1.0e-12:
                    u = u*wp.min(1.0, lim/(k_c*lu))
                ranc[r, i] = wp.vec3(pi[0] - u[0], pi[1] - u[1], 0.0)
            fc = fc + ft
            fn_sum += f_n
            held = -1
        # rigid shapes near the node
        c = cell_of(pi, org, inv_cs, nx, ny)
        if c >= 0:
            for k in range(KC):
                j = cells[c, k]
                if j < 0:
                    break
                if coll_tool[j] == t or coll_who[j] == 2:
                    continue
                pl = wp.quat_rotate_inv(coll_q[j], pi - coll_x[j])
                dist, nl = sd_collider(coll_type[j], coll_size[j], pl)
                pen_c = rad - dist
                if pen_c > 0.0:
                    n = wp.quat_rotate(coll_q[j], nl)
                    vc = coll_v[j] + wp.cross(coll_w[j], pi - coll_x[j])
                    vr = vi - vc
                    vn = wp.dot(vr, n)
                    f_n = wp.max(0.0, k_c*wp.min(pen_c, 4.0*rad) - c_c*vn)
                    vt = vr - n*vn
                    sp = wp.length(vt)
                    f = n*f_n
                    muc = wp.min(mu, coll_mu[j])
                    if held == -2 or held == j:
                        if raj[r, i] != j:
                            ranc[r, i] = pl
                            raj[r, i] = j
                        aw = coll_x[j] + wp.quat_rotate(coll_q[j], ranc[r, i])
                        u = pi - aw
                        u = u - n*wp.dot(u, n)
                        ft = u*(-k_c) - vt*c_c
                        lim = muc*f_n
                        lf = wp.length(ft)
                        if lf > lim:
                            ft = ft*(lim/wp.max(lf, 1.0e-12))
                            lu = wp.length(u)
                            if lu > 1.0e-12:
                                u = u*wp.min(1.0, lim/(k_c*lu))
                            ranc[r, i] = wp.quat_rotate_inv(coll_q[j], pi - u - coll_x[j])
                        f = f + ft
                        held = j
                    elif sp > 1.0e-9:
                        f = f - vt*(muc*f_n*wp.min(1.0, sp/v_eps)/sp)
                    fc = fc + f
                    wp.atomic_add(coll_f, j, -f)
                    wp.atomic_add(coll_tq, j, wp.cross(pi - coll_x[j], -f))
        if held == -2:
            raj[r, i] = -2            # in the air: the next thing it lies on gives it a new anchor
        acc = acc + fc/m
        f_tool = f_tool + fc
        tq_tool = tq_tool + wp.cross(pi - tx, fc)
        # neighbouring clusters keep their spacing (same node index)
        for s in range(NB):
            o = rod_nb[r, s]
            if o >= 0:
                dv = pi - xo[o, i]
                dist = wp.length(dv)
                if dist < 2.0*rad and dist > 1.0e-9:
                    acc = acc + dv*(0.25*k_c*(2.0*rad - dist)/(dist*m))
        vi = vi + acc*dt
        vi = vi - (vi - v0)*rod_damp[r]*dt
        if tool_on[t] == 2:
            # THE BANK OF A TOOL THAT HAS NOT MOVED FOR A TENTH OF A SECOND loses half of each node's speed against its
            # block in every substep (world._push_poses sets the flag: the net speed of each of the tool's parts over
            # the last 0.1 s under 0.25 mm/s). A NUMERICAL RULE, like the sleep of a piece: with a broom lying dead
            # still on debris one node in twenty of its bank went on moving at 5 to 40 mm/s, with static friction
            # (above) as without it, and the pieces under those nodes never rested; with the rule the nodes of a
            # resting bank move at 2 to 4.5 mm/s at the most. A real filament loses such a
            # tremble to its own material damping within a few cycles of some hundred hertz; the chain's damping
            # here (rod_damp) is set for the stroke and cannot be raised without stiffening the bank in motion
            # The rule acts on no tool that moves, however slowly: a stroke at 1 mm/s is not touched.
            vi = v0 + (vi - v0)*0.5
        x[r, i] = pi + vi*dt
    for it in range(4):
        for i in range(2, n_nodes):
            a = x[r, i - 1]
            b = x[r, i]
            dvec = b - a
            ln = wp.length(dvec)
            if ln > 1.0e-9:
                corr = dvec*((ln - seg)/ln)
                if i - 1 <= 1:
                    x[r, i] = b - corr
                else:
                    x[r, i - 1] = a + 0.5*corr
                    x[r, i] = b - 0.5*corr
    for i in range(2, n_nodes):
        v[r, i] = (x[r, i] - xo[r, i])/dt
    wp.atomic_add(tool_f, t, f_tool)
    wp.atomic_add(tool_tq, t, tq_tool)
    rod_fn[r] = fn_sum


@wp.kernel
def rod_apply(x: wp.array2d(dtype=wp.vec3), v: wp.array2d(dtype=wp.vec3), rdx: wp.array2d(dtype=wp.vec3),
              rcnt: wp.array2d(dtype=int), rcnt_prev: wp.array2d(dtype=int), rod_tool: wp.array(dtype=int),
              rod_m: wp.array(dtype=float), tool_x: wp.array(dtype=wp.vec3), dt: float, tool_f: wp.array(dtype=wp.vec3),
              tool_tq: wp.array(dtype=wp.vec3)):
    """The loose pieces' reactions on the bristle nodes (summed: see contact_rods) and, through them, on the tool."""
    r, i = wp.tid()
    c = rcnt[r, i]
    rcnt_prev[r, i] = c
    if c > 0:
        if i >= 2:
            dx = rdx[r, i]
            x[r, i] = x[r, i] + dx
            v[r, i] = v[r, i] + dx/dt
            t = rod_tool[r]
            f = dx*(rod_m[r]/(dt*dt))
            wp.atomic_add(tool_f, t, f)
            wp.atomic_add(tool_tq, t, wp.cross(x[r, i] - tool_x[t], f))
        rdx[r, i] = wp.vec3(0.0, 0.0, 0.0)
        rcnt[r, i] = 0


# ------------------------------------------------------------------------------------------------ debris
@wp.kernel
def body_predict(bx: wp.array(dtype=wp.vec3), bq: wp.array(dtype=wp.quat), bv: wp.array(dtype=wp.vec3),
                 bw: wp.array(dtype=wp.vec3), bxo: wp.array(dtype=wp.vec3), bqo: wp.array(dtype=wp.quat),
                 bvp: wp.array(dtype=wp.vec3), bres: wp.array(dtype=wp.vec3), asleep: wp.array(dtype=int),
                 bairk: wp.array(dtype=float), gravity: float, dt: float):
    b = wp.tid()
    bxo[b] = bx[b]
    bqo[b] = bq[b]
    if asleep[b] != 0:
        bv[b] = wp.vec3(0.0, 0.0, 0.0)
        bw[b] = wp.vec3(0.0, 0.0, 0.0)
        bvp[b] = wp.vec3(0.0, 0.0, 0.0)
        return
    vv = bv[b] + wp.vec3(0.0, 0.0, -gravity*dt)
    # still air: a = -(rho C_d A / 2 m) |v| v. Nothing for a grain creeping along the floor; a third of g for a
    # rice grain flicked at 3 m/s, and what stops a paper scrap within a hand's width
    sp = wp.length(vv)
    vv = vv*(1.0/(1.0 + bairk[b]*sp*dt))
    bvp[b] = vv
    bx[b] = bx[b] + (vv*dt + bres[b])
    w = bw[b]
    q = bq[b]
    bq[b] = wp.normalize(q + wp.quat(w[0], w[1], w[2], 0.0)*q*(0.5*dt))


@wp.kernel
def sph_update(bx: wp.array(dtype=wp.vec3), bq: wp.array(dtype=wp.quat), sph_body: wp.array(dtype=int),
               sph_local: wp.array(dtype=wp.vec3), sph_x: wp.array(dtype=wp.vec3)):
    s = wp.tid()
    b = sph_body[s]
    sph_x[s] = bx[b] + wp.quat_rotate(bq[b], sph_local[s])


@wp.kernel
def gather(src: wp.array(dtype=wp.vec3), idx: wp.array(dtype=int), dst: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    dst[i] = src[idx[i]]


@wp.kernel
def count_pairs(grid: wp.uint64, q_idx: wp.array(dtype=int), t_idx: wp.array(dtype=int), t_rmax: float, two_sided: int,
                sph_x: wp.array(dtype=wp.vec3), sph_r: wp.array(dtype=float), sph_body: wp.array(dtype=int),
                asleep: wp.array(dtype=int), bA: wp.array(dtype=wp.mat33)):
    """First of two passes over the contacts between pieces: for every piece, along which directions it is about
    to be corrected and by how many contacts (the sum of n n^T, of which the trace, the number, is used). See
    contact_pairs."""
    k = wp.tid()
    i = q_idx[k]
    a = sph_body[i]
    xi = sph_x[i]
    ri = sph_r[i]
    query = wp.hash_grid_query(grid, xi, ri + t_rmax)
    jj = int(0)
    a_sleep = asleep[a]
    while wp.hash_grid_query_next(query, jj):
        j = t_idx[jj]
        b = sph_body[j]
        if b == a:
            continue
        dvec = xi - sph_x[j]
        dist = wp.length(dvec)
        if dist >= ri + sph_r[j] or dist < 1.0e-9:
            continue
        b_sleep = asleep[b]
        if a_sleep != 0 and (b_sleep != 0 or two_sided == 0):
            continue
        n = dvec/dist
        nn = wp.outer(n, n)
        if a_sleep == 0:
            wp.atomic_add(bA, a, nn)
        if two_sided != 0 and b_sleep == 0:
            wp.atomic_add(bA, b, nn)


@wp.kernel
def piece_top(sph_x: wp.array(dtype=wp.vec3), sph_r: wp.array(dtype=float), sph_body: wp.array(dtype=int),
              btop: wp.array(dtype=float)):
    """The highest point of every piece above the room's floor (see wheel_passes)."""
    s = wp.tid()
    wp.atomic_max(btop, sph_body[s], sph_x[s][2] + sph_r[s])


@wp.kernel
def count_world(sph_x: wp.array(dtype=wp.vec3), sph_r: wp.array(dtype=float), sph_body: wp.array(dtype=int),
                bx: wp.array(dtype=wp.vec3), bq: wp.array(dtype=wp.quat), bxo: wp.array(dtype=wp.vec3),
                bqo: wp.array(dtype=wp.quat), asleep: wp.array(dtype=int), bsoft: wp.array(dtype=float),
                coll_type: wp.array(dtype=int), coll_size: wp.array(dtype=wp.vec3), coll_x: wp.array(dtype=wp.vec3),
                coll_q: wp.array(dtype=wp.quat), coll_v: wp.array(dtype=wp.vec3), coll_w: wp.array(dtype=wp.vec3),
                coll_who: wp.array(dtype=int), cells: wp.array2d(dtype=int), org: wp.vec3, inv_cs: float, nx: int,
                ny: int, dt: float, bA: wp.array(dtype=wp.mat33), coll_invm: wp.array(dtype=float),
                bwb: wp.array2d(dtype=float), nb_body: wp.array(dtype=int), coll_bid: wp.array(dtype=int),
                sph_mj: wp.array2d(dtype=int), sph_mf: wp.array2d(dtype=int), coll_roll: wp.array(dtype=int),
                btop: wp.array(dtype=float)):
    """First of two passes over the contacts with the floor and the rigid shapes (see count_pairs); and, for every
    rigid BODY, on how many held pieces its shapes press together (see contact_world, "a piece that is held")."""
    s = wp.tid()
    a = sph_body[s]
    awake = int(1)
    if asleep[a] != 0:
        awake = 0
    xs = sph_x[s]
    rs = sph_r[s]
    if awake != 0 and rs - xs[2] > 0.0:
        wp.atomic_add(bA, a, wp.outer(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 1.0)))
    c = cell_of(xs, org, inv_cs, nx, ny)
    if c < 0:
        return
    xo = bxo[a] + wp.quat_rotate(bqo[a], wp.quat_rotate_inv(bq[a], xs - bx[a]))
    for k in range(KC):
        j = cells[c, k]
        if j < 0:
            break
        if coll_who[j] == 1:
            continue
        face = int(-1)
        if sph_mj[s, 0] == j:
            face = sph_mf[s, 0]
        elif sph_mj[s, 1] == j:
            face = sph_mf[s, 1]
        wheel = coll_roll[j]
        mem = face
        if wheel != 0:
            face = -1
        pen_c, n, fo = shape_hit(coll_type[j], coll_size[j], coll_x[j], coll_q[j], coll_v[j], coll_w[j], xs, rs, xo, dt, face)
        if pen_c <= 0.0:
            continue
        if wheel != 0:
            # a wheel: nothing if it passes over the piece, a push along the floor otherwise; the chassis: a push along
            # the floor (contact_world)
            if mem == WHEEL_OVER:
                continue
            if wheel == 1 and wheel_passes(mem, bsoft[a], btop[a], coll_size[j][0], bwb[a, 4]) != 0:
                continue
            nh = wp.vec3(n[0], n[1], 0.0)
            lh = wp.length(nh)
            if lh < WHEEL_FLAT:
                continue
            n = nh/lh
        if bsoft[a] > 0.0:
            n = soft_normal(coll_type[j], coll_size[j], coll_x[j], coll_q[j], bx[a], n)
        if awake != 0:
            wp.atomic_add(bA, a, wp.outer(n, n))
        if coll_invm[j] > 0.0 and bsoft[a] == 0.0 and wheel == 0:
            kb = bin_of(-n)
            if kb >= 0:
                key = bwb[a, kb]
                if key < 0.5*BACK_NONE and back_body(key) != coll_bid[j] and rs - xs[2] < SUNK*rs:
                    wp.atomic_add(nb_body, coll_bid[j], 1)


@wp.kernel
def contact_pairs(grid: wp.uint64, q_idx: wp.array(dtype=int), t_idx: wp.array(dtype=int), t_rmax: float, two_sided: int,
                  sph_x: wp.array(dtype=wp.vec3), sph_r: wp.array(dtype=float), sph_body: wp.array(dtype=int),
                  bx: wp.array(dtype=wp.vec3), bq: wp.array(dtype=wp.quat), bax: wp.array(dtype=wp.vec3),
                  baq: wp.array(dtype=wp.quat), binvm: wp.array(dtype=float), binvi: wp.array(dtype=wp.vec3),
                  bxo: wp.array(dtype=wp.vec3), bqo: wp.array(dtype=wp.quat), bdrag: wp.array(dtype=int),
                  btimer: wp.array(dtype=int), wake_steps: int, dep: float,
                  bmu: wp.array(dtype=float), bmur: wp.array(dtype=float), asleep: wp.array(dtype=int), wake_pen: float,
                  kin: float, bdx: wp.array(dtype=wp.vec3), bdth: wp.array(dtype=wp.vec3), bcnt: wp.array(dtype=int),
                  bwake: wp.array(dtype=int), bstick: wp.array(dtype=wp.vec4i),
                  bcar: wp.array(dtype=int), bcar_n: wp.array(dtype=int), bcar_s: wp.array(dtype=int),
                  bpj: wp.array(dtype=int), bpf: wp.array(dtype=wp.vec3), bptq: wp.array(dtype=wp.vec3),
                  bA: wp.array(dtype=wp.mat33), bwb: wp.array2d(dtype=float), bwb_n: wp.array2d(dtype=float),
                  bAs: wp.array(dtype=wp.mat33)):
    """Sphere-sphere contacts between pieces. One-sided: every sphere corrects its own piece (each pair is met from
    both sides). Two-sided: the query set (large spheres) corrects both pieces (the small ones are not queried).
    The slip that friction opposes is measured from each piece's anchor pose (bax, baq; see body_finish).

    How a piece's contacts share its correction. All contacts of a pass are found from the same poses and corrected
    at once, so they must not all be applied in full: each is divided by the piece's NUMBER of contacts (counted in
    a first pass: the trace of the sum of n n^T). Between two awake pieces each side is divided by its own number,
    and the momentum the pair would gain from that is handed to what carries it (below).

    Two other divisors fail. The number of the piece's contacts along the contact's own direction,
    n^T (sum of n n^T) n: the four-sphere stack, which falls below a friction of 0.32 and stands above it with the
    plain count, then stands or falls from one friction to the next between 0.33 and 0.45 (scanned in sixteen
    steps). And one number for both pieces of a pair, the larger: the stack is right again but poured grit lies at
    22 degrees instead of 34."""
    k = wp.tid()
    i = q_idx[k]
    a = sph_body[i]
    xi = sph_x[i]
    ri = sph_r[i]
    query = wp.hash_grid_query(grid, xi, ri + t_rmax)
    jj = int(0)
    a_sleep = asleep[a]
    while wp.hash_grid_query_next(query, jj):
        j = t_idx[jj]
        b = sph_body[j]
        if b == a:
            continue
        dvec = xi - sph_x[j]
        dist = wp.length(dvec)
        rsum = ri + sph_r[j]
        if dist >= rsum or dist < 1.0e-9:
            continue
        n = dvec/dist
        pen = rsum - dist
        b_sleep = asleep[b]
        if a_sleep != 0 and b_sleep != 0:
            continue
        # A sleeping piece wakes when a piece touching it has moved (left its rest reference) within the last
        # wake_steps substeps, or is pressed deep into it. A neighbour that merely leans on a sleeper does not wake
        # it, and a woken piece passes the waking on only if it moves itself.
        # "deep" is a fifth of the smaller sphere (at least wake_pen): a light chip under a heavy shard stands 0.14 mm
        # into the floor in this solver (corrections are shared by inverse mass), and a fixed 0.1 mm woke it for ever
        # A deep overlap wakes only while the piece that presses is itself on the move (it left its rest reference
        # within the last 12 wake windows, 24 ms). A grain caught under a shard that lies on the floor on its own
        # points stands in it for good (the grain is thicker than the gap); with "deep" alone each of the two woke
        # the other the moment it fell asleep, and a few grains in every heap of shards rattled at 2 to 20 mm/s for
        # as long as they were watched, above the evaluator's 5 mm/s for a scene at rest.
        deep = wp.max(wake_pen, 0.2*wp.min(ri, sph_r[j]))
        if a_sleep != 0:
            if btimer[b] < wake_steps or (pen > deep and btimer[b] < 12*wake_steps):
                wp.atomic_add(bwake, a, 1)
            if two_sided == 0:
                continue
        if b_sleep != 0 and two_sided != 0:
            if btimer[a] < wake_steps or (pen > deep and btimer[a] < 12*wake_steps):
                wp.atomic_add(bwake, b, 1)
        cp = sph_x[j] + n*(sph_r[j] - 0.5*pen)
        ra = cp - bx[a]
        rb = cp - bx[b]
        # A sleeping piece is not moved, but the awake piece in contact with it keeps the share of the correction it
        # had while both were awake (the sleeper's own share is dropped). Counting the sleeper as immovable gave the
        # awake piece the whole correction from the substep its neighbour fell asleep: a rice grain resting on a
        # grit grain a twentieth of its mass stands 13 um into it (one pass, corrections shared by inverse mass),
        # and each time the grit grain fell asleep the rice grain was thrown up at 50 mm/s, landed, woke it, and was
        # thrown again 0.11 s later; a pile of 9,401 pieces took 7 to more than 30 s to sleep (traced substep by
        # substep).
        wa = gen_w(binvm[a], binvi[a], bq[a], ra, n)
        wb = gen_w(binvm[b], binvi[b], bq[b], rb, n)
        # (A heavy piece still sinks into a much lighter one under it, until it lies on its own points. Letting the
        # lower piece of a pair give way downward only as far as its carrier does stops that, but puts 2.9 % too
        # much weight on a pail of shards and keeps heaps in light containers awake.)
        if wa + wb <= 0.0:
            continue
        # How far the two surfaces came toward each other in this substep, AT THE CONTACT: the pieces' turning counts.
        # Read at the centres, the end of a tumbling shard that swung into a thin wall or into another piece at
        # 1.5 m/s would count as an overlap that had been there before and be let out at the depenetration speed
        # only (1 m/s).
        sub_a = (bx[a] + ra) - (bxo[a] + wp.quat_rotate(bqo[a], wp.quat_rotate_inv(bq[a], ra)))
        sub_b = (bx[b] + rb) - (bxo[b] + wp.quat_rotate(bqo[b], wp.quat_rotate_inv(bq[b], rb)))
        closing = -wp.dot(sub_a - sub_b, n)
        lam = pen_limited(pen, closing, dep)/(wa + wb)
        pa = n*lam
        # friction on the relative displacement of the two material points now in contact
        # Two pieces that both sit at their anchors: slip is the relative displacement since they came to rest (held
        # elastically, no creep). As soon as either one is moving, slip is that of this substep, so that friction
        # opposes the actual relative motion and can never drive it.
        dpa = wp.vec3(0.0, 0.0, 0.0)
        dpb = wp.vec3(0.0, 0.0, 0.0)
        anchored = int(0)
        if bdrag[a] == 0 and bdrag[b] == 0:
            anchored = 1
        if anchored != 0:
            if a_sleep == 0:
                dpa = (bx[a] + ra) - (bax[a] + wp.quat_rotate(baq[a], wp.quat_rotate_inv(bq[a], ra)))
            if b_sleep == 0:
                dpb = (bx[b] + rb) - (bax[b] + wp.quat_rotate(baq[b], wp.quat_rotate_inv(bq[b], rb)))
        else:
            if a_sleep == 0:
                dpa = (bx[a] + ra) - (bxo[a] + wp.quat_rotate(bqo[a], wp.quat_rotate_inv(bq[a], ra)))
            if b_sleep == 0:
                dpb = (bx[b] + rb) - (bxo[b] + wp.quat_rotate(bqo[b], wp.quat_rotate_inv(bq[b], rb)))
        rel = dpa - dpb
        rel_t = rel - n*wp.dot(rel, n)
        lt = wp.length(rel_t)
        pt = wp.vec3(0.0, 0.0, 0.0)
        stuck = int(1)
        if lt > 1.0e-12:
            tdir = rel_t/lt
            mu_s = wp.min(bmu[a], bmu[b])
            wta = gen_w(binvm[a], binvi[a], bq[a], ra, tdir)
            wtb = gen_w(binvm[b], binvi[b], bq[b], rb, tdir)
            ft, stuck = friction_imp(lt, wta + wtb, lam, mu_s, kin*mu_s)
            pt = tdir*(-ft)
        # rolling resistance against the relative turning (same reference poses as the slip)
        tha = wp.vec3(0.0, 0.0, 0.0)
        thb = wp.vec3(0.0, 0.0, 0.0)
        if anchored != 0:
            if a_sleep == 0:
                tha = small_rot(bq[a], baq[a])
            if b_sleep == 0:
                thb = small_rot(bq[b], baq[b])
        else:
            if a_sleep == 0:
                tha = small_rot(bq[a], bqo[a])
            if b_sleep == 0:
                thb = small_rot(bq[b], bqo[b])
        th = tha - thb
        lth = wp.length(th)
        wra = float(0.0)
        wrb = float(0.0)
        if lth > 1.0e-12:
            wra = ang_w(binvi[a], bq[a], th/lth)
            wrb = ang_w(binvi[b], bq[b], th/lth)
        gain = float(1.0)
        if anchored != 0:
            gain = 0.3
        couple, rstuck = roll_couple(th, wra, wrb, lam, wp.min(bmur[a], bmur[b]), wp.min(ri, sph_r[j]), gain)
        # each piece's own number of contacts (see "How a piece's contacts share its correction" above)
        # A sleeping piece keeps the weight it had when it fell asleep (bAs), as it keeps its share of the
        # correction: with the sleeper's weight dropped to 1, a grit grain under a shard of 30 contacts was pushed
        # fifteen times harder from the substep the shard fell asleep, moved, woke the shard, and was let go
        # again: a few grains in every heap of shards rattled at 2 to 20 mm/s for as long as they were watched
        # (traced in a pail pour).
        mA = bA[a]
        if a_sleep != 0:
            mA = bAs[a]
        mB = bA[b]
        if b_sleep != 0:
            mB = bAs[b]
        wA = wp.max(1.0, wp.trace(mA))
        wB = wp.max(1.0, wp.trace(mB))
        # Momentum between two awake pieces. Piece a is moved by this push divided by its number of contacts, piece b
        # by the opposite push divided by its own: the pair gains p (1/N_a - 1/N_b) from nothing whenever the
        # numbers differ. That difference is handed to what carries the pieces, half from each side.
        hand = -0.5*(1.0/wA - 1.0/wB)
        if a_sleep == 0:
            wp.atomic_add(bstick, a, wp.vec4i(stuck, 1 - stuck, rstuck, 1 - rstuck))
            wp.atomic_add(bdx, a, (pa + pt)*(binvm[a]/wA))
            wp.atomic_add(bdth, a, (ang_corr(binvi[a], bq[a], ra, pa + pt) + ang_turn(binvi[a], bq[a], couple))*(1.0/wA))
            wp.atomic_add(bcnt, a, 1)
            # the shape that carries the piece under this one carries this one too (see carried_reaction)
            cb = bcar[b]
            if cb >= 0:
                if n[2] > 0.1:
                    wp.atomic_max(bcar_n, a, cb)
                else:
                    wp.atomic_max(bcar_s, a, cb)
                if cb < ON_FLOOR:
                    wp.atomic_max(bpj, a, cb)
            # held from this side by what holds the piece that pushes it (see contact_world, "a piece that is held")
            ka = bin_of(n)
            if ka >= 0:
                if bwb[b, ka] < 0.5*BACK_NONE:
                    wp.atomic_min(bwb_n, a, ka, bwb[b, ka])
            # A sleeper is not moved and loses nothing: what this piece got from it goes on to the sleeper's
            # carrier (between two awake pieces nothing is left over: both were moved by the same share)
            if b_sleep != 0:
                wp.atomic_add(bpf, a, (pa + pt)*(-1.0/wA))
                wp.atomic_add(bptq, a, wp.cross(ra, (pa + pt)*(-1.0/wA)))
            else:
                wp.atomic_add(bpf, a, (pa + pt)*hand)
                wp.atomic_add(bptq, a, wp.cross(ra, (pa + pt)*hand))
        if two_sided != 0 and b_sleep == 0:
            wp.atomic_add(bstick, b, wp.vec4i(stuck, 1 - stuck, rstuck, 1 - rstuck))
            wp.atomic_add(bdx, b, (pa + pt)*(-binvm[b]/wB))
            wp.atomic_add(bdth, b, (ang_corr(binvi[b], bq[b], rb, -(pa + pt)) - ang_turn(binvi[b], bq[b], couple))*(1.0/wB))
            wp.atomic_add(bcnt, b, 1)
            ca = bcar[a]
            if ca >= 0:
                if n[2] < -0.1:
                    wp.atomic_max(bcar_n, b, ca)
                else:
                    wp.atomic_max(bcar_s, b, ca)
                if ca < ON_FLOOR:
                    wp.atomic_max(bpj, b, ca)
            kb = bin_of(-n)
            if kb >= 0:
                if bwb[a, kb] < 0.5*BACK_NONE:
                    wp.atomic_min(bwb_n, b, kb, bwb[a, kb])
            if a_sleep != 0:
                wp.atomic_add(bpf, b, (pa + pt)*(1.0/wB))
                wp.atomic_add(bptq, b, wp.cross(rb, (pa + pt)*(1.0/wB)))
            else:
                wp.atomic_add(bpf, b, (pa + pt)*hand)
                wp.atomic_add(bptq, b, wp.cross(rb, (pa + pt)*hand))


@wp.kernel
def contact_world(sph_x: wp.array(dtype=wp.vec3), sph_r: wp.array(dtype=float), sph_body: wp.array(dtype=int),
                  bx: wp.array(dtype=wp.vec3), bq: wp.array(dtype=wp.quat), bxo: wp.array(dtype=wp.vec3),
                  bqo: wp.array(dtype=wp.quat), bvp: wp.array(dtype=wp.vec3), binvm: wp.array(dtype=float),
                  binvi: wp.array(dtype=wp.vec3), bmu_floor: wp.array(dtype=float), bmu: wp.array(dtype=float),
                  bmur: wp.array(dtype=float), brest: wp.array(dtype=float), asleep: wp.array(dtype=int),
                  bax: wp.array(dtype=wp.vec3), baq: wp.array(dtype=wp.quat), bdrag: wp.array(dtype=int), v_still: float,
                  dep: float,
                  coll_type: wp.array(dtype=int), coll_size: wp.array(dtype=wp.vec3), coll_x: wp.array(dtype=wp.vec3),
                  coll_q: wp.array(dtype=wp.quat), coll_v: wp.array(dtype=wp.vec3), coll_w: wp.array(dtype=wp.vec3),
                  coll_mu: wp.array(dtype=float), coll_who: wp.array(dtype=int), cells: wp.array2d(dtype=int), org: wp.vec3,
                  inv_cs: float, nx: int, ny: int,
                  dt: float, kin: float, v_bounce: float, v_wake: float, wake_pen: float, first: int,
                  bdx: wp.array(dtype=wp.vec3), bdth: wp.array(dtype=wp.vec3), bcnt: wp.array(dtype=int),
                  bdv: wp.array(dtype=wp.vec3), bsup: wp.array(dtype=wp.vec3), bsupn: wp.array(dtype=int),
                  bwake: wp.array(dtype=int), bstick: wp.array(dtype=wp.vec4i), coll_f: wp.array(dtype=wp.vec3),
                  coll_tq: wp.array(dtype=wp.vec3), bcar_n: wp.array(dtype=int),
                  bcar_s: wp.array(dtype=int), bsoft: wp.array(dtype=float),
                  coll_pri: wp.array(dtype=int), brdx: wp.array(dtype=wp.vec3), brth: wp.array(dtype=wp.vec3),
                  bsn: wp.array(dtype=int), bA: wp.array(dtype=wp.mat33), coll_invm: wp.array(dtype=float),
                  nb_body: wp.array(dtype=int), bwb: wp.array2d(dtype=float), bwb_n: wp.array2d(dtype=float),
                  dt_tick: float, kappa: float, coll_bid: wp.array(dtype=int),
                  sph_mj: wp.array2d(dtype=int), sph_mf: wp.array2d(dtype=int), coll_roll: wp.array(dtype=int),
                  btop: wp.array(dtype=float), coll_lcom: wp.array(dtype=wp.vec3), coll_iloc: wp.array(dtype=wp.mat33),
                  bsh: wp.array(dtype=int), coll_vn: wp.array(dtype=float)):
    """Spheres against the floor plane z = 0 and the rigid shapes of the cell they are in.

    NET SPEED. Whether the shape a piece lies on or under "moves" decides two things: a piece on a moving support does
    not fall asleep, and a moving shape wakes a sleeping piece it touches. The shape's velocity of the moment cannot
    decide it. A light tool lying on a few grains, or a pail with shards in it, is never still while the pieces under
    it are awake: it trembles on them by micrometres at 1 to 6 mm/s, so the pieces never sleep, so it goes on
    trembling (a counter brush on table crumbs, a broom on floor grit, a mug in the pail), and the task's rest test
    stays false. Both read coll_vn, the shape's net speed over the last tenth of a second (how far its centre went and
    how far it turned, over that time; world._push_poses): a thing that trembles in place has none, a pan carried at a
    quarter of a millimetre a second has. The limit for a support is the pieces' own rest criterion, rest_disp in
    rest_time. Each contact's correction is divided by the number of the piece's contacts (see contact_pairs), and the
    shape gets the same share back: Newton's third law with what the piece actually got. (Added at full size, a rice
    grain lying on two of its spheres pressed on a dustpan with twice its weight: 0.2731 N from 0.1372 N, measured.)"""
    s = wp.tid()
    a = sph_body[s]
    xs = sph_x[s]
    rs = sph_r[s]
    a_sleep = asleep[a]
    # ---- floor
    pen = rs - xs[2]
    if pen > 0.0 and a_sleep == 0:
        n = wp.vec3(0.0, 0.0, 1.0)
        cp = xs - n*rs
        ra = cp - bx[a]
        wa = gen_w(binvm[a], binvi[a], bq[a], ra, n)
        dsub = (bx[a] + ra) - (bxo[a] + wp.quat_rotate(bqo[a], wp.quat_rotate_inv(bq[a], ra)))
        lam = pen_limited(pen, -dsub[2], dep)/wa                    # the approach at the contact (see contact_pairs)
        pa = n*lam
        dpa = (bx[a] + ra) - (bax[a] + wp.quat_rotate(baq[a], wp.quat_rotate_inv(bq[a], ra)))
        if bdrag[a] != 0:
            dpa = (bx[a] + ra) - (bxo[a] + wp.quat_rotate(bqo[a], wp.quat_rotate_inv(bq[a], ra)))
        rel_t = dpa - n*wp.dot(dpa, n)
        lt = wp.length(rel_t)
        pt = wp.vec3(0.0, 0.0, 0.0)
        stuck = int(1)
        if lt > 1.0e-12:
            tdir = rel_t/lt
            mu_s = bmu_floor[a]
            ft, stuck = friction_imp(lt, gen_w(binvm[a], binvi[a], bq[a], ra, tdir), lam, mu_s, kin*mu_s)
            pt = tdir*(-ft)
        th = small_rot(bq[a], baq[a])
        gain = float(0.3)
        if bdrag[a] != 0:
            th = small_rot(bq[a], bqo[a])
            gain = 1.0
        lth = wp.length(th)
        wra = float(0.0)
        if lth > 1.0e-12:
            wra = ang_w(binvi[a], bq[a], th/lth)
        couple, rstuck = roll_couple(th, wra, 0.0, lam, bmur[a], rs, gain)
        wF = wp.max(1.0, wp.trace(bA[a]))
        wp.atomic_add(bstick, a, wp.vec4i(stuck, 1 - stuck, rstuck, 1 - rstuck))
        wp.atomic_add(bdx, a, (pa + pt)*(binvm[a]/wF))
        wp.atomic_add(bdth, a, (ang_corr(binvi[a], bq[a], ra, pa + pt) + ang_turn(binvi[a], bq[a], couple))*(1.0/wF))
        wp.atomic_add(bcnt, a, 1)
        wp.atomic_max(bcar_n, a, ON_FLOOR)
        wp.atomic_min(bwb_n, a, 4, 0.0)
        if first != 0:
            wp.atomic_add(bsupn, a, 1)
            vn = bvp[a][2]
            if vn < -v_bounce:
                wp.atomic_add(bdv, a, n*(-brest[a]*vn))
    # ---- rigid shapes
    c = cell_of(xs, org, inv_cs, nx, ny)
    if c < 0:
        return
    xo = bxo[a] + wp.quat_rotate(bqo[a], wp.quat_rotate_inv(bq[a], xs - bx[a]))
    seen0 = int(0)
    seen1 = int(0)
    for k in range(KC):
        j = cells[c, k]
        if j < 0:
            break
        if coll_who[j] == 1:
            continue
        slot = int(-1)
        if sph_mj[s, 0] == j:
            slot = 0
        elif sph_mj[s, 1] == j:
            slot = 1
        face = int(-1)
        if slot >= 0:
            face = sph_mf[s, slot]
        wheel = coll_roll[j]
        mem = face
        if wheel != 0:
            face = -1
        pen_c, n, fo = shape_hit(coll_type[j], coll_size[j], coll_x[j], coll_q[j], coll_v[j], coll_w[j], xs, rs, xo, dt, face)
        over = int(0)
        if wheel != 0 and pen_c > 0.0:
            # What the wheel does to this piece is decided at the first overlap and remembered like a box's face. And a
            # piece that a wheel or the chassis is pushing and that comes under it all the same (the contact turns
            # flat: the piece did not get out of the way) is passed over from then on, for as long as it overlaps:
            # without that it was pushed again the moment it came out behind, from deep inside, at the depenetration
            # speed (a bead left a base moving at 0.3 m/s at 7 m/s, a mug piece rose 0.1 m).
            if mem == WHEEL_OVER:
                over = 1
            elif wheel == 1:
                over = wheel_passes(mem, bsoft[a], btop[a], coll_size[j][0], bwb[a, 4])
            if over == 0:
                if wp.length(wp.vec3(n[0], n[1], 0.0)) < WHEEL_FLAT:
                    over = 1
            fo = WHEEL_PUSH
            if over != 0:
                fo = WHEEL_OVER
        # the face this sphere is at the box by is remembered while they overlap (two boxes a sphere; see shape_hit)
        if fo >= 0 and pen_c > 0.0:
            if slot < 0:
                if sph_mj[s, 0] < 0:
                    slot = 0
                elif sph_mj[s, 1] < 0:
                    slot = 1
            if slot >= 0:
                sph_mj[s, slot] = j
                sph_mf[s, slot] = fo
                if slot == 0:
                    seen0 = 1
                else:
                    seen1 = 1
        if pen_c <= 0.0:
            # A SLEEPING PIECE AND A SHAPE THAT HAS JUST LEFT IT. A sleeping piece stays where it is in the room; a
            # shape that moves wakes it (below), but that was asked only where the two overlap, and since a shape is
            # known to move by its NET speed, which lags, the floor of a hand dustpan tipping away had gone a tenth of a
            # millimetre from the crumbs asleep on it before its net speed reached v_wake: 37 crumbs were left asleep in
            # the air where the pan's floor had been, and counted as "in the pan" (a fifth of the load never left a pan
            # tipped 60 degrees; traced on every tenth of a second of the tip). So the question is asked out to
            # WAKE_REACH: a shape that near a sleeping piece wakes it once it moves faster than v_wake; and a shape
            # UNDER the piece (its face looks upward at it) that has gone further from it than the wake-up tolerance
            # wakes it once it moves faster than what counts as still for a support (a quarter of v_wake, sup_still in
            # body_finish). A woken piece with nothing under it falls; one that still lies on something sleeps again
            # within a tenth of a second. (The second question was first asked of every shape, at a tenth of v_wake: a
            # lobby broom lying on debris, trembling at 0.1 to 0.25 mm/s net, then woke the grains beside its handle on
            # every substep, and the evaluator found that floor at rest in 10 of 16 readings.)
            if a_sleep != 0 and wheel == 0 and pen_c > -WAKE_REACH:
                if coll_vn[j] > v_wake or (n[2] > 0.1 and pen_c < -wake_pen and coll_vn[j] > 0.25*v_wake):
                    wp.atomic_add(bwake, a, 1)
            continue
        if wheel != 0:
            if over != 0:
                continue
            nh = wp.vec3(n[0], n[1], 0.0)
            n = nh/wp.length(nh)
        wp.atomic_add(bsh, a, 1)
        cp = xs - n*rs
        vc = coll_v[j] + wp.cross(coll_w[j], cp - coll_x[j])
        if wheel == 1:
            vc = coll_v[j]
        # A piece that is held. The loose pieces are moved by positional corrections scaled by their own mass, and
        # what such a correction puts on a shape can never be more than about the piece's mass times 1 m/s in a
        # substep: a 3.5 kg barrel set on a heap went down through it to 1 mm above the floor, a fingertip pad
        # closed through a 5 g pencil, and a wheel squeezed grains out sideways at up to 2.8 m/s instead of
        # riding over them. But a piece that is held from the other side (it lies on the
        # floor, or against a fixed shape, or on pieces that do, or between two moving shapes) cannot give way,
        # whatever it weighs: it is a strut. A moving shape pressing on such a piece is therefore pushed back by a
        # spring, critically damped, whose stiffness is set by what the shape's own body (and the body behind the
        # piece) can be stopped with in a tick, kappa/((w_shape + w_behind) dt^2), shared among the held pieces
        # the shape presses on. What holds the piece has to be another rigid body than the shape's own (back_key): a
        # pail does not squeeze what lies between its own walls.
        # A piece that the shape presses against the ROOM'S FLOOR is not moved by the shape at all: it cannot go
        # down. (Corrected as above, it was pushed into the floor by the shape and out of it by the floor in
        # every substep, each push at its own point.) Against any other holder (two finger pads, a pan and a
        # table) the piece is corrected as above.
        # The robot's CHASSIS (the skirt box of its base) pushes what stands taller than its clearance along the floor and
        # nothing else: it does not press a piece down, is not lifted by one and is not held up on one. (As an
        # ordinary shape its leading lower edge met the top of a mug piece 40 mm tall at 45 degrees; first the
        # spring of a held piece threw it back with 20,000 N, and with that spring made vertical the base, a
        # fifth of whose weight then stood on the piece, was held by the friction of it: in a band of three broken
        # mugs it stopped after 0.2 m. A 217 kg base does not stop for
        # a kilogram of crockery.)
        # A WHEEL of the robot is a case of its own. It rolls over a piece that lies on the floor and is low enough to
        # come under it: a sphere whose top is lower than 0.16 of the wheel's radius (4.9 mm under a caster, 8.1 mm
        # under a drive wheel; such a sphere meets the rim within 32 degrees of its lowest point). The piece is not
        # moved and the wheel is not lifted (a tyre gives; this cylinder does not have to). A taller piece is
        # pushed like by any shape. Three other answers were tried. Lifting the robot onto the grain
        # with the spring below: a rigid wheel has to climb a rigid grain, and the base lost a third of its way
        # over a band of rice; while the spring's force still went to the caster's own 0.2 kg link, the caster
        # whirled on its swivel at 270 rad/s and its rim hit pieces at 10 m/s. Rolling over only while the contact
        # is within 32 degrees of the lowest point: the piece, by then deep in the wheel, was thrown out behind it
        # at 3.5 m/s the moment the contact left that cone.
        pinned = int(0)
        clamped = int(0)
        if wheel == 0 and coll_invm[j] > 0.0 and bsoft[a] == 0.0:
            kb = bin_of(-n)
            if kb >= 0:
                key = bwb[a, kb]
                # (a sphere that is itself sunk a third of its radius into the room's floor is not standing on the
                # floor and nothing stands on it: a grain the bristles have pressed into the floor in front of a
                # dustpan's lip must not lift the pan, see floor_apply)
                if key < 0.5*BACK_NONE and back_body(key) != coll_bid[j] and rs - xs[2] < SUNK*rs:
                    # What the shape's body can be stopped with AT THIS POINT: its inverse mass and what its turning
                    # adds there, and shared among ALL the held pieces the body presses on, through whichever of
                    # its shapes. With the inverse mass alone, shared shape by shape, a broom let go lying 30 mm
                    # above debris, which meets the grains with its block, its ferrule and its handle and far from
                    # its centre of mass, got 500 to 6,000 N for single ticks, bounced higher each time and left the
                    # room.
                    rcm = cp - (coll_x[j] + wp.quat_rotate(coll_q[j], coll_lcom[j]))
                    # Along which line the piece is a strut: toward what holds it. For the room's floor that is the
                    # vertical, whatever the direction the shape touches the piece in: a shape that runs into a standing
                    # piece from the side does not press it into the floor, it pushes it along. (With the spring along
                    # the contact's own direction and the shape's whole speed, the leading lower edge of the robot's
                    # chassis met the top of a mug piece 40 mm tall at 45 degrees, was pushed back and up with 20,000 to
                    # 30,000 N, and the base stood still in a band of mug pieces.) Between two rigid bodies the line is
                    # the contact's.
                    hs = n
                    if key < 0.5:
                        hs = wp.vec3(0.0, 0.0, -1.0)
                    along = wp.max(0.0, wp.dot(n, hs))
                    rxn = wp.quat_rotate_inv(coll_q[j], wp.cross(rcm, hs))
                    wrot = wp.dot(rxn, coll_iloc[j]*rxn)
                    wr = (back_invm(key) + coll_invm[j] + wrot)*float(wp.max(1, nb_body[coll_bid[j]]))
                    u = wp.dot(vc, hs)
                    fn = wp.max(0.0, (kappa*pen_c*along/(dt_tick*dt_tick) + 2.0*wp.sqrt(kappa)*u/dt_tick)/wr)
                    if along <= 0.0:
                        fn = 0.0
                    fb = hs*(-fn)
                    vt = vc - hs*u
                    sp = wp.length(vt)
                    if sp > 1.0e-9:
                        fb = fb - vt*(wp.min(bmu[a], coll_mu[j])*fn/wp.max(sp, 1.0e-3))
                    wp.atomic_add(coll_f, j, fb)
                    wp.atomic_add(coll_tq, j, wp.cross(cp - coll_x[j], fb))
                    if key < 0.5:
                        pinned = 1
                    else:
                        clamped = 1
        if pinned != 0:
            # the piece cannot go down; along the floor it is pushed by what the contact has of that direction
            nh = wp.vec3(n[0], n[1], 0.0)
            lh = wp.length(nh)
            if lh > 0.05:
                n = nh/lh
                pen_c = pen_c*lh
                cp = xs - n*rs
                pinned = 0
        if a_sleep != 0:
            # a shape that moves, or one that has crept this deep into a sleeping piece, wakes it
            # (deep in it: only while the shape moves at all; a shape at rest in a sleeper leaves it asleep)
            if coll_vn[j] > v_wake or (pen_c > wp.max(wake_pen, 0.2*rs) and coll_vn[j] > 0.1*v_wake):
                wp.atomic_add(bwake, a, 1)
            continue
        ra = cp - bx[a]
        # A piece that yields (crumpled paper). Where a rigid shape presses on it the paper gives, so the push is
        # along the shape's normal at the wad's centre, not at each lump; each sphere in contact is a spring, half
        # critically damped, up to half its radius, and rigid beyond. The springs' pushes are SUMMED with the
        # piece's other corrections, not averaged: two pads closing on a wad each push with their own force. (A
        # rigid lumpy ball between two pads was squeezed out sideways like a pip.)
        ks = bsoft[a]
        soft = int(0)
        if ks > 0.0:
            soft = 1
            n = soft_normal(coll_type[j], coll_size[j], coll_x[j], coll_q[j], bx[a], n)
        wS = wp.max(1.0, wp.trace(bA[a]))
        wa = gen_w(binvm[a], binvi[a], bq[a], ra, n)
        # the approach at the contact (see contact_pairs): the piece's turning counts, and the shape's own motion
        csub = (bx[a] + ra) - (bxo[a] + wp.quat_rotate(bqo[a], wp.quat_rotate_inv(bq[a], ra))) - vc*dt
        close_c = -wp.dot(csub, n)
        lam = pen_limited(pen_c, close_c, dep)/wa
        if wheel != 0:
            # a wheel or the chassis moves a piece as fast as it comes at it and no faster: an overlap that is already
            # there (a piece that could not get out of the way) is not pushed out at the depenetration speed on top.
            # (With it, a blocked mug piece left a base moving at 0.3 m/s at 1.8 m/s and knocked beads on at 5 m/s.)
            lam = pen_limited(pen_c, close_c, 0.0)/wa
        if pinned != 0:
            lam = 0.0
        if soft != 0:
            crush = 0.5*rs
            f_n = wp.max(0.0, ks*wp.min(pen_c, crush) - wp.sqrt(ks/binvm[a])*wp.dot(bvp[a] - vc, n))
            lam_s = f_n*dt*dt
            if pen_c > crush:
                lam_s = lam_s + 0.25*pen_limited(pen_c - crush, close_c, dep)/wa
            lam = wp.min(lam, lam_s)
        pa = n*lam
        # a shape at rest holds the piece at its anchor; against a moving shape the slip is that of this substep
        rel = (bx[a] + ra) - (bax[a] + wp.quat_rotate(baq[a], wp.quat_rotate_inv(bq[a], ra)))
        moving = int(0)
        if wp.length(vc) > v_still or bdrag[a] != 0:
            moving = 1
            rel = (bx[a] + ra) - (bxo[a] + wp.quat_rotate(bqo[a], wp.quat_rotate_inv(bq[a], ra))) - vc*dt
        rel_t = rel - n*wp.dot(rel, n)
        lt = wp.length(rel_t)
        pt = wp.vec3(0.0, 0.0, 0.0)
        stuck = int(1)
        if lt > 1.0e-12:
            tdir = rel_t/lt
            mu_s = wp.min(bmu[a], coll_mu[j])
            if coll_pri[j] != 0:
                mu_s = coll_mu[j]            # a fingertip pad: the rubber governs, as in the rigid world
            ft, stuck = friction_imp(lt, gen_w(binvm[a], binvi[a], bq[a], ra, tdir), lam, mu_s, kin*mu_s)
            pt = tdir*(-ft)
        if soft != 0 or clamped != 0:
            # A piece CLAMPED between two rigid bodies (two finger pads on a shard) is carried in the same way: the
            # two opposed friction forces are a pair, and each computed alone at its contact point takes back only
            # 1/(1 + m r^2/I) of a slip (the pads stood at their 6 N on a shard and on a pencil, rose, and the piece
            # stayed on the floor).
            # A yielding ball lies on the shape with a patch and is carried by it: friction holds its CENTRE against
            # the shape (its turning is the patch couple's business, below). With friction at each contact point
            # and the corrections averaged, two opposed pads took back only 1/(1 + m r^2/I) of a slip per substep,
            # and a wad held by friction alone crept down between rising pads at 6 mm/s (= g dt (m w_t - 1),
            # measured) until it left them.
            relc = bx[a] - bax[a]
            if moving != 0:
                relc = bx[a] - bxo[a] - vc*dt
            relc_t = relc - n*wp.dot(relc, n)
            ltc = wp.length(relc_t)
            pt = wp.vec3(0.0, 0.0, 0.0)
            stuck = int(1)
            if ltc > 1.0e-12:
                mu_c = wp.min(bmu[a], coll_mu[j])
                if coll_pri[j] != 0:
                    mu_c = coll_mu[j]
                ftc, stuck = friction_imp(ltc, binvm[a], lam, mu_c, kin*mu_c)
                pt = relc_t*(-ftc/ltc)
        th = small_rot(bq[a], baq[a])
        gain = float(0.3)
        if moving != 0:
            th = small_rot(bq[a], bqo[a]) - coll_w[j]*dt
            gain = 1.0
        lth = wp.length(th)
        wra = float(0.0)
        if lth > 1.0e-12:
            wra = ang_w(binvi[a], bq[a], th/lth)
        mur_c = bmur[a]
        grip = wp.vec3(0.0, 0.0, 0.0)
        if clamped != 0 and soft == 0:
            # Friction at the contact point of a rigid piece held between two bodies also resists its TURNING there;
            # the centre's friction above leaves that out. Two parts.
            # (1) The normal force of this contact turns the piece when it does not point at the piece's centre
            # (two pads on an irregular shard are never exactly opposite). Friction at the point takes that couple
            # up as long as the whole contact force can pass through the centre, that is while the centre lies
            # within the friction cone: a couple of at most mu N |ra . n|. It is applied here, in the same pass as
            # the normal force. Taken back one substep later instead (as the turn it had become) the piece crept
            # round at tau dt / I for as long as it was held: 90 degrees a second for a 32 g shard under 5 N a pad,
            # which shortened its chord faster than the pads closed, and they lost it 0.1 s into the rise. Without
            # any of this the same shard left the pads 0.9 s into the rise.
            # (2) What is left of the cone resists a turn the piece already has, at the arm |axis x ra|.
            mu_t = wp.min(bmu[a], coll_mu[j])
            if coll_pri[j] != 0:
                mu_t = coll_mu[j]
            tq = wp.cross(ra, pa)
            ltq = wp.length(tq)
            cap = mu_t*lam*wp.abs(wp.dot(ra, n))
            left = float(1.0)
            if ltq > 1.0e-20:
                share = wp.min(1.0, cap/ltq)
                grip = tq*(-share)
                left = wp.max(0.0, 1.0 - ltq/wp.max(cap, 1.0e-20))
            if lth > 1.0e-12:
                mur_c = wp.max(mur_c, left*mu_t*wp.length(wp.cross(th, ra))/(1.4*rs*lth))
        if soft != 0:
            # a ball that yields lies against a flat shape on a patch, not on a point: the couple that resists its
            # turning has the patch's radius for an arm (never more than the shape's own smallest half size)
            pc = wp.min(pen_c, 0.5*rs)
            patch = wp.sqrt(wp.max(0.0, 2.0*rs*pc - pc*pc))
            if coll_type[j] == BOX:
                patch = wp.min(patch, wp.min(coll_size[j][0], wp.min(coll_size[j][1], coll_size[j][2])))
            mur_c = wp.max(mur_c, patch/(1.4*rs))
        couple, rstuck = roll_couple(th, wra, 0.0, lam, mur_c, rs, gain)
        wp.atomic_add(bstick, a, wp.vec4i(stuck, 1 - stuck, rstuck, 1 - rstuck))
        if soft != 0:
            # the spring is a force and forces add up; friction and the rolling couple are corrections that each
            # take back the whole slip of the substep, and are averaged with the piece's other contacts (summed,
            # two pads took the slip back twice over and the wad shot out sideways at 1 m/s under 5 N a pad)
            wp.atomic_add(brdx, a, pa*binvm[a])
            wp.atomic_add(brth, a, ang_corr(binvi[a], bq[a], ra, pa))
            wp.atomic_add(bsn, a, 1)
            wp.atomic_add(bdx, a, pt*(binvm[a]/wS))
            wp.atomic_add(bdth, a, ang_turn(binvi[a], bq[a], couple)*(1.0/wS))
            wp.atomic_add(bcnt, a, 1)
        else:
            wp.atomic_add(bdx, a, (pa + pt)*(binvm[a]/wS))
            wp.atomic_add(bdth, a, (ang_corr(binvi[a], bq[a], ra, pa + pt) + ang_turn(binvi[a], bq[a], couple + grip))*(1.0/wS))
            wp.atomic_add(bcnt, a, 1)
        if first != 0:
            wp.atomic_add(bsup, a, wp.vec3(coll_vn[j], 0.0, 0.0))      # net speeds, summed (never negative)
            wp.atomic_add(bsupn, a, 1)
            vn = wp.dot(bvp[a] - vc, n)
            if vn < -v_bounce and soft == 0:
                wp.atomic_add(bdv, a, n*(-brest[a]*vn))
        if n[2] > 0.1:
            wp.atomic_max(bcar_n, a, j)
        else:
            wp.atomic_max(bcar_s, a, j)
        ks2 = bin_of(n)
        if ks2 >= 0 and wheel == 0:
            wp.atomic_min(bwb_n, a, ks2, back_key(coll_invm[j], coll_bid[j]))
        # what the shape gets back: the share the piece got (a spring's push, which is summed on the piece, in full)
        f = (pa + pt)*(-1.0/(wS*dt*dt))
        if soft != 0:
            f = pa*(-1.0/(dt*dt)) + pt*(-1.0/(wS*dt*dt))
        wp.atomic_add(coll_f, j, f)
        wp.atomic_add(coll_tq, j, wp.cross(cp - coll_x[j], f))
    # a box this sphere no longer overlaps is forgotten
    if seen0 == 0:
        sph_mj[s, 0] = -1
    if seen1 == 0:
        sph_mj[s, 1] = -1


@wp.kernel
def carried_reaction(bx: wp.array(dtype=wp.vec3), binvm: wp.array(dtype=float), asleep: wp.array(dtype=int),
                     bcnt: wp.array(dtype=int), bcar: wp.array(dtype=int), bpj: wp.array(dtype=int),
                     bpf: wp.array(dtype=wp.vec3), bptq: wp.array(dtype=wp.vec3), coll_x: wp.array(dtype=wp.vec3),
                     relax: float, dt: float, gravity: float, first: int, coll_f: wp.array(dtype=wp.vec3),
                     coll_tq: wp.array(dtype=wp.vec3)):
    """The weight of what lies still. A sleeping piece takes no part in contacts, so it would press on nothing: a
    dustpan with 14 g of sleeping rice stood on the floor with its own weight only (measured). Each piece remembers
    what carries it (bcar): the room's floor if it touches it, else a shape that pushes it upward, else the carrier
    of a piece under it, and only if there is none of these (a grain wedged between a wall and its neighbours) a
    shape or a neighbour's carrier it merely touches. While it sleeps, its weight acts on that shape at the piece's
    own centre, which is statics. An awake piece lying on a sleeper hands what it got from the sleeper to the
    sleeper's carrier (contact_pairs), so that a heap and what carries it keep their momentum together."""
    b = wp.tid()
    c = bcar[b]
    # what the piece's contacts with other pieces hand on (contact_pairs) goes to the piece's own carrier; a piece
    # that has none yet (it has just landed on a heap) hands it to the carrier of what it touches; what is carried
    # by the room's floor hands it to the floor, which takes no force
    j = c
    if j < 0:
        j = bpj[b]
    if j >= 0 and j < ON_FLOOR and asleep[b] == 0:
        w = relax/(dt*dt)
        f = bpf[b]*w
        wp.atomic_add(coll_f, j, f)
        wp.atomic_add(coll_tq, j, bptq[b]*w + wp.cross(bx[b] - coll_x[j], f))
    bpj[b] = -1
    bpf[b] = wp.vec3(0.0, 0.0, 0.0)
    bptq[b] = wp.vec3(0.0, 0.0, 0.0)
    if first != 0 and asleep[b] != 0 and c >= 0 and c < ON_FLOOR and binvm[b] > 0.0:
        wgt = wp.vec3(0.0, 0.0, -gravity/binvm[b])
        wp.atomic_add(coll_f, c, wgt)
        wp.atomic_add(coll_tq, c, wp.cross(bx[b] - coll_x[c], wgt))


@wp.kernel
def contact_rods(grid: wp.uint64, t_idx: wp.array(dtype=int), t_rmax: float, n_seg: int,
                 x: wp.array2d(dtype=wp.vec3), xo: wp.array2d(dtype=wp.vec3), rod_tool: wp.array(dtype=int),
                 rod_n: wp.array(dtype=int), rod_seg: wp.array(dtype=float), rod_m: wp.array(dtype=float),
                 rod_rad: wp.array(dtype=float), tool_on: wp.array(dtype=int),
                 sph_x: wp.array(dtype=wp.vec3), sph_r: wp.array(dtype=float), sph_body: wp.array(dtype=int),
                 bx: wp.array(dtype=wp.vec3), bq: wp.array(dtype=wp.quat), bxo: wp.array(dtype=wp.vec3),
                 bqo: wp.array(dtype=wp.quat), binvm: wp.array(dtype=float), binvi: wp.array(dtype=wp.vec3),
                 bmu_b: wp.array(dtype=float), bmur: wp.array(dtype=float), asleep: wp.array(dtype=int), wake_pen: float,
                 kin: float, dep: float, wake_move: float, brdx: wp.array(dtype=wp.vec3), brth: wp.array(dtype=wp.vec3),
                 brn: wp.array(dtype=int), brn_prev: wp.array(dtype=int), bwake: wp.array(dtype=int),
                 rdx: wp.array2d(dtype=wp.vec3), rcnt: wp.array2d(dtype=int), rcnt_prev: wp.array2d(dtype=int)):
    """Bristle segments (capsules) against debris spheres."""
    tid = wp.tid()
    r = tid/n_seg
    i = tid % n_seg + 1
    if i + 1 >= rod_n[r] or tool_on[rod_tool[r]] == 0:
        return
    pa0 = x[r, i]
    pb0 = x[r, i + 1]
    ab = pb0 - pa0
    ab2 = wp.dot(ab, ab)
    rc = rod_rad[r]
    w_node = 1.0/rod_m[r]
    wa = w_node
    if i <= 1:
        wa = 0.0
    wb = w_node
    mid = 0.5*(pa0 + pb0)
    query = wp.hash_grid_query(grid, mid, 0.5*rod_seg[r] + rc + t_rmax)
    jj = int(0)
    while wp.hash_grid_query_next(query, jj):
        s = t_idx[jj]
        c = sph_x[s]
        t = wp.clamp(wp.dot(c - pa0, ab)/ab2, 0.0, 1.0)
        p = pa0 + ab*t
        dvec = c - p
        dist = wp.length(dvec)
        rsum = sph_r[s] + rc
        if dist >= rsum or dist < 1.0e-9:
            continue
        n = dvec/dist
        pen = rsum - dist
        b = sph_body[s]
        b_sleep = asleep[b]
        # a bristle that moves wakes the sleeping piece it touches (as a moving shape does). Waking only on a deep
        # press left sleeping beads standing like posts in front of a slow stroke: at 0.125 m/s a cluster moves 31 um
        # a substep, is pushed back out by the sleeper each time and never gets 0.1 mm in (97 % of the beads left
        # where they lay)
        moved = (pa0 - xo[r, i])*(1.0 - t) + (pb0 - xo[r, i + 1])*t
        # (a deep press wakes only while the bristle moves at all: under a broom lying still on the floor thirty
        # pieces were kept awake for as long as they were watched)
        if b_sleep != 0 and ((pen > wake_pen and wp.length(moved) > 0.1*wake_move) or wp.length(moved) > wake_move):
            wp.atomic_add(bwake, b, 1)
        cp = c - n*sph_r[s]
        rb = cp - bx[b]
        wbody = gen_w(binvm[b], binvi[b], bq[b], rb, n)
        if b_sleep != 0:
            wbody = 0.0
        # the same on the bristle's side: a node touched by several pieces takes the sum of their reactions, each
        # with the node's inverse mass multiplied by the number of pieces on it (previous substep). With the reactions
        # averaged, a bank felt one grain however many it pushed: the angle broom carried 100 g of rice at the drag
        # of an empty stroke
        wrod = (1.0 - t)*(1.0 - t)*wa*float(wp.max(1, rcnt_prev[r, i])) + t*t*wb*float(wp.max(1, rcnt_prev[r, i + 1]))
        # A piece is pushed by every cluster that touches it: the corrections add up, each with the piece's inverse
        # mass multiplied by the number of clusters on it (counted in the previous substep), so that a grain as
        # light as a cluster node still gets the average and a 30 g shard gets the sum. Averaged with the piece's
        # other contacts, fifty clusters push a shard as one does, and a push broom leaves 64 % of a broken mug
        # where it lies.
        split = float(wp.max(1, brn_prev[b]))
        wsum = wbody*split + wrod
        if wsum <= 0.0:
            continue
        dpb = (bx[b] + rb) - (bxo[b] + wp.quat_rotate(bqo[b], wp.quat_rotate_inv(bq[b], rb)))
        drod = (pa0 - xo[r, i])*(1.0 - t) + (pb0 - xo[r, i + 1])*t
        rel = dpb - drod
        # an overlap left from earlier substeps (a grain caught under the bank) is pushed out at the depenetration
        # speed, not in one substep: a first leading stroke shot grains off at 8 m/s
        lam = pen_limited(pen, -wp.dot(rel, n), dep)/wsum
        pb = n*lam
        rel_t = rel - n*wp.dot(rel, n)
        lt = wp.length(rel_t)
        pt = wp.vec3(0.0, 0.0, 0.0)
        if lt > 1.0e-12:
            tdir = rel_t/lt
            mu_s = bmu_b[b]
            wtb = float(0.0)
            if b_sleep == 0:
                wtb = gen_w(binvm[b], binvi[b], bq[b], rb, tdir)
            ft, stuck = friction_imp(lt, wtb*split + wrod, lam, mu_s, kin*mu_s)
            pt = tdir*(-ft)
        if b_sleep == 0:
            wp.atomic_add(brdx, b, (pb + pt)*binvm[b])
            wp.atomic_add(brth, b, ang_corr(binvi[b], bq[b], rb, pb + pt))
            wp.atomic_add(brn, b, 1)
        if wa > 0.0:
            wp.atomic_add(rdx, r, i, (pb + pt)*(-wa*(1.0 - t)))
            wp.atomic_add(rcnt, r, i, 1)
        wp.atomic_add(rdx, r, i + 1, (pb + pt)*(-wb*t))
        wp.atomic_add(rcnt, r, i + 1, 1)


@wp.kernel
def body_iterate(bx: wp.array(dtype=wp.vec3), bq: wp.array(dtype=wp.quat), asleep: wp.array(dtype=int),
                 bdx: wp.array(dtype=wp.vec3), bdth: wp.array(dtype=wp.vec3), bcnt: wp.array(dtype=int),
                 brdx: wp.array(dtype=wp.vec3), brth: wp.array(dtype=wp.vec3), brn: wp.array(dtype=int),
                 brn_prev: wp.array(dtype=int), last: int,
                 btx: wp.array(dtype=wp.vec3), btth: wp.array(dtype=wp.vec3), btouch: wp.array(dtype=int), bA: wp.array(dtype=wp.mat33), relax: float,
                 bsn: wp.array(dtype=int), bAs: wp.array(dtype=wp.mat33)):
    """One solver pass: move the predicted pose by the corrections of this pass and add them to the totals of the
    substep (the velocity update uses the totals). Contacts with other pieces, the floor and rigid shapes arrive
    already shared out among the piece's contacts (contact_pairs); the bristles' corrections (shared out among the clusters on
    the piece) are added."""
    b = wp.tid()
    cnt = bcnt[b]
    nr = brn[b]
    if (cnt > 0 or nr > 0 or bsn[b] > 0) and asleep[b] == 0:
        dx = brdx[b]
        th = brth[b]
        if cnt > 0:
            dx = dx + bdx[b]*relax
            th = th + bdth[b]*relax
        bx[b] = bx[b] + dx
        q = bq[b]
        bq[b] = wp.normalize(q + wp.quat(th[0], th[1], th[2], 0.0)*q*0.5)
        btx[b] = btx[b] + dx
        btth[b] = btth[b] + th
        btouch[b] = btouch[b] + 1
    bdx[b] = wp.vec3(0.0, 0.0, 0.0)
    bdth[b] = wp.vec3(0.0, 0.0, 0.0)
    if asleep[b] == 0:
        bAs[b] = bA[b]                 # kept through sleep (see contact_pairs)
    bA[b] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    bcnt[b] = 0
    brdx[b] = wp.vec3(0.0, 0.0, 0.0)
    brth[b] = wp.vec3(0.0, 0.0, 0.0)
    if last != 0:
        brn_prev[b] = nr
    brn[b] = 0
    bsn[b] = 0


@wp.kernel
def body_finish(bx: wp.array(dtype=wp.vec3), bq: wp.array(dtype=wp.quat), bv: wp.array(dtype=wp.vec3),
                bw: wp.array(dtype=wp.vec3), bxo: wp.array(dtype=wp.vec3), bqo: wp.array(dtype=wp.quat),
                bvp: wp.array(dtype=wp.vec3), bres: wp.array(dtype=wp.vec3),
                broll: wp.array(dtype=float), bsize: wp.array(dtype=float), asleep: wp.array(dtype=int),
                btx: wp.array(dtype=wp.vec3), btth: wp.array(dtype=wp.vec3), btouch: wp.array(dtype=int),
                bdv: wp.array(dtype=wp.vec3), bsup: wp.array(dtype=wp.vec3), bsupn: wp.array(dtype=int),
                bwake: wp.array(dtype=int), bref: wp.array(dtype=wp.vec3), bqref: wp.array(dtype=wp.quat),
                btimer: wp.array(dtype=int), bdrag: wp.array(dtype=int), brmin: wp.array(dtype=float),
                banc: wp.array(dtype=wp.vec3), bqanc: wp.array(dtype=wp.quat), bstick: wp.array(dtype=wp.vec4i),
                bslip: wp.array(dtype=int), slip_steps: int, wake_steps: int, bair: wp.array(dtype=int), air_steps: int,
                slip_disp: float, dt: float,
                rest_disp: float, rest_steps: int, v_max: float, fault: wp.array(dtype=int),
                bcar: wp.array(dtype=int), bcar_n: wp.array(dtype=int), bcar_s: wp.array(dtype=int),
                bwb: wp.array2d(dtype=float), bwb_n: wp.array2d(dtype=float), sup_still: float):
    """Velocities are the carried state: the contact corrections of this substep (small vectors, exact in single
    precision) change the predicted velocity, and the position is advanced from the previous one with the part lost
    to rounding carried over (a piece 8 m from the origin moving 0.1 mm/s would otherwise not move at all, and a
    deceleration of 0.2 m/s2 would vanish in the rounding of a velocity taken from position differences)."""
    b = wp.tid()
    cnt = btouch[b]
    # what carries the piece (see carried_reaction): renewed while it is awake and touches something, forgotten once
    # it is in the air (below), kept as it was while it sleeps
    if asleep[b] == 0:
        car = bcar_n[b]
        if car < 0:
            car = bcar_s[b]
        if car >= 0:
            bcar[b] = car
    bcar_n[b] = -1
    bcar_s[b] = -1
    # what holds the piece from each side (contact_world, "a piece that is held"): renewed while it is awake and
    # touched, kept while it sleeps
    if asleep[b] == 0 and cnt > 0:
        for k in range(6):
            bwb[b, k] = bwb_n[b, k]
    for k in range(6):
        bwb_n[b, k] = BACK_NONE
    if asleep[b] != 0:
        if bwake[b] > 0:
            # a woken piece has not moved yet: it wakes its own sleeping neighbours only once it does
            asleep[b] = 0
            btimer[b] = wake_steps
            bref[b] = bx[b]
            bqref[b] = bq[b]
            banc[b] = bx[b]
            bqanc[b] = bq[b]
        bdrag[b] = 0
        bstick[b] = wp.vec4i(0, 0, 0, 0)
        bwake[b] = 0
        btx[b] = wp.vec3(0.0, 0.0, 0.0)
        btth[b] = wp.vec3(0.0, 0.0, 0.0)
        bdv[b] = wp.vec3(0.0, 0.0, 0.0)
        bsup[b] = wp.vec3(0.0, 0.0, 0.0)
        btouch[b] = 0
        bsupn[b] = 0
        return
    vv = bvp[b]
    w = bw[b]
    if cnt > 0:
        vv = vv + btx[b]/dt + bdv[b]/float(wp.max(1, bsupn[b]))
        w = w + btth[b]/dt
    sp = wp.length(vv)
    if not (sp < v_max):
        wp.atomic_add(fault, 0, 1)
        vv = wp.vec3(0.0, 0.0, 0.0)
    d = vv*dt + bres[b]
    x = bxo[b] + d
    bres[b] = d - (x - bxo[b])
    qo = bqo[b]
    q = wp.normalize(qo + wp.quat(w[0], w[1], w[2], 0.0)*qo*(0.5*dt))
    sup_moving = 0
    if bsupn[b] > 0:
        if wp.length(bsup[b])/float(bsupn[b]) > sup_still:        # mean net speed of what it lies on or under (contact_world)
            sup_moving = 1
    # Friction anchor (banc, bqanc): the pose at which static friction holds the piece. Contact slip is measured
    # from the anchor, so a push that friction can carry leaves a standing offset (micrometres) instead of a creep
    # that adds up substep after substep. The piece is sliding when it has left the anchor by more than the
    # pre-sliding distance slip_disp AND friction is saturated at every one of its contacts: then the anchor
    # follows it (bdrag = 1) and its contacts use the slip of the substep, kinetic friction against the actual
    # relative motion. Saturation alone is not sliding (a sphere wedged between three others is held by their
    # normals with friction at its limit), and neither is distance alone (deep in a pile the standing offset that
    # mobilises friction grows with the load). Rolling is treated the same way with the rolling couples: turned by
    # more than slip_disp of arc with every couple at its limit. A piece that has turned 0.05 rad or moved a fifth
    # of its smallest sphere re-anchors in any case.
    st = bstick[b]
    da = x - banc[b]
    dqa = q*wp.quat_inverse(bqanc[b])
    ha = wp.length(wp.vec3(dqa[0], dqa[1], dqa[2]))
    air = 0
    if cnt == 0:
        bair[b] = bair[b] + 1
        if bair[b] >= air_steps:       # a resting piece's lightest contact opens for a substep now and then
            air = 1
            bcar[b] = -1
            for k in range(6):
                bwb[b, k] = BACK_NONE
    else:
        bair[b] = 0
    drag = 0       # reason code, kept for diagnosis: 1 airborne, 2 moving support, 3 sliding, 4 rolling, 5 far
    if air != 0:
        drag = 1
    elif sup_moving != 0:
        drag = 2
    elif wp.length(da) > 0.2*brmin[b] or ha > 0.025:
        drag = 5
    else:
        # saturated at every contact for slip_steps substeps in a row: in a loaded pile the solver's residual makes
        # all contacts of a piece read saturated for a substep now and then, and letting go on that would throw
        # away the standing offsets that hold the pile (measured: 45 um/s of drift in a pile at rest)
        sliding = 0
        if st[0] == 0 and st[1] > 0 and wp.length(da) > slip_disp:
            sliding = 3
        elif st[2] == 0 and st[3] > 0 and 2.0*ha*brmin[b] > slip_disp:
            sliding = 4
        if sliding != 0:
            bslip[b] = bslip[b] + 1
            if bslip[b] >= slip_steps:
                drag = sliding
        else:
            bslip[b] = 0
    if drag != 0:
        banc[b] = x
        bqanc[b] = q
    bdrag[b] = drag
    # rest: no net motion over rest_steps substeps (on a moving support the piece stays awake)
    moved = wp.length(x - bref[b])
    dqr = q*wp.quat_inverse(bqref[b])
    turned = 2.0*wp.length(wp.vec3(dqr[0], dqr[1], dqr[2]))*bsize[b]
    if moved > rest_disp or turned > rest_disp or air != 0 or sup_moving != 0:
        bref[b] = x
        bqref[b] = q
        btimer[b] = 0
    else:
        btimer[b] = btimer[b] + 1
        if btimer[b] >= rest_steps:
            asleep[b] = 1
            vv = wp.vec3(0.0, 0.0, 0.0)
            w = wp.vec3(0.0, 0.0, 0.0)
            bres[b] = wp.vec3(0.0, 0.0, 0.0)
            banc[b] = x
            bqanc[b] = q
    bstick[b] = wp.vec4i(0, 0, 0, 0)
    bx[b] = x
    bq[b] = q
    bv[b] = vv
    bw[b] = w
    btx[b] = wp.vec3(0.0, 0.0, 0.0)
    btth[b] = wp.vec3(0.0, 0.0, 0.0)
    bdv[b] = wp.vec3(0.0, 0.0, 0.0)
    bsup[b] = wp.vec3(0.0, 0.0, 0.0)
    btouch[b] = 0
    bsupn[b] = 0
    bwake[b] = 0


# ------------------------------------------------------------------------------------------------ packed transfer
# On a GPU shared with other processes every host<->device copy that has to wait for queued kernels costs a whole
# time slice (measured: 9.6 ms a wait with five other jobs running, 0.03 ms a launch). One upload and one download a
# tick instead of thirteen.
@wp.kernel
def floor_deep(bx: wp.array(dtype=wp.vec3), bq: wp.array(dtype=wp.quat), binvm: wp.array(dtype=float),
               binvi: wp.array(dtype=wp.vec3), sph_body: wp.array(dtype=int), sph_local: wp.array(dtype=wp.vec3),
               sph_r: wp.array(dtype=float), asleep: wp.array(dtype=int), tol: float,
               fdx: wp.array(dtype=wp.vec3), fdth: wp.array(dtype=wp.vec3), fcnt: wp.array(dtype=int)):
    """The room's floor, once more and last: for every sphere of an awake piece that the pass has left deeper in the
    floor than `tol`, the push that sets that sphere back onto it, at the sphere (see floor_apply)."""
    s = wp.tid()
    b = sph_body[s]
    if asleep[b] != 0:
        return
    r = wp.quat_rotate(bq[b], sph_local[s])
    d = sph_r[s] - (bx[b][2] + r[2]) - tol
    if d <= 0.0:
        return
    up = wp.vec3(0.0, 0.0, 1.0)
    rc = r - up*sph_r[s]
    lam = d/gen_w(binvm[b], binvi[b], bq[b], rc, up)
    wp.atomic_add(fdx, b, up*(lam*binvm[b]))
    wp.atomic_add(fdth, b, ang_corr(binvi[b], bq[b], rc, up*lam))
    wp.atomic_add(fcnt, b, 1)


@wp.kernel
def shape_deep(mode: int, bx: wp.array(dtype=wp.vec3), bq: wp.array(dtype=wp.quat), bxo: wp.array(dtype=wp.vec3),
               bqo: wp.array(dtype=wp.quat), binvm: wp.array(dtype=float), binvi: wp.array(dtype=wp.vec3),
               sph_body: wp.array(dtype=int), sph_local: wp.array(dtype=wp.vec3), sph_r: wp.array(dtype=float),
               asleep: wp.array(dtype=int), bsoft: wp.array(dtype=float), bsh: wp.array(dtype=int),
               coll_type: wp.array(dtype=int), coll_size: wp.array(dtype=wp.vec3), coll_x: wp.array(dtype=wp.vec3),
               coll_q: wp.array(dtype=wp.quat), coll_v: wp.array(dtype=wp.vec3), coll_w: wp.array(dtype=wp.vec3),
               coll_who: wp.array(dtype=int), coll_roll: wp.array(dtype=int), coll_bid: wp.array(dtype=int),
               coll_pri: wp.array(dtype=int),
               cells: wp.array2d(dtype=int), org: wp.vec3, inv_cs: float, nx: int, ny: int, dt: float, tol: float,
               dep: float, bwb: wp.array2d(dtype=float), sph_mj: wp.array2d(dtype=int), sph_mf: wp.array2d(dtype=int),
               scnt: wp.array(dtype=int), sdx: wp.array(dtype=wp.vec3), sdth: wp.array(dtype=wp.vec3),
               svx: wp.array(dtype=wp.vec3), svth: wp.array(dtype=wp.vec3), coll_f: wp.array(dtype=wp.vec3),
               coll_tq: wp.array(dtype=wp.vec3)):
    """A rigid shape is hard, like the room's floor (floor_apply). Its push in contact_world is one correction among
    a piece's others, and a light piece under load loses to them: a grit grain of 11 mg under a 10 g shard on the
    bottom of a pail was pushed into the bottom by the shard in one substep and out of it by the bottom in the
    next, 75 micrometres each way, with a velocity of 0.3 m/s and no motion; the shards crept at 0.4 to 0.9 mm/s
    on such grains and no piece in the pail ever slept (traced on one grain).

    So after the pass, for every sphere of an awake rigid piece that is still deeper than `tol` in a shape it
    touched: the push that sets that sphere back onto the shape (mode 0 counts such spheres per piece, mode 1 adds
    the pushes, shared by that count). The push counts in the piece's velocity in full for what the sphere came
    toward the shape in this substep and at the depenetration speed for an overlap that was there before, and the
    shape gets that part back as a force. Not here: a piece squeezed between two rigid bodies (that is the springs'
    business in contact_world; two surfaces closer than the piece cannot both be cleared), wheels, crumpled paper,
    a fingertip pad (rubber: with this pass a 50 g shard pinched at its rim was pushed up out of the pinch like a pip
    and rode on the closed pads), and the room's floor, which has the last word after this."""
    s = wp.tid()
    a = sph_body[s]
    if asleep[a] != 0 or bsh[a] == 0 or bsoft[a] > 0.0:
        return
    r = wp.quat_rotate(bq[a], sph_local[s])
    xs = bx[a] + r
    rs = sph_r[s]
    c = cell_of(xs, org, inv_cs, nx, ny)
    if c < 0:
        return
    xo = bxo[a] + wp.quat_rotate(bqo[a], sph_local[s])
    for k in range(KC):
        j = cells[c, k]
        if j < 0:
            break
        if coll_who[j] == 1 or coll_roll[j] != 0 or coll_pri[j] != 0:
            continue
        face = int(-1)
        if sph_mj[s, 0] == j:
            face = sph_mf[s, 0]
        elif sph_mj[s, 1] == j:
            face = sph_mf[s, 1]
        pen_c, n, fo = shape_hit(coll_type[j], coll_size[j], coll_x[j], coll_q[j], coll_v[j], coll_w[j], xs, rs, xo, dt, face)
        d = pen_c - tol
        if d <= 0.0:
            continue
        kb = bin_of(-n)
        if kb >= 0:
            key = bwb[a, kb]
            if key < 0.5*BACK_NONE and back_body(key) != coll_bid[j]:
                continue
        if mode == 0:
            wp.atomic_add(scnt, a, 1)
            continue
        cp = xs - n*rs
        rc = cp - bx[a]
        vc = coll_v[j] + wp.cross(coll_w[j], cp - coll_x[j])
        lam = d/gen_w(binvm[a], binvi[a], bq[a], rc, n)
        csub = cp - (bxo[a] + wp.quat_rotate(bqo[a], wp.quat_rotate_inv(bq[a], rc))) - vc*dt
        f = wp.min(1.0, (wp.max(0.0, -wp.dot(csub, n)) + dep)/d)
        share = 1.0/float(wp.max(1, scnt[a]))
        dth = ang_corr(binvi[a], bq[a], rc, n*lam)
        wp.atomic_add(sdx, a, n*(lam*binvm[a]*share))
        wp.atomic_add(sdth, a, dth*share)
        wp.atomic_add(svx, a, n*(lam*binvm[a]*share*f))
        wp.atomic_add(svth, a, dth*(share*f))
        fr = n*(-lam*share*f/(dt*dt))
        wp.atomic_add(coll_f, j, fr)
        wp.atomic_add(coll_tq, j, wp.cross(cp - coll_x[j], fr))


@wp.kernel
def shape_apply(bx: wp.array(dtype=wp.vec3), bq: wp.array(dtype=wp.quat), scnt: wp.array(dtype=int),
                sdx: wp.array(dtype=wp.vec3), sdth: wp.array(dtype=wp.vec3), svx: wp.array(dtype=wp.vec3),
                svth: wp.array(dtype=wp.vec3), btx: wp.array(dtype=wp.vec3), btth: wp.array(dtype=wp.vec3),
                btouch: wp.array(dtype=int), bsh: wp.array(dtype=int)):
    """See shape_deep."""
    b = wp.tid()
    if scnt[b] > 0:
        bx[b] = bx[b] + sdx[b]
        q = bq[b]
        th = sdth[b]
        bq[b] = wp.normalize(q + wp.quat(th[0], th[1], th[2], 0.0)*q*0.5)
        btx[b] = btx[b] + svx[b]
        btth[b] = btth[b] + svth[b]
        btouch[b] = btouch[b] + 1
    scnt[b] = 0
    sdx[b] = wp.vec3(0.0, 0.0, 0.0)
    sdth[b] = wp.vec3(0.0, 0.0, 0.0)
    svx[b] = wp.vec3(0.0, 0.0, 0.0)
    svth[b] = wp.vec3(0.0, 0.0, 0.0)
    bsh[b] = 0


@wp.kernel
def floor_apply(bx: wp.array(dtype=wp.vec3), bq: wp.array(dtype=wp.quat), bxo: wp.array(dtype=wp.vec3),
                fdx: wp.array(dtype=wp.vec3), fdth: wp.array(dtype=wp.vec3), fcnt: wp.array(dtype=int),
                btx: wp.array(dtype=wp.vec3), btth: wp.array(dtype=wp.vec3), btouch: wp.array(dtype=int), dep: float):
    """The room's floor is hard. The floor's push in contact_world is one correction among a piece's others, and a
    light piece under load loses to them: a grit grain of 0.2 mg under a bristle bank stood 0.4 mm deep in the
    floor, half its height, for as long as the bank was on it. With the pan's floor and lip as plain boxes, such a
    grain slips under the 0.5 mm edge of the lip, the pan is lifted onto it (a piece held from below is a strut for a
    shape), more grains follow, and a third of a swept line ends under the pan (traced tick by tick).

    So after the pass every sphere still deeper than the tolerance is set back onto the floor by a push at that
    sphere (the mean of the pushes if several of a piece's spheres are deep), and the push counts in the velocity
    like any other correction of the substep: in full for what the piece came down in this substep, and at the
    depenetration speed for an overlap that was there before (a piece that starts in the floor is not thrown out
    of it). Three other forms of this fail: the push capped at the depenetration speed
    altogether (a grain that bristles press 0.4 mm down in one substep stayed 0.15 mm deep); the piece moved up as a whole and its velocity left alone (it was a quarter millimetre deep
    again when the next substep looked at its contacts); the piece moved up as a whole and its downward speed
    taken away (a rice grain with one end under a wheel was turned a little more by the wheel in every substep
    and never turned back by the floor: it spun up to 1,000 rad/s and left at 9 m/s)."""
    b = wp.tid()
    k = fcnt[b]
    if k > 0:
        dx = fdx[b]/float(k)
        th = fdth[b]/float(k)
        bx[b] = bx[b] + dx
        q = bq[b]
        bq[b] = wp.normalize(q + wp.quat(th[0], th[1], th[2], 0.0)*q*0.5)
        came_down = wp.max(0.0, bxo[b][2] - (bx[b][2] - dx[2]))
        f = wp.min(1.0, (came_down + dep)/wp.max(wp.length(dx), 1.0e-12))
        btx[b] = btx[b] + dx*f
        btth[b] = btth[b] + th*f
        btouch[b] = btouch[b] + 1
    fdx[b] = wp.vec3(0.0, 0.0, 0.0)
    fdth[b] = wp.vec3(0.0, 0.0, 0.0)
    fcnt[b] = 0


@wp.kernel
def unpack_in(buf: wp.array(dtype=float), n_coll: int, n_tool: int,
              coll_x: wp.array(dtype=wp.vec3), coll_q: wp.array(dtype=wp.quat), coll_v: wp.array(dtype=wp.vec3),
              coll_w: wp.array(dtype=wp.vec3), tool_x: wp.array(dtype=wp.vec3), tool_q: wp.array(dtype=wp.quat),
              tool_v: wp.array(dtype=wp.vec3), tool_w: wp.array(dtype=wp.vec3), tool_on: wp.array(dtype=int),
              coll_vn: wp.array(dtype=float)):
    i = wp.tid()
    if i < n_coll:
        o = 14*i
        coll_vn[i] = buf[o + 13]
        coll_x[i] = wp.vec3(buf[o], buf[o + 1], buf[o + 2])
        coll_q[i] = wp.quat(buf[o + 3], buf[o + 4], buf[o + 5], buf[o + 6])
        coll_v[i] = wp.vec3(buf[o + 7], buf[o + 8], buf[o + 9])
        coll_w[i] = wp.vec3(buf[o + 10], buf[o + 11], buf[o + 12])
    else:
        t = i - n_coll
        o = 14*n_coll + 14*t
        tool_x[t] = wp.vec3(buf[o], buf[o + 1], buf[o + 2])
        tool_q[t] = wp.quat(buf[o + 3], buf[o + 4], buf[o + 5], buf[o + 6])
        tool_v[t] = wp.vec3(buf[o + 7], buf[o + 8], buf[o + 9])
        tool_w[t] = wp.vec3(buf[o + 10], buf[o + 11], buf[o + 12])
        tool_on[t] = int(buf[o + 13])


@wp.kernel
def pack_out(n_coll: int, n_tool: int, tool_f: wp.array(dtype=wp.vec3), tool_tq: wp.array(dtype=wp.vec3),
             coll_f: wp.array(dtype=wp.vec3), coll_tq: wp.array(dtype=wp.vec3), fault: wp.array(dtype=int),
             awake: wp.array(dtype=int), buf: wp.array(dtype=float)):
    i = wp.tid()
    f = wp.vec3(0.0, 0.0, 0.0)
    q = wp.vec3(0.0, 0.0, 0.0)
    if i < n_coll:
        f = coll_f[i]
        q = coll_tq[i]
    elif i < n_coll + n_tool:
        f = tool_f[i - n_coll]
        q = tool_tq[i - n_coll]
    else:
        f = wp.vec3(float(fault[0]), float(awake[0]), 0.0)
    o = 6*i
    buf[o] = f[0]
    buf[o + 1] = f[1]
    buf[o + 2] = f[2]
    buf[o + 3] = q[0]
    buf[o + 4] = q[1]
    buf[o + 5] = q[2]


@wp.kernel
def count_awake(asleep: wp.array(dtype=int), awake: wp.array(dtype=int)):
    b = wp.tid()
    if asleep[b] == 0:
        wp.atomic_add(awake, 0, 1)


# ------------------------------------------------------------------------------------------------ host side
@dataclass
class Bank:
    """A tool's bristle bank in its block frame (the rigid body the bristles are set in)."""
    name: str
    roots: np.ndarray            # (R, 3) where each cluster leaves the block face
    dirs: np.ndarray             # (R, 3) unit directions of the straight clusters
    length: np.ndarray           # (R,) free length of each cluster (an angled trim varies it)
    nodes: int                   # nodes per cluster, including the two the block holds
    filaments: float             # filaments a cluster stands for
    d: float                     # filament diameter
    E: float                     # filament modulus
    rho: float                   # filament density
    mu: float                    # bristle tip friction on floor and shapes (kinetic)
    zeta: float                  # damping ratio of the first bending mode
    per_row: int                 # clusters per row (neighbour topology)
    packing: float = 0.6

    @property
    def EI(self):
        return self.E*math.pi*self.d**4/64.0

    @property
    def lam(self):
        return self.rho*math.pi*self.d**2/4.0

    @property
    def radius(self):
        return 0.5*self.d*math.sqrt(self.filaments/self.packing)


@dataclass
class Piece:
    """One loose piece: spheres (local centres, radii) rigidly joined; density gives mass and inertia."""
    cls: str
    centres: np.ndarray          # (n, 3) in an arbitrary local frame
    radii: np.ndarray            # (n,)
    mass: float
    pos: tuple = (0., 0., 0.)
    quat: tuple = (0., 0., 0., 1.)   # xyzw, of the local frame given above
    tag: str = ""


@dataclass
class DebrisClass:
    name: str
    mu_floor: float              # static friction on the floor
    mu: float                    # static friction on rigid shapes and on other pieces (the smaller of the pair governs)
    mu_bristle: float            # static friction on bristles
    restitution: float
    mu_roll: float               # rolling resistance coefficient (deceleration mu_roll g of a rolling sphere)
    soft_k: float = 0.0          # N/m: each sphere of a piece that yields under a rigid shape (paper); 0 = rigid


def principal(centres, radii, mass):
    """Centre of mass, principal moments and rotation (columns = principal axes) of a union of solid spheres whose
    masses share `mass` by volume (overlaps are not subtracted)."""
    c = np.asarray(centres, float).reshape(-1, 3)
    r = np.asarray(radii, float).reshape(-1)
    m = r**3
    m = mass*m/m.sum()
    com = (m[:, None]*c).sum(0)/mass
    d = c - com
    I = np.zeros((3, 3))
    for mi, di, ri in zip(m, d, r):
        I += mi*((di @ di)*np.eye(3) - np.outer(di, di)) + 0.4*mi*ri*ri*np.eye(3)
    w, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0:
        V[:, 2] = -V[:, 2]
    return com, w, V


class Sweeper:
    def __init__(self, banks, classes, pieces, *, dt_tick, room, device="cuda:0", small_r=2.0e-3,
                 kin=0.8, rest_disp=2.5e-5, rest_time=0.1, slip_disp=1.0e-5, wake_pen=1.0e-4, relax=1.0, gravity=9.81, cell=0.1, passes=1,
                 capacity=None, air_density=1.2, kv_zeta=0.05):
        """banks: [Bank] (index = tool id); classes: [DebrisClass]; pieces: [Piece];
        room: (x0, y0, x1, y1) floor extent for the collider cells."""
        self.device = device
        self.dt = dt_tick/SUBSTEPS
        self.gravity = gravity
        self.kin, self.relax, self.passes = kin, relax, int(passes)
        self.rest_disp, self.rest_steps, self.wake_pen = rest_disp, int(round(rest_time/self.dt)), wake_pen
        self.slip_disp = slip_disp
        self.small_r = small_r
        self.kv_zeta = float(kv_zeta)       # damping ratio of the bristle chains' shortest wave (see rod_step)
        self.v_limit = V_MAX                # a piece faster than this is a fault (see push)
        self.banks = list(banks)
        self.classes = list(classes)
        self.air_density = air_density
        self.T = max(1, len(self.banks))
        self._build_rods()
        self._build_debris(pieces, capacity)
        x0, y0, x1, y1 = room
        self.cs = cell
        self.org = (x0, y0, 0.0)
        self.nx, self.ny = int(math.ceil((x1 - x0)/cell)), int(math.ceil((y1 - y0)/cell))
        self.set_colliders([])
        self.fault = wp.zeros(1, dtype=int, device=device)
        self._awake = wp.zeros(1, dtype=int, device=device)
        self._io_in, self._io_out = None, None
        self.substeps_run = 0
        self.skipped_ticks = 0

    # ------------------------------------------------------------------ bristles
    def _build_rods(self):
        dev = self.device
        T = max(1, len(self.banks))
        nmax = max([b.nodes for b in self.banks], default=2)
        roots, dirs, tool, nn, seg, kb, mass, rad, mu, damp, kv, nbs = [], [], [], [], [], [], [], [], [], [], [], []
        self.rod_slices = []
        base = 0
        for t, b in enumerate(self.banks):
            R = len(b.roots)
            s = b.length/(b.nodes - 1.5)
            roots.append(b.roots); dirs.append(b.dirs); tool.append(np.full(R, t)); nn.append(np.full(R, b.nodes))
            seg.append(s)
            kb.append(b.EI/(b.lam*s**4))
            mass.append(b.filaments*b.lam*s)
            rad.append(np.full(R, b.radius))
            mu.append(np.full(R, b.mu))
            # first bending frequency of the cluster; damping c with zeta of critical in that mode
            w1 = 1.8751**2*np.sqrt(b.EI/(b.lam*b.length**4))
            # the chain's shortest wave (fourth difference 16): w_top = 4 sqrt(kb). Kelvin-Voigt damping with the
            # ratio kv_zeta in that wave is kb*beta = kv_zeta sqrt(kb)/2 on the velocity stencil; in the first mode
            # it adds kv_zeta w1/w_top, which is taken off the mass-proportional part so that mode keeps b.zeta
            w_top = 4.0*np.sqrt(kb[-1])
            kv.append(0.5*self.kv_zeta*np.sqrt(kb[-1]))
            damp.append(2.0*np.maximum(0.0, b.zeta - self.kv_zeta*w1/w_top)*w1)
            nb = -np.ones((R, NB), int)
            idx = np.arange(R)
            left, right = idx - 1, idx + 1
            nb[:, 0] = np.where(idx % b.per_row == 0, -1, left + base)
            nb[:, 1] = np.where(right % b.per_row == 0, -1, right + base)
            nb[:, 1] = np.where(right >= R, -1, nb[:, 1])
            up, dn = idx - b.per_row, idx + b.per_row
            nb[:, 2] = np.where(up < 0, -1, up + base)
            nb[:, 3] = np.where(dn >= R, -1, dn + base)
            nbs.append(nb)
            self.rod_slices.append(slice(base, base + R))
            base += R
        self.R, self.N, self.T = base, nmax, T
        cat = lambda a, dt_=float: np.concatenate(a).astype(dt_) if a else np.zeros(0, dt_)
        Rn = max(1, self.R)
        def arr(a, dtype, shape=None):
            if not len(a):
                return wp.zeros(shape or Rn, dtype=dtype, device=dev)
            return wp.array(a, dtype=dtype, device=dev)
        self.rod_root = arr(np.concatenate(roots) if roots else [], wp.vec3)
        self.rod_dir = arr(np.concatenate(dirs) if dirs else [], wp.vec3)
        self.rod_tool = arr(cat(tool, int), int)
        self.rod_n = arr(cat(nn, int), int)
        self.rod_seg = arr(cat(seg), float)
        self.rod_kb = arr(cat(kb), float)
        self.rod_m = arr(cat(mass), float)
        self.rod_rad = arr(cat(rad), float)
        self.rod_mu = arr(cat(mu), float)
        self.rod_damp = arr(cat(damp), float)
        self.rod_kv = arr(cat(kv), float)
        self.rod_nb = wp.array(np.concatenate(nbs) if nbs else -np.ones((Rn, NB), int), dtype=int, device=dev)
        self.rod_x = wp.zeros((Rn, self.N), dtype=wp.vec3, device=dev)
        self.rod_v = wp.zeros((Rn, self.N), dtype=wp.vec3, device=dev)
        self.rod_xo = wp.zeros((Rn, self.N), dtype=wp.vec3, device=dev)
        self.rod_anc = wp.zeros((Rn, self.N), dtype=wp.vec3, device=dev)        # friction anchor of each node (rod_step)
        self.rod_aj = wp.full((Rn, self.N), -2, dtype=int, device=dev)          # ... and what it is on
        self.rdx = wp.zeros((Rn, self.N), dtype=wp.vec3, device=dev)
        self.rcnt = wp.zeros((Rn, self.N), dtype=int, device=dev)
        self.rcnt_prev = wp.zeros((Rn, self.N), dtype=int, device=dev)
        self.rod_fn = wp.zeros(Rn, dtype=float, device=dev)
        self.tool_x = wp.zeros(T, dtype=wp.vec3, device=dev)
        self.tool_q = wp.array(np.tile([0., 0., 0., 1.], (T, 1)), dtype=wp.quat, device=dev)
        self.tool_v = wp.zeros(T, dtype=wp.vec3, device=dev)
        self.tool_w = wp.zeros(T, dtype=wp.vec3, device=dev)
        self.tool_on = wp.zeros(T, dtype=int, device=dev)
        self.tool_f = wp.zeros(T, dtype=wp.vec3, device=dev)
        self.tool_tq = wp.zeros(T, dtype=wp.vec3, device=dev)
        self._tool_on = np.zeros(T, int)
        self._rod_seg_max = float(max([s.max() for s in seg], default=0.01))
        self._rod_rad_max = float(max([r.max() for r in rad], default=0.001))

    # ------------------------------------------------------------------ debris
    def _build_debris(self, pieces, capacity):
        dev = self.device
        cls_id = {c.name: k for k, c in enumerate(self.classes)}
        B = len(pieces)
        bx, bq, binvm, binvi, bmu_f, bmu, bmu_b, brest, broll, bsize, bcls, bmur, brmin, bairk, bsoft = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        sb, sl, sr = [], [], []
        from scipy.spatial.transform import Rotation
        for k, p in enumerate(pieces):
            com, w, V = principal(p.centres, p.radii, p.mass)
            R0 = Rotation.from_quat(p.quat)
            q = (R0*Rotation.from_matrix(V)).as_quat()
            bx.append(np.asarray(p.pos, float) + R0.apply(com))
            bq.append(q)
            binvm.append(1.0/p.mass)
            binvi.append(1.0/np.maximum(w, 1e-18))
            c = self.classes[cls_id[p.cls]]
            bmu_f.append(c.mu_floor); bmu.append(c.mu); bmu_b.append(c.mu_bristle); brest.append(c.restitution)
            loc = (np.asarray(p.centres, float).reshape(-1, 3) - com) @ V
            rr = np.asarray(p.radii, float).reshape(-1)
            ext = float((np.linalg.norm(loc, axis=1) + rr).max())
            # rolling: a sphere of the piece's rolling radius decelerates at mu_roll g (tau = 1.4 mu m g r for a
            # sphere, so d(omega)/dt = 3.5 mu g / r applied to the spin while friction keeps the contact point still)
            r_roll = float(rr.mean()) if len(rr) > 1 else float(rr[0])
            broll.append(3.5*c.mu_roll*self.gravity/r_roll)
            bmur.append(c.mu_roll)
            bsoft.append(c.soft_k)
            brmin.append(float(rr.min()))
            # frontal area for air drag: the mean of the piece's three projected silhouettes (bounding boxes of its
            # spheres), drag coefficient 1.0 (bluff bodies at Reynolds numbers of a few hundred)
            lo, hi = (loc - rr[:, None]).min(0), (loc + rr[:, None]).max(0)
            e = np.maximum(hi - lo, 2*rr.min())
            area = (e[0]*e[1] + e[1]*e[2] + e[0]*e[2])/3.0*(math.pi/4 if len(rr) == 1 else 0.8)
            bairk.append(0.5*self.air_density*1.0*area/p.mass)
            bsize.append(ext)
            bcls.append(cls_id[p.cls])
            sb += [k]*len(rr); sl += list(loc); sr += list(rr)
        self.B, self.S = B, len(sr)
        Bn, Sn = max(1, B), max(1, len(sr))
        f32 = np.float32
        def arr(a, dtype, n):
            if not len(a):
                return wp.zeros(n, dtype=dtype, device=dev)
            return wp.array(np.asarray(a), dtype=dtype, device=dev)
        self.bx = arr(bx, wp.vec3, Bn)
        self.bq = arr(bq, wp.quat, Bn) if B else wp.array(np.array([[0., 0., 0., 1.]]), dtype=wp.quat, device=dev)
        self.bv = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.bw = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.bxo = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.bqo = wp.zeros(Bn, dtype=wp.quat, device=dev)
        self.bvp = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.bres = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.binvm = arr(np.asarray(binvm, f32), float, Bn)
        self.binvi = arr(np.asarray(binvi, f32), wp.vec3, Bn)
        self.bmu_floor = arr(np.asarray(bmu_f, f32), float, Bn)
        self.bmu = arr(np.asarray(bmu, f32), float, Bn)
        self.bmu_b = arr(np.asarray(bmu_b, f32), float, Bn)
        self.brest = arr(np.asarray(brest, f32), float, Bn)
        self.broll = arr(np.asarray(broll, f32), float, Bn)
        self.bmur = arr(np.asarray(bmur, f32), float, Bn)
        self.brmin = arr(np.asarray(brmin, f32), float, Bn)
        self.bairk = arr(np.asarray(bairk, f32), float, Bn)
        self.bsoft = arr(np.asarray(bsoft, f32), float, Bn)
        self.bsn = wp.zeros(Bn, dtype=int, device=dev)
        self.bdrag = wp.zeros(Bn, dtype=int, device=dev)
        self.bsize = arr(np.asarray(bsize, f32), float, Bn)
        self.asleep = wp.zeros(Bn, dtype=int, device=dev)
        self.bdx = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.bdth = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.bcnt = wp.zeros(Bn, dtype=int, device=dev)
        self.btx = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.btth = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.btouch = wp.zeros(Bn, dtype=int, device=dev)
        self.bdv = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.bsup = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.bsupn = wp.zeros(Bn, dtype=int, device=dev)
        self.bwake = wp.zeros(Bn, dtype=int, device=dev)
        self.bcar = wp.full(Bn, -1, dtype=int, device=dev)
        self.bA = wp.zeros(Bn, dtype=wp.mat33, device=dev)
        self.bAs = wp.zeros(Bn, dtype=wp.mat33, device=dev)
        self.fdx = wp.zeros(Bn, dtype=wp.vec3, device=dev)        # the floor's last word of a substep (floor_apply)
        self.fdth = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.fcnt = wp.zeros(Bn, dtype=int, device=dev)
        self.scnt = wp.zeros(Bn, dtype=int, device=dev)           # a shape's last word of a substep (shape_deep)
        self.sdx = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.sdth = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.svx = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.svth = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.bsh = wp.zeros(Bn, dtype=int, device=dev)            # the piece touched a rigid shape in this substep
        self.btop = wp.zeros(Bn, dtype=float, device=dev)         # its highest point above the room's floor
        self.bwb = wp.full((Bn, 6), BACK_NONE, dtype=float, device=dev)
        self.bwb_n = wp.full((Bn, 6), BACK_NONE, dtype=float, device=dev)
        self.kappa = 0.1             # stiffness of a held piece under a moving shape, as a share of m/dt^2
        self.bcar_n = wp.full(Bn, -1, dtype=int, device=dev)
        self.bcar_s = wp.full(Bn, -1, dtype=int, device=dev)
        self.bpj = wp.full(Bn, -1, dtype=int, device=dev)
        self.bpf = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.bptq = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.v_still = 1.0e-3
        self._snap_a = self._snap_b = None
        self._age_a = self._age_b = 0
        self.v_dep = 1.0
        self.wake_steps = 8
        self.banc = wp.clone(self.bx)
        self.bqanc = wp.clone(self.bq)
        self.bstick = wp.zeros(Bn, dtype=wp.vec4i, device=dev)
        self.bslip = wp.zeros(Bn, dtype=int, device=dev)
        self.brdx = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.brth = wp.zeros(Bn, dtype=wp.vec3, device=dev)
        self.brn = wp.zeros(Bn, dtype=int, device=dev)
        self.brn_prev = wp.zeros(Bn, dtype=int, device=dev)
        self.bair = wp.zeros(Bn, dtype=int, device=dev)
        self.slip_steps = 8
        self.bref = wp.clone(self.bx)
        self.bqref = wp.clone(self.bq)
        self.btimer = wp.zeros(Bn, dtype=int, device=dev)
        self.body_class = np.asarray(bcls, int)
        self.body_mass = 1.0/np.asarray(binvm, float) if B else np.zeros(0)
        self.body_tag = [p.tag for p in pieces]
        self.sph_body = arr(np.asarray(sb, np.int32), int, Sn)
        self.sph_local = arr(np.asarray(sl, f32), wp.vec3, Sn)
        self.sph_r = arr(np.asarray(sr, f32), float, Sn)
        self.sph_x = wp.zeros(Sn, dtype=wp.vec3, device=dev)
        # the box and the face each sphere is at a box by (two boxes a sphere; -1: none): see shape_hit
        self.sph_mj = wp.full((Sn, 2), -1, dtype=int, device=dev)
        self.sph_mf = wp.full((Sn, 2), -1, dtype=int, device=dev)
        self._sph_body_np = np.asarray(sb, int)
        self._sph_r_np = np.asarray(sr, float)
        sr_np = np.asarray(sr, float)
        small = np.flatnonzero(sr_np <= self.small_r)
        large = np.flatnonzero(sr_np > self.small_r)
        self.n_small, self.n_large = len(small), len(large)
        self.idx_all = wp.array(np.arange(Sn, dtype=np.int32), dtype=int, device=dev)
        self.idx_small = wp.array(small.astype(np.int32) if len(small) else np.zeros(1, np.int32), dtype=int, device=dev)
        self.idx_large = wp.array(large.astype(np.int32) if len(large) else np.zeros(1, np.int32), dtype=int, device=dev)
        self.pos_small = wp.zeros(max(1, len(small)), dtype=wp.vec3, device=dev)
        self.pos_large = wp.zeros(max(1, len(large)), dtype=wp.vec3, device=dev)
        self.r_small = float(sr_np[small].max()) if len(small) else 0.0
        self.r_large = float(sr_np[large].max()) if len(large) else 0.0
        self.grid_small = wp.HashGrid(128, 128, 32, device=dev)
        self.grid_large = wp.HashGrid(64, 64, 32, device=dev)
        if self.S:
            wp.launch(sph_update, dim=self.S, device=dev, inputs=[self.bx, self.bq, self.sph_body, self.sph_local, self.sph_x])

    # ------------------------------------------------------------------ colliders
    def set_colliders(self, colliders):
        """colliders: list of dict(type, size, mu, tool, who) (tool: index of the tool whose bank ignores this shape,
        or -1; who: BOTH, RODS_ONLY or DEBRIS_ONLY);
        poses and velocities come every tick through update_colliders."""
        dev = self.device
        K = max(1, len(colliders))
        self.K = len(colliders)
        self.coll_type = wp.array(np.array([c["type"] for c in colliders] or [0], np.int32), dtype=int, device=dev)
        self.coll_size = wp.array(np.array([c["size"] for c in colliders] or [[0., 0., 0.]], np.float32), dtype=wp.vec3, device=dev)
        self.coll_mu = wp.array(np.array([c["mu"] for c in colliders] or [0.], np.float32), dtype=float, device=dev)
        self.coll_tool = wp.array(np.array([c.get("tool", -1) for c in colliders] or [-1], np.int32), dtype=int, device=dev)
        self.coll_who = wp.array(np.array([c.get("who", BOTH) for c in colliders] or [0], np.int32), dtype=int, device=dev)
        self.coll_pri = wp.array(np.array([int(bool(c.get("pad", False))) for c in colliders] or [0], np.int32), dtype=int, device=dev)
        # inverse mass of the rigid body a shape belongs to (0: fixed, or not to be pushed back by held pieces)
        self.coll_invm = wp.array(np.array([float(c.get("invm", 0.0)) for c in colliders] or [0.], np.float32), dtype=float, device=dev)
        # on how many held pieces each rigid body presses (count_world), by the body's number as back_key has it
        self.nb_body = wp.zeros(int(BACK_IDS), dtype=int, device=dev)
        # for a shape of a free body: where the body's centre of mass is in the shape's own frame, and the body's
        # inverse inertia in that frame (zero where none is given: the body is then stopped with its mass alone)
        self.coll_lcom = wp.array(np.array([np.asarray(c.get("lcom", (0., 0., 0.)), np.float32) for c in colliders] or [[0., 0., 0.]], np.float32),
                                  dtype=wp.vec3, device=dev)
        self.coll_iloc = wp.array(np.array([np.asarray(c.get("iinv", np.zeros((3, 3))), np.float32) for c in colliders] or [np.zeros((3, 3))],
                                           np.float32), dtype=wp.mat33, device=dev)
        # 1: a wheel of the robot; 2: its chassis (see contact_world, "A WHEEL" and "The robot's CHASSIS")
        self.coll_roll = wp.array(np.array([int(c.get("roll", 0)) for c in colliders] or [0], np.int32), dtype=int, device=dev)
        # which rigid body a shape belongs to, as back_key records it: 0 the room, body + 1 otherwise; a shape that
        # can be pushed back and names no body (a probe's bare pad) counts as a body of its own
        bid = [0 if (c.get("body", -1) < 0 and not c.get("invm", 0.0)) else
               (min(int(c["body"]) + 1, 199) if c.get("body", -1) >= 0 else 200 + k % 55) for k, c in enumerate(colliders)]
        self.coll_bid = wp.array(np.array(bid or [0], np.int32), dtype=int, device=dev)
        self.coll_x = wp.zeros(K, dtype=wp.vec3, device=dev)
        self.coll_q = wp.array(np.tile([0., 0., 0., 1.], (K, 1)), dtype=wp.quat, device=dev)
        self.coll_v = wp.zeros(K, dtype=wp.vec3, device=dev)
        self.coll_w = wp.zeros(K, dtype=wp.vec3, device=dev)
        self.coll_vn = wp.zeros(K, dtype=float, device=dev)
        self.coll_f = wp.zeros(K, dtype=wp.vec3, device=dev)
        self.coll_tq = wp.zeros(K, dtype=wp.vec3, device=dev)
        self._coll_size_np = np.array([c["size"] for c in colliders] or [[0., 0., 0.]], float)
        self._coll_type_np = np.array([c["type"] for c in colliders] or [0], int)
        self._coll_bound = np.where(self._coll_type_np == BOX, np.linalg.norm(self._coll_size_np, axis=1),
                                    self._coll_size_np[:, 0] + self._coll_size_np[:, 1])
        self._cells_np = -np.ones((self.nx*self.ny, KC), np.int32)
        self.cells = wp.array(self._cells_np, dtype=int, device=dev)
        self._cell_pos = None
        self.cell_overflow = 0

    def push(self, pos, quat, vel, omega, tool_pos, tool_quat, tool_vel, tool_omega, active, margin=0.03, rebuild_move=0.015, net=None):
        """Poses (xyzw) and velocities of every collider and tool at the end of this rigid tick, in one upload.
        `net`: each collider's net speed over the last tenth of a second (see NET SPEED at contact_world); its
        speed of the moment where not given."""
        K, T = self.K, self.T
        buf = np.zeros(14*K + 14*T, np.float32)
        if K:
            if net is None:
                net = np.linalg.norm(vel, axis=1)
            buf[:14*K] = np.concatenate([pos, quat, vel, omega, np.asarray(net, float)[:, None]], 1).astype(np.float32).reshape(-1)
        nt = len(tool_pos)
        if nt:
            tb = np.zeros((T, 14), np.float32)
            tb[:, 6] = 1.0
            tb[:nt] = np.concatenate([tool_pos, tool_quat, tool_vel, tool_omega, np.asarray(active, float)[:, None]], 1)
            buf[14*K:] = tb.reshape(-1)
        self._tool_on = np.zeros(T, int)
        self._tool_on[:nt] = np.asarray(active, int)
        # The speed above which a piece is a fault. While a bristle bank is at work: V_MAX (a filament let go flicks a
        # grain at several metres a second). With every bank at rest a piece can have what a fall gives it in a room
        # (V_FALL: 3.3 m of fall) and what a rigid shape gives it, at most twice that shape's speed.
        if nt and np.any(np.asarray(active, int)):
            self.v_limit = V_MAX
        else:
            fastest = float((np.linalg.norm(vel, axis=1) + V_ARM*np.linalg.norm(omega, axis=1)).max()) if K else 0.0
            self.v_limit = min(V_MAX, V_FALL + 2.0*fastest)
        if not len(buf):
            return
        if self._io_in is None or self._io_in.shape[0] != len(buf):
            self._io_in = wp.zeros(len(buf), dtype=float, device=self.device)
        self._io_in.assign(buf)
        wp.launch(unpack_in, dim=K + T, device=self.device,
                  inputs=[self._io_in, K, T, self.coll_x, self.coll_q, self.coll_v, self.coll_w, self.tool_x, self.tool_q,
                          self.tool_v, self.tool_w, self.tool_on, self.coll_vn])
        if K and (self._cell_pos is None or np.abs(np.asarray(pos, np.float32) - self._cell_pos).max() > rebuild_move):
            self._rebuild_cells(np.asarray(pos, np.float32), np.asarray(quat, float), margin)

    def pull(self):
        """Mean force and moment over the tick on each collider (about its own origin) and each tool (about its block
        origin), the number of faulted piece-substeps and the number of pieces awake: one download."""
        K, T = self.K, self.T
        n = 6*(K + T + 1)
        if self._io_out is None or self._io_out.shape[0] != n:
            self._io_out = wp.zeros(n, dtype=float, device=self.device)
        self._awake.zero_()
        if self.B:
            wp.launch(count_awake, dim=self.B, device=self.device, inputs=[self.asleep, self._awake])
        wp.launch(pack_out, dim=K + T + 1, device=self.device,
                  inputs=[K, T, self.tool_f, self.tool_tq, self.coll_f, self.coll_tq, self.fault, self._awake, self._io_out])
        out = self._io_out.numpy().reshape(-1, 6)
        s = 1.0/SUBSTEPS
        return (out[K:K + T, :3]*s, out[K:K + T, 3:]*s, out[:K, :3]*s, out[:K, 3:]*s, int(out[K + T, 0]), int(out[K + T, 1]))

    def update_colliders(self, pos, quat, vel, omega, margin=0.03, rebuild_move=0.015, net=None):
        """World poses (xyzw) and velocities of every collider at the end of this tick (`net`: see push)."""
        if not self.K:
            return
        pos = np.asarray(pos, np.float32)
        self.coll_x.assign(pos)
        self.coll_q.assign(np.asarray(quat, np.float32))
        self.coll_v.assign(np.asarray(vel, np.float32))
        self.coll_w.assign(np.asarray(omega, np.float32))
        self.coll_vn.assign(np.asarray(np.linalg.norm(vel, axis=1) if net is None else net, np.float32))
        if self._cell_pos is None or np.abs(pos - self._cell_pos).max() > rebuild_move:
            self._rebuild_cells(pos, np.asarray(quat, float), margin)

    def _rebuild_cells(self, pos, quat, margin):
        """Floor cells -> shapes whose padded world bounding box reaches the cell (plan view; height is ignored)."""
        from scipy.spatial.transform import Rotation
        self._cell_pos = pos.copy()
        cells = -np.ones((self.nx*self.ny, KC), np.int32)
        fill = np.zeros(self.nx*self.ny, int)
        x0, y0 = self.org[0], self.org[1]
        Rm = np.abs(Rotation.from_quat(quat[:self.K]).as_matrix())
        h = self._coll_size_np.copy()
        cap = self._coll_type_np == CAPSULE
        h[cap] = np.stack([h[cap, 0], h[cap, 0], h[cap, 0] + h[cap, 1]], 1)       # capsule: radius, radius, half length
        cyl = self._coll_type_np == CYLINDER
        h[cyl] = np.stack([h[cyl, 0], h[cyl, 0], h[cyl, 1]], 1)
        half = np.einsum("kij,kj->ki", Rm, h)[:, :2] + margin
        lo = np.floor((pos[:self.K, :2] - half - (x0, y0))/self.cs).astype(int)
        hi = np.floor((pos[:self.K, :2] + half - (x0, y0))/self.cs).astype(int)
        over = 0
        for j in range(self.K):
            for iy in range(max(lo[j, 1], 0), min(hi[j, 1], self.ny - 1) + 1):
                for ix in range(max(lo[j, 0], 0), min(hi[j, 0], self.nx - 1) + 1):
                    c = iy*self.nx + ix
                    if fill[c] < KC:
                        cells[c, fill[c]] = j
                        fill[c] += 1
                    else:
                        over += 1
        self.cell_overflow = over
        self.cell_fill_max = int(fill.max(initial=0))
        self._cells_np = cells
        self.cells.assign(cells)

    # ------------------------------------------------------------------ stepping
    def set_tools(self, pos, quat, vel, omega, active):
        self.tool_x.assign(np.asarray(pos, np.float32))
        self.tool_q.assign(np.asarray(quat, np.float32))
        self.tool_v.assign(np.asarray(vel, np.float32))
        self.tool_w.assign(np.asarray(omega, np.float32))
        self._tool_on = np.asarray(active, int)
        self.tool_on.assign(self._tool_on.astype(np.int32))

    def place_rods(self):
        """Put every bank in its straight shape at the current tool poses (start of an episode)."""
        if not self.R:
            return
        on = self._tool_on.copy()
        self.tool_on.assign(np.zeros(self.T, np.int32))
        self._rod_step()
        self.tool_on.assign(on.astype(np.int32))
        self.tool_f.zero_(); self.tool_tq.zero_()

    def _rod_step(self):
        wp.launch(rod_step, dim=self.R, device=self.device,
                  inputs=[self.rod_x, self.rod_v, self.rod_xo, self.rod_root, self.rod_dir, self.rod_tool, self.rod_n,
                          self.rod_seg, self.rod_kb, self.rod_m, self.rod_rad, self.rod_mu, self.rod_damp, self.rod_kv, self.rod_nb,
                          self.tool_x, self.tool_q, self.tool_v, self.tool_w, self.tool_on,
                          self.coll_type, self.coll_size, self.coll_x, self.coll_q, self.coll_v, self.coll_w, self.coll_mu,
                          self.coll_tool, self.coll_who, self.cells, wp.vec3(*self.org), 1.0/self.cs, self.nx, self.ny,
                          self.dt, self.gravity, 0.4, 1.0e-3, self.tool_f, self.tool_tq, self.coll_f, self.coll_tq, self.rod_fn,
                          self.rod_anc, self.rod_aj])

    def substep(self):
        dev, dt = self.device, self.dt
        rods = self.R > 0
        if rods:
            self._rod_step()
        if self.B:
            body = [self.bx, self.bq, self.bxo, self.bqo, self.binvm, self.binvi]
            wp.launch(body_predict, dim=self.B, device=dev,
                      inputs=[self.bx, self.bq, self.bv, self.bw, self.bxo, self.bqo, self.bvp, self.bres, self.asleep,
                              self.bairk, self.gravity, dt])
            anch = [self.bx, self.bq, self.banc, self.bqanc, self.binvm, self.binvi]
            dep = self.v_dep*dt
            acc = [self.bdx, self.bdth, self.bcnt, self.bwake, self.bstick]
            car = [self.bcar, self.bcar_n, self.bcar_s, self.bpj, self.bpf, self.bptq, self.bA, self.bwb, self.bwb_n, self.bAs]
            brush = rods and bool(self._tool_on.any())
            pair = [self.sph_x, self.sph_r, self.sph_body, *anch, self.bxo, self.bqo, self.bdrag, self.btimer, self.wake_steps, dep,
                    self.bmu, self.bmur, self.asleep, self.wake_pen, self.kin, *acc, *car]
            cnt = [self.sph_x, self.sph_r, self.sph_body, self.asleep, self.bA]
            for it in range(self.passes):
                wp.launch(sph_update, dim=self.S, device=dev, inputs=[self.bx, self.bq, self.sph_body, self.sph_local, self.sph_x])
                if it == 0:
                    if self.n_small:
                        wp.launch(gather, dim=self.n_small, device=dev, inputs=[self.sph_x, self.idx_small, self.pos_small])
                        self.grid_small.build(self.pos_small, 2.0*self.r_small)
                    if self.n_large:
                        wp.launch(gather, dim=self.n_large, device=dev, inputs=[self.sph_x, self.idx_large, self.pos_large])
                        self.grid_large.build(self.pos_large, 2.0*self.r_large)
                # the three sets of pairs: small with small, large with large, and small spheres looking the large ones
                # up (a few coarse cells each; the other way round a 24 mm sphere walks 9,000 fine cells, and four
                # paper wads cost 44 ms a substep, measured)
                sets = []
                if self.n_small:
                    sets.append((self.n_small, [self.grid_small.id, self.idx_small, self.idx_small, self.r_small*1.05, 0]))
                if self.n_large:
                    sets.append((self.n_large, [self.grid_large.id, self.idx_large, self.idx_large, self.r_large*1.05, 0]))
                    if self.n_small:
                        sets.append((self.n_small, [self.grid_large.id, self.idx_small, self.idx_large, self.r_large*1.05, 1]))
                # first pass: count every piece's contacts; second pass: correct
                self.nb_body.zero_()
                self.btop.fill_(-1.0)
                wp.launch(piece_top, dim=self.S, device=dev, inputs=[self.sph_x, self.sph_r, self.sph_body, self.btop])
                for dim, head in sets:
                    wp.launch(count_pairs, dim=dim, device=dev, inputs=[*head, *cnt])
                wp.launch(count_world, dim=self.S, device=dev,
                          inputs=[self.sph_x, self.sph_r, self.sph_body, self.bx, self.bq, self.bxo, self.bqo, self.asleep, self.bsoft,
                                  self.coll_type, self.coll_size, self.coll_x, self.coll_q, self.coll_v, self.coll_w, self.coll_who,
                                  self.cells, wp.vec3(*self.org), 1.0/self.cs, self.nx, self.ny, dt, self.bA, self.coll_invm,
                                  self.bwb, self.nb_body, self.coll_bid, self.sph_mj, self.sph_mf, self.coll_roll, self.btop])
                for dim, head in sets:
                    wp.launch(contact_pairs, dim=dim, device=dev, inputs=[*head, *pair])
                wp.launch(contact_world, dim=self.S, device=dev,
                          inputs=[self.sph_x, self.sph_r, self.sph_body, self.bx, self.bq, self.bxo, self.bqo, self.bvp,
                                  self.binvm, self.binvi, self.bmu_floor, self.bmu, self.bmur, self.brest, self.asleep,
                                  self.banc, self.bqanc, self.bdrag, self.v_still, dep, self.coll_type, self.coll_size, self.coll_x, self.coll_q, self.coll_v, self.coll_w,
                                  self.coll_mu, self.coll_who, self.cells, wp.vec3(*self.org), 1.0/self.cs, self.nx, self.ny, dt, self.kin,
                                  0.15, 1.0e-3, self.wake_pen, 1 if it == 0 else 0,
                                  self.bdx, self.bdth, self.bcnt, self.bdv, self.bsup, self.bsupn, self.bwake,
                                  self.bstick, self.coll_f, self.coll_tq,
                                  self.bcar_n, self.bcar_s, self.bsoft, self.coll_pri, self.brdx, self.brth, self.bsn, self.bA,
                                  self.coll_invm, self.nb_body, self.bwb, self.bwb_n, dt*SUBSTEPS, self.kappa, self.coll_bid,
                                  self.sph_mj, self.sph_mf, self.coll_roll, self.btop, self.coll_lcom, self.coll_iloc, self.bsh, self.coll_vn])
                if brush and it == 0:
                    n_seg = self.N - 2
                    rod = [self.rod_x, self.rod_xo, self.rod_tool, self.rod_n, self.rod_seg, self.rod_m, self.rod_rad, self.tool_on]
                    tail = [self.sph_x, self.sph_r, self.sph_body, *body, self.bmu_b, self.bmur, self.asleep, self.wake_pen, self.kin, dep, 1.0e-3*dt,
                            self.brdx, self.brth, self.brn, self.brn_prev, self.bwake, self.rdx, self.rcnt, self.rcnt_prev]
                    if self.n_small:
                        wp.launch(contact_rods, dim=self.R*n_seg, device=dev,
                                  inputs=[self.grid_small.id, self.idx_small, self.r_small, n_seg, *rod, *tail])
                    if self.n_large:
                        wp.launch(contact_rods, dim=self.R*n_seg, device=dev,
                                  inputs=[self.grid_large.id, self.idx_large, self.r_large, n_seg, *rod, *tail])
                wp.launch(carried_reaction, dim=self.B, device=dev,
                          inputs=[self.bx, self.binvm, self.asleep, self.bcnt, self.bcar, self.bpj, self.bpf, self.bptq,
                                  self.coll_x, self.relax, dt, self.gravity, 1 if it == 0 else 0, self.coll_f, self.coll_tq])
                wp.launch(body_iterate, dim=self.B, device=dev,
                          inputs=[self.bx, self.bq, self.asleep, self.bdx, self.bdth, self.bcnt, self.brdx, self.brth, self.brn,
                                  self.brn_prev, 1 if it == 0 else 0, self.btx, self.btth, self.btouch, self.bA, self.relax, self.bsn, self.bAs])
            if self.S:
                shp = [self.bx, self.bq, self.bxo, self.bqo, self.binvm, self.binvi, self.sph_body, self.sph_local, self.sph_r,
                       self.asleep, self.bsoft, self.bsh, self.coll_type, self.coll_size, self.coll_x, self.coll_q, self.coll_v,
                       self.coll_w, self.coll_who, self.coll_roll, self.coll_bid, self.coll_pri, self.cells, wp.vec3(*self.org), 1.0/self.cs,
                       self.nx, self.ny, dt, SHAPE_TOL, dep, self.bwb, self.sph_mj, self.sph_mf, self.scnt, self.sdx, self.sdth,
                       self.svx, self.svth, self.coll_f, self.coll_tq]
                wp.launch(shape_deep, dim=self.S, device=dev, inputs=[0, *shp])
                wp.launch(shape_deep, dim=self.S, device=dev, inputs=[1, *shp])
                wp.launch(shape_apply, dim=self.B, device=dev,
                          inputs=[self.bx, self.bq, self.scnt, self.sdx, self.sdth, self.svx, self.svth, self.btx, self.btth,
                                  self.btouch, self.bsh])
                wp.launch(floor_deep, dim=self.S, device=dev,
                          inputs=[self.bx, self.bq, self.binvm, self.binvi, self.sph_body, self.sph_local, self.sph_r, self.asleep,
                                  FLOOR_TOL, self.fdx, self.fdth, self.fcnt])
                wp.launch(floor_apply, dim=self.B, device=dev,
                          inputs=[self.bx, self.bq, self.bxo, self.fdx, self.fdth, self.fcnt, self.btx, self.btth, self.btouch, dep])
            wp.launch(body_finish, dim=self.B, device=dev,
                      inputs=[self.bx, self.bq, self.bv, self.bw, self.bxo, self.bqo, self.bvp, self.bres, self.broll,
                              self.bsize, self.asleep, self.btx, self.btth, self.btouch, self.bdv, self.bsup, self.bsupn,
                              self.bwake, self.bref, self.bqref, self.btimer, self.bdrag, self.brmin, self.banc, self.bqanc, self.bstick, self.bslip, self.slip_steps, self.wake_steps, self.bair, 4, self.slip_disp, dt,
                              self.rest_disp, self.rest_steps, self.v_limit,
                              self.fault, self.bcar, self.bcar_n, self.bcar_s, self.bwb, self.bwb_n,
                              self.rest_disp/(self.rest_steps*dt)])
            if brush:
                wp.launch(rod_apply, dim=(self.R, self.N), device=dev,
                          inputs=[self.rod_x, self.rod_v, self.rdx, self.rcnt, self.rcnt_prev, self.rod_tool, self.rod_m, self.tool_x,
                                  dt, self.tool_f, self.tool_tq])
        self.substeps_run += 1

    def begin_tick(self):
        self.tool_f.zero_(); self.tool_tq.zero_()
        self.coll_f.zero_(); self.coll_tq.zero_()
        # two snapshots of the pieces' places, the older one 10 to 20 stepped ticks old: bodies() reads velocities
        # from it (see there)
        if self.B:
            if self._snap_a is None:
                self._snap_a, self._snap_b = wp.clone(self.bx), wp.clone(self.bx)
                self._age_a = self._age_b = 0
            elif self._age_b >= NET_TICKS:
                wp.copy(self._snap_a, self._snap_b)
                wp.copy(self._snap_b, self.bx)
                self._age_a, self._age_b = self._age_b, 0
            self._age_a += 1
            self._age_b += 1

    def end_tick(self):
        """Mean force and moment over the tick: on each tool (about its block origin) and on each collider (about its
        own origin), world frame."""
        s = 1.0/SUBSTEPS
        return (self.tool_f.numpy()*s, self.tool_tq.numpy()*s, self.coll_f.numpy()[:self.K]*s, self.coll_tq.numpy()[:self.K]*s)

    # ------------------------------------------------------------------ state
    def carriers(self):
        """What carries each piece: -1 nothing known (in flight), -2 the room's floor, else the index of a shape."""
        c = self.bcar.numpy()[:self.B].copy()
        c[c >= ON_FLOOR] = -2
        return c

    def all_asleep(self):
        return (not self.B) or bool(self.asleep.numpy()[:self.B].all())

    def sync_spheres(self):
        """After a host write of piece poses."""
        if self.S:
            wp.launch(sph_update, dim=self.S, device=self.device, inputs=[self.bx, self.bq, self.sph_body, self.sph_local, self.sph_x])

    def state(self):
        keys = ("rod_x", "rod_v", "rod_anc", "rod_aj", "bx", "bq", "bv", "bw", "bres", "asleep", "bref", "bqref", "btimer", "banc", "bqanc", "bdrag", "bslip", "bcar", "bwb", "bAs", "sph_mj", "sph_mf")
        return {k: getattr(self, k).numpy().copy() for k in keys}

    def restore(self, snap):
        for k, v in snap.items():
            getattr(self, k).assign(v)
        self._snap_a = self._snap_b = None          # the places the velocities are read from belong to the old state
        self._age_a = self._age_b = 0
        if "rod_aj" not in snap:
            self.rod_aj.fill_(-2)
        if "bcar" not in snap:
            self.bcar.fill_(-1)
        for a in (self.bcar_n, self.bcar_s, self.bpj):
            a.fill_(-1)
        if "sph_mj" not in snap:
            self.sph_mj.fill_(-1)
        if "bwb" not in snap:
            self.bwb.fill_(BACK_NONE)
        self.bwb_n.fill_(BACK_NONE)
        for a in (self.bpf, self.bptq, self.bA):
            a.zero_()
        for a in (self.bdx, self.bdth, self.bcnt, self.btx, self.btth, self.btouch, self.bdv, self.bsup, self.bsupn, self.bwake, self.rdx, self.rcnt,
                  self.brdx, self.brth, self.brn, self.brn_prev):
            a.zero_()
        if self.S:
            wp.launch(sph_update, dim=self.S, device=self.device, inputs=[self.bx, self.bq, self.sph_body, self.sph_local, self.sph_x])

    def bodies(self, state_velocity=False):
        """(positions, quaternions xyzw, velocities, asleep) of the pieces as numpy arrays.

        The velocity is how far the piece went over the last 5 to 10 ms of stepped time, divided by that time. It is
        NOT the solver's own velocity state (state_velocity=True gives that): a piece squeezed between a shape and
        what it lies on is pushed into the surface in one substep and set back in the next, and its state reads 30 to
        400 mm/s for hours while the piece moves a thousandth of a millimetre (a 7 mg crumb under a brush at 58 to
        61 mm/s "straight up", 0.004 mm in two seconds). Everything that reports a
        piece's speed, the evaluator's rest test first of all, reads what the piece does."""
        x = self.bx.numpy()[:self.B].copy()
        if state_velocity or self._snap_a is None or self._age_a < 1:
            v = self.bv.numpy()[:self.B].copy()
        else:
            v = (x - self._snap_a.numpy()[:self.B])/(self._age_a*self.dt*SUBSTEPS)
        return (x, self.bq.numpy()[:self.B].copy(), v, self.asleep.numpy()[:self.B].copy())
