"""Start of tool-joints: the block infill's joints struck flush, 20-45 min old by course at 21 C, crumbs
of squeeze-out on the faces, one head joint 3 mm short; mortar on the board beside it.

setup(seed, temp_c, air, bleed, skin_s) is called by layout.setup with the seed's site conditions
(variants.site) and returns the task's wet-material regions in their start state ({name: paste.Region}).
STATION is the task's robot start, cart, tools on the cart and props (layout.station).
"""
import math

import numpy as np

from concrete_finishing import params as P
from concrete_finishing.layout import RY, _noise
from concrete_finishing.materials import MasonryMortar, masco_factor
from concrete_finishing.paste import Region

STATION = dict(robot=(4.50, 3.63, math.pi/2), cart=(5.65, 3.55, math.pi/2),
               tools=[("jointer", -0.12, 0.05, 0.0), ("margin_trowel", 0.12, 0.05, 0.0)], board=True)


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


def board_region(temp_c, seed):
    """Mortar on the mortar board on its stand next to the infill (the board's face 0.75 m up)."""
    rng = np.random.default_rng(8000 + seed)
    cx, cy, _ = STATION["cart"]
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


def setup(seed, temp_c, air, bleed, skin_s):
    regions = {}
    regions["infill"] = infill_region(seed, temp_c)
    regions["board"] = board_region(temp_c, seed)
    return regions
