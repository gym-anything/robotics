"""Start of patch-drywall: a spackled mesh patch with knife ridges, 4 min old at 21 C, the anchor holes
beside it still open; compound on the hawk on the cart.

setup(seed, temp_c, air, bleed, skin_s) is called by layout.setup with the seed's site conditions
(variants.site) and returns the task's wet-material regions in their start state ({name: paste.Region}).
STATION is the task's robot start, cart, tools on the cart and props (layout.station).
"""
import math

import numpy as np

from concrete_finishing import params as P
from concrete_finishing.layout import CART_TOP, _noise
from concrete_finishing.materials import Compound, masco_factor
from concrete_finishing.paste import Region

STATION = dict(robot=(1.82, 1.25, -math.pi/2), cart=(2.90, 0.55, 0.0),
               # the knives clear of the hawk (local x -0.415..-0.085): at (-0.10, 0.04) the joint knife lay
               # over the hawk's edge and the hand that took it touched the hawk
               tools=[("joint_knife", 0.08, -0.08, 0.0), ("putty_knife", 0.30, -0.08, 0.0)], hawk=True)
AGE_21C = 4*60.0       # s: the start's age at 21 C (layout.AGES_21C), scaled by the seed's temperature (masco_factor)


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
    age = AGE_21C*masco_factor(temp_c)
    reg.t0[:] = -age
    reg.volume0 = reg.volume()
    reg.stage = dict(task="patch-drywall", holes=len(P.DW_HOLES))
    return reg


def hawk_region(temp_c, seed):
    """Compound on the hawk (the hawk lies on the cart top, its face 15 mm above it)."""
    rng = np.random.default_rng(7000 + seed)
    cx, cy, cyaw = STATION["cart"]
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


def setup(seed, temp_c, air, bleed, skin_s):
    regions = {}
    regions["wallboard"] = drywall_region(seed, temp_c)
    regions["hawk"] = hawk_region(temp_c, seed)
    return regions
