"""Start of tool-sealant: a gunned sealant bead in the control joint, 10 min old at 21 C, proud, with two
gun stops and a 40 mm thin stretch that does not touch the joint's north wall.

setup(seed, temp_c, air, bleed, skin_s) is called by layout.setup with the seed's site conditions
(variants.site) and returns the task's wet-material regions in their start state ({name: paste.Region}).
STATION is the task's robot start, cart, tools on the cart and props (layout.station).
"""
import numpy as np

from concrete_finishing import params as P
from concrete_finishing.layout import RX, _noise
from concrete_finishing.materials import Sealant, masco_factor
from concrete_finishing.paste import Region

STATION = dict(robot=(4.85, 1.75, 0.0), cart=(4.45, 2.85, 0.0),
               tools=[("sealant_tool", 0.0, 0.0, 0.0)], rag=(-0.24, 0.06))
AGE_21C = 10*60.0      # s: the start's age at 21 C (layout.AGES_21C), scaled by the seed's temperature (masco_factor)


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
    age = AGE_21C*masco_factor(temp_c)
    reg.t0[:] = -age
    reg.volume0 = reg.volume()
    reg.stage = dict(task="tool-sealant", segment=(za, zb))
    return reg


def setup(seed, temp_c, air, bleed, skin_s):
    regions = {}
    regions["joint"] = joint_region(seed, temp_c, skin_s)
    return regions
