"""Start of patch-flush: a floor patch of repair mortar 6 min old at 21 C, overfilled, with a void left
along its west edge and smear on the slab beside its north and east sides.

setup(seed, temp_c, air, bleed, skin_s) is called by layout.setup with the seed's site conditions
(variants.site) and returns the task's wet-material regions in their start state ({name: paste.Region}).
STATION is the task's robot start, cart, tools on the cart and props (layout.station).
"""
import numpy as np

from concrete_finishing import params as P
from concrete_finishing.layout import _noise
from concrete_finishing.materials import RepairMortar, masco_factor
from concrete_finishing.paste import Region

STATION = dict(robot=(3.75, 1.25, 0.0), cart=(3.45, 0.45, 0.0),
               tools=[("margin_trowel", 0.14, 0.12, 0.0), ("mag_float", -0.18, 0.12, 0.0), ("wood_float", -0.18, -0.10, 0.0)])
AGE_21C = 6*60.0       # s: the start's age at 21 C (layout.AGES_21C), scaled by the seed's temperature (masco_factor)


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
    age = AGE_21C*masco_factor(temp_c)
    reg.t0[:] = -age
    reg.volume0 = reg.volume()
    reg.stage = dict(task="patch-flush", recess=dict(centre=(cx, cy), size=(L, W)))
    return reg


def setup(seed, temp_c, air, bleed, skin_s):
    regions = {}
    regions["patch"] = patch_region(seed, temp_c)
    return regions
