"""Start of spray-stencil: the spray preset's folding table and open spray box with a crate side lying in it under a
stencil cut with a "3", and the can of red gloss enamel held in the left hand.

layout.setup reads HELD, the tools in hand at the start (arm -> tool), and PAINT, the aerosol's product, and calls
setup(spec, jit, ang) last with the episode spec it has built and its seeded draws (jit(s): an offset within +-s m,
ang(s): an angle within +-s deg, both 0 for seed 0). setup returns nothing; it puts the crate side and the stencil
on it into the box (spec["items"]), shifted and turned together by the seed."""
from painting import params as P
from painting.layout import yaw_quat

HELD = dict(left="aerosol")
PAINT = dict(kind="enamel", color="gloss_red")


def setup(spec, jit, ang):
    items, zt = spec["items"], P.TABLE["height"]
    cj, sj, a = jit(0.02), jit(0.02), ang(3)
    items["crate"] = dict(kind="crate", pose=(-0.10 + cj, 1.33 + sj, zt + 0.004, yaw_quat(a)))
    items["stencil"] = dict(kind="stencil", pose=(-0.10 + cj, 1.33 + sj, zt + 0.004 + 0.012, yaw_quat(a)))
