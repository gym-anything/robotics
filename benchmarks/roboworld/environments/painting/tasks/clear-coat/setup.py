"""Start of clear-coat: the spray preset's folding table and open spray box with a 10 x 8 in photo print lying on the
box's floor, and the can of clear lacquer held in the left hand.

layout.setup reads HELD, the tools in hand at the start (arm -> tool), and PAINT, the aerosol's product, and calls
setup(spec, jit, ang) last with the episode spec it has built and its seeded draws. setup returns nothing; it puts the
print in the box (spec["print"]: system.PaintSystem charts it)."""

HELD = dict(left="aerosol")
PAINT = dict(kind="lacquer", color="clear")


def setup(spec, jit, ang):
    spec["print"] = True
