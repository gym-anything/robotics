"""Start of load-roller: the wall preset's room (the paint tray, cut pail and gallon can on the drop cloth) with the
roller held in the right hand, its cover dry.

layout.setup reads HELD, the tools in hand at the start (arm -> tool), and calls setup(spec, jit, ang) last with the
episode spec it has built and its seeded draws (jit(s): an offset within +-s m, ang(s): an angle within +-s deg, both
0 for seed 0). It returns nothing and adds nothing: the task's start is the wall preset's."""

HELD = dict(right="roller")


def setup(spec, jit, ang):
    """Nothing beyond the wall preset."""
