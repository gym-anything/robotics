"""Start of cut-in-trim: the wall preset's room with the angled sash brush held in the right hand, the roller lying on
the drop cloth and the cut pail on the floor.

layout.setup reads HELD, the tools in hand at the start (arm -> tool), and calls setup(spec, jit, ang) last with the
episode spec it has built and its seeded draws. It returns nothing and adds nothing: the task's start is the wall
preset's."""

HELD = dict(right="sash")


def setup(spec, jit, ang):
    """Nothing beyond the wall preset."""
