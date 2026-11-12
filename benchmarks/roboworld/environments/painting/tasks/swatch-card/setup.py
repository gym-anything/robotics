"""Start of swatch-card: the craft preset's folding table with the round brush held in the right hand and a 50 x 50 mm
square pencilled on the watercolour paper.

layout.setup reads HELD, the tools in hand at the start (arm -> tool), and calls setup(spec, jit, ang) last with the
episode spec it has built and its seeded draws; it returns nothing and adds nothing to the craft preset.
system.PaintSystem draws the template with pencil(ch, X, Y, line): the paper chart, its cell centres (paper frame, m)
and the line's half width, returning the mask of graphite cells."""
import numpy as np

HELD = dict(right="round8")


def setup(spec, jit, ang):
    """Nothing beyond the craft preset."""


def pencil(ch, X, Y, line):
    m = ((np.abs(np.abs(X) - 0.025) < line) & (np.abs(Y) <= 0.025 + line)) | \
        ((np.abs(np.abs(Y) - 0.025) < line) & (np.abs(X) <= 0.025 + line))
    return m
