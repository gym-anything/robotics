"""Start of brush-stroke: the craft preset's folding table with the round brush held in the right hand and the outline
of an S-stroke pencilled on the watercolour paper, swelling from 2 mm to 6 mm wide and back.

layout.setup reads HELD, the tools in hand at the start (arm -> tool), and calls setup(spec, jit, ang) last with the
episode spec it has built and its seeded draws; it returns nothing and adds nothing to the craft preset.
system.PaintSystem draws the template with pencil(ch, X, Y, line): the paper chart, its cell centres (paper frame, m)
and the line's half width, returning the mask of graphite cells. S_CURVE and stroke_width are the template (the
verifier's target follows them)."""
import math

import numpy as np

HELD = dict(right="round8")

S_CURVE = lambda t: np.stack([-0.10 + 0.20*t, 0.035*np.sin(2*math.pi*t)], -1)        # paper frame, m


def stroke_width(t):
    """Template width along the S stroke: 2 -> 6 -> 2 mm (half-sine swell): a #8 round's thick-and-thin stroke, from a
    hairline on the tip to the belly pressed to ~6 mm (its 6-8 mm spread)."""
    return 2e-3 + 4e-3*np.sin(math.pi*t)


def setup(spec, jit, ang):
    """Nothing beyond the craft preset."""


def pencil(ch, X, Y, line):
    m = np.zeros(X.shape, bool)
    ts = np.linspace(0, 1, 800)
    c = S_CURVE(ts)
    tang = np.gradient(c, axis=0)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)/np.linalg.norm(tang, axis=1, keepdims=True)
    for side in (-1, 1):
        pts = c + side*nrm*(stroke_width(ts)/2)[:, None]
        ii = np.clip(((pts[:, 0] + ch.size[0]/2)/ch.dx).astype(int), 0, ch.nu - 1)
        jj = np.clip(((pts[:, 1] + ch.size[1]/2)/ch.dx).astype(int), 0, ch.nv - 1)
        m[ii, jj] = True
    return m
