"""Start of roll-strip: the wall preset's room with the roller held in the right hand; on the work wall the bands
already brushed under the ceiling and above the baseboard and, on the left, the strip rolled two minutes earlier,
still wet; blue tape marks at the strip's two edges.

layout.setup reads HELD, the tools in hand at the start (arm -> tool), and calls setup(spec, jit, ang) last with the
episode spec it has built and its seeded draws. setup returns nothing; it puts the wet films in spec["prepainted"]
(system.PaintSystem lays them, aged). STRIP is the marked strip (x0, x1, z0, z1 in world metres; the verifier's
target) and TAPE the render's tape marks (x, z) at its edges."""
from painting.layout import BB, Z

HELD = dict(right="roller")
# the strip on the wall chart (x0, x1, z0, z1) in world metres
STRIP = dict(x0=-0.75, x1=-0.15, z0=0.15, z1=Z - 0.06)
# blue painter's tape marks for the strip (on the wall, render-only marks)
TAPE = [(STRIP["x0"], 0.30), (STRIP["x1"], 0.30), (STRIP["x0"], 2.20), (STRIP["x1"], 2.20)]


def setup(spec, jit, ang):
    # wet bands already brushed for the roll-strip task (painter's order: cut in first, roll into it), and the strip
    # rolled two minutes earlier on the left (seeded wet-edge age)
    spec["prepainted"] = [dict(region=(STRIP["x0"], STRIP["x1"] + 0.6, Z - 0.075, Z), h=110e-6, age=60.0),
                          dict(region=(STRIP["x0"], STRIP["x1"] + 0.6, BB["h"], BB["h"] + 0.075), h=110e-6, age=60.0),
                          dict(region=(STRIP["x0"] - 0.6, STRIP["x0"], BB["h"], Z), h=105e-6, age=120.0)]
