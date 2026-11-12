"""Start of panel-holes: the steel enclosure cover lies strapped on its plywood backer on the bench with its four hole
marks (each within 10 x 15 mm of its place on the drawing); a 3 mm pilot drill, a 6 mm drill and the centre punch lie
beside it.

setup(s) receives layout.BenchStart (layout.bench_layout calls it), appends the cover and the backer to s.parts and the
two drills and the punch to s.tools, and puts the four marks in s.L["marks"]; it returns nothing."""
import math

from drilling import params as P
from drilling.layout import BENCH, PUNCH_REST_Z


def setup(s):
    L, tools, parts, rv, jit, wear = s.L, s.tools, s.parts, s.rv, s.jit, s.wear
    # the panel lies on its plywood backer on the bench, both free; two strap clamps hold the panel down (clamped:
    # stations.build_bench, clamps.py)
    parts.append(dict(kind="panel", name="cover_panel", material="steel", clamped=True,
                      xyz=(3.52, -0.62, BENCH["top"] + 0.018 + 0.00075), half=(0.10, 0.125, 0.00075)))
    parts.append(dict(kind="panel", name="ply_backer", material="plywood",
                      xyz=(3.52, -0.62, BENCH["top"] + 0.009), half=(0.11, 0.14, 0.009)))
    tools.append(dict(kind="drill", name="pilot_drill", xyz=(3.36, -0.95, BENCH["top"] + 0.159), rpy=(0., 0., math.pi/2),
                      bit_d_mm=3.0, bit_len=0.045, gear=2, clutch=16, wear=wear))
    tools.append(dict(kind="drill", name="drill", xyz=(3.36, -0.30, BENCH["top"] + 0.159), rpy=(0., 0., math.pi/2),
                      bit_d_mm=6.0, bit_len=0.055, gear=2, clutch=16, wear=wear))
    tools.append(dict(kind="punch", name="centre_punch", xyz=(3.66, -0.95, BENCH["top"] + PUNCH_REST_Z),
                      rpy=(0., 0., math.pi/2), holder=True, release_n=float(rv.uniform(*P.PUNCH_RELEASE_RANGE_N))))
    # the four holes of a cover's drawing: each within 10 x 15 mm of its nominal place (clear of the two clamps at
    # opposite corners and at least 25 mm from an edge)
    L["marks"] = [dict(part="cover_panel", local=(x + jit(0.010), y + jit(0.015), 0.00075))
                  for x in (-0.06, 0.06) for y in (-0.08, 0.08)]
