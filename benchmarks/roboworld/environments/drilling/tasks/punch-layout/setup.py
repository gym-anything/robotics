"""Start of punch-layout: the steel flat bar lies loose on the bench (within 10 mm and 6 degrees of its place) with its
four layout marks along its centre line, and the centre punch lies on its rest, its cap set by the seed (60 to 130 N).

setup(s) receives layout.BenchStart (layout.bench_layout calls it), appends the bar to s.parts and the punch to s.tools
and puts the four marks in s.L["marks"]; it returns nothing."""
import math

from drilling import params as P
from drilling.layout import BENCH, PUNCH_REST_Z


def setup(s):
    L, tools, parts, rng, rv, jit = s.L, s.tools, s.parts, s.rng, s.rv, s.jit
    parts.append(dict(kind="bar", name="flat_bar", material="steel",
                      xyz=(3.50 + jit(0.010), -0.62 + jit(0.010), BENCH["top"] + 0.003),
                      half=(0.025, 0.075, 0.003), rpy=(0., 0., float(rng.uniform(-0.1, 0.1)))))
    tools.append(dict(kind="punch", name="centre_punch", xyz=(3.40, -0.90, BENCH["top"] + PUNCH_REST_Z),
                      rpy=(0., 0., math.pi/2), holder=True, release_n=float(rv.uniform(*P.PUNCH_RELEASE_RANGE_N))))
    L["marks"] = [dict(part="flat_bar", local=(0., y, 0.003)) for y in (-0.050, -0.017, 0.017, 0.050)]
