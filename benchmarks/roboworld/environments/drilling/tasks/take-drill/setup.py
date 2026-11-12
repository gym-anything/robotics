"""Start of take-drill: the cordless drill with its 3 mm bit stands on the bench where the last user left it, within
10 mm and 3 degrees of its place, beside the drill press.

setup(s) receives layout.BenchStart (layout.bench_layout calls it) and appends the drill to s.tools; it returns
nothing."""
import math

from drilling.layout import BENCH


def setup(s):
    tools, jit, wear = s.tools, s.jit, s.wear
    # the drill stands where the last user left it: within 10 mm and 3 degrees of its place
    tools.append(dict(kind="drill", name="drill", xyz=(3.42 + jit(0.010), -1.00 + jit(0.010), BENCH["top"] + 0.159),
                      rpy=(0., 0., math.pi/2 + jit(0.05)), bit_d_mm=3.0, bit_len=0.045, gear=2, clutch=16, wear=wear))
