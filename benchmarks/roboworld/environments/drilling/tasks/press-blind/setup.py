"""Start of press-blind: the aluminium block stands clamped in the drill-press vise under the spindle, its mark
punched; the press holds the 6.8 mm M8 tap drill. One scene at every seed.

setup(s) receives layout.BenchStart (layout.bench_layout calls it), appends the block to s.parts and puts its mark in
s.L["marks"]; it returns nothing. VISE_OPENING is the vise's opening in the layout (m); VISE_PRECLAMPED makes
env.DrillingEnv._prepare crank the vise closed on the block before the scene settles."""
from drilling.layout import PRESS_SPINDLE

VISE_OPENING = 0.050
VISE_PRECLAMPED = True


def setup(s):
    L, parts, vise_top = s.L, s.parts, s.vise_top
    sx, sy = PRESS_SPINDLE
    parts.append(dict(kind="block", name="alu_block", material="aluminium", in_vise=True,
                      xyz=(sx, sy, vise_top + 0.058 + 0.0126), half=(0.025, 0.025, 0.0125), rpy=(0., 0., 0.),
                      dimples=[(0., 0.)]))
    L["marks"] = [dict(part="alu_block", local=(0., 0., 0.0125))]
