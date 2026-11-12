"""Start of timber-bolt: two Douglas-fir boards lie stacked on a plywood backer, strapped down, with the bolt's mark on
the top board (within 10 mm across and 30 mm along it); the cordless drill with a 9/16 in brad-point bit and the 1/2 in
carriage bolt lie beside them.

setup(s) receives layout.BenchStart (layout.bench_layout calls it), appends the backer, the boards and the bolt to
s.parts and the drill to s.tools, and puts the mark in s.L["marks"]; it returns nothing."""
import math

from drilling.layout import BENCH


def setup(s):
    L, tools, parts, jit, wear = s.L, s.tools, s.parts, s.jit, s.wear
    # the boards lie on a sacrificial plywood backer, as a carpenter backs a through hole: laid on the steel bench
    # itself, a phantom-held drill pushed at 60 N went through the top board and then stalled with its tip on the
    # bench for the rest of 35 s - the bottom board could never be drilled through.
    # Backer and boards are free bodies; two strap clamps over the top board hold the stack down (clamps.py).
    # The bottom board lies 2 and 3 mm out of flush with the top one, as boards stacked by hand do. It also keeps the
    # stack clear of a defect of MuJoCo 3.11's box-box test: for two equal footprints exactly flush it can return a
    # contact tens of millimetres deep (29 mm at the poses captured at tick 20 of a flush stack, which threw the
    # top board up at 0.6 m/s; in bare MuJoCo: those poses again, 1 in 20 000 random near-flush poses, none in
    # 20 000 with the footprints 0.1 mm or more apart). Under the world's solve (params.SOLVER_LS_ITERATIONS) a
    # flush stack ran 6 s without one, but the collision test still answers the captured poses the same way
    parts.append(dict(kind="panel", name="ply_backer", material="plywood",
                      xyz=(3.52, -0.62, BENCH["top"] + 0.009), half=(0.085, 0.17, 0.009)))
    for k in range(2):
        dx, dy = ((0.002, 0.003), (0., 0.))[k]
        parts.append(dict(kind="board", name=f"board{k}", material="douglas_fir", clamped=k == 1,
                          xyz=(3.52 + dx, -0.62 + dy, BENCH["top"] + 0.018 + 0.019 + 0.038*k),
                          half=(0.070, 0.150, 0.019)))
    tools.append(dict(kind="drill", name="drill", xyz=(3.36, -0.95, BENCH["top"] + 0.159), rpy=(0., 0., math.pi/2),
                      bit_d_mm=14.3, bit_len=0.150, bit_type="brad", gear=1, clutch=16, wear=wear))
    parts.append(dict(kind="carriage_bolt", name="carriage_bolt", d=0.0127, length=0.1016,
                      xyz=(3.36, -0.40, BENCH["top"] + 0.0141), rpy=(0., 0., math.pi/2)))
    # the bolt's place on the boards: within 10 mm across and 30 mm along them
    L["marks"] = [dict(part="board1", local=(jit(0.010), jit(0.030), 0.019))]
