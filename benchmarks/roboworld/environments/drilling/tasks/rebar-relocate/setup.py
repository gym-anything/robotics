"""Start of rebar-relocate: two channel lines with two anchor crosses each on the concrete wall, drawn without a
reinforcement scan, so that a vertical bar of the near mat lies under one of the four crosses (seeded which); the
rotary hammer lies on the cart tray.

setup(s) receives layout.WallStart (layout.wall_layout calls it), appends the four marks to s.marks and puts the rebar
offset, the mark over the bar, the channel lines and the replacement's window in s.L; it returns nothing. STATION_X is
the station's centre along the wall before the seed moves it."""
from drilling import params as P
from drilling.layout import ANCHOR_ROWS_Z, WALL_REGION, clear_shift, near_mat, rebar_clearance_m

STATION_X = 0.55


def setup(s):
    L, rng, x_c, marks = s.L, s.rng, s.x_c, s.marks
    s_mat = P.REBAR_SPACING_M
    x0, z0 = WALL_REGION["x"][0], WALL_REGION["z"][0]
    # two horizontal strut channels (the "channel lines"), two anchors on each 400 mm apart; drawn without a scan:
    # a vertical bar of the near mat lies under one anchor (seeded which), across its channel line. The horizontal
    # bars run between the lines (a channel along a bar would be an all-steel line), and the other channel's anchors
    # stand where its slot let them clear the vertical bars (anchors slide along a slotted channel)
    lines = [z + float(rng.uniform(-0.01, 0.01)) for z in ANCHOR_ROWS_Z[:1]]
    lines.append(lines[0] + (ANCHOR_ROWS_Z[1] - ANCHOR_ROWS_Z[0]))
    for z in lines:
        off = float(rng.uniform(-0.02, 0.02))
        for dx in (-0.20, 0.20):
            marks.append(dict(x=x_c + dx + off + float(rng.uniform(-0.01, 0.01)), z=z))
    k = int(rng.integers(0, len(marks)))
    C = rebar_clearance_m(0.)
    x_bar = marks[k]["x"] + float(rng.uniform(-0.003, 0.003))
    phase = float(rng.uniform(C, s_mat - C))                          # horizontal bars between the channel lines
    L["rebar_offset"] = ((x_bar - x0) % s_mat, (lines[0] - phase - z0) % s_mat)
    xs_v, zs_h = near_mat(L["rebar_offset"])
    other = [i for i in range(4) if i // 2 != k // 2]
    so = clear_shift([marks[i]["x"] for i in other], xs_v, C)
    for i in other:
        marks[i]["x"] += so
    L["rebar_mark"] = k
    L["channel_lines"] = lines
    # where the replacement may go (task.py judges it from the aborted hole's actual depth): on the channel line,
    # at least ABORTED_HOLE_DISTANCE_FACTOR x the aborted depth and at most RELOCATE_MAX_M from it. The sites below
    # (+- along the line) are for the
    # feasibility and reach probes: twice an aborted depth of the vertical bar's face depth (cover + the horizontal
    # bar in front of it) plus 10 mm the bit runs on before it is stopped)
    d_abort = P.REBAR_COVER_M + P.REBAR_D_M + 0.010
    d_nom = max(P.ABORTED_HOLE_DISTANCE_FACTOR*d_abort, C)
    L["replacement"] = dict(z=marks[k]["z"], x_aborted=marks[k]["x"], min_m=round(d_nom, 4), max_m=P.RELOCATE_MAX_M,
                            x_sites=[marks[k]["x"] - d_nom, marks[k]["x"] + d_nom])
