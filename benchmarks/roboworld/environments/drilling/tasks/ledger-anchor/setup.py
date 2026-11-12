"""Start of ledger-anchor: a 2x8 timber ledger stands level on two temporary brackets against the concrete wall with
its four anchor points marked on it (staggered top and bottom, moved clear of the near mat's bars); the rotary hammer
with a long bit lies on the cart tray, its mode knob set by the seed, and the anchor kit, the impact wrench and the shop
vacuum stand on and beside the kit trolley.

setup(s) receives layout.WallStart (layout.wall_layout calls it), appends the four marks to s.marks, puts the ledger
and the layout shift in s.L and sets the hammer's mode and bit; it returns nothing. STATION_X is the station's centre
along the wall before the seed moves it; KIT is the (length, seat) of the four wedge anchors on the kit trolley
(layout.kit_anchors)."""
from drilling import params as P
from drilling.layout import SDS_BIT_D_MM, clear_shift, near_mat, rebar_clearance_m

STATION_X = -0.55
SDS_LONG_BIT_LEN_M = P.SDS_LONG_BIT_STICKOUT_M            # Bosch HC2084 class (1/2 x 10 x 12 in) for drilling through a ledger
LEDGER_TOP_Z = 1.20
LEDGER_THICK_M = P.LEDGER_THICK_M                         # a 2x8 ledger: 1-1/2 x 7-1/4 in dressed
LEDGER_DEPTH_M = P.LEDGER_DEPTH_M
LEDGER_BRACKET_DX = 0.53                                  # the two temporary shelf brackets under the ledger, from its centre
# KB-TZ2 1/2 x 7 (L): it takes a 0-2-3/4 in fixture, here the 1-1/2 in ledger, its washer seated on the ledger when the
# tip is at h_nom (Hilti KB-TZ2 Technical Supplement Table 47)
KIT = (P.KBTZ2_LEDGER_LENGTH_M, P.KBTZ2["h_nom"] + LEDGER_THICK_M)


def setup(s):
    L, rng, x_c, marks = s.L, s.rng, s.x_c, s.marks
    # ledger anchors staggered top / bottom (IRC Table R507.9.1.3(2): 2 in from the top edge, 3/4 in from the
    # bottom edge, rows 1-5/8 in apart, 2 in from the ends), at 300 mm; the whole pattern moved along the ledger and a
    # row moved inwards by the least that clears the near mat's bars (the scan)
    top = LEDGER_TOP_Z
    z_top, z_bot = top - P.LEDGER_EDGE_TOP_M, top - LEDGER_DEPTH_M + P.LEDGER_EDGE_BOTTOM_M
    C = rebar_clearance_m(LEDGER_THICK_M)
    xs_v, zs_h = near_mat(L["rebar_offset"])
    xs = [x_c + dx for dx in (-0.45, -0.15, 0.15, 0.45)]
    # the pattern may move as far as the ledger's ends allow (it is 1.2 m long) and no closer to a temporary
    # bracket (its lip stands in front of the ledger's lower edge, 25 mm either side of x_c +- LEDGER_BRACKET_DX)
    room_x = min(0.60 - 0.45 - P.LEDGER_ENDS_M, LEDGER_BRACKET_DX - 0.025 - 0.5e-3*SDS_BIT_D_MM - 0.010 - 0.45)
    sx = clear_shift(xs, xs_v, C, -room_x, room_x)
    room = (z_top - z_bot) - P.LEDGER_ROW_SPACING_M
    st = clear_shift([z_top], zs_h, C, -room, 0.)                     # the top row may only move down
    sb = clear_shift([z_bot], zs_h, C, 0., room)                      # the bottom row only up
    if st is not None and sb is not None and (z_top + st) - (z_bot + sb) < P.LEDGER_ROW_SPACING_M - 1e-9:
        st = sb = None
    for k, x in enumerate(xs):
        z = (z_top + (st or 0.)) if k % 2 == 0 else (z_bot + (sb or 0.))
        marks.append(dict(x=x + (sx or 0.), z=z))
    L["layout_shift_m"] = (None if sx is None else round(sx, 4), None if st is None else round(st, 4),
                           None if sb is None else round(sb, 4))
    L["ledger"] = dict(x_c=x_c, top_z=top, thickness=LEDGER_THICK_M, depth=LEDGER_DEPTH_M, standoff=0.,
                       bracket_dx=LEDGER_BRACKET_DX)
    s.hammer_mode = int(rng.integers(0, 3))                         # seeded mode knob
    s.bit_len = SDS_LONG_BIT_LEN_M
