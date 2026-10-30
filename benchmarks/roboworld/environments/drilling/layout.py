"""Task layouts: what stands where for (preset, task, seed).

World frame: x east, y north, z up; the room is 8 x 6 x 3 m centred on the origin. Every number here is authored
unless a comment cites another basis. Seeds vary mark positions within the station, the
rebar mat offset and the aggregate packing, the bit wear and small placements; the layout is a pure function of
(preset, task, seed, robot, variant) so a task id reproduces its world.
"""

import hashlib
import math

import numpy as np

from . import params as P
from .cameras import bench_cameras, wall_cameras


def _rng(*key):
    h = int.from_bytes(hashlib.sha256("/".join(map(str, key)).encode()).digest()[:8], "little")
    return np.random.default_rng(h)


# ----------------------------------------------------------------------------------------------------- wall bay
WALL_REGION = dict(x=(-1.2, 1.2), z=(0.6, 2.2))          # drillable region of the north wall
DOCK_W_Y = 2.45                                           # cart front edge
SDS_BIT_D_MM = P.SDS_BIT_D_MM                             # 1/2 in SDS-plus carbide tip (ledger)
SDS_BIT_LEN_M = P.SDS_BIT_STICKOUT_M                      # Bosch HC2083 class (1/2 x 6 x 8 in): working length 152.4 mm
SDS_LONG_BIT_LEN_M = P.SDS_LONG_BIT_STICKOUT_M            # Bosch HC2084 class (1/2 x 10 x 12 in) for drilling through a ledger
ANCHOR_ROWS_Z = (1.15, 1.45)                              # the two marked rows / channel lines
LEDGER_TOP_Z = 1.20
KIT_HAMMER_HEAD_R = (P.KIT_HAMMER["head_kg"]/(3.141592653589793*P.KIT_HAMMER["head_len_m"]*7850.))**0.5   # parts.ballpeen_head_r
LEDGER_THICK_M = P.LEDGER_THICK_M                         # a 2x8 ledger: 1-1/2 x 7-1/4 in dressed
LEDGER_DEPTH_M = P.LEDGER_DEPTH_M
LEDGER_BRACKET_DX = 0.53                                  # the two temporary shelf brackets under the ledger, from its centre
# placement tolerances the evaluator (task.py) applies to every wall hole
WALL_MARK_TOL_M = P.WALL_MARK_TOL_M                       # entry within 3 mm of the mark
WALL_AXIS_TOL_DEG = P.WALL_AXIS_TOL_DEG                   # axis within 3 deg of the wall normal
MAX_LAYOUT_SHIFT_M = P.MAX_LAYOUT_SHIFT_M                 # half a rebar spacing: a shift that always clears a 300 mm mat


def rebar_clearance_m(fixture_m=0.):
    """Smallest in-plane distance from a mark to a near-mat bar's axis at which a hole drilled anywhere inside the
    evaluator's tolerances cannot touch that bar: bit radius + bar radius + placement tolerance + the lateral drift of a
    hole tilted by the axis tolerance at the near mat's far side (through a fixture of thickness `fixture_m` first) +
    REBAR_CLEAR_MARGIN_M. A real layout is drawn after the wall is scanned for reinforcement (cover meter or GPR)."""
    near_back = fixture_m + P.REBAR_COVER_M + 2*P.REBAR_D_M          # horizontal bars, then vertical bars behind them
    return (0.5e-3*SDS_BIT_D_MM + 0.5*P.REBAR_D_M + WALL_MARK_TOL_M +
            near_back*math.tan(math.radians(WALL_AXIS_TOL_DEG)) + P.REBAR_CLEAR_MARGIN_M)


def near_mat(rebar_offset):
    """Axes of the near-face reinforcement mat over the drillable region: (x of the vertical bars, z of the horizontal
    bars), exactly as concrete.ConcreteFields lays them."""
    from .concrete import ConcreteFields
    cf = ConcreteFields(0, WALL_REGION["x"], WALL_REGION["z"], thickness=P.WALL_THICK_M, rebar_offset=rebar_offset)
    return ([b["c"] for b in cf.bars if b["mat"] == "near" and b["axis"] == "z"],
            [b["c"] for b in cf.bars if b["mat"] == "near" and b["axis"] == "x"])


def _distance(v, bars):
    return min((abs(v - b) for b in bars), default=math.inf)


def clear_shift(values, bars, clearance, lo=-MAX_LAYOUT_SHIFT_M, hi=MAX_LAYOUT_SHIFT_M, step=0.0005):
    """The smallest shift s (in [lo, hi], nearest to 0 first) that puts every value + s at least `clearance` from every
    bar; None if there is none."""
    n = int(round(max(abs(lo), abs(hi))/step))
    for k in range(n + 1):
        for s in ((0.,) if k == 0 else (k*step, -k*step)):
            if lo - 1e-12 <= s <= hi + 1e-12 and all(_distance(v + s, bars) >= clearance for v in values):
                return s
    return None


def wall_layout(task, seed, robot, variant):
    rng = _rng("wall", task, seed)
    x_c = -0.55 if task == "ledger-anchor" else 0.55
    x_c += float(rng.uniform(-0.05, 0.05))
    arms = {"left_arm_mount": dict(xyz=(x_c - 0.5*P.ARM_SPACING_M, DOCK_W_Y - 0.05, P.CART_DECK_Z), yaw=math.pi/2),
            "right_arm_mount": dict(xyz=(x_c + 0.5*P.ARM_SPACING_M, DOCK_W_Y - 0.05, P.CART_DECK_Z), yaw=math.pi/2)}
    # left/right as seen by the robot facing the wall (+y): its left is -x
    L = dict(preset="wall", task=task, seed=seed, x_c=x_c, mounts=arms,
             cart=dict(center=(x_c, DOCK_W_Y - 0.35, 0.5*P.CART_DECK_Z), half=(0.45, 0.35, 0.5*P.CART_DECK_Z),
                       tray=((x_c, DOCK_W_Y - 0.35), (0.17, 0.33))),
             # folded ready pose: hands 0.3 m ahead of the bases, clear of the marks and the station camera
             ready_pose={"left": (0.0, -0.8, 0.0, -2.6, 0.0, 1.9, 0.785, 0.04, 0.04),
                         "right": (0.0, -0.8, 0.0, -2.6, 0.0, 1.9, 0.785, 0.04, 0.04)},
             rebar_offset=(float(rng.uniform(0., P.REBAR_SPACING_M)), float(rng.uniform(0., P.REBAR_SPACING_M))),
             field_seed=int(rng.integers(0, 2**31)))
    s_mat = P.REBAR_SPACING_M
    x0, z0 = WALL_REGION["x"][0], WALL_REGION["z"][0]
    marks = []
    if task in ("anchor-holes", "clean-and-set"):
        # a 2 x 2 fixture pattern (400 x 300 mm), its crosses drawn by hand (+-10 mm), then moved as a whole off the
        # near mat's bars the scan found (the smallest shift that clears every bar)
        for z in ANCHOR_ROWS_Z:
            for dx in (-0.20, 0.20):
                marks.append(dict(x=x_c + dx + float(rng.uniform(-0.01, 0.01)), z=z + float(rng.uniform(-0.01, 0.01))))
        C = rebar_clearance_m(0.)
        xs_v, zs_h = near_mat(L["rebar_offset"])
        sx = clear_shift([m["x"] for m in marks], xs_v, C)
        sz = clear_shift([m["z"] for m in marks], zs_h, C)
        for m in marks:
            m["x"] += sx
            m["z"] += sz
        L["layout_shift_m"] = (round(sx, 4), round(sz, 4))
        L["channel_lines"] = [m["z"] for m in marks[::2]]
    elif task == "rebar-relocate":
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
    else:
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
    L["marks"] = marks
    # tools on the cart tray (between the arms, near the cart's front edge) and the anchor kit
    deck = P.CART_DECK_Z
    hammer_mode = 1
    if task == "ledger-anchor":
        hammer_mode = int(rng.integers(0, 3))                           # seeded mode knob
    wear = 1.0 if variant != "worn-bit" else 1.58                       # Antonucci 2017: 58 % longer per hole at 57 m
    bit_len = SDS_LONG_BIT_LEN_M if task == "ledger-anchor" else SDS_BIT_LEN_M
    tools = [dict(kind="hammer", name="rotary_hammer", xyz=(x_c - 0.02, DOCK_W_Y - 0.22, deck + 0.012 + 0.200),
                  rpy=(0., 0., math.pi/2), bit_d_mm=SDS_BIT_D_MM, bit_len=bit_len, mode=hammer_mode, wear=wear)]
    parts = []
    if task in ("clean-and-set", "ledger-anchor"):
        # the kit trolley beside the right arm (stations.build_wall): top at z 0.792, centred (x_c + 0.66, 2.20)
        tx, kz = x_c + 0.66, 0.792
        # every kit part lies along y (the trolley's long side), side by side along x
        parts += [dict(kind="nozzle", name="vacuum_nozzle", xyz=(tx - 0.15, 2.29, kz + 0.019), rpy=(0., 0., -math.pi/2)),
                  # (the brush on its own rest, its axis 14 mm above the tray: stations.build_wall)
                  dict(kind="brush", name="hole_brush", xyz=(tx - 0.09, 2.30, kz + 0.0141), rpy=(0., 0., -math.pi/2)),
                  dict(kind="ballpeen", name="hammer", xyz=(tx - 0.02, 2.20, kz + KIT_HAMMER_HEAD_R + 0.0005), rpy=(0., 0., math.pi/2))]
        # KB-TZ2 1/2 in: the washer seats on the fixture when the tip is at h_nom (Hilti KB-TZ2 Technical Supplement
        # Table 47: 1/2 x 4-1/2 (G) takes a 0-1/4 in fixture at h_nom 3-3/4 in, the washer on the bare wall here;
        # 1/2 x 7 (L) takes 0-2-3/4 in, the 1-1/2 in ledger)
        if task == "clean-and-set":
            length, seat = P.KBTZ2_WALL_LENGTH_M, P.KBTZ2["h_nom"]
        else:
            length, seat = P.KBTZ2_LEDGER_LENGTH_M, P.KBTZ2["h_nom"] + LEDGER_THICK_M
        for k in range(4):
            parts.append(dict(kind="anchor", name=f"anchor{k}", xyz=(tx + 0.05 + 0.037*k, 2.27, kz + 0.0182),
                              rpy=(0., 0., -math.pi/2), length=length, seat=seat))
        L["vacuum"] = vacuum_spec(x_c)
        # the setting tool (ESR-4266 4.2.3: a torque wrench, or the Hilti impact wrench with its adaptive-torque module,
        # permitted for the 1/2 in KB-TZ2 - but Hilti's installation instructions 2274242-03.2025 add "AT-System not
        # applicable for grout-filled CMU and wood fastening applications", so on ledger-anchor's timber ledger it is a
        # disclosed deviation from the maker's method, whose sanctioned tool there is a torque wrench): a SIW 6AT-A22
        # class wrench with its SI-AT-A22 module set for these anchors and a 3/4 in deep socket on its anvil, standing
        # on its battery along the
        # trolley's near edge (y 1.955-2.035, x tx..tx+0.12: clear of the hammer's handle end at y 2.035 and of the
        # anchors' tips at y >= 2.09), its anvil along +x to the trolley's end (tools.ImpactWrench: origin at the
        # socket's mouth, battery 0.21 m below it)
        tools.append(dict(kind="impact_wrench", name="impact_wrench", xyz=(tx + 0.20, 1.995, kz + 0.2105 + 0.0005),
                          rpy=(0., 0., 0.), target_nm=P.KBTZ2["torque"]))
    if task == "clean-and-set":
        L["predrilled"] = [dict(mark=k, depth=0.112 + 0.004*float(rng.uniform()), dust_mm=float(rng.uniform(14., 22.)))
                           for k in range(4)]
    L["tools"], L["parts"] = tools, parts
    L["cameras"] = wall_cameras(x_c, sum(m["z"] for m in marks)/len(marks))
    L["focus"] = (x_c, 2.9, 1.3)
    return L


# (x - x_c, y, yaw) of the shop vacuum (vacuum.py). Yaw 0 points its hose inlet along the wall into the open floor: at
# -pi/2 the inlet faces the kit trolley's back 2 cm away, where no hose could leave it
VACUUM_AT = (0.66, 2.72, 0.)


def vacuum_spec(x_c):
    """The shop vacuum on the floor beside the cart, behind the kit trolley, its inlet along the wall into the room."""
    from .vacuum import geometry
    g = geometry()
    dx, y, yaw = VACUUM_AT
    return dict(xyz=[x_c + dx, y, 0.], yaw=yaw, port_local=[round(float(v), 4) for v in g["port_local"]],
                switch_local=[round(float(v), 4) for v in g["switch_local"]],
                switch_off_local=[round(float(v), 4) for v in g["switch_off_local"]], switch_normal_local=[0., 0., 1.])


# ----------------------------------------------------------------------------------------------------- bench
BENCH = dict(x=(3.24, 4.0), y=(-1.18, 0.65), top=P.BENCH_TOP_Z)
# the centre punch on its two-rail rest (rails 8 mm high, 10 mm apart): axis height above the bench, 0.2 mm clear
PUNCH_REST_Z = 0.008 + math.sqrt((0.5*P.PUNCH_BODY_M[1])**2 - 0.005**2) + 0.0002
DOCK_B_X = 3.0
PRESS_SPINDLE = (3.42, -1.45)
PRESS_BITS = P.PRESS_BITS


def bench_layout(task, seed, robot, variant):
    rng = _rng("bench", task, seed)
    # what the seed varies on the bench beyond the loose bar's yaw (a stream of its own, so that the draws above keep
    # their values): where a loose part or tool lies, where the marks are, how the punch's cap is set
    rv = _rng("bench-vary", task, seed)
    jit = lambda a: float(rv.uniform(-a, a))
    yB = -1.10
    arms = {"left_arm_mount": dict(xyz=(DOCK_B_X, yB + 0.5*P.ARM_SPACING_M, P.CART_DECK_Z), yaw=0.),
            "right_arm_mount": dict(xyz=(DOCK_B_X, yB - 0.5*P.ARM_SPACING_M, P.CART_DECK_Z), yaw=0.)}
    L = dict(preset="bench", task=task, seed=seed, mounts=arms,
             cart=dict(center=(DOCK_B_X - 0.30, yB, 0.5*P.CART_DECK_Z), half=(0.35, 0.45, 0.5*P.CART_DECK_Z),
                       tray=((DOCK_B_X - 0.30, yB), (0.33, 0.17))),
             # the right arm starts turned towards the bench, its hand clear of the drill-press head
             ready_pose={"left": (-0.2, -0.8, 0.0, -2.6, 0.0, 1.9, 0.785, 0.04, 0.04),
                         "right": (0.7, -0.8, 0.0, -2.6, 0.0, 1.9, 0.785, 0.04, 0.04)})
    tools, parts = [], []
    sx, sy = PRESS_SPINDLE
    bit_d, bit_len, belt = PRESS_BITS.get(task, (9.0, 0.081, 1000.))
    # drill press with yaw pi: its front (local +x) faces the robot (-x); its spindle at PRESS_SPINDLE
    tools.append(dict(kind="press", name="drill_press", xy=(sx + 0.19, sy), yaw=math.pi, belt_rpm=belt,
                      bit_d_mm=bit_d, bit_len=bit_len))
    vise_top = P.PRESS_STAND_M + 0.204                   # the press table (tools.DrillPress, measured on its model)
    loose = variant == "loose-vise"
    opening = 0.050 if task == "press-blind" else 0.075
    # vise frame x points at the robot (yaw pi); a 50 mm part clamped against the fixed jaw (face at local x = -0.07)
    # has its centre line at local x = -0.045: the vise is bolted with that line under the spindle
    vise_pose = dict(kind="vise", name="press_vise", xyz=(sx - 0.045, sy, vise_top), rpy=(0., 0., math.pi),
                     opening=opening, bolted=not loose)
    tools.append(vise_pose)
    mat_xy = (3.50, -0.55)
    wear = 1.3 if variant == "worn-bit" else 1.0
    if task == "take-drill":
        # the drill stands where the last user left it: within 10 mm and 3 degrees of its place
        tools.append(dict(kind="drill", name="drill", xyz=(3.42 + jit(0.010), -1.00 + jit(0.010), BENCH["top"] + 0.159),
                          rpy=(0., 0., math.pi/2 + jit(0.05)), bit_d_mm=3.0, bit_len=0.045, gear=2, clutch=16, wear=wear))
    if task == "punch-layout":
        parts.append(dict(kind="bar", name="flat_bar", material="steel",
                          xyz=(3.50 + jit(0.010), -0.62 + jit(0.010), BENCH["top"] + 0.003),
                          half=(0.025, 0.075, 0.003), rpy=(0., 0., float(rng.uniform(-0.1, 0.1)))))
        tools.append(dict(kind="punch", name="centre_punch", xyz=(3.40, -0.90, BENCH["top"] + PUNCH_REST_Z),
                          rpy=(0., 0., math.pi/2), holder=True, release_n=float(rv.uniform(*P.PUNCH_RELEASE_RANGE_N))))
        L["marks"] = [dict(part="flat_bar", local=(0., y, 0.003)) for y in (-0.050, -0.017, 0.017, 0.050)]
    if task in ("press-drill", "match-bolt"):
        punched = task == "press-drill"
        parts.append(dict(kind="bar", name="flat_bar", material="steel", xyz=(3.47, -1.05, BENCH["top"] + 0.003),
                          half=(0.025, 0.075, 0.003), rpy=(0., 0., math.pi/2 + float(rng.uniform(-0.1, 0.1))),
                          dimples=[(0., -0.030)] if punched else []))
        L["marks"] = [dict(part="flat_bar", local=(0., -0.030, 0.003))]
    if task == "press-blind":
        parts.append(dict(kind="block", name="alu_block", material="aluminium", in_vise=True,
                          xyz=(sx, sy, vise_top + 0.058 + 0.0126), half=(0.025, 0.025, 0.0125), rpy=(0., 0., 0.),
                          dimples=[(0., 0.)]))
        L["marks"] = [dict(part="alu_block", local=(0., 0., 0.0125))]
    if task == "match-bolt":
        # a 50 x 50 x 5 equal angle (EN 10056-1) on the 50 x 6 flat (EN 10058): the same width, so that the vise's
        # jaws hold BOTH parts by their edges while they are match-drilled. With a 40 mm angle the jaws hold only the
        # wider bar and nothing clamps the bracket
        parts.append(dict(kind="angle", name="angle_bracket", material="steel", xyz=(3.52, -0.85, BENCH["top"] + 0.0025),
                          half=(0.025, 0.060, 0.0025), rpy=(0., 0., math.pi/2)))
        for k in range(2):
            parts.append(dict(kind="bolt", name=f"m8_bolt{k}", d=0.008, length=0.030, xyz=(3.36 + 0.03*k, -0.70,
                              BENCH["top"] + 0.0066), rpy=(0., 0., math.pi/2)))
        L["marks"] = [dict(part="angle_bracket", local=(0., y, 0.0025)) for y in (-0.030, 0.030)]
    if task == "panel-holes":
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
    if task == "timber-bolt":
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
    L["mat"] = dict(xy=mat_xy)
    L["tools"], L["parts"] = tools, parts
    focus = (parts or tools)[-1 if not parts else 0]["xyz"]          # the first workpiece, else the hand tool
    marked = {m["part"] for m in L.get("marks", []) if "part" in m}
    if marked:                                                       # ... or the workpiece that carries the marks
        focus = next((p_["xyz"] for p_ in parts if p_.get("name") in marked), focus)
    L["cameras"] = bench_cameras(sx, sy, focus)
    L["focus"] = (sx, sy, 1.05)
    return L


def task_layout(config):
    if config.preset == "wall":
        return wall_layout(config.task, config.seed, config.robot, config.variant)
    if config.preset == "bench":
        return bench_layout(config.task, config.seed, config.robot, config.variant)
    raise ValueError(f"preset {config.preset} has no generated layout")
