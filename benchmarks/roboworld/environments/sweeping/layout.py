"""Room 112: installed parts, furniture, where the tools hang, and the start state of every task.

World frame as room 114 (classroom_arrangement): origin on the floor at the middle of the board wall, x across the
room (+x toward the door side), y from the board wall to the back wall, z up.
Free bodies: six activity tables table_<A|B><1-3> with four chairs each (chair_<table>_<1-4>), the teacher's desk and
chair, the tools, the containers, and on the craft table the supplies of task T6. Everything else is installed.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation
from classroom_arrangement import params as CP
from classroom_arrangement import layout as CL
from classroom_arrangement.furniture import Part
from . import params as P
from . import toolkit as T
from . import debris as D

X0, Y0, X1, Y1 = P.ROOM
TABLE_H = CP.DESK_HEIGHT
PRONG_R = 0.004
PRONG_LEN = 0.085
RAIL_X = X1 - 0.045              # handle axis of a hanging broom: 45 mm off the wall

# ---------------------------------------------------------------------------------------------- installed parts
# (name, centre, half extents, material, friction): colliders of the rigid world and of the debris solver, and drawn
_cw = P.CASEWORK
_cb = P.CUBBIES
_cove_h, _cove_t = P.COVE_BASE
_toe_h, _toe_d = P.TOE_SPACE
STATIC = [(n, c, h, m, 0.4) for n, c, h, m in CL.STATIC if n not in ("bookcase", "cubbies")]
STATIC += [
    # casework along the back wall: carcass above the toe space, kick board set back 75 mm, counter top
    ("casework_carcass", ((_cw["x"][0] + _cw["x"][1])/2, Y1 - _cw["depth"]/2, (_toe_h + _cw["top"] - 0.03)/2),
     ((_cw["x"][1] - _cw["x"][0])/2, _cw["depth"]/2, (_cw["top"] - 0.03 - _toe_h)/2), "casework", 0.4),
    ("casework_kick", ((_cw["x"][0] + _cw["x"][1])/2, Y1 - (_cw["depth"] - _toe_d)/2, _toe_h/2),
     ((_cw["x"][1] - _cw["x"][0])/2 - 0.02, (_cw["depth"] - _toe_d)/2, _toe_h/2), "kick", 0.5),
    ("casework_top", ((_cw["x"][0] + _cw["x"][1])/2, Y1 - _cw["depth"]/2 - 0.0125, _cw["top"] - 0.015),
     ((_cw["x"][1] - _cw["x"][0])/2 + 0.0125, _cw["depth"]/2 + 0.0125, 0.015), "counter", 0.4),
    # cubbies on the right wall: plinth and shelving
    ("cubby_plinth", (X1 - (_cb["depth"] - 0.02)/2, (_cb["y"][0] + _cb["y"][1])/2, _cb["plinth"]/2),
     ((_cb["depth"] - 0.02)/2, (_cb["y"][1] - _cb["y"][0])/2 - 0.01, _cb["plinth"]/2), "kick", 0.5),
    ("cubby_body", (X1 - _cb["depth"]/2, (_cb["y"][0] + _cb["y"][1])/2, _cb["plinth"] + (_cb["height"] - _cb["plinth"])/2),
     (_cb["depth"]/2, (_cb["y"][1] - _cb["y"][0])/2, (_cb["height"] - _cb["plinth"])/2), "cubby", 0.4),
    # rubber cove base (10 mm proud of the wall) where a broom or a bead can reach it
    ("cove_right_a", (X1 - _cove_t/2, (Y0 + _cb["y"][0])/2, _cove_h/2), (_cove_t/2, (_cb["y"][0] - Y0)/2, _cove_h/2), "cove", 0.6),
    ("cove_right_b", (X1 - _cove_t/2, (_cb["y"][1] + Y1)/2, _cove_h/2), (_cove_t/2, (Y1 - _cb["y"][1])/2, _cove_h/2), "cove", 0.6),
    ("cove_back_r", ((_cw["x"][1] + X1)/2, Y1 - _cove_t/2, _cove_h/2), ((X1 - _cw["x"][1])/2, _cove_t/2, _cove_h/2), "cove", 0.6),
    ("cove_back_l", ((X0 + _cw["x"][0])/2, Y1 - _cove_t/2, _cove_h/2), ((_cw["x"][0] - X0)/2, _cove_t/2, _cove_h/2), "cove", 0.6),
    ("cove_left", (X0 + _cove_t/2, (Y0 + Y1)/2, _cove_h/2), (_cove_t/2, (Y1 - Y0)/2, _cove_h/2), "cove", 0.6),
    ("cove_front_l", ((X0 + CP.DOOR_X - CP.DOOR_LEAF/2 - 0.05)/2, Y0 + _cove_t/2, _cove_h/2),
     ((CP.DOOR_X - CP.DOOR_LEAF/2 - 0.05 - X0)/2, _cove_t/2, _cove_h/2), "cove", 0.6),
    ("cove_front_r", ((CP.DOOR_X + CP.DOOR_LEAF/2 + 0.05 + X1)/2, Y0 + _cove_t/2, _cove_h/2),
     ((X1 - CP.DOOR_X - CP.DOOR_LEAF/2 - 0.05)/2, _cove_t/2, _cove_h/2), "cove", 0.6),
    ("threshold", (CP.DOOR_X, Y0 + 0.045, 0.003), (CP.DOOR_LEAF/2, 0.045, 0.003), "aluminum", 0.4),
    # entrance mat: carpet 6 mm thick (its bevelled border is drawn; the collider is the slab)
    ("mat", (*P.MAT["centre"], P.MAT["pile"]/2), (P.MAT["size"][0]/2, P.MAT["size"][1]/2, P.MAT["pile"]/2), "mat", 0.7),
    # tool rail: a steel strip on the wall; the prongs are in PRONGS
    ("rail_low", (X1 - 0.006, (P.RAIL["y"][0] + P.RAIL["y"][1])/2, P.RAIL["z"]), (0.006, (P.RAIL["y"][1] - P.RAIL["y"][0])/2, 0.035),
     "rail_steel", 0.4),
    ("rail_high", (X1 - 0.006, 2.20, 1.80), (0.006, 0.18, 0.035), "rail_steel", 0.4),
]
# sand and water table (installed here: four legs, a tub)
_sx, _sy = P.SAND_TABLE["centre"]
_shx, _shy = P.SAND_TABLE["size"][0]/2, P.SAND_TABLE["size"][1]/2
_sh = P.SAND_TABLE["height"]
STATIC += [(f"sand_leg_{k}", (_sx + a*(_shx - 0.03), _sy + b*(_shy - 0.03), (_sh - 0.23)/2), (0.02, 0.02, (_sh - 0.23)/2), "sand_frame", 0.4)
           for k, (a, b) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))]
STATIC += [("sand_tub", (_sx, _sy, _sh - 0.115), (_shx, _shy, 0.115), "sand_tub", 0.4)]

# where each broom hangs: (tool, y on the right wall, prong height); the cap of the handle rests on two prongs
HANG = dict(angle_broom=(1.55, P.RAIL["z"]), lobby_broom=(1.90, P.RAIL["z"]), push_broom=(2.20, 1.80))
SQUARE = P.SQUARE
STATIC_CAPSULES = []          # (name, end a, end b, radius, material, friction)


def _prongs():
    out = []
    for tool, (y, z) in HANG.items():
        d = dict(angle_broom=P.ANGLE_BROOM, lobby_broom=P.LOBBY_BROOM, push_broom=P.PUSH_BROOM)[tool]["handle_d"]
        sep = d + 2*PRONG_R + 0.004
        for k, s in enumerate((-1, 1)):
            out.append((f"prong_{tool}_{k}", (X1 - PRONG_LEN, y + s*sep/2, z), (X1, y + s*sep/2, z), PRONG_R, "rail_steel", 0.5))
            # an upturned tip keeps a swinging handle from walking off the prongs
            out.append((f"prong_tip_{tool}_{k}", (X1 - PRONG_LEN, y + s*sep/2, z), (X1 - PRONG_LEN - 0.004, y + s*sep/2, z + 0.014),
                        PRONG_R, "rail_steel", 0.5))
    return out


STATIC_CAPSULES += _prongs()


def cap_height(tool):
    """Height of the centre of the cap's LOWER END (a sphere of the cap's radius) when it sits on its two prongs."""
    spec = dict(angle_broom=P.ANGLE_BROOM, lobby_broom=P.LOBBY_BROOM, push_broom=P.PUSH_BROOM)[tool]
    sep = spec["handle_d"] + 2*PRONG_R + 0.004
    return HANG[tool][1] + math.sqrt((T.cap_radius(spec) + PRONG_R)**2 - (sep/2)**2)


def hang_pose(tool):
    """World pose (pos, quat xyzw) of a broom's body frame hanging from its prongs: handle vertical, cap on the
    prongs, the block's long side along the wall."""
    spec = T.BUILDERS[tool]()
    cap_part = next(p for p in spec["parts"] if p.name == "cap")
    cap_local = np.asarray(cap_part.pos, float)
    # at rest the centre of mass hangs straight below the cap (the push broom's head pulls its handle 3 deg off
    # plumb; started with the handle plumb it swung on its prongs for as long as it was watched)
    c = np.asarray(T.summary(spec)["com"], float) - cap_local
    theta = math.atan2(-c[1], -c[2])                         # about body x: brings c onto -z
    R = Rotation.from_euler("z", math.pi/2)*Rotation.from_euler("x", theta)
    y, _ = HANG[tool]
    # The cap is a capsule along the handle: what rests on the prongs is its lower end, half a capsule length below
    # its centre.
    cap = np.array([RAIL_X, y, cap_height(tool) + float(cap_part.size[1]) + 0.0008])
    return tuple(cap - R.apply(cap_local)), tuple(R.as_quat())


# ---------------------------------------------------------------------------------------------- furniture
def furniture():
    """name -> dict(kind, pose (x, y, yaw)): six tables with four tucked chairs each, the teacher's desk and chair."""
    out = {}
    hy = CP.TABLE_TOP[1]/2
    for name, (x, y) in P.TABLES.items():
        out[f"table_{name}"] = dict(kind="table", pose=(x, y, 0.))
        for k, (dx, side) in enumerate(((-0.38, -1), (0.38, -1), (-0.38, 1), (0.38, 1)), 1):
            out[f"chair_{name}_{k}"] = dict(kind="chair", pose=(x + dx, y + side*(hy + 0.10), math.pi if side < 0 else 0.),
                                           table=f"table_{name}")
    tx, ty = P.TEACHER_DESK
    out["teacher_desk"] = dict(kind="teacher_desk", pose=(tx, ty, 0.))
    out["teacher_chair"] = dict(kind="chair", pose=(tx - 0.18, ty - 0.50, math.pi))
    return out


def supplies():
    """Parts of the things that stay on the craft table in T6 (kind -> parts, body frame on the table top)."""
    return dict(
        # the cup and the bottle collide as square prisms (drawn round): MuJoCo gives a cylinder standing on a box a
        # single contact point, and the bottle fell over on the table during the first reset settle
        cup=[Part("cup", "box", (0.035, 0.035, 0.050), (0., 0., 0.050), mass=0.22, material="cup", role="supply", mu=0.4)],
        ream=[Part("ream", "box", (0.108, 0.140, 0.026), (0., 0., 0.026), mass=2.3, material="ream", role="supply", mu=0.4)],
        scissors=[Part("blade", "box", (0.012, 0.100, 0.004), (0., 0., 0.004), mass=0.07, material="scissors", role="supply", mu=0.35)],
        glue=[Part("bottle", "box", (0.023, 0.023, 0.065), (0., 0., 0.065), mass=0.16, material="glue", role="supply", mu=0.4)])


# ---------------------------------------------------------------------------------------------- tasks
TASKS = {
    "gather-open": dict(tier="entry-intermediate", sim_s=420, instruction=(
        "Loose debris is scattered over the open floor in the right-hand lane: rice, grit, cereal rings, crumpled paper "
        "and paper scraps. Take a broom from the rail and sweep it all into the square marked with blue tape near the "
        "door. A crumpled paper may be picked up by hand instead.")),
    "corner-and-edge": dict(tier="intermediate", sim_s=420, instruction=(
        "Grit, rice and cereal lie along the walls and in the corner of the alcove at the back right of the room, "
        "between the cubbies and the cabinets. Sweep it out of the corner and from along the walls into the taped "
        "square in front of the alcove.")),
    "under-table": dict(tier="intermediate-hard", sim_s=600, instruction=(
        "Cereal, rice, grit and paper scraps lie under and around the table nearest the door (table B1). Sweep up under "
        "and around the table, gather it all in the taped square, and put the four chairs back where they were.")),
    "pan-and-barrel": dict(tier="hard", sim_s=480, instruction=(
        "A pile of rice, grit and cereal has been swept into the taped square. Pick it up with a dustpan and empty the "
        "pan into the grey barrel, then hang the tools back on the rail and stand the dustpan under it.")),
    "broken-mug": dict(tier="hard", sim_s=600, instruction=(
        "A mug broke on the floor in front of the teacher's desk. Collect every piece and put the pieces in the blue "
        "pail (not in the bagged barrel): sweep them up with a broom and a dustpan; a large piece may also be picked "
        "up by hand. Leave the desk and the chair where they were.")),
    "table-scraps": dict(tier="intermediate", sim_s=360, instruction=(
        "The craft table is covered in paper scraps, eraser crumbs and beads. The counter brush and the hand dustpan lie "
        "at the end of the table by the wastebasket, the dustpan's mouth toward the scraps. Brush the scraps into the "
        "dustpan with the counter brush (the dustpan may be moved or held where it is needed) and empty it into the "
        "wastebasket beside the table. Leave the pencil cup, the paper, the scissors and the glue where they are.")),
    "heavy-spill": dict(tier="hard", sim_s=720, instruction=(
        "The rice from the sensory table has spilled over the floor in front of it, some of it under the table. "
        "Sweep it up and put it in the grey barrel.")),
    "beads": dict(tier="hard", sim_s=600, instruction=(
        "Sixty pony beads spilled from table B2 and rolled across the floor, some under the chairs. Collect the beads "
        "into the blue pail and put the chairs back.")),
    "closing-sweep": dict(tier="hard (extended)", sim_s=1800, instruction=(
        "Sweep the classroom floor at the end of the day. Most of the mess is in three places: the open floor along the "
        "wall with the tool rail, between the barrel and the cubbies; along the walls and in the corner of the alcove at "
        "the back, between the cubbies and the cabinets; and under and around table B1, the table nearest the door. Fine "
        "dust lies over that open floor and has been tracked in by the door. Put what you sweep up in the grey barrel, put "
        "back any chairs you move, and put the tools away where they belong.")),
}
VARIANTS = ("default", "push-broom-only", "hand-pan", "one-arm", "felt-glides", "steel-glides", "dusty")
PRESETS = {"room112": tuple(TASKS)}
ZONES = {            # where the debris of a task starts (x0, y0, x1, y1), and the target square's centre
    "gather-open": (P.OPEN_ZONE, P.SQUARE["centre"]),
    "corner-and-edge": ((_cw["x"][1], _cb["y"][1], X1, Y1), P.SQUARE_ALCOVE["centre"]),
    "under-table": ((P.TABLES["B1"][0] - 1.10, P.TABLES["B1"][1] - 0.95, P.TABLES["B1"][0] + 1.10, P.TABLES["B1"][1] + 0.95),
                    P.SQUARE["centre"]),
    "pan-and-barrel": ((P.SQUARE["centre"][0] - 0.40, P.SQUARE["centre"][1] - 0.40, P.SQUARE["centre"][0] + 0.40,
                        P.SQUARE["centre"][1] + 0.40), P.SQUARE["centre"]),
    "broken-mug": ((P.TEACHER_DESK[0] - 0.9, P.TEACHER_DESK[1] - 0.75, P.TEACHER_DESK[0] + 1.3, P.TEACHER_DESK[1] + 0.95),
                   P.SQUARE["centre"]),
    "table-scraps": ((P.TABLES["A3"][0] - 0.762, P.TABLES["A3"][1] - 0.381, P.TABLES["A3"][0] + 0.762, P.TABLES["A3"][1] + 0.381),
                     P.SQUARE["centre"]),
    "heavy-spill": ((-3.00, 7.55, -2.00, 8.95), P.SQUARE["centre"]),
    "beads": ((P.TABLES["B2"][0] - 1.6, P.TABLES["B2"][1] - 1.4, P.TABLES["B2"][0] + 1.6, P.TABLES["B2"][1] + 1.4),
              P.SQUARE["centre"]),
    "closing-sweep": ((X0, Y0, X1, Y1), P.SQUARE["centre"]),
}


def robot_start(task, variant="default"):
    return P.DOCK_START if task == "table-scraps" else P.ROBOT_START


def _yaw_quat(yaw):
    return (0., 0., math.sin(yaw/2), math.cos(yaw/2))


def _keepout(items, tools, robot):
    """Circles (x, y, r) a piece may not start on: glides and legs, tools and containers on the floor, the robot."""
    out = []
    for name, it in items.items():
        if it["kind"] in ("chair", "table", "teacher_desk"):
            for gx, gy in CL.glide_points(it["kind"], it["pose"]):
                out.append((gx, gy, 0.030))
        if it["kind"] == "teacher_desk":                     # pedestal reaches the floor
            x, y, yaw = it["pose"]
            out.append((x + 0.40*math.cos(yaw), y + 0.40*math.sin(yaw), 0.42))
    for name, t in tools.items():
        if t["pos"][2] < 0.2:
            out.append((t["pos"][0], t["pos"][1], 0.33 if t["kind"] in ("barrel",) else 0.24))
    x, y, yaw = robot
    for a in np.linspace(-0.30, 0.62, 5):
        out.append((x + a*math.cos(yaw), y + a*math.sin(yaw), 0.42))
    return out


def setup(task, seed, *, variant="default", glides="nylon", amount=1.0):
    """Start state of one (task, seed, variant). Deterministic in its arguments.
    Returns dict(items, glides, books, robot, tools, debris, task, zone, square, statics, variant, targets)."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose {sorted(TASKS)}")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose {VARIANTS}")
    rng = np.random.default_rng(seed*7919 + sum(map(ord, task)) + 13*VARIANTS.index(variant))
    items = furniture()
    robot = robot_start(task, variant)
    if variant in ("felt-glides", "steel-glides"):
        glides = variant.split("-")[0]
    # chairs: every seed leaves the chairs of the task's table a little pulled out and turned
    loose = {"under-table": "B1", "beads": "B2", "closing-sweep": "B1"}.get(task)
    if loose:
        for k in range(1, 5):
            n = f"chair_{loose}_{k}"
            x, y, yaw = items[n]["pose"]
            out = rng.uniform(0.0, 0.30)
            side = -1 if k <= 2 else 1
            # a chair that is still under the table cannot stand turned: its back legs would be in the table's (a
            # seed left one 52 mm from its place at the reset). It turns as far as it is out.
            turn = rng.uniform(-15, 15)*min(1.0, out/0.20)
            items[n] = dict(items[n], pose=(x + rng.uniform(-0.03, 0.03), y + side*out, yaw + math.radians(turn)))
    if task == "table-scraps":
        # the robot docks at the near (-y) long side of craft table A3: its two chairs stand against the window wall
        items["chair_A3_1"] = dict(items["chair_A3_1"], pose=(X0 + 0.42, 6.05, math.pi/2))
        items["chair_A3_2"] = dict(items["chair_A3_2"], pose=(X0 + 0.42, 6.65, math.pi/2))
        # and the two chairs of table A2 that have their backs to the robot are pushed in under their table: where
        # they stand in the other tasks, the docked base (0.8 m long) would overlap them by 0.09 m and the reset would
        # throw them 0.8 m
        for n in ("chair_A2_3", "chair_A2_4"):
            x, y, yaw = items[n]["pose"]
            items[n] = dict(items[n], pose=(x, y - 0.12, yaw))
    # ---- tools and containers
    tools = {}
    rail = ["angle_broom", "lobby_broom", "push_broom"]
    if variant == "push-broom-only":
        rail = ["push_broom"]
    for n in rail:
        pos, q = hang_pose(n)
        tools[n] = dict(kind=n, pos=pos, quat=q, hung=True)
    if variant != "hand-pan":
        # the handle starts on its back stop (8 deg). Started at 6 deg, it fell onto the stop in the first
        # tenth of a second of every reset (0.33 m/s at its pivot end) and rocked the pan 1.3 deg twice
        tools["lobby_pan"] = dict(kind="lobby_pan", pos=(X1 - 0.30, 2.62, 0.0006), quat=_yaw_quat(math.pi/2), hinge=math.radians(8.0))
    # the barrel's wall stays 50 mm clear of the cove base (x = 3.69): drawn between 3.10 and 3.39. Drawn up to
    # 3.50, its bottom started inside the room's wall in one seed in six
    bx, by = 3.245 + rng.uniform(-0.145, 0.145), 3.30 + rng.uniform(-0.15, 0.25)
    tools["barrel"] = dict(kind="barrel", pos=(bx, by, 0.0006), quat=_yaw_quat(rng.uniform(0, 0.4)))
    tools["pail"] = dict(kind="pail", pos=(3.42, 1.02, 0.0006) if task != "broken-mug" else (-1.15, 1.75, 0.0006), quat=_yaw_quat(0.))
    ax, ay = P.TABLES["A3"]
    if task == "table-scraps":
        tools["wastebasket"] = dict(kind="wastebasket", pos=(ax + 0.95, ay - 0.30, 0.0006), quat=_yaw_quat(0.))
        zt = TABLE_H + 0.0006
        tools["counter_brush"] = dict(kind="counter_brush", pos=(ax + 0.40, ay - 0.24, zt + 0.0155),
                                      quat=tuple((Rotation.from_euler("z", 0.2)*Rotation.from_euler("x", math.pi/2)).as_quat()))
        if variant != "one-arm":
            # the hand dustpan lies with its mouth toward the length of the table, where the scraps are, and its
            # handle toward the table's end by the wastebasket
            tools["hand_pan"] = dict(kind="hand_pan", pos=(ax + 0.38, ay + 0.04, zt), quat=_yaw_quat(math.pi/2))
    else:
        tools["wastebasket"] = dict(kind="wastebasket", pos=(-1.45, 0.60, 0.0006), quat=_yaw_quat(0.))
        zc = P.CASEWORK["top"] + 0.0006
        tools["counter_brush"] = dict(kind="counter_brush", pos=(1.55, Y1 - 0.30, zc + 0.016),
                                      quat=tuple((Rotation.from_euler("z", 0.1)*Rotation.from_euler("x", math.pi/2)).as_quat()))
        tools["hand_pan"] = dict(kind="hand_pan", pos=(1.95, Y1 - 0.32, zc), quat=_yaw_quat(math.pi))
    # ---- supplies on the craft table (T6 only: bodies; elsewhere the table is bare)
    extra = {}
    if task == "table-scraps":
        spots = dict(cup=(ax - 0.58, ay + 0.22), ream=(ax - 0.28, ay + 0.16), scissors=(ax + 0.05, ay + 0.26), glue=(ax - 0.52, ay - 0.10))
        for k, (x, y) in spots.items():
            extra[k] = dict(kind=k, pose=(x + rng.uniform(-0.04, 0.04), y + rng.uniform(-0.03, 0.03), rng.uniform(-0.4, 0.4)), z=TABLE_H)
    # ---- debris
    keep = _keepout(items, tools, robot)
    # installed parts that stand on the floor (the sand table's legs, the cubbies' plinth, the mat): no piece starts
    # inside their footprint. Without this, twelve rice grains of heavy-spill started under a leg of the sand table
    # and stayed pressed into the floor, awake for ever
    standing = [(c[0] - h[0], c[1] - h[1], c[0] + h[0], c[1] + h[1]) for _, c, h, *_ in STATIC if c[2] - h[2] < 0.05]

    def avoid(xy, r):
        if not (X0 + 0.02 + r < xy[0] < X1 - 0.02 - r and Y0 + 0.02 + r < xy[1] < Y1 - 0.02 - r):
            return True
        m = r + 0.002
        if any(x0 - m < xy[0] < x1 + m and y0 - m < xy[1] < y1 + m for x0, y0, x1, y1 in standing):
            return True
        return any((xy[0] - kx)**2 + (xy[1] - ky)**2 < (kr + r)**2 for kx, ky, kr in keep)

    def amt(g):
        return g*amount*rng.uniform(0.8, 1.2)

    pieces = []
    floor = {}                  # what lies on the floor so far: shared by every scatter call on it (debris.scatter, `taken`)
    zone, square = ZONES[task]
    info = dict(zone=zone, square=dict(centre=square, side=P.SQUARE["side"]))
    if task in ("gather-open", "closing-sweep"):
        f = 0.6 if task == "closing-sweep" else 1.0
        z = P.OPEN_ZONE
        pieces += D.scatter(rng, D.rice(rng, amt(20*f)) + D.grit(rng, amt(15*f)) + D.cereal(rng, amt(10*f)) + D.wads(rng, 4 if f == 1 else 2)
                            + D.scraps(rng, 12 if f == 1 else 7), z, avoid=avoid, taken=floor)
    if task in ("corner-and-edge", "closing-sweep"):
        f = 0.6 if task == "closing-sweep" else 1.0
        ax0, ay0 = _cw["x"][1], _cb["y"][1]

        def edge(rng_):
            # within 60 mm of the back wall, the right wall or the cabinet's end, or in the corner itself
            w = rng_.integers(0, 4)
            d = _cove_t + rng_.uniform(0.004, 0.060)
            if w == 0:
                return rng_.uniform(ax0 + 0.02, X1 - 0.02), Y1 - d
            if w == 1:
                return X1 - d, rng_.uniform(ay0 + 0.02, Y1 - 0.02)
            if w == 2:
                return ax0 + (d - _cove_t), rng_.uniform(Y1 - _cw["depth"] + 0.02, Y1 - 0.02)
            return X1 - _cove_t - rng_.uniform(0.004, 0.16), Y1 - _cove_t - rng_.uniform(0.004, 0.16)
        pieces += D.scatter(rng, D.grit(rng, amt(12*f)) + D.rice(rng, amt(8*f)) + D.cereal(rng, amt(6*f)), edge, avoid=avoid, tries=200, taken=floor)
    if task in ("under-table", "closing-sweep"):
        f = 0.6 if task == "closing-sweep" else 1.0
        tx, ty = P.TABLES["B1"]
        pieces += D.scatter(rng, D.cereal(rng, amt(15*f)) + D.rice(rng, amt(15*f)) + D.grit(rng, amt(5*f)) + D.scraps(rng, 6 if f == 1 else 4),
                            (tx - 0.95, ty - 0.80, tx + 0.95, ty + 0.80), avoid=avoid, taken=floor)
    if task == "closing-sweep":
        pieces += D.scatter(rng, D.pencils(rng, 1) + D.beads(rng, 2), (0.9, 3.4, 3.3, 6.5), avoid=avoid, taken=floor)
    if task == "pan-and-barrel":
        cx, cy = square
        # a swept-up pile: pieces start in layers above a disc and settle when the world is reset; layers are 4 mm
        # apart so nothing starts overlapping
        ps = D.rice(rng, amt(30)) + D.grit(rng, amt(25)) + D.cereal(rng, amt(15))
        order = rng.permutation(len(ps))
        layer, left = 0, [ps[k] for k in order]
        while left and layer < 40:
            placed = D.scatter(rng, left, (cx, cy, 0.105), z0=0.004*layer, tries=6, taken=floor if layer == 0 else None)
            ids = {id(p) for p in placed}
            pieces += placed
            left = [p for p in left if id(p) not in ids]
            layer += 1
    if task == "broken-mug":
        tx, ty = P.TEACHER_DESK
        hit = (tx + rng.uniform(0.1, 0.5), ty + 0.38 + rng.uniform(0.25, 0.55))
        m = D.mug(rng)
        big = [p for p in m if p.tag != "chip"]
        chips = [p for p in m if p.tag == "chip"]

        def spray(rng_):
            r, a = abs(rng_.normal(0, 0.28)), rng_.uniform(0, 2*math.pi)
            return hit[0] + r*math.cos(a), hit[1] + r*math.sin(a) - (0.5 if rng_.random() < 0.33 else 0.0)*rng_.random()
        # No piece is laid under the desk's top, behind a modesty panel 0.30 m above the floor. A real mug's pieces do
        # go under a desk; this task does not ask for that.
        edge = ty + CP.TEACHER_TOP[1]/2 + 0.01

        def in_front(xy, r):
            return avoid(xy, r) or (xy[1] - r < edge and abs(xy[0] - tx) < CP.TEACHER_TOP[0]/2 + r)
        pieces += (D.scatter(rng, big, spray, avoid=in_front, tries=300, taken=floor)
                   + D.scatter(rng, chips, spray, avoid=in_front, tries=300, taken=floor))
        info["hit"] = hit
    if task == "table-scraps":
        hx, hy = CP.TABLE_TOP[0]/2 - 0.04, CP.TABLE_TOP[1]/2 - 0.04
        sup = [(e["pose"][0], e["pose"][1], dict(cup=0.05, ream=0.19, scissors=0.11, glue=0.04)[k]) for k, e in extra.items()]
        sup += [(tools[n]["pos"][0], tools[n]["pos"][1], 0.19) for n in ("counter_brush", "hand_pan") if n in tools]

        def avoid_t(xy, r):
            return any((xy[0] - kx)**2 + (xy[1] - ky)**2 < (kr + r)**2 for kx, ky, kr in sup)
        ps = D.scraps(rng, 25) + D.crumbs(rng, amt(8)) + D.beads(rng, 30) + D.pencils(rng, 1)
        # laid 0.6 mm above the table's nominal top, as the tools on it are: the table is built 0.5 mm above the floor
        # and comes down in the reset's settle. Laid at the nominal height, every piece started
        # 0.48 mm inside the table top, was pushed out at the solver's cap of 1 m/s and hopped 53 mm at every reset
        # (measured tick by tick; the settle hid it)
        pieces += D.scatter(rng, ps, (ax - hx, ay - hy, ax + hx, ay + hy), z0=TABLE_H + 0.0006, avoid=avoid_t, tries=200)
    if task == "heavy-spill":
        sx, sy = P.SAND_TABLE["centre"]

        def spill(rng_):
            r, a = 0.42*math.sqrt(rng_.random()), rng_.uniform(0, 2*math.pi)
            return sx + 0.62 + r*math.cos(a)*1.0, sy + 0.10 + r*math.sin(a)*1.6
        ps = D.rice(rng, amt(400))
        layer, left = 0, ps
        while left and layer < 12:
            placed = D.scatter(rng, left, spill, z0=0.004*layer, avoid=avoid, tries=4, taken=floor if layer == 0 else None)
            ids = {id(p) for p in placed}
            pieces += placed
            left = [p for p in left if id(p) not in ids]
            layer += 1
    if task == "beads":
        tx, ty = P.TABLES["B2"]

        def roll(rng_):
            r, a = 0.25 + abs(rng_.normal(0, 0.55)), rng_.uniform(0, 2*math.pi)
            return tx + r*math.cos(a), ty + r*math.sin(a)
        pieces += D.scatter(rng, D.beads(rng, 60), roll, avoid=avoid, tries=300, taken=floor)
    for k, p in enumerate(pieces):
        p.tag = p.tag or p.cls
    # ---- fine dust (a film, sweeping/dust.py): around the swept pile; in the lane and by the door at closing time;
    # over a task's floor in the `dusty` variant. Never under an installed part or on the mat.
    dust = []
    if task == "pan-and-barrel":
        dust.append(dict(kind="disc", centre=tuple(square), radius=0.40, grams=P.DUST_LOADS["pile_g"]*amount, avoid=standing))
    if task == "closing-sweep":
        dust.append(dict(kind="rect", box=tuple(P.OPEN_ZONE), g_per_m2=P.DUST_LOADS["open_g_m2"]*amount, avoid=standing))
        dust.append(dict(kind="rect", box=(CP.DOOR_X - 0.75, Y0 + 0.12, CP.DOOR_X + 0.75, Y0 + 1.60), g_per_m2=P.DUST_LOADS["door_g_m2"]*amount,
                         avoid=standing))
    if variant == "dusty":
        dust.append(dict(kind="rect", box=tuple(zone), g_per_m2=P.DUST_LOADS["dusty_g_m2"]*amount, avoid=standing))
    info["dust"] = dust
    return dict(items=items, glides=glides, books={}, robot=robot, tools=tools, debris=pieces, task=task, seed=seed,
                variant=variant, supplies=extra, **info)
