"""Room 114: the default layout L0 (paired rows), task start states and target layouts.

World frame: x across the room (+x toward the door side), y from the front (board) wall toward the back, z up,
floor at z = 0. Students face -y: a desk at yaw 0 has its open (student) side toward +y, its chair behind it (+y).
Item names: desk_<row><col> and chair_<row><col> (rows 1-4 front to back, columns 1-6 from the window side),
table and tchair_1..4 (tutoring table), teacher_desk and teacher_chair, tote, basket.
"""

import math
import numpy as np
from . import params as P
from .furniture import TUCK_DY, USE_DY

# three pairs at -2.157, 0, +2.157 with 10 mm between paired desks: the two main and two side aisles are equal, 0.928 m,
# 13 mm over the ADA 403.5.1 accessible-route minimum (a 7.40 m room fits three pairs and four such aisles exactly)
COLS_X = (-2.467, -1.847, -0.31, 0.31, 1.847, 2.467)
ROWS_Y = (2.40, 3.80, 5.20, 6.60)                            # pitch 1.40 m
TABLE = (-2.00, 8.90, 0.)                                     # tutoring table, long side along x, 0.32 m off the back wall
TEACHER = (-2.35, 1.20, 0.)                                   # teacher's desk: modesty panel (+y) toward the class
BASKET_HOME = (-1.50, 0.85)                                   # wastebasket's place beside the teacher's desk
CUBBY_ZONE = ((0.40, 2.80), (8.75, 9.25))                     # floor in front of the cubbies (x range, y range)
DOOR_ZONE = ((P.DOOR_X - P.DOOR_LEAF/2 - 0.05, P.DOOR_X + P.DOOR_LEAF/2 + 0.05), (0.0, P.DOOR_LEAF + 0.05))

# static room parts (name, centre, half extents, material): collide and are drawn
WALL_T = 0.15
STATIC = [
    ("wall_front", (0., -WALL_T/2, P.ROOM_Z/2), (P.ROOM_X/2 + WALL_T, WALL_T/2, P.ROOM_Z/2), "wall"),
    ("wall_back", (0., P.ROOM_Y + WALL_T/2, P.ROOM_Z/2), (P.ROOM_X/2 + WALL_T, WALL_T/2, P.ROOM_Z/2), "wall"),
    ("wall_left_low", (-P.ROOM_X/2 - WALL_T/2, P.ROOM_Y/2, 0.45), (WALL_T/2, P.ROOM_Y/2, 0.45), "wall"),
    # above the 0.90 m sill: piers, wall head, frames and the closed glazing of the three windows as one collider (the
    # render draws the piers, frames and open panes)
    ("wall_left_upper", (-P.ROOM_X/2 - WALL_T/2, P.ROOM_Y/2, (0.90 + P.ROOM_Z)/2), (WALL_T/2, P.ROOM_Y/2, (P.ROOM_Z - 0.90)/2),
     "wall"),
    *[(f"wall_sill_{k}", (-P.ROOM_X/2 + 0.04, c, 0.90), (0.11, 1.18, 0.02), "sill") for k, c in enumerate((1.70, 4.80, 7.90))],
    ("wall_right", (P.ROOM_X/2 + WALL_T/2, P.ROOM_Y/2, P.ROOM_Z/2), (WALL_T/2, P.ROOM_Y/2, P.ROOM_Z/2), "wall"),
    ("board", (0., 0.012, 1.524), (1.22, 0.012, 0.61), "board"),
    ("board_tray", (0., 0.045, 0.905), (1.22, 0.04, 0.008), "aluminum"),
    ("bookcase", (2.95, 9.42, 0.90), (0.45, 0.17, 0.90), "bookcase"),
    ("cubbies", (1.60, 9.40, 0.46), (0.95, 0.19, 0.46), "cubby"),
    ("door_leaf", (P.DOOR_X, 0.022, 1.04), (P.DOOR_LEAF/2, 0.021, 1.04), "door"),
]


def yaw_quat(yaw):
    return (0., 0., math.sin(yaw/2), math.cos(yaw/2))


def L0():
    """name -> dict(kind, pose (x, y, yaw)) of every free body in the default layout (chairs tucked)."""
    out = {}
    for r, y in enumerate(ROWS_Y, 1):
        for c, x in enumerate(COLS_X, 1):
            out[f"desk_{r}{c}"] = dict(kind="desk", pose=(x, y, 0.))
            out[f"chair_{r}{c}"] = dict(kind="chair", pose=(x, y + TUCK_DY, 0.))
    tx, ty, _ = TABLE
    out["table"] = dict(kind="table", pose=TABLE)
    out.update(tutoring_chairs_at_table())
    out["teacher_desk"] = dict(kind="teacher_desk", pose=TEACHER)
    # in the desk's knee space (between the pedestal at local +x and the legs), facing the class
    out["teacher_chair"] = dict(kind="chair", pose=(TEACHER[0] - 0.18, TEACHER[1] - 0.50, math.pi))
    out["tote"] = dict(kind="tote", pose=(1.10, 9.00, 0.))
    out["basket"] = dict(kind="basket", pose=(*BASKET_HOME, 0.))
    return out


def tutoring_chairs_at_table(pulled=0.0):
    """Four chairs at the tutoring table: two on the room side, one at each end, each facing the table; `pulled` is
    how far the seat front is from the table edge beyond tucking (0: tucked to 0.10 m under)."""
    tx, ty, _ = TABLE
    hx, hy = P.TABLE_TOP[0]/2, P.TABLE_TOP[1]/2
    d = 0.10 + pulled                    # chair centre from the table edge (seat front 0.115 m under at 0)
    return {"tchair_1": dict(kind="chair", pose=(tx - 0.33, ty - hy - d, math.pi)),
            "tchair_2": dict(kind="chair", pose=(tx + 0.33, ty - hy - d, math.pi)),
            "tchair_3": dict(kind="chair", pose=(tx - hx - d, ty, math.pi/2)),
            "tchair_4": dict(kind="chair", pose=(tx + hx + d, ty, -math.pi/2))}


def glide_points(kind, pose):
    """World xy of an item's glides at a pose (used for floor dots and the evaluator)."""
    from . import furniture as F
    parts = {"desk": F.desk, "chair": F.chair, "table": F.table, "teacher_desk": F.teacher_desk}[kind]()
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c*p.pos[0] - s*p.pos[1], y + s*p.pos[0] + c*p.pos[1]) for p in parts if p.role == "glide"]


# ---------------------------------------------------------------------------------------------- start-state validity
HALF = {"desk": (P.DESK_TOP[0]/2, P.DESK_TOP[1]/2), "chair": (0.238, 0.25), "table": (P.TABLE_TOP[0]/2, P.TABLE_TOP[1]/2),
        "teacher_desk": (P.TEACHER_TOP[0]/2, P.TEACHER_TOP[1]/2), "tote": (P.TOTE[0]/2, P.TOTE[1]/2),
        "basket": (P.BASKET[0], P.BASKET[0]), "robot": (0.56, 0.36)}
ROBOT_FOOT_DX = 0.16          # the robot's plan footprint is x -0.40..+0.72 m, y -0.36..+0.36 m in base_link: the TMR is
                              # 800 x 583 mm, but the Duo mount angles the arms 51 deg outward and the stowed arms'
                              # shoulders (link1-link3) stand out to about 0.35 m either side (E5: base_link stops 0.348 m
                              # from a wall with the left arm against it). The 0.72 m ahead
                              # keeps room for unfolding the arms: centre 0.16 m ahead of the origin


def _corners(kind, pose, pad=0.):
    hx, hy = HALF[kind]
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    if kind == "robot":
        x, y = x + c*ROBOT_FOOT_DX, y + s*ROBOT_FOOT_DX
    return np.array([(x + c*a - s*b, y + s*a + c*b) for a, b in ((-hx - pad, -hy - pad), (hx + pad, -hy - pad),
                                                                  (hx + pad, hy + pad), (-hx - pad, hy + pad))])


def _sat(A, B):
    for poly in (A, B):
        for k in range(4):
            e = poly[(k + 1) % 4] - poly[k]
            n = np.array([-e[1], e[0]])
            pa, pb = A @ n, B @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def conflicts(items, names=None):
    """Pairs of items whose start poses would interpenetrate: tops of desks/tables and chair footprints by rectangles
    (5 mm clearance), a chair against a desk by its glides (45 mm between glide centres) and its backrest (which rises
    through the height of a desk top) against the top's rectangle; everything inside the room."""
    out = []
    keys = sorted(items)
    check = set(names or keys)
    for i, a in enumerate(keys):
        A = items[a]
        if A.get("inverted_on"):
            continue
        ca = _corners(A["kind"], A["pose"])
        if a in check and (np.abs(ca[:, 0]).max() > P.ROOM_X/2 - 0.01 or ca[:, 1].min() < 0.01 or ca[:, 1].max() > P.ROOM_Y - 0.01):
            out.append((a, "wall"))
        for b in keys[i + 1:]:
            B = items[b]
            if B.get("inverted_on") or (a not in check and b not in check):
                continue
            kinds = {A["kind"], B["kind"]}
            if kinds == {"chair", "desk"} or kinds == {"chair", "table"} or kinds == {"chair", "teacher_desk"}:
                ch, dk = (A, B) if A["kind"] == "chair" else (B, A)
                g1 = np.array(glide_points("chair", ch["pose"]))
                g2 = np.array(glide_points(dk["kind"], dk["pose"]))
                close = np.linalg.norm(g1[:, None] - g2[None], axis=2).min() < 0.045
                x, y, yaw = ch["pose"]
                back = [(x - math.sin(yaw)*0.24 + math.cos(yaw)*u, y + math.cos(yaw)*0.24 + math.sin(yaw)*u) for u in (-0.2, 0., 0.2)]
                D = _corners(dk["kind"], dk["pose"])
                inside = any(_sat(np.array([pt, pt, pt, pt]), D) for pt in back)
                if dk["kind"] == "teacher_desk":           # the steel pedestal reaches the floor: the chair may not overlap it
                    dx, dy, dyaw = dk["pose"]
                    ped = (dx + math.cos(dyaw)*0.40, dy + math.sin(dyaw)*0.40, dyaw)
                    hx, hy = HALF["chair"]
                    PED = np.array([(ped[0] + math.cos(dyaw)*a - math.sin(dyaw)*b, ped[1] + math.sin(dyaw)*a + math.cos(dyaw)*b)
                                    for a, b in ((-0.2, -0.36), (0.2, -0.36), (0.2, 0.36), (-0.2, 0.36))])
                    inside = inside or _sat(_corners("chair", ch["pose"]), PED)
                if close or inside:
                    out.append((a, b))
            elif _sat(_corners(A["kind"], A["pose"], 0.0025), _corners(B["kind"], B["pose"], 0.0025)):
                out.append((a, b))
    return out


# ---------------------------------------------------------------------------------------------- tasks
TASKS = {
    "straighten-row": dict(tier="entry", sim_s=180, instruction=(
        "The desks in row 3 (the third row from the board) have drifted off their places. Put each of the six desks "
        "back on its floor dots (the coloured dots under its four feet), squared to the board, and push its chair in "
        "under the desk. Leave the rest of the room as it is.")),
    "push-in-chairs": dict(tier="entry", sim_s=120, instruction=(
        "Class is over and the six chairs of row 2 were left pulled out. Push each chair in under its own desk, "
        "straight, so the walkway behind the row is clear. Do not move the desks.")),
    "seat-tutoring-group": dict(tier="entry-intermediate", sim_s=180, instruction=(
        "A tutoring group of four is coming. Set the four loose chairs at the tutoring table at the back left of the room: "
        "two along the long side facing the room and one at each end, each facing the table and pulled out ready to sit "
        "(seat front 20-35 cm from the table edge).")),
    "make-pod": dict(tier="intermediate-hard", sim_s=600, instruction=(
        "Make a pod of four for group work from the middle pair of desks in rows 1 and 2: turn the two row-1 desks "
        "around and push the four desks together front edge to front edge on the pod's floor dots, with each chair "
        "pulled out behind its own desk, outside the pod.")),
    "exam-row": dict(tier="intermediate", sim_s=600, instruction=(
        "Set row 2 up for a test: spread its six desks onto the exam dots, 1.25 m apart across the room, facing the "
        "board, each chair pulled out behind its desk ready to sit.")),
    "chairs-up": dict(tier="hard", sim_s=720, instruction=(
        "The cleaners come tonight. Put the six chairs of row 4 up on their own desks, upside down: seat on the desk top, "
        "legs up, the backrest hanging over the student side of the desk. Leave the desks on their dots.")),
    "chairs-down": dict(tier="hard", sim_s=720, instruction=(
        "Morning: the six chairs of row 4 are upside down on their desks. Take each one down, set it upright on its glides "
        "and push it in under its own desk, without moving the desks off their dots.")),
    "clear-floor": dict(tier="intermediate", sim_s=300, instruction=(
        "Clear the floor before the fire drill: carry or slide the book tote from the aisle to the floor in front of the "
        "cubbies at the back, put the wastebasket back in its place beside the teacher's desk, and push the stray chair "
        "in the right-hand aisle in under its desk, so a 0.915 m wide route is open from the door to the back of the room.")),
    "move-teacher-desk": dict(tier="hard", sim_s=360, instruction=(
        "Make room for a projector cart at the front: turn the teacher's desk a quarter turn and slide it toward the "
        "window wall onto its floor dots, so its long side runs along the windows. Its chair goes behind it, on the "
        "room side.")),
    "group-work": dict(tier="hard (extended)", sim_s=1800, instruction=(
        "Group-work day: turn the two front rows (rows 1 and 2) into three pods of four, one for each pair of desks and "
        "the pair behind it. In each pod the pair nearer the board faces the pair behind it, and the four desks meet "
        "front edge to front edge on the pod's floor dots. Give each desk a chair on the outside of the pod, facing it "
        "and pulled out ready to sit. Leave a 0.915 m (36 in) wide route clear from the door to the back of the room, "
        "and leave the rest of the room as it is.")),
}
PRESETS = {"room114": tuple(TASKS)}


def robot_start(task):
    """(x, y, yaw) of the robot's base_link: parked in an aisle near the task area, arms stowed, spine down."""
    return {"straighten-row": (3.23, 4.45, math.pi/2), "push-in-chairs": (3.23, 4.45, math.pi/2),
            "seat-tutoring-group": (0.90, 8.30, math.pi), "make-pod": (1.075, 1.35, math.pi/2),
            "exam-row": (3.23, 1.35, math.pi/2), "chairs-up": (0.90, 7.90, math.pi), "chairs-down": (0.90, 7.90, math.pi),
            "clear-floor": (3.23, 1.35, math.pi/2), "move-teacher-desk": (-0.80, 1.47, math.pi),
            "group-work": (0.60, 1.35, math.pi)}[task]


def pod_targets(pair):
    """Pod of the given column pair (0, 1, 2) from rows 1-2: desks meet at y_p = 3.10."""
    yp = (ROWS_Y[0] + ROWS_Y[1])/2
    hy = P.DESK_TOP[1]/2
    c0, c1 = 2*pair + 1, 2*pair + 2
    t = {}
    for c in (c0, c1):
        x = COLS_X[c - 1]
        t[f"desk_1{c}"] = (x, yp - hy, math.pi)
        t[f"desk_2{c}"] = (x, yp + hy, 0.)
        t[f"chair_1{c}"] = (x, yp - hy - USE_DY, math.pi)
        t[f"chair_2{c}"] = (x, yp + hy + USE_DY, 0.)
    return t


EXAM_X = (-3.125, -1.875, -0.625, 0.625, 1.875, 3.125)


def setup(task, seed, *, glides="nylon", contents=0):
    """Start overrides, targets and the evaluator's criteria for one (task, seed). Deterministic in seed."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose {sorted(TASKS)}")
    rng = np.random.default_rng(seed*7919 + sum(map(ord, task)))
    items = L0()
    items["__robot__"] = dict(kind="robot", pose=robot_start(task))
    start, targets, inverted, check = {}, {}, set(), {}

    def place(name, sample):
        """Draw a start pose until it conflicts with nothing already placed (layout.conflicts), 500 tries."""
        for _ in range(500):
            p = sample()
            trial = dict(items)
            trial[name] = dict(items[name], pose=p)
            if not conflicts(trial, [name]):
                items[name] = trial[name]
                start[name] = p
                return p
        raise RuntimeError(f"no conflict-free start pose for {name} ({task}, seed {seed})")

    def jitter(name, lo, hi, yaw_lo_deg, yaw_hi_deg):
        x, y, yaw = items[name]["pose"]

        def sample():
            r, a = rng.uniform(lo, hi), rng.uniform(0, 2*math.pi)
            dyaw = math.radians(rng.uniform(yaw_lo_deg, yaw_hi_deg))*rng.choice((-1, 1))
            return (x + r*math.cos(a), y + r*math.sin(a), yaw + dyaw)
        return place(name, sample)

    if task == "straighten-row":
        for c in range(1, 7):
            d, ch = f"desk_3{c}", f"chair_3{c}"
            targets[d] = items[d]["pose"]
            targets[ch] = items[ch]["pose"]
            items[ch]["pose"] = (items[ch]["pose"][0], items[ch]["pose"][1] + 0.6, 0.)     # out of the way while desks are drawn
        for c in range(1, 7):
            d, ch = f"desk_3{c}", f"chair_3{c}"
            x, y, yaw = jitter(d, 0.05, 0.20, 2, 10)
            place(ch, lambda: (x - math.sin(yaw)*USE_DY + rng.uniform(-0.05, 0.05), y + math.cos(yaw)*USE_DY + rng.uniform(-0.03, 0.08),
                               yaw + math.radians(rng.uniform(-8, 8))))
            check[d] = "on_marks"
            check[ch] = ("tucked", d)
    elif task == "push-in-chairs":
        for c in range(1, 7):
            ch, d = f"chair_2{c}", f"desk_2{c}"
            x, y, _ = items[ch]["pose"]
            targets[ch] = items[ch]["pose"]
            place(ch, lambda: (x + rng.uniform(-0.12, 0.12), y + rng.uniform(0.40, 0.80), math.radians(rng.uniform(-30, 30))))
            check[ch] = ("tucked", d)
            check[d] = "unmoved"
    elif task == "seat-tutoring-group":
        tx, ty, _ = TABLE
        goal = tutoring_chairs_at_table(pulled=0.39)      # seat front 0.275 m from the edge
        for k in range(1, 5):
            items[f"tchair_{k}"]["pose"] = (-3.0 + 0.6*k, 0.4, 0.)          # parked out of the way while drawing
        for k in range(1, 5):
            n = f"tchair_{k}"
            place(n, lambda: (tx + rng.uniform(-0.8, 2.0), ty + rng.uniform(-1.75, -0.95), rng.uniform(-math.pi, math.pi)))
            targets[n] = goal[n]["pose"]
            check[n] = ("ready", "table")
        check["table"] = "unmoved"
    elif task == "make-pod":
        targets = pod_targets(1)
        for n in targets:
            check[n] = "on_marks" if n.startswith("desk") else ("in_use", "desk" + n[5:])
        check["pod"] = ["desk_13", "desk_14", "desk_23", "desk_24"]
        for n in targets:
            if n.startswith("chair"):
                x, y, yaw = items[n]["pose"]
                items[n]["pose"] = start[n] = (x, y + 0.30, yaw)        # chairs pulled out by the students (in use)
    elif task == "exam-row":
        for c in range(1, 7):
            d, ch = f"desk_2{c}", f"chair_2{c}"
            targets[d] = (EXAM_X[c - 1], ROWS_Y[1], 0.)
            targets[ch] = (EXAM_X[c - 1], ROWS_Y[1] + USE_DY, 0.)
            check[d] = "on_marks"
            check[ch] = ("in_use", d)
        check["exam"] = [f"chair_2{c}" for c in range(1, 7)]
    elif task in ("chairs-up", "chairs-down"):
        for c in range(1, 7):
            d, ch = f"desk_4{c}", f"chair_4{c}"
            check[d] = "on_marks"
            targets[d] = items[d]["pose"]
            if task == "chairs-up":
                check[ch] = ("up_on", d)
            else:
                inverted.add(ch)
                targets[ch] = items[ch]["pose"]
                check[ch] = ("tucked", d)
    elif task == "clear-floor":
        place("tote", lambda: (-1.075 + rng.uniform(-0.1, 0.1), 4.50 + rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)))
        place("basket", lambda: (1.075 + rng.uniform(-0.1, 0.1), 3.10 + rng.uniform(-0.3, 0.3), 0.))
        x, y, _ = items["chair_36"]["pose"]
        targets["chair_36"] = items["chair_36"]["pose"]
        place("chair_36", lambda: (3.235 + rng.uniform(-0.04, 0.04), y + rng.uniform(0.2, 0.5), math.pi/2 + rng.uniform(-0.5, 0.5)))
        check["tote"] = ("in_zone", CUBBY_ZONE)
        check["basket"] = ("near", BASKET_HOME, 0.20)
        check["chair_36"] = ("tucked", "desk_36")
        check["route"] = [(2.65, 0.6), (0., 8.5)]
    elif task == "move-teacher-desk":
        # a quarter turn: the modesty panel faces the windows, the teacher sits on the room side facing them
        targets["teacher_desk"] = (-2.95, 1.30, math.pi/2)
        targets["teacher_chair"] = (-2.95 + 0.381 + 0.30, 1.30 - 0.18, -math.pi/2)
        check["teacher_desk"] = "on_marks"
        check["teacher_chair"] = ("behind", "teacher_desk", (0.35, 0.95))
    elif task == "group-work":
        # The checks name pod places, not furniture: slot_<r><c> is the place desk_<r><c>'s target puts a desk (its dots,
        # its facing) and seat_<r><c> a chair at the desk in that place. The evaluator matches the room's identical
        # desks and chairs to the places (task.Evaluator.by_place).
        for pair in range(3):
            targets.update(pod_targets(pair))
        for r in (1, 2):
            for c in range(1, 7):
                check[f"slot_{r}{c}"] = ("slot", f"desk_{r}{c}")
                check[f"seat_{r}{c}"] = ("seat", f"slot_{r}{c}")
        for pair in range(3):
            check[f"pod_{pair}"] = [f"slot_{r}{c}" for r in (1, 2) for c in (2*pair + 1, 2*pair + 2)]
        check["rest_of_room"] = "as_found"
        check["route"] = [(2.65, 0.6), (0., 8.5)]
        # the rows 1-2 chairs where the students left them, pulled out 0.48-0.66 m from tucked (seat front 0.06-0.24 m
        # beyond the desk edge), up to 3 cm sideways and 10 deg turned: clear of the desk's rear legs (a turned chair
        # nearer than ~0.45 m puts its seat through them, which layout.conflicts does not see) and at most 3 mm wider
        # than the desk each side, so the aisles keep >= 0.922 m.
        for r in (1, 2):
            for c in range(1, 7):
                ch = f"chair_{r}{c}"
                x, y, yaw = items[ch]["pose"]
                place(ch, lambda: (x + rng.uniform(-0.03, 0.03), y + rng.uniform(0.48, 0.66),
                                   yaw + math.radians(rng.uniform(-10, 10))))
    for n in inverted:
        items[n]["inverted_on"] = "desk_" + n[6:]
    involved = sorted(set(targets) | {n for n in check if n in items})
    books = {}
    if contents:
        desks = [n for n in involved if n.startswith("desk")] or [n for n in items if n.startswith("desk_3")]
        for d in desks:
            books[d] = int(contents)
    slot_refs = {v[1] for v in check.values() if isinstance(v, tuple) and v[0] == "slot"}
    dots = {n: glide_points(items[n]["kind"], t) for n, t in targets.items()
            if items[n]["kind"] in ("desk", "teacher_desk") and (check.get(n) == "on_marks" or n in slot_refs)}
    bad = conflicts(items)
    if bad:
        raise RuntimeError(f"start layout conflicts ({task}, seed {seed}): {bad[:6]}")
    del items["__robot__"]
    return dict(task=task, seed=int(seed), items=items, targets=targets, check=check, involved=involved,
                robot=robot_start(task), glides=glides, books=books, dots=dots,
                instruction=TASKS[task]["instruction"])
