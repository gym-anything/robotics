"""The front yard, its six presets and the task catalogue.

World frame: x east along the walkway, y north toward the house, z up; the walkway slab's broom-finished top is the
datum z = 0 (the robot stands on it) and finished grade is GRADE = 20 mm lower. The soil patches are MPM blocks set
flush into the static ground; their side and bottom faces are rough far-field colliders.

setup(task, seed, **variant) returns the episode spec: robot pose, tools and props with poses, held tools, the soil
spec (blocks, layers, piles, carved holes, static colliders), marks, the tarp and the evaluator's regions.
"""

import math
import numpy as np

from . import params as P

GRADE = -0.020
SLAB = dict(x=(-3.0, 3.0), y=(-1.22, 0.0), z=(-0.12, 0.0))
YARD = dict(x=(-3.5, 3.5), y=(-3.6, 1.80))
HOUSE_Y = 1.80
# the property around the yard (render.py draws it; context_boxes makes it physical): wooden privacy fences at
# x = +-FENCE_X from the street hedge at y = STREET_Y to the house, the house wall above its foundation, the stoop at
# the front door. The physical ground runs to the fences, so a base strafed past x = +-3.5 m does not fall.
FENCE_X = 9.0
STREET_Y = -11.0
DOOR_X = 3.9
ROBOT_NORTH = (0.0, -0.55, math.pi/2)          # base_link on the slab, facing the bed
ROBOT_SOUTH = (0.0, -0.67, -math.pi/2)         # facing the lawn

# ------------------------------------------------------------------ presets (minor setup changes of one yard)
PRESETS = {
    "bed": dict(block=((-0.60, 0.0), (0.60, 1.20)), depth=0.45, voxel="yard",
                layers=[(0.05, "mulch"), (0.25, "topsoil"), (None, "subsoil")], tarp=((0.55, -1.20), (1.75, -0.02)),
                robot=ROBOT_NORTH, mark=(0.0, 0.50)),
    "trench": dict(block=((-1.00, -1.92), (1.00, -1.22)), depth=0.35, voxel="yard",
                   layers=[(0.04, "turf"), (0.20, "topsoil"), (None, "subsoil")], tarp=((-1.00, -2.65), (1.00, -1.97)),
                   robot=ROBOT_SOUTH, line=((-0.75, -1.52), (0.75, -1.52))),
    "post": dict(block=((2.15, -2.12), (3.05, -1.22)), depth=0.80, voxel="yard",
                 layers=[(0.04, "turf"), (0.11, "topsoil"), (None, "yard")], tarp=((1.10, -2.25), (2.10, -1.25)),
                 robot=(2.60, -0.67, -math.pi/2), mark=(2.60, -1.62)),
    "irrigation": dict(block=((-0.70, 0.0), (0.70, 1.00)), depth=0.50, voxel="yard",
                       layers=[(0.05, "mulch"), (0.20, "topsoil"), (None, "subsoil")], tarp=((0.65, -1.20), (1.85, -0.02)),
                       robot=ROBOT_NORTH, pipe=dict(y=0.48, z=GRADE - P.PIPE["depth"] - P.PIPE["od"]/2)),
    "hardpan": dict(block=((-0.50, -2.22), (0.50, -1.22)), depth=0.40, voxel="yard",
                    layers=[(0.12, "hardpan"), (None, "subsoil")], tarp=((0.55, -2.25), (1.55, -1.25)),
                    robot=ROBOT_SOUTH, square=((-0.20, -1.85), (0.20, -1.45))),
    "bulbs": dict(block=((-0.30, 0.0), (0.30, 0.50)), depth=0.30, voxel="fine",
                  layers=[(0.03, "mulch"), (None, "topsoil")], tarp=((0.40, -1.20), (1.60, -0.02)),
                  robot=ROBOT_NORTH, marks=[(-0.12, 0.20), (0.0, 0.20), (0.12, 0.20), (-0.06, 0.32), (0.06, 0.32)]),
}

# ------------------------------------------------------------------ tasks
TASKS = {
    "penetrate-marked": dict(preset="bed", held=dict(both="spade"), robots=("mobile-fr3-duo",),
        instruction="Push the spade blade vertically into the bed at the white paint mark until the blade's cutting edge "
                    "is at least 15 cm below the surface, keeping the blade within 10 degrees of vertical and both hands "
                    "on the spade, without a protective stop."),
    "spit-to-tarp": dict(preset="bed", held=dict(both="spade"), robots=("mobile-fr3-duo",),
        instruction="Dig one spadeful (a spit) of soil from the bed at the white mark and dump it on the blue tarp on the "
                    "walkway: at least 1.5 kg of bed soil must end up on the tarp and no more than 15 % of what was lifted "
                    "may be spilled elsewhere."),
    "planting-hole": dict(preset="bed", held=dict(both="spade"), robots=("mobile-fr3-duo",),
        instruction="Dig a planting hole for a #1 container shrub (root ball 165 mm across, 160 mm tall) centred on the "
                    "white mark: 160 +- 20 mm deep, 330-500 mm across at the top, with a firm floor of undisturbed soil "
                    "(no loose crumbs left on the bottom); put the spoil on the tarp, not on the walkway."),
    "plant-and-backfill": dict(preset="bed", held=dict(right="trowel"), robots=("mobile-fr3-duo", "mobile-fr3-single"),
        predug=dict(kind="planting_hole", rootball=True), instruction=
        "The shrub has been set in its planting hole. Backfill around the root ball with the soil on the tarp and firm it: "
        "leave no air pockets, keep the root ball's top 0-50 mm above grade and free of soil, and finish the backfill "
        "level with the surrounding grade (+-20 mm)."),
    "cable-trench": dict(preset="trench", held=dict(both="trench_shovel"), robots=("mobile-fr3-duo",),
        instruction="Dig a trench for a low-voltage path-light cable along the white line between the two path lights: "
                    "at least 16 cm deep (150 mm cover) over 90 % of its 1.5 m, 6-20 cm wide, centred within 5 cm of the "
                    "line, the turf cut through, the spoil on the lawn-side tarp, and the robot staying on the walkway."),
    "post-hole": dict(preset="post", held=dict(both="digger"), robots=("mobile-fr3-duo",),
        instruction="Dig a post hole at the stake with the clamshell post-hole digger: at least 60 cm deep at the centre, "
                    "20-30 cm across at mid-depth, the floor firm, the spoil on the tarp."),
    # the verifier's operationalisation of "plumb and solid on its own" (brace release, push load, tolerances) is in
    # task.Evaluator._set_post and params.POST_TEST, not in the instruction
    "set-post": dict(preset="post", held=dict(right="trowel", left="tamper"), robots=("mobile-fr3-duo",),
        predug=dict(kind="post_hole", post=True), kit=["trench_shovel"], instruction=
        "The 4x4 post stands in its dug hole, held plumb by temporary braces. Backfill around it with the soil on the "
        "tarp, back up to grade, tamping the fill firm with the 2x4, so that the post stands plumb and solid on its own "
        "once the braces come off."),
    "break-hardpan": dict(preset="hardpan", held=dict(right="mattock"), robots=("mobile-fr3-duo", "mobile-fr3-single"),
        stones=(3, 6), instruction=
        "Break up the compacted ground inside the marked square with the hand mattock: loosen it at least 8 cm deep over "
        "80 % of the square and put at least 4 kg of the broken soil on the tarp."),
    "auger-holes": dict(preset="bulbs", held=dict(both="drill"), robots=("mobile-fr3-duo",),
        instruction="Drill five bulb holes at the five white marks with the drill-mounted planting auger: each at least "
                    "18 cm deep and 70-95 mm wide at 5 cm depth, centred within 2 cm of its mark and left open; keep hold "
                    "of the drill and avoid protective stops. The trigger is squeezed by the gripper."),
    "backfill-pipe": dict(preset="irrigation", held=dict(right="trowel", left="tamper"), robots=("mobile-fr3-duo",),
        predug=dict(kind="pipe_trench"), instruction=
        "The repaired irrigation lateral lies exposed in its trench. Bed and haunch the pipe with the sand in the pail, "
        "cover it with at least 5 cm of sand, then backfill the native soil from the tarp in lifts no thicker than 10 cm, "
        "tamping each, to grade (+-20 mm). Do not strike the pipe."),
    "expose-line": dict(preset="irrigation", held=dict(right="trowel"), robots=("mobile-fr3-duo", "mobile-fr3-single"),
        instruction="Hand-dig to expose the marked irrigation lateral (purple marks, about 30 cm deep): uncover its crown "
                    "over at least 15 cm of length without striking it, digging only inside the marked tolerance zone; "
                    "put the spoil on the tarp."),
}
TASK_ORDER = tuple(TASKS)


def preset_of(task):
    return TASKS[task]["preset"]


# ------------------------------------------------------------------ static ground
def _rect_minus(outer, holes):
    """Axis-aligned rectangle tiling of outer minus holes: [(x0, y0, x1, y1)]."""
    xs = sorted({outer[0], outer[2], *[h[0] for h in holes], *[h[2] for h in holes]})
    ys = sorted({outer[1], outer[3], *[h[1] for h in holes], *[h[3] for h in holes]})
    xs = [x for x in xs if outer[0] <= x <= outer[2]]
    ys = [y for y in ys if outer[1] <= y <= outer[3]]
    cells = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = 0.5*(xs[i] + xs[i + 1]), 0.5*(ys[j] + ys[j + 1])
            if any(h[0] <= cx <= h[2] and h[1] <= cy <= h[3] for h in holes):
                continue
            cells.append([xs[i], ys[j], xs[i + 1], ys[j + 1]])
    # merge along x where the y-range matches
    merged = True
    while merged:
        merged = False
        for a in range(len(cells)):
            for b in range(len(cells)):
                A, B = cells[a], cells[b]
                if a != b and A[1] == B[1] and A[3] == B[3] and abs(A[2] - B[0]) < 1e-9:
                    cells[a] = [A[0], A[1], B[2], A[3]]
                    del cells[b]
                    merged = True
                    break
            if merged:
                break
    return [tuple(c) for c in cells]


def ground_boxes(preset):
    """Static boxes of the yard outside the soil block: slab, its base course, turf/mulch ground at grade, block floor."""
    pr = PRESETS[preset]
    (bx0, by0), (bx1, by1) = pr["block"]
    depth = pr["depth"]
    out = []
    s = SLAB
    out.append(dict(name="slab", lo=(s["x"][0], s["y"][0], s["z"][0]), hi=(s["x"][1], s["y"][1], s["z"][1]),
                    surface="concrete"))
    out.append(dict(name="base_course", lo=(YARD["x"][0], s["y"][0], -0.60), hi=(YARD["x"][1], s["y"][1], s["z"][0]),
                    surface="ground"))
    # slab-length strips beyond the slab ends (x outside +-3 m on the walkway line)
    for x0, x1 in ((YARD["x"][0], s["x"][0]), (s["x"][1], YARD["x"][1])):
        out.append(dict(name=f"walk_end_{'w' if x0 < 0 else 'e'}", lo=(x0, s["y"][0], s["z"][0]), hi=(x1, s["y"][1], GRADE),
                        surface="ground"))
    for side, (y0, y1) in (("north", (s["y"][1], YARD["y"][1])), ("south", (YARD["y"][0], s["y"][0]))):
        holes = []
        if by0 >= y0 - 1e-9 and by1 <= y1 + 1e-9:
            holes.append((bx0, by0, bx1, by1))
        for k, (x0, yy0, x1, yy1) in enumerate(_rect_minus((YARD["x"][0], y0, YARD["x"][1], y1), holes)):
            out.append(dict(name=f"ground_{side}_{k}", lo=(x0, yy0, -0.60), hi=(x1, yy1, GRADE),
                            surface="bed" if side == "north" else "lawn"))
    out.append(dict(name="block_floor", lo=(bx0 - 0.05, by0 - 0.05, -1.0), hi=(bx1 + 0.05, by1 + 0.05, GRADE - depth),
                    surface="far_field"))
    out.append(dict(name="foundation", lo=(YARD["x"][0], HOUSE_Y, GRADE - 0.3), hi=(YARD["x"][1], HOUSE_Y + 0.25, 0.30),
                    surface="concrete"))
    return out


def context_boxes():
    """Static boxes of the property beyond the simulated yard (rigid world only; the soil never reaches them): lawn and
    foundation bed at grade out to the side fences and the street hedge, the house wall and foundation along the whole
    front, the front-door stoop, the fences (1.8 m) and the hedge (a dense 1.2 m hedge stops a base like a wall)."""
    X0, X1 = YARD["x"]
    Y0 = YARD["y"][0]
    F, S, H = FENCE_X, STREET_Y, HOUSE_Y
    out = [dict(name="ctx_lawn_south", lo=(-F, S, -0.60), hi=(F, Y0, GRADE), surface="lawn"),
           dict(name="ctx_lawn_w", lo=(-F, Y0, -0.60), hi=(X0, 0.0, GRADE), surface="lawn"),
           dict(name="ctx_lawn_e", lo=(X1, Y0, -0.60), hi=(F, 0.0, GRADE), surface="lawn"),
           dict(name="ctx_bed_w", lo=(-F, 0.0, -0.60), hi=(X0, H, GRADE), surface="bed"),
           dict(name="ctx_bed_e", lo=(X1, 0.0, -0.60), hi=(F, H, GRADE), surface="bed"),
           dict(name="ctx_foundation_w", lo=(-F, H, GRADE - 0.3), hi=(X0, H + 0.25, 0.30), surface="concrete"),
           dict(name="ctx_foundation_e", lo=(X1, H, GRADE - 0.3), hi=(F, H + 0.25, 0.30), surface="concrete"),
           dict(name="ctx_house_wall", lo=(-F, H, 0.30), hi=(F, H + 0.25, 2.9), surface="concrete"),
           dict(name="ctx_stoop_top", lo=(DOOR_X - 0.75, H - 0.90, 0.0), hi=(DOOR_X + 0.75, H, 0.30), surface="concrete"),
           dict(name="ctx_stoop_step", lo=(DOOR_X - 0.75, H - 1.255, 0.0), hi=(DOOR_X + 0.75, H - 0.905, 0.15),
                surface="concrete"),
           dict(name="ctx_fence_w", lo=(-F - 0.02, S, GRADE), hi=(-F + 0.10, H, 1.80), surface="wood"),
           dict(name="ctx_fence_e", lo=(F - 0.10, S, GRADE), hi=(F + 0.02, H, 1.80), surface="wood"),
           dict(name="ctx_hedge", lo=(-F, S - 0.5, GRADE), hi=(F, S + 0.5, 1.20), surface="hedge")]
    return out


def adjacent_to_block(box, preset):
    (bx0, by0), (bx1, by1) = PRESETS[preset]["block"]
    lo, hi = box["lo"], box["hi"]
    tol = 1e-6
    touch_x = (abs(hi[0] - bx0) < tol or abs(lo[0] - bx1) < tol) and lo[1] < by1 and hi[1] > by0
    touch_y = (abs(hi[1] - by0) < tol or abs(lo[1] - by1) < tol) and lo[0] < bx1 and hi[0] > bx0
    return touch_x or touch_y or box["name"] == "block_floor"


# ------------------------------------------------------------------ seeded variation
def _rng(task, seed):
    import hashlib
    return np.random.default_rng(int(hashlib.sha256(f"{task}/{seed}".encode()).hexdigest()[:8], 16))


def _stones(rng, n, region, z_range, sizes=(0.04, 0.09)):
    out = []
    tries = 0
    while len(out) < n and tries < 400:
        tries += 1
        a = rng.uniform(*sizes)
        r = (a, a*rng.uniform(0.6, 0.9), a*rng.uniform(0.45, 0.7))
        p = (rng.uniform(region[0][0] + 0.06, region[1][0] - 0.06), rng.uniform(region[0][1] + 0.06, region[1][1] - 0.06),
             rng.uniform(*z_range))
        if all(np.linalg.norm(np.subtract(p, o["pos"])) > a + max(o["radii"]) + 0.03 for o in out):
            yaw = rng.uniform(0, math.pi)
            from scipy.spatial.transform import Rotation
            q = tuple((Rotation.from_euler("z", yaw)*Rotation.from_euler("x", rng.uniform(-0.3, 0.3))).as_quat())
            out.append(dict(radii=r, pos=p, quat=q))
    return out


def setup(task, seed=0, moisture=None, topsoil=None, stones=None):
    """The episode spec of one task variation (seeded; the same task/seed/variant always gives the same yard)."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(TASKS)}")
    T = TASKS[task]
    preset = T["preset"]
    pr = PRESETS[preset]
    rng = _rng(task, seed)
    moisture = moisture or "moist"
    if moisture not in P.MOISTURE:
        raise ValueError(f"moisture must be one of {sorted(P.MOISTURE)}")
    layers = [list(l) for l in pr["layers"]]
    if topsoil is not None:
        if not 0.12 <= float(topsoil) <= 0.32:
            raise ValueError("topsoil depth must be 0.12-0.32 m")
        for l in layers:
            if l[1] == "topsoil":
                l[0] = float(topsoil)
    (bx0, by0), (bx1, by1) = pr["block"]
    depth = pr["depth"]
    spec = dict(task=task, preset=preset, seed=int(seed), moisture=moisture, instruction=T["instruction"],
                robot=tuple(pr["robot"]), held=dict(T["held"]), block=dict(lo=(bx0, by0, GRADE - depth), hi=(bx1, by1, GRADE)),
                layers=[(l[0], l[1]) for l in layers], voxel=P.VOXEL[pr["voxel"]], tarp=pr["tarp"], grade=GRADE,
                items={}, marks=[], props={}, predug=T.get("predug"))
    # marks (render-only paint; the evaluator reads the same coordinates)
    jitter = lambda s: rng.uniform(-s, s)
    if "mark" in pr:
        mx, my = pr["mark"]
        if preset == "bed":
            mx, my = mx + jitter(0.08), my + jitter(0.04)
        spec["marks"].append(dict(kind="dot", pos=(mx, my), r=0.03, color="white"))
        spec["target"] = (mx, my)
    if "line" in pr:
        (x0, y0), (x1, y1) = pr["line"]
        spec["marks"].append(dict(kind="line", a=(x0, y0), b=(x1, y1), w=0.04, color="white"))
        spec["target_line"] = ((x0, y0), (x1, y1))
        for k, x in enumerate((x0, x1)):
            spec["props"][f"path_light_{k}"] = dict(kind="path_light", pos=(x, y0 + 0.0), static=True)
    if "square" in pr:
        (x0, y0), (x1, y1) = pr["square"]
        spec["marks"].append(dict(kind="square", lo=(x0, y0), hi=(x1, y1), w=0.03, color="white"))
        spec["target_square"] = ((x0, y0), (x1, y1))
    if "marks" in pr:
        dx, dy = jitter(0.02), jitter(0.02)
        pts = [(x + dx, y + dy) for x, y in pr["marks"]]
        spec["targets"] = pts
        for p in pts:
            spec["marks"].append(dict(kind="dot", pos=p, r=0.015, color="white"))
    if "pipe" in pr:
        pp = pr["pipe"]
        spec["pipe"] = dict(y=pp["y"], z=pp["z"], r=P.PIPE["od"]/2, x=(-3.0, 3.0))
        for x in np.linspace(bx0 + 0.1, bx1 - 0.1, 4):
            for dy in (-0.457, 0.457):
                spec["marks"].append(dict(kind="flag", pos=(float(x), pp["y"] + dy), color="purple"))
        spec["marks"].append(dict(kind="line", a=(bx0 + 0.05, pp["y"]), b=(bx1 - 0.05, pp["y"]), w=0.03, color="purple"))
        spec["tolerance_zone"] = (pp["y"] - 0.457, pp["y"] + 0.457)
        spec["props"]["valve_box"] = dict(kind="valve_box", pos=(bx1 - 0.18, 0.80), static=True)
    if preset == "post":
        spec["props"]["stake"] = dict(kind="stake", pos=pr["mark"], static=True)
    # stones (seeded): E always, A/C a few by default
    n_st = stones
    if n_st is None:
        n_st = int(rng.integers(*T["stones"])) if "stones" in T else (int(rng.integers(0, 3)) if preset in ("bed", "post") else 0)
    if n_st:
        top = layers[0][0] if preset == "hardpan" else 0.12
        region = pr.get("square", pr["block"]) if preset == "hardpan" else pr["block"]
        if preset == "hardpan":
            region = ((region[0][0] - 0.05, region[0][1] - 0.05), (region[1][0] + 0.05, region[1][1] + 0.05))
        sts = _stones(rng, n_st, region, (GRADE - 0.20, GRADE - 0.05) if preset != "hardpan" else (GRADE - 0.13, GRADE - 0.05),
                      sizes=(0.04, 0.10) if preset == "hardpan" else (0.03, 0.06))
        for k, s in enumerate(sts):
            spec["items"][f"stone_{k}"] = dict(kind="stone", radii=s["radii"], pose=(*s["pos"], s["quat"]), coupled=True)
    # held tools are placed into the hands by the environment; the rest of the kit lies on the walkway. A task may name
    # its own kit: set-post's ~35 L backfill is shovel work, not 100+ trowel loads
    spec["kit"] = list(T["kit"]) if "kit" in T else dict(bed=["trowel"], trench=["spade"], post=["tamper"],
                                                         irrigation=["spade"], hardpan=["trench_shovel"],
                                                         bulbs=[]).get(preset, [])
    if T.get("predug"):
        predug(spec, T["predug"])
        if T["predug"]["kind"] == "post_hole":
            # a seeded stone whose body reaches into the pre-dug post hole would hang in the open ring (or cut the
            # post) at the start: 124 of set-post seeds 1-999 seed one inside the cut.
            # Such stones are left out; the others keep their seeded poses (seed 0 has none inside).
            cut = spec["cuts"][-1]
            for name in [n for n, it in spec["items"].items() if it["kind"] == "stone"]:
                st = spec["items"][name]
                grow = dict(cut, size=(cut["size"][0] + max(st["radii"]) + 0.01, cut["size"][1]))
                if dug_wall(np.asarray([st["pose"][:3]], float), grow)[0]:
                    del spec["items"][name]
    spec["regions"] = regions(spec)
    spec["soil"] = soil_spec(spec)
    return spec


def tarp_top(spec):
    (x0, y0), (x1, y1) = spec["tarp"]
    on_slab = y1 <= 0.0 and y0 >= SLAB["y"][0]
    return (0.0 if on_slab else GRADE) + 0.001


def dug_wall(pos, cut):
    """Inside a hand-dug excavation: the nominal cut (cylinder or box) widened outward at its walls by a hand-dug
    irregularity -- spade-width bulges around the wall, a slight taper (wider at the top) and a ragged lip within 4 cm
    of grade -- never inward, so the nominal cavity (the root ball, the post and its gravel base, the pipe's bedding)
    stays clear. The floor stays at the nominal depth. A dug hole or trench does not have machine-perfect
    cylindrical or vertical-box walls."""
    pos = np.asarray(pos, float)
    c = np.asarray(cut["pos"], float)
    z_bot = c[2] - cut["size"][-1]                    # cylinder size (r, half height); box (hx, hy, hz)
    rng = np.random.default_rng(int(cut.get("hand_seed", 0)))
    ph = rng.uniform(0, 2*math.pi, 6)
    zf = np.clip((pos[:, 2] - z_bot)/max(GRADE - z_bot, 1e-6), 0.0, 1.0)
    lip = (pos[:, 2] > GRADE - 0.04)
    if cut["kind"] == "cylinder":
        r = cut["size"][0]
        d = pos[:, :2] - c[:2]
        th = np.arctan2(d[:, 1], d[:, 0])
        e = np.clip(0.010 + 0.010*np.sin(3*th + ph[0]) + 0.006*np.sin(7*th + ph[1]), 0.0, None) + 0.025*zf
        e = e + lip*0.018*(0.5 + 0.5*np.sin(11*th + ph[2]))
        return (np.hypot(d[:, 0], d[:, 1]) < r + e) & (pos[:, 2] > z_bot) & (pos[:, 2] < c[2] + cut["size"][1])
    hx, hy, hz = cut["size"]
    x = pos[:, 0] - c[0]
    e = np.clip(0.008 + 0.010*np.sin(9.0*x + ph[3]) + 0.006*np.sin(23.0*x + ph[4]), 0.0, None) + 0.02*zf
    e = e + lip*0.015*(0.5 + 0.5*np.sin(31.0*x + ph[5]))
    return (np.abs(x) < hx + 0.6*e) & (np.abs(pos[:, 1] - c[1]) < hy + e) & (pos[:, 2] > z_bot) & (pos[:, 2] < c[2] + hz)


def predug(spec, pd):
    """Assisted start states: the excavation was dug before tick 0 (its particles relocated loose onto the tarp,
    mixed as dug spoil is) and the plant / post / pipe is in place."""
    (tx0, ty0), (tx1, ty1) = spec["tarp"]
    tc = ((tx0 + tx1)/2, (ty0 + ty1)/2)
    zt = tarp_top(spec)
    spec["cuts"] = []
    spec["fills"] = []
    if pd["kind"] == "planting_hole":
        mx, my = spec["target"]
        R = P.ROOTBALL
        depth = R["height"] - 0.025                  # the ball's top 25 mm above grade (CSU / Ohio: 1-2 in)
        r = 0.21                                     # a 420 mm hole: 2.5 x the 165 mm ball
        spec["cuts"].append(dict(kind="cylinder", size=(r, (depth + 0.10)/2), pos=(mx, my, GRADE - depth + (depth + 0.10)/2),
                                 quat=(0., 0., 0., 1.), dest=dict(center=tc, radius=0.28, z=zt, slope=32.0),
                                 hand_seed=spec["seed"] + 101))
        spec["items"]["rootball"] = dict(kind="rootball", pose=(mx, my, GRADE - depth + 0.002, (0., 0., 0., 1.)), coupled=True)
        spec["predug"] = dict(pd, radius=r, depth=depth)
    elif pd["kind"] == "post_hole":
        mx, my = spec["target"]
        r, depth, base = 0.125, 0.65, 0.10
        spec["cuts"].append(dict(kind="cylinder", size=(r, (depth + 0.10)/2), pos=(mx, my, GRADE - depth + (depth + 0.10)/2),
                                 quat=(0., 0., 0., 1.), dest=dict(center=tc, radius=0.35, z=zt, slope=32.0),
                                 hand_seed=spec["seed"] + 102))
        floor = GRADE - depth

        cut = spec["cuts"][-1]

        def gravel(h, mx=mx, my=my, floor=floor, r=r, base=base, cut=cut):
            f = P.CRITICAL_FRACTION
            hl = h/f**(1/3)
            R = r + 0.06
            g = np.arange(-R, R, hl) + hl/2
            nz = max(1, round(base/hl))
            # Fill the specified thickness with complete equal cells; avoid
            # truncating the top gravel row and leaving an artificial void.
            zz = floor + (np.arange(nz) + 0.5)*(base/nz)
            X, Y, Z = np.meshgrid(g, g, zz, indexing="ij")
            pts = np.stack([X.ravel() + mx, Y.ravel() + my, Z.ravel()], 1)
            # fill the dug cavity, half a spacing in from its wall
            inn = dug_wall(pts, dict(cut, size=(cut["size"][0] - 0.5*hl, cut["size"][1])))
            return pts[inn].astype(np.float32)
        spec["fills"].append(dict(material="gravel", points=gravel))
        a = P.POST["section"]
        z_post = floor + base  # seated on the nominal gravel surface; no authored air gap
        spec["items"]["post"] = dict(kind="post", pose=(mx, my, z_post, (0., 0., 0., 1.)), coupled=True)
        Bc = P.BRACE
        zl = GRADE + Bc["height"] - z_post
        # braces on two adjacent faces, staked in the lawn away from the robot and the tarp: east (+x) and south (-y).
        # A brace on the north (+y) face would put its stake on the concrete walkway inside the robot's base footprint.
        spec["braces"] = [((a/2, 0., zl), (mx + Bc["reach"], my, GRADE + 0.05)),
                          ((0., -a/2, zl), (mx, my - Bc["reach"], GRADE + 0.05))]
        # the 2x4 waits 10 cm nearer the robot: at the default spot its face stood 5.5 mm from the east brace's line
        spec["start_offset"] = dict(tamper=(0., 0.10, 0.))
        spec["predug"] = dict(pd, radius=r, depth=depth, base=base)
    elif pd["kind"] == "pipe_trench":
        pp = spec["pipe"]
        b = spec["block"]
        x0, x1, w = b["lo"][0] + 0.25, b["hi"][0] - 0.25, 0.30
        bottom = pp["z"] - pp["r"]
        spec["cuts"].append(dict(kind="box", size=((x1 - x0)/2, w/2, (GRADE - bottom + 0.10)/2),
                                 pos=((x0 + x1)/2, pp["y"], (GRADE + 0.10 + bottom)/2), quat=(0., 0., 0., 1.),
                                 dest=dict(center=tc, radius=0.40, z=zt, slope=32.0), hand_seed=spec["seed"] + 103))
        # bedding sand in a 2-gallon pail on the walkway beyond the tarp (beside the robot's base it touched the
        # base hull and the held tamper at tick 0: protective stop in the first stills run)
        px, py = tx1 + 0.25, (ty0 + ty1)/2
        spec["props"]["sand_pail"] = dict(kind="pail", pos=(px, py), static=True)

        def sand(h, px=px, py=py):
            f = P.CRITICAL_FRACTION
            hl = h/f**(1/3)
            r = P.PAIL["r"] - 0.006
            g = np.arange(-r, r, hl) + hl/2
            zz = 0.006 + (np.arange(int(P.PAIL["fill"]/hl)) + 0.5)*hl
            X, Y, Z = np.meshgrid(g, g, zz, indexing="ij")
            keep = np.hypot(X, Y) < r - 0.5*hl
            return np.stack([X[keep] + px, Y[keep] + py, Z[keep]], 1).astype(np.float32)
        spec["fills"].append(dict(material="sand", points=sand))
        spec["predug"] = dict(pd, x=(x0, x1), width=w, bottom=bottom)


def regions(spec):
    """Evaluator regions (owner side): the bed/hole region, the tarp, the walkway, and per-task targets."""
    b = spec["block"]
    (tx0, ty0), (tx1, ty1) = spec["tarp"]
    return dict(block=((b["lo"][0], b["lo"][1]), (b["hi"][0], b["hi"][1])), tarp=((tx0, ty0), (tx1, ty1)),
                walkway=((SLAB["x"][0], SLAB["y"][0]), (SLAB["x"][1], SLAB["y"][1])))


def soil_spec(spec):
    """Inputs of soil.SoilSystem for the episode (particles are generated by world.build_soil from this)."""
    return dict(voxel=spec["voxel"], moisture=spec["moisture"], block=spec["block"], layers=spec["layers"],
                seed=spec["seed"])
