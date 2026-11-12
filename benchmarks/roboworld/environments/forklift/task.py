"""The task catalogue: start states, work orders and seed variation. The evaluators are in evaluator.py.

A start state fixes the truck's pose and condition (engine, key, parking brake, direction lever, fork height and tilt),
the loads and their poses, and the floor condition along the route. Seeds vary them within stated ranges;
every resolved start is built, settled by physics (env.settle) and checked before tick 0. Nothing in a start state is
authored geometry standing in for earlier work: a load "on the forks" is placed in the fork openings and settled.

A benchmark task's own start is in its task folder (tasks/<task>/setup.py) and its own checks in
tasks/<task>/verifier.py; resolve() and evaluator.evaluate() call them. The other tasks' starts are here.
"""

import math
import zlib

import numpy as np

from gym_anything_robotics import task_folders

from . import params as P

PACKAGE = "forklift"


def task_setup(task):
    """The benchmark task's setup.py (tasks/<task>/), or None for a task outside the benchmark."""
    return task_folders.setup(PACKAGE, task) if task_folders.exists(PACKAGE, task) else None


def task_verifier(task):
    """The benchmark task's verifier.py (tasks/<task>/), or None for a task outside the benchmark."""
    return task_folders.verifier(PACKAGE, task) if task_folders.exists(PACKAGE, task) else None


INSTRUCTIONS = {
    "start-and-park": "Start the truck, release the parking brake, drive to the truck parking bay at the west end of the "
                      "main aisle and park it there inside the painted lines: forks fully lowered and tilted forward until "
                      "their tips rest on the floor, direction lever in neutral, parking brake set, engine off and key "
                      "turned off.",
    "aisle-cruise": "Drive the main aisle {direction} at the posted 8 km/h, slow down and sound the horn at both "
                    "intersections, and stop at the white line at the far end of the aisle.",
    "drive-to-pallet": "Drive to the pallet in staging lane {lane} along the dock wall and stop square in front of it, "
                       "ready to pick it up: forks level, at the pallet's openings, just short of the pallet.",
    "carry-to-floor-bay": "Take the pallet on your forks to floor storage bay {bay}, set it down inside the painted lines, "
                          "back out and lower the forks.",
    "carry-loose-stack": "Move the pallet of loose cartons on your forks to floor storage bay {bay} without spilling any "
                         "carton, set it down inside the lines, back out and lower the forks.",
    "put-away-rack": "Put the pallet on your forks away in rack location {loc} (aisle {aisle}, bay {bay}, on the {ordinal} "
                     "beam level, {height} m above the floor, the {hand}-hand pallet position as you face the rack), then "
                     "back out and lower the forks.",
    "feed-conveyor": "Set the coil pallet on your forks on the press infeed conveyor between its guide rails so that the "
                     "conveyor takes it to the press, then back away and lower the forks.",
    "load-trailer": "Load the pallet on your forks into the trailer at door {door}: {side} side, directly behind the two "
                    "pallets at the nose. Then back out onto the dock. Before you drive onto the dock leveler, check the "
                    "restraint signal beside the door: enter only on green. On red the trailer is not secured: drive up "
                    "to the door and wait there, past the end of the staging lane and short of the leveler, with the "
                    "direction lever in neutral and the parking brake set.",
    "load-trailer-chocked": "Load the pallet on your forks into the trailer at door {door}: {side} side, directly behind "
                            "the two pallets at the nose, then back out onto the dock. The door has no vehicle restraint: "
                            "the trailer is held by its own spring brakes and the wheel chocks.",
    "load-bales": "Load two cardboard bales from the bale row into the trailer at door {door}: set them side by side on the "
                  "dunnage strips laid at the nose, against the front wall, then back out onto the dock.",
    "unload-trailer": "Unload the {side}-hand pallet at the nose of the trailer at door {door} and bring it out; back out "
                      "of it, lower the forks and stop once it is set down in staging lane {lane}.",
    "ramp-up-loaded": "Drive up the yard ramp into the trailer with the pallet on your forks and set it down at the nose, "
                      "on the {side} side against the side wall (left and right as you face the nose).",
    "ramp-down-loaded": "Bring the pallet on your forks out of the trailer, down the yard ramp, and set it down in the "
                        "painted yard drop zone {zone} of the ramp, then back out and lower the forks.",
    "yard-route": "Drive the marked yard lane {route} and stop square to {what} at the yellow line.",
}
DIFFICULTY = {"start-and-park": "easy", "aisle-cruise": "easy", "drive-to-pallet": "medium", "carry-to-floor-bay": "medium",
              "carry-loose-stack": "very hard", "put-away-rack": "hard", "feed-conveyor": "hard",
              "load-trailer": "hard", "load-trailer-chocked": "hard", "load-bales": "very hard",
              "unload-trailer": "very hard", "ramp-up-loaded": "hard", "ramp-down-loaded": "hard", "yard-route": "easy"}
TIME_LIMIT_S = {"start-and-park": 90.0, "aisle-cruise": 120.0, "drive-to-pallet": 120.0, "carry-to-floor-bay": 180.0,
                "carry-loose-stack": 240.0, "put-away-rack": 240.0, "feed-conveyor": 180.0, "load-trailer": 240.0,
                "load-trailer-chocked": 240.0, "load-bales": 480.0, "unload-trailer": 300.0, "ramp-up-loaded": 240.0,
                "ramp-down-loaded": 240.0, "yard-route": 150.0}
SPLITS = dict(train=(0, 799), validation=(800, 899), test=(900, 999))
# Drawn unit loads (wrapped, on a GMA pallet): load mass 600-1,200 kg and height 0.9-1.5 m within the task's type
# (authored), around a nominal 900 kg / 1.2 m unit (Walmart's < 953 kg, < 72 in, read: a retail supplier's limit,
# which a general warehouse's loads may exceed); the yard
# tasks' 900-1,300 kg about a 1,200 kg load (authored); the loads already in a trailer 500-900 kg.
# Every drawn height keeps the unit under 72 in (1.83 m) with its 5.5 in pallet.


def blade_bottom_for(lift):
    """Joint values (carriage, mast_int) that put the fork blades' bottom `lift` m above the floor (free lift first)."""
    c = lift - 0.009 + P.CARRIAGE_LOWER
    if c <= P.CARRIAGE_FREE_LIFT:
        return c, 0.0
    return P.CARRIAGE_FREE_LIFT, (c - P.CARRIAGE_FREE_LIFT) / 2.0


# Seed variation of the work order: which lane, bay, rack location, dock door, trailer side. Drawn from
# a stream of its own, so that facility.build (the rack location must be left empty) and resolve agree without a layout.
# A benchmark task's own draws are its setup.py's: target(rng), and put-away-rack's rack location.
FLOOR_BAYS = ("B-01", "B-02", "B-03", "B-04", "B-05", "B-07", "B-08", "B-09", "B-10", "B-11")
# (B-06 and B-12, the bays beside lane 2, are not targets: from the receiving lanes the truck leaves lane 2 with a
# right-angle turn and would have to turn again into those bays within ~2 m, with no straight run-in to square up)
LOAD_DOORS = ("door1", "door2", "door3", "door4")           # load-trailer, load-trailer-chocked, unload-trailer
# feed-conveyor, ramp-up-loaded and yard-route vary within what the facility offers
FEED_STARTS = ("lane 2", "R1", "R2", "R3")  # feed-conveyor: from travel lane 2 north of the conveyor, or a receiving lane
RAMP_SIDES = ("L", "R")                    # ramp-up-loaded: the nose pallet position, left or right facing the nose
YARD_ROUTES = ("to the ramp", "to the apron")  # yard-route: the marked lane either way, to the line at its far end


def _target_rng(config):
    return np.random.default_rng([int(config.seed), zlib.crc32(("target:" + config.task).encode())])


def reserved_locations(config):
    """Rack locations the task needs empty (the facility leaves them free of stored loads): those a benchmark task's
    setup.py names (its reserved_locations(config)); none for every other task."""
    folder = task_setup(config.task)
    if folder is not None and hasattr(folder, "reserved_locations"):
        return folder.reserved_locations(config)
    return set()


def _target(config):
    """The work order's target of (task, seed) for the tasks whose target varies (drawn from _target_rng). A benchmark
    task's own draws are its setup.py's target(rng)."""
    rng = _target_rng(config)
    t = config.task
    folder = task_setup(t)
    if t in ("carry-to-floor-bay", "carry-loose-stack"):
        return dict(bay=FLOOR_BAYS[int(rng.integers(0, len(FLOOR_BAYS)))])
    if t in ("load-trailer", "load-trailer-chocked", "unload-trailer"):
        out = dict(door=LOAD_DOORS[int(rng.integers(0, len(LOAD_DOORS)))], side="LR"[int(rng.integers(0, 2))])
        if folder is not None:                   # load-trailer: whether the trailer is secured (its setup.py)
            out.update(folder.target(rng))
        return out
    if folder is not None:
        return folder.target(rng) if hasattr(folder, "target") else {}
    if t == "feed-conveyor":
        return dict(start_from=FEED_STARTS[int(rng.integers(0, len(FEED_STARTS)))])
    if t == "ramp-up-loaded":
        return dict(side=RAMP_SIDES[int(rng.integers(0, len(RAMP_SIDES)))])
    if t == "yard-route":
        return dict(route=YARD_ROUTES[int(rng.integers(0, len(YARD_ROUTES)))])
    return {}


TRUCK_REACH = dict(x=(-2.14, 1.62), y=0.66)     # the truck's footprint about its reference point (evaluator.TRUCK_HALF)


def truck_box(pose, margin=0.0):
    """Axis-aligned box ((x0, x1), (y0, y1)) around the truck's footprint at pose (x, y, yaw)."""
    x, y, yaw = pose
    c, s_ = math.cos(yaw), math.sin(yaw)
    pts = [(x + c * fx - s_ * fy, y + s_ * fx + c * fy) for fx in TRUCK_REACH["x"]
           for fy in (-TRUCK_REACH["y"], TRUCK_REACH["y"])]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs) - margin, max(xs) + margin), (min(ys) - margin, max(ys) + margin)


def wet_sign_box(centre, half):
    """Where scene.py stands the wet-floor sign of a warehouse floor patch (its static box at x - hx - 0.4, y + hy - 0.1,
    half 0.16 x 0.13): a start state keeps it off the truck and off the truck's route."""
    cx, cy = centre[0] - half[0] - 0.4, centre[1] + half[1] - 0.1
    return (cx - 0.16, cx + 0.16), (cy - 0.13, cy + 0.13)


def _overlap(a, b):
    return a[0][0] < b[0][1] and b[0][0] < a[0][1] and a[1][0] < b[1][1] and b[1][0] < a[1][1]


def _wet_patch(rng, s, keep_clear, x_range=(22.0, 34.0)):
    """A wet patch in the main aisle with its sign at the aisle's north edge (as a spill is signed at the walkway's
    edge, clear of a truck on the aisle centreline): the patch reaches the travelled half of the aisle; redrawn (up to
    20 times) while its sign would stand in one of the keep_clear boxes; none if no draw fits."""
    half = (1.5, 1.2)
    for _ in range(20):
        c = (float(rng.uniform(*x_range)), float(rng.uniform(28.2, 29.2)))
        if not any(_overlap(wet_sign_box(c, half), k) for k in keep_clear):
            s["floor"]["wet_patch"] = dict(centre=c, half=half)
            return


class Start:
    """What a benchmark task's setup.py builds its start on. resolve() makes one and calls the task's setup(start) where
    the task's branch is (carry-to-floor-bay after the start it shares with carry-loose-stack, the dock tasks once the
    trailer bed is drawn).

    config      the session's Config (preset, task, seed)
    s           the start state so far (resolve's dict); setup adds the truck's pose and condition, loads and targets
    rng, j      the generator seeded by (task, seed), and j(a): a draw uniform in +-a from it
    zones       the facility layout's zones
    target      the work order's target of (task, seed) (_target)
    keep_clear  boxes the wet patch's sign must stay out of (truck start, routes); setup adds its own
    wet_x       the x range of a warehouse wet patch; setup may narrow it
    dock tasks also: bed (the trailer floor relative to the dock floor), door (the work order's door) and door_zone (its
    zone), nose_y (the trailer's nose at that door) and trailers (door -> bed_z, restraint, chocks: the work order's
    door, restrained and chocked, unless setup replaces it)"""

    def __init__(self, config, s, rng, j, zones, target, keep_clear, wet_x):
        self.config, self.s, self.rng, self.j = config, s, rng, j
        self.zones, self.target, self.keep_clear, self.wet_x = zones, target, keep_clear, wet_x


def resolve(config, layout):
    """Resolved start state of (preset, task, seed). A benchmark task's own part is its setup.py (Start above)."""
    rng = np.random.default_rng(config.seed * 7919 + zlib.crc32(config.task.encode()) % 1000)
    t = config.task
    j = lambda a: float(rng.uniform(-a, a))
    s = dict(task=t, seed=config.seed, engine="idle", key="on", park=True, direction="N", lift=0.009, tilt_deg=0.0,
             loads=[], load_on_forks=None, floor={}, belt=True)
    Z = layout.zones
    tg = _target(config)
    keep_clear = []                      # boxes the wet patch's sign must stay out of (truck start, routes)
    wet_x = (22.0, 34.0)
    folder = task_setup(t)
    st = Start(config, s, rng, j, Z, tg, keep_clear, wet_x)
    if t in ("carry-to-floor-bay", "carry-loose-stack"):
        lane = Z["receiving"]["R2"]
        s.update(truck=(45.5 + j(0.5), (lane["y"][0] + lane["y"][1]) / 2 + j(0.3), math.pi + j(math.radians(6))),
                 lift=0.15, tilt_deg=-3.0)
        if folder is not None:                   # carry-to-floor-bay: its setup.py puts its load on the forks
            folder.setup(st)
        else:
            s["load_on_forks"] = dict(kind="stack", layers=5)
        s["bay"] = tg["bay"]
    elif t == "feed-conveyor":
        s["start_from"] = tg["start_from"]
        if tg["start_from"] == "lane 2":                     # in travel lane 2 north of the conveyor, heading south
            s.update(truck=(36.0 + j(0.4), 20.0 + j(0.4), -math.pi / 2 + j(math.radians(6))), lift=0.15, tilt_deg=-3.0)
        else:                                                # in a receiving lane (the coil just received), heading west
            rl = Z["receiving"][tg["start_from"]]
            s.update(truck=(45.5 + j(0.5), (rl["y"][0] + rl["y"][1]) / 2 + j(0.3), math.pi + j(math.radians(6))),
                     lift=0.15, tilt_deg=-3.0)
        s["load_on_forks"] = dict(kind="coil", mass=2000.0)
    elif config.preset == "dock":
        IN = 0.0254
        bed = float(rng.uniform(44.0, 52.0)) * IN - P.DOCK_HEIGHT          # trailer floor relative to the dock floor
        g = P.GMA_PALLET
        Wi = P.TRAILER_INSIDE["W"]
        from .facility import BUMPER
        door = tg["door"]
        d = Z["doors"][door]
        nose_y = d["face_y"] + BUMPER + 0.106 + P.TRAILER_INSIDE["L"]
        trailers = {door: dict(bed_z=bed, restraint=(t != "load-trailer-chocked"), chocks=True)}
        if folder is not None:                   # load-trailer, load-bales: their trailers (and load-bales' work order)
            st.bed, st.door, st.door_zone, st.nose_y, st.trailers = bed, door, d, nose_y, trailers
            folder.setup(st)
            trailers = st.trailers
        if t in ("load-trailer", "load-trailer-chocked", "unload-trailer"):
            rows = 1            # one row across the nose: the unload target must be reachable without moving another load
            for r in range(rows):
                # left and right as seen facing the nose from the dock (the trailer's +y; world -x at these doors)
                for side, sx in (("L", -1), ("R", 1)):
                    # pallets loaded 48 in deep along the trailer, two across (pinwheel-free straight load)
                    px = d["x"] + sx * (Wi / 2 - g["W"] / 2 - 0.03)
                    py = nose_y - 0.03 - g["L"] / 2 - r * (g["L"] + 0.05)
                    s["loads"].append(dict(kind="pallet", name=f"trailer_{r + 1}{side}", pose=(px, py, math.pi / 2),
                                           z0=bed, load=dict(kind="wrapped", mass=float(rng.uniform(500.0, 900.0)),
                                                             size=(g["L"] - 0.04, g["W"] - 0.04, float(rng.uniform(1.0, 1.4))))))
        if t in ("load-trailer", "load-trailer-chocked"):
            # in the staging lane in front of the door, heading for it
            s.update(truck=(d["x"] + j(0.3), 9.0 + j(0.5), math.pi / 2 + j(math.radians(5))), lift=0.15, tilt_deg=-3.0)
            s["load_on_forks"] = dict(kind="wrapped", mass=float(rng.uniform(600.0, 1100.0)), height=float(rng.uniform(1.0, 1.4)))
            s["target"] = dict(door=door, side=tg["side"], behind=f"trailer_1{tg['side']}")
        elif t == "unload-trailer":
            s.update(truck=(d["x"] + j(0.3), 12.0 + j(0.4), math.pi / 2 + j(math.radians(4))), lift=0.10)
            s["target"] = dict(door=door, pallet=f"trailer_1{tg['side']}", lane=f"lane{door[-1]}")
        s["trailers"] = trailers
        s["bed_z"] = bed
    elif config.preset == "yard":
        rp = Z["ramp"]
        yt = Z["yard_trailer"]
        s["yard_trailers"] = {"yard_trailer": dict(rear=yt["rear"], yaw=yt["yaw"], bed_z=yt["bed_z"])}
        if t == "ramp-up-loaded":
            # on the yard lane's last straight, 3-6 m from the ramp foot, heading up the ramp
            s.update(truck=(rp["x"] + j(0.2), rp["y_foot"] + float(rng.uniform(3.0, 6.0)), -math.pi / 2 + j(math.radians(3))),
                     lift=0.15, tilt_deg=-3.0)
            s["load_on_forks"] = dict(kind="wrapped", mass=float(rng.uniform(900.0, 1300.0)), height=float(rng.uniform(1.0, 1.3)))
            s["target"] = dict(side=tg["side"])
        elif folder is not None:                 # ramp-down-loaded: its setup.py
            folder.setup(st)
        elif t == "yard-route":
            s["route"] = tg["route"]
            if tg["route"] == "to the ramp":
                # on the dock apron at the lane's start, heading into the lane: the front axle 1.3-4.8 m down the lane
                # from its painted head (y 21.8; the counterweight 2.3-5.8 m clear of the building wall), within +-0.6 m
                # and +-8 deg of its centreline (the footprint at least 0.4 m inside its 4 m width)
                s.update(truck=(36.0 + j(0.6), float(rng.uniform(17.0, 20.5)), -math.pi / 2 + j(math.radians(8))), lift=0.10)
            else:
                # at the ramp foot, heading away from the ramp up the lane's last straight: the front axle 3.0-4.5 m from
                # the foot (the counterweight 0.9-2.4 m clear of the incline), within +-0.3 m and +-6 deg of the lane's
                # centreline (the ramp's cones stand 2.3 m either side of it)
                s.update(truck=(rp["x"] + j(0.3), rp["y_foot"] + float(rng.uniform(3.0, 4.5)), math.pi / 2 + j(math.radians(6))),
                         lift=0.10)
            if rng.random() < 0.4:
                s["floor"]["wet_yard"] = dict(centre=(24.0 + j(4.0), 10.0), half=(3.0, 2.2))
    elif folder is not None:                     # start-and-park, aisle-cruise, drive-to-pallet, put-away-rack
        folder.setup(st)
        wet_x = st.wet_x
    else:
        raise ValueError(f"task {t!r} has no start state")
    if config.preset == "yard" and "wet_yard" in s["floor"]:
        s["floor"]["wet_yard"]["material"] = "asphalt_wet"
    # floor condition along the route: a wet patch with its sign in a third of the warehouse seeds
    if rng.random() < 1 / 3 and config.preset == "warehouse":
        _wet_patch(rng, s, keep_clear + [truck_box(s["truck"], 0.4)], wet_x)
    return s


def instruction(start):
    t = start["task"]
    text = INSTRUCTIONS[t]
    if t == "put-away-rack":
        row, face, bay, level, slot = start["rack_location"].split("-")
        from .facility import BEAM_LEVELS
        # pallet positions on a beam pair: slot 0 at the lower y. Facing a west face (the rack east of the aisle) the
        # truck looks along +x, so +y is on its left; facing an east face it looks along -x
        hand = ("right", "left")[int(slot)] if face == "w" else ("left", "right")[int(slot)]
        text = text.format(loc=f"{row}-{bay}-{int(level) + 1}", aisle=start.get("aisle", "2"), bay=int(bay),
                           ordinal=("first", "second")[int(level) - 1], height=f"{BEAM_LEVELS[int(level) - 1]:.2f}",
                           hand=hand)
    elif t == "drive-to-pallet":
        text = text.format(lane=start["lane"])
    elif t in ("carry-to-floor-bay", "carry-loose-stack"):
        text = text.format(bay=start["bay"])
    elif t in ("load-trailer", "load-trailer-chocked"):
        text = text.format(door=start["target"]["door"][4:], side={"L": "left", "R": "right"}[start["target"]["side"]])
    elif t == "aisle-cruise":
        text = text.format(direction=start.get("cruise", "east") + "ward")
    elif t == "ramp-up-loaded":
        text = text.format(side={"L": "left", "R": "right"}[start.get("target", {}).get("side", "L")])
    elif t == "ramp-down-loaded":
        text = text.format(zone=start.get("drop_zone", "east"))
    elif t == "yard-route":
        if start.get("route", "to the ramp") == "to the ramp":
            text = text.format(route="from the dock apron to the foot of the yard ramp", what="the ramp")
        else:
            text = text.format(route="from the foot of the yard ramp to the dock apron",
                               what="the building's drive-in door")
    elif t == "load-bales":
        text = text.format(door=start["target"]["door"][4:])
    elif t == "unload-trailer":
        tg = start["target"]
        text = text.format(side={"L": "left", "R": "right"}[tg["pallet"][-1]], door=tg["door"][4:], lane=tg["lane"][4:])
    return text
