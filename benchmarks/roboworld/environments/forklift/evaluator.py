"""Task evaluation from physical state and per-tick latches only. Never the robot's path or
commands. Returns a gar.evaluation.v1 dict: success, score (fraction of criteria met), and every criterion's measured
value, unit, threshold and source. Thresholds are authored unless a rule or standard is named.
"""

import math

import numpy as np

from . import params as P
from .events import STRIKE_N, STRIKE_NS          # 2,000 N, 50 N s: the strike force and impulse (the latches use them too)
from .mathx import quat_to_R
from .task import TIME_LIMIT_S, task_verifier

IMPACT_SPEED = 0.10
STRIKE_CLASSES = {"rack", "column", "wall", "conveyor", "conveyor_guide", "conveyor_rollers", "machine", "structure",
                  "trailer_wall", "trailer_floor", "bumper", "curb", "parked",
                  "pit",                                                 # pit: a leveler pit's floor is reached only by falling
                  "dressing"}                                            # facility.dressing_colliders: every drawn object in reach
DRIVEN_ON = {"trailer_floor"}                 # the truck drives on it: only loads are judged against it
# A load set down on rack beams (put-away, and the stored loads resting on theirs) is in contact with a watched
# structure for as long as it stays there, so its contact episode always exceeds the strike impulse and its per-contact
# peak can exceed the strike force by its weight alone. Such an episode is a set-down, not a strike, only if (a) the
# part the load bears on at the episode's peak is a beam (its top: what a pallet is set on), (b) the load came onto it at
# most SET_DOWN_SPEED (the load's speed at the episode's first contact), and (c) the peak contact force stays within
# SET_DOWN_PEAK_N. Anything else (a load driven into an upright, a guard or a stored load, dropped onto the beams, or
# pushed across them by the truck) is judged like any strike.
SET_DOWN_SPEED = 0.10                          # m/s: IMPACT_SPEED, the speed below which a load's contact is not an impact
# Measured (put-away seed 0, an 865 kg unit set on 3.20 m beams by the assisted autopilot, contact sweeps every 2 ms):
# lowered onto the beams at 0.045 m/s the largest single contact force was 12.9 kN (16.9-21.4 kN in all, 2.0-2.5 x the
# weight; resting: 2.1-3.6 kN per contact, 8.5 kN in all); at 0.29 m/s 53.2 kN (97 kN in all); the first plan's full-rate
# lowering met the beams at 0.43 m/s with 56.2 kN (109 kN). Interpolated linearly in the impact speed to SET_DOWN_SPEED
# (21.9 kN) and scaled by sqrt(mass) (a stiff contact: F ~ v sqrt(k m)) to the heaviest put-away unit, 1,020 kg: 23.8 kN.
SET_DOWN_PEAK_N = 25000.0
# Moving set-down surfaces: a load comes onto a trailer floor (load-trailer, ramp-up,
# a bale's dunnage strips) or the conveyor's rollers (feed-conveyor) and stays on it, riding the trailer's suspension or
# the rollers. They are watched like the rack beams: a load's episode on one is a set-down, not a strike, only if the load came onto it at most SET_DOWN_SPEED
# along the contact normal, relative to the surface (onset_normal: a load riding the rollers at 0.2 m/s, or a floor
# bouncing on its suspension, is not approaching it), and its peak per-contact force stays within the surface's bound.
# The force bound is the most force a careful set-down can make in this world, from its contact law: a load of mass m
# that meets the support at SET_DOWN_SPEED is arrested by MuJoCo's soft contact, whose reference acceleration is
# -B v - K r with B = 2 / (d_max timeconst) (P.CONTACT_TIMECONST 4 ms, d_max = P.CONTACT_SOLIMP[1]), so its first
# contact force is m B v, and it then rests with m g: F_max = m (g + B SET_DOWN_SPEED), the whole load's momentum and
# weight on a single contact (an upper bound for any one contact). With m the heaviest load in the episode's scene
# (what rests on a dunnage strip is a bale): B v = 50.5 m/s^2, so 60.3 N per kg; for the heaviest load each surface
# takes, ramp-up's 1,300 kg unit on its pallet (1,320 kg) on a trailer floor 79.6 kN and the 2,020 kg coil pallet on the
# rollers 121.8 kN.
# The per-contact peaks scatter widely at one speed (measured: the coil peaked at 32.4 kN on one roller at 0.059 m/s,
# and at 29-64 kN at 0.10-0.17 m/s in E10), because which board meets which roller first decides how the load
# shares out. The speed rule is the set-down rule; the force rule catches what speed alone does not (a load pressed or
# dropped onto the support by something else), at the force the contact law allows a careful set-down.
SET_DOWN_DECEL = 2.0 * SET_DOWN_SPEED / (P.CONTACT_SOLIMP[1] * P.CONTACT_TIMECONST)    # m/s^2: B v at the speed rule


def surface_bound(world):
    """F_max (N) for a set-down on a trailer floor or the rollers in this scene: the heaviest load's mass x (g + B v)."""
    loads = getattr(getattr(world, "scene", None), "loads", None) or {}
    m = max((float(info.get("mass", 0.0)) for info in loads.values()), default=0.0)
    return m * (P.G + SET_DOWN_DECEL) if m > 0 else None


TRAILER_FLOOR_PEAK_N = 1320.0 * (P.G + SET_DOWN_DECEL)    # the heaviest load set on a trailer floor (ramp-up's 1,320 kg)
ROLLER_PEAK_N = 2020.0 * (P.G + SET_DOWN_DECEL)           # the coil pallet on the rollers
SET_DOWN_SURFACES = {"trailer_floor": TRAILER_FLOOR_PEAK_N, "conveyor_rollers": ROLLER_PEAK_N}
LIMIT_KMH = P.POSTED_SPEED * 1.1 * 3.6         # posted 8 km/h (5 mph) + 10 % (the 10 % authored), inside and in the yard
YARD_LIMIT_KMH = LIMIT_KMH
TRUCK_HALF = dict(x=(-2.14, 1.62), y=0.66)         # truck footprint in its own frame (counterweight to fork tips)
# Closest a carried 40 in (1,016 mm) pallet can come to a trailer wall with this truck: at full side-shift (100 mm) its
# edge is 0.608 m from the truck's centreline, the asset's rear tyres 0.682 m (measured; the Forklift C is 1.36 m wide
# across them against the H2.5A's 1.16 m), 74 mm further out; with 50 mm between tyre and wall the pallet stands
# 124 mm from it. The first criterion (50 mm, authored before the width was checked) no operator of this truck could meet.
WALL_GAP_MAX = 0.125


def C(name, value, ok, unit, threshold, source):
    v = value
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    return dict(name=name, measured=v, passed=bool(ok), unit=unit, threshold=threshold, source=source)


# ------------------------------------------------------------------------------------------------ geometry helpers
def body_frame(world, body):
    q = world.state.body_q.numpy()[body].astype(float)
    return q[:3], quat_to_R(q[None, 3:7])[0]


def pallet_corners(world, name, z=0.0):
    info = world.scene.loads[name]
    o, R = body_frame(world, info["body"])
    g = P.GMA_PALLET
    pts = [o + R @ np.array([sx * g["L"] / 2, sy * g["W"] / 2, z]) for sx in (-1, 1) for sy in (-1, 1)]
    return np.array(pts), o, R


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def ang_diff(a, b):
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


def inside_rect(p, xr, yr, margin=0.0):
    return xr[0] + margin <= p[0] <= xr[1] - margin and yr[0] + margin <= p[1] <= yr[1] - margin


def blades(world):
    """Fork blades in world: for each fork (y sign) the tip point, the heel point, the bottom height and the blade axis."""
    T = world.T
    o, R = body_frame(world, T.bodies["sideshift"])
    a = np.asarray(T.anchors["sideshift"]) - np.asarray(T.shifts.get("sideshift", (0, 0, 0)))
    out = []
    for yc in (0.293, -0.293):
        tip = o + R @ (np.array([1.601, yc, 0.104]) - a)
        heel = o + R @ (np.array([0.55, yc, 0.104]) - a)
        out.append(dict(tip=tip, heel=heel, y=yc))
    axis = R[:, 0]
    return out, axis


def footprint(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for fx in TRUCK_HALF["x"]:
        for fy in (-TRUCK_HALF["y"], TRUCK_HALF["y"]):
            pts.append((x + c * fx - s * fy, y + s * fx + c * fy))
    return pts


# ------------------------------------------------------------------------------------------------ shared criteria
def episode_load(mover, e):
    """The load a load episode belongs to: the name in its key when the latches key loads by name ('load:<name>'),
    else the load whose part carried the episode's peak, else the load that opened it."""
    if mover.startswith("load:"):
        return mover[5:]
    parts = e.get("parts") or []
    if parts and ":" in str(parts[0]):
        return str(parts[0]).split(":")[0]
    mv = e.get("mover") or ""
    return mv[5:] if mv.startswith("load:") else None


def on_beam(e):
    """The structure part at the episode's peak is a rack beam (facility: '<row><frame>_beam<bay-1>_<level>_<x>')."""
    parts = e.get("parts") or []
    return len(parts) > 1 and "_beam" in str(parts[1])


def gentle_set_down(e):
    """A load set down on rack beams, or resting on them since the start: see SET_DOWN_SPEED and SET_DOWN_PEAK_N."""
    return on_beam(e) and e["onset_speed"] <= SET_DOWN_SPEED and e["peak_N"] <= SET_DOWN_PEAK_N


def onset_normal(e):
    """The episode's approach speed along the contact normal, relative to the surface (events.Events._normal_speed);
    an episode latched without it reads its onset speed."""
    v = e.get("onset_normal")
    return float(e["onset_speed"] if v is None else v)


def gentle_on_surface(cls, e, world=None):
    """A load set down on (or resting on, or riding) a moving set-down surface: see SET_DOWN_DECEL. The bound is the
    scene's (surface_bound) when the scene has loads, else the surface's heaviest task load (SET_DOWN_SURFACES)."""
    bound = (surface_bound(world) if world is not None else None) or SET_DOWN_SURFACES[cls]
    return onset_normal(e) <= SET_DOWN_SPEED and e["peak_N"] <= bound


def lean_limit(world, name):
    """The largest contact force a load at rest can keep on a structure by itself: its weight (a preload held by its own
    friction on its support is at most mu x weight); more means something drives it. Never below the strike force."""
    info = getattr(world.scene, "loads", {}).get(name) or {}
    return max(STRIKE_N, float(info.get("mass", 0.0)) * P.G)


def load_exempt(world, mover, cls, e):
    """A load's contact that is not a strike: set gently on (resting on, riding) a trailer floor or the conveyor's
    rollers, set gently on rack beams, or resting or leaning against a structure (slower than an impact at its onset,
    never pressed harder than its own weight, e.g. a bale set against a trailer's nose wall as the work order asks)."""
    if cls in SET_DOWN_SURFACES:
        return gentle_on_surface(cls, e, world)
    if gentle_set_down(e):
        return True
    return (not on_beam(e) and e["onset_speed"] < IMPACT_SPEED
            and e["peak_N"] <= lean_limit(world, episode_load(mover, e)))


def shared(world, start, dest_floor=None, strike_rule=None):
    """The shared site rules. strike_rule(world, mover, cls, e), if given, decides a task's own contact episodes: True
    (a strike), False (not one) or None (the shared rule decides)."""
    ev = world.events
    out = []
    strikes = []
    for mover, cls, e in ev.episodes():
        if cls not in STRIKE_CLASSES:
            continue
        if cls in DRIVEN_ON and not mover.startswith("load"):
            continue
        own = strike_rule(world, mover, cls, e) if strike_rule is not None else None
        if own is False or (own is None and mover.startswith("load") and load_exempt(world, mover, cls, e)):
            continue
        if own or e["peak_N"] > STRIKE_N or e["impulse_Ns"] > STRIKE_NS:
            strikes.append(dict(mover=mover, structure=cls, **{k: v for k, v in e.items() if k != "mover"},
                                load=episode_load(mover, e) if mover.startswith("load") else None))
    out.append(C("no_strike", len(strikes), not strikes, "strikes", f"none with > {STRIKE_N:.0f} N or > {STRIKE_NS:.0f} N s "
                 f"(a load's contact slower than {IMPACT_SPEED:.2f} m/s at onset counts only above its weight; a load set "
                 f"on rack beams only above {SET_DOWN_SPEED:.2f} m/s or {SET_DOWN_PEAK_N / 1000:.0f} kN, on a trailer floor "
                 f"or the conveyor's rollers only above {SET_DOWN_SPEED:.2f} m/s or the heaviest load's weight plus its "
                 f"arrest at that speed by the contact law ({(surface_bound(world) or 0) / 1000:.0f} kN here))",
                 "authored; beam bound measured (evaluator.SET_DOWN_PEAK_N); trailer-floor and roller bound "
                 "derived from the contact law (evaluator.SET_DOWN_DECEL)"))
    lift = max(ev.summary()["wheel_lift_max_s"].values() or [0.0])
    out.append(C("no_wheel_lift", lift, lift <= 0.3, "s", "<= 0.3 s off the ground", "authored"))
    out.append(C("roll_under_6deg", ev.max_roll, ev.max_roll < 6.0 and ev.overturn is None, "deg", "< 6 deg, no overturn",
                 "authored"))
    dropped = []
    for name, lst in ev.load_floor.items():
        for t, xy in lst:
            if dest_floor is not None and dest_floor(name, xy):
                continue
            dropped.append((name, t, xy))
    out.append(C("no_dropped_load", len(dropped), not dropped, "floor contacts", "load touches the floor only at its "
                 "destination", "authored"))
    if world.scene.c.preset in ("warehouse", "dock"):
        out.append(C("speed_limit", ev.max_speed_inside * 3.6, ev.over_limit_s == 0.0, "km/h", f"<= {LIMIT_KMH:.1f} km/h inside",
                     "posted 8 km/h + 10 % ('usually 5 mph', OSHAcademy, read)"))
    else:
        v = ev.max_speed * 3.6
        out.append(C("speed_limit", v, v <= YARD_LIMIT_KMH, "km/h", f"<= {YARD_LIMIT_KMH:.1f} km/h in the yard",
                     "posted 8 km/h + 10 %: 'usually 5 mph' at the workplace (OSHAcademy 619, read), '5 mph is the safe "
                     "maximum speed limit generally accepted by the forklift industry' (Warren Forensics 2022, read); "
                     "8-10 mph for open outdoor areas without pedestrians (search, not read) does not fit a yard lane "
                     "between parked trailers"))
    out.append(C("operator_seated", ev.seat_open_max_s, ev.seat_open_max_s <= 1.0, "s", "seat switch open <= 1 s while moving",
                 "authored"))
    # the seated operator's head (any robot body) against the overhead guard (measured: a hard
    # stop met the rear slat at 3.1 kN; an ordinary 0.16 g stop brushed it at 17 N)
    hg = getattr(ev, "head_guard", None) or dict(peak_N=0.0, t=None, parts=None)
    out.append(C("head_clear_of_guard", hg["peak_N"], hg["peak_N"] <= P.HEAD_CONTACT_LIMIT, "N",
                 f"no robot contact with the overhead guard above {P.HEAD_CONTACT_LIMIT:.0f} N",
                 "ISO/TS 15066 Table A.2, skull and forehead (params.HEAD_CONTACT_LIMIT, read)"))
    limit = TIME_LIMIT_S[start["task"]]
    out.append(C("time_limit", world.sim_time, world.sim_time <= limit, "s", f"<= {limit:.0f} s", "authored (~3x expert)"))
    return out, dict(strikes=strikes, dropped=dropped,
                     head_guard=dict(peak_N=hg["peak_N"], t=hg.get("t"), parts=hg.get("parts")))


def parked(world, rest_needed=2.0):
    ev = world.events
    return ev.rest_s


# ------------------------------------------------------------------------------------------------ tasks
def evaluate(world, start):
    """The episode's evaluation: a benchmark task's criteria are its verifier.py's verify(world, start) (tasks/<task>/),
    every other task's its function below."""
    t = start["task"]
    folder = task_verifier(t)
    fn = folder.verify if folder is not None else globals()["_" + t.replace("-", "_")]
    crit, details = fn(world, start)
    passed = sum(c["passed"] for c in crit)
    # a faulted world (non-finite or unphysical state) ends the episode without success, whatever the criteria read
    success = bool(crit) and passed == len(crit) and world.fault is None
    out = dict(schema="gar.evaluation.v1", task=t, success=success, score=passed / max(len(crit), 1), criteria=crit,
               details=details, fault=world.fault, sim_time=world.sim_time)
    rw = details.get("reward")
    if rw is not None:      # a task with its own reward (load-bales): its progress, not the fraction of criteria met
        # a faulted world keeps no progress
        score = 1.0 if success else 0.0 if world.fault is not None else rw["fail_scale"] * rw["progress"] * rw["workmanship"]
        out.update(score=score, progress=rw["progress"], score_rule=rw["rule"])
    return out


def _forks_low(world, limit):
    bl, _ = blades(world)
    h = max(b["tip"][2] for b in bl) - world.scene.ground_height(*bl[0]["tip"][:2])
    return h, h <= limit


def _set_down_in_bay(world, start, extra_cartons=False):
    bay = world.scene.zones["bays"][start["bay"]]

    def dest(name, xy):
        return inside_rect(xy, (bay["x"][0] - 0.05, bay["x"][1] + 0.05), (bay["y"][0] - 0.05, bay["y"][1] + 0.05)) and \
            name == "carried"
    crit, det = shared(world, start, dest_floor=dest)
    corners, o, R = pallet_corners(world, "carried")
    inside = all(inside_rect(c, bay["x"], bay["y"]) for c in corners)
    crit.append(C("inside_bay_lines", [list(np.round(c[:2], 3)) for c in corners], inside, "m",
                  f"all four corners inside {start['bay']}", "authored (90 mm margins)"))
    yerr = abs(math.degrees(ang_diff(yaw_of(R), math.pi / 2)))
    yerr = min(yerr, 180.0 - yerr)
    crit.append(C("square_to_lines", yerr, yerr <= 5.0, "deg", "pallet yaw within 5 deg of the bay lines", "authored"))
    fz = [c[2] - world.scene.ground_height(*c[:2]) for c in corners]
    crit.append(C("resting_on_floor", fz, all(abs(z) <= 0.010 for z in fz), "m", "corners within 10 mm of the floor",
                  "authored"))
    p, Rc = world.body_pose("chassis")
    bl, _ = blades(world)
    gap = min(float(np.linalg.norm(b["tip"][:2] - o[:2])) for b in bl) - P.GMA_PALLET["L"] / 2
    crit.append(C("backed_out", gap, gap >= 0.5, "m", "fork tips >= 0.5 m from the pallet", "authored"))
    h, ok = _forks_low(world, 0.15)
    crit.append(C("forks_lowered", h, ok, "m", "fork tips at or below travel height (0.15 m)", "authored"))
    crit.append(C("stationary", world.events.rest_s, world.events.rest_s >= 1.0, "s", ">= 1 s at rest", "authored"))
    if extra_cartons:
        moved = []
        for name, info in world.scene.loads.items():
            if not name.startswith("carried_c"):
                continue
            pc, Rcar = body_frame(world, info["body"])
            rel = R.T @ (pc - o)
            moved.append(rel)
        rel0 = world.scene.carton_rel0 if hasattr(world.scene, "carton_rel0") else None
        if rel0 is not None:
            d = [float(np.linalg.norm(a - b)) for a, b in zip(moved, rel0)]
            crit.append(C("cartons_in_place", max(d), max(d) <= 0.030, "m", "no carton moved > 30 mm on the pallet",
                          "authored"))
        touched = [n for n in world.events.load_floor if n.startswith("carried_c")]
        crit.append(C("no_carton_on_floor", touched, not touched, "-", "no carton ever touches the floor", "authored"))
    return crit, {}


def _carry_loose_stack(world, start):
    return _set_down_in_bay(world, start, extra_cartons=True)


def _feed_conveyor(world, start):
    crit, det = shared(world, start)
    conv = world.scene.zones["conveyor"]
    corners, o, R = pallet_corners(world, "carried")
    inside = all(inside_rect(c, conv["x"], conv["y"]) for c in corners)
    cx, cy = [float(c[0]) for c in corners], [float(c[1]) for c in corners]
    crit.append(C("on_conveyor", dict(inside=inside, x=(min(cx), max(cx)), y=(min(cy), max(cy)),
                                      frame_x=tuple(conv["x"]), frame_y=tuple(conv["y"])), inside, "-",
                  "pallet inside the conveyor frame", "authored"))
    yerr = abs(math.degrees(ang_diff(yaw_of(R), 0.0)))
    yerr = min(yerr, 180 - yerr)
    crit.append(C("square_to_rollers", yerr, yerr <= 3.0, "deg", "stringers within 3 deg of the conveyor axis", "authored"))
    ends = getattr(world, "conveyor", None)
    ev = ends.end_events if ends else []
    crit.append(C("reached_end_stop", len(ev), len(ev) >= 1, "events", "end photo-eye registered the load", "authored"))
    bl, _ = blades(world)
    out_ = min(float(np.linalg.norm(b["tip"][:2] - o[:2])) for b in bl)
    crit.append(C("forks_withdrawn", out_, out_ > 1.0, "m", "fork tips clear of the pallet", "authored"))
    # the work order's last step, "lower the forks": as after a floor-bay set-down
    h, ok = _forks_low(world, 0.15)
    crit.append(C("forks_lowered", h, ok, "m", "fork tips at or below travel height (0.15 m)", "authored (as the floor bays)"))
    return crit, det


def _trailer_frame(world, name):
    tr = world.scene.trailers[name]
    o, R = body_frame(world, tr["body"])
    return tr, o, R


def _in_trailer_slot(world, trailer, xy, x_range, y_range):
    """Whether a world point (x, y) lies, in the trailer's own frame (x from the rear sill, y left), inside the given
    ranges (its floor contact positions are recorded in world coordinates; the trailer moves millimetres)."""
    tr, to, TR = _trailer_frame(world, trailer)
    rel = TR.T @ (np.array([xy[0], xy[1], to[2]]) - to)
    return x_range[0] <= rel[0] <= x_range[1] and y_range[0] <= rel[1] <= y_range[1]


SLOT_MARGIN = 0.30      # m: a load's floor contact counts as at its destination within this of the target footprint


def _load_trailer(world, start):
    """A pallet loaded into a secured trailer: load-trailer-chocked, and load-trailer (its verifier.py judges an
    unsecured trailer itself)."""
    door = start["target"]["door"]
    g = P.GMA_PALLET

    def dest(name, xy):
        # the trailer floor is a floor (events): the carried pallet touches it only in its slot, behind the nose pallets
        # on its side (the footprint of a pallet set against them and the wall, SLOT_MARGIN all round)
        if name != "carried":
            return False
        tr, to, TR = _trailer_frame(world, door)
        ac, _, _ = pallet_corners(world, start["target"]["behind"])
        x_ahead = float(min((TR.T @ (c - to))[0] for c in ac))
        side = 1.0 if start["target"].get("side", "L") == "L" else -1.0
        ys = sorted((side * (tr["W"] / 2), side * (tr["W"] / 2 - WALL_GAP_MAX - g["W"])))
        return _in_trailer_slot(world, door, xy, (x_ahead - g["L"] - SLOT_MARGIN, x_ahead + 0.05),
                                (ys[0] - SLOT_MARGIN, ys[1] + SLOT_MARGIN))
    crit, det = shared(world, start, dest_floor=dest)
    tr, to, TR = _trailer_frame(world, start["target"]["door"])
    corners, o, R = pallet_corners(world, "carried")
    loc = np.array([TR.T @ (c - to) for c in corners])          # trailer frame
    W, L = tr["W"], tr["L"]
    inside = bool(np.all((loc[:, 0] > 0) & (loc[:, 0] < L) & (np.abs(loc[:, 1]) < W / 2)))
    on_floor = bool(np.all(np.abs(loc[:, 2]) < 0.015))
    crit.append(C("in_trailer_on_floor", dict(inside=inside, on_floor=on_floor), inside and on_floor, "-",
                  "inside the trailer, resting on its floor", "authored"))
    ahead = world.scene.loads[start["target"]["behind"]]
    ac, ao, AR = pallet_corners(world, start["target"]["behind"])
    a_loc = np.array([TR.T @ (c - to) for c in ac])
    gap = float(a_loc[:, 0].min() - loc[:, 0].max())
    crit.append(C("gap_to_pallets_ahead", gap, -0.005 <= gap <= 0.050, "m", "<= 50 mm behind the pallets at the nose",
                  "authored"))
    # the side as seen facing the nose from the dock: left is the trailer's +y
    side = start["target"].get("side", "L")
    wall_gap = float(W / 2 - loc[:, 1].max()) if side == "L" else float(W / 2 + loc[:, 1].min())
    crit.append(C("gap_to_side_wall", wall_gap, 0.0 <= wall_gap <= WALL_GAP_MAX, "m",
                  f"<= {WALL_GAP_MAX * 1000:.0f} mm to the {'left' if side == 'L' else 'right'} wall",
                  "derived (truck width at full side-shift + 50 mm)"))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, (TR[:, 2] @ R[:, 2])))))
    crit.append(C("level", tilt, tilt <= 2.0, "deg", "within 2 deg of the trailer floor", "authored"))
    p, _ = world.body_pose("chassis")
    on_dock = p[1] < world.scene.zones["doors"][start["target"]["door"]]["face_y"] - 1.0
    crit.append(C("truck_back_on_dock", float(p[1]), bool(on_dock), "m", "chassis back inside the building", "authored"))
    mv = world.events.trailer_max_move.get(start["target"]["door"], 0.0)
    crit.append(C("trailer_movement", mv, mv <= 0.025, "m", "<= 25 mm from its spotted position", "authored"))
    return crit, det


def _load_trailer_chocked(world, start):
    return _load_trailer(world, start)


# load-bales reward. The criteria average would score doing nothing 0.909 (10 of 11 rules hold untouched), one bale placed or both placed with the truck still in the
# trailer the same, and entering the trailer, which the job needs, 0.818. So: score = 1 on success, 0 for a faulted
# world, else FAIL_SCALE x progress x workmanship, so every failure stays below every success and an untouched scene
# scores 0. progress = (1 - EXIT_SHARE) P + EXIT_SHARE E g(P): P the mean placement progress of the two best bales (any
# two, either side, any order or route), E the whole truck out of the trailer, g(P) = clip((P - 0.9) / 0.1, 0, 1). With
# a bale still on the forks P is at most 0.9 (that bale's carry credit is at most CARRY_CAP), so entering the trailer
# with a bale never lowers the score, a truck that never left the dock earns nothing for it, and a near miss with the
# truck out still earns its share of the exit.
# A bale's placement progress reads its current state only:
#   0 if it rests on a floor now (on a bare floor straight forks cannot get under it again) or lies over (> CARRY_TILT_DEG);
#   1 if placed (two_bales_at_nose's rule for one bale);
#   resting on strips in the trailer otherwise, 1 - (1 - CARRY_CAP) min(1, (gap - 75 mm) / SHORT_DECAY_M), gap its
#   farther front edge's (the same measure as placed), times max(0, 1 - (tilt - 5 deg) / (CARRY_TILT_DEG - 5 deg)) above
#   5 deg;
#   otherwise CARRY_CAP (d0 - d - b) / (d0 - b), d its centre's plan distance to the trailer's nose row (from outside, to
#   the rear opening and on; from inside the other trailer, out through its rear sill first), d0 the same from its start
#   centre and b = CARRY_DEADBAND_M (a bale settling or resting where it stands earns nothing: seed 900's settled start
#   reads 8e-7 without it).
# A floor contact is charged once, in workmanship, per floor-contact event (latched entries less than FLOOR_MERGE_S
# apart are one continuous contact: entries are 0.25 s slices of one contact), and graded by how the bale met the floor,
# the same on every floor: a contact it came off again 0.8 + 0.1 / (1 + its duration / 1 s),
# between 0.9 for a brush and 0.8; a bale left lying on a floor (or over) STRAND_FACTOR 0.8, its loss being in its
# progress already; each times the approach-speed factor g(v) = 1 up to SET_DOWN_SPEED, else max(0.3, 1 / (1 + log10
# (v / SET_DOWN_SPEED))). A bale's arrival on the trailer floor, also latched as a trailer_floor strike, is graded here
# once, not again as a strike. So: one brush then placed > a longer scrape then placed > a bale stranded gently > a bale
# dropped, its cost rising with speed. A contact latched without an approach speed reads as gentle.
# Workmanship is the product of a factor per broken site rule, graded by how far it was broken (1 when it holds: a rule
# kept earns nothing by itself). Strikes are graded per contact event, episodes of one mover on one structure class less
# than STRIKE_MERGE_S apart merged (the 2 ms contact sweep splits one contact into many), by the approach
# speed's kinetic-energy ratio (v / 0.10 m/s)^2 when it approached at 0.10 m/s or more, else by its peak force over 2 kN
# (a press), never by the impulse a resting contact accumulates (peak forces scatter widely at one speed and put every
# collision in one narrow band). The time limit stays a success condition only. Every number
# here is authored. The reward is computed in tasks/load-bales/verifier.py (_load_bales_reward); the workmanship of
# forklift-tasks-v1 (task_progress.py) grades the other tasks with the same numbers and the two event functions below.
BALE_REWARD = dict(contract="load-bales-reward-v1", fail_scale=0.9, exit_share=0.1, exit_gate_from=0.9,
                   carry_cap=0.8, short_decay_m=0.5, carry_tilt_deg=20.0, carry_deadband_m=0.02, strike_merge_s=0.25,
                   floor_merge_s=0.3, strand_factor=0.8, brush_factor=0.9, brush_decay_s=1.0, speed_floor=0.3)


def _strike_events(strikes, merge_s):
    """Strike episodes as contact events: per (mover, structure class), episodes less than merge_s apart are one event.
    An event's severity is its worst episode's (approach speed / IMPACT_SPEED)^2, or for an episode that approached
    slower (a press) max(1, peak / STRIKE_N)."""
    out = []
    for s in sorted(strikes, key=lambda s: (s["mover"], s["structure"], s.get("t") or 0.0)):
        v = onset_normal(s)
        sev = (v / IMPACT_SPEED) ** 2 if v >= IMPACT_SPEED else max(1.0, s["peak_N"] / STRIKE_N)
        t0 = float(s.get("t") or 0.0)
        t1 = math.inf if s.get("open") or s.get("t_end") is None else float(s["t_end"])
        last = out[-1] if out else None
        if last and (last["mover"], last["structure"]) == (s["mover"], s["structure"]) and t0 - last["t_end"] < merge_s:
            last.update(severity=max(last["severity"], sev), t_end=max(last["t_end"], t1), episodes=last["episodes"] + 1)
        else:
            out.append(dict(mover=s["mover"], structure=s["structure"], t=t0, t_end=t1, severity=sev, episodes=1))
    return out


def _floor_events(world, det, merge_s):
    """Each bale's latched floor contacts as contact events: entries less than merge_s apart are one continuous contact
    (events.py appends one every 0.25 s while it lasts). An event's approach speed is the largest of the onsets latched
    in it (events.load_floor_onset) and of the bale's trailer_floor strike episodes that began in it, which are folded
    in here and returned so that they are not charged again as strikes."""
    onsets = getattr(world.events, "load_floor_onset", None) or {}
    out, folded = {}, set()
    for name, t, _ in sorted(det["dropped"], key=lambda r: (r[0], r[1])):
        evs = out.setdefault(name, [])
        if evs and t - evs[-1]["t1"] < merge_s:
            evs[-1]["t1"] = float(t)
        else:
            evs.append(dict(t0=float(t), t1=float(t), v=0.0))
    for name, evs in out.items():
        for e in evs:
            e["v"] = max([float(v) for t, v in onsets.get(name, []) if e["t0"] - merge_s <= t <= e["t1"] + merge_s] or [0.0])
        for i, s in enumerate(det["strikes"]):
            t = float(s.get("t") or 0.0)
            hit = [e for e in evs if s["structure"] == "trailer_floor" and s["mover"] == "load:" + name
                   and e["t0"] - merge_s <= t <= e["t1"] + merge_s]
            if hit:
                hit[0]["v"] = max(hit[0]["v"], onset_normal(s))
                folded.add(i)
    return out, folded


def _unload_trailer(world, start):
    lane = world.scene.zones["staging"][start["target"]["lane"]]
    name = start["target"]["pallet"]

    def dest(n, xy):
        return n == name and inside_rect(xy, lane["x"], lane["y"])
    crit, det = shared(world, start, dest_floor=dest)
    corners, o, R = pallet_corners(world, name)
    inside = all(inside_rect(c, lane["x"], lane["y"]) for c in corners)
    fz = [c[2] - world.scene.ground_height(*c[:2]) for c in corners]
    crit.append(C("in_staging_lane", inside, inside and all(abs(z) < 0.015 for z in fz), "-",
                  f"inside staging lane {start['target']['lane'][4:]}, on the floor", "authored"))
    other = "trailer_1L" if name.endswith("R") else "trailer_1R"
    oo, _ = body_frame(world, world.scene.loads[other]["body"])
    d = float(np.linalg.norm(oo - np.asarray(world.scene.load_start_pose[other])))
    crit.append(C("other_pallet_undisturbed", d, d <= 0.030, "m", "<= 30 mm", "authored"))
    # set down as in the floor bays: the forks out of the pallet and lowered, the truck at rest
    bl, _ = blades(world)
    gap = min(float(np.linalg.norm(b["tip"][:2] - o[:2])) for b in bl) - P.GMA_PALLET["L"] / 2
    crit.append(C("backed_out", gap, gap >= 0.5, "m", "fork tips >= 0.5 m from the pallet", "authored (as the floor bays)"))
    h, ok = _forks_low(world, 0.15)
    crit.append(C("forks_lowered", h, ok, "m", "fork tips at or below travel height (0.15 m)", "authored (as the floor bays)"))
    crit.append(C("stationary", world.events.rest_s, world.events.rest_s >= 1.0, "s", ">= 1 s at rest", "authored"))
    return crit, det


def _yard_trailer_nose(world):
    tr, to, TR = _trailer_frame(world, "yard_trailer")
    corners, o, R = pallet_corners(world, "carried")
    loc = np.array([TR.T @ (c - to) for c in corners])
    return tr, loc, R, TR


def _ramp_up_loaded(world, start):
    g = P.GMA_PALLET
    side = (start.get("target") or {}).get("side")

    def dest(name, xy):
        # the trailer floor is a floor (events): the pallet touches it only at the nose (on its side, if one is named)
        if name != "carried":
            return False
        tr = world.scene.trailers["yard_trailer"]
        if side is None:
            ys = (-tr["W"] / 2, tr["W"] / 2)
        else:
            sg = 1.0 if side == "L" else -1.0
            ys = sorted((sg * tr["W"] / 2, sg * (tr["W"] / 2 - WALL_GAP_MAX - g["W"])))
        return _in_trailer_slot(world, "yard_trailer", xy, (tr["L"] - 0.075 - g["L"] - SLOT_MARGIN, tr["L"]),
                                (ys[0] - SLOT_MARGIN, ys[1] + SLOT_MARGIN))
    crit, det = shared(world, start, dest_floor=dest)
    tr, loc, R, TR = _yard_trailer_nose(world)
    gap = float(tr["L"] - loc[:, 0].max())
    on_floor = bool(np.all(np.abs(loc[:, 2]) < 0.015))
    crit.append(C("pallet_at_nose", gap, 0.0 <= gap <= 0.075 and on_floor, "m", "<= 75 mm from the front wall, on the floor",
                  "authored"))
    if side is not None:
        # the side named in the work order, against the wall, as in load-trailer (two pallets across a 102 in van)
        W = tr["W"]
        wall_gap = float(W / 2 - loc[:, 1].max()) if side == "L" else float(W / 2 + loc[:, 1].min())
        crit.append(C("gap_to_side_wall", wall_gap, 0.0 <= wall_gap <= WALL_GAP_MAX, "m",
                      f"<= {WALL_GAP_MAX * 1000:.0f} mm to the {'left' if side == 'L' else 'right'} wall",
                      "derived (truck width at full side-shift + 50 mm, as load-trailer)"))
    rb = world.events.rollback
    crit.append(C("no_rollback", rb, rb <= 0.3, "m", "rolled back <= 0.3 m at a hill start", "authored"))
    return crit, det


def _yard_route(world, start):
    crit, det = shared(world, start)
    lane = world.scene.zones["yard_lane"]
    pts = np.array(lane["points"]); half = lane["width"] / 2
    traj = np.array(world.events.traj) if world.events.traj else np.zeros((0, 5))
    worst = 0.0
    for t, x, y, yaw, v in traj[::5]:
        for px, py in footprint(x, y, yaw):
            dmin = min(_seg_dist(np.array([px, py]), pts[k], pts[k + 1]) for k in range(len(pts) - 1))
            worst = max(worst, dmin - half)
    crit.append(C("inside_lane", worst, worst <= 0.0, "m", "footprint never crosses the lane's outer lines", "authored"))
    p, Rc = world.body_pose("chassis")
    tips, axis = blades(world)
    if start.get("route", "to the ramp") == "to the ramp":
        rz = world.scene.zones["ramp"]
        err = float(min(t_["tip"][1] for t_ in tips) - (rz["y_foot"] + 0.3))
        yerr = abs(math.degrees(ang_diff(yaw_of(Rc), -math.pi / 2)))
        name, what = "stopped_at_ramp_foot", "the foot line"
    else:
        # back to the building: the fork tips at the yellow line across the lane's head, square to the building
        err = float(max(t_["tip"][1] for t_ in tips) - world.scene.zones["apron_line"]["y"])
        yerr = abs(math.degrees(ang_diff(yaw_of(Rc), math.pi / 2)))
        name, what = "stopped_at_apron_line", "the apron line at the lane's head"
    crit.append(C(name, dict(line_error_m=err, yaw_error_deg=yerr, rest_s=world.events.rest_s),
                  abs(err) <= 0.3 and yerr <= 3.0 and world.events.rest_s >= 1.0, "-",
                  f"fork tips within +-0.3 m of {what}, yaw <= 3 deg, stopped", "authored"))
    return crit, det


def _seg_dist(p, a, b):
    ab = b - a
    t = max(0.0, min(1.0, float((p - a) @ ab / (ab @ ab))))
    return float(np.linalg.norm(p - (a + t * ab)))
