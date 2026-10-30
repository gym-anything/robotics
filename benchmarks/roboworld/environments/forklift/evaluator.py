"""Task evaluation from physical state and per-tick latches only. Never the robot's path or
commands. Returns a gar.evaluation.v1 dict: success, score (fraction of criteria met), and every criterion's measured
value, unit, threshold and source. Thresholds are authored unless a rule or standard is named.
"""

import math

import numpy as np

from . import params as P
from .events import STRIKE_N, STRIKE_NS          # 2,000 N, 50 N s: the strike force and impulse (the latches use them too)
from .mathx import quat_to_R
from .task import TIME_LIMIT_S

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


FORK_HALF_W = 0.051                 # half the blade width (asset blades 102 mm, measured)


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


def front_axle_line(world):
    t = world.tyres.telemetry
    xs = [np.asarray(t[n]["contact"]) for n in ("wheel_fl", "wheel_fr") if t[n]["contact"] is not None]
    if xs:
        return np.mean(xs, axis=0)
    p, R = world.body_pose("chassis")
    return p


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
    t = start["task"]
    fn = globals()["_" + t.replace("-", "_")]
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


def _start_and_park(world, start):
    crit, det = shared(world, start)
    bay = world.scene.zones["parking"]
    p, R = world.body_pose("chassis")
    fp = footprint(float(p[0]), float(p[1]), yaw_of(R))
    worst = max(max(bay["x"][0] - x, x - bay["x"][1], bay["y"][0] - y, y - bay["y"][1]) for x, y in fp)
    crit.append(C("in_parking_bay", dict(outside_by_m=worst, footprint=[[round(x, 3), round(y, 3)] for x, y in fp]),
                  worst <= 0.0, "m", "the truck's footprint (counterweight to fork tips) inside the parking bay's lines",
                  "authored"))
    sig = world.signals
    rest = world.events.rest_s
    crit.append(C("stationary_2s", rest, rest >= 2.0, "s", ">= 2 s at rest", "authored"))
    crit.append(C("parking_brake_set", sig.get("park", 0.0), sig.get("park", 0.0) > 0.5, "-", "applied",
                  "OSHA 1910.178(m)(5)(i): brakes set (read)"))
    crit.append(C("direction_neutral", sig.get("direction"), sig.get("direction") == 0, "-", "neutral",
                  "OSHA 1910.178(m)(5)(i): controls neutral (read)"))
    crit.append(C("engine_off", world.drivetrain.running, not world.drivetrain.running, "-", "off",
                  "OSHA 1910.178(m)(5)(i): power shut off (read)"))
    crit.append(C("key_off", sig.get("key"), sig.get("key") == "off", "-", "key in the OFF position",
                  "Adaptalift 'How to park a forklift safely': turn the key or keyless switch to the off position (read)"))
    h, ok = _forks_low(world, 0.020)
    crit.append(C("forks_lowered", h, ok, "m", "fork tips <= 20 mm above the floor",
                  "OSHA 1910.178(m)(5)(i): load engaging means fully lowered (read)"))
    bl, axis = blades(world)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(axis[2])))))
    tips = [float(b["tip"][2] - world.scene.ground_height(*b["tip"][:2])) for b in bl]
    # at the carriage's lower stop the blades lie almost flat: tilting the mast forward puts the tips on the floor and
    # stalls there (blade pitch -0.07 deg with the tips touching, seed-0 demo); tilted back, the tips stand off it
    crit.append(C("forks_tilted_to_floor", dict(blade_pitch_deg=pitch, tip_heights_m=tips),
                  pitch <= 0.0 and max(tips) <= 0.005, "-", "both fork tips <= 5 mm above the floor (resting on it), the "
                  "blades sloping down to them (pitch <= 0)",
                  "Adaptalift: tilt the mast forward so the tyne tips are touching the ground; Dalhousie safe work "
                  "instructions (2018): forks lowered to the floor surface and tilted slightly forward (both read)"))
    return crit, det


# the straight sections judged for the cruise speed, in the order driven: between the start and the first
# intersection's slow-down, between the intersections, and between the second intersection and the stop (the autopilot
# slows from 1.7 m before an intersection's entry edge and brakes to the far line at 0.6 m/s^2). Westward the start is
# 9 m from the first intersection (16.5 m eastward), so its first section is the 3 m in which a truck accelerating at
# 0.5 m/s^2 from the start is at speed
CRUISE_SECTIONS = {"east": ((6.0, 18.5), (25.0, 32.5), (39.0, 45.5)),
                   "west": ((40.0, 43.0), (26.0, 33.5), (10.0, 19.0))}
CRUISE_END = {"east": "W1_end", "west": "west_end"}       # facility.warehouse stop_lines


def _aisle_cruise(world, start):
    crit, det = shared(world, start)
    ev = world.events
    traj = np.array(ev.traj) if ev.traj else np.zeros((0, 5))
    way = start.get("cruise", "east")
    sections = CRUISE_SECTIONS[way]
    for k, (a, b) in enumerate(sections):
        m = (traj[:, 1] >= a) & (traj[:, 1] <= b) if len(traj) else np.zeros(0, bool)
        if m.sum() < 2:
            crit.append(C(f"cruise_section_{k + 1}", 0.0, False, "fraction", ">= 0.80 of the section at 6.5-8.0 km/h",
                          "authored"))
            continue
        x = traj[m, 1]; v = np.abs(traj[m, 4]) * 3.6
        dx = np.abs(np.diff(x, prepend=x[0]))
        inband = ((v >= 6.5) & (v <= 8.0)) * dx
        frac = float(inband.sum() / max(dx.sum(), 1e-6))
        crit.append(C(f"cruise_section_{k + 1}", frac, frac >= 0.8, "fraction", ">= 0.80 at 6.5-8.0 km/h", "authored"))
    for zn in ("I1", "I2"):
        cr = [c for c in ev.crossings if c["zone"] == zn]
        if not cr:
            crit.append(C(f"intersection_{zn}", None, False, "-", "crossed at <= 5 km/h, horn within 3 s before",
                          "OSHA 1910.178(n)(4): slow down and sound the horn (read); numbers authored"))
            continue
        c0 = cr[0]
        ok = c0["speed"] * 3.6 <= 5.0 and c0["since_horn"] is not None and c0["since_horn"] <= 3.0
        crit.append(C(f"intersection_{zn}", dict(speed_kmh=c0["speed"] * 3.6, since_horn_s=c0["since_horn"]), ok, "-",
                      "<= 5 km/h, horn within 3 s before",
                      "OSHA 1910.178(n)(4): slow down and sound the horn (read); numbers authored"))
    ycl = sum(world.scene.zones["main_aisle"]["y"]) / 2
    if len(traj):
        inside = traj[(traj[:, 1] > 4.0) & (traj[:, 1] < 46.0)]
        dev = float(np.max(np.abs(inside[:, 2] - ycl))) if len(inside) else 99.0
    else:
        dev = 99.0
    crit.append(C("aisle_centreline", dev, dev <= 0.6, "m", "truck centre within 0.6 m of the aisle centreline", "authored"))
    fa = front_axle_line(world)
    err = float(fa[0] - world.scene.zones["stop_lines"][CRUISE_END[way]]["x"])
    crit.append(C("at_far_line", err, abs(err) <= 0.2 and ev.rest_s >= 1.0, "m", "stopped, front wheels within +-0.2 m "
                  f"of the {way} end's white line", "authored"))
    return crit, det


def _drive_to_pallet(world, start):
    crit, det = shared(world, start)
    ev = world.events
    crit.append(C("stationary_2s", ev.rest_s, ev.rest_s >= 2.0, "s", ">= 2 s at rest", "authored"))
    corners, o, R = pallet_corners(world, "target")
    g = P.GMA_PALLET
    bl, axis = blades(world)
    # pallet face toward the truck: the pallet end nearest the fork tips
    px = R[:, 0]
    ends = [o + px * g["L"] / 2, o - px * g["L"] / 2]
    tipc = (bl[0]["tip"] + bl[1]["tip"]) / 2
    face = min(ends, key=lambda e: np.linalg.norm(e[:2] - tipc[:2]))
    dist = float(np.linalg.norm(face[:2] - tipc[:2]))
    crit.append(C("tips_short_of_pallet", dist, 0.20 <= dist <= 0.30, "m", "0.20-0.30 m short of the stringer ends",
                  "OSHA eTool: stop 8-12 in short (read)"))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, axis[2]))))
    crit.append(C("blades_level", pitch, abs(pitch) <= 1.0, "deg", "within +-1 deg", "derived (17 mm over the blade)"))
    zb = [b["tip"][2] - world.scene.ground_height(*b["tip"][:2]) for b in bl]
    crit.append(C("blade_height", zb, all(0.020 <= z <= 0.060 for z in zb), "m", "blade bottoms 20-60 mm above the floor",
                  "derived from the GMA opening (89 mm on a 16 mm deck)"))
    from .loads import fork_openings
    # each blade inside its opening clear of the stringers (the asset's forks are fixed at +-293 mm, the GMA openings are
    # centred at +-245 mm and 451 mm wide, so a centred truck has each blade 48 mm outboard of its opening centre), and
    # the forks centred on the pallet (the load centred on the carriage)
    clear, lat = [], []
    for b in bl:
        rel = R.T @ (b["tip"] - o)
        lat.append(rel[1])
        clear.append(max(max(ow / 2 - abs(rel[1] - oc) for oc, ow, *_ in fork_openings()) - FORK_HALF_W, -1.0))
    crit.append(C("forks_in_openings", clear, all(v >= 0.040 for v in clear), "m",
                  ">= 40 mm from each blade to the stringers", "authored (blade 102 mm, opening 451 mm)"))
    centre = float(np.mean(lat))
    crit.append(C("centred_on_pallet", centre, abs(centre) <= 0.050, "m", "fork centre within +-50 mm of the pallet centre",
                  "authored"))
    yaw_err = abs(math.degrees(ang_diff(math.atan2(axis[1], axis[0]), math.atan2(px[1], px[0]))))
    yaw_err = min(yaw_err, 180.0 - yaw_err)
    crit.append(C("square_to_pallet", yaw_err, yaw_err <= 3.0, "deg", "fork axis within 3 deg of the pallet axis", "authored"))
    return crit, det


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


def _carry_to_floor_bay(world, start):
    return _set_down_in_bay(world, start)


def _carry_loose_stack(world, start):
    return _set_down_in_bay(world, start, extra_cartons=True)


def _put_away_rack(world, start):
    crit, det = shared(world, start)
    loc = world.scene.zones["rack_locations"][start["rack_location"]]
    corners, o, R = pallet_corners(world, "carried")
    from .facility import BEAM_W
    beam_x = (loc["beams"][0] + BEAM_W / 2, loc["beams"][1] - BEAM_W / 2)     # the location's front and rear beams
    m, d = world.solver.mj_model, world.solver.mj_data
    pal_geoms = {g for g in range(m.ngeom) if world.mj_geom_label(g).startswith("carried:")}
    beams, other_beams = set(), set()
    for i in range(d.ncon):
        c = d.contact[i]
        for ga, gb in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if ga in pal_geoms:
                lab = world.mj_geom_label(gb)
                if "_beam" in lab:
                    parts = lab.split(":")[0].split("_")          # <row><frame>_beam<bay-1>_<level>_<x>
                    try:
                        bay, lv, bx = int(parts[1][4:]) + 1, int(parts[2]), float(parts[-1])
                    except (IndexError, ValueError):
                        continue
                    mine = bay == loc["bay"] and lv == loc["level"] and min(abs(bx - x) for x in beam_x) < 0.01
                    (beams if mine else other_beams).add(parts[-1])
    crit.append(C("on_both_beams", dict(target_location=sorted(beams), elsewhere=sorted(other_beams)),
                  len(beams) >= 2 and not other_beams, "-",
                  f"contact on the front and the rear beam of bay {loc['bay']} at beam level {loc['level']} "
                  f"({loc['z']:.2f} m), on no other beam", "authored"))
    fz = [float(cz) for cz in corners[:, 2]]
    crit.append(C("at_beam_level", fz, all(abs(z - loc["z"]) <= 0.03 for z in fz), "m",
                  f"pallet bottom corners within 30 mm of the beam tops at {loc['z']:.2f} m", "authored"))
    from .facility import RACK_BEAM, BAY_PITCH
    xs = corners[:, 0]
    overhang = float(loc["aisle_x"] - xs.min()) if loc["face"] == "w" else float(xs.max() - loc["aisle_x"])
    crit.append(C("front_overhang", overhang, 0.051 <= overhang <= 0.101, "m", "51-101 mm past the front beam",
                  "Steel King: 3 in nominal +-1 in (read)"))
    slot = int(start["rack_location"].split("-")[-1])
    yc = loc["y"] + (RACK_BEAM / 4 if slot == 0 else -RACK_BEAM / 4)
    ys = corners[:, 1]
    lo_face, hi_face = yc - RACK_BEAM / 2, yc + RACK_BEAM / 2
    other = start["rack_location"][:-1] + str(1 - slot)
    gaps = [float(ys.min() - lo_face) if slot == 0 else float(hi_face - ys.max())]
    if other in world.scene.layout.zones["rack_locations"] and other not in world.scene.layout.reserved:
        oy = world.scene.layout.zones["rack_locations"][other]["y"]
        gaps.append(float((oy - P.GMA_PALLET["W"] / 2) - ys.max()) if slot == 0 else float(ys.min() - (oy + P.GMA_PALLET["W"] / 2)))
    else:
        gaps.append(float(hi_face - ys.max()) if slot == 0 else float(ys.min() - lo_face))
    clear = min(gaps)
    crit.append(C("side_clearance", clear, clear >= 0.05, "m", ">= 50 mm to the upright and the neighbouring pallet",
                  "3-6 in between loads (read)"))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
    crit.append(C("level", tilt, tilt <= 2.0, "deg", "within 2 deg", "authored"))
    # the stored loads that are bodies (at and beside the slot): where they were
    moved = {n: float(np.linalg.norm(body_frame(world, info["body"])[0] - np.asarray(world.scene.load_start_pose[n])))
             for n, info in world.scene.loads.items() if n != "carried" and not n.startswith("carried_")}
    worst = max(moved.values()) if moved else 0.0
    crit.append(C("stored_loads_undisturbed", dict(max_m=worst, moved={k: v for k, v in moved.items() if v > 0.005}),
                  worst <= 0.030, "m", "every stored load within 30 mm of where it stood", "authored (as unload-trailer)"))
    bl, _ = blades(world)
    out_ = min(float(np.linalg.norm(b["tip"][:2] - o[:2])) for b in bl)
    h, ok = _forks_low(world, 0.30)
    crit.append(C("forks_withdrawn_lowered", dict(tip_to_pallet_m=out_, tip_height_m=h), out_ > 1.0 and ok, "-",
                  "tips clear of the location and at travel height", "authored"))
    return crit, det


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
    tr0 = start["trailers"][start["target"]["door"]]
    if start["task"] == "load-trailer" and not tr0.get("restraint") and not tr0.get("chocks"):
        return _unsecured_trailer(world, start)
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


REACH_LOADED = 1.80     # m: a carried 48 in pallet's far edge ahead of the front axle (fork face 0.556 m + 1.219 m)


def _unsecured_trailer(world, start):
    """load-trailer with the trailer unsecured (restraint signal red): the job is not to board it. The truck and its
    load never reach the leveler (the pit's inner edge), and the truck is left stopped short of it, in neutral, the
    parking brake set, the load still on the forks."""
    crit, det = shared(world, start)
    d = world.scene.zones["doors"][start["target"]["door"]]
    edge = d["face_y"] - P.LEVELER["length"]
    traj = np.array(world.events.traj) if world.events.traj else np.zeros((0, 5))
    reach = float(np.max(traj[:, 2] + REACH_LOADED * np.sin(traj[:, 3]))) if len(traj) else -99.0
    crit.append(C("stayed_off_the_leveler", reach - edge, reach < edge, "m",
                  "the truck's loaded front never past the leveler's inner edge (OSHA 1910.178(m)(7): trucks boarded "
                  "only with brakes set and wheel chocks in place, read)", "read; reach derived"))
    # the job is done at the door: the load brought up to it, waiting for the signal, not left where the truck started.
    # The waiting place is between the staging lane's painted end and the leveler.
    lane = world.scene.zones["staging"]["lane" + start["target"]["door"][4:]]
    if len(traj):
        t_, x_, y_, yaw_, v_ = traj[-1]
        front = float(y_ + REACH_LOADED * math.sin(yaw_))
    else:
        front = -99.0
    crit.append(C("waiting_at_the_door", front - lane["y"][1], front >= lane["y"][1], "m",
                  "where the truck stopped, its loaded front past the end of the staging lane's lines (the load brought "
                  "up to the door to wait for the green signal)", "authored (the work order)"))
    sig = world.signals
    rest = world.events.rest_s
    ok = rest >= 2.0 and sig.get("park", 0.0) > 0.5 and sig.get("direction") == 0
    crit.append(C("stopped_short", dict(rest_s=rest, park=sig.get("park"), direction=sig.get("direction")), ok, "-",
                  ">= 2 s at rest, parking brake set, direction neutral", "authored"))
    corners, o, R = pallet_corners(world, "carried")
    bl, _ = blades(world)
    on_forks = min(float(np.linalg.norm(b["tip"][:2] - o[:2])) for b in bl) < P.GMA_PALLET["L"]
    crit.append(C("load_on_forks", on_forks, on_forks, "-", "the pallet still on the forks", "authored"))
    return crit, det


def _load_trailer_chocked(world, start):
    return _load_trailer(world, start)


def box_corners(world, name):
    """The 8 corners (world) of a box load from its pose and its built size."""
    info = world.scene.loads[name]
    o, R = body_frame(world, info["body"])
    lx, ly, lz = info["size"]
    return np.array([o + R @ np.array([sx * lx / 2, sy * ly / 2, sz * lz / 2])
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]), o, R


# load-bales: the placement is judged from the bales' and the dunnage strips' current poses, on oriented plan
# footprints.
BALE_NOSE_GAP = (-0.005, 0.075)    # m: front face to the nose wall; 5 mm into its plane allowed (a resting contact sits a
#                                    fraction of a millimetre inside it)
BALE_UPRIGHT_DEG = 5.0             # its 48 in axis within 5 deg of the trailer's vertical
BALE_REST_TOL = 0.03               # m: its bottom to the top of the strips it rests on
BALE_OVERLAP_TOL = 0.005           # m: penetration allowed between the two bales' footprints (contact)


def _hull(pts):
    """Convex hull of plan points, counter-clockwise (monotone chain); fewer than 3 distinct points are returned as they are."""
    pts = sorted(set(map(tuple, np.round(np.asarray(pts, float).reshape(-1, 2), 9))))
    if len(pts) < 3:
        return np.array(pts, float).reshape(-1, 2)

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]
    return np.array(half(pts) + half(pts[::-1]), float)


def _clip(poly, by):
    """The part of convex polygon poly inside convex counter-clockwise polygon by (Sutherland-Hodgman)."""
    out = list(np.asarray(poly, float))
    for i in range(len(by)):
        a, b = by[i], by[(i + 1) % len(by)]
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        inp, out = out, []
        for j, p in enumerate(inp):
            q = inp[(j + 1) % len(inp)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if sp * sq < 0:
                out.append(p + (q - p) * sp / (sp - sq))
        if not out:
            break
    return np.array(out, float).reshape(-1, 2)


def _in_convex(p, poly):
    return len(poly) >= 3 and all((poly[(i + 1) % len(poly)][0] - poly[i][0]) * (p[1] - poly[i][1]) -
                                  (poly[(i + 1) % len(poly)][1] - poly[i][1]) * (p[0] - poly[i][0]) >= -1e-9
                                  for i in range(len(poly)))


def _plan_gap(A, B):
    """Signed plan distance between convex polygons A and B: their clearance (> 0) or minus their penetration depth."""
    sep = []
    for Q in (A, B):
        for i in range(len(Q)):
            e = Q[(i + 1) % len(Q)] - Q[i]
            n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            sep.append(max((B @ n).min() - (A @ n).max(), (A @ n).min() - (B @ n).max()))
    if max(sep) <= 0:
        return float(max(sep))

    def seg(p, a, b):
        t = max(0.0, min(1.0, float((p - a) @ (b - a) / ((b - a) @ (b - a)))))
        return float(np.linalg.norm(p - a - t * (b - a)))
    return min(seg(p, Q[i], Q[(i + 1) % len(Q)]) for S, Q in ((A, B), (B, A)) for p in S for i in range(len(Q)))


def _dunnage_tops(world, door):
    """Every dunnage strip's top face in the target trailer's frame (the nose strips and the row's: a strip carried in
    is a strip): name, height and plan rectangle."""
    tr, to, TR = _trailer_frame(world, door)
    out = []
    for n in sorted(world.scene.loads):
        if "dunnage" in n:
            loc = (box_corners(world, n)[0] - to) @ TR
            top = loc[np.argsort(loc[:, 2])[4:]]
            out.append(dict(name=n, top=float(top[:, 2].mean()), rect=_hull(top[:, :2])))
    return out


BOX_SIGNS = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])     # box_corners' order


def _floor_z(world, start, x, y):
    """The dock floor's height under a world point: 0, or on a deployed leveler's deck, rising from its rear hinge to the
    docked trailer's bed at the door face."""
    Ld, Wd = P.LEVELER["length"], P.LEVELER["width"]
    for dn, t in (start.get("trailers") or {}).items():
        d = world.scene.zones["doors"][dn]
        if abs(x - d["x"]) <= Wd / 2 and d["face_y"] - Ld <= y <= d["face_y"]:
            return float(t["bed_z"]) * (y - (d["face_y"] - Ld)) / Ld
    return 0.0


def _bale_state(world, start, name, strips, floored):
    """One bale against the work order, in the target trailer's frame (x from the rear sill, y left, z up). floored: the
    bales with a latched floor contact."""
    door = start["target"]["door"]
    tr, to, TR = _trailer_frame(world, door)
    corners, o, R = box_corners(world, name)
    loc = (corners - to) @ TR
    c = (o - to) @ TR
    tilt = math.degrees(math.acos(min(1.0, abs(float(TR[:, 2] @ R[:, 2])))))      # either way up
    low4 = loc[np.argsort(loc[:, 2])[:4]]
    base = _hull(low4[:, :2])                                                      # the face it stands on
    # the strips it bears on: under its bottom face, the face's plane within BALE_REST_TOL of the strip's top at the
    # closest point of their overlap (read at its lowest corner, a bale tilted past ~6 deg on the strips' edges hangs
    # more than 30 mm below their tops and reads as not resting on them)
    nb = TR.T @ R[:, 2]
    nb = nb if nb[2] > 0 else -nb
    plane = lambda q: low4[0, 2] - (nb[0] * (q[:, 0] - low4[0, 0]) + nb[1] * (q[:, 1] - low4[0, 1])) / nb[2]
    under = [(s["name"], s["top"], _clip(base, s["rect"])) for s in strips]
    under = [(n, a) for n, top, a in under if len(a) >= 3 and abs(float(plane(a).min()) - top) < BALE_REST_TOL]
    # resting on strips: its centre of mass over the support they give it (the hull of its contact patches on them)
    on_strips = bool(under) and _in_convex(c[:2], _hull(np.vstack([a for _, a in under])))
    nose_gap = float(tr["L"] - loc[:, 0].max())
    # against the front wall: the vertical face turned toward the nose (either end or side of the bale), at both of its
    # vertical edges, each read at its nearer corner. The nearest single corner would let a bale turned 25 deg, touching
    # the wall at one corner, pass.
    ax = TR.T @ R[:, :2]
    k, s = max(((k, s) for k in (0, 1) for s in (-1, 1)), key=lambda ks: ks[1] * ax[0, ks[0]])
    edges = [float(tr["L"] - loc[(BOX_SIGNS[:, k] == s) & (BOX_SIGNS[:, 1 - k] == e), 0].max()) for e in (-1, 1)]
    inside = bool(np.all((loc[:, 0] > 0) & (loc[:, 0] < tr["L"] + 0.005) & (np.abs(loc[:, 1]) < tr["W"] / 2)))
    at_wall = all(BALE_NOSE_GAP[0] <= g <= BALE_NOSE_GAP[1] for g in edges)
    placed = inside and on_strips and tilt <= BALE_UPRIGHT_DEG and at_wall
    # resting on a floor now: a latched floor contact and, not borne by strips, its lowest corner within the rest tolerance of the floor under it (a trailer's, in that trailer's frame, or the dock's)
    low = corners[int(np.argmin(corners[:, 2]))]
    on_floor = False
    if name in floored and not on_strips:
        for tn in world.scene.trailers:
            t_, o_, R_ = _trailer_frame(world, tn)
            cc = (o - o_) @ R_
            if 0.0 < cc[0] < t_["L"] and abs(cc[1]) < t_["W"] / 2:
                on_floor = float(((corners - o_) @ R_)[:, 2].min()) < BALE_REST_TOL
                break
        else:
            on_floor = float(low[2]) - _floor_z(world, start, float(low[0]), float(low[1])) < BALE_REST_TOL
    return dict(name=name, tilt=tilt, nose_gap=nose_gap, front_edge_gaps=edges, inside=inside, on_strips=on_strips,
                strips=[n for n, _ in under], placed=bool(placed), on_floor=bool(on_floor),
                in_trailer=bool(0.0 < c[0] < tr["L"] and abs(c[1]) < tr["W"] / 2),
                centre_trailer=[float(c[0]), float(c[1])], xy=[float(o[0]), float(o[1])], plan=_hull(loc[:, :2]))


def _nose_wall_rule(door):
    """load-bales' rule for a bale's contact with the target trailer's nose wall, the wall the work order sets the bales
    against. The shared rule reads the bale's total speed at first contact, so a bale lowered onto its strips while
    touching the wall would read as an impact, and it counts a resting contact's accumulated impulse, which any contact
    that stays passes. Here: not a strike if the bale approached the wall slower than
    IMPACT_SPEED (onset_normal, its speed toward the wall) and never pressed on it harder than its own weight; otherwise a
    strike only above STRIKE_N. The same for a dunnage strip there; a strip's weight bound is
    STRIKE_N. Side walls, other structures and other tasks keep the shared rule."""
    def rule(world, mover, cls, e):
        parts = e.get("parts") or []
        if (cls != "trailer_wall" or not mover.startswith(("load:bale_", "load:nose_dunnage_", "load:dunnage_"))
                or len(parts) < 2 or str(parts[1]) != f"{door}:nose"):
            return None
        if onset_normal(e) < IMPACT_SPEED and e["peak_N"] <= lean_limit(world, episode_load(mover, e)):
            return False
        return e["peak_N"] > STRIKE_N
    return rule


def _truck_past_face(world, door):
    """How far the truck's farthest point (footprint corners, counterweight to fork tips, and both blade tips) stands
    past the door's face toward the trailer (m; < 0: the whole truck behind it, on the dock or its leveler deck)."""
    p, R = world.body_pose("chassis")
    ys = [y for _, y in footprint(float(p[0]), float(p[1]), yaw_of(R))] + [float(b["tip"][1]) for b in blades(world)[0]]
    return float(max(ys) - world.scene.zones["doors"][door]["face_y"])


def _load_bales(world, start):
    door = start["target"]["door"]
    crit, det = shared(world, start, dest_floor=lambda n, xy: not n.startswith("bale"), strike_rule=_nose_wall_rule(door))
    ns = next(c for c in crit if c["name"] == "no_strike")
    ns["threshold"] += (f"; load-bales: a bale's or a dunnage strip's contact with the trailer's nose wall counts only if it "
                        f"approached the wall at >= {IMPACT_SPEED:.2f} m/s or pressed on it harder than its own weight or "
                        f"{STRIKE_N:.0f} N, whichever is more, and then only above {STRIKE_N:.0f} N (not by the impulse a "
                        f"resting contact accumulates)")
    ns["source"] += "; load-bales nose-wall rule: authored"
    strips = _dunnage_tops(world, door)
    floored = {n for n, _, _ in det["dropped"]}
    bales = [_bale_state(world, start, n, strips, floored) for n in sorted(n for n in world.scene.loads if n.startswith("bale_"))]
    at_nose = [b for b in bales if b["placed"]]
    loaded = [b["name"] for b in bales if b["in_trailer"]]
    gaps = [_plan_gap(a["plan"], b["plan"]) for i, a in enumerate(at_nose) for b in at_nose[i + 1:]]
    clear = max(gaps) if gaps else None
    ok = clear is not None and clear >= -BALE_OVERLAP_TOL and len(loaded) == 2
    crit.append(C("two_bales_at_nose", dict(at_nose=[b["name"] for b in at_nose], bales_in_trailer=loaded, clearance_m=clear),
                  ok, "-", "exactly two bales in the trailer, each upright (<= 5 deg) and resting on dunnage strips (its "
                  "bottom within 30 mm of their tops, its centre over the strips it bears on), its front face <= 75 mm "
                  "from the front wall at both front edges (5 mm into it allowed); the two not overlapping (<= 5 mm, "
                  "oriented footprints); any two bales, either side, at any spacing",
                  "authored"))
    past = _truck_past_face(world, door)
    crit.append(C("truck_back_on_dock", past, past < 0.0, "m", "the whole truck, counterweight to fork tips, behind the "
                  "door face (measured: its farthest point past the face)", "the work order ('back out onto the dock')"))
    # the trailer held where it was spotted while the laden truck drove in and out twice, as in load-trailer
    mv = world.events.trailer_max_move.get(door, 0.0)
    crit.append(C("trailer_movement", mv, mv <= 0.025, "m", "<= 25 mm from its spotted position", "authored (as load-trailer)"))
    reward = _load_bales_reward(world, start, crit, det, bales)
    for b in bales:
        b.pop("plan")
        b["progress"] = reward["bales"][b["name"]]
    return crit, dict(det, bales=bales, reward=reward)


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
# here is authored.
BALE_REWARD = dict(contract="load-bales-reward-v1", fail_scale=0.9, exit_share=0.1, exit_gate_from=0.9,
                   carry_cap=0.8, short_decay_m=0.5, carry_tilt_deg=20.0, carry_deadband_m=0.02, strike_merge_s=0.25,
                   floor_merge_s=0.3, strand_factor=0.8, brush_factor=0.9, brush_decay_s=1.0, speed_floor=0.3)
BALE_WORKMANSHIP = {
    "no_strike": "per contact event 1 / (1 + 0.5 log10 s) of its worst episode, s = (approach speed / 0.10 m/s)^2 if it "
                 "approached at >= 0.10 m/s, else max(1, peak / 2 kN); episodes of one mover on one structure class "
                 "< 0.25 s apart are one event; a bale's trailer-floor arrival is graded under no_dropped_load instead",
    "no_dropped_load": "per floor-contact event of a bale (latched entries < 0.3 s apart are one): 0.8 + 0.1 / (1 + duration "
                       "/ 1 s) if the bale came off the floor again, 0.8 if it rests on a floor or lies over, each times "
                       "g(v) = 1 for an approach speed v <= 0.10 m/s, else max(0.3, 1 / (1 + log10 (v / 0.10 m/s)))",
    "no_wheel_lift": "1 / (1 + (t - 0.3 s) / 1 s)",
    "roll_under_6deg": "1 / (1 + (roll - 6 deg) / 6 deg); 0 on overturn",
    "speed_limit": "1 / (1 + t_over / 10 s + 2 (v_max / limit - 1))",
    "operator_seated": "1 / (1 + (t_open - 1 s) / 1 s)",
    "head_clear_of_guard": "limit / peak force",
    "trailer_movement": "1 / (1 + (move - 25 mm) / 25 mm)",
}


def _to_nose_row(world, door, xy):
    """Plan distance (m) from a world point to the target trailer's nose row (a bale centre's place against its front
    wall, anywhere across): straight inside that trailer; from outside it, to its rear opening and on; from inside the
    other trailer, out through that one's rear sill first."""
    tr, to, TR = _trailer_frame(world, door)
    lx, ly = P.BALE["size"][:2]
    xg, yg = tr["L"] - lx / 2, (tr["W"] - ly) / 2

    def local(name, p):
        _, o_, R_ = _trailer_frame(world, name)
        return (R_.T @ (np.array([p[0], p[1], o_[2]]) - o_))[:2]

    def inside(name, r):
        t_ = world.scene.trailers[name]
        return 0.0 <= r[0] <= t_["L"] and abs(r[1]) <= t_["W"] / 2
    p, d = np.asarray(xy[:2], float), 0.0
    for name in world.scene.trailers:
        if name != door and inside(name, local(name, p)):
            d, p = d + float(np.linalg.norm(local(name, p))), _trailer_frame(world, name)[1][:2]
    r = local(door, p)
    if not inside(door, r):
        q = np.array([0.0, float(np.clip(r[1], -tr["W"] / 2, tr["W"] / 2))])
        d, r = d + float(np.linalg.norm(r - q)), q
    return d + float(np.hypot(abs(xg - r[0]), max(0.0, abs(r[1]) - yg)))


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


def _load_bales_reward(world, start, crit, det, bales):
    RW = BALE_REWARD
    floor_events, folded = _floor_events(world, det, RW["floor_merge_s"])
    starts = {ld["name"]: ld["centre"] for ld in start["loads"]}
    door = start["target"]["door"]
    p = {}
    for b in bales:
        lean = 1.0 - max(0.0, b["tilt"] - BALE_UPRIGHT_DEG) / (RW["carry_tilt_deg"] - BALE_UPRIGHT_DEG)
        if b["on_floor"] or lean <= 0.0:
            p[b["name"]] = 0.0
        elif b["placed"]:
            p[b["name"]] = 1.0
        elif b["inside"] and b["on_strips"]:
            short = max(0.0, max(b["front_edge_gaps"]) - BALE_NOSE_GAP[1]) / RW["short_decay_m"]
            p[b["name"]] = (1.0 - (1.0 - RW["carry_cap"]) * min(1.0, short)) * min(1.0, lean)
        else:
            d0 = _to_nose_row(world, door, starts[b["name"]]) - RW["carry_deadband_m"]
            p[b["name"]] = RW["carry_cap"] * max(0.0, (d0 - _to_nose_row(world, door, b["xy"])) / d0)
    placement = float(np.mean(sorted(p.values())[-2:]))
    by = {c["name"]: c for c in crit}
    gate = min(1.0, max(0.0, (placement - RW["exit_gate_from"]) / (1.0 - RW["exit_gate_from"])))
    out_ = float(by["truck_back_on_dock"]["passed"])
    progress = (1.0 - RW["exit_share"]) * placement + RW["exit_share"] * out_ * gate
    events = _strike_events([s for i, s in enumerate(det["strikes"]) if i not in folded], RW["strike_merge_s"])
    lying = {b["name"] for b in bales if b["on_floor"] or b["tilt"] > RW["carry_tilt_deg"]}
    speed = lambda v: 1.0 if v <= SET_DOWN_SPEED else max(RW["speed_floor"], 1.0 / (1.0 + math.log10(v / SET_DOWN_SPEED)))
    brush = lambda e: RW["strand_factor"] + (RW["brush_factor"] - RW["strand_factor"]) / (
        1.0 + (e["t1"] - e["t0"]) / RW["brush_decay_s"])

    def floor_factor(name, evs):
        f = float(np.prod([speed(e["v"]) for e in evs]))
        return f * (RW["strand_factor"] if name in lying else float(np.prod([brush(e) for e in evs])))
    ev = world.events
    m = lambda name: by[name]["measured"]
    grade = {
        "no_strike": lambda: float(np.prod([1.0 / (1.0 + 0.5 * math.log10(e["severity"])) for e in events])),
        "no_dropped_load": lambda: float(np.prod([floor_factor(n, evs) for n, evs in floor_events.items()])),
        "no_wheel_lift": lambda: 1.0 / (1.0 + max(0.0, m("no_wheel_lift") - 0.3)),
        "roll_under_6deg": lambda: 0.0 if ev.overturn is not None else 1.0 / (1.0 + max(0.0, ev.max_roll - 6.0) / 6.0),
        "speed_limit": lambda: 1.0 / (1.0 + ev.over_limit_s / 10.0 + 2.0 * max(0.0, m("speed_limit") / LIMIT_KMH - 1.0)),
        "operator_seated": lambda: 1.0 / (1.0 + max(0.0, m("operator_seated") - 1.0)),
        "head_clear_of_guard": lambda: min(1.0, P.HEAD_CONTACT_LIMIT / max(m("head_clear_of_guard"), 1e-9)),
        "trailer_movement": lambda: 1.0 / (1.0 + max(0.0, m("trailer_movement") - 0.025) / 0.025),
    }
    factors = {k: float(f()) for k, f in grade.items() if not by[k]["passed"]}
    return dict(contract=RW["contract"], rule="1 if success, 0 if the world faulted, else 0.9 x progress x workmanship",
                progress=progress, placement=placement, truck_out=out_, exit_gate=gate,
                workmanship=float(np.prod(list(factors.values()) or [1.0])), factors=factors, bales=p,
                strike_events=[dict(e, t_end=None if math.isinf(e["t_end"]) else e["t_end"]) for e in events],
                floor_events={n: [dict(e) for e in evs] for n, evs in floor_events.items()},
                fail_scale=RW["fail_scale"], workmanship_rules=BALE_WORKMANSHIP)


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


DROP_ZONE = {"east": "drop_zone", "west": "drop_zone_w"}       # facility.yard zones


def _ramp_down_loaded(world, start):
    dz = world.scene.zones[DROP_ZONE[start.get("drop_zone", "east")]]

    def dest(n, xy):
        return n == "carried" and inside_rect(xy, dz["x"], dz["y"])
    crit, det = shared(world, start, dest_floor=dest)
    corners, o, R = pallet_corners(world, "carried")
    inside = all(inside_rect(c, dz["x"], dz["y"]) for c in corners)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
    crit.append(C("in_drop_zone", inside, inside and tilt < 5.0, "-", "inside the drop zone, upright", "authored"))
    # set down, as in the floor bays: resting on the asphalt, the forks out of it and lowered, the truck at rest
    fz = [c[2] - world.scene.ground_height(*c[:2]) for c in corners]
    crit.append(C("resting_on_floor", fz, all(abs(z) <= 0.010 for z in fz), "m", "corners within 10 mm of the ground",
                  "authored (as the floor bays)"))
    bl, _ = blades(world)
    gap = min(float(np.linalg.norm(b["tip"][:2] - o[:2])) for b in bl) - P.GMA_PALLET["L"] / 2
    crit.append(C("backed_out", gap, gap >= 0.5, "m", "fork tips >= 0.5 m from the pallet", "authored (as the floor bays)"))
    h, ok = _forks_low(world, 0.15)
    crit.append(C("forks_lowered", h, ok, "m", "fork tips at or below travel height (0.15 m)", "authored (as the floor bays)"))
    crit.append(C("stationary", world.events.rest_s, world.events.rest_s >= 1.0, "s", ">= 1 s at rest", "authored"))
    v = world.events.ramp_max_speed * 3.6
    crit.append(C("slow_on_ramp", v, v <= 6.0, "km/h", "<= 6 km/h on the ramp", "OSHA 1910.178(n)(7): slowly (read); number authored"))
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
