"""The building, dock and yard of each preset: drivable surfaces (tyre model + native contact),
static structures (colliders with their visual: what can be touched is drawn as it collides), painted markings
(render only), dressing props (render only, kept off the routes or given a collider where they sit on a route edge), and
the evaluator's zone geometry. Units metres, world z up; the interior floor is z = 0 in the warehouse and dock presets,
the yard asphalt z = 0 in the yard preset.

Warehouse preset (50 x 36 m section of a distribution centre):
  racks: rows A (single, west wall) | aisle 1 | B (double) | aisle 2 | C (double) | aisle 3 | D (single), 42 in frames,
         96 in beams, beam levels at 1.60 and 3.20 m (floor level 0), seven bays per row, 4.50 m working aisles between the
         stored loads' faces (VDI 2198 Ast 4.329 m for this truck with a 48 in pallet lengthways, derived: params.AISLE_RACK),
         the beam faces 4.65 m apart (the loads overhang them by 3 in);
  cross aisle north of the racks, main travel aisle (4.5 m) along the dock wall, two north-south travel lanes with
  intersections I1 (x 20-24) and I2 (x 34-38), stop lines and convex mirrors;
  floor storage: two rows of six painted bays 1.4 x 1.2 m (B-01..B-12) with a 5.2 m access lane;
  production edge: a powered roller conveyor (1.2 m wide, 0.45 m roller top) feeding a press, photo-eye and end stop;
  receiving lanes R1-R3, staging lanes S1-S4 along the dock wall, a parking bay at the west end of the main aisle.
"""

import math

import numpy as np

from . import params as P

IN, FT = 0.0254, 0.3048
WALL_H = P.FACILITY["wall_h"]


class Layout:
    """Everything a preset places, by name, for the scene, the renderer and the evaluator."""

    def __init__(self):
        self.boxes = []          # dict(name, centre, half, yaw, material, collide, watch)
        self.lines = []          # painted markings: dict(name, a, b, width, colour)
        self.props = []          # kit USD references: dict(name, usd, p, yaw, scale)
        self.zones = {}          # evaluator geometry
        self.lights = []
        self.starts = {}         # task start poses (x, y, yaw)
        self.surfaces = []       # drivable rectangles (scene.floor_rect kwargs)
        self.bodies = []         # movable fixtures built later (conveyor rollers, levelers, trailer)

    def box(self, name, centre, half, material, yaw=0.0, collide=True, watch=None):
        self.boxes.append(dict(name=name, centre=tuple(map(float, centre)), half=tuple(map(float, half)), yaw=float(yaw),
                               material=material, collide=collide, watch=watch))

    def line(self, name, a, b, width=0.10, colour="yellow"):
        self.lines.append(dict(name=name, a=tuple(map(float, a)), b=tuple(map(float, b)), width=float(width), colour=colour))

    def rect_lines(self, name, x0, x1, y0, y1, width=0.10, colour="yellow", sides="nsew"):
        if "s" in sides: self.line(name + "_s", (x0, y0), (x1, y0), width, colour)
        if "n" in sides: self.line(name + "_n", (x0, y1), (x1, y1), width, colour)
        if "w" in sides: self.line(name + "_w", (x0, y0), (x0, y1), width, colour)
        if "e" in sides: self.line(name + "_e", (x1, y0), (x1, y1), width, colour)


# ---------------------------------------------------------------------------------------------------- racks
RACK_DEPTH = P.RACK["frame_depth"]
# a stored 48 in pallet stands centred on its 42 in frame, overhanging the front and the rear beam by 3 in each (Interlake
# Mecalux, Pallet Rack Selection Guide p. 2, the depth figure: 3" either side of "Frame D"; the put-away evaluator's
# front overhang band 51-101 mm), so its aisle face stands 76 mm into the aisle
STORED_OVERHANG = (P.GMA_PALLET["L"] - RACK_DEPTH) / 2
RACK_BEAM = P.RACK["beam"]
UPRIGHT = P.RACK["upright"]
FRAME_GAP = P.RACK["frame_gap"]           # between a double row's back-to-back frames (row spacers)
BAY_PITCH = RACK_BEAM + UPRIGHT
BEAM_LEVELS = tuple(P.RACK["levels"][1:])
BEAM_H, BEAM_W = P.RACK["beam_h"], P.RACK["beam_w"]
UPRIGHT_H = P.FACILITY["upright_h"]


def rack_row(L, name, x0, y0, n_bays, double, faces, stored, rng):
    """A selective rack row along +y from (x0, y0): x0 is the west face of the row. double: back-to-back frames on
    12 in row spacers, leaving a 6 in flue between the loads' rear overhangs. faces: which aisles it serves ('w'/'e'). stored: callable(face, bay, level) -> True if a stored load sits
    there (level 0 = floor). Records rack locations in L.zones['rack'] as name-face-bay-level."""
    frames = [(x0, x0 + RACK_DEPTH)]
    if double:
        frames.append((x0 + RACK_DEPTH + FRAME_GAP, x0 + 2 * RACK_DEPTH + FRAME_GAP))
    locs = L.zones.setdefault("rack_locations", {})
    for fi, (fa, fb) in enumerate(frames):
        face = ("w" if fi == 0 else "e") if double else faces
        aisle_x = fa if face == "w" else fb                    # the beam face toward the aisle
        for k in range(n_bays + 1):
            y = y0 + k * BAY_PITCH
            for xx in (fa + UPRIGHT / 2, fb - UPRIGHT / 2):
                L.box(f"{name}{fi}_up{k}_{xx:.2f}", (xx, y, UPRIGHT_H / 2), (UPRIGHT / 2, UPRIGHT / 2, UPRIGHT_H / 2),
                      "rack_upright", watch="rack")
        for k in range(n_bays):
            yc = y0 + (k + 0.5) * BAY_PITCH
            for lv, z in enumerate(BEAM_LEVELS, start=1):
                for xx in (fa + BEAM_W / 2, fb - BEAM_W / 2):
                    L.box(f"{name}{fi}_beam{k}_{lv}_{xx:.2f}", (xx, yc, z - BEAM_H / 2), (BEAM_W / 2, RACK_BEAM / 2, BEAM_H / 2),
                          "rack_beam", watch="rack")
            for lv in range(3):
                z = 0.0 if lv == 0 else BEAM_LEVELS[lv - 1]
                for slot, ys in enumerate((yc - RACK_BEAM / 4, yc + RACK_BEAM / 4)):
                    loc = f"{name}-{face}-{k + 1:02d}-{lv}-{slot}"
                    locs[loc] = dict(x0=fa, x1=fb, y=ys, z=z, face=face, aisle_x=aisle_x, beam_front=aisle_x,
                                     beams=(fa, fb), level=lv, bay=k + 1)
                    if loc not in L.reserved and stored(face, k + 1, lv, slot):
                        _stored_unit(L, loc, (fa + fb) / 2, ys, z, rng)
        # column protector at the aisle end of the frame
        for yy in (y0 - 0.12, y0 + n_bays * BAY_PITCH + 0.12):
            L.box(f"{name}{fi}_guard_{yy:.1f}", ((fa + fb) / 2, yy, 0.25), (RACK_DEPTH / 2, 0.06, 0.25), "guard_yellow",
                  watch="rack")


def _dynamic_location(L, loc):
    """True for a rack location whose stored load is a body resting on its beams by contact: around a reserved
    (put-away) slot, every level of the slot's face in its bay and the bays either side, and across a double row's flue
    the same bays at the slot's level: what a misplaced or misaligned pallet can push."""
    r_, f_, b_, v_, _ = loc.split("-")
    for res in L.reserved:
        row, face, bay, lv, _ = res.split("-")
        if r_ == row and abs(int(b_) - int(bay)) <= 1 and (f_ == face or v_ == lv):
            return True
    return False


# Vertical lift-off clearance between the top of a stored unit load and the underside of the beam above it: 6 in
# (params.RACK_LIFT_OFF, read: Interlake Mecalux's rack selection guide, Cisco-Eagle's clearance guide).
STORED_LIFT_OFF = P.RACK_LIFT_OFF
# the tallest load a location takes: the smallest opening between a beam level's top (or the floor) and the underside
# of the beam above, less the pallet and the lift-off clearance (1.473 - 0.121 - 0.152 = 1.200 m); every level takes the
# same population of unit loads, the top one too (it has no beam above; its loads top out at 4.52 m, under the 16 ft
# upright). The lower bound, 0.9 m, is authored (unchanged), the same as the put-away task's carried loads (0.9-1.2 m)
_LEVELS = P.RACK["levels"]
STORED_LOAD_H = (0.9, min(b - a for a, b in zip(_LEVELS, _LEVELS[1:])) - BEAM_H - P.GMA_PALLET["H"] - STORED_LIFT_OFF)


def _stored_unit(L, loc, x, y, z, rng):
    """A stored wrapped unit load: pallet + load box, the load's height drawn from STORED_LOAD_H (it fits its opening
    with the lift-off clearance). Static boxes (part of the structure) except around a put-away slot, where it is a body
    built in build_bodies (_dynamic_location)."""
    g = P.GMA_PALLET
    h = float(rng.uniform(*STORED_LOAD_H))         # one draw, as before: every other draw of the layout is unchanged
    m = float(rng.uniform(*P.STORED_LOAD_MASS))
    if _dynamic_location(L, loc):
        L.bodies.append(dict(kind="stored_unit", loc=loc, x=float(x), y=float(y), z=float(z), h=h, mass=m))
        return
    L.box(f"stored_{loc}_pallet", (x, y, z + g["H"] / 2), (g["L"] / 2, g["W"] / 2, g["H"] / 2), "pallet_wood", watch="rack")
    L.box(f"stored_{loc}_load", (x, y, z + g["H"] + h / 2), (g["L"] / 2 - 0.02, g["W"] / 2 - 0.02, h / 2), "wrapped",
          watch="rack")


# ---------------------------------------------------------------------------------------------------- warehouse
def warehouse(rng, reserved=frozenset()):
    L = Layout()
    L.reserved = set(reserved)
    X1, Y1 = 50.0, 36.0
    L.surfaces.append(dict(name="floor", centre=(X1 / 2, Y1 / 2), half=(X1 / 2 + 1, Y1 / 2 + 1), material="concrete_dry",
                           thickness=0.5))
    for nm, c, h in (("wall_s", (X1 / 2, -0.15), (X1 / 2 + 0.3, 0.15)), ("wall_n", (X1 / 2, Y1 + 0.15), (X1 / 2 + 0.3, 0.15)),
                     ("wall_w", (-0.15, Y1 / 2), (0.15, Y1 / 2)), ("wall_e", (X1 + 0.15, Y1 / 2), (0.15, Y1 / 2))):
        L.box(nm, (c[0], c[1], WALL_H / 2), (h[0], h[1], WALL_H / 2), "wall", watch="wall")
    # columns with guards
    for i, x in enumerate((12.0, 24.0, 36.0)):
        L.box(f"column{i}", (x, 33.3, WALL_H / 2), (0.2, 0.2, WALL_H / 2), "concrete_column", watch="column")
        L.box(f"column{i}_guard", (x, 33.3, 0.45), (0.32, 0.32, 0.45), "guard_yellow", watch="column")
    # racks: seven bays from y = 4.0
    y0, nb = 4.0, 7
    x = 0.6

    def stored(face, bay, lv, slot, row=None):
        return rng.random() < 0.8
    rack_row(L, "A", x, y0, nb, False, "e", stored, rng); xa1 = x + RACK_DEPTH
    # three rows (A single facing east, B back-to-back, C single facing west) on working aisles of AISLE_RACK between the
    # stored loads' faces (VDI 2198 / Hyster's Ast: the obstacles the truck turns between are the loads, which overhang
    # the beams by STORED_OVERHANG on the aisle side), so the beam faces stand AISLE_RACK + 2 x STORED_OVERHANG apart.
    # Four rows at this aisle width would run into the first cross aisle
    A = P.AISLE_RACK + 2 * STORED_OVERHANG
    rack_row(L, "B", xa1 + A, y0, nb, True, None, stored, rng); xb1 = xa1 + A + 2 * RACK_DEPTH + FRAME_GAP
    rack_row(L, "C", xb1 + A, y0, nb, False, "w", stored, rng)
    L.zones["aisles"] = {"1": (xa1, xa1 + A), "2": (xb1, xb1 + A)}
    y_rack_end = y0 + nb * BAY_PITCH
    # main aisle (y 26-30.5) and cross aisle; lanes x 20-24 and 34-38
    L.rect_lines("main_aisle", 0.3, X1 - 0.3, 26.0, 30.5, colour="yellow", sides="ns")
    for nm, (xa, xb) in (("lane1", (20.0, 24.0)), ("lane2", (34.0, 38.0))):
        L.line(nm + "_w", (xa, 2.0), (xa, 26.0)); L.line(nm + "_e", (xb, 2.0), (xb, 26.0))
    L.zones["main_aisle"] = dict(x=(0.3, X1 - 0.3), y=(26.0, 30.5))
    L.zones["intersections"] = {"I1": dict(x=(20.0, 24.0), y=(26.0, 30.5), stop_x=19.4),
                                "I2": dict(x=(34.0, 38.0), y=(26.0, 30.5), stop_x=33.4)}
    for nm, z in L.zones["intersections"].items():
        L.line(f"stop_{nm}_w", (z["stop_x"], 26.0), (z["stop_x"], 30.5), 0.30, "white")
        L.line(f"stop_{nm}_e", (z["x"][1] + 0.6, 26.0), (z["x"][1] + 0.6, 30.5), 0.30, "white")
        L.props.append(dict(name=f"mirror_{nm}", kind="convex_mirror", p=(z["x"][0] - 0.3, 31.2, 2.6), yaw=-math.pi / 2))
    # floor storage bays
    bays = {}
    for i in range(6):
        xc = 24.8 + 1.5 * i
        bays[f"B-{i + 1:02d}"] = dict(x=(xc - 0.6, xc + 0.6), y=(8.0, 9.4), approach=math.pi / 2 * -1)
        bays[f"B-{i + 7:02d}"] = dict(x=(xc - 0.6, xc + 0.6), y=(14.6, 16.0), approach=math.pi / 2)
    for nm, b in bays.items():
        L.rect_lines("bay_" + nm, b["x"][0], b["x"][1], b["y"][0], b["y"][1], 0.075, "white")
    L.zones["bays"] = bays
    # production edge: conveyor along +x at y 8.4-9.6, entry at x = 41, end stop at x = 47.2; press beyond
    conv = dict(x=(41.0, 47.2), y=(8.4, 9.6), top=0.45, roller_pitch=0.15, roller_r=0.03, eye_x=46.6)
    L.zones["conveyor"] = conv
    L.bodies.append(dict(kind="conveyor", **conv))
    for side, yy in (("s", conv["y"][0] - 0.06), ("n", conv["y"][1] + 0.06)):
        L.box(f"conv_frame_{side}", ((conv["x"][0] + conv["x"][1]) / 2, yy, conv["top"] / 2 - 0.02),
              ((conv["x"][1] - conv["x"][0]) / 2, 0.04, conv["top"] / 2 - 0.02), "machine_grey", watch="conveyor")
        # guide rails above the roller tops (the pallet must go between them)
        L.box(f"conv_guide_{side}", ((conv["x"][0] + conv["x"][1]) / 2, yy, conv["top"] + 0.08),
              ((conv["x"][1] - conv["x"][0]) / 2, 0.02, 0.025), "guard_yellow", watch="conveyor_guide")
    L.box("conv_endstop", (conv["x"][1] + 0.05, sum(conv["y"]) / 2, conv["top"] + 0.12), (0.05, 0.65, 0.12), "guard_yellow",
          watch="conveyor")
    L.box("press", (48.6, sum(conv["y"]) / 2, 1.6), (1.2, 1.4, 1.6), "machine_green", watch="machine")
    L.box("press_guard", (47.5, sum(conv["y"]) / 2, 1.1), (0.04, 0.9, 1.1), "guard_yellow", watch="machine")
    L.rect_lines("drop_zone", 40.0, 44.0, 4.0, 6.5, 0.10, "yellow")
    # receiving lanes (start states for loaded trucks) and staging lanes along the dock wall
    for i in range(3):
        yc = 17.5 + 2.4 * i
        L.rect_lines(f"R{i + 1}", 40.0, 49.5, yc - 1.0, yc + 1.0, 0.10, "white", sides="ns")
    L.zones["receiving"] = {f"R{i + 1}": dict(x=(40.0, 49.5), y=(16.5 + 2.4 * i, 18.5 + 2.4 * i)) for i in range(3)}
    for i in range(4):
        xc = 30.0 + 4.0 * i
        L.rect_lines(f"S{i + 1}", xc - 1.6, xc + 1.6, 31.2, 35.7, 0.10, "white", sides="we")
    L.zones["staging"] = {f"S{i + 1}": dict(x=(30.0 + 4.0 * i - 1.6, 30.0 + 4.0 * i + 1.6), y=(31.2, 35.7)) for i in range(4)}
    # parking bay and LPG cage at the west end of the main aisle
    L.rect_lines("parking", 0.5, 5.6, 26.5, 30.0, 0.10, "yellow", sides="nsw")
    L.zones["parking"] = dict(x=(0.5, 5.6), y=(26.5, 30.0))
    L.box("lpg_cage", (1.0, 32.5, 0.9), (0.6, 0.9, 0.9), "cage_wire", watch="structure")
    L.zones["stop_lines"] = {"W0": dict(x=19.4, y=(26.0, 30.5)), "W1_end": dict(x=47.0, y=(26.0, 30.5)),
                             "west_end": dict(x=7.5, y=(26.0, 30.5))}
    L.line("stop_W1_end", (47.0, 26.0), (47.0, 30.5), 0.30, "white")
    # the westward cruise's far line (aisle-cruise, task.CRUISE_DIRECTIONS): 1.9 m east of the parking bay's open end, so a
    # truck stopped with its front wheels on it, heading west, keeps its fork tips (1.62 m ahead) out of the bay
    L.line("stop_west_end", (7.5, 26.0), (7.5, 30.5), 0.30, "white")
    # dock wall: four closed doors (visual) and dressing
    for i in range(4):
        L.props.append(dict(name=f"dock_door{i}", kind="dock_door_closed", p=(30.0 + 4.0 * i, Y1, 0.0), yaw=0.0))
    L.props += [dict(name="sign_speed", kind="sign_speed8", p=(2.0, 30.9, 2.2), yaw=-math.pi / 2),
                dict(name="extinguisher0", kind="extinguisher", p=(12.6, 33.0, 0.0), yaw=0.0),
                dict(name="first_aid", kind="first_aid", p=(24.6, 33.0, 1.5), yaw=0.0)]
    # lights
    for xx in np.arange(6.0, X1, 12.0):
        for yy in np.arange(6.0, Y1, 12.0):
            L.lights.append(dict(p=(float(xx), float(yy), WALL_H - 0.6)))
    L.extent = (0.0, X1, 0.0, Y1)
    return L


# ---------------------------------------------------------------------------------------------------- dock
DOCK_FACE_Y = P.FACILITY["dock_face_y"]
DOOR_X = tuple(5.0 + k * P.FACILITY["door_pitch"] for k in range(4))
DOOR_W, DOOR_H = P.FACILITY["door"]
BUMPER = P.FACILITY["dock_bumper"]               # moulded rubber dock bumper projection
REAR_FRAME = P.FACILITY["trailer_rear_frame"]     # trailer rear frame (posts, sill) behind the floor's rear edge (dock.add_trailer)
RAMP_BUMPER = P.FACILITY["ramp_bumper"]           # the yard ramp's laminated bumpers, 4 x 10 in


def dock(rng, reserved=frozenset()):
    """Shipping dock section: interior x 0-26, y 0-20 at z = 0; dock wall at y = 20 with four 9 x 10 ft doors on 12 ft
    centres, a hydraulic leveler in a pit at each door, rubber bumpers; the apron outside at -48 in."""
    L = Layout()
    L.reserved = set(reserved)
    X1, Y1 = 26.0, DOCK_FACE_Y
    Ld, Wd = P.LEVELER["length"], P.LEVELER["width"]
    apron = -P.DOCK_HEIGHT
    L.zones["apron_z"] = apron
    pit_y0 = Y1 - Ld - 0.01
    L.surfaces.append(dict(name="floor", centre=(X1 / 2, pit_y0 / 2), half=(X1 / 2, pit_y0 / 2), material="concrete_dry",
                           thickness=0.5))
    xs = [0.0]
    for x in DOOR_X:
        xs += [x - Wd / 2 - 0.01, x + Wd / 2 + 0.01]
    xs.append(X1)
    for k in range(0, len(xs), 2):
        a, b_ = xs[k], xs[k + 1]
        L.surfaces.append(dict(name=f"floor_dock{k // 2}", centre=((a + b_) / 2, (pit_y0 + Y1) / 2),
                               half=((b_ - a) / 2, (Y1 - pit_y0) / 2), material="concrete_dusty", thickness=0.5))
    for i, x in enumerate(DOOR_X):
        # pit walls and floor (a deck that falls stops on its velocity fuse before reaching them)
        L.box(f"pit{i}_floor", (x, (pit_y0 + Y1) / 2, -0.62), (Wd / 2 + 0.01, Ld / 2, 0.02), "concrete_column", watch="pit")
    L.surfaces.append(dict(name="apron", centre=(X1 / 2, Y1 + 15.0), half=(X1 / 2 + 4.0, 15.0), material="concrete_dry",
                           z_top=apron, thickness=0.5))
    # dock wall (inside face y = 19.7, outer face = the dock face y = 20.0) with door openings and headers; under each door
    # the dock face continues below the deck's lowest travel, with 4.5 in rubber bumpers projecting from it
    WT = 0.30
    edges = [0.0]
    for x in DOOR_X:
        edges += [x - DOOR_W / 2, x + DOOR_W / 2]
    edges.append(X1)
    for k in range(0, len(edges), 2):
        a, b_ = edges[k], edges[k + 1]
        L.box(f"dockwall{k // 2}", ((a + b_) / 2, Y1 - WT / 2, (WALL_H + apron) / 2),
              ((b_ - a) / 2, WT / 2, (WALL_H - apron) / 2), "wall", watch="wall")
    for i, x in enumerate(DOOR_X):
        L.box(f"dockheader{i}", (x, Y1 - WT / 2, (DOOR_H + WALL_H) / 2), (DOOR_W / 2, WT / 2, (WALL_H - DOOR_H) / 2), "wall",
              watch="wall")
        L.box(f"dockface{i}", (x, Y1 - WT / 2, (apron - 0.45) / 2), (DOOR_W / 2, WT / 2, (-0.45 - apron) / 2),
              "concrete_column", watch="wall")
        for sgn in (-1, 1):
            # bumpers outboard of the leveler pit (deck half-width 0.99 m): at +-0.95 m they overlapped the deck's sides
            # by 0.16 m and jammed it with ~2.9 MN at rest, loading the trailer through the lip
            L.box(f"bumper{i}_{sgn}", (x + sgn * (Wd / 2 + 0.03 + 0.12), Y1 + BUMPER / 2, -0.22), (0.12, BUMPER / 2, 0.18),
                  "rubber_black", watch="bumper")
            # foam dock seal pads (visual: they compress around a trailer and carry no load worth modelling)
            L.box(f"seal{i}_{sgn}", (x + sgn * (DOOR_W / 2 + 0.15), Y1 + 0.15, DOOR_H / 2), (0.15, 0.15, DOOR_H / 2),
                  "seal_foam", collide=False)
    for nm, c, h in (("wall_s", (X1 / 2, -0.15), (X1 / 2 + 0.3, 0.15)), ("wall_w", (-0.15, Y1 / 2), (0.15, Y1 / 2)),
                     ("wall_e", (X1 + 0.15, Y1 / 2), (0.15, Y1 / 2))):
        L.box(nm, (c[0], c[1], WALL_H / 2), (h[0], h[1], WALL_H / 2), "wall", watch="wall")
    L.zones["doors"] = {f"door{i + 1}": dict(name=f"door{i + 1}", x=x, face_y=Y1 + 0.30, leveler=True)
                        for i, x in enumerate(DOOR_X)}
    for d in L.zones["doors"].values():
        d["face_y"] = Y1
    # staging lanes in front of the doors
    L.zones["staging"] = {}
    for i, x in enumerate(DOOR_X):
        L.rect_lines(f"lane{i + 1}", x - 1.4, x + 1.4, 6.0, 15.5, 0.10, "white", sides="we")
        L.zones["staging"][f"lane{i + 1}"] = dict(x=(x - 1.4, x + 1.4), y=(6.0, 15.5))
    L.line("dock_edge", (0.2, Y1 - 0.05), (X1 - 0.2, Y1 - 0.05), 0.15, "yellow")
    # bale row and dressing
    L.zones["bale_row"] = dict(x=(19.0, 25.0), y=(3.0, 9.0))
    L.rect_lines("bale_row", 19.0, 25.0, 3.0, 9.0, 0.10, "yellow")
    for i in range(5):
        L.box(f"pallet_stack{i}", (1.2, 3.0 + 1.5 * i, 0.6), (0.61, 0.51, 0.6), "pallet_stack", watch="structure")
    for xx in np.arange(4.0, X1, 8.0):
        for yy in np.arange(4.0, Y1, 8.0):
            L.lights.append(dict(p=(float(xx), float(yy), WALL_H - 0.6)))
    L.extent = (0.0, X1, 0.0, Y1)
    return L


# ---------------------------------------------------------------------------------------------------- yard
def yard(rng, reserved=frozenset()):
    """Outdoor yard at z = 0: asphalt, a concrete apron along the building wall (y 15-25), a 53 ft trailer spotted with
    its rear at a mobile yard ramp (30 ft incline + 6 ft level-off, 84 in wide, steel grating, 8 in curbs), parked
    trailers forming a lane, painted lane lines and a drop zone."""
    L = Layout()
    L.reserved = set(reserved)
    L.zones["apron_z"] = 0.0
    L.surfaces.append(dict(name="asphalt", centre=(15.0, 0.0), half=(32.0, 30.0), material="asphalt_dry", thickness=0.5))
    L.surfaces.append(dict(name="apron", centre=(15.0, 20.0), half=(32.0, 5.0), material="concrete_dry", thickness=0.02,
                           priority=1, collider=False))
    L.box("building_wall", (15.0, 25.15, WALL_H / 2), (32.0, 0.15, WALL_H / 2), "wall", watch="wall")
    R = P.YARD_RAMP
    inc, lev, w, rise = R["incline"], R["level"], R["width"], R["rise"]       # rise: the trailer's floor height
    # the ramp's legs are set so its hinged lip plates lie on the trailer floor: the level-off one plate thickness above
    # it (a plate pinned at the deck's top edge and resting on a floor flush with the deck stood 9.6 deg up on the sill's
    # edge with its toe 61 mm in the air; Bluff manual: set the level-off to the trailer, then lower the plates onto it)
    deck = rise + P.YARD_RAMP_LIP["plate"]
    x_r = 10.0
    y_rear = -8.0                           # trailer rear sill
    y_top = y_rear + lev                    # incline top
    run = math.sqrt(inc ** 2 - deck ** 2)
    y_foot = y_top + run
    L.zones["ramp"] = dict(x=x_r, y_foot=y_foot, y_top=y_top, y_rear=y_rear, width=w, rise=rise, deck=deck, grade=deck / run)
    # ramp surfaces: incline rising toward -y, level-off at the deck height (steel grating)
    L.surfaces.append(dict(name="ramp_incline", centre=(x_r, (y_foot + y_top) / 2), half=(inc / 2, w / 2), material="grating",
                           z_top=deck / 2, grade=deck / run, axis=(0.0, -1.0), priority=2, thickness=0.15))
    # the level-off ends at the ramp's 4 in dock bumpers (Bluff manual p. 7), which meet the trailer's rear frame
    gap = REAR_FRAME + RAMP_BUMPER
    L.zones["ramp"]["level_end"] = y_rear + gap
    L.surfaces.append(dict(name="ramp_level", centre=(x_r, (y_top + y_rear + gap) / 2), half=((lev - gap) / 2, w / 2),
                           material="grating", z_top=deck, priority=2, thickness=0.15, axis=(0.0, -1.0)))
    for sgn in (-1, 1):
        # two 4 x 10 in laminated bumpers under the deck's front edge, 3 mm off the trailer's rear frame, their tops 3 cm
        # under the lip plates (which swing down 17 mm over them at the 3 in stop)
        L.box(f"ramp_bumper_{sgn}", (x_r + sgn * 0.70, y_rear + REAR_FRAME + 0.003 + (RAMP_BUMPER - 0.003) / 2, rise - 0.03 - 0.127),
              (0.13, (RAMP_BUMPER - 0.003) / 2, 0.127), "rubber_black", watch="bumper")
    # the multi-piece hinged lip: plates pinned along the level-off's front edge, lowered onto the trailer floor and
    # resting on it as the sprung trailer sinks and rises (dock.add_ramp_lip; built with the trailer in build_bodies)
    L.bodies.append(dict(kind="ramp_lip", name="yard_ramp", hinge=(x_r, y_rear + gap - P.YARD_RAMP_LIP["hinge_gap"], deck),
                         width=w - 0.10, yaw=math.pi, trailer="yard_trailer"))
    ang = math.atan(deck / run)
    for sgn in (-1, 1):
        # curbs: 8 in steel channels along both edges of the incline and the level-off
        cy = (y_foot + y_top) / 2
        L.box(f"curb_incline_{sgn}", (x_r + sgn * (w / 2 + 0.05), cy, deck / 2 + 0.10 * math.cos(ang)),
              (0.05, inc / 2, 0.10), "ramp_yellow", yaw=0.0, watch="curb")
        L.boxes[-1]["pitch"] = -ang                       # rises toward -y
        L.box(f"curb_level_{sgn}", (x_r + sgn * (w / 2 + 0.05), (y_top + y_rear + gap) / 2, deck + 0.10),
              (0.05, (lev - gap) / 2, 0.10), "ramp_yellow", watch="curb")
    L.zones["yard_trailer"] = dict(rear=(x_r, y_rear), yaw=-math.pi / 2, bed_z=rise)
    # parked trailers (static: not part of any task's physics) forming the lane
    # the same van as the spotted one (params.TRAILER, TRAILER_INSIDE): body from 0.22 m under the 48 in floor to the
    # roof, rear doors at y = -13, nose toward +y, landing gear gear_from_rear behind the doors
    Ti, Tp = P.TRAILER_INSIDE, P.TRAILER
    z0, z1 = rise - Tp["subframe"], rise + Ti["H"] + 0.06
    half_l = (Ti["L"] + 2 * Tp["wall"]) / 2
    for i, x in enumerate((22.0, 26.0, 30.0)):
        L.box(f"parked_trailer{i}", (x, -13.0 + half_l, (z0 + z1) / 2), (Tp["outside_W"] / 2, half_l, (z1 - z0) / 2),
              "trailer_wall", watch="parked")
        L.box(f"parked_trailer{i}_gear", (x, -13.0 + Tp["gear_from_rear"], z0 / 2), (1.0, 0.2, z0 / 2), "steel", watch="parked")
    # yard lane (Y3): from the apron at x 36 to the ramp foot, 4 m wide, painted edges; its last 3.5 m run straight in
    # line with the ramp so a truck can stop square to it inside the lines
    L.zones["yard_lane"] = dict(points=[(36.0, 21.8), (36.0, 10.0), (18.0, 10.0), (x_r, y_foot + 6.5), (x_r, y_foot + 2.0)],
                                width=4.0)       # starts 2.8 m behind the start pose: the counterweight starts inside
    pts = L.zones["yard_lane"]["points"]
    for k in range(len(pts) - 1):
        (ax, ay), (bx, by) = pts[k], pts[k + 1]
        d = np.array([bx - ax, by - ay]); d /= np.linalg.norm(d); nrm = np.array([-d[1], d[0]]) * 2.0
        L.line(f"lane_l{k}", (ax + nrm[0], ay + nrm[1]), (bx + nrm[0], by + nrm[1]), 0.10, "white")
        L.line(f"lane_r{k}", (ax - nrm[0], ay - nrm[1]), (bx - nrm[0], by - nrm[1]), 0.10, "white")
    L.line("ramp_foot_line", (x_r - 1.5, y_foot + 0.3), (x_r + 1.5, y_foot + 0.3), 0.15, "yellow")
    # the lane's other end (yard-route back to the building, task.YARD_ROUTES): a yellow line across its head on the apron,
    # 3.2 m from the building's drive-in door
    hx, hy = L.zones["yard_lane"]["points"][0]
    L.zones["apron_line"] = dict(x=hx, y=hy)
    L.line("apron_line", (hx - 1.5, hy), (hx + 1.5, hy), 0.15, "yellow")
    L.zones["drop_zone"] = dict(x=(14.0, 17.0), y=(5.0, 8.0))
    L.rect_lines("drop_zone", 14.0, 17.0, 5.0, 8.0, 0.10, "yellow")
    # a second drop zone, the first mirrored about the ramp's centreline (ramp-down-loaded, task.DROP_ZONES): open asphalt
    # west of the ramp foot, clear of its cones (x 7.7, y 3.5) and of the yard lane (its west line at x 8.0 there)
    L.zones["drop_zone_w"] = dict(x=(2 * x_r - 17.0, 2 * x_r - 14.0), y=(5.0, 8.0))
    L.rect_lines("drop_zone_w", 2 * x_r - 17.0, 2 * x_r - 14.0, 5.0, 8.0, 0.10, "yellow")
    L.extent = (-15.0, 45.0, -25.0, 25.0)
    return L


def _cluster(name):
    """Cluster key of a static box: rack frames per row and frame, stored loads per row and bay, else by kind."""
    if name.startswith("stored_"):
        row, face, bay = name.split("_")[1].split("-")[:3]
        return f"stored_{row}{face}{bay}"
    for prefix in ("A0", "B0", "B1", "C0", "C1", "D0"):
        if name.startswith(prefix + "_"):
            part = name.split("_")[1]
            k = int("".join(ch for ch in part if ch.isdigit()) or 0)
            return f"rack_{prefix}_{k // 3}"
    if name.startswith("wall"):
        return name
    if name.startswith("column"):
        return name.split("_")[0]
    return name.split("_")[0]


def build(scene):
    """Layout and static structures (the task's start is resolved after this, from the layout's zones)."""
    from . import task as TK
    rng = np.random.default_rng(1000 + scene.c.seed)
    L = {"warehouse": warehouse, "dock": dock, "yard": yard}[scene.c.preset](rng, reserved=TK.reserved_locations(scene.c))
    scene.layout = L
    scene.zones = L.zones
    for s in L.surfaces:
        scene.floor_rect(**s)
    from .scene import quat_axis_angle
    # static structures grouped into fixed bodies by cluster (rack row frame, walls, conveyor, ...) for the broadphase
    clusters = {}
    for bx in L.boxes:
        if bx["collide"]:
            clusters.setdefault(_cluster(bx["name"]), []).append(bx)
    for cname, members in clusters.items():
        c = np.mean([m["centre"] for m in members], axis=0)
        body = scene.static_body(cname, c)
        for bx in members:
            rot = None
            if bx.get("pitch"):
                rot = scene.wp.quat(*quat_axis_angle((1, 0, 0), bx["pitch"]))
            elif bx["yaw"]:
                rot = scene.wp.quat(*quat_axis_angle((0, 0, 1), bx["yaw"]))
            scene.static_box(bx["name"], bx["centre"], bx["half"], bx["material"], rot=rot, watch=bx["watch"], body=body,
                             origin=c)


DRESSING_REACH = P.FACILITY["dressing_reach"]   # the mast's top at full lift is about 5.6 m; nothing drawn below this may be air


def _aabb(centre, half, yaw=0.0):
    c, h = np.asarray(centre, float), np.asarray(half, float)
    cy, sy = abs(math.cos(yaw)), abs(math.sin(yaw))
    hx, hy = cy * h[0] + sy * h[1], sy * h[0] + cy * h[1]
    return np.array([c[0] - hx, c[1] - hy, c[2] - h[2]]), np.array([c[0] + hx, c[1] + hy, c[2] + h[2]])


def _keep_out(scene, start, trailers):
    """Boxes (lo, hi) the moving bodies occupy at the start or sweep by design: each trailer's van, each leveler's deck
    and lip, the truck with its load at its start pose. A drawn object standing in one of them (a foam seal a trailer
    presses into, a jack stand under its nose, a chain to the ramp) cannot be a rigid collider: it would be inside a
    body at tick 0."""
    out = []
    Ti, Tp = P.TRAILER_INSIDE, P.TRAILER
    for name, rear, yaw, bed_z, ground_z in trailers:
        c, s_ = math.cos(yaw), math.sin(yaw)
        half_l = (Ti["L"] + 0.3) / 2
        centre = (rear[0] + c * (half_l - 0.15), rear[1] + s_ * (half_l - 0.15), bed_z + (Ti["H"] + 0.2 - 0.7) / 2)
        out.append(_aabb(centre, (half_l, Tp["outside_W"] / 2 + 0.12, (Ti["H"] + 0.2 + 0.7) / 2), yaw))
    docked = {t[0] for t in trailers}
    for door in scene.layout.zones.get("doors", {}).values():
        if door.get("leveler"):
            Lv = P.LEVELER
            y0, y1 = door["face_y"] - Lv["length"] - 0.05, door["face_y"] + Lv["lip"] + 0.10
            top = 0.45 if door["name"] in docked else -0.05      # a stowed leveler stays in its pit, under the door
            out.append((np.array([door["x"] - Lv["width"] / 2 - 0.05, y0, -0.75]), np.array([door["x"] + Lv["width"] / 2 + 0.05, y1, top])))
    x, y, yaw = scene.truck_pose
    c, s_ = math.cos(yaw), math.sin(yaw)
    out.append(_aabb((x + c * -0.35, y + s_ * -0.35, 1.4), (2.35, 0.75, 1.5), yaw))
    return out


def dressing_colliders(scene, start, keep_out=()):
    """Everything the visual modules draw without a simulated counterpart (visuals.*.dressing) and that a truck, its
    mast or its load could reach gets a fixed box collider, so that nothing visible can be driven through: bins,
    drums, pallet trucks, cones, bollards, track guards, closed dock doors, fences, poles, the parked trailers' running
    gear. Not given one: paint, parts inside a rack's or a machine's own colliders, flat wall signs, anything that
    starts above DRESSING_REACH, and items their module marks collide=False. An item with a 'door' key is the closed
    leaf of that dock door: no collider where a trailer stands at it (the door is raised). Fixed, not movable: striking
    one is recorded like striking a structure."""
    from .visuals import dockyard, interior
    from .scene import quat_axis_angle
    L = scene.layout
    docked = set((start or {}).get("trailers", {}))
    items, skipped = [], []
    for mod in (interior, dockyard):
        fn = getattr(mod, "dressing", None)
        if fn is None:
            continue
        for it in fn(L, scene.c.preset, scene.c.seed):
            half = np.asarray(it["half"], float)
            if it.get("collide") is False or it.get("where") in ("paint", "rack", "machine"):
                continue
            if it["p"][2] - half[2] >= DRESSING_REACH or half.min() < 0.015:
                continue
            if it.get("where") == "wall" and half.min() < 0.05:
                continue
            if it.get("kind") == "door_panel" and it.get("door") in docked:
                continue
            if it.get("kind") == "door_panel":                  # the closed leaf stands on the stowed deck: 20 mm clear
                it = dict(it, p=(it["p"][0], it["p"][1], it["p"][2] + 0.01), half=(half[0], half[1], half[2] - 0.01))
                half = np.asarray(it["half"], float)
            lo, hi = _aabb(it["p"], half, float(it.get("yaw", 0.0)))
            if any((lo < k_hi + 0.03).all() and (hi > k_lo - 0.03).all() for k_lo, k_hi in keep_out):
                skipped.append(it["name"])
                continue
            items.append(it)
    cells = {}
    for it in items:
        cells.setdefault((int(it["p"][0] // 12.0), int(it["p"][1] // 12.0)), []).append(it)
    n = 0
    for cell, members in sorted(cells.items()):
        c = np.mean([m["p"] for m in members], axis=0)
        body = scene.static_body(f"dressing_{cell[0]}_{cell[1]}", c)
        for it in members:
            yaw = float(it.get("yaw", 0.0))
            rot = scene.wp.quat(*quat_axis_angle((0, 0, 1), yaw)) if yaw else None
            scene.static_box(f"dressing_{it['name']}", it["p"], it["half"], "dressing", rot=rot, watch="dressing",
                             body=body, origin=c)
            n += 1
    scene.dressing_colliders = n
    scene.dressing_without_collider = skipped          # drawn where a moving body stands at the start
    return n


def build_bodies(scene, start):
    """Movable fixtures of the layout, configured by the task's start: conveyor rollers, levelers, trailers."""
    from . import machines, dock as D
    L = scene.layout
    for item in L.bodies:
        if item["kind"] == "conveyor":
            machines.add_conveyor(scene, item, powered=(scene.c.task == "feed-conveyor"))
        elif item["kind"] == "stored_unit":
            from . import loads as LD
            g = P.GMA_PALLET
            name = f"stored_{item['loc']}"
            LD.add_pallet(scene, name, (item["x"], item["y"], 0.0), z0=item["z"],
                          load=dict(kind="wrapped", mass=item["mass"], size=(g["L"] - 0.04, g["W"] - 0.04, item["h"])))
            # watched like the rack: a carried pallet or the truck striking it is a strike; its own resting contact on
            # its beams is a set-down (evaluator)
            scene.statics.setdefault("rack", []).extend(scene.loads[name]["shapes"])
    docked = {}
    for door in L.zones.get("doors", {}).values():
        t = start.get("trailers", {}).get(door["name"])
        if t is not None:
            docked[door["name"]] = ((door["x"], door["face_y"] + BUMPER + REAR_FRAME + 0.006), math.pi / 2, t["bed_z"],
                                    L.zones["apron_z"])
    yard_trailers = {name: (t["rear"], t["yaw"], t["bed_z"], 0.0) for name, t in start.get("yard_trailers", {}).items()}
    dressing_colliders(scene, start, _keep_out(scene, start, [(n, *v) for n, v in {**docked, **yard_trailers}.items()]))
    for door in L.zones.get("doors", {}).values():
        t = start.get("trailers", {}).get(door["name"])
        if t is not None:
            rear, yaw, bed_z, ground_z = docked[door["name"]]
            tr = D.add_trailer(scene, door["name"], rear, yaw, bed_z, ground_z, restraint=t.get("restraint", False),
                               chocks=t.get("chocks", True))
            if door.get("leveler"):
                D.add_leveler(scene, door["name"], door["x"], door["face_y"], t["bed_z"], deployed=True)
        elif door.get("leveler"):
            D.add_leveler(scene, door["name"], door["x"], door["face_y"], 0.0, deployed=False)
    for name, (rear, yaw, bed_z, ground_z) in yard_trailers.items():
        D.add_trailer(scene, name, rear, yaw, bed_z, ground_z, restraint=False, chocks=True, stands=True)
    for item in L.bodies:
        if item["kind"] == "ramp_lip" and item["trailer"] in scene.trailers:
            D.add_ramp_lip(scene, item["name"], item["hinge"], item["width"], item["yaw"],
                           scene.trailers[item["trailer"]]["bed_z"], item["trailer"])
