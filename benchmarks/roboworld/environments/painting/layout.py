"""Room, presets and tasks. World frame: origin at the floor centre of the 3.66 x 3.35 m bedroom, x along
the work (north) wall to the right, y toward it, z up. The work wall's interior face is y = 1.675.

Presets are minor setup changes of the same room:
  wall     A: repaint of the work wall (roller, tray, sash brush, cut pail, gallon can on a drop cloth; the dog-leg
           long-reach brush in hand for behind-radiator)
  craft    C: folding table at the wall with watercolour paper on a drawing board, palette, round brush
  spray    D: folding table with an open spray box, a crate side under a "3" stencil and an aerosol can
`setup(task, seed)` returns the episode spec: robot start, item poses, paints, wall colours, climate, held tools.
"""

import math
import numpy as np

from . import params as P

R = P.ROOM
X, Y, Z, WT = R["x"], R["y"], R["z"], R["wall_t"]
YW = Y/2                                         # work wall interior face
WIN = P.WINDOW
WX0, WX1 = WIN["x"] - WIN["w"]/2, WIN["x"] + WIN["w"]/2
CS, CT = WIN["casing"], WIN["casing_t"]
BB = P.BASEBOARD
OUT = P.OUTLET
RAD = P.RADIATOR


def static_boxes():
    """(name, centre, half extents, material) of the room's installed parts (static colliders)."""
    s = []
    # work wall with the window opening (four pieces), the other walls and the ceiling
    yc = YW + WT/2
    s.append(("wall_n_left", (-X/2 + (WX0 + X/2)/2 - WT/2, yc, Z/2), ((WX0 + X/2 + WT)/2, WT/2, Z/2), "drywall"))
    s.append(("wall_n_right", ((WX1 + X/2 + WT)/2, yc, Z/2), ((X/2 + WT - WX1)/2, WT/2, Z/2), "drywall"))
    s.append(("wall_n_low", (WIN["x"], yc, WIN["sill"]/2), (WIN["w"]/2, WT/2, WIN["sill"]/2), "drywall"))
    s.append(("wall_n_head", (WIN["x"], yc, (WIN["head"] + Z)/2), (WIN["w"]/2, WT/2, (Z - WIN["head"])/2), "drywall"))
    s.append(("glazing", (WIN["x"], YW + 0.07, (WIN["sill"] + WIN["head"])/2), (WIN["w"]/2, 0.006, (WIN["head"] - WIN["sill"])/2),
              "glass"))
    s.append(("wall_s", (0., -YW - WT/2, Z/2), (X/2 + WT, WT/2, Z/2), "drywall"))
    s.append(("wall_e", (X/2 + WT/2, 0., Z/2), (WT/2, Y/2, Z/2), "drywall"))
    s.append(("wall_w", (-X/2 - WT/2, 0., Z/2), (WT/2, Y/2, Z/2), "drywall"))
    s.append(("ceiling", (0., 0., Z + 0.05), (X/2 + WT, Y/2 + WT, 0.05), "ceiling"))
    # window trim: casing legs and head (2-1/4 in, 11/16 in proud), stool and apron
    yt = YW - CT/2
    s.append(("casing_l", (WX0 - CS/2, yt, (WIN["sill"] + WIN["head"] + CS)/2), (CS/2, CT/2, (WIN["head"] + CS - WIN["sill"])/2), "trim"))
    s.append(("casing_r", (WX1 + CS/2, yt, (WIN["sill"] + WIN["head"] + CS)/2), (CS/2, CT/2, (WIN["head"] + CS - WIN["sill"])/2), "trim"))
    s.append(("casing_head", (WIN["x"], yt, WIN["head"] + CS/2), (WIN["w"]/2, CT/2, CS/2), "trim"))
    sw = WIN["w"]/2 + CS + 0.02
    s.append(("stool", (WIN["x"], YW - WIN["stool_d"]/2 + 0.03, WIN["sill"] - WIN["stool_t"]/2), (sw, WIN["stool_d"]/2 + 0.03,
                                                                                                    WIN["stool_t"]/2), "trim"))
    s.append(("apron", (WIN["x"], yt, WIN["sill"] - WIN["stool_t"] - WIN["apron"]/2), (WIN["w"]/2 + CS, CT/2, WIN["apron"]/2), "trim"))
    # baseboard along the work wall and the side walls
    s.append(("base_n", (0., YW - BB["t"]/2, BB["h"]/2), (X/2, BB["t"]/2, BB["h"]/2), "trim"))
    s.append(("base_e", (X/2 - BB["t"]/2, 0., BB["h"]/2), (BB["t"]/2, Y/2, BB["h"]/2), "trim"))
    s.append(("base_w", (-X/2 + BB["t"]/2, 0., BB["h"]/2), (BB["t"]/2, Y/2, BB["h"]/2), "trim"))
    # duplex receptacle plate
    s.append(("outlet_plate", (OUT["x"], YW - OUT["t"]/2, OUT["z"]), (OUT["w"]/2, OUT["t"]/2, OUT["h"]/2), "plate"))
    # cast-iron radiator under the window: 10 sections at a 2.5 in pitch (4 columns each, drawn), hubs top and bottom, feet, supply pipe
    ry1 = YW - RAD["gap"]
    ry0 = ry1 - RAD["depth"]
    pitch = RAD["len"]/RAD["sections"]
    for k in range(RAD["sections"]):
        x = WIN["x"] - RAD["len"]/2 + (k + 0.5)*pitch
        s.append((f"rad_sec_{k}", (x, (ry0 + ry1)/2, 0.06 + (RAD["height"] - 0.06)/2), (pitch*0.38, RAD["depth"]/2,
                                                                                        (RAD["height"] - 0.06)/2), "radiator"))
    for k, z in enumerate((0.11, RAD["height"] - 0.05)):
        s.append((f"rad_hub_{k}", (WIN["x"], (ry0 + ry1)/2, z), (RAD["len"]/2, 0.025, 0.02), "radiator"))
    for k, x in enumerate((WIN["x"] - RAD["len"]/2 + 0.02, WIN["x"] + RAD["len"]/2 - 0.02)):
        s.append((f"rad_foot_{k}", (x, (ry0 + ry1)/2, 0.03), (0.018, RAD["depth"]/2, 0.03), "radiator"))
    return s


RADIATOR_BOX = (WIN["x"] - RAD["len"]/2, WIN["x"] + RAD["len"]/2, YW - RAD["gap"] - RAD["depth"], YW - RAD["gap"], 0.0,
                RAD["height"])
PIPE = dict(x=WIN["x"] - RAD["len"]/2 - 0.03, y=YW - RAD["gap"] - RAD["depth"]/2, r=RAD["pipe_r"], z1=0.11)
CLOTH = dict(x0=-1.62, x1=1.10, y0=0.47, y1=YW - BB["t"])     # 9 x 4 ft canvas drop cloth along the wall

PAINT_COLORS = ("naval", "lullaby", "alabaster")          # SW 6244 accent, SW 9136, SW 7008 (Sherwin-Williams colours)
OLD_COLORS = ("accessible_beige", "naval", "primer")
DEEP_COLORS = ("naval",)                                    # deep/accent tint base (painting.paint.tint_base)

PRESETS = {
    "wall": ("roll-strip", "load-roller", "cut-in-ceiling", "cut-in-trim", "behind-radiator", "touch-up"),
    "craft": ("fill-circle", "brush-stroke", "swatch-card"),
    "spray": ("spray-stencil", "clear-coat"),
}

TASKS = {
    "load-roller": dict(preset="wall", held=dict(right="roller"), tier="entry-medium", instruction=(
        "Load the dry roller cover in the tray: roll it in the paint well and on the ramp until it carries at least 50 g "
        "of paint spread around the nap, then hold it still above the tray for 10 s without a drip.")),
    "roll-strip": dict(preset="wall", held=dict(right="roller"), tier="medium-hard", instruction=(
        "Roll the marked 0.6 m-wide strip of the work wall (between the two blue tape marks, from the brushed band under "
        "the ceiling to the brushed band above the baseboard) to full hide, keeping a wet edge with the strip on its left, "
        "without runs, lap marks or paint on the ceiling, trim or outlet plate; then rest the roller in the tray.")),
    "cut-in-ceiling": dict(preset="wall", held=dict(right="sash"), tier="hard", instruction=(
        "With the angled sash brush, cut in a band at least 50 mm wide along 1.2 m of the wall-ceiling corner between "
        "the blue tape marks, reaching the corner without putting paint on the ceiling; load the brush from the cut pail.")),
    "cut-in-trim": dict(preset="wall", held=dict(right="sash"), tier="hard", instruction=(
        "With the angled sash brush, cut in along the left side of the window casing and around the outlet plate, a band "
        "at least 40 mm wide, without paint on the casing or the plate.")),
    "behind-radiator": dict(preset="wall", held=dict(right="longreach"), tier="hard", instruction=(
        "With the long-reach dog-leg brush (load it from the cut pail), paint the wall behind the radiator -- the strip "
        "between the baseboard and the window apron -- to full hide, without getting paint on the radiator or the "
        "trim.")),
    "touch-up": dict(preset="wall", held=dict(right="sash"), tier="medium", painted=True, instruction=(
        "Hide the three scuffs on the dry painted wall (marked with blue tape arrows) with thin feathered touch-ups that "
        "leave no visible ridge.")),
    "fill-circle": dict(preset="craft", held=dict(right="round8"), tier="medium", instruction=(
        "With the round brush, fill the pencilled 60 mm circle on the watercolour paper evenly with the blue acrylic from "
        "the palette without crossing the line by more than 1 mm.")),
    "brush-stroke": dict(preset="craft", held=dict(right="round8"), tier="hard", instruction=(
        "With the round brush, paint the pencilled S-stroke on the paper with the red acrylic, swelling from 2 mm to 6 mm "
        "and back to 2 mm wide, staying within 1 mm of the template width.")),
    "swatch-card": dict(preset="craft", held=dict(right="round8"), tier="medium", instruction=(
        "Paint the pencilled 50 x 50 mm square on the paper to full hide with the blue acrylic.")),
    "spray-stencil": dict(preset="spray", held=dict(left="aerosol"), tier="hard", instruction=(
        "Spray the number 3 through the stencil onto the crate side in the spray box, 25-40 cm from the target (the "
        "can's 10-16 in), building an even 25-50 um dry coat (the can's 1-2 mil) from light overlapping passes, with "
        "sharp edges and no runs, keeping overspray inside the box.")),
    "clear-coat": dict(preset="spray", held=dict(left="aerosol"), tier="medium-hard", lacquer=True, instruction=(
        "Clear-coat the photo print lying in the spray box evenly (25-50 um dry, built from light overlapping passes, "
        "25-40 cm from it) without runs, keeping overspray inside the box.")),
}

ROBOT_START = dict(wall=(-0.35, 0.70, math.pi/2), craft=(-0.35, 0.30, math.pi/2), spray=(-0.35, 0.30, math.pi/2))

# strip / band targets on the wall chart (x0, x1, z0, z1) in world metres
STRIP = dict(x0=-0.75, x1=-0.15, z0=0.15, z1=Z - 0.06)
CEILING_BAND = dict(x0=-1.30, x1=-0.10, z0=Z - 0.050, z1=Z)
CASING_BAND = dict(x0=WX0 - CS - 0.04, x1=WX0 - CS, z0=WIN["sill"], z1=WIN["head"] + CS)
OUTLET_BAND = dict(x0=OUT["x"] - OUT["w"]/2 - 0.04, x1=OUT["x"] + OUT["w"]/2 + 0.04, z0=OUT["z"] - OUT["h"]/2 - 0.04,
                   z1=OUT["z"] + OUT["h"]/2 + 0.04)
BEHIND_RAD = dict(x0=WIN["x"] - RAD["len"]/2, x1=WIN["x"] + RAD["len"]/2, z0=BB["h"], z1=WIN["sill"] - WIN["stool_t"] - WIN["apron"])


def yaw_quat(yaw):
    return (0., 0., math.sin(yaw/2), math.cos(yaw/2))


def lying_quat(yaw):
    """Roller frame lying down: handle (frame -z) along world -x rotated by yaw, cover axis (frame y) horizontal."""
    from scipy.spatial.transform import Rotation
    return tuple((Rotation.from_euler("z", yaw)*Rotation.from_euler("y", math.pi/2)).as_quat())


def flat_quat(yaw):
    """A brush lying on its flat side: thickness (body y) vertical, bristles (body z) along world -y rotated by yaw."""
    from scipy.spatial.transform import Rotation
    return tuple((Rotation.from_euler("z", yaw)*Rotation.from_euler("x", math.pi/2)).as_quat())


def setup(task, seed=0):
    """Episode spec for a task and seed. Seeds vary paint and wall colours, tray fill, tool placement (+-5 cm, +-10
    deg), climate (18-27 C, RH 30-70 %) and, on the wall, the two spackle patches."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(TASKS)}")
    t = TASKS[task]
    preset = t["preset"]
    rng = np.random.default_rng(1000 + int(seed))
    jit = (lambda s=0.05: float(rng.uniform(-s, s))) if seed else (lambda s=0.05: 0.0)
    ang = (lambda s=10: math.radians(float(rng.uniform(-s, s)))) if seed else (lambda s=10: 0.0)
    climate = dict(T_C=23.0, RH=0.50) if not seed else dict(T_C=float(rng.uniform(18, 27)), RH=float(rng.uniform(0.30, 0.70)))
    spec = dict(task=task, preset=preset, seed=int(seed), glides="nylon", climate=climate, items={}, books={},
                robot=ROBOT_START[preset], held=dict(t["held"]), instruction=t["instruction"])
    if preset == "wall":
        spec["paint"] = dict(kind="latex", color=PAINT_COLORS[seed % 3])
        spec["old_color"] = OLD_COLORS[(seed//3) % 3] if seed else "accessible_beige"
        if spec["old_color"] == spec["paint"]["color"]:
            # a repaint in the wall's own colour would start hidden (the layout sweep found seed 3 solved at the start)
            spec["old_color"] = OLD_COLORS[(seed//3 + 1) % 3]
        spec["first_coat"] = 0.0
        if spec["paint"]["color"] in DEEP_COLORS:
            # a deep colour goes over the tinted primer its product sheet asks for ('use the recommended P-Shade primer
            # to optimize hide', ProMar 200) and needs two coats over it (one 43 um coat does not hide, two do;
            # practitioners' guidance for SW 6244 Naval). The wall already carries its first coat, dry and a
            # little patchy; the tasks lay the finish coat
            spec["old_color"] = "pshade_primer"
            spec["first_coat"] = 43e-6
            if task != "load-roller":
                spec["instruction"] = spec["instruction"] + (
                    " The wall already has its first coat of this colour (dry); this is the second, finish coat.")
        spec["painted"] = bool(t.get("painted"))
        spec["patches"] = [] if not seed else [(float(rng.uniform(-1.5, -0.2)), float(rng.uniform(0.4, 2.0)), 0.06)
                                               for _ in range(2)]
        fill = 1.0e-3 if not seed else float(rng.uniform(0.6e-3, 1.2e-3))
        items = spec["items"]
        items["tray"] = dict(kind="tray", pose=(-1.05 + jit(), 1.25 + jit(), 0.0, yaw_quat(math.pi/2 + ang())), fill=fill)
        items["pail"] = dict(kind="pail", pose=(-0.02 + jit(), 1.30 + jit(0.03), 0.0, yaw_quat(ang(30))), fill=0.6e-3)
        items["can"] = dict(kind="can", pose=(-1.45 + jit(), 1.42 + jit(0.03), 0.0, yaw_quat(0.0)), fill=P.CAN["fill"])
        if "roller" not in t["held"].values():
            # lying on the drop cloth: handle (frame -z) along the floor, cover axis horizontal
            items["roller"] = dict(kind="roller", pose=(-0.55 + jit(), 1.22 + jit(0.03), 0.0315, lying_quat(ang())))
        if "sash" not in t["held"].values():
            # lying flat on the cloth beside the pail (thickness vertical), bristles toward the room
            items["sash"] = dict(kind="sash", pose=(0.17 + jit(0.03), 1.20 + jit(0.03), 0.0103, flat_quat(ang(20))))
        # wet bands already brushed for the roll-strip task (painter's order: cut in first, roll into it), and the strip
        # rolled two minutes earlier on the left (seeded wet-edge age)
        spec["prepainted"] = []
        if task == "roll-strip":
            spec["prepainted"] = [dict(region=(STRIP["x0"], STRIP["x1"] + 0.6, Z - 0.075, Z), h=110e-6, age=60.0),
                                  dict(region=(STRIP["x0"], STRIP["x1"] + 0.6, BB["h"], BB["h"] + 0.075), h=110e-6, age=60.0),
                                  dict(region=(STRIP["x0"] - 0.6, STRIP["x0"], BB["h"], Z), h=105e-6, age=120.0)]
        if task == "touch-up":
            spec["scuffs"] = [(-1.2 + 0.4*k + jit(0.05), 1.0 + 0.25*k + jit(0.05)) for k in range(3)]
    elif preset == "craft":
        spec["paint"] = dict(kind="acrylic", colors=("ultramarine", "cadmium_red"))
        spec["old_color"] = "accessible_beige"
        items = spec["items"]
        items["table"] = dict(kind="table", pose=(-0.35, 1.30, 0.0, yaw_quat(0.0)))
        zt = P.TABLE["height"]
        items["board"] = dict(kind="board", pose=(-0.30 + jit(0.02), 1.28 + jit(0.02), zt, yaw_quat(ang(3))))
        items["palette"] = dict(kind="palette", pose=(-0.80 + jit(0.03), 1.20 + jit(0.03), zt, yaw_quat(ang(10))),
                                wells={0: ("ultramarine", 9e-6), 1: ("cadmium_red", 9e-6)})
        if "round8" not in t["held"].values():
            items["round8"] = dict(kind="round8", pose=(0.12, 1.15, zt + 0.0046, flat_quat(math.pi/2)))
    else:
        lac = bool(t.get("lacquer"))
        spec["paint"] = dict(kind="lacquer" if lac else "enamel", color="clear" if lac else "gloss_red")
        spec["old_color"] = "accessible_beige"
        items = spec["items"]
        items["table"] = dict(kind="table", pose=(-0.35, 1.30, 0.0, yaw_quat(0.0)))
        zt = P.TABLE["height"]
        items["spray_box"] = dict(kind="spray_box", pose=(-0.10, 1.33, zt, yaw_quat(0.0)))
        if lac:
            spec["print"] = True
        else:
            cj, sj, a = jit(0.02), jit(0.02), ang(3)
            items["crate"] = dict(kind="crate", pose=(-0.10 + cj, 1.33 + sj, zt + 0.004, yaw_quat(a)))
            items["stencil"] = dict(kind="stencil", pose=(-0.10 + cj, 1.33 + sj, zt + 0.004 + 0.012, yaw_quat(a)))
        if "aerosol" not in t["held"].values():
            items["aerosol"] = dict(kind="aerosol", pose=(-0.80, 1.20, zt, yaw_quat(0.0)))
    return spec
