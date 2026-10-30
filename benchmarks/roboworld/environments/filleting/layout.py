"""Physical layout of the filleting cell. One source for the physics world, the render
dressing and the evaluator. Units: metres, kilograms, seconds. World frame: z up; x along the table's front
edge (robot's right is +x); y from the table's front edge into the table (toward the back wall); origin on
the floor directly below the middle of the table's front edge.

Every value carries a provenance note: read (manufacturer/standard read), literature, estimate, authored.
"""

import numpy as np

# ------------------------------------------------------------------------------------------------ room
ROOM = dict(size=(5.0, 4.0), ceiling=2.8,                      # authored
            x_range=(-2.2, 2.8), y_range=(-2.3, 1.7),           # table near the back wall, room for robots
            wall_tile_height=2.0)

# ------------------------------------------------------------------------------------------------ bench
TABLE = dict(size=(1.80, 0.75), top_z=0.900, top_thickness=0.040,   # authored; catering-table heights
             center=(0.0, 0.375), undershelf_z=0.25, leg=0.040)
TABLE_FRONT_Y = 0.0

# NSF HDPE board 610 x 457 x 12.7 mm (read: common NSF sizes), blue for fish, red for meat; on a damp cloth
BOARD = dict(size=(0.610, 0.457, 0.0127), center=(0.0, 0.28), cloth_thickness=0.001)
BOARD_TOP_Z = TABLE["top_z"] + BOARD["cloth_thickness"] + BOARD["size"][2]      # 0.9137

# ------------------------------------------------------------------------------------------------ robots
# Two FR3 on one pedestal along the table's front edge, bases at the bench height, 0.10 m in front of the edge,
# at x = -0.30 (left: holding hand) and +0.30 (right: knife hand); both face +y. (authored)
ROBOT_BASES = {"left": dict(pos=(-0.30, -0.10, 0.900), yaw_deg=90.0),
               "right": dict(pos=(0.30, -0.10, 0.900), yaw_deg=90.0)}
PEDESTAL = dict(size=(0.90, 0.30, 0.900), center=(0.0, -0.162))      # welded stainless pedestal (authored); its back
# face stops 12 mm short of the bench's front edge

# ------------------------------------------------------------------------------------------------ vessels
# GN 1/1 (530 x 325 mm, EN 631, literature) with crushed ice and stock trout: left of the board (dressing)
ICE_PAN = dict(size=(0.530, 0.325, 0.065), center=(-0.62, 0.40), presets=("A",))
# perforated GN 1/2 (325 x 265 mm): the fillet tray (task target), right of the board, standing on the bench (the
# research's 'on ice' is not modelled: a chilled drain insert on the bench)
FILLET_TRAY = dict(size=(0.325, 0.265, 0.040), center=(0.50, 0.30), wall=0.0008)
# 240 mm stainless bowl for frames and heads, behind the fillet tray (the target of no task: fillet-whole-fish has
# its own frame pan; still on every preset-A bench)
BOWL = dict(diameter=0.240, depth=0.080, center=(0.52, 0.60))
# fillet-whole-fish (A3) only: an empty solid GN 1/2 pan, 100 mm deep (EN 631 1/2, 325 x 265 mm; drawn and collided as
# dressing.gn_pan_geometry: 12 mm rim flange, walls tapering 18 mm to the floor, so the inner opening is 301 x 241 mm and
# the floor 265 x 205 mm), for the frame and head, at the front-left of the bench. The bowl above is 204.8 mm across
# inside its collider (scene.py) while the seed-0 trout's bones run 275.6 mm snout to hypural, so a straight frame
# cannot lie in it; and only the knife hand reaches the bowl while only the holding hand can pinch a fillet. Here the
# left (holding) arm reaches every point 60 mm inside the rim at six hand yaws and the right arm does not (kinematics
# only).
FRAME_PAN = dict(size=(0.325, 0.265, 0.100), center=(-0.5675, 0.1425), rim=0.012, taper=0.018, wall=0.0008,
                 tasks=("A3",), presets=("A",))
# fillet-whole-fish (A3) only: the perforated fillet tray stands on the holding-arm side, where the stock ice pan stands
# on the common bench, 325 mm along x, so the only hand that can pinch a fillet (the left: the right fingers carry the
# knife-handle inserts) can lay both fillets flat side by side. At the common position no side-by-side set-down point
# solves for the left arm; here every point does (kinematics only). The stock ice pan takes the tray's common place,
# right of the board. Their geometry is unchanged.
FILLET_TRAY_CENTER_OF_TASK = {"A3": (-0.505, 0.42)}
ICE_PAN_CENTER_OF_TASK = {"A3": (0.605, 0.30)}
# bench knife block at the right end of the board; knives handle up. 200 mm tall so the 173 mm fillet blade fits
# handle-up (a 120 mm block cannot hold it); clear of the bowl
KNIFE_BLOCK = dict(size=(0.10, 0.16, 0.20), center=(0.33, 0.60),
                   # slot centres across the block (x, block frame): fillet knife, boning knife. 70 mm apart so a
                   # Franka Hand with the knife inserts fits between the handles: to pass the inserts' lips with a
                   # margin the fingers open 22 mm each side of the 19 mm handle, and a finger's own collision geometry
                   # then reaches ~48 mm from the handle's centre (44 mm apart: no grasp fitted at all; 54 mm: the
                   # finger struck the boning knife's handle on the way in)
                   slot_x=(-0.032, 0.038))
# ASSISTED runs only (Config.phantom_rig): the phantom rig parks the knife behind the board, edge down, blade toward
# +x, 120 mm above the board, clear of the knife block, the bowl and both hands at the ready pose (TCPs at z 1.306 m)
PHANTOM_KNIFE_PARK = dict(handle_com=(-0.12, 0.62, BOARD_TOP_Z + 0.12), yaw_deg=0.0)
# tweezer tray (preset B) and the bench portion scale (presets B, C): 300 x 250 mm platform, 6 kg x 1 g
TWEEZER_TRAY = dict(size=(0.20, 0.10, 0.015), center=(0.45, 0.07))       # clear of the board and the fillet tray
SCALE = dict(size=(0.30, 0.25, 0.045), center=(-0.55, 0.40), presets=("B", "C"))   # where preset A has the ice pan

# ------------------------------------------------------------------------------------------------ cameras
# overhead 1.00 m above the board, looking down, 60 deg horizontal FOV; supervisor from the wall side at
# 1.6 m eye height looking at the board and both hands; one D405 per wrist (87 deg, shared accessory)
CAMERAS = {
    "overhead": dict(pos=(0.0, 0.28, BOARD_TOP_Z + 1.00), look_at=(0.0, 0.28, BOARD_TOP_Z), hfov_deg=60.0,
                     width=1280, height=720),
    "supervisor": dict(pos=(0.15, 1.45, 1.60), look_at=(0.0, 0.20, BOARD_TOP_Z), hfov_deg=55.0,
                       width=1280, height=720),
}

# ------------------------------------------------------------------------------------------------ workpieces
# Preset A: trout on its side on the board, head toward the left arm (-x), back toward the robots (-y).
# Preset C: the tenderloin with its thick (head) end toward the knife hand (+x), as a right-handed cook lays it to
# slice medallions from the thick end while the other hand holds the rest; the chain then lies toward the robots.
# (With the head end toward the left arm, the right arm has to reach across the holding arm, and the arms collide.)
WORKPIECE_POSE = {
    "A": dict(center=(0.0, 0.28), yaw_deg=0.0),
    "B": dict(center=(0.0, 0.28), yaw_deg=0.0),
    "C": dict(center=(0.0, 0.28), yaw_deg=180.0),
}


def yaw_quat(deg):
    a = np.radians(deg)/2
    return np.array([0., 0., np.sin(a), np.cos(a)])            # xyzw


def ice_pan_center(task=None):
    """The ice pan's centre in a task's scene (ICE_PAN_CENTER_OF_TASK)."""
    return tuple(ICE_PAN_CENTER_OF_TASK.get(task, ICE_PAN["center"]))


def fillet_tray(task=None):
    """The fillet tray's spec in a task's scene (FILLET_TRAY at FILLET_TRAY_CENTER_OF_TASK)."""
    return dict(FILLET_TRAY, center=tuple(FILLET_TRAY_CENTER_OF_TASK.get(task, FILLET_TRAY["center"])))


def task_spec(name, task=None):
    """A bench item's layout spec in a task's scene ("FILLET_TRAY" and "ICE_PAN" move in some tasks)."""
    if name == "FILLET_TRAY":
        return fillet_tray(task)
    if name == "ICE_PAN":
        return dict(ICE_PAN, center=ice_pan_center(task))
    return globals()[name]


def frame_pan_inner(spec=None):
    """(centre xy, inner half extents at the rim xy, floor top z, rim z) of the frame pan (FRAME_PAN)."""
    spec = spec or FRAME_PAN
    (sx, sy, sz), r = spec["size"], spec["rim"]
    return (tuple(spec["center"]), (sx/2 - r, sy/2 - r), TABLE["top_z"] + spec["wall"], TABLE["top_z"] + sz)
