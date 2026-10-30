"""Robotic GMAW cell, preset A ('steel-cell'): authored placements of the table, robots, coupon, clamps and cameras.
World frame: z up, origin on the floor under the fixture table's centre, x along the table.

Sources: [read] = taken from a maker's datasheet or the cited reference; [authored] = placed by this build (a real cell
is laid out this way, the numbers are chosen); [derived] = computed from read values.
"""

import math

import numpy as np

# Siegmund System 16 class fixture table [read: 1200 x 800 mm top, 850 mm working height, 12 mm top; holes 16 mm on a
# 50 mm grid]; legs 80 mm square tube [authored]
TABLE = dict(size=(1.2, .8), top_z=.850, top_thickness=.012, leg=.08, center=(0., 0.))

# robots on the table's back corners (as cobot welding carts mount them) [authored]: the torch arm
# (UR10e) back-right, the handling arm (FR3) back-left; both face the front (-y)
ROBOT_BASES = {"right": dict(pos=(.42, .26, .850), yaw_deg=-90.),
               "left": dict(pos=(-.42, .26, .850), yaw_deg=-90.)}

# the coupon (flagship A2): A36 base plate 200 x 150 x 10 mm and bracket 150 x 75 x 8 mm; the weld
# frame (welding/process.py: x along the weld, y out of the bracket's weld face, z up) has its origin at the root of
# the near-side joint's start. Placed so the plate is centred on the table 6 cm in front of its centre line and the
# weld face looks toward the robots (+y) [authored]
WELD_ORIGIN = np.array([-.075, -.060, TABLE["top_z"] + .010])

# DESTACO 225-U class hold-down toggle clamps [read: 2.2 kN holding capacity, neoprene spindle cap] on the table in
# front of the plate, their spindle pads on the plate's front margin [authored placement]. Pad force: the spindle is
# set to press with CLAMP_FORCE when locked (a fraction of the holding capacity) [authored estimate]
CLAMPS = [dict(name="clamp_left", pad_xy=(-.06, -.118)), dict(name="clamp_right", pad_xy=(.06, -.118))]
CLAMP_FORCE = 400.            # N per clamp
CLAMP_PAD = dict(radius=.012, height=.010)     # neoprene cap ~24 mm [estimate from catalogue drawings]

# magnetic ground (work) clamp, Strong Hand GM205 class [read: 500 A, 490 N pull]: on the table top front-right
WORK_CLAMP = dict(pos=(.30, -.22), size=(.075, .060, .040), mass=1.3, pull_n=490.)

# welding power source on its running gear at the table's right end, front panel toward the aisle (-y), the wire
# feeder on its top at the back, the shielding-gas cylinder chained in the gear's rack behind it. Dimensions: Miller
# Auto-Continuum 350 source [read: Miller spec sheet AU/10.0, Mar. 2022, p. 8: H 27.19 x W 17.5 x D 28.22 in = 691 x
# 444 x 717 mm, 130 lb = 59.4 kg; Miller's web page lists 27.2 x 17.8 x 27.9 in, 127 lb, as for the Continuum 350];
# Continuum class feeder [read: H 13.81 x W 16.31 in = 351 x 414 mm; its length
# is not published: 600 mm, estimate]; size-300 cylinder [read: 9.25 x 55 in = 235 x 1397 mm]; running gear deck at
# 160 mm and its cylinder platform at 50 mm [estimate]. The torch arm can reach all of it: each part is a static
# collider (scene.py), drawn as the same boxes and cylinder (shop.py) [authored placement]
SOURCE = dict(x=1.10, front_y=-.20, deck_z=.16, case=(.444, .717, .691), feeder=(.414, .600, .351),
              cylinder=dict(d=.235, h=1.397, z0=.05, gap=.02))


def source_boxes():
    """Static collider boxes of the welding equipment: (name, center, half extents)."""
    S = SOURCE
    w, l, h = S["case"]
    fw, fl, fh = S["feeder"]
    y0, z0 = S["front_y"], S["deck_z"]
    top = z0 + h
    cyl = S["cylinder"]
    yc = y0 + l + cyl["gap"] + cyl["d"]/2
    ya, yb = y0 - .04, yc + cyl["d"]/2 + .02                    # the gear's deck, a lip in front, the rack behind
    return [("source_gear", (S["x"], (ya + yb)/2, z0/2), (.25, (yb - ya)/2, z0/2)),
            ("source_case", (S["x"], y0 + l/2, z0 + h/2), (w/2, l/2, h/2)),
            ("wire_feeder", (S["x"], y0 + l - fl/2, top + fh/2), (fw/2, fl/2, fh/2)),
            ("gas_cylinder", (S["x"], yc, cyl["z0"] + cyl["h"]/2), (cyl["d"]/2, cyl["d"]/2, cyl["h"]/2))]


def source_top():
    """The power source's exposed steel top in front of the feeder: (x0, x1, y0, y1, z)."""
    S = SOURCE
    w, l, h = S["case"]
    return (S["x"] - w/2, S["x"] + w/2, S["front_y"], S["front_y"] + l - S["feeder"][1], S["deck_z"] + h)

# orange transparent welding screens (EN ISO 25980 class, 6 x 6 ft frames) around the cell, outside the sweeps
SCREENS = [dict(center=(0., -1.15), yaw_deg=0., width=1.83, height=1.83),
           dict(center=(-1.25, -.20), yaw_deg=90., width=1.83, height=1.83),
           dict(center=(1.55, -.40), yaw_deg=90., width=1.83, height=1.83)]

# cameras [authored]: bench camera on a boom outside the screens looking down the seam; side camera at joint level;
# weld camera on the gun body (render.py places it from the torch frame)
CAMERAS = {
    # policy views: a boom camera above and behind the seam (between the arms' bases, above their sweeps), and a side
    # camera at joint height looking along the seam from its start end
    # the bench camera looks straight down onto the joint from 0.94 m above it: from above the line between the two
    # robot bases, eye (-0.03, 0.26, 1.62), the 855 mm FR3s reach the seam with their elbows under it and their links
    # fill four fifths of the frame; from above it is clear on every deployment rendered
    "bench": dict(eye=(0., -.02, 1.80), target=(0., -.065, .865), hfov=40., role="policy"),
    "side": dict(eye=(-.40, .07, .93), target=(.06, -.055, .868), hfov=40., role="policy"),
    # diagnostic views
    "hero": dict(eye=(.85, -1.05, 1.45), target=(.02, .06, .93), hfov=55., role="diagnostic"),
    "seam": dict(eye=(-.045, .085, .925), target=(-.045, -.058, .864), hfov=40., role="diagnostic"),
}
# weld camera on the gun body (tool0 frame of the UR10e: x across, z along the flange axis): beside the neck, looking
# at the work point [authored, Xiris-class process camera placement]
# process camera on the gun (Xiris XVC class): rigid on the torch, trailing it low along the seam on the plate side;
# its place is given relative to the torch work point at the procedure's angles (45 deg work, 7.5 deg push) in the weld
# frame (x along the weld, y out of the bracket face, z up), so it rides the same way on either gun. Chosen by line of
# sight: from 110 mm, 75 deg behind and 20 deg up, the arc column, the pool and the bead behind clear the nozzle by
# > 2 mm (beside the neck it sits above the bracket's top edge and sees only the bracket; 60 mm out over the plate the
# nozzle hides the arc) [authored]
WELD_CAMERA = dict(eye_work=(-.100, .027, .038), procedure_angles=(45., 7.5), hfov=20.)
# camera mounts [authored]: the bench camera hangs on a 30 mm drop rod from a strut across the extraction canopy's
# mouth (z 2.8 m; a mast standing on the table between the bases would overlap the UR10e's forearm in its start pose by
# 71 mm); the side camera sits on a magnetic base on the table top. Each camera body is a 60 mm box
# just behind its lens (the eye looks out of the box's face). Within the arms' reach, so static colliders (scene.py)
# of the drawn shapes: (name, center, half extents)
def _behind(cam, back=.05):
    e, t = np.asarray(CAMERAS[cam]["eye"], float), np.asarray(CAMERAS[cam]["target"], float)
    return tuple(e - (t - e)/np.linalg.norm(t - e)*back)


_BC, _SC = _behind("bench"), _behind("side")
CANOPY_STRUT_Z = 2.8
CAMERA_MOUNTS = [("bench_drop_rod", (_BC[0], _BC[1], (_BC[2] + .03 + CANOPY_STRUT_Z)/2),
                  (.015, .015, (CANOPY_STRUT_Z - _BC[2] - .03)/2)),
                 ("bench_camera_body", _BC, (.03, .03, .03)),
                 ("side_base", (_SC[0], _SC[1], .850 + .01), (.03, .03, .01)),
                 ("side_post", (_SC[0], _SC[1], (.870 + _SC[2] - .022)/2), (.008, .008, (_SC[2] - .022 - .870)/2)),
                 ("side_camera_body", _SC, (.03, .025, .022))]

# welding screens: the vinyl curtain and its frame within the arms' reach are static colliders [authored]
SCREEN_THICKNESS = .02


def screen_boxes():
    """Static collider boxes of the welding screens: (name, center, half extents, yaw_deg)."""
    return [(f"screen_{k}", (S["center"][0], S["center"][1], .25 + S["height"]/2),
             (S["width"]/2, SCREEN_THICKNESS/2, S["height"]/2), S["yaw_deg"]) for k, S in enumerate(SCREENS)]


def yaw_quat(deg):
    a = math.radians(deg)/2
    return np.array([0., 0., math.sin(a), math.cos(a)])        # xyzw


def weld_to_world(p):
    """Weld-frame point -> world, with the coupon at its authored pose (the live pose comes from the base plate body)."""
    return WELD_ORIGIN + np.asarray(p, float)
