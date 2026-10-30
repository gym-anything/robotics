"""Geometry of the sewing cell (preset A, sample and alteration room).

Frame: the origin is where the needle axis meets the throat-plate top; +x to the operator's right (toward the machine's
arm and handwheel), +y away from the operator (the feed direction), +z up. The bed is flush with the table top, so the
table top is z = 0 and the floor is at z = -0.762 m. Metres throughout.

Labels: [read] from a manufacturer drawing or a cited source, [estimate] authored where the source is silent (each is
listed in params.py), [authored] a layout choice.
"""

import math

import numpy as np

from . import params as P

TABLE_H = P.TABLE_H                                   # 0.762 m [read typical]
TX, TY, TZ = P.TABLE_TOP                              # 1.200 x 0.533 x 0.030 m [read]
TABLE = dict(x=(-P.NEEDLE_FROM_LEFT, -P.NEEDLE_FROM_LEFT + TX), y=(-P.NEEDLE_FROM_FRONT, -P.NEEDLE_FROM_FRONT + TY),
             thickness=TZ, leg=0.04)

# Juki DDL-9000C class head [read: bed 517 x 178 mm, needle-to-arm 303 mm; the rest estimate from the manufacturer's
# product photographs and the class's catalogue drawings: arm clearance, head and pillar section]
BED = dict(x=(-0.120, 0.397), y=(-0.089, 0.089))                     # flush with the table top
PILLAR = dict(x=(P.THROAT, P.THROAT + 0.090), y=(-0.058, 0.058), z=(0.0, 0.300))
ARM = dict(x=(-0.030, P.THROAT + 0.090), y=(-0.045, 0.045), z=(0.150, 0.300))
HEAD = dict(x=(-0.058, 0.055), y=(-0.052, 0.052), z=(0.095, 0.300))   # face plate side, needle bar housing
NEEDLE_BAR = dict(radius=4.0e-3, z_bottom_up=0.042)                  # bar end above the plate with the needle up
PRESSER_BAR = dict(x=(0.0010, 0.0120), y=(-0.006, 0.006), z=(0.012, 0.095))  # behind the needle, carries the foot
HANDWHEEL = dict(center=(P.THROAT + 0.105, 0.0, 0.205), radius=0.062, width=0.030)
FOOT_KEEP_OUT = dict(x=(-0.0095, 0.0095), y=(-0.016, 0.012), z=(0.0035, 0.012))  # the foot's shank above the sole

# robots: two arms on one pedestal in the operator's seat position, both yawed to face +y [authored].
# Reach preflight E0b: with the bases 0.42 m in front of the needle at table height, 5 of 110 hand-down key poses over the bed
# and the left table are unreachable and others sit on joint limits (joint 4 at -3.0 of -3.07 rad); 0.55 m in front and 0.20 m above the table (a seated operator's shoulder
# height) reaches all 110 with >= 0.46 rad of joint-limit margin (median 0.72).
ROBOT_BASES = {"left": dict(pos=(-0.30, -0.55, 0.20), yaw_deg=90.0),
               "right": dict(pos=(0.10, -0.55, 0.20), yaw_deg=90.0)}
PEDESTAL = dict(x=(-0.42, 0.22), y=(-0.65, -0.45), z=(-TABLE_H, 0.20))

# where the plies start on the bed for a seam (needle 10 mm from the right edge, 5 mm from the near end) [authored,
# task A1]
SEAM_ALLOWANCE = 0.010
LEAD = 0.008            # leading edge behind the needle at the start (under the foot, past the needle)

# cameras (pinhole, metres, degrees) [authored]
CAMERAS = {
    # the operator's eyes: above and between the shoulders (the arms' pedestal), looking down at the needle
    "operator": dict(eye=(-0.08, -0.72, 0.68), target=(-0.03, -0.06, 0.0), hfov=50.0),
    "needle": dict(eye=(-0.085, -0.085, 0.095), target=(0.0, -0.015, 0.0), hfov=62.0),   # D405 on the face plate
    # a D435-class cell camera on a boom above the table's left front, looking down at the work in front of the
    # needle: the operator view is mostly arms and head. The boom is not a collider (0.55 m above the table top, above
    # the arms' working heights over the stack)
    "overhead": dict(eye=(-0.17, -0.22, 0.55), target=(-0.06, -0.05, 0.0), hfov=48.0),
}
DIAGNOSTIC_CAMERAS = {
    "hero": dict(eye=(-0.82, -0.18, 0.50), target=(0.0, -0.08, 0.08), hfov=55.0),
    # from the left and behind the needle, looking back at the face plate, the needle and the robots
    "machine": dict(eye=(-0.55, 0.30, 0.45), target=(0.05, -0.08, 0.10), hfov=55.0),
    "macro": dict(eye=(-0.045, -0.060, 0.030), target=(0.0, 0.0, 0.0), hfov=40.0),
    # behind the foot, looking down at the sewn work as it leaves the needle (the stitches)
    "seam": dict(eye=(0.045, 0.075, 0.060), target=(-0.004, 0.012, 0.0), hfov=40.0),
    # the sample room from its front-right corner at a standing person's eye height (1.9 m above the floor)
    "room": dict(eye=(3.25, -2.30, 1.15), target=(-0.20, 0.45, -0.25), hfov=72.0),
    # the whole cell from behind the table's left end at standing height (1.7 m above the floor): the plies on the
    # table, the machine, both arms beyond it (from the robots' side the arms hide the work)
    "cell": dict(eye=(-1.50, 0.80, 0.95), target=(-0.05, -0.25, 0.05), hfov=52.0),
    # the work from the table's left side, low: the plies, the foot, the hands as they reach the work
    "work": dict(eye=(-0.62, 0.10, 0.30), target=(-0.06, -0.07, 0.0), hfov=50.0),
}


def yaw_quat(deg):
    a = math.radians(deg)
    return np.array([0.0, 0.0, math.sin(a/2), math.cos(a/2)])
