"""Start of exam-row: row 2 as the class left it; its six desks' targets are the exam dots, 1.25 m apart across the
room, each chair's 0.45 m behind its desk (in use).

setup(s) builds on layout.Start (layout.setup); ROBOT is the robot's base_link start (x, y, yaw), at the front of the
room, arms stowed, spine down."""
import math

from classroom_arrangement.furniture import USE_DY
from classroom_arrangement.layout import ROWS_Y

ROBOT = (3.23, 1.35, math.pi/2)
EXAM_X = (-3.125, -1.875, -0.625, 0.625, 1.875, 3.125)


def setup(s):
    targets = s.targets
    for c in range(1, 7):
        targets[f"desk_2{c}"] = (EXAM_X[c - 1], ROWS_Y[1], 0.)
        targets[f"chair_2{c}"] = (EXAM_X[c - 1], ROWS_Y[1] + USE_DY, 0.)
