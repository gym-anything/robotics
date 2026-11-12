"""Start of move-teacher-desk: the room as laid out; the teacher's desk's target is a quarter turn on its floor dots
by the window wall, its chair behind it on the room side.

setup(s) builds on layout.Start (layout.setup); ROBOT is the robot's base_link start (x, y, yaw), beside the
teacher's desk, arms stowed, spine down."""
import math

ROBOT = (-0.80, 1.47, math.pi)


def setup(s):
    # a quarter turn: the modesty panel faces the windows, the teacher sits on the room side facing them
    s.targets["teacher_desk"] = (-2.95, 1.30, math.pi/2)
    s.targets["teacher_chair"] = (-2.95 + 0.381 + 0.30, 1.30 - 0.18, -math.pi/2)
