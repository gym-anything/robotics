"""Start of chairs-up: row 4 as the class left it, the six desks on their dots and their chairs tucked in.

setup(s) builds on layout.Start (layout.setup); ROBOT is the robot's base_link start (x, y, yaw), near the back of the
room, arms stowed, spine down."""
import math

ROBOT = (0.90, 7.90, math.pi)


def setup(s):
    for c in range(1, 7):
        d = f"desk_4{c}"
        s.targets[d] = s.items[d]["pose"]
