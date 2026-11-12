"""Start of push-in-chairs: the six chairs of row 2 left pulled out 0.40-0.80 m behind their desks, up to 12 cm
sideways and turned up to 30 deg.

setup(s) builds on layout.Start (layout.setup); ROBOT is the robot's base_link start (x, y, yaw), parked in the
aisle beside rows 2-3, arms stowed, spine down."""
import math

ROBOT = (3.23, 4.45, math.pi/2)


def setup(s):
    items, rng, targets = s.items, s.rng, s.targets
    for c in range(1, 7):
        ch = f"chair_2{c}"
        x, y, _ = items[ch]["pose"]
        targets[ch] = items[ch]["pose"]
        s.place(ch, lambda: (x + rng.uniform(-0.12, 0.12), y + rng.uniform(0.40, 0.80), math.radians(rng.uniform(-30, 30))))
