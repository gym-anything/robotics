"""Start of straighten-row: the six desks of row 3 have drifted 5-20 cm and 2-10 deg off their floor dots, each
chair standing behind its desk, pulled out and turned up to 8 deg.

setup(s) builds on layout.Start (layout.setup); ROBOT is the robot's base_link start (x, y, yaw), parked in the
aisle beside row 3, arms stowed, spine down."""
import math

from classroom_arrangement.furniture import USE_DY

ROBOT = (3.23, 4.45, math.pi/2)


def setup(s):
    items, rng, targets = s.items, s.rng, s.targets
    for c in range(1, 7):
        d, ch = f"desk_3{c}", f"chair_3{c}"
        targets[d] = items[d]["pose"]
        targets[ch] = items[ch]["pose"]
        items[ch]["pose"] = (items[ch]["pose"][0], items[ch]["pose"][1] + 0.6, 0.)     # out of the way while desks are drawn
    for c in range(1, 7):
        d, ch = f"desk_3{c}", f"chair_3{c}"
        x, y, yaw = s.jitter(d, 0.05, 0.20, 2, 10)
        s.place(ch, lambda: (x - math.sin(yaw)*USE_DY + rng.uniform(-0.05, 0.05), y + math.cos(yaw)*USE_DY + rng.uniform(-0.03, 0.08),
                             yaw + math.radians(rng.uniform(-8, 8))))
