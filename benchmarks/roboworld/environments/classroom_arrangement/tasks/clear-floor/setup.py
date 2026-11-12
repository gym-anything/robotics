"""Start of clear-floor: the book tote left in the left-hand aisle, the wastebasket beside the middle desks and the
row-3 chair of column 6 pulled out 0.2-0.5 m into the right-hand aisle, turned across it.

setup(s) builds on layout.Start (layout.setup); ROBOT is the robot's base_link start (x, y, yaw), at the front of the
room, arms stowed, spine down."""
import math

ROBOT = (3.23, 1.35, math.pi/2)


def setup(s):
    items, rng, targets = s.items, s.rng, s.targets
    s.place("tote", lambda: (-1.075 + rng.uniform(-0.1, 0.1), 4.50 + rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)))
    s.place("basket", lambda: (1.075 + rng.uniform(-0.1, 0.1), 3.10 + rng.uniform(-0.3, 0.3), 0.))
    x, y, _ = items["chair_36"]["pose"]
    targets["chair_36"] = items["chair_36"]["pose"]
    s.place("chair_36", lambda: (3.235 + rng.uniform(-0.04, 0.04), y + rng.uniform(0.2, 0.5), math.pi/2 + rng.uniform(-0.5, 0.5)))
