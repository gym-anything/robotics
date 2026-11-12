"""Start of seat-tutoring-group: the four tutoring chairs standing loose 0.95-1.75 m in front of the tutoring table,
anywhere along it and turned any way; their targets are the four places at the table, seat fronts 0.275 m from its
edge.

setup(s) builds on layout.Start (layout.setup); ROBOT is the robot's base_link start (x, y, yaw), near the back of
the room, arms stowed, spine down."""
import math

from classroom_arrangement.layout import TABLE, tutoring_chairs_at_table

ROBOT = (0.90, 8.30, math.pi)


def setup(s):
    items, rng, targets = s.items, s.rng, s.targets
    tx, ty, _ = TABLE
    goal = tutoring_chairs_at_table(pulled=0.39)      # seat front 0.275 m from the edge
    for k in range(1, 5):
        items[f"tchair_{k}"]["pose"] = (-3.0 + 0.6*k, 0.4, 0.)          # parked out of the way while drawing
    for k in range(1, 5):
        n = f"tchair_{k}"
        s.place(n, lambda: (tx + rng.uniform(-0.8, 2.0), ty + rng.uniform(-1.75, -0.95), rng.uniform(-math.pi, math.pi)))
        targets[n] = goal[n]["pose"]
