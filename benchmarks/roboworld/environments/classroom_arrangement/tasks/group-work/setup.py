"""Start of group-work: rows 1 and 2 as the class left them, their twelve chairs pulled out 0.48-0.66 m from tucked,
up to 3 cm sideways and turned up to 10 deg; the targets are the three pods of four (layout.pod_targets).

setup(s) builds on layout.Start (layout.setup); ROBOT is the robot's base_link start (x, y, yaw), at the front of the
room, arms stowed, spine down."""
import math

from classroom_arrangement.layout import pod_targets

ROBOT = (0.60, 1.35, math.pi)


def setup(s):
    items, rng = s.items, s.rng
    for pair in range(3):
        s.targets.update(pod_targets(pair))
    # the rows 1-2 chairs where the students left them, pulled out 0.48-0.66 m from tucked (seat front 0.06-0.24 m
    # beyond the desk edge), up to 3 cm sideways and 10 deg turned: clear of the desk's rear legs (a turned chair
    # nearer than ~0.45 m puts its seat through them, which layout.conflicts does not see) and at most 3 mm wider
    # than the desk each side, so the aisles keep >= 0.922 m.
    for r in (1, 2):
        for c in range(1, 7):
            ch = f"chair_{r}{c}"
            x, y, yaw = items[ch]["pose"]
            s.place(ch, lambda: (x + rng.uniform(-0.03, 0.03), y + rng.uniform(0.48, 0.66),
                                 yaw + math.radians(rng.uniform(-10, 10))))
