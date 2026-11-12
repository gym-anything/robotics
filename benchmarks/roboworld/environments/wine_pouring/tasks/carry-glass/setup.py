"""Start of carry-glass: a Bordeaux glass holding 150 mL of red wine stands on the bar near the robot, and a seeded
delivery spot (1 to 4) is where it goes.

setup(s) receives layout.Start (layout.task_layout calls it), appends the glass to s.items and returns its goal
(layout["goal"]): the glass delivered to its spot (deliver_spot) still holding its wine (keep)."""
from wine_pouring.layout import _jit


def setup(s):
    rng, items = s.rng, s.items
    k = int(rng.integers(1, 5))
    items.append(dict(name="glass", kind="glass_pure", xy=_jit(rng, (0.10, 0.14)), liquid="red_wine",
                      volume=150e-6, role="carried"))
    goal = dict(target="glass", deliver_spot=k, keep=150e-6)
    return goal
