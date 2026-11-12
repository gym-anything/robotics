"""Start of water-tumbler: a 1 L carafe with 500-800 mL of water stands on the bar and an empty tumbler stands at a
seeded delivery spot (1 to 4).

setup(s) receives layout.Start (layout.task_layout calls it), appends the carafe and the tumbler to s.items and returns
its goal (layout["goal"]): 250 mL (+- 25 mL) of water in the tumbler on its spot."""
from wine_pouring.layout import _jit


def setup(s):
    rng, items, spots = s.rng, s.items, s.spots
    k = int(rng.integers(1, 5))
    items.append(dict(name="carafe", kind="carafe_1l", xy=_jit(rng, (-0.10, 0.14)), liquid="water",
                      volume=float(rng.uniform(500e-6, 800e-6)), role="source"))
    items.append(dict(name="tumbler", kind="tumbler", xy=_jit(rng, spots[k]), role="target", spot=k))
    goal = dict(target="tumbler", volume=250e-6, tol=25e-6, spot=k)
    return goal
