"""Start of pour-glass: the bottle of red wine (450-750 mL) stands on its coaster on the bar and an empty Bordeaux glass
stands at a seeded delivery spot (1 to 4).

setup(s) receives layout.Start (layout.task_layout calls it), appends the bottle on its coaster and the glass to s.items
and returns its goal (layout["goal"]): 150 mL (+- 15 mL) in the glass on its spot."""
from wine_pouring.layout import _bottle, _jit


def setup(s):
    rng, items, spots, bottle_xy = s.rng, s.items, s.spots, s.bottle_xy
    k = int(rng.integers(1, 5))
    fill = float(rng.uniform(450e-6, 750e-6))
    items += _bottle(rng, bottle_xy, fill)
    items.append(dict(name="glass", kind="glass_pure", xy=_jit(rng, spots[k]), role="target", spot=k))
    goal = dict(target="glass", volume=150e-6, tol=15e-6, spot=k)
    return goal
