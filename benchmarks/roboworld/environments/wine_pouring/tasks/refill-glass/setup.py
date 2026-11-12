"""Start of refill-glass: the bottle of red wine (450-700 mL) stands on its coaster on the bar and the guest's Bordeaux
glass, still holding 40-90 mL of red wine, stands at a seeded delivery spot (1 to 4).

setup(s) receives layout.Start (layout.task_layout calls it), appends the bottle on its coaster and the glass to s.items
and returns its goal (layout["goal"]): 150 mL (+- 15 mL) in the glass on its spot."""
from wine_pouring.layout import _bottle, _jit


def setup(s):
    rng, items, spots, bottle_xy = s.rng, s.items, s.spots, s.bottle_xy
    k = int(rng.integers(1, 5))
    items += _bottle(rng, bottle_xy, float(rng.uniform(450e-6, 700e-6)))
    items.append(dict(name="glass", kind="glass_pure", xy=_jit(rng, spots[k]), role="target", spot=k,
                      liquid="red_wine", volume=float(rng.uniform(40e-6, 90e-6))))
    goal = dict(target="glass", volume=150e-6, tol=15e-6, spot=k)
    return goal
