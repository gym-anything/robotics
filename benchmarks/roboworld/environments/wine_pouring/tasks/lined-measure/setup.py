"""Start of lined-measure: in the lined preset, the bottle of red wine (500-750 mL) stands on its coaster and an empty
lined glass, etched with its 175 mL line, stands at a seeded delivery spot (1 to 4).

setup(s) receives layout.Start (layout.task_layout calls it), appends the bottle on its coaster and the glass to s.items
and returns its goal (layout["goal"]): the glass filled to its etched line (line_175) on its spot."""
from wine_pouring.layout import _bottle, _jit


def setup(s):
    rng, items, spots = s.rng, s.items, s.spots
    k = int(rng.integers(1, 5))
    items += _bottle(rng, _jit(rng, (0.02, 0.15)), float(rng.uniform(500e-6, 750e-6)))
    items.append(dict(name="glass", kind="glass_lined", xy=_jit(rng, spots[k]), role="target", spot=k))
    goal = dict(target="glass", line="line_175", tol=5e-6, spot=k)
    return goal
