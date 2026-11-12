"""Start of split-bottle: a black-glass bottle with 250-450 mL of red wine left stands on its coaster on the bar, and
two empty Bordeaux glasses stand at two neighbouring delivery spots (the first seeded, 1 to 3).

setup(s) receives layout.Start (layout.task_layout calls it), appends the bottle on its coaster and the two glasses to
s.items and returns its goal (layout["goal"]): the two glasses (targets), their share tolerance and the most wine the
finished bottle may keep (leave_max)."""
from wine_pouring.layout import _bottle, _jit


def setup(s):
    rng, items, spots, bottle_xy = s.rng, s.items, s.spots, s.bottle_xy
    k = int(rng.integers(1, 4))
    fill = float(rng.uniform(250e-6, 450e-6))
    # black glass, so the wine left cannot be seen (a green bottle shows its level from the side). Appearance only: the same LEONOR geometry, mass and contact as every bottle.
    # LEONOR geometry in black glass: Saverglass sells a black glass (Onyx), but catalogues LEONOR itself only in
    # antique green and extra-white flint
    items += _bottle(rng, bottle_xy, fill, color="glass_black")
    items.append(dict(name="glass_a", kind="glass_pure", xy=_jit(rng, spots[k]), role="target", spot=k))
    items.append(dict(name="glass_b", kind="glass_pure", xy=_jit(rng, spots[k + 1]), role="target", spot=k + 1))
    goal = dict(targets=["glass_a", "glass_b"], share_tol=0.10, leave_max=15e-6)
    return goal
