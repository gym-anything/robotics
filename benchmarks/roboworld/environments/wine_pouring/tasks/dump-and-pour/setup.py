"""Start of dump-and-pour: in the tasting preset, a bottle of white wine (600-750 mL) stands on its coaster, an ISO
tasting glass still holds 20-60 mL of red wine from the last tasting, and the dump bucket stands beside them.

setup(s) receives layout.Start (layout.task_layout calls it), appends the bottle on its coaster, the glass and the
bucket to s.items and returns its goal (layout["goal"]): the old red wine dumped (at most old_max left in the glass) and
50 mL (+- 8 mL) of the fresh white wine in it."""
from wine_pouring.layout import _bottle, _jit


def setup(s):
    rng, items = s.rng, s.items
    items += _bottle(rng, _jit(rng, (0.30, 0.14)), float(rng.uniform(600e-6, 750e-6)), liquid="white_wine")
    items.append(dict(name="iso", kind="glass_iso", xy=_jit(rng, (0.0, 0.38), 0.01), role="target",
                      liquid="red_wine", volume=float(rng.uniform(20e-6, 60e-6))))
    items.append(dict(name="dump", kind="dump_bucket", xy=_jit(rng, (-0.32, 0.16)), role="dump"))
    goal = dict(target="iso", fresh="white_wine", old="red_wine", volume=50e-6, tol=8e-6, old_max=2e-6,
                dump="dump")
    return goal
