"""Start of pancake-whisk: the 5 qt bowl holds beaten egg, milk and oil with the dry ingredients heaped in the middle
(the spelt variant: whole spelt flour in place of the wheat flour); the whisk, spatula and spoon stand in the crock.

setup(s) puts the ingredients in on a task.Start (task.start_state calls it) and sets s.fill and s.batch. STATION is
what the task puts on the counter (scene.task_layout), PROPS what stands beside the work (scene.Scene.props: kind, x
from the station's centre, y from the counter's front edge, yaw in degrees), RECIPE the recipe card's parts of
params.RECIPES (dressing), MATERIAL the recipe's solid and pastes (task.TASKS) and VARIANT_MATERIAL what a variant
changes in them (task.task_spec)."""
from dataclasses import replace

from ingredient_mixing.task import FLOUR, NONE, _Fill, blend

SPELT = replace(FLOUR, name="whole spelt flour", law_b=-2.43, n_slope=2.52)

STATION = dict(vessel="bowl5", tools=("whisk", "spatula", "spoon"))
PROPS = (("flour_bag", -0.87, 0.53, -5), ("sugar_jar", -0.73, 0.515, 0), ("baking_tin", -0.61, 0.51, 0),
         ("milk_carton", 0.90, 0.54, 10), ("egg_carton", 1.08, 0.50, 5), ("oil_bottle", 1.25, 0.55, 0),
         ("jug", 1.30, 0.40, 20))
RECIPE = ("pancake_dry", "pancake_wet")
MATERIAL = dict(solid=FLOUR, pastes=(NONE, NONE, NONE))
VARIANT_MATERIAL = {"spelt": dict(solid=SPELT)}


def setup(s):
    v, sp, rng, L, R, scale = s.vessel, s.sp, s.rng, s.layout, s.recipes, s.scale
    volume_of, off = s.volume_of, s.off
    fill = _Fill(v, sp, rng, 0.9*v.depth, level=L.get("start") == "level")
    # the dry ingredients heaped in the middle of the beaten egg, milk and oil
    dry = blend(R["pancake_dry"])*scale
    wet = blend(R["pancake_wet"])*scale
    vd, vw = volume_of(dry), volume_of(wet)
    fill.mound(dry, vd, off, 0.55)
    fill.pool(wet, vw)
    s.fill, s.batch = fill, dry + wet
