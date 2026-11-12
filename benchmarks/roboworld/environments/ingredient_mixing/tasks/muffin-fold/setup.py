"""Start of muffin-fold: the 5 qt bowl holds the blended dry ingredients with a well made in the middle and the beaten
milk, egg and oil poured into the well; the spatula, whisk and spoon stand in the crock.

setup(s) puts the ingredients in on a task.Start (task.start_state calls it) and sets s.fill and s.batch. STATION is
what the task puts on the counter (scene.task_layout), PROPS what stands beside the work (scene.Scene.props: kind, x
from the station's centre, y from the counter's front edge, yaw in degrees), RECIPE the recipe card's parts of
params.RECIPES (dressing) and MATERIAL the recipe's solid and pastes (task.TASKS)."""
from ingredient_mixing.task import FLOUR, NONE, _Fill, blend

STATION = dict(vessel="bowl5", tools=("spatula", "whisk", "spoon"))
PROPS = (("flour_bag", -0.87, 0.53, 6), ("sugar_jar", -0.73, 0.515, 0), ("baking_tin", -0.62, 0.50, 0),
         ("milk_carton", 0.90, 0.54, 12), ("egg_carton", 1.07, 0.50, -8), ("oil_bottle", 1.24, 0.55, 0),
         ("jug", 1.32, 0.40, 30))
RECIPE = ("dry_mix", "muffin_wet")
MATERIAL = dict(solid=FLOUR, pastes=(NONE, NONE, NONE))


def setup(s):
    config, v, sp, rng, L, R, scale = s.config, s.vessel, s.sp, s.rng, s.layout, s.recipes, s.scale
    volume_of, off = s.volume_of, s.off
    fill = _Fill(v, sp, rng, 0.9*v.depth, level=L.get("start") == "level")
    # the blended dry ingredients with a well made in the middle; the beaten liquid poured into the well
    dry = blend(R["dry_mix"])*scale
    wet = blend(R["muffin_wet"])*scale
    vd, vw = volume_of(dry), volume_of(wet)
    rw = 0.082 + (rng.uniform(-0.006, 0.006) if config.seed else 0.)
    fill.ring(dry, vd, off, rw, 0.50)
    fill.pool(wet, vw, off, rw)
    s.fill, s.batch = fill, dry + wet
