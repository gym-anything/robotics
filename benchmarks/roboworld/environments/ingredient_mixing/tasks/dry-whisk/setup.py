"""Start of dry-whisk: the 5 qt bowl holds a mound of flour poured from the measuring cup, with the sugar and the
leavening with the salt tipped onto its flanks; the flour, sugar, baking powder and salt stand beside the work.

setup(s) puts the ingredients in on a task.Start (task.start_state calls it) and sets s.fill and s.batch. STATION is
what the task puts on the counter (scene.task_layout), PROPS what stands beside the work (scene.Scene.props: kind, x
from the station's centre, y from the counter's front edge, yaw in degrees), RECIPE the recipe card's parts of
params.RECIPES (dressing) and MATERIAL the recipe's solid and pastes (task.TASKS)."""
from ingredient_mixing.task import FLOUR, NONE, _Fill, blend

STATION = dict(vessel="bowl5", tools=("whisk", "spatula", "spoon"))
PROPS = (("flour_bag", -0.86, 0.52, 8), ("sugar_jar", -0.71, 0.53, 0), ("baking_tin", -0.60, 0.50, 0),
         ("salt", -0.50, 0.57, 0), ("scale", 1.05, 0.47, 4), ("towel", 1.30, 0.40, -12))
RECIPE = ("dry_mix",)
MATERIAL = dict(solid=FLOUR, pastes=(NONE, NONE, NONE))


def setup(s):
    v, sp, rng, L, R, scale = s.vessel, s.sp, s.rng, s.layout, s.recipes, s.scale
    volume_of, polar, a0, off = s.volume_of, s.polar, s.a0, s.off
    # flour poured from the measuring cup, then the sugar and the leavening with the salt tipped onto its flanks
    flour = blend(dict(flour=R["dry_mix"]["flour"]))*scale
    sugar = blend(dict(sugar=R["dry_mix"]["sugar"]))*scale
    minor = blend(dict(baking_powder=R["dry_mix"]["baking_powder"], salt=R["dry_mix"]["salt"]))*scale
    fill = _Fill(v, sp, rng, 0.9*v.depth, level=L.get("start") == "level")
    fill.mound(flour, volume_of(flour), off, 0.50)
    fill.mound(sugar, volume_of(sugar), off + polar(0.052, a0), 0.60, round_r=0.006, bump=0.0008)
    fill.mound(minor, volume_of(minor), off + polar(0.048, a0 + 2.3), 0.60, round_r=0.006, bump=0.0008)
    s.fill, s.batch = fill, flour + sugar + minor
