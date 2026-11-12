"""Start of reheat-porridge: the saucepan on the hob holds cold cooked porridge (5 C) spooned out of its tub, lying as
it fell; the hob is on (level 7, one either side by seed), the stirrer, spoon and probe thermometer stand in the crock.

setup(s) puts the porridge in on a task.Start (task.start_state calls it) and sets s.fill, s.batch and s.temp.
STATION is what the task puts on the counter (scene.task_layout; hob: the knob's level), PROPS what stands beside the
work (scene.Scene.props: kind, x from the station's centre, y from the counter's front edge, yaw in degrees), RECIPE
the recipe card's parts of params.RECIPES (dressing) and MATERIAL the recipe's solid and pastes (task.TASKS)."""
from ingredient_mixing import materials as M
from ingredient_mixing.task import FLOUR, NONE, _Fill, blend

PORRIDGE = M.Paste("oat porridge", 1060., 30., 30., 0.45, 3700., (0.86, 0.80, 0.68))

# hob: the knob's position at the start (params.HOB levels_w: 12 is the manual's level 6.0, 1000 W; 7 is 3.5, 580 W)
STATION = dict(vessel="pan", tools=("stirrer", "spoon", "thermometer"), hob=7)
PROPS = (("tub", -1.40, 0.50, 8), ("milk_carton", -1.20, 0.55, -10), ("towel", -1.02, 0.40, 14))
RECIPE = ("porridge",)
MATERIAL = dict(solid=FLOUR, pastes=(PORRIDGE, NONE, NONE))


def setup(s):
    v, sp, rng, L, R, scale, volume_of, off = s.vessel, s.sp, s.rng, s.layout, s.recipes, s.scale, s.volume_of, s.off
    # yesterday's porridge spooned out of its tub into the pan: it lies as it fell
    por = blend(dict(porridge=R["porridge"]["porridge"]))*scale
    s.temp = 5.0
    fill = _Fill(v, sp, rng, 0.9*v.depth, level=L.get("start") == "level")
    fill.spread(por, volume_of(por), off*0.5, 0.006, slope=0.05, scale=0.045)
    s.fill, s.batch = fill, por
