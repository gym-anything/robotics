"""Start of pudding-whisk: the 5 qt bowl holds two cups of milk (at the layout's milk temperature) with a packet of
instant pudding mix heaped in the middle; the whisk, spatula and spoon stand in the crock.

setup(s) puts the ingredients in on a task.Start (task.start_state calls it) and sets s.fill, s.batch and s.temp.
STATION is what the task puts on the counter (scene.task_layout), PROPS what stands beside the work
(scene.Scene.props: kind, x from the station's centre, y from the counter's front edge, yaw in degrees), RECIPE the
recipe card's parts of params.RECIPES (dressing) and MATERIAL the recipe's solid and pastes (task.TASKS)."""
from dataclasses import replace

from ingredient_mixing.task import FLOUR, NONE, _Fill, blend

PUDDING = replace(FLOUR, name="instant pudding mix", rho=1550., void=0.94, grain=150e-6, law_b=-3.95, yield_x0=0.5,
                  yield_x1=0.8, hyd_gain=0., set_tau=100., set_k=20., set_n=0.4, set_yield=30., set_xref=0.1745,
                  starch=0.25, friction_deg=35., work_scale=0.)

STATION = dict(vessel="bowl5", tools=("whisk", "spatula", "spoon"))
PROPS = (("pudding_box", -0.62, 0.50, 20), ("milk_carton", -0.80, 0.54, -8), ("jug", 1.00, 0.48, 25),
         ("towel", 1.28, 0.42, 10))
RECIPE = ("pudding",)
MATERIAL = dict(solid=PUDDING, pastes=(NONE, NONE, NONE))


def setup(s):
    v, sp, rng, L, R, scale = s.vessel, s.sp, s.rng, s.layout, s.recipes, s.scale
    volume_of, off = s.volume_of, s.off
    fill = _Fill(v, sp, rng, 0.9*v.depth, level=L.get("start") == "level")
    # the packet of pudding mix tipped into the cold milk
    dry = blend(dict(mix=R["pudding"]["mix"]))*scale
    wet = blend(dict(milk=R["pudding"]["milk"]))*scale
    s.temp = L.get("milk_c", 20.0)
    vd, vw = volume_of(dry), volume_of(wet)
    fill.mound(dry, vd, off, 0.55)
    fill.pool(wet, vw)
    s.fill, s.batch = fill, dry + wet
