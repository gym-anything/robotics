"""Start of cake-mixer: the 5 qt bowl holds cake mix tipped in, water and oil poured round it and three eggs broken
onto it; the hand mixer stands on its heel beside the bowl, switched off.

setup(s) puts the ingredients in on a task.Start (task.start_state calls it) and sets s.fill and s.batch. STATION is
what the task puts on the counter (scene.task_layout; mixer: the hand mixer), PROPS what stands beside the work
(scene.Scene.props: kind, x from the station's centre, y from the counter's front edge, yaw in degrees), RECIPE the
recipe card's parts of params.RECIPES (dressing) and MATERIAL the recipe's solid and pastes (task.TASKS)."""
from dataclasses import replace

from ingredient_mixing.task import FLOUR, NONE, _Fill, blend

CAKE = replace(FLOUR, name="cake mix", rho=1500., void=1.5, law_b=-4.92, n_slope=0.85, yield_x0=0.5, yield_x1=0.8)

STATION = dict(vessel="bowl5", tools=("spatula", "whisk", "spoon"), mixer=True)
PROPS = (("cake_box", -0.80, 0.535, 4), ("egg_carton", 0.97, 0.50, -6), ("oil_bottle", 1.15, 0.55, 0),
         ("jug", 1.30, 0.44, 35))
RECIPE = ("cake",)
MATERIAL = dict(solid=CAKE, pastes=(NONE, NONE, NONE))


def setup(s):
    v, sp, rng, L, R, scale = s.vessel, s.sp, s.rng, s.layout, s.recipes, s.scale
    volume_of, polar, a0, off = s.volume_of, s.polar, s.a0, s.off
    fill = _Fill(v, sp, rng, 0.9*v.depth, level=L.get("start") == "level")
    # the cake mix tipped in; water and oil poured round it; three eggs broken onto it
    dry = blend(dict(mix=R["cake"]["mix"]))*scale
    pour = blend(dict(water=R["cake"]["water"], oil=R["cake"]["oil"]))*scale
    egg = blend(dict(egg=R["cake"]["egg"]))*scale
    vd, vw, ve = volume_of(dry), volume_of(pour), volume_of(egg)
    fill.mound(dry, vd, off, 0.50)
    fill.pool(pour, vw)
    for k in range(3):
        fill.dollop(egg/3, ve/3, off + polar(0.030, a0 + k*2.1), flat=2.2)
    wet = pour + egg
    s.fill, s.batch = fill, dry + wet
