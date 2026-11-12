"""Start of stir-sauce: the saucepan on the hob holds a cold slurry of milk, corn starch and sugar that has stood for a
few minutes, so the top layer has given its starch to the bottom 4 mm; the hob is on (level 12, one either side by
seed), the stirrer, whisk and probe thermometer stand in the crock.

setup(s) lays the sauce on a task.Start (task.start_state calls it) and sets s.points, s.species, s.batch and s.temp.
STATION is what the task puts on the counter (scene.task_layout; hob: the knob's level), PROPS what stands beside the
work (scene.Scene.props: kind, x from the station's centre, y from the counter's front edge, yaw in degrees), RECIPE
the recipe card's parts of params.RECIPES (dressing) and MATERIAL the recipe's solid and pastes (task.TASKS)."""
from dataclasses import replace

import numpy as np

from ingredient_mixing import geometry as G
from ingredient_mixing.task import FLOUR, NONE, _lattice, blend

STARCH = replace(FLOUR, name="corn starch", rho=1500., void=1.5, grain=15e-6, starch=1.0, gel_a=4.1e20, hyd_gain=0.,
                 work_scale=0.)

# hob: the knob's position at the start (params.HOB levels_w: 12 is the manual's level 6.0, 1000 W; 7 is 3.5, 580 W)
STATION = dict(vessel="pan", tools=("stirrer", "whisk", "thermometer"), hob=12)
PROPS = (("milk_carton", -1.48, 0.54, 10), ("starch_box", -1.33, 0.55, -6), ("sugar_jar", -1.18, 0.53, 0),
         ("jug", -1.05, 0.42, 25))
RECIPE = ("stir_sauce",)
MATERIAL = dict(solid=STARCH, pastes=(NONE, NONE, NONE))


def setup(s):
    config, v, sp, vp, L, R, scale = s.config, s.vessel, s.sp, s.vp, s.layout, s.recipes, s.scale
    volume_of = s.volume_of
    liquid = blend(dict(milk=R["stir_sauce"]["milk"], sugar=R["stir_sauce"]["sugar"]))*scale
    starch = blend(dict(starch=R["stir_sauce"]["starch"]))*scale
    s.temp = L.get("milk_c", 20.0)
    total = liquid + starch
    level = G.level_for(v.profile, volume_of(total))
    pts = _lattice(v, sp, level, seed=config.seed)
    z = pts[:, 2]
    # settled for stand_min minutes at the hindered Stokes speed (1.7 mm/min): the top layer has given its starch
    # to the bottom 4 mm
    clear = min(0.8*level, 1.7e-3*L.get("stand_min", 2.0))
    w = np.where(z > level - clear, 0.0, 1.0)
    bottom = z < 0.004
    w[bottom] += (w.size - w.sum())/max(1, bottom.sum())
    comps = np.tile(liquid/volume_of(total)*vp, (len(pts), 1))
    comps += np.outer(w, starch/volume_of(total)*vp)
    s.points, s.species, s.batch = pts, comps, total
