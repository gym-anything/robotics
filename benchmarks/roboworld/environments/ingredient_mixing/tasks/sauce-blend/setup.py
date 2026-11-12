"""Start of sauce-blend: the small glass bowl holds mayonnaise spooned in and pushed about, with a squeeze of ketchup
and a spoonful of relish set on it side by side (honey variant: honey and mustard; pet-meal variant: canned food with
a powdered supplement sprinkled over one side of it).

setup(s) puts the ingredients in on a task.Start (task.start_state calls it) and sets s.fill and s.batch. STATION is
what the task puts on the counter (scene.task_layout), PROPS what stands beside the work (scene.Scene.props: kind, x
from the station's centre, y from the counter's front edge, yaw in degrees), RECIPE the recipe card's parts of
params.RECIPES and VARIANT_RECIPE a variant's own (dressing), MATERIAL the recipe's solid and pastes (task.TASKS) and
VARIANT_MATERIAL what a variant changes in them (task.task_spec)."""
import math

import numpy as np

from ingredient_mixing import materials as M
from ingredient_mixing.task import FLOUR, NONE, _Fill, blend

MAYO = M.Paste("mayonnaise", 990., 60., 40., 0.40, 2800., (0.96, 0.93, 0.80))
KETCHUP = M.Paste("ketchup", 1040., 25., 5., 0.44, 3500., (0.72, 0.08, 0.05))
RELISH = M.Paste("relish", 1100., 40., 20., 0.45, 3400., (0.42, 0.52, 0.16))
HONEY = M.Paste("honey", 1430., 0., 20., 0.95, 2300., (0.86, 0.60, 0.12))
MUSTARD = M.Paste("mustard", 1170., 12., 3., 0.45, 3400., (0.90, 0.75, 0.10))
PETFOOD = M.Paste("canned pet food", 1050., 200., 80., 0.40, 3500., (0.52, 0.34, 0.24))

STATION = dict(vessel="bowl2", tools=("spoon", "spatula", "whisk"))
PROPS = (("mayo_jar", -0.82, 0.52, 0), ("ketchup", -0.68, 0.54, 0), ("relish_jar", -0.57, 0.50, 0),
         ("towel", 1.10, 0.42, 6))
RECIPE = ("sauce",)
VARIANT_RECIPE = {"honey": ("sauce_honey",), "pet-meal": ("pet_meal",)}
MATERIAL = dict(solid=FLOUR, pastes=(MAYO, KETCHUP, RELISH))
VARIANT_MATERIAL = {"honey": dict(pastes=(MAYO, HONEY, MUSTARD)), "pet-meal": dict(pastes=(PETFOOD, NONE, NONE))}


def setup(s):
    config, v, sp, rng, L, R, scale = s.config, s.vessel, s.sp, s.rng, s.layout, s.recipes, s.scale
    volume_of, polar, a0, off = s.volume_of, s.polar, s.a0, s.off
    fill = _Fill(v, sp, rng, 0.9*v.depth, level=L.get("start") == "level")
    if config.variant == "honey":
        parts = [blend(dict(mayonnaise=R["sauce_honey"]["mayonnaise"])), blend(dict(honey=R["sauce_honey"]["honey"])),
                 blend(dict(mustard=R["sauce_honey"]["mustard"]))]
    elif config.variant == "pet-meal":
        parts = [blend(dict(canned=R["pet_meal"]["canned"])), blend(dict(supplement=R["pet_meal"]["supplement"]))]
    else:
        parts = [blend(dict(mayonnaise=R["sauce"]["mayonnaise"])), blend(dict(ketchup=R["sauce"]["ketchup"])),
                 blend(dict(relish=R["sauce"]["relish"]))]
    parts = [q*scale for q in parts]
    vols = [volume_of(q) for q in parts]
    # the first part spooned in and pushed about (it keeps the relief its yield stress holds) ...
    fill.spread(parts[0], vols[0], off, 0.003)
    if config.variant == "pet-meal":
        # ... the powder sprinkled over one side of it
        c = off + polar(0.030, a0)
        zs = fill.surface_at(c, 0.03)
        fill.take(parts[1], vols[1], (fill.z - zs) + 0.25*fill.dist(c) + fill.bumps(0.001, 0.02))
    else:
        # ... a squeeze of the second and a spoonful of the third set on it side by side
        rad = 0.38*float(np.interp(fill.surface_at(off), v.profile[1:, 1], v.profile[1:, 0]))
        for k in range(1, len(parts)):
            fill.dollop(parts[k], vols[k], off + polar(rad, a0 + (k - 1)*math.pi*(1.0 if len(parts) == 3 else 0.0)),
                        flat=1.3 if k == 1 else 1.7)
    s.fill, s.batch = fill, sum(parts)
