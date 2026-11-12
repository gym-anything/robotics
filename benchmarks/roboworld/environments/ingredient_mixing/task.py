"""Tasks: instructions, recipes as material, seeded start states and the outcome evaluator.

Every criterion is measured from the material points and the rigid state, never from a "mixed" flag or the path a tool
took. A wrong outcome is a valid result; a non-finite state or a tripped guard is a backend fault and is reported as not
ready (unified contract, invariant 10).
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation

from gym_anything_robotics import task_folders
from . import params as P
from . import geometry as G
from . import materials as M
from .materials import AQ, FAT, SUGAR, MINOR, SOLID_D, SOLID_L, SOLID_S, SOLID_C, PASTE_A, PASTE_B, PASTE_C, NS

# Work one fold stroke of the muffin method puts into the batter where its wetted flour is (J per kg, the points'
# plastic and viscous work averaged over the wetted flour's mass), taken over the method as it is carried out: from
# the dry mix and the liquid in its well to "just moistened", a spatula drawn 160 mm across the 5 qt bowl's floor at
# 0.2 m/s, each stroke in a new direction (E15 muffin, a phantom carriage). Measured: 0.30 J/kg a
# stroke over the first 12 strokes, 0.33 over 16 and 0.34 over 17; 0.33 is used. The over-mixing
# limit is 25 of these. (A stroke through batter that is already mixed puts in only 0.20 J/kg: the first-wetted flour of
# a batter that is being combined is worked by every stroke and takes more, and with that reference a cook who followed
# the method's 15 to 20 strokes would read as having made 26 to 34.)
REFERENCE_STROKE_J_KG = 0.33
# Gluten: the work at which a flour batter has stiffened by a tenth is 25 reference strokes' worth
# (1 - exp(-25 w_ref / w_d) = 0.1), an estimate of how much an over-mixed batter tightens
FLOUR = M.Solid(work_scale=25.0*REFERENCE_STROKE_J_KG/0.10536)
NONE = M.Paste()

MILK = {AQ: 0.965, FAT: 0.035}
EGG = {AQ: 1.0 - P.EGG_FAT, FAT: P.EGG_FAT}
PARTS = dict(flour={SOLID_D: 1.}, sugar={SUGAR: 1.}, baking_powder={MINOR: 1.}, salt={MINOR: 1.}, milk=MILK, egg=EGG,
             oil={FAT: 1.}, water={AQ: 1.}, mix={SOLID_D: 1.}, starch={SOLID_S: 1.}, mayonnaise={PASTE_A: 1.},
             ketchup={PASTE_B: 1.}, relish={PASTE_C: 1.}, honey={PASTE_B: 1.}, mustard={PASTE_C: 1.},
             canned={PASTE_A: 1.}, supplement={MINOR: 1.}, porridge={PASTE_A: 1.})


def blend(grams):
    """Species masses (kg) of a set of ingredients {name: grams}."""
    v = np.zeros(NS)
    for name, g in grams.items():
        for k, f in PARTS[name].items():
            v[k] += 1e-3*g*f
    return v


# criteria: name -> (tasks' threshold table is in TASKS[...]["criteria"], from each task's verifier.py JUDGING).
# Where the limits come from:
#   uniform cv 0.10   the feed industry's mixer test: a coefficient of variation of a tracer under 10 % is "excellent"
#                     (Herrman and Behnke 1994, in Kansas State MF3393, opened; the pharmacopoeia's content uniformity
#                     is 6 %). No baking standard for leavening or salt was found.
#   overmix 25        fold strokes' worth of work (REFERENCE_STROKE_J_KG, measured over the method's own strokes): "15
#                     to 20 light strokes" is the muffin method's direction (Quaker Oats Baking 101, opened); 25 is
#                     that with a margin. Authored.
#   dry 0.02          "just moistened": no dry flour to be seen; 2 % of the flour is authored. The wetting constant
#                     (materials.Recipe.wet_d) is calibrated so that the method's strokes reach it: 17.6 strokes in
#                     the calibration run, about 21 in the same run on the physics as delivered.
#   hot min_c 74      165 F: "The FDA Food Code requires that all leftover foods ... be reheated to 165 F for 15
#                     seconds within 2 hours" (Institute of Child Nutrition, Reheating Foods; opened) and "at least 74 C (165 F)" (Government of Nunavut, Leftover Food Factsheet; opened); the
#                     UK FSA's own table, opened, has 70 C held for 2 minutes or 75 C for 30 s. spread_k 15: authored.
#   cooked at 88 C    the sauce must have reached 88 C (190 F), the lower end of the maker's range for a
#                     starch-thickened milk sauce: "The pastry cream should reach 190 F to 200 F" after it has thickened
#                     and bubbled and been whisked "for 1 additional minute" (King Arthur Baking, Pastry Cream; opened).
#                     The minute at
#                     heat is in the other half of the criterion: 95 % of the starch cooked, by the gelatinisation
#                     kinetics. 88 C is below this hob's simmer (no vapour
#                     leaves as bubbles below a mean of params.HOB simmer_c, 96 C): the instruction names the simmer as
#                     the cook's sign of done, and the verifier asks only for the maker's lower figure.
#   The sauce's temperature is the highest the sauce REACHED: the higher of the hob's record of the mean of the sauce
#   in the pan, kept every step (heat.Hob.food_peak_c, in every checkpoint), and the mean in the pan now. A cooked
#   sauce cools once the hob is off, and the task ends at "turn the hob off": read at the moment of judging, the same
#   cooked, unburnt sauce would pass or fail by how long after the knob went to 0 it was judged. The porridge's coldest
#   portion is still read at the moment of judging. (The mean over the food of the highest
#   temperature each POINT has had is not a temperature of the food: a point passes through the hot layer on the pan's
#   bottom and keeps that as its highest. On that rule a porridge passed with its coldest portion at 66.6 C and no
#   portion over 72.8 C, the statistic reading 76.3 C. evaluate() reports the statistic, by name; it decides nothing. The mean the hob records is the whole
#   sauce's mean in the pan at each step, which a point's passage through the bottom layer does not raise.)
#   lumps 0.02, cooked 0.95, contained 0.01, vessel within 50 mm: authored tolerances with no outside figure (none was
#                     found for a batter's lumps). stir-sauce reads its cooked share over the starch in the pan, and its
#                     cooked share and spilt share are judged as they are reported.
#   the pan on the hob  stir-sauce: upright, at the hob's height, its whole base (the rigid base disc) on the hob's
#                     glass (320 x 520 mm about the zone), as the instruction says; the bowls' 50 mm matches nothing on
#                     the hob.
#   boiled on 180 s   stir-sauce: "Then turn the hob off" has a late side. "Starch-thickened puddings, creams or sauces
#                     can handle maybe 2-3 minutes of boiling, but that's all. Beyond that they start to become
#                     thinner" (Joe Pastry; opened); the recipes boil it 1 minute.
#                     The generous end, 3 minutes in total at the boil while the hob heats, is judged from the hob's
#                     record (heat.Hob.boil_s), because this world has no starch breakdown.
#   scorch 2 cm2      authored: the pan's bottom is 8 rings of 12 patches whose area grows outward (0.24 cm2 at the
#                     centre to 3.5 cm2 at the rim; heat.Hob.area), so one burnt patch passes in the inner four rings
#                     and fails in the outer four, where the hob's heating ring puts 80 % of its power.
#   stir-sauce does not judge lumps: its starch is suspended in the milk from the start and nothing in the physics
#   gathers it (points carry their composition; the only exchange is a convex mix with the neighbourhood). The most
#   uneven state the start allows is its settled bottom layer gelled where it lies, 1.46 times the mean in a 6 mm cell,
#   and the rule of twice the mean could never fail once the starch was cooked. A threshold that tells an unstirred layer from a stirred sauce would have to be measured on stirred
#   and whisked sauces' compositions, which no record holds.
# stir-sauce's score (reward="cook", its verifier.py score()): 1 on success, else FAIL_SCORE_SCALE x the progress of
# cook_progress(), read from the final state. Every other task's score is the share of its criteria met.
# TASKS (tasks()): each task's TEXT with its setup.py MATERIAL and its verifier.py JUDGING (tasks/<task>/)
TEXT = {
    "dry-whisk": dict(
        preset="bake", title="Whisk the dry ingredients",
        instruction="The bowl holds flour with sugar, baking powder and salt dumped on it in separate places. Whisk "
                    "the dry ingredients until they are evenly combined. Keep them in the bowl."),
    "muffin-fold": dict(
        preset="bake", title="Fold the muffin batter",
        instruction="The bowl holds the blended dry ingredients with milk, egg and oil poured into a well in the "
                    "middle. Fold the wet and dry ingredients together until just combined: no dry flour left, and "
                    "no more mixing than that. Keep the batter in the bowl."),
    "pancake-whisk": dict(
        preset="bake", title="Whisk the pancake batter",
        # the recipe the batch is measured from says "Gently and quickly mix into the egg and milk mixture" and lets
        # the batter stand; a pancake batter is mixed until just combined and its small lumps are left: whisking
        # until smooth is not what the recipe wants.
        instruction="The bowl holds beaten egg, milk and oil with the dry ingredients heaped in the middle. Whisk the "
                    "dry ingredients into the liquid gently and quickly, until just combined: no dry flour left, and "
                    "no more mixing than that; small lumps may stay. Keep the batter in the bowl."),
    "pudding-whisk": dict(
        preset="bake", title="Whisk the instant pudding",
        instruction="The bowl holds two cups of cold milk with a packet of instant pudding mix heaped in the middle. "
                    "Whisk for about two minutes, until the pudding is smooth, then take the whisk out. Keep the "
                    "pudding in the bowl."),
    "cake-mixer": dict(
        preset="bake", title="Beat the cake batter with the hand mixer",
        instruction="The bowl holds cake mix with water, oil and eggs. The hand mixer stands on its heel beside the "
                    "bowl, switched off. Beat the batter with the mixer: lowest speed until the mix is moistened, "
                    "then a medium speed for about two minutes. Leave the mixer switched off and set down on the "
                    "counter, out of the bowl. Keep the batter in the bowl."),
    "sauce-blend": dict(
        preset="bake", title="Blend the sauce",
        instruction="The small bowl holds mayonnaise with a dollop of ketchup and a dollop of relish on it. Stir the sauce until it is "
                    "evenly blended. Keep it in the bowl."),
    "stir-sauce": dict(
        preset="hob", title="Cook the sauce, stirring",
        # the simmer is the recipe's sign of done ("Bring to boil ... stirring constantly", Argo; "until thickened
        # and bubbles are forming", King Arthur); steam comes minutes before it and is not. The thermometer is a thing
        # in the scene, not part of that sign. Lumps are not named: they are not judged (see above TASKS).
        instruction="The saucepan on the hob holds a cold slurry of milk, corn starch and sugar; the hob is on. Cook "
                    "the sauce, stirring, until it thickens and comes to a simmer, without letting it catch on the "
                    "bottom. Then turn the hob off. Keep the sauce in the pan and the pan on the hob. A probe "
                    "thermometer stands in the crock."),
    "reheat-porridge": dict(
        preset="hob", title="Reheat the porridge",
        instruction="The saucepan on the hob holds cold cooked porridge; the hob is on. Stir until the porridge is "
                    "hot all the way through (74 C, 165 F, everywhere; a probe thermometer stands in the crock), "
                    "without letting it catch on the bottom. Then turn the hob off (knob to 0). Keep the porridge "
                    "in the pan and the pan on the hob."),
}
PACKAGE = "ingredient_mixing"


def task_setup(task):
    """The benchmark task's setup.py (tasks/<task>/), or None for a task outside the benchmark."""
    return task_folders.setup(PACKAGE, task) if task_folders.exists(PACKAGE, task) else None


def task_verifier(task):
    """The benchmark task's verifier.py (tasks/<task>/), or None for a task outside the benchmark."""
    return task_folders.verifier(PACKAGE, task) if task_folders.exists(PACKAGE, task) else None


_TASKS = {}


def tasks():
    """name -> the task's preset, title, instruction, solid, pastes, portions or portion_ml, criteria (and reward),
    built once: its folder's modules import this one."""
    if not _TASKS:
        _TASKS.update({name: dict(text, **task_setup(name).MATERIAL, **task_verifier(name).JUDGING)
                       for name, text in TEXT.items()})
    return _TASKS


def __getattr__(name):
    if name == "TASKS":
        return tasks()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def task_spec(config):
    t = dict(tasks()[config.task])
    t.update(getattr(task_setup(config.task), "VARIANT_MATERIAL", {}).get(config.variant, {}))
    judging = getattr(task_verifier(config.task), "variant_judging", None)
    if judging is not None:
        judging(t, config.variant)
    if config.variant == "pet-meal":
        t["instruction"] = ("The small bowl holds a portion of canned food with a powdered supplement sprinkled on one "
                            "side. Mix the supplement evenly into the food. Keep it in the bowl.")
    return t


def instruction(config):
    """The task's instruction as a policy receives it: what is in the vessel, what to do, what counts. A variant
    that puts something else in the vessel, or the vessel on something else, says so (pet-meal has its own text in
    task_spec)."""
    t = task_spec(config)
    text = t["instruction"]
    if config.variant == "honey":
        text = ("The small bowl holds mayonnaise with a squeeze of honey and a spoonful of mustard on it. Stir the sauce "
                "until it is evenly blended. Keep it in the bowl.")
    elif config.variant == "spelt":
        text = text.replace("with the dry ingredients heaped", "with whole-spelt flour and the other dry ingredients heaped")
    elif config.variant == "mat":
        text += " The bowl stands on a silicone mat."
    elif config.variant == "sleeves":
        text += " The hand tools carry square non-slip sleeves on their handles."
    if config.robot.startswith("single"):
        text += " One arm is available."
    return text


def recipe_for(config):
    t = task_spec(config)
    return M.Recipe(solid=t["solid"], pastes=t["pastes"])


# ====================================================================== start states
def _lattice(vessel, sp, level, jitter=0.1, seed=0):
    prof = vessel.profile
    inside = lambda r, z, p: r < np.interp(z, prof[1:, 1], prof[1:, 0], left=prof[1, 0])
    return G.seed_revolved(inside, sp, vessel.rim_r, 0.0, level, jitter=jitter, seed=seed)


class _Fill:
    """Ingredients put into the vessel one after another, as a cook does: each takes, of the lattice points still free,
    exactly the number its rest volume needs, those lowest in a potential that describes how it was put in:

      mound   poured from a cup or a packet: z + slope * distance from the pouring point (a cone standing on whatever
              is there, at the slope given; the top rounded)
      ring    the same poured in a ring: a mound with a well in its middle
      pool    a liquid: the lowest free points, inside a radius if given (a well) or anywhere
      dollop  a spoonful or a squeeze of paste set on the surface: the points nearest a centre on that surface
      spread  a paste spooned in and pushed about: nearly level, with the relief its yield stress holds

    A smooth bump field (a few seeded sine waves) is added so that no outline is a circle and no top is a plane. The
    shapes are start states, not results: the settling that follows (env._settle) lets the material take what its
    strength allows."""

    def __init__(self, vessel, sp, rng, top, level=False):
        self.sp, self.vp, self.rng = sp, sp**3, rng
        self.level = bool(level)           # benches: every part laid in level layers (layout["start"] == "level")
        self.pts = _lattice(vessel, sp, top, seed=int(rng.integers(1 << 30)))
        self.free = np.ones(len(self.pts), bool)
        self.owner = np.full(len(self.pts), -1, int)
        self.parts = []
        self.xy, self.z = self.pts[:, :2], self.pts[:, 2]

    def bumps(self, amp, scale, k=5):
        out = np.zeros(len(self.pts))
        for _ in range(k):
            a = self.rng.uniform(0, 2*math.pi)
            lam = scale*self.rng.uniform(0.6, 1.6)
            out += np.sin((self.xy[:, 0]*math.cos(a) + self.xy[:, 1]*math.sin(a))*2*math.pi/lam + self.rng.uniform(0, 2*math.pi))
        return out*(amp/math.sqrt(k))

    def dist(self, c):
        return np.hypot(self.xy[:, 0] - c[0], self.xy[:, 1] - c[1])

    def surface_at(self, c, radius=0.012):
        """Height of what has been put in so far around the point c (the vessel's floor where nothing has)."""
        near = (~self.free) & (self.dist(c) < radius)
        if near.any():
            return float(self.z[near].max())
        near = self.dist(c) < radius
        return float(self.z[near].min()) if near.any() else 0.0

    def take(self, comp, volume, phi, where=None):
        n = int(round(volume/self.vp))
        if self.level:
            phi, where = self.z + 1e-4*np.hypot(self.xy[:, 0], self.xy[:, 1]), None
        ok = self.free if where is None else (self.free & where)
        idx = np.flatnonzero(ok)
        if len(idx) < n:
            raise ValueError("the vessel does not hold this start state")
        pick = idx[np.argpartition(phi[idx], n - 1)[:n]]
        self.free[pick] = False
        self.owner[pick] = len(self.parts)
        self.parts.append((np.asarray(comp, float), float(volume)))
        return pick

    def mound(self, comp, volume, centre, slope, round_r=0.012, bump=0.0015, scale=0.03):
        d = np.sqrt(self.dist(centre)**2 + round_r**2)
        return self.take(comp, volume, self.z + slope*d + self.bumps(bump, scale))

    def ring(self, comp, volume, centre, radius, slope, bump=0.0015, scale=0.03):
        d = np.sqrt((self.dist(centre) - radius)**2 + 0.010**2)
        return self.take(comp, volume, self.z + slope*d + self.bumps(bump, scale))

    def pool(self, comp, volume, centre=None, radius=None):
        where = None if radius is None else self.dist(centre) < radius
        return self.take(comp, volume, self.z.copy(), where)

    def dollop(self, comp, volume, centre, flat=1.5, bump=0.0015, scale=0.02):
        zs = self.surface_at(centre)
        d = np.sqrt(self.dist(centre)**2 + (flat*(self.z - zs))**2)
        return self.take(comp, volume, d + self.bumps(bump, scale))

    def spread(self, comp, volume, centre, relief, slope=0.10, scale=0.035):
        return self.take(comp, volume, self.z + slope*self.dist(centre) + self.bumps(relief, scale))

    def result(self):
        keep = self.owner >= 0
        comps = np.zeros((len(self.pts), NS))
        for k, (comp, vol) in enumerate(self.parts):
            comps[self.owner == k] = comp/vol*self.vp
        return self.pts[keep], comps[keep]


class Start:
    """What a benchmark task's setup.py puts its ingredients in on (start_state below calls its setup(s)).

      config, layout, vessel   the selection, scene.task_layout's layout and the vessel (tools.Vessel)
      recipes, scale           params.RECIPES and the layout's batch scale
      sp, vp, rng              the lattice spacing, a point's volume and the start's seeded generator
      a0, off                  where the cook happened to put things: an angle, and an offset from the vessel's axis
      volume_of, polar         a composition's rest volume; a vector from a radius and an angle

    setup(s) sets batch (the recipe's species masses), temp (C; 20 unless it sets it) and either fill (a _Fill) or
    points and species (vessel frame)."""

    def __init__(self, config, layout, vessel, recipes, scale, sp, vp, rng, a0, off, volume_of, polar):
        self.config, self.layout, self.vessel, self.recipes, self.scale = config, layout, vessel, recipes, scale
        self.sp, self.vp, self.rng, self.a0, self.off = sp, vp, rng, a0, off
        self.volume_of, self.polar = volume_of, polar
        self.temp, self.fill, self.batch, self.points, self.species = 20.0, None, None, None, None


def start_state(config, scene):
    """Points (vessel frame), their species masses and temperatures, and the batch record for the evaluator."""
    t = task_spec(config)
    L, v = scene.layout, scene.vessel
    R = P.RECIPES
    scale = L.get("scale", 1.0)
    recipe = recipe_for(config)
    sp = config.cell_m/(2 if config.points_per_cell == 8 else 1)
    vp = sp**3
    rng = np.random.default_rng([config.seed, 20261010])
    task = config.task

    def volume_of(comp):
        return M.rest_volume(recipe, {k: m for k, m in enumerate(comp) if m > 0})[0]

    def polar(radius, angle):
        return np.array([radius*math.cos(angle), radius*math.sin(angle)])
    a0 = rng.uniform(0, 2*math.pi) if config.seed else 0.3                       # where the cook happened to put things
    off = rng.uniform(-0.012, 0.012, 2) if config.seed else np.array([0.006, -0.004])
    folder = task_setup(task)
    if folder is None:
        raise ValueError(f"No continuum start state for task {task!r}")
    s = Start(config, L, v, R, scale, sp, vp, rng, a0, off, volume_of, polar)
    folder.setup(s)
    temp, batch = s.temp, s.batch
    if s.fill is not None:
        pts, comps = s.fill.result()
    else:
        pts, comps = s.points, s.species
    # the lattice holds slightly more or less than the recipe: scale every point's mass so the batch is the recipe's
    comps = comps*(batch.sum()/comps.sum())
    return dict(points=pts, species=comps, temp=temp, batch=batch, recipe=recipe, spacing=sp, level=float(pts[:, 2].max()))


# ====================================================================== evaluator
SPECIES_GROUP = dict(minor=(MINOR,), solid=(SOLID_D, SOLID_L, SOLID_S, SOLID_C), sugar=(SUGAR,), paste_b=(PASTE_B,),
                     paste_a=(PASTE_A,), paste_c=(PASTE_C,), fat=(FAT,))
CUTS = (0.0, 22.5, 45.0, 67.5)        # degrees about the vessel's axis: the frames the portions are cut in


def equal_portions(local, mass, n):
    """Cut a set of points into n portions of equal mass: the set is halved (by mass, in the proportion of the
    portions each side is to hold) across its longest extent, and each part again, until n are left. Every point is
    in exactly one portion. Returns the portion index of each point."""
    part = np.zeros(len(local), np.int32)
    todo = [(np.arange(len(local)), 0, n)]
    while todo:
        idx, first, k = todo.pop()
        if k == 1 or len(idx) < 2:
            part[idx] = first
            continue
        p = local[idx]
        axis = int(np.argmax(p.max(0) - p.min(0)))
        order = np.argsort(p[:, axis], kind="stable")
        cm = np.cumsum(mass[idx][order])
        ka = k//2
        cut = int(np.searchsorted(cm, cm[-1]*ka/k))
        cut = min(max(cut, 1), len(idx) - 1)
        todo.append((idx[order[:cut]], first, ka))
        todo.append((idx[order[cut:]], first + ka, k - ka))
    return part


def uniformity(local, spec, mass, groups, n):
    """How unevenly the species of `groups` are spread through the material in the vessel: the material is cut into
    n portions of EQUAL MASS (equal_portions), a species' mass fraction is read in each, and its coefficient of
    variation over the portions is taken. That is done in four frames turned about the vessel's axis (CUTS) and the
    four are pooled (root mean square), so that the reading does not hang on where the first cut happens to fall.
    Returns {species: dict(cv, lowest and highest of the four frames)}. All of the material is in the portions, in
    every state: none of it comes into the reading or leaves it as it moves. (Cubes of the grid fixed in the vessel,
    counted when at least half full, come in and go out as the powder is pushed about, and the reading of a dry mix
    jumps by tens of percent between two looks at nearly the same state.)"""
    out = {}
    if len(local) < 2*n:
        return {g: dict(cv=None, lowest=None, highest=None) for g in groups}
    per = {g: [] for g in groups}
    for deg in CUTS:
        a = math.radians(deg)
        c, s_ = math.cos(a), math.sin(a)
        rot = np.stack([c*local[:, 0] + s_*local[:, 1], -s_*local[:, 0] + c*local[:, 1], local[:, 2]], 1)
        part = equal_portions(rot, mass, n)
        m_all = np.bincount(part, weights=mass, minlength=n)
        for g in groups:
            m_sp = np.bincount(part, weights=spec[:, list(SPECIES_GROUP[g])].sum(1), minlength=n)
            frac = m_sp/np.maximum(m_all, 1e-30)
            mean = float(m_sp.sum()/max(m_all.sum(), 1e-30))
            per[g].append(float(frac.std()/mean) if mean > 0 else None)
    for g in groups:
        v = per[g]
        if any(x is None for x in v):
            out[g] = dict(cv=None, lowest=None, highest=None)
        else:
            out[g] = dict(cv=float(np.sqrt(np.mean(np.square(v)))), lowest=float(min(v)), highest=float(max(v)))
    return out


def evaluate(world, info, final=True):
    """Measure every criterion of the task. info: the start-state record (batch, spacing) kept by the environment."""
    c, sc = world.config, world.scene
    spec_t = task_spec(c)
    crit = spec_t["criteria"]
    cont, v = world.cont, sc.vessel
    out = dict(task=c.task, criteria={}, measures={})
    q = world.bq[v.body]
    Rv = Rotation.from_quat(q[3:])
    x, spec, flags, state, temp, mass = cont.read("x", "spec", "flags", "state", "temp", "mass")
    spec = spec.astype(np.float64)
    local = Rv.inv().apply(x.astype(np.float64) - q[:3])
    active = flags == 0
    inside = active & v.inside(local, margin=0.002)
    total = float(spec.sum())
    batch = float(np.sum(info["batch"]))
    m_in = float(spec[inside].sum())
    where = dict(in_vessel=m_in, counter=float(spec[flags == 2].sum()), floor=float(spec[flags == 3].sum()),
                 airborne=float(spec[flags == 1].sum()), elsewhere=float(spec[active & ~inside].sum()),
                 on_wires=float(spec[flags == 5].sum()))
    out["measures"]["mass_kg"] = dict(batch=batch, total=total, **{k: round(val, 6) for k, val in where.items()})
    # how what is outside the vessel got there (continuum.k_exit keeps where each point left the vessel's envelope):
    # above the rim plane, or below it. Nothing may leave below it: that would be through the wall or the base
    xout, tout = cont.read("xout", "tout")
    left = ~inside & (tout >= 0.)
    over = left & (xout[:, 2] >= v.depth - 2.0*cont.dx)
    out["measures"]["left_vessel"] = dict(over_rim_kg=round(float(spec[over].sum()), 6),
                                          through_wall_kg=round(float(spec[left & ~over].sum()), 6),
                                          no_record_kg=round(float(spec[~inside & (tout < 0.)].sum()), 6))
    out["measures"]["mass_balance_error"] = abs(total - batch)/batch
    spilled = (where["counter"] + where["floor"] + where["airborne"])/batch
    # material on a tool or the rim is not spilt; material neither in the vessel nor on the counter is reported
    out["measures"]["outside_fraction"] = round((total - m_in)/batch, 5)
    vp = info["spacing"]**3
    portion = (info["volume_m3"]/spec_t["portions"]) if "portions" in spec_t else spec_t["portion_ml"]*1e-6
    solid = spec[:, [SOLID_D, SOLID_L, SOLID_S, SOLID_C]].sum(1)
    solid_total = float(solid.sum())
    C = out["criteria"]
    if "uniform" in crit:
        u = crit["uniform"]
        groups = (u["species"],) if isinstance(u["species"], str) else tuple(u["species"])
        n_portions = max(8, int(round(info["volume_m3"]/portion)))
        res = uniformity(local[inside], spec[inside], spec[inside].sum(1), groups, n_portions)
        # every named ingredient is read and the least evenly spread one is judged: a sauce is not blended because
        # its ketchup is. The portions are cut from what is in the vessel: at
        # least half of the batch must be there, or a batch smeared over the rim and the tools would be read as even
        # on the little that is left (the half is authored)
        share = m_in/batch
        cvs = {g: res[g]["cv"] for g in groups}
        worst = None if any(x is None for x in cvs.values()) else max(cvs, key=cvs.get)
        cv = None if worst is None else cvs[worst]
        C["uniform"] = dict(value=None if cv is None else round(cv, 4), limit=u["cv"],
                            species=worst if worst is not None else groups[0],
                            each={g: None if cvs[g] is None else round(cvs[g], 4) for g in groups},
                            across_the_four_cuts={g: None if res[g]["cv"] is None else
                                                  [round(res[g]["lowest"], 4), round(res[g]["highest"], 4)] for g in groups},
                            portions=n_portions, batch_share_in_the_vessel=round(share, 3),
                            share_limit=0.5, ok=bool(cv is not None and round(cv, 4) <= u["cv"] and share >= 0.5))
    if "dry" in crit:
        f = float(spec[:, SOLID_D].sum())/max(solid_total, 1e-12)
        C["no_dry_powder"] = dict(value=round(f, 4), limit=crit["dry"], ok=bool(f <= crit["dry"]))
    if "lumps" in crit:
        f = float(spec[:, SOLID_L].sum())/max(solid_total, 1e-12)
        C["smooth"] = dict(value=round(f, 4), limit=crit["lumps"], ok=bool(f <= crit["lumps"]))
    if "overmix" in crit:
        wetted = spec[:, [SOLID_L, SOLID_S, SOLID_C]].sum(1)
        W = float((wetted*state[:, M.ST_WORK]).sum()/max(wetted.sum(), 1e-12))
        limit = crit["overmix"]*REFERENCE_STROKE_J_KG
        C["not_overmixed"] = dict(value=round(W, 3), limit=round(limit, 3), unit="J/kg",
                                  reference_strokes=round(W/REFERENCE_STROKE_J_KG, 2), ok=bool(W <= limit))
    layers = None
    if "cooked" in crit:
        k = crit["cooked"]
        st = spec[:, [SOLID_L, SOLID_S, SOLID_C]].sum(1)
        # the gelatinised share of the starch IN THE PAN, judged as it is reported (to 1e-4): starch on the counter,
        # the floor or beside the pan is not the sauce, and spill is judged once, in `contained`. As a share of
        # all the batch's starch, raw slurry spilt before the cook would keep a cooked pan from reading cooked, and
        # compared unrounded, 0.9499999992 would read "0.95 of 0.95, not met"
        frac = round(float(spec[inside, SOLID_C].sum()/max(st[inside].sum(), 1e-12)), 4) if inside.any() else 0.
        frac_batch = float(spec[:, SOLID_C].sum()/max(st.sum(), 1e-12))
        layers = {n: layer_shares(local[inside, 2], spec[inside, SOLID_C], st[inside], mass[inside], n) for n in (2, 3, 4)}
        tmean = float((temp[inside]*mass[inside]).sum()/max(mass[inside].sum(), 1e-12)) if inside.any() else 0.
        # judged on the highest mean temperature the sauce reached (see the note above TASKS): the hob's record of the
        # food's mean, kept every step, or the mean in the pan now if that is higher (a state written into the
        # simulator has not been stepped). The cooked share does not go back and neither does this; the verdict no
        # longer hangs on how long after the hob was turned off it is read. mean_temp_c is the mean now, reported.
        # mean_of_point_peaks_c is reported and decides nothing: see the note above TASKS.
        tpeak = np.maximum(state[:, M.ST_PEAK], temp)
        peaks = float((tpeak[inside]*mass[inside]).sum()/max(mass[inside].sum(), 1e-12)) if inside.any() else 0.
        recorded = getattr(world.heat, "food_peak_c", None) if world.heat is not None else None
        # the temperature is judged as it is reported, to a hundredth of a degree: a sauce written to 88 C everywhere
        # has a mean of 87.99999 in single precision, and unrounded it would be reported as "88.0 of 88.0"
        # and "not cooked" in one breath
        reached = round(max(tmean, -math.inf if recorded is None else float(recorded)), 2)
        tmean = round(tmean, 2)
        C["cooked"] = dict(value=frac, limit=k["fraction"], reached_temp_c=reached, temp_limit_c=k["temp_c"],
                           mean_temp_c=tmean, of_all_the_batch=round(frac_batch, 4), mean_of_point_peaks_c=round(peaks, 2),
                           ok=bool(frac >= k["fraction"] and reached >= k["temp_c"]))
    if "hot" in crit:
        k = crit["hot"]
        # every portion must BE at the reheating temperature at the moment of judging, as a thermometer pushed into
        # it reads (the task hands the cook one); the spread is of the same temperatures.
        # coldest_portion_mean_of_point_peaks_c is reported and decides nothing: see the note above TASKS.
        tmin, tmax, cells = portion_temperatures(local[inside], temp[inside], mass[inside], portion, vp)
        tpeak = np.maximum(state[:, M.ST_PEAK], temp)
        pmin, _, _ = portion_temperatures(local[inside], tpeak[inside], mass[inside], portion, vp)
        tmin, tmax = round(tmin, 2), round(tmax, 2)              # judged as reported (see `cooked` above)
        C["hot_through"] = dict(coldest_c=tmin, hottest_c=tmax, cells=cells, limit_c=k["min_c"],
                                spread_limit_k=k["spread_k"], coldest_portion_mean_of_point_peaks_c=round(pmin, 2),
                                ok=bool(tmin >= k["min_c"] and tmax - tmin <= k["spread_k"]))
    if "scorch_cm2" in crit:
        area = world.heat.burnt_area()*1e4 if world.heat is not None else 0.
        C["not_scorched"] = dict(value=round(area, 3), limit=crit["scorch_cm2"], unit="cm2", ok=bool(area <= crit["scorch_cm2"]))
    kc = crit["contained"]
    lim = kc["limit"] if isinstance(kc, dict) else kc
    beside = None
    if isinstance(kc, dict) and kc.get("below_rim"):
        # stir-sauce: sauce outside the pan below its rim (on the hob's glass beside it, down its outside, on the
        # counter near it) is out of the pan as well. It stays active mixture there (flag 0): the counter's flag is
        # given only to what leaves the grid region, 108 mm from the pan's axis, and 3 % of the batch lying on the hob
        # by the pan would read as nothing spilt. On the rim's top, on a
        # tool held over the pan, or in the air above the rim, it is not counted
        low = active & ~inside & (local[:, 2] < v.depth - 2.0*cont.dx)
        beside = float(spec[low].sum())/batch
        spilled += beside
        spilled = round(spilled, 5)                 # judged as it is reported
    C["contained"] = dict(value=round(spilled, 5), limit=lim, ok=bool(spilled <= lim))
    if beside is not None:
        C["contained"]["of_it_outside_the_pan_below_its_rim"] = round(beside, 5)
    p_now = q[:3]
    up = Rv.apply([0., 0., 1.])[2]
    moved = float(np.linalg.norm(p_now[:2] - sc.vessel_start[:2]))
    standing = bool(up > 0.985 and abs(p_now[2] - sc.vessel_start[2]) < 0.01)
    kv = crit["vessel_in_place"]
    tilt, height = round(math.degrees(math.acos(min(1., abs(up)))), 2), round(float(p_now[2] - sc.vessel_start[2]), 4)
    if isinstance(kv, dict) and kv.get("on_hob"):
        # stir-sauce: "the pan on the hob" is the pan standing upright, at the hob's height, with its whole base on
        # the hob's glass (Scene's static box "hob_glass"). The base is the pan's rigid base disc (tools.Vessel: a
        # dodecagon whose flats lie on the profile's flat floor, 64 mm; its corners at 66.3 mm), not its outer wall,
        # and not the bowls' "within 50 mm of where it started"
        _, (gx, gy, _), (hx, hy, _), _, _ = next(b for b in sc.static_boxes if b[0] == "hob_glass")
        base_r = float(v.profile[1, 0])/math.cos(math.pi/12)
        off = (abs(float(p_now[0]) - gx), abs(float(p_now[1]) - gy))
        on = bool(off[0] <= hx - base_r and off[1] <= hy - base_r)
        C["vessel_in_place"] = dict(tilt_deg=tilt, tilt_limit_deg=round(math.degrees(math.acos(0.985)), 2), height_m=height,
                                    height_limit_m=0.01, base_on_the_hob_glass=on,
                                    from_the_glass_centre_m=[round(off[0], 4), round(off[1], 4)],
                                    allowed_m=[round(hx - base_r, 4), round(hy - base_r, 4)], ok=bool(on and standing))
    else:
        C["vessel_in_place"] = dict(moved_m=round(moved, 4), tilt_deg=tilt, height_m=height, limit_m=kv,
                                    ok=bool(moved <= kv and standing))
    if crit.get("appliance_off") == "mixer":
        mx = sc.tools["mixer"]
        st = mx.state()
        pm, Rm = world.body_pose(mx.body)
        hands = [world._mj_to_newton[b] for fs in getattr(world, "_arm_fingers", {}).values() for b in fs]
        held = bool(world.pair_contact_forces(hands, mx.bodies)) if hands else False
        away = float(np.linalg.norm(pm[:2] - q[:2])) > v.rim_r + 0.04
        still = float(np.linalg.norm(world.bqd[mx.body, :3])) < 0.005
        C["appliance_off"] = dict(setting=st["setting"], beater_rpm=st["beater_rpm"], set_down=bool(away and still and not held),
                                  ok=bool(st["setting"] == 0 and max(map(abs, st["beater_rpm"])) < 5. and away and still
                                          and not held))
    if crit.get("appliance_off") == "knob":
        st = sc.tools["knob"].state()
        C["appliance_off"] = dict(level=st["level"], ok=bool(st["level"] == 0))
    if "boil_s" in crit:
        # stir-sauce: "until it ... comes to a simmer. Then turn the hob off." The seconds the sauce in the pan was at
        # the boil (its mean at simmer_c, where the bubbles start) while the hob heated, in total (heat.Hob.boil_s,
        # in every checkpoint). A starch sauce boiled on thins: "Starch-thickened puddings, creams or sauces can
        # handle maybe 2-3 minutes of boiling, but that's all" (Joe Pastry, opened);
        # the recipes boil it "1 minute" (Argo) or "1 additional minute" (King Arthur). This world has no starch
        # breakdown, so the sauce would show nothing: the hob's record is judged. No minimum
        bs = round(float(getattr(world.heat, "boil_s", 0.0)), 1) if world.heat is not None else 0.
        C["not_boiled_on"] = dict(value=bs, limit=crit["boil_s"], unit="s at the boil with the hob on",
                                  ok=bool(bs <= crit["boil_s"]))
    if crit.get("tool_out"):
        inside_tools = tools_in_vessel(world)
        C["tool_out"] = dict(tools_in_vessel=inside_tools, ok=not inside_tools)
    out["success"] = bool(all(k["ok"] for k in C.values()))
    share = round(sum(bool(k["ok"]) for k in C.values())/len(C), 4)               # the share of the criteria met
    out["score"] = share
    if spec_t.get("reward") == "cook":
        task_verifier(c.task).score(world, out, C, crit, layers, share)
    out["measures"]["points"] = int(len(x))
    out["measures"]["solid_states"] = {n: round(float(spec[:, k].sum()), 6) for n, k in
                                       (("dry", SOLID_D), ("lumped", SOLID_L), ("suspended", SOLID_S), ("cooked", SOLID_C))}
    if inside.any():
        out["measures"]["mean_temp_c"] = round(float((temp[inside]*mass[inside]).sum()/mass[inside].sum()), 3)
    return out


FAIL_SCORE_SCALE = 0.9           # a completion that is not a success scores at most 0.9 (label_application's rule)


def layer_shares(z, cooked, wetted, mass, n):
    """The gelatinised share of the starch in each of n layers of the sauce of equal mass, cut by height in the pan's
    frame, from the bottom up (each to 1e-4). A sauce thickens as a whole; a layer cooked on an unstirred bottom under
    cool milk is not that."""
    if len(z) == 0:
        return [0.0]*n
    order = np.argsort(z, kind="stable")
    m = np.asarray(mass, np.float64)[order]
    cum = np.cumsum(m)
    layer = np.minimum((cum - 0.5*m)/cum[-1]*n, n - 1).astype(np.int64)
    c = np.bincount(layer, weights=cooked[order], minlength=n)
    s = np.bincount(layer, weights=wetted[order], minlength=n)
    return [round(float(c[i]/s[i]), 4) if s[i] > 0 else 0.0 for i in range(n)]


def portion_temperatures(local, temp, mass, portion_m3, vp):
    edge = portion_m3**(1/3)
    ijk = np.floor(local/edge).astype(np.int64)
    key = (ijk[:, 0] + 1000)*4_000_000 + (ijk[:, 1] + 1000)*2000 + (ijk[:, 2] + 1000)
    uniq, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv)
    t = np.bincount(inv, weights=temp*mass)/np.maximum(np.bincount(inv, weights=mass), 1e-15)
    full = n*vp >= 0.5*edge**3
    if not full.any():
        return float(temp.min()), float(temp.max()), 0
    return float(t[full].min()), float(t[full].max()), int(full.sum())


def tools_in_vessel(world):
    """Names of the tools with any part inside the vessel's interior."""
    sc = world.scene
    v = sc.vessel
    q = world.bq[v.body]
    Rv = Rotation.from_quat(q[3:])
    out = []
    for name, t in sc.tools.items():
        if not hasattr(t, "tip"):
            continue
        body, off = t.tip
        qb = world.bq[body]
        Rb = Rotation.from_quat(qb[3:])
        # the working end and three points up the tool toward its handle (a whisk leaning in a bowl has its end on
        # the floor; one lying across the rim has only its middle inside)
        pts = np.array([qb[:3] + Rb.apply(np.asarray(off, float)*f) for f in (1.0, 0.85, 0.7, 0.55)])
        if v.inside(Rv.inv().apply(pts - q[:3]), margin=0.0).any():
            out.append(name)
    return out
