"""No runtime: real catalogue/evaluator against NumPy material arrays and rigid-state stand-ins.

Material dataclasses/constants are loaded from source, with Warp kernels omitted. No simulator module is imported.
Hob measurement/checkpoint methods are exercised on stand-ins; no physics step or server is run.
"""
import ast
import copy
import importlib
import json
import sys
import threading
import types
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ingredient_mixing import scoring as S, variants as V
from ingredient_mixing.scoring import TaskConfig as Config
from tests.packages import PACKAGES

ROOT = PACKAGES
ENV = ROOT / "ingredient_mixing"


@pytest.fixture(scope="module")
def task_module():
    # Only the actual material schema; no import/decorator/kernel from Warp is executed.
    tree = ast.parse((ENV / "materials.py").read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Law":
            break
        if isinstance(node, ast.Import) and any(a.name == "warp" for a in node.names):
            continue
        nodes.append(node)
    material = types.ModuleType("ingredient_mixing.materials")
    import ingredient_mixing
    with patch.dict(sys.modules, {"ingredient_mixing.materials": material}):
        exec(compile(ast.Module(nodes, type_ignores=[]), str(ENV / "materials.py"), "exec"), material.__dict__)
        with patch.object(ingredient_mixing, "materials", material, create=True):
            sys.modules.pop("ingredient_mixing.task", None)
            task = importlib.import_module("ingredient_mixing.task")
            yield task
            sys.modules.pop("ingredient_mixing.task", None)
            sys.modules.pop("ingredient_mixing.env", None)
            sys.modules.pop("ingredient_mixing.task_runtime", None)
            for name in ("task", "env", "task_runtime"):
                if hasattr(ingredient_mixing, name):
                    delattr(ingredient_mixing, name)


class Vessel:
    body, depth, rim_r = 0, .12, .149
    profile = np.array([[0., 0.], [.064, 0.], [.076, .12]])

    def inside(self, local, margin=0.):
        return ((np.linalg.norm(local[:, :2], axis=1) < .076 + margin)
                & (local[:, 2] >= -margin) & (local[:, 2] <= self.depth + margin))


class Continuum:
    dx = .0025

    def __init__(self, arrays):
        self.arrays = arrays
        self.n = len(arrays["x"])

    def read(self, *names):
        values = [self.arrays[n].copy() for n in names]
        return values[0] if len(names) == 1 else values


def fixture(task, name, contract=None):
    config = Config(preset="hob" if name in ("stir-sauce", "reheat-porridge") else "bake", task=name,
                    contract=contract or ("v1" if name == "stir-sauce" else S.CONTRACT))
    local = np.array([(x, y, z) for x in np.linspace(-.04, .04, 8) for y in np.linspace(-.04, .04, 8)
                      for z in np.linspace(.005, .065, 8)])
    n = len(local)
    species = np.zeros((n, task.NS))
    half = local[:, 0] > 0.
    if name == "dry-whisk":
        species[:, task.SOLID_D] = .8
        species[half, task.MINOR] = .2
        species[~half, task.SUGAR] = .2
    elif name == "sauce-blend":
        species[:, task.PASTE_A] = .6
        species[half, task.PASTE_B] = .4
        species[~half, task.PASTE_C] = .4
        species[half, task.PASTE_A] += .1
        species[half, task.PASTE_B] -= .1
    elif name == "reheat-porridge":
        species[:, task.PASTE_A] = 1.
    elif name == "stir-sauce":
        species[:, task.AQ], species[:, task.SOLID_S] = .95, .05
    else:
        species[:, task.AQ] = 1.
        species[half, task.AQ], species[half, task.SOLID_D] = .2, .8
    species *= .001
    arrays = dict(x=local + [0., .30, .914], spec=species, mass=species.sum(1), flags=np.zeros(n, dtype=int),
                  state=np.zeros((n, 6)), temp=np.full(n, 5. if config.preset == "hob" else 20.),
                  xout=np.zeros((n, 3)), tout=np.full(n, -1.))
    knob = types.SimpleNamespace(s_level=7)
    knob.state = lambda: dict(level=knob.s_level)
    mixer = types.SimpleNamespace(body=1, bodies=[1], setting=0, rpm=0.)
    mixer.state = lambda: dict(setting=mixer.setting, beater_rpm=[mixer.rpm, -mixer.rpm])
    scene = types.SimpleNamespace(vessel=Vessel(), vessel_start=np.array([0., .30, .914]),
                                 static_boxes=[("hob_glass", (0., .30, .9), (.16, .26, .01), None, None)],
                                 tools={"knob": knob} if config.preset == "hob" else
                                 {"mixer": mixer} if name == "cake-mixer" else {})
    heat = types.SimpleNamespace(area=np.ones(96)*.0181/96, burnt=0., food_peak_c=None, boil_s=0.,
                                knob=knob, pan=scene.vessel, hot_record=None, task_baseline=None, power_w=580.)
    heat.burnt_area = lambda: heat.burnt/1e4
    world = types.SimpleNamespace(config=config, scene=scene, cont=Continuum(arrays),
                                  bq=np.array([[0., .30, .914, 0., 0., 0., 1.], [-.34, .38, 1.02, 0., 0., 0., 1.]]),
                                  bqd=np.zeros((2, 6)), heat=heat if config.preset == "hob" else None)
    heat.w = world
    world.body_pose = lambda body: (world.bq[body, :3], Rotation.from_quat(world.bq[body, 3:]).as_matrix())
    info = dict(batch=species.sum(0), spacing=.01, volume_m3=480e-6)
    env = types.SimpleNamespace(config=config, world=world, info=info)
    S.capture_start(env)
    if world.heat is not None:
        heat.record_food_peak = types.MethodType(hob_methods()["record_food_peak"], heat)
        if S.enabled(config) and config.task == "reheat-porridge":
            from ingredient_mixing.task_runtime import install_hot_recorder
            install_hot_recorder(heat)
    return env


def evaluate(task, env):
    return S.evaluate(env.world, env.info)


def mix(task, env, fraction=1., dry=0., lump=0., work=0.):
    a = env.world.cont.arrays
    spec = a["spec"]
    spec[:] = (1.-fraction)*spec + fraction*spec.mean(0)
    if env.config.task not in ("dry-whisk", "sauce-blend", "reheat-porridge", "stir-sauce"):
        solid = spec[:, [task.SOLID_D, task.SOLID_L, task.SOLID_S, task.SOLID_C]].sum(1)
        spec[:, task.SOLID_D], spec[:, task.SOLID_L] = solid*dry, solid*lump
        spec[:, task.SOLID_S], spec[:, task.SOLID_C] = solid*(1.-dry-lump), 0.
    a["state"][:, task.M.ST_WORK] = work


def finish(task, env):
    if env.config.task == "reheat-porridge":
        env.world.cont.arrays["temp"][:] = 75.
        env.world.scene.tools["knob"].s_level = 0
    elif env.config.task == "stir-sauce":
        a = env.world.cont.arrays
        a["spec"][:, task.SOLID_C] = a["spec"][:, task.SOLID_S]
        a["spec"][:, task.SOLID_S] = 0.
        a["temp"][:] = 90.
        env.world.scene.tools["knob"].s_level = 0
    else:
        mix(task, env)


@pytest.mark.parametrize("name", S.ORDER)
def test_untouched_zero_and_valid_end_state_one(task_module, name):
    env = fixture(task_module, name)
    initial = evaluate(task_module, env)
    assert not initial["success"] and initial["score"] == 0.
    assert initial["contract"] == (S.STIR_SAUCE_CONTRACT if name == "stir-sauce" else S.CONTRACT)
    finish(task_module, env)
    result = evaluate(task_module, env)
    assert result["success"] and result["score"] == 1., result


@pytest.mark.parametrize("name", S.ORDER[:-2])
def test_partial_material_progress_and_damage(task_module, name):
    env = fixture(task_module, name)
    mix(task_module, env, .45, dry=.5, lump=.025)
    ev = evaluate(task_module, env)
    assert not ev["success"] and 0. < ev["score"] < .9
    before = ev["score"]
    env.world.cont.arrays["flags"][:64] = 2
    ev = evaluate(task_module, env)
    assert not ev["success"] and ev["score"] < before
    env.world.cont.arrays["flags"][:] = 2
    assert evaluate(task_module, env)["score"] == 0.


@pytest.mark.parametrize("name", ("muffin-fold", "pancake-whisk"))
def test_overmix_and_containment_failures(task_module, name):
    env = fixture(task_module, name)
    mix(task_module, env, work=8.25)
    assert evaluate(task_module, env)["success"]
    mix(task_module, env, work=8.26)
    assert 0. < evaluate(task_module, env)["score"] < .9
    mix(task_module, env, work=16.5)
    assert evaluate(task_module, env)["score"] == 0.


def test_closing_steps_never_replace_mixing(task_module):
    cake = fixture(task_module, "cake-mixer")
    assert evaluate(task_module, cake)["score"] == 0.
    finish(task_module, cake)
    cake.world.scene.tools["mixer"].setting = 1
    ev = evaluate(task_module, cake)
    assert not ev["success"] and ev["score"] == pytest.approx(.9*.85)
    pudding = fixture(task_module, "pudding-whisk")
    finish(task_module, pudding)
    with patch.object(task_module, "tools_in_vessel", return_value=["whisk"]):
        ev = evaluate(task_module, pudding)
        assert not ev["success"] and ev["score"] == pytest.approx(.9*.85)


def test_uniformity_credits_every_ingredient_and_rejects_missing(task_module):
    env = fixture(task_module, "sauce-blend")
    a = env.world.cont.arrays["spec"]
    a[:, task_module.PASTE_A] = a[:, task_module.PASTE_A].mean()
    assert evaluate(task_module, env)["score"] == 0.  # the other ingredients did not improve
    a[:, task_module.PASTE_C] = 0.
    assert evaluate(task_module, env)["score"] == 0.


def test_baseline_is_required_and_evaluation_never_rebases(task_module):
    env = fixture(task_module, "muffin-fold")
    baseline = copy.deepcopy(env.info["task_baseline"])
    mix(task_module, env, .5, dry=.5)
    ev = evaluate(task_module, env)
    assert env.info["task_baseline"] == baseline
    assert evaluate(task_module, env) == ev
    del env.info["task_baseline"]
    with pytest.raises(ValueError, match="hand-over baseline"):
        evaluate(task_module, env)


def hob_methods():
    tree = ast.parse((ENV / "heat.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Hob")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
               ("record_food_peak", "snapshot", "restore")]
    from ingredient_mixing import params
    namespace = dict(__name__="ingredient_mixing.heat_stub", __package__="ingredient_mixing", np=np,
                     Rotation=Rotation, P=params)
    exec(compile(ast.Module(methods, type_ignores=[]), str(ENV / "heat.py"), "exec"), namespace)
    return namespace


def test_reheat_records_simultaneous_reading_and_survives_cooling(task_module):
    env = fixture(task_module, "reheat-porridge")
    a, heat = env.world.cont.arrays, env.world.heat
    a["state"][:, task_module.M.ST_PEAK] = 90.  # every point once visited the bottom: NOT hot through
    a["temp"][:] = 67.
    heat.record_food_peak(.01)
    heat.knob.s_level = 0
    ev = evaluate(task_module, env)
    assert not ev["success"] and 0. < ev["score"] < .9
    heat.knob.s_level = 7
    a["temp"][:] = 75.
    heat.record_food_peak(.01)
    record = copy.deepcopy(heat.hot_record)
    a["temp"][:] = 20.
    heat.knob.s_level = 0
    ev = evaluate(task_module, env)
    assert ev["success"] and ev["score"] == 1.
    assert ev["criteria"]["hot_through"]["coldest_c"] == 75.
    assert ev["criteria"]["hot_through"]["current_coldest_c"] == 20.
    assert heat.hot_record == record  # evaluation is read-only
    heat.burnt = 181.
    assert evaluate(task_module, env)["score"] == pytest.approx(0.)


def test_reheat_spread_small_batch_and_off_hob_samples(task_module):
    env = fixture(task_module, "reheat-porridge")
    a, heat = env.world.cont.arrays, env.world.heat
    a["temp"][:] = 76.
    a["temp"][:256] = 99.
    heat.record_food_peak(.01)
    heat.knob.s_level = 0
    assert not evaluate(task_module, env)["success"]
    assert S.best_hot(None, dict(coldest_c=80., hottest_c=80., batch_share=.5), 5.) is None
    old = copy.deepcopy(heat.hot_record)
    a["temp"][:] = 80.
    heat.record_food_peak(.01)
    assert heat.hot_record == old
    a["flags"][:] = 2
    assert not evaluate(task_module, env)["success"]  # empty vessel is a result, not min(empty) fault


def test_reheat_hob_footprint_and_below_rim_spill(task_module):
    env = fixture(task_module, "reheat-porridge")
    finish(task_module, env)
    env.world.bq[0, 0] += .07
    env.world.cont.arrays["x"][:, 0] += .07
    assert evaluate(task_module, env)["success"]  # >50mm but base remains on hob glass
    env.world.cont.arrays["x"][:20, 0] += .15
    ev = evaluate(task_module, env)
    assert not ev["success"] and not ev["criteria"]["contained"]["ok"]
    env.world.bq[0, 0] += .1
    ev = evaluate(task_module, env)
    assert not ev["criteria"]["vessel_in_place"]["ok"]


def test_best_hot_does_not_combine_frames_or_erase_completed_reheat():
    cold = dict(coldest_c=70., hottest_c=72., batch_share=1.)
    uneven = dict(coldest_c=78., hottest_c=100., batch_share=1.)
    reading = S.best_hot(cold, uneven, 5.)
    assert not S.hot_ok(reading)
    good = dict(coldest_c=75., hottest_c=80., batch_share=1.)
    assert S.best_hot(good, uneven, 5.) == good


def test_legacy_meaning_and_stir_sauce_instruction(task_module):
    for name in S.ORDER[:-1]:
        env = fixture(task_module, name, "v1")
        ev = evaluate(task_module, env)
        assert ev["score"] == round(sum(c["ok"] for c in ev["criteria"].values())/len(ev["criteria"]), 4)
        assert "task_baseline" not in env.info
        assert ev["contract"] == "v1"
    default = V.task_resolve({})
    assert default["task_id"] == "ingredient-mixing/v1/bake/muffin-fold/seed-0"
    assert "contract" not in default["backend_options"]
    c = Config(preset="hob", task="stir-sauce")
    assert task_module.task_spec(c) == task_module.TASKS["stir-sauce"]
    assert "until it thickens and comes to a simmer" in task_module.instruction(c)
    assert S.contract_for(c) == S.STIR_SAUCE_CONTRACT


def test_catalogue_roundtrips_exact_selections(task_module):
    bench = json.loads((ENV / "benchmark.json").read_text())
    for key, value in V.catalogue_fields().items():
        assert bench[key] == value
    selections = bench["task_set"]["selections"]
    assert selections == V.task_set_selections() and len(selections) == 8
    for selection in selections:
        resolved = V.task_resolve(selection)
        c = Config(**resolved["backend_options"])
        assert c.task == selection["task"]
        assert c.contract == selection["contract"]
        assert resolved["description"] == S.instruction(c)
        assert resolved["split"] == V.split_of_seed(c.task, c.seed)
        assert ("contract-" in resolved["task_id"]) == (c.contract != "v1")
    assert "about two minutes" not in V.task_resolve(selections[3])["description"]
    assert "lowest speed" not in V.task_resolve(selections[5])["description"]


@pytest.mark.parametrize("selection", [{"contract":"typo"}, {"task":"stir-sauce", "contract":S.CONTRACT},
                                       {"seed":True}, {"seed":1.5}, {"task":"bogus"}, {"typo":True},
                                       {"task":"stir-sauce", "variant":"mat"}])
def test_invalid_selections_refused(task_module, selection):
    with pytest.raises(ValueError):
        V.task_resolve(selection)


def test_task_checkpoint_restores_record_and_reset_clears_it(task_module):
    from ingredient_mixing.env import MixEnv
    from ingredient_mixing.task_runtime import TaskEnv
    base = fixture(task_module, "reheat-porridge")
    env = TaskEnv.__new__(TaskEnv)
    env.__dict__.update(base.__dict__)
    env.lock, env.generation = threading.RLock(), 0
    env.get_state = lambda: {}
    heat = env.world.heat
    with patch.object(MixEnv, "checkpoint", side_effect=lambda: {"physical": "unchanged"}), \
            patch.object(MixEnv, "restore") as restore:
        initial = env._initial = env.checkpoint()
        heat.hot_record = dict(coldest_c=75., hottest_c=80., cells=8, batch_share=1.)
        snap = env.checkpoint()
        heat.hot_record["coldest_c"] = 2.
        assert snap["task_judging"]["hot_record"]["coldest_c"] == 75.
        env.restore(snap)
        assert heat.hot_record == snap["task_judging"]["hot_record"]
        env.reset()
        assert env.generation == 1 and heat.hot_record is None
        assert env.info["task_baseline"] == initial["task_judging"]["baseline"]
        env.info["task_baseline"]["cv"]["extra"] = 99
        assert "extra" not in initial["task_judging"]["baseline"]["cv"]
        called = restore.call_count
        for bad in ({}, dict(task_judging=dict(contract="v1"))):
            with pytest.raises(ValueError, match="nothing restored"):
                env.restore(bad)
        assert restore.call_count == called


def test_build_captures_after_settle_without_recopying_physical_checkpoint(task_module):
    from ingredient_mixing.env import MixEnv
    from ingredient_mixing.task_runtime import TaskEnv
    base = fixture(task_module, "muffin-fold")
    env = TaskEnv.__new__(TaskEnv)
    env.__dict__.update(base.__dict__)
    w = env.world
    w.scene.layout, w.targets = {}, np.zeros(2)
    w.attach_material = lambda *args, **kwargs: w.cont
    w.cont.arrays["vol0"] = np.full(w.cont.n, 480e-6/w.cont.n)
    st = dict(batch=env.info["batch"], spacing=env.info["spacing"], points=w.cont.arrays["x"] - w.bq[0, :3],
              species=w.cont.arrays["spec"], temp=20., level=.06,
              recipe=types.SimpleNamespace(as_dict=lambda: {}))
    calls = []

    def settle(rigid_only):
        calls.append(rigid_only)
        if not rigid_only:
            mix(task_module, env, .2, dry=.7)

    env._settle = settle
    world_stub = types.ModuleType("ingredient_mixing.world")
    world_stub.World = lambda config: w
    with patch.dict(sys.modules, {"ingredient_mixing.world": world_stub}), \
            patch.object(task_module, "start_state", return_value=st), \
            patch.object(MixEnv, "checkpoint", return_value={"physical": "same buffers"}) as checkpoint:
        env._build()
    assert calls == [True, False] and checkpoint.call_count == 1
    assert env._initial["task_judging"]["baseline"] == env.info["task_baseline"]
    assert env.info["task_baseline"]["dry"] == .7
    assert evaluate(task_module, env)["score"] == 0.


def test_backend_reset_and_variation_preserve_explicit_contract(task_module):
    from ingredient_mixing import robot_setup
    from gym_anything_robotics.backends.newton_ingredient_mixing import NewtonIngredientMixingBackend as B
    b = B.__new__(B)
    b.config = Config(task="muffin-fold", contract=S.CONTRACT)
    with patch.object(robot_setup, "check_task", return_value=None):
        assert b._reset_selection(7, {})["contract"] == S.CONTRACT
        assert b._reset_selection(None, {"contract":"v1"}).get("contract", "v1") == "v1"
        assert b._reset_selection(None, {"task":"stir-sauce"}).get("contract", "v1") == "v1"
        with pytest.raises(ValueError):
            b._reset_selection(None, {"task":"stir-sauce", "contract":S.CONTRACT})
        for selection in V.task_set_selections():
            request = b.select_variation(selection)["reset_request"]
            selected = b._reset_selection(request["seed"], request["options"])
            assert selected.get("contract", "v1") == selection["contract"]
            assert selected["task"] == selection["task"]
        assert b._variation_key()["contract"] == S.CONTRACT


def test_all_variants_still_resolve_and_new_contracts_do_not_change_recipe(task_module):
    for preset, task, variant in V.pairs():
        old = Config(preset=preset, task=task, variant=variant)
        legacy = V.task_resolve(dict(preset=preset, task=task, variant=variant))
        assert legacy["task_id"] == V.task_id(dict(preset=preset, task=task, seed=0, variant=variant))
        if task == "stir-sauce":
            continue
        new = Config(preset=preset, task=task, variant=variant, contract=S.CONTRACT)
        old_spec, new_spec = task_module.task_spec(old), S.task_spec(new)
        assert task_module.recipe_for(old) == task_module.recipe_for(new)
        for key in old_spec.keys() - {"instruction", "criteria"}:
            assert old_spec[key] == new_spec[key]


def test_external_temperature_recorder_is_isolated_and_does_not_write_physics(task_module):
    from ingredient_mixing.task_runtime import install_hot_recorder
    revised = fixture(task_module, "reheat-porridge")
    legacy = fixture(task_module, "reheat-porridge", "v1")
    for env in (revised, legacy):
        env.world.cont.arrays["temp"][:] = 75.
        before = {k: v.copy() for k, v in env.world.cont.arrays.items()}
        env.world.heat.record_food_peak(.01)
        for name, values in before.items():
            np.testing.assert_array_equal(env.world.cont.arrays[name], values)
    assert revised.world.heat.hot_record["coldest_c"] == 75.
    assert legacy.world.heat.hot_record is None
    assert revised.world.heat.food_peak_c == legacy.world.heat.food_peak_c
    assert revised.world.heat.food_peak_c == pytest.approx(75.)
    assert revised.world.heat.boil_s == legacy.world.heat.boil_s == 0.
    assert not getattr(legacy.world.heat, "_task_recorder_installed", False)
    with pytest.raises(ValueError, match="already installed"):
        install_hot_recorder(revised.world.heat)
    # A completed heat inside a long command is retained even if subsequent samples are cooler. No evaluator
    # call is needed between these two samples, and a reset/restore keeps the wrapper bound to this same instance.
    revised.world.cont.arrays["temp"][:] = 20.
    revised.world.heat.record_food_peak(.01)
    revised.world.heat.knob.s_level = 0
    assert evaluate(task_module, revised)["success"]


def test_task_config_leaves_all_physical_fields_and_defaults_unchanged():
    from dataclasses import asdict
    from ingredient_mixing.config import Config as PhysicalConfig
    physical, selected = asdict(PhysicalConfig()), asdict(Config())
    assert selected.pop("contract") == "v1"
    assert selected == physical
    with pytest.raises(ValueError, match="Unknown configuration fields"):
        PhysicalConfig.from_dict(dict(contract=S.CONTRACT))
    assert Config.from_dict(dict(contract=S.CONTRACT)).contract == S.CONTRACT


