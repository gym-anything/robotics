"""gift-bags-tasks-v2: the render-checked eight-task set. Constructed states only, ported from the render-check fix
branch. No simulator import, server, physics step or renderer is allowed here."""
import copy
import importlib
import itertools
import json
import math
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from gift_bags import task as legacy_task, variants
from gift_bags import task_set_catalog as V1, task_set_v2_catalog as C, task_set_v2_scoring as S
from gift_bags.task_set_v2_evaluator import Evaluator
from gift_bags.task_set_v2_presets import prepare
from tests.packages import PACKAGES, REPO

ROOT = PACKAGES


def _stub(monkeypatch, name, value):
    """Replace a submodule for one test, also as its package's attribute (an earlier test may have imported it)."""
    monkeypatch.setitem(sys.modules, name, value)
    package, _, attr = name.rpartition(".")
    if package in sys.modules:
        monkeypatch.setattr(sys.modules[package], attr, value, raising=False)


@pytest.fixture(autouse=True)
def no_runtime(monkeypatch):
    """Poison runtime imports. Only explicit stubs below may replace these sentinels."""
    for name in ("newton", "warp", "mujoco", "gift_bags.ipcworld", "gift_bags.world", "gift_bags.rig",
                 "gift_bags.scene", "gift_bags.carton", "gift_bags.bagnet"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(variants, "_AVAILABILITY", {p: dict(available=True, reason=None) for p in C.POLICY_PRESETS})


def _robots(monkeypatch):
    monkeypatch.setitem(sys.modules, "robots", NS())
    monkeypatch.setitem(sys.modules, "robots.catalog", NS(default_robot=lambda env: "dual-fr3",
                       environment_robots=lambda env: ["dual-fr3"], ENVIRONMENTS={}))


# ----------------------------------------------------------------------------------------- catalogue and resolver
def test_v2_selections_resolve_and_v1_is_unchanged(monkeypatch):
    _robots(monkeypatch)
    from gym_anything_robotics import tasks
    raw = json.loads((ROOT/"gift_bags/benchmark.json").read_text())
    v2 = raw["task_set"]
    assert v2["selections"] == C.TASK_SET and v2["contract"] == C.CONTRACT and len(C.TASK_SET) == 8
    assert [e["selection"] for e in raw["task_set_v1"]["tasks"]] == V1.TASK_SET
    assert C.CONTRACT in raw["selector_choices"]["contract"]
    test_split = json.loads((REPO/"benchmarks/roboworld/splits/test.json").read_text())["tasks"]
    hardest = next(t["selection"] for t in test_split if t["environment_id"] == "gift-bags")
    assert C.TASK_SET[-1] == hardest
    ids = []
    for selection, task in zip(C.TASK_SET, V1.ORDER):
        resolved = tasks.resolve("gift-bags", selection)
        ids.append(resolved.task_id)
        assert resolved.description == V1.instruction(task, V1.CONTRACT if task != "bag-purchase" else None)
        if task == "bag-purchase":
            assert resolved.to_dict() == tasks.resolve("gift-bags", hardest).to_dict()
            assert resolved.launch.backend_options == dict(preset="counter", task="bag-purchase", seed=0)
        else:
            assert resolved.launch.backend_options == selection
            v1 = tasks.resolve("gift-bags", dict(selection, contract=V1.CONTRACT))
            assert v1.task_id == resolved.task_id.replace(C.CONTRACT, V1.CONTRACT)
            assert v1.description == resolved.description      # instructions are v1's
    assert ids == v2["task_ids"]
    assert v2["task_contract"] == [C.evaluation_contract(t, C.CONTRACT) for t in C.TASKS_V2] + [V1.PURCHASE_CONTRACT]


@pytest.mark.parametrize("bad", [dict(task="bag-purchase", contract=C.CONTRACT),
                                dict(preset="counter", task="bag-purchase", contract=C.CONTRACT),
                                dict(task="open-present", contract=C.CONTRACT),
                                dict(task="open-flaps", preset="home", contract=C.CONTRACT),
                                dict(task="lift-lid", seed=True, contract=C.CONTRACT),
                                dict(task="lift-lid", seed=-1, contract=C.CONTRACT)])
def test_bad_v2_selections(bad):
    with pytest.raises((ValueError, TypeError)):
        V1.task_resolve(bad)


def test_v2_catalogue_reports_the_bare_contract_and_hands_other_contracts_to_v1():
    assert [C.evaluation_contract(t, C.CONTRACT) for t in C.TASKS_V2] == [C.CONTRACT]*7
    for task, contract in (("open-flaps", None), ("bag-purchase", None), ("open-bag", None), ("lift-lid", V1.CONTRACT)):
        assert C.evaluation_contract(task, contract) == V1.evaluation_contract(task, contract)
        assert C.instruction(task, contract) == V1.instruction(task, contract)
        assert C.resolve("home" if task == "lift-lid" else None, task, 0, contract) == V1.resolve(
            "home" if task == "lift-lid" else None, task, 0, contract)
    assert C.completes_window("bag-candle", C.CONTRACT) and C.completes_window("open-bag", C.CONTRACT)
    assert not C.completes_window("open-flaps", C.CONTRACT)
    # The released catalogue itself does not know v2: only task_resolve dispatches to it.
    with pytest.raises(ValueError):
        V1.selected_contract("lift-lid", C.CONTRACT)
    out = C.contract()
    assert out["task_set"] == C.TASK_SET and out["task_set_v1"] == V1.TASK_SET
    assert out["contracts"]["open-flaps"] == [V1.CONTRACT, C.CONTRACT] and out["contracts"]["bag-purchase"] == [
        V1.PURCHASE_CONTRACT]
    assert V1.contract()["task_set"] == V1.TASK_SET and "task_set_v1" not in V1.contract()


def test_v2_config_validates_like_the_released_one():
    from gift_bags.config import Config as Old
    from dataclasses import asdict
    for seed in (True, 0, 17):
        new = asdict(C.Config(preset="home", task="lift-lid", seed=seed))
        assert new.pop("contract") == C.CONTRACT
        assert new == asdict(Old(preset="home", task="lift-lid", seed=seed))
    with pytest.raises(TypeError):
        C.Config(typo=True)
    for bad in (dict(preset="home", task="open-flaps"), dict(preset="counter", task="bag-purchase"),
                dict(preset="home", task="lift-lid", contract=V1.CONTRACT), dict(preset="home", task="lift-lid",
                                                                                   contract=None)):
        with pytest.raises(ValueError):
            C.Config(**bad)


# ----------------------------------------------------------------------------------------- verifier and progress
class Counter:
    """A small synthetic tote and rigid poses; no equilibrium claim."""
    def __init__(self, monkeypatch, task="open-bag"):
        self.W, self.G, self.H = .2032, .1016, .254
        _stub(monkeypatch, "gift_bags.bagnet", NS(W_BAG=self.W, G_BAG=self.G, H_BAG=self.H))
        _stub(monkeypatch, "gift_bags.ipcworld", NS(hinge_angles=lambda *a: None))
        X = [[x, y, z] for z in (0., self.H) for x, y in
             ((-self.W/2, -self.G/2), (self.W/2, -self.G/2), (self.W/2, self.G/2), (-self.W/2, self.G/2))]
        X += [[0., y, z] for z in (self.H/2, self.H) for y in (-self.G/2, self.G/2)]
        self.X = np.array(X)
        self.open = self.X + [0., .17, .9181]
        sheet = NS(X2=self.X, tris=np.array([[0, 1, 2], [0, 2, 3]]),
                   render={"kinds": np.array(["bottom", "bottom"])}, crease_names={})
        self.positions = {"board_0": np.array([-.38, .17, .9153]), "candle": np.array([0., -.07, .97005])}
        self.rotations = {n: np.eye(3) for n in self.positions}
        self.x = self.open.copy()
        if task == "open-bag":
            self.x[:, 1] = .17 + (self.x[:, 1] - .17)*.002
            self.x = self.x @ np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]).T + [0., 1.3, .745]
        else:
            self.positions["board_0"] = np.array([0., .17, .91845])
        self.names = list(self.positions)
        self.crack = 0.
        ipc = NS(sheets=[sheet], sheet_slices=[slice(0, len(self.x))], caliper={"rest_sheet_rigid": .0001},
                 bodies=[NS(name=n, kind="dynamic") for n in self.names], x_free=lambda: self.x,
                 body_pose=lambda i: (self.positions[self.names[i]], self.rotations[self.names[i]]),
                 reactions={}, stats={}, crack_length=lambda k: (self.crack, 0.))
        geo = dict(mat=dict(center=(0., .17, .918), half=(.2, .15)), board=dict(half=(.1, .0495, .00025)),
                   candle=dict(half=(.045, .045, .055)))
        self.env = NS(config=NS(task="bag-purchase", dt=.01), world=NS(ipc=ipc, sim_time=0.,
                        scene=NS(preset="counter", geometry=geo, surface_z=.915)))
        self.ev = Evaluator(self.env, C.Config(preset="counter", task=task))

    def solve(self):
        self.x = self.open.copy()
        self.positions["board_0"] = np.array([0., .17, .91845])
        self.positions["candle"] = np.array([0., .17, .97375])
        for _ in range(self.ev.history.n):
            self.ev.record()
        return self.ev.evaluate()


@pytest.mark.parametrize("task", ["open-bag", "bag-candle"])
def test_counter_start_progress_success_and_collateral(monkeypatch, task):
    w = Counter(monkeypatch, task)
    out = w.ev.evaluate()
    assert out["score"] == 0. and not out["success"]
    assert (out["task"], out["contract"]) == (task, C.CONTRACT)
    assert w.solve()["score"] == 1.
    w.ev.history.rows[-1]["contact"] = True
    held = w.ev.evaluate()
    assert not held["success"] and 0. < held["score"] <= .9
    w.ev.history.rows[-1]["contact"] = False
    w.positions["board_0"][2] -= .001
    assert not w.ev.evaluate()["checks"]["board_in_bag"]
    w.solve()
    if task == "bag-candle":
        w.rotations["candle"] = np.diag([1., -1., -1.])
        assert not w.ev.evaluate()["checks"]["candle_upright"]
    w.crack = .026
    out = w.ev.evaluate()
    assert not out["success"] and out["score"] == 0.


def yaw(degrees):
    a = math.radians(degrees)
    return np.array([[math.cos(a), -math.sin(a), 0.], [math.sin(a), math.cos(a), 0.], [0., 0., 1.]])


class Home:
    def __init__(self, task):
        self.task = task
        self.g = dict(hx=.09, hy=.065, base_h=.055, lid_h=.03, wall=.0015, clearance=.0005)
        self.pos = [np.array([0., .17, .75]), np.array([0., .17, .7765]), np.array([0., .17, .76])]
        self.rot = [np.eye(3) for _ in self.pos]
        names = ["gift_base", "gift_lid", "photo_frame"]
        self.bodies = [NS(name=n, kind="dynamic", render={"com_offset": np.zeros(3)}) for n in names]
        ipc = NS(sheets=[], bodies=self.bodies, reactions={}, body_pose=lambda i: (self.pos[i], self.rot[i]))
        self.env = NS(config=NS(task="lift-lid", dt=.01), world=NS(ipc=ipc, sim_time=0.,
                        scene=NS(preset="home", geometry={"box": self.g}, surface_z=.75)))
        if task == "put-lid-on":
            self.pos[1] = np.array([0., 0., .75])
        # Stub only geometry-to-world vertex plumbing; run the real lid verifier and reward.
        class E(Evaluator):
            def _lowest_z(ev, i):
                return float(self.pos[i][2])
            def _bottom_high_z(ev, i):
                return float(self.pos[i][2])
        self.ev = E(self.env, C.Config(preset="home", task=task))

    def window(self):
        for _ in range(self.ev.history.n):
            self.ev.record()


@pytest.mark.parametrize("task", ["lift-lid", "put-lid-on"])
def test_lid_progress_and_success(task):
    w = Home(task)
    assert w.ev.evaluate()["score"] == 0.
    if task == "lift-lid":
        w.pos[1][2] += .01
        assert 0. < w.ev.evaluate()["score"] < .9
        w.pos[1] = np.array([0., 0., .75])
    else:
        w.pos[1] = np.array([0., .10, .80])
        assert 0. < w.ev.evaluate()["score"] < .9
        w.pos[1] = np.array([0., .17, .7765])
    w.window()
    assert w.ev.evaluate()["score"] == 1.
    w.ev.history.rows[-1]["contact"] = True
    assert w.ev.evaluate()["score"] <= .9


@pytest.mark.parametrize("offset,rotation", [(np.array([.02, 0., 0.]), np.eye(3)),
                                            (np.zeros(3), yaw(10.)),
                                            (np.zeros(3), np.diag([1., -1., -1.])),
                                            (np.array([0., 0., -.04]), np.eye(3))])
def test_lid_on_rejects_wrong_pose(offset, rotation):
    w = Home("put-lid-on")
    w.pos[1] = np.array([0., .17, .7765]) + offset
    w.rot[1] = rotation
    w.window()
    assert not w.ev.evaluate()["success"]


def test_lid_on_engagement_needs_the_rim_inside_the_skirt():
    # Review: the vertical lid overlap reads full for a lid beside the box, so a lid half over the base
    # earned the whole engagement term (0.45 here) without being on it.
    w = Home("put-lid-on")
    w.pos[1] = np.array([0., .17 - .067, .75])           # half over the base in plan, not on it
    r = w.ev.evaluate()
    assert r["metrics"]["lid_overlap_mm"] > 20. and r["metrics"]["lid_skirt_engagement_mm"] == 0.
    assert r["metrics"]["lid_plan_fraction"] == pytest.approx(.5)
    assert r["score"] == pytest.approx(.9*.25)
    w.pos[1] = np.array([0., .17, .805 + .002])          # aligned, 2 mm above the rim: plan credit only
    assert w.ev.evaluate()["score"] == pytest.approx(.9*.5)
    w.pos[1] = np.array([0., .17, .805 - .010])          # aligned, skirt 10 mm down over the base
    r = w.ev.evaluate()
    assert r["metrics"]["lid_skirt_engagement_mm"] == pytest.approx(10.)
    assert r["score"] == pytest.approx(.9*.75) and not r["success"]
    assert S.lid_on(dict(lid_plan_fraction=1., lid_skirt_engagement_mm=0., lid_overlap_mm=55.,
                         base_lowest_above_table_mm=0., base_tilt_signed_deg=0.),
                    dict(lid_plan_fraction=0., lid_skirt_engagement_mm=0., lid_overlap_mm=55.), 1.) == .5


def test_lid_off_honours_whole_episode_damage():
    w = Home("lift-lid")
    w.pos[1] = np.array([0., 0., .75])
    w.window()
    w.ev.history.peak["base_lift_mm"] = 25.
    assert w.ev.evaluate()["score"] == 0.


FLAPS = ("major_a", "major_b", "minor_a", "minor_b")


class Carton:
    def __init__(self, monkeypatch, task):
        self.angles = dict.fromkeys(FLAPS, 1.)
        self.joined = 1.
        names = ["carton_body", "tea"]
        self.pos = [np.array([0., .15, .86]), np.array([0., .15, .91])]
        self.rot = [np.eye(3), np.eye(3)]
        _stub(monkeypatch, "gift_bags.carton", NS(FLAPS=FLAPS, CONTENTS=("tea",), L_IN=.35,
                            W_IN=.25, T_B=.004, Z_END=.21, flap_angles=lambda ipc: dict(self.angles),
                            tape_status=lambda ipc: {"seam_joined_fraction": self.joined}))
        ipc = NS(sheets=[], bodies=[NS(name=n, kind="dynamic", render={"com_offset": np.zeros(3)}) for n in names],
                 body_pose=lambda i: (self.pos[i], self.rot[i]), reactions={})
        env = NS(config=NS(task="open-carton", dt=.01), world=NS(ipc=ipc, sim_time=0.,
                     scene=NS(preset="receiving", geometry={}, surface_z=.86)))
        self.ev = Evaluator(env, C.Config(preset="receiving", task=task))


@pytest.mark.parametrize("task", ["open-flaps", "open-carton"])
def test_carton_zero_partial_release_window_and_success(monkeypatch, task):
    w = Carton(monkeypatch, task)
    assert w.ev.evaluate()["score"] == 0.
    w.angles[FLAPS[0]] = 45.
    partial = w.ev.evaluate()
    assert 0. < partial["score"] < .9
    w.angles = dict.fromkeys(FLAPS, 95.)
    w.joined = 0.
    for _ in range(w.ev.history.n):
        w.ev.record()
    assert w.ev.evaluate()["score"] == 1.
    w.ev.history.rows[0]["flaps"][FLAPS[0]] = 89.
    assert not w.ev.evaluate()["success"]
    w.pos[1][0] += .02
    assert not w.ev.evaluate()["checks"]["contents_in_place"]


@pytest.mark.parametrize("task", ["open-flaps", "open-carton"])
def test_carton_idle_springback_earns_nothing(monkeypatch, task):
    # Render-check probe: the untaped start's majors crept 0.61 -> 1.09 deg over 3 s hands-off (0.0025).
    w = Carton(monkeypatch, task)
    w.angles = dict(zip(FLAPS, (.605, .605, .034, .034)))
    w.ev = Evaluator(w.ev.env, w.ev.selection)
    first = w.ev.evaluate()
    assert first["score"] == 0.
    assert (first["task"], first["contract"]) == (task, C.CONTRACT)   # bare contract; the task is its own field
    w.angles = dict(zip(FLAPS, (1.087, 1.087, .053, .053)))
    for _ in range(w.ev.history.n):
        w.ev.record()
    assert w.ev.evaluate()["score"] == 0.
    w.angles[FLAPS[0]] = 45.
    assert 0. < w.ev.evaluate()["score"] < .9


def test_v1_scoring_is_unchanged():
    from gift_bags import task_set_scoring as S1
    m = dict(lid_plan_fraction=1., lid_skirt_engagement_mm=0., lid_overlap_mm=55., base_lowest_above_table_mm=0.,
             base_tilt_signed_deg=0.)
    assert S1.lid_on(m, dict(lid_plan_fraction=0., lid_overlap_mm=55.), 1.) == 1.
    assert not hasattr(S1, "FLAP_SPRINGBACK_DEG") and S.FLAP_SPRINGBACK_DEG == 10.


def test_unwrap_reward_zero_monotone_and_bad_end_state():
    initial = dict(box_coverage=.97)
    def result(coverage, overlap=26., tilt=0.):
        return S.unwrap(dict(box_coverage=coverage, lid_overlap_mm=overlap, box_tilt_deg=tilt), initial)
    assert result(.97) == 0.
    assert 0. < result(.6) < result(.3) < result(.05) == 1.
    assert result(.05, overlap=0.) == 0.
    assert result(.05, tilt=60.) == 0.
    assert S.score(True, 0.) == 1.
    assert S.score(False, 1.) == .9


# ----------------------------------------------------------------------------------------- start states
def test_bag_start_is_the_specs_worked_crease_open_tote(monkeypatch):
    cal = {"rest_sheet_rigid": .0001}
    _stub(monkeypatch, "gift_bags.ipcworld", NS(contact_caliper=lambda mat: cal))
    hook = lambda tag: NS(gift_bags_hook=tag)
    hinges, membrane, handles = map(hook, ("hinge_state", "membrane_set", "handles"))
    bag = NS(X2=np.array([[0., 0., 0.], [.1, .05, .254]]), mats=[object()], crease_rest={"a": 1.}, x0=None)
    bodies = [NS(name="board_0", p=np.zeros(3), R=np.eye(3)), NS(name="candle"), NS(name="knot_0")]
    spec = NS(sheets=[bag], rods=[object()], bodies=bodies, ipc_setup=[hinges, membrane, handles],
              geometry={"mat": {"center": (0., .17, .918)}, "board": {"half": (.1, .0495, .00025)}, "bag": {}})
    material = bag.mats
    released = copy.deepcopy(spec)
    prepare(spec, "bag-candle")
    # Spec (stand_open, bag_stand.py "worked"): crease rest = the open shape, shipping hinge memory and handles
    # dropped; material and the membrane/slack history kept.
    assert bag.mats is material and bag.crease_rest == {}
    assert spec.ipc_setup == [membrane]
    assert [b.name for b in spec.bodies] == ["board_0", "candle"]
    assert spec.bodies[0].p[2] > bag.x0[0, 2]
    from gift_bags.task_set_presets import prepare as prepare_v1
    prepare_v1(released, "bag-candle")
    assert released.sheets[0].crease_rest == {"a": 1.}       # released v1 keeps factory memory
    np.testing.assert_array_equal(released.sheets[0].x0, bag.x0)


def test_flaps_start_omits_only_tape_and_names_the_taped_worlds_caliper_stock(monkeypatch):
    st = object()
    calls = []
    _stub(monkeypatch, "gift_bags.carton", NS(settled=lambda z, xy: st,
                       install_hinges=lambda ipc, state: calls.append((ipc, state))))
    _stub(monkeypatch, "gift_bags.scene", NS(CARTON_XY=(0., .15)))
    thin, thick = NS(t=4.6e-5, name="tape"), NS(t=9.2e-5, name="tab")
    spec = NS(surface_z=.86, sheets=[NS(name="seam_tape", mats=[thin, thick])], ipc_setup=[], geometry={},
              bodies=[object()], contact_stock=None)
    before = list(spec.bodies)
    prepare(spec, "open-flaps")
    spec.ipc_setup[0]("ipc", None)
    assert calls == [("ipc", st)] and spec.sheets == [] and spec.bodies == before
    # Render check: without a sheet the world refused to build. The caliper stock is the one the taped
    # world takes from its thickest sheet stock.
    assert spec.contact_stock is thick
    assert spec.geometry["task_start"]["contact_stock"] == "tab"


def test_untaped_carton_keeps_the_receiving_worlds_contact_caliper():
    # Real carton/ipcworld arithmetic (pure NumPy) in a separate interpreter, so this module's import poison stays.
    code = """
import sys
from types import SimpleNamespace as NS
from gift_bags import carton as CT, ipcworld as W
from gift_bags.task_set_v2_presets import prepare
X2, T, tm, info = CT.tape_net()
tape = W.Sheet("seam_tape", X2, T, CT.tape_place(X2, info), [CT.Pm.BOPP_TAPE, CT.TAB_MAT], tm, md_angle=0.)
spec = NS(surface_z=.86, sheets=[tape], ipc_setup=[], geometry={}, bodies=[], contact_stock=None)
CT.settled = lambda z, xy: object()          # the cached settle is not needed for the caliper
prepare(spec, "open-flaps")
assert spec.sheets == [] and W.contact_caliper(spec.contact_stock) == CT.caliper(), spec.contact_stock
assert not any(m in sys.modules for m in ("warp", "newton", "mujoco"))
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_new_lid_placement_is_the_released_one():
    from gift_bags.task_set_presets import prepare as prepare_v1
    g = dict(hx=.09, hy=.065, wall=.0015, clearance=.0005)
    verts = np.array(list(itertools.product((-.092, .092), (-.067, .067), (0., .03))))
    specs = []
    for _ in range(2):
        lid = NS(name="gift_lid", p=np.array([0., .17, .75]), R=yaw(5.), verts=verts, render={"com_offset": np.zeros(3)})
        base = NS(name="gift_base", p=np.array([0., .17, .75]), R=yaw(5.), render={"com_offset": np.zeros(3)})
        specs.append(NS(bodies=[base, lid], surface_z=.75, geometry={"box": dict(g)}))
    prepare(specs[0], "put-lid-on")
    prepare_v1(specs[1], "put-lid-on")
    np.testing.assert_array_equal(specs[0].bodies[1].p, specs[1].bodies[1].p)
    assert specs[0].geometry == specs[1].geometry


# ----------------------------------------------------------------------------------------- env adapter and backend
def test_v2_adapter_maps_start_evaluator_and_reset_without_global_edits(monkeypatch):
    import threading
    from gift_bags.config import Config as OldConfig

    class Base:
        def __init__(self, config, *, output_dir, assisted_scene_hook):
            self.config, self.hook = config, assisted_scene_hook
            self.world = NS(c=config)
            self.lock, self.generation, self.initial = threading.RLock(), 0, object()
        def get_state(self, evaluate=True):
            return {"evaluation": {}}
        def restore(self, initial):
            assert initial is self.initial

    class Ev:
        def __init__(self, env, selection):
            self.env, self.selection = env, selection

    _stub(monkeypatch, "gift_bags.env", NS(GiftBagsEnv=Base))
    # Load a private copy so the stub superclass cannot contaminate other tests.
    spec = importlib.util.spec_from_file_location("gift_bags._task_set_v2_env_test", ROOT/"gift_bags/task_set_v2_env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "Evaluator", Ev)
    for entry in C.TASK_SET[:-1]:
        cfg = C.Config(**entry)
        e = module.GiftBagsEnv(cfg)
        assert isinstance(e.config, OldConfig) and type(e.config) is OldConfig
        assert e.config.task == ("bag-purchase" if cfg.task == "open-bag" else C.BASE_TASK.get(cfg.task, cfg.task))
        assert e.world.c is cfg
        assert (e.hook is not None) == (cfg.task in C.BASE_TASK or cfg.task == "open-bag")
        before = e.evaluator
        state = e.reset()
        assert e.generation == 1 and e.evaluator is not before
        assert state["instruction"] == V1.INSTRUCTIONS[cfg.task]
        assert state["config"]["task"] == cfg.task and state["config"]["contract"] == C.CONTRACT
        assert state["evaluation"]["contract"] == C.CONTRACT


def test_backend_builds_the_v2_env_only_for_the_v2_contract(monkeypatch, tmp_path):
    """Only gift-bags-tasks-v2 reaches gift_bags/task_set_v2_*.py; every other launch builds the released adapter."""
    class Built(Exception):
        pass

    def env(kind):
        def make(config, *, output_dir):
            raise Built(kind, config)
        return NS(GiftBagsEnv=make)

    _robots(monkeypatch)
    monkeypatch.setitem(sys.modules, "gift_bags.task_set_env", env("v1"))
    monkeypatch.setitem(sys.modules, "gift_bags.task_set_v2_env", env("v2"))
    monkeypatch.setitem(sys.modules, "gift_bags.kinematics", NS(ArmKinematics=None))
    monkeypatch.setitem(sys.modules, "gift_bags.robot_setup", NS(validate=lambda robot: robot, SINGLE=()))
    monkeypatch.setitem(sys.modules, "gift_bags.scene", NS(G1_LAYOUT={}))
    from gym_anything_robotics.backends.newton_gift_bags import NewtonGiftBagsBackend
    cases = [(dict(preset="home", task="lift-lid"), "v1"), (dict(preset="counter", task="bag-purchase"), "v1"),
             (dict(preset="counter"), "v1"), (dict(preset="receiving", task="open-carton", contract="legacy"), "v1"),
             (dict(preset="counter", task="bag-purchase", contract=V1.PURCHASE_CONTRACT), "v1"),
             (dict(preset="receiving", task="open-flaps"), "v1"),
             (dict(preset="home", task="lift-lid", contract=V1.CONTRACT), "v1"),
             *((dict(sel), "v2") for sel in C.TASK_SET[:-1]),
             (dict(preset="home", task="put-lid-on", config={"contract": C.CONTRACT}), "v2")]
    for n, (options, kind) in enumerate(cases):
        with pytest.raises(Built) as built:
            NewtonGiftBagsBackend(**options, output_dir=tmp_path/str(n), recording=False, cameras=())
        got, config = built.value.args
        assert got == kind, options
        if kind == "v1":
            assert type(config) is V1.Config, options
        else:
            assert type(config) is C.Config and config.contract == C.CONTRACT, options
    for bad in (dict(preset="counter", task="bag-purchase", contract=C.CONTRACT),
                dict(preset="home", task="lift-lid", contract=C.CONTRACT, config={"contract": V1.CONTRACT})):
        with pytest.raises(ValueError):
            NewtonGiftBagsBackend(**bad, output_dir=tmp_path/"bad", recording=False, cameras=())


def test_backend_variation_previews_in_a_v2_session():
    from gym_anything_robotics.backends.newton_gift_bags import NewtonGiftBagsBackend
    b = NewtonGiftBagsBackend.__new__(NewtonGiftBagsBackend)
    b.config = C.Config(preset="home", task="lift-lid")
    assert b._variation() == dict(preset="home", task="lift-lid", seed=0, contract=C.CONTRACT)
    same = b.select_variation(dict(task="lift-lid", contract=C.CONTRACT))
    assert same["applies_in_this_session"] and same["task_id"] == "gift-bags/v1/home/lift-lid/gift-bags-tasks-v2/seed-0"
    for other in (dict(task="lift-lid"), dict(task="lift-lid", contract=V1.CONTRACT)):
        assert not b.select_variation(other)["applies_in_this_session"]
    new = b.select_variation(dict(task="open-flaps", contract=C.CONTRACT))
    assert new["variation"] == dict(preset="receiving", task="open-flaps", seed=0, contract=C.CONTRACT)
    # A released session keeps the released catalogue.
    b.config = V1.Config(preset="home", task="lift-lid", contract=V1.CONTRACT)
    assert b._variation() == dict(preset="home", task="lift-lid", seed=0, contract=V1.CONTRACT)
    with pytest.raises(ValueError):
        b.select_variation(dict(task="lift-lid", contract=C.CONTRACT))


def test_delegated_bag_purchase_evaluator_is_untouched_by_v2(monkeypatch):
    # The eighth task is the released launch; the v2 verifier is never built for it.
    with pytest.raises(ValueError):
        Evaluator(NS(), C.Config(preset="counter", task="bag-purchase"))
    assert legacy_task.INSTRUCTIONS["bag-purchase"] == V1.instruction("bag-purchase")
