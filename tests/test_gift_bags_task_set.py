"""Constructed states only. No simulator import, server, physics step or renderer is allowed here."""
import copy
import importlib
import itertools
import json
import math
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from gift_bags import task as legacy_task, variants
from gift_bags import task_set_catalog as C, task_set_scoring as S
from gift_bags.task_set_evaluator import Evaluator, rectangle_coverage
from gift_bags.task_set_presets import prepare
from tests.packages import PACKAGES

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


def test_catalogue_resolver_and_legacy_defaults(monkeypatch):
    monkeypatch.setitem(sys.modules, "robots", NS())
    monkeypatch.setitem(sys.modules, "robots.catalog", NS(default_robot=lambda env: "dual-fr3",
                       environment_robots=lambda env: ["dual-fr3"], ENVIRONMENTS={}))
    from gym_anything_robotics import tasks
    raw = json.loads((ROOT/"gift_bags/benchmark.json").read_text())
    entries = raw["task_set"]["tasks"]
    assert len(entries) == 8
    assert [e["selection"] for e in entries] == C.TASK_SET
    assert raw["count"] == 9
    for e in entries:
        resolved = tasks.resolve("gift-bags", e["selection"])
        assert resolved.task_id == e["task_id"]
        assert resolved.launch.backend_options == e["selection"]
        cfg = C.Config(**resolved.launch.backend_options)
        assert C.evaluation_contract(cfg.task, cfg.contract) == e["task_contract"]
    for p, ts in C.LEGACY_TASKS.items():
        if p == "dev":
            continue
        for t in ts:
            for seed in (0, 17):
                assert C.task_resolve(dict(preset=p, task=t, seed=seed)) == variants.task_resolve(dict(preset=p, task=t, seed=seed))
    assert C.task_resolve({}) == variants.task_resolve({})
    assert C.contract()["task_set"] == C.TASK_SET
    assert C.INSTRUCTIONS["bag-purchase"] == legacy_task.INSTRUCTIONS["bag-purchase"]


@pytest.mark.parametrize("bad", [dict(seed=True), dict(seed="1"), dict(seed=-1), dict(seed=10000),
                                dict(task="open-flaps", preset="home"), dict(task="bag-candle", contract="legacy"),
                                dict(task="bag-purchase", contract=C.CONTRACT), dict(task="open-present", contract=C.CONTRACT)])
def test_bad_selections(bad):
    with pytest.raises((ValueError, TypeError)):
        C.task_resolve(bad)


def test_config_validation_still_uses_original_types_and_defaults():
    from gift_bags.config import Config as Old
    from dataclasses import asdict
    for seed in (True, 0, 17):  # Config historically accepts bool, unlike the catalogue; preserve that distinction.
        old = asdict(Old(seed=seed))
        new = asdict(C.Config(seed=seed))
        new.pop("contract")
        assert new == old
    with pytest.raises(TypeError):
        C.Config(typo=True)
    with pytest.raises(ValueError):
        C.Config(preset="home", task="open-flaps")


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
        self.ev = Evaluator(self.env, C.Config(task=task, contract=C.CONTRACT))

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
    assert w.solve()["score"] == 1.
    # Release is required even if geometry is complete.
    w.ev.history.rows[-1]["contact"] = True
    held = w.ev.evaluate()
    assert not held["success"] and 0. < held["score"] <= .9
    w.ev.history.rows[-1]["contact"] = False
    # A board under the bag cannot count as inside.
    w.positions["board_0"][2] -= .001
    assert not w.ev.evaluate()["checks"]["board_in_bag"]
    w.solve()
    # Inverted candle cannot pass purchase; either hand can leave the valid state.
    if task == "bag-candle":
        w.rotations["candle"] = np.diag([1., -1., -1.])
        assert not w.ev.evaluate()["checks"]["candle_upright"]
    w.crack = .026
    out = w.ev.evaluate()
    assert not out["success"] and out["score"] == 0.


def test_counter_pinch_inversion_and_board_choice(monkeypatch):
    w = Counter(monkeypatch)
    w.solve()
    # Mid-body pinch: hull still has full width/depth; the face-gap guard must fail.
    w.x[8:10, 1] = .17 + np.array([-.002, .002])
    assert not w.ev.evaluate()["checks"]["mouth_open"]
    w.x = w.open.copy()
    w.x[:, 2] = 2*.9181 + w.H - w.x[:, 2]
    assert not w.ev.evaluate()["checks"]["standing"]


def test_original_evaluator_is_exactly_delegated(monkeypatch):
    w = Counter(monkeypatch)
    w.ev.selection = C.Config(task="bag-purchase")
    w.ev.modern = False
    assert w.ev.evaluate()["score"] == 0.
    w.solve()
    got = w.ev.evaluate()
    expected = legacy_task.Evaluator.evaluate(w.ev)
    assert got.pop("contract") == C.PURCHASE_CONTRACT
    assert got == expected


def test_bag_and_candle_partial_states_get_credit(monkeypatch):
    w = Counter(monkeypatch)
    w.x = w.open.copy()
    w.x[:, 1] = .17 + .55*(w.x[:, 1] - .17)
    partial = w.ev.evaluate()["score"]
    assert 0. < partial < .9
    w.x = w.open.copy()
    assert w.ev.evaluate()["score"] > partial
    w = Counter(monkeypatch, "bag-candle")
    w.positions["candle"] = np.array([0., .17, 1.15])  # partly below the rim, still high above the board
    assert 0. < w.ev.evaluate()["score"] < .9
    assert w.solve()["score"] == 1.


def yaw(degrees):
    a = math.radians(degrees)
    return np.array([[math.cos(a), -math.sin(a), 0.], [math.sin(a), math.cos(a), 0.], [0., 0., 1.]])


class Home:
    def __init__(self, task):
        self.task = task
        self.g = dict(hx=.09, hy=.065, base_h=.055, lid_h=.03, wall=.0015, clearance=.0005)
        self.pos = [np.array([0., .17, .75]), np.array([0., .17, .7765]), np.array([0., .17, .76])]
        self.rot = [np.eye(3) for _ in self.pos]
        self.half = [(.09, .065, .055), (.092, .067, .03), (.085, .06, .015)]
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
        self.ev = E(self.env, C.Config(preset="home", task=task, contract=C.CONTRACT))

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
        self.ev = Evaluator(env, C.Config(preset="receiving", task=task, contract=C.CONTRACT))


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


def test_unwrap_evaluator_reads_paper_state():
    w = Home("lift-lid")
    w.env.config.task = "unwrap-gift"
    samples = []
    for i, b in enumerate(w.bodies[:2]):
        hx, hy, height = w.half[i]
        half, centre = np.array([hx, hy, height/2]), np.array([0., 0., height/2])
        b.render["parts"] = [(half, centre)]
        for ax, sign in itertools.product(range(3), (-1., 1.)):
            other = [a for a in range(3) if a != ax]
            for u, v in itertools.product(np.linspace(-1., 1., 7), repeat=2):
                p = centre.copy()
                p[ax] += sign*half[ax]
                p[other[0]] += u*half[other[0]]
                p[other[1]] += v*half[other[1]]
                samples.append(p + w.pos[i])
    # Tiny triangles at surface samples model intact covering paper; no contact or dynamics engine involved.
    x = np.array([p + d for p in samples for d in ([0., 0., 0.], [.0001, 0., 0.], [0., .0001, 0.])])
    ipc = w.env.world.ipc
    ipc.sheets = [NS(tris=np.arange(len(x)).reshape(-1, 3))]
    ipc.sheet_slices = [slice(0, len(x))]
    ipc.sheet_of_tri = np.zeros(len(x)//3, int)
    ipc.damage = np.zeros(len(x)//3)
    ipc.stats = {}
    ipc.crack_length = lambda k: (0., 0.)
    ipc.x_free = lambda: x
    ev = Evaluator(w.env, C.Config(preset="home", task="unwrap-gift", contract=C.CONTRACT))
    initial = ev.evaluate()
    assert initial["metrics"]["box_coverage"] > .95
    assert initial["score"] == 0. and not initial["success"]
    x[:len(x)//2] += [0., 2., 0.]
    assert 0. < ev.evaluate()["score"] < .9
    x[len(x)//2:] += [0., 2., 0.]
    for _ in range(ev.history.n):
        ev.history.add(dict(contact=False), {})
    assert ev.evaluate()["score"] == 1.
    w.pos[1][2] += .03
    assert not ev.evaluate()["checks"]["box_closed"]


def test_rectangle_overlap_is_continuous():
    rect = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    assert rectangle_coverage(rect, 1., 1.) == 1.
    assert rectangle_coverage(rect + [1., 0.], 1., 1.) == .5
    assert rectangle_coverage(rect + [2., 0.], 1., 1.) == 0.


def test_bag_start_preserves_material_and_factory_memory(monkeypatch):
    cal = {"rest_sheet_rigid": .0001}
    _stub(monkeypatch, "gift_bags.ipcworld", NS(contact_caliper=lambda mat: cal))
    hook = lambda tag: NS(gift_bags_hook=tag)
    hinges, membrane, handles = map(hook, ("hinge_state", "membrane_set", "handles"))
    bag = NS(X2=np.array([[0., 0., 0.], [.1, .05, .254]]), mats=[object()], crease_rest={"a": 1.}, x0=None)
    bodies = [NS(name="board_0", p=np.zeros(3), R=np.eye(3)), NS(name="candle"), NS(name="knot_0")]
    spec = NS(sheets=[bag], rods=[object()], bodies=bodies, ipc_setup=[hinges, membrane, handles],
              geometry={"mat": {"center": (0., .17, .918)}, "board": {"half": (.1, .0495, .00025)}, "bag": {}})
    material, crease = bag.mats, copy.deepcopy(bag.crease_rest)
    prepare(spec, "bag-candle")
    assert bag.mats is material and bag.crease_rest == crease
    assert spec.ipc_setup == [hinges, membrane]
    assert [b.name for b in spec.bodies] == ["board_0", "candle"]
    assert spec.bodies[0].p[2] > bag.x0[0, 2]


def test_flaps_start_omits_only_tape_and_preserves_hinges(monkeypatch):
    st = object()
    calls = []
    _stub(monkeypatch, "gift_bags.carton", NS(settled=lambda z, xy: st,
                       install_hinges=lambda ipc, state: calls.append((ipc, state))))
    _stub(monkeypatch, "gift_bags.scene", NS(CARTON_XY=(0., .15)))
    spec = NS(surface_z=.86, sheets=[NS(name="seam_tape")], ipc_setup=[], geometry={}, bodies=[object()])
    before = list(spec.bodies)
    prepare(spec, "open-flaps")
    spec.ipc_setup[0]("ipc", None)
    assert calls == [("ipc", st)] and spec.sheets == [] and spec.bodies == before


def test_new_lid_placement_table_clearance_and_reach_bounds():
    # All box sizes, extreme seeds and clearances; these are geometric bounds, not an IK certificate.
    sizes = [(.09, .065, .0015), (7.5*.0254/2 + .0015, 7.5*.0254/2 + .0015, .0015),
             (11*.0254/2 + .032*.0254, 8.5*.0254/2 + .032*.0254, .032*.0254)]
    for (hx, hy, wall), dx, dy, angle, clearance in itertools.product(sizes, (-.03, .03), (-.02, .02), (-15., 15.), (.0005, .001)):
        R = yaw(angle)
        g = dict(hx=hx, hy=hy, wall=wall, clearance=clearance)
        origin = np.array([dx, .17 + dy, .75])
        verts = np.array(list(itertools.product((-(hx+clearance+wall), hx+clearance+wall),
                                               (-(hy+clearance+wall), hy+clearance+wall), (0., .03))))
        lid = NS(name="gift_lid", p=origin.copy(), R=R, verts=verts, render={"com_offset": np.zeros(3)})
        base = NS(name="gift_base", p=origin, R=R, render={"com_offset": np.zeros(3)})
        spec = NS(bodies=[base, lid], surface_z=.75, geometry={"box": g})
        prepare(spec, "put-lid-on")
        corners = verts @ R.T + lid.p
        assert np.max(np.abs(corners[:, 0])) < .75
        assert corners[:, 1].min() > -.47 and corners[:, 1].max() < .43
        local = (lid.p - origin) @ R
        assert local[1] + hy + clearance + wall < -hy
        for sign in (-1., 1.):
            grasp = lid.p + R @ np.array([sign*(hx + clearance + wall + .02), 0., .018])
            mount = np.array([sign*.30, .55, .75])
            assert np.linalg.norm(grasp - mount) < .855


def test_new_counter_placements_within_robot_reach_envelope():
    mounts = [np.array([sign*.30, .55, .915]) for sign in (-1., 1.)]
    # Open tote's mouth/rim and the preplaced board: no target farther away than these corners.
    for x, y, z in itertools.product((-.1016, .1016), (.17-.0508, .17+.0508), (.918, 1.172)):
        assert min(np.linalg.norm(np.array([x, y, z]) - m) for m in mounts) < .855
    # Customer candle's two opposed end grasps, including translation/yaw extremes and lift to mouth height.
    for dx, dy, a, z in itertools.product((-.02, .02), (-.02, .02), (-30., 30.), (.97, 1.20)):
        centre, R = np.array([dx, -.07+dy, z]), yaw(a)
        for sign, mount in zip((-1., 1.), mounts):
            grasp = centre + R @ np.array([sign*.065, 0., 0.])
            assert np.linalg.norm(grasp - mount) < .855


def test_backend_variation_contract_previews_are_distinct():
    from gym_anything_robotics.backends.newton_gift_bags import NewtonGiftBagsBackend
    b = NewtonGiftBagsBackend.__new__(NewtonGiftBagsBackend)
    b.config = C.Config(preset="home", task="lift-lid", contract=C.CONTRACT)
    same = b.select_variation(dict(task="lift-lid", contract=C.CONTRACT))
    assert same["applies_in_this_session"]
    old = b.select_variation(dict(task="lift-lid"))
    assert not old["applies_in_this_session"]
    new = b.select_variation(dict(task="open-flaps"))
    assert new["variation"]["preset"] == "receiving"
    assert new["variation"]["contract"] == C.CONTRACT


@pytest.mark.parametrize("method", ["test_refused_commands_raise_action_rejected_and_write_nothing",
                                    "test_legal_commands_write_grip_then_targets",
                                    "test_variation_contract_is_the_catalogue_and_selection_is_launch_time"])
def test_existing_stubbed_backend_regressions(method, tmp_path):
    from tests.test_gift_bags_backend import GiftBagsAdapterContractTest
    case = GiftBagsAdapterContractTest(method)
    case.tmp = NS(name=str(tmp_path))
    getattr(case, method)()


def test_adapter_maps_start_evaluator_and_reset_without_global_edits(monkeypatch):
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
    spec = importlib.util.spec_from_file_location("gift_bags._task_set_env_test", ROOT/"gift_bags/task_set_env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "Evaluator", Ev)
    for entry in C.TASK_SET:
        cfg = C.Config(**entry)
        e = module.GiftBagsEnv(cfg)
        assert isinstance(e.config, OldConfig)
        assert e.config.task == ("bag-purchase" if cfg.task == "open-bag" else C.BASE_TASK.get(cfg.task, cfg.task))
        assert e.world.c is cfg
        before = e.evaluator
        state = e.reset()
        assert e.generation == 1 and e.evaluator is not before
        assert state["instruction"] == C.instruction(cfg.task, cfg.contract)
        assert state["config"]["task"] == cfg.task
        assert state["evaluation"]["contract"] == C.evaluation_contract(cfg.task, cfg.contract)
    # Bare open-bag keeps its original world and evaluator task (including old score and contact window).
    e = module.GiftBagsEnv(C.Config())
    assert e.config.task == "open-bag" and e.hook is None
