"""filleting-tasks-v2 (the render-checked set, filleting/task_set_v2_*.py): the fix branch's tests on the v2 contract.

Constructed-state tests only. No simulator, server, renderer or robot runtime.

The legacy module's geometry imports are stubbed with physical poses/arrays;
NumPy/SciPy run the real graph and evaluator arithmetic.
"""
import copy
import json
import sys
import threading
from collections import deque
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from filleting import layout as LY, variants as legacy
from filleting.task import check
from filleting import task_set_catalogue as v1
# task_resolve is the catalogue's resolver entry point (benchmark.json "resolver"), which dispatches v2 selections
from filleting.task_set_catalogue import task_resolve
from filleting.task_set_v2_catalogue import (CONTRACT, EIGHT, INSTRUCTIONS, NEW_TASK, REVISED,
                                             TASK_SET, TASK_SLUG, resolve, start_options)
from filleting.task_set_v2_evaluator import TaskSetV2Evaluator
from filleting.task_set_progress import Connections, min_connection, knife_return_geometry, gained, ramp

from tests.packages import PACKAGES, REPO

ROOT = PACKAGES


class Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


@pytest.fixture
def geometry(monkeypatch):
    workpiece = ModuleType("filleting.workpiece")
    workpiece.blade_segment_poses = lambda w: w.poses
    workpiece._R = lambda q: np.eye(3)
    scene = ModuleType("filleting.scene")
    scene.INSERT = {"lip_height": .004}
    scene.HANDLE_H = .032
    world = ModuleType("filleting.world")
    world.KNIFE_PINCH_N = .5
    monkeypatch.setitem(sys.modules, "filleting.workpiece", workpiece)
    monkeypatch.setitem(sys.modules, "filleting.scene", scene)
    monkeypatch.setitem(sys.modules, "filleting.world", world)


@pytest.fixture
def robot_catalog(monkeypatch):
    robots, catalog = ModuleType("robots"), ModuleType("robots.catalog")
    catalog.ENVIRONMENTS = {"filleting": {"aliases": ["dual-fr3", "single-fr3", "dual-panda"]}}
    catalog.default_robot = lambda name: "dual-fr3"
    catalog.environment_robots = lambda name: ["dual-fr3", "single-fr3", "dual-panda"]
    catalog.select_for_backend = lambda name, robot: robot
    robots.catalog = catalog
    monkeypatch.setitem(sys.modules, "robots", robots)
    monkeypatch.setitem(sys.modules, "robots.catalog", catalog)


def test_exact_catalogue_and_legacy_defaults(robot_catalog):
    from gym_anything_robotics import tasks
    data = json.loads((ROOT/"filleting/benchmark.json").read_text())
    selections = data["task_set"]["selections"]
    assert selections == list(TASK_SET)
    # Seven under the v2 contract; the audited fillet-whole-fish keeps its released selection, exactly as in the split.
    split = json.loads((REPO/"benchmarks/roboworld/splits/test.json").read_text())["tasks"]
    assert selections[-1] == next(t["selection"] for t in split if t["environment_id"] == "filleting")
    assert [s.get("contract") for s in selections] == [CONTRACT]*7 + [None]
    ids = set()
    for selection in TASK_SET:
        result = tasks.resolve("filleting", selection)
        ids.add(result.task_id)
        # A task scored under the new contract has the contract in its id; the audited A3 keeps its id.
        assert (f"/{CONTRACT}/" in result.task_id) == ("contract" in selection)
        assert result.description == INSTRUCTIONS[legacy.SLUG_TASK.get(selection["task"], NEW_TASK)]
        assert result.launch.backend_options.get("contract") == selection.get("contract")
        assert tasks.TaskSpec.from_dict(result.to_dict()).to_dict() == result.to_dict()
    assert len(ids) == 8
    # The released set is unchanged: task_set_v1 resolves to its released ids under filleting-tasks-v1.
    assert [r["selection"] for r in data["task_set_v1"]["tasks"]] == list(v1.TASK_SET)
    for row in data["task_set_v1"]["tasks"]:
        assert tasks.resolve("filleting", row["selection"]).task_id == row["task_id"]
    assert task_resolve({}) == legacy.task_resolve({})
    for code in legacy.INSTRUCTIONS:
        if code not in legacy.UNSUPPORTED:
            for seed in (0, 7, 999):
                selection = {"task": code, "seed": seed}
                assert task_resolve(selection) == legacy.task_resolve(selection)
                assert task_resolve(dict(selection, contract="legacy")) == legacy.task_resolve(selection)
    assert task_resolve(TASK_SET[-1]) == legacy.task_resolve(TASK_SET[-1])
    assert len(data["selector_choices"]["task"]) == data["count"] == 12
    assert data["selector_choices"]["contract"] == ["legacy", v1.CONTRACT, CONTRACT]
    for code in (*REVISED, NEW_TASK):
        for seed in (0, 7):
            legacy_id = legacy.task_resolve({"task": TASK_SLUG["A0" if code == NEW_TASK else code],
                                             "seed": seed})["task_id"].replace("/take-knife/", f"/{TASK_SLUG[code]}/")
            released = task_resolve({"task": TASK_SLUG[code], "seed": seed, "contract": v1.CONTRACT})
            new = task_resolve({"task": TASK_SLUG[code], "seed": seed, "contract": CONTRACT})
            assert released["task_id"] == legacy_id      # filleting-tasks-v1 kept the legacy ids
            assert new["task_id"] == legacy_id.replace(f"/seed-{seed}", f"/{CONTRACT}/seed-{seed}")
            # Same task, start and instruction as filleting-tasks-v1; only the contract differs.
            assert new["description"] == released["description"]
            assert new["selection"] == dict(released["selection"], contract=CONTRACT)
            assert new["backend_options"] == dict(released["backend_options"], contract=CONTRACT)


@pytest.mark.parametrize("seed", [True, False, -1, 1000, 1.5, "3", None])
@pytest.mark.parametrize("task", ["take-knife", "put-away-knife", "fillet-whole-fish"])
def test_seed_validation_unchanged(seed, task):
    with pytest.raises(ValueError):
        resolve(task=task, seed=seed)
    with pytest.raises(ValueError):
        task_resolve(dict(task=task, seed=seed, contract=CONTRACT))


@pytest.mark.parametrize("selection", [dict(task="pin-bone"), dict(task="skin"), dict(task="missing"),
    dict(task="put-away-knife", preset="salmon-side"), dict(task="take-knife", contract="missing"),
    dict(task="fillet-whole-fish", contract=CONTRACT), dict(task="put-away-knife", contract="legacy"),
    dict(task="pin-bone", contract=CONTRACT), dict(task="put-away-knife", preset="salmon-side", contract=CONTRACT),
    dict(task="fillet-one-side", contract=CONTRACT), dict(task="trim-belly", contract=CONTRACT)])
def test_invalid_selections(selection):
    with pytest.raises(ValueError):
        task_resolve(selection)


def test_start_has_no_new_placements_and_does_not_mutate_config():
    c = {"collision_profile": "contact", "knife_start": None}
    carrier, values = start_options(resolve(task="put-away-knife"), c)
    assert carrier == "A0" and values["knife_start"] == "hand"
    assert c["knife_start"] is None
    for task in EIGHT:
        if task != NEW_TASK:
            chosen = resolve(task=task) if task in REVISED else v1.resolve(task=task)
            assert start_options(chosen, c) == (task, c)
    with pytest.raises(ValueError):
        start_options(resolve(task=NEW_TASK), {"knife_start": "block"})


def test_graph_reads_parallel_connections_bonds_and_joints():
    # Two independent paths from head to body: joint 0-3, flesh 0-1-2-3.
    edges = np.array([(0, 3), (0, 1), (1, 2), (2, 3)])
    assert min_connection(4, edges, [0], [3]) == 2
    assert min_connection(4, edges[1:], [0], [3]) == 1
    assert min_connection(4, edges[[0, 1, 3]], [0], [3]) == 1
    assert min_connection(4, edges[[1, 3]], [0], [3]) == 0
    assert min_connection(4, edges[::-1, ::-1], [0], [3]) == 2
    with pytest.raises(ValueError):
        min_connection(4, edges, [0], [0])


def connection_world():
    # Two tets split at an authored interface; a cohesive bond joins them.
    ts = NS(n=8, hactive=np.ones(2), htets=np.arange(8).reshape(2, 4), hX=np.zeros((8, 3)),
            n_bonds=1, bonds={"a": Array([0]), "b": Array([4]), "damage": Array([0.])},
            n_bties=2, bties={"node": Array([0, 4]), "body": Array([0, 1]), "damage": Array([0., 0.])},
            n_joints=1, joints={"a": Array([0]), "b": Array([1]), "on": Array([1])},
            body_states=lambda: {"p": np.zeros((2, 3))})
    cut = NS(orig=ts.htets.copy(), _nbr=np.array([[1, -1, -1, -1], [0, -1, -1, -1]]), cut_faces={(0, 1)})
    return NS(tissue=ts, cut=cut)


def test_actual_graph_uses_damage_and_intact_backbone():
    w = connection_world()
    g = Connections(w)
    assert g.remaining([2], [3]) == 2
    w.tissue.bonds["damage"].value[:] = 1.  # tearing is a valid separation too
    assert g.remaining([2], [3]) == 1
    w.tissue.joints["on"].value[:] = 0
    assert g.remaining([2], [3]) == 0
    assert g.remaining([2], [3], start=True) == 2
    w.tissue.bonds["damage"].value[:] = 0.
    w.tissue.bties["damage"].value[0] = 1.
    assert g.remaining([2], [3]) == 0


def test_partial_slice_follows_actual_cut_and_does_not_pay_completed_cuts():
    g = object.__new__(Connections)
    g.n = g.nt = 4
    g.centres = np.array([[-.02, 0, 0], [0, 0, 0], [0, 0, 0], [.02, 0, 0]])
    g.start = np.array([(0, 1), (1, 3), (0, 2), (2, 3)])
    g.initial_faces = set()
    g.world = NS(cut=NS(cut_faces={(0, 1)}))
    lab, frags = np.zeros(4, int), {0: frag(.15, [0, 1, 2, 3])}
    assert g.partial_slice(lab, frags, g.start[1:], .001) == .5
    g.world.cut.cut_faces = {(0, 2)}
    assert g.partial_slice(lab, frags, g.start[[0, 1, 3]], .001) == .5
    g.world.cut.cut_faces = {(0, 1), (0, 2)}
    lab = np.array([0, 1, 1, 1])
    assert g.partial_slice(lab, {0: frag(.15, [0]), 1: frag(.15, [1, 2, 3])}, g.start[[1, 3]], .001) == 0.


class KnifeWorld:
    def __init__(self):
        # Slot at the real block location with the real down/back 70 degree axis.
        theta = np.radians(70.)
        xdir = np.array([0., np.cos(theta), -np.sin(theta)])
        zdir = np.array([0., np.sin(theta), np.cos(theta)])
        target_R = np.column_stack((xdir, np.cross(zdir, xdir), zdir))
        self.target = np.array([.298, .55, 1.075])
        self.scene = NS(blade=NS(L=.173, H=.025, seg_len=.173/5, segments=5),
                        block_knife_pose=(self.target, target_R))
        self.config = NS(task="A0", rigid_dt=.001, mesh_h=.0035)
        self.rig = NS(instances={"left": None, "right": None})
        self.kin = NS(forward=lambda arm: np.eye(4))
        self.cut = NS(cut_faces=set())
        self.assembly = {"interface_faces": set()}
        self.fault = None
        self.knife_mode, self.knife_start = "free", "block"
        self.pinch = {"left": [0., 0.], "right": [0., 0.]}
        self.robot_force = 0.
        self.offset, self.axis = np.zeros(3), 0.
        self.tissue = NS(body_states=lambda: {"p": np.vstack((self.handle, self.centres))})
        self.place(self.target, target_R)

    def place(self, heel, R):
        self.heel, self.R = np.array(heel), np.array(R)
        self.poses = [(self.R, self.heel + self.R[:, 0]*((i + .5)*.173/5)) for i in range(5)]
        self.centres = np.array([p + self.R[:, 2]*.0125 for _, p in self.poses])
        self.handle = self.heel - self.R[:, 0]*.06 + self.R[:, 2]*.025

    def pinch_forces(self):
        return self.pinch

    def body_pose(self, name):
        return self.handle, self.R

    def knife_in_grip(self):
        return self.offset, self.axis

    def _knife_contacts(self):
        return self.robot_force, 0.


def knife_env(task):
    w = KnifeWorld()
    if task == NEW_TASK:
        w.place([0., .28, 1.15], np.eye(3))
        w.pinch["right"] = [35., 35.]
    return NS(world=w, config=w.config, sim_time=0., steps=0, _robot_boxes=[],
              blade_robot_contacts=0, blade_left_hand_contacts=0, knife_held_since=None, grip_history=deque(maxlen=64))


def record_grip(env):
    """The base env's 50 ms right-hand grip record (FilletingEnv._record), which the original A0 verdict reads."""
    w = env.world
    w.knife_mode = "held" if any(min(f) > .5 for f in w.pinch.values()) else "free"
    if w.knife_mode == "held" and min(w.pinch["right"]) > .5:
        if env.knife_held_since is None:
            env.knife_held_since = env.sim_time
        off, ang = w.knife_in_grip()
        env.grip_history.append((env.sim_time, np.asarray(off, float), float(ang)))
    else:
        env.knife_held_since = None
        env.grip_history.clear()


def advance(ev, end, step=.05):
    while ev.env.sim_time < end - 1e-9:
        ev.env.sim_time = round(ev.env.sim_time + step, 9)
        ev.env.steps = round(ev.env.sim_time/ev.env.config.rigid_dt)
        record_grip(ev.env)
        ev.monitor()


def test_take_knife_start_idle_partial_and_success(geometry):
    env = knife_env("A0")
    ev = TaskSetV2Evaluator(env, "A0")
    assert ev.evaluate()["score"] == 0.
    advance(ev, 4.)
    assert ev.evaluate()["score"] == 0.
    env.world.place([-.08, .28, 1.], np.eye(3))
    env.world.pinch["right"] = [20., 20.]
    advance(ev, 5.)
    part = ev.evaluate()
    assert 0 < part["score"] < .95 and not part["success"]
    advance(ev, 8.1)
    result = ev.evaluate()
    assert result["success"] and result["score"] == 1.
    assert result["contract"] == CONTRACT
    env.blade_robot_contacts = 1
    result = ev.evaluate()
    assert not result["success"] and result["score"] <= .475


def test_take_knife_success_is_the_original_right_hand_verdict(geometry):
    # The spec's success is the current _eval_A0, check for check: no left-hand alternative.
    env = knife_env("A0")
    ev = TaskSetV2Evaluator(env, "A0")
    env.world.place([-.08, .28, 1.], np.eye(3))
    env.world.pinch["right"] = [20., 20.]
    advance(ev, 4.)
    result = ev.evaluate()
    original, _ = LY.task_verifier("A0").released_checks(ev)
    assert result["success"] and set(result["checks"]) == set(original) | {"time"}
    assert {k: c["passed"] for k, c in result["checks"].items()} == {k: c["passed"] for k, c in original.items()}
    # The left hand touching the knife fails the original not_touched_by_left_hand check.
    env.world.pinch["left"] = [0., 1.]
    result = ev.evaluate()
    assert not result["success"] and not result["checks"]["not_touched_by_left_hand"]["passed"]
    env.world.pinch["left"] = [0., 0.]
    env.blade_left_hand_contacts = 1
    assert not ev.evaluate()["checks"]["not_touched_by_left_hand"]["passed"]


def test_take_knife_progress_measures_the_knife_whichever_hand_holds_it(geometry):
    # Recipe rule 4: progress is independent of hand. The same steady hold edge-down over the board earns the same
    # progress in either hand; success stays the original right-hand verdict.
    results = {}
    for arm in ("right", "left"):
        env = knife_env("A0")
        ev = TaskSetV2Evaluator(env, "A0")
        env.world.place([-.08, .28, 1.], np.eye(3))
        env.world.pinch[arm] = [20., 20.]
        advance(ev, 1.5)
        results[arm] = ev.evaluate()
        assert results[arm]["metrics"]["progress_hand"] == arm
    assert not results["right"]["success"] and not results["left"]["success"]      # held 1.5 s of 3 s
    assert 0 < results["left"]["score"] < .95
    assert results["left"]["score"] == pytest.approx(results["right"]["score"])


def test_take_knife_left_hand_hold_earns_progress_never_success(geometry):
    env = knife_env("A0")
    ev = TaskSetV2Evaluator(env, "A0")
    env.world.place([-.08, .28, 1.], np.eye(3))
    env.world.pinch["left"] = [20., 20.]
    advance(ev, 8.)
    result = ev.evaluate()
    # A full, steady, edge-down left-hand hold: all the progress, the failure score's 0.95 ceiling, no success.
    assert not result["success"] and result["score"] == pytest.approx(.95)
    assert result["metrics"]["progress"]["value"] == pytest.approx(1.)
    assert not result["checks"]["held_by_right_hand_s"]["passed"]
    assert not result["checks"]["not_touched_by_left_hand"]["passed"]
    # Released, or back in the block, it earns nothing again.
    env.world.pinch["left"] = [0., 0.]
    advance(ev, 8.1)
    assert ev.evaluate()["score"] == 0.


def test_grip_slip_and_release_destroy_hold(geometry):
    env = knife_env("A0")
    ev = TaskSetV2Evaluator(env, "A0")
    env.world.place([-.08, .28, 1.], np.eye(3))
    env.world.pinch["right"] = [20., 20.]
    advance(ev, 4.)
    assert ev.evaluate()["success"]
    env.world.offset = np.array([.01, 0., 0.])   # the handle slides 10 mm along the grip
    env.world.place([-.07, .28, 1.], np.eye(3))
    advance(ev, 4.1)
    assert not ev.evaluate()["success"]
    assert not ev.evaluate()["checks"]["slip_mm_over_3s"]["passed"]
    env.world.pinch["right"] = [0., 0.]
    advance(ev, 4.2)
    assert ev.evaluate()["score"] == 0.


def test_return_start_success_hold_cut_floor_and_time(geometry):
    env = knife_env(NEW_TASK)
    ev = TaskSetV2Evaluator(env, NEW_TASK)
    assert ev.evaluate()["score"] == 0.
    advance(ev, 2.)
    assert ev.evaluate()["score"] == 0.
    env.world.place(*env.world.scene.block_knife_pose)
    advance(ev, 3.2)
    held = ev.evaluate()
    assert not held["success"] and 0 < held["score"] <= .95
    env.world.pinch["right"] = [0., 0.]
    result = ev.evaluate()
    assert result["success"] and result["score"] == 1.
    env.world.robot_force = 1.
    assert not ev.evaluate()["success"]
    env.world.robot_force = 0.
    env.world.cut.cut_faces.add((1, 2))
    assert ev.evaluate()["score"] == 0.
    env.world.cut.cut_faces.clear()
    env.sim_time = 60.01
    assert not ev.evaluate()["success"]
    env.world.place([0., .28, .1], np.eye(3))
    assert ev.evaluate()["score"] == 0.


def test_return_stillness_reads_whole_trailing_window(geometry):
    env = knife_env(NEW_TASK)
    ev = TaskSetV2Evaluator(env, NEW_TASK)
    env.world.pinch["right"] = [0., 0.]
    env.world.place(*env.world.scene.block_knife_pose)
    advance(ev, 1.2)
    assert ev.evaluate()["success"]
    target, R = env.world.scene.block_knife_pose
    env.world.place(target + [0., .01, 0.], R)
    advance(ev, 1.3)
    env.world.place(target, R)
    advance(ev, 1.4)
    assert not ev.evaluate()["success"]  # identical endpoints cannot hide the excursion
    advance(ev, 2.5)
    assert ev.evaluate()["success"]


def test_slot_is_finite_and_bending_is_visible():
    centres = np.array([[.03, 0., .0125], [.14, 0., .0125]])
    def geom(c, heel=np.zeros(3), axis=np.array([1., 0., 0.])):
        return knife_return_geometry(heel, axis, c, np.zeros(3), np.eye(3), .173, .025)
    assert geom(centres)["inside"]
    for shift in ([0., .004, 0.], [.3, 0., 0.], [0., 0., .1]):
        assert not geom(centres + shift)["inside"]
    bent = centres.copy()
    bent[-1, 1] = .01
    assert not geom(bent)["inside"]
    assert geom(centres, axis=np.array([-1., 0., 0.]))["axis_error"] == 180.


class CutEvaluator(TaskSetV2Evaluator):
    """Use physical fragment digests; stub only collider geometry absent in this checkout."""
    def _parts(self):
        return self.env.lab, self.env.frags, self.env.main

    def fragments(self):
        return self.env.lab, self.env.frags

    def fragment_nodes(self, frag):
        return frag["tets"]

    def _blade_touch(self, pts, margin=.002):
        return np.zeros(len(pts), bool)

    def _robot_touch(self, pts):
        return np.zeros(len(pts), bool)

    def _resting(self, frag):
        return True

    def _cut_faces_between(self, lab, a, b=None, inside=False):
        if inside:
            return self.env.inner
        return self.env.ends.get(a, np.zeros((0, 3)))


def frag(mass, tets, reg=None, skin=0., refs=None):
    return dict(mass=mass, tets=np.array(tets, int), by_reg=np.array(reg or [0.]*8),
                by_ref=np.array(refs or [0.]*8), skin_m2=skin)


def cut_evaluator(task, monkeypatch):
    # The actual legacy success predicates for B3/C1/B5 operate on these digests.
    w = NS(config=NS(task=task, mesh_h=.0035, rigid_dt=.001), fault=None,
           cut=NS(cut_faces=set(), surface=[]), assembly={"interface_faces": set()},
           workpiece=NS(region_names=["main", "chain", "melanin_spot"], reference_names=["loin"]),
           tissue=NS(n=5, kind=Array([0]*5), x=Array([[0., .28, LY.BOARD_TOP_Z]]*5)))
    w.tissue.hX = np.array([[x*.01, 0., 0.] for x in range(5)])
    w.tissue.htets = np.repeat(np.arange(5)[:, None], 4, axis=1)
    env = NS(world=w, config=w.config, sim_time=0., blade_robot_contacts=0, blade_left_hand_contacts=0,
             piece_speed_max=(0., None, None), tissue_speed_max=(0., None, None),
             frags={0: frag(.5, [0, 1, 2, 3, 4], [.45, .04, .01, 0, 0, 0, 0, 0], skin=.02)},
             lab=np.zeros(5, int), main=0, inner=np.zeros((0, 3)), ends={}, scale_log=[])
    ev = object.__new__(CutEvaluator)
    ev.env, ev.task = env, task
    ev.graph = NS(edges=lambda: [], remaining=lambda *args: env.remaining,
                  partial_slice=lambda *args: env.partial, centres=np.array([[x, 0., 0.] for x in range(5)]))
    ev.source, ev.target, ev.r0, ev.raw_start = [0], [1], 10, 0.
    ev.target_count = 3
    ev.weighed, ev.weigh_windows = set(), {}
    env.remaining, env.partial = 10, 0.
    return ev


@pytest.mark.parametrize("task", ["B3", "C1", "B5"])
def test_cut_start_and_idle_zero(task, monkeypatch):
    ev = cut_evaluator(task, monkeypatch)
    for t in (0., 5., 30.):
        ev.env.sim_time = t
        result = ev.evaluate()
        assert result["score"] == 0. and not result["success"]


@pytest.mark.parametrize("task", ["B3", "C1", "B5", "C3"])
def test_only_head_off_accepts_a_knife_put_down(task, monkeypatch):
    # The spec relaxes the knife rule for head-off only; the other cutting tasks keep their original checks.
    ev = cut_evaluator(task, monkeypatch)
    ev.env.world.cut.orig = ev.env.world.tissue.htets.copy()
    folder = LY.task_verifier(task)
    original = folder.released_checks
    legacy_checks, _ = original(ev)
    assert "knife_not_dropped" not in legacy_checks       # none of these four has a knife check of its own
    assert set(legacy_checks) <= set(ev.evaluate()["checks"])

    def with_knife_check(e):
        checks, metrics = original(e)
        checks["knife_not_dropped"] = check("knife_not_dropped", 20., 15., "mm", False)[1]
        return checks, metrics

    monkeypatch.setattr(folder, "released_checks", with_knife_check)
    monkeypatch.setattr(TaskSetV2Evaluator, "_knife_put_down", lambda self, *args: (True, "put down"))
    assert not ev.evaluate()["checks"]["knife_not_dropped"]["passed"]


@pytest.mark.parametrize("task", ["B3", "C1"])
def test_seam_or_spot_progress_is_continuous_and_success_one(task, monkeypatch):
    ev = cut_evaluator(task, monkeypatch)
    values = []
    for remaining in (10, 8, 5, 2):
        ev.env.remaining = remaining
        values.append(ev.evaluate()["score"])
    assert values[0] == 0 and all(a < b for a, b in zip(values, values[1:]))
    # Identical physical outcome regardless of fragment id, order or tool path.
    ev.env.frags = {7: frag(.45, [0, 1, 2], [.45, 0., 0., 0, 0, 0, 0, 0], skin=.02),
                    3: frag(.05, [3, 4], [0., .04, .01, 0, 0, 0, 0, 0])}
    ev.env.main = 7
    if task == "B3":
        # Spot .01 kg and .01 kg good flesh only; no skin lost.
        ev.env.frags[7] = frag(.48, [0, 1, 2], [.45, .03, 0., 0, 0, 0, 0, 0], skin=.02)
        ev.env.frags[3] = frag(.02, [3, 4], [0., .01, .01, 0, 0, 0, 0, 0])
    ev.env.remaining = 0
    result = ev.evaluate()
    assert result["success"] and result["score"] == 1.
    ev.env.blade_robot_contacts = 1
    result = ev.evaluate()
    assert not result["success"] and result["score"] <= .475


def test_spot_good_flesh_and_skin_penalties(monkeypatch):
    ev = cut_evaluator("B3", monkeypatch)
    ev.env.remaining = 5
    before = ev.evaluate()["score"]
    # Removing only distant good flesh earns nothing and reduces useful progress.
    ev.env.frags = {0: frag(.46, [0, 1, 2], [.41, .04, .01, 0, 0, 0, 0, 0], skin=.02),
                    1: frag(.04, [3, 4], [.04, 0., 0., 0, 0, 0, 0, 0])}
    assert 0 < ev.evaluate()["score"] < before
    ev.env.remaining = 10
    assert ev.evaluate()["score"] == 0.


@pytest.mark.parametrize("task", ["B3", "C1"])
def test_missing_target_is_not_progress_or_success(task, monkeypatch):
    ev = cut_evaluator(task, monkeypatch)
    ev.env.frags[0]["by_reg"] = np.array([.5, 0., 0., 0., 0., 0., 0., 0.])
    ev.env.remaining = 0
    result = ev.evaluate()
    assert not result["success"] and result["score"] == 0.


def test_portions_need_piece_specific_scale_evidence(monkeypatch):
    ev = cut_evaluator("B5", monkeypatch)
    ev.env.frags = {0: frag(.02, [0]), 1: frag(.15, [1, 2]), 2: frag(.02, [3, 4])}
    ev.env.world.cut.surface = [[[0, 0, 0], [0, 1, 0], [0, 0, 1]]]
    ev.env.scale_log = [(0, 150.), (.05, 150.), (.1, 150.)]
    result = ev.evaluate()
    assert result["score"] > 0 and not result["success"]
    assert not result["checks"]["weighed"]["passed"]
    ev.weighed.add((3, 4))  # another identically-sized fragment cannot stand for this one
    assert not ev.evaluate()["success"]
    ev.weighed.add((1, 2))
    assert ev.evaluate()["success"] and ev.evaluate()["score"] == 1.


def test_partial_cut_gets_credit_before_first_portion(monkeypatch):
    ev = cut_evaluator("B5", monkeypatch)
    ev.env.world.cut.surface = [[[0, 0, 0], [0, 1, 0], [0, 0, 1]]]
    ev.env.partial = .5
    assert 0 < ev.evaluate()["score"] < .95
    ev.env.piece_speed_max = (3., 1., 5)
    assert ev.evaluate()["score"] == 0.


def test_head_off_start_partial_and_success(geometry, monkeypatch):
    ev = cut_evaluator("A1", monkeypatch)
    env, w = ev.env, ev.env.world
    w.config.phantom_rig = False
    w.knife_in_grip = lambda: (np.zeros(3), 0.)
    w.pinch_forces = lambda: {"left": [20., 20.]}
    w.assembly["n_knife_bodies"] = 0
    w.workpiece.rigid_bodies = [{"name": "head"}, {"name": "segment_00"}]
    w.tissue.joints = {"on": Array([1])}
    w.tissue.joint_desc = [{"name": "head->segment_00"}]
    ev.bone_attachment = lambda lab: {0: {env.head: 5}, 1: {0: 5}}
    ev.head_fragment = lambda attach: env.head
    ev.frame_fragment = lambda *args: 0
    env.head = 0
    assert ev.evaluate()["score"] == 0.
    env.remaining = 5
    result = ev.evaluate()
    assert 0 < result["score"] < .95 and not result["success"]
    env.frags = {0: frag(.45, [0, 1, 2, 3]), 1: frag(.05, [4], refs=[.001, .002, 0, .047, 0, 0, 0, 0])}
    env.head, env.remaining = 1, 0
    w.tissue.joints["on"].value[:] = 0
    result = ev.evaluate()
    assert result["success"] and result["score"] == 1.
    w.tissue.joints["on"].value[:] = 1  # no bone-joint separation: not a heading success
    assert not ev.evaluate()["success"]


def test_medallions_start_partial_success_and_thickness_boundary(monkeypatch):
    ev = cut_evaluator("C3", monkeypatch)
    env, w = ev.env, ev.env.world
    ts = w.tissue
    ts.n = 12
    ts.htets = np.repeat(np.arange(12)[:, None], 4, axis=1)
    ts.hX = np.column_stack((np.r_[np.linspace(0, .02325, 6), np.linspace(.10, .30, 6)], np.zeros((12, 2))))
    ts.kind = Array([0]*12)
    ts.x = Array([[0., .28, LY.BOARD_TOP_Z]]*12)
    w.cut.orig = ts.htets.copy()
    env.frags = {0: frag(.5, np.arange(12))}
    env.lab = np.zeros(12, int)
    assert ev.evaluate()["score"] == 0.
    w.cut.surface = [[[0, 0, 0], [0, 1, 0], [0, 0, 1]]]
    env.partial = .5
    assert 0 < ev.evaluate()["score"] < .95
    env.frags = {0: frag(.04, np.arange(6)), 1: frag(.46, np.arange(6, 12))}
    env.main = 1
    env.ends[0] = np.array([[-.005, 0., 0.], [.03, 0., 0.]])
    result = ev.evaluate()
    assert result["success"] and result["score"] == 1.
    ts.hX[5, 0] = .029  # 30.75 mm > nominal + 5 mm
    result = ev.evaluate()
    assert not result["success"] and 0 < result["score"] < .95


def test_scale_monitor_requires_one_released_supported_piece(monkeypatch):
    ev = cut_evaluator("B5", monkeypatch)
    w = ev.env.world
    w.scale_reading_g = lambda: 150.
    ev.env.frags = {0: frag(.15, [0, 1, 2])}
    w.tissue.x.value[:] = [*LY.SCALE["center"], LY.TABLE["top_z"] + LY.SCALE["size"][2]]
    for _ in range(3):
        LY.task_verifier("B5")._monitor_scale(ev)
    assert ev.weighed == {(0, 1, 2)}
    ev.weighed.clear()
    w.tissue.x.value[:, 2] += .1
    for _ in range(3):
        LY.task_verifier("B5")._monitor_scale(ev)
    assert not ev.weighed
    w.tissue.x.value[:, 2] -= .1
    ev._robot_touch = lambda pts: np.ones(len(pts), bool)
    for _ in range(3):
        LY.task_verifier("B5")._monitor_scale(ev)
    assert not ev.weighed


def test_bad_fault_reports_contract_and_zero(geometry):
    ev = TaskSetV2Evaluator(knife_env(NEW_TASK), NEW_TASK)
    ev.env.world.fault = "constructed fault"
    result = ev.evaluate()
    assert result["contract"] == CONTRACT and result["score"] == 0 and not result["success"]


def test_all_graph_task_baselines_are_primed_before_first_evaluation(monkeypatch):
    import filleting.task_set_v2_evaluator as module
    raw = [.2]
    monkeypatch.setattr(module, "Connections", lambda world: object())
    monkeypatch.setattr(TaskSetV2Evaluator, "_prime_product", lambda self: None)
    monkeypatch.setattr(TaskSetV2Evaluator, "_cut_task", lambda self: ({"job": {"passed": False}}, {}, raw[0]))
    monkeypatch.setattr(TaskSetV2Evaluator, "monitor", lambda self: None)
    for task in ("A1", "B3", "C1", "B5", "C3"):
        raw[0] = .2
        env = knife_env(task)
        ev = TaskSetV2Evaluator(env, task)
        assert ev.evaluate()["score"] == 0.
        raw[0] = .6
        assert ev.evaluate()["score"] == pytest.approx(.95*.5)


def test_actual_authored_slot_geometry_and_reach_without_renderer_import():
    import ast
    import math
    from filleting import params
    tree = ast.parse((ROOT/"filleting/dressing.py").read_text())
    functions = {"_block_profile", "_inside_distance", "knife_block_fit"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions]
    constants = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if getattr(t, "id", None) in ("BLOCK_CHAMFER", "BLOCK_MARGIN"):
                    constants[t.id] = ast.literal_eval(n.value)
    namespace = dict(np=np, math=math, L=LY, PR=params, **constants)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pure-slot-geometry", "exec"), namespace)
    fit = namespace["knife_block_fit"]()
    heel = np.array([*LY.KNIFE_BLOCK["center"], LY.TABLE["top_z"]]) + np.array([LY.KNIFE_BLOCK["slot_x"][0], *fit["heels"]["fillet"]])
    R = np.column_stack((fit["xdir"], np.cross(fit["zdir"], fit["xdir"]), fit["zdir"]))
    L, H = params.v(params.FILLET_BLADE)["length"], params.v(params.FILLET_BLADE)["height"]
    centres = np.array([heel + R @ [(k + .5)*L/5, 0, H/2] for k in range(5)])
    assert knife_return_geometry(heel, R[:, 0], centres, heel, R, L, H)["inside"]
    handle = heel + R @ [-params.v(params.HANDLE)["length"]/2, 0, H]
    distance = np.linalg.norm(handle - LY.ROBOT_BASES["right"]["pos"])
    assert distance == pytest.approx(.736709, abs=1e-6)
    assert centres[:, 2].max() < LY.TABLE["top_z"] + LY.KNIFE_BLOCK["size"][2]


def test_progress_helpers_are_bounded_and_zero_relative_to_start():
    assert gained(.2, .2) == 0.
    assert gained(1., .2) == 1.
    assert gained(.5, .2) == pytest.approx(.375)
    assert ramp(None, 0., 1.) == 0.
    for v in (float("nan"), float("inf"), -1., 0., .2, 1., 20.):
        assert 0 <= ramp(v, 0., 1.) <= 1
        assert 0 <= gained(v, .2) <= 1


class EngineStub:
    def __init__(self, env):
        self.__dict__.update(vars(env))
        self.lock = threading.RLock()
        self.generation = 0
        self.trace_callback = None
        self.initial = None

    def checkpoint(self):
        return dict(time=self.sim_time, steps=self.steps, knife=(self.world.heel.copy(), self.world.R.copy()),
                    pinch=copy.deepcopy(self.world.pinch))

    def restore(self, snap):
        self.sim_time, self.steps = snap["time"], snap["steps"]
        self.world.place(*snap["knife"])
        self.world.pinch = copy.deepcopy(snap["pinch"])

    def get_state(self, evaluate=True):
        return dict(config={"task": "A0"}, instruction="carrier", terminated=False,
                    evaluation=self.evaluator.evaluate() if evaluate else {})

    def act(self, targets, *, ticks, guard=None, state=True):
        for _ in range(ticks):
            self.steps += 1
            self.sim_time = self.steps*self.config.rigid_dt
            self.trace_callback(self)
            if guard and guard(self):
                break
        return self.get_state() if state else None


def test_facade_checkpoint_reset_histories_and_time_limit(geometry):
    from filleting.task_set_v2_env import TaskSetV2Env
    base = EngineStub(knife_env(NEW_TASK))
    env = TaskSetV2Env(base, NEW_TASK)
    assert isinstance(base.evaluator, TaskSetV2Evaluator)
    assert env.get_state()["config"]["task"] == NEW_TASK
    assert env.get_state()["config"]["task_contract"] == CONTRACT
    assert env.get_state()["evaluation"]["contract"] == CONTRACT
    assert env.get_state()["evaluation"]["score"] == 0.
    env.act({}, ticks=100)
    snap = env.checkpoint()
    env.evaluator.weighed.add((1, 2))
    env.act({}, ticks=100)
    env.restore(snap)
    assert env.steps == 100 and env.evaluator.knife_history[-1][0] == .1 and not env.evaluator.weighed
    env.reset()
    assert env.steps == 0 and env.generation == 1 and env.get_state()["evaluation"]["score"] == 0.
    env.steps = 59999
    env.act({}, ticks=10)
    assert env.steps == 60000 and env.get_state()["terminated"]
    env.act({}, ticks=1)
    assert env.steps == 60000
    for bad in (True, 0, -1, 1.5):
        with pytest.raises(ValueError):
            env.act({}, ticks=bad)


@pytest.fixture
def engine(geometry, robot_catalog, monkeypatch):
    """The registered backend (newton_filleting_task_set.py over newton_filleting.py) built for real on a constructed engine: filleting.world.Config and
    filleting.env.FilletingEnv are replaced by stubs that record the configuration the backend asks for."""
    built = []

    class Config:
        def __init__(self, **kw):
            self.__dict__.update(dict(rigid_dt=.001, tissue_dt=1e-5, collision_profile="default", grip_force=20.,
                                      workpiece="trout", knife_start="block", device=None), **kw)

    def make(config, output_dir):
        built.append(config)
        engine = EngineStub(knife_env("A0"))
        engine.config = config
        engine.world.config = config
        engine.world.knife_start = config.knife_start
        engine.world.gripper_joints = lambda arm: ("finger",)
        engine.world.joints = {"finger": dict(effort_limit=70., upper=.04)}
        engine.settle_s = 1.
        engine.joints = {"fr3_joint1": dict(lower=-2.7, upper=2.7)}
        return engine

    monkeypatch.setattr(sys.modules["filleting.world"], "Config", Config, raising=False)
    env_module = ModuleType("filleting.env")
    env_module.FilletingEnv = make
    monkeypatch.setitem(sys.modules, "filleting.env", env_module)
    return built


def test_backend_wiring_uses_carrier_but_reports_new_task(engine, tmp_path):
    from gym_anything_robotics.backends.newton_filleting_task_set import NewtonFilletingTaskSetBackend as Backend
    from gym_anything_robotics.errors import UnsupportedConfigurationError
    from gym_anything_robotics.registry import _newton_filleting_factory
    from filleting.task import INSTRUCTIONS as OLD_INSTRUCTIONS, TIME_LIMIT_S
    from filleting.task_set_env import TaskSetEnv
    from filleting.task_set_v2_env import TaskSetV2Env
    # The gateway's path: the catalogue's backend_options through the registry's (unchanged) factory.
    options = task_resolve({"task": "put-away-knife", "contract": CONTRACT})["backend_options"]
    backend = _newton_filleting_factory(**options, output_dir=tmp_path/"a")
    assert isinstance(backend, Backend) and isinstance(backend.env, TaskSetV2Env)
    assert engine[-1].task == "A0" and engine[-1].knife_start == "hand"
    assert backend.policy_access().instruction() == INSTRUCTIONS[NEW_TASK]
    assert backend.policy_access().hardware["time_limit_s"] == 60.
    assert backend.evaluate()["contract"] == CONTRACT and backend.evaluate()["task"] == NEW_TASK
    assert "put-away-knife (A5)" in backend.capabilities.notes["task"]
    assert CONTRACT in backend.capabilities.notes["task"] and CONTRACT in backend.capabilities.notes["evaluator"]
    revised = Backend(task="take-knife", contract=CONTRACT, output_dir=tmp_path/"b")
    assert isinstance(revised.env, TaskSetV2Env) and revised.evaluate()["task"] == "A0"
    assert revised.evaluate()["contract"] == CONTRACT
    assert engine[-1].task == "A0" and engine[-1].knife_start == "block"
    with pytest.raises(UnsupportedConfigurationError):
        Backend(task="put-away-knife", contract=CONTRACT, robot="single-fr3", output_dir=tmp_path/"c")
    with pytest.raises(ValueError):
        Backend(task="fillet-whole-fish", contract=CONTRACT, output_dir=tmp_path/"d")
    # The released task set is untouched: bare put-away-knife and filleting-tasks-v1 use its facade and contract.
    for kwargs in (dict(task="put-away-knife"), dict(task="take-knife", contract=v1.CONTRACT)):
        released = Backend(**kwargs, output_dir=tmp_path/f"v{len(engine)}")
        assert type(released.env) is TaskSetEnv and released.evaluate()["contract"] == v1.CONTRACT
        assert released.capabilities.notes["evaluator"].endswith("(filleting/task_set_evaluator.py)")
    # Without a contract the original path runs: no facade, the original instruction, limit and evaluator.
    for kwargs in (dict(task="fillet-whole-fish", config={"collision_profile": "default"}),
                   dict(task="take-knife"), dict(task="take-knife", contract="legacy")):
        legacy_backend = Backend(**kwargs, output_dir=tmp_path/f"e{len(engine)}")
        assert "contract" not in legacy_backend.task_selection and isinstance(legacy_backend.env, EngineStub)
        code = engine[-1].task
        assert legacy_backend.policy_access().instruction() == OLD_INSTRUCTIONS[code]
        assert legacy_backend.policy_access().hardware["time_limit_s"] == TIME_LIMIT_S[code]
        assert "contract" not in legacy_backend.capabilities.notes["task"]
    assert engine[-3].collision_profile == "default"


def test_legacy_catalogue_regressions(robot_catalog):
    # Existing interface tests, without selecting any runtime-dependent classes.
    import unittest
    from tests.test_filleting import CatalogueTest
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CatalogueTest)
    result = unittest.TestResult()
    suite.run(result)
    assert result.wasSuccessful(), result.errors + result.failures
