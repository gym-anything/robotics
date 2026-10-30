"""Constructed-state tests only. No simulator, server, renderer or robot runtime.

The legacy module's geometry imports are stubbed with physical poses/arrays;
NumPy/SciPy run the real graph and evaluator arithmetic.
"""
import copy
import json
import sys
import threading
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from filleting import layout as LY, variants as legacy
from filleting.task_set_catalogue import (CONTRACT, EIGHT, INSTRUCTIONS, NEW_TASK,
                                          TASK_SET, resolve, start_options, task_resolve)
from filleting.task_set_evaluator import TaskSetEvaluator
from filleting.task_set_progress import Connections, min_connection, knife_return_geometry, gained, ramp
from tests.packages import PACKAGES

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
    entries = data["task_set"]["tasks"]
    assert [r["selection"] for r in entries] == list(TASK_SET)
    assert len({r["task_id"] for r in entries}) == 8
    for selection, row in zip(TASK_SET, entries):
        result = tasks.resolve("filleting", selection)
        assert result.task_id == row["task_id"]
        assert result.description == INSTRUCTIONS[legacy.SLUG_TASK.get(selection["task"], NEW_TASK)]
        assert tasks.TaskSpec.from_dict(result.to_dict()).to_dict() == result.to_dict()
    assert task_resolve({}) == legacy.task_resolve({})
    for code in legacy.INSTRUCTIONS:
        if code not in legacy.UNSUPPORTED:
            for seed in (0, 7, 999):
                selection = {"task": code, "seed": seed}
                assert task_resolve(selection) == legacy.task_resolve(selection)
                assert task_resolve(dict(selection, contract="legacy")) == legacy.task_resolve(selection)
    assert task_resolve(TASK_SET[-1]) == legacy.task_resolve(TASK_SET[-1])
    assert len(data["selector_choices"]["task"]) == data["count"] == 12


@pytest.mark.parametrize("seed", [True, False, -1, 1000, 1.5, "3", None])
@pytest.mark.parametrize("task", ["take-knife", "put-away-knife", "fillet-whole-fish"])
def test_seed_validation_unchanged(seed, task):
    with pytest.raises(ValueError):
        resolve(task=task, seed=seed)


@pytest.mark.parametrize("selection", [dict(task="pin-bone"), dict(task="skin"), dict(task="missing"),
    dict(task="put-away-knife", preset="salmon-side"), dict(task="take-knife", contract="missing"),
    dict(task="fillet-whole-fish", contract=CONTRACT), dict(task="put-away-knife", contract="legacy")])
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
            assert start_options(resolve(task=task), c) == (task, c)
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
              blade_robot_contacts=0, blade_left_hand_contacts=0)


def advance(ev, end, step=.05):
    while ev.env.sim_time < end - 1e-9:
        ev.env.sim_time = round(ev.env.sim_time + step, 9)
        ev.env.steps = round(ev.env.sim_time/ev.env.config.rigid_dt)
        ev.monitor()


@pytest.mark.parametrize("arm", ["left", "right"])
def test_take_knife_start_idle_partial_and_success(geometry, arm):
    env = knife_env("A0")
    ev = TaskSetEvaluator(env, "A0")
    assert ev.evaluate()["score"] == 0.
    advance(ev, 4.)
    assert ev.evaluate()["score"] == 0.
    env.world.place([-.08, .28, 1.], np.eye(3))
    env.world.pinch[arm] = [20., 20.]
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


def test_grip_slip_and_release_destroy_hold(geometry):
    env = knife_env("A0")
    ev = TaskSetEvaluator(env, "A0")
    env.world.place([-.08, .28, 1.], np.eye(3))
    env.world.pinch["right"] = [20., 20.]
    advance(ev, 4.)
    assert ev.evaluate()["success"]
    env.world.place([-.07, .28, 1.], np.eye(3))
    advance(ev, 4.1)
    assert not ev.evaluate()["success"]
    env.world.pinch["right"] = [0., 0.]
    advance(ev, 4.2)
    assert ev.evaluate()["score"] == 0.


def test_return_start_success_hold_cut_floor_and_time(geometry):
    env = knife_env(NEW_TASK)
    ev = TaskSetEvaluator(env, NEW_TASK)
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
    ev = TaskSetEvaluator(env, NEW_TASK)
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


class CutEvaluator(TaskSetEvaluator):
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
        ev._monitor_scale()
    assert ev.weighed == {(0, 1, 2)}
    ev.weighed.clear()
    w.tissue.x.value[:, 2] += .1
    for _ in range(3):
        ev._monitor_scale()
    assert not ev.weighed
    w.tissue.x.value[:, 2] -= .1
    ev._robot_touch = lambda pts: np.ones(len(pts), bool)
    for _ in range(3):
        ev._monitor_scale()
    assert not ev.weighed


def test_bad_fault_reports_contract_and_zero(geometry):
    ev = TaskSetEvaluator(knife_env(NEW_TASK), NEW_TASK)
    ev.env.world.fault = "constructed fault"
    result = ev.evaluate()
    assert result["contract"] == CONTRACT and result["score"] == 0 and not result["success"]


def test_all_graph_task_baselines_are_primed_before_first_evaluation(monkeypatch):
    import filleting.task_set_evaluator as module
    raw = [.2]
    monkeypatch.setattr(module, "Connections", lambda world: object())
    monkeypatch.setattr(TaskSetEvaluator, "_prime_product", lambda self: None)
    monkeypatch.setattr(TaskSetEvaluator, "_cut_task", lambda self: ({"job": {"passed": False}}, {}, raw[0]))
    monkeypatch.setattr(TaskSetEvaluator, "monitor", lambda self: None)
    for task in ("A1", "B3", "C1", "B5", "C3"):
        raw[0] = .2
        env = knife_env(task)
        ev = TaskSetEvaluator(env, task)
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
    from filleting.task_set_env import TaskSetEnv
    base = EngineStub(knife_env(NEW_TASK))
    env = TaskSetEnv(base, NEW_TASK)
    assert env.get_state()["config"]["task"] == NEW_TASK
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


def test_backend_wiring_uses_carrier_but_reports_new_task(geometry, robot_catalog, monkeypatch):
    from gym_anything_robotics.backends.newton_filleting import NewtonFilletingBackend
    from gym_anything_robotics.backends.newton_filleting_task_set import NewtonFilletingTaskSetBackend
    from gym_anything_robotics.types import Capabilities
    from gym_anything_robotics.policy_access import PolicyAccess
    calls = []

    def init(self, **kw):
        calls.append(kw)
        self.env = EngineStub(knife_env(NEW_TASK))
        self.config = self.env.config
        self.capabilities = Capabilities("filleting", "newton", "dual-fr3", frozenset(), frozenset(),
                                         True, False, True, (), (), notes={})

    monkeypatch.setattr(NewtonFilletingBackend, "__init__", init)
    monkeypatch.setattr(NewtonFilletingBackend, "policy_access", lambda self:
                        PolicyAccess("dual-fr3", (), {}, frozenset(), lambda: "carrier", {}))
    from gym_anything_robotics.registry import _newton_filleting_factory
    backend = _newton_filleting_factory(task="put-away-knife")
    assert isinstance(backend, NewtonFilletingTaskSetBackend)
    assert calls[-1]["task"] == "A0" and calls[-1]["config"]["knife_start"] == "hand"
    assert backend.policy_access().instruction() == INSTRUCTIONS[NEW_TASK]
    assert backend.policy_access().hardware["time_limit_s"] == 60.
    assert backend.evaluate()["contract"] == CONTRACT
    legacy_backend = _newton_filleting_factory(task="fillet-whole-fish", config={"collision_profile": "default"})
    assert isinstance(legacy_backend.env, EngineStub)  # no wrapper, no changes to A3
    assert calls[-1]["task"] == "fillet-whole-fish"
    assert calls[-1]["config"] == {"collision_profile": "default"}


def test_legacy_catalogue_regressions(robot_catalog):
    # Existing interface tests, without selecting any runtime-dependent classes.
    import unittest
    from tests.test_filleting import CatalogueTest
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CatalogueTest)
    result = unittest.TestResult()
    suite.run(result)
    assert result.wasSuccessful(), result.errors + result.failures
