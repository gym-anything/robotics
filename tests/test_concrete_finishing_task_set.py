"""Pure NumPy state fixtures and stubbed backend lifecycle: NO simulator calls.

Run this file alone. No world/env/robot construction, stepping, materials.work,
slow_step, server, renderer, Newton, Warp or MuJoCo runtime is needed.
"""

import copy
import json
from types import SimpleNamespace as NS, ModuleType
import sys
import threading

import numpy as np
import pytest

from concrete_finishing import layout as L, params as P, task_set as C, variants
from concrete_finishing.task_set_progress import Evaluator, band, falloff
from concrete_finishing.task import Evaluator as Legacy, INSTRUCTIONS
from tests.packages import PACKAGES

ROOT = PACKAGES


def env_for(task, seed=0, variant="default"):
    """Authored data arrays only. Rigid state/contacts are explicit stubs."""
    spec = L.setup(task, seed, variant=variant, **variants.site(task, seed, variant))
    poses = {n: (np.array(p), np.eye(3)) for n, (p, _) in spec["tools"].items()}
    world = NS(regions=spec["regions"], form_peak_n=0.0, fault=None, tool_names=list(poses),
               tool_pose=lambda n: poses[n], poses=poses,
               state=NS(body_q=NS(numpy=lambda: np.zeros((0, 7)))), scene=NS(robot_body={}),
               contactors=[], plate_force={})
    env = NS(config=NS(task=task, seed=seed, preset="b012", variant=variant), world=world,
             material_clock=lambda: float(next(iter(world.regions.values())).clock), lock=threading.RLock())
    env.evaluator = Legacy(env)
    return env


@pytest.fixture(scope="module")
def starts():
    return {task: env_for(task) for task in C.TASKS[:-1]}


def fresh(starts, task):
    # Locks and lambdas are deliberately not deep-copied. The regions are the
    # only mutable material state; reset/lifecycle have separate tests below.
    source = starts[task]
    env = copy.copy(source)
    env.world = copy.copy(source.world)
    env.world.regions = copy.deepcopy(source.world.regions)
    env.material_clock = lambda: float(next(iter(env.world.regions.values())).clock)
    env.evaluator = Evaluator(env)
    return env


def set_height(r, height, mask=None):
    mask = np.ones(r.b.shape, bool) if mask is None else mask
    height = np.broadcast_to(height, r.b.shape)
    r.tb[mask] = height[mask] - r.b[mask]
    r.ts[mask] = 0
    r.bl[mask] = 0


def completed(env):
    """Construct verifier-positive material states, not demonstrations/solutions."""
    t, w = env.config.task, env.world
    if t.startswith("pad"):
        r = w.regions["pad"]
        if t == "pad-darby":
            set_height(r, P.FORM[1], r.paste_mask)
            r.E[:] = 0.01
        elif t == "pad-edge":
            d = env.evaluator.pad_d
            rad = P.TOOLS["edger"]["radius"]
            arc = P.FORM[1] - (rad - np.sqrt(np.maximum(rad*rad - (rad - d)**2, 0)))
            set_height(r, arc, r.paste_mask & (d < rad))
            r.compact[r.paste_mask & (d < 0.0025)] = 1
        else:
            r.D[r.paste_mask] = 0.9
            r.a[r.paste_mask] = 4e-5
    elif t == "patch-flush":
        r = w.regions["patch"]
        set_height(r, 0)
        r.a[r.paste_mask] = 3e-4
        r.worked[r.paste_mask] = 1
        r.tau_w[r.paste_mask] = 10e3
    elif t == "tool-joints":
        r = w.regions["infill"]
        set_height(r, np.where(r.paste_mask, -0.002, 0))
        r.compact[r.paste_mask] = 0.8
        r.tau_w[r.paste_mask] = 10e3
        r.worked[r.paste_mask] = 1
    elif t == "tool-sealant":
        r = w.regions["joint"]
        set_height(r, np.where(r.paste_mask, -0.002, 0))
        r.contact_l[r.edge_l] = 1
        r.contact_r[r.edge_r] = 1
        r.worked[env.evaluator.sj_seg] = 1
        r.seal_age[:] = 600
        r.seal_t[:] = 0
    else:
        r = w.regions["wallboard"]
        set_height(r, np.where(r.mesh, P.MESH_T + 1e-5, 0))


def bad_keys(out):
    return [k for k, c in out["criteria"].items() if not c["passed"]]


def test_catalogue_exact_selections_and_legacy_defaults():
    b = json.loads((ROOT/"concrete_finishing/benchmark.json").read_text())
    rows = b["task_set"]["tasks"]
    assert [r["selection"] for r in rows] == C.selections()
    assert len(rows) == len({r["task_id"] for r in rows}) == 8
    assert C.task_resolve({}) == variants.task_resolve({})
    assert C.task_resolve({})["task_id"] == b["default_task_id"]
    assert b["count"] == 9 and "pad-float" in C.contract()["tasks"]["b012"]
    for row in rows:
        s = row["selection"]
        resolved = C.task_resolve(s)
        assert resolved["task_id"] == row["task_id"]
        assert resolved["backend_options"] == s
        assert resolved["description"] == INSTRUCTIONS[s["task"]]
        assert C.selected_contract(s["task"], s.get("contract")) == row["reward_version"]
    for task in L.TASKS:
        assert C.task_resolve({"task": task}) == variants.task_resolve({"task": task})
        assert C.task_resolve({"task": task, "contract": C.selected_contract(task)}) == variants.task_resolve({"task": task})


@pytest.mark.parametrize("bad", [True, 1.0, "1", -1, 10000])
def test_original_seed_validation(bad):
    with pytest.raises(ValueError):
        C.task_resolve({"task": "pad-darby", "seed": bad, "contract": C.CONTRACT})


@pytest.mark.parametrize("selection", [
    {"task": "pad-finish", "contract": C.CONTRACT}, {"task": "pad-float", "contract": C.CONTRACT},
    {"task": "pad-darby", "contract": "typo"}, {"task": "pad-edge", "variant": "fast-sealant"},
    {"task": "tool-joints", "variant": "fan"}])
def test_rejected_selections(selection):
    with pytest.raises(ValueError):
        C.task_resolve(selection)


@pytest.mark.parametrize("task", C.TASKS[:-1])
def test_untouched_clock_only_and_success(starts, task):
    env = fresh(starts, task)
    ev = env.evaluator
    out = ev.evaluate()
    assert not out["success"] and out["score"] == out["progress"]["value"] == 0
    for r in env.world.regions.values():
        r.clock += 7200  # Read a later clock; do NOT run slow_step or simulate waiting.
    assert ev.evaluate()["score"] == 0
    completed(env)
    out = ev.evaluate()
    assert out["success"], bad_keys(out)
    assert out["score"] == 1
    assert out["reward_version"] == C.CONTRACT


@pytest.mark.parametrize("task", C.TASKS[:-1])
def test_continuous_partial_improvement(starts, task):
    env = fresh(starts, task)
    ev = env.evaluator
    if task == "pad-darby":
        r = env.world.regions["pad"]
        change = lambda: r.E.__isub__(0.002)
    elif task == "pad-edge":
        r = env.world.regions["pad"]
        d = ev.pad_d
        rad = P.TOOLS["edger"]["radius"]
        arc = P.FORM[1] - (rad - np.sqrt(np.maximum(rad*rad - (rad - d)**2, 0)))
        sel = r.paste_mask & (d < rad)
        change = lambda: set_height(r, r.h + 0.05*(arc - r.h), sel)
    elif task == "pad-trowel":
        r = env.world.regions["pad"]
        r.a[r.paste_mask] = 1.4e-4
        change = lambda: r.D.__iadd__(0.01)
    elif task == "patch-flush":
        r = env.world.regions["patch"]
        change = lambda: set_height(r, r.h*0.98)
    elif task == "tool-joints":
        r = env.world.regions["infill"]
        change = lambda: r.compact.__iadd__(0.01)
    elif task == "tool-sealant":
        r = env.world.regions["joint"]
        change = lambda: set_height(r, r.h - 0.00005, ev.sj_seg)
    else:
        r = env.world.regions["wallboard"]
        change = lambda: set_height(r, r.h + 0.00005, r.holes)
    scores = []
    for _ in range(3):
        change()
        out = ev.evaluate()
        assert not out["success"]
        assert out["score"] == pytest.approx(0.95*out["progress"]["value"])
        scores.append(out["score"])
    assert 0 < scores[0] < scores[1] < scores[2] < 0.95, scores


@pytest.mark.parametrize("task,damage,criterion", [
    ("pad-darby", "wet_worked", "no_water_worked"),
    ("pad-darby", "S", "not_sealed"),
    ("pad-edge", "wet_worked", "edge_not_worked_wet"),
    ("pad-trowel", "q", "delamination_risk"),
    ("pad-trowel", "dwc", "surface_wc"),
    ("tool-sealant", "skin_torn", "tooled_before_skin"),
])
def test_collateral_prevents_success_and_reduces_progress(starts, task, damage, criterion):
    env = fresh(starts, task)
    completed(env)
    before = env.evaluator.evaluate()
    assert before["success"]
    r = next(iter(env.world.regions.values()))
    getattr(r, damage)[:] = 1
    out = env.evaluator.evaluate()
    assert not out["success"] and not out["criteria"][criterion]["passed"]
    assert out["score"] < 0.95*before["progress"]["value"]


@pytest.mark.parametrize("task,criterion", [("patch-flush", "no_feather_edge"), ("tool-joints", "smear"),
    ("tool-sealant", "smear"), ("patch-drywall", "no_stray_compound")])
def test_catch_deposits_penalized_and_recoverable(starts, task, criterion):
    env = fresh(starts, task)
    completed(env)
    r = next(iter(env.world.regions.values()))
    r.off_m3 = 1e-5
    dirty = env.evaluator.evaluate()
    assert not dirty["success"] and not dirty["criteria"][criterion]["passed"]
    r.off_m3 = 0
    assert env.evaluator.evaluate()["success"]


def test_patch_late_work_and_masonry_timing(starts):
    for task, key in [("patch-flush", "finished_in_time"), ("tool-joints", "tooled_thumbprint_hard")]:
        env = fresh(starts, task)
        completed(env)
        r = next(iter(env.world.regions.values()))
        r.tau_w[:] = 50e3
        out = env.evaluator.evaluate()
        assert not out["success"] and not out["criteria"][key]["passed"]
        assert out["score"] < 0.95


def test_predicates_preserved_except_documented_pad_alignment(starts):
    for task in C.TASKS[:-1]:
        env = fresh(starts, task)
        old = Legacy(env)
        for finish in (False, True):
            if finish:
                completed(env)
            a, b = old.evaluate()["criteria"], env.evaluator.evaluate()["criteria"]
            for key in a:
                if key == "tools_back" or task == "pad-trowel" and key in ("no_blister", "delamination_risk"):
                    continue
                assert a[key] == b[key], (task, key)


def test_start_discrepancies_are_work_or_keeps_from_actual_state(starts):
    patch = fresh(starts, "patch-flush").evaluator
    joint = fresh(starts, "tool-joints").evaluator
    assert not patch.start["no_feather_edge"].passed
    assert not joint.start["smear"].passed
    assert joint.start["no_void"].passed  # 3 mm short is less than the 4.3 mm void cutoff.


def test_shelves_grip_support_and_robot_success_checks(starts, monkeypatch):
    env = fresh(starts, "pad-darby")
    completed(env)
    ev = env.evaluator
    # Stub only the rigid contacts, not the shelf geometry or acceptance logic.
    poses = {n: (p.copy(), r.copy()) for n, (p, r) in env.world.poses.items()}
    env.world.tool_pose = lambda n: poses[n]
    for p, _ in poses.values():
        p[2] = 0.50  # middle shelf (surface 0.495 m)
    assert ev.evaluate()["success"]
    assert not Legacy(env).evaluate()["criteria"]["tools_back"]["passed"]
    tool = env.world.tool_names[0]
    monkeypatch.setattr(ev, "_tools_held", lambda: [tool])
    assert not ev.evaluate()["criteria"]["tools_back"]["passed"]
    monkeypatch.setattr(ev, "_tools_held", lambda: [])
    monkeypatch.setattr(ev, "_tools_unsupported", lambda: [tool])
    assert not ev.evaluate()["criteria"]["tools_back"]["passed"]
    monkeypatch.setattr(ev, "_tools_unsupported", lambda: [])
    env.world.form_peak_n = P.FORM_DISTURB_N + 1
    assert not ev.evaluate()["success"]


def test_committed_water_prevents_early_stop(starts):
    env = fresh(starts, "pad-trowel")
    completed(env)
    r = env.world.regions["pad"]
    r.clock = 0
    r.t0[:] = -60*60  # constructed younger sealed state, not a changed preset
    r.S[:] = 1
    r.q[:] = 0
    assert Legacy(env).evaluate()["criteria"]["delamination_risk"]["passed"]
    assert not env.evaluator.evaluate()["criteria"]["delamination_risk"]["passed"]


@pytest.mark.parametrize("task,variant", [("pad-darby", "fan"), ("tool-sealant", "fast-sealant"), ("patch-drywall", "cool")])
def test_seed_specific_baseline(task, variant):
    env = env_for(task, 17, variant)
    env.evaluator = Evaluator(env)
    assert env.evaluator.evaluate()["score"] == 0
    resolved = C.task_resolve(dict(task=task, seed=17, variant=variant, contract=C.CONTRACT))
    assert resolved["description"] == variants.task_resolve(dict(task=task, seed=17, variant=variant))["description"]


def test_state_only_no_progress_memory(starts):
    env = fresh(starts, "patch-drywall")
    saved = copy.deepcopy(env.world.regions)
    completed(env)
    assert env.evaluator.evaluate()["score"] == 1
    env.world.regions = saved
    assert env.evaluator.evaluate()["score"] == 0


def test_adapter_resets_preview_validation_and_pad_finish_passthrough(starts, monkeypatch):
    from gym_anything_robotics.backends.newton_concrete_finishing import NewtonConcreteFinishingBackend as Base
    from gym_anything_robotics.backends.newton_concrete_finishing_task_set import NewtonConcreteFinishingBackend as Adapter
    from gym_anything_robotics.errors import UnsupportedActionError

    def build(self, options):
        env = env_for(options.get("task") or "pad-darby", options.get("seed", 0), options.get("variant", "default"))
        initial = copy.deepcopy(env.world.regions)
        def reset():
            env.world.regions = copy.deepcopy(initial)
            return env.get_state()
        env.reset = reset
        env.get_state = lambda: {"evaluation": env.evaluator.evaluate()}
        self.env, self.config = env, env.config
    monkeypatch.setattr(Base, "__init__", lambda self, **options: build(self, options))
    monkeypatch.setattr(Base, "_rebuild", build)
    # Original pad-finish imports just this scene constant when checking robot
    # clearance; the actual scene module would import the missing rigid runtime.
    scene = ModuleType("concrete_finishing.scene")
    scene.PAD_TIP_Z = 0.055
    monkeypatch.setitem(sys.modules, scene.__name__, scene)

    b = Adapter(task="pad-darby", contract=C.CONTRACT)
    assert isinstance(b.env.evaluator, Evaluator)
    completed(b.env)
    assert b.evaluate()["score"] == 1
    b.reset()
    assert b.evaluate()["score"] == 0
    old_env = b.env
    for options in ({"seed": True}, {"seed": "3"}, {"contract": "bogus"}, {"variant": "fast-sealant"}, {"typo": 1}):
        with pytest.raises(UnsupportedActionError):
            b.reset(options=options)
        assert b.env is old_env and b.evaluate()["score"] == 0
    preview = b.select_variation({"contract": C.LEGACY})
    b.reset(**preview["reset_request"])
    assert type(b.env.evaluator) is Legacy
    assert b.evaluate()["reward_version"] == C.LEGACY
    b.reset(options={"contract": C.CONTRACT})
    assert b.evaluate()["score"] == 0
    b.reset(seed=17)
    assert b._variation()["contract"] == C.CONTRACT and b.config.seed == 17
    assert b.evaluate()["score"] == 0
    b.reset(options={"task": "pad-finish", "seed": 0})
    assert type(b.env.evaluator) is Legacy
    assert b.evaluate() == b.env.evaluator.evaluate()  # byte-for-byte dictionary semantics
    assert b.evaluate()["reward_version"] == C.PAD_FINISH
    assert b.evaluate()["score"] == 0
    b.reset(options={"task": "patch-flush", "contract": C.CONTRACT})
    assert b.evaluate()["score"] == 0
    b.reset(options={"task": "pad-float"})
    assert type(b.env.evaluator) is Legacy


def test_public_gar_resolution_with_robot_metadata_stub(monkeypatch):
    # Stub the robot metadata only, so the test does not need robots/; exercise
    # the real GAR catalogue, LaunchSpec and TaskSpec round trip.
    robots = ModuleType("robots")
    catalog = ModuleType("robots.catalog")
    catalog.default_robot = lambda _: "mobile-fr3-duo"
    catalog.environment_robots = lambda _: []
    catalog.ENVIRONMENTS = {}
    robots.catalog = catalog
    monkeypatch.setitem(sys.modules, "robots", robots)
    monkeypatch.setitem(sys.modules, "robots.catalog", catalog)
    from gym_anything_robotics.tasks import resolve, TaskSpec
    for selection in C.selections():
        task = resolve("concrete-finishing", selection)
        assert task.launch.backend_options == selection
        assert task.selection == selection
        assert TaskSpec.from_dict(task.to_dict()).to_dict() == task.to_dict()


def test_nonfinite_defect_state_refused(starts):
    env = fresh(starts, "pad-darby")
    env.world.regions["pad"].S[:] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        env.evaluator.evaluate()


def test_scalar_helpers():
    assert [falloff(x, 2, 6) for x in (0, 2, 4, 6, 8)] == [1, 1, 0.5, 0, 0]
    assert band(np.array([0, 1, 2, 3, 4]), 1, 3, 2).tolist() == [0.5, 1, 1, 1, 0.5]
