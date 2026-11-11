"""concrete-tasks-v2, the render-checked eight-task set: pure NumPy state fixtures and a stubbed backend lifecycle.

NO simulator calls: no world/env/robot construction, stepping, server, renderer, Newton, Warp or MuJoCo runtime.
The material-state fixtures are the released task-set test's (tests/test_concrete_finishing_task_set.py).
"""

import copy
import json
import os
import re
import subprocess
from types import ModuleType
import sys

import pytest

from concrete_finishing import layout as L, task_set as C1, task_set_v2 as C, variants
from concrete_finishing import task_set_progress
from concrete_finishing.task_set_v2_progress import Evaluator, install
from concrete_finishing.task import Evaluator as Legacy, INSTRUCTIONS
from tests.packages import PACKAGES, REPO
from tests.test_concrete_finishing_task_set import env_for, completed, set_height

ROOT = PACKAGES
BENCH = json.loads((ROOT/"concrete_finishing/benchmark.json").read_text())


@pytest.fixture(scope="module")
def starts():
    return {task: env_for(task) for task in C.TASKS[:-1]}


def fresh(starts, task):
    source = starts[task]
    env = copy.copy(source)
    env.world = copy.copy(source.world)
    env.world.regions = copy.deepcopy(source.world.regions)
    env.material_clock = lambda: float(next(iter(env.world.regions.values())).clock)
    install(env)
    return env


def test_benchmark_lists_the_eight_exact_selections_and_keeps_v1():
    assert C.CONTRACT == "concrete-tasks-v2"
    v2 = BENCH["task_set"]
    assert v2["selections"] == C.selections() and v2["note"]
    assert [s["task"] for s in v2["selections"]] == list(C.TASKS)
    # the audited hardest task keeps its released test-split selection, exactly
    split = json.loads((REPO/"benchmarks/roboworld/splits/test.json").read_text())["tasks"]
    assert v2["selections"][-1] in [t["selection"] for t in split if t["environment_id"] == "concrete-finishing"]
    # the released set is kept under task_set_v1 and still resolves as released
    for row in BENCH["task_set_v1"]["tasks"]:
        assert C1.task_resolve(row["selection"])["task_id"] == row["task_id"]
        assert C.task_resolve(row["selection"]) == C1.task_resolve(row["selection"])


def test_v2_selections_resolve_with_their_own_ids():
    released = {C1.task_resolve(r["selection"])["task_id"] for r in BENCH["task_set_v1"]["tasks"]}
    legacy = {variants.task_resolve({"task": t})["task_id"] for t in L.TASKS}
    ids = []
    for s in C.selections():
        r = C1.task_resolve(s)                      # the resolver benchmark.json names dispatches the contract
        assert r == C.task_resolve(s)
        assert r["description"] == INSTRUCTIONS[s["task"]]
        if s["task"] == "pad-finish":
            assert r == variants.task_resolve(s) and r["task_id"] == "concrete-finishing/v1/b012/pad-finish/seed-0"
        else:
            assert r["task_id"] == f"concrete-finishing/v1/b012/{s['task']}/{C.CONTRACT}/seed-0"
            assert r["task_id"] not in released | legacy
            assert r["selection"] == r["backend_options"] == s
            assert r["split"] == variants.task_resolve(s)["split"]
        ids.append(r["task_id"])
    assert len(set(ids)) == 8
    s = dict(task="pad-darby", seed=17, variant="fan", contract=C.CONTRACT)
    assert C1.task_resolve(s)["task_id"] == f"concrete-finishing/v1/b012/pad-darby/{C.CONTRACT}/seed-17/fan"


@pytest.mark.parametrize("selection", [
    {"task": "pad-finish", "contract": "concrete-tasks-v2"}, {"task": "pad-float", "contract": "concrete-tasks-v2"},
    {"task": "pad-darby", "seed": True, "contract": "concrete-tasks-v2"},
    {"task": "pad-edge", "variant": "fast-sealant", "contract": "concrete-tasks-v2"},
    {"task": "pad-darby", "contract": "concrete-tasks-v2/progress-v1"}])
def test_rejected_v2_selections(selection):
    with pytest.raises(ValueError):
        C1.task_resolve(selection)


def test_contract_choice_is_validated_like_the_released_one():
    for task in L.TASKS:
        for contract in (None, C1.LEGACY, C1.PAD_FINISH, C1.CONTRACT, "typo"):
            try:
                want = C1.selected_contract(task, contract)
            except ValueError as exc:
                with pytest.raises(ValueError, match=re.escape(str(exc).split(";")[0])):
                    C.selected_contract(task, contract)
            else:
                assert C.selected_contract(task, contract) == want
    assert [C.selected_contract(t, C.CONTRACT) for t in C.TASKS[:-1]] == [C.CONTRACT]*7


@pytest.mark.parametrize("task", C.TASKS[:-1])
def test_untouched_start_scores_zero_and_completed_scores_one(starts, task):
    env = fresh(starts, task)
    ev = env.evaluator
    out = ev.evaluate()
    assert not out["success"] and out["score"] == out["progress"]["value"] == 0
    assert out["reward_version"] == C.CONTRACT
    for r in env.world.regions.values():
        r.clock += 7200
    assert ev.evaluate()["score"] == 0
    completed(env)
    out = ev.evaluate()
    assert out["success"] and out["score"] == 1 and out["reward_version"] == C.CONTRACT


@pytest.mark.parametrize("task", C.TASKS[:-1])
def test_same_verdict_as_the_released_evaluator_but_its_own_contract(starts, task):
    env = fresh(starts, task)
    released = task_set_progress.Evaluator(env)
    for finish in (False, True):
        if finish:
            completed(env)
        a, b = released.evaluate(), env.evaluator.evaluate()
        assert a["reward_version"] == C1.CONTRACT and b["reward_version"] == C.CONTRACT
        assert list(a) == list(b) and {**a, "reward_version": None} == {**b, "reward_version": None}


def test_partial_progress_is_continuous_and_0_95_times_progress(starts):
    env = fresh(starts, "patch-flush")
    r = env.world.regions["patch"]
    scores = []
    for _ in range(3):
        set_height(r, r.h*0.98)
        out = env.evaluator.evaluate()
        assert not out["success"] and out["score"] == pytest.approx(0.95*out["progress"]["value"])
        scores.append(out["score"])
    assert 0 < scores[0] < scores[1] < scores[2] < 0.95


class CachingEnv:
    """The classroom env's evaluation cache (one evaluation per tick and generation), around an env_for stub."""

    def __init__(self, env):
        self.__dict__.update(vars(env))
        self.tick, self.generation = 0, 0

    def evaluation(self):
        key = (self.tick, self.generation)
        if getattr(self, "_eval_key", None) != key:
            self._eval_cache, self._eval_key = self.evaluator.evaluate(), key
        return self._eval_cache

    def get_state(self):
        return {"evaluation": self.evaluation()}


def test_install_drops_the_cached_evaluation():
    env = CachingEnv(env_for("pad-darby"))
    assert "reward_version" not in env.get_state()["evaluation"]          # the env's own evaluator, cached
    install(env)
    assert env.get_state()["evaluation"]["reward_version"] == C.CONTRACT


def test_backend_wiring_resets_preview_validation_and_pad_finish_passthrough(monkeypatch):
    # The released adapter (registry: newton_concrete_finishing_task_set) serves the contract; its parent's constructor
    # needs the rigid runtime, so the world build is stubbed with a caching state like the classroom env's.
    from gym_anything_robotics.backends.newton_concrete_finishing import NewtonConcreteFinishingBackend as Base
    from gym_anything_robotics.backends.newton_concrete_finishing_task_set import NewtonConcreteFinishingBackend as Adapter
    from gym_anything_robotics.errors import UnsupportedActionError, UnsupportedConfigurationError

    def build(self, options):
        env = CachingEnv(env_for(options.get("task") or "pad-darby", options.get("seed", 0),
                                 options.get("variant", "default")))
        initial = copy.deepcopy(env.world.regions)

        def reset():
            env.world.regions = copy.deepcopy(initial)
            env.tick += 1
            return env.get_state()
        env.reset = reset
        env.get_state()                                                      # the build reads its state once
        self.env, self.config = env, env.config
    monkeypatch.setattr(Base, "__init__", lambda self, **options: build(self, options))
    monkeypatch.setattr(Base, "_rebuild", build)
    scene = ModuleType("concrete_finishing.scene")
    scene.PAD_TIP_Z = 0.055
    monkeypatch.setitem(sys.modules, scene.__name__, scene)

    with pytest.raises(UnsupportedConfigurationError):
        Adapter(task="pad-finish", contract=C.CONTRACT)
    b = Adapter(task="pad-darby", contract=C.CONTRACT)
    assert isinstance(b.env.evaluator, Evaluator)
    assert b.env.get_state()["evaluation"]["reward_version"] == C.CONTRACT  # the first state read is v2's
    assert b._variation()["contract"] == C.CONTRACT
    vc = b.variation_contract()
    assert vc["task_set"] == C.selections() and C.CONTRACT in vc["contract_choices"]["pad-darby"]
    assert C.CONTRACT not in vc["contract_choices"]["pad-finish"]
    completed(b.env)
    assert b.evaluate()["score"] == 1 and b.evaluate()["reward_version"] == C.CONTRACT
    b.reset()
    assert b.evaluate()["score"] == 0 and isinstance(b.env.evaluator, Evaluator)
    old_env = b.env
    for options in ({"seed": True}, {"contract": "bogus"}, {"variant": "fast-sealant"}, {"typo": 1}):
        with pytest.raises(UnsupportedActionError):
            b.reset(options=options)
        assert b.env is old_env and b.evaluate()["score"] == 0
    # a seed switch keeps the contract and rebuilds
    b.reset(seed=17)
    assert b._variation()["contract"] == C.CONTRACT and b.config.seed == 17 and isinstance(b.env.evaluator, Evaluator)
    assert b.evaluate()["score"] == 0
    # a switch back to the released contract on the same task, and forward again, without a rebuild
    preview = b.select_variation({"contract": C1.CONTRACT})
    assert preview["reset_request"]["options"]["contract"] == C1.CONTRACT
    b.reset(**preview["reset_request"])
    assert type(b.env.evaluator) is task_set_progress.Evaluator and b.evaluate()["reward_version"] == C1.CONTRACT
    assert b.variation_contract()["task_set"] == C1.selections()
    b.reset(options={"contract": C.CONTRACT})
    assert isinstance(b.env.evaluator, Evaluator) and b.env.get_state()["evaluation"]["reward_version"] == C.CONTRACT
    preview = b.select_variation({"seed": 3})
    assert preview["variation"]["contract"] == C.CONTRACT
    preview = b.select_variation({"contract": C1.LEGACY})
    b.reset(**preview["reset_request"])
    assert type(b.env.evaluator) is Legacy
    b.reset(options={"contract": C.CONTRACT})
    b.reset(options={"task": "pad-finish", "seed": 0})
    assert type(b.env.evaluator) is Legacy and "contract" not in b._variation()
    assert b.evaluate()["reward_version"] == C1.PAD_FINISH and b.evaluate()["score"] == 0
    b.reset(options={"task": "patch-flush", "contract": C.CONTRACT})
    assert isinstance(b.env.evaluator, Evaluator) and b.evaluate()["score"] == 0


def test_released_adapter_session_is_unchanged_for_released_selections(monkeypatch):
    from gym_anything_robotics.backends.newton_concrete_finishing import NewtonConcreteFinishingBackend as Base
    from gym_anything_robotics.backends.newton_concrete_finishing_task_set import NewtonConcreteFinishingBackend as Adapter

    def build(self, options):
        env = env_for(options.get("task") or "pad-darby", options.get("seed", 0), options.get("variant", "default"))
        env.reset = lambda: env.get_state()
        env.get_state = lambda: {"evaluation": env.evaluator.evaluate()}
        self.env, self.config = env, env.config
    monkeypatch.setattr(Base, "__init__", lambda self, **options: build(self, options))
    monkeypatch.setattr(Base, "_rebuild", build)
    b = Adapter(task="pad-darby", contract=C1.CONTRACT)
    assert type(b.env.evaluator) is task_set_progress.Evaluator
    assert b.variation_contract() == dict(C1.contract(), current=b._variation())
    assert b.evaluate()["reward_version"] == C1.CONTRACT
    legacy = Adapter(task="pad-darby")
    assert type(legacy.env.evaluator) is Legacy and "contract" not in legacy._variation()
    assert legacy.evaluate()["reward_version"] == C1.LEGACY


def test_public_gar_resolution_with_robot_metadata_stub(monkeypatch):
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
        assert task.selection == C.task_resolve(selection)["selection"]
        assert TaskSpec.from_dict(task.to_dict()).to_dict() == task.to_dict()


def test_v2_resolution_does_not_load_simulator():
    # A fresh interpreter: other test modules collected in the same run may already have imported warp.
    code = """import sys
from types import ModuleType
robots, catalog = ModuleType("robots"), ModuleType("robots.catalog")
catalog.default_robot = lambda _: "mobile-fr3-duo"
catalog.environment_robots = lambda _: []
catalog.ENVIRONMENTS = {}
robots.catalog = catalog
sys.modules.update({"robots": robots, "robots.catalog": catalog})
from gym_anything_robotics.tasks import resolve
from concrete_finishing import task_set_v2 as C
for selection in C.selections():
    resolve("concrete-finishing", selection)
loaded = {"newton", "warp", "mujoco", "concrete_finishing.world", "concrete_finishing.env"} & set(sys.modules)
assert not loaded, loaded
"""
    paths = [str(REPO / "src"), str(REPO / "benchmarks/roboworld/environments"), str(REPO)]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")]))
    subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, check=True)
