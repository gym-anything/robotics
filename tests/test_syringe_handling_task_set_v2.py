"""syringe-tasks-v2, the render-checked eight-task set. No simulator: real catalogue/evaluator/ledger with the stubs of
the released task-set test (tests/environments/syringe_handling/test_task_set.py), whose reward tests also run here
under this contract.
"""
import json
import sys
from types import ModuleType
from unittest import mock

import pytest

from syringe_handling import task as T, task_set_v2 as V2, variants as V
from syringe_handling.config import Config, EIGHT_TASK_CONTRACT, LEGACY_CONTRACT, VIAL_TO_PAD_CONTRACT
from syringe_handling.scene import task_layout
from tests.environments.syringe_handling import test_task_set as R
from tests.packages import PACKAGES, REPO

CONTRACT = V2.CONTRACT
BENCH = json.loads((PACKAGES/"syringe_handling/benchmark.json").read_text())
SELECTIONS = V2.selections()
SEVEN = SELECTIONS[:-1]


def test_benchmark_lists_the_eight_and_keeps_the_released_set():
    assert CONTRACT == "syringe-tasks-v2" and EIGHT_TASK_CONTRACT == "syringe-tasks-v1"
    assert BENCH["task_set"] == dict(selections=SELECTIONS, note=V2.NOTE)
    assert [s["task"] for s in SELECTIONS] == list(V.EIGHT_TASKS)
    assert all(s["contract"] == CONTRACT for s in SEVEN)
    split = json.loads((REPO/"benchmarks/roboworld/splits/test.json").read_text())["tasks"]
    assert [t["selection"] for t in split if t["environment_id"] == "syringe-handling"] == [SELECTIONS[-1]]
    assert BENCH["task_set_v1"]["selections"] == V.eight_task_selections() == V.contract()["task_set"]
    assert BENCH["selector_choices"]["contract"] == [LEGACY_CONTRACT, EIGHT_TASK_CONTRACT, CONTRACT]
    assert {k: BENCH[k] for k in V2.catalogue_fields()} == V2.catalogue_fields()    # variants.main rewrites nothing


def test_rewriting_the_catalogue_fields_changes_nothing(capsys):
    """python -m syringe_handling.variants (variants.main) writes the file as it is, both sets included."""
    import pathlib
    written = {}
    with mock.patch.object(pathlib.Path, "write_text", lambda self, text, *a, **k: written.update({self: text})):
        V.main()
    assert written == {(PACKAGES/"syringe_handling/benchmark.json").resolve():
                       (PACKAGES/"syringe_handling/benchmark.json").read_text()}


@pytest.mark.parametrize("selection", SEVEN, ids=lambda s: s["task"])
def test_the_seven_resolve_with_their_own_ids_and_v1_tasks(selection):
    r = V.task_resolve(selection)
    v1 = V.task_resolve(dict(selection, contract=EIGHT_TASK_CONTRACT))
    assert r["task_id"] == f"syringe-handling/v1/{selection['preset']}/{selection['task']}/contract-{CONTRACT}/seed-0"
    assert r["task_id"] not in (v1["task_id"], V.task_resolve({"task": selection["task"]})["task_id"])
    assert r["selection"] == selection
    assert r["backend_options"] == dict(v1["backend_options"], contract=CONTRACT)
    assert r["description"] == v1["description"] and r["split"] == v1["split"]
    assert V.task_resolve(r["selection"]) == r
    cfg = Config(**r["backend_options"])
    assert cfg.contract == CONTRACT
    assert task_layout(cfg) == task_layout(Config(**v1["backend_options"]))
    spec, spec1 = T.task_spec(cfg, task_layout(cfg)), T.task_spec(Config(**v1["backend_options"]), task_layout(cfg))
    assert spec.pop("contract") == CONTRACT and spec1.pop("contract") == EIGHT_TASK_CONTRACT and spec == spec1
    assert "measured entering" not in r["description"] and "Scoring" not in r["description"]


def test_hardest_task_resolves_as_released():
    r = V.task_resolve(SELECTIONS[-1])
    assert r["task_id"] == "syringe-handling/v1/injection/vial-to-pad/seed-0"
    assert r["backend_options"] == dict(preset="injection", task="vial-to-pad", seed=0)
    assert Config(**r["backend_options"]).contract == LEGACY_CONTRACT
    ids = [V.task_resolve(s)["task_id"] for s in SELECTIONS]
    assert len(set(ids)) == 8 and BENCH["default_task_id"] not in ids
    assert V.task_resolve(dict(task="slow-push", seed=123, variant="short", contract=CONTRACT))["task_id"] == \
        f"syringe-handling/v1/iv-lock/slow-push/variant-short/contract-{CONTRACT}/seed-123"


@pytest.mark.parametrize("selection", [
    {"task": "vial-to-pad", "contract": CONTRACT}, {"contract": "syringe-tasks-v3"},
    {"task": "inject", "contract": CONTRACT, "typo": 1}, {"task": "inject", "variant": "blocked", "contract": CONTRACT},
    {"task": "inject", "seed": True, "contract": CONTRACT}])
def test_invalid_selection_rejected(selection):
    with pytest.raises(ValueError):
        V.task_resolve(selection)


def test_config_and_contract_choices():
    with pytest.raises(ValueError):
        Config(task="vial-to-pad", contract=CONTRACT)
    assert Config(task="inject").contract == LEGACY_CONTRACT
    c = V2.contract()
    assert c["task_set"] == SELECTIONS and list(c["contracts"]) == [LEGACY_CONTRACT, EIGHT_TASK_CONTRACT, CONTRACT]
    released = V.contract()
    assert {k: v for k, v in c.items() if k not in ("contracts", "task_set")} == \
        {k: v for k, v in released.items() if k not in ("contracts", "task_set")}


class TestRewardsUnderV2(R.Rewards):
    """The released task-set reward tests, with this contract where they select the task-set contract."""
    def setUp(self):
        patcher = mock.patch.object(R, "NEW", CONTRACT)
        patcher.start()
        self.addCleanup(patcher.stop)


@pytest.mark.parametrize("task,variant", [(t, v) for t in V.EIGHT_TASKS[:-1] for v in ("", *V.VARIANTS)
                                          if v and t in V.variant_tasks(v) or not v])
def test_same_verdict_and_instruction_as_v1_but_its_own_contract(task, variant):
    for seed in (0, 7):
        a, b = R.fake(task, variant, seed, EIGHT_TASK_CONTRACT), R.fake(task, variant, seed, CONTRACT)
        assert T.instruction(a.config) == T.instruction(b.config)
        for step in range(3):                                   # at the start, finished, then a protective stop
            if step == 1:
                R.finish(a), R.finish(b)
            if step == 2:
                a.world.reflexes = b.world.reflexes = ["physical reflex"]
            ea, eb = T.evaluate(a), T.evaluate(b)
            assert ea.pop("contract") == EIGHT_TASK_CONTRACT and eb.pop("contract") == CONTRACT
            assert ea == eb
            if not variant:
                assert eb["success"] == (eb["score"] == 1.) == (step == 1)


def test_vial_to_pad_and_legacy_untouched():
    env = R.fake("vial-to-pad")
    R.finish(env)
    e = T.evaluate(env)
    assert e["contract"] == VIAL_TO_PAD_CONTRACT and e["score"] == 1.
    e = T.evaluate(R.fake("set-dose", contract=LEGACY_CONTRACT))
    assert e["contract"] == LEGACY_CONTRACT and "progress" not in e


def _backend(config):
    from gym_anything_robotics.backends.newton_syringe_handling import NewtonSyringeHandlingBackend as Backend
    b = Backend.__new__(Backend)
    b.config = config
    return b


def test_backend_reset_select_and_variation_contract_under_v2():
    b = _backend(Config(task="inject", contract=CONTRACT))
    assert b._reset_selection(5, {})["contract"] == CONTRACT                    # a seed alone keeps the contract
    assert b._reset_selection(None, {"variant": "subcut"})["contract"] == CONTRACT
    assert b._reset_selection(None, {"contract": EIGHT_TASK_CONTRACT})["contract"] == EIGHT_TASK_CONTRACT
    assert b._reset_selection(None, {"task": "set-dose"})["contract"] == LEGACY_CONTRACT   # a new task: v1
    with pytest.raises(ValueError):
        b._reset_selection(None, {"task": "vial-to-pad", "contract": CONTRACT})
    row = b.select_variation({"task": "flush-lock", "contract": CONTRACT})
    assert row["reset_request"]["options"]["contract"] == CONTRACT
    assert row["task"] == V.task_resolve({"task": "flush-lock", "contract": CONTRACT})["description"]
    vc = b.variation_contract()
    assert vc["current"]["contract"] == CONTRACT and vc["task_set"] == SELECTIONS and CONTRACT in vc["contracts"]
    released = _backend(Config(task="inject", contract=EIGHT_TASK_CONTRACT)).variation_contract()
    assert released["task_set"] == V.eight_task_selections() and CONTRACT not in released["contracts"]
    assert {k: v for k, v in vc.items() if k not in ("contracts", "task_set", "current")} == \
        {k: v for k, v in released.items() if k not in ("contracts", "task_set", "current")}


class _Built(Exception):
    pass


def test_launch_builds_the_v2_config(tmp_path, monkeypatch):
    """A launch up to the world: the selection's backend options reach SyringeEnv as a v2 config (stubbed, so no world
    is built)."""
    from gym_anything_robotics.backends.newton_syringe_handling import NewtonSyringeHandlingBackend as Backend
    from gym_anything_robotics.errors import UnsupportedConfigurationError
    built = []

    def make(config, output_dir=None):
        built.append(config)
        raise _Built
    env_stub, kin_stub = ModuleType("syringe_handling.env"), ModuleType("syringe_handling.kinematics")
    env_stub.SyringeEnv, kin_stub.ArmKinematics = make, object
    monkeypatch.setitem(sys.modules, "syringe_handling.env", env_stub)
    monkeypatch.setitem(sys.modules, "syringe_handling.kinematics", kin_stub)
    for i, s in enumerate(SELECTIONS):
        with pytest.raises(_Built):
            Backend(**V.task_resolve(s)["backend_options"], output_dir=tmp_path/str(i), recording=False)
        assert (built[-1].task, built[-1].contract) == (s["task"], s.get("contract", LEGACY_CONTRACT))
    with pytest.raises(UnsupportedConfigurationError):
        Backend(task="vial-to-pad", contract=CONTRACT, output_dir=tmp_path/"x", recording=False)
    assert len(built) == 8
