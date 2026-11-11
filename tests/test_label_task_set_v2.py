"""label-tasks-v2: the render-checked eight-task set, through the public resolver and the real reward arithmetic.

No world, solver, backend, artwork or renderer is imported. The score fixture is the released task-set test's
(tests/test_label_task_set.py); these snapshots test catalogue dispatch and arithmetic, not physical solvability.
"""
import json
import unittest

from gym_anything_robotics import tasks
from label_application import task_set_v2 as V2, variants
from label_application.config import Config
from label_application.task import INSTRUCTIONS, REWARD_CONTRACTS
from tests.packages import PACKAGES, REPO
from tests.test_label_task_set import EXPECTED, apply_sheet, score_fixture

ROOT = PACKAGES
ENV = "label-application"


def bench():
    return json.loads((ROOT / "label_application/benchmark.json").read_text())


class CatalogueV2Test(unittest.TestCase):
    def test_benchmark_lists_the_eight_exact_selections(self):
        raw = bench()
        self.assertEqual(raw["task_set"]["contract"], V2.CONTRACT)
        self.assertEqual(raw["task_set"]["selections"], [dict(s) for s in V2.SELECTIONS])
        self.assertIn("contract", raw["selectors"])
        # the same eight jobs as the released set, in the same order
        self.assertEqual([(s["preset"], s["task"]) for s in raw["task_set"]["selections"]],
                         [(p, t) for p, t, _, _ in EXPECTED])
        self.assertEqual([e["selection"] for e in raw["task_set_v1"]["tasks"]], [dict(s) for s in variants.TASK_SET])

    def test_audited_task_keeps_its_released_test_split_selection(self):
        test = json.loads((REPO / "benchmarks/roboworld/splits/test.json").read_text())["tasks"]
        audited = [t["selection"] for t in test if t["environment_id"] == ENV]
        self.assertEqual(audited, [V2.SELECTIONS[-1]])
        self.assertEqual(V2.SELECTIONS[-1], dict(preset="specimen", task="tube-label"))
        self.assertNotIn("contract", V2.SELECTIONS[-1])

    def test_v2_selections_resolve_with_contract_in_id_and_the_released_launch(self):
        ids = set()
        for selection, (preset, task, _, reward) in zip(V2.SELECTIONS, EXPECTED):
            with self.subTest(task=task):
                spec = tasks.resolve(ENV, dict(selection))
                released = tasks.resolve(ENV, dict(preset=preset, task=task, seed=0))
                if task == V2.AUDITED:
                    self.assertEqual(spec.to_dict(), tasks.resolve(ENV, dict(preset=preset, task=task)).to_dict())
                    self.assertEqual(spec.task_id, f"label-application/v1/{preset}/{task}/seed-0")
                else:
                    self.assertEqual(spec.task_id, f"label-application/v1/{preset}/{task}/{V2.CONTRACT}/seed-0")
                    self.assertEqual(spec.selection, dict(preset=preset, task=task, seed=0, contract=V2.CONTRACT))
                # same instruction, same backend options (the released code path), same evaluator description
                self.assertEqual(spec.description, INSTRUCTIONS[task])
                self.assertEqual(spec.description, released.description)
                self.assertEqual(spec.launch.to_dict(), released.launch.to_dict())
                self.assertEqual(spec.launch.backend_options, dict(preset=preset, task=task, seed=0))
                self.assertEqual(spec.evaluator, released.evaluator)
                self.assertEqual(Config(**spec.launch.backend_options).task, task)
                self.assertEqual(REWARD_CONTRACTS[task], reward)
                ids.add(spec.task_id)
                # a resolved selection resolves to itself
                self.assertEqual(tasks.resolve(ENV, spec.selection).to_dict(), spec.to_dict())
        self.assertEqual(len(ids), 8)

    def test_released_resolution_is_unchanged(self):
        for entry in bench()["task_set_v1"]["tasks"]:
            r = variants.task_resolve(entry["selection"])
            self.assertEqual(r["task_id"], entry["task_id"])
            self.assertEqual(r["backend_options"], entry["selection"])
            self.assertNotIn("contract", r["selection"])
        for selection in ({}, dict(preset="specimen"), dict(preset="parcel", task="cover-old-label", seed=4)):
            self.assertNotIn(V2.CONTRACT, variants.task_resolve(selection)["task_id"])
        self.assertEqual(variants.task_resolve(dict(preset="parcel", contract=None)), variants.task_resolve({}))

    def test_contract_refusals(self):
        for selection in (dict(preset="specimen", task="tube-label", contract=V2.CONTRACT),
                          dict(preset="specimen", task="tube-wrap", contract=V2.CONTRACT),
                          dict(preset="parcel", task="cover-old-label", contract=V2.CONTRACT),
                          dict(preset="parcel", task="smooth-corner", contract="label-tasks-v1"),
                          dict(preset="parcel", task="smooth-corner", contract="v2"),
                          dict(preset="parcel", task="smooth-corner", seed=-1, contract=V2.CONTRACT),
                          dict(preset="part", task="part-id-steel", contract=V2.CONTRACT)):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                tasks.resolve(ENV, selection)
        # other seeds stay layout variations of the same task under the contract
        spec = tasks.resolve(ENV, dict(preset="parcel", task="press-down", seed=7, contract=V2.CONTRACT))
        self.assertEqual(spec.task_id, f"label-application/v1/parcel/press-down/{V2.CONTRACT}/seed-7")


class ScoreV2Test(unittest.TestCase):
    """The v2 launch builds the released evaluator for each task: untouched starts fail at 0, finished sheets score 1."""

    def test_untouched_zero_and_finished_one_for_each_v2_selection(self):
        for selection in V2.SELECTIONS:
            task = tasks.resolve(ENV, dict(selection)).launch.backend_options["task"]
            with self.subTest(task=task):
                ev, lab, reader, ref = score_fixture(task)
                r = ev.evaluate(optical={"match": True})
                self.assertEqual((r["success"], r["progress"], r["score"]), (False, 0., 0.))
                self.assertEqual(r["reward"]["contract"], REWARD_CONTRACTS[task])
                apply_sheet(lab, reader, ref)
                r = ev.evaluate(optical={"match": True})
                self.assertTrue(r["success"], r["checks"])
                self.assertEqual(r["score"], 1.)
                lab.torn = True
                r = ev.evaluate(optical={"match": True})
                self.assertEqual((r["success"], r["score"]), (False, 0.))


if __name__ == "__main__":
    unittest.main()
