"""whiteboard-tasks-v2: catalogue, verifier and backend wiring on constructed states and a stubbed world.

Never a simulator or policy solution. The fixtures are the released task-set tests' (test_catalogue.py, test_adapter.py).
"""
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

from gym_anything_robotics import tasks
from gym_anything_robotics.backends.newton_whiteboard_task_set import NewtonWhiteboardTaskSetBackend
from whiteboard import task_set_v2 as V2
from whiteboard.task_set import catalogue as V1
from whiteboard.task_set.catalogue import CatalogueTask, INSTRUCTIONS, NAMES, raster
from whiteboard.task_set.presets import seed_board
from whiteboard.task_set.verifier import evaluate as evaluate_v1
from whiteboard.tasks import Task
from tests.environments.whiteboard.test_adapter import FakeEnv, fake_env_module, robot_modules
from tests.environments.whiteboard.test_catalogue import paint, state, surface
from tests.packages import PACKAGES, REPO

ROOT = PACKAGES
NAMED = NAMES[:-1]


def judge(task, ink, mechanical=None):
    return V2.evaluate(task, ink, V2.preset_config(task.name)[0], mechanical or state(task))


class CatalogueV2Tests(unittest.TestCase):
    def test_benchmark_lists_the_eight_exact_selections(self):
        benchmark = json.loads((ROOT / "whiteboard/benchmark.json").read_text())
        self.assertEqual(benchmark["task_set"]["contract"], V2.CONTRACT)
        self.assertEqual(benchmark["task_set"]["selections"], list(V2.SELECTIONS))
        self.assertEqual(benchmark["task_set_v1"]["selections"], list(V1.SELECTIONS))
        self.assertIn("contract", benchmark["selectors"])
        test = json.loads((REPO / "benchmarks/roboworld/splits/test.json").read_text())["tasks"]
        self.assertEqual([t["selection"] for t in test if t["environment_id"] == "whiteboard"], [V2.SELECTIONS[-1]])

    def test_contract_name_task_ids_options_and_instructions(self):
        self.assertEqual(V2.CONTRACT, "whiteboard-tasks-v2")
        ids = set()
        for selection in V2.SELECTIONS:
            spec = tasks.resolve("whiteboard", dict(selection))
            ids.add(spec.task_id)
            if "variation_id" in selection:
                self.assertEqual(spec.to_dict(), tasks.resolve("whiteboard", {"variation_id": 0}).to_dict())
                continue
            name = selection["task"]
            released = tasks.resolve("whiteboard", {"task": name, "contract": V1.CONTRACT})
            self.assertEqual(spec.task_id, f"whiteboard/{V2.CONTRACT}/{name}")
            self.assertEqual(spec.launch.backend_options, {"catalogue_task": name, "task_contract": V2.CONTRACT})
            self.assertEqual(spec.description, INSTRUCTIONS[name])
            self.assertEqual(spec.description, released.description)
            self.assertEqual(spec.evaluator, released.evaluator)
            self.assertEqual({k: v for k, v in spec.selection.items() if k != "contract"},
                             {k: v for k, v in released.selection.items() if k != "contract"})
        self.assertEqual(len(ids), 8)

    def test_refusals_and_released_resolution(self):
        for selection in ({"task": "bridge", "contract": V2.CONTRACT}, {"task": "bad", "contract": V2.CONTRACT},
                          {"contract": V2.CONTRACT}, {"task": "write-text", "contract": V2.CONTRACT, "variation_id": 1},
                          {"task": "write-text", "contract": "whiteboard-tasks8"}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                tasks.resolve("whiteboard", selection)
        for selection in V1.SELECTIONS:
            self.assertEqual(V1.resolve(selection)["task_id"],
                             f"whiteboard/{V1.CONTRACT}/{selection['task']}" if "task" in selection
                             else V1.resolve({"variation_id": 0})["task_id"])
        with self.assertRaises(ValueError):
            V2.TaskV2("erase-board", contract=V1.CONTRACT)
        self.assertEqual(V2.TaskV2("draw-table").strokes(), CatalogueTask("draw-table").strokes())
        self.assertEqual(V2.preset_config("recap-and-tidy")[0], V1.preset_config("recap-and-tidy")[0])

    def test_source_identity_covers_v2_and_leaves_v1_set(self):
        v1, v2 = V1.source_identity(), V2.source_identity()
        self.assertFalse(any("task_set_v2" in name for name in v1["files"]))
        self.assertEqual(set(v2["files"]) - set(v1["files"]),
                         {"whiteboard/task_set_v2.py", "whiteboard/task_set_v2_backend.py"})
        for name, digest in v1["files"].items():
            self.assertEqual(v2["files"][name], digest)


class PigmentV2Tests(unittest.TestCase):
    def test_all_seven_untouched_starts_zero_with_the_v2_contract(self):
        for name in NAMED:
            with self.subTest(name=name):
                task, ink = V2.TaskV2(name), surface()
                seed_board(ink, task)
                out = judge(task, ink)
                self.assertEqual((out["success"], out["score"], out["progress"]), (False, 0., 0.))
                self.assertEqual(out["contract"], V2.CONTRACT)              # draw-table too: unsuffixed
                self.assertEqual(out["task_id"], f"whiteboard/{V2.CONTRACT}/{name}")

    def test_all_seven_constructed_end_states_pass(self):
        for name in NAMED:
            with self.subTest(name=name):
                task = V2.TaskV2(name)
                out = judge(task, surface(task), state(task, capped=True))
                self.assertEqual((out["success"], out["score"], out["progress"]), (True, 1., 1.), out)

    def test_v2_judges_exactly_as_v1_apart_from_the_contract_labels(self):
        labels = {"task_id", "contract", "revision", "metric"}
        for name in NAMED:
            with self.subTest(name=name):
                v1_task, v2_task = CatalogueTask(name), V2.TaskV2(name)
                untouched = surface()
                seed_board(untouched, v1_task)
                partial = surface()
                paint(partial, raster(v1_task.strokes()[:2], partial.board, partial.width, partial.height))
                for ink, mechanical in ((untouched, state(v1_task)), (partial, state(v1_task)),
                                        (surface(v1_task), state(v1_task, capped=True))):
                    old = evaluate_v1(v1_task, ink, V1.preset_config(name)[0], mechanical)
                    new = V2.evaluate(v2_task, ink, V2.preset_config(name)[0], mechanical)
                    self.assertEqual({k: v for k, v in new.items() if k not in labels},
                                     {k: v for k, v in old.items() if k not in labels})
                    self.assertEqual(new["contract"], V2.CONTRACT)
                    self.assertEqual(old["contract"], V1.CONTRACT + ("/draw-table" if name == "draw-table" else ""))
                    if name == "draw-table":
                        self.assertEqual(new["revision"], V2.CONTRACT + "/draw-table")
                        self.assertTrue(new["metric"].startswith(V2.CONTRACT + "/draw-table: "))
                        self.assertEqual(new["metric"].split(": ", 1)[1], old["metric"].split(": ", 1)[1])
                    else:
                        self.assertEqual(new["revision"], V2.CONTRACT)

    def test_table_partial_copy_is_graded(self):
        task, ink = V2.TaskV2("draw-table"), surface()
        strokes = task.strokes()
        scores = []
        for count in (0, 2, 4, 6):
            paint(ink, raster(strokes[:count], ink.board, ink.width, ink.height))
            out = judge(task, ink)
            self.assertFalse(out["success"])
            scores.append(out["progress"])
        self.assertTrue(all(a < b for a, b in zip(scores, scores[1:])), scores)


class BackendV2Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.module_patch = patch.dict(sys.modules, {"whiteboard.env": fake_env_module(), **robot_modules()})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def backend(self, name="erase-board", contract=V2.CONTRACT, **kwargs):
        return NewtonWhiteboardTaskSetBackend(catalogue_task=name, task_contract=contract, recording=False,
                    cameras=(), output_dir=self.directory.name, texture_width=1024, texture_height=512, **kwargs)

    def test_the_adapter_builds_the_v2_backend_only_for_v2(self):
        from whiteboard.task_set_v2_backend import NewtonWhiteboardTaskSetV2Backend
        v2, v1 = self.backend(), self.backend(contract=V1.CONTRACT)
        self.assertIs(type(v2), NewtonWhiteboardTaskSetV2Backend)
        self.assertIs(type(v1), NewtonWhiteboardTaskSetBackend)
        self.assertIsInstance(v2.env.task, V2.TaskV2)
        self.assertEqual(v2.task_source, V2.source_identity())
        self.assertEqual(v1.task_source, V1.source_identity())
        result = v2.evaluate()
        self.assertEqual((result["score"], result["contract"]), (0., V2.CONTRACT))
        self.assertEqual(result["task_source_sha256"], V2.source_identity()["sha256"])
        self.assertEqual(v1.evaluate()["contract"], V1.CONTRACT)
        identity = [e for e in v2.env.trace.events if e["kind"] == "task_set_identity"]
        self.assertEqual([e["contract"] for e in identity], [V2.CONTRACT])
        for backend in (v1, v2):
            backend.close()
        with self.assertRaises(ValueError):
            NewtonWhiteboardTaskSetV2Backend(catalogue_task="erase-board", task_contract=V1.CONTRACT, recording=False,
                                             cameras=(), output_dir=self.directory.name)

    def test_constructor_reset_and_seed_reset_keep_authored_starts(self):
        for name in ("erase-board", "erase-except-boxed", "recap-and-tidy", "correct-misspelling"):
            backend = self.backend(name)
            before = backend.env.snapshot()
            self.assertEqual(backend.evaluate()["score"], 0)
            backend.env.ink.tiles.clear()
            backend.env.tick = 7
            backend.reset()
            self.assertEqual(backend.evaluate()["score"], 0)
            self.assertEqual(backend.env.tick, 0)
            self.assertEqual(backend.env.state_0.body_q.numpy().tolist(), before["arrays"]["body_q"].tolist())
            old = backend.env
            backend.reset(seed=12)
            self.assertTrue(old.closed)
            self.assertEqual(backend.config.seed, 12)
            self.assertIsInstance(backend.env.task, V2.TaskV2)
            self.assertEqual(backend.evaluate()["score"], 0)
            backend.close()

    def test_invalid_options_fail_before_constructing(self):
        before = FakeEnv.builds
        for extra in ({"task": Task()}, {"variation_id": 0}, {"robot": "bad"}, {"name": "bridge"}, {"name": "bad"}):
            with self.assertRaises((ValueError, RuntimeError)):
                self.backend(**extra)
        self.assertEqual(FakeEnv.builds, before)

    def test_native_variation_reset_uses_native_task_and_evaluation(self):
        backend = self.backend()
        backend.reset(options={"variation_id": 0})
        self.assertEqual(backend.env.task.kind, "bridge")
        self.assertEqual(backend.evaluate(), backend.env.evaluate())

    def test_policy_gets_reference_and_teacher_reward_is_delta(self):
        for name in NAMED:
            backend = self.backend(name)
            access = backend.policy_access()
            self.assertEqual(access.instruction(), INSTRUCTIONS[name])
            self.assertTrue(access.task_images()["reference"][1].startswith(b"\x89PNG"))
            backend.close()
        backend = self.backend()

        def erase(action, role):
            backend.env.ink.tiles.clear()
            backend.env.tick += 1
            return {"ticks_executed": 1}
        backend._checked = erase
        self.assertEqual(backend.act(object()).reward, 1.)
        self.assertEqual(backend.act(object()).reward, 0.)

    def test_catalogue_launch_and_serialized_launch_construct_v2(self):
        from gym_anything_robotics.launch import LaunchSpec
        from gym_anything_robotics.tasks import TaskSpec
        from whiteboard.task_set_v2_backend import NewtonWhiteboardTaskSetV2Backend
        with patch("gym_anything_robotics.env.RoboticsEnv", side_effect=lambda backend: backend):
            for selection in V2.SELECTIONS[:-1]:
                base = LaunchSpec("whiteboard", backend_options={"recording": False, "cameras": [],
                    "output_dir": self.directory.name, "texture_width": 1024, "texture_height": 512})
                spec = tasks.resolve("whiteboard", selection, base=base)
                backend = TaskSpec.from_dict(json.loads(json.dumps(spec.to_dict()))).create()
                self.assertIsInstance(backend, NewtonWhiteboardTaskSetV2Backend)
                self.assertEqual(backend.env.task, V2.TaskV2(selection["task"]))
                result = backend.evaluate()
                self.assertEqual((result["score"], result["contract"]), (0., V2.CONTRACT))
                backend.close()

    def test_durable_checkpoints_preserve_the_v2_task(self):
        from whiteboard.storage import snapshot_bytes, restore_bytes
        backend = self.backend("correct-misspelling")
        env = backend.env
        env.fixture_mode = False
        env._mechanics_fingerprint = "stubbed-frozen-physics"
        before = snapshot_bytes(env)
        env.ink.tiles.clear()
        restore_bytes(env, before)
        self.assertEqual(backend.evaluate()["score"], 0)
        env.task = CatalogueTask("correct-misspelling")          # the v1 task is a different task
        with self.assertRaises(ValueError):
            restore_bytes(env, before)


if __name__ == "__main__":
    unittest.main()
