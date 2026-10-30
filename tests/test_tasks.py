"""Task identity and evaluator transport checks, not physical qualification."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from gym_anything_robotics import LaunchSpec
from gym_anything_robotics import tasks, sessions
from gym_anything_robotics.cli import main
from gym_anything_robotics.env import RoboticsEnv
from gym_anything_robotics.evaluation import normalize
from gym_anything_robotics.remote import RemoteEnv, RemoteError
from gym_anything_robotics.session_service import SessionService, SessionServer
from tests.test_http_server import HTTPFakeBackend


class TaskTests(unittest.TestCase):
    def test_catalogs_have_no_physics_imports(self):
        with patch("gym_anything_robotics.registry.create") as create:
            values = tasks.catalogs()
            from gym_anything_robotics.registry import available_environments
            expected = {item["environment_id"] for item in available_environments()}
            self.assertEqual({item["environment_id"] for item in values}, expected)
            self.assertEqual(len(values), len(expected))
            self.assertEqual(tasks.catalog("whiteboard")["splits"], ["train", "validation", "test"])
            self.assertEqual(tasks.catalog("titration")["splits"], [])
            create.assert_not_called()

    def test_deposit_cans_exact_native_inventories(self):
        from deposit_cans.variants import PRESETS, TASKS_V2_CONTRACT, resolve
        # A bare selection keeps the default contract (reference: v1).
        for name, total in (("reference", 72), ("high_fill", 144)):
            spec = tasks.resolve("deposit-cans", {"variant": name, "seed": 7})
            self.assertEqual(spec.selection, {"variant": name, "seed": 7})
            self.assertEqual(spec.launch.backend_options, {"variant": name, "config": {"seed": 7}})
            inventory = resolve(name, 7)["config"]
            self.assertEqual(inventory, dict(PRESETS[name], seed=7))
            self.assertEqual(inventory["target_count"] + inventory["preload_count"], total)
            self.assertEqual(inventory["tub_count"], 16)
            self.assertIsNone(spec.split)
            self.assertEqual(spec.task_id, f"deposit-cans/v1/{name}/seed-7")
        self.assertEqual(set(PRESETS), {"reference", "high_fill", "single_can", "lid_cans", "sort_cups", "sort_waste",
                                        "empty_tub", "tub_cups"})
        self.assertEqual(tasks.resolve("deposit-cans").task_id, tasks.catalog("deposit-cans")["default_task_id"])
        self.assertEqual(tasks.catalog("deposit-cans")["default_task_id"], "deposit-cans/v1/reference/seed-0")
        self.assertEqual(tasks.resolve("deposit-cans", {"variant": "reference", "contract": "v1", "seed": 7}).to_dict(),
                         tasks.resolve("deposit-cans", {"variant": "reference", "seed": 7}).to_dict())
        # The eight-task catalogue's reference entry selects its contract explicitly.
        spec = tasks.resolve("deposit-cans", {"variant": "reference", "contract": TASKS_V2_CONTRACT, "seed": 7})
        self.assertEqual(spec.selection, {"variant": "reference", "seed": 7, "contract": TASKS_V2_CONTRACT})
        self.assertEqual(spec.launch.backend_options, {"variant": "reference", "contract": TASKS_V2_CONTRACT, "config": {"seed": 7}})
        self.assertEqual(spec.task_id, f"deposit-cans/v1/reference/{TASKS_V2_CONTRACT}/seed-7")
        with self.assertRaises(ValueError): tasks.resolve("deposit-cans", {"variant": "high_fill", "contract": "v1"})
        self.assertEqual(tasks.catalog("deposit-cans-into-container")["environment_id"], "deposit-cans")
        self.assertIn("Component-qualified v6/scene7", tasks.catalog("deposit-cans")["qualification"])
        for selection in ({"seed": True}, {"seed": -1}, {"seed": 2**32}, {"variant": "missing"}, {"extra": 0}):
            with self.assertRaises(ValueError): tasks.resolve("deposit-cans", selection)
        with self.assertRaises(ValueError):
            tasks.resolve("deposit-cans", {"variant": "high_fill"},
                          base=LaunchSpec("deposit-cans", backend_options={"variant": "reference"}))


    def test_fixed_roundtrip_and_exact_merge(self):
        base = LaunchSpec("football", backend_options={"config": {"dt": .002, "iterations": 64}})
        spec = tasks.resolve("football", base=base)
        self.assertEqual(tasks.TaskSpec.from_dict(spec.to_dict()).to_dict(), spec.to_dict())
        self.assertEqual(spec.launch.to_dict(), base.to_dict())
        with self.assertRaises(ValueError):
            tasks.resolve("football", {"seed": 3})
        with self.assertRaises(ValueError):
            tasks.merge_exact({"config": {"dt": .1}}, {"config": {"dt": .2}})

    def test_laundry_all_native_tasks(self):
        for garment in ("tee", "shirt", "jeans", "towel", "asymmetric"):
            spec = tasks.resolve("laundry", {"garment": garment})
            self.assertEqual(spec.launch.backend_options["config"]["garment"], garment)
            self.assertIn("laundry/" + garment + "/", spec.task_id)
            self.assertIsNone(spec.split)
        with self.assertRaises(ValueError):
            tasks.resolve("laundry", {"garment": "hat"})
        with self.assertRaises(ValueError):
            tasks.resolve("laundry", {"garment": "tee"},
                          base=LaunchSpec("laundry", backend_options={"config": {"garment": "towel"}}))

    def test_whiteboard_native_identity_and_split(self):
        from whiteboard.variations import specification
        for index in (0, 17, 9999):
            spec = tasks.resolve("whiteboard", {"variation_id": index})
            self.assertEqual(spec.selection, json.loads(json.dumps(specification(index))))
            self.assertEqual(spec.split, specification(index)["split"])
        with self.assertRaises(ValueError):
            tasks.resolve("whiteboard", {"variation_id": 10000})
        native = specification(0)
        other_split = "test" if native["split"] != "test" else "train"
        with self.assertRaises(ValueError):
            tasks.resolve("whiteboard", {"variation_id": 0, "split": other_split})

    def test_titration_inventory_and_resolution_preserved(self):
        spec = tasks.resolve("titration")
        self.assertEqual(spec.launch.backend_options["liquid_numerics"]["spacing_m"], .000625)
        self.assertEqual(spec.launch.backend_options["liquid_numerics"]["dt_s"], .000125)
        reagents = spec.launch.backend_options["reagents"]
        self.assertEqual(sum(reagents[k] for k in ("aliquot_ml", "dilution_ml", "burette_ml", "acid_bottle_ml", "base_bottle_ml")), 85.)
        varied = tasks.resolve("titration", {"constraints": {"aliquot_ml": 5.}})
        self.assertEqual(varied.launch.backend_options["reagents"]["aliquot_ml"], 5.)
        with self.assertRaises(ValueError):
            tasks.resolve("titration", {"variation_id": 1100})


    def test_cli_resolve_and_task_launch_are_non_destructive(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            path = Path(d) / "task.json"
            self.assertEqual(main(["tasks", "resolve", "football", "--output", str(path)]), 0)
            task = tasks.TaskSpec.load(path)
            with patch("gym_anything_robotics.sessions.prepare", return_value=Path(d)) as prepare, patch("gym_anything_robotics.sessions.start", return_value={}):
                # This test stubs the process; live readiness is covered separately.
                self.assertEqual(main(["run", "--task-file", str(path), "--no-wait"]), 0)
                self.assertEqual(prepare.call_args.kwargs["task"].to_dict(), task.to_dict())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["tasks", "resolve", "football", "--output", str(path)]), 1)
                self.assertEqual(main(["run", "football", "--task-file", str(path)]), 1)

    def test_task_manifest_unknown_fields_rejected(self):
        value = tasks.resolve("football").to_dict()
        value["seed"] = 4
        with self.assertRaises(ValueError):
            tasks.TaskSpec.from_dict(value)

    def test_no_session_before_task_launch_validation(self):
        with patch("gym_anything_robotics.sessions.session_root") as root:
            with self.assertRaises(ValueError):
                sessions.prepare(LaunchSpec("pencap"), task=tasks.resolve("football"))
            root.assert_not_called()


class EvaluationTests(unittest.TestCase):
    def test_missing_not_false_success(self):
        value = normalize(None)
        self.assertFalse(value["available"])
        self.assertIsNone(value["success"])

    def test_invalid_scores_not_success(self):
        for value in ({"success": 1, "score": 1}, {"success": True, "score": float("nan")},
                      {"success": True, "score": 1, "broken": float("inf")}):
            with self.assertRaises(ValueError):
                normalize(value)

    def test_exact_score_no_new_threshold(self):
        result = normalize({"success": False, "score": .92, "stable_s": .12})
        self.assertFalse(result["success"])
        self.assertEqual(result["score"], .92)

    def test_live_http_evaluation_owner_only_and_no_ticks(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        backend = HTTPFakeBackend(Path(temporary.name))
        from gym_anything_robotics.policy_access import PolicyAccess
        backend.policy_access = lambda: PolicyAccess(
            "yam", backend.capabilities.camera_specs, {("tick",): None},
            frozenset({"cartesian_delta"}), lambda: "Test task", {})
        env = RoboticsEnv(backend)
        with patch.object(backend, "evaluate", return_value={"success": False, "score": .2}) as evaluator:
            service = SessionService(env, session_id="evaluation-test")
            server = SessionServer(("127.0.0.1", 0), service, api_token="owner", policy_api_token="student")
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                owner = RemoteEnv(url, api_key="owner")
                policy = RemoteEnv(url, api_key="student")
                before = env.get_state()
                self.assertEqual(owner.evaluate()["score"], .2)
                self.assertEqual(owner.evaluate()["score"], .2)
                self.assertEqual(env.get_state(), before)
                calls = evaluator.call_count
                with self.assertRaises(RemoteError) as error:
                    policy.evaluate()
                self.assertEqual(error.exception.status, 403)
                self.assertEqual(evaluator.call_count, calls)
                self.assertNotIn("GET /v1/evaluation", policy.request("/v1/help")["routes"])
                with contextlib.redirect_stdout(io.StringIO()), patch("gym_anything_robotics.cli._client", return_value=owner):
                    self.assertEqual(main(["verify", "--url", url]), 1)
                with RemoteEnv(url, api_key="owner") as fresh:
                    evaluator.side_effect = ValueError("invalid evaluator state")
                    with self.assertRaises(RemoteError) as fault:
                        fresh.evaluate()
                    self.assertEqual(fault.exception.status, 400)
                    self.assertFalse(fresh.request("/v1/health")["ready"])
            finally:
                server.shutdown()
                worker.join()
                server.server_close()
                service.history.store.close()
                env.close()
