"""Contract checks for sewing: catalogue, registration, robot selection, machine I/O validation, restricted access.

The first classes need no GPU and no physics (seconds). The physics class builds the real world for
sewing/v1/sample-room/plain-seam/seed-0 with short plies (~1 minute) and is skipped unless GAR_SEWING_PHYSICS_TESTS=1
(the repository's convention for GPU tests). These are interface checks, not physical qualification or a task
solution.
"""

import json
import os
import unittest

from gym_anything_robotics import tasks
from gym_anything_robotics.errors import ActionRejectedError, UnsupportedConfigurationError
from tests.packages import PACKAGES

ROOT = PACKAGES


class CatalogueTest(unittest.TestCase):
    def test_benchmark_identity_and_schema(self):
        raw = json.loads((ROOT/"sewing"/"benchmark.json").read_text(), parse_constant=lambda c: self.fail(c))
        self.assertEqual(tasks.catalog("sewing"), raw)
        self.assertEqual(raw["schema"], "gar.task-catalog.v1")
        self.assertEqual(raw["selectors"], ["preset", "task", "seed", "variation_id", "contract"])
        self.assertEqual(raw["resolver"], "sewing.variations:task_resolve")
        self.assertTrue((ROOT/raw["evaluator"]["source"]).is_file())
        self.assertEqual(tasks.resolve("sewing").task_id, raw["default_task_id"])
        for alias in ("sew", "sewing-machine", "lockstitch", "sew-plain-seam"):
            self.assertEqual(tasks.catalog(alias)["environment_id"], "sewing")

    def test_tasks_resolve_and_round_trip(self):
        from sewing.task import INSTRUCTIONS, TASKS, TIME_LIMIT_S, TASK_SETUP, Evaluator
        self.assertEqual(tasks.catalog("sewing")["count"], len(TASKS))
        self.assertEqual(set(INSTRUCTIONS), set(TASKS))
        self.assertEqual(set(TIME_LIMIT_S), set(TASKS))
        self.assertEqual(set(TASK_SETUP), set(TASKS))
        ids = set()
        for task, code in TASKS.items():
            from sewing.task_set import NEW_TASKS
            self.assertTrue(task in NEW_TASKS or hasattr(Evaluator, "_eval_" + task.replace("-", "_")), task)
            for sel in ({"task": task, "seed": 3}, {"task": code, "seed": 3}, {"preset": "sample-room", "task": task,
                                                                               "seed": 3}):
                spec = tasks.resolve("sewing", sel)
                self.assertEqual(spec.task_id, f"sewing/v1/sample-room/{task}/seed-3")
                self.assertEqual(spec.launch.backend_options, {"preset": "sample-room", "task": task, "seed": 3})
                self.assertEqual(spec.description, INSTRUCTIONS[task])
                again = tasks.TaskSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
                self.assertEqual(again.to_dict(), spec.to_dict())
                ids.add(spec.task_id)
        self.assertEqual(len(ids), len(TASKS))
        with self.assertRaises(ValueError):
            tasks.resolve("sewing", {"task": "curtain-hem"})
        with self.assertRaises(ValueError):
            tasks.resolve("sewing", {"task": "plain-seam", "seed": 99})

    def test_a_catalogue_variation_is_selected_at_launch(self):
        # `gar run sewing --variation 12` and `--select variation_id=12` select an entry of the 50 in
        # sewing/variations.py
        from argparse import Namespace
        from gym_anything_robotics import cli_launch
        from sewing import variations as VR
        self.assertEqual(tasks.catalog("sewing")["variation_catalogue"]["count"], VR.COUNT)
        for index in range(VR.COUNT):
            entry = VR.specification(index)
            spec = tasks.resolve("sewing", {"variation_id": index})
            self.assertEqual(spec.task_id, entry["task_id"])
            options = {"preset": "sample-room", "task": entry["task"], "seed": entry["seed"]}
            self.assertEqual(spec.launch.backend_options, options)
            # the launch a task and seed selects, with the entry named in the selection
            by_name = tasks.resolve("sewing", {"task": entry["task"], "seed": entry["seed"]})
            self.assertEqual(spec.launch.to_dict(), by_name.launch.to_dict())
            self.assertEqual(spec.description, by_name.description)
            self.assertEqual(spec.selection, {**by_name.selection, "variation_id": index, "catalogue": VR.VERSION,
                                              "sha256": entry["sha256"]})
            self.assertEqual(tasks.TaskSpec.from_dict(json.loads(json.dumps(spec.to_dict()))).to_dict(),
                             spec.to_dict())
        self.assertEqual(tasks.resolve("sewing", {"variation_id": 12}).task_id,
                         "sewing/v1/sample-room/plain-seam/seed-2")
        # a task or seed beside the id must be the entry's; ids outside the catalogue and non-integers are refused
        self.assertEqual(tasks.resolve("sewing", {"variation_id": 12, "task": "plain-seam", "seed": 2}).task_id,
                         "sewing/v1/sample-room/plain-seam/seed-2")
        for bad in ({"variation_id": 12, "task": "curve"}, {"variation_id": 12, "seed": 3}, {"variation_id": 90},
                    {"variation_id": -1}, {"variation_id": "12"}, {"variation_id": True}, {"variation_id": 12.0},
                    {"variation_id": 12, "preset": "home"}, {"variation_id": 12, "split": "test"}):
            with self.subTest(selection=bad), self.assertRaises(ValueError):
                tasks.resolve("sewing", bad)
        # the command line's two spellings (gym_anything_robotics/cli_launch.py), and with a robot and options beside
        blank = dict(task_file=None, config=None, backend_options_json=None, robot=None, backend=None, device=None,
                     task=None, select=[], selector_json=None, variation=None)
        for extra in (dict(variation=12), dict(select=["variation_id=12"])):
            with self.subTest(**extra):
                launch, task = cli_launch.launch(Namespace(**{**blank, "environment": "sewing", **extra}))
                self.assertEqual(task.task_id, "sewing/v1/sample-room/plain-seam/seed-2")
                self.assertEqual(launch.backend_options, {"preset": "sample-room", "task": "plain-seam", "seed": 2})
        launch, task = cli_launch.launch(Namespace(**{**blank, "environment": "sewing", "variation": 3, "robot": "g1",
                                                      "backend_options_json": '{"cameras": ["needle"]}'}))
        self.assertEqual((launch.robot, task.task_id), ("g1", "sewing/v1/sample-room/load/seed-3"))
        self.assertEqual(launch.backend_options, {"cameras": ["needle"], "preset": "sample-room", "task": "load",
                                                  "seed": 3})
        with self.assertRaises(ValueError):                 # a task that is not the entry's
            cli_launch.launch(Namespace(**{**blank, "environment": "sewing", "variation": 12, "task": "curve"}))

    def test_seeds_are_deterministic_and_bounded(self):
        from sewing.variants import resolve
        a, b = resolve(task="plain-seam", seed=5), resolve(task="plain-seam", seed=5)
        self.assertEqual(a, b)
        (dx, dy, yaw), mis = a["ply_offset"], a["misalign"]
        self.assertTrue(abs(dx) <= 1e-3 and abs(dy) <= 2e-3 and abs(yaw) <= .5 and 0 <= mis <= .5e-3)
        self.assertEqual(resolve(task="plain-seam", seed=0)["ply_offset"], (0., 0., 0.))


class RegistrationTest(unittest.TestCase):
    def test_registry_launch_and_robots(self):
        from gym_anything_robotics import launch, registry
        self.assertIn(("sewing", "newton"), {(e["environment_id"], e["backend"])
                                             for e in registry.available_environments()})
        d = launch.describe("sewing")
        self.assertEqual(d["default_robot"], "dual-fr3")
        self.assertEqual({r["id"] for r in d["robots"]}, {"dual-fr3", "dual-panda", "single-fr3", "g1"})
        from robots.catalog import select_for_backend, for_environment, resolve
        for alias, rig in (("dual-fr3", "sewing-fr3"), ("dual-panda", "sewing-panda"),
                           ("single-fr3", "sewing-fr3-single"), ("g1", "sewing-g1")):
            self.assertEqual(for_environment("sewing", alias).to_dict()["id"], rig)
            self.assertEqual(select_for_backend("sewing", resolve(rig).to_dict()), alias)
        with self.assertRaises(ValueError):
            select_for_backend("sewing", resolve("filleting-fr3").to_dict())


class FaultedStateTest(unittest.TestCase):
    def test_a_faulted_world_is_not_reported_as_a_score(self):
        """The backend's state of a faulted world carries no numeric score; it is unavailable, with the fault and the
        last judged evaluation, marked interrupted. A state that is not faulted is unchanged."""
        from types import SimpleNamespace
        from gym_anything_robotics.backends.newton_sewing import NewtonSewingBackend
        b = SimpleNamespace(_last_valid_evaluation=None, task_contract="v1")
        ok = dict(tick=10, fault=None, evaluation=dict(success=False, score=0.25, checks={"x": {"ok": True}}))
        self.assertIs(NewtonSewingBackend._unjudged_fault(b, ok), ok)
        self.assertEqual(ok["evaluation"]["score"], 0.25)
        bad = NewtonSewingBackend._unjudged_fault(b, dict(tick=11, fault="FloatingPointError: x",
                                                          evaluation=dict(success=False, score=0.0, checks={})))
        ev = bad["evaluation"]
        self.assertIsNone(ev["score"])
        self.assertFalse(ev["available"])
        self.assertTrue(ev["interrupted"])
        self.assertEqual(ev["fault"], "FloatingPointError: x")
        self.assertEqual(ev["last_valid"], dict(tick=10, evaluation=ok["evaluation"]))


class HistoryTest(unittest.TestCase):
    def test_reflex_counted_once(self):
        """The evaluator's reflex count: one per protective stop recorded while the machine runs, including two arms
        stopping in one tick. No physics."""
        from types import SimpleNamespace
        from sewing.env import SewingEnv
        env = SewingEnv.__new__(SewingEnv)
        w = SimpleNamespace(machine=SimpleNamespace(mode="sew", speed_ramped=600.), config=SimpleNamespace(dt=5e-4),
                            reflexes=[], tick=0, hand_arms=[])
        env.world = w
        env._reset_histories()
        stops = {10: ["left"], 100: ["left"], 150: ["left", "right"], 151: ["left"]}
        for t in range(1, 200):
            w.tick = t
            w.reflexes += [dict(tick=t, arm=a) for a in stops.get(t, ())]
            env.steps += 1
            env._record()
        self.assertEqual(env.history["reflex_while_sewing"], 5)


@unittest.skipUnless(os.environ.get("GAR_SEWING_PHYSICS_TESTS") == "1", "set GAR_SEWING_PHYSICS_TESTS=1 (GPU)")
class PhysicsInterfaceTest(unittest.TestCase):
    def test_live_session_contract(self):
        import gym_anything_robotics as gar
        from gym_anything_robotics.actions import ToolIO, NamedJointTarget
        env = gar.create("sewing", task="plain-seam", seed=0, cameras=(), config=dict(ply_length=0.12))
        try:
            env.reset()
            with self.assertRaises(ActionRejectedError):
                env.act(ToolIO(arm="right", outputs={"speed_spm": 9000.0}, ticks=1))
            with self.assertRaises(ActionRejectedError):
                env.act(ToolIO(arm="right", outputs={"pedal": 1.0}, ticks=1))
            with self.assertRaises(ActionRejectedError):
                env.act(NamedJointTarget({"right/fr3_joint1": 99.0}, ticks=1))
            before = env.get_state()["tick"]
            self.assertEqual(before, 0)
            r = env.act(ToolIO(arm="left", outputs={"speed_spm": 600.0}, ticks=200))
            self.assertEqual(r.info["executed_ticks"], 200)
            self.assertIsNotNone(env.evaluate()["details"]["checks"])
            pa = env.policy_access()
            self.assertIn("plain seam", pa.instruction())
        finally:
            env.close()
        with self.assertRaises(UnsupportedConfigurationError):
            gar.create("sewing", task="plain-seam", config=dict(not_a_field=1))
