"""classroom-tasks-v2: the render-checked eight-task set. Catalogue, backend and grading checks ported from the
fix branch; NO simulator dependencies.

The worlds are tests/environments/classroom_arrangement/test_task_set.py's stubs (rigid transforms and authored
contacts: no physics, robot, renderer or GPU). These constructed states test the verifier, not physical solvability.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import json
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from classroom_arrangement import layout as L
from classroom_arrangement.catalogue import CONTRACT as V1, LEGACY, TASK_SET as TASK_SET_V1, description
from classroom_arrangement.config import Config
from classroom_arrangement.task import instruction
from classroom_arrangement.task_set import reduction as reduction_v1
from classroom_arrangement.task_set_v2 import START_ROUNDING, reduction
from classroom_arrangement.task_set_v2_catalogue import CONTRACT, TASK_SET, TASKS_V2
from classroom_arrangement.variants import contract as variation_catalogue, task_resolve
from tests.environments.classroom_arrangement import test_task_set as released
from tests.packages import PACKAGES, REPO

ORIGINAL_FIXTURE = released.fixture


def fixture(task, seed=0, contract=CONTRACT):
    return ORIGINAL_FIXTURE(task, seed, contract)


def without_contract(row):
    """A resolved row with its contract taken out of the id, options and selection."""
    strip = lambda d: {k: v for k, v in d.items() if k != "contract"}
    return dict(row, task_id=row["task_id"].replace(f"/{row['selection']['contract']}", ""),
                backend_options=strip(row["backend_options"]), selection=strip(row["selection"]))


class CatalogueTests(unittest.TestCase):

    def test_benchmark_lists_the_eight_v2_selections(self):
        b = json.loads((PACKAGES/"classroom_arrangement"/"benchmark.json").read_text())
        ts = b["task_set"]
        self.assertEqual(ts["contract"], CONTRACT)
        self.assertEqual(ts["selections"], [dict(s) for s in TASK_SET])
        self.assertEqual(ts["task_ids"], [task_resolve(dict(s))["task_id"] for s in TASK_SET])
        self.assertEqual(len(set(ts["task_ids"])), 8)
        self.assertEqual(b["task_set_v1"], list(TASK_SET_V1))
        # the audited hardest task keeps its released legacy selection, exactly as in splits/test.json
        test = json.loads((REPO/"benchmarks"/"roboworld"/"splits"/"test.json").read_text())["tasks"]
        self.assertEqual([t["selection"] for t in test if t["environment_id"] == b["environment_id"]], [ts["selections"][-1]])
        self.assertEqual(ts["selections"][-1], {"task": "group-work"})

    def test_v2_rows_are_v1_rows_with_the_v2_contract(self):
        for t in TASKS_V2:
            row = task_resolve(dict(task=t, seed=0, contract=CONTRACT))
            self.assertEqual(row["task_id"], f"classroom-arrangement/v1/room114/{t}/{CONTRACT}/seed-0")
            self.assertEqual(row["description"], description(t, V1))
            self.assertEqual(row["selection"], row["backend_options"])
            self.assertEqual(Config(**row["backend_options"]).contract, CONTRACT)
            self.assertEqual(without_contract(row), without_contract(task_resolve(dict(task=t, seed=0, contract=V1))))

    def test_released_selections_resolve_as_released(self):
        b = json.loads((PACKAGES/"classroom_arrangement"/"benchmark.json").read_text())
        self.assertEqual(task_resolve({})["task_id"], b["default_task_id"])
        for t in L.TASKS:
            old = task_resolve(dict(task=t, seed=7))
            self.assertEqual(old, task_resolve(dict(task=t, seed=7, contract=LEGACY)))
            self.assertEqual(old["task_id"], f"classroom-arrangement/v1/room114/{t}/seed-7")
            self.assertEqual(old["description"], L.TASKS[t]["instruction"])
        for selection in TASK_SET_V1:
            row = task_resolve(selection)
            self.assertNotIn(CONTRACT, row["task_id"])
            self.assertEqual(Config(**row["backend_options"]).contract, selection.get("contract", LEGACY))

    def test_invalid_v2_selections_fail_loudly(self):
        for t in ("group-work", "chairs-down", "make-pod"):
            with self.assertRaises(ValueError):
                task_resolve(dict(task=t, contract=CONTRACT))
            with self.assertRaises(ValueError):
                Config(task=t, contract=CONTRACT)
        # the released refusal for an unknown contract reads as before
        with self.assertRaisesRegex(ValueError, r"^contract must be 'v1' or 'classroom-tasks-v1'$"):
            task_resolve(dict(task="chairs-up", contract="bad"))

    def test_instructions_reach_both_surfaces(self):
        for t in TASKS_V2:
            env = fixture(t)
            self.assertEqual(instruction(env.config, env.spec), instruction(Config(task=t, contract=V1), env.spec))
            self.assertEqual(description(t, CONTRACT), description(t, V1))
        self.assertIn("20-35 cm", instruction(Config(task="seat-tutoring-group"), fixture("seat-tutoring-group").spec))
        self.assertNotIn("20-35 cm", description("seat-tutoring-group", CONTRACT))

    def test_backend_selection_reset_and_catalogue_without_runtime(self):
        from gym_anything_robotics.backends.newton_classroom_arrangement import NewtonClassroomArrangementBackend as Backend
        from gym_anything_robotics.errors import UnsupportedActionError

        @dataclass
        class Caps:
            notes: dict

        def new_env(config, **kwargs):
            return NS(config=config, world=object(), lock=nullcontext(), generation=0,
                      get_state=lambda: {}, reset=lambda: {})

        backend = Backend.__new__(Backend)
        backend.config = Config()
        backend.env = new_env(backend.config)
        backend.trace, backend.output, backend.recording = None, None, False
        backend._release = lambda: None
        backend._notes = lambda: {}
        backend.capabilities = Caps({})
        modules = {"classroom_arrangement.env": NS(ClassroomEnv=new_env),
                   "classroom_arrangement.kinematics": NS(ArmKinematics=lambda world: None)}
        before = set(sys.modules)
        with patch.dict(sys.modules, modules):
            # a legacy episode: catalogue and refusals as released
            self.assertEqual(backend.variation_contract(), dict(**variation_catalogue(), current=backend._variation()))
            with self.assertRaisesRegex(ValueError, r"^Selectors are preset, task, seed, glides, contents and contract$"):
                backend.select_variation(dict(colour="red"))
            for selection in TASK_SET:
                request = backend.select_variation(selection)
                backend.reset(**request["reset_request"])
                self.assertEqual(backend._variation(), task_resolve(selection)["selection"])
                self.assertEqual(backend.env.config.contract, selection.get("contract", LEGACY))
                self.assertEqual(request["task"], task_resolve(selection)["description"])
            backend.reset(options=dict(task="push-in-chairs", contract=CONTRACT))
            backend.reset(seed=1)
            self.assertEqual(backend.config.contract, CONTRACT)
            # a v2 episode lists the v2 set and keeps the released one beside it
            catalogue = backend.variation_contract()
            self.assertEqual(catalogue["contracts"], [LEGACY, V1, CONTRACT])
            self.assertEqual(catalogue["task_set"], list(TASK_SET))
            self.assertEqual(catalogue["task_set_v1"], list(TASK_SET_V1))
            self.assertEqual(catalogue["current"], dict(preset="room114", task="push-in-chairs", seed=1, glides="nylon",
                                                        contents=0, contract=CONTRACT))
            with self.assertRaisesRegex(UnsupportedActionError, r"^Unknown reset options \['colour'\]; selectors are preset, "
                                        r"task, seed, glides, contents, contract; nothing changed$"):
                backend.reset(options=dict(colour="red"))
            current = backend.env
            with self.assertRaises(UnsupportedActionError):
                backend.reset(options=dict(task="group-work", contract=CONTRACT))
            self.assertIs(backend.env, current)
            backend.reset(options=dict(contract=V1))
            self.assertEqual(backend.variation_contract(), dict(**variation_catalogue(), current=backend._variation()))
            backend.reset(options=dict(contract=LEGACY))
            self.assertNotIn("contract", backend._variation())
            self.assertNotIn("task_set_v1", backend.variation_contract())
            # nothing above imported the simulator (other test files may have loaded it before)
            self.assertFalse({"newton", "warp", "mujoco", "classroom_arrangement.world"} & (sys.modules.keys() - before))


class FixTests(unittest.TestCase):
    """The render-check fix: exact-zero untouched starts."""

    def test_reduction_exact_zero_untouched_and_exact_goal(self):
        self.assertIsNone(reduction(0., 0.))
        self.assertAlmostEqual(reduction(0.25, 0.5), 0.5, places=8)
        # an untouched start is exactly 0 through the start's last-bit rounding; the goal is exactly 1
        self.assertEqual(reduction(0.5 - 2e-14, 0.5), 0.)
        self.assertEqual(reduction(0.5, 0.5), 0.)
        self.assertEqual(reduction(0.02, 0.5, 0.02), 1.)
        self.assertEqual(reduction(0., 0.5), 1.)
        self.assertEqual(START_ROUNDING, 1e-9)
        # the released v1 reduction is unchanged, last-bit credit included
        self.assertEqual(reduction_v1(0.25, 0.5), 0.5)
        self.assertGreater(reduction_v1(0.5 - 2e-14, 0.5), 0.)

    def test_last_bit_start_difference_scores_exactly_zero(self):
        # the render check's case: straighten-row's desk_34 a few 1e-14 m off its recorded post-settle start
        for contract in (V1, CONTRACT):
            env = fixture("straighten-row", contract=contract)
            a = np.array(env.world.pose2d("desk_34")[0])
            d = np.array(env.spec["targets"]["desk_34"]) - a
            env.world.put("desk_34", a + 1e-14*d/np.linalg.norm(d[:2]))
            ev = env.evaluator.evaluate()
            self.assertEqual(ev["contract"], contract)
            if contract == V1:
                self.assertGreater(ev["progress"], 0.)
                self.assertLess(ev["progress"], 1e-12)
            else:
                self.assertEqual((ev["progress"], ev["score"]), (0., 0.))

    def test_v2_grades_as_v1_away_from_the_start(self):
        for t, n in (("push-in-chairs", "chair_21"), ("straighten-row", "desk_31"), ("exam-row", "desk_21"),
                     ("seat-tutoring-group", "tchair_3"), ("clear-floor", "tote"), ("move-teacher-desk", "teacher_desk")):
            with self.subTest(task=t):
                envs = {c: fixture(t, contract=c) for c in (V1, CONTRACT)}
                a = np.array(envs[V1].spec["items"][n]["pose"])
                b = np.array(envs[V1].spec["targets"].get(n, (1.1, 9., 0.)))
                for alpha in (0.25, 0.5, 1.):
                    ev = {}
                    for c, env in envs.items():
                        env.world.put(n, a + alpha*(b - a))
                        ev[c] = env.evaluator.evaluate()
                    self.assertEqual(ev[CONTRACT]["succeeded"], ev[V1]["succeeded"])
                    self.assertEqual(ev[CONTRACT]["reward"]["contract"], CONTRACT)
                    self.assertLess(abs(ev[CONTRACT]["score"] - ev[V1]["score"]), 1e-6)

    def test_group_work_keeps_its_released_selection_and_evaluator(self):
        row = task_resolve(TASK_SET[-1])
        self.assertEqual(row, task_resolve(dict(task="group-work")))
        self.assertEqual(Config(**row["backend_options"]).contract, LEGACY)
        env = fixture("group-work", contract=LEGACY)
        motion, window = env.window_motion()
        self.assertEqual(env.evaluator.evaluate(), env.evaluator.by_place(motion, window))


class ReleasedEvaluationTestsUnderV2(released.EvaluationTests):
    """Every released classroom-tasks-v1 grading test, run on classroom-tasks-v2 (as the fix branch ran them)."""

    def setUp(self):
        for name, value in (("fixture", fixture), ("CONTRACT", CONTRACT)):
            p = patch.object(released, name, value)
            p.start()
            self.addCleanup(p.stop)


if __name__ == "__main__":
    unittest.main()
