"""Pure Python geometry and stub-world tests; never import/start the simulator.

python -m pytest tests/environments/classroom_arrangement
These constructed states test the verifier, not physical solvability.
"""
import json
import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from classroom_arrangement import layout as L, params as P
from classroom_arrangement.catalogue import CONTRACT, SET_TASKS, TASK_SET, description
from classroom_arrangement.config import Config
from classroom_arrangement.scene import BUILDERS, chair_up_pose
from classroom_arrangement.task import Evaluator, instruction
from classroom_arrangement.task_set import reduction
from classroom_arrangement.variants import task_resolve
from tests.packages import PACKAGES


class Array:
    def __init__(self, data):
        self.data = np.asarray(data)

    def numpy(self):
        return self.data


class World:
    """Rigid transforms, catalogue geometry, and explicitly authored contact records.

    No gravity, integration, dynamics, solver, robot, rendering or runtime package.
    """
    def __init__(self, spec):
        self.scene = NS(items={}, bodies={}, parts={})
        self.transforms, self.floor, self.support = {}, set(), {}
        sizes, coms = [(0., 0., 0.)], []
        for n, it in spec["items"].items():
            body = len(self.scene.bodies)
            parts = BUILDERS[it["kind"]]()
            self.scene.bodies[n] = body
            self.scene.parts[n] = parts
            shapes, glides = [], []
            for part in parts:
                if not part.collide:
                    continue
                s = len(sizes)
                sizes.append(tuple(part.size) + (0.,)*(3 - len(part.size)))
                shapes.append(s)
                if part.role == "glide":
                    glides.append(s)
            self.scene.items[n] = dict(kind=it["kind"], body=body, shapes=shapes, glides=glides)
            coms.append(sum(np.asarray(p.pos)*p.mass for p in parts)/sum(p.mass for p in parts))
            x, y, yaw = it["pose"]
            self.transforms[n] = (np.array([x, y, 0.0005]), Rotation.from_euler("z", yaw).as_matrix())
            self.floor.add(n)
        self._shape_geom = np.arange(len(sizes))
        self.mjm = NS(geom_size=np.asarray(sizes))
        self.model = NS(body_com=Array(coms))
        self.state = NS(body_q=Array(np.zeros((len(coms), 7))))
        self.mjd = NS(geom_xpos=np.zeros((len(sizes), 3)), contact=[], ncon=0)
        self.sync()

    def body_pose(self, n):
        return self.transforms[n]

    def pose2d(self, n):
        p, R = self.transforms[n]
        return (float(p[0]), float(p[1]), math.atan2(R[1, 0], R[0, 0])), math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))

    def put(self, n, pose, z=0.0005, rotation=None):
        x, y, yaw = pose
        self.transforms[n] = (np.array([x, y, z]), Rotation.from_euler("z", yaw).as_matrix() if rotation is None else rotation)
        self.support.pop(n, None)
        self.floor.discard(n)
        if z == 0.0005:
            self.floor.add(n)
        self.sync()

    def put_up(self, n, desk, contact=True, rise=0.):
        p, quat = chair_up_pose(self.pose2d(desk)[0])
        self.put(n, (p[0], p[1], 0.), z=p[2] + rise, rotation=Rotation.from_quat(quat).as_matrix())
        if contact:
            self.support[n] = desk
        self.sync()

    def swap(self, a, b):
        self.transforms[a], self.transforms[b] = self.transforms[b], self.transforms[a]
        for s in (self.floor,):
            has_a, has_b = a in s, b in s
            s.discard(a); s.discard(b)
            if has_a:
                s.add(b)
            if has_b:
                s.add(a)
        da, db = self.support.pop(a, None), self.support.pop(b, None)
        if da:
            self.support[b] = da
        if db:
            self.support[a] = db
        # Contacts refer to bodies: swapped desks carry their support assignment.
        self.support = {n: b if d == a else a if d == b else d for n, d in self.support.items()}
        self.sync()

    def sync(self):
        contacts = []
        for n, it in self.scene.items.items():
            p, R = self.transforms[n]
            self.state.body_q.data[it["body"], :3] = p
            for s, part in zip(it["shapes"], [p for p in self.scene.parts[n] if p.collide]):
                self.mjd.geom_xpos[s] = p + R @ np.asarray(part.pos)
            if n in self.floor:
                contacts.append(NS(geom=(0, it["shapes"][0])))
            if n in self.support:
                d = self.support[n]
                contacts.append(NS(geom=(self.scene.items[d]["shapes"][0], it["shapes"][0])))
        self.mjd.contact, self.mjd.ncon = contacts, len(contacts)


def fixture(task, seed=0, contract=CONTRACT):
    spec = L.setup(task, seed)
    world = World(spec)
    env = NS(config=Config(task=task, seed=seed, contract=contract), spec=spec, world=world, complete=True,
             motion={n: 0. for n in spec["items"]})
    env.window_motion = lambda: (env.motion, env.complete)
    env.evaluator = Evaluator(env)
    return env


def finish(env):
    w, spec = env.world, env.spec
    for n, p in spec["targets"].items():
        w.put(n, p)
    task = spec["task"]
    if task == "chairs-up":
        for c in range(1, 7):
            w.put_up(f"chair_4{c}", f"desk_4{c}")
    elif task == "clear-floor":
        w.put("tote", (1.1, 9., 0.))
        w.put("basket", (*L.BASKET_HOME, 0.))


class CatalogueTests(unittest.TestCase):

    def test_eight_exact_selections_and_legacy_identity(self):
        b = json.loads((PACKAGES/"classroom_arrangement"/"benchmark.json").read_text())
        self.assertEqual(b["task_set"], list(TASK_SET))
        self.assertEqual(len(TASK_SET), 8)
        self.assertEqual(b["count"], 10)
        self.assertEqual(task_resolve({})["task_id"], b["default_task_id"])
        for t in L.TASKS:
            old = task_resolve(dict(task=t, seed=7))
            self.assertEqual(old, task_resolve(dict(task=t, seed=7, contract="v1")))
            self.assertEqual(old["task_id"], f"classroom-arrangement/v1/room114/{t}/seed-7")
            self.assertEqual(old["description"], L.TASKS[t]["instruction"])
            self.assertNotIn("contract", old["backend_options"])
        for selection in TASK_SET:
            row = task_resolve(selection)
            self.assertEqual(row["selection"], row["backend_options"])
            config = Config(**row["backend_options"])
            self.assertEqual(config.contract, selection.get("contract", "v1"))
            if config.contract == CONTRACT:
                self.assertIn(CONTRACT, row["task_id"])
                self.assertNotIn("20-35 cm", row["description"])
                self.assertIn("Leave the rest of the room", row["description"])
        self.assertEqual(len({task_resolve(s)["task_id"] for s in TASK_SET}), 8)

    def test_invalid_contract_fails_loudly(self):
        for t, c in (("group-work", CONTRACT), ("chairs-down", CONTRACT), ("make-pod", CONTRACT), ("chairs-up", "bad")):
            with self.assertRaises(ValueError):
                task_resolve(dict(task=t, contract=c))
            with self.assertRaises(ValueError):
                Config(task=t, contract=c)
        with self.assertRaises(ValueError):
            task_resolve({"typo": True})

    def test_instructions_reach_both_surfaces(self):
        env = fixture("seat-tutoring-group")
        self.assertEqual(instruction(env.config, env.spec), description("seat-tutoring-group", CONTRACT))
        self.assertIn("20-35 cm", instruction(Config(task="seat-tutoring-group"), env.spec))

    def test_existing_placements_and_mobile_base_clearance(self):
        # No new placement, target, prop or start generator is introduced. Check
        # all eight's actual seed-0 geometry, including the conservative Duo box.
        for selection in TASK_SET:
            s = L.setup(selection["task"], 0)
            items = dict(s["items"], __robot__=dict(kind="robot", pose=s["robot"]))
            self.assertEqual(L.conflicts(items), [], selection)
            for n, pose in s["targets"].items():
                corners = L._corners(s["items"][n]["kind"], pose)
                self.assertLess(float(abs(corners[:, 0]).max()), P.ROOM_X/2)
                self.assertGreater(float(corners[:, 1].min()), 0.)
                self.assertLess(float(corners[:, 1].max()), P.ROOM_Y)
        aisle = (L.COLS_X[2] - P.DESK_TOP[0]/2) - (L.COLS_X[1] + P.DESK_TOP[0]/2)
        self.assertGreater(aisle, 2*L.HALF["robot"][1])


class EvaluationTests(unittest.TestCase):
    def test_untouched_zero_across_seeds(self):
        for seed in (0, 1, 7):
            for t in SET_TASKS:
                with self.subTest(task=t, seed=seed):
                    env = fixture(t, seed, "v1" if t == "group-work" else CONTRACT)
                    for complete in (False, True):
                        env.complete = complete
                        ev = env.evaluator.evaluate()
                        self.assertFalse(ev["succeeded"])
                        self.assertEqual(ev["score"], 0.)
                        self.assertEqual(ev["progress"], 0.)
                        self.assertTrue(ev["criteria"]["rest_of_room"]["ok"])
                        if t != "group-work":
                            self.assertEqual(ev["contract"], CONTRACT)

    def test_finished_states_and_rest_window(self):
        for t in SET_TASKS:
            with self.subTest(task=t):
                env = fixture(t, contract="v1" if t == "group-work" else CONTRACT)
                finish(env)
                ev = env.evaluator.evaluate()
                self.assertTrue(ev["succeeded"], {n: c for n, c in ev["criteria"].items() if not c["ok"]})
                self.assertEqual(ev["score"], 1.)
                env.complete = False
                ev = env.evaluator.evaluate()
                self.assertFalse(ev["succeeded"])
                self.assertLessEqual(ev["score"], 0.9)
                if t != "group-work":
                    self.assertAlmostEqual(ev["score"], 0.9*ev["progress"])

    def test_identical_furniture_can_be_swapped_including_bystanders(self):
        for t in SET_TASKS[:-1]:
            with self.subTest(task=t):
                env = fixture(t)
                finish(env)
                chairs = [n for n, c in env.spec["check"].items() if n in env.spec["items"] and env.spec["items"][n]["kind"] == "chair"]
                env.world.swap(chairs[0], "chair_11")
                desks = [n for n, c in env.spec["check"].items() if c == "on_marks" and env.spec["items"][n]["kind"] == "desk"]
                if desks:
                    env.world.swap(desks[0], "desk_11")
                ev = env.evaluator.evaluate()
                self.assertTrue(ev["succeeded"], {n: c for n, c in ev["criteria"].items() if not c["ok"]})
                self.assertEqual(ev["score"], 1.)

    def test_partial_moves_are_continuous_and_reversible(self):
        for t, n in (("push-in-chairs", "chair_21"), ("straighten-row", "desk_31"),
                     ("exam-row", "desk_21"), ("seat-tutoring-group", "tchair_3"), ("clear-floor", "tote"),
                     ("move-teacher-desk", "teacher_desk")):
            with self.subTest(task=t):
                env = fixture(t)
                a = np.array(env.spec["items"][n]["pose"])
                b = np.array(env.spec["targets"].get(n, (1.1, 9., 0.)))
                values = []
                for alpha in (0., 0.25, 0.5, 0.50001, 0.75, 1.):
                    env.world.put(n, a + alpha*(b - a))
                    values.append(env.evaluator.evaluate()["score"])
                self.assertEqual(values[0], 0.)
                self.assertGreater(values[-1], 0.)
                self.assertTrue(all(a <= b + 1e-9 for a, b in zip(values, values[1:])), values)
                self.assertLess(abs(values[3] - values[2]), 0.001)
                env.world.put(n, a)
                self.assertEqual(env.evaluator.evaluate()["score"], 0.)

    def test_collateral_and_safety_costs(self):
        env = fixture("push-in-chairs")
        finish(env)
        env.complete = False
        full = env.evaluator.evaluate()
        p = np.array(env.world.pose2d("desk_11")[0])
        env.world.put("desk_11", p + (0.18, 0., 0.))
        ev = env.evaluator.evaluate()
        self.assertFalse(ev["succeeded"])
        self.assertLess(ev["score"], full["score"])
        self.assertGreater(ev["reward"]["rest_of_room_deduction"], 0.)
        env.world.put("desk_11", p)
        self.assertEqual(env.evaluator.evaluate()["score"], full["score"])
        env.world.put("chair_11", (2.65, 0.5, 0.))
        ev = env.evaluator.evaluate()
        self.assertEqual(ev["reward"]["safety_penalty"], 0.5)

    def test_close_but_wrong_states_fail(self):
        for t, n, offset in (("straighten-row", "desk_31", (0.045, 0, 0)),
                              ("exam-row", "desk_21", (0, 0, math.radians(4))),
                              ("push-in-chairs", "chair_21", (0, 0.25, 0)),
                              ("push-in-chairs", "desk_21", (0.055, 0, 0)),
                              ("move-teacher-desk", "teacher_chair", (0, 0, math.pi))):
            with self.subTest(task=t, item=n):
                env = fixture(t)
                finish(env)
                p = np.asarray(env.world.pose2d(n)[0])
                env.world.put(n, p + offset)
                self.assertFalse(env.evaluator.evaluate()["succeeded"])

    def test_chair_up_requires_top_support_and_side_and_one_per_desk(self):
        env = fixture("chairs-up")
        finish(env)
        env.world.put_up("chair_41", "desk_41", contact=False)
        self.assertFalse(env.evaluator.evaluate()["succeeded"])
        env.world.put_up("chair_41", "desk_41", rise=0.6, contact=False)
        ev = env.evaluator.evaluate()
        self.assertEqual(ev["criteria"]["chair_41"]["progress"], 0.)
        env.world.put_up("chair_41", "desk_42")
        self.assertFalse(env.evaluator.evaluate()["succeeded"])
        env.world.put("chair_41", (L.COLS_X[0], L.ROWS_Y[3], 0.), rotation=Rotation.from_euler("x", math.pi/2).as_matrix())
        self.assertFalse(env.evaluator.evaluate()["succeeded"])

    def test_chair_up_partial_credit_uses_unrounded_geometry(self):
        env = fixture("chairs-up")
        finish(env)
        env.complete = False
        p, R = env.world.body_pose("chair_41")
        p = p.copy()
        values = []
        for offset in (0.28, 0.28001):
            env.world.transforms["chair_41"] = (p + (offset, 0., 0.), R)
            env.world.sync()
            values.append(env.evaluator.evaluate()["score"])
        self.assertGreater(values[0], values[1])
        self.assertLess(values[0] - values[1], 1e-4)

    def test_route_and_floor_destinations(self):
        env = fixture("clear-floor")
        finish(env)
        env.world.floor.discard("tote")
        env.world.sync()
        self.assertFalse(env.evaluator.evaluate()["succeeded"])
        env.world.floor.add("tote")
        env.world.sync()
        with patch.object(env.evaluator, "route", return_value=False), patch.object(env.evaluator, "route_width", return_value=0.815):
            ev = env.evaluator.evaluate()
            self.assertFalse(ev["succeeded"])
            self.assertEqual(ev["reward"]["route_penalty"], 0.5)

    def test_tutoring_side_plan_and_exam_spacing(self):
        env = fixture("seat-tutoring-group")
        finish(env)
        env.world.put("tchair_3", (-2., 8.03, math.pi))  # a third chair on the room side
        self.assertFalse(env.evaluator.evaluate()["succeeded"])
        env = fixture("exam-row")
        finish(env)
        p = np.asarray(env.world.pose2d("chair_21")[0])
        env.world.put("chair_21", p + (0.06, 0., 0.))  # within seat lateral tolerance but spacing too small
        ev = env.evaluator.evaluate()
        self.assertTrue(ev["criteria"]["chair_21"]["ok"])
        self.assertFalse(ev["criteria"]["exam"]["ok"])

    def test_post_settle_baseline_and_state_only_replay(self):
        env = fixture("straighten-row")
        # Simulate a different settled start by editing transforms ONLY, before
        # constructing the evaluator. No simulation is executed.
        p = np.asarray(env.world.pose2d("desk_31")[0])
        env.world.put("desk_31", p + (0.002, -0.003, 0.001))
        env.evaluator = Evaluator(env)
        self.assertEqual(env.evaluator.evaluate()["score"], 0.)
        finish(env)
        expected = env.evaluator.evaluate()
        self.assertEqual(json.loads(json.dumps(expected))["score"], 1.)
        env.world.put("desk_31", p)
        finish(env)
        self.assertEqual(env.evaluator.evaluate(), expected)

    def test_group_work_dispatch_is_unchanged(self):
        env = fixture("group-work", contract="v1")
        motion, window = env.window_motion()
        self.assertEqual(env.evaluator.evaluate(), env.evaluator.by_place(motion, window))
        self.assertIsNone(reduction(0., 0.))
        self.assertEqual(reduction(0.25, 0.5), 0.5)


if __name__ == "__main__":
    unittest.main()
