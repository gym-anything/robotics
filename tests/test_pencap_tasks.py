"""Catalogue, pose, force-buffer and lifecycle checks; NO simulator dependencies."""

from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

from pencap.newton15.config import Config
from pencap.newton15.task import DESCRIPTION, Evaluator as LegacyEvaluator, measure
from pencap.tasks import TASKS, EIGHT_TASKS, CONTRACTS, INSTRUCTIONS, task_resolve
from pencap.task_presets import start_poses, default_poses, fixture_boxes, reach_report
from pencap.task_scoring import SeatCapEvaluator, TaskEvaluator, contact_forces
from tests.packages import PACKAGES

ROOT = PACKAGES


class Array:
    def __init__(self, value):
        self.value = np.array(value, dtype=float)

    def numpy(self):
        return self.value.copy()

    def assign(self, value):
        self.value = np.array(value.numpy() if isinstance(value, Array) else value, dtype=float)


def contacts():
    return NS(rigid_contact_count=Array([0]), rigid_contact_shape0=Array([]),
              rigid_contact_shape1=Array([]), force=Array(np.zeros((0, 6))))


class PoseEnv:
    def __init__(self, task, config=None):
        self.config = config or Config(scene="studio")
        self.task_start = task
        self.tick = 0
        self.bodies = {"pen": 0, "cap": 1, "notebook": 2}
        self.state = NS(body_q=Array(list(start_poses(task, self.config).values())))
        self.builder = NS(shape_body=[0, 1, 2, 3, 4],
                          body_label=["pen", "cap", "notebook", "left/panda/panda_link4", "right/panda/panda_rightfinger"])
        self.touch = {}
        self.pipeline = NS(contacts=contacts)
        self.solver = NS(update_contacts=self.update_contacts)

    def update_contacts(self, solved):
        pairs = [(self.bodies[n], arm, f) for n, (arm, f) in self.touch.items()]
        solved.rigid_contact_count.assign([len(pairs)])
        solved.rigid_contact_shape0.assign([p[0] for p in pairs])
        solved.rigid_contact_shape1.assign([p[1] for p in pairs])
        solved.force.assign([[p[2], 0., 0., 0., 0., 0.] for p in pairs] if pairs else np.zeros((0, 6)))

    def poses(self, poses):
        self.state.body_q.assign([poses[n] for n in self.bodies])


def goal_poses(task, c):
    poses = default_poses(c)
    if task in ("pen-on-notebook", "pass-pen"):
        poses["pen"][:3] = [.33-c.pen_length/2, .24, poses["notebook"][2]+.007+.0065]
    elif task == "load-fixture":
        poses = start_poses("cap-in-fixture", c)
    elif task == "cap-in-fixture":
        poses = start_poses("unload-fixture", c)
    elif task in ("reseat-cap", "unload-fixture", "seat-cap"):
        poses = start_poses("uncap", c)
    return poses


def run_hold(env, evaluator, seconds=.51):
    result = None
    for _ in range(round(seconds/env.config.dt)):
        env.tick += 1
        result = evaluator.sample(env)
    return result


class CatalogueTests(unittest.TestCase):
    def setUp(self):
        # The test needs no robot assets: stub only the robot catalogue's discovery
        # metadata, never the task resolver being tested.
        catalog = NS(default_robot=lambda name: "dual-panda", environment_robots=lambda name: ["dual-panda"],
                     select_for_backend=lambda name, robot: robot, ENVIRONMENTS={})
        self.stub = patch.dict(sys.modules, {"robots": NS(), "robots.catalog": catalog})
        self.stub.start()
        self.addCleanup(self.stub.stop)

    def test_exact_selections_default_and_public_roundtrip(self):
        from gym_anything_robotics import resolve_task, LaunchSpec
        from gym_anything_robotics.tasks import TaskSpec
        b = json.loads((ROOT/"pencap/newton15/benchmark.json").read_text())
        selections = b["task_set"]["selections"]
        self.assertEqual(selections, [{"task": t} for t in EIGHT_TASKS])
        self.assertEqual(b["count"], 8)
        self.assertEqual([t["task_id"] for t in b["tasks"]], ["pencap/"+t for t in TASKS])
        self.assertEqual(INSTRUCTIONS["seat-cap"], DESCRIPTION)
        default = resolve_task("pencap")
        self.assertEqual(default.task_id, "pencap/seat-cap")
        self.assertEqual(default.selection, {})
        self.assertEqual(default.launch.backend_options, {})
        for selection in selections:
            task = selection["task"]
            resolved = resolve_task("pencap", selection)
            self.assertEqual(resolved.description, INSTRUCTIONS[task])
            self.assertEqual(resolved.task_id, "pencap/"+task)
            self.assertEqual(TaskSpec.from_dict(resolved.to_dict()), resolved)
            self.assertEqual(resolved.evaluator["contracts"][task], CONTRACTS[task])
        base = LaunchSpec("pencap", backend_options={"config": {"seed": 12}})
        self.assertEqual(resolve_task("pencap", base=base).launch.backend_options, {"config": {"seed": 12}})
        with self.assertRaisesRegex(ValueError, "conflict"):
            resolve_task("pencap", {"task": "uncap"}, base=LaunchSpec("pencap", backend_options={"task": "pass-pen"}))

    def test_bad_selectors_and_types_are_not_coerced(self):
        for selection in ({"task": ""}, {"task": None}, {"task": 2}, {"task": []}, {"task": "pencap/uncap"}, {"seed": 0}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                task_resolve(selection)


    def test_imports_do_not_load_simulator(self):
        import subprocess
        import sys
        code = """import sys
from types import SimpleNamespace as NS
sys.modules['robots'] = NS()
sys.modules['robots.catalog'] = NS(default_robot=lambda name: 'dual-panda', environment_robots=lambda name: ['dual-panda'], ENVIRONMENTS={})
from gym_anything_robotics import resolve_task
from pencap.tasks import EIGHT_TASKS
for t in EIGHT_TASKS: resolve_task('pencap', {'task': t})
assert not any(k in sys.modules for k in ('warp', 'newton', 'mujoco', 'mujoco_warp', 'pxr'))
"""
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


class ScoringTests(unittest.TestCase):
    def test_float32_native_start_poses_score_exactly_zero(self):
        for task in EIGHT_TASKS:
            with self.subTest(task=task):
                env = PoseEnv(task)
                env.state.body_q.value = env.state.body_q.value.astype(np.float32)
                evaluator = SeatCapEvaluator() if task == "seat-cap" else TaskEvaluator(task)
                self.assertEqual(evaluator.sample(env)["score"], 0.)
                result = run_hold(env, evaluator, .01)
                self.assertEqual(result["score"], 0.)
                json.dumps(result, allow_nan=False)

    def test_all_starts_remain_zero_untouched_and_goals_succeed(self):
        for task in EIGHT_TASKS:
            with self.subTest(task=task):
                env = PoseEnv(task)
                evaluator = SeatCapEvaluator() if task == "seat-cap" else TaskEvaluator(task)
                for result in (evaluator.sample(env), run_hold(env, evaluator)):
                    self.assertFalse(result["success"])
                    self.assertEqual(result["score"], 0.)
                    self.assertEqual(result["contract"], CONTRACTS[task])
                env.poses(goal_poses(task, env.config))
                early = run_hold(env, evaluator, .1)
                self.assertFalse(early["success"])
                self.assertGreater(early["score"], 0.)
                self.assertLessEqual(early["score"], .9)
                result = run_hold(env, evaluator)
                self.assertTrue(result["success"], result)
                self.assertEqual(result["score"], 1.)
                env.poses(start_poses(task, env.config))
                result = run_hold(env, evaluator, .01)
                self.assertFalse(result["success"])
                self.assertEqual(result["score"], 0.)

    def test_seat_cap_result_fields_unchanged_over_trajectory(self):
        env = PoseEnv("seat-cap")
        old, new = LegacyEvaluator(), SeatCapEvaluator()
        for task in ("seat-cap", "reseat-cap", "uncap", "seat-cap"):
            env.poses(start_poses(task, env.config))
            for _ in range(500):
                a, b = old.sample(env), new.sample(env)
                self.assertEqual(a, {k: v for k, v in b.items() if k not in ("contract", "task_id")})

    def test_reseat_snap_and_rest_verdict_matches_seat_cap(self):
        env = PoseEnv("reseat-cap")
        old, new = LegacyEvaluator(), TaskEvaluator("reseat-cap")
        self.assertEqual(old.sample(env)["success"], new.sample(env)["success"])
        for task in ("reseat-cap", "uncap", "seat-cap"):
            env.poses(start_poses(task, env.config))
            for _ in range(490):
                env.tick += 1
                self.assertEqual(old.sample(env)["success"], new.sample(env)["success"])

    def check_failure(self, task, change):
        env = PoseEnv(task)
        evaluator = TaskEvaluator(task)
        evaluator.sample(env)
        p = goal_poses(task, env.config)
        change(p, env)
        env.poses(p)
        result = run_hold(env, evaluator)
        self.assertFalse(result["success"], (task, result))
        self.assertGreaterEqual(result["score"], 0.)
        self.assertLessEqual(result["score"], .9)
        return result

    def test_held_goal_fails_for_every_set_down_task_and_either_arm(self):
        for task in ("pen-on-notebook", "pass-pen", "unload-fixture", "load-fixture", "uncap"):
            for arm in (3, 4):
                with self.subTest(task=task, arm=arm):
                    self.check_failure(task, lambda p, e: e.touch.update(pen=(arm, .051)))
        for task in ("unload-fixture", "uncap"):
            self.check_failure(task, lambda p, e: e.touch.update(cap=(4, .051)))
        env = PoseEnv("reseat-cap")
        evaluator = TaskEvaluator("reseat-cap")
        evaluator.sample(env)
        env.touch = {"pen": (3, 3.), "cap": (4, 3.)}
        env.poses(goal_poses("reseat-cap", env.config))
        self.assertTrue(run_hold(env, evaluator)["success"])

    def test_placement_support_orientation_and_notebook_are_checked(self):
        for task in ("pen-on-notebook", "pass-pen"):
            self.check_failure(task, lambda p, e: p["pen"].__setitem__(2, 1.))
            self.check_failure(task, lambda p, e: p["pen"].__setitem__(2, .74))
            self.check_failure(task, lambda p, e: p["pen"].__setitem__(0, .7))
            self.check_failure(task, lambda p, e: p["notebook"].__setitem__(slice(3, 7), Rotation.from_euler("x", 180, degrees=True).as_quat()))
            def lift_book(p, e):
                p["notebook"][2] += .10
                p["pen"][2] += .10
            self.check_failure(task, lift_book)

    def test_fixture_requires_real_bore_and_no_floor_penetration(self):
        for task in ("load-fixture", "cap-in-fixture"):
            self.check_failure(task, lambda p, e: p["pen"].__setitem__(0, 0.))
            self.check_failure(task, lambda p, e: p["pen"].__setitem__(2, .86))
            self.check_failure(task, lambda p, e: p["pen"].__setitem__(slice(3, 7), Rotation.from_euler("x", 90, degrees=True).as_quat()))
        self.check_failure("unload-fixture", lambda p, e: p.update(start_poses("unload-fixture", e.config)))
        self.check_failure("unload-fixture", lambda p, e: p.update(cap=default_poses(e.config)["cap"]))

    def test_uncap_requires_both_parts_down_and_clear_tip(self):
        self.check_failure("uncap", lambda p, e: p.update(start_poses("uncap", e.config)))
        self.check_failure("uncap", lambda p, e: p["cap"].__setitem__(2, 1.))
        self.check_failure("uncap", lambda p, e: p["pen"].__setitem__(2, .4))

    def test_collateral_blocks_success(self):
        for task in EIGHT_TASKS[:-1]:
            # The notebook is a named loose object in every new task.
            self.check_failure(task, lambda p, e: p["notebook"].__setitem__(2, .3))

    def test_stillness_and_release_window_are_real_time_not_evaluate_calls(self):
        env = PoseEnv("pen-on-notebook")
        evaluator = TaskEvaluator("pen-on-notebook")
        evaluator.sample(env)
        env.poses(goal_poses("pen-on-notebook", env.config))
        early = run_hold(env, evaluator, .10)
        for _ in range(1000):
            self.assertEqual(evaluator.sample(env), early)
        self.assertFalse(early["success"])
        env.touch = {"pen": (3, 1.)}
        run_hold(env, evaluator, .3)
        env.touch = {}
        self.assertFalse(run_hold(env, evaluator, .2)["success"])
        self.assertTrue(run_hold(env, evaluator, .31)["success"])
        for i in range(600):
            env.state.body_q.value[0, 0] += .00001
            result = run_hold(env, evaluator, env.config.dt)
        self.assertFalse(result["success"])

    def test_missing_preset_or_skipped_ticks_rejected(self):
        env = PoseEnv("uncap")
        env.task_start = "seat-cap"
        with self.assertRaisesRegex(ValueError, "preset"):
            TaskEvaluator("uncap").sample(env)
        env.task_start = "uncap"
        evaluator = TaskEvaluator("uncap")
        evaluator.sample(env)
        env.tick = 2
        with self.assertRaisesRegex(ValueError, "every physics tick"):
            evaluator.sample(env)

    def test_new_capping_progress_does_not_jump_at_nub_snap(self):
        for task in ("reseat-cap", "cap-in-fixture"):
            scores = []
            for depth in np.linspace(.03964, .03969, 20):
                env = PoseEnv(task)
                evaluator = TaskEvaluator(task)
                evaluator.sample(env)
                p = goal_poses(task, env.config)
                rp = Rotation.from_quat(p["pen"][3:])
                p["cap"][:3] = p["pen"][:3]+rp.apply([0., depth, 0.])
                env.poses(p)
                scores.append(run_hold(env, evaluator, env.config.dt)["score"])
            self.assertTrue(all(b >= a for a, b in zip(scores, scores[1:])))
            self.assertLess(max(np.diff(scores)), .01)

    def test_progress_improves_at_intermediate_work_states(self):
        for task in EIGHT_TASKS[:-1]:
            env = PoseEnv(task)
            evaluator = TaskEvaluator(task)
            evaluator.sample(env)
            p = start_poses(task, env.config)
            goal = goal_poses(task, env.config)
            # Construct a partial state, not a simulated trajectory.
            for name in ("pen", "cap"):
                p[name][:3] = .5*(p[name][:3]+goal[name][:3])
                p[name][3:] = goal[name][3:]
            if task == "cap-in-fixture":
                # Assembly shaping starts near the writing tip, as in seat-cap;
                # the halfway transport pose is still outside that neighbourhood.
                rp = Rotation.from_quat(p["pen"][3:])
                p["cap"][:3] = p["pen"][:3]+rp.apply([0., .020, 0.])
            env.poses(p)
            result = run_hold(env, evaluator, env.config.dt)
            self.assertGreater(result["score"], 0., (task, result))
            self.assertLessEqual(result["score"], .9)


class PresetTests(unittest.TestCase):
    def test_reach_screen_for_every_start_and_destination(self):
        report = reach_report(Config(scene="studio"))
        for task, points in report.items():
            for name, point in points.items():
                with self.subTest(task=task, point=name):
                    self.assertLess(min(v["shoulder_m"] for v in point["arms"].values()), .855)
        # A distance screen supports the design; it does not establish reach exclusivity by IK.
        self.assertGreater(report["pass-pen"]["pen_end"]["arms"]["left"]["shoulder_m"], .855)
        self.assertLess(report["load-fixture"]["fixture_mouth"]["arms"]["left"]["shoulder_m"], .70)

    def test_nest_clearance_start_snap_and_no_table_penetration(self):
        c = Config(scene="studio")
        boxes = fixture_boxes(c)
        self.assertEqual(len(boxes), 5)
        for _, position, half in boxes:
            self.assertGreaterEqual(position[2]-half[2], c.table_top-1e-12)
            self.assertLessEqual(position[2]+half[2], c.table_top+.070+1e-12)
        for task in ("uncap", "unload-fixture"):
            p = start_poses(task, c)
            self.assertTrue(measure(c, p["pen"], p["cap"])["seated"])
        p = start_poses("reseat-cap", c)
        self.assertTrue(measure(c, p["pen"], p["cap"])["engaged"])
        self.assertFalse(measure(c, p["pen"], p["cap"])["seated"])
        for task in EIGHT_TASKS:
            p = start_poses(task, c)
            for name in ("pen", "cap"):
                length = c.pen_length if name == "pen" else c.cap_length
                axis = Rotation.from_quat(p[name][3:]).as_matrix()[:, 1]
                ends = [p[name][:3], p[name][:3]+axis*length]
                radial_z = .0067*np.sqrt(max(0., 1-axis[2]**2))
                self.assertGreater(min(e[2] for e in ends)-radial_z, c.table_top)

    def test_solved_force_buffer_any_link_both_pair_directions(self):
        env = PoseEnv("load-fixture")
        env.tick = 1
        env.touch = {"pen": (3, .1), "cap": (4, .2)}
        self.assertEqual(contact_forces(env, contacts()), {"pen": .1, "cap": .2, "notebook": 0.})
        original = env.solver.update_contacts
        def reverse(buffer):
            original(buffer)
            buffer.rigid_contact_shape0, buffer.rigid_contact_shape1 = buffer.rigid_contact_shape1, buffer.rigid_contact_shape0
        env.solver.update_contacts = reverse
        self.assertEqual(contact_forces(env, contacts()), {"pen": .1, "cap": .2, "notebook": 0.})


class FakeState:
    def __init__(self, poses):
        self.body_q = Array(poses)
        self.body_qd = Array(np.zeros((len(poses), 6)))
        self.joint_q, self.joint_qd = Array([]), Array([])

    def assign(self, other):
        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            getattr(self, name).assign(getattr(other, name))


class FakeBuilder:
    def __init__(self, c):
        self.body_q = list(default_poses(c).values())
        self.body_label = ["pen", "cap", "notebook"]
        self.shape_body = [0, 1, 2]
        self.added = []

    def add_shape_box(self, body, **kw):
        self.added.append((body, kw))
        self.shape_body.append(body)
        return len(self.shape_body)-1


class LifecycleTests(unittest.TestCase):
    def test_adapter_applies_preset_before_finalize_and_restores_it_on_reset(self):
        from pencap.task_env import TaskPenCapEnv
        c = Config(scene="studio")
        wp = NS(ScopedDevice=lambda device: nullcontext(),
                transform=lambda p, q: np.r_[p, q], quat_identity=lambda: [0., 0., 0., 1.])
        newton = NS(ModelBuilder=NS(ShapeConfig=lambda **kw: kw))
        native = NS(reset_contact_matching=lambda: None)
        def build(config):
            return FakeBuilder(config), {"pen": 0, "cap": 1, "notebook": 2}, {}, {0: "pen", 1: "cap", 2: "paper"}
        def solver():
            return NS(mj_model=NS(opt=NS(integrator=2)), update_contacts=lambda buffer: None)
        def create(builder, config, joints):
            model = NS(state=lambda: FakeState(builder.body_q))
            return (model, NS(contacts=contacts, native=native), solver(), model.state(), model.state(),
                    NS(joint_target_q=Array([])), contacts())
        with patch("pencap.task_env.require_engine", return_value=(newton, wp)), \
             patch("pencap.task_env.build", side_effect=build), \
             patch("pencap.task_env.create", side_effect=create), \
             patch("pencap.newton15.env.make_solver", side_effect=lambda *args: solver()):
            for task in EIGHT_TASKS:
                with self.subTest(task=task):
                    env = TaskPenCapEnv(c, task)
                    expected = np.array(list(start_poses(task, c).values()))
                    np.testing.assert_allclose(env.state.body_q.numpy(), expected, atol=1e-12)
                    self.assertEqual(env.evaluation["score"], 0.)
                    self.assertEqual(len(env.builder.added), 5 if "fixture" in task else 0)
                    for body, kw in env.builder.added:
                        self.assertEqual(body, -1)
                        self.assertEqual(kw["cfg"], dict(gap=c.contact_gap, mu=.35, ke=1e7, kd=5000., mu_torsional=0., mu_rolling=0.))
                    self.assertEqual({i: env.materials[i] for i in range(3)}, {0: "pen", 1: "cap", 2: "paper"})
                    env.state.body_q.assign(list(goal_poses(task, c).values()))
                    old_evaluator = env.evaluator
                    env.tick = 27
                    result = env.reset()
                    np.testing.assert_array_equal(env.state.body_q.numpy(), expected)
                    self.assertEqual(result["evaluation"]["score"], 0.)
                    self.assertEqual(result["evaluation"]["contract"], CONTRACTS[task])
                    self.assertEqual((env.tick, env.generation), (0, 1))
                    self.assertIsNot(env.evaluator, old_evaluator)

    def test_backend_preserves_original_config_handling_and_policy_instruction(self):
        from pencap.task_backend import NewtonPenCapTaskBackend
        from gym_anything_robotics.backends.newton_pencap import NewtonPenCapBackend
        from gym_anything_robotics import registry
        cases = [dict(config={}, cameras=()), dict(config=None, cameras=[]),
                 dict(config={"seed": True}), dict(config={"dt": 1/480}),
                 dict(config={"scene": "baseline"}), dict(cameras=("bench_rgb", "bench_rgb")),
                 dict(robot=None), dict(config={"unknown": 1}),
                 dict(device="cpu", config=Config(scene="studio"), cameras=())]
        def fake_env(config, task="seat-cap"):
            return NS(config=config, task=task, joints={})
        with patch("pencap.newton15.env.PenCapEnv", side_effect=fake_env), \
             patch("pencap.task_env.TaskPenCapEnv", side_effect=fake_env), \
             patch("pencap.newton15.catalog.Catalog", return_value=NS()), \
             patch("pencap.newton15.kinematics.Kinematics", return_value=NS()), \
             tempfile.TemporaryDirectory() as tmp:
            for n, options in enumerate(cases):
                outcomes = []
                for i, backend in enumerate((NewtonPenCapBackend, NewtonPenCapTaskBackend)):
                    try:
                        obj = backend(**options, recording=False, output_dir=Path(tmp)/f"{n}-{i}")
                        outcomes.append((obj.env.config, obj.capabilities))
                    except (TypeError, ValueError, RuntimeError) as exc:
                        outcomes.append((type(exc), str(exc)))
                self.assertEqual(*outcomes, options)
            for task in EIGHT_TASKS:
                obj = registry._REGISTRY[("pencap", "newton")](task=task, cameras=(), recording=False,
                                                               output_dir=Path(tmp)/task)
                self.assertEqual(obj.policy_access().instruction(), INSTRUCTIONS[task])
                self.assertEqual(obj.env.task, task)


if __name__ == "__main__":
    unittest.main()
