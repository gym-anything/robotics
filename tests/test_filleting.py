"""Contract checks for filleting: catalogue, registration, robot selection, action preparation, restricted access.

The first classes need no GPU, no physics and no renderer (seconds). The physics class builds the real Newton +
tissue world for filleting/v1/trout/fillet-one-side/seed-0 (about 2-3 minutes) and is skipped unless
GAR_FILLETING_PHYSICS_TESTS=1 (the repository's convention for GPU tests, as GAR_LABEL_PHYSICS_TESTS). These are
interface checks, not physical qualification or a task solution.
"""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gym_anything_robotics import tasks
from gym_anything_robotics.errors import (ActionRejectedError, UnsupportedActionError,
                                          UnsupportedConfigurationError)
from tests.packages import PACKAGES

ROOT = PACKAGES
PRESETS = {"A": "trout", "B": "salmon-side", "C": "pork-tenderloin"}


class CatalogueTest(unittest.TestCase):
    def test_benchmark_identity_and_schema(self):
        raw = json.loads((ROOT/"filleting"/"benchmark.json").read_text(), parse_constant=lambda c: self.fail(c))
        self.assertEqual(tasks.catalog("filleting"), raw)
        self.assertEqual(raw["schema"], "gar.task-catalog.v1")
        self.assertEqual(raw["environment_id"], "filleting")
        self.assertEqual(raw["selectors"], ["preset", "task", "seed", "contract"])
        self.assertEqual(raw["resolver"], "filleting.task_set_catalogue:task_resolve")
        self.assertEqual(raw["splits"], [])
        self.assertIs(type(raw["evaluator"]["available"]), bool)
        self.assertTrue((ROOT/raw["evaluator"]["source"]).is_file())
        for key in ("description", "count_scope", "qualification", "split_note"):
            self.assertTrue(raw[key].strip(), key)
        self.assertEqual(tasks.resolve("filleting").task_id, raw["default_task_id"])
        self.assertEqual(raw["default_task_id"], "filleting/v1/trout/fillet-one-side/seed-0")
        for alias in ("fillet", "fillet-fish", "fish-filleting"):
            self.assertEqual(tasks.catalog(alias)["environment_id"], "filleting")

    def test_supported_tasks_resolve_and_round_trip(self):
        """Of the 13 tasks, the 11 this build supports resolve and round-trip, each with an evaluator."""
        from filleting.task import INSTRUCTIONS, PRESET_OF, TIME_LIMIT_S, Evaluator
        from filleting.variants import TASK_SLUG, UNSUPPORTED
        self.assertEqual(len(INSTRUCTIONS), 13)
        self.assertEqual(set(UNSUPPORTED), {"B1", "B4"})
        supported = {c: i for c, i in INSTRUCTIONS.items() if c not in UNSUPPORTED}
        self.assertEqual(tasks.catalog("filleting")["count"], len(supported) + 1)  # and put-away-knife
        self.assertEqual(set(TASK_SLUG), set(INSTRUCTIONS))
        self.assertEqual(set(TIME_LIMIT_S), set(INSTRUCTIONS))
        for code in supported:
            self.assertTrue(hasattr(Evaluator, f"_eval_{code}"), code)
        ids = set()
        for code, instruction in supported.items():
            preset = PRESET_OF[code]
            for selection in ({"task": TASK_SLUG[code], "seed": 7}, {"preset": PRESETS[preset], "task": code, "seed": 7},
                              {"preset": preset, "task": TASK_SLUG[code], "seed": 7}):
                with self.subTest(selection=selection):
                    spec = tasks.resolve("filleting", selection)
                    self.assertEqual(spec.task_id, f"filleting/v1/{PRESETS[preset]}/{TASK_SLUG[code]}/seed-7")
                    self.assertEqual(spec.launch.environment_id, "filleting")
                    self.assertEqual(spec.launch.backend_options, {"preset": preset, "task": code, "seed": 7})
                    self.assertEqual(spec.description, instruction)
                    self.assertIsNone(spec.split)
                    self.assertTrue(spec.evaluator["available"])
                    again = tasks.TaskSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
                    self.assertEqual(again.to_dict(), spec.to_dict())
                    ids.add(spec.task_id)
        self.assertEqual(len(ids), 11)

    def test_unsupported_tasks_rejected_with_their_reason(self):
        from filleting.variants import UNSUPPORTED, TASK_SLUG
        for code, reason in UNSUPPORTED.items():
            with self.subTest(task=code), self.assertRaises(ValueError) as cm:
                tasks.resolve("filleting", {"task": TASK_SLUG[code]})
            self.assertIn("not supported", str(cm.exception))
            self.assertIn(reason[:30], str(cm.exception))

    def test_bad_selections_rejected(self):
        for selection in ({"task": "debone-chicken"}, {"preset": "tuna", "task": "skin"},
                          {"preset": "trout", "task": "skin"},
                          {"seed": -1}, {"seed": 1000}, {"extra": 1}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                tasks.resolve("filleting", selection)
        from gym_anything_robotics.launch import LaunchSpec
        with self.assertRaises(ValueError):          # a task recipe never overwrites an explicit launch value
            tasks.resolve("filleting", {"task": "skin"}, base=LaunchSpec("filleting", backend_options={"seed": 3}))

    def test_resolver_rejects_non_integer_seeds(self):
        """1.5, True and "3" are rejected, not coerced (as label_application's resolver does)."""
        for seed in (1.5, True, "3"):
            with self.assertRaises(ValueError):
                tasks.resolve("filleting", {"seed": seed})

    def test_resolver_rejects_unknown_preset_with_value_error(self):
        """An unknown preset is a ValueError from the resolver itself."""
        with self.assertRaises(ValueError):
            tasks.resolve("filleting", {"preset": "tuna"})


class RegistrationTest(unittest.TestCase):
    def test_registered_with_aliases_and_descriptor(self):
        from gym_anything_robotics import registry
        from gym_anything_robotics.launch import describe
        from gym_anything_robotics.backends.newton_filleting import NewtonFilletingBackend
        from gym_anything_robotics.backends.base import RoboticsBackend
        self.assertTrue(issubclass(NewtonFilletingBackend, RoboticsBackend))
        self.assertIn(("filleting", "newton"), registry._REGISTRY)
        for alias in ("filleting", "fillet", "fillet-fish", "fish-filleting"):
            d = describe(alias)
            self.assertEqual((d["environment_id"], d["default_robot"], d["implementation_package"],
                              d["engine_version"]), ("filleting", "dual-fr3", "filleting", "1.5"))
            self.assertEqual([r["id"] for r in d["robots"]], ["dual-fr3", "dual-panda", "single-fr3"])

    def test_create_passes_the_catalog_default_and_task_options(self):
        from gym_anything_robotics import registry
        factory = Mock()
        spec = tasks.resolve("filleting", {"task": "fillet-one-side", "seed": 0})
        with patch.dict(registry._REGISTRY, {("filleting", "newton"): factory}), \
                patch.object(registry, "RoboticsEnv"):
            spec.create()
            registry.create("fillet", cameras=())
        self.assertEqual(factory.call_args_list[0].kwargs, {"robot": "dual-fr3", "preset": "A", "task": "A2", "seed": 0})
        self.assertEqual(factory.call_args_list[1].kwargs, {"robot": "dual-fr3", "cameras": ()})

    def test_unsupported_robots_rejected_with_catalog_reasons(self):
        from robots.catalog import ENVIRONMENTS
        from gym_anything_robotics.launch import LaunchSpec
        from gym_anything_robotics.backends.newton_filleting import NewtonFilletingBackend
        reasons = ENVIRONMENTS["filleting"]["unsupported"]
        self.assertEqual(set(reasons), {"g1"})
        for robot, reason in reasons.items():
            with self.subTest(robot=robot):
                with self.assertRaises(ValueError) as caught:
                    LaunchSpec("filleting", robot=robot)
                self.assertIn(f"does not support robot {robot!r}", str(caught.exception))
                self.assertIn(reason, str(caught.exception))
                with patch("filleting.env.FilletingEnv") as env, self.assertRaises(UnsupportedConfigurationError) as caught:
                    NewtonFilletingBackend(robot=robot)
                self.assertIn(reason, str(caught.exception))
                env.assert_not_called()
        for robot in ({"instances": {}}, "yam"):
            with self.subTest(robot=robot), self.assertRaises(UnsupportedConfigurationError):
                NewtonFilletingBackend(robot=robot)

    def test_dual_panda_is_a_qualified_deployment(self):
        from gym_anything_robotics.backends.newton_filleting import _check_robot
        self.assertEqual(_check_robot("dual-panda"), "dual-panda")
        self.assertEqual(_check_robot("dual-fr3"), "dual-fr3")
        self.assertEqual(_check_robot("single-fr3"), "single-fr3")

    def test_setup_objects_of_the_three_rigs_accepted_others_refused(self):
        # the rigs as gar.robot-setup.v1 objects, rig ids and rig paths select their alias
        from robots.catalog import resolve, select_for_backend
        from gym_anything_robotics.launch import LaunchSpec
        for alias, rig in (("dual-fr3", "filleting-fr3"), ("dual-panda", "filleting-panda"),
                           ("single-fr3", "filleting-fr3-single")):
            for sel in (rig, resolve(rig).to_dict(), str(PACKAGES / "robots" / "rigs" / f"{rig}.json")):
                with self.subTest(sel=str(sel)[:40]):
                    self.assertEqual(select_for_backend("filleting", sel), alias)
            LaunchSpec("filleting", robot=resolve(rig).to_dict())
        with self.assertRaises(ValueError) as caught:
            select_for_backend("filleting", resolve("scan-fr3").to_dict())
        self.assertIn("hosts its three catalog rigs", str(caught.exception))


    def test_collision_profile_is_a_deployment_option(self):
        from gym_anything_robotics.backends.newton_filleting import NewtonFilletingBackend
        from filleting.world import Config
        self.assertEqual(Config().collision_profile, "default")
        with patch("filleting.env.FilletingEnv") as env, patch("filleting.world.World"):
            try:
                NewtonFilletingBackend(task="medallions", config={"collision_profile": "contact"})
            except Exception:
                pass                      # the mocked environment may not satisfy the rest of the constructor
            self.assertEqual(env.call_args[0][0].collision_profile, "contact")

    def test_cutting_tasks_default_to_the_contact_profile(self):
        # a cutting task defaults to the contact profile; take-knife keeps the default, an explicit value is kept
        from gym_anything_robotics.backends.newton_filleting import NewtonFilletingBackend
        for task, config, want in (("medallions", {}, "contact"), ("fillet-one-side", {}, "contact"),
                                   ("take-knife", {}, "default"),
                                   ("medallions", {"collision_profile": "default"}, "default")):
            with self.subTest(task=task, config=config), patch("filleting.env.FilletingEnv") as env, \
                    patch("filleting.world.World"):
                try:
                    NewtonFilletingBackend(task=task, config=dict(config))
                except Exception:
                    pass
                self.assertEqual(env.call_args[0][0].collision_profile, want)

    def test_dual_panda_only_for_take_knife(self):
        # the Panda is qualified for holding and resting only; cutting tasks are refused before physics
        from gym_anything_robotics.backends.newton_filleting import NewtonFilletingBackend
        for task in ("fillet-one-side", "portion", "medallions"):
            with self.subTest(task=task), patch("filleting.env.FilletingEnv") as env, \
                    self.assertRaises(UnsupportedConfigurationError) as caught:
                NewtonFilletingBackend(robot="dual-panda", task=task)
            self.assertIn("dual-panda is offered for take-knife only", str(caught.exception))
            env.assert_not_called()

    def test_crumb_control_flood_fill_matches_full_labelling(self):
        # the crumb rule's local flood fill finds the same small fragments as labelling the whole mesh, on random cuts
        # of a lattice block
        import types
        import numpy as np
        from filleting import mesher, topology as TP
        X, tets = mesher.mesh_sdf(lambda p: (np.abs(p) - np.array([.012, .008, .005])).max(1),
                                  np.array([-.013, -.009, -.006]), np.array([.013, .009, .006]), 2.5e-3)
        nt = len(tets)
        ts = types.SimpleNamespace(hactive=np.ones(nt, np.int32), htets=tets, nt=nt,
                                   hV0=np.random.default_rng(0).uniform(.5, 2, nt)*1e-9)
        cr = TP.CrackRecorder.__new__(TP.CrackRecorder)
        cr.ts, cr.orig, cr._nbr, cr.min_fragment_m3 = ts, tets.copy(), None, 30e-9
        cr._adjacency()
        faces = [(a, int(b)) for a in range(nt) for b in cr._nbr[a] if b > a]
        rng = np.random.default_rng(1)
        vol = np.abs(ts.hV0)
        for _ in range(10):
            cr.cut_faces = {faces[i] for i in rng.choice(len(faces), rng.integers(0, len(faces)//2), replace=False)}
            pool = [f for f in faces if f not in cr.cut_faces]
            new = {pool[i]: None for i in rng.choice(len(pool), min(len(pool), 40), replace=False)}
            lab = cr._fragments(new.keys())
            fv = np.bincount(lab[lab >= 0], weights=vol[lab >= 0], minlength=nt)
            touched = {lab[a] for a, b in new} | {lab[b] for a, b in new}
            full = sorted(sorted(np.nonzero(lab == c)[0].tolist()) for c in touched if c >= 0 and fv[c] < 30e-9)
            local = {}
            for t, c in cr._small_fragments(new, vol).items():
                local.setdefault(c, []).append(t)
            self.assertEqual(full, sorted(sorted(v) for v in local.values()))

    def test_single_fr3_only_for_one_hand_tasks(self):
        from gym_anything_robotics.backends.newton_filleting import NewtonFilletingBackend
        from robots.catalog import for_environment
        self.assertEqual(list(for_environment("filleting", "single-fr3").to_dict()["instances"]), ["right"])
        for task in ("head-off", "fillet-one-side", "remove-belly-bones", "trim-belly", "portion", "remove-chain",
                     "denude", "medallions"):
            with self.subTest(task=task), patch("filleting.env.FilletingEnv") as env, \
                    self.assertRaises(UnsupportedConfigurationError) as caught:
                NewtonFilletingBackend(robot="single-fr3", task=task)
            self.assertIn("needs a second hand", str(caught.exception))
            env.assert_not_called()

    def test_invalid_configuration_fails_before_the_world_is_built(self):
        from gym_anything_robotics.backends.newton_filleting import NewtonFilletingBackend
        bad = (dict(cameras=("hero",)), dict(cameras=("overhead", "overhead")), dict(config={"task": "A1"}),
               dict(config={"device": "cpu"}, device="cuda:0"), dict(renderer_options={"exposure": 2}),
               dict(seed=True), dict(seed=1.5), dict(task="debone-chicken"), dict(preset="A", task="skin"),
               dict(preset="tuna"),
               dict(config={"blade_magic": 1}))
        with patch("filleting.env.FilletingEnv") as env:
            for options in bad:
                with self.subTest(options=options), self.assertRaises((ValueError, TypeError,
                                                                        UnsupportedConfigurationError)):
                    NewtonFilletingBackend(output_dir="/nonexistent/never-created", **options)
            env.assert_not_called()


def _stub_backend(task="A2"):
    """A backend over a stub world (joint table, kinematics) to check action preparation without physics."""
    from gym_anything_robotics.backends.newton_filleting import NewtonFilletingBackend
    from filleting.world import Config
    joints = {}
    for arm in ("left", "right"):
        for k in range(1, 8):
            joints[f"{arm}/fr3_joint{k}"] = dict(q=0, dof=0, lower=-2.5, upper=2.5, effort_limit=87.,
                                                 velocity_limit=2., native_label="x")
        for k in (1, 2):
            joints[f"{arm}/fr3_finger_joint{k}"] = dict(q=0, dof=0, lower=0., upper=.04, effort_limit=35.,
                                                        velocity_limit=.2, native_label="x")
    import numpy as np
    kin = SimpleNamespace(forward=lambda arm: np.eye(4), solve=Mock(side_effect=ValueError(
        "IK unreachable (812.00 mm, 0.00 deg); no action executed")))
    world = SimpleNamespace(joints=joints, fault=None, kin=kin, set_grip_force=Mock(),
                            gripper_joints=lambda arm: [f"{arm}/fr3_finger_joint1", f"{arm}/fr3_finger_joint2"])
    b = object.__new__(NewtonFilletingBackend)
    b.config = Config(task=task, robot="dual-fr3")
    b.env = SimpleNamespace(world=world, joints=joints, lock=__import__("threading").RLock(), act=Mock(), steps=0)
    b.arms, b.cameras, b.camera_specs = ("left", "right"), (), {}
    b.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta"})
    b.finger_effort_limit, b._prestart, b._reapers, b.last_timing = 35., None, [], {}
    return b


class ActionPreparationTest(unittest.TestCase):
    def test_rejections_execute_nothing(self):
        from gym_anything_robotics.actions import (CartesianDelta, CartesianPose, Gripper, MotionSafety,
                                                   NamedJointTarget)
        b = _stub_backend()
        rejected = (Gripper("left", half_width_m=.05), Gripper("left", half_width_m=.02, duration_s=.0005),
                    Gripper("left", inward_effort_n=80.), NamedJointTarget({"left/fr3_joint1": 3.}, 5),
                    NamedJointTarget({"knife_free": 0.}, 5), CartesianPose("left", (5., 5., 5.)),
                    CartesianDelta("right", (0., 0., .01), gripper_half_target_m=.05))
        for action in rejected:
            with self.subTest(action=action), self.assertRaises(ActionRejectedError) as caught:
                b.act(action)
            self.assertIn("no action executed", str(caught.exception))
        with self.assertRaises(UnsupportedActionError):
            b.act(CartesianDelta("left", (0., 0., .01), safety=MotionSafety(stop_on_contact=True)))
        b.env.act.assert_not_called()
        b.env.world.set_grip_force.assert_not_called()

    def test_gripper_semantics(self):
        from gym_anything_robotics.actions import Gripper
        b = _stub_backend()
        targets, ticks, grip = b._prepare(Gripper("right", inward_effort_n=70., duration_s=.05))
        self.assertEqual((ticks, grip), (50, ("right", 35.)))
        self.assertEqual(targets, {"right/fr3_finger_joint1": 0., "right/fr3_finger_joint2": 0.})
        targets, ticks, grip = b._prepare(Gripper("left", half_width_m=.02, duration_s=.1))
        self.assertEqual((ticks, grip), (100, ("left", 0.)))
        self.assertEqual(set(targets.values()), {.02})
        targets, _, grip = b._prepare(Gripper("left", inward_effort_n=0.))
        self.assertEqual((set(targets.values()), grip), ({.04}, ("left", 0.)))


class PolicyAccessTest(unittest.TestCase):
    def test_instruction_lookup_and_redaction(self):
        from filleting.task import INSTRUCTIONS
        for code in INSTRUCTIONS:
            access = _stub_backend(code).policy_access()
            self.assertEqual(access.instruction(), INSTRUCTIONS[code])
        caps = access.capabilities()
        self.assertEqual(caps["action_types"], ["cartesian_delta", "cartesian_pose", "gripper", "named_joint_target"])
        self.assertFalse(caps["privileged_state"])
        joints = caps["hardware"]["joints"]
        self.assertEqual(joints["left/fr3_finger_joint1"]["unit"], "metre")
        self.assertEqual(joints["right/fr3_joint4"]["unit"], "radian")
        self.assertFalse({"q", "dof", "native_label"} & set(joints["left/fr3_joint1"]))
        self.assertEqual(len(caps["proprioception_fields"]), 2*18)
        with self.assertRaises(Exception):
            access.decode_actions({"type": "cartesian_delta", "arm": "left", "delta_m": [0, 0, .01],
                                   "safety": {"stop_on_contact": True}})
        with self.assertRaises(Exception):
            access.decode_actions({"type": "keyboard", "keys": ["W"], "ticks": 1})


@unittest.skipUnless(os.environ.get("GAR_FILLETING_PHYSICS_TESTS") == "1",
                     "set GAR_FILLETING_PHYSICS_TESTS=1 (builds the Newton + tissue world, ~2-3 min, GPU)")
class PhysicsTest(unittest.TestCase):
    """The default task through the public constructor: reset, one gripper command, one small Cartesian move,
    one rejected unreachable target, evaluation, checkpoint restore. No RGB cameras (no Isaac worker)."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from gym_anything_robotics.launch import LaunchSpec
        cls.tmp = tempfile.TemporaryDirectory()
        cls.spec = tasks.resolve("filleting", {"preset": "trout", "task": "fillet-one-side", "seed": 0})
        launch = LaunchSpec("filleting", backend_options={**cls.spec.launch.backend_options, "cameras": [],
                                                          "output_dir": str(Path(cls.tmp.name)/"session")})
        from gym_anything_robotics import registry
        cls.env = registry.create(launch.environment_id, auto_reset=False, **launch.backend_options)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        cls.tmp.cleanup()

    def test_default_task_round_trip(self):
        from gym_anything_robotics.actions import CartesianDelta, CartesianPose, Gripper
        env = self.env
        self.assertEqual(self.spec.task_id, "filleting/v1/trout/fillet-one-side/seed-0")
        observation, info = env.reset()
        self.assertEqual(info["continuity"], "reset")
        state = env.get_state()
        self.assertEqual((state["tick"], state["config"]["task"], state["config"]["workpiece"]), (0, "A2", "trout"))
        self.assertTrue(state["ready"])
        world = env._backend.env.world
        self.assertLess(float(sum(v*v for v in state["knife"]["grasp_offset_m"]))**.5, .015)   # knife held
        finger = "left/fr3_finger_joint1"
        q0 = state["joints"][finger]["position"]

        step = env.act(Gripper("left", half_width_m=.02, duration_s=.05))
        self.assertEqual((step.info["requested_ticks"], step.info["executed_ticks"]), (50, 50))
        q1 = step.observation.proprioception["joints"][finger]["position"]
        self.assertAlmostEqual(step.observation.proprioception["joints"][finger]["target"], .02)
        self.assertLess(q1, q0 - 1e-3, f"left finger did not close: {q0:.4f} -> {q1:.4f} m")

        z0 = float(world.kin.forward("left")[2, 3])
        step = env.act(CartesianDelta("left", (0., 0., .01), duration_s=.1))
        self.assertEqual(step.info["executed_ticks"], 100)
        z1 = float(world.kin.forward("left")[2, 3])
        self.assertGreater(z1 - z0, 2e-3, f"left TCP rose only {1e3*(z1 - z0):.2f} mm")
        print(f"\n[filleting physics] finger {q0*1e3:.2f} -> {q1*1e3:.2f} mm in 50 ticks; left TCP +"
              f"{1e3*(z1 - z0):.2f} mm in 100 ticks; physics {step.info['physics_s']:.1f} s; knife offset "
              f"{1e3*float(sum(v*v for v in step.info['knife']['grasp_offset_m']))**.5:.2f} mm")

        tick = env.get_state()["tick"]
        with self.assertRaises(ActionRejectedError):
            env.act(CartesianPose("left", (3., 3., 3.)))
        self.assertEqual(env.get_state()["tick"], tick)

        result = env.evaluate()
        self.assertTrue(result["available"])
        self.assertIs(type(result["success"]), bool)
        self.assertTrue(math.isfinite(result["score"]))
        self.assertEqual(result["details"]["task"], "A2")
        print(f"[filleting physics] evaluation success={result['success']} score={result['score']:.3f} checks="
              f"{ {k: v['passed'] for k, v in result['details']['checks'].items()} }")

        saved = env.save_checkpoint(label="after-moves")
        self.assertEqual(saved["tick"], tick)
        env.act(Gripper("left", half_width_m=.04, duration_s=.01))
        self.assertEqual(env.get_state()["tick"], tick + 10)
        observation, info = env.restore_checkpoint("after-moves")
        self.assertEqual(info["continuity"], "restored_discontinuity")
        self.assertEqual(env.get_state()["tick"], tick)
        self.assertEqual(env.list_checkpoints(), ({"label": "after-moves", "tick": tick},))


if __name__ == "__main__":
    unittest.main()
