"""Contract checks for sewing's four robot deployments (dual-fr3, dual-panda, single-fr3, g1) through the real backend
constructor over a stub world shaped like each deployment's (sewing/world.py): the arms a command may name, the action
kinds (a gripper only where there is a Franka Hand), the policy cameras and the refusal of the diagnostic ones, the
machine's control box on any arm, a command to an arm the rig does not have, selection by alias, rig id, rig path and
setup object, and the environment's doctor. No GPU, no physics, no renderer (seconds).
"""

import tempfile
import threading
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from gym_anything_robotics.errors import UnsupportedActionError, UnsupportedConfigurationError
from tests.packages import PACKAGES

ROOT = PACKAGES
RIGS = {"dual-fr3": "sewing-fr3", "dual-panda": "sewing-panda", "single-fr3": "sewing-fr3-single", "g1": "sewing-g1"}


def _stub_world(robot):
    """The parts of sewing.world.World the backend's constructor reads, as each deployment builds them."""
    from robots.catalog import for_environment
    data = for_environment("sewing", robot).to_dict()
    g1 = robot == "g1"
    instances = dict.fromkeys(data["instances"])
    finger = "panda_finger_joint" if robot == "dual-panda" else "fr3_finger_joint"
    hand_arms = ("left", "right") if g1 else tuple(a for a in instances)
    joints = {f"{a}/{finger}{k}": dict(q=0, dof=0, lower=0., upper=.04, effort_limit=35., velocity_limit=.2)
              for a in ([] if g1 else hand_arms) for k in (1, 2)}
    # the arms' joints with a MuJoCo dof (world.py builds this for the collision reflex; the G1 has none)
    arm_dofs = {} if g1 else {a: [(f"{a}/joint{k}", 7*i + k) for k in range(1, 8)] for i, a in enumerate(instances)}
    return SimpleNamespace(rig=SimpleNamespace(instances=instances), hand_arms=hand_arms, joints=joints, g1=g1,
                           _arm_mj_dofs=arm_dofs,
                           gripper_joints=lambda arm: [] if g1 else [f"{arm}/{finger}1", f"{arm}/{finger}2"])


def _backend(robot, cameras=None):
    from gym_anything_robotics.backends.newton_sewing import NewtonSewingBackend
    selected = robot if isinstance(robot, str) and robot in RIGS else None
    world = _stub_world(selected or "dual-fr3")
    env = SimpleNamespace(world=world, joints=world.joints, start_state="under_foot", lock=threading.RLock(),
                          act=Mock(), steps=0)
    out = Path(tempfile.mkdtemp())/"session"
    with patch("sewing.task_set_env.make_environment", return_value=env):
        return NewtonSewingBackend(robot=robot, task="plain-seam", cameras=cameras, output_dir=str(out))


class DeploymentContractTest(unittest.TestCase):
    EXPECTED = {  # arms a command may name, gripper offered, policy cameras
        "dual-fr3": (("left", "right"), True, ("operator", "overhead", "needle")),
        "dual-panda": (("left", "right"), True, ("operator", "overhead", "needle")),
        "single-fr3": (("right",), True, ("operator", "overhead", "needle")),
        "g1": (("left", "right", "left+waist", "right+waist"), False, ("head_rgb", "overhead", "needle")),
    }

    def test_arms_kinds_and_cameras_per_deployment(self):
        for robot, (arms, gripper, cams) in self.EXPECTED.items():
            with self.subTest(robot=robot):
                b = _backend(robot)
                self.assertEqual(b.arms, arms)
                self.assertEqual("gripper" in b.kinds, gripper)
                self.assertTrue({"named_joint_target", "cartesian_pose", "cartesian_delta", "tool_io"} <= b.kinds)
                self.assertEqual(b.policy_cameras, cams)
                self.assertEqual(b.cameras, cams)              # every policy camera in the default observation
                pa = b.policy_access()
                self.assertEqual([c.name for c in pa.cameras], list(cams))
                self.assertEqual(pa.capabilities()["hardware"]["arms"], list(arms))
                if robot == "g1":
                    self.assertIn("Dex3-1 hands", b.capabilities.notes["gripper"])
                    self.assertIn("no balance controller", b.capabilities.notes["control"])
                else:
                    self.assertIn("Franka Hand per arm", b.capabilities.notes["gripper"])
                    self.assertIn("libfranka", b.capabilities.notes["control"])

    def test_the_arms_external_torque_estimate_is_proprioception_and_the_g1_has_none(self):
        for robot, (arms, _, _) in self.EXPECTED.items():
            with self.subTest(robot=robot):
                b = _backend(robot)
                fields = b.policy_access().capabilities()["proprioception_fields"]
                tau = [f for f in fields if f.endswith(".external_torque")]
                if robot == "g1":
                    self.assertEqual(tau, [])
                    self.assertIn("No force or tactile channel", b.capabilities.notes["force"])
                else:
                    self.assertEqual(len(tau), 7*len(arms))
                    self.assertIn("tau_ext_hat", b.capabilities.notes["force"])

    def test_diagnostic_cameras_are_refused_to_the_policy(self):
        from gym_anything_robotics.backends.newton_sewing import DIAGNOSTIC
        for robot in self.EXPECTED:
            for cam in DIAGNOSTIC:
                with self.subTest(robot=robot, camera=cam), self.assertRaises(ValueError) as caught:
                    _backend(robot, cameras=(cam,))
                self.assertIn("privileged diagnostics", str(caught.exception))
        with self.assertRaises(ValueError):                                 # the G1's head camera is the G1's only
            _backend("dual-fr3", cameras=("head_rgb",))
        with self.assertRaises(ValueError):                                 # the arms' operator camera is not the G1's
            _backend("g1", cameras=("operator",))
        self.assertEqual(_backend("g1", cameras=("head_rgb", "needle")).cameras, ("head_rgb", "needle"))

    def test_an_arm_the_rig_lacks_is_refused_before_any_tick(self):
        from gym_anything_robotics.actions import ToolIO
        b = _backend("single-fr3")
        with self.assertRaises(UnsupportedActionError):
            b._arm("left")
        with self.assertRaises(ValueError) as caught:                       # the server answers 422, nothing runs
            b._machine_outputs(ToolIO(arm="left", outputs={"speed_spm": 600.0}, ticks=10))
        self.assertIn("('right',)", str(caught.exception))
        b.env.act.assert_not_called()

    def test_a_command_is_capped_to_fit_the_declared_client_timeout(self):
        from gym_anything_robotics.actions import ToolIO
        from gym_anything_robotics.backends.newton_sewing import CLIENT_TIMEOUT_S, MAX_COMMAND_S
        b = _backend("dual-fr3")
        caps = b.policy_access().capabilities()["hardware"]
        self.assertEqual((caps["client_timeout_s"], caps["max_command_s"]), (CLIENT_TIMEOUT_S, MAX_COMMAND_S))
        # the slowest rate measured on the shared host (sewing/REPORT.pdf §14) keeps the longest command inside the
        # timeout
        self.assertLess(MAX_COMMAND_S*395, CLIENT_TIMEOUT_S)
        ticks = int(round(MAX_COMMAND_S/b.config.dt))
        self.assertEqual(b._prepare(ToolIO(arm="left", outputs={"speed_spm": 600.0}, ticks=ticks))[1], ticks)
        with self.assertRaises(ValueError):
            b._prepare(ToolIO(arm="left", outputs={"speed_spm": 600.0}, ticks=ticks + 1))
        b.env.act.assert_not_called()

    def test_selection_by_alias_rig_id_path_and_setup_object(self):
        from robots.catalog import resolve
        for alias, rig in RIGS.items():
            for sel in (alias, rig, str(ROOT/"robots"/"rigs"/f"{rig}.json"), resolve(rig).to_dict()):
                with self.subTest(sel=str(sel)[:50]):
                    from robots.catalog import select_for_backend
                    self.assertEqual(select_for_backend("sewing", sel), alias)

    def test_a_refused_reset_is_a_refusal_not_a_fault(self):
        # the GAR server latches a runtime fault for any ValueError from reset, after which the session refuses every
        # command until the owner recovers it; a refused selection change must be an ActionRejectedError, which it
        # passes through with the session still ready
        from gym_anything_robotics.errors import ActionRejectedError
        import inspect
        from gym_anything_robotics import http_server
        src = inspect.getsource(http_server.GARHTTPService.reset)
        self.assertIn("except (ActionRejectedError, UnsupportedActionError)", src)
        b = _backend("dual-fr3")
        for kwargs in (dict(seed=3), dict(options={"task": "pivot"})):
            with self.subTest(**kwargs):
                with self.assertRaises(ActionRejectedError):
                    b.reset(**kwargs)
        b.env.act.assert_not_called()

    def test_other_environments_rigs_and_unknown_robots_are_refused(self):
        from gym_anything_robotics.backends.newton_sewing import _check_robot
        from robots.catalog import resolve
        for robot in ("ur10e", "welding-g1", resolve("filleting-panda").to_dict()):
            with self.subTest(robot=str(robot)[:30]), self.assertRaises(UnsupportedConfigurationError):
                _check_robot(robot)

    def test_a_rig_with_a_sewing_rigs_structure_is_accepted_whatever_its_name(self):
        # a copy of a rig under another id or description is accepted
        import copy
        from sewing.robot_setup import validate_setup
        from robots.catalog import resolve
        for alias, rig in RIGS.items():
            d = resolve(rig).to_dict()
            renamed = dict(copy.deepcopy(d), id=f"my-{rig}", description="a copy of the sewing rig under my own name")
            with self.subTest(alias=alias):
                self.assertEqual(validate_setup(renamed), alias)
                from gym_anything_robotics.backends.newton_sewing import _check_robot
                self.assertEqual(_check_robot(renamed), alias)         # the backend's own route
        # the default rig with its models changed to Panda is the dual-Panda rig's structure
        d = copy.deepcopy(resolve("sewing-fr3").to_dict())
        for inst in d["instances"].values():
            inst["model"] = inst["model"].replace("fr3", "panda")
        d["id"] = "fr3-pedestal-with-pandas"
        self.assertEqual(validate_setup(d), "dual-panda")
        # a structure no rig has is refused: the left arm alone, a single Panda
        lone = copy.deepcopy(resolve("sewing-fr3").to_dict())
        lone["id"] = "left-fr3-alone"
        del lone["instances"]["right"]
        lone["attachments"] = {k: v for k, v in lone["attachments"].items() if not v["parent"].startswith("right/")}
        single_panda = copy.deepcopy(resolve("sewing-fr3-single").to_dict())
        single_panda["id"] = "single-panda"
        for inst in single_panda["instances"].values():
            inst["model"] = inst["model"].replace("fr3", "panda")
        for d in (lone, single_panda):
            with self.subTest(setup=d["id"]), self.assertRaises(ValueError):
                validate_setup(d)


class StandInInterfaceTest(unittest.TestCase):
    """Over the stand-in world (no GPU, no physics): the renderer options are checked before anything is built, a
    batch's commands are checked without a write or a tick, and the arm names a deployment offers are the names its
    requests may carry."""

    def test_renderer_options_are_checked_before_the_world_is_built(self):
        from gym_anything_robotics.backends.newton_sewing import IDLE_RELEASE_S, NewtonSewingBackend
        bad = ({"idle_release_s": "abc"}, {"idle_release_s": -1}, {"idle_release_s": True},
               {"idle_release_s": float("inf")}, {"idle_release_s": float("nan")}, {"idle_release_s": [20]},
               {"width": "abc"}, {"width": 0}, {"height": 720.5}, {"samples": 0}, {"samples": True}, {"exposure": 1},
               "idle_release_s=20")
        for options in bad:
            with self.subTest(options=options), patch("sewing.task_set_env.make_environment") as build:
                out = Path(tempfile.mkdtemp())/"session"
                with self.assertRaises((ValueError, UnsupportedConfigurationError)) as caught:
                    NewtonSewingBackend(robot="dual-fr3", task="plain-seam", output_dir=str(out),
                                        renderer_options=options)
                self.assertIn("renderer_options", str(caught.exception))
                build.assert_not_called()                       # no world, and no session folder either
                self.assertFalse(out.exists())
        self.assertIn("idle_release_s must be a number of seconds >= 0",
                      str(self._refusal({"idle_release_s": "abc"})))
        for options, idle in (({}, IDLE_RELEASE_S), ({"idle_release_s": 0}, 0.), ({"idle_release_s": 20}, 20.),
                              ({"idle_release_s": 1800.5, "width": 640, "height": 360, "samples": 4}, 1800.5),
                              ({"idle_release_s": None}, IDLE_RELEASE_S)):
            with self.subTest(options=options):
                out = Path(tempfile.mkdtemp())/"session"
                world = _stub_world("dual-fr3")
                env = SimpleNamespace(world=world, joints=world.joints, start_state="under_foot",
                                      lock=threading.RLock(), act=Mock(), steps=0)
                with patch("sewing.task_set_env.make_environment", return_value=env):
                    b = NewtonSewingBackend(robot="dual-fr3", task="plain-seam", output_dir=str(out),
                                            renderer_options=options)
                # what the backend reads when it puts a render worker away (_end_operation)
                self.assertEqual(float(b.options.get("idle_release_s", IDLE_RELEASE_S) or 0.), idle)
                self.assertEqual(b.options, {k: v for k, v in options.items() if v is not None})

    def _refusal(self, options):
        from gym_anything_robotics.backends.newton_sewing import NewtonSewingBackend
        with self.assertRaises(ValueError) as caught:
            NewtonSewingBackend(robot="dual-fr3", task="plain-seam", renderer_options=options,
                                output_dir=str(Path(tempfile.mkdtemp())/"session"))
        return caught.exception

    def test_a_batch_is_refused_whole_before_any_command_runs(self):
        from gym_anything_robotics.actions import (CartesianDelta, CartesianPose, Gripper, MotionSafety,
                                                   NamedJointTarget, RawAction, ToolIO)
        from gym_anything_robotics.env import RoboticsEnv
        from gym_anything_robotics.errors import ActionRejectedError
        b = _backend("dual-fr3")
        finger = "left/fr3_finger_joint1"
        good = ToolIO(arm="left", outputs={"speed_spm": 600.0}, ticks=10)
        # the stand-in world has no kinematics: a Cartesian command passes its checks without the IK being consulted
        for action in (good, NamedJointTarget({finger: 0.02}, 4), Gripper(arm="right", half_width_m=0.02),
                       CartesianDelta(arm="left", delta_m=(0., 0., .01), duration_s=0.05),
                       CartesianPose(arm="right", position_m=(0., -.1, .2), duration_s=0.5)):
            with self.subTest(accepted=type(action).__name__):
                self.assertIsNone(b.validate_action(action))
        refused = (ToolIO(arm="left", outputs={"speed_spm": 9000.0}), ToolIO(arm="left", outputs={"knife": True}),
                   ToolIO(arm="left", outputs={"speed_spm": 0.0}, ticks=4001),
                   NamedJointTarget({finger: 0.5}, 1), NamedJointTarget({"left/no_such_joint": 0.}, 1),
                   Gripper(arm="left", inward_effort_n=500.), Gripper(arm="left", half_width_m=0.2),
                   CartesianDelta(arm="left", delta_m=(0., 0., .01), duration_s=0.00031),
                   CartesianDelta(arm="left", delta_m=(0., 0., .01), duration_s=2.5),
                   RawAction({"op": "joint_impedance", "arm": "left", "stiffness_nm_per_rad": 99999.0}),
                   RawAction({"op": "joint_impedance", "arm": "left", "stiffness_nm_per_rad": 100.0, "ticks": 0}))
        for action in refused:
            with self.subTest(refused=repr(action)[:60]):
                with self.assertRaises(ActionRejectedError) as caught:
                    b.validate_action(action)
                self.assertIn("no action executed", str(caught.exception))
                self.assertIsNone(b._pending_impedance)
        for action in (CartesianDelta(arm="left", delta_m=(0., 0., .01), safety=MotionSafety(stop_on_contact=True)),
                       RawAction({"op": "teleport"})):
            with self.subTest(unsupported=repr(action)[:60]), self.assertRaises(UnsupportedActionError):
                b.validate_action(action)
        # GAR's environment calls it for every command of a batch before the first (gym_anything_robotics/env.py)
        env = RoboticsEnv(b, auto_reset=False)
        b.act = Mock(side_effect=AssertionError("a command of a refused batch ran"))
        for batch in ([good, refused[0]], [good, good, refused[5]], [good, RawAction({"op": "teleport"})]):
            with self.subTest(batch=len(batch)), self.assertRaises((ActionRejectedError, UnsupportedActionError)):
                env.act(batch)
        b.act.assert_not_called()
        b.env.act.assert_not_called()

    def test_the_arm_names_a_deployment_offers_are_the_names_a_request_may_carry(self):
        from gym_anything_robotics.action_codec import actions_from_request
        from gym_anything_robotics.actions import ARM_NAMES, CartesianDelta, CartesianPose, arm_names
        delta = {"type": "cartesian_delta", "arm": "left+waist", "delta_m": [0.0, 0.0, 0.01], "duration_s": 0.05}
        pose = {"type": "cartesian_pose", "arm": "right+waist", "position_m": [0.0, -0.2, 0.1]}
        for robot, (arms, _, _) in DeploymentContractTest.EXPECTED.items():
            with self.subTest(robot=robot):
                b = _backend(robot)
                extra = tuple(a for a in arms if a not in ARM_NAMES)
                self.assertEqual(b.extra_arm_names, extra)
                self.assertEqual(set(ARM_NAMES) | set(b.extra_arm_names), set(arms) | set(ARM_NAMES))
                access = b.policy_access()
                self.assertEqual(access.extra_arm_names, extra)
                if robot == "g1":
                    # the policy's decoding (PolicyAccess.decode_actions) and the owner's (inside arm_names, as the
                    # HTTP service does with RoboticsEnv.extra_arm_names)
                    one = access.decode_actions(delta)
                    self.assertEqual((type(one), one.arm), (CartesianDelta, "left+waist"))
                    both = access.decode_actions({"actions": [delta, pose]})
                    self.assertEqual([a.arm for a in both], ["left+waist", "right+waist"])
                    with arm_names(b.extra_arm_names):
                        self.assertEqual(actions_from_request(pose), CartesianPose("right+waist", (0.0, -0.2, 0.1)))
                    for arm in arms:
                        self.assertEqual(b._arm(arm), arm)
                else:
                    for payload in (delta, pose):
                        with self.assertRaises(ValueError) as caught:
                            access.decode_actions(payload)
                        self.assertEqual(str(caught.exception), "arm must be 'left' or 'right'")
        # outside a backend's declaration nothing changed, and one backend's names do not outlive its block
        with self.assertRaises(ValueError) as caught:
            actions_from_request(delta)
        self.assertEqual(str(caught.exception), "arm must be 'left' or 'right'")
        with arm_names(("left+waist",)):
            with self.assertRaises(ValueError) as caught:
                actions_from_request(pose)
            self.assertIn("this robot also accepts 'left+waist'", str(caught.exception))
        with self.assertRaises(ValueError):
            CartesianDelta(arm="left+waist", delta_m=(0., 0., .01))


class DoctorTest(unittest.TestCase):
    def test_doctor_reports_the_renderer_modules_and_every_deployments_assets(self):
        from sewing.doctor import CROSS_ENV, MODELS, checks
        r = checks(SimpleNamespace(robot=None))
        names = [c["check"] for c in r["checks"]]
        for mod in CROSS_ENV:
            self.assertIn(f"module {mod} (renderer)", names)
        for model in {m for ms in MODELS.values() for m in ms}:
            self.assertTrue(any(n.startswith(f"{model} assets") for n in names), model)
        self.assertEqual(set(MODELS), set(RIGS))
        # only the selected deployment's assets are required
        g1 = next(c for c in r["checks"] if c["check"].startswith("g1 assets"))
        self.assertFalse(g1["required"])
        g1_sel = next(c for c in checks(SimpleNamespace(robot="g1"))["checks"] if c["check"].startswith("g1 assets"))
        self.assertTrue(g1_sel["required"])


if __name__ == "__main__":
    unittest.main()


class WallTimeNoteTest(unittest.TestCase):
    """The runtime note a restricted policy receives (capabilities.hardware.wall_time) is built from E16's four result
    files when the backend is imported, and the command cap is held to the slowest rate they measured."""

    def _records(self):
        import json
        from gym_anything_robotics.backends import newton_sewing as NS
        paths = NS.e16_result_paths()
        return NS, {r: json.loads(paths[r].read_text()) for r, _ in NS.E16_DEPLOYMENTS}

    def _pause(self):
        import json
        from gym_anything_robotics.backends import newton_sewing as NS
        paths = NS.e16b_result_paths()
        if not all(p.exists() for p in paths.values()):
            return None
        return {r: json.loads(paths[r].read_text()) for r, _ in NS.E16_DEPLOYMENTS}

    def _quiet(self):
        import json
        from gym_anything_robotics.backends import newton_sewing as NS
        paths = NS.e16_result_paths("quiet")
        if not all(p.exists() for p in paths.values()):
            return None
        return {r: json.loads(paths[r].read_text()) for r, _ in NS.E16_DEPLOYMENTS}

    def test_note_is_derived_from_every_deployments_e16_record(self):
        NS, records = self._records()
        self.assertEqual(NS.e16_result_paths()["dual-fr3"].resolve(),
                         (ROOT/"sewing/data/runtime/e16_runtime_api.json").resolve())
        quiet, pause = self._quiet(), self._pause()
        figures = NS.wall_time_figures(records, quiet, pause)
        self.assertEqual(NS.WALL_TIME_NOTE, NS._wall_time_note(figures))
        self.assertEqual("On a quiet host" in NS.WALL_TIME_NOTE, quiet is not None)
        self.assertEqual("on the shared host (E16b" in NS.WALL_TIME_NOTE, pause is not None)
        self.assertNotEqual(NS.WALL_TIME_NOTE, NS.WALL_TIME_FALLBACK)
        for key, figure in figures.items():
            with self.subTest(key=key):
                if key in ("shortest_s", "quiet_shortest_s") and figure == "none":
                    self.assertIn("no command length stayed within", NS.WALL_TIME_NOTE)
                else:
                    self.assertIn(figure, NS.WALL_TIME_NOTE)
        # spot checks against the files' own fields, independent of the derivation's code
        worst = max(e["worst_action_wall_per_sim"] for e in records.values())
        self.assertEqual(figures["worst"], f"{worst:.0f}")
        self.assertNotEqual(figures["physics_hash"], "differs between deployments")
        for robot, e in records.items():
            with self.subTest(robot=robot):
                self.assertEqual(e["robot"], robot)
                sew = [r["wall_s"]/r["sim_s"] for r in e["rows"] if r["step"].startswith("sewing")]
                self.assertTrue(sew)
                self.assertIn(f"{max(sew):,.0f}", figures[f"{robot}_sewing"])

    def test_quiet_host_records_are_optional_and_quoted_from_their_files(self):
        import json
        NS, records = self._records()
        shared = NS.e16_result_paths()
        with tempfile.TemporaryDirectory() as tmp:
            # no quiet records: the note is the shared-host one, not the fallback
            missing = {r: Path(tmp)/f"missing_{r}.json" for r, _ in NS.E16_DEPLOYMENTS}
            note = NS.load_wall_time_note(shared, missing)
            self.assertNotEqual(note, NS.WALL_TIME_FALLBACK)
            self.assertNotIn("On a quiet host", note)
            self.assertNotIn("idle_release_s", note)
            # quiet records with a measured pause cost: their sewing rates and the pause cost are quoted
            quiet = {}
            for r, _ in NS.E16_DEPLOYMENTS:
                e = dict(records[r], pause_beyond_release_adds_s=4.25 + len(r))
                quiet[r] = Path(tmp)/f"quiet_{r}.json"
                quiet[r].write_text(json.dumps(e))
            note = NS.load_wall_time_note(shared, quiet)
            self.assertIn("On a quiet host", note)
            pauses = sorted(4.25 + len(r) for r, _ in NS.E16_DEPLOYMENTS)
            self.assertIn(f"took {pauses[0]:,.1f}-{pauses[-1]:,.1f} s longer", note)
            self.assertIn(f"default {NS.IDLE_RELEASE_S:g} s", note)

    def test_pause_records_are_optional_and_quoted_from_their_files(self):
        import json
        NS, records = self._records()
        shared = NS.e16_result_paths()
        with tempfile.TemporaryDirectory() as tmp:
            missing = {r: Path(tmp)/f"missing_{r}.json" for r, _ in NS.E16_DEPLOYMENTS}
            note = NS.load_wall_time_note(shared, missing, missing)
            self.assertNotIn("idle_release_s", note)
            pause = {}
            for i, (r, _) in enumerate(NS.E16_DEPLOYMENTS):
                e = dict(robot=r, physics_hash=records[r]["physics_hash"], pause_beyond_release_adds_s=6.0 + i,
                         pause_inside_release_adds_s=0.1*i,
                         rows=[dict(step="x", wall_s=1.0, host=dict(loadavg=[5.0 + i, 0, 0]))])
                pause[r] = Path(tmp)/f"pause_{r}.json"
                pause[r].write_text(json.dumps(e))
            note = NS.load_wall_time_note(shared, missing, pause)
            self.assertIn("took 6.0-9.0 s longer than one sent at once on the shared host", note)
            self.assertIn("after a 5 s pause, inside the release, +0.0 to +0.3 s", note)
            self.assertIn(f"default {NS.IDLE_RELEASE_S:g} s", note)
            # a record of another deployment is refused (the note falls back to the shared-host figures alone)
            bad = json.loads(pause["g1"].read_text())
            bad["robot"] = "dual-fr3"
            pause["g1"].write_text(json.dumps(bad))
            self.assertNotIn("idle_release_s", NS.load_wall_time_note(shared, missing, pause))

    def test_static_statements_of_the_note(self):
        # the note does not say "five minutes" (the launcher waits three) or give an example that "raises"
        # idle_release_s below the default, and it does not present the 5 s and 15 s pauses as a release
        NS, records = self._records()
        note = NS.WALL_TIME_NOTE
        self.assertNotIn("five minutes", note)
        self.assertNotIn('"idle_release_s": 120}', note)
        if self._quiet() is not None:
            self.assertIn("each run started after three minutes of 1-minute load under 4 and GPU use under 30 %", note)
        if self._quiet() is not None or self._pause() is not None:
            import re
            example = re.search(r'"idle_release_s": ([0-9.]+)\}', note)
            self.assertIsNotNone(example)
            self.assertGreater(float(example.group(1)), NS.IDLE_RELEASE_S)          # raising means above the default
            self.assertIn(f"kept for renderer_options idle_release_s (default {NS.IDLE_RELEASE_S:g} s)", note)
        pause = self._pause()
        if pause is not None and not all("idle_release_s" in e for e in pause.values()):
            # records without idle_release_s: pauses of 5 s and 15 s under the default release, which they do not reach
            self.assertLess(15.0, NS.IDLE_RELEASE_S)
            self.assertIn("both inside the default release, so these records do not hold what a new render worker "
                          "costs", note)
            self.assertNotIn("beyond the release", note)
        self.assertIn("tens of milliseconds on the load task", note)
        self.assertIn("every further simulation on the same card slows it in proportion", note)


    def test_the_backend_imports_with_the_package_records(self):
        # the note is built at import from whatever records the package has: an import-order error (a constant the
        # note uses defined below it) only shows in a fresh process
        import subprocess
        r = subprocess.run([sys.executable, "-c", "import gym_anything_robotics.backends.newton_sewing as NS; "
                            "assert NS.WALL_TIME_NOTE"], capture_output=True, text=True, cwd=ROOT, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])

    def test_the_command_cap_returns_inside_the_client_timeout_at_the_slowest_measured_rate(self):
        NS, records = self._records()
        every = list(records.values()) + list((self._quiet() or {}).values())
        slowest = max(r["wall_s"]/r["sim_s"] for e in every for r in e["rows"]
                      if r["sim_s"] and r["sim_s"] >= 0.3)
        self.assertLess(NS.MAX_COMMAND_S*slowest, NS.CLIENT_TIMEOUT_S)

    def test_missing_or_broken_records_give_the_fallback(self):
        from gym_anything_robotics.backends import newton_sewing as NS
        with tempfile.TemporaryDirectory() as tmp:
            paths = {r: Path(tmp)/f"{r}.json" for r, _ in NS.E16_DEPLOYMENTS}
            self.assertEqual(NS.load_wall_time_note(paths), NS.WALL_TIME_FALLBACK)
            for p in paths.values():
                p.write_text('{"rows": []}')
            self.assertEqual(NS.load_wall_time_note(paths), NS.WALL_TIME_FALLBACK)
        self.assertIn("sewing/REPORT.pdf", NS.WALL_TIME_FALLBACK)
        self.assertIn("client_timeout_s", NS.WALL_TIME_FALLBACK)

    def test_a_record_of_another_deployment_is_refused(self):
        import json
        from gym_anything_robotics.backends import newton_sewing as NS
        rec = dict(robot="g1", rows=[dict(step="sewing 0", wall_s=40., sim_s=.5)])
        with tempfile.TemporaryDirectory() as tmp:
            paths = {r: Path(tmp)/f"{r}.json" for r, _ in NS.E16_DEPLOYMENTS}
            for p in paths.values():
                p.write_text(json.dumps(rec))
            self.assertEqual(NS.load_wall_time_note(paths), NS.WALL_TIME_FALLBACK)

    def test_policy_capabilities_carry_the_note_the_cap_and_the_timeout(self):
        from gym_anything_robotics.backends import newton_sewing as NS
        hw = _backend("dual-fr3").policy_access().capabilities()["hardware"]
        self.assertEqual(hw["wall_time"], NS.WALL_TIME_NOTE)
        self.assertEqual(hw["client_timeout_s"], NS.CLIENT_TIMEOUT_S)
        self.assertEqual(hw["max_command_s"], NS.MAX_COMMAND_S)
