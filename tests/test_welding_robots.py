"""Contract checks for welding's robot deployments (the humanoid g1 and the two-Panda dual-panda) and the launch-time
robot/task refusal: registration, validation of rig objects, refusals before any session exists, and capabilities
(arms, torch arm, hand present or not) read from the catalog, the controller profiles and the backend constructor over
a stub environment. No GPU, no physics, no renderer (seconds).

The optional class at the end builds the deployments' rigid cells (CPU, about a minute) and is skipped unless
GAR_WELDING_PHYSICS_TESTS=1.
"""

import json
import math
import os
import tempfile
import threading
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from gym_anything_robotics.errors import UnsupportedActionError, UnsupportedConfigurationError
from tests.packages import PACKAGES

ROOT = PACKAGES
NEW = {"dual-panda": "welding-panda", "g1": "welding-g1"}
ALL = ("ur10e-fr3", "ur10e", "dual-fr3", "single-fr3", "dual-panda", "g1")


class RegistrationTest(unittest.TestCase):
    def test_deployments_listed_in_order_with_their_rigs(self):
        from robots.catalog import ENVIRONMENTS, for_environment
        from gym_anything_robotics.launch import describe
        d = describe("welding")
        self.assertEqual([r["id"] for r in d["robots"]], list(ALL))
        self.assertEqual(d["default_robot"], "ur10e-fr3")
        self.assertNotIn("unsupported", ENVIRONMENTS["welding"])
        for alias, rig in NEW.items():
            with self.subTest(alias=alias):
                self.assertEqual(ENVIRONMENTS["welding"]["aliases"][alias], rig)
                self.assertEqual(for_environment("welding", alias).to_dict()["id"], rig)

    def test_rigs_resolve_and_differ_from_other_environments_rigs(self):
        from robots.catalog import resolve
        prints = {r: resolve(r).fingerprint for r in ("welding-panda", "welding-g1", "sewing-g1", "scan-g1",
                                                         "sewing-panda", "filleting-panda", "welding-fr3")}
        self.assertEqual(len(set(prints.values())), len(prints))

    def test_robot_setup_aliases_match_the_catalog(self):
        from robots.catalog import ENVIRONMENTS
        from welding.robot_setup import ALIASES
        self.assertEqual(ALIASES, ENVIRONMENTS["welding"]["aliases"])


class ValidationTest(unittest.TestCase):
    def test_new_rigs_accepted_as_alias_id_object_and_path(self):
        from robots.catalog import resolve, select_for_backend
        from gym_anything_robotics.launch import LaunchSpec
        for alias, rig in NEW.items():
            for sel in (alias, rig, resolve(rig).to_dict(), str(PACKAGES / "robots" / "rigs" / f"{rig}.json")):
                with self.subTest(sel=str(sel)[:40]):
                    self.assertEqual(select_for_backend("welding", sel), alias if sel != alias else alias)
            LaunchSpec("welding", robot=resolve(rig).to_dict())

    def test_other_environments_g1_and_panda_rigs_refused(self):
        from robots.catalog import resolve, select_for_backend
        for rig in ("sewing-g1", "scan-g1", "sewing-panda", "filleting-panda"):
            with self.subTest(rig=rig), self.assertRaises(ValueError) as caught:
                select_for_backend("welding", resolve(rig).to_dict())
            self.assertIn("hosts its six catalog rigs", str(caught.exception))


class LaunchRefusalTest(unittest.TestCase):
    """A torch-only rig with a task that needs a hand is refused when the launch is specified, before any session
    record or server exists."""

    def test_torch_only_rigs_refused_for_tack_at_launch(self):
        from gym_anything_robotics.launch import LaunchSpec
        from robots.catalog import resolve
        for robot in ("ur10e", "single-fr3", "welding-ur10e", resolve("welding-fr3-single").to_dict()):
            for task in ("tack", "A1"):
                with self.subTest(robot=str(robot)[:30], task=task), self.assertRaises(ValueError) as caught:
                    LaunchSpec("welding", robot=robot, backend_options={"task": task})
                self.assertIn("needs the FR3's hand", str(caught.exception))
                self.assertIn("torch arm only", str(caught.exception))

    def test_task_resolution_with_a_torch_only_base_is_refused(self):
        # the path of `gar run welding --robot ur10e --task tack`: cli_launch builds the base spec, then tasks.resolve
        from gym_anything_robotics import tasks
        from gym_anything_robotics.launch import LaunchSpec
        base = LaunchSpec("welding", robot="ur10e")
        with self.assertRaises(ValueError):
            tasks.resolve("welding", {"task": "tack"}, base=base)
        tasks.resolve("welding", {"task": "fillet-2f"}, base=base)

    def test_hand_deployments_and_default_accepted_for_every_task(self):
        from gym_anything_robotics.launch import LaunchSpec
        for robot in (None, "ur10e-fr3", "dual-fr3", "dual-panda", "g1"):
            for task in ("position-torch", "tack", "fillet-2f"):
                if robot == "g1" and task == "tack":
                    continue
                with self.subTest(robot=robot, task=task):
                    LaunchSpec("welding", robot=robot, backend_options={"task": task})
        # the G1's left Dex3 has not been shown holding the bracket: tack is refused on it (variants.NO_HOLD)
        with self.assertRaises(ValueError) as caught:
            LaunchSpec("welding", robot="g1", backend_options={"task": "tack"})
        self.assertIn("not offered on it", str(caught.exception))
        for robot in ("ur10e", "single-fr3"):
            for task in ("position-torch", "fillet-2f", None):
                with self.subTest(robot=robot, task=task):
                    LaunchSpec("welding", robot=robot, backend_options={} if task is None else {"task": task})

    def test_check_launch_resolves_robots_and_task_codes(self):
        from welding.variants import check_launch
        check_launch(None, {"task": "tack"})
        check_launch("g1", {"task": "fillet-2f"})
        with self.assertRaises(ValueError):
            check_launch("g1", {"task": "A1"})                       # tack's code: refused on the G1 (NO_HOLD)
        with self.assertRaises(ValueError):
            check_launch("ur10e", {"task": "A1"})
        with self.assertRaises(ValueError):
            check_launch({"instances": {}}, {})                      # not a rig welding hosts


    def test_backend_still_refuses_in_process(self):
        from gym_anything_robotics.backends.newton_welding import NewtonWeldingBackend
        for robot in ("ur10e", "single-fr3"):
            with self.subTest(robot=robot), patch("welding.env.WeldingEnv") as env, \
                    self.assertRaises(UnsupportedConfigurationError):
                NewtonWeldingBackend(robot=robot, task="tack")
            env.assert_not_called()


class ProfileAndMountTest(unittest.TestCase):
    """What the deployments are built from: the Panda's hand-less torch profile (additive), the torches, the G1's
    clamp, stance and ready posture."""

    def test_panda_profiles_added_without_changing_the_others(self):
        import subprocess
        from robots.catalog import model
        spec = model("panda").to_dict()
        prof, poses = spec["actuation"]["profiles"], spec["poses"]
        torch = prof["welding_torch"]
        self.assertEqual(sorted(torch["joints"]), [f"panda_joint{i}" for i in range(1, 8)])
        self.assertIsNone(torch["default_tool"])
        self.assertTrue(torch["gravity_compensation"])
        self.assertEqual(prof["welding"]["default_tool"], "franka-hand")
        fr3 = model("fr3").to_dict()["actuation"]["profiles"]
        for name in ("welding", "welding_torch"):
            ours = {k: v for k, v in prof[name].items() if k not in ("joints", "initial_pose")}
            theirs = {k: v for k, v in fr3[name].items() if k not in ("joints", "initial_pose")}
            self.assertEqual(ours, theirs, name)
            self.assertEqual(set(poses[name]), set(prof[name]["joints"]))
        try:
            old = json.loads(subprocess.run(["git", "show", "HEAD:robots/models/panda/robot.json"], cwd=ROOT,
                                            capture_output=True, text=True, check=True).stdout)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.skipTest("no git history")
        for name, value in old["actuation"]["profiles"].items():
            self.assertEqual(prof[name], value, name)
        for name, value in old["poses"].items():
            self.assertEqual(poses[name], value, name)

    def test_bare_panda_urdf_drops_only_the_hand(self):
        from robots.assets import resolve
        from robots.backends.newton import _panda_without_hand, PANDA_HAND_LINKS
        try:
            path = resolve("panda")
        except (ValueError, FileNotFoundError, OSError) as exc:
            self.skipTest(f"Panda assets not installed: {exc}")
        full, bare = ET.parse(path).getroot(), ET.fromstring(_panda_without_hand(path))
        links = lambda x: {l.get("name") for l in x.findall("link")}
        self.assertEqual(links(full) - links(bare), set(PANDA_HAND_LINKS))
        moving = [j.get("name") for j in bare.findall("joint") if j.get("type") != "fixed"]
        self.assertEqual(moving, [f"panda_joint{i}" for i in range(1, 8)])
        for mesh in bare.iter("mesh"):
            self.assertTrue(Path(mesh.get("filename")).is_file(), mesh.get("filename"))

    def test_torches(self):
        import numpy as np
        from welding import torch as TO
        self.assertIs(TO.GUNS["panda"], TO.RM2)
        self.assertIs(TO.GUNS["g1"], TO.RM2)
        self.assertEqual((TO.GUNS["ur10e"], TO.GUNS["fr3"]), (TO.BA1, TO.RM2))
        self.assertAlmostEqual(TO.RM2.mass_properties()[0], 1.60, places=6)
        M = TO.G1_TORCH_MOUNT
        R = M.rotation()
        self.assertTrue(np.allclose(R[:, 2], [1., 0., 0.]))          # along the hand
        self.assertTrue(np.allclose(R[:, 1], [0., -1., 0.]))         # the neck bends away from the fingers
        self.assertAlmostEqual(M.clamp_mass, .15)
        # the KS-2 body clears the Dex3 palm's back (y = -24.4 mm in the link frame) by more than 1 mm
        self.assertGreater(-(M.origin[1] + TO.RM2.mount[0]), .0244 + 1e-3)

    def test_g1_stance_and_ready_posture_within_the_urdf_limits(self):
        from robots.assets import resolve
        from welding import humanoid as HU
        from welding import layout as LY
        self.assertGreaterEqual(HU.STAND["y"] - LY.TABLE["size"][1]/2, .10 - 1e-9)    # pelvis behind the table's edge
        try:
            root = ET.parse(resolve("g1")).getroot()
        except (ValueError, FileNotFoundError, OSError) as exc:
            self.skipTest(f"G1 assets not installed: {exc}")
        limits = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
                  for j in root.findall("joint") if j.get("type") == "revolute"}
        for name, value in HU.READY.items():
            short = name.split("/", 1)[1]
            with self.subTest(joint=short):
                lo, hi = limits[short]
                self.assertTrue(lo <= value <= hi)
        # the stance keeps the soles flat and the pelvis upright: hip + knee + ankle pitch = 0 on both legs
        for side in ("left", "right"):
            self.assertAlmostEqual(sum(HU.READY[HU.joint(side, j)] for j in ("hip_pitch", "knee", "ankle_pitch")), 0.)


def _stub_backend(robot, task="fillet-2f", cameras=(), **options):
    """The real backend constructor over a stub environment shaped like the deployment's world (welding/world.py):
    arms and hands as scene.py and world.py derive them (no GPU, no physics)."""
    from robots.catalog import for_environment
    from gym_anything_robotics.backends.newton_welding import NewtonWeldingBackend
    data = for_environment("welding", robot).to_dict()
    g1 = robot == "g1"
    arm_hw = {name: {"model": inst["model"].split("@")[0]} for name, inst in data["instances"].items()}
    for att in data.get("attachments", {}).values():
        arm_hw[att["parent"].split("/")[0]]["hand"] = att["model"]
    hand_arms = ("left",) if g1 else tuple(a for a in arm_hw if arm_hw[a].get("hand"))
    finger = "panda_finger_joint" if robot == "dual-panda" else "fr3_finger_joint"
    joints = {} if g1 else {f"{a}/{finger}{k}": dict(q=0, dof=0, lower=0., upper=.04, effort_limit=35.,
                                                    velocity_limit=.2) for a in hand_arms for k in (1, 2)}
    kin = SimpleNamespace(arms={a: (None, []) for a in ("right", "right+waist", "left", "left+waist")} if g1 else {})
    world = SimpleNamespace(rig=SimpleNamespace(instances=dict.fromkeys(arm_hw)), torch_arm="right", g1=g1, kin=kin,
                            hand_arms=hand_arms, scene=SimpleNamespace(arm_hw=arm_hw), joints=joints, fault=None,
                            gripper_joints=lambda arm: [] if g1 else [f"{arm}/{finger}1", f"{arm}/{finger}2"])
    env = SimpleNamespace(world=world, joints=joints, start_state="tacked", lock=threading.RLock(), act=Mock(),
                          tool_io=Mock(), steps=0)
    out = Path(tempfile.mkdtemp())/"session"
    with patch.object(NewtonWeldingBackend, "_make_env", return_value=env):
        return NewtonWeldingBackend(robot=robot, task=task, trace=False, cameras=cameras, output_dir=str(out),
                                    **options)


class CapabilityTest(unittest.TestCase):
    def test_arms_torch_arm_and_hands_per_deployment(self):
        expected = {  # arms offered to Cartesian commands, gripper action offered
            "ur10e-fr3": (("left", "right"), True), "dual-fr3": (("left", "right"), True),
            "ur10e": (("right",), False), "single-fr3": (("right",), False),
            "dual-panda": (("left", "right"), True),
            "g1": (("right", "right+waist", "left", "left+waist"), False)}
        for robot, (arms, gripper) in expected.items():
            with self.subTest(robot=robot):
                b = _stub_backend(robot, task="tack" if gripper else "fillet-2f")
                self.assertEqual(b.arms, arms)
                self.assertEqual(b.torch_arm, "right")
                hw = b.policy_access().capabilities()["hardware"]
                self.assertEqual((hw["arms"], hw["torch_arm"]), (list(arms), "right"))
                self.assertEqual("gripper" in b.kinds, gripper)
                if robot == "g1":
                    self.assertIn("Dex3-1 hands", b.capabilities.notes["gripper"])
                    self.assertIn("G1's low-level joint interface has no collision", b.capabilities.notes["control"])
                elif gripper:
                    self.assertIn("Franka Hand on 'left'", b.capabilities.notes["gripper"])

    def test_g1_dex3_is_not_a_gripper(self):
        from gym_anything_robotics.actions import Gripper
        b = _stub_backend("g1", task="fillet-2f")
        for arm in ("left", "right"):
            with self.assertRaises(UnsupportedActionError):
                b.act(Gripper(arm, half_width_m=.02))
        b.env.act.assert_not_called()

    def test_rig_hands(self):
        from robots.catalog import for_environment, model
        for robot, hand in (("dual-panda", {"left": "franka-hand@1"}), ("g1", {}), ("ur10e", {}),
                            ("single-fr3", {})):
            data = for_environment("welding", robot).to_dict()
            got = {a["parent"].split("/")[0]: a["model"] for a in data.get("attachments", {}).values()}
            self.assertEqual(got, hand, robot)
        g1 = for_environment("welding", "g1").to_dict()["instances"]["g1"]
        groups = model(g1["model"]).to_dict()["groups"]
        self.assertEqual(len(groups["hands"]["joints"]), 14)          # two Dex3-1 hands, seven joints each
        panda = for_environment("welding", "dual-panda").to_dict()["instances"]
        self.assertEqual((panda["right"]["controller_profile"], panda["left"]["controller_profile"]),
                         ("welding_torch", "welding"))


class RobotCameraTest(unittest.TestCase):
    """The cameras the deployments' robots carry: each FR3 or Panda arm's wrist D405 (<arm>_wrist_rgb), the G1's head
    D435 (head_rgb), none on a UR10e; offered beside the cell's cameras and in every default session, refused where the
    robot is not, and the world built the same whichever cameras a session selects (no GPU, no physics)."""

    OFFERED = {"ur10e-fr3": ("left_wrist_rgb",), "ur10e": (), "dual-fr3": ("left_wrist_rgb", "right_wrist_rgb"),
               "single-fr3": ("right_wrist_rgb",), "dual-panda": ("left_wrist_rgb", "right_wrist_rgb"),
               "g1": ("head_rgb",)}

    def test_offered_per_deployment_and_in_the_default(self):
        from gym_anything_robotics.backends import newton_welding as NW
        self.assertEqual(NW.POLICY_CAMERAS, ("bench", "side", "weld_camera"))
        self.assertEqual(set(self.OFFERED), set(ALL))
        for robot, offered in self.OFFERED.items():
            with self.subTest(robot=robot):
                self.assertEqual(NW.robot_cameras(robot), offered)
                self.assertEqual(NW.default_cameras(robot), ("bench",) + offered)
                self.assertEqual(NW.check_options(robot)[4], ("bench",) + offered)
                self.assertEqual(NW.check_options(robot, cameras=None)[4], ("bench",) + offered)
                self.assertEqual(NW.check_options(robot, cameras=[])[4], ())
                b = _stub_backend(robot, cameras=None)
                self.assertEqual(b.capabilities.robot_cameras, ("bench",) + offered)
                self.assertEqual([c.name for c in b.policy_access().cameras], ["bench", *offered])
                b = _stub_backend(robot, cameras=("bench",))
                self.assertEqual(b.capabilities.robot_cameras, ("bench",))
                # every FR3 and Panda arm carries its D405 whether or not its camera is selected
                note = b.capabilities.notes["cameras"]
                wrists = [c[:-len("_wrist_rgb")] for c in offered if c.endswith("_wrist_rgb")]
                self.assertEqual("D405" in note, bool(wrists))
                if wrists:
                    self.assertIn(f"each FR3 or Panda arm ({', '.join(wrists)}) carries its wrist D405, the 60 g case "
                                  "and the 10 g bracket on link7 as collision geometry, whether or not its camera is "
                                  "selected", note)

    def test_camera_specs_and_notes(self):
        for robot, offered in self.OFFERED.items():
            for name in offered:
                with self.subTest(robot=robot, camera=name):
                    b = _stub_backend(robot, cameras=("bench", name))
                    self.assertEqual(b.capabilities.robot_cameras, ("bench", name))
                    spec = b.capabilities.camera_specs[1]
                    hfov = 69. if name == "head_rgb" else 87.
                    self.assertEqual((spec.name, spec.mount_frame), (name, "g1_torso_d435" if name == "head_rgb"
                                                                     else "link7_d405_envelope"))
                    self.assertEqual((spec.width_px, spec.height_px, spec.native_width_px, spec.native_height_px),
                                     (1280, 720, 1280, 720))
                    self.assertAlmostEqual(math.degrees(2*math.atan(spec.sensor_size_m[0]/2/spec.focal_length_m[0])),
                                           hfov, places=9)
                    self.assertEqual([c["name"] for c in b.policy_access().capabilities()["camera_specs"]],
                                     ["bench", name])
                    note = b.capabilities.notes["cameras"]
                    if name == "head_rgb":
                        self.assertIn("head_rgb: the G1 head D435 mount (URDF d435_joint), 69 deg HFOV", note)
                    else:
                        self.assertIn(f"{name}: an Intel RealSense D405", note)
                        self.assertIn("carries its wrist D405, the 60 g case and the 10 g bracket on link7", note)
                    with self.assertRaises(ValueError) as caught:
                        b.render(label="bogus")
                    self.assertIn(name, str(caught.exception))
        b = _stub_backend("dual-fr3", cameras=("right_wrist_rgb",), renderer_options={"width": 640, "height": 360})
        spec = b.capabilities.camera_specs[0]
        self.assertEqual((spec.width_px, spec.height_px, spec.sensor_size_m), (640, 360, (.036, .036*360/640)))

    def test_the_world_is_built_the_same_whichever_cameras(self):
        from gym_anything_robotics.backends.newton_welding import NewtonWeldingBackend

        class Built(Exception):
            pass
        for robot, cameras in (("dual-fr3", ["bench", "right_wrist_rgb"]), ("dual-panda", None),
                               ("dual-panda", ["left_wrist_rgb", "right_wrist_rgb"]), ("g1", ["head_rgb"]),
                               ("ur10e-fr3", None), ("dual-fr3", ["bench", "side", "weld_camera"]), ("g1", []),
                               ("single-fr3", [])):
            with self.subTest(robot=robot, cameras=cameras), tempfile.TemporaryDirectory() as d, \
                    patch("welding.env.WeldingEnv", side_effect=Built) as env, self.assertRaises(Built):
                NewtonWeldingBackend(robot=robot, output_dir=d + "/s",
                                     **({} if cameras is None else {"cameras": cameras}))
            self.assertEqual(set(env.call_args.kwargs), {"output_dir"})
            self.assertEqual(env.call_args.args[0].robot, robot)

    def test_cameras_a_deployment_does_not_carry_are_refused_before_the_world(self):
        from gym_anything_robotics.backends.newton_welding import NewtonWeldingBackend, ROBOT_CAMERAS
        from gym_anything_robotics.launch import LaunchSpec
        why = {("single-fr3", "left_wrist_rgb"): "has no 'left' arm",
               ("ur10e", "left_wrist_rgb"): "has no 'left' arm",
               ("ur10e", "right_wrist_rgb"): "is a UR10e, which carries no camera",
               ("ur10e-fr3", "right_wrist_rgb"): "is a UR10e, which carries no camera",
               ("g1", "left_wrist_rgb"): "is the G1 (no Franka arm)", ("g1", "right_wrist_rgb"): "is the G1",
               ("dual-fr3", "head_rgb"): "has no G1", ("ur10e", "head_rgb"): "has no G1"}
        for robot, offered in self.OFFERED.items():
            for name in ROBOT_CAMERAS:
                if name in offered:
                    LaunchSpec("welding", robot=robot, backend_options={"cameras": ["bench", name]})
                    continue
                with self.subTest(robot=robot, camera=name):
                    with patch("welding.env.WeldingEnv") as env, \
                            self.assertRaises(UnsupportedConfigurationError) as caught:
                        NewtonWeldingBackend(robot=robot, cameras=["bench", name], output_dir="/nonexistent/never")
                    env.assert_not_called()
                    self.assertIn(f"Robot cameras of {robot!r}: {', '.join(offered) or 'none'}", str(caught.exception))
                    if (robot, name) in why:
                        self.assertIn(why[robot, name], str(caught.exception))
                    with self.assertRaises(ValueError):
                        LaunchSpec("welding", robot=robot, backend_options={"cameras": [name]})

    def test_errors_of_camera_lists_without_robot_cameras_unchanged(self):
        from gym_anything_robotics.backends.newton_welding import check_options
        for robot in ALL:
            for cameras in (["hero"], ["bench", "bench"], ["front"]):
                with self.subTest(robot=robot, cameras=cameras), self.assertRaises(ValueError) as caught:
                    check_options(robot, cameras=cameras)
                self.assertEqual(str(caught.exception), "Policy cameras are ('bench', 'side', 'weld_camera'); "
                                                        "('hero', 'seam') are privileged diagnostics")
        with self.assertRaises(ValueError) as caught:
            _stub_backend("ur10e", cameras=("bench",)).render(label="bogus")
        self.assertEqual(str(caught.exception), "Render label must be one of ('bench', 'side', 'weld_camera', 'hero', "
                                                "'seam')")
        # a deployment's robot cameras are in its scene, selected or not
        with self.assertRaises(ValueError) as caught:
            _stub_backend("dual-fr3", cameras=("bench",)).render(label="bogus")
        self.assertEqual(str(caught.exception), "Render label must be one of ('bench', 'side', 'weld_camera', 'hero', "
                                                "'seam', 'left_wrist_rgb', 'right_wrist_rgb')")


class DoctorTest(unittest.TestCase):
    """`gar doctor welding` covers the robot assets of every offered deployment."""

    def _models(self):
        from robots.catalog import for_environment
        out = {}
        for alias in ALL:
            for inst in for_environment("welding", alias).to_dict()["instances"].values():
                out.setdefault(inst["model"].split("@")[0], set()).add(alias)
        return out

    def test_doctor_has_a_row_for_every_robot_model_naming_its_deployments(self):
        from welding import doctor
        models = self._models()
        self.assertEqual(set(doctor.ROBOT_ASSETS), set(models))
        for m, aliases in models.items():
            with self.subTest(model=m):
                self.assertEqual(set(doctor.ROBOT_ASSETS[m][0]), aliases)


@unittest.skipUnless(os.environ.get("GAR_WELDING_PHYSICS_TESTS") == "1", "rigid-cell build (set GAR_WELDING_PHYSICS_TESTS=1)")
class RigidCellTest(unittest.TestCase):
    """The deployments' rigid cells (CPU MuJoCo, no weld process): the torch on the right arm's flange or clamp, its
    work point where the kinematics put it, the start pose at the procedure's angles, the G1's Dex3 as its hand."""

    def test_rigid_cells(self):
        import numpy as np
        import warp as wp
        wp.config.quiet = True
        from welding.world import World, Config
        for robot, gun_mass in (("dual-panda", 1.60), ("g1", 1.60 + .15)):
            with self.subTest(robot=robot):
                w = World(Config(process=False, robot=robot))
                self.assertEqual(w.torch_arm, "right")
                self.assertEqual(w.hand_arms, ("left",))
                o, R = w.weld_frame()
                tip, axis, tcp = w.torch_pose()
                a = R.T @ axis
                self.assertAlmostEqual(math.degrees(math.atan2(-a[2], -a[1])), 45., places=1)
                self.assertAlmostEqual(math.degrees(math.asin(a[0])), 7.5, places=1)
                self.assertLess(np.linalg.norm(w.kin.forward("right")[:3, 3] - tcp), 1e-3)
                link = w.model.body_mass.numpy()[w.scene.gun_body]
                self.assertGreater(link, gun_mass)
                self.assertEqual(w.reflexes, [])
                self.assertTrue(w.hand_bodies())

    def test_robot_cameras_on_the_rigid_cells(self):
        """Every FR3 and Panda arm carries its wrist D405's case and bracket (reference pose, 60 g + 10 g) on link7 and
        nothing else differs from the arms without them (the scene built with Scene._wrist_camera switched off); the
        case stays clear of the torch sharing that link; the G1's head camera adds nothing to the model."""
        import numpy as np
        import warp as wp
        wp.config.quiet = True
        from welding.world import World, Config
        from welding.scene import D405, Scene
        from welding.render import scene_meta
        from gym_anything_robotics.backends.newton_welding import robot_cameras

        def arrays(w):
            m = w.model
            return {k: getattr(m, k).numpy() for k in ("body_mass", "body_inertia", "body_com", "shape_body",
                                                     "shape_transform", "shape_scale", "shape_collision_group")}
        for robot in ALL:
            cams = robot_cameras(robot)
            with self.subTest(robot=robot):
                with patch.object(Scene, "_wrist_camera", lambda *args: None):
                    bare = World(Config(process=False, robot=robot, device="cpu"))
                w = World(Config(process=False, robot=robot, device="cpu"))
                a, b = arrays(bare), arrays(w)
                labels = list(w.model.body_label)
                self.assertEqual(labels, list(bare.model.body_label))
                self.assertEqual(set(w.scene.cameras), set(cams))
                self.assertEqual(set(scene_meta(w).get("mounted_cameras", {})), set(cams))
                wrists = [c for c in cams if c.endswith("_wrist_rgb")]
                shape_labels = list(w.model.shape_label)
                self.assertEqual(len(shape_labels), len(bare.model.shape_label) + 2*len(wrists))
                if not wrists:
                    for k in a:
                        self.assertTrue(np.array_equal(a[k], b[k]), k)
                if "head_rgb" in cams:
                    cam = w.scene.cameras["head_rgb"]
                    self.assertTrue(labels[cam["body"]].endswith("/torso_link"))
                    self.assertEqual(cam["fov"], 69.)
                links = set()
                for name in wrists:
                    arm = name[:-len("_wrist_rgb")]
                    cam = w.scene.cameras[name]
                    body = cam["body"]
                    links.add(body)
                    self.assertRegex(labels[body], rf"^{arm}/(fr3|panda)/(fr3|panda)_link7$")
                    self.assertAlmostEqual(float(b["body_mass"][body] - a["body_mass"][body]), .070, places=6)
                    for part in ("case", "bracket"):
                        i = shape_labels.index(f"{arm}_camera_{part}")
                        self.assertEqual(int(b["shape_body"][i]), body)
                        self.assertTrue(np.allclose(b["shape_transform"][i], [*D405[part][0], 0., 0., 0., 1.]))
                        self.assertTrue(np.allclose(b["shape_scale"][i], D405[part][1]))
                    self.assertEqual((cam["eye"], cam["target"], cam["fov"]),
                                     (list(D405["eye"]), list(D405["target"]), 87.))
                    if arm == w.torch_arm:
                        mj, m, d = w.solver._mujoco, w.solver.mj_model, w.solver.mj_data
                        g2s = np.asarray(w.solver.mjc_geom_to_newton_shape.numpy()).reshape(-1)
                        s2g = {int(sh): g for g, sh in enumerate(g2s) if sh >= 0}
                        for part in ("case", "bracket"):
                            g = s2g[shape_labels.index(f"{arm}_camera_{part}")]
                            for gun in w.scene.gun_shapes:
                                self.assertGreater(mj.mj_geomDistance(m, d, g, s2g[gun], .2, np.zeros(6)), .03)
                others = [k for k in range(len(labels)) if k not in links]
                for k in ("body_mass", "body_inertia", "body_com"):
                    self.assertTrue(np.array_equal(a[k][others], b[k][others]), k)


if __name__ == "__main__":
    unittest.main()
