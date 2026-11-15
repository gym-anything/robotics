"""Wrist D405 policy camera (panda_wrist_rgb): selection, refusal, the accessory's geometry and mass, the mounted
render camera, and that a session without it builds the released laboratory.

Only TitrationWristModelTest builds Newton models (CPU); it needs the pinned Panda assets and is skipped without them.
"""

from contextlib import ExitStack
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from pxr import Gf, Usd, UsdGeom

from titrate.newton15 import wrist_camera
from titrate.newton15.lab_config import LabConfig

from gym_anything_robotics.errors import UnsupportedConfigurationError
from gym_anything_robotics.types import CameraSpec


class Session:
    instances = []

    def __init__(self, config=None, **kwargs):
        self.config = config or LabConfig()
        self.kwargs = kwargs
        self.dt_s = kwargs.get('dt_s', .001)
        self.output = Path('/unused')
        self.renderer_options = kwargs.get('renderer_options')
        self.liquid = None
        self.closed, self.tick, self.trace_liquid_stride = False, 0, 20
        self.world = NS(lab=NS(robot_joints={}))
        self.instances.append(self)

    def close(self):
        self.closed = True

    def get_state(self):
        return {'tick': self.tick}


class BackendCameraTest(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        Session.instances = []
        stack.enter_context(patch.dict(sys.modules, {
            'titrate.newton15.session': NS(ApparatusSession=Session),
            'titrate.newton15.geometry_api': NS(GeometryCatalog=lambda world: NS()),
            'titrate.newton15.session_trace': NS(SessionTrace=Mock()),
            'whiteboard.storage': NS(Store=Mock()),
        }))

    def backend(self, **kwargs):
        from gym_anything_robotics.backends.newton_titration import NewtonTitrationBackend
        return NewtonTitrationBackend(apparatus_fixture=True, recording=False, **kwargs)

    def test_default_and_bench_only_sessions_are_unchanged(self):
        bench = CameraSpec("bench_rgb", "authored_external_stand", 1280, 720, 1280, 720, (.036, .036*720/1280),
                           (.028, .028))
        for cameras, specs in ((None, (bench,)), (("bench_rgb",), (bench,)), ((), ())):
            b = self.backend(**({} if cameras is None else {'cameras': cameras}))
            self.assertNotIn('wrist_cameras', Session.instances[-1].kwargs)
            self.assertEqual(b.capabilities.camera_specs, specs)
            self.assertEqual(b.policy_access().cameras, specs)
            self.assertNotIn('wrist_camera', b.capabilities.notes)
            b.reset(options={'variation_id': 3})
            self.assertNotIn('wrist_cameras', Session.instances[-1].kwargs)

    def test_wrist_camera_spec_session_and_reset(self):
        b = self.backend(cameras=("bench_rgb", "panda_wrist_rgb"), renderer_options={'width': 640, 'height': 480})
        self.assertEqual(Session.instances[-1].kwargs['wrist_cameras'], ("panda_wrist_rgb",))
        focal = .018/math.tan(math.radians(87/2))
        wrist = CameraSpec("panda_wrist_rgb", "link7_d405_envelope", 640, 480, 1280, 720, (.036, .036*480/640),
                           (focal, focal))
        self.assertEqual(b.capabilities.robot_cameras, ("bench_rgb", "panda_wrist_rgb"))
        self.assertEqual(b.capabilities.camera_specs[1], wrist)
        self.assertEqual(b.policy_access().cameras[1], wrist)
        self.assertEqual(b.capabilities.camera_specs[0].name, "bench_rgb")
        note = b.capabilities.notes['wrist_camera']
        for text in ("D405", "87 deg", "1280x720", "60 g case", "10 g", "without it"):
            self.assertIn(text, note)
        # A reset to another variation builds the new session with the same accessory.
        b.reset(options={'variation_id': 7})
        self.assertEqual(Session.instances[-1].kwargs['wrist_cameras'], ("panda_wrist_rgb",))

    def test_absent_arm_and_unknown_cameras_are_refused_before_building(self):
        for cameras in (("left_wrist_rgb",), ("bench_rgb", "right_wrist_rgb"), ("panda_wrist_rgb", "left_wrist_rgb")):
            with self.assertRaisesRegex(UnsupportedConfigurationError, "one arm, panda"):
                self.backend(cameras=cameras)
        for cameras in (("head_rgb",), ("panda_wrist_rgb", "panda_wrist_rgb"), ("overview",)):
            with self.assertRaises(ValueError):
                self.backend(cameras=cameras)
        self.assertEqual(Session.instances, [])


class Lab:
    def __init__(self, robots):
        self.builder = NS(_gar_robots={arm: NS(frames={'wrist': {'body': body}}) for arm, body in robots.items()})
        self.boxes = []

    def box(self, name, pos, size, material, body=-1, density=0.):
        self.boxes.append(dict(name=name, pos=pos, size=size, material=material, body=body, density=density))


class InstallTest(unittest.TestCase):
    def test_reference_case_bracket_and_optics_on_the_wrist_link(self):
        lab = Lab({'panda': 18})
        mounts = wrist_camera.install(lab, ("panda_wrist_rgb",))
        case, bracket = lab.boxes
        # drilling/scene.py _wrist_camera: centres, half extents, 60 g case and 10 g bracket on link7.
        self.assertEqual((case['name'], case['body'], case['pos'], case['material']),
                         ("panda_camera_case", 18, (.075, 0., .07), "camera_dark"))
        self.assertEqual(tuple(s/2 for s in case['size']), (.021, .021, .0115))
        self.assertEqual(case['density'], .060/(.042*.042*.023))
        self.assertEqual((bracket['name'], bracket['body'], bracket['pos'], bracket['material']),
                         ("panda_camera_bracket", 18, (.05, 0., .046), "aluminum"))
        self.assertEqual(tuple(s/2 for s in bracket['size']), (.025, .013, .003))
        self.assertEqual(bracket['density'], .010/(.050*.026*.006))
        for box, mass in ((case, .060), (bracket, .010)):
            self.assertAlmostEqual(box['density']*np.prod(box['size']), mass, places=15)
        self.assertEqual(mounts, {"panda_wrist_rgb": dict(body=18, eye=(.075, -.012, .083), target=(.075, -.012, .283),
                                                          up=(0., 1., 0.), horizontal_fov_deg=87.)})
        self.assertIs(lab.camera_mounts, mounts)

    def test_only_selected_arms_and_no_absent_arm(self):
        lab = Lab({'panda': 18})
        self.assertEqual(wrist_camera.install(lab, ()), {})
        self.assertEqual(lab.boxes, [])
        with self.assertRaisesRegex(ValueError, "No arm carries left_wrist_rgb"):
            wrist_camera.install(Lab({'panda': 18}), ("left_wrist_rgb",))


class SessionCameraNamesTest(unittest.TestCase):
    def session(self, lab):
        from titrate.newton15.session import ApparatusSession
        s = ApparatusSession.__new__(ApparatusSession)
        s._lock, s.closed, s.world = threading.RLock(), True, NS(lab=lab)
        return s

    def test_mounted_camera_is_an_authored_camera_only_where_mounted(self):
        with self.assertRaisesRegex(ValueError, "authored cameras"):
            self.session(NS()).observe(("panda_wrist_rgb",))
        # Past the name check, the closed session refuses.
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.session(NS(camera_mounts={"panda_wrist_rgb": {}})).observe(("bench_rgb", "panda_wrist_rgb"))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.session(NS()).observe(("bench_rgb",))


def pose_body(stage, index, position, quat_xyzw):
    prim = UsdGeom.Xform.Define(stage, f"/Lab/body_{index}")
    prim.AddTranslateOp().Set(Gf.Vec3d(*position))
    prim.AddOrientOp().Set(Gf.Quatf(quat_xyzw[3], Gf.Vec3f(*quat_xyzw[:3])))


class MountedRenderCameraTest(unittest.TestCase):
    def stage(self):
        from titrate.newton15 import usd_lab
        stage = Usd.Stage.CreateInMemory()
        # The wrist link turned 90 deg about world x: its +z (the optical axis) points along world -y.
        q = (math.sin(math.pi/4), 0., 0., math.cos(math.pi/4))
        pose_body(stage, 18, (.3, -.1, 1.2), q)
        mount = dict(body=18, **wrist_camera.OPTICS, horizontal_fov_deg=87.)
        usd_lab.mounted_camera(stage, "panda_wrist_rgb", mount)
        usd_lab.camera(stage, "bench_rgb", (.68, -.60, 1.65), (.15, .035, 1.18), 28., role="policy")
        return stage

    def build(self, stage, name):
        from titrate.newton15.transport_scene import LaboratoryScene
        scene = LaboratoryScene.__new__(LaboratoryScene)
        scene.stage, scene.prims, scene.materials, scene.mi = stage, [], {}, MagicMock()
        with patch('titrate.newton15.volume_transport.medium', lambda field: None):
            scene.build(name, 1280, 720, None, None)
        sensor = scene.mi.load_dict.call_args[0][0]['sensor']
        return sensor, np.asarray(scene.mi.ScalarTransform4f.call_args[0][0])

    def test_wrist_camera_follows_its_body_with_d405_fov(self):
        stage = self.stage()
        camera = UsdGeom.Camera(stage.GetPrimAtPath("/Lab/body_18/panda_wrist_rgb"))
        self.assertEqual(camera.GetPrim().GetCustomDataByKey("observation_role"), "policy")
        self.assertEqual(stage.GetPrimAtPath("/Cameras/panda_wrist_rgb").GetCustomDataByKey("mounted_camera_path"),
                         "/Lab/body_18/panda_wrist_rgb")
        sensor, to_world = self.build(stage, "panda_wrist_rgb")
        self.assertAlmostEqual(sensor['fov'], 87., places=4)
        self.assertEqual(sensor['fov_axis'], 'x')
        # Body frame eye (.075, -.012, .083) in the world, looking along the body's +z (world -y), image up along the
        # body's +y (world +z); Mitsuba looks along its +z.
        np.testing.assert_allclose(to_world[:3, 3], (.3+.075, -.1-.083, 1.2-.012), atol=1e-6)
        np.testing.assert_allclose(to_world[:3, 2], (0., -1., 0.), atol=1e-6)
        np.testing.assert_allclose(to_world[:3, 1], (0., 0., 1.), atol=1e-6)

    def test_authored_cameras_resolve_as_before_and_bad_mounts_are_refused(self):
        stage = self.stage()
        sensor, to_world = self.build(stage, "bench_rgb")
        np.testing.assert_allclose(to_world[:3, 3], (.68, -.60, 1.65), atol=1e-6)
        self.assertAlmostEqual(sensor['fov'], math.degrees(2*math.atan(18/28)), places=4)
        with self.assertRaisesRegex(ValueError, "Unknown camera"):
            self.build(stage, "head_rgb")
        UsdGeom.Xform.Define(stage, "/Cameras/elsewhere").GetPrim().SetCustomDataByKey("mounted_camera_path", "/Other")
        with self.assertRaisesRegex(ValueError, "Invalid mounted camera path"):
            self.build(stage, "elsewhere")


def panda_assets():
    try:
        from robots.assets import resolve
        import newton  # noqa: F401
        resolve("panda")
        return True
    except Exception:
        return False


@unittest.skipUnless(panda_assets(), "needs Newton and the pinned Panda assets")
class TitrationWristModelTest(unittest.TestCase):
    """CPU build: the wrist model is the released model plus two shapes on panda_link7 and 70 g on that link."""

    @classmethod
    def setUpClass(cls):
        from titrate.newton15.mechanics import LaboratoryMechanics
        config = LabConfig(device="cpu")
        cls.plain = LaboratoryMechanics(config)
        cls.wrist = wrist_camera.WristCameraMechanics(config, ("panda_wrist_rgb",))

    def test_same_construction_and_released_model_plus_the_accessory(self):
        plain, wrist = self.plain, self.wrist
        self.assertEqual(set(vars(plain)), set(vars(wrist)))
        self.assertEqual((wrist.model.body_count, wrist.model.joint_count), (plain.model.body_count,
                                                                             plain.model.joint_count))
        n = plain.model.shape_count
        self.assertEqual(wrist.model.shape_count, n+2)
        b = wrist.lab.builder
        link7 = b._gar_robots["panda"].frames["wrist"]["body"]
        self.assertEqual(b.body_label[link7], "panda/panda_link7")
        self.assertEqual(b.shape_label[n:], ["panda_camera_case", "panda_camera_bracket"])
        self.assertEqual(list(b.shape_body[n:]), [link7, link7])
        np.testing.assert_allclose(np.asarray(b.shape_transform[n])[:3], (.075, 0., .07))
        np.testing.assert_allclose(np.asarray(b.shape_transform[n+1])[:3], (.05, 0., .046))
        for name in ("shape_transform", "shape_scale", "shape_body", "shape_flags", "shape_collision_group",
                     "shape_material_mu", "shape_type"):
            np.testing.assert_array_equal(getattr(plain.model, name).numpy(), getattr(wrist.model, name).numpy()[:n])
        mass = wrist.model.body_mass.numpy()-plain.model.body_mass.numpy()
        self.assertAlmostEqual(mass[link7], .070, places=6)
        self.assertEqual(np.count_nonzero(mass), 1)
        for name in ("body_q", "joint_q"):
            np.testing.assert_array_equal(getattr(plain.state, name).numpy(), getattr(wrist.state, name).numpy())
        self.assertEqual(wrist.lab.camera_mounts["panda_wrist_rgb"]["body"], link7)
        self.assertFalse(hasattr(plain.lab, "camera_mounts"))
