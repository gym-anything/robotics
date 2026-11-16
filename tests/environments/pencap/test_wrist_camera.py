"""The D405 wrist cameras of the Panda simulations (pencap/newton15/wrist_camera.py) and pencap's camera selection.

The accessory checks build on a Newton ModelBuilder on the CPU (no simulator step); the Panda itself only where its
pinned description assets are installed. The backend checks replace the environment with a recording double: the
default session serves every robot camera, and the environment is built the same whichever cameras are selected.
"""

import math
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pencap.newton15 import wrist_camera

try:
    import newton
    import warp as wp
    HAVE_NEWTON = True
except Exception:  # pragma: no cover - the engine venv is required for the builder checks
    HAVE_NEWTON = False
try:
    from pxr import Gf, Usd, UsdGeom
    HAVE_USD = True
except Exception:  # pragma: no cover
    HAVE_USD = False

FOCAL_M = .018/math.tan(math.radians(87/2))


def panda_assets():
    try:
        from robots import assets
        assets.resolve("panda")
        return True
    except Exception:
        return False


def chain():
    """link6 -> link7 -> finger, one box each, registered as robot instance 'left' with link7 as its wrist."""
    b = newton.ModelBuilder()
    bodies = []
    for name, z in (("link6", 0.), ("link7", .1), ("finger", .2)):
        body = b.add_link(xform=wp.transform((0., 0., z), wp.quat_identity()), label=f"left/panda/panda_{name}",
                          mass=1.)
        b.add_shape_box(body, hx=.03, hy=.03, hz=.03, label=f"left/panda_{name}")
        bodies.append(body)
    joints = [b.add_joint_revolute(-1, bodies[0])]
    joints += [b.add_joint_revolute(bodies[i], bodies[i+1], parent_xform=wp.transform((0., 0., .1),
                                                                                       wp.quat_identity()))
               for i in range(2)]
    b.add_articulation(joints)
    b._gar_robots = {"left": NS(frames={"wrist": {"body": bodies[1]}})}
    return b, bodies


def contact():
    return newton.ModelBuilder.ShapeConfig(mu=.4, ke=1e7, kd=5000., gap=4e-5)


@unittest.skipUnless(HAVE_NEWTON, "Newton engine not importable in this interpreter")
class AccessoryTests(unittest.TestCase):
    def test_every_arm_gets_the_reference_case_and_bracket_on_link7(self):
        b, bodies = chain()
        cfg, materials = contact(), {}
        mass = b.body_mass[bodies[1]]
        wrist_camera.install(b, "left", cfg, materials)
        case, bracket = b.shape_count-2, b.shape_count-1
        self.assertEqual([b.shape_label[case], b.shape_label[bracket]], ["left_camera_case", "left_camera_bracket"])
        self.assertEqual([b.shape_body[case], b.shape_body[bracket]], [bodies[1]]*2)
        for shape, center, half in ((case, (.075, 0., .07), (.021, .021, .0115)),
                                    (bracket, (.05, 0., .046), (.025, .013, .003))):
            self.assertEqual(b.shape_type[shape], newton.GeoType.BOX)
            for got, want in zip(list(b.shape_transform[shape]), (*center, 0., 0., 0., 1.)):
                self.assertAlmostEqual(got, want, places=7)
            for got, want in zip(list(b.shape_scale[shape]), half):
                self.assertAlmostEqual(got, want, places=7)
            self.assertAlmostEqual(b.shape_material_mu[shape], .4, places=6)
            self.assertEqual((b.shape_material_ke[shape], b.shape_material_kd[shape]), (1e7, 5000.))
        # 60 g D405 case and 10 g bracket join link7; the caller's settings are not changed.
        self.assertAlmostEqual(b.body_mass[bodies[1]]-mass, .070, places=6)
        self.assertEqual(cfg.density, newton.ModelBuilder.ShapeConfig().density)
        self.assertEqual(materials, {case: "robot_dark", bracket: "aluminum"})
        self.assertEqual(b.wrist_cameras, {"left_wrist_rgb": bodies[1]})
        # Collisions with the arm's own neighbours (link6, the fingers) are filtered as for link7's own shapes.
        filtered = {b.shape_body[j if i == shape else i] for shape in (case, bracket)
                    for i, j in b.shape_collision_filter_pairs if shape in (i, j)}
        self.assertTrue({bodies[0], bodies[2]} <= filtered)

    def test_a_named_camera_is_recorded_under_its_name(self):
        # lockkey's single Panda is robot instance 'panda'; the actions call it 'left'.
        b, bodies = chain()
        b._gar_robots = {"panda": b._gar_robots["left"]}
        wrist_camera.install(b, "panda", contact(), {}, "left_wrist_rgb")
        self.assertEqual(b.wrist_cameras, {"left_wrist_rgb": bodies[1]})
        self.assertEqual(list(b.shape_label[-2:]), ["panda_camera_case", "panda_camera_bracket"])

    def test_spec_and_note(self):
        spec = wrist_camera.spec("right_wrist_rgb", 640, 480)
        self.assertEqual((spec.name, spec.mount_frame), ("right_wrist_rgb", "link7_d405_envelope"))
        self.assertEqual((spec.width_px, spec.height_px, spec.native_width_px, spec.native_height_px),
                         (640, 480, 1280, 720))
        self.assertEqual(spec.sensor_size_m, (.036, .036*480/640))
        self.assertEqual(spec.focal_length_m, (FOCAL_M, FOCAL_M))
        self.assertAlmostEqual(2*math.degrees(math.atan(.018/spec.focal_length_m[0])), 87.)
        note = wrist_camera.note(["left_wrist_rgb"])
        self.assertTrue(note.startswith("left_wrist_rgb: Intel RealSense D405"))
        self.assertIn("60 g case and its 10 g bracket", note)
        self.assertIn("whichever cameras the session selects", note)

    @unittest.skipUnless(panda_assets(), "pinned Panda description assets are not installed")
    def test_on_the_panda_the_parts_ride_panda_link7_beside_the_hand(self):
        from robots.backends.newton import add_robot, mount
        from pencap.newton15.runtime import require_engine
        require_engine()
        b = newton.ModelBuilder()
        add_robot(b, "panda", "pencap", instance="left", placement=mount((0., .55, .75), -math.pi/2))
        before = (b.shape_count, list(b.body_mass))
        wrist_camera.install(b, "left", contact(), {})
        self.assertEqual(b.shape_count, before[0]+2)
        link7 = b.wrist_cameras["left_wrist_rgb"]
        self.assertTrue(b.body_label[link7].endswith("panda_link7"))
        self.assertAlmostEqual(b.body_mass[link7]-before[1][link7], .070, places=6)
        filtered = {b.body_label[b.shape_body[j if i == b.shape_count-2 else i]]
                    for i, j in b.shape_collision_filter_pairs if b.shape_count-2 in (i, j)}
        self.assertTrue(any(n.endswith("panda_link6") for n in filtered))
        self.assertTrue(any(n.endswith("finger") for n in filtered))


@unittest.skipUnless(HAVE_USD, "OpenUSD (pxr) not importable in this interpreter")
class ExportTests(unittest.TestCase):
    def test_camera_rides_its_body_prim_with_the_alias_the_render_workers_resolve(self):
        stage = Usd.Stage.CreateInMemory()
        body = UsdGeom.Xform.Define(stage, "/Lab/body_8")
        body.AddTranslateOp().Set(Gf.Vec3d(.3, -.2, 1.1))
        body.AddOrientOp().Set(Gf.Quatf(.5, Gf.Vec3f(.5, .5, .5)))
        wrist_camera.export(stage, NS())
        self.assertFalse(stage.GetPrimAtPath("/Cameras/left_wrist_rgb"))
        wrist_camera.export(stage, NS(wrist_cameras={"left_wrist_rgb": 8}))
        alias = stage.GetPrimAtPath("/Cameras/left_wrist_rgb")
        path = alias.GetCustomDataByKey("mounted_camera_path")
        self.assertEqual(path, "/Lab/body_8/left_wrist_rgb")
        camera = UsdGeom.Camera(stage.GetPrimAtPath(path))
        self.assertEqual(camera.GetPrim().GetCustomDataByKey("observation_role"), "policy")
        self.assertAlmostEqual(camera.GetFocalLengthAttr().Get(), 1e3*FOCAL_M, places=4)
        self.assertAlmostEqual(camera.GetHorizontalApertureAttr().Get(), 36.)
        # In the body frame: optical centre at the case front, looking along +z with image up +y.
        local = camera.GetLocalTransformation()
        for got, want in zip(local.ExtractTranslation(), wrist_camera.EYE):
            self.assertAlmostEqual(got, want, places=6)
        for direction, want in (((0, 0, -1), (0, 0, 1)), ((0, 1, 0), (0, 1, 0))):
            for got, w in zip(local.TransformDir(Gf.Vec3d(*direction)), want):
                self.assertAlmostEqual(got, w, places=6)


class Recorder:
    """Stands in for the pen-cap environment and records what each build receives."""
    def __init__(self):
        self.builds = []

    def __call__(self, config, task="seat-cap"):
        self.builds.append((config, task))
        return NS(config=config, task=task, joints={}, lock=threading.RLock(), tick=0, generation=0)


class BackendTests(unittest.TestCase):
    def backends(self):
        from gym_anything_robotics.backends.newton_pencap import NewtonPenCapBackend
        from pencap.task_backend import NewtonPenCapTaskBackend
        return NewtonPenCapBackend, NewtonPenCapTaskBackend

    def test_the_default_serves_every_robot_camera_and_any_subset_can_be_selected(self):
        # The real modules load before any patch: pencap.task_env subclasses PenCapEnv at import.
        import pencap.newton15.catalog, pencap.newton15.env, pencap.newton15.kinematics, pencap.task_env  # noqa: F401
        backends = self.backends()
        record = Recorder()
        everything = ("bench_rgb", "left_wrist_rgb", "right_wrist_rgb")
        with patch("pencap.newton15.env.PenCapEnv", side_effect=record), \
             patch("pencap.task_env.TaskPenCapEnv", side_effect=record), \
             patch("pencap.newton15.catalog.Catalog", return_value=NS()), \
             patch("pencap.newton15.kinematics.Kinematics", return_value=NS()), \
             tempfile.TemporaryDirectory() as tmp:
            for i, backend in enumerate(backends):
                default = backend(recording=False, output_dir=Path(tmp)/f"{i}-default")
                self.assertEqual(default.capabilities.robot_cameras, everything)
                self.assertEqual([s.name for s in default.capabilities.camera_specs], list(everything))
                self.assertEqual([s.name for s in default.policy_access().cameras], list(everything))
                self.assertEqual(default.capabilities.camera_specs[0].mount_frame, "floor_stand_D435_enclosure")
                for n, cameras in enumerate(((), ("left_wrist_rgb",), ("bench_rgb", "right_wrist_rgb"),
                                             ("right_wrist_rgb", "bench_rgb", "left_wrist_rgb"))):
                    obj = backend(cameras=cameras, recording=False, output_dir=Path(tmp)/f"{i}-{n}")
                    self.assertEqual(obj.capabilities.robot_cameras, cameras)
                    self.assertEqual([s.name for s in obj.capabilities.camera_specs], list(cameras))
                    self.assertEqual([s.name for s in obj.policy_access().cameras], list(cameras))
                    for spec in obj.capabilities.camera_specs:
                        if spec.name != "bench_rgb":
                            self.assertEqual(spec, wrist_camera.spec(spec.name, 1280, 720))
                    # Both arms carry their D405 in every session, so the note names both.
                    self.assertTrue(obj.capabilities.notes["wrist_cameras"].startswith(
                        "left_wrist_rgb, right_wrist_rgb: "))
                    frames = {name: Path(tmp)/f"{name}.png" for name in everything}
                    with patch.object(obj, "_render", side_effect=lambda names: {c: frames[c] for c in names}):
                        state = {"tick": 0, "generation": 0, "time_s": 0., "joints": {}}
                        observation = obj.observation_from_state(state, sequence=1)
                        for name in cameras:
                            self.assertEqual(observation.images[name].camera, obj.camera_specs[name])
                        for name in wrist_camera.NAMES:
                            self.assertEqual(obj.render(label=name), frames[name])
                        with self.assertRaisesRegex(ValueError, "Render label must be"):
                            obj.render(label="head_rgb")
                for bad in (("head_rgb",), ("left_wrist_rgb", "left_wrist_rgb")):
                    with self.assertRaisesRegex(ValueError, "Policy cameras are bench_rgb, left_wrist_rgb, "
                                                            "right_wrist_rgb"):
                        backend(cameras=bad, recording=False, output_dir=Path(tmp)/f"{i}-bad")
                for cameras in (everything, ("left_wrist_rgb",)):
                    with self.assertRaisesRegex(ValueError, "requires the studio scene"):
                        backend(config={"scene": "baseline"}, cameras=cameras, recording=False,
                                output_dir=Path(tmp)/f"{i}-baseline")
                with self.assertRaisesRegex(ValueError, "requires the studio scene"):
                    backend(config={"scene": "baseline"}, recording=False, output_dir=Path(tmp)/f"{i}-baseline")
                backend(config={"scene": "baseline"}, cameras=(), recording=False,
                        output_dir=Path(tmp)/f"{i}-physics-only")
            # One build per accepted session, each receiving only its configuration and task: the camera list does
            # not reach the environment.
            self.assertEqual(len(record.builds), 2*6)
            self.assertEqual({(c.scene, t) for c, t in record.builds}, {("studio", "seat-cap"),
                                                                        ("baseline", "seat-cap")})


if __name__ == "__main__":
    unittest.main()
