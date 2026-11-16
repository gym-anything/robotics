"""laundry's camera selection: the default serves the bench camera and both Pandas' D405 wrist cameras.

The environment is replaced by a double that records what each build receives; the D405 itself, on every arm whichever
cameras are selected, is checked in tests/environments/pencap/test_wrist_camera.py.
"""

from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pencap.newton15 import wrist_camera


class BackendTests(unittest.TestCase):
    def test_the_default_serves_every_robot_camera_and_any_subset_can_be_selected(self):
        from gym_anything_robotics.backends.newton_laundry import NewtonLaundryBackend
        # The real modules load before sys.modules is patched, so they stay loaded afterwards.
        import laundry.newton15.config, laundry.task_set, laundry.task_set_v2  # noqa: F401
        builds = []

        def env(config, *, task=None, contract=None):
            builds.append((config, task, contract))
            return NS(config=config, joints={}, lock=threading.RLock(), tick=0, generation=0,
                      task=NS(spec=NS(label="Fold the towel")))
        modules = {}
        for name in ("laundry.task_set_env", "laundry.task_set_v2_env"):
            modules[name] = ModuleType(name)
            modules[name].CatalogueLaundryEnv = env
        modules["laundry.newton15.catalog"] = ModuleType("laundry.newton15.catalog")
        modules["laundry.newton15.catalog"].Catalog = lambda env: NS()
        modules["laundry.newton15.kinematics"] = ModuleType("laundry.newton15.kinematics")
        modules["laundry.newton15.kinematics"].Kinematics = lambda base_y: NS()
        everything = ("bench_rgb", "left_wrist_rgb", "right_wrist_rgb")
        with patch.dict(sys.modules, modules), tempfile.TemporaryDirectory() as tmp:
            for contract in (None, "laundry-tasks-v2"):
                options = {} if contract is None else {"task": "half_fold", "contract": contract}
                default = NewtonLaundryBackend(recording=False, output_dir=Path(tmp)/f"default-{contract}", **options)
                self.assertEqual(default.capabilities.robot_cameras, everything)
                self.assertEqual([s.name for s in default.capabilities.camera_specs], list(everything))
                self.assertEqual([s.name for s in default.policy_access().cameras], list(everything))
                self.assertEqual(default.capabilities.camera_specs[0].mount_frame, "external_floor_stand")
            for n, cameras in enumerate(((), ("right_wrist_rgb",), everything)):
                for contract in (None, "laundry-tasks-v2"):
                    options = {} if contract is None else {"task": "half_fold", "contract": contract}
                    obj = NewtonLaundryBackend(cameras=cameras, recording=False,
                                               output_dir=Path(tmp)/f"{n}-{contract}", **options)
                    self.assertEqual(obj.capabilities.robot_cameras, cameras)
                    self.assertEqual([s.name for s in obj.policy_access().cameras], list(cameras))
                    for name in wrist_camera.NAMES:
                        self.assertEqual(obj.camera_specs[name], wrist_camera.spec(name, 1280, 720))
                    # Both arms carry their D405 in every session, so the note names both.
                    self.assertTrue(obj.capabilities.notes["wrist_cameras"].startswith(
                        "left_wrist_rgb, right_wrist_rgb: "))
                    frames = {name: Path(tmp)/f"{name}.png" for name in everything}
                    with patch.object(obj, "_render", side_effect=lambda names: {c: frames[c] for c in names}):
                        images = obj.observation_from_state({"tick": 0, "generation": 0, "time_s": 0.,
                                                             "joints": {}}, sequence=1).images
                        self.assertEqual({k: v.camera for k, v in images.items()},
                                         {k: obj.camera_specs[k] for k in cameras})
                        for name in wrist_camera.NAMES:
                            self.assertEqual(obj.render(label=name), frames[name])
                        with self.assertRaisesRegex(ValueError, "Unknown camera"):
                            obj.render(label="head_rgb")
            policy = "Policy cameras are bench_rgb, left_wrist_rgb, right_wrist_rgb;"
            for bad in (("head_rgb",), ("right_wrist_rgb", "right_wrist_rgb")):
                with self.assertRaisesRegex(ValueError, policy):
                    NewtonLaundryBackend(cameras=bad, recording=False, output_dir=Path(tmp)/"bad")
            self.assertFalse((Path(tmp)/"bad").exists())
            # One build per accepted session, each receiving only its configuration, task and contract: the camera
            # list does not reach the environment.
            self.assertEqual(len(builds), 2+3*2)
            self.assertEqual({(task, contract) for _, task, contract in builds},
                             {(None, None), ("half_fold", "laundry-tasks-v2")})


if __name__ == "__main__":
    unittest.main()
