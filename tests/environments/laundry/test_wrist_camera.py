"""laundry's wrist camera selection: each Panda's D405 is built only when its camera is selected.

The environment is replaced by a double that records the wrist cameras selected while it is built; the D405 itself
is checked in tests/environments/pencap/test_wrist_camera.py.
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
    def test_wrist_cameras_select_their_arms_and_the_bench_camera_is_unchanged(self):
        from gym_anything_robotics.backends.newton_laundry import NewtonLaundryBackend
        # The real modules load before sys.modules is patched, so they stay loaded afterwards.
        import laundry.newton15.config, laundry.task_set, laundry.task_set_v2  # noqa: F401
        selections = []

        def env(config, *, task=None, contract=None):
            selections.append(wrist_camera._selected.get())
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
        with patch.dict(sys.modules, modules), tempfile.TemporaryDirectory() as tmp:
            plain = NewtonLaundryBackend(recording=False, output_dir=Path(tmp)/"plain")
            self.assertEqual(selections.pop(), {})
            self.assertEqual([s.name for s in plain.capabilities.camera_specs], ["bench_rgb"])
            self.assertEqual(plain.capabilities.camera_specs[0].mount_frame, "external_floor_stand")
            self.assertNotIn("wrist_cameras", plain.capabilities.notes)
            with patch.object(plain, "_render", return_value={}), self.assertRaisesRegex(ValueError, "Unknown camera"):
                plain.render(label="left_wrist_rgb")
            for n, (cameras, arms) in enumerate(((("right_wrist_rgb",), {"right": "right_wrist_rgb"}),
                                                 (("bench_rgb", "left_wrist_rgb", "right_wrist_rgb"),
                                                  {"left": "left_wrist_rgb", "right": "right_wrist_rgb"}))):
                for contract in (None, "laundry-tasks-v2"):
                    options = {} if contract is None else {"task": "half_fold", "contract": contract}
                    obj = NewtonLaundryBackend(cameras=cameras, recording=False,
                                               output_dir=Path(tmp)/f"{n}-{contract}", **options)
                    self.assertEqual(selections.pop(), arms)
                    self.assertEqual(obj.capabilities.robot_cameras, cameras)
                    self.assertEqual([s.name for s in obj.policy_access().cameras], list(cameras))
                    for name in arms.values():
                        self.assertEqual(obj.camera_specs[name], wrist_camera.spec(name, 1280, 720))
                    self.assertTrue(obj.capabilities.notes["wrist_cameras"].startswith(", ".join(arms.values())))
                    frames = {name: Path(tmp)/f"{name}.png" for name in cameras}
                    with patch.object(obj, "_render", return_value=frames):
                        images = obj.observation_from_state({"tick": 0, "generation": 0, "time_s": 0.,
                                                             "joints": {}}, sequence=1).images
                        self.assertEqual({k: v.camera for k, v in images.items()},
                                         {k: obj.camera_specs[k] for k in cameras})
                        for name in arms.values():
                            self.assertEqual(obj.render(label=name), frames[name])
            policy = "Policy cameras are bench_rgb, left_wrist_rgb, right_wrist_rgb;"
            for bad in (("head_rgb",), ("right_wrist_rgb", "right_wrist_rgb")):
                with self.assertRaisesRegex(ValueError, policy):
                    NewtonLaundryBackend(cameras=bad, recording=False, output_dir=Path(tmp)/"bad")
            self.assertFalse((Path(tmp)/"bad").exists())
            self.assertEqual(selections, [])


if __name__ == "__main__":
    unittest.main()
