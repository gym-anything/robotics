"""lockkey's wrist camera selection: the single Panda (the actions' 'left' arm) offers left_wrist_rgb only.

The environment is replaced by a double that records the wrist cameras selected while it is built; the D405 itself
is checked in tests/environments/pencap/test_wrist_camera.py.
"""

from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from gym_anything_robotics.errors import UnsupportedConfigurationError
from pencap.newton15 import wrist_camera


class Recorder:
    def __init__(self):
        self.selections = []

    def setup(self, config, contract=None):
        self.selections.append(wrist_camera._selected.get())
        return NS(config=config, joints={}, lock=threading.RLock(), tick=0, generation=0)


class BackendTests(unittest.TestCase):
    def test_left_wrist_camera_selects_the_panda_and_right_is_refused(self):
        from gym_anything_robotics.backends.newton_lockkey import NewtonLockKeyBackend
        # The real modules load before any patch, so none of them keeps a double.
        import lockkey.newton15.catalog, lockkey.newton15.env, lockkey.newton15.kinematics  # noqa: F401
        import lockkey.task_set.catalogue  # noqa: F401
        record = Recorder()
        with patch("lockkey.task_set.catalogue.task_setup", return_value=record), \
             patch("lockkey.newton15.catalog.Catalog", return_value=NS()), \
             patch("lockkey.newton15.kinematics.Kinematics", return_value=NS()), \
             tempfile.TemporaryDirectory() as tmp:
            plain = NewtonLockKeyBackend(recording=False, output_dir=Path(tmp)/"plain")
            self.assertEqual(record.selections.pop(), {})
            self.assertEqual([s.name for s in plain.capabilities.camera_specs], ["bench_rgb"])
            self.assertNotIn("wrist_cameras", plain.capabilities.notes)
            with patch.object(plain, "_render", return_value={}), \
                    self.assertRaisesRegex(ValueError, "^Render label must be room, bench, lock, keys, keyway, grasp, "
                                                       "or bench_rgb$"):
                plain.render(label="left_wrist_rgb")
            for n, cameras in enumerate((("left_wrist_rgb",), ("left_wrist_rgb", "bench_rgb"))):
                obj = NewtonLockKeyBackend(cameras=cameras, recording=False, output_dir=Path(tmp)/str(n))
                self.assertEqual(record.selections.pop(), {"panda": "left_wrist_rgb"})
                self.assertEqual(obj.capabilities.robot_cameras, cameras)
                self.assertEqual([s.name for s in obj.policy_access().cameras], list(cameras))
                self.assertEqual(obj.camera_specs["left_wrist_rgb"], wrist_camera.spec("left_wrist_rgb", 1280, 720))
                self.assertTrue(obj.capabilities.notes["wrist_cameras"].startswith("left_wrist_rgb: "))
                frames = {name: Path(tmp)/f"{name}.png" for name in cameras}
                with patch.object(obj, "_render", return_value=frames):
                    images = obj.observation_from_state({"tick": 0, "generation": 0, "time_s": 0., "joints": {}},
                                                        sequence=1).images
                    self.assertEqual({k: v.camera for k, v in images.items()},
                                     {k: obj.camera_specs[k] for k in cameras})
                    self.assertEqual(obj.render(label="left_wrist_rgb"), frames["left_wrist_rgb"])
            for bad in (("right_wrist_rgb",), ("bench_rgb", "right_wrist_rgb"), ("left_wrist_rgb", "right_wrist_rgb")):
                with self.assertRaisesRegex(UnsupportedConfigurationError, "one arm.*bench_rgb, left_wrist_rgb$"):
                    NewtonLockKeyBackend(cameras=bad, recording=False, output_dir=Path(tmp)/"bad")
            for bad in (("head_rgb",), ("left_wrist_rgb", "left_wrist_rgb")):
                with self.assertRaisesRegex(ValueError, "Policy cameras are bench_rgb, left_wrist_rgb;"):
                    NewtonLockKeyBackend(cameras=bad, recording=False, output_dir=Path(tmp)/"bad")
            self.assertFalse((Path(tmp)/"bad").exists())
            self.assertEqual(record.selections, [])


if __name__ == "__main__":
    unittest.main()
