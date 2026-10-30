"""CPU-only schema/discovery/launch contracts; no physical claims."""

import contextlib
import io
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch
from robots import RobotSpec, RobotSetup
from robots.catalog import resolve, model, list_models, list_setups, select_for_backend
from gym_anything_robotics.launch import LaunchSpec
from gym_anything_robotics.cli import main


class CatalogTests(unittest.TestCase):
    def test_every_environment_launches_with_its_declared_default(self):
        from gym_anything_robotics import registry
        from gym_anything_robotics.launch import describe
        from robots.catalog import default_robot, environment_robots

        for item in registry.available_environments():
            name, backend = item["environment_id"], item["backend"]
            with self.subTest(environment=name, backend=backend):
                expected = default_robot(name)
                self.assertTrue(expected)
                self.assertEqual(describe(name, backend)["default_robot"], expected)
                defaults = [x["id"] for x in environment_robots(name) if x["default"]]
                self.assertEqual(defaults, [expected])
                factory = Mock()
                with patch.dict(registry._REGISTRY, {(name, backend): factory}):
                    with patch.object(registry, "RoboticsEnv"):
                        LaunchSpec(name, backend=backend).create()
                factory.assert_called_once_with(robot=expected)

    def test_roundtrips(self):
        for item in list_models():
            a = RobotSpec.from_dict(item)
            self.assertEqual(a.to_dict(), item)
            self.assertEqual(
                a.fingerprint, RobotSpec.from_dict(a.to_dict()).fingerprint
            )
        for item in list_setups():
            self.assertEqual(resolve(item).to_dict(), item)

    def test_detached(self):
        data = resolve("deposit-fr3").to_dict()
        a = RobotSetup.from_dict(data)
        data["id"] = "changed"
        self.assertNotEqual(a.id, data["id"])
        b = a.to_dict()
        b["instances"]["left"]["model"] = "bad"
        self.assertEqual(a.to_dict()["instances"]["left"]["model"], "fr3@1")

    def test_unknown_profile(self):
        x = resolve("deposit-fr3").to_dict()
        x["instances"]["left"]["controller_profile"] = "bad"
        with self.assertRaises(ValueError):
            resolve(x)

    def test_bad_pose(self):
        x = resolve("laundry-panda").to_dict()
        item = x["instances"]["left"]
        item.pop("mount")
        item["pose"] = {"orientation_wxyz": [2, 0, 0, 0]}
        with self.assertRaises(ValueError):
            resolve(x)

    def test_cycle(self):
        x = resolve("deposit-fr3").to_dict()
        x["attachments"]["left_hand"]["parent"] = "right_hand/tcp"
        x["attachments"]["right_hand"]["parent"] = "left_hand/tcp"
        with self.assertRaises(ValueError):
            resolve(x)

    def test_revision(self):
        with self.assertRaises(ValueError):
            model("fr3@999")

    def test_preset_compatibility(self):
        self.assertEqual(select_for_backend("laundry", "laundry-panda"), "dual-panda")
        with self.assertRaises(ValueError):
            select_for_backend("laundry", "g1")

    def test_deposit_structured_launch(self):
        setup = resolve("deposit-panda").to_dict()
        s = LaunchSpec("deposit-cans", robot=setup)
        self.assertEqual(LaunchSpec.from_dict(s.to_dict()).robot, setup)
        setup["id"] = "mutation"
        self.assertNotEqual(s.robot["id"], "mutation")

    def test_invalid_launch_before_physics(self):
        with self.assertRaises(ValueError):
            LaunchSpec("deposit-cans", robot="g1")
        x = resolve("deposit-fr3").to_dict()
        x["instances"]["left"]["parameters"] = {"finger_effort_limit": 900}
        with self.assertRaises(ValueError):
            LaunchSpec("deposit-cans", robot=x)

    def test_cli(self):
        for args in (
            ["robots"],
            ["robots", "list"],
            ["robots", "inspect", "fr3"],
            ["robots", "inspect", "deposit-panda", "--json"],
            ["inspect", "deposit-cans"],
        ):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(args), 0)
            self.assertTrue(output.getvalue())

    def test_discovery_does_not_load_physics(self):
        code = "import robots,sys; robots.list_models(); robots.list_setups(); assert 'newton' not in sys.modules and 'warp' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
