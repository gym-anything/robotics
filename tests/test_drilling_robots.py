"""Drilling robot configurations: every qualified rig is accepted, every other configuration is refused with a stated
reason whose pointers hold what they are cited for (pure Python: the catalogue, drilling/robot_setup.py and the task
check).

The path tested is robots.catalog.select_for_backend, which `gar run --robot` and gar.create use before the backend."""

from __future__ import annotations

import copy
import unittest
from tests.packages import PACKAGES


REPO = PACKAGES
QUALIFIED = {"dual-fr3": "drill-fr3", "dual-panda": "drill-panda", "single-fr3": "drill-fr3-single",
             "single-panda": "drill-panda-single"}
SINGLE = ("single-fr3", "single-panda")
SINGLE_TASKS = ("punch-layout", "press-drill", "press-blind")
REPORT = "drilling/REPORT.pdf"


def select(robot):
    from robots.catalog import select_for_backend
    return select_for_backend("drilling", robot)


def refusal(test, robot):
    with test.assertRaises(ValueError) as ctx:
        select(robot)
    return str(ctx.exception)


class DrillingRobotsTest(unittest.TestCase):
    def test_qualified_aliases_and_rigs(self) -> None:
        from robots.catalog import ENVIRONMENTS, resolve
        item = ENVIRONMENTS["drilling"]
        self.assertEqual(item["aliases"], QUALIFIED)
        self.assertEqual(item["default"], "dual-fr3")
        for alias, rig in QUALIFIED.items():
            self.assertTrue((REPO/"robots"/"rigs"/f"{rig}.json").is_file(), rig)   # the rig file the catalogue names
            self.assertEqual(select(alias), alias)
            self.assertEqual(select(rig), alias)                          # the rig id resolves to its alias
            d = resolve(rig).to_dict()
            self.assertEqual(select(d), alias)                            # the gar.robot-setup.v1 object itself
            renamed = copy.deepcopy(d)
            renamed["id"] = f"copy-of-{rig}"
            self.assertEqual(select(renamed), alias)                      # content, not the id, is compared

    def test_every_unsupported_alias_has_a_reason(self) -> None:
        from robots.catalog import ENVIRONMENTS
        unsupported = ENVIRONMENTS["drilling"]["unsupported"]
        for name in ("g1", "mobile-fr3-duo", "mobile-fr3-single", "panda", "dual-panda-wsg50", "yam"):
            self.assertIn(name, unsupported)
        self.assertNotIn("single-panda", unsupported)                      # a qualified rig
        for name, reason in unsupported.items():
            self.assertGreater(len(reason), 40, name)
            self.assertNotIn("REPORT.md", reason)
            self.assertNotIn("section 5", reason, name)                    # a section that never mentioned them
            msg = refusal(self, name)
            self.assertIn(f"drilling does not support robot {name!r}: ", msg)
            self.assertIn(reason, msg)
        self.assertIn("5 N m", unsupported["g1"])                          # the G1 wrist pitch/yaw rating
        self.assertIn("rocker", unsupported["mobile-fr3-single"])
        self.assertIn("single-panda", unsupported["panda"])                # the name to use instead
        self.assertIn("Franka Hand", unsupported["dual-panda-wsg50"])


    def test_unknown_name_lists_the_choices(self) -> None:
        msg = refusal(self, "no-such-robot")
        self.assertIn("drilling has no robot setup 'no-such-robot'", msg)
        for alias in QUALIFIED:
            self.assertIn(alias, msg)
        self.assertIn(REPORT, msg)

    def test_custom_setups_are_refused_with_reasons(self) -> None:
        from robots.catalog import resolve
        single = resolve("drill-fr3-single").to_dict()
        dual = resolve("drill-fr3").to_dict()
        cases = {}
        a = copy.deepcopy(single)                                          # a single Panda equals the qualified rig
        a["id"], a["instances"]["right"]["model"] = "t-single-panda", "panda@1"
        a["instances"]["right"]["controller_profile"] = "drilling"         # the Panda profile with the one hand
        self.assertEqual(select(a), "single-panda")
        g = copy.deepcopy(a)                                               # the Panda's own barcode_scanning profile:
        g["id"] = "t-single-panda-weak-hand"                               # the hand limited to 20 N per finger
        g["instances"]["right"]["controller_profile"] = "barcode_scanning"
        cases["panda with the 20 N hand profile"] = (g, "limits the hand to 20 N per finger")
        h = copy.deepcopy(dual)                                            # and the FR3 has no 'drilling' profile
        h["id"] = "t-fr3-drilling-profile"
        h["instances"]["left"]["controller_profile"] = "drilling"
        cases["fr3 with the panda's profile name"] = (h, "controller profile 'drilling' for fr3@1")
        f = copy.deepcopy(a)                                               # ... but not on the left mount
        f["id"] = "t-single-panda-left"
        f["instances"] = {"left": dict(a["instances"]["right"], mount="left_arm_mount")}
        f["attachments"] = {"left_hand": dict(a["attachments"]["right_hand"], parent="left/flange")}
        cases["single panda, left mount"] = (f, "the single arm on the right mount")
        b = copy.deepcopy(dual)                                            # an end-effector swap
        b["id"] = "t-fr3-wsg50"
        for k in b["attachments"]:
            b["attachments"][k]["model"] = "wsg50@1"
        cases["wsg50"] = (b, "built and measured for the Franka Hand (franka-hand@1) only")
        c = copy.deepcopy(dual)                                            # mixed arm models
        c["id"], c["instances"]["left"]["model"] = "t-mixed", "panda@1"
        cases["mixed"] = (c, "arms of one model")
        d = copy.deepcopy(dual)                                            # another controller profile
        d["id"] = "t-profile"
        d["instances"]["right"]["controller_profile"] = "wine_pouring"
        cases["profile"] = (d, "controller profile 'wine_pouring' for fr3@1: the arm servos are those qualified in "
                               "barcode_scanning and the Franka Hand's controller is the same on both arm models")
        e = copy.deepcopy(dual)                                            # one arm on the left mount only
        e["id"] = "t-left-only"
        del e["instances"]["right"]
        e["attachments"] = {k: v for k, v in e["attachments"].items() if v["parent"].startswith("left/")}
        cases["left only"] = (e, "the single arm on the right mount")
        for name, (setup, expected) in cases.items():
            msg = refusal(self, setup)
            self.assertIn(expected, msg, name)
            self.assertNotIn("Unknown mount profile", msg, name)
            self.assertNotIn("REPORT.md", msg, name)


    def test_tasks_per_configuration(self) -> None:
        from drilling.robot_setup import check_task
        from drilling.variants import PUBLIC
        for preset, tasks in PUBLIC.items():
            for task in tasks:
                check_task("dual-fr3", preset, task)
                check_task("dual-panda", preset, task)
                for single, arm in zip(SINGLE, ("FR3", "Panda")):
                    if task in SINGLE_TASKS:
                        check_task(single, preset, task)
                        continue
                    with self.assertRaises(ValueError) as ctx:
                        check_task(single, preset, task)
                    msg = str(ctx.exception)
                    self.assertIn("needs two arms: ", msg)
                    self.assertIn(f"one {arm}'s 2.27 kg tool payload" if preset == "wall" else "one Franka Hand pinch", msg)
                    self.assertIn(REPORT, msg)
                    self.assertNotIn("section 5", msg)
                    self.assertIn(f"{single} is qualified for " + ", ".join(SINGLE_TASKS), msg)

    def test_launch_check_decides_robot_and_task_before_a_session(self) -> None:
        """drilling.robot_setup.check_launch is the one call that refuses an impossible robot/task pair from the
        launch options alone (no engine, no session folder); the backend applies the same two checks when built."""
        from drilling.robot_setup import check_launch
        self.assertEqual(check_launch("dual-fr3"), "dual-fr3")
        self.assertEqual(check_launch("drill-panda", {"preset": "wall", "task": "ledger-anchor", "seed": 3}), "dual-panda")
        for single in SINGLE:
            for task in SINGLE_TASKS:
                self.assertEqual(check_launch(single, {"preset": "bench", "task": task, "seed": 0}), single)
                self.assertEqual(check_launch(single, {"task": task}), single)          # the preset follows the task
                self.assertEqual(check_launch(single, {"config": {"preset": "bench", "task": task}}), single)
            for options in ({"preset": "wall", "task": "anchor-holes", "seed": 0}, {"task": "match-bolt"},
                            {"preset": "wall"}, {"preset": "bench"}, {}, None):
                with self.assertRaises(ValueError, msg=f"{single} {options}") as ctx:
                    check_launch(single, options)                                       # default launch: wall preset
                self.assertIn("needs two arms: ", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            check_launch("g1", {"preset": "bench", "task": "press-drill"})
        self.assertIn("drilling does not support robot 'g1'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
