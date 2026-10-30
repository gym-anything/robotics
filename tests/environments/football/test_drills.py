"""Catalogue wiring and constructed regression cases for the football drills (variants.py, drills.py).

No physics: ball-centre paths are fed tick by tick (500 Hz) as in test_goal_task.py. They test what the rules do
with a path, not whether the robot can produce it.
"""

import json
import math
import unittest

from football.football_specs import BallSpec
from football.newton15 import variants
from football.newton15.config import Config
from football.newton15.drills import (
    CONTRACT, PLAYED_AWAY_M, Clearance, CornerTarget, PassGate, PassSquare, Save, Trap, make_verifier,
)
from football.newton15.task import FAIL_SCORE_SCALE, PitchAwareGoalDetector
from tests.packages import PACKAGES

R = BallSpec().radius_m
DT = 0.002
HERE = PACKAGES / "football" / "newton15"
UPRIGHT = {"pelvis_m": (-11.42, 0.0, 0.79), "pelvis_up_z": 1.0, "feet_midpoint_m": (-11.40, 0.0, 0.05)}
FALLEN = {"pelvis_m": (-10.6, 0.0, 0.18), "pelvis_up_z": 0.1, "feet_midpoint_m": (-11.9, 0.0, 0.2)}


def line(start, end, speed):
    n = max(1, round(math.dist(start, end) / (speed * DT)))
    return [tuple(a + (b - a) * i / n for a, b in zip(start, end)) for i in range(n + 1)]


def path(*segments):
    points = list(segments[0])
    for segment in segments[1:]:
        points += segment[1:]
    return points


def rest(point, seconds):
    return [point] * (round(seconds / DT) + 1)


def feed(detector, points, touches=None, robot=None, velocities=None):
    """Feed a path; velocities default to finite differences; `robot` is a pose dict or a function of the tick."""
    touches = touches or {}
    for i, p in enumerate(points):
        if velocities is not None:
            v = velocities[i]
        else:
            v = (0.0, 0.0, 0.0) if i == 0 else tuple((b - a) / DT for a, b in zip(points[i - 1], p))
        if robot is not None:
            detector.robot = robot(i) if callable(robot) else robot
        if i == 0 and detector.previous_position_m is None:
            detector.update(0.0, p, v)
        else:
            detector.update_supported(i * DT, p, v, normal_force_n=0.0, compression_m=0.0,
                                      robot_touches=touches.get(i, ()))
    return detector


def drill(task, seed=0):
    return make_verifier(Config(task=task, seed=seed), variants.layout(task, seed))


class CatalogueTests(unittest.TestCase):
    def test_benchmark_names_the_resolver_and_every_task(self):
        bench = json.loads((HERE / "benchmark.json").read_text())
        self.assertEqual(bench["selectors"], ["task", "seed"])
        self.assertEqual(bench["resolver"], "football.newton15.variants:task_resolve")
        self.assertEqual(bench["selector_choices"]["task"], list(variants.TASKS))
        self.assertEqual(bench["count"], len(variants.TASKS))
        self.assertEqual(bench["default_task_id"], variants.task_id("goal"))
        self.assertEqual(variants.TASKS[0], "goal")

    def test_empty_selection_is_the_default_goal_launch(self):
        from gym_anything_robotics.backends.newton_football import DESCRIPTION

        resolved = variants.task_resolve({})
        self.assertEqual((resolved["task_id"], resolved["backend_options"]), ("football/goal", {}))
        self.assertEqual(resolved["description"], DESCRIPTION)
        self.assertEqual(variants.task_resolve({"task": "goal", "seed": 0})["backend_options"], {})
        goal = variants.layout("goal")
        self.assertEqual((goal["props"], goal["markers"]), ([], []))
        default = Config()
        for name, value in goal["config"].items():
            self.assertEqual(tuple(value) if isinstance(value, tuple) else value, getattr(default, name), name)
        self.assertIs(type(make_verifier(Config(), goal)), PitchAwareGoalDetector)
        self.assertTrue(make_verifier(Config(), goal).require_kick)

    def test_selections_resolve_to_backend_options(self):
        for task in variants.TASKS[1:]:
            with self.subTest(task=task):
                r = variants.task_resolve({"task": task})
                self.assertEqual(r["backend_options"], {"task": task, "seed": 0})
                self.assertEqual(r["task_id"], f"football/{task}")
                self.assertEqual(r["description"], variants.INSTRUCTIONS[task])
        self.assertEqual(variants.task_resolve({"task": "trap", "seed": 12})["task_id"], "football/trap/seed-12")

    def test_invalid_selections_are_refused(self):
        for task, seed in (("dribble", 0), ("goal", 1), ("corner-target", 2), ("free-kick", 1), ("trap", -1),
                           ("trap", 1000), ("save", True), ("clearance", 1.5)):
            with self.subTest(task=task, seed=seed), self.assertRaises(ValueError):
                variants.validate(task, seed)
        with self.assertRaises(ValueError):
            Config(task="dribble")
        with self.assertRaises(ValueError):
            Config(seed=3)

    def test_apply_sets_the_start_state_and_keeps_other_fields(self):
        config = variants.apply(Config(dt=0.002, net_layout="back"), "save", 0)
        layout = variants.layout("save")
        self.assertEqual((config.task, config.seed, config.net_layout), ("save", 0, "back"))
        for name, value in layout["config"].items():
            self.assertEqual(getattr(config, name), value)

    def test_seed_zero_geometry_is_the_specified_one(self):
        gate = [m["position_m"][:2] for m in variants.layout("pass-gate")["markers"]]
        for got, want in zip(gate, ((-6.4, -2.2), (-7.6, -3.8))):
            self.assertAlmostEqual(math.dist(got, want), 0.0, places=9)
        square = sorted(m["position_m"][:2] for m in variants.layout("pass-square")["markers"])
        self.assertEqual(square, [(-6.0, 1.5), (-6.0, 3.5), (-4.0, 1.5), (-4.0, 3.5)])
        clearance = variants.layout("clearance")
        self.assertEqual(clearance["config"]["ball_position"][:2], (-10.0, 1.0))
        self.assertEqual(clearance["config"]["robot_position"], (-9.58, 1.0, 0.80))
        self.assertAlmostEqual(clearance["config"]["robot_yaw"], math.pi)
        line_marker = clearance["markers"][0]
        self.assertAlmostEqual(line_marker["position_m"][0] - line_marker["size_m"][0] / 2, -16.5)
        trap = variants.layout("trap")["config"]
        self.assertEqual(trap["ball_position"][:2], (-5.0, -0.12))
        self.assertAlmostEqual(trap["ball_omega"][1] * R, trap["ball_velocity"][0])  # pure rolling
        save = variants.layout("save")["config"]
        self.assertEqual((save["robot_position"], save["robot_yaw"]), ((-0.45, 0.0, 0.80), math.pi))
        sheet = {p["name"]: p for p in variants.layout("corner-target")["props"]}
        top, middle = sheet["target_sheet_top"], sheet["target_sheet_middle"]
        self.assertAlmostEqual(top["position_m"][2] - top["half_extents_m"][2], 1.00)
        self.assertAlmostEqual(middle["half_extents_m"][1], 2.40)
        self.assertAlmostEqual(top["position_m"][0] - top["half_extents_m"][0], 0.065)

    def test_seeds_are_deterministic_and_in_range(self):
        for seed in range(1, 60):
            self.assertEqual(variants.layout("trap", seed), variants.layout("trap", seed))
            c = variants.layout("clearance", seed)["summary"]
            self.assertTrue(-1.0 <= c["ball_y_m"] <= 1.0)
            self.assertTrue(25.0 <= variants.layout("pass-gate", seed)["summary"]["angle_right_deg"] <= 45.0)
            t = variants.layout("trap", seed)["summary"]
            self.assertTrue(4.0 <= t["speed_m_s"] <= 4.6 and -0.20 <= t["aim_y_m"] <= -0.05)
            cx, cy = variants.layout("pass-square", seed)["summary"]["square_centre_m"]
            distance, angle = math.hypot(cx + 11.0, cy), math.degrees(math.atan2(cy, cx + 11.0))
            self.assertTrue(5.5 <= distance <= 7.5 and 15.0 <= angle <= 30.0)
            y, z = variants.layout("save", seed)["summary"]["aim_m"]
            self.assertTrue(0.6 <= y <= 0.9 and 0.2 <= z <= 0.8)


class ClearanceTests(unittest.TestCase):
    START = (-10.0, 1.0, R + 0.001)
    LINE = -16.5 - R

    def test_untouched_and_idle_push_earn_nothing(self):
        e = feed(drill("clearance"), rest(self.START, 1.0)).evaluation()
        self.assertEqual((e["success"], e["score"], e["progress"]), (False, 0.0, 0.0))
        self.assertEqual(e["reward"]["contract"], CONTRACT)
        # goal's idle fall: thighs on the ball, pushed 0.44 m and pinned
        push = path(rest(self.START, 1.0), line(self.START, (-10.44, 1.0, 0.17), 1.0), rest((-10.44, 1.0, 0.17), 1))
        touches = {i: ("left_hip_yaw_link", "right_hip_yaw_link") for i in range(500, len(push))}
        self.assertEqual(feed(drill("clearance"), push, touches).evaluation()["score"], 0.0)

    def test_a_kick_past_the_line_clears(self):
        points = line(self.START, (-20.0, 1.0, R), 8.0)
        d = feed(drill("clearance"), points, {1: ("right_ankle_roll_link",)})
        e = d.evaluation()
        self.assertTrue(e["success"])
        self.assertEqual((e["score"], e["reward"]["outcome"]), (1.0, "cleared"))
        self.assertAlmostEqual(e["reward"]["line_crossing"]["center_world_m"][0], self.LINE, places=9)

    def test_partial_credit_is_the_distance_played_beyond_a_metre(self):
        points = path(line(self.START, (-13.0, 1.0, R), 6.0), rest((-13.0, 1.0, R), 0.5))
        e = feed(drill("clearance"), points, {1: ("left_knee_link",)}).evaluation()
        d0, d = -10.0 - self.LINE, -13.0 - self.LINE
        self.assertAlmostEqual(e["progress"], 1.0 - d / (d0 - PLAYED_AWAY_M), places=9)
        self.assertAlmostEqual(e["score"], FAIL_SCORE_SCALE * e["progress"], places=12)

    def test_trunk_last_is_not_a_clearance_and_own_goal_scores_zero(self):
        points = line(self.START, (-20.0, 1.0, R), 8.0)
        e = feed(drill("clearance"), points, {1: ("right_ankle_roll_link",), 300: ("torso_link",)}).evaluation()
        self.assertEqual((e["success"], e["reward"]["outcome"]), (False, "not_kicked"))
        self.assertLess(e["progress"], 1.0)
        own = feed(drill("clearance"), line(self.START, (0.6, 0.5, R), 10.0), {1: ("right_ankle_roll_link",)})
        self.assertEqual((own.evaluation()["reward"]["outcome"], own.evaluation()["score"]), ("own_goal", 0.0))

    def test_handball_and_leaving_the_sector_freeze_progress(self):
        points = path(line(self.START, (-14.0, 1.0, R), 6.0), line((-14.0, 1.0, R), (-12.0, 1.0, R), 2.0))
        first = len(line(self.START, (-14.0, 1.0, R), 6.0))
        e = feed(drill("clearance"), points, {1: ("right_ankle_roll_link",), first: ("left_hand_index_0_link",)})
        self.assertEqual(e.evaluation()["reward"]["outcome"], "handball")
        before = 1.0 - (points[first - 1][0] - self.LINE) / (-10.0 - self.LINE - PLAYED_AWAY_M)
        self.assertAlmostEqual(e.evaluation()["progress"], before, places=9)
        out = feed(drill("clearance"), line(self.START, (-14.0, 13.0, R), 8.0), {1: ("right_ankle_roll_link",)})
        self.assertEqual(out.evaluation()["reward"]["outcome"], "out_of_sector")
        self.assertTrue(out.evaluation()["reward"]["attempt_over"])


class PassGateTests(unittest.TestCase):
    SPOT = variants.SPOT

    def test_ground_pass_through_the_gate_succeeds(self):
        points = line(self.SPOT, (-5.4, -4.2, R), 6.0)
        e = feed(drill("pass-gate"), points, {1: ("right_ankle_roll_link",)}).evaluation()
        self.assertTrue(e["success"])
        self.assertEqual(e["reward"]["outcome"], "through_gate")
        self.assertAlmostEqual(e["reward"]["gate_crossing"]["along_gate_m"], 0.0, places=6)

    def test_wide_and_lofted_passes_fail_with_their_miss(self):
        d = feed(drill("pass-gate"), line(self.SPOT, (-3.0, -3.0, R), 6.0), {1: ("right_ankle_roll_link",)})
        e = d.evaluation()
        self.assertEqual(e["reward"]["outcome"], "wide_of_gate")
        miss = e["reward"]["gate_crossing"]["miss_m"]
        self.assertGreater(miss, 0.0)
        self.assertAlmostEqual(e["progress"], max(0.0, 1.0 - miss / (5.0 - PLAYED_AWAY_M)), places=9)
        lob = path(line(self.SPOT, (-7.0, -3.0, 0.9), 6.0), line((-7.0, -3.0, 0.9), (-5.4, -4.2, R), 6.0))
        e = feed(drill("pass-gate"), lob, {1: ("right_ankle_roll_link",)}).evaluation()
        self.assertEqual((e["success"], e["reward"]["outcome"]), (False, "over_gate"))

    def test_progress_toward_the_gate_and_untouched_start(self):
        self.assertEqual(feed(drill("pass-gate"), rest(self.SPOT, 0.5)).evaluation()["progress"], 0.0)
        points = path(line(self.SPOT, (-8.6, -1.8, R), 5.0), rest((-8.6, -1.8, R), 0.2))
        e = feed(drill("pass-gate"), points, {1: ("right_ankle_roll_link",)}).evaluation()
        self.assertAlmostEqual(e["progress"], 1.0 - 2.0 / 4.0, places=6)  # 3 m of 5 m played, 1 m not counted


class TrapTests(unittest.TestCase):
    START = (-5.0, -0.12, R + 0.001)
    FOOT = (-11.30, -0.12, R)

    def received(self, touches, robot=UPRIGHT, settle=(-11.0, -0.12, R), hold=1.2):
        points = path(line(self.START, self.FOOT, 2.5), line(self.FOOT, settle, 0.3), rest(settle, hold))
        index = len(line(self.START, self.FOOT, 2.5)) - 1
        return feed(drill("trap"), points, {index + k: t for k, t in touches.items()}, robot=robot), index

    def test_a_pass_rolling_past_earns_nothing(self):
        points = line(self.START, (-14.0, -0.12, R), 2.5)
        e = feed(drill("trap"), points, robot=UPRIGHT).evaluation()
        self.assertEqual((e["success"], e["progress"]), (False, 0.0))

    def test_a_cushioned_stop_at_the_feet_while_upright_succeeds(self):
        for links in (("right_ankle_roll_link",), ("right_knee_link",), ("torso_link",)):
            with self.subTest(links=links):
                d, _ = self.received({0: links})
                e = d.evaluation()
                self.assertTrue(e["success"])
                self.assertEqual(e["reward"]["outcome"], "controlled")

    def test_the_hold_takes_a_second(self):
        d, _ = self.received({0: ("right_ankle_roll_link",)}, hold=0.9)
        e = d.evaluation()
        self.assertFalse(e["success"])
        self.assertGreater(e["progress"], 0.9)

    def test_a_fallen_robot_earns_nothing_and_handball_freezes(self):
        d, _ = self.received({0: ("right_ankle_roll_link",)}, robot=FALLEN)
        self.assertEqual((d.evaluation()["success"], d.evaluation()["progress"]), (False, 0.0))
        d, index = self.received({0: ("right_ankle_roll_link",), 50: ("right_hand_thumb_0_link",)})
        e = d.evaluation()
        self.assertEqual((e["success"], e["reward"]["outcome"]), (False, "handball"))

    def test_partial_credit_factors(self):
        points = path(line(self.START, self.FOOT, 2.5), line(self.FOOT, (-9.0, -0.12, R), 1.0))
        index = len(line(self.START, self.FOOT, 2.5)) - 1
        e = feed(drill("trap"), points, {index: ("right_ankle_roll_link",)}, robot=UPRIGHT).evaluation()
        f = e["reward"]["factors"]
        d0 = math.dist(self.START[:2], UPRIGHT["feet_midpoint_m"][:2])
        self.assertAlmostEqual(f["C"], 1.0 - (f["feet_distance_m"] - 1.0) / (d0 - 1.0), places=9)
        self.assertAlmostEqual(e["progress"], f["T"] * f["C"] * f["S"] * f["U"], places=12)
        self.assertEqual((f["T"], f["U"]), (1.0, 1.0))


class PassSquareTests(unittest.TestCase):
    SPOT = variants.SPOT
    INSIDE = (-5.2, 2.4, R)

    def test_ball_stopping_in_the_square_succeeds_after_a_second(self):
        points = path(line(self.SPOT, self.INSIDE, 5.0), rest(self.INSIDE, 1.1))
        e = feed(drill("pass-square"), points, {1: ("right_ankle_roll_link",)}).evaluation()
        self.assertTrue(e["success"])
        self.assertEqual(e["reward"]["outcome"], "stopped_in_square")

    def test_overhit_scores_low_while_fast_and_by_where_it_rests(self):
        points = line(self.SPOT, (-2.0, 3.5, R), 6.0)
        half = len(points) // 2 + 120
        moving = feed(drill("pass-square"), points[:half], {1: ("right_ankle_roll_link",)}).evaluation()
        self.assertEqual(moving["reward"]["factors"]["distance_to_square_m"], 0.0)
        self.assertEqual(moving["progress"], 0.0)  # inside at 6 m/s: S = 0
        long = feed(drill("pass-square"), path(points, rest(points[-1], 1.2)), {1: ("right_ankle_roll_link",)})
        e = long.evaluation()
        self.assertFalse(e["success"])
        e0 = math.hypot(5.0, 1.5)
        self.assertAlmostEqual(e["progress"], 1.0 - 2.0 / (e0 - PLAYED_AWAY_M), places=6)

    def test_trunk_last_caps_and_cannot_succeed(self):
        points = path(line(self.SPOT, self.INSIDE, 5.0), rest(self.INSIDE, 1.5))
        touches = {1: ("right_ankle_roll_link",), 600: ("pelvis",)}
        e = feed(drill("pass-square"), points, touches).evaluation()
        self.assertFalse(e["success"])
        self.assertLess(e["progress"], 1.0)
        self.assertEqual(feed(drill("pass-square"), rest(self.SPOT, 1.0)).evaluation()["progress"], 0.0)


class CornerTargetTests(unittest.TestCase):
    SPOT = variants.SPOT

    def test_goal_through_a_corner_opening(self):
        for y in (3.0, -3.0, 2.52, -3.54):
            with self.subTest(y=y):
                points = path(line(self.SPOT, (-1.0, y, 0.4), 15.0), line((-1.0, y, 0.4), (0.6, y, 0.4), 15.0))
                e = feed(drill("corner-target"), points, {1: ("right_ankle_roll_link",)}).evaluation()
                self.assertTrue(e["success"])
                self.assertEqual(e["reward"]["outcome"], "goal_through_opening")
                self.assertTrue(e["reward"]["sheet_entry"]["through_opening"])

    def test_the_covered_goal_and_misses_fail(self):
        # a path through the sheet is impossible physically (the panel stops it); the guard rejects it anyway
        centre = feed(drill("corner-target"), line(self.SPOT, (0.6, 0.0, 0.4), 15.0), {1: ("right_ankle_roll_link",)})
        self.assertEqual(centre.evaluation()["reward"]["outcome"], "not_through_opening")
        self.assertFalse(centre.evaluation()["success"])
        high = feed(drill("corner-target"), line(self.SPOT, (0.6, 3.0, 1.5), 15.0), {1: ("right_ankle_roll_link",)})
        self.assertEqual(high.evaluation()["reward"]["outcome"], "not_through_opening")
        wide = feed(drill("corner-target"), line(self.SPOT, (0.6, 4.2, 0.4), 15.0), {1: ("right_ankle_roll_link",)})
        e = wide.evaluation()
        self.assertEqual(e["reward"]["outcome"], "wide_or_over")
        d0 = math.hypot(0.06 + R + 11.0, 2.40 + R)
        self.assertAlmostEqual(e["reward"]["start_distance_to_opening_m"], d0, places=9)
        self.assertLess(e["score"], FAIL_SCORE_SCALE)

    def test_idle_scale_progress(self):
        push = path(rest(self.SPOT, 1.0), line(self.SPOT, (-10.56, 0.0, 0.17), 1.0))
        touches = {i: ("left_hip_yaw_link",) for i in range(500, len(push))}
        self.assertLess(feed(drill("corner-target"), push, touches).evaluation()["score"], 0.05)


class SaveTests(unittest.TestCase):
    SPOT = variants.SPOT

    def shot(self, *segments):
        return path(*segments)

    def test_untouched_reference_shot_is_conceded_with_nothing(self):
        e = feed(drill("save"), line(self.SPOT, (0.6, 0.75, 0.3), 10.0)).evaluation()
        self.assertEqual((e["success"], e["score"], e["reward"]["outcome"]), (False, 0.0, "goal_conceded"))

    def test_deflections_and_stops_are_saves(self):
        towards = line(self.SPOT, (-0.6, 0.72, 0.5), 10.0)
        at = len(towards) - 1
        wide = self.shot(towards, line((-0.6, 0.72, 0.5), (-0.3, 5.0, 0.4), 6.0), line((-0.3, 5.0, 0.4),
                                                                                          (0.6, 5.0, 0.4), 6.0))
        e = feed(drill("save"), wide, {at: ("right_hand_index_0_link",)}).evaluation()
        self.assertEqual((e["success"], e["reward"]["outcome"]), (True, "saved_out_over_goal_line"))
        stopped = self.shot(towards, line((-0.6, 0.72, 0.5), (-1.0, 0.8, R), 1.0), rest((-1.0, 0.8, R), 1.1))
        e = feed(drill("save"), stopped, {at: ("right_knee_link",)}).evaluation()
        self.assertEqual((e["success"], e["reward"]["outcome"]), (True, "saved_ball_at_rest"))
        parried = self.shot(towards, line((-0.6, 0.72, 0.5), (-7.0, 2.0, R), 8.0))
        e = feed(drill("save"), parried, {at: ("pelvis",)}).evaluation()
        self.assertEqual((e["success"], e["reward"]["outcome"]), (True, "saved_cleared_upfield"))
        slow = self.shot(towards, line((-0.6, 0.72, 0.5), (-0.3, 0.7, R), 0.08))
        e = feed(drill("save"), slow, {at: ("left_ankle_roll_link",)}).evaluation()
        self.assertEqual((e["success"], e["reward"]["outcome"]), (True, "saved_until_deadline"))

    def test_a_slowed_goal_earns_partial_credit_and_a_live_parry_reads_half(self):
        towards = line(self.SPOT, (-0.6, 0.72, 0.5), 10.0)
        at = len(towards) - 1
        slowed = self.shot(towards, line((-0.6, 0.72, 0.5), (0.6, 0.7, 0.3), 4.0))
        e = feed(drill("save"), slowed, {at: ("right_hand_index_0_link",)}).evaluation()
        self.assertEqual(e["reward"]["outcome"], "goal_conceded")
        self.assertAlmostEqual(e["progress"], 0.5 * (1.0 - e["reward"]["goal_speed_m_s"] / 10.0), delta=0.02)
        back = self.shot(towards, line((-0.6, 0.72, 0.5), (-3.0, 1.5, R), 5.0))
        e = feed(drill("save"), back, {at: ("right_elbow_link",)}).evaluation()
        self.assertEqual((e["success"], e["progress"], e["reward"]["attempt_over"]), (False, 0.5, False))


class BackendSelectionTests(unittest.TestCase):
    def test_select_applies_the_task_and_refuses_conflicts(self):
        from gym_anything_robotics.backends.newton_football import select_task

        config = select_task(Config(net_layout="enclosure"), "trap", 3)
        self.assertEqual((config.task, config.seed), ("trap", 3))
        self.assertEqual(config.ball_velocity, variants.layout("trap", 3)["config"]["ball_velocity"])
        self.assertEqual(select_task(Config(dt=0.002), "goal", 0, frozenset({"dt"})), Config(dt=0.002))
        with self.assertRaises(ValueError):
            select_task(Config(ball_position=(-9.0, 0.0, 0.2)), "pass-gate", 0, frozenset({"ball_position"}))
        with self.assertRaises(ValueError):
            select_task(Config(), "goal", 4)

    def test_reset_refuses_bad_selections_without_touching_the_world(self):
        from types import SimpleNamespace
        from gym_anything_robotics.backends.newton_football import DESCRIPTION, NewtonFootballBackend
        from gym_anything_robotics.errors import ActionRejectedError

        backend = NewtonFootballBackend.__new__(NewtonFootballBackend)
        backend.env, backend.trace = SimpleNamespace(config=Config(), tick=5, fault=None), None
        for kwargs in ({"options": {"variant": "x"}}, {"options": {"task": "dribble"}}, {"seed": 2},
                       {"options": {"task": "free-kick"}, "seed": 1}):
            with self.subTest(**kwargs), self.assertRaises(ActionRejectedError):
                backend.reset(**kwargs)
        self.assertEqual((backend.env.tick, backend.env.config), (5, Config()))
        chosen = backend.select_variation({"task": "save", "seed": 4})
        self.assertEqual(chosen["reset_request"], {"seed": 4, "options": {"task": "save"}})
        self.assertEqual(chosen["task"], variants.INSTRUCTIONS["save"])
        self.assertEqual(backend.description, DESCRIPTION)
        self.assertEqual(backend.variation_contract()["current"], {"task": "goal", "seed": 0})


class CheckpointTests(unittest.TestCase):
    def test_drill_state_survives_a_checkpoint(self):
        from gym_anything_robotics.physics_trace import pack, unpack
        from football.newton15.checkpoint import decode, encode

        start, foot = TrapTests.START, TrapTests.FOOT
        points = path(line(start, foot, 2.5), line(foot, (-11.0, -0.12, R), 0.3), rest((-11.0, -0.12, R), 1.2))
        index = len(line(start, foot, 2.5)) - 1
        cut = index + 200
        first = feed(drill("trap"), points[:cut], {index: ("right_ankle_roll_link",)}, robot=UPRIGHT)
        data = vars(first).copy()
        data.pop("ball"), data.pop("field")
        restored = drill("trap")
        vars(restored).update(decode(unpack(pack(encode(data)))["data"]))
        self.assertEqual(json.dumps(restored.evaluation()), json.dumps(first.evaluation()))
        for detector in (first, restored):
            for i in range(cut, len(points)):
                detector.robot = UPRIGHT
                v = tuple((b - a) / DT for a, b in zip(points[i - 1], points[i]))
                detector.update_supported(i * DT, points[i], v, normal_force_n=0.0, compression_m=0.0)
        self.assertEqual(json.dumps(restored.evaluation()), json.dumps(first.evaluation()))
        self.assertTrue(first.evaluation()["success"])

    def test_every_task_builds_its_verifier(self):
        kinds = {"goal": PitchAwareGoalDetector, "free-kick": PitchAwareGoalDetector, "clearance": Clearance,
                 "pass-gate": PassGate, "trap": Trap, "pass-square": PassSquare, "corner-target": CornerTarget,
                 "save": Save}
        for task, kind in kinds.items():
            with self.subTest(task=task):
                verifier = drill(task)
                self.assertIs(type(verifier), kind)
                self.assertEqual(verifier.require_kick, task not in ("trap", "save"))
                json.dumps(feed(verifier, rest(variants.layout(task)["config"]["ball_position"], 0.1),
                                robot=UPRIGHT if task == "trap" else None).evaluation())


if __name__ == "__main__":
    unittest.main()
