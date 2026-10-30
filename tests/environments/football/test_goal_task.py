"""Constructed regression cases for the football/goal verifier and reward (football/newton15/task.py).

Ball-centre paths are fed tick by tick (500 Hz) with no physics. They test what the rules do with a path, not
whether the robot can produce it.
"""

import json
import math
import unittest

from football.football_specs import BallSpec, FieldSpec
from football.football_task import GoalDetector
from football.newton15.task import FAIL_SCORE_SCALE, REWARD_CONTRACT, PitchAwareGoalDetector, link_group

R = BallSpec().radius_m
DT = 0.002
WHOLE = 0.06 + R
SPOT = (-11.0, 0.0, R + 0.001)
D0 = WHOLE - SPOT[0]
G1_BODIES = (
    "pelvis left_hip_pitch_link left_hip_roll_link left_hip_yaw_link left_knee_link left_ankle_pitch_link "
    "left_ankle_roll_link right_hip_pitch_link right_hip_roll_link right_hip_yaw_link right_knee_link "
    "right_ankle_pitch_link right_ankle_roll_link waist_yaw_link waist_roll_link torso_link left_shoulder_pitch_link "
    "left_shoulder_roll_link left_shoulder_yaw_link left_elbow_link left_wrist_roll_link left_wrist_pitch_link "
    "left_wrist_yaw_link left_hand_thumb_0_link left_hand_thumb_1_link left_hand_thumb_2_link "
    "left_hand_middle_0_link left_hand_middle_1_link left_hand_index_0_link left_hand_index_1_link "
    "right_shoulder_pitch_link right_shoulder_roll_link right_shoulder_yaw_link right_elbow_link "
    "right_wrist_roll_link right_wrist_pitch_link right_wrist_yaw_link right_hand_thumb_0_link "
    "right_hand_thumb_1_link right_hand_thumb_2_link right_hand_middle_0_link right_hand_middle_1_link "
    "right_hand_index_0_link right_hand_index_1_link"
).split()


def line(start, end, speed):
    n = max(1, round(math.dist(start, end) / (speed * DT)))
    return [tuple(a + (b - a) * i / n for a, b in zip(start, end)) for i in range(n + 1)]


def path(*segments):
    points = list(segments[0])
    for segment in segments[1:]:
        points += segment[1:]
    return points


def feed(detector, points, touches=None, force=0.0, compression=0.0):
    touches = touches or {}
    for i, p in enumerate(points):
        v = (0.0, 0.0, 0.0) if i == 0 else tuple((b - a) / DT for a, b in zip(points[i - 1], p))
        if i == 0 and detector.previous_position_m is None:
            detector.update(0.0, p, v)
        else:
            detector.update_supported(i * DT, p, v, normal_force_n=force, compression_m=compression,
                                      robot_touches=touches.get(i, ()))
    return detector


def kicked(points, at=1, link="right_ankle_roll_link", extra=None):
    """A robot-world detector fed a path that the robot touched with `link` at tick `at`."""
    touches = {at: (link,)}
    touches.update(extra or {})
    return feed(PitchAwareGoalDetector(require_kick=True), points, touches)


def shot_through(y0=0.0, z0=1.0, angle_deg=0.0, rise_deg=0.0, speed=15.0, x0=-0.5, x1=0.5):
    """Straight path whose centre passes x = 0 at (y0, z0)."""
    ty, tz = math.tan(math.radians(angle_deg)), math.tan(math.radians(rise_deg))
    return line((x0, y0 + ty * x0, z0 + tz * x0), (x1, y0 + ty * x1, z0 + tz * x1), speed)


class GoalRuleTests(unittest.TestCase):
    def test_ordinary_goal_is_decided_at_the_back_edge_of_the_line(self):
        d = feed(PitchAwareGoalDetector(), shot_through())
        self.assertTrue(d.crossing.scored)
        self.assertAlmostEqual(d.crossing.center_world_m[0], WHOLE, places=9)
        self.assertEqual(d.evaluation()["score"], 1.0)

    def test_goals_beside_the_frame_score_although_the_net_has_yielded(self):
        # centre inside the frame where it passes the posts/bar; the undeflected line then runs beyond the
        # net plane (the net yields). The base GoalDetector's plane test counts these as outside.
        cases = (
            dict(y0=3.535, z0=0.6, angle_deg=17.9),        # 6.5 mm from the inner post face
            dict(y0=3.520, z0=0.6, angle_deg=17.9),        # 21 mm
            dict(y0=3.535, z0=R + 0.001, angle_deg=17.9),  # ground shot into the side rail and net
            dict(y0=0.0, z0=2.320, rise_deg=15.0),         # 10 mm under the bar
            dict(y0=-2.9, z0=1.8),                         # off-centre goal
        )
        for case in cases:
            with self.subTest(**case):
                points = shot_through(**case)
                self.assertTrue(feed(PitchAwareGoalDetector(), points).crossing.scored)
                self.assertTrue(kicked(points).evaluation()["success"])
        base = GoalDetector()
        for i, p in enumerate(shot_through(y0=3.535, z0=0.6, angle_deg=17.9)):
            base.update(i * DT, p, (15.0, 0.0, 0.0))
        self.assertFalse(base.crossing.scored)

    def test_balls_passing_outside_the_frame_do_not_score_and_the_miss_is_final(self):
        for case in (dict(y0=3.90, z0=0.6, angle_deg=17.9), dict(y0=-3.95, z0=R + 0.001),
                     dict(y0=0.0, z0=2.70, rise_deg=10.0)):
            with self.subTest(**case):
                d = kicked(shot_through(**case))
                self.assertFalse(d.crossing.scored)
                self.assertEqual(d.outcome, "wide_or_over")
                self.assertTrue(d.evaluation()["reward"]["attempt_over"])
        # wholly over the line wide, then brought back and rolled into the goal: still out of play
        points = path(line((-0.5, 4.2, R), (0.5, 4.2, R), 10.0), line((0.5, 4.2, R), (-1.0, 0.0, R), 5.0),
                      line((-1.0, 0.0, R), (0.6, 0.0, R), 8.0))
        d = kicked(points)
        self.assertFalse(d.evaluation()["success"])
        self.assertAlmostEqual(d.crossing.center_world_m[1], 4.2, places=6)

    def test_ball_on_the_drawn_line_is_not_a_goal(self):
        # trailing surface comes to rest 20 mm onto the 0.12 m line; and one that reaches 10 mm and comes back
        resting = path(line((-1.0, 0.0, R), (0.02 + R, 0.0, R), 3.0), [(0.02 + R, 0.0, R)] * 50)
        d = kicked(resting)
        self.assertIsNone(d.crossing)
        self.assertFalse(d.evaluation()["success"])
        self.assertLess(d.evaluation()["score"], FAIL_SCORE_SCALE)
        back = path(line((-1.0, 0.0, R), (0.01 + R, 0.0, R), 3.0), line((0.01 + R, 0.0, R), (-0.48, 0.0, R), 2.0))
        d = kicked(back)
        self.assertIsNone(d.crossing)
        self.assertIsNone(d.frame_entry)
        self.assertFalse(d.evaluation()["success"])

    def test_goal_stays_a_goal_when_the_ball_comes_back_out(self):
        points = path(shot_through(x1=0.4), line((0.4, 0.0, 1.0), (-2.0, 0.5, 0.2), 6.0))
        self.assertTrue(kicked(points).evaluation()["success"])

    def test_in_off_the_post(self):
        # enters inside after deflecting off the inside of the post in front of the frame plane
        points = path(line((-3.0, 2.5, 0.5), (-0.08, 3.53, 0.5), 15.0), line((-0.08, 3.53, 0.5), (0.6, 3.2, 0.5), 12.0))
        self.assertTrue(kicked(points).evaluation()["success"])

    def test_compliant_ground_goal_needs_pitch_support(self):
        low = R - 0.0039
        points = line((-0.5, 0.0, low), (0.5, 0.0, low), 12.0)
        self.assertTrue(feed(PitchAwareGoalDetector(), points, force=10.0, compression=0.004).crossing.scored)
        self.assertFalse(feed(PitchAwareGoalDetector(), points).crossing.scored)

    def test_fast_shot_crossing_both_planes_in_one_tick(self):
        d = PitchAwareGoalDetector()
        d.update(0.0, (-0.05, 1.0, 1.0), (150.0, 0.0, 0.0))
        d.update_supported(DT, (0.25, 1.0, 1.0), (150.0, 0.0, 0.0), normal_force_n=0.0, compression_m=0.0)
        self.assertTrue(d.crossing.scored)
        self.assertAlmostEqual(d.crossing.time_s, DT * (WHOLE + 0.05) / 0.30, places=12)

    def test_recorded_margins_are_measured_to_the_decision_boundaries(self):
        # a goal that glanced the post must not report a negative margin
        glance_in = kicked(line((-0.5, 3.553, 0.6), (0.5, 3.553, 0.6), 15.0))
        entry = glance_in.evaluation()["reward"]["frame_entry"]
        self.assertTrue(glance_in.evaluation()["success"])
        self.assertAlmostEqual(entry["post_axis_margin_m"], 3.72 - 3.553, places=9)
        self.assertAlmostEqual(glance_in.crossing.lateral_clearance_m, entry["post_axis_margin_m"], places=12)
        self.assertAlmostEqual(glance_in.crossing.upper_clearance_m, 2.50 - 0.6, places=9)
        glance_out = kicked(line((-0.5, 3.886, 0.6), (0.5, 3.886, 0.6), 15.0))
        self.assertEqual(glance_out.outcome, "wide_or_over")
        self.assertAlmostEqual(glance_out.crossing.lateral_clearance_m, 3.72 - 3.886, places=9)
        low = R - 0.0039
        points = line((-0.5, 0.0, low), (0.5, 0.0, low), 12.0)
        supported = feed(PitchAwareGoalDetector(), points, force=10.0, compression=0.004).frame_entry
        self.assertAlmostEqual(supported["floor_margin_m"], 0.006 - 0.0039, places=9)
        self.assertLess(feed(PitchAwareGoalDetector(), points).frame_entry["floor_margin_m"], 0.0)

    def test_ball_over_the_bar_that_rolls_back_off_the_roof_is_still_in_play(self):
        # over the bar at the frame, but never wholly over the line; then kicked in
        points = path(line((-0.5, 0.0, 2.55), (0.12, 0.0, 2.70), 10.0), line((0.12, 0.0, 2.70), (-1.0, 0.0, R), 4.0),
                      line((-1.0, 0.0, R), (0.6, 0.0, R), 8.0))
        late = len(points) - 100
        d = kicked(points, extra={late: ("left_ankle_roll_link",)})
        self.assertTrue(d.evaluation()["success"])

    def test_entry_that_comes_back_out_is_forgotten(self):
        # in at the frame, back out before wholly over the line, then out wide
        points = path(line((-0.5, 3.0, 0.5), (0.10, 3.0, 0.5), 10.0), line((0.10, 3.0, 0.5), (-0.5, 4.3, 0.5), 5.0),
                      line((-0.5, 4.3, 0.5), (0.5, 4.3, 0.5), 10.0))
        d = kicked(points)
        self.assertEqual(d.outcome, "wide_or_over")
        self.assertAlmostEqual(d.frame_entry["center_world_m"][1], 4.3, places=6)


class KickRuleTests(unittest.TestCase):
    def test_any_leg_link_may_kick(self):
        for link in ("right_ankle_roll_link", "left_ankle_roll_link", "right_knee_link", "left_hip_pitch_link"):
            with self.subTest(link=link):
                self.assertTrue(kicked(shot_through(), link=link).evaluation()["success"])

    def test_handball_ends_the_attempt(self):
        points = line(SPOT, (0.6, 0.0, R), 12.0)
        d = kicked(points, extra={400: ("left_hand_index_1_link",)})
        e = d.evaluation()
        self.assertFalse(e["success"])
        self.assertIsNone(d.crossing)
        self.assertEqual(e["reward"]["outcome"], "handball")
        self.assertTrue(e["reward"]["attempt_over"])
        before = math.dist(points[399], (WHOLE, 0.0, points[399][2]))
        self.assertAlmostEqual(e["progress"], 1.0 - before / D0, places=9)
        # ball carried in the hands at 0.9 m into the goal
        carried = line((-11.0, 0.0, 0.9), (0.5, 0.0, 0.9), 0.5)
        d = feed(PitchAwareGoalDetector(require_kick=True), carried,
                 {i: ("right_hand_thumb_1_link", "left_hand_index_0_link") for i in range(1, len(carried))})
        self.assertFalse(d.evaluation()["success"])
        self.assertEqual(d.evaluation()["progress"], 0.0)

    def test_goal_needs_a_leg_as_the_last_touch(self):
        points = line(SPOT, (0.6, 0.0, R), 12.0)
        self.assertLess(points[450][0], 0.0)
        trunk_last = kicked(points, extra={450: ("pelvis",)})
        e = trunk_last.evaluation()
        self.assertFalse(e["success"])
        self.assertEqual(e["reward"]["outcome"], "not_kicked")
        self.assertTrue(trunk_last.crossing.inside_opening)
        onset = 1.0 - math.dist(points[449], (WHOLE, 0.0, points[449][2])) / D0
        self.assertAlmostEqual(e["progress"], onset, places=9)
        self.assertTrue(kicked(points, extra={400: ("torso_link",), 450: ("left_knee_link",)}).evaluation()["success"])
        self.assertFalse(feed(PitchAwareGoalDetector(require_kick=True), points).evaluation()["success"])

    def test_link_groups_cover_the_g1(self):
        groups = [link_group(name) for name in G1_BODIES]
        self.assertEqual((groups.count("leg"), groups.count("trunk"), groups.count("arm")), (12, 8, 24))
        for name in ("left_shoulder_pitch_link", "right_shoulder_roll_link"):
            self.assertEqual(link_group(name), "trunk")
        for name in ("left_shoulder_yaw_link", "right_elbow_link", "left_hand_thumb_0_link"):
            self.assertEqual(link_group(name), "arm")
        with self.assertRaises(ValueError):
            link_group("tail_link")

    def test_shoulder_joint_housing_brush_then_kick_scores_but_an_arm_brush_does_not(self):
        # the housings sit above IFAB Law 12's armpit line; the upper arm does not
        points = path([SPOT] * 300, line(SPOT, (0.6, 0.38, R), 12.0))

        def brush(*links):
            return feed(PitchAwareGoalDetector(require_kick=True), points,
                        {100: links, 300: ("right_ankle_roll_link",)}).evaluation()

        self.assertTrue(brush("left_shoulder_pitch_link")["success"])
        self.assertTrue(brush("right_shoulder_roll_link")["success"])
        for links in (("left_shoulder_yaw_link",), ("left_hand_index_1_link",),
                      ("left_shoulder_pitch_link", "left_elbow_link")):
            with self.subTest(links=links):
                e = brush(*links)
                self.assertEqual((e["success"], e["reward"]["outcome"], e["score"]), (False, "handball", 0.0))

    def test_touches_after_the_attempt_ends_are_ignored(self):
        points = shot_through()
        late = len(points) - 2
        self.assertTrue(kicked(points, extra={late - 1: ("pelvis",), late: ("left_hand_index_0_link",)})
                        .evaluation()["success"])
        e = kicked(line(SPOT, (0.6, 0.0, R), 12.0), extra={100: ("right_wrist_yaw_link",), 200: ("left_knee_link",)})
        self.assertEqual(e.evaluation()["reward"]["outcome"], "handball")
        self.assertEqual(e.last_touch["group"], "arm")

    def test_ball_carried_between_the_knees_scores(self):
        # the documented, accepted consequence of reading "kick" as leg play; progress rises with the carry
        points = path(line(SPOT, (-10.9, 0.0, 0.35), 1.0), line((-10.9, 0.0, 0.35), (0.6, 0.0, 0.35), 1.0))
        knees = {i: ("left_knee_link", "right_knee_link") for i in range(1, len(points))}
        half = feed(PitchAwareGoalDetector(require_kick=True), points[: len(points) // 2], knees).evaluation()
        self.assertGreater(half["progress"], 0.4)
        self.assertTrue(feed(PitchAwareGoalDetector(require_kick=True), points, knees).evaluation()["success"])

    def test_thigh_contact_counts_as_leg_play(self):
        # a push by the pelvis with a thigh also on the ball is leg play; the pelvis alone is not
        first = line(SPOT, (-8.0, 0.0, R), 6.0)
        points = path(first, line((-8.0, 0.0, R), (-5.0, 0.0, R), 2.0))
        both = {i: ("pelvis", "left_hip_yaw_link") for i in range(len(first), len(points))}
        alone = {i: ("pelvis",) for i in range(len(first), len(points))}
        self.assertAlmostEqual(kicked(points, extra=both).evaluation()["progress"], 1.0 - (WHOLE + 5.0) / D0, places=9)
        self.assertAlmostEqual(kicked(points, extra=alone).evaluation()["progress"], 1.0 - (WHOLE + 8.0) / D0, places=9)


class RewardTests(unittest.TestCase):
    def progress(self, points, **kwargs):
        return kicked(points, **kwargs).evaluation()

    def test_untouched_ball_scores_zero(self):
        e = feed(PitchAwareGoalDetector(require_kick=True), [SPOT] * 100).evaluation()
        self.assertEqual((e["success"], e["score"], e["progress"]), (False, 0.0, 0.0))
        self.assertEqual(e["reward"]["contract"], REWARD_CONTRACT)
        self.assertAlmostEqual(e["reward"]["start_distance_to_goal_window_m"], D0, places=9)

    def test_progress_is_the_distance_won_whatever_the_route(self):
        direct = self.progress(line(SPOT, (-5.0, 0.0, R), 6.0))
        detour = self.progress(path(line(SPOT, (-8.0, 2.5, R), 6.0), line((-8.0, 2.5, R), (-5.0, 0.0, R), 3.0)))
        expected = 1.0 - (WHOLE + 5.0) / D0
        self.assertAlmostEqual(direct["progress"], expected, places=9)
        self.assertAlmostEqual(detour["progress"], expected, places=9)
        self.assertAlmostEqual(direct["score"], FAIL_SCORE_SCALE * expected, places=12)
        self.assertEqual(self.progress(line(SPOT, (-14.0, 0.0, R), 3.0))["progress"], 0.0)

    def test_lost_progress_shows(self):
        e = self.progress(path(line(SPOT, (-2.0, 0.0, R), 6.0), line((-2.0, 0.0, R), (-7.0, 1.0, R), 3.0)))
        self.assertAlmostEqual(e["progress"], 1.0 - math.dist((-7.0, 1.0), (WHOLE, 1.0)) / D0, places=9)
        self.assertAlmostEqual(e["reward"]["closest_approach_m"], WHOLE + 2.0, places=9)

    def test_a_miss_keeps_its_margin_and_every_failure_stays_below_a_goal(self):
        e = self.progress(path(line(SPOT, (-0.5, 3.95, R), 15.0), line((-0.5, 3.95, R), (0.6, 3.95, R), 15.0)))
        self.assertAlmostEqual(e["progress"], 1.0 - (3.95 - (FieldSpec().half_goal_width_m - R)) / D0, places=9)
        self.assertLess(e["score"], 0.9)
        self.assertTrue(e["reward"]["attempt_over"])
        on_line = self.progress(path(line(SPOT, (0.005 + R, 0.0, R), 6.0), [(0.005 + R, 0.0, R)] * 10))
        self.assertLess(on_line["score"], 1.0)
        self.assertGreater(on_line["progress"], 0.99)

    def test_failures_are_credited_where_the_ball_ends_up(self):
        # a ball that hits the post and rebounds into play is credited where it lies;
        # closest_approach_m keeps the near thing
        at_post, back = (-0.17, 3.72, R), (-3.0, 3.0, R)
        d = kicked(path(line(SPOT, at_post, 12.0), line(at_post, back, 4.0)))
        e = d.evaluation()
        self.assertEqual(e["reward"]["outcome"], "in_play")
        self.assertAlmostEqual(e["progress"], 1.0 - d.window_distance(back) / D0, places=9)
        self.assertAlmostEqual(e["reward"]["credited_distance_to_goal_window_m"], d.window_distance(back), places=9)
        self.assertAlmostEqual(e["reward"]["closest_approach_m"], d.window_distance(at_post), places=9)

    def test_a_leg_touch_releases_the_trunk_cap_where_the_ball_lies(self):
        # kicked to -9 m, pushed by the pelvis to -5 m (capped), then a shoe touch on the resting
        # ball credits it where it lies, and a roll in from there is a goal
        kick, push = line(SPOT, (-9.0, 0.0, R), 6.0), line((-9.0, 0.0, R), (-5.0, 0.0, R), 2.0)
        rest = [(-5.0, 0.0, R)] * 20
        points = path(kick, push, rest, line((-5.0, 0.0, R), (0.6, 0.0, R), 8.0))
        onset, touch = len(kick), len(kick) + len(push) - 1 + 5
        touches = {i: ("pelvis",) for i in range(onset, onset + len(push) - 1)}
        touches[touch] = ("right_ankle_roll_link",)

        def at(n):
            return kicked(points[:n], extra={i: t for i, t in touches.items() if i < n}).evaluation()["progress"]

        self.assertAlmostEqual(at(touch), 1.0 - (WHOLE + 9.0) / D0, places=9)
        self.assertAlmostEqual(at(touch + 1), 1.0 - (WHOLE + 5.0) / D0, places=9)
        self.assertTrue(kicked(points, extra=touches).evaluation()["success"])

    def test_handball_freezes_progress_against_later_loss(self):
        # the handball keeps the legitimately kicked progress and nothing after it
        first = line(SPOT, (-2.0, 0.0, R), 6.0)
        points = path(first, line((-2.0, 0.0, R), (-6.0, 0.5, R), 3.0))
        e = kicked(points, extra={len(first): ("left_wrist_yaw_link",)}).evaluation()
        self.assertEqual(e["reward"]["outcome"], "handball")
        self.assertAlmostEqual(e["progress"], 1.0 - (WHOLE + 2.0) / D0, places=9)
        self.assertAlmostEqual(e["reward"]["credited_distance_to_goal_window_m"], WHOLE + 2.0, places=9)

    def test_trunk_play_earns_nothing_until_a_leg_touches_again(self):
        points = line(SPOT, (-3.0, 0.0, R), 4.0)
        d = feed(PitchAwareGoalDetector(require_kick=True), points, {1: ("pelvis",)})
        self.assertEqual(d.evaluation()["progress"], 0.0)
        d = kicked(points, extra={500: ("torso_link",)})
        capped = 1.0 - (WHOLE - points[499][0]) / D0
        self.assertAlmostEqual(d.evaluation()["progress"], capped, places=9)

    def test_evaluation_is_valid_json_and_survives_a_checkpoint(self):
        from gym_anything_robotics.evaluation import normalize
        from gym_anything_robotics.physics_trace import pack, unpack
        from football.newton15.checkpoint import decode, encode

        points = line(SPOT, (0.6, 0.0, R), 12.0)
        touches = {1: ("right_ankle_roll_link",), 300: ("pelvis",)}
        first = feed(PitchAwareGoalDetector(require_kick=True), points[:400], touches)
        data = vars(first).copy()
        data.pop("ball"), data.pop("field")
        restored = PitchAwareGoalDetector(require_kick=True)
        vars(restored).update(decode(unpack(pack(encode(data)))["data"]))
        self.assertEqual(json.dumps(restored.evaluation()), json.dumps(first.evaluation()))
        for detector in (first, restored):
            for i in range(400, len(points)):
                detector.update_supported(i * DT, points[i], (12.0, 0.0, 0.0), normal_force_n=0.0,
                                          compression_m=0.0, robot_touches={420: ("left_knee_link",)}.get(i, ()))
        self.assertEqual(json.dumps(restored.evaluation()), json.dumps(first.evaluation()))
        self.assertTrue(first.evaluation()["success"])
        normalized = normalize({**first.evaluation(), "instruction": "x"})
        self.assertTrue(normalized["available"] and normalized["success"])


if __name__ == "__main__":
    unittest.main()
