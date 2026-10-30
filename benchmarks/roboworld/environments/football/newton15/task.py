"""Goal adjudication and reward for the football/goal task (its deterministic verifier).

Instruction: "The football is on the penalty mark in front of the Unitree G1 humanoid. Kick it into the goal."
The rules below are the verifier's; none of them is part of the instruction. Every 2 ms physics tick the world
passes the ball's end-of-tick centre and velocity, the pitch's support for that tick and the robot links that
touched the ball (FootballWorld.robot_ball_touches). Field coordinates: the shooter is at negative x, the
goal-line and frame axes at x = 0, the goal behind it.

Goal (success). All three must hold; the first two are IFAB Law 10.1 for this frame.

1. Whole ball over the goal line. The drawn goal line is as wide as the posts (0.12 m) and centred on the post and
   crossbar axes, so its back edge is x = +0.06 m (IFAB Law 1.2: lines belong to the area they bound). The whole
   ball is over the line when its trailing surface passes that edge, i.e. its centre passes x = 0.06 + r
   (r = 0.1098 m). The instant is interpolated between ticks, and the first time the whole ball is over the line,
   inside or outside the goal, decides the attempt: a ball out of play over the goal line cannot later be rolled
   in, and a goal stays a goal if the ball comes back out.
2. Between the posts and under the crossbar, judged where the ball passes the frame: the ball centre's (y, z) at
   the instant it crosses the plane of the post and crossbar axes (x = 0) on the way in is compared with those
   axes, |y| < 3.72 m and z < 2.50 m. The posts and crossbar are solid cylinders of radius 0.06 m on that plane,
   so no ball centre can be there within r + 0.06 m of an axis: a ball passing inside has |y| <= 3.55 m and
   z <= 2.33 m, one passing outside has |y| >= 3.89 m or z >= 2.67 m. The verdict therefore cannot depend on how
   far the side or roof netting behind the line has yielded. A ball that comes back out in front of the frame
   before it is wholly over the line loses that entry. Ground shots: the lower surface may be below z = 0 at
   that instant only by the pitch's peak compression in that tick, if the pitch pushed on the ball in it, plus 2 mm.
   Recorded margins (frame_entry and goal_crossing) are measured to these decision boundaries at the frame
   plane: post_axis_margin_m = 3.72 - |y| (goal_crossing.lateral_clearance_m), crossbar_axis_margin_m = 2.50 - z
   (upper_clearance_m) and floor_margin_m = z - r + the floor allowance (lower_clearance_m). The ball passed inside
   exactly when all three are positive (the floor margin may be zero).
3. Kicked, not handled (worlds with the robot). A robot touch is a native contact with the ball that carries a
   positive normal force, or the right shoe's constitutive contact. Any touch by an arm or hand link (upper arm
   from shoulder_yaw_link down, elbow, wrist, hand) ends the attempt without a goal from that tick on (handball).
   The goal counts only if the robot's last touch before the whole ball was over the line was with a leg link:
   the hip links (hip-joint housing and thigh), knee (shin), ankle and foot. A ball last played by the trunk
   (pelvis, waist, torso, head, and the shoulder-joint housings shoulder_pitch_link and shoulder_roll_link, which
   sit above the upper arm, i.e. above IFAB Law 12's armpit line) was not kicked in. Either foot, the shin, knee
   or thigh, any number of touches, any order, any approach and any robot posture are accepted. Within one tick,
   arm outranks leg and leg outranks trunk. Robot-free fixtures have no kick rule. Unlike rules 1, 2 and the arm
   boundary, this leg rule is not IFAB's definition: it reads "kick" as play with any leg link, deliberately wider
   than the IFAB glossary's "The ball is kicked when a player makes contact with it with the foot and/or the
   ankle". On the G1 the shin meets the ball together with the shoe in an instep strike (both executed robot
   goals touched shoe and shin), and a shot that goes in off the shin or thigh is still the kick the task asks
   for; the left foot's only collision shapes are four 5 mm sole spheres, so a left-foot strike registers mostly
   on ankle and shin. Accepted consequences: a ball carried between the knees or pushed along by the shins or
   thighs into the goal also scores (excluding it would need a contact-duration or displacement threshold); and
   because the hip-joint housing (hip_pitch_link) is a leg link, the leg/trunk boundary sits beside the pelvis, so
   a ball bumped in off the side of the hip counts while the same bump on the pelvis does not. Accepted
   limitation: an incidental arm or hand touch earlier in the play also ends the attempt, although IFAB Law 12
   would let a later separate kick stand; telling a brush from handling (carrying, placing, a hand slap before a
   tap-in) would need a duration, impulse or displacement threshold.

Reward, contract REWARD_CONTRACT: score = 1 on a goal, else 0.9 x progress, so every failure stays below every
goal (the convention of label_application/task.py). progress = clip(1 - d / d0, 0, 1): d is the distance in metres
from the ball centre to the goal window, the centre positions at which the whole ball is over the line inside the
frame (x >= 0.06 + r, |y| <= 3.66 - r, z <= 2.44 - r), and d0 is that distance at the episode's start (11.17 m
from the penalty mark). It is read from the current ball state while the attempt is live, so progress that is
lost again shows, and nothing in it depends on how the robot moved. It honours rule 3, with the same wide leg
reading as success (so a leg carry also earns progress): ball movement after a trunk touch earns nothing until a
leg touches the ball again (it can still lose progress); from that leg touch the ball is credited where it then
lies, because a ball that now goes in from there would be a goal. Before the first leg touch nothing is earned.
When the attempt ends, progress freezes: a goal at 1; a ball wholly over the line
outside the frame at clip(1 - m / d0, 0, 1), m its distance from the window in the plane where it passed the
frame; a ball wholly over the line inside the frame but not kicked in at its credited value; a handball at its
value just before the handling touch. Failures are therefore ranked by where the ball is when the attempt ends
(or when the score is read), not by the best moment of its path: a ball that hits the post and rebounds into
play is credited where it lies, and can rank below a clean miss; reward.closest_approach_m reports such near
things. reward.credited_distance_to_goal_window_m is the d behind the current progress and freezes with it (for
a miss, m). The score is read when the evaluator is asked: a ball still travelling is credited by where it is.
The owner's per-action reward from act() is this score level at the end of the action, not an increment; a
learner wanting a return equal to the final score should sum per-action differences of the score, which add up
to the final score because the score freezes when the attempt ends. Idle baseline: 5 s of empty-target actions
from the task start, in which the unbalanced robot falls onto the ball with its thighs and pelvis, measured
progress 0.0385 and score 0.0346; keep it at most 0.05.
"""

import math

from football.football_math import is_finite, lerp
from football.football_task import GoalCrossing, GoalDetector

REWARD_CONTRACT = "football-goal-progress-v1"
FAIL_SCORE_SCALE = 0.9
# Robot link groups for the kick and handball rules, matched in this order on G1 body names: the shoulder-joint
# housings are trunk before "shoulder" makes the rest of the arm an arm, and arm words come before leg words so
# that no hand link can be read as a leg. Every robot body must match one group (FootballWorld checks this at
# build). TOUCH_PRECEDENCE decides the group of a tick in which several groups touch the ball.
LINK_GROUPS = (
    ("trunk", ("pelvis", "waist", "torso", "head", "shoulder_pitch", "shoulder_roll")),
    ("arm", ("shoulder", "elbow", "wrist", "hand")),
    ("leg", ("hip", "knee", "ankle")),
)
TOUCH_PRECEDENCE = ("arm", "leg", "trunk")


def link_group(name):
    for group, words in LINK_GROUPS:
        if any(word in name for word in words):
            return group
    raise ValueError(f"Robot link {name!r} has no kick/handball group")


class PitchAwareGoalDetector(GoalDetector):
    def __init__(self, *args, require_kick=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.require_kick = bool(require_kick)
        self._reset_attempt()

    def _reset_attempt(self):
        self.frame_entry = None
        self.last_touch = None
        self.handball = None
        self.outcome = "in_play"
        self.start_distance_m = None
        self.distance_m = None
        self.best_distance_m = None
        self.credit_cap = 0.0 if self.require_kick else None
        self.progress = 0.0

    def reset(self):
        super().reset()
        self._reset_attempt()

    @property
    def attempt_over(self):
        return self.crossing is not None or self.handball is not None

    @property
    def post_radius_m(self):
        return 0.5 * self.field.goal_post_diameter_m

    @property
    def whole_ball_x_m(self):
        return self.field.goal_plane_x_m + self.post_radius_m + self.ball.radius_m

    @property
    def window_half_width_m(self):
        return self.field.half_goal_width_m - self.ball.radius_m

    @property
    def window_height_m(self):
        return self.field.goal_inside_height_m - self.ball.radius_m

    def window_distance(self, position):
        x, y, z = position
        return math.sqrt(
            max(self.whole_ball_x_m - x, 0.0) ** 2
            + max(abs(y) - self.window_half_width_m, 0.0) ** 2
            + max(z - self.window_height_m, 0.0) ** 2
        )

    def update_supported(
        self, time_s, position, velocity, *, normal_force_n, compression_m, robot_touches=()
    ):
        # IFAB's opening has posts and a crossbar, not a solid mathematical
        # plate at z=0. A supported ball can indent the compliant pitch. Admit
        # only the current simulated compression plus the existing 2 mm
        # geometric margin, not an unconditional wider goal tolerance.
        self.lower_surface_tolerance_m = (
            0.002 + max(0.0, compression_m) if normal_force_n > 0 else 0.002
        )
        self.touch(time_s, robot_touches)
        return self.update(time_s, position, velocity)

    def touch(self, time_s, links):
        """Record the robot links that touched the ball in this tick (rule 3)."""
        if not links or self.attempt_over:
            return
        groups = {}
        for name in links:
            groups.setdefault(link_group(name), []).append(str(name))
        group = next(g for g in TOUCH_PRECEDENCE if g in groups)
        self.last_touch = {"time_s": float(time_s), "group": group, "links": sorted(groups[group])}
        if group == "arm":
            self.handball = dict(self.last_touch)
            self.outcome = "handball"
        elif group == "leg":
            self.credit_cap = None
        elif self.credit_cap is None:
            self.credit_cap = self.progress

    def update(self, time_s, position_world_m, velocity_world_m_s):
        if self.attempt_over:
            return self.crossing
        if not is_finite(position_world_m) or not is_finite(velocity_world_m_s) or not math.isfinite(time_s):
            raise ValueError("goal detector received non-finite state")
        p0, v0, t0 = self.previous_position_m, self.previous_velocity_m_s, self.previous_time_s
        self.previous_position_m = position_world_m
        self.previous_velocity_m_s = velocity_world_m_s
        self.previous_time_s = time_s
        if p0 is None or v0 is None or t0 is None:
            if self.start_distance_m is None:
                self.start_distance_m = self.window_distance(position_world_m)
            self._credit(position_world_m)
            return None

        def at(plane):
            dx = position_world_m[0] - p0[0]
            alpha = 0.0 if dx == 0.0 else min(1.0, max(0.0, (plane - p0[0]) / dx))
            return (
                t0 + alpha * (time_s - t0),
                lerp(p0, position_world_m, alpha),
                lerp(v0, velocity_world_m_s, alpha),
            )

        frame = self.field.goal_plane_x_m
        if p0[0] <= frame < position_world_m[0]:
            t, center, _ = at(frame)
            self.frame_entry = self._frame_record(t, center)
        elif position_world_m[0] <= frame < p0[0]:
            self.frame_entry = None
        whole = self.whole_ball_x_m
        if p0[0] <= whole < position_world_m[0]:
            self._latch(*at(whole))
        else:
            self._credit(position_world_m)
        return self.crossing

    def _frame_record(self, time_s, center):
        post = self.field.half_goal_width_m + self.post_radius_m - abs(center[1])
        bar = self.field.goal_inside_height_m + self.post_radius_m - center[2]
        floor = center[2] - self.ball.radius_m + self.lower_surface_tolerance_m
        return {
            "time_s": float(time_s),
            "center_world_m": tuple(float(c) for c in center),
            "inside": bool(post > 0.0 and bar > 0.0 and floor >= 0.0),
            "post_axis_margin_m": post,
            "crossbar_axis_margin_m": bar,
            "floor_margin_m": floor,
            "floor_allowance_m": self.lower_surface_tolerance_m,
        }

    def _latch(self, time_s, center, velocity):
        entry = self.frame_entry or self._frame_record(time_s, center)
        inside = entry["inside"]
        kicked = not self.require_kick or (self.last_touch is not None and self.last_touch["group"] == "leg")
        scored = inside and kicked
        if scored:
            self.outcome, reason = "goal", "whole ball crossed inside the opening"
        elif inside:
            self.outcome = "not_kicked"
            reason = "whole ball crossed inside the opening, but the robot's last touch was not with a leg"
        else:
            self.outcome, reason = "wide_or_over", "whole ball crossed outside the opening"
        self.crossing = GoalCrossing(
            time_s=time_s,
            center_world_m=center,
            velocity_world_m_s=velocity,
            whole_ball_crossed=True,
            inside_opening=inside,
            scored=scored,
            lateral_clearance_m=entry["post_axis_margin_m"],
            lower_clearance_m=entry["floor_margin_m"],
            upper_clearance_m=entry["crossbar_axis_margin_m"],
            reason=reason,
        )
        _, y, z = entry["center_world_m"]
        miss = 0.0 if inside else math.hypot(
            max(abs(y) - self.window_half_width_m, 0.0), max(z - self.window_height_m, 0.0)
        )
        self._set_distance(miss)
        self.progress = 1.0 if scored else self._capped(self._raw(miss))

    def _credit(self, position):
        d = self.window_distance(position)
        self._set_distance(d)
        self.progress = self._capped(self._raw(d))

    def _set_distance(self, d):
        self.distance_m = d
        self.best_distance_m = d if self.best_distance_m is None else min(self.best_distance_m, d)

    def _raw(self, d):
        d0 = self.start_distance_m
        return min(1.0, max(0.0, 1.0 - d / d0)) if d0 else float(d <= 0.0)

    def _capped(self, value):
        return value if self.credit_cap is None else min(value, self.credit_cap)

    def evaluation(self):
        success = bool(self.crossing is not None and self.crossing.scored)
        progress = 1.0 if success else float(self.progress)
        return {
            "success": success,
            "score": 1.0 if success else FAIL_SCORE_SCALE * progress,
            "progress": progress,
            "reward": {
                "contract": REWARD_CONTRACT,
                "score_rule": "1 if success else 0.9 x progress",
                "outcome": self.outcome,
                "attempt_over": self.attempt_over,
                "credited_distance_to_goal_window_m": self.distance_m,
                "start_distance_to_goal_window_m": self.start_distance_m,
                "closest_approach_m": self.best_distance_m,
                "kick_rule": self.require_kick,
                "last_robot_touch": self.last_touch,
                "handball": self.handball,
                "frame_entry": self.frame_entry,
            },
        }
