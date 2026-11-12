"""Deterministic verifiers and reward for the football drills (catalogue: football/newton15/variants.py).

The `goal` and the `free-kick` (whose verifier is goal's, unchanged) use task.PitchAwareGoalDetector. The
other six drills subclass it through Drill below, each in its task folder (tasks/<task>/verifier.py): every tick
FootballWorld passes the same inputs as for goal (ball centre and velocity at the end of the tick, the pitch's
support, the robot links that pushed on the ball), plus, for trap, the robot's pelvis and feet
(FootballWorld.robot_pose). Instructions state the job; the rules below are the verifier's.

Shared rules.
- Score = 1 on success, else 0.9 x progress (goal's contract), so every failure stays below every success.
  progress is in [0, 1], 0 at the untouched start, read from the current state while the attempt is live (lost
  progress shows) and frozen when the attempt ends. Nothing in it depends on how the robot moved.
- Touches use goal's link groups and precedence (task.LINK_GROUPS, task.TOUCH_PRECEDENCE). Rule "kick" (clearance,
  pass-gate, pass-square, corner-target): exactly goal's rule 3. An arm or hand touch ends the attempt (handball,
  progress frozen at its value before that touch); nothing is earned before the first leg touch; a trunk touch caps
  credit at its value until a leg touches again; success needs a leg as the last touch. Rule "receive" (trap): a
  trunk touch is legal as well (a chest or thigh trap); arm or hand is handball. Rule "keeper" (save): every link
  is legal (a goalkeeper in his own area); there is no handball.
- The goal line (whole ball over x = 0.06 + r, goal's rule 1, with goal's frame-plane opening test) ends every
  drill: a goal in corner-target, a goal conceded in save, an own goal in clearance, otherwise the ball is out.
- The ball leaving the sector (centre |y| > 12 - r or x < -34 + r) ends the attempt with progress frozen.
- Played away (clearance, pass-gate, pass-square). Their targets are 5.0-6.6 m away, and goal's idle baseline (5 s of
  empty targets; the robot falls onto the ball with its thighs) moves the ball 0.44 m. Measured against
  the full start distance that fall alone would read 0.065-0.079 progress (score 0.059-0.071, above the 0.05 the
  goal task keeps). These drills therefore count the approach from 1.0 m on (trap's "at your feet" radius): with d
  the distance still to go and d0 its start value, progress = clip(1 - d / (d0 - 1 m)). A ball within a metre of
  where it lay has not been played away.

Per-task rules (code in tasks/<task>/verifier.py, thresholds from tasks/<task>/setup.py via variants.layout):
- clearance: success when the whole ball passes the penalty-area line's outer edge upfield (centre x < -16.5 - r,
  interpolated, the first crossing final) with a leg as the last touch; the ball need not stop. d = how far the
  centre still is from x = -16.5 - r. A whole-ball crossing of the goal line inside the frame is an own goal
  (progress 0).
- pass-gate: success when the centre crosses the gate line away from the start side between the marker centres
  (|s| <= 1.0 m along the gate) with the centre at most r + 0.25 m above the pitch (along the ground), a leg the
  last touch; interpolated, the first crossing final. d = distance to the through-window (the half-plane beyond the
  line, |s| <= 1.0 m, centre height <= r + 0.25 m), goal's window_distance construction; a crossing outside the gate
  freezes progress at d = the miss in the gate plane.
- trap: success when, held for the trailing 1 s, the ball's speed is < 0.05 m/s, its centre is within 1.0 m
  horizontally of the midpoint of the two ankle-roll links, the pelvis is at z >= 0.55 m with its up axis within
  45 deg of vertical, a leg or trunk has touched the ball and no arm or hand has. progress = T x C x S x U: T = 1
  from the first legal touch (0 before, so a ball rolling past earns nothing), C = clip(1 - (d - 1.0) / (d0 - 1.0))
  with d the ball-to-feet distance (d0 at the start), S = clip(1 - |v| / 1.0 m/s), U = clip((z_pelvis - 0.35) /
  (0.55 - 0.35)). A fallen robot earns U = 0.
- pass-square: success when, held for the trailing 1 s, the ball's speed is < 0.05 m/s and its centre is inside the
  2 x 2 m square, a leg the last touch. progress = capped(C x S): C = clip(1 - e / (e0 - 1 m)), e the horizontal
  distance from the centre to the square (0 inside), S = clip(1 - |v| / 1.0 m/s); kick rule as above, so an overhit
  ball crossing the square scores low and the reading rises as it stops.
- corner-target: goal's success, and the ball passed the sheet through an opening: where its centre crosses the
  sheet's mid-plane (x = 0.075 m) |y| >= 2.40 + r and z <= 1.00 - r, less a 0.01 m contact allowance. The rigid
  sheet is the physical barrier (a ball meeting it cannot carry its centre past 0.065 - r); this is a guard. d = the
  distance to the nearer opening window (whole ball over the line, |y| in [2.40 + r, 3.66 - r], z <= 1.00 - r);
  progress = clip(1 - d / d0) as in goal (d0 = 11.45 m); a miss outside the frame freezes at d = its distance from
  the opening windows in the frame plane.
- save: success when the shot is resolved with no goal: the ball at rest (< 0.05 m/s) for 1 s, back beyond the goal
  area line (x < -5.5) moving away, wholly over the goal line outside the frame, out of the sector, or 5 s after
  launch. Any whole-ball crossing inside the frame is a goal conceded, whoever touched last. Progress only after the
  keeper's first touch (0 untouched, so the reference shot earns nothing), measured against the shot's speed s_in
  just before that touch: unresolved, 0.5 x clip(1 - max(v_x, 0) / s_in); a goal conceded, 0.5 x clip(1 - |v| /
  s_in) at the whole-ball crossing (a parry that slows the shot earns something). The spec's reference-profile
  measure (robot-free crossing speed and v_ref(x)) is replaced by s_in so that no fitted profile is needed and an
  untouched shot reads exactly 0.
"""

import math

from football.football_math import is_finite, lerp
from .task import FAIL_SCORE_SCALE, TOUCH_PRECEDENCE, PitchAwareGoalDetector, link_group
from .variants import task_verifier

CONTRACT = "football-drills-v1"
PLAYED_AWAY_M = 1.0
SECTOR_HALF_WIDTH_M = 12.0
SECTOR_END_X_M = -34.0


def clip01(value):
    return min(1.0, max(0.0, value))


def between(p0, v0, t0, p1, v1, t1, alpha):
    return t0 + alpha * (t1 - t0), lerp(p0, p1, alpha), lerp(v0, v1, alpha)


class Drill(PitchAwareGoalDetector):
    """Goal-line, touch, sector and scoring machinery shared by the drills; subclasses add their task's rules."""

    task = None
    touch_rule = "kick"
    uses_robot_pose = False

    def __init__(self, params=None, *, require_kick=True):
        self.params = dict(params or {})
        super().__init__(require_kick=require_kick)

    def _reset_attempt(self):
        super()._reset_attempt()
        self.success = False
        self.end = None
        self.start_m = None
        self.robot = None
        self.first_touch = None
        self._reset_drill()

    def _reset_drill(self):
        pass

    @property
    def attempt_over(self):
        return self.end is not None

    def kicked(self):
        return not self.require_kick or (self.last_touch is not None and self.last_touch["group"] == "leg")

    def finish(self, time_s, outcome, *, success=False):
        self.end = {"time_s": float(time_s), "outcome": outcome}
        self.outcome, self.success = outcome, bool(success)
        if success:
            self.progress = 1.0

    def approach(self, distance_m, start_m, played_away=PLAYED_AWAY_M):
        span = start_m - played_away
        return clip01(1.0 - distance_m / span) if span > 0.0 else float(distance_m <= 0.0)

    def left_sector(self, position):
        return abs(position[1]) > SECTOR_HALF_WIDTH_M - self.ball.radius_m or (
            position[0] < SECTOR_END_X_M + self.ball.radius_m
        )

    def touch(self, time_s, links):
        if not links or self.attempt_over:
            return
        groups = {}
        for name in links:
            groups.setdefault(link_group(name), []).append(str(name))
        group = next(g for g in TOUCH_PRECEDENCE if g in groups)
        self.last_touch = {"time_s": float(time_s), "group": group, "links": sorted(groups[group])}
        if self.first_touch is None:
            incoming = self.previous_velocity_m_s or (0.0, 0.0, 0.0)
            self.first_touch = dict(self.last_touch, incoming_velocity_m_s=tuple(map(float, incoming)))
        if self.touch_rule == "keeper":
            return
        if group == "arm":
            self.handball = dict(self.last_touch)
            self.finish(time_s, "handball")
        elif group == "leg" or (group == "trunk" and self.touch_rule == "receive"):
            self.credit_cap = None
            self.on_legal_touch()
        elif self.credit_cap is None:
            self.credit_cap = self.progress

    def on_legal_touch(self):
        pass

    def update(self, time_s, position_world_m, velocity_world_m_s):
        if self.attempt_over:
            return self.crossing
        if not is_finite(position_world_m) or not is_finite(velocity_world_m_s) or not math.isfinite(time_s):
            raise ValueError("drill verifier received non-finite state")
        p0, v0, t0 = self.previous_position_m, self.previous_velocity_m_s, self.previous_time_s
        p1, v1, t1 = position_world_m, velocity_world_m_s, time_s
        self.previous_position_m, self.previous_velocity_m_s, self.previous_time_s = p1, v1, t1
        if p0 is None or v0 is None or t0 is None:
            if self.start_m is None:
                self.start_m = tuple(float(c) for c in p1)
                self.begin(p1)
            self.live(t1, p1, v1)
            return None

        def at_x(plane):
            dx = p1[0] - p0[0]
            return between(p0, v0, t0, p1, v1, t1, 0.0 if dx == 0.0 else clip01((plane - p0[0]) / dx))

        frame = self.field.goal_plane_x_m
        if p0[0] <= frame < p1[0]:
            t, centre, _ = at_x(frame)
            self.frame_entry = self._frame_record(t, centre)
        elif p1[0] <= frame < p0[0]:
            self.frame_entry = None
        self.planes(p0, v0, t0, p1, v1, t1, at_x)
        whole = self.whole_ball_x_m
        if p0[0] <= whole < p1[0]:
            t, centre, velocity = at_x(whole)
            progress, distance = self.progress, self.distance_m
            # records self.crossing (GoalCrossing) with goal's margins; the drill decides outcome and progress
            PitchAwareGoalDetector._latch(self, t, centre, velocity)
            self.progress, self.distance_m = progress, distance
            self.goal_line(t, centre, velocity, self.frame_entry or self._frame_record(t, centre))
            return self.crossing
        self.advance(p0, v0, t0, p1, v1, t1)
        return self.crossing

    def begin(self, position):
        pass

    def planes(self, p0, v0, t0, p1, v1, t1, at_x):
        pass

    def goal_line(self, time_s, centre, velocity, entry):
        self.finish(time_s, "out_over_goal_line")

    def live(self, time_s, position, velocity):
        raise NotImplementedError

    def advance(self, p0, v0, t0, p1, v1, t1):
        self.live(t1, p1, v1)
        if not self.attempt_over and self.left_sector(p1):
            self.finish(t1, "out_of_sector")

    def details(self):
        return {}

    def evaluation(self):
        success = bool(self.success)
        progress = 1.0 if success else float(self.progress)
        return {
            "success": success,
            "score": 1.0 if success else FAIL_SCORE_SCALE * progress,
            "progress": progress,
            "reward": {
                "contract": CONTRACT,
                "task": self.task,
                "score_rule": "1 if success else 0.9 x progress",
                "outcome": self.outcome,
                "attempt_over": self.attempt_over,
                "end": self.end,
                "touch_rule": self.touch_rule,
                "kick_rule": self.require_kick,
                "last_robot_touch": self.last_touch,
                "first_robot_touch": self.first_touch,
                "handball": self.handball,
                "credit_cap": self.credit_cap,
                **self.details(),
            },
        }


class Hold:
    """Mixin: a condition held continuously for `hold_s` latches success (trap, pass-square, save at rest)."""

    def held(self, time_s, condition, key="hold_since_s"):
        if not condition:
            setattr(self, key, None)
            return False
        if getattr(self, key) is None:
            setattr(self, key, float(time_s))
        return time_s - getattr(self, key) >= self.params["hold_s"] - 1e-9


def make_verifier(config, layout):
    """The task's verifier: verify(config, layout) of its folder (tasks/<task>/verifier.py); goal and free-kick get
    goal's detector exactly as before, the other drills their Drill subclass (drill_verifier)."""
    return task_verifier(config.task).verify(config, layout)


def drill_verifier(cls, config, layout):
    """A drill's verifier with its layout's parameters; the kick rule holds for rule "kick" drills with a robot."""
    return cls(layout["verifier"], require_kick=bool(config.robot) and cls.touch_rule == "kick")
