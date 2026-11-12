"""Outcome of corner-target: success when goal's success holds and the ball passed the target sheet
through one of its two bottom-corner openings; progress is the approach to the opening windows.

verify(config, layout) receives the world's Config and the task's layout (variants.layout, whose "verifier"
entry holds the parameters from setup.py) and returns the verifier FootballWorld builds at the start, feeds
every tick (update, update_supported) and asks for evaluation(): success, score, progress and the
football-drills-v1 reward record. The class builds on the drills' shared machinery (drills.Drill)."""
import math

from football.newton15.drills import Drill, drill_verifier


class CornerTarget(Drill):
    task = "corner-target"

    def _reset_drill(self):
        self.start_distance_m = None
        self.sheet_entry = None

    @property
    def opening_y_m(self):
        r = self.ball.radius_m
        return self.params["opening_inner_y_m"] + r, self.params["opening_outer_y_m"] - r

    @property
    def opening_z_m(self):
        return self.params["opening_top_m"] - self.ball.radius_m

    def opening_miss(self, y, z):
        low, high = self.opening_y_m
        return math.hypot(max(low - abs(y), 0.0, abs(y) - high), max(z - self.opening_z_m, 0.0))

    def window_distance(self, position):
        x, y, z = position
        return math.hypot(max(self.whole_ball_x_m - x, 0.0), self.opening_miss(y, z))

    def begin(self, position):
        self.start_distance_m = self.window_distance(position)

    def live(self, time_s, position, velocity):
        self.distance_m = self.window_distance(position)
        self.progress = self._capped(self.approach(self.distance_m, self.start_distance_m, played_away=0.0))

    def planes(self, p0, v0, t0, p1, v1, t1, at_x):
        sheet = self.params["sheet_x_m"]
        if p0[0] <= sheet < p1[0]:
            t, centre, _ = at_x(sheet)
            allowance, r = self.params["contact_allowance_m"], self.ball.radius_m
            through = (abs(centre[1]) >= self.params["opening_inner_y_m"] + r - allowance
                       and centre[2] <= self.params["opening_top_m"] - r + allowance)
            self.sheet_entry = {"time_s": float(t), "center_world_m": tuple(map(float, centre)),
                                "through_opening": bool(through)}
        elif p1[0] <= sheet < p0[0]:
            self.sheet_entry = None

    def goal_line(self, time_s, centre, velocity, entry):
        through = self.sheet_entry is not None and self.sheet_entry["through_opening"]
        # a miss is measured where the ball passed the frame, or the sheet if the frame let it through
        reference = self.sheet_entry if entry["inside"] and self.sheet_entry else entry
        _, y, z = reference["center_world_m"]
        miss = 0.0 if entry["inside"] and through else self.opening_miss(y, z)
        self.distance_m = miss
        self.progress = self._capped(self.approach(miss, self.start_distance_m, played_away=0.0))
        if not entry["inside"]:
            self.finish(time_s, "wide_or_over")
        elif not through:
            self.finish(time_s, "not_through_opening")
        elif self.kicked():
            self.finish(time_s, "goal_through_opening", success=True)
        else:
            self.finish(time_s, "not_kicked")

    def details(self):
        return {"credited_distance_to_opening_m": self.distance_m,
                "start_distance_to_opening_m": self.start_distance_m,
                "frame_entry": self.frame_entry, "sheet_entry": self.sheet_entry}


def verify(config, layout):
    return drill_verifier(CornerTarget, config, layout)
