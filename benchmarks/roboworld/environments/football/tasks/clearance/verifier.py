"""Outcome of clearance: success when the whole ball passes the penalty-area line's outer edge upfield
with a leg as the last touch; progress is the approach to that line, counted from 1 m on.

verify(config, layout) receives the world's Config and the task's layout (variants.layout, whose "verifier"
entry holds the parameters from setup.py) and returns the verifier FootballWorld builds at the start, feeds
every tick (update, update_supported) and asks for evaluation(): success, score, progress and the
football-drills-v1 reward record. The class builds on the drills' shared machinery (drills.Drill)."""
from football.newton15.drills import Drill, between, drill_verifier


class Clearance(Drill):
    task = "clearance"

    def _reset_drill(self):
        self.line_x_m = None
        self.start_distance_m = None
        self.line_crossing = None

    def begin(self, position):
        self.line_x_m = self.params["line_outer_x_m"] - self.ball.radius_m
        self.start_distance_m = position[0] - self.line_x_m

    def distance(self, x):
        return max(x - self.line_x_m, 0.0)

    def live(self, time_s, position, velocity):
        self.distance_m = self.distance(position[0])
        self.progress = self._capped(self.approach(self.distance_m, self.start_distance_m))

    def advance(self, p0, v0, t0, p1, v1, t1):
        if p0[0] > self.line_x_m >= p1[0]:
            t, centre, velocity = between(p0, v0, t0, p1, v1, t1, (p0[0] - self.line_x_m) / (p0[0] - p1[0]))
            self.line_crossing = {"time_s": t, "center_world_m": centre, "velocity_world_m_s": velocity}
            if self.kicked():
                self.finish(t, "cleared", success=True)
            else:
                self.live(t, centre, velocity)
                self.finish(t, "not_kicked")
            return
        super().advance(p0, v0, t0, p1, v1, t1)

    def goal_line(self, time_s, centre, velocity, entry):
        if entry["inside"]:
            self.progress = 0.0
            self.finish(time_s, "own_goal")
        else:
            self.finish(time_s, "out_over_goal_line")

    def details(self):
        return {"line_center_x_m": self.line_x_m, "credited_distance_to_line_m": self.distance_m,
                "start_distance_to_line_m": self.start_distance_m, "line_crossing": self.line_crossing}


def verify(config, layout):
    return drill_verifier(Clearance, config, layout)
