"""Outcome of pass-square: success when the ball has stopped inside the 2 x 2 m square for 1 s with a leg
as the last touch; progress is the approach to the square times how slow the ball is.

verify(config, layout) receives the world's Config and the task's layout (variants.layout, whose "verifier"
entry holds the parameters from setup.py) and returns the verifier FootballWorld builds at the start, feeds
every tick (update, update_supported) and asks for evaluation(): success, score, progress and the
football-drills-v1 reward record. The class builds on the drills' shared machinery (drills.Drill)."""
import math

from football.football_math import norm
from football.newton15.drills import Drill, Hold, clip01, drill_verifier


class PassSquare(Hold, Drill):
    task = "pass-square"

    def _reset_drill(self):
        self.start_distance_m = None
        self.hold_since_s = None
        self.factors = None

    def square_distance(self, p):
        (cx, cy), h = self.params["centre_m"], self.params["half_side_m"]
        return math.hypot(max(abs(p[0] - cx) - h, 0.0), max(abs(p[1] - cy) - h, 0.0))

    def begin(self, position):
        self.start_distance_m = self.square_distance(position)

    def live(self, time_s, position, velocity):
        e, speed = self.square_distance(position), norm(velocity)
        self.factors = {"C": self.approach(e, self.start_distance_m),
                        "S": clip01(1.0 - speed / self.params["speed_scale_m_s"]),
                        "distance_to_square_m": e, "speed_m_s": speed}
        self.progress = self._capped(self.factors["C"] * self.factors["S"])
        if self.held(time_s, e == 0.0 and speed < self.params["rest_speed_m_s"] and self.kicked()):
            self.finish(time_s, "stopped_in_square", success=True)

    def details(self):
        return {"factors": self.factors, "hold_since_s": self.hold_since_s,
                "start_distance_to_square_m": self.start_distance_m}


def verify(config, layout):
    return drill_verifier(PassSquare, config, layout)
