"""Outcome of pass-gate: success when the ball's centre crosses the gate line between the two discs, along
the ground (at most r + 0.25 m up), with a leg as the last touch; progress is the approach to the gate.

verify(config, layout) receives the world's Config and the task's layout (variants.layout, whose "verifier"
entry holds the parameters from setup.py) and returns the verifier FootballWorld builds at the start, feeds
every tick (update, update_supported) and asks for evaluation(): success, score, progress and the
football-drills-v1 reward record. The class builds on the drills' shared machinery (drills.Drill)."""
import math

from football.newton15.drills import Drill, between, drill_verifier


class PassGate(Drill):
    task = "pass-gate"

    def _reset_drill(self):
        self.start_distance_m = None
        self.gate_crossing = None

    def side(self, p):
        (cx, cy), (nx, ny) = self.params["centre_m"], self.params["normal"]
        return (p[0] - cx) * nx + (p[1] - cy) * ny

    def along(self, p):
        (cx, cy), (tx, ty) = self.params["centre_m"], self.params["tangent"]
        return (p[0] - cx) * tx + (p[1] - cy) * ty

    def miss(self, p):
        return math.hypot(max(abs(self.along(p)) - self.params["half_width_m"], 0.0),
                          max(p[2] - self.params["max_height_m"], 0.0))

    def window_distance(self, position):
        return math.hypot(max(-self.side(position), 0.0), self.miss(position))

    def begin(self, position):
        self.start_distance_m = self.window_distance(position)

    def live(self, time_s, position, velocity):
        self.distance_m = self.window_distance(position)
        self.progress = self._capped(self.approach(self.distance_m, self.start_distance_m))

    def advance(self, p0, v0, t0, p1, v1, t1):
        s0, s1 = self.side(p0), self.side(p1)
        if s0 < 0.0 <= s1:
            t, centre, velocity = between(p0, v0, t0, p1, v1, t1, s0 / (s0 - s1))
            miss = self.miss(centre)
            self.gate_crossing = {"time_s": t, "center_world_m": centre, "velocity_world_m_s": velocity,
                                  "along_gate_m": self.along(centre), "miss_m": miss}
            if miss == 0.0 and self.kicked():
                self.finish(t, "through_gate", success=True)
                return
            self.distance_m = miss
            self.progress = self._capped(self.approach(miss, self.start_distance_m))
            wide = abs(self.along(centre)) > self.params["half_width_m"]
            self.finish(t, "wide_of_gate" if wide else "over_gate" if miss > 0.0 else "not_kicked")
            return
        super().advance(p0, v0, t0, p1, v1, t1)

    def details(self):
        return {"credited_distance_to_gate_m": self.distance_m, "start_distance_to_gate_m": self.start_distance_m,
                "gate_crossing": self.gate_crossing}


def verify(config, layout):
    return drill_verifier(PassGate, config, layout)
