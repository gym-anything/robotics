"""Outcome of trap: success when the ball has been received (a leg or trunk touch, no arm or hand) and held
at rest within 1 m of the feet for 1 s with the robot upright; progress multiplies the touch, the closeness to the
feet, the slowness of the ball and how upright the robot is (T x C x S x U).

verify(config, layout) receives the world's Config and the task's layout (variants.layout, whose "verifier"
entry holds the parameters from setup.py) and returns the verifier FootballWorld builds at the start, feeds
every tick (update, update_supported) and asks for evaluation(): success, score, progress and the
football-drills-v1 reward record. The class builds on the drills' shared machinery (drills.Drill)."""
import math

from football.football_math import norm
from football.newton15.drills import Drill, Hold, clip01, drill_verifier


class Trap(Hold, Drill):
    task = "trap"
    touch_rule = "receive"
    uses_robot_pose = True

    def _reset_drill(self):
        self.received = False
        self.hold_since_s = None
        self.start_feet_distance_m = None
        self.factors = None

    def on_legal_touch(self):
        self.received = True

    def feet_distance(self, position):
        if self.robot is None:
            return None
        feet = self.robot["feet_midpoint_m"]
        return math.hypot(position[0] - feet[0], position[1] - feet[1])

    def begin(self, position):
        self.start_feet_distance_m = self.feet_distance(position)

    def live(self, time_s, position, velocity):
        d = self.feet_distance(position)
        if d is None or self.start_feet_distance_m is None:
            self.progress, self.factors = 0.0, None
            return
        p = self.params
        radius, speed = p["feet_radius_m"], norm(velocity)
        pelvis_z, up_z = self.robot["pelvis_m"][2], self.robot["pelvis_up_z"]
        span = self.start_feet_distance_m - radius
        self.factors = {
            "T": 1.0 if self.received else 0.0,
            "C": clip01(1.0 - (d - radius) / span) if span > 0.0 else float(d <= radius),
            "S": clip01(1.0 - speed / p["speed_scale_m_s"]),
            "U": clip01((pelvis_z - p["pelvis_zero_z_m"]) / (p["pelvis_min_z_m"] - p["pelvis_zero_z_m"])),
            "feet_distance_m": d,
            "speed_m_s": speed,
            "pelvis_z_m": pelvis_z,
            "pelvis_up_z": up_z,
        }
        f = self.factors
        self.progress = f["T"] * f["C"] * f["S"] * f["U"]
        upright = pelvis_z >= p["pelvis_min_z_m"] and up_z >= math.cos(math.radians(p["max_tilt_deg"]))
        if self.held(time_s, self.received and speed < p["rest_speed_m_s"] and d <= radius and upright):
            self.finish(time_s, "controlled", success=True)

    def details(self):
        return {"received": self.received, "factors": self.factors, "hold_since_s": self.hold_since_s,
                "start_feet_distance_m": self.start_feet_distance_m}


def verify(config, layout):
    return drill_verifier(Trap, config, layout)
