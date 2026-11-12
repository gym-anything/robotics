"""Outcome of save: success when the machine's shot is resolved without a goal (the ball at rest for 1 s, back
beyond the goal-area line moving away, out of play, or 5 s after launch); progress only after the keeper's
first touch.

verify(config, layout) receives the world's Config and the task's layout (variants.layout, whose "verifier"
entry holds the parameters from setup.py) and returns the verifier FootballWorld builds at the start, feeds
every tick (update, update_supported) and asks for evaluation(): success, score, progress and the
football-drills-v1 reward record. The class builds on the drills' shared machinery (drills.Drill)."""
from football.football_math import norm
from football.newton15.drills import Drill, Hold, clip01, drill_verifier


class Save(Hold, Drill):
    task = "save"
    touch_rule = "keeper"

    def _reset_drill(self):
        self.rest_since_s = None
        self.goal_speed_m_s = None

    def incoming_speed(self):
        return max(norm(self.first_touch["incoming_velocity_m_s"]), self.params["rest_speed_m_s"])

    def live(self, time_s, position, velocity):
        if self.first_touch is None:
            self.progress = 0.0
        else:
            self.progress = 0.5 * clip01(1.0 - max(velocity[0], 0.0) / self.incoming_speed())

    def advance(self, p0, v0, t0, p1, v1, t1):
        self.live(t1, p1, v1)
        p = self.params
        if self.held(t1, norm(v1) < p["rest_speed_m_s"], key="rest_since_s"):
            self.finish(t1, "saved_ball_at_rest", success=True)
        elif p1[0] < p["area_line_x_m"] and v1[0] < 0.0:
            self.finish(t1, "saved_cleared_upfield", success=True)
        elif self.left_sector(p1):
            self.finish(t1, "saved_out_of_play", success=True)
        elif t1 >= p["deadline_s"] - 1e-9:
            self.finish(t1, "saved_until_deadline", success=True)

    def goal_line(self, time_s, centre, velocity, entry):
        if not entry["inside"]:
            self.finish(time_s, "saved_out_over_goal_line", success=True)
            return
        self.goal_speed_m_s = norm(velocity)
        self.progress = 0.0 if self.first_touch is None else 0.5 * clip01(
            1.0 - self.goal_speed_m_s / self.incoming_speed())
        self.finish(time_s, "goal_conceded")

    def details(self):
        return {"incoming_speed_m_s": None if self.first_touch is None else self.incoming_speed(),
                "goal_speed_m_s": self.goal_speed_m_s, "rest_since_s": self.rest_since_s,
                "frame_entry": self.frame_entry}


def verify(config, layout):
    return drill_verifier(Save, config, layout)
