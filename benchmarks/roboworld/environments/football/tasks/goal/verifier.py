"""Outcome of goal: success when the whole ball crosses the goal line between the posts and under the
crossbar with a leg as the robot's last touch and no arm or hand touch; score 1 on a goal, else 0.9 x
progress toward the goal window.

verify(config, layout) receives the world's Config and the task's layout and returns the verifier
FootballWorld builds at the start, feeds every tick (update, update_supported) and asks for evaluation():
goal's detector, task.PitchAwareGoalDetector (contract football-goal-progress-v1), with the kick rule when a
robot plays. Its rules stay in task.py, shared: free-kick uses it unchanged and every drill builds on it."""
from football.newton15.task import PitchAwareGoalDetector


def verify(config, layout):
    return PitchAwareGoalDetector(require_kick=config.robot)
