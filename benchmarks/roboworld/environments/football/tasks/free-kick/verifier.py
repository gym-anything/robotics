"""Outcome of free-kick: goal's success, unchanged, from the free-kick spot past the wall: the whole ball
over the goal line inside the frame with a leg as the robot's last touch and no arm or hand touch.

verify(config, layout) receives the world's Config and the task's layout and returns the verifier
FootballWorld builds at the start, feeds every tick (update, update_supported) and asks for evaluation():
goal's detector, task.PitchAwareGoalDetector (contract football-goal-progress-v1), with the kick rule when a
robot plays. Its rules stay in task.py, shared: goal uses it unchanged and every drill builds on it."""
from football.newton15.task import PitchAwareGoalDetector


def verify(config, layout):
    return PitchAwareGoalDetector(require_kick=config.robot)
