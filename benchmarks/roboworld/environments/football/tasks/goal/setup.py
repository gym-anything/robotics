"""Start of goal: the start every task builds on, unchanged: the ball at rest on the penalty mark, the robot's
pelvis 0.42 m behind it facing the goal; no props, markers or verifier parameters.

setup(task, seed, start) receives the task name, the seed and the start every task builds on (variants.layout:
the ball at rest on the penalty mark, the robot's pelvis 0.42 m behind it, yaw 0), updates that start in place
and returns (props, markers, verifier, summary): static props, render-only markers, the verifier's parameters
and the seed's drawn values."""


def setup(task, seed, start):
    return [], [], {}, {}
