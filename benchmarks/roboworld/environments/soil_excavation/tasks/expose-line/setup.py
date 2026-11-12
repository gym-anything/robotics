"""Start of expose-line: the irrigation preset, the lateral about 30 cm under the bed marked by purple flags and a
line with its tolerance zone, and the trowel in the right hand over the line in front of the robot.

setup() returns the task's own entries of its start, which layout.task_entry adds to its TASKS entry and
layout.setup builds the episode spec from: held, the tool each hand slot holds at the start ("both": the two hands)."""


def setup():
    return dict(held=dict(right="trowel"))
