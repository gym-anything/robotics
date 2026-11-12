"""Start of cable-trench: the trench preset, the white line between the two path lights in the lawn south of the
walkway, with the trenching shovel held in both hands over the point of the line in front of the robot and the tarp on
the lawn side.

setup() returns the task's own entries of its start, which layout.task_entry adds to its TASKS entry and
layout.setup builds the episode spec from: held, the tool each hand slot holds at the start ("both": the two hands)."""


def setup():
    return dict(held=dict(both="trench_shovel"))
