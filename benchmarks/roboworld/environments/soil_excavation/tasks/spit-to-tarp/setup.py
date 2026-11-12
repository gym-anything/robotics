"""Start of spit-to-tarp: the bed preset, with the spade held upright in both hands a few cm above the white mark
and the blue tarp spread on the walkway beside the robot.

setup() returns the task's own entries of its start, which layout.task_entry adds to its TASKS entry and
layout.setup builds the episode spec from: held, the tool each hand slot holds at the start ("both": the two hands)."""


def setup():
    return dict(held=dict(both="spade"))
