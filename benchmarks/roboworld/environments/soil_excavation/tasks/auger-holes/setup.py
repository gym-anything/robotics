"""Start of auger-holes: the bulbs preset, five white marks the seed shifts together by up to 2 cm, with the
drill-mounted planting auger held in both hands (the side handle gripped, the pistol grip resting in the open right
hand).

setup() returns the task's own entries of its start, which layout.task_entry adds to its TASKS entry and
layout.setup builds the episode spec from: held, the tool each hand slot holds at the start ("both": the two hands)."""


def setup():
    return dict(held=dict(both="drill"))
