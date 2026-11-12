"""Start of planting-hole: the bed preset, with the spade held upright in both hands a few cm above the white
mark the hole is centred on, and the tarp on the walkway for the spoil.

setup() returns the task's own entries of its start, which layout.task_entry adds to its TASKS entry and
layout.setup builds the episode spec from: held, the tool each hand slot holds at the start ("both": the two hands)."""


def setup():
    return dict(held=dict(both="spade"))
