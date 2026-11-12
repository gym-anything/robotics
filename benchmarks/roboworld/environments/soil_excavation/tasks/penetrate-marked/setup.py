"""Start of penetrate-marked: the bed preset, with the stainless digging spade held upright in both hands a few
cm above the white paint mark (the seed moves the mark up to 8 cm across and 4 cm along the bed).

setup() returns the task's own entries of its start, which layout.task_entry adds to its TASKS entry and
layout.setup builds the episode spec from: held, the tool each hand slot holds at the start ("both": the two hands)."""


def setup():
    return dict(held=dict(both="spade"))
