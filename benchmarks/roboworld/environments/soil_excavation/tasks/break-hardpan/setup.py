"""Start of break-hardpan: the hardpan preset, a compacted 12 cm layer over subsoil with the marked square,
3 to 5 seeded stones in and around the square, and the hand mattock carried head down in the right hand.

setup() returns the task's own entries of its start, which layout.task_entry adds to its TASKS entry and
layout.setup builds the episode spec from: held, the tool each hand slot holds at the start ("both": the two hands),
and stones, the range its seeded stone count is drawn from."""


def setup():
    return dict(held=dict(right="mattock"), stones=(3, 6))
