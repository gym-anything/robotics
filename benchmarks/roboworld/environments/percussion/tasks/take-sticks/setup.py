"""Start of take-sticks: the gait-training session (the drum in front of the robot, the two drumsticks, the mallets and
the beater lying in their saddles on two trap tables beside it); both sticks are to be held ready over the drumhead for
hold_s and put back.

setup(cfg, rng, lay) is called by layout.preset_layout (through layout.goal) when it lays out the start: cfg the
episode's Config, rng the layout's generator (seeded by task and seed; the instruments' placements are drawn from it
first), lay the layout so far; it returns the task's goal (layout["goal"]): how long to hold (hold_s), the sticks and
what to hold them over."""


def setup(cfg, rng, lay):
    return dict(hold_s=5.0, sticks=["stick_left", "stick_right"], over="drum")
