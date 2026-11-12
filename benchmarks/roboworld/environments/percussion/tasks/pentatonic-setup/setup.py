"""Start of pentatonic-setup: the therapy room with the accessory bar tray on the table beside the xylophone; the F4, B4
and F5 bars are to be lifted off their pins into the tray.

setup(cfg, rng, lay) is called by layout.preset_layout (through layout.goal) when it lays out the start: cfg the
episode's Config, rng the layout's generator (seeded by task and seed; the instruments' placements are drawn from it
first), lay the layout so far; it returns the task's goal (layout["goal"]): the bars to remove and where to put them."""


def setup(cfg, rng, lay):
    return dict(remove=["F4", "B4", "F5"], tray="bar_tray")
