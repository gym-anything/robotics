"""Start of finger-tap: the therapy room (the xylophone and two resonator bars on the table, the drum on its stand
beside it, the tools in their tray); the drum is to be tapped with a fingertip, softly, at a slow steady tempo.

setup(cfg, rng, lay) is called by layout.preset_layout (through layout.goal) when it lays out the start: cfg the
episode's Config, rng the layout's generator (seeded by task and seed; the instruments' placements are drawn from it
first), lay the layout so far; it returns the task's goal (layout["goal"]): the taps asked (count), their tempo
(bpm), dynamic and tool."""


def setup(cfg, rng, lay):
    return dict(count=8, bpm=80, dynamic="mp", tool="fingertip")
