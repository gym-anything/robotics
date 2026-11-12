"""Start of accent-ostinato: the gait-training session (the drum in front of the robot, the tools on two trap tables
beside it); an ostinato of three soft strokes and an accent per bar, at 116 BPM for 8 bars.

setup(cfg, rng, lay) is called by layout.preset_layout (through layout.goal) when it lays out the start: cfg the
episode's Config, rng the layout's generator (seeded by task and seed; the instruments' placements are drawn from it
first), lay the layout so far; it returns the task's goal (layout["goal"]): the tempo (bpm), the bars, the accent
pattern within a bar (1: the accent) and the accent's contrast (accent_db)."""


def setup(cfg, rng, lay):
    return dict(bpm=116, bars=8, pattern=[0, 0, 0, 1], accent_db=6.0)
