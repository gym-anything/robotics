"""Start of steady-beat: the gait-training session (no instrument table: the drum stands in front of the robot, the
sticks, mallets and beater lie on two trap tables beside it); the beat follows the client's walking cadence, a seeded
80-120 BPM, then speeds up by 10 %.

setup(cfg, rng, lay) is called by layout.preset_layout (through layout.goal) when it lays out the start: cfg the
episode's Config, rng the layout's generator (seeded by task and seed; the instruments' placements are drawn from it
first), lay the layout so far; it returns the task's goal (layout["goal"]): the cadence (bpm), the beats at it, the
speed-up (faster_pct) and the beats after it."""


def setup(cfg, rng, lay):
    cad = int(rng.integers(80, 121))
    return dict(bpm=cad, beats=32, faster_pct=10, faster_beats=16)
