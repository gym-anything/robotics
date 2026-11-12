"""Start of melody: the therapy room (the xylophone on the table, two resonator bars, the drum on its stand, the mallets
in the tool tray); a seeded tune (one of two from TUNES) is to be played on the xylophone at a seeded 80-100 BPM.

setup(cfg, rng, lay) is called by layout.preset_layout (through layout.goal) when it lays out the start: cfg the
episode's Config, rng the layout's generator (seeded by task and seed; the instruments' placements are drawn from it
first), lay the layout so far; it returns the task's goal (layout["goal"]): the tune, its notes (note, beats), the
tempo (bpm) and the instrument. SAME_GOAL_ON_G1: on the G1 the goal is drawn again from the seed stream as it stood
before the G1's own drum placement, so a seed asks for the same tune at the same tempo on every robot
(layout.preset_layout)."""

TUNES = {
    # (note, beats) - Orff/preschool repertoire in C
    "hot-cross-buns": [("E4", 1), ("D4", 1), ("C4", 2), ("E4", 1), ("D4", 1), ("C4", 2), ("C4", .5), ("C4", .5),
                       ("C4", .5), ("C4", .5), ("D4", .5), ("D4", .5), ("D4", .5), ("D4", .5), ("E4", 1), ("D4", 1),
                       ("C4", 2)],
    "mary-had-a-little-lamb": [("E4", 1), ("D4", 1), ("C4", 1), ("D4", 1), ("E4", 1), ("E4", 1), ("E4", 2),
                               ("D4", 1), ("D4", 1), ("D4", 2), ("E4", 1), ("G4", 1), ("G4", 2)],
    "pentatonic-call": [("G4", 1), ("E4", 1), ("G4", 1), ("E4", 1), ("A4", .5), ("G4", .5), ("E4", 1), ("D4", 2)],
}
SAME_GOAL_ON_G1 = True


def setup(cfg, rng, lay):
    name = ("hot-cross-buns", "mary-had-a-little-lamb")[int(rng.integers(0, 2))]
    return dict(tune=name, notes=TUNES[name], bpm=int(rng.integers(80, 101)), instrument="xylophone")
