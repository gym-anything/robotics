"""Start of load-trailer: the start it shares with load-trailer-chocked (task.resolve: a trailer at one of the four
doors with two loaded pallets across its nose, the truck in the staging lane in front of the door with a wrapped unit
on its forks), except that in one seed of five the trailer is not secured: no restraint hooked and no chocks.

target(rng) returns dict(secured=...), drawn after the door and side (task._target); setup(start) gets the task.Start
that task.resolve builds on once the trailer bed is drawn, and replaces start.trailers for an unsecured trailer. It
returns nothing."""

UNSECURED_SHARE = 0.2                                       # load-trailer seeds whose trailer arrives unsecured


def target(rng):
    # in one seed of five the trailer at the door is not secured (no restraint hooked, no chocks): the door's
    # restraint signal shows red and the job is to stop short of the leveler
    return dict(secured=bool(rng.random() >= UNSECURED_SHARE))


def setup(start):
    door, bed, tg = start.door, start.bed, start.target
    if not tg["secured"]:
        start.trailers = {door: dict(bed_z=bed, restraint=False, chocks=False)}  # held by its own spring brakes only
