"""Start of start-and-park: the truck stopped at a travel position with the engine off and the key off, in the main
aisle east or west of the parking bay heading west toward it, or in travel lane 1 heading north toward the main aisle;
the job is to start it and park it in the bay.

target(rng) returns the work order's target, dict(start_from=...), drawn from task._target's stream; setup(start) gets
the task.Start that task.resolve builds on, adds the truck's pose and condition to start.s, keeps the parking bay clear
of a wet patch's sign and narrows start.wet_x to the way to the bay. It returns nothing."""
import math

PARK_STARTS = ("main aisle east", "main aisle west", "lane 1")


def target(rng):
    return dict(start_from=PARK_STARTS[int(rng.integers(0, len(PARK_STARTS)))])


def setup(start):
    s, j, tg, keep_clear = start.s, start.j, start.target, start.keep_clear
    # the truck stopped at a travel position, key off; the job: start it and return it to the parking bay
    s["start_from"] = tg["start_from"]
    if tg["start_from"] == "lane 1":                     # in travel lane 1, heading north toward the main aisle
        s.update(truck=(22.0 + j(0.3), 19.5 + j(1.5), math.pi / 2 + j(math.radians(4))))
        start.wet_x = (9.0, 17.0)
    else:                                                # in the main aisle, heading west toward the bay
        x0 = 28.5 if tg["start_from"] == "main aisle east" else 14.5
        s.update(truck=(x0 + j(2.5), 28.25 + j(0.25), math.pi + j(math.radians(4))))
        start.wet_x = (9.0, max(9.5, s["truck"][0] - 4.5))     # on the way to the bay
    s.update(engine="off", key="off")
    keep_clear.append(((0.0, 6.5), (26.0, 30.5)))       # the parking bay
