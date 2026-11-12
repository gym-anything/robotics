"""Start of aisle-cruise: the truck at one end of the main aisle facing along it, parking brake set and forks at
0.10 m: in the parking bay at the west end for an eastward cruise, or at the aisle's east end for a westward one.

target(rng) returns the work order's target, dict(cruise=...), drawn from task._target's stream; setup(start) gets the
task.Start that task.resolve builds on and adds the cruise direction and the truck's pose to start.s. It returns
nothing."""
import math

CRUISE_DIRECTIONS = ("east", "west")       # aisle-cruise: the main aisle either way, to the far end's white line


def target(rng):
    return dict(cruise=CRUISE_DIRECTIONS[int(rng.integers(0, len(CRUISE_DIRECTIONS)))])


def setup(start):
    s, j, tg = start.s, start.j, start.target
    s["cruise"] = tg["cruise"]
    if tg["cruise"] == "east":                           # from the parking bay at the aisle's west end
        s.update(truck=(3.5 + j(0.4), 28.25 + j(0.3), j(math.radians(5))), park=True, lift=0.10)
    else:                                                # from the aisle's east end (counterweight >= 0.5 m off the
        s.update(truck=(46.8 + j(0.3), 28.25 + j(0.3), math.pi + j(math.radians(5))), park=True, lift=0.10)  # wall)
