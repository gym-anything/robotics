"""Start of drive-to-pallet: a pallet standing in one of the four staging lanes along the dock wall, within 0.3 m and
10 deg of the lane's middle, and the truck in the main aisle about 9 m west of it, heading east with its forks down.

target(rng) returns the work order's target, dict(lane=...), drawn from task._target's stream; setup(start) gets the
task.Start that task.resolve builds on, adds the pallet and the truck's pose to start.s and keeps the turn into the
lane clear of a wet patch's sign. It returns nothing."""
import math

STAGING_LANES = ("S1", "S2", "S3", "S4")


def target(rng):
    return dict(lane=STAGING_LANES[int(rng.integers(0, len(STAGING_LANES)))])


def setup(start):
    s, j, tg, Z, keep_clear = start.s, start.j, start.target, start.zones, start.keep_clear
    s["lane"] = tg["lane"]
    st = Z["staging"][tg["lane"]]
    px = (st["x"][0] + st["x"][1]) / 2 + j(0.3)
    py = 34.6 + j(0.2)
    # the pallet's pose: +-0.3 m and +-10 deg
    s["loads"].append(dict(kind="pallet", name="target", pose=(px, py, math.pi / 2 + j(math.radians(10)))))
    s.update(truck=(px - 9.0 + j(1.5), 28.25 + j(0.4), j(math.radians(10))))
    keep_clear.append(((px - 4.0, px + 1.5), (26.0, 31.0)))     # the turn into the lane
