"""Start of ramp-down-loaded: the truck inside the yard trailer at the top of the ramp, forks toward the nose, 3-9 m in
from the rear sill, a wrapped unit load of 900-1,300 kg on its forks; the drop zone to set it down in lies east or west
of the ramp's foot.

target(rng) returns the work order's target, dict(drop_zone=...), drawn from task._target's stream; setup(start) gets
the task.Start that task.resolve builds on, after the yard's trailer, and adds the truck's pose, its load and the drop
zone to start.s. It returns nothing."""
import math

DROP_ZONES = ("east", "west")              # ramp-down-loaded: the painted yard drop zone either side of the ramp's lane


def target(rng):
    return dict(drop_zone=DROP_ZONES[int(rng.integers(0, len(DROP_ZONES)))])


def setup(start):
    s, rng, j, tg = start.s, start.rng, start.j, start.target
    rp = start.zones["ramp"]
    # inside the trailer, forks toward the nose, where the load was picked: 3-9 m in from the rear sill (the
    # truck's rear axle at least 1.4 m inside it; its forks at least 4 m short of the nose)
    s.update(truck=(rp["x"] + j(0.1), rp["y_rear"] - float(rng.uniform(3.0, 9.0)), -math.pi / 2 + j(math.radians(2))),
             lift=0.15, tilt_deg=-3.0)
    s["load_on_forks"] = dict(kind="wrapped", mass=float(rng.uniform(900.0, 1300.0)), height=float(rng.uniform(1.0, 1.3)))
    s["truck_z"] = rp["rise"]
    s["drop_zone"] = tg["drop_zone"]
