"""Start of put-away-rack: the truck at the entry of one of the two working aisles from the main aisle, heading down
it with a wrapped unit load of 600-1,000 kg and 0.9-1.2 m on its forks at travel height; the target rack location (a
rack face along that aisle, bay 1-7, beam level, pallet position) is drawn per seed and left empty.

rack_target(config) draws the location from task._target's stream and reserved_locations(config) names it for the
facility to leave empty (task.reserved_locations). setup(start) gets the task.Start that task.resolve builds on and
adds the truck's pose, its load, the rack location and the aisle to start.s. It returns nothing."""
import math

from forklift.task import _target_rng

RACK_FACES = {"1": ("A-e", "B-w"), "2": ("B-e", "C-w")}     # the two rack faces along each working aisle (row-face)


def rack_target(config):
    """put-away-rack's location of (task, seed): one of the four rack faces along the two working aisles, bay 1-7, the
    first beam level in two seeds of three and the second in one, either pallet position. Returns (name, aisle)."""
    rng = _target_rng(config)
    aisle = ("1", "2")[int(rng.integers(0, 2))]
    rowface = RACK_FACES[aisle][int(rng.integers(0, 2))]
    bay = int(rng.integers(1, 8))
    level = 1 if rng.random() < 2 / 3 else 2
    slot = int(rng.integers(0, 2))
    return f"{rowface}-{bay:02d}-{level}-{slot}", aisle


def reserved_locations(config):
    """The put-away slot only; its neighbour on the same beams is stocked like any other location (a stored load in
    four of five)."""
    return {rack_target(config)[0]}


def setup(start):
    s, rng, j, Z = start.s, start.rng, start.j, start.zones
    loc, aisle = rack_target(start.config)
    xa, xb = Z["aisles"][aisle]
    # at the aisle's entry from the main aisle, heading down it
    s.update(truck=((xa + xb) / 2 + j(0.25), 24.6 + j(0.3), -math.pi / 2 + j(math.radians(6))), lift=0.15, tilt_deg=-3.0)
    s["load_on_forks"] = dict(kind="wrapped", mass=float(rng.uniform(600.0, 1000.0)), height=float(rng.uniform(0.9, 1.2)))
    s["rack_location"] = loc
    s["aisle"] = aisle
