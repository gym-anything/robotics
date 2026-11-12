"""Start of load-bales: three cardboard bales standing on dunnage strips in the bale row, the trailers at doors 3 and 4
both restrained and chocked, a pair of strips pre-laid at the nose of the target trailer for each bale position, and
the truck on the dock floor west of the bale row with empty forks at 0.10 m.

target(rng) returns the work order's target, dict(door=...), drawn from task._target's stream; setup(start) gets the
task.Start that task.resolve builds on once the trailer bed is drawn, gives start.trailers and adds the bales, the
strips, the truck's pose and the target to start.s. It returns nothing."""
import math

from forklift import params as P

BALE_DOORS = ("door3", "door4")                             # load-bales: the two doors nearest the bale row, both spotted
DUNNAGE_Y = 0.13      # bale dunnage strips at +-0.13 m from the bale centre line: inside the blades (inner edges 0.242 m)


def target(rng):
    return dict(door=BALE_DOORS[int(rng.integers(0, len(BALE_DOORS)))])


def setup(start):
    s, rng, j, Z = start.s, start.rng, start.j, start.zones
    door, d, nose_y, bed = start.door, start.door_zone, start.nose_y, start.bed
    start.trailers = {dn: dict(bed_z=bed, restraint=True, chocks=True) for dn in BALE_DOORS}
    br = Z["bale_row"]
    bs = P.BALE["size"]                         # 60 x 30 x 48 in: the 60 in along the forks (42 in blades)
    dl, dw, dh = P.DUNNAGE["size"][2], P.DUNNAGE["size"][0], P.DUNNAGE["size"][1]
    for k in range(3):
        # side by side along the row, every bale's 60 in end facing the dock floor to the west
        bx = br["x"][0] + 1.5
        by = (br["y"][0] + br["y"][1]) / 2 + 1.8 * (k - 1)
        # two 4x4 strips along the fork direction, inside the blades (+-293 mm) so the forks can enter
        for dd, sy in (("a", -1), ("b", 1)):
            s["loads"].append(dict(kind="box", name=f"dunnage_{k}{dd}", centre=(bx, by + sy * DUNNAGE_Y, 0.0),
                                   size=(dl, dw, dh), mass=P.DUNNAGE["mass"], mu=0.5, material="dunnage"))
        s["loads"].append(dict(kind="box", name=f"bale_{k}", centre=(bx, by, dh),
                               size=(bs[0], bs[1], bs[2]), mass=float(rng.uniform(*P.BALE["mass"])),
                               mu=P.BALE["mu"], material="bale"))
    # the same strips pre-laid at the target trailer's nose, one pair per bale position, side by side across the
    # trailer (left and right as seen facing the nose; strips along the trailer)
    for side, sx in (("L", -1), ("R", 1)):
        cx = d["x"] + sx * (bs[1] / 2 + 0.03)
        for dd, sy in (("a", -1), ("b", 1)):
            s["loads"].append(dict(kind="box", name=f"nose_dunnage_{side}{dd}",
                                   centre=(cx + sy * DUNNAGE_Y, nose_y - 0.03 - dl / 2, bed),
                                   size=(dw, dl, dh), mass=P.DUNNAGE["mass"], mu=0.5, material="dunnage"))
    s.update(truck=(16.0 + j(0.5), 6.0 + j(0.4), 0.0 + j(math.radians(5))), lift=0.10)
    s["target"] = dict(door=door)
