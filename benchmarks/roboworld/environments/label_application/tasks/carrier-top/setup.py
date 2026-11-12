"""Start of carrier-top: the shipping pack bench (scene.parcel) with the sealed 16 x 12 x 8 in carton at its seeded
place and turn on the bench, and the desktop printer presenting the 4 x 6 in shipping label it has just printed,
carrying the seed's 18-digit carrier tracking number (payload). Nothing is on the carton yet.

setup(world) gets the task's world (world.World), built and not yet settled, and returns the start information LabelEnv
keeps as start_info (env.prepare_start calls it). carrier-top starts from the printer's presented label as the scene
builds it: setup constructs nothing and returns {}. payload(seed) is the barcode payload printed for the seed
(layouts.payload calls it): PAYLOAD at seed 0, a drawn tracking number otherwise."""
from label_application.layouts import _rng

PAYLOAD = "961201950123456789"


def payload(seed):
    if seed == 0:
        return PAYLOAD
    r = _rng(seed, 1)
    return "9612019" + "".join(str(d) for d in r.integers(0, 10, 11))           # 18-digit tracking number


def setup(world):
    return {}
