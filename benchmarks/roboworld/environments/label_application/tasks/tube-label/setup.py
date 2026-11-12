"""Start of tube-label: the blood collection station's bench (scene.specimen) with the blood collection tube in its
cradle, turned to the seed's roll about its own axis (layouts.tube_roll: any angle, seed 0 manufacturer label up), and
the printer presenting the 2 x 1 in patient label it has just printed, carrying the seed's accession number (payload).
It is tube-wrap's bench, tube and label, seed for seed.

setup(world) gets the task's world (world.World), built and not yet settled, and returns the start information LabelEnv
keeps as start_info (env.prepare_start calls it). tube-label starts from the printer's presented label as the scene
builds it: setup constructs nothing and returns {}. payload(seed) is the barcode payload printed for the seed
(layouts.payload calls it): an accession number, as on tube-wrap. WRIST_CAMERAS: the backend turns the wrist cameras on
by default."""
from label_application.layouts import _rng

# wrist cameras on by default (backend), as on tube-wrap: the manufacturer label's edge for 0-2 mm registration (the
# 13 mm tube is ~11 px across overhead)
WRIST_CAMERAS = True


def payload(seed):
    return "S260923" + "".join(str(d) for d in _rng(seed, 13).integers(0, 10, 4))


def setup(world):
    return {}
