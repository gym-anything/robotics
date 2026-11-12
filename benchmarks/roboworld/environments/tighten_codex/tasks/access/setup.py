"""Start of access: as frame (the corner loose, the rail drooped onto its upward M4 washer, both M4 screws prestarted),
with a removable service guard and its return lip narrowing the way to the upward M4 (guard).

setup(builder, items, materials, config) gets the scene scene.build made (the Newton ModelBuilder and its item and
material tables) and the Config; presets.build calls it once the scene is built, and it returns nothing. scene.build
calls guard(box) with its box helper where the guard's shapes belong among the scene's; FRAME_PRETIGHTENED is read by
config.frame_is_pretightened."""

FRAME_PRETIGHTENED = False


def guard(box):
    # Guard panel behind the corner and a bent lower return lip beside
    # the upward M4 corridor. Its nearest corner is 5.41 mm from the
    # top-M4 axis: 2.24 mm radial clearance for the 3.17 mm shaft.
    box("removable_access_guard", (.075,.1245,1.064),
        (.050,.002,.042), "safety_yellow")
    box("removable_access_guard_return_lip", (.075,.1065,1.059),
        (.050,.002,.009), "safety_yellow")


def setup(builder, items, materials, config):
    """Nothing to add to the built scene."""
