"""Start of support: as frame (the corner loose, the rail drooped onto its upward M4 washer, both M4 screws prestarted),
with the rail 1.5 mm along its slot (RAIL_SLOT_OFFSET_M).

setup(builder, items, materials, config) gets the scene scene.build made (the Newton ModelBuilder and its item and
material tables) and the Config; presets.build calls it once the scene is built, and it returns nothing.
RAIL_SLOT_OFFSET_M is read by scene.build (the rail joint's slide coordinate); FRAME_PRETIGHTENED by
config.frame_is_pretightened."""

FRAME_PRETIGHTENED = False
RAIL_SLOT_OFFSET_M = .0015


def setup(builder, items, materials, config):
    """Nothing to add to the built scene."""
