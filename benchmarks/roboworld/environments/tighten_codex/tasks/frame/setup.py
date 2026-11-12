"""Start of frame: the extrusion corner loose, the rail drooped nose-down onto its upward M4 washer, both M4 screws
prestarted; the scene builds it as it is (scene.build), with nothing added for this task.

setup(builder, items, materials, config) gets the scene scene.build made (the Newton ModelBuilder and its item and
material tables) and the Config; presets.build calls it once the scene is built, and it returns nothing.
FRAME_PRETIGHTENED is read by config.frame_is_pretightened."""

# The corner is the job: its M4s start loose.
FRAME_PRETIGHTENED = False


def setup(builder, items, materials, config):
    """Nothing to add to the built scene."""
