"""Start of sensor: the frame corner already secured (the construction settle tightens both M4s first) and the staged
W4F replacement sensor loose on its bracket at the end of its arc slot, both M3 screws prestarted, the reference flag at
its home.

setup(builder, items, materials, config) gets the scene scene.build made (the Newton ModelBuilder and its item and
material tables) and the Config; presets.build calls it once the scene is built, and it returns nothing. Under the
eight-task contracts it adds the flag's home stop (presets.add_flag_home_stop). FRAME_PRETIGHTENED is read by
config.frame_is_pretightened."""
from tighten_codex.presets import add_flag_home_stop
from tighten_codex.task import uses_eight

# Sensor replacement is done on a secured frame: both M4s start tightened.
FRAME_PRETIGHTENED = True


def setup(builder, items, materials, config):
    if uses_eight(config):
        add_flag_home_stop(builder, materials, config)
