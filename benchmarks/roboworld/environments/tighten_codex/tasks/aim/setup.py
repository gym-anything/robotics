"""Start of aim: as sensor (the frame corner secured, the W4F loose at the end of its arc slot, both M3 screws
prestarted), with the reference flag's home 3.5 mm along its carriage (FLAG_HOME_M), so the light spot misses it.

setup(builder, items, materials, config) gets the scene scene.build made (the Newton ModelBuilder and its item and
material tables) and the Config; presets.build calls it once the scene is built, and it returns nothing. Under the
eight-task contracts it adds the flag's home stop at that home (presets.add_flag_home_stop). FLAG_HOME_M is the flag's
start and home (presets.flag_home, scene.build); FRAME_PRETIGHTENED is read by config.frame_is_pretightened."""
from tighten_codex.presets import add_flag_home_stop
from tighten_codex.task import uses_eight

# Aiming is done on a secured frame: both M4s start tightened.
FRAME_PRETIGHTENED = True
FLAG_HOME_M = -.0035


def setup(builder, items, materials, config):
    if uses_eight(config):
        add_flag_home_stop(builder, materials, config)
