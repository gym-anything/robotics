"""Start of find-loose: one screw loose, chosen by the seed from both M3s and the forward M4 (loose_screw), and the
other three tightened by the construction settle's prior assembly (presets.settle_initial), the rail level, the
reference flag at its home with the sensor detecting it.

setup(builder, items, materials, config) gets the scene scene.build made (the Newton ModelBuilder and its item and
material tables) and the Config; presets.build calls it once the scene is built, and it returns nothing. It sets the
level rail and the screws' start coordinates (presets.apply_start_coordinates) and adds the flag's home stop
(presets.add_flag_home_stop). loose_screw(seed) names the seed's loose screw (presets.loose_screw); FRAME_PRETIGHTENED
is read by config.frame_is_pretightened."""
import hashlib

from tighten_codex.presets import add_flag_home_stop, apply_start_coordinates
from tighten_codex.task import uses_eight

LOOSE_CHOICES = ("sensor_left", "sensor_right", "frame_front")
# The frame's prior assembly is the single-loose settle, not the secured-frame ramp.
FRAME_PRETIGHTENED = False


def loose_screw(seed):
    digest = hashlib.sha256(f"tighten-find-loose-v1:{seed}".encode()).digest()
    return LOOSE_CHOICES[int.from_bytes(digest[:8], "big") % len(LOOSE_CHOICES)]


def setup(builder, items, materials, config):
    apply_start_coordinates(builder, items, config)
    if uses_eight(config):
        add_flag_home_stop(builder, materials, config)
