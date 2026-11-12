"""Start of centre-flask: the flask, with the stir bar in it, nudged 13-16.5 mm off the centre of the stirrer plate in
a seeded direction, so the burette tip no longer points into its mouth; the whole flask footprint stays on the plate.

arrange(s) moves the flask on task_presets.Start (task_presets.preset, seeded by the variation); setup(config, **kwargs)
receives task_session.make_session's arguments and returns the task's session (task_session.catalogue_session)."""
import math

from titrate.newton15.task_session import catalogue_session


def arrange(s):
    config, rng, poses = s.config, s.rng, s.poses
    # Keep the entire 67 mm flask footprint on the 100 mm plate.
    radius, angle = rng.uniform(.013, .0165), rng.uniform(0., 2*math.pi)
    dx, dy = radius*math.cos(angle), radius*math.sin(angle)
    x, y, z = config.flask_position
    poses['flask'] = (x+dx, y+dy, z)
    poses['stir_bar'] = (x+dx, y+dy, z+.005)


def setup(config=None, **kwargs):
    return catalogue_session('centre-flask', config, **kwargs)
