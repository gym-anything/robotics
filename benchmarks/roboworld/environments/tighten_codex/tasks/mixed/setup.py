"""Start of mixed: the W4F's two M3 screws and the corner's two M4 screws loose and prestarted, the rail drooped onto
its upward M4 washer, the reference flag at its home with a home stop on the worktop at its track's return end
(home_stop).

setup(builder, items, materials, config) gets the scene scene.build made (the Newton ModelBuilder and its item and
material tables) and the Config; presets.build calls it once the scene is built, and it returns nothing. scene.build
calls home_stop(box, flag_x, cfg) with its box and shape-config helpers and the flag's x after every other shape of the
scene; FRAME_PRETIGHTENED is read by config.frame_is_pretightened."""

FRAME_PRETIGHTENED = False


def home_stop(box, flag_x, cfg):
    # Task prop (added last so every existing shape keeps its index): a
    # steel home stop with a damped bumper face on the worktop at the
    # track's return end. Sliding the flag back until its foot meets the
    # stop leaves it 0.1 mm past its home, where the sensor sees it;
    # without it the only return-side stop was the slide's -6 mm limit,
    # where the beam misses. The flag's own joint, limits, damping and
    # friction are unchanged. Contact gains of this prop only: Newton's
    # force-space mapping scales the pair's mean ke/kd by the 0.16 kg
    # carriage's body_invweight0 x (1 - dmax) = 2.09 x 0.05; with the
    # default 2e6/150 the pair had a 128 ms time constant and damping
    # ratio 0.017 and threw the flag back out of the beam (measured).
    # These give 4 dt and a damping ratio of ~1.
    box("flag_home_stop",(flag_x-.020-.0001-.004,-.063,.785),(.004,.010,.005),"robot_dark",
        shape_cfg=cfg(mu=.45,ke=2.5e7,kd=4.6e4))


def setup(builder, items, materials, config):
    """Nothing to add to the built scene."""
