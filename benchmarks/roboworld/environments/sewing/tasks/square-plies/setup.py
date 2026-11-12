"""Start of square-plies: the load start (BASE: the two 200 x 420 mm plies stacked on the table left of the machine,
the foot down on the empty plate, with load's seeded stack offsets) with the top ply slipped on the stack, moved
40 mm right and 25 mm away and turned 6 deg about its centre before the start's relaxation.

BASE is the task whose plies and settings this one reuses (task.TASK_SETUP copies them; task_set.BASE_TASK). setup(env,
config) is called by the task adapter (task_set_env.TaskEnvironment, and its sewing-tasks-v2 subclass) with the
episode's Config before the world is built; it returns the Config the world is built from: the episode's own as
task_set_env.SquareConfig, whose task_setup_revision (REVISION) keys a cached start state of its own, never the
aligned load start's. prepare(env, make) is the adapter's start preparation (TaskEnvironment._make_start): it slips the
top ply (slip_top), prepares the start (make: the relaxation, or the cached state) and records the initial condition.
START_STATE names the start; TASK_START describes it in the variation catalogue (variations.py)."""
import numpy as np

BASE = "load"
REVISION = "tray-slipped-40x25mm-yaw6-v1"
START_STATE = "tray_slipped"
TASK_START = dict(name=START_STATE, top_ply_translation_m=[.040, .025], yaw_deg=6., revision=REVISION,
                  preparation="before existing relaxation; separate cache")


def slip_top(positions):
    """Rigid initial placement: +40 mm x, +25 mm y, 6 degrees about the panel centre.

    Using the bounding-box centre makes the placement independent of mesh density.
    It changes positions only, before any episode or preparation tick.
    """
    out = np.array(positions, copy=True)
    centre = (out[:, :2].min(axis=0) + out[:, :2].max(axis=0))/2
    a = np.deg2rad(6.)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    out[:, :2] = (out[:, :2] - centre) @ rotation.T + centre + [.040, .025]
    return out


def setup(env, config):
    from sewing.task_set_env import SquareConfig
    # Unique cache: never restore an aligned load state as the slipped one.
    return SquareConfig(**vars(config))


def prepare(env, make):
    if env.world.cloth is not None:
        cl = env.world.cloth
        p = cl.plies[env.world.ply_ids[1]]
        snap = cl.snapshot()
        idx = slice(p["v0"], p["v0"] + p["nv"])
        snap["x"][idx] = slip_top(snap["x"][idx])
        cl.restore(snap)
    make()
    env.assisted_log.append(dict(initial_condition=REVISION,
                                 note="top ply displaced before relaxation; no physical parameters changed"))
