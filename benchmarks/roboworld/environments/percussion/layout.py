"""Scene layout per preset and seed: which instruments stand where, the tools in the robot's
rack, the task goal. Seeds vary drum tension (Wagner's low/mid/high), placements (+-20 mm instruments and drum),
the client's cadence, tunes, tempi and the echo rhythm. All placements are authored; none is moved to suit a robot.
The goal of a benchmark task is its folder's setup (tasks/<task>/setup.py); the other tasks' goals are below.
"""
import math
import numpy as np

from gym_anything_robotics import task_folders

from . import params as P

PACKAGE = "percussion"
BORDUNS = {"C": ("C4", "G4"), "F": ("F4", "C5")}
DRUM_YAW = math.radians(30.0)           # stand rotation jitter: no tripod leg toward the table legs (authored)


def _rng(cfg):
    return np.random.default_rng(int(cfg.seed)*7919 + sum(map(ord, cfg.task)))


def is_g1(cfg):
    return str(getattr(cfg, "robot", "")) in ("g1", "percussion-g1")


def task_setup(task):
    """The benchmark task's setup.py (tasks/<task>/), or None for a task outside the benchmark."""
    return task_folders.setup(PACKAGE, task) if task_folders.exists(PACKAGE, task) else None


def task_verifier(task):
    """The benchmark task's verifier.py (tasks/<task>/), or None for a task outside the benchmark."""
    return task_folders.verifier(PACKAGE, task) if task_folders.exists(PACKAGE, task) else None


def preset_layout(cfg):
    rng = _rng(cfg)
    j = lambda s: float(rng.uniform(-s, s))
    lay = dict(preset=cfg.preset, task=cfg.task, seed=cfg.seed, table=True, xylophone=None, resonator_bars=[],
               drum=None, bar_tray=None, speaker=None)
    tension = ("low", "mid", "high")[int(rng.integers(0, 3))]
    if cfg.preset in ("therapy", "pentatonic"):
        lay["xylophone"] = dict(center=(-0.05 + j(0.02), 0.75 + j(0.02)), yaw=j(math.radians(2.0)),
                                notes=list(P.XYLO_NOTES))
        lay["resonator_bars"] = [dict(note="C4", xy=(-0.47 + j(0.008), 0.775 + j(0.015))),
                                 dict(note="G4", xy=(-0.55 + j(0.008), 0.775 + j(0.015)))]
        lay["drum"] = dict(xy=(0.82 + j(0.02), 0.38 + j(0.03)), head_z=0.74, tension=tension, yaw=j(DRUM_YAW))
    if cfg.preset == "pentatonic":
        lay["bar_tray"] = dict(xy=(0.40, 0.76), half=(0.075, 0.18, 0.012))
    if cfg.preset == "assessment":
        lay["resonator_bars"] = [dict(note="C4", xy=(-0.12 + j(0.03), 0.775 + j(0.015))),
                                 dict(note="G4", xy=(0.02 + j(0.03), 0.775 + j(0.015)))]
        lay["drum"] = dict(xy=(0.82 + j(0.02), 0.38 + j(0.03)), head_z=0.74, tension=tension, yaw=j(DRUM_YAW))
        lay["speaker"] = dict(xy=(-0.48, 0.88), size=(0.15, 0.17, 0.24))
    if cfg.preset == "gait":
        lay["table"] = False                                        # moved to the wall (dressing only)
        lay["drum"] = dict(xy=(0.0 + j(0.04), 0.44 + j(0.03)), head_z=0.74, tension=tension, yaw=j(DRUM_YAW))
    goal_rng_state = rng.bit_generator.state
    if is_g1(cfg) and lay["drum"] is not None:
        from . import g1 as G1
        moved = G1.LAYOUT[cfg.preset]["drum"]
        if moved is not None:                    # the humanoid cannot walk: the drum stands at its right hand
            lay["drum"]["xy"] = (moved[0] + j(0.02), moved[1] + j(0.005))
    lay["tools"], lay["tool_supports"] = tool_rack(cfg.preset, g1=is_g1(cfg))
    lay["goal"] = goal(cfg, rng, lay)
    if getattr(task_setup(cfg.task), "SAME_GOAL_ON_G1", False) and is_g1(cfg):
        goal_rng = np.random.default_rng()
        goal_rng.bit_generator.state = goal_rng_state
        lay["goal"] = goal(cfg, goal_rng, lay)
    lay["lighting_lux"] = float(rng.uniform(300., 500.))
    return lay


def tool_rack(preset, g1=False):
    """Where the tools lie at the start: in a beech stick tray at the near edge of the instrument table,
    0.55-0.65 m in front of the arm bases (closer, the FR3 folds onto its elbow limit to reach down), both sticks tip to
    tip in the first groove (butts outward: each arm takes the stick on its side at the fulcrum), the mallets in the
    second, the felt beater in the third. Gait sessions have no table: two percussionist's trap tables beside the drum.
    Poses: tool frame x along the tool from its butt/handle end; z = the tool's axis height."""
    if preset == "gait":
        zt = TRAP_TABLE["top_z"] + SADDLE_AXIS
        # the G1 stands behind the drum: its trap tables 0.50 m to each side (clear of its hanging hands), 4 cm nearer
        ox, oy = (0.03, -0.04) if g1 else (0., 0.)
        sup = [dict(kind="trap_table", center=(-0.47 - ox, 0.34 + oy), half=TRAP_TABLE["half"], top_z=TRAP_TABLE["top_z"]),
               dict(kind="trap_table", center=(0.47 + ox, 0.34 + oy), half=TRAP_TABLE["half"], top_z=TRAP_TABLE["top_z"])]
        tools = [dict(name="stick_left", kind="stick", butt=(-0.67 - ox, 0.30 + oy, zt), dir=(1., 0., 0.)),
                 dict(name="stick_right", kind="stick", butt=(0.67 + ox, 0.30 + oy, zt), dir=(-1., 0., 0.)),
                 dict(name="mallet_left", kind="mallet", butt=(-0.62 - ox, 0.38 + oy, zt), dir=(1., 0., 0.)),
                 dict(name="mallet_right", kind="mallet", butt=(0.62 + ox, 0.38 + oy, zt), dir=(-1., 0., 0.)),
                 dict(name="beater", kind="beater", butt=(0.62 + ox, 0.44 + oy, zt), dir=(-1., 0., 0.))]
        return tools, sup
    zt = P.TABLE["top_z"] + TRAY["floor"] + SADDLE_AXIS
    sup = [dict(kind="stick_tray", center=(0.0, TRAY["cy"]), half=TRAY["half"], top_z=P.TABLE["top_z"] + TRAY["floor"])]
    y = TRAY_GROOVES
    tools = [dict(name="stick_left", kind="stick", butt=(-0.405, y[0], zt), dir=(1., 0., 0.)),
             dict(name="stick_right", kind="stick", butt=(0.405, y[0], zt), dir=(-1., 0., 0.)),
             dict(name="mallet_left", kind="mallet", butt=(-0.30, y[1], zt), dir=(1., 0., 0.)),
             dict(name="mallet_right", kind="mallet", butt=(0.30, y[1], zt), dir=(-1., 0., 0.)),
             dict(name="beater", kind="beater", butt=(0.15, y[2], zt), dir=(-1., 0., 0.))]
    return tools, sup


TRAY = dict(cy=0.47, half=(0.42, 0.095, 0.018), floor=0.006)    # beech stick tray on the table's near edge (authored)
TRAY_GROOVES = (0.40, 0.47, 0.54)       # groove lines, 70 mm apart: a Franka Hand opened to 50 mm clears the next tool
TRAP_TABLE = dict(half=(0.22, 0.13), top_z=0.78)                  # percussionist's trap table (authored, ~45 x 25 cm)
SADDLE_AXIS = 0.024                     # tool axis height above the support (authored saddles)


def goal(cfg, rng, lay):
    t = cfg.task
    start = task_setup(t)
    if start is not None:
        return start.setup(cfg, rng, lay)
    if t == "bordun":
        key = ("C", "F")[int(rng.integers(0, 2))]
        return dict(key=key, notes=list(BORDUNS[key]), count=8, bpm=72, max_async_s=0.015, instrument="xylophone")
    if t == "roll":
        return dict(bar="C4", rate_hz=8.0, duration_s=2.0, level_sd_db=3.0)
    if t == "single-tone":
        bar = ("C4", "G4")[int(rng.integers(0, 2))]
        return dict(bar=bar, dynamic="mf")
    if t == "dynamics":
        return dict(sequence=["p", "mf", "f"], instrument="drum", min_step_db=6.0)
    raise ValueError(t)
