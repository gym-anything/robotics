"""Task selection: preset / task / seed -> the world configuration (resolving a task never starts physics)."""

from .task import INSTRUCTIONS, PRESET_OF, WORKPIECE_OF, WORKPIECE_OF_TASK

# designed but not supported by this build: the physics cannot yet represent the job faithfully
UNSUPPORTED = {
    "B1": "pin-boning needs the fish-bone tweezers as a physical tool (jaws on a spring hinge gripped by the Franka Hand "
          "and pinching 0.3 mm pin bones); in this build the tweezers are a static obstacle in their tray, not a tool a "
          "hand can pick up",
    "B4": "skinning needs the skin to separate with under 10 % of its mass in flesh; the tissue mesh "
          "separates along 3.5 mm tet faces, so any removed skin carries at least one tet layer of flesh",
}
# the one-arm deployment (single-fr3: the right FR3 with the knife inserts) is offered for the tasks one knife hand can
# do: taking the knife (A0). Every cutting task needs a second hand on the workpiece: measured on the salmon side (B5),
# an unheld 944 g side on the board (skin on HDPE, mu 0.15) was dragged 75 mm toward the robots by six
# rock-and-draw cuts and no portion came free; the unheld rib stroke (A4's control) slid the fillet 35 mm. A task is
# added here only after a run without a holding hand shows the workpiece staying put.
SINGLE_ARM_TASKS = ("A0",)
# the two-Panda deployment is qualified for holding the knife and resting only (rest probe, robot_a0_panda): its
# take-knife run fails (the wrist ends the lift 3-5 deg short and never turns the knife edge-down over the board, score
# 0.78), and no cutting run or timing exists on it. It is offered for take-knife alone until one does.
PANDA_TASKS = ("A0",)
# cutting tasks run with libfranka's contact-task collision thresholds (world.Config.collision_profile 'contact'):
# every robot cutting demonstration and every runtime measurement used them; the default 10/20 N m thresholds stop an
# arm pressing a knife into meat
CONTACT_TASKS = ("A1", "A2", "A3", "A4", "B2", "B3", "B5", "C1", "C2", "C3")


def check_robot(robot, task):
    """Raise ValueError if the robot deployment cannot do the task in this build."""
    if robot == "single-fr3" and task not in SINGLE_ARM_TASKS:
        raise ValueError(f"single-fr3 is offered for {', '.join(TASK_SLUG[t] for t in SINGLE_ARM_TASKS)} only: "
                         f"{TASK_SLUG.get(task, task)} ({task}) needs a second hand holding the workpiece beside the "
                         f"knife hand")
    if robot == "dual-panda" and task not in PANDA_TASKS:
        raise ValueError(f"dual-panda is offered for {', '.join(TASK_SLUG[t] for t in PANDA_TASKS)} only: it is "
                         f"qualified for holding the knife and resting; no cutting run or timing exists on it")


TASKS = {p: [t for t in INSTRUCTIONS if PRESET_OF[t] == p and t not in UNSUPPORTED] for p in ("A", "B", "C")}
DEFAULT = dict(preset="A", task="A2", seed=0)
TASK_SLUG = {"A0": "take-knife", "A1": "head-off", "A2": "fillet-one-side", "A3": "fillet-whole-fish",
             "A4": "remove-belly-bones", "B1": "pin-bone", "B2": "trim-belly", "B3": "excise-defect", "B4": "skin",
             "B5": "portion", "C1": "remove-chain", "C2": "denude", "C3": "medallions"}
SLUG_TASK = {v: k for k, v in TASK_SLUG.items()}
PRESET_SLUG = {"A": "trout", "B": "salmon-side", "C": "pork-tenderloin"}
SLUG_PRESET = {v: k for k, v in PRESET_SLUG.items()}


def resolve(preset=None, task=None, seed=0):
    if preset is not None and SLUG_PRESET.get(preset, preset) not in PRESET_SLUG:
        raise ValueError(f"Unknown preset {preset!r}; presets: {', '.join(PRESET_SLUG.values())}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    preset = SLUG_PRESET.get(preset, preset) if preset else None
    task = SLUG_TASK.get(task, task) if task else None
    if task is None:
        preset = preset or DEFAULT["preset"]
        task = TASKS[preset][2 if preset == "A" else 0]
    if task in UNSUPPORTED:
        raise ValueError(f"Task {TASK_SLUG[task]} ({task}) is not supported by this build: {UNSUPPORTED[task]}")
    if task not in INSTRUCTIONS:
        raise ValueError(f"Unknown task {task!r}; tasks: "
                         f"{', '.join(TASK_SLUG[t] for t in INSTRUCTIONS if t not in UNSUPPORTED)}")
    if preset is not None and PRESET_OF[task] != preset:
        raise ValueError(f"Task {task} belongs to preset {PRESET_OF[task]}, not {preset}")
    if not 0 <= seed < 1000:
        raise ValueError("seed must be in [0, 1000)")
    return dict(preset=PRESET_OF[task], task=task, seed=seed,
                workpiece=WORKPIECE_OF_TASK.get(task) or WORKPIECE_OF[PRESET_OF[task]])


def task_resolve(selection):
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0))
    return dict(task_id=f"filleting/v1/{PRESET_SLUG[v['preset']]}/{TASK_SLUG[v['task']]}/seed-{v['seed']}",
                description=INSTRUCTIONS[v["task"]],
                backend_options=dict(preset=v["preset"], task=v["task"], seed=v["seed"]),
                selection=v, split=None)
