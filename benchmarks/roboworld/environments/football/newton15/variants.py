"""Football task catalogue: eight tasks on one training sector, a pure function of (task, seed); no physics.

`goal` is the default task (verifier football/newton15/task.py, contract football-goal-progress-v1): an
empty selection resolves to it with no backend options. The other seven are drills verified by
football/newton15/drills.py. A task is

- a start state: the Config fields ball_position, ball_velocity, ball_omega, robot_position and robot_yaw;
- static props: rigid boxes on the world body (body -1), added by FootballWorld after every existing shape, so the
  ball meets them through the same native contact as the goal frame. No existing object, material, contact law,
  solver setting or robot property changes;
- render-only markers (flat discs, lines, a ball machine) with no collider, drawn by render.py;
- verifier parameters (drills.py).

Each task's own part (its start, props, markers, verifier parameters and seeded draws) is the setup(task, seed, start)
of its folder, tasks/<task>/setup.py, which layout calls on the start every task builds on; its verifier is the
verify(config, layout) of tasks/<task>/verifier.py (drills.make_verifier). This module keeps the catalogue: tasks,
seeds, ids, difficulty, instructions, resolve and contract.

Seed 0 is each task's nominal configuration. clearance, pass-gate, trap, pass-square and save also accept seeds
1..999, which draw the spec's proposed ranges from SHA-256 of ("football/<task>/<seed>/<quantity>"); those seeds are
proposals, not qualified configurations. goal, corner-target and free-kick have seed 0 only. Task ids:
football/<task> for seed 0 (so the default is football/goal) and football/<task>/seed-<n> otherwise.

  python -m football.newton15.variants        # print each task's seed-0 layout
"""

import hashlib
import json

from football.football_specs import BallSpec
from gym_anything_robotics import task_folders

R = BallSpec().radius_m
REST_Z = R + 0.001                       # the ball's centre resting on the pitch (Config's default height)
SPOT = (-11.0, 0.0, REST_Z)              # penalty mark
GOAL_ROBOT = (-11.42, 0.0, 0.80)         # the goal start: pelvis 0.42 m behind the ball, yaw 0
DEFAULT_TASK = "goal"
PACKAGE = "football"
TASKS = ("goal", "clearance", "pass-gate", "trap", "pass-square", "corner-target", "save", "free-kick")
SEEDED = ("clearance", "pass-gate", "trap", "pass-square", "save")
MAX_SEED = 999
DIFFICULTY = {
    "clearance": "entry", "pass-gate": "entry", "goal": "medium", "trap": "medium", "pass-square": "medium",
    "corner-target": "hard", "save": "hard", "free-kick": "hardest",
}
INSTRUCTIONS = {
    "goal": "The football is on the penalty mark in front of the Unitree G1 humanoid. Kick it into the goal.",
    "clearance": "The ball is loose in your own penalty area and you are facing upfield. Clear it: kick it out of "
                 "the penalty area.",
    "pass-gate": "Pass the ball along the ground between the two orange markers to your front right.",
    "trap": "A teammate has passed the ball to you along the ground. Receive it and bring it under control at "
            "your feet.",
    "pass-square": "Pass the ball into the square marked by the four discs ahead of you on your left, so that it "
                   "stops inside it.",
    "corner-target": "A target sheet covers the goal except its two bottom corners. Kick the ball into the goal "
                     "through one of the corner openings.",
    "save": "You are the goalkeeper, standing on your goal line. The ball machine on the penalty mark is about to "
            "shoot. Stop the shot from going in.",
    "free-kick": "Direct free kick from 24 metres, with a four-man training wall 9.15 metres in front of the ball. "
                 "Score directly from the free kick.",
}

MARKER_DISC = (0.095, 0.004)             # flat training disc: radius, height


def validate(task, seed):
    if task not in TASKS:
        raise ValueError(f"Unknown football task {task!r}; choose one of {', '.join(TASKS)}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Football seed must be an integer")
    if task in SEEDED:
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"{task} seeds are 0..{MAX_SEED}")
    elif seed != 0:
        raise ValueError(f"{task} has only seed 0")


def task_id(task, seed=0):
    validate(task, seed)
    return f"football/{task}" if seed == 0 else f"football/{task}/seed-{seed}"


def uniform(task, seed, quantity, low, high):
    """Deterministic draw in [low, high) for one seeded quantity."""
    digest = hashlib.sha256(f"football/{task}/{seed}/{quantity}".encode()).digest()
    return low + (high - low) * int.from_bytes(digest[:8], "big") / 2.0**64


def task_setup(task):
    """The task's setup.py (tasks/<task>/)."""
    return task_folders.setup(PACKAGE, task)


def task_verifier(task):
    """The task's verifier.py (tasks/<task>/)."""
    return task_folders.verifier(PACKAGE, task)


def disc(name, x, y, material="marker"):
    return {"name": name, "shape": "disc", "position_m": (x, y, 0.5 * MARKER_DISC[1]),
            "radius_m": MARKER_DISC[0], "height_m": MARKER_DISC[1], "material": material}


def box(name, centre, size, material):
    return {"name": name, "shape": "box", "position_m": tuple(centre), "size_m": tuple(size), "material": material}


def layout(task, seed=0):
    """Start state, props, render-only markers and verifier parameters of one task and seed (all plain data); the
    task's own part is its folder's setup(task, seed, start) (tasks/<task>/setup.py)."""
    validate(task, seed)
    start = {"ball_position": SPOT, "ball_velocity": (0.0, 0.0, 0.0), "ball_omega": (0.0, 0.0, 0.0),
             "robot_position": GOAL_ROBOT, "robot_yaw": 0.0}
    props, markers, verifier, summary = task_setup(task).setup(task, seed, start)
    return {"task": task, "seed": seed, "task_id": task_id(task, seed), "difficulty": DIFFICULTY[task],
            "instruction": INSTRUCTIONS[task], "config": start, "props": props, "markers": markers,
            "verifier": verifier, "summary": summary}


def apply(config, task, seed=0):
    """`config` with the task's start state, task and seed (every other field, e.g. dt or net_layout, kept)."""
    from dataclasses import replace

    return replace(config, task=task, seed=seed, **layout(task, seed)["config"])


def task_resolve(selection):
    task, seed = selection.get("task", DEFAULT_TASK), selection.get("seed", 0)
    validate(task, seed)
    # goal, seed 0: no backend options, the launch the single-task catalogue had (its default backend task)
    options = {} if (task, seed) == (DEFAULT_TASK, 0) else {"task": task, "seed": seed}
    return dict(task_id=task_id(task, seed), description=INSTRUCTIONS[task], backend_options=options,
                selection={"task": task, "seed": seed}, split=None)


def contract():
    return {"supported": True, "selectors": ["task", "seed"], "tasks": list(TASKS), "default_task": DEFAULT_TASK,
            "seeded_tasks": list(SEEDED), "seeds": f"0..{MAX_SEED} for seeded tasks, 0 otherwise",
            "difficulty": dict(DIFFICULTY)}


if __name__ == "__main__":
    print(json.dumps({task: layout(task) for task in TASKS}, indent=1, default=float))
