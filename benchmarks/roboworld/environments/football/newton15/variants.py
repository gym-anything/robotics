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

Seed 0 is each task's nominal configuration. clearance, pass-gate, trap, pass-square and save also accept seeds
1..999, which draw the spec's proposed ranges from SHA-256 of ("football/<task>/<seed>/<quantity>"); those seeds are
proposals, not qualified configurations. goal, corner-target and free-kick have seed 0 only. Task ids:
football/<task> for seed 0 (so the default is football/goal) and football/<task>/seed-<n> otherwise.

  python -m football.newton15.variants        # print each task's seed-0 layout
"""

import hashlib
import json
import math

from football.football_specs import BallSpec, FieldSpec

R = BallSpec().radius_m
REST_Z = R + 0.001                       # the ball's centre resting on the pitch (Config's default height)
SPOT = (-11.0, 0.0, REST_Z)              # penalty mark
GOAL_ROBOT = (-11.42, 0.0, 0.80)         # the goal start: pelvis 0.42 m behind the ball, yaw 0
DEFAULT_TASK = "goal"
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

# clearance: IFAB Law 1 penalty-area line, 16.5 m from the goal line, 0.12 m wide (the goal line's width here);
# lines belong to the area they bound, so its outer edge is at x = -16.5.
PENALTY_AREA_OUTER_X = -16.5
LINE_WIDTH = 0.12
# trap: a pass rolling in from 6.4 m. The spec's 3.0 m/s cannot arrive: rolling on this pitch decelerates by at least
# c g / (1 + k) = 0.13 x 9.807 / (5/3) = 0.77 m/s^2 (football/pitch_model.py rolling_resistance_coefficient 0.130,
# thin-shell k = 2/3) plus drag, so a 3.0 m/s ball stops after <= 5.4 m, short of the feet (a 2 m/s slide stops
# in 1.01 m). At 4.2 m/s it reaches the feet at 1.5-2.4 m/s after 1.9-2.2 s and, untouched,
# stops 1-3.5 m behind them.
TRAP_SPEED = 4.2
# save: an airborne shot from the penalty mark with 25 rad/s of machine backspin, crossing the frame plane at
# (y, z) = (0.75, 0.30) m 1.1 s after launch. Fitted (Gauss-Newton) with the package's dependency-free flight model
# (football/flight_lab.midpoint_step at 500 Hz, the same aerodynamics FootballWorld applies); other aims use the
# fit's Jacobian, which hit the corners of the seed range within 2 mm. The robot-free fixture confirms the goal.
SAVE_AIM = (0.75, 0.30)
SAVE_CROSSING_S = 1.1
SAVE_BACKSPIN = 25.0
SAVE_VELOCITY = (11.790941, 0.803927, 4.844771)
SAVE_DV_DY = (0.006333, 1.072335, -0.001898)
SAVE_DV_DZ = (0.104053, 0.007094, 1.092878)
# corner-target: a rigid sheet just behind the posts, in front of the net; two bottom-corner openings.
SHEET_X = (0.065, 0.085)
SHEET_HALF_WIDTH = 3.66
SHEET_TOP = 2.44
OPENING_INNER_Y = 2.40
OPENING_TOP = 1.00
# free-kick: the visible-curl start and a four-mannequin wall.
FREE_KICK_BALL = (-24.0, -4.0, REST_Z)
FREE_KICK_AIM, FREE_KICK_SETBACK, FREE_KICK_LATERAL, FREE_KICK_PELVIS = -0.6, 0.22, -0.095, 0.815
WALL_DISTANCE = 9.15                     # IFAB Law 13: the wall's front face 9.15 m from the ball
MANNEQUIN = (0.20, 0.50, 1.80)           # depth, width, height
WALL_OUTSIDE_NEAR_POST = 0.40            # outer end beyond the ball-to-near-post line
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


def save_launch(aim_y, aim_z):
    """Initial ball velocity and backspin of the save shot aimed to cross the frame plane at (aim_y, aim_z)."""
    dy, dz = aim_y - SAVE_AIM[0], aim_z - SAVE_AIM[1]
    v = tuple(v0 + a * dy + b * dz for v0, a, b in zip(SAVE_VELOCITY, SAVE_DV_DY, SAVE_DV_DZ))
    h = math.hypot(v[0], v[1])
    return v, (SAVE_BACKSPIN * v[1] / h, -SAVE_BACKSPIN * v[0] / h, 0.0)


def disc(name, x, y, material="marker"):
    return {"name": name, "shape": "disc", "position_m": (x, y, 0.5 * MARKER_DISC[1]),
            "radius_m": MARKER_DISC[0], "height_m": MARKER_DISC[1], "material": material}


def box(name, centre, size, material):
    return {"name": name, "shape": "box", "position_m": tuple(centre), "size_m": tuple(size), "material": material}


def free_kick_robot():
    """Pelvis position of the visible-curl profile (curl_experiment.Kick.config with its recorded settings)."""
    c, s = math.cos(FREE_KICK_AIM), math.sin(FREE_KICK_AIM)
    dx, dy = c * FREE_KICK_SETBACK - s * FREE_KICK_LATERAL, s * FREE_KICK_SETBACK + c * FREE_KICK_LATERAL
    return (FREE_KICK_BALL[0] - dx, FREE_KICK_BALL[1] - dy, FREE_KICK_PELVIS)


def free_kick_wall():
    """Four mannequins square to the ball-goal-centre line, front face 9.15 m from the ball, outer end 0.40 m
    outside the ball-to-near-post line, so a straight ground shot at the near half of the goal meets the wall."""
    bx, by = FREE_KICK_BALL[:2]
    length = math.hypot(bx, by)
    ux, uy = -bx / length, -by / length
    wx, wy = -uy, ux
    px, py = -bx, math.copysign(FieldSpec().half_goal_width_m + 0.06, by) - by
    near_post_lateral = (px * wx + py * wy) * WALL_DISTANCE / (px * ux + py * uy)
    outward = math.copysign(1.0, near_post_lateral)
    outer = near_post_lateral + outward * WALL_OUTSIDE_NEAR_POST
    depth, width, height = MANNEQUIN
    along = WALL_DISTANCE + 0.5 * depth
    props = []
    for i in range(4):
        lateral = outer - outward * width * (i + 0.5)
        props.append({"name": f"mannequin_{i}", "half_extents_m": (0.5 * depth, 0.5 * width, 0.5 * height),
                      "position_m": (bx + along * ux + lateral * wx, by + along * uy + lateral * wy, 0.5 * height),
                      "yaw_rad": math.atan2(uy, ux), "material": "mannequin"})
    return props


def layout(task, seed=0):
    """Start state, props, render-only markers and verifier parameters of one task and seed (all plain data)."""
    validate(task, seed)
    start = {"ball_position": SPOT, "ball_velocity": (0.0, 0.0, 0.0), "ball_omega": (0.0, 0.0, 0.0),
             "robot_position": GOAL_ROBOT, "robot_yaw": 0.0}
    props, markers, verifier, summary = [], [], {}, {}
    if task == "clearance":
        y = 1.0 if seed == 0 else uniform(task, seed, "ball_y", -1.0, 1.0)
        start.update(ball_position=(-10.0, y, REST_Z), robot_position=(-9.58, y, 0.80), robot_yaw=math.pi)
        markers.append(box("penalty_area_line", (PENALTY_AREA_OUTER_X + 0.5 * LINE_WIDTH, 0.0, 0.0008),
                           (LINE_WIDTH, 24.0, 0.001), "line"))
        verifier = {"line_outer_x_m": PENALTY_AREA_OUTER_X}
        summary = {"ball_y_m": y}
    elif task == "pass-gate":
        angle = math.atan2(3.0, 4.0) if seed == 0 else math.radians(uniform(task, seed, "angle_deg", 25.0, 45.0))
        n = (math.cos(angle), -math.sin(angle))
        t = (-n[1], n[0])
        centre = (SPOT[0] + 5.0 * n[0], SPOT[1] + 5.0 * n[1])
        for name, sign in (("gate_left", 1.0), ("gate_right", -1.0)):
            markers.append(disc(name, centre[0] + sign * t[0], centre[1] + sign * t[1]))
        verifier = {"centre_m": centre, "normal": n, "tangent": t, "half_width_m": 1.0, "max_height_m": R + 0.25}
        summary = {"angle_right_deg": math.degrees(angle), "gate_centre_m": centre}
    elif task == "trap":
        speed = TRAP_SPEED if seed == 0 else uniform(task, seed, "speed_m_s", 4.0, 4.6)
        y = -0.12 if seed == 0 else uniform(task, seed, "aim_y_m", -0.20, -0.05)
        start.update(ball_position=(-5.0, y, REST_Z), ball_velocity=(-speed, 0.0, 0.0),
                     ball_omega=(0.0, -speed / R, 0.0))
        verifier = {"feet_radius_m": 1.0, "rest_speed_m_s": 0.05, "hold_s": 1.0, "speed_scale_m_s": 1.0,
                    "pelvis_min_z_m": 0.55, "pelvis_zero_z_m": 0.35, "max_tilt_deg": 45.0}
        summary = {"speed_m_s": speed, "aim_y_m": y}
    elif task == "pass-square":
        if seed == 0:
            centre = (-5.0, 2.5)
        else:
            distance = uniform(task, seed, "distance_m", 5.5, 7.5)
            angle = math.radians(uniform(task, seed, "angle_deg", 15.0, 30.0))
            centre = (SPOT[0] + distance * math.cos(angle), SPOT[1] + distance * math.sin(angle))
        for i, (dx, dy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            markers.append(disc(f"square_{i}", centre[0] + dx, centre[1] + dy))
        verifier = {"centre_m": centre, "half_side_m": 1.0, "rest_speed_m_s": 0.05, "hold_s": 1.0,
                    "speed_scale_m_s": 1.0}
        summary = {"square_centre_m": centre}
    elif task == "corner-target":
        thickness, x = SHEET_X[1] - SHEET_X[0], 0.5 * (SHEET_X[0] + SHEET_X[1])
        props = [
            {"name": "target_sheet_top", "half_extents_m": (0.5 * thickness, SHEET_HALF_WIDTH,
                                                            0.5 * (SHEET_TOP - OPENING_TOP)),
             "position_m": (x, 0.0, 0.5 * (SHEET_TOP + OPENING_TOP)), "yaw_rad": 0.0, "material": "sheet"},
            {"name": "target_sheet_middle", "half_extents_m": (0.5 * thickness, OPENING_INNER_Y, 0.5 * OPENING_TOP),
             "position_m": (x, 0.0, 0.5 * OPENING_TOP), "yaw_rad": 0.0, "material": "sheet"},
        ]
        # contrasting 5 cm edging painted on the sheet's front face along each opening's inner side and top
        edge, front = 0.05, SHEET_X[0] - 0.002
        for side, label in ((-1.0, "right"), (1.0, "left")):
            markers.append(box(f"opening_edge_{label}_side", (front, side * (OPENING_INNER_Y - 0.5 * edge),
                                                              0.5 * (OPENING_TOP + edge)),
                               (0.003, edge, OPENING_TOP + edge), "edge"))
            markers.append(box(f"opening_edge_{label}_top",
                               (front, side * 0.5 * (OPENING_INNER_Y + SHEET_HALF_WIDTH), OPENING_TOP + 0.5 * edge),
                               (0.003, SHEET_HALF_WIDTH - OPENING_INNER_Y, edge), "edge"))
        verifier = {"sheet_x_m": x, "opening_inner_y_m": OPENING_INNER_Y, "opening_outer_y_m": SHEET_HALF_WIDTH,
                    "opening_top_m": OPENING_TOP, "contact_allowance_m": 0.01}
    elif task == "save":
        aim = SAVE_AIM if seed == 0 else (uniform(task, seed, "aim_y_m", 0.6, 0.9),
                                          uniform(task, seed, "aim_z_m", 0.2, 0.8))
        velocity, omega = save_launch(*aim)
        start.update(ball_velocity=velocity, ball_omega=omega, robot_position=(-0.45, 0.0, 0.80), robot_yaw=math.pi)
        markers += [box("ball_machine_body", (-11.62, 0.0, 0.62), (0.60, 0.52, 0.42), "machine"),
                    box("ball_machine_stand", (-11.62, 0.0, 0.205), (0.12, 0.12, 0.41), "machine"),
                    box("ball_machine_base", (-11.62, 0.0, 0.01), (0.60, 0.50, 0.02), "machine"),
                    box("ball_machine_chute", (-11.24, 0.0, 0.30), (0.20, 0.24, 0.03), "machine")]
        verifier = {"rest_speed_m_s": 0.05, "hold_s": 1.0, "area_line_x_m": -5.5, "deadline_s": 5.0,
                    "aim_m": aim, "crossing_time_s": SAVE_CROSSING_S}
        summary = {"aim_m": aim, "launch_speed_m_s": math.hypot(*velocity)}
    elif task == "free-kick":
        start.update(ball_position=FREE_KICK_BALL, robot_position=free_kick_robot(), robot_yaw=FREE_KICK_AIM)
        props = free_kick_wall()
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
