"""Start of save: the robot on its goal line facing upfield and a ball machine on the penalty mark whose shot
is launched at the start (aimed at (0.75, 0.30) m in the frame plane at seed 0, at a drawn aim otherwise).

setup(task, seed, start) receives the task name, the seed and the start every task builds on (variants.layout:
the ball at rest on the penalty mark, the robot's pelvis 0.42 m behind it, yaw 0), updates that start in place
and returns (props, markers, verifier, summary): static props, render-only markers, the verifier's parameters
and the seed's drawn values."""
import math

from football.newton15.variants import box, uniform

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


def save_launch(aim_y, aim_z):
    """Initial ball velocity and backspin of the save shot aimed to cross the frame plane at (aim_y, aim_z)."""
    dy, dz = aim_y - SAVE_AIM[0], aim_z - SAVE_AIM[1]
    v = tuple(v0 + a * dy + b * dz for v0, a, b in zip(SAVE_VELOCITY, SAVE_DV_DY, SAVE_DV_DZ))
    h = math.hypot(v[0], v[1])
    return v, (SAVE_BACKSPIN * v[1] / h, -SAVE_BACKSPIN * v[0] / h, 0.0)


def setup(task, seed, start):
    props, markers, verifier, summary = [], [], {}, {}
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
    return props, markers, verifier, summary
