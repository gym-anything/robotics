"""Start of trap: a pass rolling along the ground toward the robot's feet from 6.4 m in front (4.2 m/s at seed 0,
drawn in 4.0-4.6 m/s with a drawn lateral aim otherwise).

setup(task, seed, start) receives the task name, the seed and the start every task builds on (variants.layout:
the ball at rest on the penalty mark, the robot's pelvis 0.42 m behind it, yaw 0), updates that start in place
and returns (props, markers, verifier, summary): static props, render-only markers, the verifier's parameters
and the seed's drawn values."""
from football.newton15.variants import R, REST_Z, uniform

# trap: a pass rolling in from 6.4 m. The spec's 3.0 m/s cannot arrive: rolling on this pitch decelerates by at least
# c g / (1 + k) = 0.13 x 9.807 / (5/3) = 0.77 m/s^2 (football/pitch_model.py rolling_resistance_coefficient 0.130,
# thin-shell k = 2/3) plus drag, so a 3.0 m/s ball stops after <= 5.4 m, short of the feet (a 2 m/s slide stops
# in 1.01 m). At 4.2 m/s it reaches the feet at 1.5-2.4 m/s after 1.9-2.2 s and, untouched,
# stops 1-3.5 m behind them.
TRAP_SPEED = 4.2


def setup(task, seed, start):
    props, markers, verifier, summary = [], [], {}, {}
    speed = TRAP_SPEED if seed == 0 else uniform(task, seed, "speed_m_s", 4.0, 4.6)
    y = -0.12 if seed == 0 else uniform(task, seed, "aim_y_m", -0.20, -0.05)
    start.update(ball_position=(-5.0, y, REST_Z), ball_velocity=(-speed, 0.0, 0.0),
                 ball_omega=(0.0, -speed / R, 0.0))
    verifier = {"feet_radius_m": 1.0, "rest_speed_m_s": 0.05, "hold_s": 1.0, "speed_scale_m_s": 1.0,
                "pelvis_min_z_m": 0.55, "pelvis_zero_z_m": 0.35, "max_tilt_deg": 45.0}
    summary = {"speed_m_s": speed, "aim_y_m": y}
    return props, markers, verifier, summary
