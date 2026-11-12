"""Start of pass-square: the ball on the penalty mark and a 2 x 2 m square of four discs ahead on the left
(centred at (-5.0, 2.5) m at seed 0, at a drawn 5.5-7.5 m and 15-30 deg from the mark otherwise).

setup(task, seed, start) receives the task name, the seed and the start every task builds on (variants.layout:
the ball at rest on the penalty mark, the robot's pelvis 0.42 m behind it, yaw 0), updates that start in place
and returns (props, markers, verifier, summary): static props, render-only markers, the verifier's parameters
and the seed's drawn values."""
import math

from football.newton15.variants import SPOT, disc, uniform


def setup(task, seed, start):
    props, markers, verifier, summary = [], [], {}, {}
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
    return props, markers, verifier, summary
