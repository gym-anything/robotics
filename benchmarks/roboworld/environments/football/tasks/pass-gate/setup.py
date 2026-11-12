"""Start of pass-gate: the ball on the penalty mark and a 2 m gate of two discs 5 m away to the robot's front
right (36.9 deg, atan(3/4), to the right at seed 0, drawn in 25-45 deg otherwise).

setup(task, seed, start) receives the task name, the seed and the start every task builds on (variants.layout:
the ball at rest on the penalty mark, the robot's pelvis 0.42 m behind it, yaw 0), updates that start in place
and returns (props, markers, verifier, summary): static props, render-only markers, the verifier's parameters
and the seed's drawn values."""
import math

from football.newton15.variants import R, SPOT, disc, uniform


def setup(task, seed, start):
    props, markers, verifier, summary = [], [], {}, {}
    angle = math.atan2(3.0, 4.0) if seed == 0 else math.radians(uniform(task, seed, "angle_deg", 25.0, 45.0))
    n = (math.cos(angle), -math.sin(angle))
    t = (-n[1], n[0])
    centre = (SPOT[0] + 5.0 * n[0], SPOT[1] + 5.0 * n[1])
    for name, sign in (("gate_left", 1.0), ("gate_right", -1.0)):
        markers.append(disc(name, centre[0] + sign * t[0], centre[1] + sign * t[1]))
    verifier = {"centre_m": centre, "normal": n, "tangent": t, "half_width_m": 1.0, "max_height_m": R + 0.25}
    summary = {"angle_right_deg": math.degrees(angle), "gate_centre_m": centre}
    return props, markers, verifier, summary
