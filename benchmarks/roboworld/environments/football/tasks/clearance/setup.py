"""Start of clearance: the ball loose 10 m out in the robot's own penalty area (y = 1.0 m at seed 0, drawn in
[-1, 1) m otherwise), the robot just goal-side of it facing upfield, and the penalty-area line 16.5 m out.

setup(task, seed, start) receives the task name, the seed and the start every task builds on (variants.layout:
the ball at rest on the penalty mark, the robot's pelvis 0.42 m behind it, yaw 0), updates that start in place
and returns (props, markers, verifier, summary): static props, render-only markers, the verifier's parameters
and the seed's drawn values."""
import math

from football.newton15.variants import REST_Z, box, uniform

# clearance: IFAB Law 1 penalty-area line, 16.5 m from the goal line, 0.12 m wide (the goal line's width here);
# lines belong to the area they bound, so its outer edge is at x = -16.5.
PENALTY_AREA_OUTER_X = -16.5
LINE_WIDTH = 0.12


def setup(task, seed, start):
    props, markers, verifier, summary = [], [], {}, {}
    y = 1.0 if seed == 0 else uniform(task, seed, "ball_y", -1.0, 1.0)
    start.update(ball_position=(-10.0, y, REST_Z), robot_position=(-9.58, y, 0.80), robot_yaw=math.pi)
    markers.append(box("penalty_area_line", (PENALTY_AREA_OUTER_X + 0.5 * LINE_WIDTH, 0.0, 0.0008),
                       (LINE_WIDTH, 24.0, 0.001), "line"))
    verifier = {"line_outer_x_m": PENALTY_AREA_OUTER_X}
    summary = {"ball_y_m": y}
    return props, markers, verifier, summary
