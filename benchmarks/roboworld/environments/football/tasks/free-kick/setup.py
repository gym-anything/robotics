"""Start of free-kick: the ball 24 m from the goal line at y = -4 m with the robot set up for a curling kick,
and a wall of four mannequins 9.15 m in front of the ball.

setup(task, seed, start) receives the task name, the seed and the start every task builds on (variants.layout:
the ball at rest on the penalty mark, the robot's pelvis 0.42 m behind it, yaw 0), updates that start in place
and returns (props, markers, verifier, summary): static props, render-only markers, the verifier's parameters
and the seed's drawn values."""
import math

from football.football_specs import FieldSpec
from football.newton15.variants import REST_Z

# free-kick: the visible-curl start and a four-mannequin wall.
FREE_KICK_BALL = (-24.0, -4.0, REST_Z)
FREE_KICK_AIM, FREE_KICK_SETBACK, FREE_KICK_LATERAL, FREE_KICK_PELVIS = -0.6, 0.22, -0.095, 0.815
WALL_DISTANCE = 9.15                     # IFAB Law 13: the wall's front face 9.15 m from the ball
MANNEQUIN = (0.20, 0.50, 1.80)           # depth, width, height
WALL_OUTSIDE_NEAR_POST = 0.40            # outer end beyond the ball-to-near-post line


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


def setup(task, seed, start):
    props, markers, verifier, summary = [], [], {}, {}
    start.update(ball_position=FREE_KICK_BALL, robot_position=free_kick_robot(), robot_yaw=FREE_KICK_AIM)
    props = free_kick_wall()
    return props, markers, verifier, summary
