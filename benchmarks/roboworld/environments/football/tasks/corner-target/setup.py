"""Start of corner-target: the ball on the penalty mark and a rigid target sheet just behind the posts that
covers the goal except its two bottom-corner openings, with their edging painted on.

setup(task, seed, start) receives the task name, the seed and the start every task builds on (variants.layout:
the ball at rest on the penalty mark, the robot's pelvis 0.42 m behind it, yaw 0), updates that start in place
and returns (props, markers, verifier, summary): static props, render-only markers, the verifier's parameters
and the seed's drawn values."""
from football.newton15.variants import box

# corner-target: a rigid sheet just behind the posts, in front of the net; two bottom-corner openings.
SHEET_X = (0.065, 0.085)
SHEET_HALF_WIDTH = 3.66
SHEET_TOP = 2.44
OPENING_INNER_Y = 2.40
OPENING_TOP = 1.00


def setup(task, seed, start):
    props, markers, verifier, summary = [], [], {}, {}
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
    return props, markers, verifier, summary
