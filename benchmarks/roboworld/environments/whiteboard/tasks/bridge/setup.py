"""Start of bridge: the native classroom of the selected variation (variations.specification: robot, tray order, wiper,
seed, the diagram's place and width), the board clean and the markers in the tray; the task hands over the
colour-coded Golden Gate Bridge diagram (bridge_strokes) as its reference picture, the whole writing surface as seen
from the front.

setup(between, width) gives a bridge variation's diagram centre height (center_z, metres) when the variation catalogue
builds its spec: between(field, lo, hi) is that variation's hash draw, width its diagram width (width_m).
reference_view(task, ink) gives the reference picture (whiteboard.tasks.Task.reference_view, which the backend's task
images and the server's task route hand over): an RGB array of ink's texture size, the diagram drawn where task places
it (center_x, center_z) at its width_m, with the texture's columns reversed (world +x is the viewer's left). diagram()
draws the diagram on Task.reference's picture; HALF_HEIGHT is the diagram's half height in widths, which
Task.validate_board checks against the writable board.
"""
from functools import lru_cache
import math

import numpy as np

# Bridge placement: a kinematic check of the pinned G1 puts the highest nib point a Dex3 hand can hold on the
# board at 1.50-1.55 m. Bridge drawings keep their top at or below this height and their bottom on the writing
# surface; the same hash draw is mapped into that range.
BRIDGE_TOP_Z_M = 1.48
HALF_HEIGHT = 0.30


def setup(between, width):
    return between("task_z", 0.955 + 0.30 * width, BRIDGE_TOP_Z_M - 0.27 * width)


@lru_cache(maxsize=1)
def bridge_strokes():
    """The bridge diagram as (element, colour, line weight, points), in drawing order.

    Points are multiples of the task's width_m from its centre, along world +x
    and +z. A person facing the board sees world +x on their left, so elements
    are named as seen from the front. Coordinates, weights and order are the
    ones the reference has always been drawn with; tests pin the raster.
    """
    def cable_y(x):
        return float(0.09 + 0.18 * (x / 0.28) ** 2)

    strokes = []
    for side, x in (("right", -0.28), ("left", 0.28)):
        strokes.append((f"{side} tower legs", "orange", "thick", ((x, -0.06), (x, 0.27))))
        strokes.append((f"{side} tower legs", "orange", "thick", ((x - 0.035, -0.06), (x - 0.035, 0.25))))
        strokes += [(f"{side} tower cross-bars", "orange", "normal", ((x - 0.035, y), (x, y))) for y in (0.02, 0.10, 0.18)]
    strokes.append(("roadway", "black", "thick", ((-0.48, -0.02), (0.48, -0.02))))
    strokes.append((
        "main cable", "red", "normal",
        tuple((float(x), cable_y(x)) for x in np.linspace(-0.28, 0.28, 150)),
    ))
    strokes.append(("right side-span cable", "red", "normal", ((-0.48, 0.0), (-0.28, 0.27))))
    strokes.append(("left side-span cable", "red", "normal", ((0.28, 0.27), (0.48, 0.0))))
    strokes += [
        ("hangers", "red", "thin", ((float(x), -0.02), (float(x), cable_y(x))))
        for x in np.linspace(-0.25, 0.25, 17)
    ]
    for name, y in (("upper water line", -0.11), ("middle water line", -0.15), ("lower water line", -0.19)):
        strokes.append((
            name, "blue", "normal",
            tuple((float(x), float(y + 0.009 * math.sin(x * 45))) for x in np.linspace(-0.48, 0.48, 150)),
        ))
    return tuple(strokes)


def diagram(draw, point, palette, line):
    """Draw the diagram (Task.reference): point(x, y) maps diagram units to texture pixels, palette colour names to
    RGB, line is the normal line width in pixels."""
    widths = {"thick": line * 2, "normal": line, "thin": max(1, line - 1)}
    for _, color, weight, points in bridge_strokes():
        draw.line(
            [point(x, y) for x, y in points],
            fill=palette[color],
            width=widths[weight],
            joint="curve",
        )


def reference_view(task, ink):
    image = task.reference(ink)
    return np.ascontiguousarray(image[:, ::-1])
