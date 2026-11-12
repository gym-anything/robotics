"""Task starts only; never change material coordinates, topology or solver data.

These are authored geometry, not measured rest states. The GAR task adapter
must crossing-check and settle them before exposing an episode. No props added.
"""
import numpy as np

from .garment_asset import PathRef
from .task_set import task_setup


def accordion_curve(s, length):
    """Two opposite rounded U-bends; unit-speed along a flat strip's centreline."""
    radius = .060 / np.pi
    first, second = .20, .40
    # Scale only the placement of bends if a different length is ever presented.
    first, second = first * length / .58, second * length / .58
    half_arc = .5 * np.pi * radius
    a, b, c, d = first - half_arc, first + half_arc, second - half_arc, second + half_arc
    along, height = np.array(s, copy=True), np.zeros_like(s)
    one = (s > a) & (s <= b)
    theta = (s[one] - a) / radius
    along[one] = a + radius * np.sin(theta)
    height[one] = radius * (1 - np.cos(theta))
    middle = (s > b) & (s <= c)
    along[middle] = a - (s[middle] - b)
    height[middle] = 2 * radius
    two = (s > c) & (s <= d)
    theta = (s[two] - c) / radius
    along[two] = a - (c - b) - radius * np.sin(theta)
    height[two] = 2 * radius + radius * (1 - np.cos(theta))
    last = s > d
    along[last] = a - (c - b) + (s[last] - d)
    height[last] = 4 * radius
    return along, height


def bunched_ties(index, positions):
    x = np.array(positions, dtype=np.float64, copy=True)
    for name in ("left_tie", "right_tie"):
        anchor = index.path_samples(PathRef(name, "anchor"), positions, 5).mean(0)
        tip = index.path_samples(PathRef(name, "tip"), positions, 5).mean(0)
        direction = tip[:2] - anchor[:2]
        length = float(np.linalg.norm(direction))
        direction /= length
        ids = index.panel_vertices((name,))
        # Only the distal strip is bent; its shared anchor and under-overlay exit stay intact.
        s = (positions[ids, :2] - anchor[:2]) @ direction
        along, height = accordion_curve(s, length)
        x[ids, :2] += (along - s)[:, None] * direction
        x[ids, 2] += height
    return x


def start_positions(entry, index, positions, table_top):
    """A task's start positions from the garment's lay as loaded. A benchmark task's setup.py (tasks/<task>/) gives
    them, except an at-rest start, which the reset loader has already placed (newton15/garment.rest_start)."""
    folder = task_setup(entry.garment, entry.name) if entry.start != "at_rest" else None
    if folder is not None:
        return folder.setup(index, positions, table_top)
    if entry.start == "bunched_ties":
        return bunched_ties(index, positions)
    return np.array(positions, dtype=np.float64, copy=True)


def reach_report(positions, *, table_top=.78, robot_base_y=.78):
    """Necessary reach bound, NOT an IK/collision or grasp qualification.

    Both bases are at tabletop height. Give every point to its nearer arm;
    no claim that a single arm can reach the whole garment.
    """
    bases = np.array([[0., robot_base_y, table_top], [0., -robot_base_y, table_top]])
    distance = np.linalg.norm(np.asarray(positions)[:, None] - bases[None], axis=2)
    nearest = distance.min(axis=1)
    return dict(maximum_nearest_base_distance_m=float(nearest.max()),
                outside_085m_count=int(np.count_nonzero(nearest > .85)), point_count=len(nearest))
