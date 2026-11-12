"""Task starts of contract `laundry-tasks-v2`; never change material coordinates, topology or solver data.

As `task_set_presets` (unchanged, `laundry-tasks-v1`), except that the ties are bunched on
their free distal parts: the first version piled both bunches back
over the anchors, under the drape overlay and on each other, and on the GPU `author_state`
found 6 triangle crossings and refused the start. The GAR task adapter still crossing-checks
and settles every authored start before exposing an episode. No props added.
"""
import numpy as np

from .garment_asset import PathRef
from .task_set import task_setup
from .task_set_presets import reach_report  # noqa: F401  (reach_report re-exported)


# Each tie's bunch lies on its own free distal part: past where the drape overlay
# covers it (to 0.20 m along the left tie, 0.12 m along the right, 20 mm plan
# margin included) and past where the ties overlap each other near their shared
# anchor (to 0.16 m). The first version piled both bunches back over the anchors,
# under the overlay and on each other: 6 triangle crossings, so author_state
# refused the start.
# (near end of the bunch along the tie, bend radius, top-layer length) per tie. Settling on the GPU moved the left tie's
# extension .436 -> .445/.450 and the right tie's .386 -> .319/.283 (once below the .30 the
# adapter requires), so the right tie's top layer is longer: both start near .44-.46.
TIE_BUNCHES = {"left_tie": (.21, .020, .010), "right_tie": (.17, .020, .065)}


def accordion_curve(s, length, near_end=.21, radius=.020, top_length=.010):
    """Two opposite rounded U-bends on the distal strip; unit speed along the centreline.

    The strip runs flat from its anchor to the first bend, back on top (layer 2) to a
    second bend whose outer face is `near_end` from the anchor, then forward again on
    top (layer 3, `top_length` long). Returns plan position along the strip and height.
    """
    back = (length - near_end - radius - top_length - 2 * np.pi * radius) / 2
    if back <= 0:
        raise ValueError("Strip too short for this bunch")
    a = near_end + back + radius
    b = a + np.pi * radius
    c = b + back
    d = c + np.pi * radius
    along, height = np.array(s, dtype=np.float64, copy=True), np.zeros_like(s, dtype=np.float64)
    one = (s > a) & (s <= b)
    theta = (s[one] - a) / radius
    along[one] = a + radius * np.sin(theta)
    height[one] = radius * (1 - np.cos(theta))
    middle = (s > b) & (s <= c)
    along[middle] = a - (s[middle] - b)
    height[middle] = 2 * radius
    two = (s > c) & (s <= d)
    theta = (s[two] - c) / radius
    along[two] = a - back - radius * np.sin(theta)
    height[two] = 2 * radius + radius * (1 - np.cos(theta))
    last = s > d
    along[last] = a - back + (s[last] - d)
    height[last] = 4 * radius
    return along, height


def bunched_ties(index, positions):
    x = np.array(positions, dtype=np.float64, copy=True)
    for name, (near_end, radius, top_length) in TIE_BUNCHES.items():
        anchor = index.path_samples(PathRef(name, "anchor"), positions, 5).mean(0)
        tip = index.path_samples(PathRef(name, "tip"), positions, 5).mean(0)
        direction = tip[:2] - anchor[:2]
        length = float(np.linalg.norm(direction))
        direction /= length
        ids = index.panel_vertices((name,))
        # Only the distal strip is bent; its shared anchor and under-overlay exit stay intact.
        s = (positions[ids, :2] - anchor[:2]) @ direction
        along, height = accordion_curve(s, length, near_end, radius, top_length)
        x[ids, :2] += (along - s)[:, None] * direction
        x[ids, 2] += height
    return x


def start_positions(entry, index, positions, table_top):
    """As `task_set_presets.start_positions`, with this module's bunched ties."""
    folder = task_setup(entry.garment, entry.name) if entry.start != "at_rest" else None
    if folder is not None:
        return folder.setup(index, positions, table_top)
    if entry.start == "bunched_ties":
        return bunched_ties(index, positions)
    return np.array(positions, dtype=np.float64, copy=True)
