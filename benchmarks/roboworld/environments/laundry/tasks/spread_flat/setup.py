"""Start of spread_flat: the bath towel (0.45 m x 0.70 m) lying folded in half on the table, its fold line along
its long axis (world Y) with a rounded bend, one layer over the other.

setup(index, positions_m, table_top_m) gets the garment's semantic index (garment_task.SemanticGarmentIndex), its
lay as the reset loader placed it (particles x 3, world frame, m) and the tabletop height (m), and returns the start's
positions (float64, a new array); the task-set adapters (task_set_env, task_set_v2_env) build the start with it, write
it into the simulation (author_state) and let it settle before the episode."""
import numpy as np


def setup(index, positions_m, table_top_m):
    return rounded_half_fold(positions_m, table_top_m)


def rounded_half_fold(positions, table_top):
    """A two-layer half fold with a rounded bend, its fold line along the towel's long (world Y) axis."""
    x = np.array(positions, dtype=np.float64, copy=True)
    s = x[:, 0] - (x[:, 0].min() + x[:, 0].max()) / 2
    lower, upper = s < -.05, s > .05
    theta = np.clip((s + .05) * np.pi / .10, 0., np.pi)
    x[:, 0] = .10 / np.pi * np.sin(theta)
    x[:, 2] = table_top + .012 + .10 / np.pi * (1 - np.cos(theta))
    x[lower, 0] = s[lower] + .05
    x[upper, 0] = -s[upper] + .05
    return x
