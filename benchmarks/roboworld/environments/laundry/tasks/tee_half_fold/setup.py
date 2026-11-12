"""Start of tee_half_fold: the T-shirt lying face up and laid out flat on the table, sleeves out to the sides, as the
garment catalogue lays it (the P13 flat lay), untouched.

setup(index, positions_m, table_top_m) gets the garment's semantic index (garment_task.SemanticGarmentIndex), its
lay as the reset loader placed it (particles x 3, world frame, m) and the tabletop height (m), and returns the start's
positions (float64, a new array); the laundry-tasks-v2 adapter (task_set_v2_env) builds the start with it: here that
lay unchanged; nothing is authored. The shirt settles before the episode (laundry-tasks-v2)."""
import numpy as np


def setup(index, positions_m, table_top_m):
    return np.array(positions_m, dtype=np.float64, copy=True)
