"""Start of sleeve_fold: the button-down shirt lying face up on the table, its long sleeves out to the sides, at rest.
Its P13 catalogue lay recoils within 0.07 s, so the task starts from that same shirt after it has come to rest
untouched on the table: world-frame positions recorded for this topology and tabletop height (REST_START, in
data/frames).

setup(config, topology) gets the workcell configuration (newton15.config.Config) and the shirt's welded topology and
returns those positions (float64, particles x 3, m). The reset loader (newton15/garment.rest_start) loads the shirt in
them, so the episode starts from the shirt as loaded; a topology or tabletop height they were not recorded for is
refused."""
from pathlib import Path

import numpy as np

REST_START = "shirt_sleeve_fold_rest_start.npz"


def setup(config, topology):
    path = Path(__file__).resolve().parents[2]/"data/frames"/REST_START
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved["physical_triangles"], topology.physical_triangles):
            raise ValueError("At-rest start topology no longer matches constructed garment")
        if float(saved["table_top_m"]) != config.table_top:
            raise ValueError("At-rest start was recorded for a different tabletop height")
        return saved["positions_m"].astype(np.float64)
