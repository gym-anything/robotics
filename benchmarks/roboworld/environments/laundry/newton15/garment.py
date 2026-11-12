"""Reconstruct the same sewn mesh; load only the accepted reset configuration."""
from pathlib import Path
import numpy as np
from ..flat_lay_seed import build_flat_lay_seed
from ..garment_patterns import get_garment_construction
from ..garment_meshing import compile_garment_pattern_mesh
from ..garment_assembly import weld_garment_topology
from .config import GARMENTS
from gym_anything_robotics import task_folders


def load(config):
    construction = get_garment_construction(GARMENTS[config.garment])
    pattern = compile_garment_pattern_mesh(construction, target_edge_length_m=config.edge_length)
    seed = build_flat_lay_seed(construction, pattern)
    raw = weld_garment_topology(construction, pattern, seed=seed.seed, relax=False)
    path = Path(__file__).resolve().parents[1]/"data/frames/p13_garment_flat_lay_catalog.npz"
    with np.load(path, allow_pickle=False) as saved:
        prefix = construction.id
        if not np.array_equal(saved[prefix+"_physical_triangles"], raw.physical_triangles):
            raise ValueError("P13 reset topology no longer matches constructed garment")
        positions = saved[prefix+"_positions_m"].copy()
    # Position initialization only. Material coordinates, masses, stitching and
    # manufactured rest geometry are unchanged. No saved runtime is resumed.
    return construction, pattern, raw, positions


# Garments whose task starts from a lay the garment rests in (world-frame positions recorded after
# the garment settled untouched), and the task folder (tasks/<name>/) whose setup.py gives that lay.
# The shirt's authored P13 lay recoils within 0.07 s,
# so its task starts from that same shirt after it has come to rest untouched on the table.
REST_STARTS = {"shirt": "sleeve_fold"}


def rest_start(config, topology):
    """At-rest world positions for this garment's task start, or None for the P13 catalogue lay."""
    name = REST_STARTS.get(config.garment)
    if name is None:
        return None
    return task_folders.setup("laundry", name).setup(config, topology)
