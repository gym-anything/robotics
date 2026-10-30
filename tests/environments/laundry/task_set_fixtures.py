"""Recorded geometry and runtime stubs for the task-set tests; never import a simulator.

Reconstruct the semantic index from the committed NPZ's material mesh. This
avoids regenerating a mesh with platform-dependent triangulation. Path samples
use the same authored arclength and SemanticGarmentIndex implementation.
This is a test fixture, not a replacement for the production reset loader.
"""
from types import SimpleNamespace
from functools import lru_cache

import numpy as np

from laundry.garment_patterns import get_garment_construction
from laundry.garment_task import SemanticGarmentIndex
from laundry.newton15.config import GARMENTS
from tests.packages import PACKAGES

FRAMES = PACKAGES / "laundry/data/frames"


@lru_cache(None)
def recorded(garment):
    construction = get_garment_construction(GARMENTS[garment])
    prefix = construction.id + "_"
    with np.load(FRAMES / "p13_garment_flat_lay_catalog.npz", allow_pickle=False) as saved:
        data = {key[len(prefix):]: saved[key].copy() for key in saved.files if key.startswith(prefix)}
    material = data["panel_material_coordinates_m"]
    panels = tuple(str(p) for p in data["panel_ids"])
    ranges, paths, fractions = {}, {}, {}
    for i, panel_id in enumerate(panels):
        ids = np.unique(data["panel_triangles"][data["triangle_panel_index"] == i])
        ranges[panel_id] = (int(ids.min()), int(ids.max()) + 1)
        panel = construction.panel_map[panel_id]
        for path in (*panel.outline_paths, *panel.internal_paths):
            line = np.asarray(path.points_m)
            lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
            offsets = np.r_[0., lengths.cumsum()]
            found = {}
            for a, b, length, offset in zip(line[:-1], line[1:], lengths, offsets[:-1]):
                alpha = np.clip((material[ids] - a) @ (b - a) / length**2, 0., 1.)
                distance = np.linalg.norm(material[ids] - (a + alpha[:, None] * (b - a)), axis=1)
                for j in np.flatnonzero(distance < 1.e-7):
                    found[int(ids[j])] = (offset + alpha[j] * length) / offsets[-1]
            ordered = sorted(found, key=found.get)
            paths[panel_id, path.id] = np.array(ordered, dtype=np.int32)
            fractions[panel_id, path.id] = np.array([found[k] for k in ordered])
    top = SimpleNamespace(**{k: data[k] for k in (
        "physical_triangles", "triangle_panel_index", "material_to_physical_vertex",
        "panel_material_coordinates_m", "panel_triangles")},
        physical_positions_m=data["positions_m"], panel_ids=panels,
        physical_vertex_count=len(data["positions_m"]), material_vertex_count=len(material),
        construction_id=construction.id)
    mesh = SimpleNamespace(construction_id=construction.id, vertex_count=len(material),
                           panel_vertex_ranges=ranges, path_vertices=paths, path_fractions=fractions)
    index = SemanticGarmentIndex(construction, mesh, top)
    x = data["positions_m"].copy()
    # Same task-space placement; radius is irrelevant to these stubbed tests.
    x[:, 2] += .790 - x[:, 2].min()
    if garment == "shirt":
        with np.load(FRAMES / "shirt_sleeve_fold_rest_start.npz", allow_pickle=False) as saved:
            x = saved["positions_m"].copy()
    return index, x


def half_fold(x):
    y = x.copy()
    left = y[:, 0] < 0.
    y[left, 0] *= -1
    y[left, 2] += .007
    return y


def thirds_fold(x, *, z_fold=True):
    """Constructed flat plies only; no assertion of physical reachability/crossings."""
    y = x.copy()
    # Slightly offset C-folds are valid too; the coarse mesh's transition triangles
    # consume too much flat overlap in the sharp, exactly coincident C construction.
    a, b = (.075, .075) if z_fold else (.085, .070)
    left, right = x[:, 0] < -a, x[:, 0] > b
    y[left, 0] = -2*a - x[left, 0]
    y[right, 0] = 2*b - x[right, 0]
    y[left, 2] += .007
    y[right, 2] += -.007 if z_fold else .014
    y[:, 2] += .790 - y[:, 2].min()
    return y
