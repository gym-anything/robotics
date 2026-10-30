"""Spatial soil measurements; no simulator commands or state changes."""
import itertools
import numpy as np


def pipe_sand_cover(positions, volumes, spacing, x_bounds, y, crown, radius,
                    required_depth=0.05, cell=0.02, minimum_packing=0.55):
    """Sand occupancy from crown to required cover, resolved along and across pipe.

    Each particle contributes its solid volume over a cube of one particle spacing.
    Exact cube/cell overlap avoids aliasing a regular lattice against bin edges.
    A station qualifies only when every transverse and vertical cell has enough
    sand. Floating sand, isolated mounds, and a gap along the crown cannot supply
    cover elsewhere. This is a finite-resolution geometric criterion, not a
    compaction or load-bearing test. Inputs must contain sand particles only.
    """
    lo = np.array([x_bounds[0], y-radius, crown], float)
    hi = np.array([x_bounds[1], y+radius, crown+required_depth], float)
    size = hi-lo
    if spacing <= 0 or (size <= 0).any():
        raise ValueError('positive particle spacing and cover dimensions required')
    n = np.maximum(1, np.minimum(np.ceil(size/cell), np.floor(size/spacing))).astype(int)
    widths = size/n
    p = np.asarray(positions, float)
    vol = np.asarray(volumes, float)
    keep = ((p+spacing/2 > lo) & (p-spacing/2 < hi)).all(axis=1)
    p, vol = p[keep], vol[keep]
    first = np.floor((p-spacing/2-lo)/widths).astype(int)
    last = np.floor((p+spacing/2-lo)/widths).astype(int)
    grid = np.zeros(n)
    for shift in itertools.product((0, 1), repeat=3):
        idx = first+shift
        valid = ((idx >= 0) & (idx < n) & (idx <= last)).all(axis=1)
        j, points = idx[valid], p[valid]
        cell_lo = lo+j*widths
        overlap = np.maximum(0, np.minimum(points+spacing/2, cell_lo+widths)
                             - np.maximum(points-spacing/2, cell_lo))
        weight = np.prod(overlap, axis=1)/spacing**3
        np.add.at(grid, tuple(j.T), vol[valid]*weight/np.prod(widths))
    # Contiguous filled layers starting at the crown; no pooled height statistic.
    layers = (grid >= minimum_packing).all(axis=1)
    depth = np.cumprod(layers, axis=1).sum(axis=1)*widths[2]
    covered = layers.all(axis=1)
    return dict(coverage_fraction=float(covered.mean()), minimum_depth_m=float(depth.min()),
                required_depth_m=required_depth, minimum_packing=minimum_packing,
                station_x_m=(lo[0]+(np.arange(n[0])+.5)*widths[0]).tolist(),
                contiguous_depth_m=depth.tolist(), packing=grid.tolist(),
                cell_widths_m=widths.tolist(), particle_spacing_m=spacing)
