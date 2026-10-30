"""Resolved liquid-plug detection for the lumped pipette gas circuit.

Cross sections use equal-area disk samples and the liquid particle resolution.
An empty or partly uncovered airway vents through its remaining area; a plug
spanning a cross section seals it. Entrained individual gas bubbles are not
resolved. This is not a volume threshold that lets any stray droplet seal a tube.
"""

import math
import numpy as np
import warp as wp
from .pipette_geometry import profile, LENGTH_M


@wp.kernel
def uncovered_sections(grid: wp.uint64, liquid: wp.array[wp.vec3], pose: wp.transform,
                       samples: wp.array2d[wp.vec3], radius: float,
                       uncovered: wp.array[float]):
    section = wp.tid()
    empty = int(0)
    for k in range(samples.shape[1]):
        point = wp.transform_point(pose, samples[section, k])
        covered = bool(False)
        for j in wp.hash_grid_query(grid, point, radius):
            if wp.length_sq(liquid[j]-point) <= radius*radius:
                covered = True
        if not covered:
            empty += 1
    uncovered[section] = float(empty)/float(samples.shape[1])


class PipetteAirway:
    def __init__(self, water):
        self.water = water
        inner = np.asarray(profile())[7:][::-1]
        dx = water.config.spacing_m
        heights = np.arange(dx, LENGTH_M-dx, dx)
        samples = np.empty((len(heights), 20, 3), np.float32)
        # Two opposite 10-point equal-area spirals retain reflection symmetry.
        for k in range(20):
            r = np.sqrt(((k//2)+.5)/10)*np.interp(heights, inner[:, 1], inner[:, 0])
            theta = (k//2)*math.pi*(3.-math.sqrt(5.))+(k % 2)*math.pi
            samples[:, k, :] = np.column_stack((r*np.cos(theta), r*np.sin(theta), heights))
        self.samples = wp.array(samples, dtype=wp.vec3, ndim=2, device=water.config.device)
        self.uncovered = wp.ones(len(heights), dtype=float, device=water.config.device)
        self.minimum = wp.zeros(1, dtype=float, device=water.config.device)
        self.last = {}

    def area(self, pose):
        w = self.water
        # Current coordinates, not a stale pre-integration hash grid.
        w.grid.build(w.state.particle_q, w.config.support_m)
        wp.launch(uncovered_sections, len(self.uncovered),
            inputs=[w.grid.id, w.state.particle_q, pose, self.samples,
                    math.sqrt(3.)/2*w.config.spacing_m], outputs=[self.uncovered],
            device=w.config.device)
        # Read only the small section array; no full particle download.
        fraction = float(self.uncovered.numpy().min())
        area = math.pi*.0006**2*fraction
        self.last = dict(uncovered_fraction=fraction, effective_air_area_m2=area,
            axial_spacing_m=w.config.spacing_m, disk_samples=20,
            liquid_coverage_radius_m=math.sqrt(3.)/2*w.config.spacing_m)
        return area
