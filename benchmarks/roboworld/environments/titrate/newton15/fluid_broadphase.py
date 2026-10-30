"""Conservative compact particle/shape candidates from real world-space AABBs.

The previous Cartesian-product allocation scaled with unrelated objects. This
only drops pairs whose inflated bounds provably cannot meet; narrow phase still
uses the same exact primitive/mesh queries. Overflow is an explicit error.
"""

import numpy as np
import warp as wp


@wp.kernel
def world_bounds(local_low: wp.array[wp.vec3], local_high: wp.array[wp.vec3], bodies: wp.array[int],
        body_q: wp.array[wp.transform], lows: wp.array[wp.vec3], highs: wp.array[wp.vec3]):
    i = wp.tid()
    low, high = wp.vec3(1.e30), wp.vec3(-1.e30)
    for corner in range(8):
        p = wp.vec3(local_low[i])
        for axis in range(3):
            if (corner & (1 << axis)) != 0:
                p[axis] = local_high[i][axis]
        if bodies[i] >= 0:
            p = wp.transform_point(body_q[bodies[i]], p)
        low, high = wp.min(low, p), wp.max(high, p)
    lows[i] = low
    highs[i] = high


@wp.kernel
def compact(q: wp.array[wp.vec3], radii: wp.array[float], shapes: wp.array[int], mesh: wp.array[int],
        lows: wp.array[wp.vec3], highs: wp.array[wp.vec3], margins: wp.array[float], extra: float,
        counts: wp.array[int], primitive_pairs: wp.array[wp.vec2i], mesh_pairs: wp.array[wp.vec2i]):
    i, s = wp.tid()
    shape = shapes[s]
    margin = extra+radii[i]+margins[shape]
    if wp.min(q[i]-lows[s]) < -margin or wp.min(highs[s]-q[i]) < -margin:
        return
    slot = wp.atomic_add(counts, mesh[s], 1)
    if mesh[s] != 0:
        if slot < mesh_pairs.shape[0]:
            mesh_pairs[slot] = wp.vec2i(i, shape)
    elif slot < primitive_pairs.shape[0]:
        primitive_pairs[slot] = wp.vec2i(i, shape)


class FluidBroadphase:
    def __init__(self, coupling, maximum_candidates=10_000_000):
        self.coupling = coupling
        m, w = coupling.model, coupling.water
        self.capacity = min(maximum_candidates, 8*w.model.particle_count)
        self.buffers = [wp.empty(self.capacity, dtype=wp.vec2i, device=m.device) for _ in range(2)]
        self.counts = wp.zeros(2, dtype=int, device=m.device)
        self.generation = None

    def update(self):
        s, (primitive, meshes) = self.coupling, self.buffers
        m, w = s.model, s.water
        if self.generation != s.samples.rebuilds:
            from .runtime import require_newton
            newton, _ = require_newton()
            selected = set(s.samples.candidate_shape_ids)
            records = [(body, shape) for body, shapes in s.samples.groups.items()
                for shape in shapes if shape.index in selected]
            self.shapes = wp.array([shape.index for body, shape in records], dtype=int, device=m.device)
            self.bodies = wp.array([body for body, shape in records], dtype=int, device=m.device)
            self.local_low = wp.array(np.array([shape.bounds[0] for body, shape in records]).reshape(-1, 3), dtype=wp.vec3, device=m.device)
            self.local_high = wp.array(np.array([shape.bounds[1] for body, shape in records]).reshape(-1, 3), dtype=wp.vec3, device=m.device)
            kinds = m.shape_type.numpy()
            self.mesh = wp.array([int(kinds[shape.index] in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH))
                for body, shape in records], dtype=int, device=m.device)
            self.lows, self.highs = wp.empty_like(self.local_low), wp.empty_like(self.local_high)
            self.generation = s.samples.rebuilds
        wp.launch(world_bounds, dim=len(self.shapes), inputs=[self.local_low, self.local_high,
            self.bodies, s.state.body_q], outputs=[self.lows, self.highs], device=m.device)
        self.counts.zero_()
        wp.launch(compact, dim=(w.model.particle_count, len(self.shapes)), inputs=[w.state.particle_q,
            w.model.particle_radius, self.shapes, self.mesh, self.lows, self.highs, m.shape_margin,
            2*w.config.spacing_m], outputs=[self.counts, primitive, meshes], device=m.device)
        counts = self.counts.numpy()
        if np.any(counts > self.capacity):
            raise ValueError(f'Fluid broadphase capacity {self.capacity} exceeded by {counts.tolist()}; no pair dropped silently')
        return primitive[:int(counts[0])], meshes[:int(counts[1])]
