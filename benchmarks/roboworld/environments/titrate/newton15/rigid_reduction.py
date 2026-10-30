"""Grouped boundary wrenches without per-node atomic contention.

The body-local quadrature and physical loads are unchanged. A stable index
permutation is rebuilt only when the sample identities change. Each block
reduces up to 256 samples for one owner; a second deterministic sum writes
one wrench per rigid body. World-anchored nodes still have no dynamic owner.
"""

import numpy as np
import warp as wp

Vec6f = wp.types.vector(length=6, dtype=wp.float32)
Vec6d = wp.types.vector(length=6, dtype=wp.float64)


@wp.kernel
def partial(order: wp.array[int], chunks: wp.array[wp.vec3i], points: wp.array[wp.vec3],
            forces: wp.array[wp.vec3], poses: wp.array[wp.transform], com: wp.array[wp.vec3],
            sums: wp.array[wp.spatial_vector]):
    block, lane = wp.tid()
    row = chunks[block]
    wrench = Vec6f(0.)
    if lane < row[2]:
        i = order[row[1]+lane]
        arm = points[i]-wp.transform_point(poses[row[0]], com[row[0]])
        wrench = wp.spatial_vector(forces[i], wp.cross(arm, forces[i]))
    values = wp.tile(wrench, preserve_type=True)
    total = wp.tile_reduce(wp.add, values)
    if lane == 0:
        sums[block] = total[0]


@wp.kernel
def finish(ranges: wp.array[wp.vec2i], sums: wp.array[wp.spatial_vector],
           out: wp.array[wp.spatial_vector]):
    body = wp.tid()
    row = ranges[body]
    total = Vec6d(wp.float64(0.))
    for j in range(row[0], row[1]):
        total += Vec6d(sums[j])
    out[body] += Vec6f(total)


class RigidReduction:
    def __init__(self, body_count, device):
        self.body_count, self.device = body_count, device
        self.identity = None

    def prepare(self, identities):
        if self.identity is identities:
            return
        ids = identities.numpy()
        if np.any(ids < -1) or np.any(ids >= self.body_count):
            raise ValueError('Boundary reaction owner outside rigid body layout')
        order = np.argsort(ids, kind='stable').astype(np.int32)
        order = order[ids[order] >= 0]
        sorted_ids = ids[order]
        chunks, ranges = [], []
        for body in range(self.body_count):
            low, high = np.searchsorted(sorted_ids, (body, body+1))
            start = len(chunks)
            chunks.extend((body, i, min(256, high-i)) for i in range(low, high, 256))
            ranges.append((start, len(chunks)))
        self.order = wp.array(order, dtype=int, device=self.device)
        self.chunks = wp.array(np.asarray(chunks, np.int32).reshape(-1, 3), dtype=wp.vec3i, device=self.device)
        self.ranges = wp.array(ranges, dtype=wp.vec2i, device=self.device)
        self.sums = wp.empty(len(chunks), dtype=wp.spatial_vector, device=self.device)
        self.identity = identities

    def add(self, points, forces, poses, com, out):
        if len(self.chunks):
            wp.launch(partial, dim=(len(self.chunks), 256), block_dim=256,
                inputs=[self.order, self.chunks, points, forces, poses, com], outputs=[self.sums], device=self.device)
        wp.launch(finish, dim=self.body_count, inputs=[self.ranges, self.sums], outputs=[out], device=self.device)
