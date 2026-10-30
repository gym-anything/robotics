"""Conservative body-local coverage checks without per-body host transforms.

Only spatial-query work changes. The world fluid AABB is exact; all eight
corners are transformed, using double precision and a normalized quaternion.
Boundary samples keep the existing body-local lattice and coverage margin.
"""

import numpy as np
import warp as wp


@wp.kernel
def check(bounds: wp.array[float], bodies: wp.array[int], poses: wp.array[wp.transform],
        coverage: wp.array2d[wp.float64], shapes: wp.array2d[wp.float64], support: wp.float64,
        current: wp.array2d[wp.float64], status: wp.array[int]):
    i = wp.tid()
    b = bodies[i]
    low, high = wp.vec3d(1.e30), wp.vec3d(-1.e30)
    translation, vector, scalar = wp.vec3d(0.), wp.vec3d(0.), wp.float64(1.)
    if b >= 0:
        pose = poses[b]
        translation = wp.vec3d(wp.transform_get_translation(pose))
        q = wp.transform_get_rotation(pose)
        norm = wp.sqrt(wp.float64(q[0])*wp.float64(q[0])+wp.float64(q[1])*wp.float64(q[1])
            +wp.float64(q[2])*wp.float64(q[2])+wp.float64(q[3])*wp.float64(q[3]))
        vector = -wp.vec3d(wp.float64(q[0]), wp.float64(q[1]), wp.float64(q[2]))/norm
        scalar = wp.float64(q[3])/norm
    for corner in range(8):
        p = wp.vec3d(0.)
        for axis in range(3):
            index = axis+3 if (corner & (1 << axis)) != 0 else axis
            p[axis] = wp.float64(bounds[index])
        p -= translation
        p += wp.float64(2.)*wp.cross(vector, wp.cross(vector, p)+scalar*p)
        low, high = wp.min(low, p), wp.max(high, p)
    intersects = bool(True)
    for axis in range(3):
        if high[axis]+support < shapes[i, axis] or low[axis]-support > shapes[i, axis+3]:
            intersects = False
    for axis in range(3):
        current[i, axis] = low[axis]
        current[i, axis+3] = high[axis]
        if not wp.isfinite(low[axis]) or not wp.isfinite(high[axis]):
            wp.atomic_max(status, 0, 2)
        elif intersects and (low[axis]-support < coverage[i, axis] or high[axis]+support > coverage[i, axis+3]):
            wp.atomic_max(status, 0, 1)


class BoundaryCoverage:
    def __init__(self, bodies, device, shape_bounds=None):
        self.identities = tuple(bodies)
        self.device = device
        self.bodies = wp.array(self.identities, dtype=int, device=device)
        self.current = wp.empty((len(self.identities), 6), dtype=wp.float64, device=device)
        self.coverage = wp.empty_like(self.current)
        # Exact broad-phase rejection only: a support-expanded fluid box
        # disjoint from the body's entire solid AABB needs no solid samples.
        # A moving body is rechecked every tick, including when it approaches.
        boxes = [np.asarray(shape_bounds[b]).reshape(6) if shape_bounds is not None
                 else np.array([-1.e30]*3+[1.e30]*3) for b in self.identities]
        self.shapes = wp.array(np.asarray(boxes, np.float64), dtype=wp.float64, device=device)
        self.status = wp.zeros(1, dtype=int, device=device)
        self.previous = None

    def needs_rebuild(self, bounds, poses, coverage, support):
        if self.previous is not coverage:
            rows = [np.asarray(coverage[b]).reshape(6) if b in coverage else
                    np.array([1.e30]*3+[-1.e30]*3) for b in self.identities]
            self.coverage.assign(np.asarray(rows, np.float64))
            self.previous = coverage
        self.status.zero_()
        wp.launch(check, dim=len(self.identities), inputs=[bounds, self.bodies, poses,
            self.coverage, self.shapes, wp.float64(support)], outputs=[self.current, self.status], device=self.device)
        result = int(self.status.numpy()[0])
        if result == 2:
            raise FloatingPointError('Nonfinite boundary coverage transform')
        return result == 1

    def read(self):
        values = self.current.numpy().reshape(-1, 2, 3)
        return {b: values[i] for i, b in enumerate(self.identities)}
