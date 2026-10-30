"""Exact per-step neighbor/gradient cache for repeated DFSPH pressure queries.

No neighbor is truncated. Overflow fails before projection and integration.
Positions do not change during a pressure solve, so its pair gradients are
invariant and can be reused without changing the physical discretization.
"""

import warp as wp
from .sph_kernels import pressure_grad, disabled_pressure_row


@wp.kernel
def build(grid: wp.uint64, q: wp.array[wp.vec3], volume: float, h: float, kernel_kind: int, ordered: int,
          ids: wp.array2d[int], gradients: wp.array2d[wp.vec3],
          counts: wp.array[int], overflow: wp.array[int]):
    i = wp.tid()
    if ordered != 0:
        i = wp.hash_grid_point_id(grid, i)
    count = int(0)
    for j in wp.hash_grid_query(grid, q[i], h):
        r = q[i]-q[j]
        if wp.length_sq(r) < h*h:
            if count < ids.shape[0]:
                ids[count, i] = j
                gradients[count, i] = pressure_grad(r, h, kernel_kind)
            else:
                wp.atomic_max(overflow, 0, count+1)
            count += 1
    counts[i] = count


@wp.kernel
def pressure_velocity(ids: wp.array2d[int], gradients: wp.array2d[wp.vec3],
                      counts: wp.array[int], volume: float, p: wp.array[float], bg: wp.array[wp.vec3],
                      active: wp.array[int], out: wp.array[wp.vec3]):
    i = wp.tid()
    if active[0] == 0:
        return
    dv = -p[i]*bg[i]
    for k in range(counts[i]):
        j = ids[k, i]
        dv -= volume*(p[i]+p[j])*gradients[k, i]
    out[i] = dv


@wp.kernel
def jacobi(ids: wp.array2d[int], gradients: wp.array2d[wp.vec3], counts: wp.array[int],
           volume: float, v: wp.array[wp.vec3], dv: wp.array[wp.vec3], bg: wp.array[wp.vec3],
           rho: wp.array[float], factor: wp.array[float], neighbors: wp.array[int],
           dt: float, density_mode: int, p: wp.array[float], active: wp.array[int], boundary_rate: wp.array[float],
           next_p: wp.array[float], residual: wp.array[float]):
    i = wp.tid()
    if active[0] == 0:
        return
    if disabled_pressure_row(factor[i], density_mode, neighbors[i]):
        next_p[i] = 0.
        residual[i] = 0.
        return
    vi = v[i]+dv[i]
    rate = wp.dot(vi, bg[i])-boundary_rate[i]
    for k in range(counts[i]):
        j = ids[k, i]
        rate += volume*wp.dot(vi-v[j]-dv[j], gradients[k, i])
    error = dt*rate
    if density_mode != 0:
        error += rho[i]-1.0
    next_p[i] = wp.max(p[i]+0.25*factor[i]*error/dt, 0.0)
    residual[i] = wp.max(error, 0.0)


@wp.kernel
def pressure_velocity_positions(ids: wp.array2d[int], q: wp.array[wp.vec3], h: float, kind: int,
        counts: wp.array[int], volume: float, p: wp.array[float], bg: wp.array[wp.vec3],
        active: wp.array[int], out: wp.array[wp.vec3]):
    i = wp.tid()
    if active[0] == 0:
        return
    dv = -p[i]*bg[i]
    for k in range(counts[i]):
        j = ids[k, i]
        dv -= volume*(p[i]+p[j])*pressure_grad(q[i]-q[j], h, kind)
    out[i] = dv


@wp.kernel
def jacobi_positions(ids: wp.array2d[int], q: wp.array[wp.vec3], h: float, kind: int,
        counts: wp.array[int], volume: float, v: wp.array[wp.vec3], dv: wp.array[wp.vec3],
        bg: wp.array[wp.vec3], rho: wp.array[float], factor: wp.array[float], neighbors: wp.array[int],
        dt: float, density_mode: int, p: wp.array[float], active: wp.array[int], boundary_rate: wp.array[float],
        next_p: wp.array[float], residual: wp.array[float]):
    i = wp.tid()
    if active[0] == 0:
        return
    if disabled_pressure_row(factor[i], density_mode, neighbors[i]):
        next_p[i] = 0.
        residual[i] = 0.
        return
    vi = v[i]+dv[i]
    rate = wp.dot(vi, bg[i])-boundary_rate[i]
    for k in range(counts[i]):
        j = ids[k, i]
        rate += volume*wp.dot(vi-v[j]-dv[j], pressure_grad(q[i]-q[j], h, kind))
    error = dt*rate
    if density_mode != 0:
        error += rho[i]-1.0
    next_p[i] = wp.max(p[i]+0.25*factor[i]*error/dt, 0.0)
    residual[i] = wp.max(error, 0.0)


class NeighborCache:
    def __init__(self, water, capacity=256):
        self.water = water
        # This output is [neighbor, particle]-major. Spatially ordered queries
        # lose coalesced writes here; same-state measurement153 found no gain.
        self.ordered_threads = False
        n, device = water.model.particle_count, water.config.device
        self.ids = wp.empty((capacity, n), dtype=int, device=device)
        self.gradients = wp.empty((capacity, n), dtype=wp.vec3, device=device)
        self.counts = wp.zeros(n, dtype=int, device=device)
        self.overflow = wp.zeros(1, dtype=int, device=device)

    def update(self):
        w = self.water
        self.overflow.zero_()
        w._launch(build, [w.grid.id, w.constraint_positions, w.volume_per_particle_m3, w.config.support_m, w.kernel_kind, int(self.ordered_threads)],
            [self.ids, self.gradients, self.counts, self.overflow])
        actual = int(self.overflow.numpy()[0])
        if actual:
            raise ValueError(f"Exact neighbor cache capacity {self.ids.shape[0]} exceeded: at least {actual}; none dropped")
