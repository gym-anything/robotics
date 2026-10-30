"""Wendland volume-field visualization; independent of the pressure kernel."""

import warp as wp


@wp.func
def volume_at(grid: wp.uint64, q: wp.array[wp.vec3], x: wp.vec3, volume: float, h: float):
    total = float(0.)
    for p in wp.hash_grid_query(grid, x, h):
        r = wp.length(x-q[p])/h
        if r < 1.:
            a = 1.-r
            total += volume*21./(2.*wp.pi*h*h*h)*a*a*a*a*(1.+4.*r)
    return total


@wp.kernel
def sample_bricks(grid: wp.uint64, q: wp.array[wp.vec3], volume: float, h: float,
                  origins: wp.array[wp.vec3], spacing: float, field: wp.array4d[float]):
    b, i, j, k = wp.tid()
    x = origins[b]+spacing*wp.vec3(float(i), float(j), float(k))
    field[b, i, j, k] = volume_at(grid, q, x, volume, h)


@wp.kernel
def at_particles(grid: wp.uint64, q: wp.array[wp.vec3], volume: float, h: float,
                 field: wp.array[float]):
    i = wp.tid()
    field[i] = volume_at(grid, q, q[i], volume, h)


@wp.kernel
def sample_volume(grid: wp.uint64, q: wp.array[wp.vec3], volume: float, h: float,
                  lower: wp.vec3, spacing: float, field: wp.array(dtype=float, ndim=3)):
    i, j, k = wp.tid()
    x = lower + spacing*wp.vec3(float(i), float(j), float(k))
    neighbors = wp.hash_grid_query(grid, x, h)
    p = int(0)
    total = float(0.0)
    while wp.hash_grid_query_next(neighbors, p):
        r = wp.length(x-q[p])/h
        if r < 1.0:
            a = 1.0-r
            total += volume * 21.0/(2.0*wp.pi*h*h*h) * a*a*a*a*(1.0+4.0*r)
    field[i, j, k] = total
