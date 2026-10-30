"""Equal/opposite liquid-solid surface-energy forces at quadrature points."""

import warp as wp


@wp.kernel
def contact_line(grid: wp.uint64, q: wp.array[wp.vec3], volumes: wp.array[float],
                  wall_q: wp.array[wp.vec3], normals: wp.array[wp.vec3], areas: wp.array[float],
                  coefficients: wp.array[float], h: float, sigma: float,
                  field: wp.array[float], forces: wp.array[wp.vec3]):
    j = wp.tid()
    query = wp.hash_grid_query(grid, wall_q[j], h)
    p = int(0)
    gradient, c = wp.vec3(0.), float(0.)
    while wp.hash_grid_query_next(query, p):
        d = wall_q[j]-q[p]
        r = wp.length(d)/h
        if r < 1.:
            a = 1.-r
            gradient += volumes[p]*(-210./(wp.pi*h*h*h*h*h))*a*a*a*d
            c += volumes[p]*21./(2.*wp.pi*h*h*h)*a*a*a*a*(1.+4.*r)
    tangent = gradient-wp.dot(gradient, normals[j])*normals[j]
    forces[j] = -sigma*coefficients[j]*areas[j]*tangent
    field[j] = c


@wp.kernel
def distribute_line(grid: wp.uint64, wall_q: wp.array[wp.vec3], field: wp.array[float], line_f: wp.array[wp.vec3],
                     q: wp.array[wp.vec3], volumes: wp.array[float], mass: wp.array[float], h: float, dt: float,
                     predicted_v: wp.array[wp.vec3], particle_f: wp.array[wp.vec3], wall_f: wp.array[wp.vec3]):
    p = wp.tid()
    query = wp.hash_grid_query(grid, q[p], h)
    j = int(0)
    accum = wp.vec3(0.)
    while wp.hash_grid_query_next(query, j):
        r = wp.length(wall_q[j]-q[p])/h
        if r < 1. and field[j] > 1.e-7:
            a = 1.-r
            weight = volumes[p]*21./(2.*wp.pi*h*h*h)*a*a*a*a*(1.+4.*r)/field[j]
            f = line_f[j]*weight
            accum += f
            wp.atomic_add(wall_f, j, -f)
    particle_f[p] += accum
    predicted_v[p] += dt*accum/mass[p]


@wp.kernel
def wall_force(grid: wp.uint64, wall_q: wp.array[wp.vec3], normals: wp.array[wp.vec3],
               areas: wp.array[float], coefficients: wp.array[float], q: wp.array[wp.vec3],
               volumes: wp.array[float], mass: wp.array[float], h: float, sigma: float, dt: float,
               predicted_v: wp.array[wp.vec3], particle_f: wp.array[wp.vec3], wall_f: wp.array[wp.vec3]):
    p = wp.tid()
    accum = wp.vec3(0.0)
    query = wp.hash_grid_query(grid, q[p], h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        d = wall_q[j]-q[p]
        r = wp.length(d)
        if r < h and wp.dot(-d, normals[j]) >= 0.0:
            a = 1.0-r/h
            grad = (-210.0/(wp.pi*h*h*h*h*h))*a*a*a*d
            f = -sigma*coefficients[j]*volumes[p]*areas[j]*grad
            accum += f
            wp.atomic_add(wall_f, j, -f)
    particle_f[p] += accum
    predicted_v[p] += dt*accum/mass[p]
