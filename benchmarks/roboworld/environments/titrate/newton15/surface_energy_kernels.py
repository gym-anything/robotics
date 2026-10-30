"""Exact Warp derivatives of the documented grid-quadrature area energy."""

import warp as wp


@wp.kernel
def sample_weighted_area(grid: wp.uint64, q: wp.array[wp.vec3], volume: wp.array[float],
                        h: float, lower: wp.vec3, dx: float, epsilon: float,
                        normals: wp.array(dtype=wp.vec3, ndim=3),
                        potential: wp.array(dtype=float, ndim=3), energies: wp.array(dtype=float, ndim=3)):
    i, j, k = wp.tid()
    x = lower+dx*wp.vec3(float(i), float(j), float(k))
    c, g = float(0.), wp.vec3(0.)
    for p in wp.hash_grid_query(grid, x, h):
        d = x-q[p]
        r = wp.length(d)/h
        if r < 1.:
            a = 1.-r
            c += volume[p]*21./(2.*wp.pi*h*h*h)*a*a*a*a*(1.+4.*r)
            g += volume[p]*(-210./(wp.pi*h*h*h*h*h))*a*a*a*d
    norm = wp.sqrt(wp.length_sq(g)+epsilon*epsilon)
    f, derivative = float(0.), float(0.)
    if c > 0. and c < 1.:
        # Integral_0^1 30 c^2(1-c)^2 dc = 1. A planar monotone interface
        # therefore still costs sigma per area. Value AND slope vanish at
        # pure phases, suppressing the bulk lattice-gradient area artifact.
        f = 30.*c*c*(1.-c)*(1.-c)
        derivative = 60.*c*(1.-c)*(1.-2.*c)
    normals[i, j, k] = f*g/norm
    potential[i, j, k] = derivative*(norm-epsilon)
    energies[i, j, k] = f*(norm-epsilon)


@wp.kernel
def sample_gradient(grid: wp.uint64, q: wp.array[wp.vec3], volume: wp.array[float],
                    h: float, lower: wp.vec3, dx: float, epsilon: float,
                    normals: wp.array(dtype=wp.vec3, ndim=3),
                    energies: wp.array(dtype=float, ndim=3)):
    i, j, k = wp.tid()
    x = lower + dx*wp.vec3(float(i), float(j), float(k))
    query = wp.hash_grid_query(grid, x, h)
    p = int(0)
    gradient = wp.vec3(0.0)
    while wp.hash_grid_query_next(query, p):
        d = x-q[p]
        r = wp.length(d)
        if r < h:
            a = 1.0-r/h
            gradient += volume[p]*(-210.0/(wp.pi*h*h*h*h*h))*a*a*a*d
    norm = wp.sqrt(wp.dot(gradient, gradient)+epsilon*epsilon)
    normals[i, j, k] = gradient/norm
    energies[i, j, k] = norm-epsilon


@wp.kernel
def area_force(q: wp.array[wp.vec3], v: wp.array[wp.vec3], mass: wp.array[float],
               volume: wp.array[float], h: float, lower: wp.vec3, dx: float,
               normals: wp.array(dtype=wp.vec3, ndim=3), potential: wp.array(dtype=float, ndim=3),
               sigma: float, dt: float,
               predicted_v: wp.array[wp.vec3], forces: wp.array[wp.vec3]):
    p = wp.tid()
    center = (q[p]-lower)/dx
    radius = int(wp.ceil(h/dx))
    accum = wp.vec3(0.0)
    for i in range(wp.max(0, int(center[0])-radius), wp.min(normals.shape[0], int(center[0])+radius+2)):
        for j in range(wp.max(0, int(center[1])-radius), wp.min(normals.shape[1], int(center[1])+radius+2)):
            for k in range(wp.max(0, int(center[2])-radius), wp.min(normals.shape[2], int(center[2])+radius+2)):
                d = lower+dx*wp.vec3(float(i), float(j), float(k))-q[p]
                r = wp.length(d)
                if r < h:
                    a = 1.0-r/h
                    n = normals[i, j, k]
                    hessian_n = (-210.0/(wp.pi*h*h*h*h*h))*a*a*a*n
                    if r > 1.0e-12:
                        hessian_n += (630.0/(wp.pi*h*h*h*h*h*h))*a*a*d*(wp.dot(d, n)/r)
                    accum += hessian_n
                    accum += potential[i, j, k]*(-210.0/(wp.pi*h*h*h*h*h))*a*a*a*d
    f = sigma*volume[p]*dx*dx*dx*accum
    forces[p] = f
    predicted_v[p] = v[p]+dt*f/mass[p]


@wp.kernel
def sample_phase(grid: wp.uint64, q: wp.array[wp.vec3], volume: wp.array[float],
                 h: float, lower: wp.vec3, dx: float, epsilon: float,
                 gradient_term: wp.array(dtype=wp.vec3, ndim=3),
                 potential: wp.array(dtype=float, ndim=3), energies: wp.array(dtype=float, ndim=3)):
    i, j, k = wp.tid()
    x = lower+dx*wp.vec3(float(i), float(j), float(k))
    c, gradient = float(0.0), wp.vec3(0.0)
    for p in wp.hash_grid_query(grid, x, h):
        r = x-q[p]
        distance = wp.length(r)
        if distance < h:
            a = 1.0-distance/h
            c += volume[p]*21.0/(2.0*wp.pi*h*h*h)*a*a*a*a*(1.0+4.0*distance/h)
            gradient += volume[p]*(-210.0/(wp.pi*h*h*h*h*h))*a*a*a*r
    normalization = 3.0*wp.sqrt(2.0)
    gradient_term[i, j, k] = normalization*epsilon*gradient
    potential[i, j, k] = normalization/epsilon*2.0*c*(1.0-c)*(1.0-2.0*c)
    energies[i, j, k] = normalization*(epsilon/2.0*wp.length_sq(gradient)+c*c*(1.0-c)*(1.0-c)/epsilon)
