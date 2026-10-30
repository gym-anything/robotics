"""Grid quadrature of CSF; the material-point liquid inventory is unchanged."""

import warp as wp


@wp.kernel
def sample(grid: wp.uint64, q: wp.array[wp.vec3], volume: wp.array[float], h: float,
            lower: wp.vec3, dx: float, real_count: int, plane: bool, angle: float,
            gradient: wp.array(dtype=wp.vec3, ndim=3), normals: wp.array(dtype=wp.vec3, ndim=3),
            field: wp.array(dtype=float, ndim=3)):
    i, j, k = wp.tid()
    x = lower+dx*wp.vec3(float(i), float(j), float(k))
    query = wp.hash_grid_query(grid, x, h)
    p = int(0)
    g, c = wp.vec3(0.), float(0.)
    while wp.hash_grid_query_next(query, p):
        d = x-q[p]
        r = wp.length(d)/h
        if r < 1.:
            a = 1.-r
            g += volume[p]*(-210./(wp.pi*h*h*h*h*h))*a*a*a*d
            if p < real_count:
                c += volume[p]*21./(2.*wp.pi*h*h*h)*a*a*a*a*(1.+4.*r)
    nhat = g/wp.sqrt(wp.dot(g, g)+(.001/h)*(.001/h))
    if plane and k == 0:
        tangent = wp.vec3(nhat[0], nhat[1], 0.)
        if wp.length(tangent) > .001:
            nhat = wp.normalize(tangent)*wp.sin(angle)-wp.vec3(0., 0., wp.cos(angle))
    gradient[i, j, k] = g
    normals[i, j, k] = nhat
    field[i, j, k] = c


@wp.kernel
def curvature_force(gradient: wp.array(dtype=wp.vec3, ndim=3), normals: wp.array(dtype=wp.vec3, ndim=3),
                     sigma: float, dx: float, plane: bool, force: wp.array(dtype=wp.vec3, ndim=3)):
    i, j, k = wp.tid()
    lo_i, hi_i = wp.max(0, i-1), wp.min(normals.shape[0]-1, i+1)
    lo_j, hi_j = wp.max(0, j-1), wp.min(normals.shape[1]-1, j+1)
    lo_k, hi_k = wp.max(0, k-1), wp.min(normals.shape[2]-1, k+1)
    divergence = (normals[hi_i, j, k][0]-normals[lo_i, j, k][0])/(float(hi_i-lo_i)*dx)
    divergence += (normals[i, hi_j, k][1]-normals[i, lo_j, k][1])/(float(hi_j-lo_j)*dx)
    divergence += (normals[i, j, hi_k][2]-normals[i, j, lo_k][2])/(float(hi_k-lo_k)*dx)
    force[i, j, k] = -sigma*divergence*gradient[i, j, k]


@wp.kernel
def distribute(q: wp.array[wp.vec3], v: wp.array[wp.vec3], mass: wp.array[float], volume: wp.array[float],
                h: float, lower: wp.vec3, dx: float, field: wp.array(dtype=float, ndim=3),
                grid_f: wp.array(dtype=wp.vec3, ndim=3), dt: float, plane: bool,
                predicted: wp.array[wp.vec3], forces: wp.array[wp.vec3]):
    p = wp.tid()
    center = (q[p]-lower)/dx
    radius = int(wp.ceil(h/dx))
    accum = wp.vec3(0.)
    for i in range(wp.max(0, int(center[0])-radius), wp.min(field.shape[0], int(center[0])+radius+2)):
        for j in range(wp.max(0, int(center[1])-radius), wp.min(field.shape[1], int(center[1])+radius+2)):
            for k in range(wp.max(0, int(center[2])-radius), wp.min(field.shape[2], int(center[2])+radius+2)):
                x = lower+dx*wp.vec3(float(i), float(j), float(k))
                r = wp.length(x-q[p])/h
                if r < 1. and field[i, j, k] > 1.e-7:
                    a = 1.-r
                    weight = volume[p]*21./(2.*wp.pi*h*h*h)*a*a*a*a*(1.+4.*r)/field[i, j, k]
                    quadrature = dx*dx*dx
                    if plane and k == 0:
                        quadrature *= .5
                    accum += grid_f[i, j, k]*weight*quadrature
    forces[p] = accum
    predicted[p] = v[p]+dt*accum/mass[p]
