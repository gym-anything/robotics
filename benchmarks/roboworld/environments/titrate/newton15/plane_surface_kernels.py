"""Mirror extension and half-space quadrature for a planar wetting audit."""

import warp as wp


@wp.kernel
def mirror(q: wp.array[wp.vec3], volumes: wp.array[float], points: wp.array[wp.vec3], out_v: wp.array[float]):
    i = wp.tid()
    points[i] = q[i]
    points[len(q)+i] = wp.vec3(q[i][0], q[i][1], -q[i][2])
    out_v[i] = volumes[i]
    out_v[len(q)+i] = volumes[i]


@wp.kernel
def apply_mirror(qd: wp.array[wp.vec3], mass: wp.array[float], f: wp.array[wp.vec3],
                 dt: float, predicted: wp.array[wp.vec3], result_f: wp.array[wp.vec3]):
    i = wp.tid()
    ghost = f[len(qd)+i]
    total = f[i]+wp.vec3(ghost[0], ghost[1], -ghost[2])
    result_f[i] = total
    predicted[i] = qd[i]+dt*total/mass[i]


@wp.kernel
def add_force(qd: wp.array[wp.vec3], mass: wp.array[float], f: wp.array[wp.vec3],
              dt: float, predicted: wp.array[wp.vec3], result_f: wp.array[wp.vec3]):
    i = wp.tid()
    result_f[i] = f[i]
    predicted[i] = qd[i]+dt*f[i]/mass[i]
