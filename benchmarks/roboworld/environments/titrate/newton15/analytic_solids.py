"""Analytic fluid boundaries for the authored bored rotary valve.

The triangle surfaces are render/rigid-decomposition approximations of these
same cylinder differences. Their independently marched sharp intersections
must not define a spurious hydraulic clearance at the zero-clearance bearing.
The CSG field has the exact solid sign and unit active-primitive normal; its
magnitude is a conservative distance bound, not an exact Euclidean distance
at Boolean intersections. No fluid state or physical dimensions are changed.
"""

import numpy as np
import warp as wp


def register(builder, shape, kind, parameters):
    if kind not in (1, 2):
        raise ValueError('Analytic kind is 1 for plug, 2 for socket')
    p = np.asarray(parameters, float)
    if p.shape != (4,) or not np.isfinite(p).all() or p[0] <= 0 or p[2] <= 0 or p[3] <= 0:
        raise ValueError('Invalid analytic valve dimensions')
    if not hasattr(builder, 'titration_analytic_solids'):
        builder.titration_analytic_solids = {}
    builder.titration_analytic_solids[shape] = (kind, p.tolist())


@wp.func
def cylinder_y(p: wp.vec3, radius: float, half_length: float):
    r = wp.sqrt(p[0]*p[0]+p[2]*p[2])
    radial = r-radius
    axial = wp.abs(p[1])-half_length
    nr = wp.vec3(1., 0., 0.)
    if r > 1.e-12:
        nr = wp.vec3(p[0]/r, 0., p[2]/r)
    na = wp.vec3(0., wp.sign(p[1]), 0.)
    dr, da = wp.max(radial, 0.), wp.max(axial, 0.)
    length = wp.sqrt(dr*dr+da*da)
    n = nr if radial >= axial else na
    if length > 0.:
        n = (dr*nr+da*na)/length
    return wp.vec4(n[0], n[1], n[2], length+wp.min(wp.max(radial, axial), 0.))


@wp.func
def field(p: wp.vec3, kind: int, dimensions: wp.vec4):
    # dimensions = (plug/inner radius, socket outer radius, half Y length, bore radius)
    outer_radius = dimensions[0] if kind == 1 else dimensions[1]
    value = cylinder_y(p, outer_radius, dimensions[2])
    if kind == 2:
        r = wp.sqrt(p[0]*p[0]+p[2]*p[2])
        cavity = dimensions[0]-r
        if cavity > value[3]:
            n = wp.vec3(-1., 0., 0.)
            if r > 1.e-12:
                n = wp.vec3(-p[0]/r, 0., -p[2]/r)
            value = wp.vec4(n[0], n[1], n[2], cavity)
    r = wp.sqrt(p[0]*p[0]+p[1]*p[1])
    bore = dimensions[3]-r
    if bore > value[3]:
        n = wp.vec3(-1., 0., 0.)
        if r > 1.e-12:
            n = wp.vec3(-p[0]/r, -p[1]/r, 0.)
        value = wp.vec4(n[0], n[1], n[2], bore)
    return value


@wp.kernel
def inspect(points: wp.array[wp.vec3], kind: int, dimensions: wp.vec4,
            values: wp.array[wp.vec4]):
    i = wp.tid()
    values[i] = field(points[i], kind, dimensions)


@wp.kernel
def inside(points: wp.array[wp.vec3], transform: wp.transform, kind: int,
           dimensions: wp.vec4, result: wp.array[int]):
    i = wp.tid()
    p = wp.transform_point(wp.transform_inverse(transform), points[i])
    if field(p, kind, dimensions)[3] < 0.:
        result[i] = 1
