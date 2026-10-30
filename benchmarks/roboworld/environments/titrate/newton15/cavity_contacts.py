"""Finite-step convex cavity constraints for analytic cylindrical passages.

An interior cylinder is a disk constraint in the velocity plane normal to its
axis. A tangent half-space alone permits O(dt^2 * v_tangent^2 / radius)
penetration. Project onto the actual predicted disk while retaining all other
contact half-spaces. Dykstra's algorithm computes the Euclidean projection of
their convex intersection, rather than successively applying untracked clamps.
Only velocity impulses change; positions are integrated once by the solver.
Boundary motion follows the existing first-order partitioned body velocity.
"""

import warp as wp
from .analytic_solids import cylinder_y

Corrections = wp.types.matrix(shape=(8, 3), dtype=wp.float64)


@wp.kernel
def build(count: wp.array[int], particles: wp.array[int], shapes: wp.array[int],
          q: wp.array[wp.vec3], body_q: wp.array[wp.transform], body_v: wp.array[wp.spatial_vector],
          body_com: wp.array[wp.vec3], shape_body: wp.array[int], shape_transform: wp.array[wp.transform],
          kinds: wp.array[int], dimensions: wp.array[wp.vec4], dt: float, skin: float,
          axis: wp.array[wp.vec3], offset: wp.array[wp.vec3], radius: wp.array[float],
          has_cavity: wp.array[int]):
    j = wp.tid()
    if j >= count[0]:
        return
    radius[j] = 0.
    shape, i = shapes[j], particles[j]
    kind = kinds[shape]
    if kind == 0:
        return
    body = shape_body[shape]
    pose = wp.transform_identity()
    if body >= 0:
        pose = body_q[body]
    world = wp.transform_multiply(pose, shape_transform[shape])
    p = wp.transform_point(wp.transform_inverse(world), q[i])
    d = dimensions[shape]
    outside = cylinder_y(p, d[0] if kind == 1 else d[1], d[2])[3]
    bore = d[3]-wp.sqrt(p[0]*p[0]+p[1]*p[1])
    inner = float(-1.e10)
    if kind == 2:
        inner = d[0]-wp.sqrt(p[0]*p[0]+p[2]*p[2])
    direction = wp.vec3(0.)
    r = float(0.)
    if bore > outside and bore >= inner:
        direction, r = wp.vec3(0., 0., 1.), d[3]-skin
    elif inner > outside:
        direction, r = wp.vec3(0., 1., 0.), d[0]-skin
    if r <= 0.:
        return
    velocity = wp.vec3(0.)
    if body >= 0:
        com = wp.transform_point(pose, body_com[body])
        velocity = wp.spatial_top(body_v[body])+wp.cross(wp.spatial_bottom(body_v[body]), q[i]-com)
    axis[j] = wp.normalize(wp.transform_vector(world, direction))
    offset[j] = q[i]-wp.transform_get_translation(world)-dt*velocity
    radius[j] = r
    has_cavity[i] = 1


@wp.func
def project_set(value: wp.vec3d, j: int, normals: wp.array[wp.vec3], lower: wp.array[float],
                axes: wp.array[wp.vec3], offsets: wp.array[wp.vec3], radii: wp.array[float], dt: wp.float64):
    r = wp.float64(radii[j])
    if r > 0.:
        axis = wp.vec3d(axes[j])
        end = wp.vec3d(offsets[j])+dt*value
        radial = end-wp.dot(end, axis)/wp.length_sq(axis)*axis
        length = wp.length(radial)
        if length > r:
            return value-(wp.float64(1.)-r/length)*radial/dt
        return value
    n = wp.vec3d(normals[j])
    return value+wp.max(wp.float64(lower[j])-wp.dot(value, n), wp.float64(0.))/wp.length_sq(n)*n


@wp.func
def project_one(i: int, ids: wp.array2d[int], sizes: wp.array[int], has_cavity: wp.array[int],
            normals: wp.array[wp.vec3], lower: wp.array[float], axes: wp.array[wp.vec3],
            offsets: wp.array[wp.vec3], radii: wp.array[float], velocity: wp.array[wp.vec3],
            dt: float, active: wp.array[int], dv: wp.array[wp.vec3], increments: wp.array[wp.vec3]):
    if active[0] == 0 or has_cavity[i] == 0:
        return
    value = wp.vec3d(velocity[i])+wp.vec3d(dv[i])
    corrections = Corrections(wp.float64(0.))
    for iteration in range(128):
        previous = value
        for k in range(sizes[i]):
            j = ids[k, i]
            old = wp.vec3d(corrections[k, 0], corrections[k, 1], corrections[k, 2])
            trial = value+old
            value = project_set(trial, j, normals, lower, axes, offsets, radii, wp.float64(dt))
            change = trial-value
            corrections[k, 0] = change[0]
            corrections[k, 1] = change[1]
            corrections[k, 2] = change[2]
        if wp.length(value-previous) < wp.float64(1.e-9):
            feasible = bool(True)
            for k in range(sizes[i]):
                j = ids[k, i]
                checked = project_set(value, j, normals, lower, axes, offsets, radii, wp.float64(dt))
                if wp.length(checked-value) > wp.float64(1.e-8):
                    feasible = False
            if feasible:
                break
    dv[i] = wp.vec3(value-wp.vec3d(velocity[i]))
    for k in range(sizes[i]):
        j = ids[k, i]
        # The sum of these per-surface corrections is exactly the total
        # contact change (up to output float rounding), preserving reactions.
        increments[j] = -wp.vec3(wp.float32(corrections[k, 0]),
            wp.float32(corrections[k, 1]), wp.float32(corrections[k, 2]))


@wp.kernel
def project(ids: wp.array2d[int], sizes: wp.array[int], has_cavity: wp.array[int],
            normals: wp.array[wp.vec3], lower: wp.array[float], axes: wp.array[wp.vec3],
            offsets: wp.array[wp.vec3], radii: wp.array[float], velocity: wp.array[wp.vec3],
            dt: float, active: wp.array[int], dv: wp.array[wp.vec3], increments: wp.array[wp.vec3]):
    project_one(wp.tid(), ids, sizes, has_cavity, normals, lower, axes,
                offsets, radii, velocity, dt, active, dv, increments)


@wp.kernel
def compact(has_cavity: wp.array[int], count: wp.array[int], particles: wp.array[int]):
    i = wp.tid()
    if has_cavity[i] != 0:
        particles[wp.atomic_add(count, 0, 1)] = i


@wp.kernel
def project_sparse(count: wp.array[int], particles: wp.array[int], workers: int,
            ids: wp.array2d[int], sizes: wp.array[int], has_cavity: wp.array[int],
            normals: wp.array[wp.vec3], lower: wp.array[float], axes: wp.array[wp.vec3],
            offsets: wp.array[wp.vec3], radii: wp.array[float], velocity: wp.array[wp.vec3],
            dt: float, active: wp.array[int], dv: wp.array[wp.vec3], increments: wp.array[wp.vec3]):
    # The compact list contains EVERY active cavity particle. A grid-stride
    # traversal changes launch occupancy, not contact capacity or arithmetic.
    for slot in range(wp.tid(), count[0], workers):
        project_one(particles[slot], ids, sizes, has_cavity, normals, lower,
                    axes, offsets, radii, velocity, dt, active, dv, increments)


@wp.kernel
def guard(ids: wp.array2d[int], sizes: wp.array[int], has_cavity: wp.array[int],
          axes: wp.array[wp.vec3], offsets: wp.array[wp.vec3], radii: wp.array[float],
          velocity: wp.array[wp.vec3], dt: float, failures: wp.array[int]):
    i = wp.tid()
    if has_cavity[i] == 0:
        return
    for k in range(sizes[i]):
        j = ids[k, i]
        if radii[j] > 0.:
            axis = wp.vec3d(axes[j])
            end = wp.vec3d(offsets[j])+wp.float64(dt)*wp.vec3d(velocity[i])
            radial = end-wp.dot(end, axis)/wp.length_sq(axis)*axis
            if wp.length(radial)-wp.float64(radii[j]) > wp.float64(1.e-7):
                wp.atomic_add(failures, 2, 1)
