"""Explicit winding/normal Newton/Warp mesh queries for laboratory liquid.

Newton 1.5 selects ray parity for watertight meshes. The recorded bore-sign
fixture reproduces a false inside result and the resulting wrong-way impulse.
Normal averaging also fails near a narrow bore's symmetry axis. This kernel
selects generalized winding for closed meshes, normal sign for open meshes;
the two methods are checked separately against known cavity/solid points. Mesh vertices,
watertight metadata, physical dimensions and all material values are unchanged.
Primitive contacts continue to use Newton's original collision kernel.
"""

import warp as wp
from newton._src.geometry.flags import MeshProperties, ParticleFlags, ShapeFlags
from .revolved_queries import inside as inside_revolved
from .analytic_solids import field as analytic_field


@wp.kernel
def create_mesh_contacts(pairs: wp.array[wp.vec2i], particle_q: wp.array[wp.vec3],
        particle_radius: wp.array[float], particle_flags: wp.array[int], particle_world: wp.array[int],
        shape_flags: wp.array[int], shape_world: wp.array[int], mesh_properties: wp.array[int],
        body_q: wp.array[wp.transform],
        shape_transform: wp.array[wp.transform], shape_body: wp.array[int],
        shape_scale: wp.array[wp.vec3], shape_mesh: wp.array[wp.uint64], shape_margin: wp.array[float],
        profile_data: wp.array[wp.vec2], profile_layout: wp.array[wp.vec3i],
        analytic_kinds: wp.array[int], analytic_dimensions: wp.array[wp.vec4],
        margin: float, capacity: int,
        count: wp.array[int], particles: wp.array[int], shapes: wp.array[int],
        indices: wp.array[wp.vec3i], barycentric: wp.array[wp.vec3],
        body_position: wp.array[wp.vec3], body_velocity: wp.array[wp.vec3], normal: wp.array[wp.vec3]):
    k = wp.tid()
    i, shape = pairs[k][0], pairs[k][1]
    if (particle_flags[i] & int(ParticleFlags.ACTIVE)) == 0 or (shape_flags[shape] & int(ShapeFlags.COLLIDE_PARTICLES)) == 0:
        return
    if particle_world[i] != -1 and shape_world[shape] != -1 and particle_world[i] != shape_world[shape]:
        return
    body = shape_body[shape]
    transform = wp.transform_identity()
    if body >= 0:
        transform = body_q[body]
    local = shape_transform[shape]
    world = wp.transform_multiply(transform, local)
    x = wp.transform_point(wp.transform_inverse(world), particle_q[i])
    scale = shape_scale[shape]
    minimum_scale = wp.min(wp.abs(scale))
    extra = shape_margin[shape] if shape_margin.shape[0] > 0 else 0.
    if analytic_kinds[shape] != 0:
        # Authored analytic valve geometry, shared with solid quadrature.
        # No marching-cubes rounding gap becomes a hydraulic passage.
        value = analytic_field(x, analytic_kinds[shape], analytic_dimensions[shape])
        if value[3] >= margin+extra+particle_radius[i]:
            return
        n = wp.vec3(value[0], value[1], value[2])
        point = x-value[3]*n
        slot = wp.atomic_add(count, 0, 1)
        if slot < capacity:
            particles[slot] = i
            shapes[slot] = shape
            indices[slot] = wp.vec3i(i, -1, -1)
            barycentric[slot] = wp.vec3(1., 0., 0.)
            body_position[slot] = wp.transform_point(local, point)
            body_velocity[slot] = wp.vec3(0.)
            normal[slot] = wp.transform_vector(world, n)
        return
    specification = profile_layout[shape]
    if specification[1] > 0:
        query = wp.mesh_query_point_no_sign(shape_mesh[shape], wp.cw_div(x, scale),
            (margin+extra+particle_radius[i])/minimum_scale)
        query.sign = -1. if inside_revolved(wp.cw_div(x, scale), profile_data,
            specification[0], specification[1], specification[2]) else 1.
    elif mesh_properties[shape] & int(MeshProperties.WATERTIGHT):
        query = wp.mesh_query_point_sign_winding_number(shape_mesh[shape], wp.cw_div(x, scale),
            (margin+extra+particle_radius[i])/minimum_scale)
    else:
        query = wp.mesh_query_point_sign_normal(shape_mesh[shape], wp.cw_div(x, scale),
            (margin+extra+particle_radius[i])/minimum_scale)
    if not query.result:
        return
    point = wp.cw_mul(wp.mesh_eval_position(shape_mesh[shape], query.face, query.u, query.v), scale)
    delta = x-point
    distance = wp.length(delta)*query.sign
    if distance >= margin+extra+particle_radius[i]:
        return
    n = wp.normalize(delta)*query.sign
    if wp.length_sq(delta) < 1.e-20:
        a = wp.cw_mul(wp.mesh_eval_position(shape_mesh[shape], query.face, 1., 0.), scale)
        b = wp.cw_mul(wp.mesh_eval_position(shape_mesh[shape], query.face, 0., 1.), scale)
        c = wp.cw_mul(wp.mesh_eval_position(shape_mesh[shape], query.face, 0., 0.), scale)
        n = wp.normalize(wp.cross(b-a, c-a))
    slot = wp.atomic_add(count, 0, 1)
    if slot < capacity:
        particles[slot] = i
        shapes[slot] = shape
        indices[slot] = wp.vec3i(i, -1, -1)
        barycentric[slot] = wp.vec3(1., 0., 0.)
        body_position[slot] = wp.transform_point(local, point)
        body_velocity[slot] = wp.transform_vector(local,
            wp.cw_mul(wp.mesh_eval_velocity(shape_mesh[shape], query.face, query.u, query.v), scale))
        normal[slot] = wp.transform_vector(world, n)
