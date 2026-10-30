"""Newton 1.5 exact particle/shape queries with unilateral liquid contact.

Volume quadrature supplies pressure, but does not by itself forbid centres
inside a thin wall. Newton's actual mesh/primitive queries supply nonpenetration
constraints. Their velocity impulses are returned to the contacted rigid body.
No point is teleported or position-corrected. Candidate/constraint overflows fail.
"""

import numpy as np
import warp as wp
from .runtime import require_newton


@wp.kernel
def candidate_pairs(shapes: wp.array[int], pairs: wp.array[wp.vec2i]):
    particle, j = wp.tid()
    pairs[particle*shapes.shape[0]+j] = wp.vec2i(particle, shapes[j])


@wp.kernel
def index_contacts(count: wp.array[int], particles: wp.array[int], shapes: wp.array[int],
                   local_points: wp.array[wp.vec3], normal: wp.array[wp.vec3], shape_body: wp.array[int],
                   body_q: wp.array[wp.transform], body_v: wp.array[wp.spatial_vector], com: wp.array[wp.vec3],
                   q: wp.array[wp.vec3], radius: float, dt: float,
                   ids: wp.array2d[int], sizes: wp.array[int], overflow: wp.array[int],
                   points: wp.array[wp.vec3], body_ids: wp.array[int], lower: wp.array[float]):
    j = wp.tid()
    if j >= count[0]:
        return
    i, b = particles[j], shape_body[shapes[j]]
    slot = wp.atomic_add(sizes, i, 1)
    if slot >= ids.shape[0]:
        wp.atomic_max(overflow, 0, slot+1)
        return
    ids[slot, i] = j
    point = wp.vec3(local_points[j])
    velocity = wp.vec3(0.)
    if b >= 0:
        point = wp.transform_point(body_q[b], point)
        velocity = wp.spatial_top(body_v[b])+wp.cross(wp.spatial_bottom(body_v[b]), point-wp.transform_point(body_q[b], com[b]))
    distance = wp.dot(q[i]-point, normal[j])
    lower[j] = (radius-distance)/dt+wp.dot(velocity, normal[j])
    points[j] = point
    body_ids[j] = b


@wp.kernel
def project_sweeps(ids: wp.array2d[int], sizes: wp.array[int], normals: wp.array[wp.vec3],
            lower: wp.array[float], v: wp.array[wp.vec3], active: wp.array[int],
            dv: wp.array[wp.vec3], increments: wp.array[wp.vec3]):
    i = wp.tid()
    if active[0] == 0:
        return
    value = v[i]+dv[i]
    for k in range(sizes[i]):
        increments[ids[k, i]] = wp.vec3(0.)
    # Intersect up to eight local half-spaces. Usually one glass face is
    # involved; repeated projections also cover inside corners.
    for iteration in range(8):
        for k in range(sizes[i]):
            j = ids[k, i]
            change = wp.max(lower[j]-wp.dot(value, normals[j]), 0.)*normals[j]
            value += change
            increments[j] += change
    dv[i] = value-v[i]


@wp.func
def feasible(value: wp.vec3d, i: int, ids: wp.array2d[int], sizes: wp.array[int],
             normals: wp.array[wp.vec3], lower: wp.array[float]):
    valid = bool(True)
    for k in range(sizes[i]):
        j = ids[k, i]
        if wp.dot(value, wp.vec3d(normals[j])) < wp.float64(lower[j])-wp.float64(1.e-8):
            valid = False
    return valid


@wp.kernel
def project(ids: wp.array2d[int], sizes: wp.array[int], normals: wp.array[wp.vec3],
            lower: wp.array[float], v: wp.array[wp.vec3], active: wp.array[int],
            dv: wp.array[wp.vec3], increments: wp.array[wp.vec3], has_cavity: wp.array[int]):
    """Exact 3D closest velocity in the local contact half-spaces.

    In three dimensions a nondegenerate optimum has at most three active
    planes. Enumerating them avoids slow alternating projection between the
    nearly opposed flask-floor and descending stir-bar normals.
    """
    i = wp.tid()
    if active[0] == 0 or sizes[i] == 0 or has_cavity[i] != 0:
        return
    # Almost parallel faces at the column/socket seam make the small Gram
    # system ill-conditioned in float32. Solve that local system in float64;
    # keep the simulator state and emitted impulses in their original dtype.
    initial = wp.vec3d(v[i])+wp.vec3d(dv[i])
    for k in range(sizes[i]):
        increments[ids[k, i]] = wp.vec3(0.)
    if feasible(initial, i, ids, sizes, normals, lower):
        return
    best, cost = wp.vec3d(initial), wp.float64(1.e30)
    best_ids, best_multipliers = wp.vec3i(-1), wp.vec3d(0.)
    for a in range(sizes[i]):
        ia = ids[a, i]
        na = wp.vec3d(normals[ia])
        aa = wp.dot(na, na)
        ra = wp.float64(lower[ia])-wp.dot(initial, na)
        if aa > 1.e-12:
            la = ra/aa
            candidate = initial+la*na
            distance = wp.length_sq(candidate-initial)
            if la >= 0. and distance < cost and feasible(candidate, i, ids, sizes, normals, lower):
                best, cost = candidate, distance
                best_ids = wp.vec3i(ia, -1, -1)
                best_multipliers = wp.vec3d(la, wp.float64(0.), wp.float64(0.))
        for b in range(a+1, sizes[i]):
            ib = ids[b, i]
            nb = wp.vec3d(normals[ib])
            bb, ab = wp.dot(nb, nb), wp.dot(na, nb)
            rb = wp.float64(lower[ib])-wp.dot(initial, nb)
            determinant = aa*bb-ab*ab
            if determinant > 1.e-10:
                la = (ra*bb-rb*ab)/determinant
                lb = (rb*aa-ra*ab)/determinant
                candidate = initial+la*na+lb*nb
                distance = wp.length_sq(candidate-initial)
                if la >= 0. and lb >= 0. and distance < cost and feasible(candidate, i, ids, sizes, normals, lower):
                    best, cost = candidate, distance
                    best_ids = wp.vec3i(ia, ib, -1)
                    best_multipliers = wp.vec3d(la, lb, wp.float64(0.))
            for c in range(b+1, sizes[i]):
                ic = ids[c, i]
                nc = wp.vec3d(normals[ic])
                cc, ac, bc = wp.dot(nc, nc), wp.dot(na, nc), wp.dot(nb, nc)
                matrix = wp.mat33d(aa, ab, ac, ab, bb, bc, ac, bc, cc)
                if wp.determinant(matrix) > 1.e-10:
                    lambdas = wp.inverse(matrix)*wp.vec3d(ra, rb, wp.float64(lower[ic])-wp.dot(initial, nc))
                    candidate = initial+lambdas[0]*na+lambdas[1]*nb+lambdas[2]*nc
                    distance = wp.length_sq(candidate-initial)
                    if wp.min(lambdas) >= 0. and distance < cost and feasible(candidate, i, ids, sizes, normals, lower):
                        best, cost = candidate, distance
                        best_ids = wp.vec3i(ia, ib, ic)
                        best_multipliers = lambdas
    if cost < 1.e30:
        dv[i] = wp.vec3(best-wp.vec3d(v[i]))
        for k in range(3):
            if best_ids[k] >= 0:
                increments[best_ids[k]] = wp.float32(best_multipliers[k])*normals[best_ids[k]]
    # No feasible active set leaves the velocity unchanged; the independent
    # integration guard rejects its residual instead of injecting a velocity.


@wp.kernel
def react(count: wp.array[int], ids: wp.array[int], points: wp.array[wp.vec3],
          increments: wp.array[wp.vec3], mass: float, dt: float,
          body_q: wp.array[wp.transform], com: wp.array[wp.vec3], forces: wp.array[wp.spatial_vector]):
    j = wp.tid()
    if j >= count[0]:
        return
    body = ids[j]
    if body >= 0:
        force = -mass/dt*increments[j]
        arm = points[j]-wp.transform_point(body_q[body], com[body])
        wp.atomic_add(forces, body, wp.spatial_vector(force, wp.cross(arm, force)))


@wp.kernel
def integration_guard(velocity: wp.array[wp.vec3], dt: float, support: float,
                      ids: wp.array2d[int], sizes: wp.array[int], normals: wp.array[wp.vec3],
                      lower: wp.array[float], cavity_radii: wp.array[float], failures: wp.array[int]):
    i = wp.tid()
    v = velocity[i]
    if not wp.isfinite(v[0]) or not wp.isfinite(v[1]) or not wp.isfinite(v[2]):
        wp.atomic_add(failures, 0, 1)
    # Neighbour and exact-contact candidates were evaluated at step start.
    # Reject a step that crosses more than half the pressure support, rather
    # than integrate through an unqueried wall or silently clip its velocity.
    if dt*wp.length(v) > .5*support:
        wp.atomic_add(failures, 1, 1)
    for k in range(sizes[i]):
        j = ids[k, i]
        if cavity_radii[j] <= 0. and dt*(lower[j]-wp.dot(v, normals[j])) > 1.e-7:
            wp.atomic_add(failures, 2, 1)


class FluidContacts:
    def __init__(self, system, capacity_per_particle=8):
        newton, _ = require_newton()
        self.system = system
        w, m = system.water, system.model
        self.capacity = w.model.particle_count*capacity_per_particle
        self.contacts = newton.Contacts(rigid_contact_max=0, soft_contact_max=self.capacity,
            soft_contact_tids_size=0, device=m.device)
        self.ids = wp.empty((capacity_per_particle, w.model.particle_count), dtype=int, device=m.device)
        self.sizes = wp.zeros(w.model.particle_count, dtype=int, device=m.device)
        self.overflow = wp.zeros(1, dtype=int, device=m.device)
        self.points = wp.empty(self.capacity, dtype=wp.vec3, device=m.device)
        self.body_ids = wp.empty(self.capacity, dtype=int, device=m.device)
        self.lower = wp.empty(self.capacity, dtype=float, device=m.device)
        self.increments = wp.zeros(self.capacity, dtype=wp.vec3, device=m.device)
        self.cavity_axes = wp.empty(self.capacity, dtype=wp.vec3, device=m.device)
        self.cavity_offsets = wp.empty(self.capacity, dtype=wp.vec3, device=m.device)
        self.cavity_radii = wp.zeros(self.capacity, dtype=float, device=m.device)
        self.has_cavity = wp.zeros(w.model.particle_count, dtype=int, device=m.device)
        self.cavity_count = wp.zeros(1, dtype=int, device=m.device)
        self.cavity_particles = wp.empty(w.model.particle_count, dtype=int, device=m.device)
        self.cavity_workers = min(w.model.particle_count, 4096)
        self.generation = None
        self.guard_failures = wp.zeros(3, dtype=int, device=m.device)
        from .fluid_broadphase import FluidBroadphase
        self.broadphase = FluidBroadphase(system)

    def update(self, query_positions=None):
        from newton._src.geometry.kernels import create_soft_contacts
        from .mesh_contacts import create_mesh_contacts
        newton, _ = require_newton()
        s = self.system
        w, m, c = s.water, s.model, self.contacts
        query_positions = w.state.particle_q if query_positions is None else query_positions
        self.pairs, self.mesh_pairs = self.broadphase.update()
        c.soft_contact_count.zero_()
        self.sizes.zero_()
        self.overflow.zero_()
        wp.launch(create_soft_contacts, dim=len(self.pairs), inputs=[self.pairs, query_positions,
            w.model.particle_radius, w.model.particle_flags, w.model.particle_world, s.state.body_q,
            m.shape_transform, m.shape_body, m.shape_type, m.shape_scale, m.shape_source_ptr,
            m._shape_mesh_properties, m.shape_world, 2*w.config.spacing_m, m.shape_margin,
            self.capacity, m.shape_flags, m.shape_heightfield_index, m.heightfield_data, m.heightfield_elevations],
            outputs=[c.soft_contact_count, c.soft_contact_particle, c.soft_contact_indices, c.soft_contact_barycentric,
                c.soft_contact_shape, c.soft_contact_body_pos, c.soft_contact_body_vel, c.soft_contact_normal, c.soft_contact_tids],
            device=m.device)
        wp.launch(create_mesh_contacts, dim=len(self.mesh_pairs), inputs=[self.mesh_pairs, query_positions,
            w.model.particle_radius, w.model.particle_flags, w.model.particle_world,
            m.shape_flags, m.shape_world, m._shape_mesh_properties, s.state.body_q,
            m.shape_transform, m.shape_body, m.shape_scale,
            s.samples.mesh_query_ids, m.shape_margin, s.samples.profile_data, s.samples.profile_layout,
            s.samples.analytic_kinds, s.samples.analytic_dimensions,
            2*w.config.spacing_m, self.capacity],
            outputs=[c.soft_contact_count, c.soft_contact_particle, c.soft_contact_shape,
                c.soft_contact_indices, c.soft_contact_barycentric, c.soft_contact_body_pos,
                c.soft_contact_body_vel, c.soft_contact_normal], device=m.device)
        count = int(c.soft_contact_count.numpy()[0])
        if count > self.capacity:
            raise ValueError("Exact fluid contact capacity exceeded; no contact silently dropped")
        wp.launch(index_contacts, dim=count, inputs=[c.soft_contact_count, c.soft_contact_particle, c.soft_contact_shape,
            c.soft_contact_body_pos, c.soft_contact_normal, m.shape_body, s.state.body_q, s.state.body_qd, m.body_com,
            w.state.particle_q, w.config.spacing_m*w.config.wall_clearance_fraction, w.config.dt_s],
            outputs=[self.ids, self.sizes, self.overflow, self.points, self.body_ids, self.lower], device=m.device)
        if self.overflow.numpy()[0]:
            raise ValueError("Per-particle contact capacity exceeded; no contact silently dropped")
        self.count = count
        from .cavity_contacts import build
        self.has_cavity.zero_()
        wp.launch(build, dim=count, inputs=[c.soft_contact_count, c.soft_contact_particle,
            c.soft_contact_shape, w.state.particle_q, s.state.body_q, s.state.body_qd,
            m.body_com, m.shape_body, m.shape_transform, s.samples.analytic_kinds,
            s.samples.analytic_dimensions, w.config.dt_s, w.config.spacing_m*w.config.wall_clearance_fraction],
            outputs=[self.cavity_axes, self.cavity_offsets, self.cavity_radii, self.has_cavity], device=m.device)
        from .cavity_contacts import compact
        self.cavity_count.zero_()
        wp.launch(compact, dim=w.model.particle_count, inputs=[self.has_cavity],
            outputs=[self.cavity_count, self.cavity_particles], device=m.device)

    def project(self, water, velocity, active):
        water._launch(project, [self.ids, self.sizes, self.contacts.soft_contact_normal,
            self.lower, velocity, active], [water.dv, self.increments, self.has_cavity])
        from .cavity_contacts import project_sparse
        wp.launch(project_sparse, dim=self.cavity_workers, inputs=[self.cavity_count,
            self.cavity_particles, self.cavity_workers, self.ids, self.sizes, self.has_cavity,
            self.contacts.soft_contact_normal, self.lower, self.cavity_axes, self.cavity_offsets,
            self.cavity_radii, velocity, water.config.dt_s, active],
            outputs=[water.dv, self.increments], device=water.config.device)

    def reaction(self, water):
        s = self.system
        wp.launch(react, dim=self.count, inputs=[self.contacts.soft_contact_count, self.body_ids,
            self.points, self.increments, water.mass_per_particle_kg, water.config.dt_s, s.state.body_q,
            s.model.body_com], outputs=[s.forces], device=s.model.device)

    def validate_integration(self, water, velocity):
        self.guard_failures.zero_()
        water._launch(integration_guard, [velocity, water.config.dt_s, water.config.support_m,
            self.ids, self.sizes, self.contacts.soft_contact_normal, self.lower,
            self.cavity_radii], [self.guard_failures])
        from .cavity_contacts import guard
        water._launch(guard, [self.ids, self.sizes, self.has_cavity, self.cavity_axes,
            self.cavity_offsets, self.cavity_radii, velocity, water.config.dt_s], [self.guard_failures])
        failures = self.guard_failures.numpy()
        if failures.any():
            raise FloatingPointError(f"Liquid step rejected before integration: nonfinite={failures[0]}, "
                f"half-support travel violations={failures[1]}, contact residuals over 0.1 micrometre={failures[2]}")
