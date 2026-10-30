"""Lagged reaction coupling using Newton's exact Style3D contact-force function.

Rigid integration consumes the preceding tick's contact wrenches; Style3D then
solves against that finite-mass robot motion. This is partitioned, not monolithic
coupling. Forces are sampled at the final nonlinear contact evaluation, not
summed over solver iterations. No cloth attachments, grasp flags or pose writes.
"""
import warp as wp
from newton._src.solvers.vbd.rigid_vbd_kernels import evaluate_body_particle_contact


@wp.kernel
def collect_reactions(dt: float, prev: wp.array[wp.vec3], pos: wp.array[wp.vec3],
    ke: float, kd: float, mu: float, eps: float, radius: wp.array[float],
    particle: wp.array[int], count: wp.array[int], capacity: int,
    shape_mu: wp.array[float], shape_body: wp.array[int], body_q: wp.array[wp.transform],
    body_prev: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], com: wp.array[wp.vec3],
    shape: wp.array[int], local_point: wp.array[wp.vec3], local_velocity: wp.array[wp.vec3],
    normal: wp.array[wp.vec3], margin: wp.array[float],
    wrench: wp.array[wp.spatial_vector], forces: wp.array[wp.vec3]):
    i = wp.tid()
    if i >= wp.min(count[0], capacity):
        return
    p = particle[i]
    f, h = evaluate_body_particle_contact(p, pos[p], prev[p], i, ke, kd, mu, eps,
        radius, shape_mu, shape_body, body_q, body_prev, body_qd, com, shape,
        local_point, local_velocity, normal, margin, dt)
    forces[i] = f
    b = shape_body[shape[i]]
    if b >= 0:
        point = wp.transform_point(body_q[b], local_point[i])
        center = wp.transform_point(body_q[b], com[b])
        wp.atomic_add(wrench, b, wp.spatial_vector(-f, wp.cross(point-center, -f)))


class ReactionCollision:
    """Delegate all native/filtered self-contact; observe its rigid contact law."""
    def __init__(self, native, simulation):
        self.native, self.simulation = native, simulation
        self.native.body_contact_max = simulation.contacts.soft_contact_max
        self.wrenches = wp.zeros(simulation.model.body_count, dtype=wp.spatial_vector,
                                 device=simulation.config.device)
        self.forces = wp.zeros(simulation.contacts.soft_contact_max, dtype=wp.vec3,
                              device=simulation.config.device)

    def __getattr__(self, name):
        return getattr(self.native, name)

    def accumulate_contact_force(self, dt, iteration, state_in, state_out, contacts,
                                  particle_forces, particle_q_prev, particle_stiff=None):
        self.native.accumulate_contact_force(dt, iteration, state_in, state_out,
            contacts, particle_forces, particle_q_prev, particle_stiff)
        if iteration != self.simulation.config.nonlinear_iterations-1:
            return
        m = self.simulation.model
        self.wrenches.zero_()
        self.forces.zero_()
        wp.launch(collect_reactions, dim=contacts.soft_contact_max, inputs=[
            dt, particle_q_prev, state_in.particle_q, m.soft_contact_ke, m.soft_contact_kd,
            m.soft_contact_mu, self.native.friction_epsilon, m.particle_radius,
            contacts.soft_contact_particle, contacts.soft_contact_count, contacts.soft_contact_max,
            m.shape_material_mu, m.shape_body, state_out.body_q, state_in.body_q,
            m.body_qd, m.body_com, contacts.soft_contact_shape, contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel, contacts.soft_contact_normal, m.shape_margin],
            outputs=[self.wrenches, self.forces], device=self.simulation.config.device)
