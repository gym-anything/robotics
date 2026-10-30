"""Partitioned Newton rigid / repository SPH volume-boundary coupling.

Boundary pressure and physical viscous stresses have equal-opposite rigid
reactions. Solids are sampled from their actual hollow collision geometry;
the capillary predictor uses the same attached boundary-volume quadrature.
This first-order partitioned coupling is evaluated by assisted component loads.
It does not weld vessels, impose fluid velocities, or repair particle positions.
"""

import numpy as np
import warp as wp
from .sph_kernels import pressure_w, pressure_grad
from .solid_sampling import SolidSamples
from .runtime import require_newton


@wp.kernel
def prepare_boundary(fluid_gradient: wp.array[wp.vec3], bg: wp.uint64, q: wp.array[wp.vec3], bq: wp.array[wp.vec3],
                     bv: wp.array[wp.vec3], volume: float, fluid_volume: float, h: float, kind: int,
                     density: wp.array[float], factor: wp.array[float], bg_out: wp.array[wp.vec3],
                     boundary_rate: wp.array[float], neighbors: wp.array[int]):
    i = wp.tid()
    gf, gb = fluid_gradient[i], wp.vec3(0.)
    rho, rate = float(0.), float(0.)
    count = int(0)
    for j in wp.hash_grid_query(bg, q[i], h):
        r = q[i]-bq[j]
        if wp.length_sq(r) < h*h:
            count += 1
            g = volume*pressure_grad(r, h, kind)
            gb += g
            rho += volume*pressure_w(r, h, kind)
            rate += wp.dot(g, bv[j])
    old = float(0.)
    if factor[i] > 0.:
        old = 1./factor[i]
    denominator = old-wp.length_sq(gf)+wp.length_sq(gf+gb)
    factor[i] = 0.
    if denominator > 1.e-6:
        factor[i] = 1./denominator
    density[i] += rho
    bg_out[i] = gb
    boundary_rate[i] = rate
    # Equivalent-volume count preserves the same deficiency criterion when
    # solid quadrature alone is refined. A gradient threshold has units and
    # incorrectly activated the isolated parcel in pressure_row094.
    neighbors[i] += int(float(count)*volume/fluid_volume)


@wp.kernel
def pressure_reaction(bg: wp.uint64, q: wp.array[wp.vec3], bq: wp.array[wp.vec3],
                      p: wp.array[float], volume: float, mass: float, h: float, kind: int,
                      dt: float, reaction: wp.array[wp.vec3]):
    i = wp.tid()
    for j in wp.hash_grid_query(bg, q[i], h):
        r = q[i]-bq[j]
        if wp.length_sq(r) < h*h:
            f = mass/dt*p[i]*volume*pressure_grad(r, h, kind)
            wp.atomic_add(reaction, j, f)


@wp.kernel
def viscous_boundary(bg: wp.uint64, q: wp.array[wp.vec3], v: wp.array[wp.vec3],
                     bq: wp.array[wp.vec3], bv: wp.array[wp.vec3], volume: float, mass: float,
                     nu: float, h: float, kind: int, dt: float,
                     velocity: wp.array[wp.vec3], reaction: wp.array[wp.vec3]):
    i = wp.tid()
    change = wp.vec3(0.)
    for j in wp.hash_grid_query(bg, q[i], h):
        r = q[i]-bq[j]
        if wp.length_sq(r) < h*h:
            a = 10.*nu*volume*wp.dot(v[i]-bv[j], r)/(wp.length_sq(r)+.01*h*h)*pressure_grad(r, h, kind)
            change += dt*a
            wp.atomic_add(reaction, j, -mass*a)
    velocity[i] += change


@wp.kernel
def distribute(ids: wp.array[int], points: wp.array[wp.vec3], forces: wp.array[wp.vec3],
               body_q: wp.array[wp.transform], com: wp.array[wp.vec3], out: wp.array[wp.spatial_vector]):
    i = wp.tid()
    b = ids[i]
    if b >= 0:
        arm = points[i]-wp.transform_point(body_q[b], com[b])
        wp.atomic_add(out, b, wp.spatial_vector(forces[i], wp.cross(arm, forces[i])))


class SphSolidCoupling:
    """Advance the liquid and return wrenches to an existing Newton world.

    Does not create a second rigid solver or advance rigid state. The caller
    applies the returned loads in its one existing mechanics step.
    """

    def __init__(self, builder, model, water, contact_angles, *, shape_contact_angles=None, particle_regions=None):
        require_newton()
        if water.ground:
            raise ValueError("Solid coupling supplies its own boundaries; do not also use the plane fixture")
        if water.config.surface_model != "interface_energy":
            raise ValueError("The sampled boundary surface energy requires interface_energy")
        self.model, self.water = model, water
        self.state = None
        self.forces = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
        self.last_applied_body_forces = np.zeros((model.body_count, 6), dtype=np.float32)
        self.samples = SolidSamples(builder, model, water.config.spacing_m/water.config.boundary_refinement,
            shape_contact_angles=shape_contact_angles, coverage_support_m=water.capillary.support_m,
            maximum_points=water.config.maximum_boundary_samples, particle_regions=particle_regions)
        self.contact_angles = dict(contact_angles)
        self.reaction = None
        self.water.boundaries = self
        from .fluid_contacts import FluidContacts
        self.contacts = FluidContacts(self)
        from .rigid_reduction import RigidReduction
        self.rigid_reduction = RigidReduction(model.body_count, model.device)

    def prepare(self, water):
        s = self.samples
        water._launch(prepare_boundary, [water.fluid_gradient, s.grid.id, water.constraint_positions, s.q, s.v,
            s.spacing**3, water.volume_per_particle_m3, water.config.support_m, water.kernel_kind],
            [water.rho, water.factor, water.bg, water.boundary_rate, water.neighbors])

    def pressure_reaction(self, water):
        s = self.samples
        water._launch(pressure_reaction, [s.grid.id, water.constraint_positions, s.q, water.p,
            s.spacing**3, water.mass_per_particle_kg, water.config.support_m,
            water.kernel_kind, water.config.dt_s], [self.reaction])
        self.contacts.reaction(water)

    def contact_projection(self, water, velocity, active):
        self.contacts.project(water, velocity, active)

    def validate_integration(self, water, velocity):
        self.contacts.validate_integration(water, velocity)

    def viscosity(self, water, velocity):
        s = self.samples
        water._launch(viscous_boundary, [s.grid.id, water.state.particle_q, velocity, s.q, s.v,
            s.spacing**3, water.mass_per_particle_kg,
            water.material.viscosity_pa_s/water.material.density_kg_m3, water.config.support_m,
            water.kernel_kind, water.config.dt_s], [water.predicted_velocity, self.reaction])

    def advance(self, rigid_state):
        self.state = rigid_state
        w, s = self.water, self.samples
        s.update(self.state, w.state.particle_q)
        self.contacts.update()
        if self.reaction is None or self.reaction.shape != s.q.shape:
            self.reaction = wp.zeros_like(s.q)
        self.reaction.zero_()
        w.capillary.attach_boundaries(s, self.contact_angles)
        self.forces.zero_()
        w.step()
        self.rigid_reduction.prepare(s.body_ids)
        for forces in (self.reaction, w.capillary.boundary_forces):
            self.rigid_reduction.add(s.q, forces, self.state.body_q, self.model.body_com, self.forces)
        self.last_applied_body_forces = self.forces.numpy()
        return self.last_applied_body_forces


class SphSolidSystem(SphSolidCoupling):
    """Standalone component fixture; a single Newton rigid solver."""

    def __init__(self, builder, model, water, contact_angles, force_models=()):
        super().__init__(builder, model, water, contact_angles)
        newton, _ = require_newton()
        from .rigid_contact import create_solver
        self.state, self.next = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state)
        self.control = model.control()
        self.solver = create_solver(model)
        self.force_models = tuple(force_models)

    def step(self):
        self.advance(self.state)
        self.state.clear_forces()
        self.state.body_f.assign(self.forces)
        for model in self.force_models:
            model.apply(self.state, self.water.config.dt_s)
        self.last_applied_body_forces = self.state.body_f.numpy()
        self.solver.step(self.state, self.next, self.control, None, self.water.config.dt_s)
        if not np.isfinite(self.next.body_q.numpy()).all() or not np.isfinite(self.next.body_qd.numpy()).all():
            raise FloatingPointError("Nonfinite rigid state in coupled SPH step")
        self.state, self.next = self.next, self.state
