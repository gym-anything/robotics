"""Coupled pressure / implicit interface-force iteration.

Uses the end-position interface detection and start-position force gradients
of Probst & Teschner (2025), equations 20--25. Pressure is the existing
constrained projection; the two blocks iterate to a measured force residual.
This is a block iteration, not a claim to reproduce their Jacobi/NNCG solver.
Rigid boundary transport is still first-order partitioned, not monolithic.
Only the accepted velocity is integrated by the ordinary water step.
"""

import numpy as np
import warp as wp
from .interface_energy import fluid_interface, sampled_boundary_interface, forces
from .sph_kernel_functions import cubic_grad
from .sph_kernels import integrate


@wp.kernel
def relaxation_bound(grid: wp.uint64, bg: wp.uint64, q: wp.array[wp.vec3],
        b: wp.array[wp.vec3], vf: float, vb: float, h: float, area: float,
        sigma: float, sv: wp.array[float], sl: wp.array[float], mass: wp.array[float],
        dt: float, values: wp.array[float]):
    i = wp.tid()
    gtotal = wp.vec3(0.)
    curvature = float(0.)
    for j in wp.hash_grid_query(grid, q[i], h):
        r = q[i]-q[j]
        if wp.length_sq(r) < h*h:
            g = vf*cubic_grad(r, h)
            gtotal += g
            curvature += sigma*wp.length_sq(g)
    for j in wp.hash_grid_query(bg, q[i], h):
        r = q[i]-b[j]
        if wp.length_sq(r) < h*h:
            g = cubic_grad(r, h)
            gtotal += vb*g
            curvature += vf*vb*(sv[j]+sl[j])*wp.length_sq(g)
    # psi''(C) <= 1/epsilon = 20. The common scalar relaxation preserves
    # equal-opposite surface forces even before exact convergence.
    curvature = 20.*area*(curvature+sigma*wp.length_sq(gtotal))
    values[i] = 1./(1.+dt*dt*curvature/(3.*mass[i]))


@wp.kernel
def add_force(v: wp.array[wp.vec3], f: wp.array[wp.vec3], mass: wp.array[float],
        dt: float, out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = v[i]+dt*f[i]/mass[i]


@wp.kernel
def relax(target: wp.array[wp.vec3], alpha: float, current: wp.array[wp.vec3]):
    i = wp.tid()
    current[i] += alpha*(target[i]-current[i])


@wp.kernel
def error(current: wp.array[wp.vec3], target: wp.array[wp.vec3], mass: wp.array[float],
        dt: float, spacing: float, result: wp.array[float]):
    i = wp.tid()
    result[i] = dt*dt*wp.length(current[i]-target[i])/(mass[i]*spacing)


class Workspace:
    def __init__(self, water):
        c = water.capillary
        self.trial_q = wp.empty_like(water.state.particle_q)
        self.trial_grid = wp.HashGrid(64, 64, 512, device=water.config.device)
        self.velocity = wp.empty_like(water.state.particle_qd)
        self.current_f = wp.empty_like(c.forces)
        self.target_f = wp.empty_like(c.forces)
        self.curvature = wp.empty_like(water.rho)
        self.residual = wp.empty_like(water.rho)
        self.boundary_pointer = None

    def boundaries(self, c):
        if self.boundary_pointer != c.boundary.ptr:
            self.current_b = wp.empty_like(c.boundary_forces)
            self.target_b = wp.empty_like(c.boundary_forces)
            self.trial_b = wp.empty_like(c.boundary)
            self.trial_bg = wp.HashGrid(64, 64, 512, device=c.model.device)
            self.boundary_pointer = c.boundary.ptr


def project(w, unconstrained):
    if w.boundaries is None:
        raise ValueError('Implicit interface currently requires the explicit apparatus-boundary adapter')
    if not hasattr(w, 'implicit_workspace'):
        w.implicit_workspace = Workspace(w)
    s, c, cfg = w.implicit_workspace, w.capillary, w.config
    s.boundaries(c)
    # Seed only the force iterate. predict() does not advance physical state.
    c.predict(w.next_state, cfg.dt_s)
    s.current_f.assign(c.forces)
    s.current_b.assign(c.boundary_forces)
    wp.launch(integrate, dim=len(c.boundary), inputs=[c.boundary, c.solid_samples.v, cfg.dt_s],
        outputs=[s.trial_b], device=cfg.device)
    s.trial_bg.build(s.trial_b, c.support_m)
    w._launch(relaxation_bound, [c.grid.id, c.bg.id, w.state.particle_q, c.boundary,
        c.volume, c.boundary_volume, c.support_m, c.area, c.sigma_n_m, c.sv, c.sl,
        w.model.particle_mass, cfg.dt_s], [s.curvature])
    alpha = .5*float(s.curvature.numpy().min())
    for iteration in range(cfg.implicit_surface_iterations):
        w._launch(add_force, [unconstrained, s.current_f, w.model.particle_mass, cfg.dt_s], [s.velocity])
        if cfg.nonlinear_density:
            from .nonlinear_density import project as project_density
            candidate = project_density(w, s.velocity, react=False)
        else:
            candidate = w._project(s.velocity, True, react=False)
        w._launch(integrate, [w.state.particle_q, candidate, cfg.dt_s], [s.trial_q])
        s.trial_grid.build(s.trial_q, c.support_m)
        w._launch(fluid_interface, [s.trial_grid.id, s.trial_bg.id, s.trial_q, s.trial_b,
            c.volume, c.boundary_volume, c.support_m, 1, int(c.ordered_threads)], [c.fv, c.fb, c.ef])
        wp.launch(sampled_boundary_interface, dim=len(c.boundary),
            inputs=[s.trial_grid.id, s.trial_bg.id, s.trial_q, s.trial_b,
                c.volume, c.boundary_volume, c.support_m, int(c.ordered_threads)], outputs=[c.bv, c.bf, c.eb], device=cfg.device)
        s.target_b.zero_()
        # Keep spatial force gradients at step start, as in paper Eq. 22.
        # Only interface weights use the pressure-corrected predicted state.
        w._launch(forces, [c.grid.id, c.bg.id, w.state.particle_q, c.boundary,
            c.fv, c.fb, c.bv, c.bf, c.volume, c.boundary_volume, c.support_m,
            c.area, c.sigma_n_m, c.sv, c.sl, 1, int(c.ordered_threads), w.model.particle_mass,
            unconstrained, cfg.dt_s], [c.predicted.particle_qd, s.target_f, s.target_b])
        w._launch(error, [s.current_f, s.target_f, w.model.particle_mass, cfg.dt_s, cfg.spacing_m], [s.residual])
        residual = s.residual.numpy()
        info = {'iterations': iteration+1, 'maximum_displacement_residual_fraction': float(residual.max()),
            'mean_displacement_residual_fraction': float(residual.mean()), 'common_relaxation': alpha}
        w.last_projection['implicit_surface'] = info
        if not np.isfinite(residual).all():
            raise FloatingPointError('Nonfinite coupled surface-force residual')
        if residual.max() <= cfg.implicit_surface_error:
            c.forces.assign(s.current_f)
            c.boundary_forces.assign(s.current_b)
            w.boundaries.pressure_reaction(w)
            return candidate
        w._launch(relax, [s.target_f, alpha], [s.current_f])
        wp.launch(relax, dim=len(c.boundary), inputs=[s.target_b, alpha], outputs=[s.current_b], device=cfg.device)
    raise FloatingPointError(f'Coupled implicit surface force did not converge: {info}')
