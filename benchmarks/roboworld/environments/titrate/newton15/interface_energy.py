"""Interface-energy force from Probst & Teschner, TOG 44(1), 2025.

Equations 8--12, cubic spline support 3*spacing, epsilon=.05 and the
authors' numerical area normalization A0=pi/4*spacing^2 (section 5.2).
This module implements their surface-energy force, not their unified implicit
pressure/friction solver. It is initially used with explicit capillary steps
and the existing DFSPH projection; that combination needs separate acceptance.

Unlike the rejected unweighted pair potentials, only detected interfaces exert
surface forces. Liquid--air, solid--air, and liquid--solid energies are explicit.
Young's relation sets gamma_SV-gamma_SL=sigma*cos(theta). These are macroscopic
interfacial energy differences, not estimates of absolute atomic surface energy.
The plane fixture has a finite sampled patch with an explicit coverage guard.
"""

import math
import numpy as np
import warp as wp
from .sph_kernel_functions import cubic_w, cubic_grad, cubic_plane_fraction


@wp.func
def soft_area(c: float):
    c = wp.max(c, 0.)
    return wp.sqrt(c*c+.0025)-.05


@wp.func
def soft_derivative(c: float):
    return wp.max(c, 0.)/wp.sqrt(c*c+.0025)


@wp.kernel
def fluid_interface(grid: wp.uint64, bg: wp.uint64, q: wp.array[wp.vec3], b: wp.array[wp.vec3],
                    volume: float, boundary_volume: float, h: float, wall: int, ordered: int,
                    fv: wp.array[float], fb: wp.array[float], energy: wp.array[wp.vec2]):
    i = wp.tid()
    if ordered != 0:
        i = wp.hash_grid_point_id(grid, i)
    liquid = float(0.)
    solid = float(0.)
    for j in wp.hash_grid_query(grid, q[i], h):
        r = q[i]-q[j]
        if wp.length_sq(r) < h*h:
            liquid += volume*cubic_w(r, h)
    if wall != 0:
        for j in wp.hash_grid_query(bg, q[i], h):
            r = q[i]-b[j]
            if wp.length_sq(r) < h*h:
                solid += boundary_volume*cubic_w(r, h)
    fv[i] = soft_derivative(1.-liquid-solid)
    fb[i] = soft_derivative(solid)
    energy[i] = wp.vec2(soft_area(1.-liquid-solid), soft_area(solid))


@wp.kernel
def boundary_interface(grid: wp.uint64, q: wp.array[wp.vec3], b: wp.array[wp.vec3],
                       volume: float, h: float,
                       bv: wp.array[float], bf: wp.array[float], energy: wp.array[wp.vec2]):
    i = wp.tid()
    liquid = float(0.)
    for j in wp.hash_grid_query(grid, b[i], h):
        r = b[i]-q[j]
        if wp.length_sq(r) < h*h:
            liquid += volume*cubic_w(r, h)
    solid = cubic_plane_fraction(-b[i][2], h)
    bv[i] = soft_derivative(1.-liquid-solid)
    bf[i] = soft_derivative(liquid)
    energy[i] = wp.vec2(soft_area(1.-liquid-solid), soft_area(liquid))


@wp.kernel
def sampled_boundary_interface(grid: wp.uint64, bg: wp.uint64,
                               q: wp.array[wp.vec3], b: wp.array[wp.vec3], volume: float, boundary_volume: float, h: float, ordered: int,
                               bv: wp.array[float], bf: wp.array[float], energy: wp.array[wp.vec2]):
    i = wp.tid()
    if ordered != 0:
        i = wp.hash_grid_point_id(bg, i)
    liquid, solid = float(0.), float(0.)
    for j in wp.hash_grid_query(grid, b[i], h):
        r = b[i]-q[j]
        if wp.length_sq(r) < h*h:
            liquid += volume*cubic_w(r, h)
    # A boundary node with no fluid in its compact support contributes no
    # fluid force. Skip its much larger solid-solid query exactly. Report
    # energy relative to that node's dry surface so activation is continuous.
    if liquid == 0.:
        bv[i] = 0.
        bf[i] = 0.
        energy[i] = wp.vec2(0.)
        return
    for j in wp.hash_grid_query(bg, b[i], h):
        r = b[i]-b[j]
        if wp.length_sq(r) < h*h:
            solid += boundary_volume*cubic_w(r, h)
    bv[i] = soft_derivative(1.-liquid-solid)
    bf[i] = soft_derivative(liquid)
    energy[i] = wp.vec2(soft_area(1.-liquid-solid)-soft_area(1.-solid), soft_area(liquid))


@wp.kernel
def forces(grid: wp.uint64, bg: wp.uint64, q: wp.array[wp.vec3], b: wp.array[wp.vec3],
           fv: wp.array[float], fb: wp.array[float], bv: wp.array[float], bf: wp.array[float],
           volume: float, boundary_volume: float, h: float, area: float, sigma: float, sv: wp.array[float], sl: wp.array[float],
           wall: int, ordered: int, mass: wp.array[float], velocity: wp.array[wp.vec3], dt: float,
           out: wp.array[wp.vec3], f: wp.array[wp.vec3], reaction: wp.array[wp.vec3]):
    i = wp.tid()
    if ordered != 0:
        i = wp.hash_grid_point_id(grid, i)
    force = wp.vec3(0.)
    for j in wp.hash_grid_query(grid, q[i], h):
        r = q[i]-q[j]
        if wp.length_sq(r) < h*h:
            coefficient = area*volume*sigma*(fv[i]+fv[j])
            if coefficient != 0.:
                force += coefficient*cubic_grad(r, h)
    if wall != 0:
        for j in wp.hash_grid_query(bg, q[i], h):
            r = q[i]-b[j]
            if wp.length_sq(r) < h*h:
                coefficient = area*boundary_volume*(sigma*fv[i]+sv[j]*bv[j]-sl[j]*bf[j])
                if coefficient != 0.:
                    pair = coefficient*cubic_grad(r, h)
                    force += pair
                    wp.atomic_add(reaction, j, -pair)
    f[i] = force
    out[i] = velocity[i]+dt*force/mass[i]


class InterfaceEnergyPredictor:
    def __init__(self, model, volume_m3, support_m, sigma_n_m, repulsive_support_m=0., repulsive_weight=1.):
        self.model, self.wp = model, wp
        self.volume = float(volume_m3)
        self.boundary_volume = self.volume
        self.support_m = float(support_m)
        self.spacing = self.volume**(1./3.)
        if not math.isclose(self.support_m, 3*self.spacing, rel_tol=1e-6):
            raise ValueError("Published interface normalization requires three-spacing surface support")
        self.area = math.pi/4*self.spacing**2
        self.sigma_n_m = float(sigma_n_m)
        self.wall_angle = None
        self.grid = wp.HashGrid(64, 64, 512, device=model.device)
        self.bg = wp.HashGrid(64, 64, 8, device=model.device)
        self.predicted = model.state()
        self.forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self.fv, self.fb = [wp.zeros(model.particle_count, dtype=float, device=model.device) for _ in range(2)]
        self.ef = wp.zeros(model.particle_count, dtype=wp.vec2, device=model.device)
        self.boundary = None
        self.patch_radius = .012
        self.last_surface_energy_j = None
        self.solid_samples = None
        self._solid_generation = None
        # Schedule neighbouring threads on neighbouring query cells without
        # reordering any material/state arrays or changing particle identities.
        # The false path is retained for exact same-state diagnostic comparison.
        self.ordered_threads = True

    def attach_boundaries(self, samples, contact_angles):
        """Bind already updated, body-attached quadrature. No state modification.

        gamma_FB=0, gamma_BF=max(-sigma*cos(theta),0),
        gamma_BV=max(sigma*cos(theta),0). Thus gamma_BV-gamma_BF
        satisfies Young's relation independently for each boundary material.
        """
        # Refine solid volume integration independently of the fluid parcels.
        # The physical interface smoothing length/normalization remain set by
        # fluid dx. Each solid node has area weight A0 * Vb / Vf, so refining
        # quadrature does not multiply wetting energy or reaction forces.
        self.boundary_volume = samples.spacing**3
        self.solid_samples = samples
        self.boundary, self.bg = samples.q, samples.grid
        if self._solid_generation == samples.rebuilds:
            return
        ids = samples.body_ids.numpy()
        if samples.sample_contact_angles is None and set(map(int, ids))-set(contact_angles):
            raise ValueError("Every contacting boundary body needs an explicit wetting material")
        angle = (samples.sample_contact_angles if samples.sample_contact_angles is not None else
                 np.array([contact_angles[int(i)] for i in ids], dtype=float))
        if not np.isfinite(angle).all() or np.any((angle <= 0.) | (angle >= 180.)):
            raise ValueError("Contact angles must lie strictly between zero and 180 degrees")
        cosine = np.cos(np.deg2rad(angle))
        self.sv = wp.array(self.sigma_n_m*np.maximum(cosine, 0.), dtype=float, device=self.model.device)
        self.sl = wp.array(self.sigma_n_m*np.maximum(-cosine, 0.), dtype=float, device=self.model.device)
        self._allocate_boundary_fields(len(ids))
        self._solid_generation = samples.rebuilds

    def _allocate_boundary_fields(self, count):
        self.bv, self.bf = [wp.zeros(count, dtype=float, device=self.model.device) for _ in range(2)]
        self.eb = wp.zeros(count, dtype=wp.vec2, device=self.model.device)
        self.boundary_forces = wp.zeros(count, dtype=wp.vec3, device=self.model.device)

    def _plane(self):
        if self.boundary is not None:
            return
        d = self.spacing
        x = (np.arange(math.ceil(2*self.patch_radius/d))+.5)*d-self.patch_radius
        z = -(np.arange(3)+.5)*d
        points = np.stack(np.meshgrid(x, x, z, indexing="ij"), axis=-1).reshape(-1, 3)
        if self.wall_angle is None:
            points = np.empty((0, 3))
        self.boundary = wp.array(points, dtype=wp.vec3, device=self.model.device)
        self.bg.build(self.boundary, self.support_m)
        self._allocate_boundary_fields(len(points))
        cosine = math.cos(math.radians(self.wall_angle or 0.))
        self.sv = wp.full(len(points), max(self.sigma_n_m*cosine, 0.), dtype=float, device=self.model.device)
        self.sl = wp.full(len(points), max(-self.sigma_n_m*cosine, 0.), dtype=float, device=self.model.device)

    def predict(self, state, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Positive finite explicit surface timestep required")
        self._plane()
        wall = int(self.wall_angle is not None or self.solid_samples is not None)
        if wall and self.solid_samples is None and np.max(np.abs(state.particle_q.numpy()[:, :2]))+self.support_m >= self.patch_radius:
            raise ValueError("Liquid left the explicitly bounded plane-interface fixture; no boundary samples silently omitted")
        from .state_copy import copy_state
        from .runtime import require_newton
        newton, _ = require_newton()
        copy_state(self.predicted, state, wp, newton)
        self.grid.build(state.particle_q, self.support_m)
        self.boundary_forces.zero_()
        wp.launch(fluid_interface, dim=self.model.particle_count,
            inputs=[self.grid.id, self.bg.id, state.particle_q, self.boundary, self.volume, self.boundary_volume, self.support_m, wall, int(self.ordered_threads)],
            outputs=[self.fv, self.fb, self.ef], device=self.model.device)
        if self.solid_samples is not None:
            wp.launch(sampled_boundary_interface, dim=len(self.boundary),
                inputs=[self.grid.id, self.bg.id, state.particle_q, self.boundary, self.volume, self.boundary_volume, self.support_m, int(self.ordered_threads)],
                outputs=[self.bv, self.bf, self.eb], device=self.model.device)
        elif wall:
            wp.launch(boundary_interface, dim=len(self.boundary),
                inputs=[self.grid.id, state.particle_q, self.boundary, self.volume, self.support_m],
                outputs=[self.bv, self.bf, self.eb], device=self.model.device)
        wp.launch(forces, dim=self.model.particle_count,
            inputs=[self.grid.id, self.bg.id, state.particle_q, self.boundary, self.fv, self.fb, self.bv, self.bf,
                self.volume, self.boundary_volume, self.support_m, self.area, self.sigma_n_m, self.sv, self.sl, wall, int(self.ordered_threads),
                self.model.particle_mass, state.particle_qd, dt],
            outputs=[self.predicted.particle_qd, self.forces, self.boundary_forces], device=self.model.device)
        return self.predicted

    def energy(self):
        ef, eb = self.ef.numpy().astype(float), self.eb.numpy().astype(float)
        return self.area*(self.sigma_n_m*ef[:, 0].sum()
            +self.boundary_volume/self.volume*((self.sv.numpy()*eb[:, 0]).sum()+(self.sl.numpy()*eb[:, 1]).sum()))
