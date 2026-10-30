"""Variational diffuse-interface surface area, independently derived.

E = sigma integral sqrt(|grad c|^2 + epsilon^2) - epsilon dx,
c(x) = sum Vp W(x-qp). The force is the exact derivative of this
quadrature energy: Fp = sigma Vp sum Hess W(x-qp) n(x) dV.
Unlike a pair-attraction force, it has no bulk pair-attraction energy.
The grid is quadrature, not a particle-position correction or mass source.
The method is distinct from Chen et al.'s.
"""

import numpy as np

from .capillary import CapillaryPredictor


class SurfaceEnergyPredictor(CapillaryPredictor):
    def predict(self, state, dt):
        from .surface_energy_kernels import sample_gradient, area_force, sample_phase, sample_weighted_area

        # Reuse the complete, non-mutating Newton-state copy. Zero the old
        # pair potential: its kernel only performs the initial velocity copy.
        sigma = self.sigma_n_m
        self.sigma_n_m = 0.0
        try:
            predicted = super().predict(state, dt)
        finally:
            self.sigma_n_m = sigma
        wp = self.wp
        q = state.particle_q.numpy()
        wall = getattr(self, "wall", None)
        plane = wall is not None and getattr(wall, "mirror_plane", False)
        if plane:
            from .plane_surface_kernels import mirror, apply_mirror

            n = self.model.particle_count
            points = wp.empty(2*n, dtype=wp.vec3, device=self.model.device)
            volumes = wp.empty(2*n, dtype=float, device=self.model.device)
            wp.launch(mirror, dim=n, inputs=[state.particle_q, self.volumes], outputs=[points, volumes],
                      device=self.model.device)
            self.grid.build(points, self.support_m)
        else:
            points, volumes = state.particle_q, self.volumes
        dx = self.support_m / 6.0
        lower = np.floor((q.min(axis=0)-self.support_m)/dx)*dx
        upper = np.ceil((q.max(axis=0)+self.support_m)/dx)*dx
        if plane:
            lower[2] = dx/2
        counts = tuple(int(v) for v in np.rint((upper-lower)/dx)+1)
        if np.prod(counts) > 2_000_000:
            raise ValueError("Surface-energy quadrature exceeds explicit 2M cell budget")
        normals = wp.empty(counts, dtype=wp.vec3, device=self.model.device)
        energies = wp.empty(counts, dtype=float, device=self.model.device)
        potential = wp.zeros(counts, dtype=float, device=self.model.device)
        phase_field = getattr(self, "phase_field", False)
        weighted = getattr(self, "weighted_area", False)
        epsilon = self.support_m/8 if phase_field else 0.001/self.support_m
        wp.launch(sample_phase if phase_field else sample_weighted_area if weighted else sample_gradient, dim=counts,
                  inputs=[self.grid.id, points, volumes,
                          self.support_m, wp.vec3(*lower), dx, epsilon],
                  outputs=[normals, potential, energies] if phase_field or weighted else [normals, energies], device=self.model.device)
        zero_v = wp.zeros(len(points), dtype=wp.vec3, device=self.model.device)
        masses = wp.full(len(points), float(self.model.particle_mass.numpy()[0]), device=self.model.device)
        all_f = wp.empty(len(points), dtype=wp.vec3, device=self.model.device)
        wp.launch(area_force, dim=len(points),
                  inputs=[points, zero_v, masses,
                          volumes, self.support_m, wp.vec3(*lower), dx,
                          normals, potential, sigma, dt],
                  outputs=[zero_v, all_f], device=self.model.device)
        if plane:
            wp.launch(apply_mirror, dim=self.model.particle_count,
                      inputs=[state.particle_qd, self.model.particle_mass, all_f, dt],
                      outputs=[predicted.particle_qd, self.forces], device=self.model.device)
        else:
            from .plane_surface_kernels import add_force

            wp.launch(add_force, dim=self.model.particle_count,
                      inputs=[state.particle_qd, self.model.particle_mass, all_f, dt],
                      outputs=[predicted.particle_qd, self.forces], device=self.model.device)
        self.energy_j = sigma*dx**3*float(energies.numpy().sum(dtype=np.float64))
        if getattr(self, "wall", None) is not None:
            if phase_field or weighted:
                from .phase_wetting import apply

                apply(self, state, dt)
            else:
                self.wall.apply(self, state, dt)
        return predicted
