"""Continuum surface force with a contact-angle normal boundary condition.

Brackbill's f=sigma*kappa*grad(c), kappa=-div(grad(c)/|grad(c)|).
Wetting changes the interface normal at the wall; it does not apply an
unbounded bulk attraction toward the solid. This is an explicit experimental
Newton MPM constitutive extension, not another dynamics engine.
"""

import numpy as np
from .capillary import CapillaryPredictor


class CsfPredictor(CapillaryPredictor):
    def predict(self, state, dt):
        from .csf_kernels import sample, curvature_force, distribute
        from .plane_surface_kernels import mirror

        sigma = self.sigma_n_m
        self.sigma_n_m = 0.
        try:
            result = super().predict(state, dt)
        finally:
            self.sigma_n_m = sigma
        wp, device = self.wp, self.model.device
        q = state.particle_q.numpy()
        n = len(q)
        dx, h = self.support_m/6., self.support_m
        lower = np.floor((q.min(axis=0)-h)/dx)*dx
        upper = np.ceil((q.max(axis=0)+h)/dx)*dx
        plane = getattr(self, "wall", None) is not None
        if plane:
            points = wp.empty(2*n, dtype=wp.vec3, device=device)
            volumes = wp.empty(2*n, dtype=float, device=device)
            wp.launch(mirror, dim=n, inputs=[state.particle_q, self.volumes], outputs=[points, volumes], device=device)
            lower[2] = 0.
        else:
            points, volumes = state.particle_q, self.volumes
        self.grid.build(points, h)
        counts = tuple(int(v) for v in np.rint((upper-lower)/dx)+1)
        if np.prod(counts) > 2_000_000:
            raise ValueError("CSF quadrature exceeds explicit 2M voxel budget")
        gradient = wp.empty(counts, dtype=wp.vec3, device=device)
        normals = wp.empty(counts, dtype=wp.vec3, device=device)
        field = wp.empty(counts, dtype=float, device=device)
        forces = wp.empty(counts, dtype=wp.vec3, device=device)
        wp.launch(sample, dim=counts,
                  inputs=[self.grid.id, points, volumes, h, wp.vec3(*lower), dx, n,
                          plane, getattr(getattr(self, "wall", None), "angle_deg", 90.)*np.pi/180.],
                  outputs=[gradient, normals, field], device=device)
        wp.launch(curvature_force, dim=counts, inputs=[gradient, normals, sigma, dx, plane],
                  outputs=[forces], device=device)
        wp.launch(distribute, dim=n,
                  inputs=[state.particle_q, state.particle_qd, self.model.particle_mass,
                          self.volumes, h, wp.vec3(*lower), dx, field, forces, dt, plane],
                  outputs=[result.particle_qd, self.forces], device=device)
        return result
