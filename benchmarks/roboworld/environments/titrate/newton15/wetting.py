"""Resolved wall quadrature for a Young surface-energy boundary condition.

The diffuse liquid/air surface energy includes the interface against a wall.
Subtract 2 sigma (1+cos(theta)) integral_wall c dA: c tends to 1/2 at a
flat filled wall as the kernel width tends to zero. Thus its net energy is
-sigma cos(theta) per wetted area, as required by Young's equation.
The finite-kernel contact angle MUST be checked, not assumed from this limit.
"""

import numpy as np
from .runtime import require_newton


class WallQuadrature:
    def __init__(self, positions, normals, areas, angles_deg, body_ids=None, device="cuda:0"):
        _, wp = require_newton()
        self.wp = wp
        n = len(positions)
        self.positions = wp.array(positions, dtype=wp.vec3, device=device)
        self.normals = wp.array(normals, dtype=wp.vec3, device=device)
        self.areas = wp.array(areas, dtype=float, device=device)
        self.coefficients = wp.array(2*(1+np.cos(np.deg2rad(np.broadcast_to(angles_deg, n)))), dtype=float, device=device)
        self.body_ids = wp.array(np.full(n, -1) if body_ids is None else body_ids, dtype=int, device=device)
        self.forces = wp.zeros(n, dtype=wp.vec3, device=device)
        self.grid = wp.HashGrid(64, 64, 64, device=device)

    @classmethod
    def plane(cls, half_width=.020, spacing=.00025, angle_deg=25., device="cuda:0"):
        x = np.arange(-half_width, half_width+spacing*.1, spacing)
        xy = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1).reshape(-1, 2)
        q = np.c_[xy, np.zeros(len(xy))]
        obj = cls(q, np.tile([0., 0., 1.], (len(q), 1)), np.full(len(q), spacing**2),
                  angle_deg, device=device)
        obj.mirror_plane = True
        obj.coefficients.fill_(2*np.cos(np.deg2rad(angle_deg)))
        return obj

    def apply(self, predictor, state, dt):
        from .wetting_kernels import wall_force, contact_line, distribute_line

        wp = self.wp
        self.grid.build(self.positions, predictor.support_m)
        self.forces.zero_()
        if getattr(self, "mirror_plane", False):
            # Young contact-line traction, projected into the impermeable
            # wall tangent plane. Mirror extension already supplies the
            # neutral (90 degree) liquid/air interface energy.
            concentration = wp.empty(len(self.positions), dtype=float, device=predictor.model.device)
            line_forces = wp.empty(len(self.positions), dtype=wp.vec3, device=predictor.model.device)
            fluid_grid = wp.HashGrid(64, 64, 64, device=predictor.model.device)
            fluid_grid.build(state.particle_q, predictor.support_m)
            wp.launch(contact_line, dim=len(self.positions),
                      inputs=[fluid_grid.id, state.particle_q, predictor.volumes,
                              self.positions, self.normals, self.areas, self.coefficients,
                              predictor.support_m, predictor.sigma_n_m],
                      outputs=[concentration, line_forces], device=predictor.model.device)
            wp.launch(distribute_line, dim=len(state.particle_q),
                      inputs=[self.grid.id, self.positions, concentration, line_forces,
                              state.particle_q, predictor.volumes, predictor.model.particle_mass,
                              predictor.support_m, dt],
                      outputs=[predictor.predicted.particle_qd, predictor.forces, self.forces],
                      device=predictor.model.device)
            return
        wp.launch(wall_force, dim=len(state.particle_q),
                  inputs=[self.grid.id, self.positions, self.normals, self.areas, self.coefficients,
                          state.particle_q, predictor.volumes, predictor.model.particle_mass,
                          predictor.support_m, predictor.sigma_n_m, dt],
                  outputs=[predictor.predicted.particle_qd, predictor.forces, self.forces],
                  device=predictor.model.device)
