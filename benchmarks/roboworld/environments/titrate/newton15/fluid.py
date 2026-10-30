"""Native Newton 1.5 MPM water bring-up. NOT a completed laboratory liquid model.

No particle speed cap, volume spawning during motion, color script, or PhysX.
An optional, explicitly experimental surface-energy force extends native MPM.
Solid wetting and laboratory acceptance are not implemented by this module.
"""

import numpy as np

from .materials import FluidNumerics, Water
from .runtime import require_newton


def lattice_box(lower, upper, spacing):
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all():
        raise ValueError("Finite 3D bounds required")
    if np.any(upper <= lower) or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Invalid particle box or spacing")
    counts = np.floor((upper - lower) / spacing + 1e-10).astype(int)
    if np.any(counts < 1):
        raise ValueError("No particles fit the specified box")
    axes = [lower[a] + (np.arange(counts[a]) + 0.5) * spacing for a in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


class NativeWater:
    def __init__(self, positions, config=None, material=None, gravity=(0.0, 0.0, -9.80665),
                 ground=False):
        self.config = config or FluidNumerics()
        self.material = material or Water()
        self.newton, self.wp = require_newton()
        from newton.solvers import SolverImplicitMPM

        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
            raise ValueError("Expected a nonempty N x 3 particle array")
        if not np.isfinite(positions).all() or not np.isfinite(gravity).all():
            raise ValueError("Nonfinite initial fluid state")
        builder = self.newton.ModelBuilder()
        builder.particle_max_velocity = float("inf")
        SolverImplicitMPM.register_custom_attributes(builder)
        spacing = self.config.spacing_m
        self.volume_per_particle_m3 = spacing**3
        self.mass_per_particle_kg = self.material.density_kg_m3 * spacing**3
        for pos in positions:
            builder.add_particle(pos=tuple(pos), vel=(0.0, 0.0, 0.0),
                                 mass=self.mass_per_particle_kg, radius=spacing / 2.0)
        if ground:
            builder.add_ground_plane(cfg=builder.ShapeConfig(mu=0.0))
        self.model = builder.finalize(device=self.config.device)
        self.model.set_gravity(gravity)
        self.model.mpm.viscosity.fill_(self.material.viscosity_pa_s)
        self.model.mpm.friction.fill_(0.0)
        self.model.mpm.yield_stress.fill_(0.0)
        self.model.mpm.yield_pressure.fill_(1.0e15)
        self.model.mpm.tensile_yield_ratio.fill_(self.config.tensile_limit_pa / 1.0e15)
        # Native incompressible/inelastic MPM limit; not an elastic water solid.
        # A finite isotropic elastic carrier retains bulk compression; its
        # deviatoric stress still flows under the zero-shear-yield fluid law.
        # K = E/[3(1-2 nu)], with nu=.3 as the carrier parameter, not a claim
        # that water sustains an elastic shear modulus.
        self.model.mpm.poisson_ratio.fill_(.3)
        self.model.mpm.young_modulus.fill_(1.0e15 if self.config.bulk_modulus_pa is None
                                          else 1.2*self.config.bulk_modulus_pa)
        options = SolverImplicitMPM.Config()
        options.voxel_size = self.config.voxel_m
        options.max_iterations = self.config.max_iterations
        options.tolerance = self.config.tolerance
        options.air_drag = self.config.air_drag
        options.critical_fraction = self.config.critical_fraction
        options.strain_basis = self.config.strain_basis
        options.velocity_basis = self.config.velocity_basis
        options.integration_scheme = self.config.integration_scheme
        self.solver = SolverImplicitMPM(self.model, config=options, enable_timers=False, verbose=self.config.verbose)
        self.state = self.model.state()
        self.next_state = self.model.state()
        self.tick = 0
        self.capillary = None
        if self.config.capillarity:
            from .capillary import CapillaryPredictor
            from .surface_energy import SurfaceEnergyPredictor
            from .csf import CsfPredictor
            from .mesh_capillary import MeshCapillaryPredictor

            capillary_dt = 0.2 * np.sqrt(self.material.density_kg_m3 *
                                         self.config.spacing_m**3 /
                                         self.material.surface_tension_n_m)
            if self.config.dt_s > capillary_dt:
                raise ValueError(f"dt={self.config.dt_s} exceeds explicit capillary bound {capillary_dt}; no clipping")
            predictor = SurfaceEnergyPredictor if self.config.surface_model == "area" else CapillaryPredictor
            if self.config.surface_model == "csf":
                predictor = CsfPredictor
            if self.config.surface_model == "mesh":
                predictor = MeshCapillaryPredictor
            if self.config.surface_model == "interface_energy":
                from .interface_energy import InterfaceEnergyPredictor
                predictor = InterfaceEnergyPredictor
            self.capillary = predictor(
                self.model, self.volume_per_particle_m3, self.config.surface_support_m,
                self.material.surface_tension_n_m,
                2.5*self.config.spacing_m if self.config.surface_model == "balanced" else 0.0)

    def step(self):
        predicted = (self.capillary.predict(self.state, self.config.dt_s)
                     if self.capillary is not None else self.state)
        self.solver.step(predicted, self.next_state, None, None, self.config.dt_s)
        self.state, self.next_state = self.next_state, self.state
        self.tick += 1

    def read(self):
        q, v = self.state.particle_q.numpy(), self.state.particle_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise FloatingPointError(f"Nonfinite Newton MPM state at tick {self.tick}")
        return q, v
