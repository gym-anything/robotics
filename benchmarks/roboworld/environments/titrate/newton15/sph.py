"""Explicitly experimental DFSPH extension using Newton 1.5 Model/State.

This is repository-owned water physics, not a claim of native Newton SPH.
It is evaluated separately from the retained native MPM experiments.
"""

from dataclasses import dataclass, asdict
import math
import numpy as np

from .materials import Water
from .runtime import require_newton


@dataclass(frozen=True)
class SphNumerics:
    spacing_m: float = .0005
    dt_s: float = .00025
    support_ratio: float = 2.0
    max_iterations: int = 100
    density_tolerance: float = .0001
    divergence_tolerance: float = .001
    device: str = "cuda:0"
    surface_model: str = "area"
    divergence_projection: bool = True
    pressure_graph: bool = False
    pressure_graph_chunk: int = 0
    neighbor_cache: bool = False
    planar_energy_normalization: bool = False
    pressure_kernel: str = "wendland"
    warm_start: bool = False
    warm_start_fraction: float = .5
    accelerated_pressure: bool = False
    hydrostatic_initial_guess: bool = False
    maximum_projection_error: float | None = None
    recompute_gradients: bool = False
    wall_clearance_fraction: float = .5
    boundary_refinement: int = 1
    maximum_boundary_samples: int = 2_000_000
    nonlinear_density: bool = False
    nonlinear_iterations: int = 12
    nonlinear_density_error: float = .005
    nonlinear_motion_error_fraction: float = .005
    nonlinear_linesearch: bool = False
    nonlinear_relaxation: float = 1.
    nonlinear_stationarity: bool = True
    implicit_surface: bool = False
    implicit_surface_iterations: int = 64
    implicit_surface_error: float = .0005

    @property
    def support_m(self):
        return self.spacing_m*self.support_ratio

    def __post_init__(self):
        for k, v in asdict(self).items():
            if type(v) in (int, float) and (not math.isfinite(v) or (v <= 0 and k not in ('wall_clearance_fraction', 'pressure_graph_chunk'))):
                raise ValueError(f"Positive finite {k} required")
        if type(self.max_iterations) is not int or self.surface_model not in ("area", "weighted_area", "mesh", "none", "attraction_only", "csf", "balanced", "pair_mirror", "core_pair", "core_mirror", "phase_field", "cohesion2013", "interface_energy"):
            raise ValueError("Invalid DFSPH iteration count or surface model")
        if self.pressure_kernel not in ("wendland", "cubic"):
            raise ValueError("Choose an explicit supported pressure kernel")
        if any(type(v) is not bool for v in (self.divergence_projection, self.pressure_graph, self.neighbor_cache, self.planar_energy_normalization, self.warm_start, self.accelerated_pressure, self.hydrostatic_initial_guess, self.recompute_gradients, self.nonlinear_density, self.nonlinear_linesearch, self.implicit_surface)):
            raise ValueError("Explicit divergence-projection selection required")
        if self.neighbor_cache and not self.pressure_graph:
            raise ValueError("Cached pressure gradients require GPU projection control")
        if type(self.pressure_graph_chunk) is not int or self.pressure_graph_chunk < 0:
            raise ValueError('Pressure graph chunk must be a nonnegative integer')
        if self.pressure_graph_chunk and (not self.pressure_graph or not self.accelerated_pressure):
            raise ValueError('Unrolled pressure graphs require accelerated GPU projection')
        if self.accelerated_pressure and not self.pressure_graph:
            raise ValueError("Accelerated pressure requires the GPU projection graph")
        if self.warm_start_fraction > 1 or (self.hydrostatic_initial_guess and not self.warm_start):
            raise ValueError("Warm fraction must be at most one; initial pressure guess requires warm starting")
        if self.maximum_projection_error is not None and not self.pressure_graph:
            raise ValueError("The explicit maximum-residual criterion currently requires the GPU pressure graph")
        if self.recompute_gradients and not self.neighbor_cache:
            raise ValueError("Position-based pressure traversal requires the neighbor identity cache")
        if isinstance(self.wall_clearance_fraction, bool) or not 0 <= self.wall_clearance_fraction <= .5:
            raise ValueError('Explicit numerical wall clearance must lie within 0..0.5 sample spacings')
        if type(self.boundary_refinement) is not int or type(self.maximum_boundary_samples) is not int:
            raise ValueError('Boundary refinement and capacity must be positive integers')
        if type(self.nonlinear_iterations) is not int:
            raise ValueError('Nonlinear iteration limit must be a positive integer')
        if type(self.nonlinear_stationarity) is not bool:
            raise ValueError('Explicit nonlinear stationarity selection required')
        if self.nonlinear_relaxation > 1.:
            raise ValueError('Nonlinear trial relaxation must lie in (0, 1]')
        if self.nonlinear_linesearch and self.nonlinear_relaxation != 1.:
            raise ValueError('Choose line search or fixed trial relaxation, not both')
        if type(self.implicit_surface_iterations) is not int:
            raise ValueError('Implicit surface iteration limit must be a positive integer')
        if self.implicit_surface and self.surface_model != 'interface_energy':
            raise ValueError('Implicit surface requires interface_energy')
        if self.planar_energy_normalization and self.surface_model != "core_pair":
            raise ValueError("Planar normalization is implemented for core_pair only")


class ProjectedWater:
    def __init__(self, positions, config=None, material=None,
                 gravity=(0., 0., -9.80665), ground=False, wall_angle=None):
        self.config, self.material = config or SphNumerics(), material or Water()
        if self.config.surface_model != "none":
            # Explicit capillary-wave screening. The Nyquist inviscid mode has
            # omega^2=sigma*(pi/dx)^3/rho. Use a conservative margin below the
            # symplectic-Euler limit 2/omega. This is not a stability guarantee
            # for all contacts or nonlinear states; do not silently alter dt.
            limit = .2*math.sqrt(self.material.density_kg_m3*self.config.spacing_m**3 /
                                self.material.surface_tension_n_m)
            if self.config.dt_s > limit:
                raise ValueError(f"Requested dt {self.config.dt_s:g}s exceeds explicit capillary screening limit {limit:g}s; no clamping")
        self.newton, self.wp = require_newton()
        wp, cfg = self.wp, self.config
        self.kernel_kind = int(cfg.pressure_kernel == "cubic")
        q = np.asarray(positions, dtype=float)
        if q.ndim != 2 or q.shape[1] != 3 or not len(q) or not np.isfinite(q).all():
            raise ValueError("Finite nonempty Nx3 fluid positions required")
        self.volume_per_particle_m3 = cfg.spacing_m**3
        self.mass_per_particle_kg = self.volume_per_particle_m3*self.material.density_kg_m3
        self.gravity, self.ground = wp.vec3(*gravity), int(ground)
        self.wall_angle = wall_angle
        builder = self.newton.ModelBuilder()
        builder.particle_max_velocity = float("inf")
        for x in q:
            builder.add_particle(tuple(x), (0., 0., 0.), self.mass_per_particle_kg, radius=cfg.spacing_m/2)
        self.model = builder.finalize(device=cfg.device)
        self.state, self.next_state = self.model.state(), self.model.state()
        self.constraint_positions = self.state.particle_q
        self.ordered_threads = True  # Query scheduling only; particle arrays/IDs remain unchanged.
        self.grid = wp.HashGrid(64, 64, 512, device=cfg.device)
        self.rho = wp.zeros(len(q), dtype=float, device=cfg.device)
        self.factor, self.p, self.next_p, self.residual = [wp.zeros_like(self.rho) for _ in range(4)]
        self.warm_density, self.warm_divergence = [wp.zeros_like(self.rho) for _ in range(2)]
        self.bg = wp.zeros(len(q), dtype=wp.vec3, device=cfg.device)
        self.fluid_gradient = wp.zeros_like(self.bg)
        self.boundary_rate = wp.zeros_like(self.rho)
        self.boundaries = None
        self.force_models = ()
        self.dv, self.velocity = wp.zeros_like(self.bg), wp.zeros_like(self.bg)
        self.predicted_velocity = wp.zeros_like(self.bg)
        self.neighbors = wp.zeros(len(q), dtype=int, device=cfg.device)
        self.tick, self.last_projection = 0, {}
        self.active = wp.ones(1, dtype=int, device=cfg.device)
        self.pressure_graphs = None
        self.neighbor_cache = None
        if cfg.neighbor_cache:
            from .neighbor_cache import NeighborCache

            self.neighbor_cache = NeighborCache(self)
        if cfg.pressure_graph:
            from .pressure_graph import PressureGraphs

            self.pressure_graphs = PressureGraphs(self)
        self.capillary = None
        self.planar_moments = None
        if cfg.surface_model != "none":
            from .surface_energy import SurfaceEnergyPredictor
            from .mesh_capillary import MeshCapillaryPredictor
            from .capillary import CapillaryPredictor
            from .csf import CsfPredictor
            from .cohesion2013 import CohesionPredictor
            from .interface_energy import InterfaceEnergyPredictor

            predictor = {"area": SurfaceEnergyPredictor, "mesh": MeshCapillaryPredictor,
                         "attraction_only": CapillaryPredictor, "csf": CsfPredictor,
                         "balanced": CapillaryPredictor, "pair_mirror": CapillaryPredictor,
                         "core_pair": CapillaryPredictor, "core_mirror": CapillaryPredictor,
                         "phase_field": SurfaceEnergyPredictor, "weighted_area": SurfaceEnergyPredictor,
                         "cohesion2013": CohesionPredictor,
                         "interface_energy": InterfaceEnergyPredictor}[cfg.surface_model]
            self.capillary = predictor(self.model, self.volume_per_particle_m3,
                                       (4 if cfg.surface_model == "balanced" else 3)*cfg.spacing_m,
                                       self.material.surface_tension_n_m,
                                       2.5*cfg.spacing_m if cfg.surface_model == "balanced" else 0.)
            if cfg.surface_model in ("core_pair", "core_mirror"):
                # Short-range numerical core: the isolated pair force changes
                # sign at one particle spacing. Planar surface energy retains
                # the SI sigma via the exact kernel first moment.
                self.capillary.repulsive_support_m = 1.5*cfg.spacing_m
                self.capillary.repulsive_weight = .25
            if cfg.planar_energy_normalization:
                from .planar_energy import moments
                self.planar_moments = moments(cfg.spacing_m, self.capillary.support_m,
                    self.capillary.repulsive_support_m, self.capillary.repulsive_weight)
                self.capillary.energy_length_m = self.planar_moments["liquid_air_m"]
            if cfg.surface_model == "phase_field":
                self.capillary.phase_field = True
            if cfg.surface_model == "weighted_area":
                self.capillary.weighted_area = True
            if cfg.surface_model in ("cohesion2013", "interface_energy"):
                self.capillary.wall_angle = wall_angle
            if wall_angle is not None and cfg.surface_model not in ("attraction_only", "balanced", "core_pair", "cohesion2013", "interface_energy"):
                from .wetting import WallQuadrature

                self.capillary.wall = WallQuadrature.plane(.012, cfg.spacing_m/2, wall_angle, cfg.device)
                self.capillary.wall.angle_deg = wall_angle

    def _launch(self, kernel, inputs, outputs):
        self.wp.launch(kernel, dim=self.model.particle_count, inputs=inputs, outputs=outputs,
                       device=self.config.device)

    def _project(self, v, density_mode, *, react=True):
        from .sph_kernels import pressure_velocity, jacobi, apply_velocity, plane_contact

        if self.pressure_graphs is not None:
            result = self.pressure_graphs.project(v, density_mode)
            if self.boundaries is not None and react:
                self.boundaries.pressure_reaction(self)
            return result
        cfg = self.config
        tolerance = cfg.density_tolerance if density_mode else cfg.divergence_tolerance
        from .pressure_graph import start_pressure, clear_disabled_rows, gate_warm_pressure
        previous = self.warm_density if density_mode else self.warm_divergence
        self._launch(start_pressure, [previous, int(cfg.warm_start), cfg.warm_start_fraction], [self.p])
        self._launch(clear_disabled_rows, [self.neighbors, self.bg, int(density_mode), self.factor], [self.p])
        if cfg.warm_start:
            self._launch(gate_warm_pressure, [self.grid.id, self.constraint_positions, v,
                self.volume_per_particle_m3, cfg.support_m, self.kernel_kind, self.bg,
                self.boundary_rate, self.rho, cfg.dt_s, int(density_mode)], [self.p])
        for iteration in range(cfg.max_iterations):
            self._launch(pressure_velocity, [self.grid.id, self.constraint_positions,
                         self.volume_per_particle_m3, cfg.support_m, self.p, self.bg, self.active, self.kernel_kind], [self.dv])
            if self.ground:
                self._launch(plane_contact, [self.state.particle_q, v, cfg.dt_s, cfg.spacing_m*cfg.wall_clearance_fraction, self.active], [self.dv])
            if self.boundaries is not None:
                self.boundaries.contact_projection(self, v, self.active)
            self._launch(jacobi, [self.grid.id, self.constraint_positions, v, self.dv,
                         self.volume_per_particle_m3, cfg.support_m, self.bg, self.rho,
                         self.factor, self.neighbors, cfg.dt_s, int(density_mode), self.p, self.active, self.kernel_kind, self.boundary_rate],
                         [self.next_p, self.residual])
            # Check the measured residual of the CURRENT iterate before swapping.
            # Mean follows DFSPH; max is also reported, never silently discarded.
            r = self.residual.numpy()
            if not np.isfinite(r).all():
                raise FloatingPointError("Nonfinite DFSPH pressure residual")
            if iteration >= 2 and float(r.mean()) <= tolerance:
                break
            self.p, self.next_p = self.next_p, self.p
        else:
            raise FloatingPointError(f"DFSPH projection did not converge: mean={r.mean():.6g}, max={r.max():.6g}")
        self.last_projection["density" if density_mode else "divergence"] = {
            "iterations": iteration+1, "mean_residual": float(r.mean()), "max_residual": float(r.max())}
        self._launch(apply_velocity, [v, self.dv], [self.velocity])
        previous.assign(self.p)
        if self.boundaries is not None and react:
            self.boundaries.pressure_reaction(self)
        return self.velocity

    def step(self):
        from .sph_kernels import prepare, nonpressure, integrate

        cfg = self.config
        self.constraint_positions = self.state.particle_q
        self.grid.build(self.state.particle_q, cfg.support_m)
        if self.neighbor_cache is not None:
            self.neighbor_cache.update()
        self._launch(prepare, [self.grid.id, self.state.particle_q, self.volume_per_particle_m3,
                     cfg.support_m, self.ground, self.kernel_kind, int(self.ordered_threads)],
                     [self.rho, self.factor, self.bg, self.neighbors, self.fluid_gradient])
        if self.boundaries is not None:
            self.boundaries.prepare(self)
        velocity = (self._project(self.state.particle_qd, False) if cfg.divergence_projection
                    else self.state.particle_qd)
        self.next_state.particle_qd.assign(velocity)
        self.next_state.particle_q.assign(self.state.particle_q)
        prediction = self.capillary.predict(self.next_state, cfg.dt_s) if self.capillary and not cfg.implicit_surface else self.next_state
        if cfg.surface_model in ("pair_mirror", "core_mirror") and self.wall_angle is not None:
            from .sph_kernels import mirror_cohesion

            self._launch(mirror_cohesion, [self.grid.id, self.state.particle_q,
                         self.volume_per_particle_m3, self.capillary.support_m,
                         self.material.surface_tension_n_m, self.material.density_kg_m3,
                         cfg.dt_s, self.capillary.repulsive_support_m, self.capillary.repulsive_weight],
                         [prediction.particle_qd])
            self.capillary.wall.apply(self.capillary, self.next_state, cfg.dt_s)
        if cfg.surface_model in ("attraction_only", "balanced", "core_pair") and self.wall_angle is not None:
            from .sph_kernels import pair_wall

            self._launch(pair_wall, [self.state.particle_q, prediction.particle_qd,
                         self.capillary.support_m, cfg.dt_s, self.material.surface_tension_n_m,
                         self.material.density_kg_m3, math.cos(math.radians(self.wall_angle)),
                         self.capillary.repulsive_support_m, self.capillary.repulsive_weight,
                         self.planar_moments["wall_m"] if self.planar_moments else 0.],
                         [prediction.particle_qd])
        self._launch(nonpressure, [self.grid.id, self.state.particle_q, prediction.particle_qd,
                     self.volume_per_particle_m3, cfg.support_m,
                     self.material.viscosity_pa_s/self.material.density_kg_m3, self.rho,
                     self.gravity, cfg.dt_s, self.kernel_kind, int(self.ordered_threads)], [self.predicted_velocity])
        if self.boundaries is not None:
            self.boundaries.viscosity(self, prediction.particle_qd)
        for force_model in self.force_models:
            force_model.apply(self)
        if cfg.implicit_surface:
            from .implicit_interface import project
            velocity = project(self, self.predicted_velocity)
        elif cfg.nonlinear_density:
            from .nonlinear_density import project
            velocity = project(self, self.predicted_velocity)
        else:
            velocity = self._project(self.predicted_velocity, True)
        if self.boundaries is not None:
            self.boundaries.validate_integration(self, velocity)
        self.next_state.particle_qd.assign(velocity)
        self._launch(integrate, [self.state.particle_q, velocity, cfg.dt_s], [self.next_state.particle_q])
        self.state, self.next_state = self.next_state, self.state
        self.constraint_positions = self.state.particle_q
        self.tick += 1

    def read(self):
        q, v = self.state.particle_q.numpy(), self.state.particle_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise FloatingPointError(f"Nonfinite DFSPH state at tick {self.tick}")
        return q, v
