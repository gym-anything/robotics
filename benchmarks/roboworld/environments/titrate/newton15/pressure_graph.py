"""GPU-resident DFSPH iteration control; no extra physics steps or looser tolerances.

Capture a device-side loop bounded by the declared maximum iteration count.
Kernels stop updating once the
same minimum-three-iterations / mean-residual test passes. One final host
readback replaces a device synchronization after every Jacobi iteration.
"""

import numpy as np
import warp as wp
from .sph_kernels import pressure_grad, disabled_pressure_row


@wp.kernel
def start_pressure(previous: wp.array[float], enabled: int, fraction: float, p: wp.array[float]):
    i = wp.tid()
    # Same fixed dt in both steps. SPlisHSPlasH/Bender-Koschier's 0.5
    # previous-pressure initial guess; convergence criteria are unchanged.
    p[i] = fraction*previous[i] if enabled else 0.


@wp.kernel
def clear_disabled_rows(neighbors: wp.array[int], bg: wp.array[wp.vec3], density_mode: int,
                        factor: wp.array[float], p: wp.array[float]):
    i = wp.tid()
    # A removed constraint has zero dual multiplier, not zero residual with
    # an arbitrary retained multiplier. Otherwise a warm-start pressure can
    # push sparse free-surface particles forever while reporting convergence.
    if disabled_pressure_row(factor[i], density_mode, neighbors[i]):
        p[i] = 0.


@wp.kernel
def gate_warm_pressure(grid: wp.uint64, q: wp.array[wp.vec3], velocity: wp.array[wp.vec3],
        volume: float, support: float, kind: int, bg: wp.array[wp.vec3],
        boundary_rate: wp.array[float], rho: wp.array[float], dt: float,
        density_mode: int, p: wp.array[float]):
    i = wp.tid()
    if p[i] == 0.:
        return
    rate = wp.dot(velocity[i], bg[i])-boundary_rate[i]
    for j in wp.hash_grid_query(grid, q[i], support):
        r = q[i]-q[j]
        if wp.length_sq(r) < support*support:
            rate += volume*wp.dot(velocity[i]-velocity[j], pressure_grad(r, support, kind))
    error = dt*rate
    if density_mode != 0:
        error += rho[i]-1.
    # Upstream DFSPH initializes warm pressure only for a compressive
    # unconstrained prediction. This is an initial guess, not a disabled row:
    # ordinary iterations may subsequently assign positive pressure here.
    if error <= 0.:
        p[i] = 0.


@wp.kernel
def projected_residual(p: wp.array[float], proposed: wp.array[float], factor: wp.array[float],
                       dt: float, residual: wp.array[float]):
    i = wp.tid()
    # Projected-gradient/KKT residual, expressed in density-ratio units.
    # Positive error alone cannot detect an accelerated pressure overshoot.
    if factor[i] > 0.:
        residual[i] = wp.abs(p[i]-proposed[i])*dt/(.25*factor[i])


@wp.kernel
def partial_sum(values: wp.array[float], partial: wp.array[float], maxima: wp.array[float]):
    block, lane = wp.tid()
    tile = wp.tile_load(values, shape=256, offset=block*256)
    total = wp.tile_sum(tile)
    high = wp.tile_max(tile)
    if lane == 0:
        partial[block] = total[0]
        maxima[block] = high[0]


@wp.kernel
def finish_sum(partial: wp.array[float], maxima: wp.array[float], total: wp.array[float], high: wp.array[float]):
    value = float(0.)
    maximum = float(0.)
    for i in range(partial.shape[0]):
        value += partial[i]
        maximum = wp.max(maximum, maxima[i])
    total[0] = value
    high[0] = maximum


@wp.kernel
def initialize(active: wp.array[int], iterations: wp.array[int], running: wp.array[int]):
    active[0] = 1
    iterations[0] = 0
    running[0] = 1


@wp.kernel
def decide(total: wp.array[float], high: wp.array[float], count: int, tolerance: float, maximum_error: float, maximum: int,
           active: wp.array[int], iterations: wp.array[int], running: wp.array[int]):
    if active[0] != 0:
        iterations[0] += 1
        if iterations[0] >= 3 and total[0]/float(count) <= tolerance and high[0] <= maximum_error:
            active[0] = 0
        if active[0] == 0 or iterations[0] >= maximum:
            running[0] = 0


@wp.kernel
def select_verification(total: wp.array[float], high: wp.array[float], count: int,
        tolerance: float, maximum_error: float, maximum: int,
        iterations: wp.array[int], verify: wp.array[int]):
    iterations[0] += 1
    # The mapping at an extrapolate is only a cheap candidate-screen, never
    # acceptance. At the declared budget also evaluate the final feasible
    # candidate, so the final diagnostics describe its actual residual.
    verify[0] = int((iterations[0] >= 3 and total[0]/float(count) <= tolerance
        and high[0] <= maximum_error) or iterations[0] >= maximum)


@wp.kernel
def accept_candidate(total: wp.array[float], high: wp.array[float], count: int,
        tolerance: float, maximum_error: float, iterations: wp.array[int], active: wp.array[int]):
    if iterations[0] >= 3 and total[0]/float(count) <= tolerance and high[0] <= maximum_error:
        active[0] = 0


@wp.kernel
def loop_status(active: wp.array[int], iterations: wp.array[int], maximum: int, running: wp.array[int]):
    running[0] = int(active[0] != 0 and iterations[0] < maximum)


@wp.kernel
def readout(total: wp.array[float], high: wp.array[float], count: int,
        active: wp.array[int], iterations: wp.array[int], result: wp.array[wp.float64]):
    result[0] = wp.float64(total[0])/wp.float64(count)
    result[1] = wp.float64(high[0])
    result[2] = wp.float64(iterations[0])
    result[3] = wp.float64(active[0])


@wp.kernel
def advance(proposed: wp.array[float], active: wp.array[int], p: wp.array[float]):
    i = wp.tid()
    if active[0] != 0:
        p[i] = proposed[i]


@wp.kernel
def extrapolate(iterations: wp.array[int], active: wp.array[int], old: wp.array[float], p: wp.array[float]):
    i = wp.tid()
    value = p[i]
    if active[0] != 0:
        # Accelerated proximal gradient with t_k=(k+2)/2. Intermediate
        # extrapolates may leave p>=0; only the subsequently projected,
        # nonnegative iterate is checked for convergence or integrated.
        k = float(iterations[0])
        p[i] = value+(k-1.)/(k+2.)*(value-old[i])
    old[i] = value


class PressureGraphs:
    def __init__(self, water):
        self.water, self.graphs = water, {}
        device = water.config.device
        self.active = wp.ones(1, dtype=int, device=device)
        self.iterations = wp.zeros(1, dtype=int, device=device)
        self.running = wp.zeros(1, dtype=int, device=device)
        self.verify = wp.zeros(1, dtype=int, device=device)
        self.readout = wp.zeros(4, dtype=wp.float64, device=device)
        self.total = wp.zeros(1, dtype=float, device=device)
        self.partial = wp.zeros((water.model.particle_count+255)//256, dtype=float, device=device)
        self.maxima = wp.zeros_like(self.partial)
        self.high = wp.zeros(1, dtype=float, device=device)
        self.old_pressure = wp.zeros(water.model.particle_count, dtype=float, device=device)

    def project(self, velocity, density_mode):
        if self.water.config.pressure_graph_chunk:
            from .pressure_unrolled import project
            return project(self, velocity, density_mode)
        from .sph_kernels import pressure_velocity, plane_contact, jacobi, apply_velocity

        w, cfg = self.water, self.water.config
        cache = w.neighbor_cache
        if cache is not None:
            from .neighbor_cache import pressure_velocity, jacobi
            if cfg.recompute_gradients:
                from .neighbor_cache import pressure_velocity_positions as pressure_velocity, jacobi_positions as jacobi
                cached_geometry = [cache.ids, w.constraint_positions, cfg.support_m, w.kernel_kind]
            else:
                cached_geometry = [cache.ids, cache.gradients]
        tolerance = cfg.density_tolerance if density_mode else cfg.divergence_tolerance
        previous = w.warm_density if density_mode else w.warm_divergence
        key = (w.constraint_positions.ptr, velocity.ptr, bool(density_mode))
        if key not in self.graphs:
            # Warm module compilation before entering CUDA capture.
            wp.load_module(module=pressure_velocity.module, device=cfg.device)
            wp.load_module(module=initialize.module, device=cfg.device)
            with wp.ScopedCapture(device=cfg.device) as capture:
                w._launch(start_pressure, [previous, int(cfg.warm_start), cfg.warm_start_fraction], [w.p])
                w._launch(clear_disabled_rows, [w.neighbors, w.bg, int(density_mode), w.factor], [w.p])
                if cfg.warm_start:
                    w._launch(gate_warm_pressure, [w.grid.id, w.constraint_positions, velocity,
                        w.volume_per_particle_m3, cfg.support_m, w.kernel_kind, w.bg,
                        w.boundary_rate, w.rho, cfg.dt_s, int(density_mode)], [w.p])
                wp.copy(self.old_pressure, w.p)
                wp.launch(initialize, dim=1, inputs=[self.active, self.iterations, self.running], device=cfg.device)
                def evaluate():
                    if cache is None:
                        w._launch(pressure_velocity, [w.grid.id, w.constraint_positions,
                            w.volume_per_particle_m3, cfg.support_m, w.p, w.bg, self.active, w.kernel_kind], [w.dv])
                    else:
                        w._launch(pressure_velocity, [*cached_geometry, cache.counts,
                            w.volume_per_particle_m3, w.p, w.bg, self.active], [w.dv])
                    if w.ground:
                        w._launch(plane_contact, [w.state.particle_q, velocity, cfg.dt_s,
                            cfg.spacing_m*cfg.wall_clearance_fraction, self.active], [w.dv])
                    if w.boundaries is not None:
                        w.boundaries.contact_projection(w, velocity, self.active)
                    if cache is None:
                        w._launch(jacobi, [w.grid.id, w.constraint_positions, velocity, w.dv,
                            w.volume_per_particle_m3, cfg.support_m, w.bg, w.rho, w.factor,
                            w.neighbors, cfg.dt_s, int(density_mode), w.p, self.active, w.kernel_kind, w.boundary_rate], [w.next_p, w.residual])
                    else:
                        w._launch(jacobi, [*cached_geometry, cache.counts, w.volume_per_particle_m3, velocity, w.dv,
                            w.bg, w.rho, w.factor, w.neighbors, cfg.dt_s, int(density_mode), w.p,
                            self.active, w.boundary_rate], [w.next_p, w.residual])
                def reduce_residual():
                    # Fixed storage: CUB's temporary allocation is forbidden
                    # inside CUDA conditional nodes. A deterministic two-stage
                    # reduction retains the same mean-residual criterion.
                    wp.launch(partial_sum, dim=(len(self.partial), 256), block_dim=256,
                        inputs=[w.residual, self.partial, self.maxima], device=cfg.device)
                    wp.launch(finish_sum, dim=1, inputs=[self.partial, self.maxima, self.total, self.high], device=cfg.device)
                def verify_candidate():
                    # Only this full evaluation of NONNEGATIVE p can accept
                    # an accelerated solution. Its dv/reactions remain the
                    # arrays used by integration; no extrapolate is applied.
                    evaluate()
                    w._launch(projected_residual, [w.p, w.next_p, w.factor, cfg.dt_s], [w.residual])
                    reduce_residual()
                    wp.launch(accept_candidate, dim=1, inputs=[self.total, self.high, w.model.particle_count,
                        tolerance, cfg.maximum_projection_error if cfg.maximum_projection_error is not None else float('inf'),
                        self.iterations, self.active], device=cfg.device)
                def iteration_body():
                    evaluate()
                    if cfg.accelerated_pressure:
                        w._launch(projected_residual, [w.p, w.next_p, w.factor, cfg.dt_s], [w.residual])
                        reduce_residual()
                        wp.copy(w.p, w.next_p)
                        wp.launch(select_verification, dim=1, inputs=[self.total, self.high, w.model.particle_count,
                            tolerance, cfg.maximum_projection_error if cfg.maximum_projection_error is not None else float('inf'),
                            cfg.max_iterations, self.iterations, self.verify], device=cfg.device)
                        wp.capture_if(self.verify, verify_candidate)
                        wp.launch(loop_status, dim=1, inputs=[self.active, self.iterations, cfg.max_iterations, self.running], device=cfg.device)
                        w._launch(extrapolate, [self.iterations, self.active], [self.old_pressure, w.p])
                    else:
                        reduce_residual()
                        wp.launch(decide, dim=1, inputs=[self.total, self.high, w.model.particle_count,
                        tolerance, cfg.maximum_projection_error if cfg.maximum_projection_error is not None else float('inf'),
                        cfg.max_iterations, self.active, self.iterations, self.running], device=cfg.device)
                        w._launch(advance, [w.next_p, self.active], [w.p])
                wp.capture_while(self.running, iteration_body)
                w._launch(apply_velocity, [velocity, w.dv], [w.velocity])
                wp.copy(previous, w.p)
                wp.launch(readout, dim=1, inputs=[self.total, self.high, w.model.particle_count,
                    self.active, self.iterations], outputs=[self.readout], device=cfg.device)
            self.graphs[key] = capture.graph
        wp.capture_launch(self.graphs[key])
        mean, maximum, iterations, active = self.readout.numpy()
        if not np.isfinite([mean, maximum]).all():
            raise FloatingPointError("Nonfinite DFSPH pressure residual")
        if active:
            raise FloatingPointError(f"DFSPH projection did not converge: mean={mean:.6g}, max={maximum:.6g}")
        w.last_projection["density" if density_mode else "divergence"] = {
            "iterations": int(iterations), "mean_residual": float(mean),
            "max_residual": float(maximum)}
        return w.velocity
