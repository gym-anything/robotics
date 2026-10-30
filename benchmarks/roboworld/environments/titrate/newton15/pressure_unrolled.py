"""Fixed launch graphs for the same guarded accelerated pressure iteration.

On hosts with expensive conditional CUDA nodes, submit small unrolled graphs.
Inactive kernels do not update state; neither the minimum/maximum iteration
counts nor the residual criteria change. Both extrapolate and feasible-candidate
operators are scheduled, but candidate kernels execute only when requested.
This is an explicit scheduling option, not a lower-accuracy pressure solver.
"""

import numpy as np
import warp as wp
from . import pressure_graph as pg


@wp.kernel
def residual(p: wp.array[float], proposed: wp.array[float], factor: wp.array[float],
             dt: float, enabled: wp.array[int], out: wp.array[float]):
    i = wp.tid()
    if enabled[0] != 0 and factor[i] > 0.:
        out[i] = wp.abs(p[i]-proposed[i])*dt/(.25*factor[i])


@wp.kernel
def select(total: wp.array[float], high: wp.array[float], count: int,
           tolerance: float, maximum_error: float, maximum: int,
           running: wp.array[int], iterations: wp.array[int], verify: wp.array[int]):
    verify[0] = 0
    if running[0] != 0:
        iterations[0] += 1
        verify[0] = int((iterations[0] >= 3 and total[0]/float(count) <= tolerance
            and high[0] <= maximum_error) or iterations[0] >= maximum)


@wp.kernel
def accept(total: wp.array[float], high: wp.array[float], count: int,
           tolerance: float, maximum_error: float, verify: wp.array[int], active: wp.array[int]):
    if verify[0] != 0 and total[0]/float(count) <= tolerance and high[0] <= maximum_error:
        active[0] = 0


def project(g, velocity, density_mode):
    from .sph_kernels import pressure_velocity, plane_contact, jacobi, apply_velocity
    w, cfg = g.water, g.water.config
    cache = w.neighbor_cache
    geometry = None
    if cache is not None:
        from .neighbor_cache import pressure_velocity, jacobi
        geometry = [cache.ids, cache.gradients]
        if cfg.recompute_gradients:
            from .neighbor_cache import pressure_velocity_positions as pressure_velocity, jacobi_positions as jacobi
            geometry = [cache.ids, w.constraint_positions, cfg.support_m, w.kernel_kind]
    tolerance = cfg.density_tolerance if density_mode else cfg.divergence_tolerance
    maximum_error = cfg.maximum_projection_error if cfg.maximum_projection_error is not None else float('inf')
    previous = w.warm_density if density_mode else w.warm_divergence
    key = (w.constraint_positions.ptr, velocity.ptr, bool(density_mode))
    if key not in g.graphs:
        for kernel in (pressure_velocity, pg.initialize, residual):
            wp.load_module(module=kernel.module, device=cfg.device)

        def evaluate(enabled):
            if cache is None:
                w._launch(pressure_velocity, [w.grid.id, w.constraint_positions, w.volume_per_particle_m3,
                    cfg.support_m, w.p, w.bg, enabled, w.kernel_kind], [w.dv])
            else:
                w._launch(pressure_velocity, [*geometry, cache.counts, w.volume_per_particle_m3,
                    w.p, w.bg, enabled], [w.dv])
            if w.ground:
                w._launch(plane_contact, [w.state.particle_q, velocity, cfg.dt_s,
                    cfg.spacing_m*cfg.wall_clearance_fraction, enabled], [w.dv])
            if w.boundaries is not None:
                w.boundaries.contact_projection(w, velocity, enabled)
            if cache is None:
                w._launch(jacobi, [w.grid.id, w.constraint_positions, velocity, w.dv,
                    w.volume_per_particle_m3, cfg.support_m, w.bg, w.rho, w.factor, w.neighbors,
                    cfg.dt_s, int(density_mode), w.p, enabled, w.kernel_kind, w.boundary_rate],
                    [w.next_p, w.residual])
            else:
                w._launch(jacobi, [*geometry, cache.counts, w.volume_per_particle_m3, velocity, w.dv,
                    w.bg, w.rho, w.factor, w.neighbors, cfg.dt_s, int(density_mode), w.p, enabled,
                    w.boundary_rate], [w.next_p, w.residual])
            w._launch(residual, [w.p, w.next_p, w.factor, cfg.dt_s, enabled], [w.residual])
            wp.launch(pg.partial_sum, dim=(len(g.partial), 256), block_dim=256,
                inputs=[w.residual, g.partial, g.maxima], device=cfg.device)
            wp.launch(pg.finish_sum, dim=1, inputs=[g.partial, g.maxima, g.total, g.high], device=cfg.device)

        with wp.ScopedCapture(device=cfg.device) as init:
            w._launch(pg.start_pressure, [previous, int(cfg.warm_start), cfg.warm_start_fraction], [w.p])
            w._launch(pg.clear_disabled_rows, [w.neighbors, w.bg, int(density_mode), w.factor], [w.p])
            if cfg.warm_start:
                w._launch(pg.gate_warm_pressure, [w.grid.id, w.constraint_positions, velocity,
                    w.volume_per_particle_m3, cfg.support_m, w.kernel_kind, w.bg, w.boundary_rate,
                    w.rho, cfg.dt_s, int(density_mode)], [w.p])
            wp.copy(g.old_pressure, w.p)
            wp.launch(pg.initialize, dim=1, inputs=[g.active, g.iterations, g.running], device=cfg.device)
        with wp.ScopedCapture(device=cfg.device) as chunk:
            for _ in range(cfg.pressure_graph_chunk):
                evaluate(g.running)
                w._launch(pg.advance, [w.next_p, g.running], [w.p])
                wp.launch(select, dim=1, inputs=[g.total, g.high, w.model.particle_count,
                    tolerance, maximum_error, cfg.max_iterations, g.running, g.iterations, g.verify], device=cfg.device)
                evaluate(g.verify)
                wp.launch(accept, dim=1, inputs=[g.total, g.high, w.model.particle_count,
                    tolerance, maximum_error, g.verify, g.active], device=cfg.device)
                wp.launch(pg.loop_status, dim=1, inputs=[g.active, g.iterations, cfg.max_iterations, g.running], device=cfg.device)
                w._launch(pg.extrapolate, [g.iterations, g.running], [g.old_pressure, w.p])
            wp.launch(pg.readout, dim=1, inputs=[g.total, g.high, w.model.particle_count,
                g.active, g.iterations], outputs=[g.readout], device=cfg.device)
        with wp.ScopedCapture(device=cfg.device) as finish:
            w._launch(apply_velocity, [velocity, w.dv], [w.velocity])
            wp.copy(previous, w.p)
        g.graphs[key] = (init.graph, chunk.graph, finish.graph)
    init, chunk, finish = g.graphs[key]
    wp.capture_launch(init)
    while True:
        wp.capture_launch(chunk)
        mean, maximum, iterations, active = g.readout.numpy()
        if not np.isfinite([mean, maximum]).all():
            raise FloatingPointError('Nonfinite DFSPH pressure residual')
        if not active:
            break
        if iterations >= cfg.max_iterations:
            raise FloatingPointError(f'DFSPH projection did not converge: mean={mean:.6g}, max={maximum:.6g}')
    wp.capture_launch(finish)
    w.last_projection['density' if density_mode else 'divergence'] = {
        'iterations': int(iterations), 'mean_residual': float(mean), 'max_residual': float(maximum)}
    return w.velocity
