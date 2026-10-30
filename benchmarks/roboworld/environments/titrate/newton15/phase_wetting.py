"""Cubic wall free energy -sigma cos(theta) (3c^2-2c^3), exact quadrature force.

Coupled with the double-well diffuse-interface energy. This experimental
particle quadrature is not claimed to implement a full Cahn-Hilliard solver.
"""

import warp as wp
from .sph_kernels import w, grad


@wp.kernel
def wall_derivative(grid: wp.uint64, q: wp.array[wp.vec3], volumes: wp.array[float],
                    points: wp.array[wp.vec3], areas: wp.array[float],
                    h: float, sigma_cos: float, weighted_area: int, derivative: wp.array[float]):
    b = wp.tid()
    c = float(0.0)
    for i in wp.hash_grid_query(grid, points[b], h):
        c += 2.0*volumes[i]*w(points[b]-q[i], h)
    derivative[b] = -sigma_cos*areas[b]*6.0*c*(1.0-c)
    if weighted_area != 0:
        # Same integrated phase weight as weighted-area liquid/air energy:
        # H(c)=10c^3-15c^4+6c^5, H'=30c^2(1-c)^2; H(0)=0,H(1)=1.
        c = wp.clamp(c, 0., 1.)
        derivative[b] = -sigma_cos*areas[b]*30.*c*c*(1.-c)*(1.-c)


@wp.kernel
def force(grid: wp.uint64, points: wp.array[wp.vec3], derivative: wp.array[float],
          q: wp.array[wp.vec3], volumes: wp.array[float], mass: wp.array[float],
          h: float, dt: float, velocity: wp.array[wp.vec3], forces: wp.array[wp.vec3],
          wall_forces: wp.array[wp.vec3]):
    i = wp.tid()
    total = wp.vec3(0.0)
    for b in wp.hash_grid_query(grid, q[i], h):
        f = 2.0*volumes[i]*derivative[b]*grad(points[b]-q[i], h)
        total += f
        wp.atomic_sub(wall_forces, b, f)
    forces[i] += total
    velocity[i] += dt*total/mass[i]


def apply(predictor, state, dt):
    import numpy as np
    wall = predictor.wall
    h = predictor.support_m
    device = predictor.model.device
    wall.grid.build(wall.positions, h)
    grid = wp.HashGrid(64, 64, 64, device=device)
    grid.build(state.particle_q, h)
    derivative = wp.empty(len(wall.positions), dtype=float, device=device)
    wp.launch(wall_derivative, dim=len(wall.positions), inputs=[grid.id, state.particle_q,
        predictor.volumes, wall.positions, wall.areas, h,
        predictor.sigma_n_m*np.cos(np.deg2rad(wall.angle_deg)), int(getattr(predictor, "weighted_area", False))], outputs=[derivative], device=device)
    wall.forces.zero_()
    wp.launch(force, dim=len(state.particle_q), inputs=[wall.grid.id, wall.positions, derivative,
        state.particle_q, predictor.volumes, predictor.model.particle_mass, h, dt],
        outputs=[predictor.predicted.particle_qd, predictor.forces, wall.forces], device=device)
