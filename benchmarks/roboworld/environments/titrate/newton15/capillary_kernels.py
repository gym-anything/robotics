"""Warp implementation of the documented surface-energy gradient."""

import warp as wp


@wp.kernel
def force_predictor(grid: wp.uint64, q: wp.array[wp.vec3], v: wp.array[wp.vec3],
                    mass: wp.array[float], volume: wp.array[float], h: float,
                    sigma: float, dt: float, hr: float, beta: float, energy_length: float, predicted_v: wp.array[wp.vec3],
                    forces: wp.array[wp.vec3]):
    i = wp.tid()
    accum = wp.vec3(0.0)
    neighbors = wp.hash_grid_query(grid, q[i], h)
    j = int(0)
    while wp.hash_grid_query_next(neighbors, j):
        d = q[i] - q[j]
        distance = wp.length(d)
        if j != i and distance < h:
            a = 1.0 - distance / h
            gradient = (-210.0 / (wp.pi * h*h*h*h*h)) * a*a*a * d
            if hr > 0.0 and distance < hr:
                ar = 1.0-distance/hr
                gradient -= beta*(-210.0 / (wp.pi * hr*hr*hr*hr*hr)) * ar*ar*ar*d
            accum += volume[j] * gradient
    moment = 5.0*(h-beta*hr)/48.0
    if energy_length > 0.0:
        moment = energy_length
    force = (2.0 * sigma / moment) * volume[i] * accum
    forces[i] = force
    predicted_v[i] = v[i] + dt * force / mass[i]
