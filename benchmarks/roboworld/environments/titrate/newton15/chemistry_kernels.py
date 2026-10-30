"""Symmetric SPH Laplacian of analytical species; all inventories float64."""

import warp as wp

Vec5d = wp.types.vector(length=5, dtype=wp.float64)

@wp.kernel
def diffuse_react(grid: wp.uint64, q: wp.array[wp.vec3], n: wp.array2d[wp.float64],
                   volume: float, h: float, dt: float, diffusion: float,
                   previous_reacted: wp.array[wp.float64], ordered: int, result: wp.array2d[wp.float64],
                   reacted: wp.array[wp.float64], invalid: wp.array[int]):
    i = wp.tid()
    if ordered != 0:
        i = wp.hash_grid_point_id(grid, i)
    values = Vec5d(0.0)
    for s in range(5):
        values[s] = n[i, s]
    query = wp.hash_grid_query(grid, q[i], h)
    j = int(0)
    weights = float(0.0)
    while wp.hash_grid_query_next(query, j):
        if i != j:
            d = q[i]-q[j]
            r2 = wp.dot(d, d)
            r = wp.sqrt(r2)
            if r < h:
                a = 1.0-r/h
                minus_r_grad = (210.0/(wp.pi*h*h*h*h*h))*a*a*a*r2
                weight = 2.0*diffusion*dt*volume*minus_r_grad/(r2+.01*h*h)
                weights += weight
                for s in range(5):
                    values[s] += wp.float64(weight)*(n[j, s]-n[i, s])
    if weights >= 1.0 or not wp.isfinite(weights):
        wp.atomic_add(invalid, 0, 1)
    reaction = wp.min(values[0], values[1])
    values[0] -= reaction
    values[1] -= reaction
    reacted[i] = previous_reacted[i]+reaction
    for s in range(5):
        if values[s] < wp.float64(0.) or not wp.isfinite(values[s]):
            wp.atomic_add(invalid, 0, 1)
        result[i, s] = values[s]
