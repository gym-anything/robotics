"""Warp operators for a Newton Model/State DFSPH extension, in SI units.

Pressure projection follows Bender & Koschier (SCA 2015), not native MPM.
The two projections constrain predicted density and velocity divergence;
particles move only by the resulting velocity, without position repair.
"""

import warp as wp
from .sph_kernel_functions import cubic_w, cubic_grad, cubic_plane_fraction, cubic_plane_gradient


@wp.func
def w(r: wp.vec3, h: float):
    u = wp.length(r)/h
    s = wp.max(1.0-u, 0.0)
    return 21.0/(2.0*wp.pi*h*h*h)*s*s*s*s*(1.0+4.0*u)


@wp.func
def grad(r: wp.vec3, h: float):
    s = wp.max(1.0-wp.length(r)/h, 0.0)
    return (-210.0/(wp.pi*h*h*h*h*h)*s*s*s)*r


@wp.func
def plane_fraction(z: float, h: float):
    # Exact integral of the normalized Wendland kernel in the half-space
    # z'<0. This is a boundary volume map, not a particle position clamp.
    s = wp.clamp(z/h, -1.0, 1.0)
    a = wp.abs(s)
    value = 0.5-21.0*(a/14.0-a*a*a/6.0+wp.pow(a, 5.0)/2.0
                     -2.0*wp.pow(a, 6.0)/3.0+5.0*wp.pow(a, 7.0)/14.0
                     -wp.pow(a, 8.0)/14.0)
    if s < 0.0:
        value = 1.0-value
    return value


@wp.func
def plane_gradient(z: float, h: float):
    a = wp.min(wp.abs(z/h), 1.0)
    value = -21.0/h*(1.0/14.0-a*a/2.0+2.5*wp.pow(a, 4.0)
                      -4.0*wp.pow(a, 5.0)+2.5*wp.pow(a, 6.0)
                      -4.0*wp.pow(a, 7.0)/7.0)
    return wp.vec3(0.0, 0.0, value)


@wp.func
def pressure_w(r: wp.vec3, h: float, kind: int):
    if kind == 1:
        return cubic_w(r, h)
    return w(r, h)


@wp.func
def pressure_grad(r: wp.vec3, h: float, kind: int):
    if kind == 1:
        return cubic_grad(r, h)
    return grad(r, h)


@wp.func
def disabled_pressure_row(factor: float, density_mode: int, support_neighbors: int):
    # DFSPH's deficiency test counts neighbours, not the magnitude of a
    # dimensional boundary gradient. A few distant solid samples must not
    # enable a bulk-divergence constraint on an isolated liquid parcel.
    # Density/contact constraints remain active independently.
    return factor == 0. or (density_mode == 0 and support_neighbors < 20)


@wp.kernel
def prepare(grid: wp.uint64, q: wp.array[wp.vec3], volume: float, h: float,
            plane: int, kernel_kind: int, ordered: int, density: wp.array[float], factor: wp.array[float],
            boundary_gradient: wp.array[wp.vec3], neighbors: wp.array[int],
            fluid_gradient: wp.array[wp.vec3]):
    i = wp.tid()
    if ordered != 0:
        i = wp.hash_grid_point_id(grid, i)
    x = q[i]
    rho = float(0.0)
    total_grad = wp.vec3(0.0)
    denom = float(0.0)
    count = int(0)
    for j in wp.hash_grid_query(grid, x, h):
        r = x-q[j]
        if wp.length_sq(r) < h*h:
            rho += volume*pressure_w(r, h, kernel_kind)
            g = volume*pressure_grad(r, h, kernel_kind)
            total_grad += g
            denom += wp.length_sq(g)
            count += 1
    gb = wp.vec3(0.0)
    if plane != 0:
        rho += plane_fraction(x[2], h)
        gb = plane_gradient(x[2], h)
        if kernel_kind == 1:
            rho += cubic_plane_fraction(x[2], h)-plane_fraction(x[2], h)
            gb = cubic_plane_gradient(x[2], h)
    fluid_gradient[i] = total_grad
    total_grad += gb
    denom += wp.length_sq(total_grad)
    density[i] = rho
    factor[i] = 0.0
    if denom > 1.0e-6:
        factor[i] = 1.0/denom
    boundary_gradient[i] = gb
    count -= 1  # The sample itself is not a neighbour.
    if plane != 0:
        # Analytic ground has no discrete sample list. Integrate its spherical
        # support volume in equivalent fluid-volume samples (a spherical cap).
        z = wp.clamp(x[2], -h, h)
        count += int(wp.pi*(h-z)*(h-z)*(2.*h+z)/(3.*volume))
    neighbors[i] = count


@wp.kernel
def nonpressure(grid: wp.uint64, q: wp.array[wp.vec3], v: wp.array[wp.vec3],
                volume: float, h: float, viscosity: float, density: wp.array[float],
                gravity: wp.vec3, dt: float, kernel_kind: int, ordered: int, out: wp.array[wp.vec3]):
    i = wp.tid()
    if ordered != 0:
        i = wp.hash_grid_point_id(grid, i)
    a = gravity
    for j in wp.hash_grid_query(grid, q[i], h):
        r = q[i]-q[j]
        # Standard angular-momentum-conserving SPH viscous stress operator.
        if wp.length_sq(r) < h*h:
            a += 10.0*viscosity*volume/wp.max(density[j], 1.0e-6)*wp.dot(v[i]-v[j], r)/(wp.length_sq(r)+0.01*h*h)*pressure_grad(r, h, kernel_kind)
    out[i] = v[i]+dt*a


@wp.kernel
def pressure_velocity(grid: wp.uint64, q: wp.array[wp.vec3], volume: float,
                      h: float, p: wp.array[float], bg: wp.array[wp.vec3],
                      active: wp.array[int], kernel_kind: int,
                      out: wp.array[wp.vec3]):
    i = wp.tid()
    if active[0] == 0:
        return
    dv = -p[i]*bg[i]
    for j in wp.hash_grid_query(grid, q[i], h):
        dv -= volume*(p[i]+p[j])*pressure_grad(q[i]-q[j], h, kernel_kind)
    out[i] = dv


@wp.kernel
def jacobi(grid: wp.uint64, q: wp.array[wp.vec3], v: wp.array[wp.vec3],
           dv: wp.array[wp.vec3], volume: float, h: float, bg: wp.array[wp.vec3],
           rho: wp.array[float], factor: wp.array[float], neighbors: wp.array[int],
           dt: float, density_mode: int, p: wp.array[float],
           active: wp.array[int], kernel_kind: int, boundary_rate: wp.array[float],
           next_p: wp.array[float], residual: wp.array[float]):
    i = wp.tid()
    if active[0] == 0:
        return
    if disabled_pressure_row(factor[i], density_mode, neighbors[i]):
        next_p[i] = 0.
        residual[i] = 0.
        return
    vi = v[i]+dv[i]
    rate = wp.dot(vi, bg[i])-boundary_rate[i]
    for j in wp.hash_grid_query(grid, q[i], h):
        rate += volume*wp.dot(vi-v[j]-dv[j], pressure_grad(q[i]-q[j], h, kernel_kind))
    error = dt*rate
    if density_mode != 0:
        error += rho[i]-1.0
    # Wendland support has a different spectrum from the paper's cubic
    # kernel. For the 3-spacing initial lattice lambda_max(D^-1 A)=5.878,
    # so omega=.5 is unstable; omega=.25 is below 2/lambda_max=.3402.
    next_p[i] = wp.max(p[i]+0.25*factor[i]*error/dt, 0.0)
    residual[i] = wp.max(error, 0.0)


@wp.kernel
def apply_velocity(v: wp.array[wp.vec3], dv: wp.array[wp.vec3],
                   result: wp.array[wp.vec3]):
    i = wp.tid()
    result[i] = v[i]+dv[i]


@wp.kernel
def plane_contact(q: wp.array[wp.vec3], v: wp.array[wp.vec3], dt: float,
                  radius: float, active: wp.array[int], dv: wp.array[wp.vec3]):
    i = wp.tid()
    if active[0] == 0:
        return
    # Unilateral, zero-restitution finite-volume sample contact. Constrain
    # predicted normal velocity before integration, never repair positions.
    # Kernel volume maps alone admit a dispersed sub-resolution monolayer.
    change = dv[i]
    lower_velocity = (radius-q[i][2])/dt
    change[2] += wp.max(lower_velocity-v[i][2]-change[2], 0.0)
    dv[i] = change


@wp.kernel
def integrate(q: wp.array[wp.vec3], v: wp.array[wp.vec3], dt: float,
              out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = q[i]+dt*v[i]


@wp.kernel
def pair_wall(q: wp.array[wp.vec3], v: wp.array[wp.vec3], h: float, dt: float,
              sigma: float, density: float, cos_angle: float, repulsive_h: float, beta: float, wall_moment: float,
              out: wp.array[wp.vec3]):
    i = wp.tid()
    # Half-space integral of the SAME normalized kernel as fluid cohesion.
    # Missing fluid neighbors cost sigma per unit planar interface area.
    # Solid adhesion strength lambda*(1+cos(theta)) gives Young's energy.
    lam = sigma/(5.0*(h-beta*repulsive_h)/48.0)
    if wall_moment > 0.0:
        lam = sigma/wall_moment
    gradient = plane_gradient(q[i][2], h)
    if repulsive_h > 0.0:
        gradient -= beta*plane_gradient(q[i][2], repulsive_h)
    a = lam*(1.0+cos_angle)/density*gradient
    out[i] = v[i]+dt*a


@wp.kernel
def mirror_cohesion(grid: wp.uint64, q: wp.array[wp.vec3], volume: float,
                    h: float, sigma: float, density: float, dt: float,
                    hr: float, beta: float,
                    velocity: wp.array[wp.vec3]):
    i = wp.tid()
    reflected = wp.vec3(q[i][0], q[i][1], -q[i][2])
    force = wp.vec3(0.0)
    for j in wp.hash_grid_query(grid, reflected, h):
        image_j = wp.vec3(q[j][0], q[j][1], -q[j][2])
        force += grad(q[i]-image_j, h)
        if hr > 0.0:
            force -= beta*grad(q[i]-image_j, hr)
    # Derivative of the mirrored pair energy in the fluid half-space.
    # This supplies a neutral wall extension, with a separate Young traction.
    velocity[i] += dt*2.0*sigma/(5.0*(h-beta*hr)/48.0)*volume/density*force
