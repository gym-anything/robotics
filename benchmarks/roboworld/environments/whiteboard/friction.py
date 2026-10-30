"""Compiled double-precision tangent impulse sweep, with a NumPy oracle.

Same finite iteration count/order as the reference; compilation removes Python
loop overhead and does not change the timestep or physical friction coefficient.
"""

import numpy as np
import warp as wp

V12 = wp.types.vector(length=12, dtype=wp.float64)
M2x12 = wp.types.matrix(shape=(2, 12), dtype=wp.float64)
M12x2 = wp.types.matrix(shape=(12, 2), dtype=wp.float64)


@wp.kernel
def _sweep(
    initial: wp.array(dtype=V12),
    jac: wp.array(dtype=M2x12),
    effective: wp.array(dtype=wp.mat22d),
    inverse: wp.array(dtype=wp.mat22d),
    response: wp.array(dtype=M12x2),
    limits: wp.array(dtype=wp.float64),
    impulses: wp.array(dtype=wp.vec2d),
    diagnostic: wp.array(dtype=wp.float64),
):
    velocity = initial[0]
    work = wp.float64(0.0)
    last = wp.float64(0.0)
    iterations = int(0)
    for sweep in range(12):
        maximum = wp.float64(0.0)
        for j in range(limits.shape[0]):
            old = impulses[j]
            before = jac[j] * velocity
            proposed = old - inverse[j] * before
            length = wp.length(proposed)
            if length > limits[j]:
                proposed = proposed * (limits[j] / length)
            delta = proposed - old
            work = (
                work + wp.float64(0.5) * wp.dot(delta, effective[j] * delta) + wp.dot(delta, before)
            )
            velocity = velocity + response[j] * delta
            impulses[j] = proposed
            maximum = wp.max(maximum, wp.length(delta))
        iterations = sweep + 1
        last = maximum
        if maximum < wp.float64(1e-9):
            break
    diagnostic[0] = work
    diagnostic[1] = wp.float64(iterations)
    diagnostic[2] = last


def solve(initial, jac, effective, inverse, response, limits):
    if len(limits) == 0:
        return np.zeros((0, 2)), (0.0, 0, 0.0)
    arrays = [
        wp.array([initial], dtype=V12, device="cpu"),
        wp.array(jac, dtype=M2x12, device="cpu"),
        wp.array(effective, dtype=wp.mat22d, device="cpu"),
        wp.array(inverse, dtype=wp.mat22d, device="cpu"),
        wp.array(response, dtype=M12x2, device="cpu"),
        wp.array(limits, dtype=wp.float64, device="cpu"),
    ]
    impulses = wp.zeros(len(limits), dtype=wp.vec2d, device="cpu")
    diagnostic = wp.zeros(3, dtype=wp.float64, device="cpu")
    wp.launch(_sweep, dim=1, inputs=arrays, outputs=[impulses, diagnostic], device="cpu")
    return impulses.numpy(), diagnostic.numpy()


def reference(initial, jac, effective, inverse, response, limits):
    velocity = np.array(initial, copy=True)
    impulses = np.zeros((len(limits), 2))
    work = 0.0
    maximum = 0.0
    for sweep in range(12):
        maximum = 0.0
        for j in range(len(limits)):
            old = impulses[j].copy()
            v = jac[j] @ velocity
            proposed = old - inverse[j] @ v
            length = np.linalg.norm(proposed)
            if length > limits[j]:
                proposed *= limits[j] / length
            delta = proposed - old
            work += float(0.5 * delta @ (effective[j] @ delta) + delta @ v)
            velocity += response[j] @ delta
            impulses[j] = proposed
            maximum = max(maximum, float(np.linalg.norm(delta)))
        if maximum < 1e-9:
            break
    return impulses, (work, sweep + 1, maximum)


def implicit_regularized(free_velocity, mass, jacobian, limits, dt, smoothing=0.001):
    """Backward-Euler integration of the existing smooth Coulomb law.

    Minimize 1/2 ||L^T(v-v_free)||^2 + dt sum(mu*N*sqrt(|Jv|^2+eps^2)).
    The strictly convex six-dimensional problem has a unique minimizer. This
    changes time integration, not the physical friction coefficient or eps.
    Native gravity/applied loads belong in v_free, exactly once.
    """
    L = np.linalg.cholesky(mass)
    unscale = np.linalg.solve(L.T, np.eye(6))
    J = np.asarray(jacobian) @ unscale
    free = L.T @ np.asarray(free_velocity)
    y = free.copy()
    limits = np.asarray(limits)

    def evaluate(value):
        slip = J @ value
        speed = np.sqrt(np.sum(slip * slip, axis=1) + smoothing**2)
        force = -limits[:, None] * slip / speed[:, None]
        objective = 0.5 * np.sum((value - free) ** 2) + dt * np.dot(limits, speed)
        gradient = value - free - dt * np.einsum("nij,ni->j", J, force)
        return objective, gradient, slip, speed, force

    for iteration in range(60):
        objective, gradient, slip, speed, force = evaluate(y)
        residual = float(np.linalg.norm(gradient))
        if residual < 1e-12:
            return force, {
                "iterations": iteration,
                "momentum_residual_scaled": residual,
                "power_w": float(np.sum(force * slip)),
            }
        H = (
            np.eye(2)[None] / speed[:, None, None]
            - slip[:, :, None] * slip[:, None, :] / speed[:, None, None] ** 3
        )
        Hessian = np.eye(6) + dt * np.einsum("nij,nik,nkl,n->jl", J, H, J, limits)
        direction = np.linalg.solve(Hessian, -gradient)
        scale = 1.0
        slope = float(gradient @ direction)
        for _ in range(40):
            trial = y + scale * direction
            trial_objective, trial_gradient, *_ = evaluate(trial)
            # Near the minimizer, subtracting objective values loses the tiny
            # Newton decrease to rounding. Accept a strictly smaller gradient
            # within the objective's floating-point error, not a looser residual.
            roundoff = 16 * np.finfo(float).eps * max(abs(objective), np.finfo(float).tiny)
            if trial_objective <= objective + 1e-4 * scale * slope or (
                abs(trial_objective - objective) <= roundoff
                and np.linalg.norm(trial_gradient) < 0.5 * residual
            ):
                y = trial
                break
            scale *= 0.5
        else:
            raise FloatingPointError("Implicit board friction line search failed")
    raise FloatingPointError(f"Implicit board friction did not converge: {residual}")
