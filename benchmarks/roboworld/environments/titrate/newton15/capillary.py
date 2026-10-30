"""Experimental non-local surface-energy forces, in SI units.

Newton MPM handles incompressibility/viscosity; these forces supply the missing
liquid-air surface energy. No solid wetting is implemented here.
"""

import math
import numpy as np
from scipy.spatial import cKDTree


def kernel(r, h):
    q = np.asarray(r, dtype=float) / h
    return 21.0 / (2.0 * math.pi * h**3) * np.maximum(1.0-q, 0.0)**4 * (1.0+4.0*q)


def kernel_gradient(displacement, h):
    displacement = np.asarray(displacement, dtype=float)
    q = np.linalg.norm(displacement, axis=-1) / h
    factor = -210.0 / (math.pi * h**5) * np.maximum(1.0-q, 0.0)**3
    return displacement * factor[..., None]


def reference(positions, volumes, support_m, sigma_n_m, repulsive_support_m=0.0, repulsive_weight=1.0):
    """Float64 energy/force reference used only in bounded component checks."""
    q = np.asarray(positions, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    if (q.ndim != 2 or q.shape[1] != 3 or volumes.shape != (len(q),)
            or not np.isfinite(q).all() or not np.isfinite(volumes).all()
            or np.any(volumes <= 0) or not np.isfinite(support_m) or support_m <= 0
            or not np.isfinite(sigma_n_m) or sigma_n_m < 0):
        raise ValueError("Invalid capillary reference input")
    pairs = cKDTree(q).query_pairs(support_m, output_type="ndarray")
    hr = float(repulsive_support_m)
    if not 0 <= hr < support_m:
        raise ValueError("Repulsive support must be nonnegative and below attractive support")
    beta = float(repulsive_weight)
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("Repulsive weight must lie in [0,1]")
    factor = sigma_n_m / (5.0 * (support_m-beta*hr) / 48.0)
    forces = np.zeros_like(q)
    energy = -factor * float((volumes**2).sum()) * float(kernel(0.0, support_m))
    if hr:
        energy += beta*factor * float((volumes**2).sum()) * float(kernel(0.0, hr))
    if len(pairs):
        i, j = pairs.T
        d = q[i] - q[j]
        products = volumes[i] * volumes[j]
        energy -= 2.0 * factor * float((products * kernel(np.linalg.norm(d, axis=1), support_m)).sum())
        f = 2.0 * factor * products[:, None] * kernel_gradient(d, support_m)
        if hr:
            energy += beta*2.0 * factor * float((products * kernel(np.linalg.norm(d, axis=1), hr)).sum())
            f -= beta*2.0 * factor * products[:, None] * kernel_gradient(d, hr)
        np.add.at(forces, i, f)
        np.add.at(forces, j, -f)
    return energy, forces


class CapillaryPredictor:
    """Compute a momentum-conserving force predictor without editing input state."""

    def __init__(self, model, volume_m3, support_m, sigma_n_m, repulsive_support_m=0.0, repulsive_weight=1.0):
        from .runtime import require_newton

        newton, wp = require_newton()
        from .capillary_kernels import force_predictor

        self.wp = wp
        self.newton = newton
        self.force_predictor = force_predictor
        self.model = model
        self.support_m = float(support_m)
        self.sigma_n_m = float(sigma_n_m)
        self.repulsive_support_m = float(repulsive_support_m)
        self.repulsive_weight = float(repulsive_weight)
        self.energy_length_m = 0.0  # Zero selects the original continuum moment.
        if (not np.isfinite([volume_m3, support_m, sigma_n_m, repulsive_support_m]).all()
                or volume_m3 <= 0 or sigma_n_m < 0
                or not 0 <= repulsive_support_m < support_m
                or not np.isfinite(repulsive_weight) or not 0 <= repulsive_weight <= 1
                or np.any(model.particle_mass.numpy() <= 0)):
            raise ValueError("Capillary predictor needs positive-mass fluid and valid SI parameters")
        self.volumes = wp.full(model.particle_count, float(volume_m3), dtype=float, device=model.device)
        self.forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self.grid = wp.HashGrid(64, 64, 64, device=model.device)
        self.predicted = model.state()

    def predict(self, state, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Predictor timestep must be finite and positive")
        # MPM appends variable-size collider impulse arrays after its first
        # step. State.assign copies but deliberately does not allocate them.
        # Mirror the complete input array layout before the non-mutating copy.
        def synchronize_layout(dst, src):
            for name, value in vars(src).items():
                current = getattr(dst, name, None)
                if isinstance(value, self.wp.array):
                    if (not isinstance(current, self.wp.array) or current.shape != value.shape
                            or current.dtype != value.dtype):
                        setattr(dst, name, self.wp.empty_like(value))
                elif isinstance(value, self.newton.Model.AttributeNamespace):
                    if current is None:
                        current = self.newton.Model.AttributeNamespace()
                        setattr(dst, name, current)
                    synchronize_layout(current, value)
                elif value is None and isinstance(current, self.wp.array):
                    setattr(dst, name, None)
        synchronize_layout(self.predicted, state)
        self.predicted.assign(state)
        self.grid.build(state.particle_q, self.support_m)
        self.wp.launch(self.force_predictor, dim=self.model.particle_count,
                       inputs=[self.grid.id, state.particle_q, state.particle_qd,
                               self.model.particle_mass, self.volumes, self.support_m,
                               self.sigma_n_m, dt, self.repulsive_support_m, self.repulsive_weight,
                               self.energy_length_m],
                       outputs=[self.predicted.particle_qd, self.forces], device=self.model.device)
        return self.predicted
