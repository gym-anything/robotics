"""Spatial conservative dilute chemistry carried by the liquid material points.

H/OH denote analytical excess acid/base, not the equilibrium water ions.
Neutralization is local. Particle motion supplies advection. Temperature is 20 C.
The default effective exchange coefficient is the user-authorized 1000x
experiment (1.5e-9 -> 1.5e-6 m^2/s), not measured molecular diffusivity.
"""

import numpy as np
from .runtime import require_newton

SPECIES = ("H", "OH", "Na", "Cl", "indicator")
KW = 10.0**-14.17
INDICATOR_PKA = 9.7  # PubChem/HSDB at 25 C; explicitly used near 20 C.


def equilibrium_ph(net_molar):
    """Solve H-OH=net, H*OH=Kw without cancellation on the alkaline branch."""
    net = np.asarray(net_molar, dtype=float)
    if not np.isfinite(net).all():
        raise ValueError("Finite analytical concentration required")
    root = np.hypot(net, 2*np.sqrt(KW))
    h = np.empty_like(net)
    acid = net >= 0
    h[acid] = .5*(net[acid]+root[acid])
    h[~acid] = 2*KW/(root[~acid]-net[~acid])
    return -np.log10(h)


class Chemistry:
    def __init__(self, count, volume_m3, device="cuda:0", diffusivity_m2_s=1.5e-6):
        _, self.wp = require_newton()
        if type(count) is not int or count < 1 or not np.isfinite([volume_m3, diffusivity_m2_s]).all() or volume_m3 <= 0 or diffusivity_m2_s <= 0:
            raise ValueError("Positive material-point count, volume and diffusivity required")
        self.volume_l = float(volume_m3*1000)
        self.volume_m3 = float(volume_m3)
        self.diffusivity = float(diffusivity_m2_s)
        self.device = device
        self.moles = self.wp.zeros((count, 5), dtype=self.wp.float64, device=device)
        self.next = self.wp.zeros_like(self.moles)
        self.reacted = self.wp.zeros(count, dtype=self.wp.float64, device=device)
        self.next_reacted = self.wp.zeros_like(self.reacted)
        self.invalid = self.wp.zeros(1, dtype=int, device=device)
        self.grid = self.wp.HashGrid(64, 64, 512, device=device)
        self.ordered_threads = True

    def charge(self, indices, *, acid_m=0., base_m=0., indicator_m=0.):
        if not np.isfinite([acid_m, base_m, indicator_m]).all() or min(acid_m, base_m, indicator_m) < 0:
            raise ValueError("Finite nonnegative initial concentrations required")
        n = self.moles.numpy()
        n[indices] = self.volume_l*np.array([acid_m, base_m, base_m, acid_m, indicator_m])
        self.moles.assign(n)

    def step(self, positions, dt, support_m):
        from .chemistry_kernels import diffuse_react

        # The specified dt is never clamped. This generous analytical bound
        # is checked before a step; actual sum-of-weights is checked in kernel.
        if not np.isfinite([dt, support_m]).all() or dt <= 0 or support_m <= 0 or dt*self.diffusivity/support_m**2 > .01:
            raise ValueError("Chemical timestep outside explicit diffusion bound")
        self.grid.build(positions, support_m)
        self.invalid.zero_()
        self.wp.launch(diffuse_react, dim=len(positions),
                       inputs=[self.grid.id, positions, self.moles, self.volume_m3,
                               support_m, dt, self.diffusivity, self.reacted, int(self.ordered_threads)],
                       outputs=[self.next, self.next_reacted, self.invalid],
                       device=self.device)
        if self.invalid.numpy()[0]:
            raise FloatingPointError("Chemical step rejected: invalid diffusion weights or inventory")
        self.moles, self.next = self.next, self.moles
        self.reacted, self.next_reacted = self.next_reacted, self.reacted

    def observe(self):
        n = self.moles.numpy()
        if not np.isfinite(n).all() or (n < -1e-25).any():
            raise FloatingPointError("Nonfinite or negative chemical inventory")
        ph = equilibrium_ph((n[:, 0]-n[:, 1])/self.volume_l)
        pink_m = n[:, 4]/self.volume_l/(1+10**(INDICATOR_PKA-ph))
        return {"moles": n, "ph": ph, "pink_molar": pink_m,
                "totals_mol": dict(zip(SPECIES, n.sum(axis=0).tolist()))}
