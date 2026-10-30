# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pinned Newton 1.5 zero-slip tangent correction, explicit experiment.

The native regularized force for u -> 0 is -2*mu*N*u/eps, but its exact-zero
branch returns K=0. The correct limit is K=2*mu*N/eps*(I-n*nT). This correction
does not change the force law at ANY displacement, including zero.
It changes the Newton linearization only, not physical Coulomb coefficients.
No installed vendor file is edited. Install before constructing any VBD solver
in this isolated environment process; never mutate an existing solver at runtime.
"""
import importlib.metadata
import warp as wp

@wp.func
def regularized_friction(mu:float,load:float,n:wp.vec3,u:wp.vec3,eps:float):
    tangent=u-n*wp.dot(n,u)
    length=wp.length(tangent)
    factor=2.0/eps
    if length>eps:factor=1.0/length
    elif length>0.0:factor=(-length/eps+2.0)/eps
    scale=mu*load*factor
    return -scale*tangent,scale*(wp.identity(3,float)-wp.outer(n,n))

def install():
    if importlib.metadata.version("newton")!="1.5.0":raise RuntimeError("Friction tangent compatibility is pinned to Newton 1.5.0")
    if importlib.metadata.version("warp-lang")!="1.16.0":raise RuntimeError("Revalidate module invalidation before changing Warp")
    from newton._src.solvers.vbd import rigid_vbd_kernels as native
    if native.compute_projected_isotropic_friction is not regularized_friction:
        native.compute_projected_isotropic_friction=regularized_friction
        # Ensure Warp recompiles callers with the new device-function dependency.
        wp.get_module(native.__name__).mark_modified()
    return dict(kind="zero_slip_tangent",force_law_changed=False,
                reference="Newton v1.5.0 rigid_vbd_kernels.py:1679",
                correction="K(0)=2*mu*N/eps*(I-n*nT), rather than zero")
