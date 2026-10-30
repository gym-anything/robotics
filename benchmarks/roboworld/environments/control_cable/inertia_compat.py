"""Keep valid product-scale inertias from being inflated by Newton's 1e-10 floor.

Newton 1.5's compiler adds 1e-6 kg m^2 to a tensor with any principal
moment below 1e-10. This is inappropriate for a valid sub-gram ferrule.
We retain all compiler warnings and restore only independently validated
original tensors. Invalid tensors still fail; no solver source is patched.
"""
import warnings
import numpy as np

def preserve_small_inertias(builder,model):
    original=np.asarray(builder.body_inertia,dtype=np.float64).reshape(-1,3,3)
    actual=model.body_inertia.numpy(); inverse=model.body_inv_inertia.numpy()
    records=[]
    for i,(tensor,mass) in enumerate(zip(original,builder.body_mass)):
        if mass<=0:continue
        symmetric=(tensor+tensor.T)/2
        values=np.linalg.eigvalsh(symmetric)
        if not np.isfinite(tensor).all() or not np.allclose(tensor,symmetric,rtol=1e-6,atol=1e-18):
            raise ValueError(f"Invalid inertia symmetry: {builder.body_label[i]}")
        if values[0]<=0 or values[0]+values[1]<values[2]*(1-2e-6):
            raise ValueError(f"Nonphysical inertia: {builder.body_label[i]} {values}")
        if not np.allclose(actual[i],tensor,rtol=1e-5,atol=1e-18):
            if values[0]>=1e-10:
                raise ValueError(f"Unexpected compiler inertia correction: {builder.body_label[i]}")
            records.append(dict(body=builder.body_label[i],principal_moments_kg_m2=values.tolist(),
                compiler_tensor_kg_m2=actual[i].tolist(),restored_tensor_kg_m2=tensor.tolist()))
            actual[i]=tensor
            inverse[i]=np.linalg.inv(symmetric)
    if records:
        warnings.warn("Restored independently validated small-body inertias after Newton 1.5 absolute-floor correction: "
            +", ".join(r["body"] for r in records),stacklevel=2)
        model.body_inertia.assign(actual);model.body_inv_inertia.assign(inverse)
    # Establish the actual arrays used by the solver, including inverse scaling.
    check=np.einsum("bij,bjk->bik",model.body_inertia.numpy(),model.body_inv_inertia.numpy())
    active=np.asarray(builder.body_mass)>0
    if not np.allclose(check[active],np.eye(3),rtol=5e-4,atol=5e-4):
        raise ValueError("Body inertia/inverse inconsistency")
    return records
