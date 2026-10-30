"""Explicit candidate ordering for native SDF edge/face contact.

Recorded-pose diagnostic278 shows the installed GPU kernel omits an intersecting
face in mixed-shape order but emits it in shape-major order and in isolation.
This workaround preserves the exact candidate multiset and native calculation.
It is not a proof of the underlying compiler/driver cause or CCD correctness.
No installed Newton code, force law, mesh, radius or contact capacity is changed.
"""
import hashlib
import numpy as np
import warp as wp


def group_soft_pairs(pipeline):
    receipt={}
    for name in ('soft_edge_rigid_pairs','soft_face_rigid_pairs'):
        source=getattr(pipeline,name);pairs=source.numpy()
        order=np.argsort(pairs[:,1],kind='stable')
        grouped=pairs[order].copy()
        # Verify a permutation, including repeated rows, not just set equality.
        recovered=np.empty_like(grouped);recovered[order]=grouped
        if not np.array_equal(recovered,pairs):raise RuntimeError('Collision candidate multiset changed')
        setattr(pipeline,name,wp.array(grouped,dtype=wp.vec2i,device=pipeline.device))
        receipt[name]=dict(count=len(pairs),original_sha256=hashlib.sha256(pairs.tobytes()).hexdigest(),
            grouped_sha256=hashlib.sha256(grouped.tobytes()).hexdigest(),permutation_verified=True)
    pipeline.wiping_order_receipt=receipt
    return pipeline
