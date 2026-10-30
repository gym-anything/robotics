"""Local, inverse-mass normal corrections before the final conservative CCD.

This is an operator-split contact projection, not a replacement textile solver.
The zero-distance constraints retain the side at the start of the step. Only
normal closing displacement is corrected; unconstrained particles are untouched.
Final exact CCD and the existing crossing audit still determine acceptance.
"""
import numpy as np


def ccd_witness_collides(mesh, origin, proposed, witness, minimum_distance=0.):
    """Test one prior EE/FV witness; a miss never certifies the whole mesh.

    Require swept AABB overlap just as the complete broadphase does, then use
    IPC's unchanged narrow-phase CCD. The immutable topology/filter is shared
    with the contact list that supplied this witness. Final acceptance still
    computes the complete collision-free fraction on the corrected proposal.
    """
    import ipctk
    if not witness or witness["kind"] not in ("EdgeEdgeCandidate","FaceVertexCandidate"):
        return False
    candidate=getattr(ipctk,witness["kind"])(*witness["ids"])
    a=candidate.dof(origin,mesh.edges,mesh.faces).reshape(-1,3)
    b=candidate.dof(proposed,mesh.edges,mesh.faces).reshape(-1,3)
    # IPC FV dofs are ordered point, triangle[0:3], not face then point.
    split=2 if witness["kind"]=="EdgeEdgeCandidate" else 1
    first=np.vstack((a[:split],b[:split])); second=np.vstack((a[split:],b[split:]))
    if np.any(first.min(0)>second.max(0)+minimum_distance) or np.any(second.min(0)>first.max(0)+minimum_distance):
        return False
    collides,fraction=candidate.ccd(a.ravel(),b.ravel(),minimum_distance)
    return bool(collides and fraction < 1.-1e-12)


def _witness(stencil):
    if hasattr(stencil,"edge0_id"):
        return dict(kind="EdgeEdgeCandidate",ids=[int(stencil.edge0_id),int(stencil.edge1_id)])
    if hasattr(stencil,"face_id"):
        return dict(kind="FaceVertexCandidate",ids=[int(stencil.face_id),int(stencil.vertex_id)])
    return None


def project(mesh, collisions, origin, proposed, inverse_mass, *, iterations=128, clearance_m=2e-6,
            table=None):
    stencils=(list(collisions.vv_collisions)+list(collisions.ev_collisions)+
              list(collisions.ee_collisions)+list(collisions.fv_collisions))
    rows=[]; active=[]
    # Nanobind exposes Eigen topology as NumPy arrays. Materialize each once,
    # not once per stencil; these arrays are immutable during this projection.
    edges,faces=mesh.edges,mesh.faces
    for s in stencils:
        ids=np.array([v for v in s.vertex_ids(edges,faces) if v>=0],dtype=int)
        dof=origin[ids].ravel()
        gap=float(np.sqrt(max(0,s.compute_distance(dof))))
        if gap < 1e-12: continue
        g=np.asarray(s.compute_distance_gradient(dof)).reshape(-1,3)/(2*gap)
        rows.append((ids,g,gap))
        active.append(s)
    if not rows: return proposed,dict(iterations=0,corrected_vertices=0,maximum_correction_m=0.)
    indices=np.zeros((len(rows),4),int); gradient=np.zeros((len(rows),4,3)); gaps=[]
    for i,(ids,g,gap) in enumerate(rows):
        indices[i,:len(ids)]=ids; gradient[i,:len(ids)]=g
        # Below 0.1 mm (well inside the cloth's native thickness contact),
        # prohibit further normal closure. Repeatedly retaining only 20% of
        # a micrometre gap otherwise drives the final CCD into global damping.
        # Two micrometres resolves the measured float32/CCD residual after
        # zero-clearance projection (CCD's default tolerance is one micrometre).
        # This is a numerical margin, not a replacement textile thickness.
        gaps.append(.8*max(gap-.0001,0.)-clearance_m)
    weights=inverse_mass[indices]
    gaps=np.asarray(gaps)
    denominator=np.sum(weights*np.sum(gradient**2,axis=2),axis=1)
    move=proposed-origin; revised=move.copy(); count=0
    table_correction=np.zeros(len(origin))
    witness=None
    if table is not None:
        from .newton_style3d_table_ccd import table_top_face_domain
        table_radius=np.broadcast_to(table['radius'],(len(origin),))
        def table_clearance(ids):
            endpoint=origin[ids]+revised[ids]
            inside=table_top_face_domain(endpoint,half_x=table['half_x'],half_y=table['half_y'],
                top=table['top'],bottom=table['bottom'])
            radius=table_radius[ids]
            return np.where(inside,endpoint[:,2]-table['top']-radius-table['epsilon'],np.inf)
        clearance=table_clearance(np.arange(len(origin)))
    # Project the most violated local half-space in the mass metric. Unlike
    # whole-garment time scaling, a pocket contact cannot arrest both legs.
    for _ in range(iterations):
        value=gaps+np.einsum('ijk,ijk->i',gradient,revised[indices])
        row=int(np.argmin(value))
        if table is not None:
            vertex=int(np.argmin(clearance))
            if clearance[vertex] < min(float(value[row]),-1e-12):
                correction=-clearance[vertex]
                revised[vertex,2]+=correction; table_correction[vertex]+=correction
                # A top-normal upward correction stays in the top-face domain.
                # Only this endpoint changed; retain every other cached value.
                radius=table_radius[vertex]
                clearance[vertex]=(origin[vertex,2]+revised[vertex,2])-table['top']-radius-table['epsilon']
                count+=1
                continue
        if value[row] >= -1e-12: break
        if denominator[row] <= 0: break
        if witness is None: witness=_witness(active[row])
        impulse=-value[row]/denominator[row]
        np.add.at(revised,indices[row],impulse*weights[row,:,None]*gradient[row])
        if table is not None:
            changed=np.unique(indices[row])
            clearance[changed]=table_clearance(changed)
        count+=1
    delta=revised-move
    return origin+revised,dict(iterations=count,corrected_vertices=int(np.count_nonzero(np.linalg.norm(delta,axis=1)>1e-12)),
        maximum_correction_m=float(np.linalg.norm(delta,axis=1).max()),
        table_corrected_vertices=int(np.count_nonzero(table_correction)),
        summed_table_normal_correction_m=float(table_correction.sum()),
        ccd_witness=witness,
        linear_residual_m=float(min(0,(gaps+np.einsum('ijk,ijk->i',gradient,revised[indices])).min())))
