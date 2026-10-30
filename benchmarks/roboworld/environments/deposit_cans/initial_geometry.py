"""Fail-closed initial can/tub triangle checks, before any physics step.

This fills a separate gap from the bag/self CCD verifier. It neither projects
initial positions nor changes contact geometry. Runtime CCD remains separate.
"""
import numpy as np
import ipctk


def intersects_can_tub(can_vertices,can_faces,tub_vertices,tub_faces):
    cv=np.asarray(can_vertices,dtype=np.float64);tv=np.asarray(tub_vertices,dtype=np.float64)
    # Exact triangle test is unnecessary for disjoint closed AABBs.
    if np.any(cv.max(0)<tv.min(0))or np.any(tv.max(0)<cv.min(0)):return False
    n=len(cv);vertices=np.vstack((cv,tv))
    faces=np.vstack((can_faces,np.asarray(tub_faces,dtype=np.int32)+n))
    edges=np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
    mesh=ipctk.CollisionMesh(vertices,edges,faces)
    # Native filter: exactly can-versus-tub, no deprecated Python callback.
    patches=np.r_[np.zeros(n,dtype=np.int32),np.ones(len(tv),dtype=np.int32)]
    mesh.can_collide=ipctk.make_vertex_patches_filter(patches)
    return bool(ipctk.has_intersections(mesh,vertices))


def audit(assembly):
    from scipy.spatial.transform import Rotation
    b=assembly.builder;pose=np.asarray(b.body_q[assembly.tub],dtype=np.float64)
    v,f=assembly.tub_surface
    # These are the same float32 mesh/pose inputs used by the native model.
    tv=Rotation.from_quat(pose[3:]).apply(v)+pose[:3]
    q=np.asarray(b.particle_q,dtype=np.float32).astype(np.float64)
    intersections=[];checked=[]
    for s in assembly.surfaces:
        if s['kind']!='can':continue
        checked.append(s['name']);p=q[s['first']:s['first']+s['count']]
        if intersects_can_tub(p,s['triangles'],tv,f):intersections.append(s['name'])
    result=dict(physics_steps=0,checked_cans=checked,intersecting_cans=intersections,
        method='Actual initial can/tub triangles, native IPC bipartite filter; no state correction')
    assembly.geometry_checks['initial_can_tub']=result
    if intersections:raise ValueError('Initial can/tub surface intersection: '+', '.join(intersections))
    return result
