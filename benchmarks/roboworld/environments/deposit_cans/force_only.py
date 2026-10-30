# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Native force assembly without unused Hessian accumulation for line work.

The same force evaluators, contact streams, multiplicity and material inputs
are used. No contact is pruned. Ignoring returned Hessians lets the compiler
discard their arithmetic, and avoids nine atomic stores per vertex/term.
"""
import warp as wp
from newton._src.solvers.vbd.particle_vbd_kernels import (
    _eval_body_particle_contact,_eval_soft_ef_contact,
    evaluate_spring_force_and_hessian_both_vertices,
    NUM_THREADS_PER_COLLISION_PRIMITIVE)
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionInfo
from .contact_accumulation import (
    evaluate_edge_edge_contact_2_vertices,
    evaluate_vertex_triangle_collision_force_hessian_4_vertices)
from .bending import bending
from wiping.membrane_hessian import membrane

@wp.kernel
def body(dt:float,color:int,prev:wp.array[wp.vec3],q:wp.array[wp.vec3],colors:wp.array[int],
    eps:float,radius:wp.array[float],indices:wp.array[wp.vec3i],count:wp.array[int],capacity:int,
    penalty:wp.array[float],ke:wp.array[float],kd:wp.array[float],mu:wp.array[float],
    shape_body:wp.array[int],bq:wp.array[wp.transform],bprev:wp.array[wp.transform],
    bqd:wp.array[wp.spatial_vector],com:wp.array[wp.vec3],shape:wp.array[int],
    point:wp.array[wp.vec3],vel:wp.array[wp.vec3],normal:wp.array[wp.vec3],margin:wp.array[float],
    barycentric:wp.array[wp.vec3],force:wp.array[wp.vec3],unused:wp.array[wp.mat33]):
    i=wp.tid()
    if i>=wp.min(capacity,count[0]):return
    corners=indices[i]
    if corners[1]<0:
        p=corners[0]
        if colors[p]==color:
            f,h=_eval_body_particle_contact(p,q[p],prev[p],i,penalty[i],kd[i],mu[i],eps,radius,
                shape_body,bq,bprev,bqd,com,shape,point,vel,normal,margin,dt)
            wp.atomic_add(force,p,f)
    else:
        bary=barycentric[i]
        f,h,cp=_eval_soft_ef_contact(i,corners,bary,q,prev,radius,penalty[i],kd[i],mu[i],eps,
            shape_body,bq,bprev,bqd,com,shape,point,vel,normal,margin,dt)
        for j in range(3):
            p=corners[j]
            if p>=0:
                if colors[p]==color:wp.atomic_add(force,p,bary[j]*f)

@wp.kernel
def springs(dt:float,color:int,prev:wp.array[wp.vec3],q:wp.array[wp.vec3],colors:wp.array[int],
    count:int,indices:wp.array[int],rest:wp.array[float],stiffness:wp.array[float],damping:wp.array[float],
    force:wp.array[wp.vec3],unused:wp.array[wp.mat33]):
    i=wp.tid()
    if i>=count:return
    a=indices[2*i];b=indices[2*i+1]
    if colors[a]==color or colors[b]==color:
        x,y,fa,fb,h=evaluate_spring_force_and_hessian_both_vertices(i,dt,q,prev,indices,rest,stiffness,damping)
        if colors[a]==color:wp.atomic_add(force,a,fa)
        if colors[b]==color:wp.atomic_add(force,b,fb)

@wp.kernel
def self_contact(dt:float,color:int,prev:wp.array[wp.vec3],q:wp.array[wp.vec3],colors:wp.array[int],
    tri:wp.array2d[int],edges:wp.array2d[int],info:wp.array[TriMeshCollisionInfo],radius:float,
    ke:float,kd:float,mu:float,eps:float,edge_epsilon:float,force:wp.array[wp.vec3],
    unused:wp.array[wp.mat33],groups:wp.array[int],unused_internal:wp.array[wp.mat33]):
    thread=wp.tid();i=thread//NUM_THREADS_PER_COLLISION_PRIMITIVE
    lane=thread%NUM_THREADS_PER_COLLISION_PRIMITIVE;c=info[0]
    if i<c.edge_colliding_edges_buffer_sizes.shape[0]:
        j=lane;offset=c.edge_colliding_edges_offsets[i]
        count=wp.min(c.edge_colliding_edges_count[i],c.edge_colliding_edges_buffer_sizes[i])
        while j<count:
            other=c.edge_colliding_edges[2*(offset+j)+1]
            if other>=0:
                a=edges[i,2];b=edges[i,3]
                if colors[a]==color or colors[b]==color:
                    hit,fa,fb,ha,hb=evaluate_edge_edge_contact_2_vertices(i,other,q,prev,edges,radius,ke,kd,mu,eps,dt,edge_epsilon)
                    if hit:
                        if colors[a]==color:wp.atomic_add(force,a,fa)
                        if colors[b]==color:wp.atomic_add(force,b,fb)
            j+=NUM_THREADS_PER_COLLISION_PRIMITIVE
    if i<c.vertex_colliding_triangles_buffer_sizes.shape[0]:
        j=lane;offset=c.vertex_colliding_triangles_offsets[i]
        count=wp.min(c.vertex_colliding_triangles_count[i],c.vertex_colliding_triangles_buffer_sizes[i])
        while j<count:
            t=c.vertex_colliding_triangles[2*(offset+j)+1]
            if t>=0:
                a=tri[t,0];b=tri[t,1];d=tri[t,2]
                if colors[i]==color or colors[a]==color or colors[b]==color or colors[d]==color:
                    hit,fa,fb,fd,fi,ha,hb,hd,hi=evaluate_vertex_triangle_collision_force_hessian_4_vertices(i,t,q,prev,tri,radius,ke,kd,mu,eps,dt)
                    if hit:
                        if colors[i]==color:wp.atomic_add(force,i,fi)
                        if colors[a]==color:wp.atomic_add(force,a,fa)
                        if colors[b]==color:wp.atomic_add(force,b,fb)
                        if colors[d]==color:wp.atomic_add(force,d,fd)
            j+=NUM_THREADS_PER_COLLISION_PRIMITIVE

@wp.kernel
def elasticity(q:wp.array[wp.vec3],prev:wp.array[wp.vec3],tri:wp.array2d[int],pose:wp.array[wp.mat22],
    area:wp.array[float],mat:wp.array2d[float],edges:wp.array2d[int],rest:wp.array[float],length:wp.array[float],
    stiffness:wp.array2d[float],dt:float,force:wp.array[wp.vec3],unused:wp.array[wp.mat33]):
    i=wp.tid()
    if i<tri.shape[0]*3:
        t=i//3;o=i%3;v=tri[t,o]
        f,h=membrane(t,o,q,prev,tri,pose[t],area[t],mat[t,0],mat[t,1],mat[t,2],dt)
        wp.atomic_add(force,v,f)
    if i<edges.shape[0]*4:
        e=i//4;o=i%4;v=edges[e,o]
        if v>=0:
            f,h=bending(e,o,q,prev,edges,rest,length,stiffness[e,0],stiffness[e,1],dt)
            wp.atomic_add(force,v,f)

@wp.kernel
def rhs(q:wp.array[wp.vec3],target:wp.array[wp.vec3],mass:wp.array[float],dt:float,
    force:wp.array[wp.vec3],unused:wp.array[wp.mat33],out:wp.array[wp.vec3]):
    i=wp.tid();out[i]=force[i]+(mass[i]/(dt*dt))*(target[i]-q[i])
    if mass[i]==0.:out[i]=wp.vec3(0.)
