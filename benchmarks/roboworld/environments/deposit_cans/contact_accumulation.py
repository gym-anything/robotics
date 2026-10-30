# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Same native contact evaluations, without looping over unused buffer slots.

Derived from Newton1.5 accumulate_self_contact_force_and_hessian. Per-primitive
lane order, pair multiplicity, colors, force laws and atomic outputs are retained.
Capacity overflow is still an explicit fault, not silently truncated evidence.
"""
import warp as wp
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionInfo
from newton._src.solvers.vbd.particle_vbd_kernels import (
    NUM_THREADS_PER_COLLISION_PRIMITIVE,
    evaluate_edge_edge_contact_2_vertices,
    evaluate_vertex_triangle_collision_force_hessian_4_vertices)
from .contact_curvature import redirect
evaluate_edge_edge_contact_2_vertices=redirect(evaluate_edge_edge_contact_2_vertices)
evaluate_vertex_triangle_collision_force_hessian_4_vertices=redirect(evaluate_vertex_triangle_collision_force_hessian_4_vertices)

@wp.kernel
def accumulate(dt:float,color:int,prev:wp.array[wp.vec3],q:wp.array[wp.vec3],colors:wp.array[int],
    tri:wp.array2d[int],edges:wp.array2d[int],info:wp.array[TriMeshCollisionInfo],radius:float,
    ke:float,kd:float,mu:float,friction_epsilon:float,edge_epsilon:float,
    force:wp.array[wp.vec3],hessian:wp.array[wp.mat33]):
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
                    hit,fa,fb,ha,hb=evaluate_edge_edge_contact_2_vertices(i,other,q,prev,edges,
                        radius,ke,kd,mu,friction_epsilon,dt,edge_epsilon)
                    if hit:
                        if colors[a]==color:wp.atomic_add(force,a,fa);wp.atomic_add(hessian,a,ha)
                        if colors[b]==color:wp.atomic_add(force,b,fb);wp.atomic_add(hessian,b,hb)
            j+=NUM_THREADS_PER_COLLISION_PRIMITIVE
    if i<c.vertex_colliding_triangles_buffer_sizes.shape[0]:
        j=lane;offset=c.vertex_colliding_triangles_offsets[i]
        count=wp.min(c.vertex_colliding_triangles_count[i],c.vertex_colliding_triangles_buffer_sizes[i])
        while j<count:
            t=c.vertex_colliding_triangles[2*(offset+j)+1]
            if t>=0:
                a=tri[t,0];b=tri[t,1];d=tri[t,2]
                if colors[i]==color or colors[a]==color or colors[b]==color or colors[d]==color:
                    hit,fa,fb,fd,fi,ha,hb,hd,hi=evaluate_vertex_triangle_collision_force_hessian_4_vertices(
                        i,t,q,prev,tri,radius,ke,kd,mu,friction_epsilon,dt)
                    if hit:
                        if colors[i]==color:wp.atomic_add(force,i,fi);wp.atomic_add(hessian,i,hi)
                        if colors[a]==color:wp.atomic_add(force,a,fa);wp.atomic_add(hessian,a,ha)
                        if colors[b]==color:wp.atomic_add(force,b,fb);wp.atomic_add(hessian,b,hb)
                        if colors[d]==color:wp.atomic_add(force,d,fd);wp.atomic_add(hessian,d,hd)
            j+=NUM_THREADS_PER_COLLISION_PRIMITIVE

@wp.kernel
def split(dt:float,color:int,prev:wp.array[wp.vec3],q:wp.array[wp.vec3],colors:wp.array[int],
    tri:wp.array2d[int],edges:wp.array2d[int],info:wp.array[TriMeshCollisionInfo],radius:float,
    ke:float,kd:float,mu:float,friction_epsilon:float,edge_epsilon:float,
    force:wp.array[wp.vec3],external:wp.array[wp.mat33],groups:wp.array[int],internal:wp.array[wp.mat33]):
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
                    hit,fa,fb,ha,hb=evaluate_edge_edge_contact_2_vertices(i,other,q,prev,edges,radius,ke,kd,mu,friction_epsilon,dt,edge_epsilon)
                    if hit:
                        same=groups[a]>=0 and groups[a]==groups[edges[other,2]]
                        if colors[a]==color:
                            wp.atomic_add(force,a,fa)
                            if same:wp.atomic_add(internal,a,ha)
                            else:wp.atomic_add(external,a,ha)
                        if colors[b]==color:
                            wp.atomic_add(force,b,fb)
                            if same:wp.atomic_add(internal,b,hb)
                            else:wp.atomic_add(external,b,hb)
            j+=NUM_THREADS_PER_COLLISION_PRIMITIVE
    if i<c.vertex_colliding_triangles_buffer_sizes.shape[0]:
        j=lane;offset=c.vertex_colliding_triangles_offsets[i]
        count=wp.min(c.vertex_colliding_triangles_count[i],c.vertex_colliding_triangles_buffer_sizes[i])
        while j<count:
            t=c.vertex_colliding_triangles[2*(offset+j)+1]
            if t>=0:
                a=tri[t,0];b=tri[t,1];d=tri[t,2]
                if colors[i]==color or colors[a]==color or colors[b]==color or colors[d]==color:
                    hit,fa,fb,fd,fi,ha,hb,hd,hi=evaluate_vertex_triangle_collision_force_hessian_4_vertices(i,t,q,prev,tri,radius,ke,kd,mu,friction_epsilon,dt)
                    if hit:
                        same=groups[i]>=0 and groups[i]==groups[a]
                        for k in range(4):
                            v=i;f=fi;h=hi
                            if k==1:v=a;f=fa;h=ha
                            elif k==2:v=b;f=fb;h=hb
                            elif k==3:v=d;f=fd;h=hd
                            if colors[v]==color:
                                wp.atomic_add(force,v,f)
                                if same:wp.atomic_add(internal,v,h)
                                else:wp.atomic_add(external,v,h)
            j+=NUM_THREADS_PER_COLLISION_PRIMITIVE

@wp.kernel
def combine(external:wp.array[wp.mat33],internal:wp.array[wp.mat33]):
    i=wp.tid();external[i]+=internal[i]

def compare(solver,q,dt):
    """Read-only numerical equivalence/timing check at an existing query."""
    import time,numpy as np
    from newton._src.solvers.vbd.particle_vbd_kernels import accumulate_self_contact_force_and_hessian as original
    m=solver.model;d=m.device;colors=wp.zeros(m.particle_count,dtype=int,device=d)
    values=[];timings=[]
    for kernel in (original,accumulate):
        f=wp.zeros(m.particle_count,dtype=wp.vec3,device=d);h=wp.zeros(m.particle_count,dtype=wp.mat33,device=d)
        arguments=[dt,0,q,q,colors,m.tri_indices,m.edge_indices,solver.trimesh_collision_info,
            solver.particle_self_contact_radius,m.soft_contact_ke,m.soft_contact_kd,m.soft_contact_mu,
            solver.friction_epsilon,solver.trimesh_collision_detector.edge_edge_parallel_epsilon,f,h]
        def launch():
            f.zero_();h.zero_();wp.launch(kernel,solver.particle_self_contact_evaluation_kernel_launch_size,
                arguments,device=d,max_blocks=d.sm_count)
        launch();wp.synchronize_device(d);start=time.perf_counter()
        for _ in range(10):launch()
        wp.synchronize_device(d);timings.append((time.perf_counter()-start)/10)
        values.append((f.numpy(),h.numpy()))
    errors={name:float(np.linalg.norm(values[0][i]-values[1][i])/max(np.linalg.norm(values[0][i]),1e-20))
        for i,name in enumerate(('relative_force_l2','relative_hessian_l2'))}
    return dict(**errors,native_seconds=timings[0],active_seconds=timings[1],
        finite=bool(all(np.isfinite(a).all() for pair in values for a in pair)))
