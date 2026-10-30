"""Read-only final-iterate shell contact contributions, using the solve's laws.

Edge entries are DIRECTED: only the two source-edge endpoint forces are stored.
The reverse candidate supplies the other edge's reaction. Vertex/triangle
entries store all four forces once. Summing entries therefore reproduces the
native all-colour accumulation without doubling edge reactions. These are
implicit end-state evaluations, not constant forces or measured sensors.
"""
import numpy as np
from types import FunctionType
import warp as wp
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionInfo
from .contact_accumulation import (
    evaluate_edge_edge_contact_2_vertices as edge_force,
    evaluate_vertex_triangle_collision_force_hessian_4_vertices as vertex_force)

@wp.kernel
def collect(dt:float,prev:wp.array[wp.vec3],q:wp.array[wp.vec3],
    tri:wp.array2d[int],edges:wp.array2d[int],info:wp.array[TriMeshCollisionInfo],
    radius:float,ke:float,kd:float,mu:float,eps:float,edge_eps:float,
    count:wp.array[int],kind:wp.array[int],primitive:wp.array2d[int],
    vertices:wp.array2d[int],forces:wp.array2d[wp.vec3]):
    i=wp.tid();c=info[0]
    if i<c.edge_colliding_edges_buffer_sizes.shape[0]:
        offset=c.edge_colliding_edges_offsets[i]
        n=wp.min(c.edge_colliding_edges_count[i],c.edge_colliding_edges_buffer_sizes[i])
        for j in range(n):
            other=c.edge_colliding_edges[2*(offset+j)+1]
            if other>=0:
                hit,fa,fb,ha,hb=edge_force(i,other,q,prev,edges,radius,ke,kd,mu,eps,dt,edge_eps)
                if hit:
                    row=wp.atomic_add(count,0,1)
                    if row<kind.shape[0]:
                        kind[row]=0;primitive[row,0]=i;primitive[row,1]=other
                        vertices[row,0]=edges[i,2];vertices[row,1]=edges[i,3]
                        vertices[row,2]=-1;vertices[row,3]=-1
                        forces[row,0]=fa;forces[row,1]=fb
                        forces[row,2]=wp.vec3(0.);forces[row,3]=wp.vec3(0.)
    if i<c.vertex_colliding_triangles_buffer_sizes.shape[0]:
        offset=c.vertex_colliding_triangles_offsets[i]
        n=wp.min(c.vertex_colliding_triangles_count[i],c.vertex_colliding_triangles_buffer_sizes[i])
        for j in range(n):
            other=c.vertex_colliding_triangles[2*(offset+j)+1]
            if other>=0:
                hit,fa,fb,fc,fi,ha,hb,hc,hi=vertex_force(i,other,q,prev,tri,radius,ke,kd,mu,eps,dt)
                if hit:
                    row=wp.atomic_add(count,0,1)
                    if row<kind.shape[0]:
                        kind[row]=1;primitive[row,0]=i;primitive[row,1]=other
                        vertices[row,0]=i;vertices[row,1]=tri[other,0]
                        vertices[row,2]=tri[other,1];vertices[row,3]=tri[other,2]
                        forces[row,0]=fi;forces[row,1]=fa;forces[row,2]=fb;forces[row,3]=fc

class SelfContacts:
    def __init__(self,solver,capacity):
        self.s=solver;self.capacity=int(capacity);d=solver.model.device
        self.count=wp.zeros(1,dtype=int,device=d)
        self.kind=wp.empty(capacity,dtype=int,device=d)
        self.primitive=wp.empty((capacity,2),dtype=int,device=d)
        self.vertices=wp.empty((capacity,4),dtype=int,device=d)
        self.forces=wp.empty((capacity,4),dtype=wp.vec3,device=d)
        self.kernel=collect;self.precision=None;self.increments=None;self.precise_geometry=False
        g=solver.global_refinement
        if g is not None and hasattr(g.contact,'precise_increment'):
            self.precise_geometry=bool(getattr(g,'precise_self_geometry',False))
            if self.precise_geometry:
                from .precise_self_geometry import ee,vt
            else:
                from .precise_increment import ee,vt
            from warp._src.context import Kernel
            f=collect.func
            namespace=dict(f.__globals__,edge_force=ee,vertex_force=vt)
            copied=FunctionType(f.__code__,namespace,f.__name__,f.__defaults__,f.__closure__)
            copied.__module__=__name__;copied.__qualname__=f.__qualname__
            copied.__annotations__=dict(f.__annotations__)
            if self.precise_geometry:copied.__annotations__['q']=wp.array[wp.vec3d]
            suffix='_precise_geometry'if self.precise_geometry else'_precise_slip'
            self.kernel=Kernel(copied,key=collect.key+suffix,module=wp.get_module(__name__),options=dict(collect.options))
            self.precision=g.precision
            self.increments=wp.zeros_like(solver.model.particle_q)
    def read(self,q,dt):
        s=self.s;m=s.model;d=s.trimesh_collision_detector
        previous=s.particle_q_prev
        if self.precision is not None:
            from .precise_increment import increments
            # Own scratch array: observation must not overwrite solver storage.
            wp.launch(increments,m.particle_count,[self.precision.current,self.precision.previous,self.increments],device=m.device)
            previous=self.increments
            if self.precise_geometry:q=self.precision.current
        self.count.zero_()
        wp.launch(self.kernel,max(m.particle_count,m.edge_count),[
            dt,previous,q,m.tri_indices,m.edge_indices,s.trimesh_collision_info,
            s.particle_self_contact_radius,m.soft_contact_ke,m.soft_contact_kd,m.soft_contact_mu,
            s.friction_epsilon,d.edge_edge_parallel_epsilon,self.count,self.kind,
            self.primitive,self.vertices,self.forces],device=m.device)
        n=int(self.count.numpy()[0])
        if n>self.capacity:raise OverflowError(f'Self-contact trace capacity exceeded: {n}>{self.capacity}; no truncated sample published')
        kind=self.kind.numpy()[:n];primitive=self.primitive.numpy()[:n]
        vertices=self.vertices.numpy()[:n];forces=self.forces.numpy()[:n]
        if not np.isfinite(forces).all():raise FloatingPointError('Nonfinite self-contact force readback')
        order=np.lexsort((primitive[:,1],primitive[:,0],kind))
        return dict(kind=kind[order],primitive_indices=primitive[order],
            vertex_indices=vertices[order],force_world_on_vertex_n=forces[order])
    @staticmethod
    def identities(sample):
        return set((int(k),int(p[0]),int(p[1]))for k,p in zip(sample['kind'],sample['primitive_indices']))
