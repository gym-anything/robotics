"""Private numerical candidate: retain within-shell contact cross blocks.

For frozen contact weights, each native PSD contact tangent contributes
w_i w_j H. The old 4D bound replaces these blocks (including film self-contact)
with a much stiffer diagonal approximation. Here each material shell retains
its principal block; cross-shell forces remain in every nonlinear evaluation.
Body/spring diagonal tangents are unscaled. Actual forces, contacts, bounds,
material constants, DOFs and acceptance tests are untouched.
Not a public solver or an assertion that friction is a conservative potential.
"""
import warp as wp
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionInfo
from newton._src.solvers.vbd.particle_vbd_kernels import triangle_closest_point
from deposit_cans.contact_accumulation import (
    evaluate_edge_edge_contact_2_vertices as edge_force,
    evaluate_vertex_triangle_collision_force_hessian_4_vertices as vertex_force)
from deposit_cans.translation import (
    accumulate_particle_body_contact_force_and_hessian as body,
    accumulate_spring_force_and_hessian as springs)
from deposit_cans.linear64 import matrix,membrane,cached_hinges,finish

@wp.kernel
def pairs(dt:float,prev:wp.array[wp.vec3],q:wp.array[wp.vec3],
    tri:wp.array2d[int],edges:wp.array2d[int],info:wp.array[TriMeshCollisionInfo],
    radius:float,ke:float,kd:float,mu:float,eps:float,edge_eps:float,
    count:wp.array[int],ids:wp.array[wp.vec4i],weights:wp.array[wp.vec4],H:wp.array[wp.mat33]):
    i=wp.tid();c=info[0]
    if i<c.edge_colliding_edges_buffer_sizes.shape[0]:
        offset=c.edge_colliding_edges_offsets[i]
        n=wp.min(c.edge_colliding_edges_count[i],c.edge_colliding_edges_buffer_sizes[i])
        for j in range(n):
            other=c.edge_colliding_edges[2*(offset+j)+1]
            if other>i:
                hit,fa,fb,ha,hb=edge_force(i,other,q,prev,edges,radius,ke,kd,mu,eps,dt,edge_eps)
                if hit:
                    a=edges[i,2];b=edges[i,3];d=edges[other,2];e=edges[other,3]
                    st=wp.closest_point_edge_edge(q[a],q[b],q[d],q[e],edge_eps)
                    w=wp.vec4(1.-st[0],st[0],-1.+st[1],-st[1])
                    h=wp.mat33(0.)
                    if w[1]>w[0]:h=hb/(w[1]*w[1])
                    else:h=ha/(w[0]*w[0])
                    row=wp.atomic_add(count,0,1)
                    if row<ids.shape[0]:ids[row]=wp.vec4i(a,b,d,e);weights[row]=w;H[row]=h
    if i<c.vertex_colliding_triangles_buffer_sizes.shape[0]:
        offset=c.vertex_colliding_triangles_offsets[i]
        n=wp.min(c.vertex_colliding_triangles_count[i],c.vertex_colliding_triangles_buffer_sizes[i])
        for j in range(n):
            t=c.vertex_colliding_triangles[2*(offset+j)+1]
            if t>=0:
                hit,fa,fb,fc,fi,ha,hb,hc,hi=vertex_force(i,t,q,prev,tri,radius,ke,kd,mu,eps,dt)
                if hit:
                    a=tri[t,0];b=tri[t,1];d=tri[t,2]
                    closest,bary,feature=triangle_closest_point(q[a],q[b],q[d],q[i])
                    row=wp.atomic_add(count,0,1)
                    if row<ids.shape[0]:
                        ids[row]=wp.vec4i(a,b,d,i);weights[row]=wp.vec4(-bary[0],-bary[1],-bary[2],1.);H[row]=hi

@wp.kernel
def begin(mass:wp.array[float],base:wp.array[wp.mat33],x:wp.array[wp.vec3d],dt:wp.float64,out:wp.array[wp.vec3d]):
    i=wp.tid();out[i]=wp.float64(mass[i])/(dt*dt)*x[i]+matrix(base[i])*x[i]

@wp.kernel
def multiply(count:wp.array[int],ids:wp.array[wp.vec4i],weights:wp.array[wp.vec4],H:wp.array[wp.mat33],
             groups:wp.array[int],coupled:bool,x:wp.array[wp.vec3d],out:wp.array[wp.vec3d]):
    row=wp.tid()
    if row>=count[0]:return
    v=ids[row];w=weights[row];h=matrix(H[row])
    for i in range(4):
        g=groups[v[i]]
        if g>=0:
            value=wp.vec3d(0.)
            for j in range(4):
                if coupled or groups[v[j]]==g:value+=wp.float64(w[j])*x[v[j]]
            wp.atomic_add(out,v[i],wp.float64(w[i])*(h*value))

class PairSearch:
    def __init__(self,g,capacity=65536,*,coupled=False):
        self.g=g;m=g.s.model;d=m.device;self.capacity=capacity
        self.coupled=coupled
        self.count=wp.zeros(1,dtype=int,device=d)
        self.ids=wp.zeros(capacity,dtype=wp.vec4i,device=d)
        self.weights=wp.zeros(capacity,dtype=wp.vec4,device=d)
        self.H=wp.zeros(capacity,dtype=wp.mat33,device=d)
        self.base=wp.zeros(m.particle_count,dtype=wp.mat33,device=d)
        self.scratch=wp.zeros_like(g.force)
        self.context=None
        original_evaluate=g.contact.evaluate
        def evaluate(q,state,c,dt,**kwargs):
            self.context=(state,c,dt)
            return original_evaluate(q,state,c,dt,**kwargs)
        g.contact.evaluate=evaluate
        original_prepare=g.linear.prepare_rhs
        def prepare():
            original_prepare();self.prepare()
        g.linear.prepare_rhs=prepare
        g.linear.matvec=self.matvec
        # CG stores a bound method at creation, so replace its operator too.
        from warp.optim.linear import LinearOperator
        old=g.cg._A
        g.cg._A=LinearOperator(old.shape,wp.mat33d,d,self.matvec,batch_offsets=g.offsets)
        if coupled:
            from warp.optim.linear import CG
            A=LinearOperator(old.shape,wp.mat33d,d,self.matvec)
            M=LinearOperator(old.shape,wp.mat33d,d,g.linear.precondition)
            g.cg=CG(A,g.linear.rhs,g.linear.delta,M=M,tol=1e-6,atol=1e-7,
                    maxiter=g.cg._maxiter,check_every=0,use_cuda_graph=True)

    def prepare(self):
        g=self.g;s=g.s;m=s.model;state,c,dt=self.context;d=m.device
        self.base.zero_();self.scratch.zero_();self.count.zero_()
        wp.launch(body,c.soft_contact_max,
            [dt,0,s.particle_q_prev,g.q,g.contact.all_colors,s.friction_epsilon,m.particle_radius,
             c.soft_contact_indices,c.soft_contact_count,c.soft_contact_max,s.body_particle_contact_penalty_k,
             s.body_particle_contact_material_ke,s.body_particle_contact_material_kd,s.body_particle_contact_material_mu,
             m.shape_body,state.body_q,s.body_q_prev,state.body_qd,m.body_com,c.soft_contact_shape,
             c.soft_contact_body_pos,c.soft_contact_body_vel,c.soft_contact_normal,m.shape_margin,c.soft_contact_barycentric],
            [self.scratch,self.base],device=d)
        if m.spring_count:
            wp.launch(springs,m.spring_count,
                [dt,0,s.particle_q_prev,g.q,g.contact.all_colors,m.spring_count,m.spring_indices,
                 m.spring_rest_length,m.spring_stiffness,m.spring_damping],[self.scratch,self.base],device=d)
        wp.launch(pairs,max(m.particle_count,m.edge_count),
            [dt,s.particle_q_prev,g.q,m.tri_indices,m.edge_indices,s.trimesh_collision_info,
             s.particle_self_contact_radius,m.soft_contact_ke,m.soft_contact_kd,m.soft_contact_mu,
             s.friction_epsilon,s.trimesh_collision_detector.edge_edge_parallel_epsilon,
             self.count,self.ids,self.weights,self.H],device=d)

    def matvec(self,x,y,z,alpha,beta):
        g=self.g;m=g.s.model;d=m.device;dt=wp.float64(g.dt);linear=g.linear
        wp.launch(begin,m.particle_count,[m.particle_mass,self.base,x,dt,linear.mv],device=d)
        wp.launch(multiply,self.capacity,[self.count,self.ids,self.weights,self.H,g.coarse.groups,self.coupled,x,linear.mv],device=d)
        if linear.elastic_cache is not None:
            linear.elastic_cache.add_product(x,linear.mv)
        else:
            wp.launch(membrane,m.tri_count,[g.q,x,m.tri_indices,m.tri_poses,m.tri_areas,m.tri_materials,dt,linear.mv],device=d)
            wp.launch(cached_hinges,m.edge_count,[linear.hinge_grad,x,m.edge_indices,m.edge_rest_length,m.edge_bending_properties,dt,linear.mv],device=d)
        wp.launch(finish,m.particle_count,[m.particle_mass,linear.mv,x,y,wp.float64(alpha),wp.float64(beta),z],device=d)

    def audit(self):
        import numpy as np
        n=int(self.count.numpy()[0])
        if n>self.capacity:raise OverflowError(f'Pair search capacity exceeded: {n}>{self.capacity}')
        if not np.isfinite(self.H.numpy()[:n]).all():raise FloatingPointError('Nonfinite search curvature')
        eig=np.linalg.eigvalsh(self.H.numpy()[:n].astype(float))
        return dict(active_pairs=n,min_pair_eigenvalue=float(eig.min())if n else None,
                    scope='Frozen contact-weight search blocks; physical forces unchanged.')
