"""Diagnostic contact search preserving interpolation cross blocks.

A face contact sees sum(w_i * dx_i), not each vertex independently. The
diagonal approximation resists motions whose contact-point displacement is
zero, including rolling. Retain this frozen-weight PSD tangent in the search
operator only; physical forces and all nonlinear/contact acceptance stay put.
"""
import warp as wp
from deposit_cans.pair_search import PairSearch,pairs,springs
from deposit_cans.precise_body import point,face
@wp.kernel
def mark_overflow(count:wp.array[int],capacity:int,flag:wp.array[int]):
    if count[0]>capacity:flag[0]=1

@wp.kernel
def body_pairs(dt:float,q:wp.array[wp.vec3d],previous:wp.array[wp.vec3d],
    indices:wp.array[wp.vec3i],bary:wp.array[wp.vec3],contact_count:wp.array[int],
    ke:wp.array[float],kd:wp.array[float],mu:wp.array[float],eps:float,radius:wp.array[float],
    shape_body:wp.array[int],bq:wp.array[wp.transform],bprev:wp.array[wp.transform],bqd:wp.array[wp.spatial_vector],
    com:wp.array[wp.vec3],shape:wp.array[int],local:wp.array[wp.vec3],vel:wp.array[wp.vec3],normal:wp.array[wp.vec3],
    margin:wp.array[float],count:wp.array[int],ids:wp.array[wp.vec4i],weights:wp.array[wp.vec4],H:wp.array[wp.mat33]):
    i=wp.tid()
    if i>=contact_count[0]:return
    v=indices[i];w=bary[i]
    if v[0]<0:return
    vertices=wp.vec4i(v[0],v[0],v[0],v[0]);coefficients=wp.vec4(1.,0.,0.,0.)
    f=wp.vec3(0.);h=wp.mat33(0.);x=wp.vec3(0.)
    if v[1]<0:
        f,h,x=point(q[v[0]],previous[v[0]],radius[v[0]],i,ke[i],kd[i],mu[i],eps,
            shape_body,bq,bprev,bqd,com,shape,local,vel,normal,margin,dt)
    else:
        f,h,x=face(i,v,w,q,previous,radius,ke[i],kd[i],mu[i],eps,
            shape_body,bq,bprev,bqd,com,shape,local,vel,normal,margin,dt)
        vertices=wp.vec4i(v[0],v[1],v[0],v[0]);coefficients=wp.vec4(w[0],w[1],0.,0.)
        if v[2]>=0:vertices[2]=v[2];coefficients[2]=w[2]
    if wp.abs(h[0,0])+wp.abs(h[1,1])+wp.abs(h[2,2])>0.:
        row=wp.atomic_add(count,0,1)
        if row<ids.shape[0]:ids[row]=vertices;weights[row]=coefficients;H[row]=h

class SurfaceSearch(PairSearch):
    def __init__(self,g):
        # Explicit workspace bound, never omission of contact records.
        super().__init__(g,capacity=131072,coupled=True)
        self.overflow=wp.zeros(1,dtype=int,device=g.s.model.device)
    def prepare(self):
        g=self.g;s=g.s;m=s.model;state,c,dt=self.context;d=m.device
        self.base.zero_();self.scratch.zero_();self.count.zero_()
        # Keep installed cuff-spring search terms. Their other side is the
        # separately solved frame, not another unknown in this particle solve.
        if m.spring_count:
            wp.launch(springs,m.spring_count,
                [dt,0,s.particle_q_prev,g.q,g.contact.all_colors,m.spring_count,m.spring_indices,
                 m.spring_rest_length,m.spring_stiffness,m.spring_damping],[self.scratch,self.base],device=d)
        wp.launch(body_pairs,c.soft_contact_max,[dt,g.precision.current,g.precision.previous,
            c.soft_contact_indices,c.soft_contact_barycentric,c.soft_contact_count,
            s.body_particle_contact_penalty_k,s.body_particle_contact_material_kd,s.body_particle_contact_material_mu,
            s.friction_epsilon,m.particle_radius,m.shape_body,state.body_q,s.body_q_prev,state.body_qd,m.body_com,
            c.soft_contact_shape,c.soft_contact_body_pos,c.soft_contact_body_vel,c.soft_contact_normal,m.shape_margin,
            self.count,self.ids,self.weights,self.H],device=d)
        wp.launch(pairs,max(m.particle_count,m.edge_count),
            [dt,s.particle_q_prev,g.q,m.tri_indices,m.edge_indices,s.trimesh_collision_info,
             s.particle_self_contact_radius,m.soft_contact_ke,m.soft_contact_kd,m.soft_contact_mu,
             s.friction_epsilon,s.trimesh_collision_detector.edge_edge_parallel_epsilon,
             self.count,self.ids,self.weights,self.H],device=d)
        wp.launch(mark_overflow,1,[self.count,self.capacity,self.overflow],device=d)
