"""Residual-decreasing island translation iterations of the existing equations.

Membrane/bending and intra-island contact are translation invariant. Aggregate
actual native contact/spring forces and implicit inertia; no artificial support,
rigid attachment, changed stiffness, shape matching or velocity rewriting.
Every accepted translation stays inside the native collision-discovery bound.
The physical local deformation solve still runs for every vertex.
"""
import numpy as np
import warp as wp
from newton._src.solvers.vbd.particle_vbd_kernels import (
    accumulate_particle_body_contact_force_and_hessian,accumulate_spring_force_and_hessian)
from .contact_accumulation import split as accumulate_self_contact_force_and_hessian,combine
from .contact_curvature import redirect
accumulate_particle_body_contact_force_and_hessian=redirect(accumulate_particle_body_contact_force_and_hessian)

@wp.kernel
def aggregate(q:wp.array[wp.vec3],target:wp.array[wp.vec3],mass:wp.array[float],groups:wp.array[int],dt:float,
    force:wp.array[wp.vec3],hess:wp.array[wp.mat33],total:wp.array[wp.vec3],matrix:wp.array[wp.mat33]):
    i=wp.tid();g=groups[i]
    if g<0:return
    k=mass[i]/(dt*dt)
    wp.atomic_add(total,g,force[i]+k*(target[i]-q[i]))
    # A contact has <=4 vertices. Cauchy-Schwarz supplies a conservative
    # aggregate diagonal bound including omitted cross-vertex blocks.
    wp.atomic_add(matrix,g,4.*hess[i]+k*wp.identity(n=3,dtype=float))

@wp.kernel
def directions(force:wp.array[wp.vec3],matrix:wp.array[wp.mat33],delta:wp.array[wp.vec3]):
    i=wp.tid();d=wp.vec3(0.)
    if wp.abs(wp.determinant(matrix[i]))>1e-12:d=wp.inverse(matrix[i])*force[i]
    l=wp.length(d)
    if l>.00025:d*=.00025/l
    delta[i]=d

@wp.kernel
def permissible(q:wp.array[wp.vec3],old:wp.array[wp.vec3],radius:float,groups:wp.array[int],
    delta:wp.array[wp.vec3],scales:wp.array[float]):
    i=wp.tid();g=groups[i]
    if g<0:return
    r=radius;d=delta[g];x=q[i]-old[i];a=wp.dot(d,d)
    if r>1e-5 and a>1e-20:
        b=wp.dot(x,d);c=wp.dot(x,x)-r*r;disc=b*b-a*c
        scale=float(0.)
        if disc>=0.:scale=wp.clamp((-b+wp.sqrt(disc))/a,0.,1.)
        wp.atomic_min(scales,g,scale)

@wp.kernel
def safe_scale(proposed:wp.array[wp.vec3],groups:wp.array[int],truncation:wp.array[float],scales:wp.array[float]):
    i=wp.tid();g=groups[i]
    if g>=0 and truncation[i]<1.-1e-7:wp.atomic_min(scales,g,0.)

@wp.kernel
def proposed(displacements:wp.array[wp.vec3],groups:wp.array[int],delta:wp.array[wp.vec3],scales:wp.array[float],out:wp.array[wp.vec3]):
    i=wp.tid();g=groups[i];out[i]=displacements[i]
    if g>=0:out[i]+=scales[g]*delta[g]

@wp.kernel
def trial(source:wp.array[wp.vec3],groups:wp.array[int],delta:wp.array[wp.vec3],bounds:wp.array[float],scale:float,
    dest:wp.array[wp.vec3]):
    i=wp.tid();g=groups[i];dest[i]=source[i]
    if g>=0:dest[i]+=scale*bounds[g]*delta[g]

@wp.kernel
def norm(force:wp.array[wp.vec3],result:wp.array[float]):
    i=wp.tid();wp.atomic_add(result,0,wp.dot(force[i],force[i]))

@wp.kernel
def accept(before:wp.array[float],after:wp.array[float],scale:float,chosen:wp.array[float]):
    if chosen[0]==0. and after[0]<before[0]:chosen[0]=scale

@wp.kernel
def apply(q:wp.array[wp.vec3],out:wp.array[wp.vec3],displacement:wp.array[wp.vec3],groups:wp.array[int],
    delta:wp.array[wp.vec3],bounds:wp.array[float],chosen:wp.array[float]):
    i=wp.tid();g=groups[i]
    if g<0:return
    d=chosen[0]*bounds[g]*delta[g];q[i]+=d;out[i]=q[i];displacement[i]+=d

class Translation:
    def __init__(self,solver,surfaces):
        self.s=solver;m=solver.model;self.device=m.device
        group=np.full(m.particle_count,-1,np.int32)
        for i,a in enumerate(surfaces):group[a['first']:a['first']+a['count']]=i
        self.groups=wp.array(group,dtype=int,device=m.device);n=len(surfaces)
        self.force=wp.zeros_like(m.particle_q);self.hessian=wp.zeros(m.particle_count,dtype=wp.mat33,device=m.device)
        self.internal=wp.zeros_like(self.hessian);self.external=wp.zeros_like(self.hessian)
        self.total=wp.zeros(n,dtype=wp.vec3,device=m.device);self.matrix=wp.zeros(n,dtype=wp.mat33,device=m.device)
        self.delta=wp.zeros_like(self.total);self.bounds=wp.ones(n,dtype=float,device=m.device)
        self.before=wp.zeros(1,dtype=float,device=m.device);self.after=wp.zeros_like(self.before);self.chosen=wp.zeros_like(self.before)
        self.trial=wp.zeros_like(m.particle_q)
        # Force assembly only: one evaluation visits every contact vertex.
        self.all_colors=wp.zeros(m.particle_count,dtype=int,device=m.device)
        self.proposed=wp.zeros_like(m.particle_q);self.truncation=wp.ones(m.particle_count,dtype=float,device=m.device)
    def evaluate(self,q,state,contacts,dt,*,aggregate_groups=True,hessians=True):
        s=self.s;m=s.model;c=contacts;self.force.zero_()
        if aggregate_groups and not hessians:raise ValueError('Translation aggregation requires contact Hessians')
        from . import force_only
        body_kernel=accumulate_particle_body_contact_force_and_hessian if hessians else force_only.body
        spring_kernel=accumulate_spring_force_and_hessian if hessians else force_only.springs
        self_kernel=accumulate_self_contact_force_and_hessian if hessians else force_only.self_contact
        if hessians:self.hessian.zero_();self.internal.zero_()
        for color in (0,):
            wp.launch(body_kernel,c.soft_contact_max,
                [dt,color,s.particle_q_prev,q,self.all_colors,s.friction_epsilon,m.particle_radius,
                 c.soft_contact_indices,c.soft_contact_count,c.soft_contact_max,s.body_particle_contact_penalty_k,
                 s.body_particle_contact_material_ke,s.body_particle_contact_material_kd,s.body_particle_contact_material_mu,
                 m.shape_body,state.body_q,s.body_q_prev,state.body_qd,m.body_com,c.soft_contact_shape,
                 c.soft_contact_body_pos,c.soft_contact_body_vel,c.soft_contact_normal,m.shape_margin,c.soft_contact_barycentric],
                [self.force,self.hessian],device=self.device)
            if m.spring_count:
                wp.launch(spring_kernel,m.spring_count,
                    [dt,color,s.particle_q_prev,q,self.all_colors,m.spring_count,m.spring_indices,m.spring_rest_length,
                     m.spring_stiffness,m.spring_damping],[self.force,self.hessian],device=self.device)
            if s.particle_enable_self_contact:
                wp.launch(self_kernel,s.particle_self_contact_evaluation_kernel_launch_size,
                    [dt,color,s.particle_q_prev,q,self.all_colors,m.tri_indices,m.edge_indices,s.trimesh_collision_info,
                     s.particle_self_contact_radius,m.soft_contact_ke,m.soft_contact_kd,m.soft_contact_mu,s.friction_epsilon,
                     s.trimesh_collision_detector.edge_edge_parallel_epsilon],[self.force,self.hessian,self.groups,self.internal],device=self.device,
                    max_blocks=m.device.sm_count)
        if hessians:
            wp.copy(self.external,self.hessian)
            wp.launch(combine,m.particle_count,[self.hessian,self.internal],device=self.device)
        if aggregate_groups:
            self.total.zero_();self.matrix.zero_()
            wp.launch(aggregate,m.particle_count,[q,s.inertia,m.particle_mass,self.groups,dt,self.force,self.hessian,self.total,self.matrix],device=self.device)
    def refine(self,state,out,c,dt):
        s=self.s;m=s.model;self.chosen.zero_();self.before.zero_()
        self.evaluate(state.particle_q,state,c,dt)
        wp.launch(norm,self.total.size,[self.total,self.before],device=self.device)
        wp.launch(directions,self.total.size,[self.total,self.matrix,self.delta],device=self.device)
        self.bounds.fill_(1.)
        wp.launch(permissible,m.particle_count,[state.particle_q,s.pos_prev_collision_detection,s.particle_self_contact_margin*s.particle_conservative_bound_relaxation*.5,
            self.groups,self.delta,self.bounds],device=self.device)
        # Retain native planar collision bounds as well as the radius bound.
        from newton._src.solvers.vbd.particle_vbd_kernels import apply_planar_truncation_parallel_by_collision
        self.truncation.fill_(1.)
        wp.launch(proposed,m.particle_count,[s.particle_displacements,self.groups,self.delta,self.bounds,self.proposed],device=self.device)
        wp.launch(apply_planar_truncation_parallel_by_collision,s.particle_self_contact_evaluation_kernel_launch_size,
            [s.pos_prev_collision_detection,self.proposed,m.tri_indices,m.edge_indices,s.trimesh_collision_info,
             s.trimesh_collision_detector.edge_edge_parallel_epsilon,s.particle_conservative_bound_relaxation,self.truncation],device=self.device)
        # Reject a candidate group if its full displacement would be truncated:
        # shrinking only the correction would not necessarily satisfy the plane.
        wp.launch(safe_scale,m.particle_count,[self.proposed,self.groups,self.truncation,self.bounds],device=self.device)
        for scale in (1.,.5,.25,.125):
            wp.launch(trial,m.particle_count,[state.particle_q,self.groups,self.delta,self.bounds,scale,self.trial],device=self.device)
            self.evaluate(self.trial,state,c,dt);self.after.zero_()
            wp.launch(norm,self.total.size,[self.total,self.after],device=self.device)
            wp.launch(accept,1,[self.before,self.after,scale,self.chosen],device=self.device)
        wp.launch(apply,m.particle_count,[state.particle_q,out.particle_q,s.particle_displacements,self.groups,
            self.delta,self.bounds,self.chosen],device=self.device)
