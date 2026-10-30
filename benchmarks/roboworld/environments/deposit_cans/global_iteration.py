# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic global correction of the existing shell/contact equations.

The residual is the actual native force plus implicit inertia. PCG uses a
positive metric-strain tangent as a SEARCH MATRIX, not a changed force law.
Bending uses its native Gauss-Newton angular block. Contact diagonal blocks
are conservative; repeated nonlinear evaluations retain actual contact forces.
Native collision truncation and residual backtracking safeguard each trial.
This is an opt-in numerical candidate until physical/performance qualification.
"""
import warp as wp
from warp.optim.linear import LinearOperator,CG
from newton._src.solvers.vbd.particle_vbd_kernels import (
    compute_normalized_vector_derivative,compute_angle_derivative,
    apply_planar_truncation_parallel_by_collision,apply_truncation_ts)
from wiping.membrane_hessian import membrane
from .bending import bending
from .translation import Translation

@wp.func
def hinge_gradient(e:int,o:int,q:wp.array[wp.vec3],edges:wp.array2d[int]):
    a=edges[e,0];b=edges[e,1];c=edges[e,2];d=edges[e,3]
    if a<0 or b<0:return wp.vec3(0.)
    x02=q[c]-q[a];x03=q[d]-q[a];x13=q[d]-q[b];x12=q[c]-q[b];edge=q[d]-q[c]
    n1=wp.cross(x02,x03);n2=wp.cross(x13,x12);l1=wp.length(n1);l2=wp.length(n2);le=wp.length(edge)
    if l1<1e-6 or l2<1e-6 or le<1e-6:return wp.vec3(0.)
    n1/=l1;n2/=l2;axis=edge/le;sine=wp.dot(wp.cross(n1,n2),axis);cosine=wp.dot(n1,n2)
    g1=wp.skew(edge)*float(o==0)-wp.skew(x03)*float(o==2)+wp.skew(x02)*float(o==3)
    g2=-wp.skew(edge)*float(o==1)+wp.skew(x13)*float(o==2)-wp.skew(x12)*float(o==3)
    d1=compute_normalized_vector_derivative(l1,n1,g1);d2=compute_normalized_vector_derivative(l2,n2,g2)
    return compute_angle_derivative(n1,n2,axis,d1,d2,sine,cosine,wp.skew(n1),wp.skew(n2))

@wp.kernel
def elasticity(q:wp.array[wp.vec3],prev:wp.array[wp.vec3],tri:wp.array2d[int],pose:wp.array[wp.mat22],
    area:wp.array[float],mat:wp.array2d[float],edges:wp.array2d[int],rest:wp.array[float],length:wp.array[float],
    stiffness:wp.array2d[float],dt:float,force:wp.array[wp.vec3],diag:wp.array[wp.mat33]):
    i=wp.tid()
    if i<tri.shape[0]*3:
        t=i//3;o=i%3;v=tri[t,o]
        f,h=membrane(t,o,q,prev,tri,pose[t],area[t],mat[t,0],mat[t,1],mat[t,2],dt)
        wp.atomic_add(force,v,f);wp.atomic_add(diag,v,h)
    if i<edges.shape[0]*4:
        e=i//4;o=i%4;v=edges[e,o]
        if v>=0:
            f,h=bending(e,o,q,prev,edges,rest,length,stiffness[e,0],stiffness[e,1],dt)
            wp.atomic_add(force,v,f);wp.atomic_add(diag,v,h)

@wp.kernel
def rhs(q:wp.array[wp.vec3],target:wp.array[wp.vec3],mass:wp.array[float],dt:float,
    force:wp.array[wp.vec3],diag:wp.array[wp.mat33],out:wp.array[wp.vec3]):
    i=wp.tid();k=mass[i]/(dt*dt)
    out[i]=force[i]+k*(target[i]-q[i]);diag[i]+=k*wp.identity(n=3,dtype=float)
    if mass[i]==0.:out[i]=wp.vec3(0.);diag[i]=wp.identity(n=3,dtype=float)

@wp.kernel
def invert(matrix:wp.array[wp.mat33],inverse:wp.array[wp.mat33]):
    i=wp.tid();inverse[i]=wp.inverse(matrix[i])

@wp.kernel
def block_mv(matrix:wp.array[wp.mat33],x:wp.array[wp.vec3],y:wp.array[wp.vec3],alpha:float,beta:float,z:wp.array[wp.vec3]):
    i=wp.tid();z[i]=alpha*(matrix[i]*x[i])+beta*y[i]

@wp.kernel
def begin_mv(mass:wp.array[float],contact:wp.array[wp.mat33],x:wp.array[wp.vec3],dt:float,out:wp.array[wp.vec3]):
    i=wp.tid();out[i]=(mass[i]/(dt*dt))*x[i]+4.*contact[i]*x[i]

@wp.kernel
def membrane_mv(q:wp.array[wp.vec3],x:wp.array[wp.vec3],tri:wp.array2d[int],pose:wp.array[wp.mat22],
    area:wp.array[float],mat:wp.array2d[float],dt:float,out:wp.array[wp.vec3]):
    t=wp.tid();a=tri[t,0];b=tri[t,1];c=tri[t,2];P=pose[t]
    f0=(q[b]-q[a])*P[0,0]+(q[c]-q[a])*P[1,0]
    f1=(q[b]-q[a])*P[0,1]+(q[c]-q[a])*P[1,1]
    d0=(x[b]-x[a])*P[0,0]+(x[c]-x[a])*P[1,0]
    d1=(x[b]-x[a])*P[0,1]+(x[c]-x[a])*P[1,1]
    c00=2.*wp.dot(f0,d0);c01=wp.dot(f0,d1)+wp.dot(f1,d0);c11=2.*wp.dot(f1,d1)
    shear=mat[t,0]+2.*mat[t,2]/dt;vol=.5*mat[t,1]*(c00+c11)
    p0=area[t]*(shear*(f0*c00+f1*c01)+f0*vol)
    p1=area[t]*(shear*(f0*c01+f1*c11)+f1*vol)
    fb=P[0,0]*p0+P[0,1]*p1;fc=P[1,0]*p0+P[1,1]*p1
    wp.atomic_add(out,a,-fb-fc);wp.atomic_add(out,b,fb);wp.atomic_add(out,c,fc)

@wp.kernel
def bending_mv(q:wp.array[wp.vec3],x:wp.array[wp.vec3],edges:wp.array2d[int],length:wp.array[float],
    stiffness:wp.array2d[float],dt:float,out:wp.array[wp.vec3]):
    e=wp.tid()
    if edges[e,0]<0 or edges[e,1]<0:return
    g0=hinge_gradient(e,0,q,edges);g1=hinge_gradient(e,1,q,edges)
    g2=hinge_gradient(e,2,q,edges);g3=hinge_gradient(e,3,q,edges)
    projection=wp.dot(g0,x[edges[e,0]])+wp.dot(g1,x[edges[e,1]])+wp.dot(g2,x[edges[e,2]])+wp.dot(g3,x[edges[e,3]])
    coefficient=length[e]*(stiffness[e,0]+stiffness[e,1]/dt)*projection
    wp.atomic_add(out,edges[e,0],coefficient*g0);wp.atomic_add(out,edges[e,1],coefficient*g1)
    wp.atomic_add(out,edges[e,2],coefficient*g2);wp.atomic_add(out,edges[e,3],coefficient*g3)

@wp.kernel
def end_mv(mass:wp.array[float],value:wp.array[wp.vec3],x:wp.array[wp.vec3],y:wp.array[wp.vec3],
    alpha:float,beta:float,z:wp.array[wp.vec3]):
    i=wp.tid();v=value[i]
    if mass[i]==0.:v=x[i]
    z[i]=alpha*v+beta*y[i]

@wp.kernel
def weighted_norm(force:wp.array[wp.vec3],inverse:wp.array[wp.mat33],out:wp.array[float]):
    i=wp.tid();wp.atomic_add(out,0,wp.dot(force[i],inverse[i]*force[i]))

@wp.kernel
def candidate(original:wp.array[wp.vec3],delta:wp.array[wp.vec3],scale:float,trial:wp.array[wp.vec3]):
    i=wp.tid();trial[i]=original[i]+scale*delta[i]

@wp.kernel
def choose(before:wp.array[float],after:wp.array[float],accepted:wp.array[int],scale:float,chosen:wp.array[float]):
    accepted[0]=int(chosen[0]==0. and after[0]<before[0])
    if accepted[0]!=0:chosen[0]=scale

@wp.kernel
def save(q:wp.array[wp.vec3],displacement:wp.array[wp.vec3],accepted:wp.array[int],bestq:wp.array[wp.vec3],bestd:wp.array[wp.vec3]):
    i=wp.tid()
    if accepted[0]!=0:bestq[i]=q[i];bestd[i]=displacement[i]

@wp.kernel
def metrics(slot:int,iterations:wp.array[int],linear_residual:wp.array[wp.float64],linear_tol:wp.array[wp.float64],
    before:wp.array[float],after:wp.array[float],chosen:wp.array[float],record:wp.array2d[float]):
    worst=int(0);ratio=float(-1.)
    for i in range(linear_residual.shape[0]):
        value=float(linear_residual[i]/wp.max(linear_tol[i],wp.float64(1e-30)))
        if value>ratio:ratio=value;worst=i
    # Preserve residual/tolerance together for the worst normalized batch.
    # Warp returns SQUARED norms. Report actual force norms, not their squares.
    record[slot,0]=float(iterations[0]);record[slot,1]=float(wp.sqrt(linear_residual[worst]));record[slot,2]=float(wp.sqrt(linear_tol[worst]))
    record[slot,3]=before[0];record[slot,4]=after[0];record[slot,5]=chosen[0]

@wp.kernel
def snapshot(slot:int,q:wp.array[wp.vec3],f:wp.array[wp.vec3],d:wp.array[wp.vec3],
             oq:wp.array2d[wp.vec3],of:wp.array2d[wp.vec3],od:wp.array2d[wp.vec3]):
    i=wp.tid();oq[slot,i]=q[i];of[slot,i]=f[i];od[slot,i]=d[i]

@wp.kernel
def line_debug(slot:int,j:int,before:wp.array[float],after:wp.array[float],values:wp.array[wp.float64],
               bound:wp.array[float],chosen:wp.array[float],record:wp.array3d[float]):
    record[slot,j,0]=before[0];record[slot,j,1]=after[0]
    record[slot,j,2]=float(values[0]);record[slot,j,3]=float(values[1])
    record[slot,j,4]=bound[0];record[slot,j,5]=chosen[0]

@wp.kernel
def needs_trial(chosen:wp.array[float],condition:wp.array[int]):
    condition[0]=int(chosen[0]==0.)

@wp.kernel
def clear_lines(slot:int,record:wp.array3d[float]):
    j,k=wp.tid();record[slot,j,k]=0.
    if k==4:record[slot,j,k]=-1.  # Explicitly not evaluated this refinement.

class GlobalIteration:
    def __init__(self,solver,surfaces):
        self.s=solver;self.contact=Translation(solver,surfaces);m=solver.model;self.device=m.device;n=m.particle_count
        self.q=wp.zeros_like(m.particle_q);self.force=wp.zeros_like(m.particle_q);self.delta=wp.zeros_like(m.particle_q)
        self.mv=wp.zeros_like(m.particle_q);self.contact_matrix=wp.zeros(n,dtype=wp.mat33,device=m.device)
        self.internal_matrix=wp.zeros_like(self.contact_matrix)
        self.inverse=wp.zeros_like(self.contact_matrix)
        self.trial=wp.zeros_like(m.particle_q);self.displacement=wp.zeros_like(m.particle_q)
        self.bestq=wp.zeros_like(m.particle_q);self.bestd=wp.zeros_like(m.particle_q)
        self.ts=wp.ones(n,dtype=float,device=m.device)
        self.before=wp.zeros(1,dtype=float,device=m.device);self.after=wp.zeros_like(self.before)
        self.accepted=wp.zeros(1,dtype=int,device=m.device);self.chosen=wp.zeros_like(self.before);self.dt=1/960
        from .coarse_modes import CoarseModes
        self.coarse=CoarseModes(m,surfaces)
        # This SEARCH operator is block diagonal by material shell: contact
        # curvature is diagonal. Keep each independent Krylov reduction separate
        # so stiff metal residuals cannot determine the film's CG step length.
        # Cross-object forces still couple the actual nonlinear evaluations.
        self.offsets=wp.array([0]+[3*a['first'] for a in surfaces[1:]]+[3*n],dtype=int,device=m.device)
        from .linear64 import Linear64
        self.linear=Linear64(self)
        A=LinearOperator((n,n),wp.mat33d,m.device,self.linear.matvec,batch_offsets=self.offsets)
        M=LinearOperator((n,n),wp.mat33d,m.device,self.linear.precondition,batch_offsets=self.offsets)
        self.cg=CG(A,self.linear.rhs,self.linear.delta,M=M,tol=1e-4,atol=1e-7,maxiter=160,check_every=0,use_cuda_graph=True)
        self.metrics=wp.zeros((256,6),dtype=float,device=m.device);self.metric_slot=0
        self.debug=None
        from .force_work import ForceWork
        self.work=ForceWork(self);self.merit='norm'
        self.need_trial=wp.ones(1,dtype=int,device=m.device);self.early_backtracking=True
        self.force_only_evaluation=False  # Enable only after same-query audit.
        self.line_trials=6
        self.precision=None
    def matvec(self,x,y,z,alpha,beta):
        s=self.s;m=s.model;dt=self.dt
        wp.launch(begin_mv,m.particle_count,[m.particle_mass,self.contact_matrix,x,dt,self.mv],device=self.device)
        wp.launch(membrane_mv,m.tri_count,[self.q,x,m.tri_indices,m.tri_poses,m.tri_areas,m.tri_materials,dt,self.mv],device=self.device)
        wp.launch(bending_mv,m.edge_count,[self.q,x,m.edge_indices,m.edge_rest_length,m.edge_bending_properties,dt,self.mv],device=self.device)
        wp.launch(end_mv,m.particle_count,[m.particle_mass,self.mv,x,y,float(alpha),float(beta),z],device=self.device)
    def precondition(self,x,y,z,alpha,beta):
        self.coarse.apply(self.q,self.inverse,x,y,z,alpha,beta)
    def evaluate(self,q,state,c,dt,*,hessians=True,contact_ready=False,precise_q=None):
        s=self.s;m=s.model
        from . import force_only
        if not contact_ready:self.contact.evaluate(q,state,c,dt,aggregate_groups=False,hessians=hessians)
        if self.precision:
            exact=self.precision.current if precise_q is None else precise_q
            self.precision.evaluate(exact,dt,self.contact.force,self.contact.hessian,self.force,hessians=hessians)
            return
        ek=elasticity if hessians else force_only.elasticity
        rk=rhs if hessians else force_only.rhs
        wp.launch(ek,max(m.tri_count*3,m.edge_count*4),[q,s.particle_q_prev,m.tri_indices,m.tri_poses,m.tri_areas,
            m.tri_materials,m.edge_indices,m.edge_rest_angle,m.edge_rest_length,m.edge_bending_properties,dt,
            self.contact.force,self.contact.hessian],device=self.device)
        wp.launch(rk,m.particle_count,[q,s.inertia,m.particle_mass,dt,self.contact.force,self.contact.hessian,self.force],device=self.device)
    def refine(self,state,out,c,dt):
        s=self.s;m=s.model;self.dt=dt;wp.copy(self.q,state.particle_q)
        if self.precision:
            wp.copy(self.precision.reference,self.precision.current)
            wp.copy(self.precision.best,self.precision.current)
        self.contact.evaluate(self.q,state,c,dt,aggregate_groups=False)
        wp.copy(self.contact_matrix,self.contact.external);wp.copy(self.internal_matrix,self.contact.internal)
        self.evaluate(self.q,state,c,dt,contact_ready=True);wp.launch(invert,m.particle_count,[self.contact.hessian,self.inverse],device=self.device)
        self.coarse.prepare(self.q,self.contact_matrix,dt,self.internal_matrix)
        self.before.zero_();wp.launch(weighted_norm,m.particle_count,[self.force,self.inverse,self.before],device=self.device)
        self.linear.prepare_rhs();cg_status=self.cg();self.linear.retrieve()
        self.chosen.zero_();self.bestq.assign(state.particle_q);self.bestd.assign(s.particle_displacements)
        slot=self.metric_slot%256;self.metric_slot+=1
        if self.debug is not None:
            wp.launch(clear_lines,(self.line_trials,6),[slot%12,self.debug['line']],device=self.device)
            wp.launch(snapshot,m.particle_count,[slot%12,self.q,self.force,self.delta,
                self.debug['q'],self.debug['rhs'],self.debug['delta']],device=self.device)
        def trial_step(j,scale):
            s.relative_step.increment(self.q,self.delta,scale,self.displacement,self.trial)
            if self.precision:
                from .elastic_precision import candidate
                wp.launch(candidate,m.particle_count,[self.precision.reference,self.delta,s.relative_step.scale,float(scale),
                    self.precision.trial,self.trial,s.pos_prev_collision_detection,self.displacement],device=self.device)
            if not (getattr(self,'work_shortcuts',False) and self.merit=='work' and self.debug is None):
                # Work acceptance does not consume this endpoint norm. Keep it
                # whenever diagnostics request it, and for the norm merit.
                self.evaluate(self.trial,state,c,dt,hessians=not self.force_only_evaluation,
                    precise_q=self.precision.trial if self.precision else None);self.after.zero_()
                wp.launch(weighted_norm,m.particle_count,[self.force,self.inverse,self.after],device=self.device)
            if self.merit=='work':
                from .force_work import choose as choose_work
                self.work.evaluate(self.trial,state,c,dt)
                wp.launch(choose_work,1,[self.work.values,self.accepted,scale,self.chosen],device=self.device)
            else:
                wp.launch(choose,1,[self.before,self.after,self.accepted,scale,self.chosen],device=self.device)
            wp.launch(save,m.particle_count,[self.trial,self.displacement,self.accepted,self.bestq,self.bestd],device=self.device)
            if self.precision:
                from .elastic_precision import accept
                wp.launch(accept,m.particle_count,[self.precision.trial,self.accepted,self.precision.best],device=self.device)
            if self.debug is not None:
                wp.launch(line_debug,1,[slot%12,j,self.before,self.after,self.work.values,
                    s.relative_step.scale,self.chosen,self.debug['line']],device=self.device)
        for j in range(self.line_trials):
            scale=2.**(-j)
            if self.early_backtracking:
                # The old path already keeps the FIRST accepted trial. Later
                # trial evaluations cannot change it; skip that identical work.
                wp.launch(needs_trial,1,[self.chosen,self.need_trial],device=self.device)
                wp.capture_if(self.need_trial,trial_step,j=j,scale=scale)
            else:trial_step(j,scale)
        wp.copy(state.particle_q,self.bestq);wp.copy(out.particle_q,self.bestq);wp.copy(s.particle_displacements,self.bestd)
        if self.precision:wp.copy(self.precision.current,self.precision.best)
        self.evaluate(self.bestq,state,c,dt,hessians=not self.force_only_evaluation);self.after.zero_()
        wp.launch(weighted_norm,m.particle_count,[self.force,self.inverse,self.after],device=self.device)
        wp.launch(metrics,1,[slot,*cg_status,self.before,self.after,self.chosen,self.metrics],device=self.device)
