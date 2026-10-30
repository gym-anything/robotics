"""Global translation-coordinate refinement of the existing cloth equations.

Translating every vertex of one separated cloth together leaves its membrane,
bending and self-contact distances unchanged. Their total force and directional Hessian
therefore vanish in this three-dimensional subspace. Evaluate the remaining
implicit inertia and EXACT existing body-contact law; never add a support force.

This is a residual-decreasing coarse numerical iteration, not a grasp constraint
or a sufficient local-convergence test. Backtracking rejects worse candidates.
The trust radius stays within the existing contact-discovery horizon.
"""
from types import FunctionType
import numpy as np
import warp as wp
from warp._src.context import Kernel
from newton._src.solvers.vbd.rigid_vbd_kernels import _eval_soft_ef_contact


@wp.kernel
def contact_totals(dt:float,pos:wp.array[wp.vec3],prev:wp.array[wp.vec3],radius:wp.array[float],
    body_q:wp.array[wp.transform],body_prev:wp.array[wp.transform],body_qd:wp.array[wp.spatial_vector],
    com:wp.array[wp.vec3],eps:float,ke:wp.array[float],kd:wp.array[float],mu:wp.array[float],
    count:wp.array[int],indices:wp.array[wp.vec3i],bary:wp.array[wp.vec3],shape:wp.array[int],
    point:wp.array[wp.vec3],velocity:wp.array[wp.vec3],normal:wp.array[wp.vec3],
    margin:wp.array[float],shape_body:wp.array[int],
    per_cloth:int,buckets:int,chosen:wp.array[float],total:wp.array[wp.vec3],hessian:wp.array[wp.mat33]):
    i=wp.tid()
    if i>=count[0]:return
    cloth=indices[i][0]//per_cloth
    if chosen[cloth]!=0.:return
    f,h,p=_eval_soft_ef_contact(i,indices[i],bary[i],pos,prev,radius,ke[i],kd[i],mu[i],eps,
        shape_body,body_q,body_prev,body_qd,com,shape,point,velocity,normal,margin,dt)
    weight=float(0.)
    for j in range(3):
        if indices[i][j]>=0:weight+=bary[i][j]
    group=cloth*buckets+i%buckets
    wp.atomic_add(total,group,weight*f)
    wp.atomic_add(hessian,group,weight*weight*h)


@wp.kernel
def inertial_totals(pos:wp.array[wp.vec3],target:wp.array[wp.vec3],mass:wp.array[float],dt:float,per_cloth:int,buckets:int,chosen:wp.array[float],
                    total:wp.array[wp.vec3],hessian:wp.array[wp.mat33]):
    i=wp.tid();k=mass[i]/(dt*dt)
    if chosen[i//per_cloth]!=0.:return
    group=(i//per_cloth)*buckets+i%buckets
    wp.atomic_add(total,group,k*(target[i]-pos[i]))
    wp.atomic_add(hessian,group,k*wp.identity(n=3,dtype=float))


@wp.kernel
def reduce_buckets(force:wp.array[wp.vec3],hess:wp.array[wp.mat33],buckets:int,
                   total:wp.array[wp.vec3],hessian:wp.array[wp.mat33]):
    g=wp.tid();f=wp.vec3(0.);h=wp.mat33(0.)
    for b in range(buckets):
        f+=force[g*buckets+b];h+=hess[g*buckets+b]
    total[g]=f;hessian[g]=h


@wp.kernel
def bounds(pos:wp.array[wp.vec3],per_cloth:int,lo:wp.array2d[float],hi:wp.array2d[float]):
    i=wp.tid();g=i//per_cloth
    for j in range(3):
        wp.atomic_min(lo,g,j,pos[i][j]);wp.atomic_max(hi,g,j,pos[i][j])


@wp.kernel
def direction(total:wp.array[wp.vec3],hessian:wp.array[wp.mat33],radius:float,
              lo:wp.array2d[float],hi:wp.array2d[float],exclusion:float,delta:wp.array[wp.vec3]):
    g=wp.tid();separated=bool(True)
    for other in range(total.shape[0]):
        if other!=g:
            gap=wp.vec3(0.)
            for j in range(3):gap[j]=wp.max(0.,wp.max(lo[g,j]-hi[other,j],lo[other,j]-hi[g,j]))
            if wp.length(gap)<=exclusion:separated=False
    d=wp.vec3(0.)
    if separated and wp.abs(wp.determinant(hessian[g]))>1.e-8:d=wp.inverse(hessian[g])*total[g]
    length=wp.length(d)
    if length>radius:d*=radius/length
    delta[g]=d


@wp.kernel
def trial(source:wp.array[wp.vec3],delta:wp.array[wp.vec3],scale:float,per_cloth:int,dest:wp.array[wp.vec3]):
    i=wp.tid();dest[i]=source[i]+scale*delta[i//per_cloth]


@wp.kernel
def accept(before:wp.array[wp.vec3],after:wp.array[wp.vec3],chosen:wp.array[float],scale:float):
    g=wp.tid()
    if chosen[g]==0. and wp.dot(after[g],after[g])<wp.dot(before[g],before[g]):chosen[g]=scale


@wp.kernel
def apply(source:wp.array[wp.vec3],out:wp.array[wp.vec3],displacements:wp.array[wp.vec3],
          delta:wp.array[wp.vec3],chosen:wp.array[float],per_cloth:int):
    i=wp.tid();g=i//per_cloth;d=chosen[g]*delta[g]
    source[i]+=d;out[i]=source[i];displacements[i]+=d


@wp.kernel
def pending_candidates(chosen:wp.array[float],remaining:wp.array[int]):
    pending=int(0)
    for i in range(chosen.shape[0]):
        if chosen[i]==0.:pending=1
    remaining[0]=pending


class TranslationRefinement:
    def __init__(self,solver,cloth_count=1):
        self.solver=solver;m=solver.model;self.device=m.device
        self.groups=cloth_count;self.per_cloth=m.particle_count//cloth_count
        if self.per_cloth*cloth_count!=m.particle_count:raise ValueError('Expected equal disjoint cloth topology')
        triangles=m.tri_indices.numpy()
        if len(triangles) and np.any(triangles.min(1)//self.per_cloth!=triangles.max(1)//self.per_cloth):
            raise ValueError('Cross-cloth membrane topology requires a coupled coarse block')
        if not np.all((m.particle_flags.numpy()&1)!=0) or not np.all(m.particle_mass.numpy()>0):
            raise ValueError('Global translation cancellation requires all cloth vertices dynamic')
        self.total=wp.zeros(cloth_count,dtype=wp.vec3,device=self.device)
        self.hessian=wp.zeros(cloth_count,dtype=wp.mat33,device=self.device)
        # Same terms, staged reduction. Thousands of threads contending on just
        # three global totals serialize the original implementation. Bucketing
        # changes floating-point summation order, not equations or resolution.
        self.buckets=64
        self.bucket_force=wp.zeros(cloth_count*self.buckets,dtype=wp.vec3,device=self.device)
        self.bucket_hessian=wp.zeros(cloth_count*self.buckets,dtype=wp.mat33,device=self.device)
        self.initial=wp.zeros_like(self.total);self.delta=wp.zeros_like(self.total)
        self.chosen=wp.zeros(cloth_count,dtype=float,device=self.device)
        # Execution-only candidate: public/default path stays unchanged until
        # matched GPU-output and full-runtime checks establish the benefit.
        self.conditional_backtracking=False
        self.pending=wp.zeros(1,dtype=int,device=self.device)
        self.batched=None
        self.lo=wp.zeros((cloth_count,3),dtype=float,device=self.device)
        self.hi=wp.zeros_like(self.lo)
        self.exclusion=4*float(m.particle_radius.numpy().max())+.0002
        self.trial=wp.zeros_like(m.particle_q)
        # Reuse the exact contact evaluator already selected for physical solve
        # and feedback, including the qualified compression-core candidate.
        selected=solver.record_contact_kernel.func.__globals__['_eval_soft_ef_contact']
        namespace=dict(contact_totals.func.__globals__);namespace['_eval_soft_ef_contact']=selected
        f=contact_totals.func
        copy=FunctionType(f.__code__,namespace,f.__name__,f.__defaults__,f.__closure__)
        copy.__annotations__=dict(f.__annotations__);copy.__module__=__name__
        self.kernel=Kernel(copy,key='contact_totals_'+selected.func.__module__.replace('.','_')+'_'+selected.key,
                           module=wp.get_module(__name__))

    def evaluate(self,positions,state_in,state_out,contacts,dt):
        s=self.solver;m=s.model;c=contacts
        if s.integrate_with_external_rigid_solver:
            body,previous,velocity=state_out.body_q,state_in.body_q,state_out.body_qd
        else:body,previous,velocity=state_in.body_q,s.body_q_prev,state_in.body_qd
        self.bucket_force.zero_();self.bucket_hessian.zero_()
        wp.launch(self.kernel,c.soft_contact_max,[dt,positions,s.particle_q_prev,m.particle_radius,
            body,previous,velocity,m.body_com,s.friction_epsilon,s.body_particle_contact_penalty_k,
            s.body_particle_contact_material_kd,s.body_particle_contact_material_mu,
            c.soft_contact_count,c.soft_contact_indices,c.soft_contact_barycentric,c.soft_contact_shape,
            c.soft_contact_body_pos,c.soft_contact_body_vel,c.soft_contact_normal,m.shape_margin,m.shape_body,
            self.per_cloth,self.buckets,self.chosen,self.bucket_force,self.bucket_hessian],device=self.device)
        wp.launch(inertial_totals,m.particle_count,[positions,s.inertia,m.particle_mass,dt,self.per_cloth,
            self.buckets,self.chosen,self.bucket_force,self.bucket_hessian],device=self.device)
        wp.launch(reduce_buckets,self.groups,[self.bucket_force,self.bucket_hessian,self.buckets,self.total,self.hessian],device=self.device)

    def refine(self,state_in,state_out,contacts,dt):
        s=self.solver;m=s.model
        # The algorithm accepts the FIRST improving backtracking candidate.
        # Later candidate evaluations cannot affect accepted groups; skip only
        # that redundant arithmetic, retaining all four scales for the others.
        self.chosen.zero_()
        self.evaluate(state_in.particle_q,state_in,state_out,contacts,dt)
        wp.copy(self.initial,self.total)
        self.lo.fill_(float('inf'));self.hi.fill_(-float('inf'))
        wp.launch(bounds,m.particle_count,[state_in.particle_q,self.per_cloth,self.lo,self.hi],device=self.device)
        wp.launch(direction,self.groups,[self.total,self.hessian,.0001,self.lo,self.hi,self.exclusion,self.delta],device=self.device)
        def candidate(scale):
            wp.launch(trial,m.particle_count,[state_in.particle_q,self.delta,scale,self.per_cloth,self.trial],device=self.device)
            self.evaluate(self.trial,state_in,state_out,contacts,dt)
            wp.launch(accept,self.groups,[self.initial,self.total,self.chosen,scale],device=self.device)
        if self.batched is not None:self.batched.evaluate(state_in,state_out,contacts,dt)
        else:
            for scale in (1.,.5,.25,.125):
                if self.conditional_backtracking and wp.get_device(self.device).is_capturing:
                    wp.launch(pending_candidates,1,[self.chosen,self.pending],device=self.device)
                    wp.capture_if(self.pending,candidate,scale=scale)
                else:candidate(scale)
        wp.launch(apply,m.particle_count,[state_in.particle_q,state_out.particle_q,
            s.particle_displacements,self.delta,self.chosen,self.per_cloth],device=self.device)
