"""Same installed springs, evaluated consistently on both sides in float64.

No new attachment or compliance: this is the twelve existing bag/frame locks.
Explicit installation is limited to the residual-Newton numerical candidate.
"""
from types import MethodType
import warp as wp
from .relative_step import double
from .elastic_precision import single
from newton._src.solvers.vbd.particle_vbd_kernels import evaluate_spring_force_and_hessian_both_vertices


@wp.func
def transform64(pose:wp.transform):
    t=wp.transform_get_translation(pose);q=wp.transform_get_rotation(pose)
    return wp.transformd(double(t),wp.quatd(wp.float64(q[0]),wp.float64(q[1]),wp.float64(q[2]),wp.float64(q[3])))


@wp.kernel
def spring_correction(dt:float,q:wp.array[wp.vec3],previous:wp.array[wp.vec3],
    exact:wp.array[wp.vec3d],old_exact:wp.array[wp.vec3d],indices:wp.array[int],
    rest:wp.array[float],stiffness:wp.array[float],damping:wp.array[float],force:wp.array[wp.vec3]):
    i=wp.tid();a,b,fa,fb,H=evaluate_spring_force_and_hessian_both_vertices(i,dt,q,previous,indices,rest,stiffness,damping)
    dx=exact[a]-exact[b];length=wp.max(wp.length(dx),wp.float64(1e-8))
    old_length=wp.max(wp.length(old_exact[a]-old_exact[b]),wp.float64(1e-8))
    value=wp.float64(stiffness[i])*(wp.float64(rest[i])/length-wp.float64(1.))*dx
    value-=dx/length*wp.float64(damping[i]/dt)*(length-old_length)
    f=single(value)
    wp.atomic_add(force,a,f-fa);wp.atomic_add(force,b,-f-fb)


@wp.kernel
def follow(anchors:wp.array[int],bodies:wp.array[int],local:wp.array[wp.vec3],
           bq:wp.array[wp.transform],q:wp.array[wp.vec3],precise:wp.array[wp.vec3d]):
    i=wp.tid();x=wp.transform_point(transform64(bq[bodies[i]]),double(local[i]))
    q[anchors[i]]=single(x);precise[anchors[i]]=x


@wp.kernel
def accumulate(group:wp.array[int],pins:wp.array[int],bodies:wp.array[int],local:wp.array[wp.vec3],
    stiffness:float,bq:wp.array[wp.transform],com:wp.array[wp.vec3],q:wp.array[wp.vec3d],
    force:wp.array[wp.vec3],torque:wp.array[wp.vec3],Hll:wp.array[wp.mat33],
    Hal:wp.array[wp.mat33],Haa:wp.array[wp.mat33]):
    i=wp.tid();body=group[i]
    if body<0:return
    f=wp.vec3(0.);t=wp.vec3(0.);ll=wp.mat33(0.);al=wp.mat33(0.);aa=wp.mat33(0.)
    for j in range(pins.shape[0]):
        if bodies[j]==body:
            pose=transform64(bq[body]);x=wp.transform_point(pose,double(local[j]))
            r=single(x-wp.transform_point(pose,double(com[body])))
            value=single(wp.float64(stiffness)*(q[pins[j]]-x));R=wp.skew(r)
            f+=value;t+=wp.cross(r,value);ll+=stiffness*wp.identity(3,float)
            al+=stiffness*R;aa+=stiffness*wp.transpose(R)*R
    force[body]+=f;torque[body]+=t;Hll[body]+=ll;Hal[body]+=al;Haa[body]+=aa


@wp.kernel
def record(pins:wp.array[int],bodies:wp.array[int],local:wp.array[wp.vec3],stiffness:float,
           bq:wp.array[wp.transform],q:wp.array[wp.vec3d],out:wp.array[wp.vec3]):
    i=wp.tid();x=wp.transform_point(transform64(bq[bodies[i]]),double(local[i]))
    out[i]=single(wp.float64(stiffness)*(q[pins[i]]-x))


def install(coupled):
    s=coupled.native;g=s.global_refinement;p=g.precision
    retention=s._implicit_retention
    def exact_follow(self,state):
        c=self.c
        wp.launch(follow,c.anchor_ids.size,[c.anchor_ids,c.pin_bodies,c.pin_local,state.body_q,
            state.particle_q,p.current],device=c.device)
    def exact_accumulate(self,group,state):
        c=self.c;m=s.model
        wp.launch(accumulate,group.size,[group,c.pin_particles,c.pin_bodies,c.pin_local,c.retention_stiffness,
            state.body_q,m.body_com,p.current,s.body_forces,s.body_torques,
            s.body_hessian_ll,s.body_hessian_al,s.body_hessian_aa],device=c.device)
    def exact_record(self,state):
        c=self.c
        wp.launch(record,c.pin_particles.size,[c.pin_particles,c.pin_bodies,c.pin_local,c.retention_stiffness,
            state.body_q,p.current,c.retention_forces],device=c.device)
    retention.follow=MethodType(exact_follow,retention)
    retention.accumulate=MethodType(exact_accumulate,retention)
    retention.record=MethodType(exact_record,retention)
    evaluate=p.evaluate
    def exact_evaluate(q,dt,force,diag,out,*,hessians):
        m=s.model
        # Replace rounded spring force BEFORE inertia/massless filtering.
        if m.spring_count:
            wp.launch(spring_correction,m.spring_count,[dt,g.contact_query,s.particle_q_prev,q,p.previous,
                m.spring_indices,m.spring_rest_length,m.spring_stiffness,m.spring_damping,force],device=m.device)
        evaluate(q,dt,force,diag,out,hessians=hessians)
    original=g.evaluate
    def remember(q,*args,**kwargs):
        g.contact_query=q
        return original(q,*args,**kwargs)
    g.evaluate=remember;p.evaluate=exact_evaluate
    g.retention_precision=True
