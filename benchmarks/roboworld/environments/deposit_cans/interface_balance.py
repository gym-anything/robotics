"""Same-query shell-to-rigid reaction balance for the endpoint block solve.

This evaluates forces; it does not integrate, change targets or introduce a
physical force. Both sides use contact_residual_precision.rigid_point.
"""
import numpy as np
import warp as wp
from .contact_residual_precision import rigid_point
from .retention_precision import transform64
from .relative_step import double


@wp.kernel
def contacts(q:wp.array[wp.vec3d],prev:wp.array[wp.vec3d],radius:wp.array[float],
    indices:wp.array[wp.vec3i],count:wp.array[int],capacity:int,ke:wp.array[float],kd:wp.array[float],mu:wp.array[float],
    shape_body:wp.array[int],bq:wp.array[wp.transform],bprev:wp.array[wp.transform],bqd:wp.array[wp.spatial_vector],
    com:wp.array[wp.vec3],shape:wp.array[int],local:wp.array[wp.vec3],velocity:wp.array[wp.vec3],
    normal:wp.array[wp.vec3],margin:wp.array[float],barycentric:wp.array[wp.vec3],eps:wp.float64,dt:wp.float64,
    force:wp.array[wp.vec3d],torque:wp.array[wp.vec3d]):
    i=wp.tid()
    if i>=wp.min(count[0],capacity):return
    ids=indices[i];a=ids[0]
    if a<0:return
    body=shape_body[shape[i]]
    if body<0:return
    p=q[a];previous=prev[a];r=wp.float64(radius[a])
    if ids[1]>=0:
        w=double(barycentric[i]);p=w[0]*q[a];previous=w[0]*prev[a]
        for j in range(1,3):
            v=ids[j]
            if v>=0:p+=w[j]*q[v];previous+=w[j]*prev[v];r=wp.max(r,wp.float64(radius[v]))
    f,bx=rigid_point(p,previous,r,i,ke[i],kd[i],mu[i],shape_body,bq,bprev,bqd,com,
        shape,local,velocity,normal,margin,eps,dt)
    center=wp.transform_point(transform64(bq[body]),double(com[body]))
    wp.atomic_add(force,body,-f);wp.atomic_add(torque,body,wp.cross(bx-center,-f))


@wp.kernel
def springs(pins:wp.array[int],bodies:wp.array[int],local:wp.array[wp.vec3],stiffness:wp.float64,
    bq:wp.array[wp.transform],com:wp.array[wp.vec3],q:wp.array[wp.vec3d],
    force:wp.array[wp.vec3d],torque:wp.array[wp.vec3d]):
    i=wp.tid();body=bodies[i];pose=transform64(bq[body])
    point=wp.transform_point(pose,double(local[i]));center=wp.transform_point(pose,double(com[body]))
    f=stiffness*(q[pins[i]]-point)
    wp.atomic_add(force,body,f);wp.atomic_add(torque,body,wp.cross(point-center,f))


class InterfaceBalance:
    def __init__(self,coupled):
        self.c=coupled;s=coupled.native;m=s.model
        self.force=wp.zeros(m.body_count,dtype=wp.vec3d,device=m.device)
        self.torque=wp.zeros_like(self.force)
        # Accumulate all body reactions, but apply the convergence comparison
        # only to dynamic bodies; kinematic anchors cannot respond.
        self.dynamic=m.body_inv_mass.numpy()>0.

    def evaluate(self,q,state,contact,dt,*,penalty=None):
        c=self.c;s=c.native;m=s.model;p=s.global_refinement.precision
        self.force.zero_();self.torque.zero_()
        wp.launch(contacts,contact.soft_contact_max,[q,p.previous,m.particle_radius,
            contact.soft_contact_indices,contact.soft_contact_count,contact.soft_contact_max,
            s.body_particle_contact_penalty_k if penalty is None else penalty,s.body_particle_contact_material_kd,s.body_particle_contact_material_mu,
            m.shape_body,state.body_q,s.body_q_prev,state.body_qd,m.body_com,
            contact.soft_contact_shape,contact.soft_contact_body_pos,contact.soft_contact_body_vel,
            contact.soft_contact_normal,m.shape_margin,contact.soft_contact_barycentric,
            wp.float64(s.friction_epsilon),wp.float64(dt),self.force,self.torque],device=m.device)
        wp.launch(springs,c.pin_particles.size,[c.pin_particles,c.pin_bodies,c.pin_local,wp.float64(float(np.float32(c.retention_stiffness))),
            state.body_q,m.body_com,q,self.force,self.torque],device=m.device)
        result=np.concatenate((self.force.numpy(),self.torque.numpy()),axis=1)
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite shell/rigid interface wrench')
        return result[self.dynamic]

    @staticmethod
    def difference(before,after):
        delta=after-before
        if not len(delta):return 0.,0.
        values=(float(np.linalg.norm(delta[:,:3],axis=1).max()),float(np.linalg.norm(delta[:,3:],axis=1).max()))
        if not np.isfinite(values).all():raise FloatingPointError('Nonfinite interface convergence change')
        return values
