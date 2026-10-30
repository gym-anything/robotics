"""Additive rigid-mode preconditioner; no pose projection or material change.

Each shell contributes translation/rotation near-null modes of elasticity.
The fine block-Jacobi term remains SPD; each additive coarse term is P C^-1 P^T.
C uses inertial/contact search curvature, including internal-contact rotation. Translation and rotation use separate
3x3 blocks (an approximation to the full 6x6 coarse inverse). This changes PCG
conditioning only; every vertex, shell force and contact remains in the solve.
Basis: PETSc MatNullSpaceCreateRigidBody / hypre elasticity near-null modes.
"""
import numpy as np
import warp as wp

@wp.kernel
def centers(q:wp.array[wp.vec3],groups:wp.array[int],counts:wp.array[float],out:wp.array[wp.vec3]):
    i=wp.tid();g=groups[i]
    if g>=0:wp.atomic_add(out,g,q[i]/counts[g])

@wp.kernel
def matrices(q:wp.array[wp.vec3],groups:wp.array[int],center:wp.array[wp.vec3],mass:wp.array[float],
    contact:wp.array[wp.mat33],dt:float,linear:wp.array[wp.mat33],angular:wp.array[wp.mat33]):
    i=wp.tid();g=groups[i]
    if g<0:return
    K=(mass[i]/(dt*dt))*wp.identity(n=3,dtype=float)+4.*contact[i]
    J=-wp.skew(q[i]-center[g])
    wp.atomic_add(linear,g,K);wp.atomic_add(angular,g,wp.transpose(J)*K*J)

@wp.kernel
def internal_rotation(q:wp.array[wp.vec3],groups:wp.array[int],center:wp.array[wp.vec3],
    contact:wp.array[wp.mat33],angular:wp.array[wp.mat33]):
    i=wp.tid();g=groups[i]
    if g<0:return
    # For rotations around the unweighted vertex centroid, mean(J*w)=0.
    # Hence R^T P^T(4D)P R = sum J_i^T(4D_i)J_i. Translation stays null.
    J=-wp.skew(q[i]-center[g])
    wp.atomic_add(angular,g,4.*wp.transpose(J)*contact[i]*J)

@wp.kernel
def inverses(linear:wp.array[wp.mat33],angular:wp.array[wp.mat33],
    il:wp.array[wp.mat33],ia:wp.array[wp.mat33]):
    g=wp.tid();il[g]=wp.inverse(linear[g]);ia[g]=wp.inverse(angular[g])

@wp.kernel
def restrict(q:wp.array[wp.vec3],groups:wp.array[int],center:wp.array[wp.vec3],x:wp.array[wp.vec3],
    linear:wp.array[wp.vec3],angular:wp.array[wp.vec3]):
    i=wp.tid();g=groups[i]
    if g<0:return
    wp.atomic_add(linear,g,x[i]);wp.atomic_add(angular,g,wp.cross(q[i]-center[g],x[i]))

@wp.kernel
def prolong(q:wp.array[wp.vec3],groups:wp.array[int],center:wp.array[wp.vec3],
    il:wp.array[wp.mat33],ia:wp.array[wp.mat33],linear:wp.array[wp.vec3],angular:wp.array[wp.vec3],
    diagonal:wp.array[wp.mat33],x:wp.array[wp.vec3],y:wp.array[wp.vec3],alpha:float,beta:float,z:wp.array[wp.vec3]):
    i=wp.tid();g=groups[i];v=diagonal[i]*x[i]
    if g>=0:v+=il[g]*linear[g]+wp.cross(ia[g]*angular[g],q[i]-center[g])
    z[i]=alpha*v+beta*y[i]

class CoarseModes:
    def __init__(self,model,surfaces):
        self.model=model;d=model.device;group=np.full(model.particle_count,-1,np.int32)
        for i,a in enumerate(surfaces):group[a['first']:a['first']+a['count']]=i
        self.groups=wp.array(group,dtype=int,device=d)
        self.starts=wp.array([a['first'] for a in surfaces],dtype=int,device=d)
        self.sizes=wp.array([a['count'] for a in surfaces],dtype=int,device=d)
        self.counts=wp.array([a['count'] for a in surfaces],dtype=float,device=d)
        n=len(surfaces);self.center=wp.zeros(n,dtype=wp.vec3,device=d)
        self.linear=wp.zeros(n,dtype=wp.mat33,device=d);self.angular=wp.zeros_like(self.linear)
        self.il=wp.zeros_like(self.linear);self.ia=wp.zeros_like(self.linear)
        self.f=wp.zeros_like(self.center);self.t=wp.zeros_like(self.center)
    def prepare(self,q,contact,dt,internal=None):
        m=self.model;d=m.device
        self.center.zero_();self.linear.zero_();self.angular.zero_()
        wp.launch(centers,m.particle_count,[q,self.groups,self.counts,self.center],device=d)
        wp.launch(matrices,m.particle_count,[q,self.groups,self.center,m.particle_mass,contact,dt,self.linear,self.angular],device=d)
        if internal is not None:
            wp.launch(internal_rotation,m.particle_count,[q,self.groups,self.center,internal,self.angular],device=d)
        wp.launch(inverses,self.center.size,[self.linear,self.angular,self.il,self.ia],device=d)
    def apply(self,q,diagonal,x,y,z,alpha,beta):
        m=self.model;d=m.device;self.f.zero_();self.t.zero_()
        wp.launch(restrict,m.particle_count,[q,self.groups,self.center,x,self.f,self.t],device=d)
        wp.launch(prolong,m.particle_count,[q,self.groups,self.center,self.il,self.ia,self.f,self.t,
            diagonal,x,y,float(alpha),float(beta),z],device=d)
