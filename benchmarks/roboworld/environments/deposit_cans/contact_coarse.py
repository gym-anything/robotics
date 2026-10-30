"""Six-mode contact-consistent candidate preconditioner; same physical forces.

For each material surface form R^T (M/dt^2 + springs + contact) R using
the SAME frozen interpolation blocks as SurfaceSearch. Retain the complete
6x6 translation/rotation block, rather than separate3x3 approximations of
an obsolete diagonal contact bound. Cross-surface coarse blocks are omitted
only from this additive preconditioner; they remain in A and all forces.
Elasticity's rigid modes are near-null modes; fine block Jacobi is retained.
No positions, velocities, coefficients or acceptance bounds are changed.
"""
import warp as wp
from .linear64 import double,matrix

# These diagnostic kernels are never differentiated. Unrolling nested6x6
# loops generated4.4MB of CUDA and minutes of compile time in255.
wp.set_module_options({'enable_backward':False,'max_unroll':0})

@wp.func
def mode(j:int,r:wp.vec3d):
    axis=wp.vec3d(0.)
    axis[j%3]=wp.float64(1.)
    if j<3:return axis
    return wp.cross(axis,r)

@wp.kernel
def mass_blocks(q:wp.array[wp.vec3],groups:wp.array[int],center:wp.array[wp.vec3],
    mass:wp.array[float],base:wp.array[wp.mat33],dt:wp.float64,C:wp.array3d[wp.float64]):
    i=wp.tid();g=groups[i]
    if g<0:return
    r=double(q[i])-double(center[g])
    H=matrix(base[i])+wp.float64(mass[i])/(dt*dt)*wp.identity(n=3,dtype=wp.float64)
    for a in range(6):
        ja=mode(a,r)
        for b in range(6):
            wp.atomic_add(C,g,a,b,wp.dot(ja,H*mode(b,r)))

@wp.kernel
def contact_blocks(q:wp.array[wp.vec3],groups:wp.array[int],center:wp.array[wp.vec3],
    count:wp.array[int],ids:wp.array[wp.vec4i],weights:wp.array[wp.vec4],H:wp.array[wp.mat33],
    C:wp.array3d[wp.float64]):
    row=wp.tid()
    if row>=wp.min(count[0],ids.shape[0]):return
    v=ids[row];w=weights[row];h=matrix(H[row])
    for k in range(4):
        g=groups[v[k]];first=bool(g>=0)
        for l in range(k):
            if groups[v[l]]==g:first=False
        if first:
            for a in range(6):
                ja=wp.vec3d(0.)
                for i in range(4):
                    if groups[v[i]]==g:ja+=wp.float64(w[i])*mode(a,double(q[v[i]])-double(center[g]))
                for b in range(6):
                    jb=wp.vec3d(0.)
                    for i in range(4):
                        if groups[v[i]]==g:jb+=wp.float64(w[i])*mode(b,double(q[v[i]])-double(center[g]))
                    wp.atomic_add(C,g,a,b,wp.dot(ja,h*jb))

@wp.kernel
def factor(C:wp.array3d[wp.float64],L:wp.array3d[wp.float64],fault:wp.array[int]):
    g=wp.tid()
    for i in range(6):
        for j in range(i+1):
            value=wp.float64(.5)*(C[g,i,j]+C[g,j,i])
            for k in range(j):value-=L[g,i,k]*L[g,j,k]
            if i==j:
                if value<=wp.float64(0.) or not wp.isfinite(value):
                    fault[0]=1
                    return
                L[g,i,j]=wp.sqrt(value)
            else:L[g,i,j]=value/L[g,j,j]

@wp.kernel
def solve(L:wp.array3d[wp.float64],linear:wp.array[wp.vec3d],angular:wp.array[wp.vec3d],
    work:wp.array2d[wp.float64],solution:wp.array2d[wp.float64]):
    g=wp.tid()
    for i in range(6):
        value=wp.float64(0.)
        if i<3:value=linear[g][i]
        else:value=angular[g][i-3]
        for j in range(i):value-=L[g,i,j]*work[g,j]
        work[g,i]=value/L[g,i,i]
    for k in range(6):
        i=5-k;value=work[g,i]
        for j in range(i+1,6):value-=L[g,j,i]*solution[g,j]
        solution[g,i]=value/L[g,i,i]

@wp.kernel
def prolong(q:wp.array[wp.vec3],groups:wp.array[int],center:wp.array[wp.vec3],
    solution:wp.array2d[wp.float64],diagonal:wp.array[wp.mat33],x:wp.array[wp.vec3d],
    y:wp.array[wp.vec3d],alpha:wp.float64,beta:wp.float64,z:wp.array[wp.vec3d]):
    i=wp.tid();g=groups[i];value=matrix(diagonal[i])*x[i]
    if g>=0:
        r=double(q[i])-double(center[g])
        for j in range(6):value+=solution[g,j]*mode(j,r)
    z[i]=alpha*value+beta*y[i]

class ContactCoarse:
    def __init__(self,g):
        self.g=g;self.search=g.surface_search;d=g.s.model.device;n=g.coarse.center.size
        self.C=wp.zeros((n,6,6),dtype=wp.float64,device=d)
        self.L=wp.zeros_like(self.C)
        self.work=wp.zeros((n,6),dtype=wp.float64,device=d);self.solution=wp.zeros_like(self.work)
        self.fault=wp.zeros(1,dtype=int,device=d)
    def prepare(self):
        g=self.g;m=g.s.model;c=g.coarse;p=self.search;d=m.device
        self.C.zero_();self.L.zero_()
        wp.launch(mass_blocks,m.particle_count,[g.q,c.groups,c.center,m.particle_mass,p.base,wp.float64(g.dt),self.C],device=d)
        wp.launch(contact_blocks,p.capacity,[g.q,c.groups,c.center,p.count,p.ids,p.weights,p.H,self.C],device=d)
        wp.launch(factor,c.center.size,[self.C,self.L,self.fault],device=d)
    def apply(self,x,y,z,alpha,beta):
        from .search_reductions import restrict
        g=self.g;m=g.s.model;c=g.coarse;l=g.linear;d=m.device
        wp.launch_tiled(restrict,dim=c.center.size,inputs=[g.q,c.starts,c.sizes,c.center,x,l.f,l.t],block_dim=128,device=d)
        wp.launch(solve,c.center.size,[self.L,l.f,l.t,self.work,self.solution],device=d)
        wp.launch(prolong,m.particle_count,[g.q,c.groups,c.center,self.solution,g.inverse,x,y,
            wp.float64(alpha),wp.float64(beta),z],device=d)
    def install(self):
        from warp.optim.linear import LinearOperator
        g=self.g;old=g.linear.prepare_rhs
        def prepare():old();self.prepare()
        g.linear.prepare_rhs=prepare
        g.cg._M=LinearOperator(g.cg._A.shape,wp.mat33d,g.s.model.device,self.apply)
    def audit(self):
        import numpy as np
        if self.fault.numpy()[0]:raise FloatingPointError('Non-positive contact coarse Cholesky pivot')
        C=self.C.numpy();L=self.L.numpy();error=np.linalg.norm(C-L@L.transpose(0,2,1))/np.linalg.norm(C)
        return dict(relative_factorization_error=float(error),minimum_eigenvalue=float(np.linalg.eigvalsh(C).min()))
