"""Projected tangent of the EXISTING area-based neo-Hookean membrane.

The prior metric-strain search loses all normal prestress curvature. Derive
the6x6 F-space Hessian from mu/2 |F|^2 + (lam+mu)/2 (J-1-mu/(lam+mu))^2,
J=|f0 cross f1|; clamp ONLY negative SEARCH eigenvalues. Physical force laws,
material constants, DOFs, damping, collisions and line acceptance are unchanged.
The exact force derivative is checked independently by finite differences.
"""
import warp as wp
import numpy as np
from .linear64 import cached_hinges
mat66=wp.types.matrix(shape=(6,6),dtype=wp.float64)
vec6=wp.types.vector(length=6,dtype=wp.float64)

@wp.func
def projected(f0:wp.vec3d,f1:wp.vec3d,d0:wp.vec3d,d1:wp.vec3d,mu:wp.float64,lam:wp.float64):
    c00=wp.dot(f0,f0);c11=wp.dot(f1,f1);c01=wp.dot(f0,f1)
    angle=wp.float64(.5)*wp.atan2(wp.float64(2.)*c01,c00-c11)
    cv=wp.cos(angle);sv=wp.sin(angle)
    u0=cv*f0+sv*f1;u1=-sv*f0+cv*f1
    a=wp.length(u0);b=wp.length(u1)
    u0/=wp.max(a,wp.float64(1e-15));u1/=wp.max(b,wp.float64(1e-15));normal=wp.normalize(wp.cross(u0,u1))
    v0=cv*d0+sv*d1;v1=-sv*d0+cv*d1
    d00=wp.dot(u0,v0);d10=wp.dot(u1,v0);d20=wp.dot(normal,v0)
    d01=wp.dot(u0,v1);d11=wp.dot(u1,v1);d21=wp.dot(normal,v1)
    bulk=lam+mu;s=bulk*(a*b-wp.float64(1.))-mu
    h00=mu+bulk*b*b;h11=mu+bulk*a*a;h01=bulk*a*b+s
    theta=wp.float64(.5)*wp.atan2(wp.float64(2.)*h01,h00-h11)
    cs=wp.cos(theta);sn=wp.sin(theta)
    mid=wp.float64(.5)*(h00+h11)
    spread=wp.sqrt(wp.float64(.25)*(h00-h11)*(h00-h11)+h01*h01)
    plus=wp.max(mid+spread,wp.float64(0.));minus=wp.max(mid-spread,wp.float64(0.))
    z0=plus*(cs*d00+sn*d11);z1=minus*(-sn*d00+cs*d11)
    t00=cs*z0-sn*z1;t11=sn*z0+cs*z1
    symmetric=wp.float64(.5)*wp.max(mu-s,wp.float64(0.))*(d10+d01)
    skew=wp.float64(.5)*wp.max(mu+s,wp.float64(0.))*(d10-d01)
    t10=symmetric+skew;t01=symmetric-skew
    t20=wp.max(mu+s*b/wp.max(a,wp.float64(1e-15)),wp.float64(0.))*d20
    t21=wp.max(mu+s*a/wp.max(b,wp.float64(1e-15)),wp.float64(0.))*d21
    p0=t00*u0+t10*u1+t20*normal;p1=t01*u0+t11*u1+t21*normal
    return cv*p0-sv*p1,sv*p0+cv*p1

@wp.func
def metric_damping(f0:wp.vec3d,f1:wp.vec3d,d0:wp.vec3d,d1:wp.vec3d,k:wp.float64):
    a=wp.float64(2.)*wp.dot(f0,d0);b=wp.dot(f0,d1)+wp.dot(f1,d0);c=wp.float64(2.)*wp.dot(f1,d1)
    return k*(f0*a+f1*b),k*(f0*b+f1*c)

@wp.kernel
def prepare(q:wp.array[wp.vec3d],tri:wp.array2d[int],pose:wp.array[wp.mat22d],
            area:wp.array[float],mat:wp.array2d[float],dt:wp.float64,H:wp.array[mat66]):
    t=wp.tid();a=tri[t,0];b=tri[t,1];c=tri[t,2];P=pose[t]
    f0=(q[b]-q[a])*P[0,0]+(q[c]-q[a])*P[1,0]
    f1=(q[b]-q[a])*P[0,1]+(q[c]-q[a])*P[1,1]
    value=mat66(0.)
    for j in range(6):
        d0=wp.vec3d(0.);d1=wp.vec3d(0.)
        if j<3:d0[j]=wp.float64(1.)
        else:d1[j-3]=wp.float64(1.)
        p0,p1=projected(f0,f1,d0,d1,wp.float64(mat[t,0]),wp.float64(mat[t,1]))
        z0,z1=metric_damping(f0,f1,d0,d1,wp.float64(2.*mat[t,2])/dt)
        for k in range(3):
            value[k,j]=wp.float64(area[t])*(p0[k]+z0[k]);value[k+3,j]=wp.float64(area[t])*(p1[k]+z1[k])
    H[t]=value

@wp.kernel
def product(H:wp.array[mat66],x:wp.array[wp.vec3d],tri:wp.array2d[int],pose:wp.array[wp.mat22d],out:wp.array[wp.vec3d]):
    t=wp.tid();a=tri[t,0];b=tri[t,1];c=tri[t,2];P=pose[t]
    d0=(x[b]-x[a])*P[0,0]+(x[c]-x[a])*P[1,0];d1=(x[b]-x[a])*P[0,1]+(x[c]-x[a])*P[1,1]
    p=H[t]*vec6(d0[0],d0[1],d0[2],d1[0],d1[1],d1[2])
    p0=wp.vec3d(p[0],p[1],p[2]);p1=wp.vec3d(p[3],p[4],p[5])
    fb=P[0,0]*p0+P[0,1]*p1;fc=P[1,0]*p0+P[1,1]*p1
    wp.atomic_add(out,a,-fb-fc);wp.atomic_add(out,b,fb);wp.atomic_add(out,c,fc)

class MembraneSearch:
    def __init__(self,g):
        self.g=g
        if not g.precision:raise ValueError('This candidate requires precise existing elastic coordinates')
        self.H=wp.zeros(g.s.model.tri_count,dtype=mat66,device=g.device)
    def prepare(self):
        g=self.g;m=g.s.model
        wp.launch(prepare,m.tri_count,[g.precision.current,m.tri_indices,g.precision.pose,m.tri_areas,m.tri_materials,wp.float64(g.dt),self.H],device=g.device)
    def add_product(self,x,out):
        g=self.g;m=g.s.model
        wp.launch(product,m.tri_count,[self.H,x,m.tri_indices,g.precision.pose,out],device=g.device)
        wp.launch(cached_hinges,m.edge_count,[g.linear.hinge_grad,x,m.edge_indices,m.edge_rest_length,m.edge_bending_properties,wp.float64(g.dt),out],device=g.device)

@wp.kernel
def sample(F:wp.array2d[wp.vec3d],output:wp.array[mat66]):
    i=wp.tid();h=mat66(0.)
    for j in range(6):
        d0=wp.vec3d(0.);d1=wp.vec3d(0.)
        if j<3:d0[j]=wp.float64(1.)
        else:d1[j-3]=wp.float64(1.)
        a,b=projected(F[i,0],F[i,1],d0,d1,wp.float64(2000.),wp.float64(8000.))
        for k in range(3):h[k,j]=a[k];h[k+3,j]=b[k]
    output[i]=h

def audit(output):
    import json
    from pathlib import Path
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(206);cases=[]
    for stretches in [(1.,1.),(1.04,1.03),(1.3,.9),(.7,.8),(2.,.3),(1.,1.00001)]:
        Q,_=np.linalg.qr(rng.normal(size=(3,3)));R,_=np.linalg.qr(rng.normal(size=(2,2)))
        cases.append(Q[:,:2]@np.diag(stretches)@R.T)
    F=np.asarray(cases);x=wp.array(F.transpose(0,2,1),dtype=wp.vec3d,device='cuda:0')
    H=wp.zeros(len(F),dtype=mat66,device='cuda:0');wp.launch(sample,len(F),[x,H],device='cuda:0');actual=H.numpy()
    def gradient(v):
        f=v.reshape(2,3).T;C=f.T@f;J=np.sqrt(np.linalg.det(C))
        return (2000*f+(10000*(J-1)-2000)*J*f@np.linalg.inv(C)).T.ravel()
    rows=[]
    for i,f in enumerate(F):
        v=f.T.ravel();fd=np.zeros((6,6));step=1e-6
        for j in range(6):
            d=np.zeros(6);d[j]=step;fd[:,j]=(gradient(v+d)-gradient(v-d))/(2*step)
        eig,U=np.linalg.eigh((fd+fd.T)/2);expected=(U*np.maximum(eig,0))@U.T
        error=float(np.linalg.norm(actual[i]-expected)/np.linalg.norm(expected))
        rows.append(dict(case=i,relative_projected_derivative_error=error,original_eigenvalues=eig.tolist(),
                         search_eigenvalues=np.linalg.eigvalsh(actual[i]).tolist()))
    result=dict(cases=rows,passed=all(v['relative_projected_derivative_error']<1e-7 for v in rows),
                scope='Search tangent derivative audit, not a physical qualification')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if not result['passed']:raise AssertionError('Membrane tangent mismatch')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();audit(a.output)
