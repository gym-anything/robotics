"""Opt-in precise elastic coordinates for the existing global shell solve.

Retains sub-float32-ULP displacements instead of converting a rigid translation
into metal strain. Public Newton/contact positions remain rounded coordinates;
their discrepancy is recorded and bounded, not a second rendered trajectory.
The membrane, wrapped hinge and objective damping laws/parameters are unchanged.
No force clipping, stiffness reduction or rigidification is performed.
"""
import numpy as np
import warp as wp
from .relative_step import double

@wp.func
def single(x:wp.vec3d):return wp.vec3(float(x[0]),float(x[1]),float(x[2]))

@wp.func
def matrix_single(a:wp.mat33d):
    return wp.mat33(float(a[0,0]),float(a[0,1]),float(a[0,2]),float(a[1,0]),float(a[1,1]),
        float(a[1,2]),float(a[2,0]),float(a[2,1]),float(a[2,2]))

@wp.func
def membrane(t:int,o:int,q:wp.array[wp.vec3d],prev:wp.array[wp.vec3d],tri:wp.array2d[int],
    P:wp.mat22d,A:wp.float64,mu:wp.float64,lam:wp.float64,damp:wp.float64,dt:wp.float64):
    a=tri[t,0];b=tri[t,1];c=tri[t,2]
    f0=(q[b]-q[a])*P[0,0]+(q[c]-q[a])*P[1,0]
    f1=(q[b]-q[a])*P[0,1]+(q[c]-q[a])*P[1,1]
    c00=wp.dot(f0,f0);c01=wp.dot(f0,f1);c11=wp.dot(f1,f1)
    J=wp.sqrt(wp.max(c00*c11-c01*c01,wp.float64(1e-20)))
    g0=(c11*f0-c01*f1)/J;g1=(c00*f1-c01*f0)/J
    s=(lam+mu)*(J-wp.float64(1.))-mu
    d0=P[0,0]*wp.float64(int(o==1)-int(o==0))+P[1,0]*wp.float64(int(o==2)-int(o==0))
    d1=P[0,1]*wp.float64(int(o==1)-int(o==0))+P[1,1]*wp.float64(int(o==2)-int(o==0))
    f=-((mu*f0+s*g0)*d0+(mu*f1+s*g1)*d1)
    r=wp.max(wp.float64(0.),s)/J;dj=d0*g0+d1*g1;w=d0*f1-d1*f0
    H=(mu*(d0*d0+d1*d1)+r*wp.dot(w,w))*wp.identity(n=3,dtype=wp.float64)
    H+=(lam+mu-r)*wp.outer(dj,dj)-r*wp.outer(w,w)
    if s<wp.float64(0.)and J>wp.float64(1e-10):
        normal=wp.cross(f0,f1)/J
        H-=wp.min(mu*(d0*d0+d1*d1),-s/J*wp.dot(w,w))*wp.outer(normal,normal)
    if damp>wp.float64(0.):
        p0=(prev[b]-prev[a])*P[0,0]+(prev[c]-prev[a])*P[1,0]
        p1=(prev[b]-prev[a])*P[0,1]+(prev[c]-prev[a])*P[1,1]
        dc00=wp.float64(2.)*d0*f0;dc01=d0*f1+d1*f0;dc11=wp.float64(2.)*d1*f1
        f-=damp/dt*((c00-wp.dot(p0,p0))*dc00+wp.float64(2.)*(c01-wp.dot(p0,p1))*dc01+(c11-wp.dot(p1,p1))*dc11)
        H+=damp/dt*(wp.outer(dc00,dc00)+wp.float64(2.)*wp.outer(dc01,dc01)+wp.outer(dc11,dc11))
    return single(A*f),matrix_single(A*H)

@wp.func
def hinge(e:int,o:int,q:wp.array[wp.vec3d],prev:wp.array[wp.vec3d],edges:wp.array2d[int],
    rest:wp.float64,length:wp.float64,k:wp.float64,damp:wp.float64,dt:wp.float64):
    a=edges[e,0];b=edges[e,1];c=edges[e,2];d=edges[e,3]
    if a<0 or b<0:return wp.vec3(0.),wp.mat33(0.)
    u=q[c]-q[a];v=q[d]-q[a];s=q[d]-q[b];t=q[c]-q[b];edge=q[d]-q[c]
    n1=wp.cross(u,v);n2=wp.cross(s,t);l1=wp.length(n1);l2=wp.length(n2);le=wp.length(edge)
    if l1<wp.float64(1e-6)or l2<wp.float64(1e-6)or le<wp.float64(1e-6):return wp.vec3(0.),wp.mat33(0.)
    n1/=l1;n2/=l2;axis=edge/le
    sine=wp.dot(wp.cross(n1,n2),axis);cosine=wp.dot(n1,n2);theta=wp.atan2(sine,cosine)
    B1=wp.skew(edge)*wp.float64(int(o==0))-wp.skew(v)*wp.float64(int(o==2))+wp.skew(u)*wp.float64(int(o==3))
    B2=-wp.skew(edge)*wp.float64(int(o==1))+wp.skew(s)*wp.float64(int(o==2))-wp.skew(t)*wp.float64(int(o==3))
    I=wp.identity(n=3,dtype=wp.float64)
    D1=(I-wp.outer(n1,n1))*B1/l1;D2=(I-wp.outer(n2,n2))*B2/l2
    ds=wp.transpose(-wp.skew(n2)*D1+wp.skew(n1)*D2)*axis
    dc=wp.transpose(D1)*n2+wp.transpose(D2)*n1
    grad=(cosine*ds-sine*dc)/(sine*sine+cosine*cosine)
    angle=wp.atan2(wp.sin(theta-rest),wp.cos(theta-rest))
    f=-k*length*angle*grad;H=k*length*wp.outer(grad,grad)
    pn1=wp.cross(prev[c]-prev[a],prev[d]-prev[a]);pn2=wp.cross(prev[d]-prev[b],prev[c]-prev[b]);pe=prev[d]-prev[c]
    if damp>wp.float64(0.)and wp.length(pn1)>=wp.float64(1e-6)and wp.length(pn2)>=wp.float64(1e-6)and wp.length(pe)>=wp.float64(1e-6):
        pn1=wp.normalize(pn1);pn2=wp.normalize(pn2);pe=wp.normalize(pe)
        old=wp.atan2(wp.dot(wp.cross(pn1,pn2),pe),wp.dot(pn1,pn2));delta=theta-old
        f-=damp*length/dt*wp.atan2(wp.sin(delta),wp.cos(delta))*grad
        H+=damp*length/dt*wp.outer(grad,grad)
    return single(f),matrix_single(H)

@wp.func
def hinge_fast(e:int,o:int,q:wp.array[wp.vec3d],prev:wp.array[wp.vec3d],edges:wp.array2d[int],
    rest:wp.float64,length:wp.float64,k:wp.float64,damp:wp.float64,dt:wp.float64):
    # Algebraically identical shared-edge angle gradient. Avoids constructing
    # and multiplying several3x3 derivative matrices for every vertex.
    a=edges[e,0];b=edges[e,1];c=edges[e,2];d=edges[e,3]
    if a<0 or b<0:return wp.vec3(0.),wp.mat33(0.)
    n1=wp.cross(q[c]-q[a],q[d]-q[a]);n2=wp.cross(q[d]-q[b],q[c]-q[b]);edge=q[d]-q[c]
    l1=wp.length(n1);l2=wp.length(n2);le=wp.length(edge)
    if l1<wp.float64(1e-6)or l2<wp.float64(1e-6)or le<wp.float64(1e-6):return wp.vec3(0.),wp.mat33(0.)
    n1/=l1;n2/=l2;axis=edge/le
    ga=-le/l1*n1;gb=-le/l2*n2
    gc=(wp.dot(q[a]-q[d],axis)*ga+wp.dot(q[b]-q[d],axis)*gb)/le
    gd=-(ga+gb+gc)
    grad=ga*wp.float64(int(o==0))+gb*wp.float64(int(o==1))+gc*wp.float64(int(o==2))+gd*wp.float64(int(o==3))
    theta=wp.atan2(wp.dot(wp.cross(n1,n2),axis),wp.dot(n1,n2))
    angle=wp.atan2(wp.sin(theta-rest),wp.cos(theta-rest))
    f=-k*length*angle*grad;H=k*length*wp.outer(grad,grad)
    pn1=wp.cross(prev[c]-prev[a],prev[d]-prev[a]);pn2=wp.cross(prev[d]-prev[b],prev[c]-prev[b]);pe=prev[d]-prev[c]
    if damp>wp.float64(0.)and wp.length(pn1)>=wp.float64(1e-6)and wp.length(pn2)>=wp.float64(1e-6)and wp.length(pe)>=wp.float64(1e-6):
        pn1=wp.normalize(pn1);pn2=wp.normalize(pn2);pe=wp.normalize(pe)
        old=wp.atan2(wp.dot(wp.cross(pn1,pn2),pe),wp.dot(pn1,pn2));delta=theta-old
        f-=damp*length/dt*wp.atan2(wp.sin(delta),wp.cos(delta))*grad
        H+=damp*length/dt*wp.outer(grad,grad)
    return single(f),matrix_single(H)

@wp.func
def membrane_force(t:int,o:int,q:wp.array[wp.vec3d],prev:wp.array[wp.vec3d],tri:wp.array2d[int],
    P:wp.mat22d,A:wp.float64,mu:wp.float64,lam:wp.float64,damp:wp.float64,dt:wp.float64):
    # Same force expression as membrane(), with NO curvature temporaries.
    a=tri[t,0];b=tri[t,1];c=tri[t,2]
    f0=(q[b]-q[a])*P[0,0]+(q[c]-q[a])*P[1,0]
    f1=(q[b]-q[a])*P[0,1]+(q[c]-q[a])*P[1,1]
    c00=wp.dot(f0,f0);c01=wp.dot(f0,f1);c11=wp.dot(f1,f1)
    J=wp.sqrt(wp.max(c00*c11-c01*c01,wp.float64(1e-20)))
    g0=(c11*f0-c01*f1)/J;g1=(c00*f1-c01*f0)/J
    s=(lam+mu)*(J-wp.float64(1.))-mu
    d0=P[0,0]*wp.float64(int(o==1)-int(o==0))+P[1,0]*wp.float64(int(o==2)-int(o==0))
    d1=P[0,1]*wp.float64(int(o==1)-int(o==0))+P[1,1]*wp.float64(int(o==2)-int(o==0))
    f=-((mu*f0+s*g0)*d0+(mu*f1+s*g1)*d1)
    if damp>wp.float64(0.):
        p0=(prev[b]-prev[a])*P[0,0]+(prev[c]-prev[a])*P[1,0]
        p1=(prev[b]-prev[a])*P[0,1]+(prev[c]-prev[a])*P[1,1]
        dc00=wp.float64(2.)*d0*f0;dc01=d0*f1+d1*f0;dc11=wp.float64(2.)*d1*f1
        f-=damp/dt*((c00-wp.dot(p0,p0))*dc00+wp.float64(2.)*(c01-wp.dot(p0,p1))*dc01+(c11-wp.dot(p1,p1))*dc11)
    return single(A*f)

@wp.func
def hinge_force(e:int,o:int,q:wp.array[wp.vec3d],prev:wp.array[wp.vec3d],edges:wp.array2d[int],
    rest:wp.float64,length:wp.float64,k:wp.float64,damp:wp.float64,dt:wp.float64):
    a=edges[e,0];b=edges[e,1];c=edges[e,2];d=edges[e,3]
    if a<0 or b<0:return wp.vec3(0.)
    n1=wp.cross(q[c]-q[a],q[d]-q[a]);n2=wp.cross(q[d]-q[b],q[c]-q[b]);edge=q[d]-q[c]
    l1=wp.length(n1);l2=wp.length(n2);le=wp.length(edge)
    if l1<wp.float64(1e-6)or l2<wp.float64(1e-6)or le<wp.float64(1e-6):return wp.vec3(0.)
    n1/=l1;n2/=l2;axis=edge/le
    ga=-le/l1*n1;gb=-le/l2*n2
    gc=(wp.dot(q[a]-q[d],axis)*ga+wp.dot(q[b]-q[d],axis)*gb)/le
    gd=-(ga+gb+gc)
    grad=ga*wp.float64(int(o==0))+gb*wp.float64(int(o==1))+gc*wp.float64(int(o==2))+gd*wp.float64(int(o==3))
    theta=wp.atan2(wp.dot(wp.cross(n1,n2),axis),wp.dot(n1,n2))
    angle=wp.atan2(wp.sin(theta-rest),wp.cos(theta-rest))
    f=-k*length*angle*grad
    pn1=wp.cross(prev[c]-prev[a],prev[d]-prev[a]);pn2=wp.cross(prev[d]-prev[b],prev[c]-prev[b]);pe=prev[d]-prev[c]
    if damp>wp.float64(0.)and wp.length(pn1)>=wp.float64(1e-6)and wp.length(pn2)>=wp.float64(1e-6)and wp.length(pe)>=wp.float64(1e-6):
        pn1=wp.normalize(pn1);pn2=wp.normalize(pn2);pe=wp.normalize(pe)
        old=wp.atan2(wp.dot(wp.cross(pn1,pn2),pe),wp.dot(pn1,pn2));delta=theta-old
        f-=damp*length/dt*wp.atan2(wp.sin(delta),wp.cos(delta))*grad
    return single(f)

@wp.kernel
def elasticity(q:wp.array[wp.vec3d],prev:wp.array[wp.vec3d],tri:wp.array2d[int],pose:wp.array[wp.mat22d],
    area:wp.array[float],mat:wp.array2d[float],edges:wp.array2d[int],rest:wp.array[float],length:wp.array[float],
    stiffness:wp.array2d[float],dt:wp.float64,force:wp.array[wp.vec3],diag:wp.array[wp.mat33],hessians:bool):
    i=wp.tid()
    if i<tri.shape[0]*3:
        t=i//3;o=i%3;v=tri[t,o]
        f,h=membrane(t,o,q,prev,tri,pose[t],wp.float64(area[t]),wp.float64(mat[t,0]),wp.float64(mat[t,1]),wp.float64(mat[t,2]),dt)
        wp.atomic_add(force,v,f)
        if hessians:wp.atomic_add(diag,v,h)
    if i<edges.shape[0]*4:
        e=i//4;o=i%4;v=edges[e,o]
        if v>=0:
            f,h=hinge_fast(e,o,q,prev,edges,wp.float64(rest[e]),wp.float64(length[e]),wp.float64(stiffness[e,0]),wp.float64(stiffness[e,1]),dt)
            wp.atomic_add(force,v,f)
            if hessians:wp.atomic_add(diag,v,h)

@wp.kernel
def prepare(q:wp.array[wp.vec3],mass:wp.array[float],current:wp.array[wp.vec3d],emitted:wp.array[wp.vec3],prev:wp.array[wp.vec3d]):
    i=wp.tid()
    # Only assembly anchors are prescribed. A mismatch on a material vertex
    # comes from an explicit checkpoint/state operation, not an attachment.
    x=current[i]+double(q[i])-double(emitted[i])
    if mass[i]==0.:x=double(q[i])
    prev[i]=x;current[i]=x;emitted[i]=q[i]

@wp.kernel
def initialize(prev:wp.array[wp.vec3d],vel:wp.array[wp.vec3],external:wp.array[wp.vec3],
    gravity:wp.array[wp.vec3],world:wp.array[int],inverse_mass:wp.array[float],flags:wp.array[int],dt:wp.float64,
    current:wp.array[wp.vec3d],inertia:wp.array[wp.vec3d]):
    i=wp.tid();current[i]=prev[i];inertia[i]=prev[i]
    if (flags[i]&1)!=0 and inverse_mass[i]>0.:
        inertia[i]+=dt*double(vel[i])+dt*dt*(double(gravity[world[i]])+wp.float64(inverse_mass[i])*double(external[i]))

@wp.kernel
def residual(q:wp.array[wp.vec3d],target:wp.array[wp.vec3d],mass:wp.array[float],dt:wp.float64,
    force:wp.array[wp.vec3],diag:wp.array[wp.mat33],out:wp.array[wp.vec3],hessians:bool):
    i=wp.tid();k=wp.float64(mass[i])/(dt*dt)
    out[i]=force[i]+single(k*(target[i]-q[i]))
    if hessians:diag[i]+=float(k)*wp.identity(n=3,dtype=float)
    if mass[i]==0.:
        out[i]=wp.vec3(0.)
        if hessians:diag[i]=wp.identity(n=3,dtype=float)

@wp.kernel
def candidate(q:wp.array[wp.vec3d],delta:wp.array[wp.vec3],bound:wp.array[float],alpha:float,
    out:wp.array[wp.vec3d],public:wp.array[wp.vec3],collision_ref:wp.array[wp.vec3],disp:wp.array[wp.vec3]):
    i=wp.tid();x=q[i]+wp.float64(bound[0])*wp.float64(alpha)*double(delta[i]);out[i]=x;public[i]=single(x)
    disp[i]=public[i]-collision_ref[i]

@wp.kernel
def interpolate(q:wp.array[wp.vec3d],end:wp.array[wp.vec3d],t:wp.float64,out:wp.array[wp.vec3d],public:wp.array[wp.vec3]):
    i=wp.tid();out[i]=q[i]+t*(end[i]-q[i]);public[i]=single(out[i])

@wp.kernel
def work(q:wp.array[wp.vec3d],end:wp.array[wp.vec3d],f:wp.array[wp.vec3],w3:wp.float64,w5:wp.float64,out:wp.array[wp.float64]):
    i=wp.tid();v=wp.dot(double(f[i]),end[i]-q[i]);wp.atomic_add(out,0,w3*v);wp.atomic_add(out,1,w5*v)

@wp.kernel
def accept(trial:wp.array[wp.vec3d],accepted:wp.array[int],best:wp.array[wp.vec3d]):
    i=wp.tid()
    if accepted[0]!=0:best[i]=trial[i]

@wp.kernel
def velocity(current:wp.array[wp.vec3d],previous:wp.array[wp.vec3d],dt:wp.float64,out:wp.array[wp.vec3]):
    i=wp.tid();out[i]=single((current[i]-previous[i])/dt)

@wp.kernel
def elasticity_force(q:wp.array[wp.vec3d],prev:wp.array[wp.vec3d],tri:wp.array2d[int],pose:wp.array[wp.mat22d],
    area:wp.array[float],mat:wp.array2d[float],edges:wp.array2d[int],rest:wp.array[float],length:wp.array[float],
    stiffness:wp.array2d[float],dt:wp.float64,force:wp.array[wp.vec3]):
    # Dedicated kernel permits dead-code elimination of the unused returned
    # curvature. The dynamic hessians flag in elasticity() cannot do that.
    i=wp.tid()
    if i<tri.shape[0]*3:
        t=i//3;o=i%3;v=tri[t,o]
        f=membrane_force(t,o,q,prev,tri,pose[t],wp.float64(area[t]),wp.float64(mat[t,0]),wp.float64(mat[t,1]),wp.float64(mat[t,2]),dt)
        wp.atomic_add(force,v,f)
    if i<edges.shape[0]*4:
        e=i//4;o=i%4;v=edges[e,o]
        if v>=0:
            f=hinge_force(e,o,q,prev,edges,wp.float64(rest[e]),wp.float64(length[e]),wp.float64(stiffness[e,0]),wp.float64(stiffness[e,1]),dt)
            wp.atomic_add(force,v,f)


class ElasticPrecision:
    def __init__(self,g):
        self.g=g;m=g.s.model;d=m.device
        q=m.particle_q.numpy().astype(float);tri=m.tri_indices.numpy();p=q[tri]
        a=p[:,1]-p[:,0];b=p[:,2]-p[:,0];L=np.linalg.norm(a,axis=1)
        along=np.sum(a*b,axis=1)/L;H=np.linalg.norm(np.cross(a,b),axis=1)/L
        P=np.zeros((len(tri),2,2));P[:,0,0]=1/L;P[:,0,1]=-along/(L*H);P[:,1,1]=1/H
        self.pose=wp.array(P,dtype=wp.mat22d,device=d)
        self.current=wp.array(q,dtype=wp.vec3d,device=d);self.previous=wp.clone(self.current)
        self.emitted=wp.clone(m.particle_q);self.inertia=wp.zeros_like(self.current)
        self.reference=wp.zeros_like(self.current);self.trial=wp.zeros_like(self.current)
        self.sample=wp.zeros_like(self.current);self.best=wp.zeros_like(self.current)
    def prepare(self,state):
        m=self.g.s.model;wp.launch(prepare,m.particle_count,[state.particle_q,m.particle_mass,self.current,self.emitted,self.previous],device=m.device)
    def initialize(self,state,dt):
        s=self.g.s;m=s.model
        wp.launch(initialize,m.particle_count,[self.previous,state.particle_qd,state.particle_f,m.gravity,m.particle_world,
            m.particle_inv_mass,m.particle_flags,wp.float64(dt),self.current,self.inertia],device=m.device)
    def evaluate(self,q,dt,force,diag,out,*,hessians):
        s=self.g.s;m=s.model
        inputs=[q,self.previous,m.tri_indices,self.pose,m.tri_areas,
            m.tri_materials,m.edge_indices,m.edge_rest_angle,m.edge_rest_length,m.edge_bending_properties,
            wp.float64(dt),force]
        if hessians:wp.launch(elasticity,max(m.tri_count*3,m.edge_count*4),[*inputs,diag,True],device=m.device)
        else:wp.launch(elasticity_force,max(m.tri_count*3,m.edge_count*4),inputs,device=m.device)
        wp.launch(residual,m.particle_count,[q,self.inertia,m.particle_mass,wp.float64(dt),force,diag,out,hessians],device=m.device)
