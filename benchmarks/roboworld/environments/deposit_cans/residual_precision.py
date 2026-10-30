"""Double-precision elastic force accumulation for force-residual derivatives.

The constitutive expressions are identical to elastic_precision.py. Keeping
only positions in double precision is insufficient: differentiating rounded
float32 force sums adds false stiffness to rigid translations of a thin shell.
Contact-law evaluation is unchanged; its precision is qualified separately.
"""
import warp as wp
from .relative_step import double
from .elastic_precision import single
from newton._src.solvers.vbd.particle_vbd_kernels import evaluate_spring_force_and_hessian_both_vertices


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
    if damp>wp.float64(0.):
        p0=(prev[b]-prev[a])*P[0,0]+(prev[c]-prev[a])*P[1,0]
        p1=(prev[b]-prev[a])*P[0,1]+(prev[c]-prev[a])*P[1,1]
        dc00=wp.float64(2.)*d0*f0;dc01=d0*f1+d1*f0;dc11=wp.float64(2.)*d1*f1
        f-=damp/dt*((c00-wp.dot(p0,p0))*dc00+wp.float64(2.)*(c01-wp.dot(p0,p1))*dc01+(c11-wp.dot(p1,p1))*dc11)
    return A*f


@wp.func
def hinge(e:int,o:int,q:wp.array[wp.vec3d],prev:wp.array[wp.vec3d],edges:wp.array2d[int],
    rest:wp.float64,length:wp.float64,k:wp.float64,damp:wp.float64,dt:wp.float64):
    a=edges[e,0];b=edges[e,1];c=edges[e,2];d=edges[e,3]
    if a<0 or b<0:return wp.vec3d(0.)
    raw1=wp.cross(q[c]-q[a],q[d]-q[a]);raw2=wp.cross(q[d]-q[b],q[c]-q[b]);edge=q[d]-q[c]
    l1=wp.length(raw1);l2=wp.length(raw2);le=wp.length(edge)
    if l1<wp.float64(1e-6)or l2<wp.float64(1e-6)or le<wp.float64(1e-6):return wp.vec3d(0.)
    n1=raw1/l1;n2=raw2/l2;axis=edge/le
    ga=-le/l1*n1;gb=-le/l2*n2
    gc=(wp.dot(q[a]-q[d],axis)*ga+wp.dot(q[b]-q[d],axis)*gb)/le
    gd=-(ga+gb+gc)
    grad=ga*wp.float64(int(o==0))+gb*wp.float64(int(o==1))+gc*wp.float64(int(o==2))+gd*wp.float64(int(o==3))
    theta=wp.atan2(wp.dot(wp.cross(n1,n2),axis),wp.dot(n1,n2))
    angle=wp.atan2(wp.sin(theta-rest),wp.cos(theta-rest))
    elastic_force=-k*length*angle*grad
    pr1=wp.cross(prev[c]-prev[a],prev[d]-prev[a]);pr2=wp.cross(prev[d]-prev[b],prev[c]-prev[b]);previous_edge=prev[d]-prev[c]
    if damp>wp.float64(0.)and wp.length(pr1)>=wp.float64(1e-6)and wp.length(pr2)>=wp.float64(1e-6)and wp.length(previous_edge)>=wp.float64(1e-6):
        pn1=wp.normalize(pr1);pn2=wp.normalize(pr2);pe=wp.normalize(previous_edge)
        old=wp.atan2(wp.dot(wp.cross(pn1,pn2),pe),wp.dot(pn1,pn2));delta=theta-old
        damping_force=-damp*length/dt*wp.atan2(wp.sin(delta),wp.cos(delta))*grad
        return elastic_force+damping_force
    return elastic_force


@wp.kernel
def begin(contact:wp.array[wp.vec3],out:wp.array[wp.vec3d]):
    i=wp.tid();out[i]=double(contact[i])


@wp.kernel
def elastic(q:wp.array[wp.vec3d],prev:wp.array[wp.vec3d],tri:wp.array2d[int],pose:wp.array[wp.mat22d],
    area:wp.array[float],mat:wp.array2d[float],edges:wp.array2d[int],rest:wp.array[float],length:wp.array[float],
    stiffness:wp.array2d[float],dt:wp.float64,force:wp.array[wp.vec3d],mode:int):
    i=wp.tid()
    if mode!=2 and i<tri.shape[0]*3:
        t=i//3;o=i%3;v=tri[t,o]
        f=membrane(t,o,q,prev,tri,pose[t],wp.float64(area[t]),wp.float64(mat[t,0]),wp.float64(mat[t,1]),wp.float64(mat[t,2]),dt)
        wp.atomic_add(force,v,f)
    if mode!=1 and i<edges.shape[0]*4:
        e=i//4;o=i%4;v=edges[e,o]
        if v>=0:
            f=hinge(e,o,q,prev,edges,wp.float64(rest[e]),wp.float64(length[e]),wp.float64(stiffness[e,0]),wp.float64(stiffness[e,1]),dt)
            wp.atomic_add(force,v,f)


@wp.kernel
def elastic_batched(q:wp.array[wp.vec3d],prev:wp.array[wp.vec3d],tri:wp.array2d[int],pose:wp.array[wp.mat22d],
    area:wp.array[float],mat:wp.array2d[float],edges:wp.array2d[int],rest:wp.array[float],length:wp.array[float],
    stiffness:wp.array2d[float],dt:wp.float64,force:wp.array[wp.vec3d],mode:int):
    # Same per-vertex functions, but all local outputs precede assembly. The
    # compiler can reuse common triangle/hinge geometry and trigonometry.
    i=wp.tid()
    if mode!=2 and i<tri.shape[0]:
        f0=membrane(i,0,q,prev,tri,pose[i],wp.float64(area[i]),wp.float64(mat[i,0]),wp.float64(mat[i,1]),wp.float64(mat[i,2]),dt)
        f1=membrane(i,1,q,prev,tri,pose[i],wp.float64(area[i]),wp.float64(mat[i,0]),wp.float64(mat[i,1]),wp.float64(mat[i,2]),dt)
        f2=membrane(i,2,q,prev,tri,pose[i],wp.float64(area[i]),wp.float64(mat[i,0]),wp.float64(mat[i,1]),wp.float64(mat[i,2]),dt)
        wp.atomic_add(force,tri[i,0],f0)
        wp.atomic_add(force,tri[i,1],f1)
        wp.atomic_add(force,tri[i,2],f2)
    if mode!=1 and i<edges.shape[0]:
        h0=hinge(i,0,q,prev,edges,wp.float64(rest[i]),wp.float64(length[i]),wp.float64(stiffness[i,0]),wp.float64(stiffness[i,1]),dt)
        h1=hinge(i,1,q,prev,edges,wp.float64(rest[i]),wp.float64(length[i]),wp.float64(stiffness[i,0]),wp.float64(stiffness[i,1]),dt)
        h2=hinge(i,2,q,prev,edges,wp.float64(rest[i]),wp.float64(length[i]),wp.float64(stiffness[i,0]),wp.float64(stiffness[i,1]),dt)
        h3=hinge(i,3,q,prev,edges,wp.float64(rest[i]),wp.float64(length[i]),wp.float64(stiffness[i,0]),wp.float64(stiffness[i,1]),dt)
        if edges[i,0]>=0:wp.atomic_add(force,edges[i,0],h0)
        if edges[i,1]>=0:wp.atomic_add(force,edges[i,1],h1)
        if edges[i,2]>=0:wp.atomic_add(force,edges[i,2],h2)
        if edges[i,3]>=0:wp.atomic_add(force,edges[i,3],h3)


@wp.kernel
def springs(dt:float,q:wp.array[wp.vec3],previous:wp.array[wp.vec3],
    exact:wp.array[wp.vec3d],old_exact:wp.array[wp.vec3d],indices:wp.array[int],
    rest:wp.array[float],stiffness:wp.array[float],damping:wp.array[float],force:wp.array[wp.vec3d]):
    i=wp.tid();a,b,fa,fb,H=evaluate_spring_force_and_hessian_both_vertices(i,dt,q,previous,indices,rest,stiffness,damping)
    dx=exact[a]-exact[b];length=wp.max(wp.length(dx),wp.float64(1e-8))
    old_length=wp.max(wp.length(old_exact[a]-old_exact[b]),wp.float64(1e-8))
    f=wp.float64(stiffness[i])*(wp.float64(rest[i])/length-wp.float64(1.))*dx
    f-=dx/length*wp.float64(damping[i]/dt)*(length-old_length)
    wp.atomic_add(force,a,f-double(fa));wp.atomic_add(force,b,-f-double(fb))


@wp.kernel
def inertia(q:wp.array[wp.vec3d],target:wp.array[wp.vec3d],mass:wp.array[float],dt:wp.float64,
            force:wp.array[wp.vec3d],out:wp.array[wp.vec3d],public:wp.array[wp.vec3],include_inertia:bool):
    i=wp.tid();out[i]=force[i]
    if include_inertia:out[i]+=wp.float64(mass[i])/(dt*dt)*(target[i]-q[i])
    if mass[i]==0.:out[i]=wp.vec3d(0.)
    public[i]=single(out[i])


class ResidualPrecision:
    def __init__(self,g,*,full_contact=False):
        self.g=g;self.force=wp.zeros_like(g.precision.current,requires_grad=True);self.full_contact=full_contact
        self.result=wp.zeros_like(g.precision.current,requires_grad=True)
        self.component='all'
    def evaluate(self,q,state,c,dt):
        g=self.g;s=g.s;m=s.model;p=g.precision
        if self.component in ('elastic','inertia','membrane','bending'):self.force.zero_()
        elif self.full_contact:
            from .contact_residual_precision import assemble
            assemble(g,q,state,c,dt,self.force,component=self.component)
        else:
            g.contact.evaluate(g.trial,state,c,dt,aggregate_groups=False,hessians=False)
            wp.launch(begin,m.particle_count,[g.contact.force,self.force],device=m.device)
            if m.spring_count:
                wp.launch(springs,m.spring_count,[dt,g.trial,s.particle_q_prev,q,p.previous,m.spring_indices,
                    m.spring_rest_length,m.spring_stiffness,m.spring_damping,self.force],device=m.device)
        if self.component in ('all','elastic','membrane','bending'):
            modes=(1,) if self.component=='membrane' else (2,) if self.component=='bending' else (1,2)
            for mode in modes:
                batched=getattr(g,'batched_elastic_force',False)
                count=(m.tri_count if mode==1 else m.edge_count)if batched else(m.tri_count*3 if mode==1 else m.edge_count*4)
                wp.launch(elastic_batched if batched else elastic,count,[q,p.previous,m.tri_indices,p.pose,m.tri_areas,
                    m.tri_materials,m.edge_indices,m.edge_rest_angle,m.edge_rest_length,m.edge_bending_properties,
                    wp.float64(dt),self.force,mode],device=m.device)
        wp.launch(inertia,m.particle_count,[q,p.inertia,m.particle_mass,wp.float64(dt),self.force,self.result,g.force,
            self.component in ('all','inertia')],device=m.device)
        return self.result
