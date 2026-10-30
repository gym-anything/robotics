"""Higher-precision arithmetic for the SAME global search matrix and PCG.

A known rigid translation had 2e-7 solution error but29% recomputed force
residual with float32 matvec. Metal-to-inertia conditioning makes that residual
gap material. This changes arithmetic, not elasticity, contact, mass or time.
Contact blocks/poses remain native inputs; matrix products and Krylov state
use float64. Physical state remains the Newton state and is not reprojected.
"""
import warp as wp
from .relative_step import double
from .global_iteration import hinge_gradient

@wp.func
def matrix(a:wp.mat33):
    return wp.mat33d(wp.float64(a[0,0]),wp.float64(a[0,1]),wp.float64(a[0,2]),
        wp.float64(a[1,0]),wp.float64(a[1,1]),wp.float64(a[1,2]),
        wp.float64(a[2,0]),wp.float64(a[2,1]),wp.float64(a[2,2]))

@wp.kernel
def widen(source:wp.array[wp.vec3],dest:wp.array[wp.vec3d]):
    i=wp.tid();dest[i]=double(source[i])

@wp.kernel
def narrow(source:wp.array[wp.vec3d],dest:wp.array[wp.vec3]):
    i=wp.tid();v=source[i];dest[i]=wp.vec3(float(v[0]),float(v[1]),float(v[2]))

@wp.kernel
def begin(mass:wp.array[float],contact:wp.array[wp.mat33],x:wp.array[wp.vec3d],dt:wp.float64,out:wp.array[wp.vec3d]):
    i=wp.tid();out[i]=(wp.float64(mass[i])/(dt*dt))*x[i]+wp.float64(4.)*matrix(contact[i])*x[i]

@wp.kernel
def membrane(q:wp.array[wp.vec3],x:wp.array[wp.vec3d],tri:wp.array2d[int],pose:wp.array[wp.mat22],
    area:wp.array[float],mat:wp.array2d[float],dt:wp.float64,out:wp.array[wp.vec3d]):
    t=wp.tid();a=tri[t,0];b=tri[t,1];c=tri[t,2];P=pose[t]
    p00=wp.float64(P[0,0]);p01=wp.float64(P[0,1]);p10=wp.float64(P[1,0]);p11=wp.float64(P[1,1])
    f0=(double(q[b])-double(q[a]))*p00+(double(q[c])-double(q[a]))*p10
    f1=(double(q[b])-double(q[a]))*p01+(double(q[c])-double(q[a]))*p11
    d0=(x[b]-x[a])*p00+(x[c]-x[a])*p10;d1=(x[b]-x[a])*p01+(x[c]-x[a])*p11
    c00=wp.float64(2.)*wp.dot(f0,d0);c01=wp.dot(f0,d1)+wp.dot(f1,d0);c11=wp.float64(2.)*wp.dot(f1,d1)
    shear=wp.float64(mat[t,0])+wp.float64(2.*mat[t,2])/dt;vol=wp.float64(.5*mat[t,1])*(c00+c11)
    p0=wp.float64(area[t])*(shear*(f0*c00+f1*c01)+f0*vol)
    p1=wp.float64(area[t])*(shear*(f0*c01+f1*c11)+f1*vol)
    fb=p00*p0+p01*p1;fc=p10*p0+p11*p1
    wp.atomic_add(out,a,-fb-fc);wp.atomic_add(out,b,fb);wp.atomic_add(out,c,fc)

@wp.kernel
def hinges(q:wp.array[wp.vec3],x:wp.array[wp.vec3d],edges:wp.array2d[int],length:wp.array[float],
    stiffness:wp.array2d[float],dt:wp.float64,out:wp.array[wp.vec3d]):
    e=wp.tid()
    if edges[e,0]<0 or edges[e,1]<0:return
    g0=double(hinge_gradient(e,0,q,edges));g1=double(hinge_gradient(e,1,q,edges))
    g2=double(hinge_gradient(e,2,q,edges));g3=-g0-g1-g2
    projection=wp.dot(g0,x[edges[e,0]])+wp.dot(g1,x[edges[e,1]])+wp.dot(g2,x[edges[e,2]])+wp.dot(g3,x[edges[e,3]])
    coefficient=wp.float64(length[e])*(wp.float64(stiffness[e,0])+wp.float64(stiffness[e,1])/dt)*projection
    wp.atomic_add(out,edges[e,0],coefficient*g0);wp.atomic_add(out,edges[e,1],coefficient*g1)
    wp.atomic_add(out,edges[e,2],coefficient*g2);wp.atomic_add(out,edges[e,3],coefficient*g3)

@wp.kernel
def finish(mass:wp.array[float],value:wp.array[wp.vec3d],x:wp.array[wp.vec3d],y:wp.array[wp.vec3d],
    alpha:wp.float64,beta:wp.float64,z:wp.array[wp.vec3d]):
    i=wp.tid();v=value[i]
    if mass[i]==0.:v=x[i]
    z[i]=alpha*v+beta*y[i]

@wp.kernel
def restrict(q:wp.array[wp.vec3],groups:wp.array[int],center:wp.array[wp.vec3],x:wp.array[wp.vec3d],
    linear:wp.array[wp.vec3d],angular:wp.array[wp.vec3d]):
    i=wp.tid();g=groups[i]
    if g<0:return
    wp.atomic_add(linear,g,x[i]);wp.atomic_add(angular,g,wp.cross(double(q[i])-double(center[g]),x[i]))

@wp.kernel
def prolong(q:wp.array[wp.vec3],groups:wp.array[int],center:wp.array[wp.vec3],
    il:wp.array[wp.mat33],ia:wp.array[wp.mat33],linear:wp.array[wp.vec3d],angular:wp.array[wp.vec3d],
    diagonal:wp.array[wp.mat33],x:wp.array[wp.vec3d],y:wp.array[wp.vec3d],alpha:wp.float64,beta:wp.float64,z:wp.array[wp.vec3d]):
    i=wp.tid();g=groups[i];v=matrix(diagonal[i])*x[i]
    if g>=0:v+=matrix(il[g])*linear[g]+wp.cross(matrix(ia[g])*angular[g],double(q[i])-double(center[g]))
    z[i]=alpha*v+beta*y[i]

@wp.kernel
def mean_vector(x:wp.array[wp.vec3d],groups:wp.array[int],counts:wp.array[float],out:wp.array[wp.vec3d]):
    i=wp.tid();g=groups[i]
    if g>=0:wp.atomic_add(out,g,x[i]/wp.float64(counts[g]))

@wp.kernel
def internal_product(D:wp.array[wp.mat33],x:wp.array[wp.vec3d],groups:wp.array[int],counts:wp.array[float],
    mean:wp.array[wp.vec3d],out:wp.array[wp.vec3d],average:wp.array[wp.vec3d]):
    i=wp.tid();g=groups[i];v=wp.vec3d(0.)
    if g>=0:
        v=wp.float64(4.)*matrix(D[i])*(x[i]-mean[g])
        wp.atomic_add(average,g,v/wp.float64(counts[g]))
    out[i]=v

@wp.kernel
def add_projected(groups:wp.array[int],value:wp.array[wp.vec3d],mean:wp.array[wp.vec3d],out:wp.array[wp.vec3d]):
    i=wp.tid();g=groups[i]
    if g>=0:out[i]+=value[i]-mean[g]

@wp.kernel
def prepare_hinges(q:wp.array[wp.vec3],edges:wp.array2d[int],out:wp.array[wp.vec3d]):
    e=wp.tid();g0=double(hinge_gradient(e,0,q,edges));g1=double(hinge_gradient(e,1,q,edges));g2=double(hinge_gradient(e,2,q,edges))
    # Exact translation invariance; independently rounded fourth derivatives
    # otherwise become an artificial stiffness on a known null direction.
    out[4*e]=g0;out[4*e+1]=g1;out[4*e+2]=g2;out[4*e+3]=-g0-g1-g2

@wp.kernel
def cached_hinges(grad:wp.array[wp.vec3d],x:wp.array[wp.vec3d],edges:wp.array2d[int],length:wp.array[float],
    stiffness:wp.array2d[float],dt:wp.float64,out:wp.array[wp.vec3d]):
    e=wp.tid()
    if edges[e,0]<0 or edges[e,1]<0:return
    g0=grad[4*e];g1=grad[4*e+1];g2=grad[4*e+2];g3=grad[4*e+3]
    projection=wp.dot(g0,x[edges[e,0]])+wp.dot(g1,x[edges[e,1]])+wp.dot(g2,x[edges[e,2]])+wp.dot(g3,x[edges[e,3]])
    coefficient=wp.float64(length[e])*(wp.float64(stiffness[e,0])+wp.float64(stiffness[e,1])/dt)*projection
    wp.atomic_add(out,edges[e,0],coefficient*g0);wp.atomic_add(out,edges[e,1],coefficient*g1)
    wp.atomic_add(out,edges[e,2],coefficient*g2);wp.atomic_add(out,edges[e,3],coefficient*g3)

class Linear64:
    def __init__(self,global_iteration):
        self.g=global_iteration;m=self.g.s.model;d=m.device
        self.rhs=wp.zeros(m.particle_count,dtype=wp.vec3d,device=d);self.delta=wp.zeros_like(self.rhs)
        self.mv=wp.zeros_like(self.rhs)
        self.f=wp.zeros(self.g.coarse.center.size,dtype=wp.vec3d,device=d);self.t=wp.zeros_like(self.f)
        self.hinge_grad=wp.zeros(4*m.edge_count,dtype=wp.vec3d,device=d)
        self.cache_geometry=True
        self.elastic_cache=None
        self.tiled_reductions=False
        self.translation_mean=wp.zeros_like(self.f);self.internal_mean=wp.zeros_like(self.f)
        self.internal_value=wp.zeros_like(self.rhs)
    def prepare_geometry(self):
        g=self.g;m=g.s.model
        wp.launch(prepare_hinges,m.edge_count,[g.q,m.edge_indices,self.hinge_grad],device=m.device)
        if self.elastic_cache is not None:self.elastic_cache.prepare()
    def matvec(self,x,y,z,alpha,beta):
        g=self.g;m=g.s.model;d=m.device;dt=wp.float64(g.dt)
        wp.launch(begin,m.particle_count,[m.particle_mass,g.contact_matrix,x,dt,self.mv],device=d)
        # P^T(4D)P retains the conservative diagonal contact bound while
        # preserving the EXACT translation nullspace of within-shell contacts.
        c=g.coarse
        if self.tiled_reductions:
            from .search_reductions import sum_vectors,internal_values
            wp.launch_tiled(sum_vectors,dim=c.center.size,inputs=[c.starts,c.sizes,x,self.translation_mean],block_dim=128,device=d)
            wp.launch(internal_values,m.particle_count,[g.internal_matrix,x,c.groups,self.translation_mean,self.internal_value],device=d)
            wp.launch_tiled(sum_vectors,dim=c.center.size,inputs=[c.starts,c.sizes,self.internal_value,self.internal_mean],block_dim=128,device=d)
        else:
            self.translation_mean.zero_();self.internal_mean.zero_()
            wp.launch(mean_vector,m.particle_count,[x,c.groups,c.counts,self.translation_mean],device=d)
            wp.launch(internal_product,m.particle_count,[g.internal_matrix,x,c.groups,c.counts,self.translation_mean,self.internal_value,self.internal_mean],device=d)
        wp.launch(add_projected,m.particle_count,[c.groups,self.internal_value,self.internal_mean,self.mv],device=d)
        if self.elastic_cache is not None:
            self.elastic_cache.add_product(x,self.mv)
        else:
            wp.launch(membrane,m.tri_count,[g.q,x,m.tri_indices,m.tri_poses,m.tri_areas,m.tri_materials,dt,self.mv],device=d)
            kernel=cached_hinges if self.cache_geometry else hinges
            data=self.hinge_grad if self.cache_geometry else g.q
            wp.launch(kernel,m.edge_count,[data,x,m.edge_indices,m.edge_rest_length,m.edge_bending_properties,dt,self.mv],device=d)
        wp.launch(finish,m.particle_count,[m.particle_mass,self.mv,x,y,wp.float64(alpha),wp.float64(beta),z],device=d)
    def precondition(self,x,y,z,alpha,beta):
        g=self.g;m=g.s.model;d=m.device;c=g.coarse
        if self.tiled_reductions:
            from .search_reductions import restrict as tiled_restrict
            wp.launch_tiled(tiled_restrict,dim=c.center.size,inputs=[g.q,c.starts,c.sizes,c.center,x,self.f,self.t],
                block_dim=128,device=d)
        else:
            self.f.zero_();self.t.zero_()
            wp.launch(restrict,m.particle_count,[g.q,c.groups,c.center,x,self.f,self.t],device=d)
        wp.launch(prolong,m.particle_count,[g.q,c.groups,c.center,c.il,c.ia,self.f,self.t,g.inverse,x,y,
            wp.float64(alpha),wp.float64(beta),z],device=d)
    def prepare_rhs(self):
        self.prepare_geometry()
        g=self.g;wp.launch(widen,g.s.model.particle_count,[g.force,self.rhs],device=g.device);self.delta.zero_()
    def retrieve(self):
        g=self.g;wp.launch(narrow,g.s.model.particle_count,[self.delta,g.delta],device=g.device)
