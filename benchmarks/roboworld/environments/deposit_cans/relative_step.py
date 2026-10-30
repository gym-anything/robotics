# SPDX-License-Identifier: Apache-2.0
"""Conservative simultaneous step using relative separating-plane motion.

For each queried pair, the initial closest-point normal separates its convex
primitives. Requiring every cross-pair projected separation to stay positive
is sufficient for the entire linear step. One common scale preserves that
condition for both primitives; independent per-vertex scales would not.
A discovery-radius bound covers pairs absent from the current query.
This is a conservative line-search filter, not a replacement contact force.
It must not be described as IPC's robust floating-point CCD implementation.
"""
import warp as wp
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionInfo
from newton._src.solvers.vbd.particle_vbd_kernels import triangle_closest_point

@wp.func
def double(v:wp.vec3):
    return wp.vec3d(wp.float64(v[0]),wp.float64(v[1]),wp.float64(v[2]))

@wp.func
def edge_separator(a:wp.vec3d,b:wp.vec3d,c:wp.vec3d,d:wp.vec3d,epsilon:wp.float64):
    # Same finite-segment closest-point cases, in local double coordinates.
    # Forming two ~1m world closest points in float32 before subtracting can
    # corrupt a20micrometre separator enough to produce a negative vertex gap.
    u=b-a;v=d-c;r=a-c;aa=wp.dot(u,u);ee=wp.dot(v,v);ff=wp.dot(v,r)
    s=wp.float64(0.);t=wp.float64(0.)
    if aa<=epsilon and ee<=epsilon:return r
    if aa<=epsilon:t=wp.clamp(ff/ee,wp.float64(0.),wp.float64(1.))
    else:
        cc=wp.dot(u,r)
        if ee<=epsilon:s=wp.clamp(-cc/aa,wp.float64(0.),wp.float64(1.))
        else:
            bb=wp.dot(u,v);cross=wp.cross(u,v);denom=wp.dot(cross,cross)
            if denom>wp.float64(1.4210854715202004e-14)*aa*ee:
                numerator=wp.dot(wp.cross(v,r),cross)
                s=wp.clamp(numerator/denom,wp.float64(0.),wp.float64(1.))
            t=(bb*s+ff)/ee
            if t<wp.float64(0.):
                t=wp.float64(0.);s=wp.clamp(-cc/aa,wp.float64(0.),wp.float64(1.))
            elif t>wp.float64(1.):
                t=wp.float64(1.);s=wp.clamp((bb-cc)/aa,wp.float64(0.),wp.float64(1.))
    return r+s*u-t*v

@wp.func
def triangle_separator(a:wp.vec3d,b:wp.vec3d,c:wp.vec3d,p:wp.vec3d):
    ab=b-a;ac=c-a;ap=p-a;d1=wp.dot(ab,ap);d2=wp.dot(ac,ap)
    if d1<=wp.float64(0.) and d2<=wp.float64(0.):return ap
    bp=p-b;d3=wp.dot(ab,bp);d4=wp.dot(ac,bp)
    if d3>=wp.float64(0.) and d4<=d3:return bp
    vc=d1*d4-d3*d2
    if vc<=wp.float64(0.) and d1>=wp.float64(0.) and d3<=wp.float64(0.):return ap-(d1/(d1-d3))*ab
    cp=p-c;d5=wp.dot(ab,cp);d6=wp.dot(ac,cp)
    if d6>=wp.float64(0.) and d5<=d6:return cp
    vb=d5*d2-d1*d6
    if vb<=wp.float64(0.) and d2>=wp.float64(0.) and d6<=wp.float64(0.):return ap-(d2/(d2-d6))*ac
    va=d3*d6-d5*d4
    if va<=wp.float64(0.) and d4-d3>=wp.float64(0.) and d5-d6>=wp.float64(0.):return bp-((d4-d3)/(d4-d3+d5-d6))*(c-b)
    denom=wp.float64(1.)/(va+vb+vc)
    return ap-(vb*denom)*ab-(vc*denom)*ac

@wp.func
def limit_pair(gap:float,relative:float,gamma:float):
    # Positive separation along one fixed normal excludes primitive crossing.
    # Allow floating point roundoff only in the separating projection, not an
    # initially intersecting pair. Degenerate initial distances are flagged.
    t=float(1.)
    if relative< -1e-10:
        t=wp.clamp(gamma*wp.max(gap,0.)/(-relative),0.,1.)
    return t

@wp.kernel
def limit_contacts(q:wp.array[wp.vec3],delta:wp.array[wp.vec3],tri:wp.array2d[int],
    edges:wp.array2d[int],info:wp.array[TriMeshCollisionInfo],gamma:float,
    epsilon:float,scale:wp.array[float],degenerate:wp.array[int],diagnostic:wp.array2d[float]):
    i=wp.tid();c=info[0];bound=float(1.);kind=float(0.);pair=float(-1.);worst_gap=float(0.);worst_relative=float(0.)
    if i<c.vertex_colliding_triangles_count.shape[0]:
        count=wp.min(c.vertex_colliding_triangles_count[i],c.vertex_colliding_triangles_buffer_sizes[i])
        offset=c.vertex_colliding_triangles_offsets[i]
        for j in range(count):
            t=c.vertex_colliding_triangles[2*(offset+j)+1]
            if t>=0:
                a=tri[t,0];b=tri[t,1];d=tri[t,2]
                n=triangle_separator(double(q[a]),double(q[b]),double(q[d]),double(q[i]));distance=wp.length(n)
                if distance<=1e-9:
                    wp.atomic_add(degenerate,0,1);bound=0.;kind=3.;pair=float(t);worst_gap=float(distance)
                else:
                    n/=distance
                    for k in range(3):
                        v=tri[t,k]
                        gap=float(wp.dot(n,double(q[i])-double(q[v])));relative=float(wp.dot(n,double(delta[i])-double(delta[v])))
                        value=limit_pair(gap,relative,gamma)
                        if value<bound:
                            bound=value;kind=1.;pair=float(t);worst_gap=gap;worst_relative=relative
    if i<c.edge_colliding_edges_count.shape[0]:
        count=wp.min(c.edge_colliding_edges_count[i],c.edge_colliding_edges_buffer_sizes[i])
        offset=c.edge_colliding_edges_offsets[i]
        for j in range(count):
            e=c.edge_colliding_edges[2*(offset+j)+1]
            if e>=0:
                a=edges[i,2];b=edges[i,3];d=edges[e,2];f=edges[e,3]
                n=edge_separator(double(q[a]),double(q[b]),double(q[d]),double(q[f]),wp.float64(epsilon));distance=wp.length(n)
                if distance<=1e-9:
                    wp.atomic_add(degenerate,0,1);bound=0.;kind=4.;pair=float(e);worst_gap=float(distance)
                else:
                    n/=distance
                    for k in range(2):
                        for l in range(2):
                            v=edges[i,2+k];w=edges[e,2+l]
                            gap=float(wp.dot(n,double(q[v])-double(q[w])));relative=float(wp.dot(n,double(delta[v])-double(delta[w])))
                            value=limit_pair(gap,relative,gamma)
                            if value<bound:
                                bound=value;kind=2.;pair=float(e);worst_gap=gap;worst_relative=relative
    diagnostic[i,0]=bound;diagnostic[i,1]=kind;diagnostic[i,2]=pair
    diagnostic[i,3]=worst_gap;diagnostic[i,4]=worst_relative
    wp.atomic_min(scale,0,bound)

@wp.kernel
def limit_discovery(delta:wp.array[wp.vec3],radius:float,scale:wp.array[float]):
    i=wp.tid();length=wp.length(delta[i])
    if length>radius:wp.atomic_min(scale,0,radius/length)

@wp.kernel
def apply(q:wp.array[wp.vec3],delta:wp.array[wp.vec3],scale:wp.array[float],
    displacement:wp.array[wp.vec3],out:wp.array[wp.vec3]):
    i=wp.tid();d=scale[0]*delta[i];displacement[i]=d;out[i]=q[i]+d

@wp.kernel
def scale_direction(delta:wp.array[wp.vec3],alpha:float,out:wp.array[wp.vec3]):
    i=wp.tid();out[i]=alpha*delta[i]

@wp.kernel
def remaining_discovery(reference:wp.array[wp.vec3],current:wp.array[wp.vec3],
    delta:wp.array[wp.vec3],radius:float,scale:wp.array[float]):
    i=wp.tid();y=double(current[i])-double(reference[i]);d=double(delta[i])
    a=wp.dot(d,d);b=wp.dot(y,d);c=wp.dot(y,y)-wp.float64(radius)*wp.float64(radius)
    bound=wp.float64(1.)
    if c>wp.float64(1e-12):bound=wp.float64(0.)
    elif a>wp.float64(1e-30):
        bound=wp.clamp((-b+wp.sqrt(wp.max(b*b-a*c,wp.float64(0.))))/a,wp.float64(0.),wp.float64(1.))
    wp.atomic_min(scale,0,float(bound))

@wp.kernel
def apply_direction(reference:wp.array[wp.vec3],current:wp.array[wp.vec3],
    delta:wp.array[wp.vec3],scale:wp.array[float],displacement:wp.array[wp.vec3],out:wp.array[wp.vec3]):
    i=wp.tid();v=current[i]+scale[0]*delta[i];out[i]=v;displacement[i]=v-reference[i]

@wp.kernel
def refresh_needed(reference:wp.array[wp.vec3],current:wp.array[wp.vec3],threshold:float,condition:wp.array[int]):
    i=wp.tid()
    if wp.length_sq(current[i]-reference[i])>=threshold*threshold:
        wp.atomic_max(condition,0,1)

class RelativeStep:
    def __init__(self,solver):
        self.s=solver;self.scale=wp.ones(1,dtype=float,device=solver.device)
        self.degenerate=wp.zeros(1,dtype=int,device=solver.device)
        self.direction=wp.zeros_like(solver.model.particle_q)
        self.epsilon=1e-12  # squared edge length:1 micrometre, NOT a parallel-angle tolerance
        self.diagnostic=wp.zeros((max(solver.model.particle_count,solver.model.edge_count),5),dtype=float,device=solver.device)
        self.refresh_condition=wp.zeros(1,dtype=int,device=solver.device)
    def refresh_if_needed(self,state):
        # Re-query after half the discovery allowance has been consumed.
        # The original conservative bound remains active on EVERY correction;
        # this only updates the search reference, never grants unchecked motion.
        s=self.s;self.refresh_condition.zero_()
        threshold=s.particle_self_contact_margin*s.particle_conservative_bound_relaxation*.25
        wp.launch(refresh_needed,s.model.particle_count,[s.pos_prev_collision_detection,state.particle_q,threshold,self.refresh_condition],device=s.device)
        wp.capture_if(self.refresh_condition,s._collision_detection_penetration_free,current_state=state)
    def apply(self,delta,displacement,out):
        self._apply_total(delta,displacement,out)
    def increment(self,current,delta,alpha,displacement,out):
        # Bound the NEW correction from the accepted iterate, not a scaled
        # displacement from the last collision query. Scaling the latter also
        # undoes old accepted motion and is not a line search along delta.
        s=self.s;m=s.model;self.scale.fill_(1.);self.degenerate.zero_()
        wp.launch(scale_direction,m.particle_count,[delta,alpha,self.direction],device=s.device)
        wp.launch(limit_contacts,max(m.particle_count,m.edge_count),
            [current,self.direction,m.tri_indices,m.edge_indices,s.trimesh_collision_info,
             s.particle_conservative_bound_relaxation,self.epsilon,self.scale,self.degenerate,self.diagnostic],device=s.device)
        wp.launch(remaining_discovery,m.particle_count,
            [s.pos_prev_collision_detection,current,self.direction,s.particle_self_contact_margin*s.particle_conservative_bound_relaxation*.5,self.scale],device=s.device)
        wp.launch(apply_direction,m.particle_count,
            [s.pos_prev_collision_detection,current,self.direction,self.scale,displacement,out],device=s.device)
    def _apply_total(self,delta,displacement,out):
        s=self.s;m=s.model;self.scale.fill_(1.);self.degenerate.zero_()
        wp.launch(limit_contacts,max(m.particle_count,m.edge_count),
            [s.pos_prev_collision_detection,delta,m.tri_indices,m.edge_indices,s.trimesh_collision_info,
             s.particle_conservative_bound_relaxation,self.epsilon,self.scale,self.degenerate,self.diagnostic],device=s.device)
        wp.launch(limit_discovery,m.particle_count,
            [delta,s.particle_self_contact_margin*s.particle_conservative_bound_relaxation*.5,self.scale],device=s.device)
        wp.launch(apply,m.particle_count,[s.pos_prev_collision_detection,delta,self.scale,displacement,out],device=s.device)
