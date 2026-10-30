"""Unchanged contact laws and candidate multiplicities with float64 forces.

Self-contact retains Newton's C2 log/quadratic penalty, its closing-only damping
threshold, and C1 Coulomb regularization. Rigid contact retains its separate
linear normal penalty. No contacts are removed, softened or attached.
"""
import warp as wp
from newton._src.solvers.vbd.particle_vbd_kernels import NUM_THREADS_PER_COLLISION_PRIMITIVE
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionInfo
from .self_geometry import triangle,edge_pair
from .relative_step import double
from .retention_precision import transform64


@wp.func
def friction(n:wp.vec3d,dx:wp.vec3d,mu:wp.float64,load:wp.float64,eps:wp.float64):
    tangent=dx-n*wp.dot(n,dx);length=wp.length(tangent)
    # Keep the analytic zero-slip derivative; returning a literal zero would
    # erase the nonzero C1 limiting tangent during automatic differentiation.
    if length==wp.float64(0.):return -wp.float64(2.)*mu*load/eps*tangent
    factor=wp.float64(1.)/length
    if length<=eps:factor=(-length/eps+wp.float64(2.))/eps
    return -mu*load*factor*tangent


@wp.func
def self_response(n:wp.vec3d,distance:wp.float64,dx:wp.vec3d,radius:wp.float64,
                  ke:wp.float64,kd:wp.float64,mu:wp.float64,eps:wp.float64,dt:wp.float64):
    tau=radius*wp.float64(.5);dmin=wp.float64(1e-5)
    normal=ke*(radius-distance)
    if tau>distance>dmin:normal=tau*tau*ke/distance
    elif distance<=dmin:normal=tau*tau*ke*(wp.float64(2.)*dmin-distance)/(dmin*dmin)
    value=normal*n+friction(n,dx,mu,normal,eps*dt)
    rate=wp.dot(n,dx)/dt
    if rate<wp.float64(-1e-6):value-=kd*rate*n
    return value


@wp.kernel
def self_contacts(q:wp.array[wp.vec3d],prev:wp.array[wp.vec3d],tri:wp.array2d[int],edges:wp.array2d[int],
    info:wp.array[TriMeshCollisionInfo],radius:wp.float64,ke:wp.float64,kd:wp.float64,mu:wp.float64,
    eps:wp.float64,dt:wp.float64,parallel:wp.float64,force:wp.array[wp.vec3d],mode:int):
    thread=wp.tid();i=thread//NUM_THREADS_PER_COLLISION_PRIMITIVE
    lane=thread%NUM_THREADS_PER_COLLISION_PRIMITIVE;c=info[0]
    if mode==1 and i<c.edge_colliding_edges_buffer_sizes.shape[0]:
        j=lane;offset=c.edge_colliding_edges_offsets[i]
        count=wp.min(c.edge_colliding_edges_count[i],c.edge_colliding_edges_buffer_sizes[i])
        while j<count:
            other=c.edge_colliding_edges[2*(offset+j)+1]
            if other>=0:
                a=edges[i,2];b=edges[i,3];u=edges[other,2];v=edges[other,3]
                st=edge_pair(i,other,q,edges,parallel);s=st[0];t=st[1];distance=st[2]
                if wp.float64(0.)<distance<radius:
                    n=((q[a]+s*(q[b]-q[a]))-(q[u]+t*(q[v]-q[u])))/distance
                    dx=(wp.float64(1.)-s)*(q[a]-prev[a])+s*(q[b]-prev[b])
                    dx-=(wp.float64(1.)-t)*(q[u]-prev[u])+t*(q[v]-prev[v])
                    f=self_response(n,distance,dx,radius,ke,kd,mu,eps,dt)
                    # Directed edge candidate: only the first edge's endpoints,
                    # exactly like the native reciprocal candidate stream.
                    wp.atomic_add(force,a,(wp.float64(1.)-s)*f);wp.atomic_add(force,b,s*f)
            j+=NUM_THREADS_PER_COLLISION_PRIMITIVE
    if mode==2 and i<c.vertex_colliding_triangles_buffer_sizes.shape[0]:
        j=lane;offset=c.vertex_colliding_triangles_offsets[i]
        count=wp.min(c.vertex_colliding_triangles_count[i],c.vertex_colliding_triangles_buffer_sizes[i])
        while j<count:
            triangle_id=c.vertex_colliding_triangles[2*(offset+j)+1]
            if triangle_id>=0:
                a=tri[triangle_id,0];b=tri[triangle_id,1];d=tri[triangle_id,2]
                point,w=triangle(q[a],q[b],q[d],q[i]);gap=q[i]-point;distance=wp.length(gap)
                if wp.float64(0.)<distance<radius:
                    dx=q[i]-prev[i]-(w[0]*(q[a]-prev[a])+w[1]*(q[b]-prev[b])+w[2]*(q[d]-prev[d]))
                    f=self_response(gap/distance,distance,dx,radius,ke,kd,mu,eps,dt)
                    wp.atomic_add(force,i,f);wp.atomic_add(force,a,-w[0]*f)
                    wp.atomic_add(force,b,-w[1]*f);wp.atomic_add(force,d,-w[2]*f)
            j+=NUM_THREADS_PER_COLLISION_PRIMITIVE


@wp.func
def rigid_point(p:wp.vec3d,previous:wp.vec3d,r:wp.float64,i:int,ke:float,kd:float,mu:float,
    shape_body:wp.array[int],bq:wp.array[wp.transform],bprev:wp.array[wp.transform],bqd:wp.array[wp.spatial_vector],
    com:wp.array[wp.vec3],shape:wp.array[int],local:wp.array[wp.vec3],velocity:wp.array[wp.vec3],
    normal:wp.array[wp.vec3],margin:wp.array[float],eps:wp.float64,dt:wp.float64):
    sh=shape[i];body=shape_body[sh];pose=wp.transform_identity();old=wp.transform_identity();cm=wp.vec3(0.)
    if body>=0:pose=bq[body];cm=com[body]
    exact=transform64(pose);bx=wp.transform_point(exact,double(local[i]));n=double(normal[i]);pad=wp.float64(0.)
    if margin.shape[0]>0:pad=wp.float64(margin[sh])
    depth=-(wp.dot(n,p-bx)-r-pad)
    if depth<=wp.float64(0.):return wp.vec3d(0.),bx
    displacement=wp.vec3d(0.)
    if bprev:
        if body>=0:old=bprev[body]
        displacement=bx-wp.transform_point(transform64(old),double(local[i]))
    else:
        speed=wp.spatial_vector()
        if body>=0:speed=bqd[body]
        displacement=dt*(double(wp.spatial_top(speed))+wp.cross(double(wp.spatial_bottom(speed)),bx-wp.transform_point(exact,double(cm))))
    displacement+=dt*wp.transform_vector(exact,double(velocity[i]))
    dx=p-previous-displacement;load=depth*wp.float64(ke);value=n*load
    rate=wp.dot(n,dx)/dt
    if rate<wp.float64(0.):value-=wp.float64(kd)*rate*n
    value+=friction(n,dx,wp.float64(mu),load,eps*dt)
    return value,bx


@wp.kernel
def rigid_contacts(q:wp.array[wp.vec3d],prev:wp.array[wp.vec3d],radius:wp.array[float],
    indices:wp.array[wp.vec3i],count:wp.array[int],capacity:int,ke:wp.array[float],kd:wp.array[float],mu:wp.array[float],
    shape_body:wp.array[int],bq:wp.array[wp.transform],bprev:wp.array[wp.transform],bqd:wp.array[wp.spatial_vector],
    com:wp.array[wp.vec3],shape:wp.array[int],local:wp.array[wp.vec3],velocity:wp.array[wp.vec3],
    normal:wp.array[wp.vec3],margin:wp.array[float],barycentric:wp.array[wp.vec3],eps:wp.float64,dt:wp.float64,
    force:wp.array[wp.vec3d]):
    i=wp.tid()
    if i>=wp.min(count[0],capacity):return
    ids=indices[i];a=ids[0]
    if a<0:return
    p=q[a];previous=prev[a];r=wp.float64(radius[a]);weights=wp.vec3d(1.,0.,0.)
    if ids[1]>=0:
        weights=double(barycentric[i]);p=weights[0]*q[a];previous=weights[0]*prev[a]
        for j in range(1,3):
            v=ids[j]
            if v>=0:p+=weights[j]*q[v];previous+=weights[j]*prev[v];r=wp.max(r,wp.float64(radius[v]))
    value,bx=rigid_point(p,previous,r,i,ke[i],kd[i],mu[i],shape_body,bq,bprev,bqd,com,
        shape,local,velocity,normal,margin,eps,dt)
    for j in range(3):
        if ids[j]>=0:wp.atomic_add(force,ids[j],weights[j]*value)


@wp.kernel
def spring_contacts(q:wp.array[wp.vec3d],prev:wp.array[wp.vec3d],indices:wp.array[int],
    rest:wp.array[float],stiffness:wp.array[float],damping:wp.array[float],dt:wp.float64,force:wp.array[wp.vec3d]):
    i=wp.tid();a=indices[2*i];b=indices[2*i+1];dx=q[a]-q[b]
    length=wp.max(wp.length(dx),wp.float64(1e-8));old=wp.max(wp.length(prev[a]-prev[b]),wp.float64(1e-8))
    f=wp.float64(stiffness[i])*(wp.float64(rest[i])/length-wp.float64(1.))*dx
    f-=dx/length*wp.float64(damping[i])/dt*(length-old)
    wp.atomic_add(force,a,f);wp.atomic_add(force,b,-f)


def assemble(g,q,state,c,dt,out,component='all'):
    s=g.s;m=s.model;p=g.precision;out.zero_()
    if component in ('all','contact','rigid_contact'):wp.launch(rigid_contacts,c.soft_contact_max,[q,p.previous,m.particle_radius,c.soft_contact_indices,c.soft_contact_count,
        c.soft_contact_max,s.body_particle_contact_penalty_k,s.body_particle_contact_material_kd,s.body_particle_contact_material_mu,
        m.shape_body,state.body_q,s.body_q_prev,state.body_qd,m.body_com,c.soft_contact_shape,c.soft_contact_body_pos,
        c.soft_contact_body_vel,c.soft_contact_normal,m.shape_margin,c.soft_contact_barycentric,
        wp.float64(s.friction_epsilon),wp.float64(dt),out],device=m.device)
    if m.spring_count and component in ('all','contact','spring_contact'):
        wp.launch(spring_contacts,m.spring_count,[q,p.previous,m.spring_indices,m.spring_rest_length,m.spring_stiffness,
            m.spring_damping,wp.float64(dt),out],device=m.device)
    if s.particle_enable_self_contact and component in ('all','contact','edge_contact','vertex_contact'):
        modes=(1,) if component=='edge_contact' else (2,) if component=='vertex_contact' else (1,2)
        for mode in modes:
            wp.launch(self_contacts,s.particle_self_contact_evaluation_kernel_launch_size,[q,p.previous,m.tri_indices,m.edge_indices,
                s.trimesh_collision_info,wp.float64(s.particle_self_contact_radius),wp.float64(m.soft_contact_ke),
                wp.float64(m.soft_contact_kd),wp.float64(m.soft_contact_mu),wp.float64(s.friction_epsilon),wp.float64(dt),
                wp.float64(s.trimesh_collision_detector.edge_edge_parallel_epsilon),out,mode],device=m.device,max_blocks=m.device.sm_count)
