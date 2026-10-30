# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Isolated same-law self-contact using precise solver displacement increments.

`previous` arguments below carry increments, not world positions. Only private
kernel copies receive them. Broad phase, normals, contact law, coefficients,
query multiplicity, truncation and all original state arrays are preserved.
"""
import inspect,textwrap
from types import FunctionType,MethodType,SimpleNamespace
import warp as wp
from newton._src.geometry.kernels import triangle_closest_point
from newton._src.math import orthonormal_basis
from newton._src.solvers.vbd.particle_vbd_kernels import mat32,evaluate_self_contact_force_norm,damp_collision
from deposit_cans.contact_curvature import friction
from deposit_cans.elastic_precision import single

@wp.kernel
def increments(q:wp.array[wp.vec3d],previous:wp.array[wp.vec3d],out:wp.array[wp.vec3]):
    i=wp.tid();out[i]=single(q[i]-previous[i])

@wp.func
def response(n:wp.vec3,dist:float,dx:wp.vec3,radius:float,ke:float,kd:float,mu:float,eps:float,dt:float):
    dEdD,d2E=evaluate_self_contact_force_norm(dist,radius,ke)
    e0,e1=orthonormal_basis(n);T=mat32(e0[0],e1[0],e0[1],e1[1],e0[2],e1[2])
    f,h=friction(mu,-dEdD,T,wp.transpose(T)*dx,eps*dt)
    return -dEdD*n+f,d2E*wp.outer(n,n)+h,wp.dot(n,dx)/dt

@wp.func
def weighted(f:wp.vec3,h:wp.mat33,rate:float,w:float,n:wp.vec3,kd:float,dt:float):
    df,dh=damp_collision(rate,w,n,kd,dt)
    return w*f+df,w*w*h+dh

@wp.func
def vt(v:int,t:int,q:wp.array[wp.vec3],inc:wp.array[wp.vec3],tri:wp.array2d[int],
    radius:float,ke:float,kd:float,mu:float,eps:float,dt:float):
    a=tri[t,0];b=tri[t,1];c=tri[t,2]
    closest,w,feature=triangle_closest_point(q[a],q[b],q[c],q[v]);gap=q[v]-closest;dist=wp.length(gap)
    if 0.<dist<radius:
        n=gap/dist;dx=inc[v]-(w[0]*inc[a]+w[1]*inc[b]+w[2]*inc[c])
        f,h,rate=response(n,dist,dx,radius,ke,kd,mu,eps,dt)
        fa,ha=weighted(f,h,rate,-w[0],n,kd,dt);fb,hb=weighted(f,h,rate,-w[1],n,kd,dt)
        fc,hc=weighted(f,h,rate,-w[2],n,kd,dt);fv,hv=weighted(f,h,rate,1.,n,kd,dt)
        return True,fa,fb,fc,fv,ha,hb,hc,hv
    f=wp.vec3(0.);h=wp.mat33(0.)
    return False,f,f,f,f,h,h,h,h

@wp.func
def ee(i:int,j:int,q:wp.array[wp.vec3],inc:wp.array[wp.vec3],edges:wp.array2d[int],
    radius:float,ke:float,kd:float,mu:float,eps:float,dt:float,parallel:float):
    a=edges[i,2];b=edges[i,3];c=edges[j,2];d=edges[j,3]
    st=wp.closest_point_edge_edge(q[a],q[b],q[c],q[d],parallel);s=st[0];t=st[1];dist=st[2]
    if 0.<dist<radius:
        p=q[a]+s*(q[b]-q[a]);r=q[c]+t*(q[d]-q[c]);n=(p-r)/dist
        dx=(1.-s)*inc[a]+s*inc[b]-((1.-t)*inc[c]+t*inc[d])
        f,h,rate=response(n,dist,dx,radius,ke,kd,mu,eps,dt)
        fa,ha=weighted(f,h,rate,1.-s,n,kd,dt);fb,hb=weighted(f,h,rate,s,n,kd,dt)
        return True,fa,fb,ha,hb
    f=wp.vec3(0.);h=wp.mat33(0.)
    return False,f,f,h,h

def kernel_copy(root):
    from warp._src.context import Kernel
    f=root.func;ns=dict(f.__globals__)
    for name,new in [('evaluate_edge_edge_contact_2_vertices',ee),('evaluate_vertex_triangle_collision_force_hessian_4_vertices',vt)]:
        if name not in ns:raise RuntimeError('Expected contact dependency absent')
        ns[name]=new
    copied=FunctionType(f.__code__,ns,f.__name__,f.__defaults__,f.__closure__)
    copied.__module__=__name__;copied.__qualname__=f.__qualname__;copied.__annotations__=dict(f.__annotations__)
    return Kernel(copied,key=root.key+'_precise_increment',module=wp.get_module(__name__),options=dict(root.options))

def install(g,*,rigid=False):
    from deposit_cans.translation import Translation
    from deposit_cans.contact_accumulation import split
    from deposit_cans import force_only
    contact=g.contact;p=g.precision
    if p is None:raise ValueError('Precise elastic coordinates required')
    contact.precise_increment=wp.zeros_like(g.q)
    force_ns=SimpleNamespace(**vars(force_only));force_ns.self_contact=kernel_copy(force_only.self_contact)
    # Copy the Python launch method only. Installed modules and another solver
    # instance are never patched. The body/spring previous-state inputs stay intact.
    source=textwrap.dedent(inspect.getsource(Translation.evaluate))
    target='[dt,color,s.particle_q_prev,q,self.all_colors,m.tri_indices'
    if source.count(target)!=1:raise RuntimeError('Self-contact launch changed; review experiment')
    source=source.replace(target,'[dt,color,self.precise_increment,q,self.all_colors,m.tri_indices')
    source=source.replace('from . import force_only','force_only=precise_force_only')
    namespace=dict(Translation.evaluate.__globals__)
    namespace['accumulate_self_contact_force_and_hessian']=kernel_copy(split)
    namespace['precise_force_only']=force_ns
    if rigid:
        from deposit_cans.precise_body import copy_kernel,install_rigid
        target='[dt,color,s.particle_q_prev,q,self.all_colors,s.friction_epsilon'
        if source.count(target)!=1:raise RuntimeError('Particle/body launch changed')
        source=source.replace(target,'[dt,color,self.precise_previous,self.precise_query,self.all_colors,s.friction_epsilon')
        namespace['accumulate_particle_body_contact_force_and_hessian']=copy_kernel(namespace['accumulate_particle_body_contact_force_and_hessian'])
        force_ns.body=copy_kernel(force_only.body)
        contact.precise_previous=p.previous
        install_rigid(g.s)
    exec(compile(source,'<precise-self-contact-launch>','exec'),namespace)
    evaluate=MethodType(namespace['evaluate'],contact)
    def wrapped(self,q,state,contacts,dt,**kwargs):
        mapping={g.q.ptr:p.current,g.trial.ptr:p.trial,g.work.q.ptr:p.sample,g.bestq.ptr:p.current}
        exact=mapping.get(q.ptr)
        if exact is None:raise ValueError('Unmapped precise contact query')
        self.precise_query=exact
        wp.launch(increments,g.s.model.particle_count,[exact,p.previous,self.precise_increment],device=g.device)
        return evaluate(q,state,contacts,dt,**kwargs)
    contact.evaluate=MethodType(wrapped,contact)
    return 'Precise self-contact slip and '+('two-sided rigid/soft geometry.'if rigid else'unchanged rigid/soft contact.')
