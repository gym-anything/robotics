# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Numerically stable relative displacement for native cloth self-contact.

The two contact functions below follow Newton1.5 particle_vbd_kernels.py;
only relative displacement evaluation is changed. Interpolate per-vertex
increments rather than subtracting differently reconstructed world points.
No material, normal-contact law, friction law, damping, state or threshold change.
Private dependency copies leave the installed package and other sessions intact.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
from types import FunctionType
import warp as wp
from newton._src.geometry.kernels import triangle_closest_point
from newton._src.math import orthonormal_basis
from newton._src.solvers.vbd.particle_vbd_kernels import (
    compute_friction,damp_collision,evaluate_self_contact_force_norm,mat32)


@wp.func
def vertex_triangle(v:int,v_order:int,tri:int,pos:wp.array[wp.vec3],pos_anchor:wp.array[wp.vec3],
    tri_indices:wp.array2d[wp.int32],collision_radius:float,collision_stiffness:float,
    collision_damping:float,friction_coefficient:float,friction_epsilon:float,dt:float):
    a=pos[tri_indices[tri,0]];b=pos[tri_indices[tri,1]];c=pos[tri_indices[tri,2]];p=pos[v]
    closest_p,bary,feature=triangle_closest_point(a,b,c,p)
    diff=p-closest_p;dis=wp.length(diff);normal=diff/dis
    if dis<collision_radius:
        bs=wp.vec4(-bary[0],-bary[1],-bary[2],1.);weight=bs[v_order]
        dEdD,d2E=evaluate_self_contact_force_norm(dis,collision_radius,collision_stiffness)
        force=-dEdD*weight*normal;hessian=d2E*weight*weight*wp.outer(normal,normal)
        dx=(p-pos_anchor[v])-(bary[0]*(a-pos_anchor[tri_indices[tri,0]])+
            bary[1]*(b-pos_anchor[tri_indices[tri,1]])+bary[2]*(c-pos_anchor[tri_indices[tri,2]]))
        e0,e1=orthonormal_basis(normal)
        tangent=mat32(e0[0],e1[0],e0[1],e1[1],e0[2],e1[2])
        u=wp.transpose(tangent)*dx
        friction,friction_h=compute_friction(friction_coefficient,-dEdD,tangent,u,friction_epsilon*dt)
        damping,damping_h=damp_collision(wp.dot(normal,dx)/dt,weight,normal,collision_damping,dt)
        force=force+damping;hessian=hessian+damping_h
        force=force+weight*friction;hessian=hessian+weight*weight*friction_h
    else:
        force=wp.vec3(0.);hessian=wp.mat33(0.)
    return force,hessian


@wp.func
def vertex_triangle_all(v:int,tri:int,pos:wp.array[wp.vec3],pos_anchor:wp.array[wp.vec3],
    tri_indices:wp.array2d[wp.int32],collision_radius:float,collision_stiffness:float,
    collision_damping:float,friction_coefficient:float,friction_epsilon:float,dt:float):
    a=pos[tri_indices[tri,0]];b=pos[tri_indices[tri,1]];c=pos[tri_indices[tri,2]];p=pos[v]
    closest_p,bary,feature=triangle_closest_point(a,b,c,p)
    diff=p-closest_p;dis=wp.length(diff);normal=diff/dis
    if 0.<dis<collision_radius:
        bs=wp.vec4(-bary[0],-bary[1],-bary[2],1.)
        dEdD,d2E=evaluate_self_contact_force_norm(dis,collision_radius,collision_stiffness)
        force=-dEdD*normal;hessian=d2E*wp.outer(normal,normal)
        dx=(p-pos_anchor[v])-(bary[0]*(a-pos_anchor[tri_indices[tri,0]])+
            bary[1]*(b-pos_anchor[tri_indices[tri,1]])+bary[2]*(c-pos_anchor[tri_indices[tri,2]]))
        e0,e1=orthonormal_basis(normal)
        tangent=mat32(e0[0],e1[0],e0[1],e1[1],e0[2],e1[2])
        u=wp.transpose(tangent)*dx
        friction,friction_h=compute_friction(friction_coefficient,-dEdD,tangent,u,friction_epsilon*dt)
        gap_rate=wp.dot(normal,dx)/dt
        f0=force*bs[0];f1=force*bs[1];f2=force*bs[2];f3=force*bs[3]
        h0=hessian*bs[0]*bs[0];h1=hessian*bs[1]*bs[1]
        h2=hessian*bs[2]*bs[2];h3=hessian*bs[3]*bs[3]
        damping,damping_h=damp_collision(gap_rate,bs[0],normal,collision_damping,dt)
        f0+=damping+bs[0]*friction;h0+=damping_h+bs[0]*bs[0]*friction_h
        damping,damping_h=damp_collision(gap_rate,bs[1],normal,collision_damping,dt)
        f1+=damping+bs[1]*friction;h1+=damping_h+bs[1]*bs[1]*friction_h
        damping,damping_h=damp_collision(gap_rate,bs[2],normal,collision_damping,dt)
        f2+=damping+bs[2]*friction;h2+=damping_h+bs[2]*bs[2]*friction_h
        damping,damping_h=damp_collision(gap_rate,bs[3],normal,collision_damping,dt)
        f3+=damping+bs[3]*friction;h3+=damping_h+bs[3]*bs[3]*friction_h
        return True,f0,f1,f2,f3,h0,h1,h2,h3
    else:
        f=wp.vec3(0.);h=wp.mat33(0.)
        return False,f,f,f,f,h,h,h,h


@lru_cache(maxsize=4)
def redirect_particle_method(base):
    from warp._src.context import Function,Kernel
    from newton._src.solvers.vbd import particle_vbd_kernels as native
    expected='095faac8c2d76ba14e02e534910206e98f7b97404c159256e69b91413eef8bcf'
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Review increment arithmetic against changed Newton source')
    memo={id(native.evaluate_vertex_triangle_collision_force_hessian):vertex_triangle,
          id(native.evaluate_vertex_triangle_collision_force_hessian_4_vertices):vertex_triangle_all}
    changed=[];visiting=set()
    def copy_function(f,namespace):
        copied=FunctionType(f.__code__,namespace,f.__name__,f.__defaults__,f.__closure__)
        copied.__kwdefaults__=f.__kwdefaults__;copied.__annotations__=dict(f.__annotations__)
        copied.__module__=__name__;copied.__qualname__=f.__qualname__
        return copied
    def dependency(item):
        if id(item) in memo:return memo[id(item)]
        if not isinstance(item,(Function,Kernel)) or item.func is None:return item
        if id(item) in visiting:raise RuntimeError('Recursive Warp self-contact dependency')
        visiting.add(id(item));f=item.func;namespace=dict(f.__globals__);modified=False
        for name in f.__code__.co_names:
            if name not in namespace:continue
            old=namespace[name];new=dependency(old)
            if new is not old:namespace[name]=new;modified=True
        visiting.remove(id(item))
        if not modified:memo[id(item)]=item;return item
        copied=copy_function(f,namespace)
        if isinstance(item,Kernel):
            result=Kernel(copied,key=item.key+'_self_increment',module=wp.get_module(__name__),options=dict(item.options))
        else:result=wp.func(copied)
        memo[id(item)]=result;changed.append(item.key);return result
    namespace=dict(base.__globals__)
    for key in base.__code__.co_names:
        if key in namespace:namespace[key]=dependency(namespace[key])
    if not changed:raise RuntimeError('No self-contact kernels redirected')
    return copy_function(base,namespace),tuple(changed)
