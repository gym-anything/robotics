# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exact per-vertex PSD membrane block; native force/damping are unchanged.

For F=[f0,f1], changing one vertex changes both columns by scalar multiples
of the same vector. Let A=d0*d0+d1*d1, w=d0*f1-d1*f0, n=normalize(f0 cross f1).
The exact elastic block has orthogonal eigenvectors w, dJ/dx, n and eigenvalues
mu*A, mu*A+lambda_NH*|w|^2, mu*A+s/J*|w|^2 respectively.
Only the last can be negative for the positive authored Lame coefficients.
Newton1.5 clamps s to zero, adding unnecessary normal curvature at rest. This
candidate clips just that negative eigenvalue. No force or damping change.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
from types import FunctionType
import warp as wp
from newton._src.solvers.vbd.particle_vbd_kernels import evaluate_neo_hookean_membrane_force_hessian as native_membrane


@wp.func
def membrane(face: int, order: int, pos: wp.array[wp.vec3], prev: wp.array[wp.vec3],
    tri: wp.array2d[int], pose: wp.mat22, area: float,
    mu: float, lmbd: float, damping: float, dt: float):
    force, hessian = native_membrane(face, order, pos, prev, tri, pose, area,
                                    mu, lmbd, damping, dt)
    a=pos[tri[face,1]]-pos[tri[face,0]]; b=pos[tri[face,2]]-pos[tri[face,0]]
    f0=a*pose[0,0]+b*pose[1,0]; f1=a*pose[0,1]+b*pose[1,1]
    cross=wp.cross(f0,f1); J=wp.length(cross)
    s=(lmbd+mu)*(J-1.)-mu
    if s<0. and J>1.e-10:
        d0=pose[0,0]*(float(order==1)-float(order==0))+pose[1,0]*(float(order==2)-float(order==0))
        d1=pose[0,1]*(float(order==1)-float(order==0))+pose[1,1]*(float(order==2)-float(order==0))
        w=f1*d0-f0*d1; normal=cross/J
        reduction=wp.min(mu*(d0*d0+d1*d1),-s/J*wp.dot(w,w))
        hessian-=area*reduction*wp.outer(normal,normal)
    return force,hessian


@lru_cache(maxsize=8)
def redirect_particle_method(base):
    from warp._src.context import Function,Kernel
    from newton._src.solvers.vbd import particle_vbd_kernels as native
    expected='095faac8c2d76ba14e02e534910206e98f7b97404c159256e69b91413eef8bcf'
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Review exact membrane block against changed Newton source')
    memo={id(native.evaluate_neo_hookean_membrane_force_hessian):membrane}
    changed=[];visiting=set()
    def copy_function(f,namespace):
        copied=FunctionType(f.__code__,namespace,f.__name__,f.__defaults__,f.__closure__)
        copied.__kwdefaults__=f.__kwdefaults__;copied.__annotations__=dict(f.__annotations__)
        copied.__module__=__name__;copied.__qualname__=f.__qualname__
        return copied
    def dependency(item):
        if id(item) in memo:return memo[id(item)]
        if not isinstance(item,(Function,Kernel)) or item.func is None:return item
        if id(item) in visiting:raise RuntimeError('Recursive Warp membrane dependency')
        visiting.add(id(item));f=item.func;namespace=dict(f.__globals__);modified=False
        for name in f.__code__.co_names:
            if name not in namespace:continue
            old=namespace[name];new=dependency(old)
            if new is not old:namespace[name]=new;modified=True
        visiting.remove(id(item))
        if not modified:memo[id(item)]=item;return item
        copied=copy_function(f,namespace)
        if isinstance(item,Kernel):
            result=Kernel(copied,key=item.key+'_exact_membrane',module=wp.get_module(__name__),options=dict(item.options))
        else:result=wp.func(copied)
        memo[id(item)]=result;changed.append(item.key);return result
    namespace=dict(base.__globals__)
    for key in base.__code__.co_names:
        if key in namespace:namespace[key]=dependency(namespace[key])
    if not changed:raise RuntimeError('No membrane solve kernels redirected')
    return copy_function(base,namespace),tuple(changed)
