# SPDX-License-Identifier: Apache-2.0
"""Continuous zero-slip limit of the existing Newton friction search block.

Newton's force is unchanged. Its u==0 branch omits the nonzero limiting
tangent 2*mu*N/eps * T*T^T. That branch makes the initial search underestimate
stiction by orders of magnitude. This privately copies only dependencies;
it never patches installed Newton or changes another environment.
Basis: IPC technical supplement section9, C1 friction; C-IPC supplement5.

Also reuse the independently audited VT increment interpolation from wiping:
stationary can/bag queries otherwise generate up to0.175N of phantom friction.
Both corrections are private; force laws and physical coefficients are unchanged.
"""
from functools import lru_cache
from types import FunctionType
import hashlib
from pathlib import Path
import warp as wp
from newton._src.solvers.vbd import particle_vbd_kernels as native
from newton._src.solvers.vbd.particle_vbd_kernels import mat32
from newton._src.solvers.vbd.rigid_vbd_kernels import compute_projected_isotropic_friction as native_projected

@wp.func
def friction(mu:float,normal:float,T:mat32,u:wp.vec2,eps:float):
    f,h=native.compute_friction(mu,normal,T,u,eps)
    if wp.length(u)==0. and eps>0.:
        h=(2.*mu*normal/eps)*(T*wp.transpose(T))
    return f,h

@wp.func
def projected(mu:float,normal:float,n:wp.vec3,u:wp.vec3,eps:float):
    f,h=native_projected(mu,normal,n,u,eps)
    tangent=u-n*wp.dot(n,u)
    if wp.length(tangent)==0. and eps>0.:
        h=(2.*mu*normal/eps)*(wp.identity(3,float)-wp.outer(n,n))
    return f,h

@lru_cache(maxsize=8)
def redirect(root):
    from warp._src.context import Function,Kernel
    expected='095faac8c2d76ba14e02e534910206e98f7b97404c159256e69b91413eef8bcf'
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Review friction tangent against changed Newton source')
    memo={id(native.compute_friction):friction,id(native_projected):projected}
    from wiping.self_contact_increment import vertex_triangle_all
    def dependency(item):
        if id(item) in memo:return memo[id(item)]
        if not isinstance(item,(Function,Kernel)) or item.func is None:return item
        # A previous private dependency copy preserves the original code object,
        # even when its Function identity differs. Match only this exact pinned
        # native function, never unrelated name strings or installed globals.
        if item.func.__code__ is native.evaluate_vertex_triangle_collision_force_hessian_4_vertices.func.__code__:
            result=dependency(vertex_triangle_all)
            memo[id(item)]=result
            return result
        f=item.func;namespace=dict(f.__globals__);modified=False
        for name in f.__code__.co_names:
            if name not in namespace:continue
            old=namespace[name];new=dependency(old)
            if new is not old:namespace[name]=new;modified=True
        if not modified:memo[id(item)]=item;return item
        copied=FunctionType(f.__code__,namespace,f.__name__,f.__defaults__,f.__closure__)
        copied.__kwdefaults__=f.__kwdefaults__;copied.__annotations__=dict(f.__annotations__)
        copied.__module__=__name__;copied.__qualname__=f.__qualname__
        result=Kernel(copied,key=item.key+'_friction_limit',module=wp.get_module(__name__),options=dict(item.options)) if isinstance(item,Kernel) else wp.func(copied)
        memo[id(item)]=result;return result
    result=dependency(root)
    if result is root:raise RuntimeError('Friction dependency was not found')
    return result

@wp.kernel
def probe(values:wp.array2d[float]):
    i=wp.tid();eps=float(.0005/960.);mu=float(.3);normal=float(1.)
    u=wp.vec2(float(i-1)*eps*1e-4,0.)
    T=mat32(1.,0.,0.,1.,0.,0.)
    oldf,oldh=native.compute_friction(mu,normal,T,u,eps)
    newf,newh=friction(mu,normal,T,u,eps)
    values[i,0]=u[0];values[i,1]=oldf[0];values[i,2]=newf[0]
    values[i,3]=oldh[0,0];values[i,4]=newh[0,0]

if __name__=='__main__':
    import numpy as np
    a=wp.zeros((3,5),dtype=float,device='cuda:0');wp.launch(probe,3,[a],device='cuda:0')
    v=a.numpy();print(v.tolist())
    derivative=-(v[2,1]-v[0,1])/(v[2,0]-v[0,0])
    assert np.array_equal(v[:,1],v[:,2]);assert abs(v[1,4]/derivative-1)<1e-4
    print('finite_difference_tangent',float(derivative),'old_zero_slip',float(v[1,3]),'corrected',float(v[1,4]))
