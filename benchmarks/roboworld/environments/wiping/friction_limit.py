"""Candidate zero-slip Hessian limit, preserving the native friction force.

Native Newton1.5 returns K=0 exactly at zero slip despite its regularized force
having a finite nonzero derivative there. This changes only that Hessian branch.
No force, coefficient, velocity or pose clamp is added. Not yet qualified.
"""
import warp as wp
from functools import lru_cache
from newton._src.solvers.vbd.rigid_vbd_kernels import compute_projected_isotropic_friction as native_friction
from newton._src.solvers.vbd.particle_vbd_kernels import compute_friction as native_self_friction,mat32


@wp.func
def friction_with_zero_limit(mu:float,normal_load:float,n:wp.vec3,u:wp.vec3,eps:float):
    force,hessian=native_friction(mu,normal_load,n,u,eps)
    tangent=u-n*wp.dot(n,u)
    if wp.length(tangent)==0.0:
        hessian=(2.0*mu*normal_load/eps)*(wp.identity(3,float)-wp.outer(n,n))
    return force,hessian


@wp.func
def self_friction_with_zero_limit(mu:float,normal_load:float,T:mat32,u:wp.vec2,eps:float):
    force,hessian=native_self_friction(mu,normal_load,T,u,eps)
    if wp.length(u)==0.0:
        hessian=(2.0*mu*normal_load/eps)*T*wp.transpose(T)
    return force,hessian


@lru_cache(maxsize=2)
def redirected_methods(*,include_self_contact=False):
    """Private dependency copies, not installed-source edits/global replacement.

    Copy only Warp callers that reach the corrected function. Their Python
    bytecode/source and options stay unchanged; new private globals resolve that
    one dependency. Unaffected functions/kernels remain the native objects.
    Two Python dispatch-method copies use those kernels for this solver only.
    """
    import hashlib
    from pathlib import Path
    from types import FunctionType
    from warp._src.context import Function,Kernel
    from newton.solvers import SolverVBD
    from newton._src.solvers.vbd import rigid_vbd_kernels as native
    expected='0a34069f3f6bc6e7b0063117371fc70f9808751b93053dfd7b3e6d401634cf3f'
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Review zero-slip correction against changed Newton source')
    original=native.compute_projected_isotropic_friction
    memo={id(original):friction_with_zero_limit};changed=[];visiting=set()
    if include_self_contact:
        from newton._src.solvers.vbd import particle_vbd_kernels as particles
        expected_self='095faac8c2d76ba14e02e534910206e98f7b97404c159256e69b91413eef8bcf'
        if hashlib.sha256(Path(particles.__file__).read_bytes()).hexdigest()!=expected_self:
            raise RuntimeError('Review zero-slip correction against changed native particle source')
        memo[id(particles.compute_friction)]=self_friction_with_zero_limit
    def python_copy(f,namespace):
        copied=FunctionType(f.__code__,namespace,f.__name__,f.__defaults__,f.__closure__)
        copied.__kwdefaults__=f.__kwdefaults__;copied.__annotations__=dict(f.__annotations__)
        copied.__module__=__name__;copied.__qualname__=f.__qualname__
        return copied
    def dependency(item):
        if id(item) in memo:return memo[id(item)]
        if not isinstance(item,(Function,Kernel)) or item.func is None:return item
        if id(item) in visiting:raise RuntimeError('Unexpected recursive Warp contact dependency')
        visiting.add(id(item));f=item.func;namespace=dict(f.__globals__);modified=False
        for name in f.__code__.co_names:
            if name not in namespace:continue
            previous=namespace[name];current=dependency(previous)
            if current is not previous:namespace[name]=current;modified=True
        visiting.remove(id(item))
        if not modified:memo[id(item)]=item;return item
        copied=python_copy(f,namespace)
        if isinstance(item,Kernel):
            result=Kernel(copied,key=item.key+'_zero_slip',module=wp.get_module(__name__),options=dict(item.options))
        else:result=wp.func(copied)
        memo[id(item)]=result;changed.append(item.key);return result
    methods=[]
    for name in ('_solve_rigid_body_iteration','_solve_particle_iteration'):
        f=getattr(SolverVBD,name);namespace=dict(f.__globals__)
        for key in f.__code__.co_names:
            if key in namespace:namespace[key]=dependency(namespace[key])
        methods.append(python_copy(f,namespace))
    if native.compute_projected_isotropic_friction is not original:
        raise RuntimeError('Native function identity unexpectedly changed')
    if not changed:raise RuntimeError('Zero-slip correction reached no solver kernel')
    return tuple(methods),tuple(changed)


if __name__=='__main__':
    import json
    print(json.dumps(dict(redirected_kernels=redirected_methods()[1],native_globals_unchanged=True)))
