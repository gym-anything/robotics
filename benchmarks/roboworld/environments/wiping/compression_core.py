"""Opt-in pile/core compression candidate; not a nonpenetration guarantee.

The historical linear1500N/m law permits the microfiber midsurface to enter a
rigid object once contact load exceeds2.25N. This candidate retains that pile
response through0.5mm compression, then adds a30000N/m core spring. The potential
is .5*k*d**2 + .5*kc*max(d-d0,0)**2: continuous force with its exact normal
tangent. Native closing-only damping and Coulomb friction use the same new load
on BOTH bodies and in force telemetry. No state/velocity projection or pinning.
These numbers are explicit numerical/constitutive estimates, not product data.
"""
from functools import lru_cache
import warp as wp
from newton._src.solvers.vbd.rigid_vbd_kernels import compute_projected_isotropic_friction

CORE_ONSET_METRES=.0005
CORE_ONSET_M=wp.constant(CORE_ONSET_METRES)
CORE_STIFFNESS_N_M=wp.constant(30000.)


@wp.func
def core_force(depth:float,n:wp.vec3,u:wp.vec3,ke:float,kd:float,mu:float,eps:float,dt:float):
    normal_load=ke*depth+CORE_STIFFNESS_N_M*wp.max(depth-CORE_ONSET_M,0.)
    tangent=ke
    if depth>CORE_ONSET_M:tangent+=CORE_STIFFNESS_N_M
    nn=wp.outer(n,n)
    force=n*normal_load;hessian=tangent*nn
    if wp.dot(n,u)<0.:
        damping=(kd/dt)*nn
        force-=damping*u;hessian+=damping
    f,k=compute_projected_isotropic_friction(mu,normal_load,n,u,eps*dt)
    return force+f,hessian+k


@lru_cache(maxsize=1)
def redirected_methods():
    """Private Warp dependency copies; native package/global functions untouched.

    Redirect both sides of the solve, proxy feedback, and the recorded force
    query together. Altering telemetry alone would not alter physical contact.
    """
    import hashlib
    from pathlib import Path
    from types import FunctionType
    from warp._src.context import Function,Kernel
    from newton.solvers import SolverVBD
    from newton._src.solvers.vbd import rigid_vbd_kernels as native
    from .vbd_contact import collect
    expected='0a34069f3f6bc6e7b0063117371fc70f9808751b93053dfd7b3e6d401634cf3f'
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Review compression candidate against changed Newton source')
    original=native._compute_body_particle_contact_force
    memo={id(original):core_force};changed=[];visiting=set()
    def python_copy(f,namespace):
        copied=FunctionType(f.__code__,namespace,f.__name__,f.__defaults__,f.__closure__)
        copied.__kwdefaults__=f.__kwdefaults__;copied.__annotations__=dict(f.__annotations__)
        copied.__module__=__name__;copied.__qualname__=f.__qualname__
        return copied
    def dependency(item):
        if id(item) in memo:return memo[id(item)]
        if not isinstance(item,(Function,Kernel)) or item.func is None:return item
        if id(item) in visiting:raise RuntimeError('Recursive Warp contact dependency')
        visiting.add(id(item));f=item.func;namespace=dict(f.__globals__);modified=False
        for name in f.__code__.co_names:
            if name not in namespace:continue
            old=namespace[name];new=dependency(old)
            if new is not old:namespace[name]=new;modified=True
        visiting.remove(id(item))
        if not modified:memo[id(item)]=item;return item
        copied=python_copy(f,namespace)
        if isinstance(item,Kernel):
            result=Kernel(copied,key=item.key+'_compression_core',module=wp.get_module(__name__),options=dict(item.options))
        else:result=wp.func(copied)
        memo[id(item)]=result;changed.append(item.key);return result
    methods=[]
    for name in ('_solve_rigid_body_iteration','_solve_particle_iteration','coupling_harvest_proxy_wrenches'):
        f=getattr(SolverVBD,name);namespace=dict(f.__globals__)
        for key in f.__code__.co_names:
            if key in namespace:namespace[key]=dependency(namespace[key])
        methods.append(python_copy(f,namespace))
    recorded=dependency(collect)
    if native._compute_body_particle_contact_force is not original:raise RuntimeError('Native function replaced')
    if recorded is collect:raise RuntimeError('Contact telemetry does not use candidate force law')
    return tuple(methods),recorded,tuple(changed)
