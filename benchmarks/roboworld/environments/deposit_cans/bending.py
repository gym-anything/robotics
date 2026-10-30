# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Newton 1.5 hinge law with a continuous principal elastic angle difference.

The native damping already wraps its angle increment; native elastic bending
does not. A rim crossing +pi/-pi must not acquire a spurious full revolution
of elastic strain. Same stiffness, gradients, Gauss-Newton Hessian and damping.
Private dependency copies do not modify Newton or other running environments.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
from types import FunctionType
import warp as wp
from newton._src.solvers.vbd.particle_vbd_kernels import (
    compute_normalized_vector_derivative,compute_angle_derivative)

@wp.func
def difference(a:float,b:float):
    return wp.atan2(wp.sin(a-b),wp.cos(a-b))

@wp.func
def bending(index:int,order:int,q:wp.array[wp.vec3],previous:wp.array[wp.vec3],
    edges:wp.array2d[int],rest:wp.array[float],length:wp.array[float],
    stiffness:float,damping:float,dt:float):
    a=edges[index,0];b=edges[index,1];c=edges[index,2];d=edges[index,3]
    if a<0 or b<0:return wp.vec3(0.),wp.mat33(0.)
    x02=q[c]-q[a];x03=q[d]-q[a];x13=q[d]-q[b];x12=q[c]-q[b];e=q[d]-q[c]
    n1=wp.cross(x02,x03);n2=wp.cross(x13,x12)
    l1=wp.length(n1);l2=wp.length(n2);le=wp.length(e)
    if l1<1e-6 or l2<1e-6 or le<1e-6:return wp.vec3(0.),wp.mat33(0.)
    n1/=l1;n2/=l2;axis=e/le
    sine=wp.dot(wp.cross(n1,n2),axis);cosine=wp.dot(n1,n2);theta=wp.atan2(sine,cosine)
    g10=compute_normalized_vector_derivative(l1,n1,wp.skew(e))
    g21=compute_normalized_vector_derivative(l2,n2,-wp.skew(e))
    g12=compute_normalized_vector_derivative(l1,n1,-wp.skew(x03))
    g22=compute_normalized_vector_derivative(l2,n2,wp.skew(x13))
    g13=compute_normalized_vector_derivative(l1,n1,wp.skew(x02))
    g23=compute_normalized_vector_derivative(l2,n2,-wp.skew(x12))
    grad0=compute_angle_derivative(n1,n2,axis,g10,wp.mat33(0.),sine,cosine,wp.skew(n1),wp.skew(n2))
    grad1=compute_angle_derivative(n1,n2,axis,wp.mat33(0.),g21,sine,cosine,wp.skew(n1),wp.skew(n2))
    grad2=compute_angle_derivative(n1,n2,axis,g12,g22,sine,cosine,wp.skew(n1),wp.skew(n2))
    grad3=compute_angle_derivative(n1,n2,axis,g13,g23,sine,cosine,wp.skew(n1),wp.skew(n2))
    grad=grad0*float(order==0)+grad1*float(order==1)+grad2*float(order==2)+grad3*float(order==3)
    k=stiffness*length[index]
    force=-k*difference(theta,rest[index])*grad;hessian=k*wp.outer(grad,grad)
    if damping>0.:
        pn1=wp.cross(previous[c]-previous[a],previous[d]-previous[a])
        pn2=wp.cross(previous[d]-previous[b],previous[c]-previous[b]);pe=previous[d]-previous[c]
        if wp.length(pn1)>=1e-6 and wp.length(pn2)>=1e-6 and wp.length(pe)>=1e-6:
            pn1=wp.normalize(pn1);pn2=wp.normalize(pn2);pe=wp.normalize(pe)
            old=wp.atan2(wp.dot(wp.cross(pn1,pn2),pe),wp.dot(pn1,pn2))
            damp=damping*length[index]/dt
            force-=damp*difference(theta,old)*grad;hessian+=damp*wp.outer(grad,grad)
    return force,hessian

@lru_cache(maxsize=2)
def redirect(base):
    from warp._src.context import Function,Kernel
    from newton._src.solvers.vbd import particle_vbd_kernels as native
    expected='095faac8c2d76ba14e02e534910206e98f7b97404c159256e69b91413eef8bcf'
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Review hinge correction against changed Newton source')
    memo={id(native.evaluate_dihedral_angle_based_bending_force_hessian):bending};changed=[]
    def copy(f,namespace):
        result=FunctionType(f.__code__,namespace,f.__name__,f.__defaults__,f.__closure__)
        result.__kwdefaults__=f.__kwdefaults__;result.__annotations__=dict(f.__annotations__)
        result.__module__=__name__;result.__qualname__=f.__qualname__;return result
    def dependency(item):
        if id(item) in memo:return memo[id(item)]
        if not isinstance(item,(Function,Kernel)) or item.func is None:return item
        f=item.func;namespace=dict(f.__globals__);modified=False
        for name in f.__code__.co_names:
            if name not in namespace:continue
            old=namespace[name];new=dependency(old)
            if new is not old:namespace[name]=new;modified=True
        if not modified:memo[id(item)]=item;return item
        function=copy(f,namespace)
        result=Kernel(function,key=item.key+'_continuous_hinge',module=wp.get_module(__name__),options=dict(item.options)) if isinstance(item,Kernel) else wp.func(function)
        memo[id(item)]=result;changed.append(item.key);return result
    namespace=dict(base.__globals__)
    for name in base.__code__.co_names:
        if name in namespace:namespace[name]=dependency(namespace[name])
    if not changed:raise RuntimeError('No bending solve kernel redirected')
    return copy(base,namespace)
