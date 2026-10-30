# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Instance-local float64 narrow-phase self-contact geometry.

Same installed closest-point branches, contact law and discovery lists. No
material, friction, collision thickness, physical DOF or acceptance change.
The installed Warp edge primitive is float32-only, hence the explicit copy.
No other simulator instance or installed module is patched.
"""
import inspect,linecache,textwrap
from types import MethodType
import warp as wp
from deposit_cans.elastic_precision import single
from deposit_cans.precise_increment import response,weighted

# One closest-feature implementation for force assembly and contact readback.
from .self_geometry import triangle,edge_pair

@wp.func
def vt(v:int,t:int,q:wp.array[wp.vec3d],inc:wp.array[wp.vec3],tri:wp.array2d[int],
       radius:float,ke:float,kd:float,mu:float,eps:float,dt:float):
    a=tri[t,0];b=tri[t,1];c=tri[t,2]
    closest,wd=triangle(q[a],q[b],q[c],q[v]);gap=q[v]-closest;dd=wp.length(gap)
    if wp.float64(0.)<dd<wp.float64(radius):
        n=single(gap/dd);w=single(wd);dx=inc[v]-(w[0]*inc[a]+w[1]*inc[b]+w[2]*inc[c])
        f,h,rate=response(n,float(dd),dx,radius,ke,kd,mu,eps,dt)
        fa,ha=weighted(f,h,rate,-w[0],n,kd,dt);fb,hb=weighted(f,h,rate,-w[1],n,kd,dt)
        fc,hc=weighted(f,h,rate,-w[2],n,kd,dt);fv,hv=weighted(f,h,rate,1.,n,kd,dt)
        return True,fa,fb,fc,fv,ha,hb,hc,hv
    f=wp.vec3(0.);h=wp.mat33(0.)
    return False,f,f,f,f,h,h,h,h

@wp.func
def ee(i:int,j:int,q:wp.array[wp.vec3d],inc:wp.array[wp.vec3],edges:wp.array2d[int],
       radius:float,ke:float,kd:float,mu:float,eps:float,dt:float,parallel:float):
    a=edges[i,2];b=edges[i,3];c=edges[j,2];d=edges[j,3]
    st=edge_pair(i,j,q,edges,wp.float64(parallel));sd=st[0];td=st[1];dd=st[2]
    if wp.float64(0.)<dd<wp.float64(radius):
        p=q[a]+sd*(q[b]-q[a]);r=q[c]+td*(q[d]-q[c]);n=single((p-r)/dd)
        s=float(sd);t=float(td);dx=(1.-s)*inc[a]+s*inc[b]-((1.-t)*inc[c]+t*inc[d])
        f,h,rate=response(n,float(dd),dx,radius,ke,kd,mu,eps,dt)
        fa,ha=weighted(f,h,rate,1.-s,n,kd,dt);fb,hb=weighted(f,h,rate,s,n,kd,dt)
        return True,fa,fb,ha,hb
    f=wp.vec3(0.);h=wp.mat33(0.)
    return False,f,f,h,h

def kernel_copy(root):
    source=textwrap.dedent(inspect.getsource(root.func))
    if source.count('q:wp.array[wp.vec3]')!=1:raise ValueError('Changed self-contact kernel signature')
    source=source.replace('q:wp.array[wp.vec3]','q:wp.array[wp.vec3d]')
    ns=dict(root.func.__globals__);ns.update(__name__=__name__,evaluate_edge_edge_contact_2_vertices=ee,
        evaluate_vertex_triangle_collision_force_hessian_4_vertices=vt)
    filename='<precise-self-geometry-'+root.key+'>'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    exec(compile(source,filename,'exec'),ns)
    return ns[root.func.__name__]

def install(g):
    from deposit_cans import precise_increment as original
    source=textwrap.dedent(inspect.getsource(original.install))
    target="source=source.replace('from . import force_only','force_only=precise_force_only')"
    if source.count(target)!=1:raise ValueError('Changed precise-increment launcher')
    source=source.replace(target,target+"\n    source=source.replace('[dt,color,self.precise_increment,q,self.all_colors,m.tri_indices','[dt,color,self.precise_increment,self.precise_query,self.all_colors,m.tri_indices')")
    # v4 already installed its precise rigid side before scheduling/ALM. Do
    # not reinstall that kernel through the subsequently transformed method.
    if source.count('install_rigid(g.s)')!=1:raise ValueError('Changed rigid installation site')
    source=source.replace('install_rigid(g.s)','pass # Keep existing v4 rigid implementation')
    ns=dict(original.install.__globals__);ns['kernel_copy']=kernel_copy
    exec(compile(source,'<precise-self-geometry-install>','exec'),ns)
    ns['install'](g,rigid=True)
    original_evaluate=g.contact.evaluate
    def evaluate(q,state,c,dt,**kwargs):
        g.surface_search.context=(state,c,dt)
        return original_evaluate(q,state,c,dt,**kwargs)
    g.contact.evaluate=evaluate
    g.precise_self_geometry=True
