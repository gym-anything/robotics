# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Instance-local rigid/shell contact using precise positions on BOTH sides.

Preserve native pairs, rigid poses, normals, margins, penalty and force laws.
Only particle interpolation, penetration projection and displacement use
existing double shell coordinates. Both sides receive the same geometry.
"""
import ast,inspect,textwrap
from types import MethodType
import warp as wp
from newton._src.solvers.vbd.rigid_vbd_kernels import _compute_body_particle_contact_force
from deposit_cans.contact_curvature import redirect
from deposit_cans.elastic_precision import single
from deposit_cans.relative_step import double
from deposit_cans.rigid_schedule import compiled
law=redirect(_compute_body_particle_contact_force)

@wp.func
def point(p:wp.vec3d,previous:wp.vec3d,radius:float,i:int,ke:float,kd:float,mu:float,eps:float,
    shape_body:wp.array[int],bq:wp.array[wp.transform],bprev:wp.array[wp.transform],bqd:wp.array[wp.spatial_vector],
    com:wp.array[wp.vec3],shape:wp.array[int],local:wp.array[wp.vec3],vel:wp.array[wp.vec3],normal:wp.array[wp.vec3],
    margin:wp.array[float],dt:float):
    sh=shape[i];body=shape_body[sh];pose=wp.transform_identity();cm=wp.vec3(0.)
    if body>=0:pose=bq[body];cm=com[body]
    bx=wp.transform_point(pose,local[i]);n=normal[i];pad=float(0.)
    if margin.shape[0]>0:pad=margin[sh]
    depth=float(-(wp.dot(double(n),p-double(bx))-wp.float64(radius)-wp.float64(pad)))
    if depth>0.:
        dx=single(p-previous)
        if bprev:
            old=wp.transform_identity()
            if body>=0:old=bprev[body]
            bv=(bx-wp.transform_point(old,local[i]))/dt+wp.transform_vector(pose,vel[i])
        else:
            velocity=wp.spatial_vector()
            if body>=0:velocity=bqd[body]
            bv=wp.spatial_top(velocity)+wp.cross(wp.spatial_bottom(velocity),bx-wp.transform_point(pose,cm))+wp.transform_vector(pose,vel[i])
        f,h=law(depth,n,dx-bv*dt,ke,kd,mu,eps,dt)
        return f,h,bx
    return wp.vec3(0.),wp.mat33(0.),bx

@wp.func
def particle(v:int,p:wp.vec3d,previous:wp.vec3d,i:int,ke:float,kd:float,mu:float,eps:float,radius:wp.array[float],
    shape_body:wp.array[int],bq:wp.array[wp.transform],bprev:wp.array[wp.transform],bqd:wp.array[wp.spatial_vector],
    com:wp.array[wp.vec3],shape:wp.array[int],local:wp.array[wp.vec3],vel:wp.array[wp.vec3],normal:wp.array[wp.vec3],
    margin:wp.array[float],dt:float):
    f,h,bx=point(p,previous,radius[v],i,ke,kd,mu,eps,shape_body,bq,bprev,bqd,com,shape,local,vel,normal,margin,dt)
    return f,h

@wp.func
def face(i:int,corners:wp.vec3i,bary:wp.vec3,q:wp.array[wp.vec3d],previous:wp.array[wp.vec3d],radius:wp.array[float],
    ke:float,kd:float,mu:float,eps:float,shape_body:wp.array[int],bq:wp.array[wp.transform],bprev:wp.array[wp.transform],
    bqd:wp.array[wp.spatial_vector],com:wp.array[wp.vec3],shape:wp.array[int],local:wp.array[wp.vec3],
    vel:wp.array[wp.vec3],normal:wp.array[wp.vec3],margin:wp.array[float],dt:float):
    a=corners[0];p=wp.float64(bary[0])*q[a];old=wp.float64(bary[0])*previous[a];r=radius[a]
    for j in range(1,3):
        v=corners[j]
        if v>=0:p+=wp.float64(bary[j])*q[v];old+=wp.float64(bary[j])*previous[v];r=wp.max(r,radius[v])
    return point(p,old,r,i,ke,kd,mu,eps,shape_body,bq,bprev,bqd,com,shape,local,vel,normal,margin,dt)

def copy_kernel(root,*,rigid=False):
    tree=ast.parse(textwrap.dedent(inspect.getsource(root.func)));f=tree.body[0];f.name+='_precise_body'
    names={'particle_q','particle_q_prev'}if rigid else{'pos','pos_anchor','q','prev'}
    changed=0
    for arg in f.args.args:
        if arg.arg in names:
            if ast.unparse(arg.annotation)!='wp.array[wp.vec3]':raise RuntimeError('Unexpected particle annotation')
            arg.annotation=ast.parse('wp.array[wp.vec3d]',mode='eval').body;changed+=1
    if changed!=2:raise RuntimeError('Expected exactly two particle coordinate inputs')
    if rigid:
        # The native rigid kernel inlines its particle-record geometry. Replace
        # that branch with the SAME point helper used on the particle side.
        class PointBranch(ast.NodeTransformer):
            count=0
            def visit_If(self,node):
                if ast.unparse(node.test)=='corners[1] < 0':
                    self.count+=1
                    node.body=ast.parse('''particle_idx = corners[0]
if particle_idx < 0: continue
f_soft, h_soft, cp_world = precise_point(particle_q[particle_idx], particle_q_prev[particle_idx], particle_radius[particle_idx], contact_idx,
    body_particle_contact_penalty_k[contact_idx], body_particle_contact_material_kd[contact_idx], body_particle_contact_material_mu[contact_idx], friction_epsilon,
    shape_body, body_q, body_q_prev, body_qd, body_com, body_particle_contact_shape, body_particle_contact_body_pos,
    body_particle_contact_body_vel, body_particle_contact_normal, shape_margin, dt)
''').body
                    return node
                return self.generic_visit(node)
        edit=PointBranch();tree=edit.visit(tree)
        if edit.count!=1:raise RuntimeError('Unexpected inline particle-contact branch')
    ns=dict(root.func.__globals__,_eval_body_particle_contact=particle,_eval_soft_ef_contact=face,precise_point=point)
    ns,source=compiled(tree,ns,'precise-body-'+root.key)
    return ns[f.name]

def install_rigid(s):
    method=s._solve_rigid_body_iteration;name='accumulate_body_particle_contacts_per_body'
    kernel=copy_kernel(method.__func__.__globals__[name],rigid=True)
    tree=ast.parse(textwrap.dedent(inspect.getsource(method)))
    class Arguments(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            if any(k.arg=='kernel'and isinstance(k.value,ast.Name)and k.value.id==name for k in node.keywords):
                self.count+=1
                for k in node.keywords:
                    if k.arg=='inputs':
                        if ast.unparse(k.value.elts[2])!='state_in.particle_q' or ast.unparse(k.value.elts[3])!='self.particle_q_prev':raise RuntimeError('Unexpected contact argument positions')
                        k.value.elts[2]=ast.parse('self.global_refinement.precision.current',mode='eval').body
                        k.value.elts[3]=ast.parse('self.global_refinement.precision.previous',mode='eval').body
            return self.generic_visit(node)
    edit=Arguments();tree=edit.visit(tree)
    if edit.count!=1:raise RuntimeError('Unexpected rigid contact launch count')
    ns,source=compiled(tree,dict(method.__func__.__globals__,**{name:kernel}),'precise-rigid-launch')
    s._solve_rigid_body_iteration=MethodType(ns[tree.body[0].name],s)
