# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Bounded implicit drives in an instance-local Newton1.5 VBD copy.

Native VBD ignores joint_effort_limit. This extension clips only motor drives,
not mechanical limit reactions, and sets their derivative to zero in saturation.
A saturated drive must retain its nonzero force even though its Hessian is zero.
Optional feedforward is included INSIDE the same effort bound. No object forces,
mass modifications, joint welds or installed-engine edits are introduced.
Installed by the unified robot controller; no installed-engine globals change.
"""
import ast,hashlib,inspect,linecache,textwrap
from functools import lru_cache
from pathlib import Path
from types import MethodType
import numpy as np
import warp as wp

def compile_tree(tree,namespace,label):
    source=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    filename='<deposit-cans-bounded-drive-'+label+'>'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    namespace=dict(namespace,__name__=__name__)
    exec(compile(source,filename,'exec'),namespace)
    return namespace,source

@lru_cache(maxsize=1)
def methods():
    from newton._src.solvers.vbd import rigid_vbd_kernels as k,solver_vbd as v
    expected='0a34069f3f6bc6e7b0063117371fc70f9808751b93053dfd7b3e6d401634cf3f'
    if hashlib.sha256(Path(k.__file__).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Review bounded drives against changed Newton source')
    expected_method='ebfff7b37a260e7068586a57c20404ba08d7269905de48af9a51f9a0d4d23d87'
    if hashlib.sha256(Path(v.__file__).read_bytes()).hexdigest()!=expected_method:
        raise RuntimeError('Review bounded launch against changed Newton source')
    params=ast.parse('def f(motor_limit:wp.array[float],motor_feedforward:wp.array[float],passive_damping:wp.array[float],passive_friction:wp.array[float],motor_evaluation:wp.array[float],passive_evaluation:wp.array[float],limit_evaluation:wp.array[float]):pass').body[0].args.args
    original=k.evaluate_joint_force_hessian
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.func)));f=tree.body[0]
    f.name='evaluate_bounded_joint_force_hessian';f.args.args.extend(params)
    class Bound(ast.NodeTransformer):
        def __init__(self):self.drives=0;self.guards=0;self.limits=0
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name)and t.id=='has_drive'for t in node.targets):
                node.value=ast.BoolOp(op=ast.Or(),values=[node.value,
                    ast.parse('passive_damping[dof_idx] > 0.0 or passive_friction[dof_idx] > 0.0',mode='eval').body])
            return node
        def passive(self,velocity):
            # C1 Coulomb regularisation at1e-4 joint-coordinate units/second.
            # Sliding magnitude is the authored friction; passive damping and
            # friction dissipate energy and are NOT capped as motor effort.
            return ast.parse(f'''u = wp.clamp(({velocity}) / 1.0e-4, -1.0, 1.0)
f_scalar += passive_damping[dof_idx] * ({velocity}) + passive_friction[dof_idx] * u * (2.0 - wp.abs(u))
passive_evaluation[dof_idx] = -(passive_damping[dof_idx] * ({velocity}) + passive_friction[dof_idx] * u * (2.0 - wp.abs(u)))
H_scalar += (passive_damping[dof_idx] + 2.0 * passive_friction[dof_idx] * wp.max(1.0-wp.abs(u),0.0) / 1.0e-4) * inv_dt
''').body
        def visit_If(self,node):
            self.generic_visit(node)
            if ast.unparse(node.test)=='mode == _DRIVE_LIMIT_MODE_DRIVE':
                # All four scalar drive branches use this DOF index. Limits
                # retain their original uncapped physical stop response.
                if not any(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='H_scalar'for t in n.targets)for n in node.body):
                    raise RuntimeError('Unexpected native drive branch')
                node.body+=ast.parse('''f_scalar -= motor_feedforward[dof_idx]
bound = wp.max(motor_limit[dof_idx], 0.0)
if wp.abs(f_scalar) > bound:
    f_scalar = wp.clamp(f_scalar, -bound, bound)
    H_scalar = 0.0
motor_evaluation[dof_idx] = -f_scalar
limit_evaluation[dof_idx] = 0.0
''').body
                node.body+=self.passive('vel_err + target_vel')
                self.drives+=1
            if ast.unparse(node.test)=='mode == _DRIVE_LIMIT_MODE_LIMIT_LOWER or mode == _DRIVE_LIMIT_MODE_LIMIT_UPPER':
                value=next(n.value for n in node.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='f_scalar'for t in n.targets))
                if not isinstance(value,ast.BinOp)or not isinstance(value.right,ast.BinOp):raise RuntimeError('Unexpected native limit force')
                velocity=ast.unparse(value.right.right)
                node.body+=ast.parse('motor_evaluation[dof_idx] = 0.0\nlimit_evaluation[dof_idx] = -f_scalar').body
                node.body+=self.passive(velocity);self.limits+=1
            if ast.unparse(node.test)=='H_scalar > 0.0':
                node.test=ast.parse('H_scalar > 0.0 or f_scalar != 0.0',mode='eval').body
                self.guards+=1
            return node
    transform=Bound();tree=transform.visit(tree)
    if (transform.drives,transform.guards,transform.limits)!=(4,4,4):raise RuntimeError('Incomplete bounded-drive coverage')
    ns,source=compile_tree(tree,original.func.__globals__,'force')
    force=ns[f.name];sources={'force':source}
    original=k.solve_rigid_body;tree=ast.parse(textwrap.dedent(inspect.getsource(original.func)));f=tree.body[0]
    f.name='solve_bounded_rigid_body';f.args.args[-1:-1]=params
    count=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id=='evaluate_joint_force_hessian':
            node.args.extend([ast.Name(id=arg.arg,ctx=ast.Load())for arg in params]);count+=1
    if count!=1:raise RuntimeError('Unexpected joint evaluator call graph')
    ns,source=compile_tree(tree,dict(original.func.__globals__,evaluate_joint_force_hessian=force),'kernel')
    kernel=ns[f.name];sources['kernel']=source
    original=v.SolverVBD._solve_rigid_body_iteration
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)));f=tree.body[0];count=0
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call):continue
        keys={item.arg:item for item in node.keywords}
        if 'kernel' in keys and ast.unparse(keys['kernel'].value)=='solve_rigid_body':
            inputs=keys['inputs'].value
            if not isinstance(inputs,ast.List):raise RuntimeError('Unexpected native launch inputs')
            inputs.elts.extend(ast.parse('[self.model.joint_effort_limit,self.motor_feedforward,self.model.joint_damping,self.model.joint_friction,self.motor_evaluation,self.passive_evaluation,self.limit_evaluation]',mode='eval').body.elts);count+=1
    if count!=1:raise RuntimeError('Unexpected rigid solve launch count')
    ns,source=compile_tree(tree,dict(original.__globals__,solve_rigid_body=kernel),'method')
    sources['method']=source
    return ns[f.name],sources

def install(solver):
    # Reject unsupported armature; preserve authored passive coefficients.
    m=solver.model
    if np.any(m.joint_armature.numpy()!=0):
        raise ValueError('Bounded VBD candidate cannot discard joint armature')
    for name in ('joint_friction','joint_damping'):
        values=getattr(m,name).numpy()
        if not np.isfinite(values).all()or np.any(values<0):raise ValueError('Invalid passive joint coefficient: '+name)
    limit=m.joint_effort_limit.numpy()
    driven=(m.joint_target_ke.numpy()!=0)|(m.joint_target_kd.numpy()!=0)
    if np.any(~np.isfinite(limit[driven]))or np.any(limit[driven]<0):raise ValueError('Finite nonnegative drive effort bounds required')
    method,sources=methods();solver.motor_feedforward=wp.zeros(m.joint_dof_count,dtype=float,device=m.device)
    for name in ('motor_evaluation','passive_evaluation','limit_evaluation'):
        setattr(solver,name,wp.zeros(m.joint_dof_count,dtype=float,device=m.device))
    solver._solve_rigid_body_iteration=MethodType(method,solver)
    return sources
