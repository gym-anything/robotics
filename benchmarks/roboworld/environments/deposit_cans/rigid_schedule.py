# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Instance-local batching of commuting Newton 1.5 rigid updates.

Preserve ascending-body Gauss-Seidel order on EVERY contact/joint edge.
Independent bodies can share a wave because their local solves do not read
each other's poses. Particle positions and duals stay fixed within this loop.
The dependency plan is rebuilt from the current candidate contacts on every
iteration. No force, material, contact, iteration or physical clock is removed.

Source-hash-guarded scheduling transformation used by corrected_solver.
Never edits installed Newton; environment-level acceptance remains separate.
"""
import ast
from functools import lru_cache
import hashlib
import inspect
import linecache
from pathlib import Path
import textwrap
from types import MethodType
import warp as wp

# This private execution path performs forward simulation, not autodiff.
# Avoid compiling unused reverse-mode kernels for the large native body solve.
# No fast-math, arithmetic, contacts or iteration count is changed.
wp.set_module_options({'enable_backward':False})

@wp.kernel
def plan(n:int,counts:wp.array[int],indices:wp.array[int],capacity:int,
         s0:wp.array[int],s1:wp.array[int],shape_body:wp.array[int],
         parents:wp.array[int],children:wp.array[int],enabled:wp.array[bool],
         levels:wp.array[int],maximum:wp.array[int],current:wp.array[int],condition:wp.array[int]):
    # One tiny serial DAG traversal. Orient every edge by the reference order.
    highest=int(-1)
    for b in range(n):
        level=int(0)
        for j in range(wp.min(counts[b],capacity)):
            c=indices[b*capacity+j]
            a=int(-1)
            x=int(-1);y=int(-1)
            if s0[c]>=0:x=shape_body[s0[c]]
            if s1[c]>=0:y=shape_body[s1[c]]
            if x==b:a=y
            elif y==b:a=x
            if a>=0 and a<b:level=wp.max(level,levels[a]+1)
        for j in range(parents.shape[0]):
            if enabled[j]:
                a=int(-1)
                if parents[j]==b:a=children[j]
                elif children[j]==b:a=parents[j]
                if a>=0 and a<b:level=wp.max(level,levels[a]+1)
        levels[b]=level;highest=wp.max(highest,level)
    maximum[0]=highest;current[0]=0;condition[0]=int(n>0)

@wp.kernel
def choose(levels:wp.array[int],current:wp.array[int],group:wp.array[int]):
    b=wp.tid();group[b]=-1
    if levels[b]==current[0]:group[b]=b

@wp.kernel
def choose_fixed(levels:wp.array[int],wave:int,group:wp.array[int]):
    b=wp.tid();group[b]=-1
    if levels[b]==wave:group[b]=b

@wp.kernel
def advance(current:wp.array[int],maximum:wp.array[int],condition:wp.array[int]):
    current[0]+=1;condition[0]=int(current[0]<=maximum[0])

def compiled(tree,namespace,label):
    source=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    filename='<deposit-cans-rigid-schedule-'+label+'>'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    namespace=dict(namespace,__name__=__name__)
    exec(compile(source,filename,'exec'),namespace)
    return namespace,source

@lru_cache(maxsize=4)
def method(bounded=False,fixed_waves=False):
    from newton._src.solvers.vbd import rigid_vbd_kernels as kernels,solver_vbd as module
    hashes={kernels:'0a34069f3f6bc6e7b0063117371fc70f9808751b93053dfd7b3e6d401634cf3f',
            module:'ebfff7b37a260e7068586a57c20404ba08d7269905de48af9a51f9a0d4d23d87'}
    for item,expected in hashes.items():
        if hashlib.sha256(Path(item.__file__).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Review scheduling against changed Newton source')
    bounded_method=None
    if bounded:
        from .bounded_drive import methods
        bounded_method,_=methods()
    replacements={};sources={}
    for name,variable in [('accumulate_body_body_contacts_per_body','body_id'),
                          ('accumulate_body_particle_contacts_per_body','body_id'),
                          ('solve_rigid_body','body_index')]:
        original=(bounded_method.__globals__[name] if bounded and name=='solve_rigid_body'
            else getattr(kernels,name))
        tree=ast.parse(textwrap.dedent(inspect.getsource(original.func)))
        f=tree.body[0];f.name=name+'_scheduled'
        assignments=[i for i,node in enumerate(f.body)if isinstance(node,ast.Assign)
            and any(isinstance(t,ast.Name)and t.id==variable for t in node.targets)]
        if len(assignments)!=1:raise RuntimeError('Unexpected body-ID assignment')
        f.body[assignments[0]+1:assignments[0]+1]=ast.parse(f'if {variable} < 0: return').body
        namespace,source=compiled(tree,original.func.__globals__,name)
        replacements[name]=namespace[f.name];sources[name]=source
    original=bounded_method or module.SolverVBD._solve_rigid_body_iteration
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)));f=tree.body[0]
    loops=[(i,node)for i,node in enumerate(f.body)if isinstance(node,ast.For)
        and isinstance(node.target,ast.Name)and node.target.id=='color']
    if len(loops)!=1:raise RuntimeError('Unexpected rigid color loop')
    i,loop=loops[0]
    if ast.unparse(loop.body[0])!='color_group = body_color_groups[color]':
        raise RuntimeError('Unexpected color assignment')
    setup=ast.parse('self._rigid_schedule.prepare(contacts)').body
    body=ast.parse('def scheduled_wave():\n    self._rigid_schedule.choose()\n    color_group = self._rigid_schedule.group\n').body[0]
    body.body+=loop.body[1:]+ast.parse('self._rigid_schedule.advance()').body
    invocation=ast.parse('wp.capture_while(self._rigid_schedule.condition, scheduled_wave)').body
    f.body[i:i+1]=setup+[body]+invocation
    if fixed_waves:
        # Same dependency levels and ascending reference order. Execute empty
        # masked waves up to body_count instead of nested CUDA conditional
        # graphs. No body is omitted or visited twice; empty waves write nothing.
        fixed=ast.parse('for wave in range(model.body_count):\n    self._rigid_schedule.choose_fixed(wave)\n    color_group = self._rigid_schedule.group\n').body[0]
        fixed.body+=loop.body[1:]
        f.body[i:i+len(setup)+2]=setup+[fixed]
    namespace,source=compiled(tree,dict(original.__globals__,**replacements),'iteration')
    sources['iteration']=source
    return namespace[f.name],sources

class Schedule:
    def __init__(self,solver):
        self.s=solver;m=solver.model;self.device=m.device
        self.group=wp.empty(m.body_count,dtype=int,device=m.device)
        self.levels=wp.zeros(m.body_count,dtype=int,device=m.device)
        self.maximum=wp.zeros(1,dtype=int,device=m.device)
        self.current=wp.zeros(1,dtype=int,device=m.device)
        self.condition=wp.zeros(1,dtype=int,device=m.device)
    def prepare(self,contacts):
        if contacts is None:raise ValueError('This diagnostic requires a contact buffer')
        s=self.s;m=s.model
        wp.launch(plan,1,[m.body_count,s.body_body_contact_counts,s.body_body_contact_indices,
            s.body_body_contact_buffer_pre_alloc,contacts.rigid_contact_shape0,contacts.rigid_contact_shape1,
            m.shape_body,m.joint_parent,m.joint_child,m.joint_enabled,
            self.levels,self.maximum,self.current,self.condition],device=self.device)
    def choose(self):wp.launch(choose,self.group.size,[self.levels,self.current,self.group],device=self.device)
    def choose_fixed(self,wave):wp.launch(choose_fixed,self.group.size,[self.levels,wave,self.group],device=self.device)
    def advance(self):wp.launch(advance,1,[self.current,self.maximum,self.condition],device=self.device)

def install(solver,*,fixed_waves=False):
    original=solver._solve_rigid_body_iteration
    replacement,sources=method(hasattr(solver,'motor_feedforward'),fixed_waves);solver._rigid_schedule=Schedule(solver)
    solver._rigid_schedule.mode='fixed_masked'if fixed_waves else'conditional_masked'
    solver._solve_rigid_body_iteration=MethodType(replacement,solver)
    return original,sources
