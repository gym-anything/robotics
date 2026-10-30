"""Instance-local parallel reduction of native rigid/soft contact records.

Newton1.5 uses four strided threads per body. A frame with many full-surface
contacts serializes thousands of evaluations there. This uses128 lanes, retaining
every record and force/Hessian law. Floating reduction order changes; audit the
same query explicitly, then compare physical trajectories. No installed edits.
"""
import ast,inspect,textwrap,time
from types import MethodType
import numpy as np
import warp as wp
from .rigid_schedule import compiled

def methods(solver):
    method=solver._solve_rigid_body_iteration
    kernel=method.__func__.__globals__['accumulate_body_particle_contacts_per_body']
    tree=ast.parse(textwrap.dedent(inspect.getsource(kernel.func)))
    tree.body[0].name='soft_contacts_128'
    class Lanes(ast.NodeTransformer):
        def visit_Name(self,node):
            return ast.copy_location(ast.Constant(value=128),node)if node.id=='_NUM_CONTACT_THREADS_PER_BODY'else node
    tree=Lanes().visit(tree)
    namespace,kernel_source=compiled(tree,kernel.func.__globals__,'soft128-kernel')
    parallel=namespace['soft_contacts_128']
    tree=ast.parse(textwrap.dedent(inspect.getsource(method)))
    class Launch(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            if any(k.arg=='kernel'and isinstance(k.value,ast.Name)and k.value.id=='accumulate_body_particle_contacts_per_body'for k in node.keywords):
                self.count+=1
                for k in node.keywords:
                    if k.arg=='dim':k.value=Lanes().visit(k.value)
            return self.generic_visit(node)
    edit=Launch();tree=edit.visit(tree)
    if edit.count!=1:raise ValueError('Unexpected rigid/soft accumulation site')
    namespace,method_source=compiled(tree,dict(method.__func__.__globals__,accumulate_body_particle_contacts_per_body=parallel),'soft128-method')
    return namespace[tree.body[0].name],kernel,parallel,kernel_source,method_source

def install(solver):
    method,old,new,_,_=methods(solver)
    solver._solve_rigid_body_iteration=MethodType(method,solver)
    solver.contact_reduction_lanes=128
