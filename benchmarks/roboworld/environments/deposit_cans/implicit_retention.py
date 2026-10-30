"""Same-force correction: solve installed bag/frame springs implicitly.

The old prepare() applies last-state spring reactions as explicit external body
forces and leaves anchors at the start-of-step frame pose. Here their SAME
equal/opposite spring force and PSD body tangent join each rigid local solve,
then anchors follow that iterate before the deformable solve. No grasp or new
physical constraint is added. Original12 installed springs,100kN/m,remain.
"""
import ast,inspect,linecache,textwrap
from types import MethodType
import warp as wp
from .relative_step import double

@wp.kernel
def accumulate(group:wp.array[int],pins:wp.array[int],bodies:wp.array[int],local:wp.array[wp.vec3],
    stiffness:float,bq:wp.array[wp.transform],com:wp.array[wp.vec3],q:wp.array[wp.vec3],
    force:wp.array[wp.vec3],torque:wp.array[wp.vec3],Hll:wp.array[wp.mat33],
    Hal:wp.array[wp.mat33],Haa:wp.array[wp.mat33]):
    i=wp.tid();body=group[i]
    if body<0:return
    f=wp.vec3(0.);t=wp.vec3(0.);ll=wp.mat33(0.);al=wp.mat33(0.);aa=wp.mat33(0.)
    for j in range(pins.shape[0]):
        if bodies[j]==body:
            x=wp.transform_point(bq[body],local[j]);r=x-wp.transform_point(bq[body],com[body])
            value=stiffness*(q[pins[j]]-x);R=wp.skew(r)
            f+=value;t+=wp.cross(r,value);ll+=stiffness*wp.identity(3,float)
            al+=stiffness*R;aa+=stiffness*wp.transpose(R)*R
    force[body]+=f;torque[body]+=t;Hll[body]+=ll;Hal[body]+=al;Haa[body]+=aa

@wp.kernel
def follow(anchors:wp.array[int],bodies:wp.array[int],local:wp.array[wp.vec3],
           bq:wp.array[wp.transform],q:wp.array[wp.vec3],precise:wp.array[wp.vec3d]):
    i=wp.tid();x=wp.transform_point(bq[bodies[i]],local[i]);q[anchors[i]]=x;precise[anchors[i]]=double(x)

@wp.kernel
def record(pins:wp.array[int],bodies:wp.array[int],local:wp.array[wp.vec3],stiffness:float,
           bq:wp.array[wp.transform],q:wp.array[wp.vec3],out:wp.array[wp.vec3]):
    i=wp.tid();out[i]=stiffness*(q[pins[i]]-wp.transform_point(bq[bodies[i]],local[i]))

class ImplicitRetention:
    def __init__(self,coupled):
        self.c=coupled;self.s=coupled.native
        if not coupled.monolithic or coupled.coupler:raise ValueError('Unified isolated candidate only')
        g=self.s.global_refinement
        if not g.precision:raise ValueError('Precise coordinates required')
        self.s._implicit_retention=self
        original=self.s._solve_rigid_body_iteration
        tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
        class Insert(ast.NodeTransformer):
            count=0
            def visit_Expr(self,node):
                call=node.value
                if isinstance(call,ast.Call)and any(k.arg=='kernel'and isinstance(k.value,ast.Name)
                    and k.value.id=='solve_rigid_body'for k in call.keywords):
                    self.count+=1
                    return ast.parse('self._implicit_retention.accumulate(color_group,state_in)').body+[node]
                return self.generic_visit(node)
        edit=Insert();tree=edit.visit(tree)
        if edit.count!=1:raise ValueError(f'Unexpected native rigid solve sites: {edit.count}')
        source=ast.unparse(ast.fix_missing_locations(tree))+'\n';filename='<deposit-cans-implicit-retention>'
        linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
        namespace=dict(original.__func__.__globals__);exec(compile(source,filename,'exec'),namespace)
        self.s._solve_rigid_body_iteration=MethodType(namespace[tree.body[0].name],self.s)
        old_particle=self.s._solve_particle_iteration
        def particle(s,state_in,state_out,contacts,dt,iteration):
            self.follow(state_in)
            old_particle(state_in,state_out,contacts,dt,iteration)
            if iteration==s.iterations-1:self.record(state_in)
        self.s._solve_particle_iteration=MethodType(particle,self.s)
        def prepare(c,state):
            if c.motor_controller:c.motor_controller.prepare(state)
            # No explicit spring reaction is added to state.body_f.
            self.follow(state)
        coupled.prepare=MethodType(prepare,coupled)
    def follow(self,state):
        c=self.c;g=self.s.global_refinement
        wp.launch(follow,c.anchor_ids.size,[c.anchor_ids,c.pin_bodies,c.pin_local,state.body_q,
                  state.particle_q,g.precision.current],device=c.device)
    def accumulate(self,group,state):
        c=self.c;s=self.s;m=s.model
        wp.launch(accumulate,group.size,[group,c.pin_particles,c.pin_bodies,c.pin_local,c.retention_stiffness,
            state.body_q,m.body_com,state.particle_q,s.body_forces,s.body_torques,
            s.body_hessian_ll,s.body_hessian_al,s.body_hessian_aa],device=c.device)
    def record(self,state):
        c=self.c
        wp.launch(record,c.pin_particles.size,[c.pin_particles,c.pin_bodies,c.pin_local,c.retention_stiffness,
            state.body_q,state.particle_q,c.retention_forces],device=c.device)
