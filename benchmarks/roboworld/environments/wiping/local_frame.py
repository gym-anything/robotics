"""Precision-preserving numerical frame for the native cloth/prop solve.

Only coordinates change. Public state, gravity, material laws, collision radii,
robot effort and simulated time do not. Local arrays retain sub-world-ULP
corrections between timesteps; externally edited positions invalidate their
cached value. All caches are checkpoint state, not hidden grasp constraints.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import warp as wp


@wp.kernel
def import_points(world:wp.array[wp.vec3],published:wp.array[wp.vec3],
                  local:wp.array[wp.vec3],valid:wp.array[int],
                  recent_world:wp.array[wp.vec3],recent_local:wp.array[wp.vec3],
                  recent_valid:wp.array[int],use_recent:int,origin:wp.vec3):
    i=wp.tid();x=world[i]
    if valid[i]!=0 and x==published[i]:return
    if use_recent!=0 and recent_valid[i]!=0 and x==recent_world[i]:local[i]=recent_local[i]
    else:local[i]=x-origin


@wp.kernel
def export_points(local:wp.array[wp.vec3],world:wp.array[wp.vec3],
                  published:wp.array[wp.vec3],valid:wp.array[int],origin:wp.vec3):
    i=wp.tid();x=local[i]+origin;world[i]=x;published[i]=x;valid[i]=1


@wp.kernel
def import_poses(world:wp.array[wp.transform],published:wp.array[wp.transform],
                 local:wp.array[wp.transform],valid:wp.array[int],
                 recent_world:wp.array[wp.transform],recent_local:wp.array[wp.transform],
                 recent_valid:wp.array[int],use_recent:int,origin:wp.vec3):
    i=wp.tid();x=world[i]
    if valid[i]!=0 and x==published[i]:return
    if use_recent!=0 and recent_valid[i]!=0 and x==recent_world[i]:local[i]=recent_local[i]
    else:local[i]=wp.transform(wp.transform_get_translation(x)-origin,wp.transform_get_rotation(x))


@wp.kernel
def export_poses(local:wp.array[wp.transform],world:wp.array[wp.transform],
                 published:wp.array[wp.transform],valid:wp.array[int],origin:wp.vec3):
    i=wp.tid();q=local[i]
    x=wp.transform(wp.transform_get_translation(q)+origin,wp.transform_get_rotation(q))
    world[i]=x;published[i]=x;valid[i]=1


@wp.kernel
def static_poses(source:wp.array[wp.transform],dest:wp.array[wp.transform],
                 parent:wp.array[int],shift:wp.vec3):
    i=wp.tid();q=source[i]
    if parent[i]<0:q=wp.transform(wp.transform_get_translation(q)+shift,wp.transform_get_rotation(q))
    dest[i]=q


@wp.kernel
def static_contacts(source:wp.array[wp.vec3],dest:wp.array[wp.vec3],
                    count:wp.array[int],shape:wp.array[int],body:wp.array[int],shift:wp.vec3):
    i=wp.tid();x=source[i]
    if i<count[0] and shape[i]>=0 and body[shape[i]]<0:x+=shift
    dest[i]=x


class LocalSolveFrame:
    def __init__(self,solver,z):
        self.solver=solver;self.device=solver.device;self.origin=wp.vec3(0.,0.,z)
        self.slots=[];self.by_pointer={};self.active=False
        m=solver.model
        self.recent={}
        for name,value in (('particles',m.particle_q),('bodies',m.body_q)):
            self.recent[name]=SimpleNamespace(local=wp.zeros_like(value),published=wp.zeros_like(value),
                                              valid=wp.zeros(len(value),dtype=int,device=self.device))
        for name in ('particle_q_prev','inertia','pos_prev_collision_detection'):
            self.register(solver,name,'points','particles')
        for name in ('body_q_prev','body_inertia_q','_coupling_body_q_prev_snapshot','previous_for_record'):
            self.register(solver,name,'poses','bodies')
        self.register(m,'body_q','poses')
        self.register(m,'shape_transform','static_poses',parent=m.shape_body)
        self.register(m,'joint_X_p','static_poses',parent=m.joint_parent)

    def register(self,obj,name,kind,recent=None,**extra):
        value=getattr(obj,name,None)
        if value is None or not len(value):return
        if value.ptr in self.by_pointer:
            slot=self.by_pointer[value.ptr]
            if slot.kind!=kind:raise ValueError('Aliased arrays have different frame semantics')
            if (obj,name) not in slot.bindings:slot.bindings.append((obj,name))
            return
        if self.device.is_capturing:raise RuntimeError('Register frame arrays before graph capture')
        slot=SimpleNamespace(world=value,local=wp.zeros_like(value),published=wp.zeros_like(value),
            valid=wp.zeros(len(value),dtype=int,device=self.device),kind=kind,recent=recent,
            bindings=[(obj,name)],extra=extra)
        self.slots.append(slot);self.by_pointer[value.ptr]=slot

    def register_state(self,state):
        if state is None:return
        self.register(state,'particle_q','points','particles')
        self.register(state,'body_q','poses','bodies')

    def register_contacts(self,c):
        m=self.solver.model
        for name,count,shape in (
            ('soft_contact_body_pos',c.soft_contact_count,c.soft_contact_shape),
            ('rigid_contact_point0',c.rigid_contact_count,c.rigid_contact_shape0),
            ('rigid_contact_point1',c.rigid_contact_count,c.rigid_contact_shape1)):
            self.register(c,name,'static_contacts',count=count,shape=shape,body=m.shape_body)

    def _transfer(self,s,enter):
        source,dest=(s.world,s.local) if enter else (s.local,s.world)
        shift=-self.origin if enter else self.origin
        if s.kind=='static_poses':
            wp.launch(static_poses,len(source),[source,dest,s.extra['parent'],shift],device=self.device)
        elif s.kind=='static_contacts':
            wp.launch(static_contacts,len(source),[source,dest,s.extra['count'],s.extra['shape'],s.extra['body'],shift],device=self.device)
        elif enter:
            recent=self.recent[s.recent] if s.recent else s
            kernel=import_points if s.kind=='points' else import_poses
            wp.launch(kernel,len(source),[source,s.published,dest,s.valid,
                recent.published,recent.local,recent.valid,int(s.recent is not None),self.origin],device=self.device)
        else:
            kernel=export_points if s.kind=='points' else export_poses
            wp.launch(kernel,len(source),[source,dest,s.published,s.valid,self.origin],device=self.device)

    @contextmanager
    def scope(self):
        if self.active:raise RuntimeError('Nested numerical frame')
        self.active=True
        for s in self.slots:
            self._transfer(s,True)
            for obj,name in s.bindings:setattr(obj,name,s.local)
        try:yield
        finally:
            for s in self.slots:
                self._transfer(s,False)
                for obj,name in s.bindings:setattr(obj,name,s.world)
            self.active=False

    def remember(self,state):
        if not self.active:raise RuntimeError('Remember local state inside its frame')
        for name,field in (('particles','particle_q'),('bodies','body_q')):
            source=getattr(state,field);cache=self.recent[name]
            if source is None or not len(source):continue
            wp.copy(cache.local,source)
            kernel=export_points if name=='particles' else export_poses
            wp.launch(kernel,len(source),[source,cache.published,cache.published,cache.valid,self.origin],device=self.device)

    def snapshot_arrays(self):
        result={}
        for i,s in enumerate(self.slots):
            for name in ('local','published','valid'):result[f'slot_{i}/{name}']=getattr(s,name)
        for key,cache in self.recent.items():
            for name in ('local','published','valid'):result[f'recent_{key}/{name}']=getattr(cache,name)
        return result
