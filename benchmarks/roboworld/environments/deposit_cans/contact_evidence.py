"""Privileged native rigid/shell contact evidence, without stepping physics.

These are simulator forces, not inferred camera observations or a fabricated
wrist sensor. Use current destination-model indices, never parent-body indices
against a reordered proxy state. Zero-force broad-phase candidates are omitted.
"""
import numpy as np
import warp as wp
from newton._src.solvers.vbd.particle_vbd_kernels import _eval_body_particle_contact,_eval_soft_ef_contact
from .precise_body import particle as precise_particle,face as precise_face
from .elastic_precision import single

@wp.kernel
def evaluate(dt:float,pos:wp.array[wp.vec3],prev:wp.array[wp.vec3],radius:wp.array[float],
    body_q:wp.array[wp.transform],body_prev:wp.array[wp.transform],body_qd:wp.array[wp.spatial_vector],
    com:wp.array[wp.vec3],eps:float,ke:wp.array[float],kd:wp.array[float],mu:wp.array[float],
    count:wp.array[int],indices:wp.array[wp.vec3i],bary:wp.array[wp.vec3],shape:wp.array[int],
    point:wp.array[wp.vec3],velocity:wp.array[wp.vec3],normal:wp.array[wp.vec3],
    margin:wp.array[float],shape_body:wp.array[int],forces:wp.array[wp.vec3],locations:wp.array[wp.vec3]):
    i=wp.tid()
    if i>=wp.min(count[0],forces.shape[0]):return
    ids=indices[i]
    if ids[1]<0:
        p=pos[ids[0]]
        f,h=_eval_body_particle_contact(ids[0],p,prev[ids[0]],i,ke[i],kd[i],mu[i],eps,radius,
            shape_body,body_q,body_prev,body_qd,com,shape,point,velocity,normal,margin,dt)
    else:
        f,h,p=_eval_soft_ef_contact(i,ids,bary[i],pos,prev,radius,ke[i],kd[i],mu[i],eps,
            shape_body,body_q,body_prev,body_qd,com,shape,point,velocity,normal,margin,dt)
    forces[i]=f;locations[i]=p

@wp.kernel
def evaluate_precise(dt:float,pos:wp.array[wp.vec3d],prev:wp.array[wp.vec3d],radius:wp.array[float],
    body_q:wp.array[wp.transform],body_prev:wp.array[wp.transform],body_qd:wp.array[wp.spatial_vector],
    com:wp.array[wp.vec3],eps:float,ke:wp.array[float],kd:wp.array[float],mu:wp.array[float],
    count:wp.array[int],indices:wp.array[wp.vec3i],bary:wp.array[wp.vec3],shape:wp.array[int],
    point:wp.array[wp.vec3],velocity:wp.array[wp.vec3],normal:wp.array[wp.vec3],
    margin:wp.array[float],shape_body:wp.array[int],forces:wp.array[wp.vec3],locations:wp.array[wp.vec3]):
    i=wp.tid()
    if i>=wp.min(count[0],forces.shape[0]):return
    ids=indices[i];p=pos[ids[0]]
    if ids[1]<0:
        f,h=precise_particle(ids[0],p,prev[ids[0]],i,ke[i],kd[i],mu[i],eps,radius,
            shape_body,body_q,body_prev,body_qd,com,shape,point,velocity,normal,margin,dt)
    else:
        f,h,body_point=precise_face(i,ids,bary[i],pos,prev,radius,ke[i],kd[i],mu[i],eps,
            shape_body,body_q,body_prev,body_qd,com,shape,point,velocity,normal,margin,dt)
        p=wp.float64(bary[i][0])*pos[ids[0]]
        for j in range(1,3):
            if ids[j]>=0:p+=wp.float64(bary[i][j])*pos[ids[j]]
    forces[i]=f;locations[i]=single(p)

class ContactEvidence:
    def __init__(self,sim):
        self.sim=sim;c=sim.solver.native if hasattr(sim.solver,'graphs') else sim.solver
        self.coupled=c;self.native=c.native;self.model=c.native.model
        self.forces=wp.zeros(sim.config.contact_capacity,dtype=wp.vec3,device=self.model.device)
        self.locations=wp.zeros_like(self.forces)
        g=self.native.global_refinement
        self.precision=g.precision if g is not None and hasattr(g.contact,'precise_previous')else None
        self.kernel=evaluate_precise if self.precision is not None else evaluate
        if g is not None and hasattr(g,'adapt_contact_reader'):g.adapt_contact_reader(self)
        self.owner=np.full(self.model.particle_count,-1,np.int32)
        for i,a in enumerate(sim.assembly.surfaces):self.owner[a['first']:a['first']+a['count']]=i
    def read(self,*,surface=None,robot_only=False,columnar=False):
        from .contact_records import columns,rows
        e=self.sim;s=self.native;m=self.model;c=self.coupled.contacts
        source=('Precise solver-coordinate endpoint contact evaluation; privileged ground truth'if self.precision is not None
            else'Current native destination contact evaluator; privileged ground truth')
        def empty():
            return columns([],[],[],np.empty((0,3),np.float32),np.empty((0,3),np.float32),source)if columnar else[]
        if c is None:return empty()
        n=int(c.soft_contact_count.numpy()[0])
        if n>self.forces.size:raise OverflowError('Contact evidence capacity exceeded')
        if not n:return empty()
        state=self.coupled.coupler.entry_state('surfaces','output') if self.coupled.coupler else e.state
        position=self.precision.current if self.precision is not None else state.particle_q
        previous=self.precision.previous if self.precision is not None else s.particle_q_prev
        wp.launch(self.kernel,n,[e.config.dt,position,previous,m.particle_radius,
            state.body_q,s.evidence_body_q_prev,state.body_qd,m.body_com,s.friction_epsilon,
            s.body_particle_contact_penalty_k,s.body_particle_contact_material_kd,s.body_particle_contact_material_mu,
            c.soft_contact_count,c.soft_contact_indices,c.soft_contact_barycentric,c.soft_contact_shape,
            c.soft_contact_body_pos,c.soft_contact_body_vel,c.soft_contact_normal,m.shape_margin,m.shape_body,
            self.forces,self.locations],device=m.device)
        f=self.forces.numpy()[:n];p=self.locations.numpy()[:n]
        if not np.isfinite(f).all() or not np.isfinite(p).all():raise FloatingPointError('Nonfinite contact evidence')
        ids=c.soft_contact_indices.numpy()[:n,0];shapes=c.soft_contact_shape.numpy()[:n]
        body=m.shape_body.numpy();selected=[];surfaces=[];labels=[];shape_labels=[]
        for i in np.flatnonzero(np.linalg.norm(f,axis=1)>1e-8):
            group=self.owner[ids[i]]
            if group<0:continue
            name=e.assembly.surfaces[group]['name'];b=int(body[shapes[i]])
            label=m.body_label[b] if b>=0 else 'world'
            if surface is not None and name!=surface:continue
            if robot_only and not label.startswith(('left/','right/')):continue
            selected.append(i);surfaces.append(name);labels.append(label);shape_labels.append(m.shape_label[int(shapes[i])])
        selected=np.asarray(selected,dtype=np.int32)
        result=columns(surfaces,labels,shape_labels,p[selected],f[selected],source)
        return result if columnar else rows(result)
    def robot_touching(self,bodies):
        """Rigid bodies among `bodies` that a robot shape touches at the current end pose.

        Uses the env's zero-gap rigid shape pairs (`active_contact_pairs`, also used by the
        backend's contact guards); detection margins are excluded there."""
        e=self.sim;owner=np.asarray(e.builder.shape_body);labels=e.builder.shape_label;wanted=set(int(b) for b in bodies)
        touched=set()
        for a,b in e.active_contact_pairs():
            for shape,other in ((a,b),(b,a)):
                if int(owner[shape]) in wanted and labels[other].startswith(('left/','right/')):touched.add(int(owner[shape]))
        return touched
