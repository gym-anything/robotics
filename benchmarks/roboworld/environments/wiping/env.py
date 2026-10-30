"""Persistent kitchen environment: policy actions address physical FR3 servos."""
from dataclasses import replace,asdict
import math
import threading
import numpy as np
import newton
from .config import Config
from .simulation import Simulation


def historical_production_config(**options):
    """Retain the withdrawn original configuration for explicit comparisons."""
    return replace(Config(cloth_solver='vbd',nonlinear_iterations=60,scene='kitchen',
                          soil=True,film_transport=True,wrist_cameras=True,
                          coupling='native',contact_capacity=65536,
                          native_rigid_iterations=10,native_grain_solver='vbd'),**options)


def production_config(**options):
    """Current physical settings; prior explicit configs retain split ownership.

    Historical recordings deserialize their own Config instead. Changing this
    default neither resets an existing environment nor rewrites old recordings.
    """
    if 'native_grain_solver' in options and 'native_rigid_ownership' not in options:
        options['native_rigid_ownership']='split'
    return replace(Config(
        nonlinear_iterations=240, contact_capacity=65536,
        membrane_shear=1000., membrane_lambda=1500., contact_kd=.05,
        joint_limit_ke=100000000., joint_limit_kd=20000.,
        friction_velocity=.0003, cloth_solver='vbd', scene='kitchen',
        soil=True, film_transport=True, wrist_cameras=True, coupling='native',
        native_rigid_iterations=10, native_adaptive_initial_guess=True,
        native_local_frame_z=1., native_translation_refinement=True,
        native_rigid_contact_onset_correction=True, native_grain_solver='mujoco',
        native_rigid_ownership='mujoco',
        scene_revision=4, self_contact_filter_hops=2,
        self_contact_material_exclusion=.006, native_rigid_linear_beta=100000.,
        native_contact_seed_mass_scale=100000000., native_sphere_contact_refresh=True,
        native_shape_grouped_contacts=True, native_analytic_box_contacts=True,
        native_stable_box_contacts=True, native_contact_compression_core=True,
        grain_contact_gap=.010), **options)


class WipingEnv(Simulation):
    def __init__(self,config=None):
        super().__init__(config or production_config())
        self.lock=threading.RLock();self.generation=0;self.renderer=None;self.trace_callback=None
        self.builder=self.assembly.builder;self.newton=newton;self.joints=self.assembly.robot_joints
        from .task import progress
        self.progress=progress(self);self.evaluation=self.progress.update(self)

    @property
    def state(self):return self.state_0

    def validate_targets(self,targets,ticks):
        if type(ticks) is not int or not 1<=ticks<=60000:raise ValueError('Request 1..60000 exact physics ticks')
        result=self.targets.copy()
        for name,value in targets.items():
            if name not in self.joints:raise ValueError('Unknown robot actuator: '+str(name))
            s=self.joints[name]
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
                raise ValueError('Finite numeric target required')
            if not s['lower']<=value<=s['upper']:raise ValueError('Target exceeds joint limit: '+name)
            result[s['q']]=value
        return result

    def _completed(self,sim):
        if self.tick%max(1,round(.1/self.config.dt))==0:
            self.evaluation=self.progress.update(self)
        if self.trace_callback:self.trace_callback(self)

    def act(self,targets,*,ticks,guard=None):
        with self.lock:
            if self.fault:raise RuntimeError(self.fault)
            self.targets=self.validate_targets(targets,ticks)
            try:
                for _ in range(ticks):
                    self.step(1,callback=self._completed)
                    if guard and guard(self):break
            except Exception as exc:
                self.fault=self.fault or f'Stopped at tick {self.tick}: {type(exc).__name__}: {exc}'
                raise
            self.evaluation=self.progress.update(self)
            return self.get_state()

    def get_state(self):
        from gym_anything_robotics.action_history import diagnostic_json
        with self.lock:
            q,v=self.state.body_q.numpy(),self.state.body_qd.numpy()
            jq,jv=self.state.joint_q.numpy(),self.state.joint_qd.numpy()
            p=self.state.particle_q.numpy()
            value=dict(tick=self.tick,generation=self.generation,time_s=self.tick*self.config.dt,
                ready=self.fault is None,fault=self.fault,config=asdict(self.config),
                joints={n:dict(position=float(jq[s['q']]),velocity=float(jv[s['dof']]),
                    target=float(self.targets[s['q']])) for n,s in self.joints.items()},
                bodies={n:dict(id=i,position_m=q[i,:3].tolist(),quaternion_xyzw=q[i,3:].tolist(),
                    velocity_linear_angular=v[i].tolist()) for i,n in enumerate(self.model.body_label)},
                cloths={a['name']:dict(vertices_m=p[a['first']:a['first']+a['count']].tolist(),
                    material_kg=self.transport.cloth[a['first']:a['first']+a['count']].sum(0).tolist())
                    for a in self.assembly.cloths},
                material=self.transport.ledger() if self.transport else None,evaluation=self.evaluation)
            nonfinite=[];value=diagnostic_json(value,nonfinite)
            if nonfinite:value['nonfinite_paths']=nonfinite
            return value

    def close(self):
        with self.lock:
            if self.renderer:self.renderer.close();self.renderer=None

    def active_contact_pairs(self):
        from scipy.spatial.transform import Rotation
        c=self.contacts;n=int(c.rigid_contact_count.numpy()[0])
        if n>c.rigid_contact_max:raise OverflowError('Rigid contact capacity exceeded')
        shapes=[getattr(c,'rigid_contact_shape'+str(i)).numpy()[:n] for i in (0,1)]
        q=self.state.body_q.numpy();world=[]
        for side in (0,1):
            p=getattr(c,'rigid_contact_point'+str(side)).numpy()[:n].copy()
            ids=np.asarray(self.builder.shape_body)[shapes[side]];moving=ids>=0
            if moving.any():p[moving]=q[ids[moving],:3]+Rotation.from_quat(q[ids[moving],3:]).apply(p[moving])
            world.append(p)
        distance=np.sum((world[1]-world[0])*c.rigid_contact_normal.numpy()[:n],axis=1)
        distance-=c.rigid_contact_margin0.numpy()[:n]+c.rigid_contact_margin1.numpy()[:n]
        # Multiple collision meshes on one rigid link may overlap by design;
        # likewise the bolted fixed base and its pedestal are both world shapes.
        # These are not collisions between independently moving bodies.
        body=np.asarray(self.builder.shape_body)
        active=(distance<=0)&(body[shapes[0]]!=body[shapes[1]])
        return list(set((int(a),int(b)) for a,b in zip(shapes[0][active],shapes[1][active])))
