"""Persistent candidate environment. Reads and renders never step physics."""
from dataclasses import asdict,replace
import math
import threading
import numpy as np
from .config import Config
from .simulation import Simulation
from . import snapshot as _checkpoint_source  # Freeze package identity at import, before any model.

def candidate_config(**options):
    # Historical public name retained; the v6/scene7 scope is documented in the environment's report.
    revision=options.get('solver_revision','bounded-shell-v6')
    residual=revision=='residual-shell-v5'
    return replace(Config(scene_revision=7 if revision in ('residual-shell-v5','bounded-shell-v6') else 5,solver_revision=revision,capture=not residual,
        work_shortcuts=revision=='bounded-shell-v6',
        shell_numerics='global',global_merit='work',dt=1/240,linear_maxiter=64,
        global_stride=8,rigid_soft_margin=.030,rigid_compression_reference=True,bag_local_contact=True,
        global_line_trials=12,force_only_evaluation=True,tiled_search_reductions=True,
        elastic_precision=True,proxy_relaxation=.25,coupling_iterations=4),**options)

class DepositCansEnv(Simulation):
    def __init__(self,config=None,*,solver_factory=None,initial_grasp=None,initial_joint_configuration=None,fixture=None,robot_setup=None):
        super().__init__(config or candidate_config(),solver_factory=solver_factory,initial_grasp=initial_grasp,
            initial_joint_configuration=initial_joint_configuration,fixture=fixture,robot_setup=robot_setup)
        self.lock=threading.RLock();self.generation=0;self.renderer=None;self.trace_callback=None
        self.joints=self.assembly.robot_joints
        # Force-query output buffers otherwise grow on the first teacher trace,
        # breaking a tick-zero checkpoint's cache layout. This empty query
        # initializes diagnostic storage only; it advances zero physics steps.
        c=self.solver.native if hasattr(self.solver,'graphs') else self.solver
        s=c.native;poses=self.state.body_q;previous=s.evidence_body_q_prev
        if c.coupler:
            poses=c.coupler.entry_state('surfaces','output').body_q
        s.collect_rigid_contact_forces(poses,previous,
            c.contacts if c.coupler else self.contacts,self.config.dt)
        from .task import Progress
        self.progress=Progress(self);self.evaluation=self.progress.update(self)
        if not self.evaluation['physical_geometry_valid']:
            raise ValueError('Initial physical bag/can geometry intersects')
    def warm_up(self):
        # Kernels a command runs outside the step graphs: the drive feedforward (solver.prepare)
        # and the action trace's self-contact and residual records. Loaded, not launched.
        from . import self_contact_evidence,trace,unified_candidate
        super().warm_up(modules=(unified_candidate,self_contact_evidence,trace))
    def validate_targets(self,targets,ticks):
        if type(ticks) is not int or not 1<=ticks<=60000:raise ValueError('Request 1..60000 exact physics ticks')
        if not isinstance(targets,dict):raise TypeError('Targets must be a dictionary')
        result=self.targets.copy()
        for name,value in targets.items():
            if name not in self.joints:raise ValueError('Unknown robot actuator: '+str(name))
            spec=self.joints[name]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('Finite numeric joint target required')
            if not spec['lower']<=value<=spec['upper']:raise ValueError('Joint limit exceeded: '+name)
            result[spec['q']]=value
        return result
    def _completed(self,sim):
        # Evaluate at every completed substep so transient loss cannot be
        # hidden between API requests. Reads return the stored evaluation.
        self.evaluation=self.progress.update(self)
        geometry_error=not self.evaluation['physical_geometry_valid']
        if geometry_error:self.fault='Invalid physical bag/can surface path at tick '+str(self.tick)
        if self.trace_callback:self.trace_callback(self)
        if geometry_error:raise FloatingPointError(self.fault)
    def act(self,targets,*,ticks,guard=None,grip_commands=None):
        with self.lock:
            if self.fault:raise RuntimeError(self.fault)
            validated=self.validate_targets(targets,ticks)
            from .gripper import validate,apply
            commands=validate(self,grip_commands)
            apply(self,targets,commands)
            self.targets=validated
            for _ in range(ticks):
                self.step(1,callback=self._completed)
                if guard and guard(self):break
            return self.get_state()
    def get_state(self):
        from gym_anything_robotics.action_history import diagnostic_json
        from .gripper import state as gripper_state
        with self.lock:
            q=self.state.body_q.numpy();v=self.state.body_qd.numpy()
            jq=self.state.joint_q.numpy();jv=self.state.joint_qd.numpy()
            p=self.state.particle_q.numpy();pv=self.state.particle_qd.numpy()
            value=dict(tick=self.tick,generation=self.generation,time_s=self.tick*self.config.dt,
                ready=self.fault is None,fault=self.fault,config=asdict(self.config),
                grippers=gripper_state(self),
                joints={n:dict(position=float(jq[s['q']]),velocity=float(jv[s['dof']]),
                    target=float(self.targets[s['q']])) for n,s in self.joints.items()},
                bodies={n:dict(id=i,position_m=q[i,:3].tolist(),quaternion_xyzw=q[i,3:].tolist(),
                    velocity_linear_angular=v[i].tolist()) for i,n in enumerate(self.model.body_label)},
                surfaces={a['name']:dict(first_particle=a['first'],vertices_m=p[a['first']:a['first']+a['count']].tolist(),
                    velocities_m_s=pv[a['first']:a['first']+a['count']].tolist()) for a in self.assembly.surfaces},
                evaluation=self.evaluation,qualification='Not yet qualified for benchmark release')
            solver=self.solver.native if hasattr(self.solver,'graphs')else self.solver
            if hasattr(solver,'residual_solver'):value['solver_convergence']=solver.residual_solver.status()
            elif hasattr(solver,'bounded_solver'):value['solver_convergence']=solver.bounded_solver.status()
            nonfinite=[];value=diagnostic_json(value,nonfinite)
            if nonfinite:value['nonfinite_paths']=nonfinite
            return value
    def active_contact_pairs(self):
        """World-model rigid candidates evaluated at the current end pose."""
        from scipy.spatial.transform import Rotation
        c=self.contacts;n=int(c.rigid_contact_count.numpy()[0])
        if n>c.rigid_contact_max:raise OverflowError('Rigid contact capacity exceeded')
        shapes=[getattr(c,'rigid_contact_shape'+str(i)).numpy()[:n]for i in (0,1)]
        q=self.state.body_q.numpy();body=np.asarray(self.builder.shape_body);world=[]
        for side in (0,1):
            p=getattr(c,'rigid_contact_point'+str(side)).numpy()[:n].copy()
            ids=body[shapes[side]];moving=ids>=0
            if moving.any():p[moving]=q[ids[moving],:3]+Rotation.from_quat(q[ids[moving],3:]).apply(p[moving])
            world.append(p)
        distance=np.sum((world[1]-world[0])*c.rigid_contact_normal.numpy()[:n],axis=1)
        distance-=c.rigid_contact_margin0.numpy()[:n]+c.rigid_contact_margin1.numpy()[:n]
        active=(distance<=0)&(body[shapes[0]]!=body[shapes[1]])
        return list(set((int(a),int(b))for a,b in zip(shapes[0][active],shapes[1][active])))
    def close(self):
        with self.lock:
            try:
                if self.renderer:self.renderer.close()
            finally:
                self.renderer=None
                super().close()
