"""One exact clock; rendering and reads cannot advance it."""
import importlib.metadata
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import newton
import warp as wp
from laundry.newton15.graph import CapturedClothStep,CapturedCollision
from .assembly import Assembly
from .config import Config,resolved_contact_capacity
from .solver import Coupled

class Simulation:
    def __init__(self,config=Config(),*,robots=True,fixture=None,initial_grasp=None,initial_joint_configuration=None,solver_factory=None,robot_setup=None):
        if importlib.metadata.version('newton')!='1.5.0':raise RuntimeError('Requires Newton exactly 1.5.0')
        config=replace(config,contact_capacity=resolved_contact_capacity(config))
        self.newton=newton
        self.config=config;self.assembly=Assembly(config,robots=robots,fixture=fixture,robot_setup=robot_setup)
        b=self.builder=self.assembly.builder;b.validate_inertia_detailed=True
        self.initial_assistance=None
        if initial_joint_configuration is not None:
            if initial_grasp is not None or not robots or not isinstance(initial_joint_configuration,dict):
                raise ValueError('Explicit initial joint fixture requires robots and no competing grasp fixture')
            for name,value in initial_joint_configuration.items():
                spec=self.assembly.robot_joints.get(name)
                if spec is None or isinstance(value,bool)or not np.isfinite(value):
                    raise ValueError('Unknown or nonfinite initial fixture joint '+str(name))
                if not spec['lower']<=value<=spec['upper']:
                    raise ValueError('Initial fixture joint exceeds actual hardware limit: '+name)
                b.joint_q[spec['q']]=b.joint_target_q[spec['q']]=float(value)
            self.initial_assistance=dict(type='prepositioned_joint_fixture',
                joint_configuration=dict(initial_joint_configuration),object_fixture=fixture,
                scope='Explicit initial joints/object placement before the first physics state; no runtime pose writes or attachments')
        if initial_grasp is not None:
            # Labelled component-fixture INITIAL condition, before finalization,
            # states or any tick exist. Never exposed as an environment action.
            if not robots or initial_grasp not in ('left','right'):raise ValueError('Initial grasp requires a valid robot arm')
            from .kinematics import Kinematics
            from scipy.spatial.transform import Rotation
            can=next(a for a in self.assembly.surfaces if a.get('origin')=='counter' and a.get('target'))
            q=np.asarray(b.particle_q)[can['first']:can['first']+can['count']]
            center=(q.min(0)+q.max(0))/2
            ids=[b._gar_robots[initial_grasp].joints[name]['q'] for name in b._gar_robots[initial_grasp].groups['arm']['joints']]
            ik=Kinematics(self.assembly);fit=ik.solve(initial_grasp,center,Rotation.from_euler('xyz',[np.pi,0,0]).as_quat(),np.asarray(b.joint_q)[ids])
            if not fit['converged']:raise ValueError('Initial component-fixture IK is unreachable: '+str(fit))
            for i,value in zip(ids,fit['joints']):b.joint_q[i]=b.joint_target_q[i]=float(value)
            self.initial_assistance=dict(type='prepositioned_open_robot',arm=initial_grasp,
                target_can=can['name'],target_m=center.tolist(),ik=fit,
                scope='Initial joint configuration only, before first physics state; no object repositioning, force or attachment')
        m=self.model=b.finalize(device=config.device)
        from robots.runtime import RobotRig
        self.robot_rig=RobotRig(self.assembly.robot_setup,getattr(b,'_gar_robots',{})).bind(m)
        m.soft_contact_ke=config.contact_ke;m.soft_contact_kd=config.contact_kd
        m.soft_contact_mu=config.friction;m.soft_contact_radius=config.contact_radius;m.soft_contact_margin=config.rigid_soft_margin
        self.state_0,self.state_1=m.state(),m.state()
        newton.eval_fk(m,m.joint_q,m.joint_qd,self.state_0);self.state_1.assign(self.state_0)
        self.control=m.control();self.targets=self.control.joint_target_q.numpy().copy()
        self.rates=np.zeros(len(self.targets),np.float32)
        for s in self.assembly.robot_joints.values():self.rates[s['q']]=s['velocity_limit']
        self.pipeline=newton.CollisionPipeline(m,broad_phase='nxn',soft_contact_margin=config.rigid_soft_margin,
            enable_rigid_soft_full_surface_contact=True,soft_contact_max=config.contact_capacity)
        self.contacts=self.pipeline.contacts()
        if solver_factory is None and config.solver_revision in ('implicit-shell-v2','implicit-shell-v3','implicit-shell-v4','residual-shell-v5','bounded-shell-v6'):
            from .corrected_solver import make
            solver_factory=make
        self.solver=(solver_factory or Coupled)(self)
        self.rigid_solver=self.solver.rigid
        self._collision_states={id(s):SimpleNamespace(body_q=s.body_q,particle_q=None) for s in (self.state_0,self.state_1)} if self.solver.coupler else None
        if config.capture:
            self.solver=CapturedClothStep(self.solver);self.pipeline=CapturedCollision(self.pipeline)
        self.tick=0;self.fault=None

    @property
    def state(self):return self.state_0

    def step(self,count=1,*,callback=None,fixture_forces=None):
        if self.fault:raise RuntimeError(self.fault)
        if type(count) is not int or count<1:raise ValueError('Positive integer exact tick count required')
        c=self.config
        for _ in range(count):
            try:
                self.state_0.clear_forces()
                if fixture_forces is not None:self.state_0.particle_f.assign(fixture_forces)
                self.solver.prepare(self.state_0)
                current=self.control.joint_target_q.numpy()
                current+=np.clip(self.targets-current,-self.rates*c.dt,self.rates*c.dt)
                self.control.joint_target_q.assign(current)
                self.pipeline.collide(self._collision_states[id(self.state_0)] if self._collision_states else self.state_0,self.contacts)
                self.solver.step(self.state_0,self.state_1,self.control,self.contacts,c.dt)
                # Validate the candidate before publishing its clock/state.
                # A failed solve may have modified work buffers or state_in;
                # this is fail-closed admission, NOT a claim of rollback.
                self.check_finite(state=self.state_1)
                self.state_0,self.state_1=self.state_1,self.state_0;self.tick+=1
                if self.rigid_solver is not None:self.rigid_solver._step=self.tick
                if callback:callback(self)
            except Exception as exc:
                self.fault=f'{type(exc).__name__} after tick {self.tick}: {exc}'
                raise

    def warm_up(self,modules=()):
        """Compile and load the next tick's kernels now; advances and changes nothing.

        With CUDA graph dispatch, the first tick records the collision and solver graphs, and
        recording compiles every Warp module they launch (minutes on an empty kernel cache).
        Recording executes no kernel. This records exactly the graphs the next tick would
        record, under the keys step() looks up, so that tick launches them unchanged.
        ``modules`` (kernels launched outside the graphs) are loaded, not launched.
        """
        if self.fault:raise RuntimeError(self.fault)
        if not self.config.capture or not wp.get_device(self.config.device).is_cuda:return
        state=self._collision_states[id(self.state_0)] if self._collision_states else self.state_0
        self.pipeline.record_graph(state,self.contacts)
        self.solver.record_graph(self.state_0,self.state_1,self.control,self.contacts,self.config.dt)
        for module in modules:wp.load_module(module,device=self.config.device)

    def check_finite(self,*,state=None):
        self.solver.check()
        state=self.state_0 if state is None else state
        for field in ('particle_q','particle_qd','body_q','body_qd','joint_q','joint_qd'):
            v=getattr(state,field).numpy()
            if not np.isfinite(v).all():raise FloatingPointError('Nonfinite '+field)
            limit=10 if field in ('particle_q','body_q') else 50
            if v.size and field in ('particle_q','body_q','particle_qd','body_qd') and np.max(abs(v[:,:3]))>limit:
                i=int(np.max(abs(v[:,:3]),axis=1).argmax())
                name=self.model.body_label[i] if field.startswith('body') else 'vertex '+str(i)
                raise FloatingPointError(f'{field} outside physical envelope: {name} {v[i].tolist()}')

    def summary(self):
        q=self.state_0.particle_q.numpy();v=self.state_0.particle_qd.numpy();mass=self.model.particle_mass.numpy()
        return dict(tick=self.tick,sim_s=self.tick*self.config.dt,fault=self.fault,particles=len(q),
            particle_kinetic_j=float(.5*np.sum(mass*np.sum(v*v,axis=1))),
            max_particle_speed_m_s=float(np.linalg.norm(v,axis=1).max()),
            retention_force_on_frame_n=self.solver.retention_forces.numpy().sum(0).tolist(),
            plastic_dissipation_j=float(self.solver.plastic_dissipation.numpy().sum()),
            objects={a['name']:dict(min_m=q[a['first']:a['first']+a['count']].min(0).tolist(),
                max_m=q[a['first']:a['first']+a['count']].max(0).tolist(),mass_kg=a['mass_kg']) for a in self.assembly.surfaces})

    def close(self):
        solver=getattr(self,'solver',None)
        close=getattr(solver,'close',None)
        if close is not None:close()
