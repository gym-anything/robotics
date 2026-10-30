"""Candidate GAR adapter. Not registered as a qualified benchmark yet."""
import base64
from collections import deque
from dataclasses import replace
import math
from pathlib import Path
import uuid
import numpy as np
from scipy.spatial.transform import Rotation
from .base import RoboticsBackend
from ..actions import NamedJointTarget,Gripper,CartesianPose,CartesianDelta,BimanualCartesian
from ..errors import UnsupportedActionError,UnsupportedConfigurationError,BackendProtocolError,ActionRejectedError
from ..types import BackendStep,CameraSpec,Capabilities,ImageFrame,Observation
from ..policy_access import PolicyAccess

class NewtonDepositCansBackend(RoboticsBackend):
    def __init__(self,*,robot='dual-fr3',config=None,device=None,variant=None,contract=None,
                 cameras=('left_wrist_rgb','right_wrist_rgb'),output_dir=None,renderer_options=None,recording=True):
        from deposit_cans.env import DepositCansEnv,candidate_config
        from deposit_cans.config import Config
        from deposit_cans.variants import resolve
        from deposit_cans.kinematics import Kinematics
        from deposit_cans.catalog import Catalog
        from deposit_cans.trace import Trace
        from whiteboard.storage import Store
        from robots.deployment import deposit_setup
        self.robot_setup=deposit_setup(robot)
        self.robot_id=robot if isinstance(robot,str) else self.robot_setup.id
        if config is None:c=candidate_config()
        elif isinstance(config,dict):c=candidate_config(**config)
        elif isinstance(config,Config):c=config
        else:raise TypeError('config must be a Config or dictionary')
        if device is not None:
            if isinstance(config,dict) and 'device' in config:raise ValueError('Device supplied twice')
            c=replace(c,device=device)
        # contract: the task contract selector (reference only: 'v1', the default, or
        # deposit-tasks-v2, the eight-task catalogue's entry). Without it the default contract applies.
        selected=resolve(variant or 'reference',c.seed,contract)
        for k,v in selected['config'].items():
            if isinstance(config,dict) and k in config and config[k]!=v:
                raise ValueError('Explicit configuration conflicts with selected inventory: '+k)
            if isinstance(config,Config) and getattr(config,k)!=v:
                raise ValueError('Config conflicts with selected inventory: '+k)
        c=replace(c,**selected['config'])
        expected_scene=7 if c.solver_revision in ('residual-shell-v5','bounded-shell-v6')else 5
        if c.scene_revision!=expected_scene:
            raise UnsupportedConfigurationError(f'{c.solver_revision} requires integrated scene revision{expected_scene}')
        self.cameras=tuple(cameras)
        if len(set(self.cameras))!=len(self.cameras) or set(self.cameras)-{'left_wrist_rgb','right_wrist_rgb'}:
            raise ValueError('Policy cameras must be mounted wrist RGB cameras')
        self.options=dict(renderer_options or {})
        if set(self.options)-{'width','height','samples'}:raise ValueError('Unknown renderer option')
        w,h=self.options.get('width',1280),self.options.get('height',720)
        if any(type(v)is not int or v<=0 for v in (w,h,self.options.get('samples',32))) or w>1280 or h>720:
            raise ValueError('Positive image sizes up to1280x720 and positive sample count required')
        if type(recording)is not bool:raise TypeError('recording must be boolean')
        self.output=Path(output_dir or Path.home()/'.cache/gar/deposit-cans'/uuid.uuid4().hex)
        self.output.mkdir(parents=True,exist_ok=False)
        self.env=None;self.trace=None;self.checkpoints=None;self._recording=recording
        try:
            self.env=DepositCansEnv(c,robot_setup=self.robot_setup);self.ik=Kinematics(self.env.assembly);self.catalog=Catalog(self.env)
            if recording:self.trace=Trace(self.env,self.output/'physics.sqlite3')
            self.checkpoints=Store(self.output/'checkpoints',budget_bytes=1024**3)
            # Compile the first command's kernels before the server reports ready; no tick or state change.
            self.env.warm_up()
        except BaseException:
            self.close();raise
        self._variation=selected;self.images=deque();focal=.018/math.tan(math.radians(87/2))
        self._pristine_initial_model=True
        self.specs={n:CameraSpec(n,self.env.builder._gar_robots[n.split('_')[0]].frames['wrist']['body_label'],w,h,1280,720,
            (.036,.036*h/w),(focal,focal))for n in self.cameras}
        self.kinds=frozenset({'named_joint_target','gripper','cartesian_pose','cartesian_delta','bimanual_cartesian'})
        self.capabilities=Capabilities('deposit-cans','newton',self.robot_id,self.kinds,
            frozenset({'proprioception',*(['rgb']if self.cameras else [])}),True,True,True,
            self.cameras,tuple(self.specs.values()),notes=dict(
                qualification=('FR3 v6/scene7 component evidence retained; shared-robot integration evidence is separately scoped.'
                    if all(x['model'].startswith('fr3@') for x in self.robot_setup.to_dict()['instances'].values())
                    else 'Alternative Panda/WSG robot integration; not a new full task/material qualification.'),
                numerical_revision=c.solver_revision,scene_revision=str(c.scene_revision),
                stepping=('Host-checked residual solve; CUDA graph capture disabled'if c.solver_revision=='residual-shell-v5'
                    else 'Bounded coupled solve; '+('CUDA graph dispatch'if c.capture else'host dispatch')
                    if c.solver_revision=='bounded-shell-v6'else'Legacy explicit revision'),
                clock='Commands advance exact ticks. Reads, images, geometry and selection advance none.',
                control='Finite-effort, rate-limited servos. IK does not guarantee a collision-free path.',
                gripper=f'WSG MOVE: half_width_m0..0.055m, {c.gripper_force:g}N total cap. GRIP: inward_effort_n5..80N TOTAL (half per jaw),50mm/s relative closure. Choose one field. Grip persists until a finger-width command; no holding detector.',
                camera='Ideal RGB pinhole on collision-bearing D405-envelope wrist mounts; no depth or noise model.',
                physics='Dry opened cans; reduced thin-shell modest denting; no liquid, puncture or fracture.',
                checkpoint='Explicit recorded-state restore, generation changes; not guaranteed deterministic future GPU replay.',
                history='Privileged every-completed-substep trace, bounded retention and explicit missing fields.'),
                robot_configuration=self.env.robot_rig.capabilities())

    def reset(self,*,seed=None,options=None):
        from deposit_cans.env import DepositCansEnv
        from deposit_cans.variants import resolve
        from deposit_cans.catalog import Catalog
        from deposit_cans.kinematics import Kinematics
        from deposit_cans.trace import Trace
        if self.trace and self.trace.current:raise RuntimeError('Finish active trace before reset')
        values=dict(options or {})
        if set(values)-{'variant','contract'}:raise ValueError('Reset selectors are variant, contract and seed only')
        variant=values.get('variant',self._variation['id'])
        # Without an explicit contract the same variant keeps its current contract; another variant takes its default.
        contract=values.get('contract',self._variation['config'].get('task_contract') if variant==self._variation['id'] else None)
        selected=resolve(variant,self.env.config.seed if seed is None else seed,contract)
        # RoboticsEnv auto-reset follows construction. The constructor already
        # created exactly this seed/variant at tick zero. Reuse it ONCE, while
        # preserving reset generation semantics; never reuse an operated state.
        # Constructing it twice unnecessarily retains two models and solver
        # workspaces and can starve rendering before the first observation.
        if (self._pristine_initial_model and self.env.tick==0 and self.env.generation==0
                and self.env.fault is None and selected==self._variation):
            self._pristine_initial_model=False;self.env.generation+=1
            return self.get_state(),dict(continuity='explicit_reset',variation=selected)
        self._pristine_initial_model=False
        # The previous variation's contract choice never carries over to a variant without one.
        candidate=DepositCansEnv(replace(self.env.config,**{'task_contract':'',**selected['config']}),robot_setup=self.robot_setup)
        try:
            ik=Kinematics(candidate.assembly);catalog=Catalog(candidate)
            trace=Trace(candidate,self.trace.store.path)if self._recording else None
            candidate.warm_up()
        except BaseException:candidate.close();raise
        old=self.env;candidate.generation=old.generation+1
        if self.trace:self.trace.close()
        self.env,self.ik,self.catalog,self.trace=candidate,ik,catalog,trace;self._variation=selected
        old.close()
        return self.get_state(),dict(continuity='explicit_reset',variation=selected)

    def get_state(self):return self.env.get_state()
    def variation_contract(self):
        from deposit_cans.variants import contract
        return dict(**contract(),current=self._variation)
    def select_variation(self,request):
        from deposit_cans.variants import resolve
        from deposit_cans.task import description
        if set(request)-{'variant','seed','contract'}:raise ValueError('Selectors are variant, seed and contract only')
        v=resolve(request.get('variant','reference'),request.get('seed',0),request.get('contract'))
        chosen=v['config'].get('task_contract')
        return dict(variation=v,task=description(v['id'],chosen),reset_request=dict(seed=v['seed'],
            options={'variant':v['id'],**({'contract':chosen} if chosen else {})}))

    def _render(self,names):
        from deposit_cans.render import Renderer
        with self.env.lock:
            try:
                self.env.renderer=Renderer(self.env,self.output/f'render_{uuid.uuid4().hex}',**self.options)
                result=self.env.renderer.observe(names)
            finally:
                if self.env.renderer:
                    try:self.env.renderer.close()
                    finally:self.env.renderer=None
            frames={}
            for name,blob in result['images'].items():
                path=self.output/f'{name}_{self.env.generation}_{self.env.tick}_{uuid.uuid4().hex}.png'
                path.write_bytes(base64.b64decode(blob));self.images.append(path);frames[name]=path
            while len(self.images)>64:
                p=self.images.popleft()
                if p.parent!=self.output:raise RuntimeError('Image retention escaped output directory')
                p.unlink(missing_ok=True)
            return frames
    def observation_from_state(self,state,*,sequence):
        with self.env.lock:
            if (state['tick'],state['generation'])!=(self.env.tick,self.env.generation):raise ValueError('Stale state/image pairing')
            frames=self._render(self.cameras)if self.cameras else {}
            proprio={'joints':{n:{k:x[k]for k in ('position','velocity')}for n,x in state['joints'].items()}}
            return Observation(sequence,state['time_s'],self.robot_id,proprio,
                {n:ImageFrame(n,p,True,camera=self.specs[n])for n,p in frames.items()})
    def _ticks(self,seconds):
        n=seconds/self.env.config.dt
        if not math.isclose(n,round(n),rel_tol=0,abs_tol=1e-8):raise ValueError('Duration must equal a whole physics-tick count')
        return int(round(n))
    def _fingers(self,arm,width):
        if arm not in ('left','right')or isinstance(width,bool)or not isinstance(width,(float,int))or not math.isfinite(width)or not 0<=width<=.055:
            raise ValueError('Valid arm and WSG half-gap in [0,.055]m required')
        return {f'{arm}/wsg_finger_joint{i}':width for i in (1,2)}
    def _cartesian(self,arm,position,delta,orientation,gripper):
        from deposit_cans.kinematics import arm_q
        seed=arm_q(self.env,arm);fk=self.ik.fk(arm,seed)
        p=fk[:3,3]+delta if delta is not None else position
        quat=Rotation.from_matrix(fk[:3,:3]).as_quat()if orientation is None else np.asarray((*orientation[1:],orientation[0]))
        fit=self.ik.solve(arm,p,quat,seed)
        if not fit['converged']:raise ValueError(f"IK unreachable: {fit['position_error_m']:.6f}m, {fit['orientation_error_rad']:.6f}rad; no motion")
        targets={name:float(x)for name,x in zip(self.ik.joint_names(arm),fit['joints'])}
        if gripper is not None:targets.update(self._fingers(arm,gripper))
        return targets
    def _prepare(self,action):
        safety=None;arm_arm=False;grip_commands={}
        if isinstance(action,NamedJointTarget):targets,ticks=dict(action.targets_si),action.ticks
        elif isinstance(action,Gripper):
            ticks=self._ticks(action.duration_s)
            if action.inward_effort_n is not None:
                if action.half_width_m is not None:raise ValueError('Choose half_width_m MOVE or inward_effort_n GRIP, not both')
                from deposit_cans.gripper import validate
                grip_commands=validate(self.env,{action.arm:action.inward_effort_n});targets={}
            else:targets=self._fingers(action.arm,action.half_width_m)
        elif isinstance(action,(CartesianPose,CartesianDelta)):
            safety=action.safety
            if safety.stop_on_grip_loss is not None:raise UnsupportedActionError('Object-specific grip guard unavailable')
            targets=self._cartesian(action.arm,action.position_m if isinstance(action,CartesianPose)else None,
                action.delta_m if isinstance(action,CartesianDelta)else None,
                action.orientation_wxyz if isinstance(action,CartesianPose)else None,action.gripper_half_target_m)
            ticks=self._ticks(action.duration_s);arm_arm=safety.stop_on_arm_arm_contact
        elif isinstance(action,BimanualCartesian):
            targets={}
            for arm in ('left','right'):
                a=getattr(action,arm);targets.update(self._cartesian(arm,a.position_m,a.delta_m,a.orientation_wxyz,a.gripper_half_target_m))
            ticks=self._ticks(action.duration_s);arm_arm=action.stop_on_arm_arm_contact
        else:raise UnsupportedActionError('Unsupported deposit-cans action')
        self.env.validate_targets(targets,ticks)
        def guard(e):
            for a,b in e.active_contact_pairs():
                labels=[e.builder.shape_label[i]for i in (a,b)]
                arms={s.split('/')[0]for s in labels if s.startswith(('left/','right/'))}
                if arm_arm and arms=={'left','right'}:return True
                if safety and (safety.stop_on_contact or safety.stop_on_robot_contact)and arms:return True
            if safety and (safety.stop_on_contact or safety.stop_on_robot_contact):
                if e.progress.contacts.read(robot_only=True):return True
            return False
        return targets,ticks,grip_commands,guard if arm_arm or (safety and (safety.stop_on_contact or safety.stop_on_robot_contact))else None
    def act(self,action):
        with self.env.lock:
            if self.env.fault:raise BackendProtocolError(self.env.fault)
            try:targets,ticks,grip_commands,guard=self._prepare(action)
            except (ValueError,TypeError)as exc:raise ActionRejectedError(str(exc))from exc
            self._pristine_initial_model=False
            before=self.env.tick
            try:state=self.env.act(targets,ticks=ticks,guard=guard,grip_commands=grip_commands)
            except Exception as exc:raise BackendProtocolError(f'Stopped after tick {self.env.tick}: {exc}')from exc
            return BackendStep(state,dict(requested_ticks=ticks,executed_ticks=self.env.tick-before,stopped_on_contact=self.env.tick-before<ticks),
                state['evaluation']['score'],state['evaluation']['success'])
    def policy_access(self):
        from deposit_cans.task import description
        return PolicyAccess(self.robot_id,tuple(self.specs.values()),
            {('joints',n,f):None for n in self.env.joints for f in ('position','velocity')},
            self.kinds,lambda:description(self._variation['id'],self._variation['config'].get('task_contract')),dict(dt_s=self.env.config.dt,
                joints={n:{k:v for k,v in s.items()if k not in ('q','dof')}for n,s in self.env.joints.items()}))
    def act_policy(self,action):
        step=self.act(action);return BackendStep(step.state,step.info,None,False,step.truncated)
    def render(self,*,label='interactive'):
        name='room'if label in ('interactive','http')else label
        if name not in ('room','station','slot','tub','bag_bottom','left_wrist_rgb','right_wrist_rgb'):raise ValueError('Unknown diagnostic camera')
        return self._render((name,))[name]
    def get_geometry_index(self):return self.catalog.index()
    def get_geometry(self,geometry_id):return self.catalog.objects[str(geometry_id)]
    def get_geometry_asset(self,asset_id):return self.catalog.assets[str(asset_id)]
    def begin_action_trace(self,context):return self.trace.begin(context)if self.trace else dict(available=False,reason='recording_disabled')
    def end_action_trace(self):return self.trace.finish()if self.trace else None
    def save_checkpoint(self,*,label):
        from deposit_cans.snapshot import capture
        if not isinstance(label,str)or not 1<=len(label)<=64:raise ValueError('Checkpoint label length must be1..64')
        with self.env.lock:
            self.checkpoints.put('checkpoint',label,capture(self.env))
            return dict(name=label,tick=self.env.tick,durable=True,continuity='unchanged')
    def restore_checkpoint(self,checkpoint):
        from deposit_cans.snapshot import restore
        with self.env.lock:
            state=restore(self.env,self.checkpoints.get('checkpoint',checkpoint,arrays=True))
            return state,dict(checkpoint=checkpoint,continuity='restored_discontinuity')
    def list_checkpoints(self):return tuple(dict(name=k)for k in self.checkpoints.keys('checkpoint'))
    def close(self):
        try:
            if self.trace:self.trace.close();self.trace=None
        finally:
            try:
                if self.checkpoints:self.checkpoints.close();self.checkpoints=None
            finally:
                if self.env:self.env.close()
