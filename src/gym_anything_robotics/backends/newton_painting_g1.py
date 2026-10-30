"""Craft-only G1 adapter. The public interface commands physical joint servos and arm IK."""
import math
import threading
import time
from collections import deque
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .newton_painting import NewtonPaintingBackend, _session_folder, SESSION_MARKER
from ..actions import NamedJointTarget, CartesianPose, CartesianDelta
from ..errors import UnsupportedConfigurationError, UnsupportedActionError, ActionRejectedError, BackendProtocolError
from ..types import Capabilities, CameraSpec, BackendStep
from ..policy_access import PolicyAccess

class NewtonPaintingG1Backend(NewtonPaintingBackend):
    def __init__(self,*,robot='g1',preset=None,task=None,seed=0,cameras=None,output_dir=None,
                 renderer_options=None,recording=True,device=None,contract=None,**options):
        from painting.config import Config
        from painting.g1 import G1Env
        from painting.variants import resolve
        from painting.robot_setup import validate,check_task
        if options:raise UnsupportedConfigurationError(f'G1 options not supported: {sorted(options)}')
        if device not in (None,'cpu'):raise UnsupportedConfigurationError('G1 painting uses the CPU rigid/film solver')
        if validate(robot)!='g1':raise UnsupportedConfigurationError('This adapter builds the painting-g1 rig only')
        v=resolve(preset,task or 'fill-circle',int(seed),contract)
        check_task('g1',v['preset'])
        self.config=Config(preset=v['preset'],task=v['task'],seed=v['seed'],robot='g1',contract=v.get('contract'))
        self.robot_name='g1';self.options=dict(renderer_options or {})
        self.output=Path(output_dir) if output_dir else _session_folder();self.output.mkdir(parents=True,exist_ok=False)
        if output_dir is None:(self.output/SESSION_MARKER).write_text('painting G1 session\n')
        self.env=G1Env(self.config,output_dir=self.output);self.ik=self.env.ik;self.arms=('left','right')
        self.policy_cameras=('head_rgb',);self.cameras=tuple(('head_rgb',) if cameras is None else cameras)
        if len(set(self.cameras))!=len(self.cameras) or set(self.cameras)-set(self.policy_cameras):
            raise UnsupportedConfigurationError('G1 policy camera: head_rgb (torso-mounted D435)')
        width,height=self.options.get('width',640),self.options.get('height',360)
        focal=.018/math.tan(math.radians(69/2))
        self.camera_specs={'head_rgb':CameraSpec('head_rgb','G1 torso D435 (pinned URDF mount)',width,height,1280,720,
            (.036,.036*height/width),(focal,focal))}
        self.min_action_s=self._min_action_s(len(self.cameras),width,height)
        self.images=deque();self._renderer=self._timer=self._scene_path=None;self._render_lock=threading.RLock()
        self.last_timing={};self.recording=bool(recording);self.trace=None
        if self.recording:
            from painting.g1 import G1Trace
            self.trace=G1Trace(self.env.world,self.output/'physics.sqlite3')
        self.kinds=frozenset({'named_joint_target','cartesian_pose','cartesian_delta'});self.keyboard_keys=()
        self.capabilities=Capabilities('painting','newton','g1',self.kinds,
            frozenset({'proprioception',*(['rgb'] if self.cameras else [])}),True,True,True,self.cameras,
            tuple(self.camera_specs[c] for c in self.cameras),notes=self._notes())
    def _notes(self):
        return dict(task=self.env.spec['instruction'],clock='0.5 ms physical ticks; no hidden settling in actions',
            robot='Pinned Unitree G1 with 29 body joints and two seven-joint Dex3 hands; shared painting-g1 rig',
            control='Floating base; standing PD legs, commanded-pose gravity feedforward on waist/arms, URDF effort and speed limits. No walking or balance policy.',
            start='Dry round brush in a physical thumb-and-palm grasp; initial placement and phantom support only while fingers close, removed before tick zero.',
            gripper='Dex3 fingers use their named joint targets; no Franka gripper action.',
            qualification='Craft tasks only.',
            min_command_s=f'Warm camera actions >= {self.min_action_s:g} s; after idle release request an observation for zero-tick readiness before actions.',
            protective_stop='No FR3 collision reflex. Nonfinite state, solver warnings or a fall stop the episode; joint commands can cause falls.')
    def policy_access(self):
        w=self.env.world
        proprio={('joints',n,f):None for n in w.joints for f in ('position','velocity')}
        proprio[('robot_status','imu_tilt_deg')]=None
        joints={n:{k:float(j[k]) for k in ('lower','upper','velocity_limit','effort_limit')} for n,j in w.joints.items()}
        return PolicyAccess('g1',tuple(self.camera_specs[c] for c in self.cameras),proprio,self.kinds,
            lambda:self.env.spec['instruction'],dict(dt_s=.0005,arms=list(self.arms),joints=joints,control=self._notes()['control'],start=self._notes()['start'],
            **({'camera_renderer':self._camera_renderer_note()} if self.cameras else {})))
    def act(self,action):
        with self.env.lock,self._render_lock:
            if self.cameras and self._renderer is None:
                raise self._released_refusal('RGB renderer released; zero ticks executed. Request an observation for readiness.')
            try:
                if isinstance(action,NamedJointTarget):
                    ticks=action.ticks;targets=dict(action.targets_si)
                    if type(ticks) is not int or not self._ticks(self.min_action_s)<=ticks<=self._ticks(5.):
                        raise ValueError('Joint command ticks must span the advertised minimum through 5 seconds')
                elif isinstance(action,(CartesianPose,CartesianDelta)):
                    self._arm(action.arm);ticks=self._ticks(action.duration_s)
                    if not .1<=action.duration_s<=5.:raise ValueError('G1 Cartesian IK actions require 0.1 through 5 seconds')
                    if action.safety.stop_on_grip_loss is not None or action.gripper_half_target_m is not None:
                        raise UnsupportedActionError('Dex3 fingers take named targets; no grip-loss guard or parallel gripper width')
                    pos,rot=self.ik.pose(action.arm)
                    if isinstance(action,CartesianDelta):pos=pos+np.asarray(action.delta_m)
                    else:
                        pos=np.asarray(action.position_m)
                        if action.orientation_wxyz is not None:
                            q=action.orientation_wxyz;rot=Rotation.from_quat((*q[1:],q[0]))
                    targets,_=self.ik.solve(action.arm,pos,rot.as_quat(),restarts=4)
                else:raise UnsupportedActionError('G1 actions: named_joint_target, cartesian_pose, cartesian_delta')
                # Validate whole command before changing any target.
                for n,v in targets.items():
                    j=self.env.world.joints.get(n)
                    if j is None or isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not j['lower']<=v<=j['upper']:
                        raise ValueError(f'Invalid G1 joint target {n}={v}')
            except ValueError as exc:raise self._planner_refusal(ActionRejectedError(str(exc))) from exc
            if self._timer is not None:self._timer.cancel();self._timer=None
            w=self.env.world;before=w.tick;t=time.perf_counter()
            try:state=self.env.act(targets,ticks=ticks)
            except (RuntimeError,FloatingPointError) as exc:raise BackendProtocolError(f'Stopped at tick {w.tick}: {exc}') from exc
            self.last_timing['physics_s']=time.perf_counter()-t
            return BackendStep(state,dict(requested_ticks=ticks,executed_ticks=w.tick-before,reflexes=[],refused_targets=0),state['evaluation']['score'],False)
    def _rebuild(self,v):
        from dataclasses import replace
        from painting.g1 import G1Env,G1Trace
        from painting.robot_setup import check_task
        check_task('g1',v['preset'])
        config=replace(self.config,**dict(v,contract=v.get('contract')))
        candidate=G1Env(config,output_dir=self.output)
        with self.env.lock:
            if self.trace:self.trace.close()
            self._release();candidate.generation=self.env.generation+1
            self.env=candidate;self.config=config;self.ik=candidate.ik
            self._scene_path=None;self._geometry=None;self._checkpoints={}
            self.trace=G1Trace(candidate.world,self.output/f'physics-{candidate.generation}.sqlite3') if self.recording else None
            self.capabilities=replace(self.capabilities,notes=self._notes())
