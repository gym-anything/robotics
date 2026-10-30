"""GAR CLI/HTTP/dashboard adapter for the industrial tightening workcell."""

import base64
from collections import deque
from dataclasses import replace
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import uuid
import numpy as np
from scipy.spatial.transform import Rotation

from .base import RoboticsBackend
from ..actions import NamedJointTarget,Gripper,CartesianPose,CartesianDelta,BimanualCartesian
from ..errors import (UnsupportedActionError,UnsupportedConfigurationError,
                      BackendProtocolError,ActionRejectedError)
from ..types import BackendStep,CameraSpec,Capabilities,ImageFrame,Observation
from ..policy_access import PolicyAccess


class NewtonTightenCodexBackend(RoboticsBackend):
    def __init__(self,*,robot="dual-fr3",config=None,device=None,
                 cameras=("bench_rgb","overview_rgb"),output_dir=None,
                 renderer_options=None,recording=True):
        from tighten_codex.config import Config
        from tighten_codex.env import TightenEnv
        from tighten_codex.catalog import Catalog
        from tighten_codex.trace import Trace
        from tighten_codex.kinematics import ArmKinematics
        from robots.catalog import ENVIRONMENTS

        from tighten_codex.kinematics import validate_setup
        aliases=ENVIRONMENTS["tighten-codex"]["aliases"]
        if isinstance(robot,str) and robot in aliases:
            setup,robot_id=aliases[robot],robot
        else:
            # Any catalog setup the bench can host (arms left/right on their
            # mounts, catalog arm with a bench TCP, Franka Hand each).
            try:
                data=validate_setup(robot)
            except (ValueError,TypeError) as exc:
                raise UnsupportedConfigurationError(
                    f"{exc}. Named setups: {', '.join(sorted(aliases))}; or pass a "
                    "gar.robot-setup.v1 with left/right arms (fr3@1 or panda@1) on "
                    "left_arm_mount/right_arm_mount, each with franka-hand@1") from exc
            setup,robot_id=data,data.get("id","custom")
        if config is None:c=Config()
        elif isinstance(config,dict):c=Config(**config)
        elif isinstance(config,Config):c=config
        else:raise TypeError("config must be a Config or dictionary")
        if device is not None:
            if isinstance(config,dict) and "device" in config:
                raise ValueError("Device specified twice")
            c=replace(c,device=device)
        if c.fixture_mode:
            raise UnsupportedConfigurationError("Assisted fixture mode is evidence-only; GAR serves the ordinary scene")
        self.cameras=tuple(cameras)
        if len(set(self.cameras))!=len(self.cameras) or set(self.cameras)-{"bench_rgb","overview_rgb"}:
            raise ValueError("Physical cameras are bench_rgb and overview_rgb")
        requested_options=dict(renderer_options or {})
        if set(requested_options)-{"width","height","samples"}:
            raise ValueError("Unknown renderer option")
        width,height=requested_options.get("width",960),requested_options.get("height",540)
        samples=requested_options.get("samples",8)
        if any(type(v) is not int or v<=0 for v in
               (width,height,samples)) or width>1280 or height>720:
            raise ValueError("Positive image size up to 1280 x 720 and positive samples required")
        # Resolve once. The public CameraSpec must describe the PNGs emitted
        # by the renderer even when callers omit renderer_options.
        self.options={"width":width,"height":height,"samples":samples}
        self.output=Path(output_dir or Path.home()/".cache/gar/tighten-codex"/uuid.uuid4().hex)
        self.output.mkdir(parents=True,exist_ok=False)
        self.env=TightenEnv(c,setup)
        self.robot=robot_id
        self.catalog=Catalog(self.env)
        self.arms=frozenset(self.env.rig.instances)
        self.ik=ArmKinematics(self.env.rig,self.env.setup)
        self.trace=Trace(self.env,self.output/"physics.sqlite3") if recording else None
        self.images=deque()
        # Render scheduling (measured bottleneck: a worker takes ~9 s to
        # start, its first render ~3.5-5.7 s to warm, its close ~2.5 s).
        # A worker is still created per operation and released right after
        # it; these only move that work off the reply's critical path.
        self._prestart=None
        self._reapers=[]
        self.specs={
            "bench_rgb":CameraSpec("bench_rgb","inspection_camera_lens",width,height,
                1280,720,(.036,.036*height/width),(.038,.038)),
            "overview_rgb":CameraSpec("overview_rgb","overview_camera_lens",width,height,
                1280,720,(.036,.036*height/width),(.020,.020))}
        self.kinds=frozenset({"named_joint_target","gripper","cartesian_pose",
                              "cartesian_delta"} |
                             ({"bimanual_cartesian"} if len(self.arms)==2 else set()))
        self.capabilities=Capabilities("tighten-codex","newton",robot_id,self.kinds,
            frozenset({"proprioception",*(["rgb"] if self.cameras else [])}),
            True,False,True,self.cameras,
            tuple(self.specs[name] for name in self.cameras),notes={
                "clock":"Exact 1/4800 s physics ticks; observation/render do not advance time",
                "control":"Physical bounded catalog-arm position servos (tighten controller profile); Cartesian IK has no hidden collision planner",
                "gripper":"Franka Hand half gap 0..0.04 m; catalog controller effort fixed",
                "camera":"Fixed inspection-camera housing with ideal RGB pinhole; no depth/noise or distortion model",
                "fasteners":"Prestarted M3/M4; native bit/socket collision, helical guidance and solved bearing force",
                "fixture":"Phantom assistance is separate and unavailable in GAR robot actions",
                "history":"Bounded privileged every-step state/contact records when recording enabled"})

    def robot_configuration(self):
        result=super().robot_configuration()
        result.update(setup_id=self.env.setup.id,
                      setup_fingerprint=self.env.setup.fingerprint)
        return result

    def reset(self,*,seed=None,options=None):
        if options or (seed is not None and seed!=self.env.config.seed):
            raise ValueError("This fixed task launch accepts no reset variation; select another task recipe")
        if self.trace and self.trace.current:
            raise RuntimeError("Finish active trace before reset")
        return self.env.reset(),{"continuity":"explicit_reset","task":self.env.config.task,
                                 "seed":self.env.config.seed}

    def get_state(self):return self.env.get_state()

    PRESTART_EXPIRY_S=60.

    def _start_renderer_early(self):
        """Start this operation's render worker while its physics runs, and
        warm it with one discarded render of the pre-action pose (shader and
        acceleration-structure build). The operation's observation consumes
        and releases it; a failed action closes it; an unconsumed one is
        closed after PRESTART_EXPIRY_S. Never more than one."""
        if (not self.cameras or self._prestart is not None or self.env.renderer is not None
                or not (self.output/"render"/"scene.usdc").is_file()):
            return
        from tighten_codex.render import open_renderer,indicator_states
        body_q=self.env.state.body_q.numpy().copy()
        tick=self.env.tick
        indicators=indicator_states(self.env)
        holder={"renderer":None,"error":None,"warm_s":None}
        def work():
            try:
                began=time.perf_counter()
                holder["renderer"]=open_renderer(self.env,self.output/"render",
                                                  reuse_scene=True,**self.options)
                holder["renderer"].observe(tick,body_q,self.cameras,indicators=indicators)
                holder["warm_s"]=time.perf_counter()-began
            except BaseException as exc:  # reported when consumed
                holder["error"]=exc
        holder["thread"]=threading.Thread(target=work,daemon=True,name="tighten-render-prestart")
        holder["thread"].start()
        self._prestart=holder
        timer=threading.Timer(self.PRESTART_EXPIRY_S,self._expire_prestart,args=(holder,))
        timer.daemon=True
        timer.start()

    def _take_prestart(self):
        holder,self._prestart=self._prestart,None
        if holder is None:
            return None
        holder["thread"].join()
        if holder["error"] is not None or holder["renderer"] is None:
            if holder["renderer"] is not None:
                holder["renderer"].close()
            return None
        return holder["renderer"]

    def _discard_prestart(self):
        renderer=self._take_prestart()
        if renderer is not None:
            self._release(renderer)

    def _expire_prestart(self,holder):
        with self.env.lock:
            if self._prestart is holder:
                self._discard_prestart()

    def _release(self,renderer):
        """Release a worker now, without holding the reply for Kit's close."""
        self._reapers=[t for t in self._reapers if t.is_alive()]
        thread=threading.Thread(target=renderer.close,daemon=False,name="tighten-render-release")
        thread.start()
        self._reapers.append(thread)

    def _render(self,names):
        from tighten_codex.render import open_renderer,indicator_states
        with self.env.lock:
            # RTX/Kit can lose a just-started worker during a concurrent render
            # workload. Retrying the same immutable pose is safe: observation
            # does not step physics. A second failure still propagates, so a
            # broken RGB stream is never reported as a successful action.
            for attempt in range(2):
                renderer=self._take_prestart() if attempt==0 else None
                try:
                    if renderer is None:
                        renderer=open_renderer(self.env,self.output/"render",
                            reuse_scene=True,**self.options)
                    self.env.renderer=renderer
                    result=renderer.observe(self.env.tick,self.env.state.body_q.numpy(),names,
                                            indicators=indicator_states(self.env))
                except (ConnectionError, BrokenPipeError, EOFError):
                    if renderer is not None:
                        renderer.close()
                    self.env.renderer=None
                    if attempt:
                        raise
                    continue
                except BaseException:
                    if renderer is not None:
                        renderer.close()
                    self.env.renderer=None
                    raise
                self.env.renderer=None
                self._release(renderer)
                break
            frames={}
            for name,blob in result["images"].items():
                path=self.output/f"{name}_{self.env.generation}_{self.env.tick}_{uuid.uuid4().hex}.png"
                path.write_bytes(base64.b64decode(blob))
                self.images.append(path)
                frames[name]=path
            while len(self.images)>64:
                old=self.images.popleft()
                if old.parent!=self.output:
                    raise RuntimeError("Image retention escaped session output folder")
                old.unlink(missing_ok=True)
            return frames

    def observation_from_state(self,state,*,sequence):
        with self.env.lock:
            if (state["tick"],state["generation"])!=(self.env.tick,self.env.generation):
                raise RuntimeError("Stale state cannot be paired with a current image")
            images=self._render(self.cameras) if self.cameras else {}
            return Observation(sequence,state["time_s"],self.robot,
                {"joints":state["joints"]},
                {name:ImageFrame(name,path,True,camera=self.specs[name])
                 for name,path in images.items()})

    def _ticks(self,seconds):
        value=seconds/self.env.config.dt
        if not math.isclose(value,round(value),rel_tol=0,abs_tol=1e-8):
            raise ValueError("Duration must be an exact number of physics ticks")
        return int(round(value))

    def _fingers(self,arm,width):
        if arm not in self.arms:
            raise ValueError(f"Arm {arm!r} is unavailable in {self.robot}")
        if isinstance(width,bool) or not isinstance(width,(float,int)) or not math.isfinite(width) or not 0<=width<=.04:
            raise ValueError("Half gap must be 0..0.04 m")
        return {name:width for name in self.ik.fingers[arm]}

    def _cartesian(self,arm,position,delta,orientation,gripper):
        if arm not in self.arms:
            raise ValueError(f"Arm {arm!r} is unavailable in {self.robot}")
        q=self.env.state.joint_q.numpy()
        names=self.ik.names[arm]
        seed=[float(q[self.env.joints[name]["q"]]) for name in names]
        fk=self.ik.fk(arm,seed)
        target=fk[:3,3]+delta if delta is not None else position
        quat=(Rotation.from_matrix(fk[:3,:3]).as_quat() if orientation is None
              else np.asarray((*orientation[1:],orientation[0])))
        fit=self.ik.solve(arm,target,quat,seed)
        if not fit["converged"]:
            raise ValueError(f"IK missed target by {fit['position_error_m']:.6f} m and "
                             f"{fit['orientation_error_rad']:.6f} rad; no motion")
        result={name:float(value) for name,value in zip(names,fit["joints"])}
        if gripper is not None:result.update(self._fingers(arm,gripper))
        return result

    def _prepare(self,action):
        with self.env.lock:
            safety=None
            arm_arm=False
            if isinstance(action,NamedJointTarget):
                targets,ticks=dict(action.targets_si),action.ticks
            elif isinstance(action,Gripper):
                if action.inward_effort_n is not None:
                    raise UnsupportedActionError("Gripper effort override is unavailable")
                targets,ticks=self._fingers(action.arm,action.half_width_m),self._ticks(action.duration_s)
            elif isinstance(action,(CartesianPose,CartesianDelta)):
                safety=action.safety
                if safety.stop_on_grip_loss is not None:
                    raise UnsupportedActionError("Grip-loss guard is unavailable")
                targets=self._cartesian(action.arm,
                    action.position_m if isinstance(action,CartesianPose) else None,
                    action.delta_m if isinstance(action,CartesianDelta) else None,
                    action.orientation_wxyz if isinstance(action,CartesianPose) else None,
                    action.gripper_half_target_m)
                ticks=self._ticks(action.duration_s)
                arm_arm=safety.stop_on_arm_arm_contact
            elif isinstance(action,BimanualCartesian):
                if len(self.arms)!=2:
                    raise UnsupportedActionError("Bimanual Cartesian requires two mounted arms")
                targets={}
                for arm in ("left","right"):
                    target=getattr(action,arm)
                    targets.update(self._cartesian(arm,target.position_m,target.delta_m,
                        target.orientation_wxyz,target.gripper_half_target_m))
                ticks=self._ticks(action.duration_s)
                arm_arm=action.stop_on_arm_arm_contact
            else:
                raise UnsupportedActionError("Supported: named joints, gripper and Cartesian hand targets")
            self.env.rig.validate(targets)
            if type(ticks) is not int or ticks<=0:
                raise ValueError("Positive exact physics tick count required")
            def guard(env):
                n=int(env.contacts.rigid_contact_count.numpy()[0])
                a=env.contacts.rigid_contact_shape0.numpy()[:n]
                b=env.contacts.rigid_contact_shape1.numpy()[:n]
                for x,y in zip(a,b):
                    labels=[env.builder.shape_label[int(i)] for i in (x,y)]
                    arms=[label.split("/")[0] if label.startswith(("left/","right/"))
                          else None for label in labels]
                    if arm_arm and set(arms)=={"left","right"}:return True
                    if safety and (safety.stop_on_contact or safety.stop_on_robot_contact) and any(arms):return True
                return False
            guarded=arm_arm or (safety and (safety.stop_on_contact or safety.stop_on_robot_contact))
            return targets,ticks,guard if guarded else None

    def act(self,action):
        with self.env.lock:
            if self.env.fault:raise BackendProtocolError(self.env.fault)
            try:targets,ticks,guard=self._prepare(action)
            except (ValueError,TypeError) as exc:
                raise ActionRejectedError(str(exc)) from exc
            before=self.env.tick
            self._start_renderer_early()
            try:state=self.env.act(targets,ticks=ticks,guard=guard)
            except (RuntimeError,FloatingPointError,OSError,ValueError) as exc:
                self._discard_prestart()
                raise BackendProtocolError(f"Stopped after tick {self.env.tick}: {exc}") from exc
            except BaseException:
                self._discard_prestart()
                raise
            info={"requested_ticks":ticks,
                "executed_ticks":self.env.tick-before,
                "stopped_on_contact":self.env.tick-before<ticks}
            if self.trace:
                info["history_retention"]=self.trace.retention()
            return BackendStep(state,info,
                state["evaluation"]["score"],state["evaluation"]["success"])

    def policy_access(self):
        from tighten_codex.task import instruction
        return PolicyAccess(self.robot,tuple(self.specs[name] for name in self.cameras),
            {("joints",name,field):None for name in self.env.joints
             for field in ("position","velocity")},self.kinds,
            lambda:instruction(self.env.config),{"dt_s":self.env.config.dt,
                "joints":{name:{key:value for key,value in spec.items()
                    if key not in ("q","dof")} for name,spec in self.env.joints.items()},
                "tools":"Free green M3 and yellow M4 torque drivers; task objects cannot be commanded"})

    def act_policy(self,action):
        result=self.act(action)
        return BackendStep(result.state,result.info,None,False,result.truncated)

    def render(self,*,label="interactive"):
        name="room" if label in ("interactive","http") else label
        if name not in ("room","bench","hardware","head","drivers","bench_rgb","overview_rgb",
                        "sensor_tool","frame_detail","frame_under","frame_back_under",
                        "frame_front_detail","hand_detail","hand_side","access_under",
                        "cradles_close","hand_sensor"):
            raise ValueError("Unknown render camera")
        return self._render((name,))[name]

    # Durable checkpoints: full simulator state and evaluator tracking in the
    # session directory (arrays .npz + JSON), restorable within this scene.
    def _checkpoint_dir(self):
        d=self.output/"checkpoints";d.mkdir(exist_ok=True);return d

    def save_checkpoint(self,*,label):
        import json,re
        if not isinstance(label,str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}",label):
            raise ValueError("Checkpoint label: 1-64 characters from A-Z a-z 0-9 _ . -")
        if self.trace and self.trace.current:
            raise RuntimeError("Finish the active trace before saving a checkpoint")
        arrays,meta=self.env.checkpoint()
        d=self._checkpoint_dir()
        np.savez_compressed(d/f"{label}.npz",**arrays)
        (d/f"{label}.json").write_text(json.dumps(meta,default=float))
        return {"name":label,"tick":meta["tick"],"durable":True,"continuity":"unchanged"}

    def restore_checkpoint(self,checkpoint):
        import json
        d=self._checkpoint_dir()
        path=d/f"{checkpoint}.json"
        if not isinstance(checkpoint,str) or not path.is_file():
            raise KeyError(checkpoint)
        if self.trace and self.trace.current:
            raise RuntimeError("Finish the active trace before restoring a checkpoint")
        meta=json.loads(path.read_text())
        with np.load(d/f"{checkpoint}.npz") as z:
            arrays={k:z[k] for k in z.files}
        state=self.env.restore(arrays,meta)
        return state,{"checkpoint":checkpoint,"tick":state["tick"],
                      "continuity":"restored_discontinuity"}

    def list_checkpoints(self):
        import json
        d=self._checkpoint_dir()
        return tuple({"name":p.stem,"tick":json.loads(p.read_text())["tick"]}
                     for p in sorted(d.glob("*.json")))

    def get_geometry_index(self):return self.catalog.index()
    def get_geometry(self,geometry_id):return self.catalog.objects[str(geometry_id)]
    def get_geometry_asset(self,asset_id):return self.catalog.assets[str(asset_id)]
    def begin_action_trace(self,context):
        return self.trace.begin(context) if self.trace else {"available":False,"reason":"recording_disabled"}
    def end_action_trace(self):return self.trace.finish() if self.trace else None

    def close(self):
        with self.env.lock:
            self._discard_prestart()
        for thread in self._reapers:
            thread.join()
        self._reapers=[]
        if self.trace:
            self.trace.close()
            self.trace=None
        self.env.close()
