"""G1 craft configuration. Shared pinned G1/Dex3 rig, free-standing PD joints, real brush contact.
No pelvis constraint, invisible grasp weld, or surface-film command.
"""
import copy
import math
import threading
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from painting import params as P, layout as L
from painting.scene import PaintingScene
from painting.world import PaintingWorld
from painting.env import PaintingEnv
from classroom_arrangement.world import World, require_engine
from percussion.g1 import G1HandIK, head_camera, pelvis_state, ARMS_READY, HAND_JOINTS, joint

class G1Scene(PaintingScene):
    def robot(self):
        import robots
        from robots.catalog import resolve
        from robots.backends.newton import mount
        # Dedicated scene mount; the 0.74 m table and its physical contents stay unchanged.
        setup=resolve('painting-g1')
        self.robot_setup=setup
        b=self.b; first,fs=b.body_count,b.shape_count
        self.rig_pending=robots.add_to(b,setup,mounts={'g1_floor_mount':mount((-.55,.84,.79),math.pi/2)})
        self.robot_bodies=list(range(first,b.body_count));self.robot_shapes=list(range(fs,b.shape_count))
        self.robot_instance='g1';self.handle=self.rig_pending.instances['g1'];self.joints=dict(self.handle.joints)
        self.arms=('left','right');self.groups={}
        self.robot_body={b.body_label[i].split('/')[-1]:i for i in self.robot_bodies}
        self.g1_hw=dict(pelvis=self.robot_body['pelvis'],torso=self.robot_body['torso_link'])
        for i in self.robot_shapes:
            coll=bool(int(b.shape_flags[i]) & int(self.newton.ShapeFlags.COLLIDE_SHAPES))
            self.materials[i]='robot' if coll else 'robot_visual'
            b.shape_gap[i]=0.
            if coll:
                b.shape_material_mu[i]=.8 if '_hand_' in b.body_label[b.shape_body[i]] else .6
                if '_hand_' in b.body_label[b.shape_body[i]]: self.pad_shapes[i]=b.body_label[b.shape_body[i]]
        head_camera(self)

class G1World:
    body_pose=World.body_pose
    body_velocity=PaintingWorld.body_velocity
    sync_solver=World.sync_solver
    set_lumped_mass=PaintingWorld.set_lumped_mass
    contact_table=World.contact_table
    contact_force=World.contact_force
    def __init__(self,config,spec):
        self.config,self.spec=config,spec
        self.newton,self.wp=newton,wp=require_engine()
        import mujoco
        self._mujoco=mujoco
        t=time.perf_counter()
        with wp.ScopedDevice('cpu'):
            self.scene=sc=G1Scene(config,newton,wp,spec)
            self.model=m=sc.b.finalize(device='cpu')
            from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
            m.mujoco.solref_mode.fill_(SOLREF_MODE_RAW)
            ref=m.mujoco.solref.numpy();ref[:]=(.002,1.);m.mujoco.solref.assign(ref)
            self.rig=sc.rig_pending.bind(m)
            self.pipeline=newton.CollisionPipeline(m,rigid_contact_max=4096,broad_phase='sap',reduce_contacts=True)
            self.contacts=self.pipeline.contacts()
            self.solver=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,use_mujoco_contacts=True,
                enable_multiccd=True,integrator='implicitfast',cone='elliptic',solver='newton',iterations=100,
                ls_iterations=20,impratio=10,nconmax=4096,njmax=16384,update_data_interval=0)
            self.state,self.next_state=m.state(),m.state();self.control=m.control()
        self.mjm,self.mjd=self.solver.mj_model,self.solver.mj_data
        self.mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        self.mjm.opt.noslip_iterations=1000;self.mjm.opt.noslip_tolerance=1e-15
        self._geom_shape=np.asarray(self.solver.mjc_geom_to_newton_shape.numpy()).reshape(-1)
        m2n=np.asarray(self.solver.mjc_body_to_newton.numpy()).reshape(-1)
        self._mj_body={int(n):i for i,n in enumerate(m2n) if n>=0}
        self._newton_to_mj=self._mj_body
        tools={it['body'] for it in sc.items.values()}; fingers={i for i in sc.robot_bodies if '_hand_' in m.body_label[i]}
        for g in range(self.mjm.ngeom):
            body=int(m2n[self.mjm.geom_bodyid[g]])
            if body in fingers:
                self.mjm.geom_priority[g]=1;self.mjm.geom_condim[g]=4
                self.mjm.geom_friction[g,:2]=(.8,.002);self.mjm.geom_solref[g]=(.001,1.)
            elif body in tools:
                self.mjm.geom_solref[g]=(.001,1.)
        self.joints=dict(sc.joints);self.arms=sc.arms;self.groups=sc.groups
        self.state.joint_q.assign(m.joint_q)
        q=self.state.joint_q.numpy();target=self.control.joint_target_q.numpy()
        # Park the unused arm down beside the hip. A flexed left elbow puts its
        # hand through the near edge of the craft board during startup.
        start={**ARMS_READY,'g1/left_elbow_joint':.15}
        for n,v in start.items():q[self.joints[n]['q']]=target[self.joints[n]['q']]=v
        self.state.joint_q.assign(q);self.control.joint_target_q.assign(target)
        newton.eval_fk(m,self.state.joint_q,self.state.joint_qd,self.state);self.next_state.assign(self.state)
        self.targets=target.copy();self._extra_mass={};self.paint=None;self.fixture=None;self.recorders=[]
        self.body_wrench=np.zeros((m.body_count,6),np.float32)
        self.tick=0;self.fault=None;self.reflexes=[];self.refused_targets=0
        self.timing=dict(physics_s=0.,control_s=0.);self.build_s=time.perf_counter()-t
        self.sync_solver()
        # Gravity feedforward on commanded waist/arm pose, with total effort kept within URDF limits.
        names={mujoco.mj_id2name(self.mjm,mujoco.mjtObj.mjOBJ_JOINT,j):j for j in range(self.mjm.njnt)}
        arm=[n for n in self.joints if any(k in n for k in ('_shoulder_','_elbow_','_wrist_','/waist_'))]
        jj=[next(j for k,j in names.items() if k and k.endswith('_'+n.split('/')[-1])) for n in arm]
        self.ff=dict(q=[self.joints[n]['q'] for n in arm],dof=[self.joints[n]['dof'] for n in arm],
            mj_q=self.mjm.jnt_qposadr[jj],mj_dof=self.mjm.jnt_dofadr[jj],j=jj,
            limit=np.array([self.joints[n]['effort_limit'] for n in arm]),data=mujoco.MjData(self.mjm),tau=np.zeros(len(arm)))
    @property
    def sim_time(self):return self.tick*P.DT
    def set_targets(self,targets):
        for n,v in targets.items():
            if n not in self.joints:raise ValueError(f'Unknown G1 joint {n}')
            j=self.joints[n]
            if not np.isfinite(v) or not j['lower']<=v<=j['upper']:raise ValueError(f'Out-of-range G1 joint {n}')
        for n,v in targets.items():self.targets[self.joints[n]['q']]=v
    def step(self):
        if self.fault:raise RuntimeError(self.fault)
        t=time.perf_counter()
        try:
            self.body_wrench[:]=0.
            if self.paint is not None:self.paint.pre_step()
            cur=self.control.joint_target_q.numpy();vel=self.control.joint_target_qd.numpy()
            for j in self.joints.values():
                i,d=j['q'],j['dof'];move=np.clip(self.targets[i]-cur[i],-j['velocity_limit']*P.DT,j['velocity_limit']*P.DT)
                cur[i]+=move;vel[d]=move/P.DT
            jf=np.zeros(self.model.joint_dof_count,np.float32);ff=self.ff
            if self.tick%10==0:
                d=ff['data'];d.qpos[:]=self.mjd.qpos;d.qvel[:]=0.;d.qpos[ff['mj_q']]=cur[ff['q']]
                self._mujoco.mj_kinematics(self.mjm,d);self._mujoco.mj_comPos(self.mjm,d)
                tau=np.zeros(self.mjm.nv);self._mujoco.mj_rne(self.mjm,d,0,tau)
                ff['tau']=np.clip(tau[ff['mj_dof']],-ff['limit'],ff['limit'])
            jf[ff['dof']]=ff['tau']
            self.mjm.jnt_actfrcrange[ff['j'],0]=-ff['limit']-ff['tau']
            self.mjm.jnt_actfrcrange[ff['j'],1]=ff['limit']-ff['tau']
            self.control.joint_target_q.assign(cur);self.control.joint_target_qd.assign(vel);self.control.joint_f.assign(jf)
            self.state.clear_forces()
            if self.fixture is not None:self.fixture(self)
            self.state.body_f.assign(self.body_wrench)
            self.solver.step(self.state,self.next_state,self.control,self.contacts,P.DT)
            if any(self.mjd.warning[i].number for i in range(len(self.mjd.warning))):
                raise RuntimeError('MuJoCo warning: G1 episode stopped')
            self.state,self.next_state=self.next_state,self.state;self.tick+=1
            if self.paint is not None:self.paint.post_step()
            bq=self.state.body_q.numpy();bv=self.state.body_qd.numpy()
            if not np.isfinite(bq).all() or not np.isfinite(bv).all():raise FloatingPointError('nonfinite G1 state')
            z,tilt=pelvis_state(self)
            if z<.5 or tilt>35:raise RuntimeError(f'G1 fallen: pelvis {z:.3f} m, tilt {tilt:.1f} deg')
            for rec in self.recorders:rec(self)
            self.timing['physics_s']+=time.perf_counter()-t
        except Exception as exc:
            self.fault=f'{type(exc).__name__}: {exc}';raise

class G1Env(PaintingEnv):
    def __init__(self,config,*,output_dir=None,spec=None,settle=True):
        self.config=config;self.lock=threading.RLock();self.output=Path(output_dir) if output_dir else None
        from .catalogue import apply_contract
        self.spec=copy.deepcopy(apply_contract(spec or L.setup(config.task,config.seed),config.contract))
        bp=list(self.spec['items']['board']['pose']);bp[0]=-.55;bp[1]-=.10;self.spec['items']['board']['pose']=tuple(bp)
        pp=list(self.spec['items']['palette']['pose']);pp[0]=-.10;pp[1]=1.13;pp[3]=(0.,0.,0.,1.);self.spec['items']['palette']['pose']=tuple(pp)
        if self.spec['preset']!='craft':raise ValueError('G1 qualifies craft only; wall reach and two-hand aerosol use not qualified')
        self.world=w=G1World(config,self.spec)
        from painting.system import PaintSystem
        self.paint=PaintSystem(w,self.spec);w.paint=self.paint
        self.ik=G1HandIK(w);self.generation=0
        self._place_held()
        if settle:
            for _ in range(2000):w.step()
            w.tick=0
        from painting.task import Evaluator
        self.evaluator=Evaluator(self);self.initial=self.checkpoint()
        self.start_layout_record={}
        for name in ('board','palette','table'):
            pos,rr=w.body_pose(w.scene.bodies[name]);requested=self.spec['items'][name]['pose']
            displacement=float(np.linalg.norm(pos-np.array(requested[:3]))*1000)
            angle=float(np.linalg.norm((Rotation.from_matrix(rr)*Rotation.from_quat(requested[3]).inv()).as_rotvec())*180/np.pi)
            self.start_layout_record[name]=dict(translation_mm=displacement,rotation_deg=angle)
            if displacement>5 or angle>2:raise RuntimeError(f'G1 startup displaced {name}: {self.start_layout_record[name]}')
        thumb=[i for n,i in w.scene.robot_body.items() if n.startswith('right_hand_thumb_')]
        grip=w.contact_force(thumb,w.scene.bodies['round8'])
        self.hold_record['initial_thumb_force_N']=float(np.linalg.norm(grip))
        if np.linalg.norm(grip)<.3:
            touching=[(w.model.body_label[a],w.model.body_label[b],float(f)) for a,b,f,_,_ in w.contact_table() if w.scene.bodies['round8'] in (a,b)]
            raise RuntimeError(f'G1 initial thumb contact absent; contacts={touching}; layout={self.start_layout_record}')

    def _place_held(self):
        from painting.probes.fixtures import PhantomMount
        w=self.world
        R=Rotation.from_matrix(np.column_stack([[0,1,0],[1,0,0],[0,0,-1]]))
        sol,err=self.ik.solve('right',[-.58,1.16,.94],R.as_quat(),restarts=12)
        preshape=dict(thumb_0=0.,thumb_1=.45,thumb_2=0.,index_0=.1,index_1=.1,middle_0=.1,middle_1=.1)
        for n,v in {**sol,**{joint('right',k):v for k,v in preshape.items()}}.items():self._set_joint(n,v)
        self._fk()
        # Let the feet and commanded arm settle before defining a world-space
        # loading fixture; otherwise the falling pelvis moves the hand off that fixture.
        for _ in range(2000):w.step()
        wrist=w.scene.robot_body['right_wrist_yaw_link'];pw,Rw=w.body_pose(wrist)
        center=pw+Rw@np.array([.095,.0215,0.]);Rb=Rw.copy()
        self._place_body('round8',center-Rb@np.array([0,0,-.115]),Rb);self._fk()
        body=w.scene.bodies['round8']
        fixture=PhantomMount(w,[body],(0,0,-.115),lambda t:(center,Rb),k=2000.,k_rot=40.)
        w.fixture=fixture.apply
        syn=dict(thumb=dict(thumb_1=-1.,thumb_2=-.6),index=dict(index_0=.7,index_1=1.),middle=dict(middle_0=.7,middle_1=1.))
        links={f:[i for i in w.scene.robot_bodies if 'right_hand_'+f+'_' in w.model.body_label[i]] for f in syn}
        state={f:'closing' for f in syn};goal={};ke=w.model.joint_target_ke.numpy()
        for _ in range(4000):
            for f,ss in syn.items():
                if state[f]!='closing':continue
                force=float(np.linalg.norm(w.contact_force(links[f],body)))
                if force>.3:
                    state[f]='squeezed'
                    for k,r in ss.items():
                        n=joint('right',k);j=w.joints[n];q=w.state.joint_q.numpy()[j['q']]
                        goal[n]=float(np.clip(q+np.sign(r)*.6*j['effort_limit']/ke[j['dof']],j['lower'],j['upper']))
                        w.targets[j['q']]=goal[n]
                else:
                    for k,r in ss.items():
                        j=w.joints[joint('right',k)]
                        w.targets[j['q']]=np.clip(w.targets[j['q']]+r*1.5*P.DT,j['lower'],j['upper'])
            w.step()
            if all(v=='squeezed' for v in state.values()):break
        for _ in range(600):w.step()
        w.fixture=None
        self.hold_record=dict(grasp='thumb against palm; index/middle curled',assistance='initial pose and phantom while fingers close; fixture removed before tick zero',close=state,targets=goal)
        print('GRASP',self.hold_record,flush=True)
    def act(self,targets=None,*,ticks=1,**kw):
        if kw:raise ValueError(f'G1 unsupported controls {list(kw)}')
        if type(ticks) is not int or ticks<1:raise ValueError('ticks must be positive integer')
        with self.lock:
            self.world.set_targets(targets or {})
            for _ in range(ticks):self.world.step()
            return self.get_state()
    def robot_status(self):
        z,tilt=pelvis_state(self.world)
        return dict(imu_tilt_deg=tilt,controller='floating G1 PD + commanded-pose arm/waist gravity feedforward')
    def get_state(self):
        w=self.world
        return dict(tick=w.tick,time_s=w.sim_time,generation=self.generation,fault=w.fault,finite=w.fault is None,
            ready=w.fault is None,joints=self.joints_state(),robot_status=self.robot_status(),tools=self.tools_state(),evaluation=self.evaluation())
    def checkpoint(self):
        w=self.world
        return dict(rigid={a:getattr(w.state,a).numpy().copy() for a in ('body_q','body_qd','joint_q','joint_qd')},
            targets=w.targets.copy(),control=w.control.joint_target_q.numpy(),control_qd=w.control.joint_target_qd.numpy(),
            tick=w.tick,paint=self.paint.snapshot(),task_evaluation=self.evaluator.checkpoint_state(),ff_tau=w.ff['tau'].copy(),mj_warm=w.mjd.qacc_warmstart.copy(),
            mass={k:(float(w.mjm.body_mass[w._mj_body[k]]),w.mjm.body_ipos[w._mj_body[k]].copy(),w.mjm.body_inertia[w._mj_body[k]].copy()) for k in w._extra_mass})
    def restore(self,snap):
        w=self.world
        with self.lock:
            for a,v in snap['rigid'].items():getattr(w.state,a).assign(v);getattr(w.next_state,a).assign(v)
            w.targets=snap['targets'].copy();w.control.joint_target_q.assign(snap['control']);w.control.joint_target_qd.assign(snap['control_qd'])
            w.tick=snap['tick'];w.fault=None;w.ff['tau']=snap['ff_tau'].copy()
            self.paint.restore(snap['paint'])
            self.evaluator.restore_state(snap.get('task_evaluation'))
            for k,(m,c,I) in snap['mass'].items():
                b=w._mj_body[k];w.mjm.body_mass[b]=m;w.mjm.body_ipos[b]=c;w.mjm.body_inertia[b]=I
            w.sync_solver();w.mjd.qacc_warmstart[:]=snap['mj_warm'];self._eval_key=None


from painting.trace import Trace
class G1Trace(Trace):
    """G1-specific durable samples; no fabricated mobile-base or Franka wrench sensors."""
    def begin(self, context):
        import shutil
        if self.current: raise RuntimeError('Action trace already active')
        if shutil.disk_usage(self.store.path.parent).free<512*1024**2:raise OSError('Trace disk reserve unavailable')
        w=self.world;key=str(context.get('action_id') or context.get('command') or f'tick-{w.tick}')
        self.store.begin(key,str(context.get('request_id',key)),w.tick,P.DT,dict(
            source='painting/G1+Newton+MuJoCo+film',body_labels=list(w.model.body_label),
            sampling='joint states/targets each tick; rigid states, tilt and paint summary every 20 ticks',
            missing='no wrist force sensors; film fields per cell in checkpoints only'))
        self.current=key
        return dict(available=True,id=key,store_path=str(self.store.path))
    def _on_tick(self,w):
        if self.current is None:return
        sample=dict(tick=w.tick,time_s=w.sim_time,joint_q=w.state.joint_q.numpy(),
            joint_qd=w.state.joint_qd.numpy(),joint_target_q=w.control.joint_target_q.numpy())
        if w.tick%self.every==0:
            sample.update(body_q=w.state.body_q.numpy(),body_qd=w.state.body_qd.numpy(),
                pelvis_height_tilt=pelvis_state(w),paint=self._paint(w))
        self.store.append(self.current,w.tick,sample,[])
