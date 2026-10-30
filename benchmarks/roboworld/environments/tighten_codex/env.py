"""Persistent Newton workcell. Only catalogue robot joints are public actions."""

from dataclasses import asdict
import math
import threading
import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig, Synchronization
from scipy.spatial.transform import Rotation

from .config import Config
from .presets import build
from .task import NEW_TASKS, contract_for, uses_eight


class TightenEnv:
    # mixed work order (see _evaluate): a screw's load must stay with no bit
    # and no hand on the work for 10 ticks (2.1 ms, the bit-moment window),
    # and its rail-level grade runs from the loose corner's measured free
    # droop to the level tolerance.
    MIXED_RETENTION_TICKS = 10
    FREE_DROOP_RAD = .0369
    # mixed: the reference flag is "at its home" within this distance of its
    # start position (the home stop holds it at -0.1 mm; free returns rest
    # at +0.004 and -0.022 mm).
    MIXED_FLAG_HOME_BAND_M = .0005
    # mixed partial credit only: a flag short of home keeps a share of the
    # detection grade that falls linearly from the band edge to 0 at 10 mm
    # (the detected-cue LED is lit to ~7.2 mm). Success still needs home.
    MIXED_FLAG_RETURN_GRADE_M = .010
    # mixed: a seated click is cancelled once the screw turns back more than
    # this past its angle at that click (loosening; a seated screw cannot back
    # off by itself). Recorded after-click back-rotation: <= 1.76 mrad.
    MIXED_BACKOFF_TOLERANCE_RAD = .010

    def __init__(self, config=None, robot_setup="tighten-fr3"):
        import newton
        import warp as wp
        from newton.solvers import SolverMuJoCo
        from newton._src.solvers.mujoco.constants import SOLREF_MODE_FORCE_SPACE
        self.config = config or Config()
        self.lock = threading.RLock()
        self.builder, self.items, self.materials, self.rig, self.setup = build(
            self.config, robot_setup)
        self.bearing_targets={}
        for name in self.screws:
            target=("sensor_mount_ear",) if name.startswith("sensor_") else (
                ("corner_bracket_top_m4_land",) if name=="frame_top" else
                ("corner_bracket_front_m4_land",))
            self.bearing_targets[name]=np.asarray(
                [i for i,label in enumerate(self.builder.shape_label) if label in target],
                dtype=np.int32)
        labels=self.builder.shape_label
        group=lambda *names:np.asarray([i for i,l in enumerate(labels) if l in names],dtype=np.int32)
        self._rail_shapes=group("horizontal_profile_top","horizontal_profile_web",
                                "horizontal_profile_side","horizontal_profile_slot_lip",
                                "horizontal_profile_lip_edge")
        self._bracket_h=group("corner_bracket_horizontal_leg")
        self._bracket_v=group("corner_bracket_vertical_leg")
        self._post=group("profile_post","profile_post_front_cheek")
        self._interface={"rail_on_bracket_n":0.,"bracket_on_post_face_n":0.,"bracket_on_post_top_n":0.}
        self.model = self.builder.finalize(device=self.config.device)
        # Everything the rail seat carries: rail (+BEF-W4-A), sensor, and the
        # screw chains anchored in them.
        tree={i for i,l in enumerate(self.builder.body_label)
              if l=="horizontal_profile" or l=="WTB4FP_sensor" or l.startswith(("sensor_left_","sensor_right_","frame_top_"))}
        self._rail_tree_bodies=np.asarray(sorted(tree),np.int32)
        self._rail_tree_shapes=np.asarray([i for i,b in enumerate(self.builder.shape_body) if b in tree],np.int32)
        self._rail_weight_n=float(sum(self.builder.body_mass[i] for i in tree))*9.81
        # Who touches the work (mixed: "hold on its own after you let go"):
        # robot links, and any driver a robot link is touching. The sensor
        # group is what the M3s clamp; the frame group is everything the rail
        # seat carries plus the bracket and the forward M4 chain.
        shapes_of=lambda bodies:np.asarray([i for i,b in enumerate(self.builder.shape_body)
                                            if b in bodies],np.int32)
        self._robot_shapes=np.asarray([i for i,l in enumerate(labels)
                                       if l.startswith(("left/","right/"))],np.int32)
        self._driver_shapes={name:shapes_of({item["grip"],item["rotor"]})
                             for name,item in self.drivers.items()}
        body_label=self.builder.body_label
        self._work_groups={
            "sensor":shapes_of({i for i,l in enumerate(body_label) if l=="WTB4FP_sensor" or
                                l.startswith(("sensor_left_","sensor_right_"))}),
            "frame":shapes_of(tree|{i for i,l in enumerate(body_label) if l=="corner_bracket" or
                                    l.startswith("frame_front_")})}
        self._held={group:False for group in self._work_groups}
        self._rail_fixture_fz=0.
        self.model.request_contact_attributes("force")
        self.model.mujoco.solref_mode.fill_(SOLREF_MODE_FORCE_SPACE)
        self.rig.bind(self.model)
        self.joints = self.rig.joints
        self.pipeline = newton.CollisionPipeline(
            self.model, rigid_contact_max=self.config.contact_capacity,
            broad_phase="sap", reduce_contacts=True,
            contact_matching="latest", contact_report=True)
        self.solver = self._make_solver()
        self._map_friction_rows()
        self.state, self.next_state = self.model.state(), self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.next_state.assign(self.state)
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        self.solved_contacts = self.pipeline.contacts()
        self.initial = self.model.state()
        self.initial.assign(self.state)
        self.initial_targets = self.control.joint_target_q.numpy().copy()
        if self.config.fixture_hand_target is not None:
            staged=self.state.joint_q.numpy()
            for spec in self.joints.values():
                self.initial_targets[spec["q"]]=staged[spec["q"]]
            self.control.joint_target_q.assign(self.initial_targets)
        self.targets = self.initial_targets.copy()
        self._setup_arm_motion()
        self.tick = 0
        self.generation = 0
        self.fault = None
        self.renderer = None
        self.trace_callback = None
        self._bracket_origin = self.state.body_q.numpy()[self.items["bracket"],:3].copy()
        self._preload = {x:0. for x in self.screws}
        self._peak_preload = self._preload.copy()
        self._peak_torque = {x:0. for x in self.screws}
        self._reset_tracking()
        self._bit_contacts = {x:0 for x in self.screws}
        self._retained_ticks = {x:0 for x in self.screws}
        self._retained_anchor = {x:None for x in self.screws}
        self._reset_free_readings()
        self.evaluation = self._evaluate()
        self._settle_initial()

    def _settle_initial(self):
        """Run the configured settle, then make that state tick 0."""
        if self.config.task in NEW_TASKS:
            from .presets import settle_initial
            return settle_initial(self)
        ticks=int(round(self.config.initial_settle_s/self.config.dt))
        if not ticks:
            return
        trace,self.trace_callback=self.trace_callback,None
        # Test-only stages (assisted scenes) hold their staged
        # positions through the settle, as the assisted harness does after.
        tq=self.control.joint_target_q.numpy();jq=self.state.joint_q.numpy()
        qstart=self.model.joint_q_start.numpy()
        for j,label in enumerate(self.builder.joint_label):
            if "TEST_ONLY" in label:
                for k in range(int(qstart[j]),int(qstart[j+1])):
                    tq[k]=jq[k]
        self.control.joint_target_q.assign(tq)
        self.initial_targets=tq.copy();self.targets=tq.copy()
        # A secured frame was tightened before the job: during the first half
        # of the settle both frame screws are turned by a torque ramped to the
        # M4 driver setting (the earlier assembly), then released; the thread
        # and under-head friction rows hold what that torque built.
        tighten=[self._screw_dof[n] for n in ("frame_top","frame_front")] \
            if self.config.frame_is_pretightened else []
        # Those screws are being turned (by the earlier assembly), so their
        # self-locking hold is released while the torque acts.
        self._assembly_turning=set(("frame_top","frame_front") if tighten else ())
        base_f=self.control.joint_f.numpy().copy()
        for k in range(ticks):
            if tighten:
                if k>=.6*ticks:
                    self._assembly_turning=set()
                f=base_f.copy()
                ramp=min(1.,k/(.3*ticks))*(k<.6*ticks)
                for dof in tighten:
                    f[dof]=ramp*self.config.m4_tool_limit_nm
                self.control.joint_f.assign(f)
            self.step()
        self.control.joint_f.assign(base_f)
        self._assembly_turning=set()
        self.trace_callback=trace
        self.initial.assign(self.state)
        self._settled_loads=(dict(self._preload),dict(self._thread_load),dict(self._interface))
        self.tick=0
        self._peak_preload=dict(self._preload)
        self._peak_torque={x:0. for x in self.screws}
        self._reset_tracking()
        self._thread_load=dict(self._settled_loads[1])
        self._bit_contacts={x:0 for x in self.screws}
        self._retained_ticks={x:0 for x in self.screws}
        self._retained_anchor={x:None for x in self.screws}
        self._bracket_origin=self.state.body_q.numpy()[self.items["bracket"],:3].copy()
        self._reset_free_readings()
        self.evaluation=self._evaluate()

    def _reset_free_readings(self):
        """mixed: the last hands-off head loads and rail pitch are the
        current ones, and nothing is touching the work."""
        self._held={group:False for group in self._work_groups}
        self._free_load=dict(self._preload)
        self._free_pitch=self._rail_attitude()[0]
        self._task_set_state={}
        if uses_eight(self.config):
            from .task_evaluation import capture_start
            capture_start(self)

    def _reset_tracking(self):
        self._bit_moment={x:0. for x in self.screws}
        self._peak_bit_moment={x:0. for x in self.screws}
        # 10-tick (2.1 ms) moving average: a torque-transducer-like reading.
        self._bit_window={x:[0.]*10 for x in self.screws}
        self._peak_bit_moment_filtered={x:0. for x in self.screws}
        self._release_slip={x:0 for x in self.screws}
        self._clicks={x:0 for x in self.screws}
        # Clicks of the screw's own preset driver; mixed voids them once the
        # hands-off head load stays below the retained floor for 10 readings.
        self._seated_clicks={x:0 for x in self.screws}
        self._subfloor_ticks={x:0 for x in self.screws}
        # Screw angle at its latest seated click (None: not yet read).
        self._click_turn={x:None for x in self.screws}
        self._thread_load={x:0. for x in self.screws}
        self._engaged_driver={x:None for x in self.screws}

    @property
    def drivers(self):
        return {name:item for name,item in self.items.items()
                if isinstance(item,dict) and "clutch_joint" in item}

    @property
    def screws(self):
        return {name: item for name,item in self.items.items()
                if isinstance(item,dict) and "turn_joint" in item}

    def _make_solver(self):
        from newton.solvers import SolverMuJoCo
        return SolverMuJoCo(self.model,use_mujoco_contacts=False,
            integrator="implicitfast",cone="elliptic",solver="newton",
            impratio=self.config.solver_impratio,
            iterations=self.config.solver_iterations,
            nconmax=self.config.contact_capacity,
            njmax=self.config.contact_capacity*3)

    def _setup_arm_motion(self, reference_q=None, reference_qd=None,
                          reference_acceleration=None):
        """Configure the same jerk-limited reference the physical arms accept."""
        reference_q=(self.control.joint_target_q.numpy() if reference_q is None else
                     np.asarray(reference_q))
        reference_qd=(self.control.joint_target_qd.numpy() if reference_qd is None else
                      np.asarray(reference_qd))
        reference_acceleration=(np.zeros(self.model.joint_dof_count)
            if reference_acceleration is None else np.asarray(reference_acceleration))
        limits={
            "fr3":((10.,)*7,(5000.,)*7),
            "panda":((15.,7.5,10.,12.5,15.,20.,20.),
                     (7500.,3750.,5000.,6250.,7500.,10000.,10000.))}
        self._arm_motion={}
        for arm,handle in self.rig.instances.items():
            names=[name for name in self.joints
                   if name.startswith(arm+"/") and "finger" not in name]
            acceleration,jerk=limits[handle.model_id]
            inp,out=InputParameter(len(names)),OutputParameter(len(names))
            inp.current_position=[float(reference_q[self.joints[n]["q"]]) for n in names]
            inp.current_velocity=[float(reference_qd[self.joints[n]["dof"]]) for n in names]
            inp.current_acceleration=[
                float(reference_acceleration[self.joints[n]["dof"]]) for n in names]
            inp.target_position=[float(self.targets[self.joints[n]["q"]]) for n in names]
            inp.max_velocity=[.95*float(self.joints[n]["velocity_limit"]) for n in names]
            inp.max_acceleration=list(acceleration)
            inp.max_jerk=list(jerk)
            inp.synchronization=Synchronization.Time
            moving=(not np.array_equal(inp.current_position,inp.target_position) or
                    any(inp.current_velocity) or any(inp.current_acceleration))
            self._arm_motion[arm]={"names":names,"input":inp,"output":out,
                "generator":Ruckig(len(names),self.config.dt),"moving":moving}

    def reset(self):
        with self.lock:
            self.state.assign(self.initial)
            self.next_state.assign(self.initial)
            self.targets=self.initial_targets.copy()
            self.control.joint_target_q.assign(self.targets)
            self.control.joint_target_qd.assign(
                np.zeros_like(self.control.joint_target_qd.numpy()))
            self.control.joint_f.assign(np.zeros_like(self.control.joint_f.numpy()))
            self.pipeline.reset_contact_matching()
            self.solver=self._make_solver()
            self.contacts=self.pipeline.contacts()
            self.solved_contacts=self.pipeline.contacts()
            self.tick=0
            self.fault=None
            self.generation+=1
            loads=getattr(self,"_settled_loads",None)
            self._preload=dict(loads[0]) if loads else {x:0. for x in self.screws}
            self._interface=dict(loads[2]) if loads else {k:0. for k in self._interface}
            self._map_friction_rows()
            self._peak_preload=self._preload.copy()
            self._peak_torque={x:0. for x in self.screws}
            self._reset_tracking()
            if loads:self._thread_load=dict(loads[1])
            self._bit_contacts={x:0 for x in self.screws}
            self._retained_ticks={x:0 for x in self.screws}
            self._retained_anchor={x:None for x in self.screws}
            self._setup_arm_motion()
            self._reset_free_readings()
            self.evaluation=self._evaluate()
            return self.get_state()

    def act(self, targets, *, ticks, guard=None):
        if type(ticks) is not int or ticks<=0:
            raise ValueError("ticks must be a positive integer")
        with self.lock:
            if self.fault:
                raise RuntimeError(f"Environment faulted; reset required: {self.fault}")
            self.rig.validate(targets)
            pending=self.targets.copy()
            for name,value in targets.items():
                pending[self.joints[name]["q"]]=value
            self.targets=pending
            for _ in range(ticks):
                self.step()
                if guard and guard(self):
                    break
            return self.get_state()

    def _map_friction_rows(self):
        """Newton DOF -> MuJoCo DOF column of the solver's friction-loss array."""
        mapping=self.solver.mjc_dof_to_newton_dof.numpy()[0]
        self._mjc_dof={int(n):i for i,n in enumerate(mapping) if n>=0}
        self._base_frictionloss=self.solver.mjw_model.dof_frictionloss.numpy().copy()
        vstarts=self.model.joint_qd_start.numpy()
        self._screw_dof={name:int(vstarts[item["turn_joint"]]) for name,item in self.screws.items()}
        self._aim_dof=(None if self.config.fixture_hold_sensor else
                       int(vstarts[self.items["sensor_aim_joint"]]))
        self._rail_dof=int(vstarts[self.items["rail_joint"]])
        self._bracket_dof=[int(vstarts[self.items["bracket_joint"]])+k
                           for k in range(3 if self.config.bracket_vertical_play else 2)]
        self._interface_capacity={}
        # Detent clutches: MuJoCo joint index (for the runtime lock range) and
        # DOF column (roll-over friction row, constraint torque read-back).
        jmap=self.solver.mjc_jnt_to_newton_jnt.numpy()[0]
        mjc_jnt={int(n):i for i,n in enumerate(jmap) if n>=0}
        self._jnt_range=self.solver.mjw_model.jnt_range.numpy().copy()
        qstart=self.model.joint_q_start.numpy()
        self._screw_lock={name:{"jnt":mjc_jnt[item["turn_joint"]],"q":int(qstart[item["turn_joint"]]),
                                "qd":int(vstarts[item["turn_joint"]]),
                                "slide_jnt":mjc_jnt[item["slide_joint"]],"slide_q":int(qstart[item["slide_joint"]]),
                                "locked_at":None,"slide_at":None}
                          for name,item in self.screws.items()}
        # Coulomb stick on the clamp coordinates (rail slide and yaw, bracket
        # x and z). Friction-loss rows alone creep viscously below their
        # capacity (the clamped rail crept at ~1.8 um/s per N, so a "breakaway"
        # ramp measured creep, not mu). Stuck: a zero-width stop at the stuck
        # position carries the load; the force it carries is read from the
        # solver's own joint-limit constraint row (checked: a stuck rail under
        # 150 N showed limit 118.7 N + friction row 31.3 N). Slips at the
        # capacity, sticks again once stopped.
        jntid=self.solver.mjw_model.dof_jntid.numpy()
        rq=int(qstart[self.items["rail_joint"]]); bq=int(qstart[self.items["bracket_joint"]])
        stick=[("rail_along_slot_n",self._rail_dof,rq,1e-4),
               ("rail_yaw_nm",self._rail_dof+3,rq+3,1e-3),
               ("bracket_along_x_n",self._bracket_dof[0],bq,1e-4)]
        if len(self._bracket_dof)>2:
            stick.append(("bracket_along_z_n",self._bracket_dof[2],bq+2,1e-4))
        self._clamp_stick=[{"capacity":key,"dof":dof,"q":q,"jnt":int(jntid[self._mjc_dof[dof]]),
                            "stick_speed":v,"locked_at":None} for key,dof,q,v in stick]
        self._head_touched={name:False for name in self.screws}
        self._assembly_turning=set()
        self._clutch={}
        for name,item in self.drivers.items():
            j=item["clutch_joint"]
            self._clutch[name]={"jnt":mjc_jnt[j],"col":self._mjc_dof[int(vstarts[j])],
                "dof":int(vstarts[j]),
                "q":int(self.model.joint_q_start.numpy()[j]),"lock":0.,"slipping":False,
                "climbing":False,"clicks":0,"torque_nm":0.,"click_ticks":[]}

    def thread_friction_per_newton(self, name):
        """Thread friction torque per newton of axial thread load, N m/N:
        (d2/2)[tan(lambda+atan(mu_t/cos30)) - tan(lambda)]. The ideal
        F p/(2 pi) helix work is supplied by the native helical equality."""
        item=self.screws[name]
        d=item["diameter_m"];p=item["pitch_m"]
        d2=.003545 if d>.003 else .002675
        lead=math.atan(p/(math.pi*d2))
        return (d2/2)*(math.tan(lead+math.atan(self.config.thread_mu/math.cos(math.pi/6)))
                       -math.tan(lead))

    def head_mu(self, name):
        return (self.config.m3_bearing_mu if name.startswith("sensor_") else
                self.config.aluminium_bearing_mu if name=="frame_front" else
                self.config.bearing_mu)

    def head_radius(self, name):
        item=self.screws[name]
        return ((.007 if name=="frame_top" else .006) if item["diameter_m"]>.003 else .0045)/2

    def friction_capacity(self, name, F_thread, F_bearing=None):
        """Static thread + under-head torque that the screw row may carry, N m.

        Thread friction from the solved axial thread load (so a prestarted
        screw is self-locking under any axial push: its lead angle is below
        the friction angle); under-head friction from the solved head load.
        """
        F_bearing=F_thread if F_bearing is None else F_bearing
        return max(0.,F_thread*self.thread_friction_per_newton(name)+
                   F_bearing*self.head_mu(name)*self.head_radius(name))

    def _limit_row_forces(self):
        """Joint-limit constraint forces of the last solve, by MuJoCo joint
        (mujoco_warp ConstraintType.LIMIT_JOINT = 3; row id = joint id)."""
        d=self.solver.mjw_data
        n=int(d.nefc.numpy()[0])
        typ=d.efc.type.numpy().reshape(-1)[:n]
        ids=d.efc.id.numpy().reshape(-1)[:n]
        force=d.efc.force.numpy().reshape(-1)[:n]
        out={}
        for t,i,f in zip(typ,ids,force):
            if int(t)==3:
                out[int(i)]=out.get(int(i),0.)+float(f)
        return out

    def _friction_wrenches(self):
        """Set each tick's implicit friction-loss capacities from solved loads.

        Screw rows: loaded thread and under-head friction (above), static and
        force-bounded in the coupled solve; the reaction goes to the nut
        anchor (sensor clamping strip, rail T-nut or post T-nut). Sensor
        pivot row: both M3s pull the housing and the clamping strip onto the
        two plate faces, so each face resists in-plane rotation with Coulomb
        friction at the screw's radius about the pivot - the pivot lobe
        annulus (1.7-3.5 mm, r_eff 2.70 mm) and the arc-slot screw at 9 mm:
        T = 2 mu_c (F_L 2.70 mm + F_R 9.0 mm). All other in-plane clamp loads
        (rail on bracket, bracket on post) are solved contact friction.
        """
        out=np.zeros((self.model.body_count,6),np.float32)
        loss=self._base_frictionloss.copy()
        for name in self.screws:
            F=self._preload[name]
            capacity=self.friction_capacity(name,self._thread_load[name],F)
            column=self._mjc_dof[self._screw_dof[name]]
            loss[0,column]=max(loss[0,column],capacity)
            self._peak_torque[name]=max(self._peak_torque[name],F*self.torque_per_newton(name))
        # Frame clamp interfaces (Coulomb, solved normal loads). The rail
        # slides on the bracket leg and its washer slides under it; the
        # bracket slides on the post face and top and under the front head.
        # Along y (toward/away from the post face) only the top interface
        # shears: the face and front-head loads are normal to that motion.
        mu=self.config.clamp_mu
        n=self._interface
        rail_x=mu*n["rail_on_bracket_n"]+self.head_mu("frame_top")*self._preload["frame_top"]
        face=mu*n["bracket_on_post_face_n"]+self.head_mu("frame_front")*self._preload["frame_front"]
        top=mu*n["bracket_on_post_top_n"]
        # Rail yaw about the leg's far edge: the clamp load is concentrated
        # around the T-nut, 13.5 mm from that edge (drawing), and both the
        # rail-on-leg and the head/washer interfaces shear as it yaws. The
        # loose rail's seat load sits at the edge itself and resists nothing.
        rail_yaw=((mu+self.head_mu("frame_top"))*self._preload["frame_top"]*
                  self.config.rail_clamp_lever_m)
        rows=[(self._rail_dof,rail_x),(self._rail_dof+3,rail_yaw),
              (self._bracket_dof[0],face+top),(self._bracket_dof[1],top)]
        if len(self._bracket_dof)>2:
            rows.append((self._bracket_dof[2],face))
        for dof,capacity in rows:
            loss[0,self._mjc_dof[dof]]+=capacity
        self._interface_capacity={"rail_along_slot_n":rail_x,"rail_yaw_nm":rail_yaw,
            "bracket_along_x_n":face+top,"bracket_along_y_n":top,"bracket_along_z_n":face}
        if self._aim_dof is not None:
            column=self._mjc_dof[self._aim_dof]
            loss[0,column]+=(self.config.sensor_face_mu+self.config.sensor_strip_mu)*(
                self._preload["sensor_left"]*.0027+self._preload["sensor_right"]*.009)
        # Detent clutches: locked = zero-width range at the lock angle; during
        # a notch the range opens one notch forward (never backward) and the
        # cam's reduced roll-over torque is a friction row.
        rng=self._jnt_range.copy()
        jq=self.state.joint_q.numpy()
        jqd=self.state.joint_qd.numpy()
        for name,lock in self._screw_lock.items():
            if name in self._assembly_turning or not self.config.screw_self_lock:
                hold=False
            elif not self._head_touched.get(name,False):
                # Self-locking thread: nothing on the head, nothing turns it.
                hold=True
            else:
                # Coulomb stick with a bit on the head: held while the solved
                # bit moment is below the thread + under-head capacity; it
                # slips at the capacity and sticks again once it has stopped
                # (MuJoCo's friction-loss row alone creeps below capacity:
                # 2.3 mrad/s at 94 % of it, measured).
                # Drive moment: the solved bit moment averaged over 2.1 ms
                # (the evaluator's window); single-tick contact noise
                # otherwise broke the stick at 94 % of the capacity.
                capacity=self.friction_capacity(name,self._thread_load[name],self._preload[name])
                drive=sum(self._bit_window[name])/len(self._bit_window[name])
                if lock["locked_at"] is not None:
                    hold=drive<capacity
                else:
                    hold=(abs(float(jqd[lock["qd"]]))<self.config.screw_stick_speed_rad_s and
                          drive<capacity)
            if not hold:
                lock["locked_at"]=lock["slide_at"]=None
            else:
                if lock["locked_at"] is None:
                    lock["locked_at"]=float(jq[lock["q"]])
                    lock["slide_at"]=float(jq[lock["slide_q"]])
                rng[0,lock["jnt"]]=(lock["locked_at"],lock["locked_at"])
                rng[0,lock["slide_jnt"]]=(lock["slide_at"],lock["slide_at"])
        joint_f=None
        if self.config.clamp_stick:
            limit_force=self._limit_row_forces() if any(
                st["locked_at"] is not None for st in self._clamp_stick) else {}
            for st in self._clamp_stick:
                capacity=self._interface_capacity[st["capacity"]]
                q=float(jq[st["q"]])
                if st["locked_at"] is not None:
                    if abs(limit_force.get(st["jnt"],0.))>=capacity:
                        st["locked_at"]=None
                elif abs(float(jqd[st["dof"]]))<st["stick_speed"]:
                    # Never inside a hard stop: sticking at a penetrated
                    # position would ratchet the coordinate through it.
                    lo,hi=self._jnt_range[0,st["jnt"]]
                    st["locked_at"]=min(max(q,float(lo)),float(hi))
                if st["locked_at"] is not None:
                    rng[0,st["jnt"]]=(st["locked_at"],st["locked_at"])
                    # Stuck: static friction is the stop alone. The kinetic
                    # friction row acts only while sliding; left on, its
                    # viscous regularisation delayed the stop's load by
                    # ~0.26 s (late breakaway, short pushes ignored).
                    if (self.config.stick_rows_off=="all" or
                            (self.config.stick_rows_off=="linear" and not st["capacity"].endswith("_nm"))):
                        col=self._mjc_dof[st["dof"]]
                        loss[0,col]=self._base_frictionloss[0,col]
        for name,c in self._clutch.items():
            limit=self.drivers[name]["limit_nm"]
            upper=c["lock"]+(self.config.clutch_notch_rad if c["slipping"] else
                             self.config.clutch_climb_rad if c["climbing"] else 0.)
            rng[0,c["jnt"]]=(c["lock"],upper)
            loss[0,c["col"]]=(self.config.clutch_rollover_fraction*limit if c["slipping"] else 0.)
            # Climbing the detent ramp: the preloaded cam spring pushes back
            # with the setting torque; below it the cam falls back and relocks.
            if joint_f is None:
                joint_f=self.control.joint_f.numpy()
            joint_f[c["dof"]]=-limit if c["climbing"] else 0.
        if joint_f is not None:
            self.control.joint_f.assign(joint_f)
        self.solver.mjw_model.jnt_range.assign(rng)
        self.solver.mjw_model.dof_frictionloss.assign(loss)
        return out

    def _bearing_loads(self):
        """Solved axial head/washer loads, bit-socket moments and engagement."""
        self.solver.update_contacts(self.solved_contacts)
        c=self.solved_contacts
        n=int(c.rigid_contact_count.numpy()[0])
        ids0=c.rigid_contact_shape0.numpy()[:n]
        ids1=c.rigid_contact_shape1.numpy()[:n]
        forces=c.force.numpy()[:n]
        if self.config.task=="mixed" or uses_eight(self.config):
            self._held=self._work_contact(ids0,ids1)
        loads={name:0. for name in self.screws}
        # Net external axial force on each screw chain (head, shank, bearing
        # points): the load its thread carries. Contact force acts on shape 1
        # (+) and on shape 0 (-).
        self._thread_load={}
        for name,item in self.screws.items():
            own=np.asarray([item["socket_shape"],item["shank_shape"],*item["bearing_shape"]],np.int32)
            on1=np.isin(ids1,own);on0=np.isin(ids0,own)
            f=forces[on1,:3].sum(axis=0)-forces[on0,:3].sum(axis=0)
            self._thread_load[name]=abs(float(np.dot(f,item["axis_world"])))
        def pair(g0,g1,axis):
            m=(np.isin(ids0,g0)&np.isin(ids1,g1))|(np.isin(ids0,g1)&np.isin(ids1,g0))
            return float(np.abs(forces[m,axis]).sum())
        bracket=np.concatenate((self._bracket_h,self._bracket_v))
        # The rail's seat on the leg is the rail joint's one-sided stops
        # (contact filtered). Its normal load, for the slot friction row, is
        # the rail's statics: weight + the upward-M4 washer pull + every other
        # downward force on the rail (contacts, e.g. fingers or a bit, and
        # the assisted hand). A penetration-based read-back of a force-space
        # limit row is not a force measurement in this solver.
        on1=np.isin(ids1,self._rail_tree_shapes);on0=np.isin(ids0,self._rail_tree_shapes)
        # Pairs inside the rail subtree cancel; the washer pull on the
        # upward-M4 bearing points is an external contact and counts. The
        # reported contact force acts on shape 0 (checked: the drooped loose
        # rail's seat then equals its weight plus the washer pull).
        fz=float(forces[on0&~on1,2].sum()-forces[on1&~on0,2].sum())
        seat=max(0.,self._rail_weight_n-fz-self._rail_fixture_fz)
        self._interface={"rail_on_bracket_n":seat,
                         "bracket_on_post_face_n":pair(self._bracket_v,self._post,1),
                         "bracket_on_post_top_n":pair(self._bracket_h,self._post,2)}
        drivers=self.drivers
        bit_of={item["bit_shape"]:name for name,item in drivers.items()}
        bits=np.asarray(list(bit_of),dtype=np.int32)
        engaged=np.isin(ids0,bits)|np.isin(ids1,bits)
        points=None
        body_q=None
        for name,item in self.screws.items():
            bearing=item["bearing_shape"]
            targets=self.bearing_targets[name]
            mask=(np.isin(ids0,bearing)&np.isin(ids1,targets)) | \
                 (np.isin(ids1,bearing)&np.isin(ids0,targets))
            if np.any(mask):
                loads[name]=float(np.abs(forces[mask,:3] @ np.asarray(
                    item["axis_world"],dtype=np.float32)).sum())
            socket=item["socket_shape"]
            self._head_touched[name]=bool(np.any((ids0==socket)|(ids1==socket)))
            m0=engaged&(ids1==socket)
            m1=engaged&(ids0==socket)
            count=int(np.count_nonzero(m0|m1))
            self._bit_contacts[name]=count
            self._engaged_driver[name]=None
            self._bit_moment[name]=0.
            if not count:
                self._bit_window[name]=self._bit_window[name][1:]+[0.]
                continue
            other=np.concatenate((ids0[m0],ids1[m1]))
            self._engaged_driver[name]=bit_of[int(other[0])]
            # Moment about the screw axis that the bit applies to the socket,
            # from the solved contact force and couple. Contact witnesses are
            # bit-shape-local; rotate the arm into world with the rotor pose.
            if points is None:
                points=(c.rigid_contact_point0.numpy()[:n],c.rigid_contact_point1.numpy()[:n])
                body_q=self.next_state.body_q.numpy()
            rotor=drivers[self._engaged_driver[name]]["rotor"]
            rot=Rotation.from_quat(body_q[rotor,3:])
            f=np.concatenate((forces[m0,:3],-forces[m1,:3]))
            tq=np.concatenate((forces[m0,3:],-forces[m1,3:]))
            arm=rot.apply(np.concatenate((points[0][m0],points[1][m1])))
            moment=float(np.dot(np.sum(np.cross(arm,f)+tq,axis=0),item["axis_world"]))
            self._bit_moment[name]=abs(moment)
            self._peak_bit_moment[name]=max(self._peak_bit_moment[name],abs(moment))
            window=self._bit_window[name]=self._bit_window[name][1:]+[abs(moment)]
            self._peak_bit_moment_filtered[name]=max(self._peak_bit_moment_filtered[name],
                                                     sum(window)/len(window))
        return loads

    def _work_contact(self, ids0, ids1):
        """Work groups touched this tick by a robot link or by a driver that a
        robot link is touching (a tool lying on its own does not count)."""
        robot0=np.isin(ids0,self._robot_shapes);robot1=np.isin(ids1,self._robot_shapes)
        hands=[self._robot_shapes]
        for shapes in self._driver_shapes.values():
            if np.any((np.isin(ids0,shapes)&robot1)|(np.isin(ids1,shapes)&robot0)):
                hands.append(shapes)
        hands=np.concatenate(hands)
        h0=np.isin(ids0,hands);h1=np.isin(ids1,hands)
        return {group:bool(np.any((h0&np.isin(ids1,shapes))|(h1&np.isin(ids0,shapes))))
                for group,shapes in self._work_groups.items()}

    def _track_clutches(self):
        """Detent clutch state and clicks per screw.

        Locked: transmitted torque is the lock stiffness times the deflection
        into the forward stop (tightening slip is +q). At the setting the cam
        releases; when the coordinate reaches the
        next notch it relocks there and that is one click. A screw counts the
        clicks of whichever driver's bit is in its socket at the time.
        """
        jq=self.state.joint_q.numpy()
        k=self.config.clutch_lock_stiffness_nm_rad
        for name,c in self._clutch.items():
            angle=float(jq[c["q"]])
            # Torque through the locked cam = lock stiffness x deflection into
            # the forward stop (checked against the solved bit-socket moment).
            limit=self.drivers[name]["limit_nm"]
            c["torque_nm"]=(self.config.clutch_rollover_fraction*limit if c["slipping"] else
                            limit if c["climbing"] else k*(angle-c["lock"]))
            if not c["slipping"] and not c["climbing"]:
                if c["torque_nm"]>=limit:
                    # Setting reached: the cam starts up its ramp.
                    c["climbing"]=True
                    c["climb_onset"]={"tick":self.tick,"torque_nm":c["torque_nm"],"angle_rad":angle}
            elif c["climbing"]:
                if angle>=c["lock"]+self.config.clutch_climb_rad-1e-4:
                    # Over the top: released; it rolls on to the next notch.
                    c["climbing"]=False
                    c["slipping"]=True
                    c["release_onset"]=dict(c["climb_onset"],over_top_tick=self.tick)
                    c["releases"]=c.get("releases",0)+1
                elif angle<=c["lock"]+.5*limit/k:
                    # Torque fell below the setting before the top: back in.
                    c["climbing"]=False
                    c["fallbacks"]=c.get("fallbacks",0)+1
            elif angle>=c["lock"]+self.config.clutch_notch_rad-1e-3:
                c["lock"]+=self.config.clutch_notch_rad
                c["slipping"]=False
                c["clicks"]+=1
                c["click_ticks"].append(self.tick)
                for screw,driver in self._engaged_driver.items():
                    if driver==name:
                        self._clicks[screw]+=1
                        if limit==self.items[screw]["limit_nm"]:
                            self._seated_clicks[screw]+=1
                            self._subfloor_ticks[screw]=0
                            self._click_turn[screw]=None
            c["angle_rad"]=angle
        for screw,driver in self._engaged_driver.items():
            self._release_slip[screw]=self._clicks[screw]

    def step(self, *, fixture_wrenches=None):
        """One physics tick; fixture_wrenches is diagnostic only, never GAR policy."""
        import newton
        with self.lock:
            if self.fault:
                raise RuntimeError(self.fault)
            self.state.clear_forces()
            forces=self._friction_wrenches()
            if fixture_wrenches is not None:
                if not self.config.fixture_mode:
                    raise PermissionError("Assisted wrenches require fixture_mode")
                fixture=np.asarray(fixture_wrenches,np.float32)
                if fixture.shape!=forces.shape or not np.isfinite(fixture).all():
                    raise ValueError("Finite (body_count,6) assisted wrench required")
                forces+=fixture
                self._rail_fixture_fz=float(fixture[self._rail_tree_bodies,2].sum())
            else:
                self._rail_fixture_fz=0.
            self.state.body_f.assign(forces)
            current=self.control.joint_target_q.numpy()
            velocity=self.control.joint_target_qd.numpy()
            arm_names={name for motion in self._arm_motion.values()
                       for name in motion["names"]}
            for arm,motion in self._arm_motion.items():
                target=[float(self.targets[self.joints[n]["q"]])
                        for n in motion["names"]]
                inp=motion["input"]
                if motion["moving"] or target!=list(inp.target_position):
                    inp.target_position=target
                    result=motion["generator"].update(inp,motion["output"])
                    if result not in (Result.Working,Result.Finished):
                        raise RuntimeError(f"{arm} motion generator: {result}")
                    motion["output"].pass_to_input(inp)
                    for n,q,qd in zip(motion["names"],inp.current_position,
                                      inp.current_velocity):
                        current[self.joints[n]["q"]]=q
                        velocity[self.joints[n]["dof"]]=qd
                    motion["moving"]=result==Result.Working
            for name,spec in self.joints.items():
                if name in arm_names:
                    continue
                q=spec["q"]
                delta=spec["velocity_limit"]*self.config.dt
                current[q]+=np.clip(self.targets[q]-current[q],-delta,delta)
            self.control.joint_target_q.assign(current)
            self.control.joint_target_qd.assign(velocity)
            try:
                self.pipeline.collide(self.state,self.contacts)
                self.solver.step(self.state,self.next_state,self.control,
                                 self.contacts,self.config.dt)
                self._validate(self.next_state)
                self._preload=self._bearing_loads()
                for name,value in self._preload.items():
                    self._peak_preload[name]=max(self._peak_preload[name],value)
                self.state,self.next_state=self.next_state,self.state
                self._track_clutches()
                self.tick+=1
                self._track_retention()
                self.evaluation=self._evaluate()
                if self.trace_callback:self.trace_callback(self)
            except Exception as exc:
                self.fault=f"{type(exc).__name__}: {exc}"
                raise

    def _track_retention(self):
        """Per-screw retention timers after a tick. mixed also keeps the last
        hands-off head loads (lowered, never raised, by readings under a bit
        or hand) and rail pitch, and voids a screw's clicks once its hands-off
        load has read below the floor for 10 consecutive readings, or once
        the screw has turned back past its click angle (at any reading)."""
        q=self.state.joint_q.numpy()
        starts=self.model.joint_q_start.numpy()
        mixed=self.config.task=="mixed" or uses_eight(self.config)
        for name,item in self.screws.items():
            advance=float(q[int(starts[item["slide_joint"]])])
            angle=float(q[int(starts[item["turn_joint"]])])
            threshold=self.minimum_load(name)
            minimum_advance=.0004 if name.startswith("sensor") else .00045
            # mixed: a reading counts only with no bit in the socket and no
            # hand (robot link or held driver) on what the screw clamps.
            free=self._bit_contacts[name]==0 and not (
                mixed and self._held["sensor" if name.startswith("sensor_") else "frame"])
            if mixed:
                if free:
                    self._free_load[name]=self._preload[name]
                    below=self._preload[name]<threshold
                    self._subfloor_ticks[name]=self._subfloor_ticks[name]+1 if below else 0
                    if self._subfloor_ticks[name]>=self.MIXED_RETENTION_TICKS:
                        self._seated_clicks[name]=0
                else:
                    # Under a bit or hand a reading may show lost load, never
                    # gained load (pushing or holding cannot inflate it).
                    self._free_load[name]=min(self._free_load[name],self._preload[name])
                if self._seated_clicks[name]>=1:
                    if self._click_turn[name] is None:
                        self._click_turn[name]=angle
                    elif angle<self._click_turn[name]-self.MIXED_BACKOFF_TOLERANCE_RAD:
                        self._seated_clicks[name]=0
                        self._click_turn[name]=None
            loaded=(self._preload[name]>=threshold and
                    advance>=minimum_advance and free)
            if not loaded:
                self._retained_ticks[name]=0
                self._retained_anchor[name]=None
            elif (self._retained_anchor[name] is None or
                  abs(angle-self._retained_anchor[name])>.001):
                self._retained_ticks[name]=1
                self._retained_anchor[name]=angle
            else:
                self._retained_ticks[name]+=1
        if mixed and not self._held["frame"]:
            self._free_pitch=self._rail_attitude()[0]
        if uses_eight(self.config):
            from .task_evaluation import track_free_rail
            track_free_rail(self)

    def _validate(self,state):
        for key in ("body_q","body_qd","joint_q","joint_qd"):
            a=getattr(state,key).numpy()
            if not np.isfinite(a).all():
                raise FloatingPointError(f"Nonfinite {key} at tick {self.tick+1}")
        q=state.body_q.numpy()[:,:3]
        v=state.body_qd.numpy()
        if np.max(np.abs(q))>100 or np.max(np.abs(v[:,:3]))>100 or np.max(np.abs(v[:,3:]))>10000:
            raise FloatingPointError(f"Runaway body at tick {self.tick+1}")
        joints=state.joint_q.numpy()
        starts=self.model.joint_q_start.numpy()
        slot=float(joints[int(starts[self.builder.joint_label.index("horizontal_profile_slot_travel")])])
        if abs(slot)>.0035:
            raise FloatingPointError(f"Rail left its captive slot at tick {self.tick+1}: {slot} m")
        for name,item in self.screws.items():
            slide=float(joints[int(starts[item["slide_joint"]])])
            turn=float(joints[int(starts[item["turn_joint"]])])
            residual=slide-item["pitch_m"]*turn/(2*math.pi)
            upper=.00255 if name.startswith("sensor_") else .00205
            if not (-.00015<=slide<=upper and abs(residual)<=.00010):
                raise FloatingPointError(f"{name} violated helical/axial bounds at tick {self.tick+1}: "
                                         f"slide={slide}, residual={residual}")

    def minimum_load(self, name):
        """Retained load floor: 80% of the load the specified preset driver
        develops when it releases (release torque / equation torque per N)."""
        item=self.screws[name]
        return .8*item["limit_nm"]/self.torque_per_newton(name)

    def torque_per_newton(self, name):
        """ISO 16047-style tightening torque per newton of preload:
        (d2/2) tan(lambda + atan(mu_t/cos30)) + mu_b D_b/2."""
        item=self.screws[name]
        d=item["diameter_m"];p=item["pitch_m"]
        d2=.003545 if d>.003 else .002675
        lead=math.atan(p/(math.pi*d2))
        return ((d2/2)*math.tan(lead+math.atan(self.config.thread_mu/math.cos(math.pi/6)))
                +self.head_mu(name)*self.head_radius(name))

    def _optical(self):
        body=self.state.body_q.numpy()[self.items["sensor"]]
        r=Rotation.from_quat(body[3:])
        start=body[:3]+r.apply(self.items["sensor_optic_local_m"])
        direction=r.apply((0,-1,0))
        if direction[1]>=-.01:
            return {"detected":False,"range_m":None,"spot_error_m":None}
        face=self.items["flag_face_y_m"]
        distance=(face-start[1])/direction[1]
        hit=start+distance*direction
        target_x=float(self.state.body_q.numpy()[self.items["flag"],0])
        target_z=self.items["flag_center_z_m"]
        error=float(np.linalg.norm(hit[[0,2]]-np.array([target_x,target_z])))
        # Other tasks: the whole nominal 4.2 mm spot on the 10 x 44 mm face.
        # mixed: most of the spot on the face (its centre on it). A
        # background-suppression sensor switches when the near-range return
        # outweighs the far one, so a spot 95-99.9 % on a white flag at 130 mm
        # is detected; no partial-coverage switching curve is sourced.
        margin=0. if self.config.task=="mixed" or uses_eight(self.config) else .0021
        detected=(.04<=distance<=.14 and
                  abs(hit[0]-target_x)<=.005-margin and
                  abs(hit[2]-target_z)<=.022-margin)
        return {"detected":bool(detected),"range_m":float(distance),
                "spot_error_m":error,"intersection_m":hit.tolist()}

    def _evaluate(self):
        jq=self.state.joint_q.numpy()
        body_q=self.state.body_q.numpy()
        starts=self.model.joint_q_start.numpy()
        metrics={}
        eight=uses_eight(self.config)
        mixed=self.config.task=="mixed" or eight
        for name,item in self.screws.items():
            slide=float(jq[int(starts[item["slide_joint"]])])
            turn=float(jq[int(starts[item["turn_joint"]])])
            floor=self.minimum_load(name)
            released=(self._seated_clicks if mixed else self._clicks)[name]>=1
            retained=(self._retained_ticks[name]>=self.MIXED_RETENTION_TICKS if mixed else
                      self._retained_ticks[name]*self.config.dt>=.25)
            metrics[name]={"turn_rad":turn,"advance_m":slide,
                "pitch_residual_m":slide-item["pitch_m"]*turn/(2*math.pi),
                "bearing_normal_force_n":self._preload[name],
                "minimum_retained_load_n":floor,
                "bit_socket_contact_count":self._bit_contacts[name],
                "bit_socket_moment_nm":self._bit_moment[name],
                "peak_bit_socket_moment_nm":self._peak_bit_moment[name],
                "peak_bit_socket_moment_2ms_nm":self._peak_bit_moment_filtered[name],
                "clutch_clicks_while_engaged":self._clicks[name],
                "axial_thread_load_n":self._thread_load[name],
                "driver_released_on_screw":bool(released),
                "post_disengagement_retention_s":self._retained_ticks[name]*self.config.dt,
                "retained_after_disengagement":retained,
                "peak_bearing_normal_force_n":self._peak_preload[name],
                "estimated_total_torque_nm":self._peak_torque[name]}
            if mixed:
                metrics[name].update(preset_driver_clicks_since_seated=self._seated_clicks[name],
                                     hands_off_load_n=self._free_load[name])
        if eight:
            from .task_evaluation import evaluate
            return evaluate(self,metrics)
        need={"sensor":("sensor_left","sensor_right"),
              "aim":("sensor_left","sensor_right"),
              "frame":("frame_front","frame_top"),
              "mixed":tuple(metrics),"support":("frame_front","frame_top"),
              "access":("frame_top",)}[self.config.task]
        secured=all(metrics[n]["retained_after_disengagement"] and
                    metrics[n]["driver_released_on_screw"] for n in need)
        optical=self._optical()
        need_optical=self.config.task in ("sensor","aim","mixed")
        rail_offset=float(body_q[self.items["rail"],0]-.14)
        # Rail attitude relative to the bracket's reference (level, square):
        # rotation about y (nose down +) and about z.
        pitch,yaw=self._rail_attitude(body_q)
        level=abs(pitch)<=self.config.rail_level_tolerance_rad
        square=level and abs(yaw)<=self.config.rail_level_tolerance_rad
        # mixed asks for the rail level only: its slot position and yaw (held
        # within 6.15 mrad by the bracket key) are not part of that job. With
        # the flag required at its home, detection there covers their effect
        # on the inspection head (the flag cannot be moved to meet the beam).
        rail_ok=((abs(rail_offset)<=.0005 and square) if self.config.task=="support" else
                 (abs(rail_offset)<=.001 and square) if self.config.task=="frame" else
                 level if mixed else True)
        # SICK W4F fixing-screw maximum, checked on the measured bit moment.
        torque_ok=all(metrics[n]["peak_bit_socket_moment_2ms_nm"]<=.4
                      for n in ("sensor_left","sensor_right"))
        # mixed: "the assembly must hold on its own after you let go", and the
        # reference flag "back at its home position".
        hands_off=not (mixed and self._held["frame"])
        flag_offset=(float(jq[int(starts[self.items["flag_joint"]])]) if mixed else 0.)
        flag_home=abs(flag_offset)<=self.MIXED_FLAG_HOME_BAND_M
        success=bool(secured and (optical["detected"] or not need_optical)
                     and rail_ok and torque_ok and hands_off and flag_home and self.fault is None)
        extra={}
        if mixed:
            score,extra=self._mixed_score(metrics,optical,success,flag_offset)
            extra.update(assembly_hands_off=hands_off,work_touched=dict(self._held),
                         flag_offset_m=flag_offset,flag_at_home=flag_home)
        else:
            score=sum(min(1.,metrics[n]["bearing_normal_force_n"]/metrics[n]["minimum_retained_load_n"]) *
                      min(1.,metrics[n]["post_disengagement_retention_s"]/.25) *
                      (1. if metrics[n]["driver_released_on_screw"] else .5)
                      for n in need)/len(need)
            if need_optical:score*=float(optical["detected"])
            score*=float(rail_ok and torque_ok)
        return {"task":self.config.task,"task_contract":contract_for(self.config.task,self.config.task_contract),
                "success":success,"score":score,
                "required_fasteners":need,"fasteners":metrics,
                "optical":optical,"rail_offset_m":rail_offset,
                "rail_slide_m":float(jq[int(starts[self.builder.joint_label.index(
                    "horizontal_profile_slot_travel")])]),
                "bracket_offset_m":(body_q[self.items["bracket"],:3]-
                                    np.asarray(self._bracket_origin)).tolist(),
                "rail_pitch_rad":pitch,"rail_yaw_rad":yaw,
                "rail_in_reference":rail_ok,
                "interface_normal_loads_n":dict(self._interface),
                "interface_friction_capacity":dict(getattr(self,"_interface_capacity",{})),
                "sensor_measured_torque_below_max":torque_ok,
                **extra,
                "evaluation_scope":self.MIXED_SCOPE if mixed else "Each required screw: its preset driver's detent clutch clicked (completed at least one notch at the setting) while the bit was in that socket; afterwards at least 80% of the equation load at that torque, carried for 0.25 s with no bit contact and under 0.001 rad angle drift. M3 solved bit moment (2.1 ms average) never above SICK's 0.4 N m. Frame tasks: rail at its slot reference and level and square within the tolerance. Geometric optical ray. Privileged truth hidden from policy"}

    MIXED_SCOPE=("mixed. Success: each of the four screws has a click of its own preset driver's detent clutch "
        "(green 0.35 N m on the M3s, yellow 0.60 N m on the M4s) with the bit in that socket, not voided since by 10 "
        "consecutive hands-off head-load readings below the floor or by the screw turning back more than 0.01 rad "
        "past its angle at that click (at any reading, bit in or not); and at least 80% of the equation load at that "
        "setting, with helical advance, held for the last 10 ticks (2.1 ms) under 0.001 rad angle drift with no "
        "bit in the socket and no robot link or robot-held driver touching what the screw clamps. M3 solved bit "
        "moment (2.1 ms average, either sense) never above SICK's 0.4 N m. Rail level within 3 mrad of pitch "
        "(rail_in_reference means level here); no separate slot-position or yaw gate (the bracket key bounds yaw "
        "at 6.15 mrad), but both move the beam and so count through detection. Reference flag at its home: "
        "carriage within 0.5 mm of its start position (its home stop holds it at -0.1 mm). Geometric optical ray "
        "at 40-140 mm with the centre of the nominal 4.2 mm spot on the 10 x 44 mm white flag (most of the spot "
        "on the face; background-suppression argument, no sourced switching curve). No robot link or robot-held "
        "driver touching the assembly. Score: 1.0 only with success, otherwise 0.95 x the mean of four screw "
        "credits. A screw's seat is 0 for an M3 whose bit moment exceeded 0.4 N m, 1 for a clicked screw whose "
        "last hands-off load is at or above the floor with advance, else 0.9 x last hands-off load / setting "
        "load. Credit: upward M4 seat x rail-level grade (last hands-off pitch: 0 at the 36.9 mrad free droop, 1 "
        "within 3 mrad); forward M4 seat; M3 seat x (0.75 + 0.25 x detection grade). Detection grade = aim x "
        "return: the aim is judged against a flag at its home position (1 when the spot centre would be on it, else "
        "twice the spot's share there), so moving the flag never raises it; the return share is 1 within the "
        "0.5 mm home band and falls linearly to 0 at 10 mm. Head loads are the last hands-off reading, lowered "
        "(never raised) by readings taken with a bit in the socket or a hand on the work, so loosening under a tool "
        "shows; pitch is read only hands-off. A driver left resting in a socket still blocks that screw's "
        "hands-off reading. Privileged truth hidden from policy")

    def _spot_coverage(self, optical, target_x=None):
        """Fraction of the nominal 4.2 mm spot on the 10 x 44 mm white face
        (centred at target_x, default the flag's current x), 0 off the face
        or outside 40-140 mm (each axis's circular-segment share, multiplied)."""
        hit=optical.get("intersection_m")
        if hit is None or not .04<=optical["range_m"]<=.14:
            return 0.
        r=.0021
        def beyond(h):
            if h>=r:return 0.
            if h<=-r:return 1.
            return (r*r*math.acos(h/r)-h*math.sqrt(r*r-h*h))/(math.pi*r*r)
        def inside(u,half):
            return max(0.,1.-beyond(half-u)-beyond(half+u))
        flag_x=(float(self.state.body_q.numpy()[self.items["flag"],0]) if target_x is None
                else target_x)
        return inside(hit[0]-flag_x,.005)*inside(hit[2]-self.items["flag_center_z_m"],.022)

    def _mixed_score(self, metrics, optical, success, flag_offset):
        """Per-deliverable partial credit from hands-off readings (see
        MIXED_SCOPE); never 1.0 without success. No term rewards motion,
        contact, tool pickup, flag retraction, support or work order."""
        tol=self.config.rail_level_tolerance_rad
        level=float(np.clip((self.FREE_DROOP_RAD-abs(self._free_pitch))/(self.FREE_DROOP_RAD-tol),0.,1.))
        coverage=self._spot_coverage(optical)
        # Detection grade = aim x return. The aim is judged against a flag at
        # its HOME x (1 when the spot centre would be on it, else twice the
        # spot share there), so moving the flag can never raise it; the return
        # share is 1 inside the home band and falls linearly to 0 at 10 mm.
        hit=optical.get("intersection_m")
        if hit is None:
            aim=float(optical["detected"])
        else:
            home_x=float(self.state.body_q.numpy()[self.items["flag"],0])-flag_offset
            on_face=(.04<=optical["range_m"]<=.14 and abs(hit[0]-home_x)<=.005 and
                     abs(hit[2]-self.items["flag_center_z_m"])<=.022)
            aim=1. if on_face else min(1.,2*self._spot_coverage(optical,home_x))
        band,reach=self.MIXED_FLAG_HOME_BAND_M,self.MIXED_FLAG_RETURN_GRADE_M
        returned=1. if abs(flag_offset)<=band else max(0.,(reach-abs(flag_offset))/(reach-band))
        seen=aim*returned
        credit={}
        for name,m in metrics.items():
            m3=name.startswith("sensor_")
            floor=m["minimum_retained_load_n"]
            if m3 and m["peak_bit_socket_moment_2ms_nm"]>.4:
                seat=0.
            elif (m["driver_released_on_screw"] and m["hands_off_load_n"]>=floor and
                  m["advance_m"]>=(.0004 if m3 else .00045)):
                seat=1.
            else:
                seat=.9*min(1.,m["hands_off_load_n"]*.8/floor)
            # Only the upward M4 sets the rail's attitude; the forward M4
            # clamps the (non-rotating) bracket to the post.
            credit[name]=seat*((.75+.25*seen) if m3 else level if name=="frame_top" else 1.)
        partial=sum(credit.values())/len(credit)
        return (1. if success else .95*partial),{
            "score_terms":{"screw_credit":credit,"rail_level_grade":level,
                           "hands_off_rail_pitch_rad":self._free_pitch,
                           "spot_on_flag_fraction":coverage,"aim_at_home_grade":aim,
                           "flag_return_grade":returned,"detection_grade":seen,
                           "partial_mean":partial}}

    _TRACKED=("_preload","_peak_preload","_peak_torque","_thread_load","_interface",
              "_bit_moment","_peak_bit_moment","_bit_window","_peak_bit_moment_filtered",
              "_release_slip","_clicks","_engaged_driver","_bit_contacts","_retained_ticks",
              "_retained_anchor","_seated_clicks","_free_load","_free_pitch","_held",
              "_subfloor_ticks","_click_turn","_task_set_state")

    def checkpoint(self):
        """Complete restartable state: arrays plus JSON-able tracking."""
        import copy
        with self.lock:
            if self.fault:
                raise RuntimeError("Faulted state cannot be checkpointed")
            arrays={f"state_{k}":getattr(self.state,k).numpy().copy()
                    for k in ("body_q","body_qd","joint_q","joint_qd")}
            arrays["control_joint_target_q"]=self.control.joint_target_q.numpy().copy()
            arrays["control_joint_target_qd"]=self.control.joint_target_qd.numpy().copy()
            arrays["control_joint_f"]=self.control.joint_f.numpy().copy()
            arrays["targets"]=np.asarray(self.targets).copy()
            arrays["bracket_origin"]=np.asarray(self._bracket_origin).copy()
            acceleration=np.zeros(self.model.joint_dof_count)
            for motion in self._arm_motion.values():
                for name,value in zip(motion["names"],
                                      motion["input"].current_acceleration):
                    acceleration[self.joints[name]["dof"]]=value
            arrays["arm_reference_acceleration"]=acceleration
            meta={"tick":self.tick,"generation":self.generation,
                  "config":asdict(self.config),"robot_setup":self.setup.id,
                  "setup_fingerprint":self.setup.fingerprint,
                  "tracking":{k:copy.deepcopy(getattr(self,k)) for k in self._TRACKED},
                  "clutch":{n:{k:v for k,v in c.items() if k not in ("jnt","col","q")}
                            for n,c in self._clutch.items()},
                  "screw_lock":{n:[l["locked_at"],l["slide_at"]] for n,l in self._screw_lock.items()},
                  "clamp_stick":[st["locked_at"] for st in self._clamp_stick],
                  "head_touched":dict(self._head_touched)}
            return arrays,meta

    def restore(self, arrays, meta):
        """Restore a checkpoint of this same scene; the solver is rebuilt, so
        continuity is a declared discontinuity (contact warm start reset)."""
        import copy
        with self.lock:
            checkpoint_config=dict(meta["config"])
            checkpoint_config.setdefault("task_contract", "")
            if meta["setup_fingerprint"]!=self.setup.fingerprint or checkpoint_config!=asdict(self.config):
                raise ValueError("Checkpoint belongs to a different scene configuration")
            for k in ("body_q","body_qd","joint_q","joint_qd"):
                getattr(self.state,k).assign(arrays[f"state_{k}"])
            self.next_state.assign(self.state)
            self.control.joint_target_q.assign(arrays["control_joint_target_q"])
            reference_qd=(arrays["control_joint_target_qd"]
                          if "control_joint_target_qd" in arrays
                          else np.zeros_like(self.control.joint_target_qd.numpy()))
            self.control.joint_target_qd.assign(reference_qd)
            self.control.joint_f.assign(arrays["control_joint_f"])
            self.targets=np.asarray(arrays["targets"]).copy()
            self._bracket_origin=np.asarray(arrays["bracket_origin"]).copy()
            self.pipeline.reset_contact_matching()
            self.solver=self._make_solver()
            self.contacts=self.pipeline.contacts()
            self.solved_contacts=self.pipeline.contacts()
            self._map_friction_rows()
            for k,v in meta["tracking"].items():
                setattr(self,k,copy.deepcopy(v))
            for n,c in meta["clutch"].items():
                self._clutch[n].update(copy.deepcopy(c))
            for n,v in meta["screw_lock"].items():
                self._screw_lock[n]["locked_at"],self._screw_lock[n]["slide_at"]=v
            for st,v in zip(self._clamp_stick,meta.get("clamp_stick",[None]*len(self._clamp_stick))):
                st["locked_at"]=v
            self._head_touched=dict(meta["head_touched"])
            self._setup_arm_motion(arrays["control_joint_target_q"],reference_qd,
                arrays["arm_reference_acceleration"]
                if "arm_reference_acceleration" in arrays
                else np.zeros(self.model.joint_dof_count))
            self.tick=int(meta["tick"])
            self.fault=None
            self.generation+=1
            self.evaluation=self._evaluate()
            return self.get_state()

    def _rail_attitude(self, body_q=None):
        body_q=self.state.body_q.numpy() if body_q is None else body_q
        r=Rotation.from_quat(body_q[self.items["rail"],3:])
        axis=r.apply((1.,0.,0.))
        return float(-math.atan2(axis[2],math.hypot(axis[0],axis[1]))),float(math.atan2(axis[1],axis[0]))

    def get_state(self):
        with self.lock:
            q=self.state.body_q.numpy();v=self.state.body_qd.numpy()
            return {"tick":self.tick,"generation":self.generation,
                "time_s":self.tick*self.config.dt,"ready":self.fault is None,
                "fault":self.fault,"config":asdict(self.config),
                "robot_setup":self.setup.id,
                "objects":{name:{"position_m":q[i,:3].tolist(),
                    "quaternion_xyzw":q[i,3:].tolist(),
                    "velocity_linear_angular":v[i].tolist()}
                    for name,i in (("rail",self.items["rail"]),
                                   ("bracket",self.items["bracket"]),
                                   ("sensor",self.items["sensor"]),
                                   ("M3_driver",self.items["M3_driver"]["grip"]),
                                   ("M4_driver",self.items["M4_driver"]["grip"]))},
                "joints":self.rig.observe(self.state)["joints"],
                "contacts":int(self.contacts.rigid_contact_count.numpy()[0]),
                "evaluation":self.evaluation}

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer=None
