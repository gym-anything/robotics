"""Persistent surface-rinsing environment: robot-only commands on exact 0.5 ms ticks, explicit faults, checkpoints.

One command = joint targets (arms, Spine, fingers), an optional base twist and grip-force changes, held for N ticks
(classroom_arrangement's command model, unchanged). Water moves only through the fixtures' joints (squeeze levers,
wrist blades, the swing spout, lever wastes) as the robot's links push them, and through the forces it computes; assisted
fixtures (probes, demos) attach to world.fixture and are never reachable from the policy interface.

Start state: the robot stands at its station with the classroom stowed arms; tools start in their real parking places
(spray valves hanging, the hose valve in its cradle), not in the hand. The reset settle lets gravity, contacts and the
water settle for >= 1 s (classroom rule) before tick 0.
"""

import copy
import threading
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from classroom_arrangement import params as CP
from . import params as P
from . import layout as L
from .config import Config
from .world import RinsingWorld


class RinsingEnv:
    def __init__(self, config=None, *, output_dir=None, spec=None, settle=True, water=True):
        self.config = c = config or Config()
        self.lock = threading.RLock()
        self.output = Path(output_dir) if output_dir else None
        self.spec = spec or L.setup(c.task, c.seed)
        from .robot_setup import check_task
        check_task(c.robot, c.task)
        self.world = w = RinsingWorld(c, self.spec)
        self.water = None
        if water:
            from .water.system import WaterSystem
            self.water = WaterSystem(w, self.spec, device=c.water_device)
            w.water = self.water
        from classroom_arrangement.kinematics import ArmKinematics
        self.ik = ArmKinematics(w)
        self.generation = 0
        self._start_pose()
        if settle:
            self._settle()
        from .task import Evaluator
        self.evaluator = Evaluator(self)
        self.initial = self.checkpoint()

    # ------------------------------------------------------------------ start state
    def _start_pose(self):
        """The Spine raised to the station's start height (layout.ROBOT_SPINE), target and state together, arms stowed."""
        w = self.world
        h = float(self.spec.get("spine", 0.0))
        if not h:
            return
        name = next(n for n in w.joints if "spine_vertical" in n)
        j = w.joints[name]["q"]
        q = w.state.joint_q.numpy(); q[j] = h; w.state.joint_q.assign(q)
        t = w.control.joint_target_q.numpy(); t[j] = h; w.control.joint_target_q.assign(t)
        w.targets[j] = h
        self._fk()

    def _settle(self):
        """Gravity settle before tick 0 (classroom rule): at least 1 s, then until every free body (items, fixtures, the
        robot base) stayed slower than 2 mm/s through a 0.1 s window, at most 5 s."""
        w, s = self.world, CP.SETTLE
        bodies = np.array(sorted(set(w._objects) | {w.scene.robot_body["base_link"]}), int)
        chunk, every = int(round(s["window_s"]/CP.DT)), int(round(0.01/CP.DT))
        n = 0
        while True:
            fastest = 0.
            for k in range(chunk):
                w.step()
                if (k + 1) % every == 0:
                    qd = w.state.body_qd.numpy()
                    fastest = max(fastest, float(np.linalg.norm(qd[bodies, :3], axis=1).max()))
            n += chunk
            if n*CP.DT >= s["min_s"] - 1e-9 and (fastest < s["speed_m_s"] or n*CP.DT >= s["max_s"] - 1e-9):
                break
        qd = w.state.body_qd.numpy()
        sp = np.linalg.norm(qd[bodies, :3], axis=1)
        self.settle_record = dict(seconds=round(n*CP.DT, 3), max_speed_m_s=round(fastest, 6),
                                  slowest_to_settle=w.model.body_label[int(bodies[int(np.argmax(sp))])])
        if w.reflexes:
            raise RuntimeError(f"robot protective stop during the reset settle: {w.reflexes[:2]}")
        w.tick = 0
        w.base.odom[:] = 0.
        w.timing = dict(physics_s=0., control_s=0.)
        if self.water is not None:
            self.water.initial = self.water.totals()
            self.water.timing = {k: 0.0 for k in self.water.timing}

    # ------------------------------------------------------------------ commands
    def act(self, targets=None, *, twist=None, ticks=1, grip=None):
        w = self.world
        if type(ticks) is not int or ticks < 1:
            raise ValueError("ticks must be a positive integer")
        with self.lock:
            if twist is not None and any(abs(v) > 0 for v in twist) and w.base_held():
                raise ValueError(f"Base command refused: an arm protective stop holds the base for {CP.REFLEX_RECOVERY_S} s")
            for arm, f in (grip or {}).items():
                w.set_grip_force(arm, f)
            if targets:
                w.set_targets(dict(targets))
            if twist is not None:
                w.base.set_twist(*twist)
            for _ in range(ticks):
                w.step()
            return self.get_state()

    def step_free(self, ticks):
        """Advance with the current targets (probes and the evaluator's settle branch)."""
        for _ in range(ticks):
            self.world.step()

    # ------------------------------------------------------------------ state
    def joints_state(self):
        w = self.world
        q, qd = w.state.joint_q.numpy(), w.state.joint_qd.numpy()
        return {n: {"position": float(q[s["q"]]), "velocity": float(qd[s["dof"]])} for n, s in w.joints.items()}

    def robot_status(self):
        w = self.world
        b = w.base
        return {"base": {"odometry": [round(float(v), 6) for v in b.odom], "twist_measured": [round(float(v), 6) for v in b.odom_twist],
                         "twist_commanded": [round(float(v), 6) for v in b.cmd]},
                "arms": {a: {"stopped_s": round(max(0., float(w._reflex_until.get(a, 0.)) - w.sim_time), 4),
                             "external_torque_nm": [round(float(v), 3) for v in w._tau_f[a]],
                             "external_wrench_hand": [round(float(v), 3) for v in w._wrench_f[a]],
                             "grip_force_n": w.grip_force[a]} for a in w.arms},
                "refused_targets": int(w.refused_targets),
                "base_sensors": self.base_sensors()}

    def base_sensors(self):
        key = (int(self.world.tick), self.generation)
        if getattr(self, "_sensor_key", None) != key:
            w = self.world
            imu = getattr(w, "imu", None) or dict(angular_velocity_rad_s=np.zeros(3), specific_force_m_s2=np.array([0., 0., CP.GRAVITY]))
            scans = w.lidar_scan()
            self._sensor_cache = {
                "imu": {k: [round(float(x), 5) for x in v] for k, v in imu.items()},
                "lidar": {k: dict(angle_min_deg=float(s["angle_min_deg"]), angle_step_deg=float(s["angle_step_deg"]),
                                  ranges_m=[float(x) for x in s["ranges_m"]]) for k, s in scans.items()}}
            self._sensor_key = key
        return self._sensor_cache

    def items_state(self):
        """Privileged: item and fixture poses, fixture joint values, outlet flows, pools (owner view; not observations)."""
        w = self.world
        out = {}
        for n, it in w.scene.items.items():
            p, R = w.body_pose(it["body"])
            out[n] = dict(kind=it["kind"], position=[round(float(x), 5) for x in p],
                          quat_xyzw=[round(float(x), 6) for x in Rotation.from_matrix(R).as_quat()])
        fx = {}
        for n, f in w.scene.fixtures.items():
            j = f.get("joint") or f.get("lever_joint")
            if j:
                fx[n] = dict(kind=f["kind"], joint_rad=round(w.joint_value(j)[0], 5))
        water = {}
        if self.water is not None:
            ws = self.water
            water = dict(outlets={n: dict(flow_gpm=round(o.Q/P.GPM, 4), T_C=round(o.T, 2), jet_m_s=round(o.v_jet, 3))
                                  for n, o in ws.outlets.items()},
                         pools={p.name: dict(volume_mL=round(p.vol*1e6, 3), T_C=round(p.T, 2),
                                             surfactant_g_L=round(p.sp[0]/p.vol, 5) if p.vol > 0 else 0.0)
                                for p in ws.pools.pools})
        return dict(items=out, fixtures=fx, water=water)

    def get_state(self):
        w = self.world
        return {"tick": int(w.tick), "time_s": round(w.sim_time, 6), "generation": self.generation, "fault": w.fault,
                "finite": w.fault is None, "ready": True, "joints": self.joints_state(), "robot_status": self.robot_status(),
                "base_pose": [round(v, 5) for v in w.base_pose()], "scene": self.items_state(), "reflexes": len(w.reflexes),
                "evaluation": self.evaluation()}

    def evaluation(self):
        if not hasattr(self, "evaluator"):
            return None
        key = (int(self.world.tick), self.generation)
        if getattr(self, "_eval_key", None) != key:
            self._eval_cache, self._eval_key = self.evaluator.progress(), key
        return self._eval_cache

    # ------------------------------------------------------------------ checkpoints
    # world state that stepping changes besides the rigid state, the controllers and the water: it is checkpointed and
    # restored too, or every restore (each reset, each evaluation's settle branch, P10's controls, an owner's restore)
    # leaks it from the branch it ends: the hold flag of a glass the robot held before the restore, the no-slip pass's
    # on/off state, the base's reflex hold, the IMU filter, the glide filter and MuJoCo's clock (P16 compares every such
    # attribute).
    STEP_STATE = ("holding", "slip_max", "noslip_off_ticks", "_base_hold_until", "_prev_glide", "glide_speed",
                  "_imu_v", "_imu_f", "imu", "feedforward")

    def checkpoint(self):
        w = self.world
        step_state = {k: copy.deepcopy(getattr(w, k)) for k in self.STEP_STATE if hasattr(w, k)}
        return dict(step_state=step_state, noslip=(int(w.mjm.opt.noslip_iterations), float(w.mjm.opt.noslip_tolerance)),
                    mj_time=float(w.mjd.time),
                    rigid={a: getattr(w.state, a).numpy().copy() for a in ("body_q", "body_qd", "joint_q", "joint_qd")},
                    targets=w.targets.copy(), control=w.control.joint_target_q.numpy().copy(),
                    control_qd=w.control.joint_target_qd.numpy().copy(), tick=w.tick,
                    grip_force=dict(w.grip_force), grasp_latched=dict(w.grasp_latched), grip_ref=dict(w._grip_ref),
                    reflex_until=dict(w._reflex_until), reflex_fired=dict(w._reflex_fired), tau_f=copy.deepcopy(w._tau_f),
                    vref=w._vref.copy(), tau_sf=copy.deepcopy(w._tau_sf), wrench_f=copy.deepcopy(w._wrench_f),
                    base=w.base.state(), refused=w.refused_targets, n_reflexes=len(w.reflexes),
                    mj_qpos=w.mjd.qpos.copy(), mj_qvel=w.mjd.qvel.copy(), mj_act=w.mjd.act.copy() if w.mjm.na else None,
                    mj_warm=w.mjd.qacc_warmstart.copy(), geom_friction=w.mjm.geom_friction.copy(),
                    water=self.water.snapshot() if self.water is not None else None,
                    evaluator=self.evaluator.snapshot() if hasattr(self, "evaluator") else None,
                    mass={k: (float(w.mjm.body_mass[w._mj_body[k]]), np.array(w.mjm.body_ipos[w._mj_body[k]]),
                              np.array(w.mjm.body_inertia[w._mj_body[k]])) for k in w._extra_mass})

    def restore(self, snap):
        w = self.world
        with self.lock:
            for a, v in snap["rigid"].items():
                getattr(w.state, a).assign(v)
                getattr(w.next_state, a).assign(v)
            w.targets = snap["targets"].copy()
            w.control.joint_target_q.assign(snap["control"])
            w.control.joint_target_qd.assign(snap["control_qd"])
            w.tick = snap["tick"]
            w.grip_force, w.grasp_latched, w._grip_ref = dict(snap["grip_force"]), dict(snap["grasp_latched"]), dict(snap["grip_ref"])
            w._reflex_until, w._reflex_fired = dict(snap["reflex_until"]), dict(snap["reflex_fired"])
            w._tau_f = copy.deepcopy(snap["tau_f"])
            w._vref = snap["vref"].copy()
            w._tau_sf = copy.deepcopy(snap["tau_sf"]); w._wrench_f = copy.deepcopy(snap["wrench_f"])
            w.base.restore(snap["base"])
            w.refused_targets = snap["refused"]
            del w.reflexes[snap["n_reflexes"]:]
            w.fault = None
            w.mjm.geom_friction[:] = snap["geom_friction"]
            for k, (m, c, I) in snap["mass"].items():
                kk = w._mj_body[k]
                w.mjm.body_mass[kk], w.mjm.body_ipos[kk], w.mjm.body_inertia[kk] = m, c, I
            w.sync_solver()
            for k in self.STEP_STATE:
                if k in snap["step_state"]:
                    setattr(w, k, copy.deepcopy(snap["step_state"][k]))
                elif hasattr(w, k):
                    delattr(w, k)                    # set lazily after the checkpoint (the base's reflex hold, the IMU)
                    if k in ("_imu_v", "_imu_f", "imu"):
                        w.__dict__.pop("_imu_body", None)        # the IMU initialises itself again on its next update
            w.mjm.opt.noslip_iterations, w.mjm.opt.noslip_tolerance = snap["noslip"]
            w.mjd.time = snap["mj_time"]
            w.mjd.qpos[:] = snap["mj_qpos"]
            w.mjd.qvel[:] = snap["mj_qvel"]
            w.mjd.qacc_warmstart[:] = snap["mj_warm"]
            if snap["mj_act"] is not None:
                w.mjd.act[:] = snap["mj_act"]
            w._mujoco.mj_forward(w.mjm, w.mjd)
            if self.water is not None and snap["water"] is not None:
                self.water.restore(snap["water"])
            if snap.get("evaluator") is not None and hasattr(self, "evaluator"):
                self.evaluator.restore(snap["evaluator"])

    # ------------------------------------------------------------------ privileged state writes (probes only)
    def _fk(self):
        w = self.world
        w.newton.eval_fk(w.model, w.state.joint_q, w.state.joint_qd, w.state)
        w.next_state.assign(w.state)
        w.sync_solver()

    def _place_body(self, name, p, R):
        """Teleport a free item (and its articulated children) to the world pose (p, R), at rest. Probe fixtures only
        (evaluator controls, P10); never reachable from the policy interface. Call _fk() after the last write."""
        w = self.world
        b = w.scene.bodies[name]
        jq, jd = w.model.joint_q_start.numpy(), w.model.joint_qd_start.numpy()
        jchild = w.model.joint_child.numpy()
        jfree = [j for j in range(len(jchild)) if jchild[j] == b][0]
        q = w.state.joint_q.numpy()
        s = int(jq[jfree])
        q[s:s + 3] = p
        q[s + 3:s + 7] = Rotation.from_matrix(R).as_quat()          # Newton free joint: (x, y, z, qx, qy, qz, qw)
        w.state.joint_q.assign(q)
        qd = w.state.joint_qd.numpy()
        qd[int(jd[jfree]):int(jd[jfree]) + 6] = 0.0
        qd[:] = np.where(np.isfinite(qd), qd, 0.0)
        w.state.joint_qd.assign(qd)

    def reset(self):
        with self.lock:
            self.restore(self.initial)
            self.generation += 1
            return self.get_state()
