"""Persistent painting environment: robot-only commands on exact 0.5 ms ticks, explicit faults, checkpoints.

One command = joint targets (arms, Spine, fingers), an optional base twist and grip-force changes, held for N ticks
(classroom_arrangement's command model, unchanged). Paint moves only through the tools' contacts and the aerosol
button; assisted fixtures (probes) attach to world.fixture and are never reachable from the policy interface.

Start state: the task's tool is already in a valid grasp (as the whiteboard task isolates writing from acquiring the
marker): the arm is put at an IK solution for the grasp pose before tick 0, the tool placed between the open fingers
and the fingers closed in force mode; the reset settle then lets gravity and finger friction take over.
"""

import copy
import math
import threading
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from classroom_arrangement import params as CP
from . import params as P
from . import layout as L
from .config import Config
from .world import PaintingWorld

# grasp poses: tool frame in the hand TCP frame (TCP: z approach, y finger travel; see _grasps) and the hand pose per task
HOLD = {
    # roller: grip axis along TCP x, grip centre at the TCP, cover's contact side (tool +x) along the approach
    "roller": dict(R=np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]]), p=(0.245, 0., 0.), half=0.01755),
    # sash brush: handle across the fingers by its 19 mm thickness, bristles along TCP +x
    "sash": dict(R=np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]), p=(0.115, 0., 0.), half=0.0095),
    # round brush: held like a pencil, bristles along TCP -x (down), so the wrist stays above and behind the hand (with
    # bristles along +x the hand had to come from the far side and its wrist met the board edge, P9)
    "round8": dict(R=np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]]), p=(-0.115, 0., 0.), half=0.0045),
    # aerosol: can axis along TCP x, held at mid height, nozzle (tool +x) along the approach
    "aerosol": dict(R=np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]]), p=(-0.09, 0., 0.), half=0.0305),
}


def _longreach_hold():
    """Long-reach dog-leg brush, held at the end of its handle as a radiator brush is (grip centre at the TCP, the pads
    over the last 22 mm of the handle's straight part): the handle axis along TCP x with the head beyond the hand, the
    approach (TCP z) toward the side the head bends to, turned 10 deg about the handle (toward the brush's width),
    so with the head on the wall behind the radiator the hand stays on the room side, outside the radiator's end or
    above its top: hand-clear, the bristle tip line then reaches all of the band but the bottom of the middle (geometry
    ~75 cm2, the most of the fixed turns scanned, -40..40 deg; phantom-held at this grip with the hand kept clear, the
    bottom row stayed unhidden over x 0.531-0.573 m). A grasp 0.21 m from the ferrule, width along the approach, puts
    the finger travel along the wall normal in every working pose: the hand housing reaches 48-50 mm into the wall."""
    from . import tools as T
    B = P.BRUSHES["longreach"]
    a = np.radians(B["dogleg_deg"])
    s, c = np.sin(a), np.cos(a)
    R = np.array([[0., s, c], [-1., 0., 0.], [0., -c, s]]) @ Rotation.from_rotvec(T.dogleg_dir("longreach")*np.radians(10.0)).as_matrix()
    g = np.array([0., 0., -B["ferrule"][2]]) + T.dogleg_dir("longreach")*(B["handle_len"] - B["handle_r"])
    return dict(R=R, p=tuple(-R @ g), half=B["handle_r"])


HOLD["longreach"] = _longreach_hold()
# hand TCP pose per preset: Spine height, TCP position (world) and orientation R = Rz(yaw) G Ry(pitch), G the classroom
# grip frame (TCP x up, y along world +x, approach z along world +y); found by an IK scan with self-contact rejection:
# the Duo mount puts the shoulders 0.53 m ahead of base_link, so the hands start
# 0.3-0.45 m ahead of the shoulders. wall: tool axis up, leaning 30 deg toward the wall; craft: TCP x up leaning 30 deg
# forward, the brush's bristles down along -x; spray: can upright, nozzle forward.
G_FRAME = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
START_HAND = {
    "wall": dict(spine=0.30, pos=(-0.10, 1.44, 1.10), yaw=0.0, pitch=-30.0),
    "craft": dict(spine=0.25, pos=(-0.15, 1.15, 1.00), yaw=0.0, pitch=-30.0),
    "spray": dict(spine=0.25, pos=(-0.75, 1.20, 0.95), yaw=0.0, pitch=0.0),
}


def start_rotation(preset):
    s = START_HAND[preset]
    return (Rotation.from_euler("z", s["yaw"], degrees=True)*Rotation.from_matrix(G_FRAME)
            * Rotation.from_euler("y", s["pitch"], degrees=True)).as_matrix()


class PaintingEnv:
    def __init__(self, config=None, *, output_dir=None, spec=None, settle=True):
        self.config = c = config or Config()
        self.lock = threading.RLock()
        self.output = Path(output_dir) if output_dir else None
        from .catalogue import apply_contract
        self.spec = apply_contract(spec or L.setup(c.task, c.seed), c.contract)
        if self.spec.get("held"):
            from .robot_setup import check_task
            check_task(c.robot, self.spec["preset"])        # e.g. one arm cannot hold the can and press its button
        self.world = w = PaintingWorld(c, self.spec)
        from .system import PaintSystem
        self.paint = PaintSystem(w, self.spec)
        w.paint = self.paint
        from classroom_arrangement.kinematics import ArmKinematics
        self.ik = ArmKinematics(w)
        self.generation = 0
        self._place_held()
        if settle:
            self._settle()
        from .task import Evaluator
        self.evaluator = Evaluator(self)
        self.initial = self.checkpoint()

    # ------------------------------------------------------------------ start state
    def _set_joint(self, name, value):
        w = self.world
        j = w.joints[name]["q"]
        q = w.state.joint_q.numpy(); q[j] = value; w.state.joint_q.assign(q)
        t = w.control.joint_target_q.numpy(); t[j] = value; w.control.joint_target_q.assign(t)
        w.targets[j] = value

    def _place_held(self):
        w, sp = self.world, self.spec
        held = sp.get("held") or {}
        if not held:
            return
        start = START_HAND[sp["preset"]]
        spine = [n for n in w.joints if "spine_vertical" in n][0]
        self._set_joint(spine, start["spine"])
        self._fk()
        for arm, tool in held.items():
            if arm not in w.arms:
                arm = w.arms[0]                      # single-arm robot: the right arm holds it
            pos = np.array(start["pos"], float)
            Rt = start_rotation(sp["preset"])
            if arm != dict(spray="left").get(sp["preset"], "right"):
                pos[0] += 0.55 if arm == "right" else -0.55        # the other arm (single-arm robot): mirror across
            # IK solutions nearest the stowed arm first; the first clear of the robot and of the scene (classroom planner)
            from classroom_arrangement.planner import ArmPlanner
            planner = ArmPlanner(w, self.ik, margin=0.01)
            names = self.ik.arms[arm][1]
            sol = None
            for q1 in self.ik.solutions(arm, pos, Rotation.from_matrix(Rt).as_quat(), seeds=48, tol_rad=0.035):
                if not self.ik.self_contact(arm, q1) and not planner.contacts(arm, q1, allow=(tool,)):
                    sol = dict(zip(names, map(float, q1)))
                    break
            if sol is None:
                raise RuntimeError(f"no collision-free start grasp for {arm} at {pos.tolist()}")
            for n, v in sol.items():
                self._set_joint(n, v)
            self._fk()
            fr = w.rig.frame_pose(f"{w.scene.robot_instance}/{arm}_tcp", w.state)
            qw = fr["orientation_wxyz"]
            Rtcp = Rotation.from_quat((qw[1], qw[2], qw[3], qw[0])).as_matrix()
            ptcp = np.asarray(fr["position_m"], float)
            g = HOLD[tool]
            Rtool = Rtcp @ g["R"]
            ptool = ptcp + Rtcp @ np.asarray(g["p"], float)
            self._place_body(tool, ptool, Rtool)
            for n in w.gripper_joints(arm):
                self._set_joint(n, min(0.04, g["half"] + 0.0005))
            self._fk()
            # start grip per finger: the 9 in roller (0.4 kg, its cover 0.12 m off the grip axis) needs a firm pinch
            # (35 N per finger holds 0.6-2.1 N m, the roller's gravity moment is ~0.5-0.7 N m); brushes and the can 35 N
            w.set_grip_force(arm, 60.0 if tool == "roller" else 35.0)
            # force mode pushes kp (ref - q) with the reference walking toward the target; with the target at 0 the push
            # on a thin handle stops at kp x its half width (12.3 N on the 9 mm round brush, not 35): aim the reference
            # past closed so the set force is what the fingers apply
            for n in w.gripper_joints(arm):
                w.targets[w.joints[n]["q"]] = -0.03
            self.hold_record = getattr(self, "hold_record", {})
            self.hold_record[arm] = dict(tool=tool, tcp=ptcp.tolist(), spine=start["spine"])
        w.sync_solver()

    def _fk(self):
        w = self.world
        w.newton.eval_fk(w.model, w.state.joint_q, w.state.joint_qd, w.state)
        w.next_state.assign(w.state)
        w.sync_solver()

    def _place_body(self, name, p, R):
        """Move a tool's free joint (and with it its articulated children) to the world pose (p, R)."""
        w = self.world
        b = w.scene.bodies[name]
        jq = w.model.joint_q_start.numpy()
        jt = w.model.joint_type.numpy()
        jchild = w.model.joint_child.numpy()
        jfree = [j for j in range(len(jt)) if jchild[j] == b][0]
        q = w.state.joint_q.numpy()
        s = int(jq[jfree])
        q[s:s + 3] = p
        q[s + 3:s + 7] = Rotation.from_matrix(R).as_quat()          # Newton free joint: (x, y, z, qx, qy, qz, qw)
        w.state.joint_q.assign(q)
        qd = w.state.joint_qd.numpy(); qd[:] = np.where(np.isfinite(qd), qd, 0.0); w.state.joint_qd.assign(qd)

    def _settle(self):
        """Gravity settle before tick 0 (classroom rule): at least 1 s, then until every free body stayed slower than
        2 mm/s through a 0.1 s window, at most 5 s."""
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
        self.settle_record = dict(seconds=round(n*CP.DT, 3), max_speed_m_s=round(fastest, 6))
        if w.reflexes:
            raise RuntimeError(f"robot protective stop during the reset settle: {w.reflexes[:2]}")
        w.tick = 0
        w.base.odom[:] = 0.
        w.timing = dict(physics_s=0., control_s=0.)

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

    def tools_state(self):
        """Privileged: tool poses and paint loads (owner view; not policy observations)."""
        w = self.world
        out = {}
        for n, it in w.scene.items.items():
            p, R = w.body_pose(it["body"])
            out[n] = dict(kind=it["kind"], position=[round(float(x), 5) for x in p],
                          quat_xyzw=[round(float(x), 6) for x in Rotation.from_matrix(R).as_quat()])
        for n, a in self.paint.applicators.items():
            out.setdefault(n, {})["paint"] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in a.state().items()
                                              if not isinstance(v, list)}
        for n, r in self.paint.reservoirs.items():
            out[f"pool:{n}"] = dict(volume_mL=round(r.volume*1e6, 3))
        return out

    def get_state(self):
        w = self.world
        return {"tick": int(w.tick), "time_s": round(w.sim_time, 6), "generation": self.generation, "fault": w.fault,
                "finite": w.fault is None, "ready": True, "joints": self.joints_state(), "robot_status": self.robot_status(),
                "base_pose": [round(v, 5) for v in w.base_pose()], "tools": self.tools_state(), "reflexes": len(w.reflexes),
                "evaluation": self.evaluation()}

    def evaluation(self):
        if not hasattr(self, "evaluator"):
            return None
        key = (int(self.world.tick), self.generation)
        if getattr(self, "_eval_key", None) != key:
            self._eval_cache, self._eval_key = self.evaluator.progress(), key
        return self._eval_cache

    # ------------------------------------------------------------------ checkpoints
    def checkpoint(self):
        w = self.world
        return dict(rigid={a: getattr(w.state, a).numpy().copy() for a in ("body_q", "body_qd", "joint_q", "joint_qd")},
                    targets=w.targets.copy(), control=w.control.joint_target_q.numpy().copy(),
                    control_qd=w.control.joint_target_qd.numpy().copy(), tick=w.tick,
                    grip_force=dict(w.grip_force), grasp_latched=dict(w.grasp_latched), grip_ref=dict(w._grip_ref),
                    reflex_until=dict(w._reflex_until), reflex_fired=dict(w._reflex_fired), tau_f=copy.deepcopy(w._tau_f),
                    vref=w._vref.copy(), tau_sf=copy.deepcopy(w._tau_sf), wrench_f=copy.deepcopy(w._wrench_f),
                    base=w.base.state(), refused=w.refused_targets, n_reflexes=len(w.reflexes),
                    mj_qpos=w.mjd.qpos.copy(), mj_qvel=w.mjd.qvel.copy(), mj_act=w.mjd.act.copy() if w.mjm.na else None,
                    mj_warm=w.mjd.qacc_warmstart.copy(), paint=self.paint.snapshot(),
                    task_evaluation=self.evaluator.checkpoint_state(),
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
            for k, (m, c, I) in snap["mass"].items():
                kk = w._mj_body[k]
                w.mjm.body_mass[kk], w.mjm.body_ipos[kk], w.mjm.body_inertia[kk] = m, c, I
            w.sync_solver()
            w.mjd.qpos[:] = snap["mj_qpos"]
            w.mjd.qvel[:] = snap["mj_qvel"]
            w.mjd.qacc_warmstart[:] = snap["mj_warm"]
            if snap["mj_act"] is not None:
                w.mjd.act[:] = snap["mj_act"]
            w._mujoco.mj_forward(w.mjm, w.mjd)
            self.paint.restore(snap["paint"])
            self.evaluator.restore_state(snap.get("task_evaluation"))

    def reset(self):
        with self.lock:
            self.restore(self.initial)
            self.generation += 1
            return self.get_state()
