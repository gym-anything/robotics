"""Persistent classroom environment: robot-only commands, exact 0.5 ms ticks, explicit faults, checkpoints.

One command = joint targets (arms, spine, fingers), an optional base twist and grip-force changes, held for N ticks.
Nothing here moves furniture except through the robot's contacts; assisted fixtures (probes) attach to
world.fixture and are never reachable from the backend's policy interface.
"""

import copy
import math
import threading
from collections import deque
from pathlib import Path
import numpy as np

from . import params as P
from . import layout as L
from .config import Config
from .world import World


class ClassroomEnv:
    def __init__(self, config=None, *, output_dir=None, spec=None):
        """`spec` (probes only): a layout.setup result edited by an experiment (robot start, item poses); the public
        backend always builds it from the configuration."""
        self.config = c = config or Config()
        self.lock = threading.RLock()
        self.output = Path(output_dir) if output_dir else None
        self.spec = spec or L.setup(c.task, c.seed, glides=c.glides, contents=c.contents)
        self.world = World(c, self.spec)
        self.generation = 0
        self._settle()
        self._new_history()
        from .task import Evaluator
        self.evaluator = Evaluator(self)
        self.initial = self.checkpoint()

    # ------------------------------------------------------------------ reset settle
    def _settle(self):
        """Gravity settle before tick 0: at least 1 s, then until every free body and the robot base stayed slower
        than 2 mm/s through a whole 0.1 s window (sampled every 10 ms), at most 5 s (as in barcode)."""
        w, s = self.world, P.SETTLE
        bodies = np.array(sorted(set(w._objects) | {w.scene.robot_body["base_link"]}), int)
        chunk, every = int(round(s["window_s"]/P.DT)), int(round(0.01/P.DT))
        n = 0
        while True:
            fastest = 0.
            for k in range(chunk):
                w.step()
                if (k + 1) % every == 0:
                    qd = w.state.body_qd.numpy()
                    fastest = max(fastest, float(np.linalg.norm(qd[bodies, :3], axis=1).max()))
            n += chunk
            if n*P.DT >= s["min_s"] - 1e-9 and (fastest < s["speed_m_s"] or n*P.DT >= s["max_s"] - 1e-9):
                break
        self.settle_record = dict(seconds=round(n*P.DT, 3), max_speed_m_s=round(fastest, 6))
        if w.reflexes:
            raise RuntimeError(f"robot protective stop during the reset settle: {w.reflexes[:2]}")
        w.tick = 0
        w.base.odom[:] = 0.
        w.timing = dict(physics_s=0., control_s=0.)

    # ------------------------------------------------------------------ history (evaluation windows)
    def _new_history(self):
        w = self.world
        self.watch = [n for n in self.spec["items"] if n in w.scene.bodies]
        self._idx = np.array([w.scene.bodies[n] for n in self.watch], int)
        self.samples = deque(maxlen=int(round(P.EVAL["window_s"]/0.05)) + 1)
        q = w.state.body_q.numpy()
        self.start_q = q[self._idx].copy()
        self.history = dict(max_tilt={n: 0. for n in self.watch}, reflexes=0, peak_wheel_slip=0.)
        self._sample()

    def _sample(self):
        q = self.world.state.body_q.numpy()[self._idx].copy()
        self.samples.append((self.world.tick, q))
        tilt = np.degrees(np.arccos(np.clip(1 - 2*(q[:, 3]**2 + q[:, 4]**2), -1, 1)))
        for n, t in zip(self.watch, tilt):
            if t > self.history["max_tilt"][n]:
                self.history["max_tilt"][n] = float(t)

    def window_motion(self):
        """Per watched item: largest displacement (m) of its body origin within the last 1 s window, and whether the
        window is complete (a fresh restore or reset has no complete window yet)."""
        if len(self.samples) < 2:
            return {n: math.inf for n in self.watch}, False
        qs = np.stack([q for _, q in self.samples])
        span = (self.samples[-1][0] - self.samples[0][0])*P.DT
        d = np.linalg.norm(qs[:, :, :3] - qs[-1][None, :, :3], axis=2).max(axis=0)
        return dict(zip(self.watch, d.tolist())), span >= P.EVAL["window_s"] - 1e-9

    # ------------------------------------------------------------------ commands
    def act(self, targets=None, *, twist=None, ticks=1, grip=None):
        w = self.world
        if type(ticks) is not int or ticks < 1:
            raise ValueError("ticks must be a positive integer")
        with self.lock:
            if twist is not None and any(abs(v) > 0 for v in twist) and w.base_held():
                raise ValueError("Base command refused: an arm protective stop holds the base for "
                                 f"{P.REFLEX_RECOVERY_S} s; no command executed")
            for arm, f in (grip or {}).items():
                w.set_grip_force(arm, f)
            if targets:
                w.set_targets(dict(targets))
            if twist is not None:
                w.base.set_twist(*twist)
            every = int(round(0.05/P.DT))
            for _ in range(ticks):
                w.step()
                if w.tick % every == 0:
                    self._sample()
            return self.get_state()

    def stop_base(self):
        self.world.base.set_twist(0., 0., 0.)

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
        """The TMR's IMU and its two corner LIDAR scans (manual R04245 5.6), cached per (tick, episode): angles in the
        base frame (x forward, y left); the four base D455 cameras are render cameras (front/rear/left/right_rgb)."""
        key = (int(self.world.tick), self.generation)
        if getattr(self, "_sensor_key", None) != key:
            w = self.world
            imu = getattr(w, "imu", None) or dict(angular_velocity_rad_s=np.zeros(3), specific_force_m_s2=np.array([0., 0., P.GRAVITY]))
            scans = w.lidar_scan()
            self._sensor_cache = {
                "imu": {k: [round(float(x), 5) for x in v] for k, v in imu.items()},
                "lidar": {k: dict(angle_min_deg=float(s["angle_min_deg"]), angle_step_deg=float(s["angle_step_deg"]),
                                  ranges_m=[float(x) for x in s["ranges_m"]]) for k, s in scans.items()}}
            self._sensor_key = key
        return self._sensor_cache

    def get_state(self):
        w = self.world
        items = {}
        for n in self.watch:
            (x, y, yaw), tilt = w.pose2d(n)
            items[n] = dict(x=round(x, 5), y=round(y, 5), yaw=round(yaw, 5), tilt_deg=round(tilt, 3))
        return {"tick": int(w.tick), "time_s": round(w.sim_time, 6), "generation": self.generation, "fault": w.fault,
                "finite": w.fault is None, "ready": True, "joints": self.joints_state(), "robot_status": self.robot_status(),
                "base_pose": [round(v, 5) for v in w.base_pose()], "items": items, "reflexes": len(w.reflexes),
                "evaluation": self.evaluation()}

    def evaluation(self):
        """The evaluator's result, computed once per (tick, generation) and reused: GAR reads the state before and after
        every command and the backend once more, and the route check of clear-floor / group-work takes 0.13 s a call.
        Nothing in it depends on more than the world state and its history."""
        if not hasattr(self, "evaluator"):
            return None
        key = (int(self.world.tick), self.generation)
        if getattr(self, "_eval_key", None) != key:
            self._eval_cache, self._eval_key = self.evaluator.evaluate(), key
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
                    base=w.base.state(), friction=w.mjm.geom_friction[w.glide_geoms, 0].copy(),
                    prev_glide=w._prev_glide.copy(), refused=w.refused_targets, n_reflexes=len(w.reflexes),
                    samples=list(self.samples), history=copy.deepcopy(self.history), mj_qpos=w.mjd.qpos.copy(),
                    mj_qvel=w.mjd.qvel.copy(), mj_act=w.mjd.act.copy() if w.mjm.na else None,
                    mj_warm=w.mjd.qacc_warmstart.copy())

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
            w.mjm.geom_friction[w.glide_geoms, 0] = snap["friction"]
            w._prev_glide = snap["prev_glide"].copy()
            w.refused_targets = snap["refused"]
            del w.reflexes[snap["n_reflexes"]:]
            w.fault = None
            w.sync_solver()
            # MuJoCo's own float64 state (update_data_interval = 0): restore it exactly, after the float32 sync
            w.mjd.qpos[:] = snap["mj_qpos"]
            w.mjd.qvel[:] = snap["mj_qvel"]
            w.mjd.qacc_warmstart[:] = snap["mj_warm"]
            if snap["mj_act"] is not None:
                w.mjd.act[:] = snap["mj_act"]
            w._mujoco.mj_forward(w.mjm, w.mjd)
            self.samples = deque(snap["samples"], maxlen=self.samples.maxlen)
            self.history = copy.deepcopy(snap["history"])

    def reset(self):
        with self.lock:
            self.restore(self.initial)
            self.generation += 1
            return self.get_state()
