"""Scripted robot motion through the normal controls, with recording for videos.

The Operator moves the arms only by setting joint position targets (from IK on Cartesian waypoints) and
grip forces -- exactly what a policy can do through `named_joint_target` / `cartesian_pose` / `gripper`.
Choosing those waypoints is the ASSISTANCE (a scripted operator, not a policy); phantom forces and mounts live
in probes/fixtures.py and are labelled separately. Every chunk of physics goes through ScanEnv.act, so the
trigger switch, seat sensor, firmware, scanner frames, decoder and host app run exactly as for a policy.

Recording: every `record_every` milliseconds of simulated time a snapshot of the rendered state (body poses, the scanner's lights and
good-read LED, the host screen) and the phase label; video.py renders those snapshots (never re-simulates).
"""

import json
import math
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .kinematics import ArmKinematics
from . import params as P

CHUNK_S = .01       # s per control update: waypoint rate and recording granularity (10 ms; independent of the step)


def hand_quat(approach, fingers):
    """Franka Hand TCP orientation (xyzw) from its approach axis (TCP z, out of the palm) and finger axis (TCP y)."""
    z = np.asarray(approach, float); z /= np.linalg.norm(z)
    y = np.asarray(fingers, float); y = y - (y @ z)*z; y /= np.linalg.norm(y)
    return Rotation.from_matrix(np.column_stack([np.cross(y, z), y, z])).as_quat()


def scanner_in_hand(op, arm):
    """Measured scanner pose in the TCP frame (position, Rotation) -- ASSISTED scripts read it from the state."""
    pt, Rt = op.tcp(arm)
    ps, Rs = op.w.body_pose("scanner")
    return Rt.inv().apply(ps - pt), Rt.inv()*Rotation.from_matrix(Rs)


def tcp_for_scanner(ps, Rs, rel):
    """TCP pose that places the held scanner's origin at ps with orientation Rs (matrix), given scanner_in_hand."""
    rel_p, rel_R = rel
    Rt = Rotation.from_matrix(Rs)*rel_R.inv()
    return ps - Rt.apply(rel_p), Rt


def place_held(op, arm, body, p_target, R_target, seconds=1.0, corrections=2, tol_m=.002):
    """ASSISTED: move so that a held body's origin reaches p_target with orientation R_target (matrix), re-measuring
    the body in the hand after each move and correcting (a policy would close this loop with its cameras)."""
    err = None
    for k in range(1 + corrections):
        pt, Rt = op.tcp(arm)
        pb, Rb = op.w.body_pose(body)
        rel = (Rt.inv().apply(pb - pt), Rt.inv()*Rotation.from_matrix(Rb))
        tp, tR = tcp_for_scanner(np.asarray(p_target, float), R_target, rel)
        if k == 0:
            op.move_joints(arm, tp, tR.as_quat(), seconds=seconds)       # the big move in joint space (no branch jumps)
        else:
            op.move(arm, tp, tR.as_quat(), seconds=.5)
        op.wait(.2)
        pb, Rb = op.w.body_pose(body)
        err = float(np.linalg.norm(pb - p_target))
        if err < tol_m:
            break
    return err


def seated_scanner_pose(scene):
    """The scanner body's pose when seated in its cradle, and the axis of the cradle's socket (world): lifted along it,
    the handle's front face leaves the lip. (The socket stands turned back from the seated handle by its
    rest lean, scene.socket_lean.)"""
    cr = scene.cradle
    Rs = Rotation.from_quat(cr["scanner_quat"]).as_matrix()
    axis = np.asarray(cr.get("socket_axis", Rotation.from_quat(cr["handle_quat"]).apply((0., 0., 1.))), float)
    return np.asarray(cr["origin"], float), Rs, axis


def reseat(op, arm, mouth=.050, drop=.010, seconds=1.6):
    """ASSISTED: steer the held scanner (from its measured in-hand pose) upright into the cradle's socket
    (scene.insertion_pose), its handle's bottom `mouth` m above the socket floor, lower it to `drop` m, and open the
    hand: the scanner drops onto the floor and tips forward onto the lip. (Lowered at its seated lean, the handle's face
    met the lip's top edge first and the scanner hung 4 mm above its seat, held between the lip and the rear wall.)"""
    rel = scanner_in_hand(op, arm)
    for k, t in ((mouth, seconds), (drop, seconds)):
        o, q = op.w.scene.insertion_pose(k)
        pt, Rt = tcp_for_scanner(o, Rotation.from_quat(q).as_matrix(), rel)
        op.move(arm, pt, Rt.as_quat(), seconds=t)
    op.open_hand(arm, seconds=.8)
    return rel


class Operator:
    def __init__(self, env, record_every=40, frames_dir=None, verbose=None):
        dt = env.world.config.rigid_dt
        self.chunk = max(1, int(round(CHUNK_S/dt)))                   # ticks per control update
        if record_every % 10:
            raise ValueError("record_every (ms) must be a multiple of the 10 ms control chunk")
        import os
        self.verbose = bool(os.environ.get("BS_OPERATOR_VERBOSE")) if verbose is None else verbose
        self._phase = "start"
        self.env, self.w = env, env.world
        self.k = ArmKinematics(env.world)
        self.record_every = max(1, int(round(record_every*1e-3/dt)))    # ticks
        self.snaps, self.log = [], []
        self._wall0 = time.perf_counter()
        self.t0 = time.perf_counter()
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.scanner_frames = []            # (tick, png path) of every processed imager frame
        if self.frames_dir:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            for old in self.frames_dir.glob("imager_*.png"):    # a re-run's folder held the earlier runs' frames too
                old.unlink()
            prev = env.frame_hook

            def hook(e, frame, rad, meta):
                from PIL import Image
                p = self.frames_dir/f"imager_{meta['tick']:07d}.png"
                Image.fromarray(frame).save(p)
                self.scanner_frames.append(dict(tick=int(meta["tick"]), path=str(p), found=meta["found"],
                                                reported=meta["reported"], exposure_s=meta["exposure_s"], gain=meta["gain"]))
                if prev:
                    prev(e, frame, rad, meta)
            env.frame_hook = hook
        self._pending = {}
        from .manifest import launch_stamp
        self.code_at_launch = launch_stamp()
        self.snapshot()

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        if value != self._phase and getattr(self, "verbose", False):
            print(f"PHASE t={self.w.sim_time:8.3f} wall={time.perf_counter() - self._wall0:7.1f}s  {value}", flush=True)
        self._phase = value

    # ---------------------------------------------------------------- stepping and recording
    def snapshot(self):
        env, w = self.env, self.w
        fw, t = env.firmware, w.sim_time
        self.snaps.append(dict(tick=int(w.tick), t=float(t), phase=self.phase, body_q=w.state.body_q.numpy().copy(),
                               joint_q=w.state.joint_q.numpy().copy(), joint_target=w.targets.copy(),
                               led_rel=float(fw.led_intensity(t)/P.LED_INTENSITY), aim=bool(fw.aim_on(t)), good_read=bool(fw.good_read_led(t)),
                               fw_state=fw.state, screen=json.loads(json.dumps(env.host.screen)),
                               trigger_mm=float(w.trigger_travel_m()*1e3), switch=bool(w.trigger.state), seated=bool(w.seated),
                               reads=len(fw.reads)))

    def _chunk(self, n=None):
        n = self.chunk if n is None else n
        n0 = len(self.w.reflexes)
        self.env.act(self._pending, ticks=n)
        self._pending = {}
        if self.verbose and len(self.w.reflexes) > n0:
            for r in self.w.reflexes[n0:]:
                print(f"REFLEX t={self.w.sim_time:8.3f} phase={self.phase!r} {json.dumps(r, default=float)}", flush=True)
        if self.w.tick % self.record_every == 0:
            self.snapshot()

    def wait(self, seconds):
        for _ in range(max(1, int(round(seconds/CHUNK_S)))):
            self._chunk()

    def set_targets(self, targets):
        self._pending.update(targets)

    # ---------------------------------------------------------------- arm and hand
    def tcp(self, arm):
        T = self.k.forward(arm)
        return T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3])

    def move(self, arm, position, quat_xyzw=None, seconds=1.0, tol_m=3e-3, tol_rad=.05):
        """Straight-line Cartesian move (smooth-step profile), IK waypoints every 10 ms seeded by the previous
        solution; a branch jump > 0.25 rad is refused. The servos do the moving."""
        p0, r0 = self.tcp(arm)
        r1 = Rotation.from_quat(quat_xyzw) if quat_xyzw is not None else r0
        slerp = Slerp([0, 1], Rotation.concatenate([r0, r1]))
        n = max(1, int(round(seconds/CHUNK_S)))
        seed = self.k.joints_now(arm)
        for i in range(1, n + 1):
            s = i/n
            s = 3*s*s - 2*s*s*s
            p = p0 + (np.asarray(position, float) - p0)*s
            targets = self.k.solve(arm, p, slerp(s).as_quat(), seed=seed, tol_m=tol_m, tol_rad=tol_rad)
            jt = np.array(list(targets.values()))
            if np.abs(jt - seed).max() > .25:
                raise ValueError(f"IK branch jump at waypoint {i}/{n}: {np.abs(jt - seed).max():.2f} rad")
            seed = jt
            self.set_targets(targets)
            self._chunk()

    def move_joints(self, arm, position, quat_xyzw, seconds=1.5, restarts=16):
        """Large reconfiguration: IK goal once (with restarts), then interpolate in joint space."""
        goal = self.k.solve(arm, position, quat_xyzw, tol_m=2e-3, tol_rad=.03, restarts=restarts)
        start = dict(zip(goal, map(float, self.k.joints_now(arm))))
        n = max(1, int(round(seconds/CHUNK_S)))
        for i in range(1, n + 1):
            s = i/n; s = 3*s*s - 2*s*s*s
            self.set_targets({j: start[j] + (goal[j] - start[j])*s for j in goal})
            self._chunk()

    def open_hand(self, arm, half_width=.04, seconds=.6):
        """Open the fingers to `half_width` each, waiting at least `seconds` and until they are there (at most 1.2 s
        more; from closed the Franka fingers need 0.8 s to open fully, and a path planned for the open hand was
        started with the fingers still half shut)."""
        self.w.set_grip_force(arm, 0.)
        self.set_targets({j: half_width for j in self.w.gripper_joints(arm)})
        self.wait(seconds)
        q = lambda: self.w.state.joint_q.numpy()
        for _ in range(12):
            if all(abs(float(q()[self.w.joints[j]["q"]]) - half_width) < .001 for j in self.w.gripper_joints(arm)):
                break
            self.wait(.1)

    def grasp(self, arm, force_per_finger, seconds=.8):
        """Franka Hand grasp: fingers close at <= 50 mm/s and squeeze with the force once blocked."""
        self.w.set_grip_force(arm, force_per_finger)
        self.set_targets({j: 0. for j in self.w.gripper_joints(arm)})
        self.wait(seconds)

    def finger_width(self, arm):
        q = self.w.state.joint_q.numpy()
        return float(sum(q[self.w.joints[j]["q"]] for j in self.w.gripper_joints(arm)))

    # ---------------------------------------------------------------- output
    def save(self, folder, name, extra=None):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        S = self.snaps
        np.savez_compressed(folder/f"{name}_frames.npz", tick=np.array([s["tick"] for s in S]),
                            body_q=np.stack([s["body_q"] for s in S]), joint_q=np.stack([s["joint_q"] for s in S]),
                            joint_target=np.stack([s["joint_target"] for s in S]), phase=np.array([s["phase"] for s in S]))
        states = [{k: v for k, v in s.items() if k not in ("body_q", "joint_q", "joint_target")} for s in S]
        from . import params as P
        from .scene import Scene
        meta = dict(name=name, sim_s=self.w.sim_time, wall_s=time.perf_counter() - self.t0, timing=self.w.timing,
                    scene_overrides=dict(STOCKROOM_CRADLE=list(Scene.STOCKROOM_CRADLE)),
                    config=vars(self.env.config), reflexes=self.w.reflexes, reads=_jsonable(self.env.firmware.reads),
                    host_record=self.env.host.record, evaluation=self.env.evaluator.evaluate(), fault=self.w.fault,
                    scanner_frames=self.scanner_frames, ledger=P.ledger(), code_at_launch=self.code_at_launch,
                    **(extra or {}))
        (folder/f"{name}_states.json").write_text(json.dumps(states, default=float))
        (folder/f"{name}_log.json").write_text(json.dumps(dict(meta=meta, log=self.log), indent=1, default=_default))
        return meta


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


def _jsonable(reads):
    return [{k: v for k, v in r.items() if k != "quad_px"} for r in reads]
