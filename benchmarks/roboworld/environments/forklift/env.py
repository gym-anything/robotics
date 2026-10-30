"""Persistent forklift environment: robot-only actions, exact ticks, explicit faults.

One action = a set of H1-2 motor command channels (motors.Motors: q*, dq*, kp, kd, tau per joint) held for N ticks of
0.5 ms. The start state is built (task.py), settled by physics until at rest over a window (World.settle) and checked
(a start state with any problem is refused: StartStateRefused names the problems); reset restores that settled, checked
state exactly (checkpoint, including the truck's systems and the event latches).
"""

import copy
import threading
from pathlib import Path

import numpy as np

from . import params as P
from .config import Config
from .world import World

START_FORCE_MAX = 1.0e5     # N: no single contact in a start state at rest carries more (the laden truck weighs ~65 kN)


class StartStateRefused(RuntimeError):
    """The start state built for (task, seed) failed its start check (ForkliftEnv._check_start): it is refused, never
    handed to an operator. Carries the check and the settle report for the probes that record it."""

    def __init__(self, config, start_check, settle_record):
        self.task, self.seed = config.task, config.seed
        self.problems = list(start_check.get("problems") or [])
        self.start_check, self.settle_record = start_check, settle_record
        super().__init__(f"forklift start state refused ({config.preset}/{config.task} seed {config.seed}): "
                         + "; ".join(self.problems))


def _default_output():
    """A fresh folder for an environment built without output_dir: under GAR's workspace (`$GAR_WORKSPACE/.gar/
    forklift-env/`) when one is set, else the system's temporary folder. Never inside the package: an installed copy
    is not writable and the checkout's folder is not ignored by Git."""
    import os
    import tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = Path(ws) / ".gar" / "forklift-env" if ws else Path(tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="forklift-env-", dir=base))


class ForkliftEnv:
    def __init__(self, config=None, *, output_dir=None, fixture=None):
        """fixture: ASSISTED probes/demos only, callable(world) -> fixture installed after construction (never reachable
        from the backend)."""
        self.config = config or Config()
        self.lock = threading.RLock()
        self.output = Path(output_dir) if output_dir else _default_output()
        self.output.mkdir(parents=True, exist_ok=True)
        self.world = World(self.config)
        self.generation = 0
        self.frame_callback = None
        self.trace_callback = None
        self.settle_record = self.world.settle()
        self.start_check = self._check_start()
        if self.start_check["problems"]:
            raise StartStateRefused(self.config, self.start_check, self.settle_record)
        self._record_start_poses()
        if fixture is not None:
            self.world.fixture = fixture(self.world)
        self.initial = self.checkpoint()
        self.world.on_tick.append(self._on_tick)

    # ------------------------------------------------------------------ start state
    def _record_start_poses(self):
        w = self.world
        sc = w.scene
        sc.load_start_pose = {n: w.state.body_q.numpy()[info["body"], :3].astype(float).tolist() for n, info in sc.loads.items()}
        if "carried" in sc.loads:
            from .evaluator import pallet_corners, body_frame
            _, o, R = pallet_corners(w, "carried")
            sc.carton_rel0 = [R.T @ (body_frame(w, info["body"])[0] - o) for n, info in sorted(sc.loads.items())
                              if n.startswith("carried_c")]

    def _check_start(self):
        """Start-state checks before tick 0: settled, operator seated (seat switch closed), no wheel off the ground, the
        carried load on the forks, every penetration within contact tolerance."""
        w = self.world
        import mujoco
        m, d = w.solver.mj_model, w.solver.mj_data
        deepest = min([float(d.contact[i].dist) for i in range(d.ncon)] or [0.0])
        # the largest contact force at rest: a leveler deck jammed into two dock bumpers carried ~2.9 MN while every
        # penetration stayed sub-millimetre; nothing standing here weighs more than ~65 kN (the laden truck)
        f6 = np.zeros(6)
        f_max, f_pair = 0.0, None
        for i in range(d.ncon):
            mujoco.mj_contactForce(m, d, i, f6)
            if abs(f6[0]) > f_max:
                f_max = abs(float(f6[0]))
                f_pair = (w.mj_geom_label(int(d.contact[i].geom1)), w.mj_geom_label(int(d.contact[i].geom2)))
        tyres = {k: round(v["Fz"] / P.G, 1) for k, v in w.tyres.telemetry.items()}
        rec = dict(settled=self.settle_record["settled"], deepest_contact_m=deepest, max_contact_force_N=f_max,
                   max_contact_pair=f_pair, tyre_loads_kg=tyres,
                   seat_switch=bool(w.signals.get("seat_switch")) if w.rig is not None else None)
        problems = []
        if not self.settle_record["settled"]:
            problems.append(f"not at rest after {self.settle_record['sim_s']:.1f} s ({self.settle_record['slowest']})")
        if any(v <= 1.0 for v in tyres.values()):
            problems.append("a wheel is off the ground")
        if deepest < -0.01:
            problems.append(f"penetration {-deepest * 1000:.1f} mm (over the 10 mm limit)")
        if f_max > START_FORCE_MAX:
            problems.append(f"contact force {f_max / 1000:.0f} kN at rest between {f_pair[0]} and {f_pair[1]}")
        if w.rig is not None and not w.signals.get("seat_switch"):
            problems.append("operator not on the seat (seat switch open)")
        rec["problems"] = problems
        return rec

    # ------------------------------------------------------------------ checkpoints
    def checkpoint(self):
        w = self.world
        d = w.solver.mj_data
        snap = dict(rigid={a: getattr(w.state, a).numpy().copy() for a in ("body_q", "body_qd", "joint_q", "joint_qd")},
                    mj=dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), act=d.act.copy(), qacc_warmstart=d.qacc_warmstart.copy(),
                            time=float(d.time)),
                    control=w.control.joint_target_q.numpy().copy(), tick=w.tick, systems=copy.deepcopy(w.systems_checkpoint()))
        if w.rig is not None:
            snap["motors"] = w.operator.motors.checkpoint()
            snap["belt"] = copy.deepcopy(w.operator.belt)
        return snap

    def restore(self, snap):
        w = self.world
        for a, v in snap["rigid"].items():
            getattr(w.state, a).assign(v)
            getattr(w.next_state, a).assign(v)
        w.control.joint_target_q.assign(snap["control"])
        d = w.solver.mj_data
        d.qpos[:], d.qvel[:], d.act[:] = snap["mj"]["qpos"], snap["mj"]["qvel"], snap["mj"]["act"]
        d.qacc_warmstart[:] = snap["mj"]["qacc_warmstart"]
        d.time = snap["mj"]["time"]
        w._mujoco.mj_forward(w.solver.mj_model, d)
        w.systems_restore(copy.deepcopy(snap["systems"]))
        w.tick = snap["tick"]
        if w.rig is not None:
            w.operator.motors.restore(snap["motors"])
            w.operator.belt = copy.deepcopy(snap["belt"])
        w.fault = None

    def reset(self):
        with self.lock:
            self.restore(self.initial)
            self.generation += 1
            return self.get_state()

    # ------------------------------------------------------------------ acting
    def act(self, targets, ticks):
        """Apply motor command channels (validated as a whole first) and run `ticks` physics ticks."""
        with self.lock:
            w = self.world
            if w.rig is None:
                raise ValueError("this session has no robot (robot='none' is a probe configuration)")
            parsed = w.operator.motors.parse(targets)
            w.operator.motors.apply(parsed)
            for _ in range(int(ticks)):
                w.step()
            return self.get_state()

    def _on_tick(self, world):
        if self.trace_callback is not None:
            self.trace_callback(self)
        if self.frame_callback is not None:
            self.frame_callback(world)

    # ------------------------------------------------------------------ state
    def joints(self):
        w = self.world
        if w.rig is None:
            return {}
        q, qd = w.state.joint_q.numpy(), w.state.joint_qd.numpy()
        mo = w.operator.motors
        tau = mo.torques()
        return {n: dict(position=float(q[mo.q_idx[i]]), velocity=float(qd[mo.dof[i]]), torque=float(tau[i]))
                for i, n in enumerate(mo.names)}

    def robot_status(self):
        """What the H1-2's own pelvis IMU reports: orientation (roll, pitch, yaw from the pelvis body's pose) and angular
        rate (gyro, pelvis frame). No linear acceleration (the observation carries none) and nothing about the truck;
        the joints' states (hands included) are in joints()."""
        w = self.world
        if w.rig is None:
            return {}
        import math
        b = w.operator.pelvis
        q = w.state.body_q.numpy()[b].astype(float)
        qd = w.state.body_qd.numpy()[b].astype(float)
        from .mathx import quat_to_R
        R = quat_to_R(q[None, 3:7])[0]
        rpy = (math.atan2(R[2, 1], R[2, 2]), math.asin(max(-1.0, min(1.0, -R[2, 0]))), math.atan2(R[1, 0], R[0, 0]))
        gyro = R.T @ qd[3:]
        return dict(imu=dict(rpy_rad=[float(v) for v in rpy], gyro_rad_s=[float(v) for v in gyro]))

    def get_state(self):
        with self.lock:
            w = self.world
            return dict(tick=w.tick, time_s=round(w.sim_time, 6), generation=self.generation, ready=True,
                        finite=w.fault is None, fault=w.fault, joints=self.joints(), robot_status=self.robot_status(),
                        settle=self.settle_record, start_check=self.start_check)

    def owner_state(self):
        """Privileged truck and world state (owner credentials, evidence, never the policy)."""
        w = self.world
        v, yaw_rate = w.truck_speed()
        p, R = w.body_pose("chassis")
        return dict(truck=dict(position=p.tolist(), speed_m_s=v, yaw_rate=yaw_rate, signals={k: v for k, v in w.signals.items()},
                               drivetrain=w.drivetrain.tele, steering=w.steering.tele, hydraulics=w.hydraulics.tele,
                               tyres={k: {kk: vv for kk, vv in t.items() if kk not in ("ribs", "contact")}
                                      for k, t in w.tyres.telemetry.items()},
                               joints={n: w.joint(n)[0] for n in ("mast_outer", "mast_int", "carriage", "sideshift",
                                                                  "steering_wheel", "knuckle_l", "knuckle_r", "seat")}),
                    loads={n: w.state.body_q.numpy()[info["body"]].astype(float).tolist() for n, info in w.scene.loads.items()},
                    events=w.events.summary())
