"""Motor model of the H1-2 and its Inspire hands: a Unitree low-level motor command per joint (q*, dq*, kp, kd, tau_ff)
with the motors' limits (the shared robots catalog's `forklift` controller profile; robots/models/h1_2/robot.json).

    tau = kp (q* - q) + kd (dq* - dq) + tau_ff,
saturated by a torque-speed envelope: motoring |tau| <= tau_max (1 - |dq|/w_nl), braking |tau| <= tau_max, with tau_max
and w_nl the URDF effort and velocity limits (a DC-motor line: an estimate; Unitree publishes no H1-2 curves).
Each robot dof's MuJoCo position actuator becomes a general actuator with gain 1 and bias (0, -kp, -kd) and
ctrl = kp q* + kd dq* + tau_ff, so the kd term is integrated implicitly by implicitfast and the saturation applies to the
summed torque. The Inspire hands' mimic followers carry zero gain: the
URDF mimic equality drives them; their channels are not commandable.
"""

import math

import numpy as np

from . import params as P

# the policy's gain bounds and the reflected rotor inertia per joint class: params.ROBOT_GAIN_BOUNDS, ROBOT_ARMATURE
GAIN_BOUNDS = P.ROBOT_GAIN_BOUNDS
ARMATURE = P.ROBOT_ARMATURE


def group_of(name):
    s = name.split("/")[-1]
    if s[:2] in ("L_", "R_"):
        return "hand"
    if any(k in s for k in ("hip", "knee", "ankle")):
        return "leg"
    if s.startswith("torso"):
        return "torso"
    if "wrist" in s:
        return "wrist"
    return "arm"


class Motors:
    def __init__(self, world, joints, profile):
        """joints: {name: address dict from the robots runtime}; profile: the catalog profile's joint table."""
        self.w = world
        m = world.solver.mj_model
        self.names = [n for n in joints if profile[n.split("/")[-1]]["kp"] > 0.0]     # commandable (not mimic followers)
        self.followers = [n for n in joints if n not in self.names]
        n = len(self.names)
        self.q_idx = np.array([joints[k]["q"] for k in self.names], int)
        self.dof = np.array([joints[k]["dof"] for k in self.names], int)
        self.lower = np.array([joints[k]["lower"] for k in self.names])
        self.upper = np.array([joints[k]["upper"] for k in self.names])
        prof = [profile[k.split("/")[-1]] for k in self.names]
        self.effort = np.array([p["effort_limit"] for p in prof], float)
        self.w_nl = np.array([p["velocity_limit"] for p in prof], float)
        self.kp_default = np.array([p["kp"] for p in prof], float)
        self.kd_default = np.array([p["kd"] for p in prof], float)
        self.group = [group_of(k) for k in self.names]
        self.kp_max = np.array([GAIN_BOUNDS[g][0] for g in self.group])
        self.kd_max = np.array([GAIN_BOUNDS[g][1] for g in self.group])
        self.index = {k: i for i, k in enumerate(self.names)}
        self.cmd = dict(q=np.zeros(n), dq=np.zeros(n), kp=self.kp_default.copy(), kd=self.kd_default.copy(),
                        tau=np.zeros(n))
        self._gains_dirty = True
        to_newton = world.solver.mjc_actuator_to_newton_idx.numpy().reshape(-1)
        pos = {int(d): a for a, d in enumerate(to_newton) if d >= 0}
        vel = {int(-(d + 2)): a for a, d in enumerate(to_newton) if d <= -2}
        self.act_pos = np.array([pos[int(d)] for d in self.dof], int)
        self.act_vel = np.array([vel.get(int(d), -1) for d in self.dof], int)
        m.actuator_gainprm[self.act_pos, :] = 0.0
        m.actuator_gainprm[self.act_pos, 0] = 1.0
        m.actuator_biasprm[self.act_pos, :] = 0.0
        m.actuator_ctrllimited[self.act_pos] = 0
        m.actuator_forcelimited[self.act_pos] = 1
        av = self.act_vel[self.act_vel >= 0]
        m.actuator_gainprm[av, :] = 0.0
        m.actuator_biasprm[av, :] = 0.0
        m.actuator_forcelimited[av] = 1
        m.actuator_forcerange[av] = 0.0
        # mimic followers: their actuators off (the equality drives them)
        fdof = [joints[k]["dof"] for k in self.followers]
        for d in fdof:
            for table in (pos, vel):
                a = table.get(int(d))
                if a is not None:
                    m.actuator_gainprm[a, :] = 0.0
                    m.actuator_biasprm[a, :] = 0.0
                    m.actuator_forcelimited[a] = 1
                    m.actuator_forcerange[a] = 0.0

    # ------------------------------------------------------------------ commands
    def hold_current(self):
        q = self.w.state.joint_q.numpy()
        self.cmd["q"][:] = np.clip(q[self.q_idx], self.lower, self.upper)
        self.cmd["dq"][:] = 0.0
        self.cmd["tau"][:] = 0.0
        self.cmd["kp"][:] = self.kp_default
        self.cmd["kd"][:] = self.kd_default
        self._gains_dirty = True

    def parse(self, targets):
        """{joint or joint.dq/.kp/.kd/.tau: value} -> [(channel, index, value)]; raises ValueError (no side effect)."""
        out = []
        for key, value in targets.items():
            name, _, suffix = key.partition(".")
            if name not in self.index:
                if name in self.followers:
                    raise ValueError(f"{name} is a mimic follower of the Inspire hand (driven by its linkage), not a motor")
                raise ValueError(f"Not a commanded motor: {name}")
            i = self.index[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            v = float(value)
            if suffix == "":
                if not self.lower[i] <= v <= self.upper[i]:
                    raise ValueError(f"{name} target {v:.4f} outside its limits [{self.lower[i]:.4f}, {self.upper[i]:.4f}]")
                out.append(("q", i, v))
            elif suffix == "dq":
                if abs(v) > self.w_nl[i]:
                    raise ValueError(f"{key} must be within +-{self.w_nl[i]:g} rad/s (the motor's speed)")
                out.append(("dq", i, v))
            elif suffix == "kp":
                if not 0.0 <= v <= self.kp_max[i]:
                    raise ValueError(f"{key} must be in [0, {self.kp_max[i]:g}]")
                out.append(("kp", i, v))
            elif suffix == "kd":
                if not 0.0 <= v <= self.kd_max[i]:
                    raise ValueError(f"{key} must be in [0, {self.kd_max[i]:g}]")
                out.append(("kd", i, v))
            elif suffix == "tau":
                if abs(v) > self.effort[i]:
                    raise ValueError(f"{key} must be within +-{self.effort[i]:g} N m (the motor's effort limit)")
                out.append(("tau", i, v))
            else:
                raise ValueError(f"Unknown command channel {key!r}; channels: <joint>, <joint>.dq/.kp/.kd/.tau")
        return out

    def apply(self, parsed):
        for ch, i, v in parsed:
            self.cmd[ch][i] = v
            if ch in ("kp", "kd"):
                self._gains_dirty = True

    def channels(self):
        out = {}
        for i, n in enumerate(self.names):
            out[n] = dict(unit="radian", lower=float(self.lower[i]), upper=float(self.upper[i]))
            out[n + ".dq"] = dict(unit="radian/s", lower=-float(self.w_nl[i]), upper=float(self.w_nl[i]))
            out[n + ".kp"] = dict(unit="N m/rad", lower=0.0, upper=float(self.kp_max[i]), default=float(self.kp_default[i]))
            out[n + ".kd"] = dict(unit="N m s/rad", lower=0.0, upper=float(self.kd_max[i]), default=float(self.kd_default[i]))
            out[n + ".tau"] = dict(unit="N m", lower=-float(self.effort[i]), upper=float(self.effort[i]))
        return out

    # ------------------------------------------------------------------ every physics step
    def write(self, control_q, qd):
        m = self.w.solver.mj_model
        c = self.cmd
        if self._gains_dirty:
            m.actuator_biasprm[self.act_pos, 1] = -c["kp"]
            m.actuator_biasprm[self.act_pos, 2] = -c["kd"]
            self._gains_dirty = False
        control_q[self.q_idx] = c["kp"] * c["q"] + c["kd"] * c["dq"] + c["tau"]
        v = qd[self.dof]
        motoring = self.effort * np.clip(1.0 - np.abs(v) / self.w_nl, 0.0, 1.0)
        m.actuator_forcerange[self.act_pos, 0] = np.where(v >= 0.0, -self.effort, -motoring)
        m.actuator_forcerange[self.act_pos, 1] = np.where(v >= 0.0, motoring, self.effort)

    def torques(self):
        return np.asarray(self.w.solver.mj_data.actuator_force)[self.act_pos].copy()

    def checkpoint(self):
        return {k: v.copy() for k, v in self.cmd.items()}

    def restore(self, snap):
        for k, v in snap.items():
            self.cmd[k][:] = v
        self._gains_dirty = True
