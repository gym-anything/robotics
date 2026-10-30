"""The arms' motion generator: what the Franka controller does to a commanded joint position before its servo follows
it (FCI "Control Interface Specification and Robot Limits"; libfranka `limitRate`, rate_limiting.h).

Every arm command (a policy action, a keyboard chord, a scripted waypoint) sets a goal for some joints and a duration.
The command is interpolated linearly from the previous goal over its duration, and a time-optimal tracker moves the
position reference toward that moving target under the model's joint velocity, acceleration and jerk limits (FR3:
10 rad/s^2 and 5000 rad/s^3 on every joint; FER/Panda: 7.5-20 rad/s^2 and 3750-10000 rad/s^3). The tracker brakes
on the curve v = sqrt(2 a e) so it stops at a goal without overshoot. A Cartesian check then scales the joint
accelerations of an arm so that its TCP stays within the FCI Cartesian limits (FR3: 3.0 m/s, 9 m/s^2, 2.5 rad/s,
17 rad/s^2; FER: 1.7 m/s, 13 m/s^2, 2.5 rad/s, 25 rad/s^2).
"""

import math
import numpy as np

# FCI joint-space limits, joints 1-7 (frankarobotics.github.io/docs/robot_specifications.html, fetched 2026-10-04)
JOINT_ACC = {"fr3": (10.,) * 7, "panda": (15., 7.5, 10., 12.5, 15., 20., 20.)}
JOINT_JERK = {"fr3": (5000.,) * 7, "panda": (7500., 3750., 5000., 6250., 7500., 10000., 10000.)}
# FCI Cartesian limits: translation velocity, acceleration; rotation velocity, acceleration
CART = {"fr3": (3.0, 9.0, 2.5, 17.0), "panda": (1.7, 13.0, 2.5, 25.0)}
BRAKE_FRACTION = 0.9          # brake on 90 % of the acceleration limit: room for the jerk-limited ramp into braking
SETTLE_TAU_S = 0.01           # last stretch of an approach: exponential with this time constant (the square-root brake
                              # curve is infinitely steep at the goal, and the jerk limit cannot follow it there: the
                              # reference circled the goal by 1e-4 rad)


def family(model_id):
    m = str(model_id).split("@")[0].lower()
    if m.startswith("fr3"):
        return "fr3"
    if m.startswith("panda") or m.startswith("fer"):
        return "panda"
    raise ValueError(f"no FCI motion limits for robot model {model_id}")


def _max_scale(a0, b, limit):
    """Largest k in [0, 1] with |a0 + k b| <= limit (vectors); 0 when even k = 0 violates it."""
    a0, b = np.asarray(a0, float), np.asarray(b, float)
    if np.linalg.norm(a0 + b) <= limit:
        return 1.
    if np.linalg.norm(a0) > limit:
        return 0.
    # |a0 + k b|^2 = limit^2: bb k^2 + 2 ab k + aa - L^2 = 0, the positive root
    bb, ab, aa = float(b @ b), float(a0 @ b), float(a0 @ a0)
    disc = ab*ab - bb*(aa - limit*limit)
    return float(np.clip((-ab + math.sqrt(max(disc, 0.)))/max(bb, 1e-30), 0., 1.))


def _rotvec(Ra, Rb):
    """Rotation vector (world frame) taking Ra to Rb."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(Rb @ Ra.T).as_rotvec()


class ArmMotion:
    """Motion generator of one arm's seven joints."""

    def __init__(self, family_name, vmax, dt, q0, fk=None):
        self.fam = family_name
        self.vmax = np.asarray(vmax, float)
        self.amax = np.asarray(JOINT_ACC[family_name], float)
        self.jmax = np.asarray(JOINT_JERK[family_name], float)
        self.cart = CART[family_name]
        self.dt = float(dt)
        self.fk = fk
        q0 = np.asarray(q0, float)
        self.qr, self.vr, self.ar = q0.copy(), np.zeros(7), np.zeros(7)
        self.start, self.goal = q0.copy(), q0.copy()
        self.t, self.n = 0, 0
        self._fk_prev = None          # TCP pose one tick ago and now: (p, R), (p, R)
        self.peak = dict(joint_acc=0., tcp_lin_acc=0., tcp_ang_acc=0., tcp_lin_vel=0., tcp_ang_vel=0.)

    def state(self):
        return dict(qr=self.qr.copy(), vr=self.vr.copy(), ar=self.ar.copy(), start=self.start.copy(),
                    goal=self.goal.copy(), t=self.t, n=self.n)

    def set_state(self, s):
        for k in ("qr", "vr", "ar", "start", "goal"):
            setattr(self, k, np.asarray(s[k], float).copy())
        self.t, self.n = int(s["t"]), int(s["n"])
        self._fk_prev = None

    def command_now(self):
        return self.start + (self.goal - self.start)*(min(1., self.t/self.n) if self.n else 1.)

    def command(self, goal, ticks, mask=None):
        """New goal for the joints in `mask` (all by default), reached by a linear ramp over `ticks` ticks."""
        now = self.command_now()
        goal = np.asarray(goal, float)
        m = np.ones(7, bool) if mask is None else np.asarray(mask, bool)
        self.start = now
        self.goal = np.where(m, goal, now)
        self.t, self.n = 0, max(int(ticks), 1)

    def hold(self, q):
        """Stop where the arm is (protective stop): reference, command and goal at q, at rest."""
        q = np.asarray(q, float)
        self.qr, self.start, self.goal = q.copy(), q.copy(), q.copy()
        self.vr[:] = 0.
        self.ar[:] = 0.
        self.t, self.n = 0, 0
        self._fk_prev = None

    def moving(self):
        return bool(np.abs(self.goal - self.qr).max() > 1e-6 or np.abs(self.vr).max() > 1e-6)

    def _tcp(self, q):
        T = self.fk(q)
        return T[:3, 3].copy(), T[:3, :3].copy()

    def step(self):
        dt = self.dt
        self.t = min(self.t + 1, self.n) if self.n else 0
        target = self.command_now()
        e = target - self.qr
        ab = BRAKE_FRACTION*self.amax
        e_lin = 2.*ab*SETTLE_TAU_S**2                      # where the brake curve and the linear approach meet
        vdes = np.where(np.abs(e) > e_lin, np.sign(e)*np.sqrt(2.*ab*np.abs(e)), e/SETTLE_TAU_S)
        vdes = np.clip(vdes, -self.vmax, self.vmax)
        ades = np.clip((vdes - self.vr)/dt, -self.amax, self.amax)
        if self.fk is not None and (np.abs(ades).max() > 0. or np.abs(self.vr).max() > 0.):
            ades = ades*self._cartesian_scale(ades)
        a = np.clip(ades, self.ar - self.jmax*dt, self.ar + self.jmax*dt)
        a = np.clip(a, -self.amax, self.amax)
        v = np.clip(self.vr + a*dt, -self.vmax, self.vmax)
        q = self.qr + 0.5*(self.vr + v)*dt
        # arrive: snap onto a goal reached at rest (the brake curve's discrete tail)
        done = (np.abs(target - q) < 1e-6) & (np.abs(v) < 1e-4) & (self.t >= self.n)
        q = np.where(done, target, q)
        v = np.where(done, 0., v)
        a = np.where(done, 0., (v - self.vr)/dt)
        self.peak["joint_acc"] = max(self.peak["joint_acc"], float(np.abs(a).max()))
        self.qr, self.vr, self.ar = q, v, a
        return q

    def _cartesian_scale(self, ades):
        """Largest k in [0, 1] such that the TCP, with joint accelerations k * ades, stays within the Cartesian
        velocity and acceleration limits over the next tick (finite differences of the forward kinematics)."""
        dt = self.dt
        if self._fk_prev is None:
            self._fk_prev = (self._tcp(self.qr - self.vr*dt), self._tcp(self.qr))
        (pm, Rm), (p0, R0) = self._fk_prev
        q_zero = self.qr + self.vr*dt
        q_full = self.qr + (self.vr + ades*dt)*dt
        pz, Rz = self._tcp(q_zero)
        pf, Rf = self._tcp(q_full)
        v_lin_m, v_ang_m = (p0 - pm)/dt, _rotvec(Rm, R0)/dt
        vz_lin, vz_ang = (pz - p0)/dt, _rotvec(R0, Rz)/dt
        vf_lin, vf_ang = (pf - p0)/dt, _rotvec(R0, Rf)/dt
        vmax_l, amax_l, vmax_a, amax_a = self.cart
        k = min(_max_scale(vz_lin, vf_lin - vz_lin, vmax_l), _max_scale(vz_ang, vf_ang - vz_ang, vmax_a),
                _max_scale((vz_lin - v_lin_m)/dt, (vf_lin - vz_lin)/dt, amax_l),
                _max_scale((vz_ang - v_ang_m)/dt, (vf_ang - vz_ang)/dt, amax_a))
        return k

    def advance_fk(self):
        """After the step: record the TCP poses the Cartesian check differences, and the peaks it reports."""
        if self.fk is None:
            return
        if self._fk_prev is not None and not self.vr.any() and not self.ar.any() and self._fk_prev[0] is self._fk_prev[1]:
            return                                       # at rest, poses unchanged
        cur = self._tcp(self.qr)
        if self._fk_prev is not None:
            (pm, Rm), (p0, R0) = self._fk_prev
            dt = self.dt
            v1, w1 = (cur[0] - p0)/dt, _rotvec(R0, cur[1])/dt
            v0, w0 = (p0 - pm)/dt, _rotvec(Rm, R0)/dt
            self.peak["tcp_lin_vel"] = max(self.peak["tcp_lin_vel"], float(np.linalg.norm(v1)))
            self.peak["tcp_ang_vel"] = max(self.peak["tcp_ang_vel"], float(np.linalg.norm(w1)))
            self.peak["tcp_lin_acc"] = max(self.peak["tcp_lin_acc"], float(np.linalg.norm(v1 - v0)/dt))
            self.peak["tcp_ang_acc"] = max(self.peak["tcp_ang_acc"], float(np.linalg.norm(w1 - w0)/dt))
            at_rest = not self.vr.any() and not self.ar.any()
            self._fk_prev = (cur, cur) if at_rest else ((p0, R0), cur)
        else:
            self._fk_prev = (cur, cur)
