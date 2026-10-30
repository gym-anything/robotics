"""TMR base controller: a base twist -> two ArgoDrive steering angles and wheel speeds (swerve inverse kinematics).

The real TMR takes a Cartesian velocity command (manual R04245 section 5.10); this is that controller's job here:
  * the commanded twist (v_x, v_y, w_z in base_link) is slewed toward the request within the software limits
    (params.BASE_LIMITS: 1.0 m/s, 1.0 rad/s, 0.8 m/s2, 1.5 rad/s2);
  * drive i at r_i = (x_i, y_i) needs the contact velocity v_i = (v_x - w_z y_i, v_y + w_z x_i): steering angle
    atan2(v_iy, v_ix) (or that +- pi with the wheel reversed, whichever is nearer the current angle; the AD-100.12
    steers without limit at 180 deg/s) and wheel speed |v_i| / r, scaled by cos(steering error) so a wheel does not
    drive across its own direction while it is still turning;
  * each wheel servo tracks a wheel-angle reference integrated from that speed; the reference never leads the wheel
    by more than the drive's torque limit / kp (16 N m when accelerating, 28.4 N m when braking, AD-100.12), so a
    blocked base does not wind up and lurch when released. At zero command the reference stops: the wheel servo
    holds its angle, up to the motor braking torque 28.4 N m (the drive's spring-applied holding brake, 35.5 N m slip
    torque, is not modelled);
  * odometry: the base twist estimated from the measured steering angles and wheel speeds (least squares), integrated
    in the odometry frame, as the TMR's own wheel odometry; it drifts when wheels slip.
"""

import math
import numpy as np
from . import params as P


def wrap(a):
    return (a + math.pi) % (2*math.pi) - math.pi


class BaseController:
    def __init__(self, joints, prefix, dt):
        self.dt = dt
        self.drives = []
        for key, steer, wheel in (("front", "tmrv0_2_joint_0", "tmrv0_2_joint_1"), ("rear", "tmrv0_2_joint_2", "tmrv0_2_joint_3")):
            s, w = joints[f"{prefix}/{steer}"], joints[f"{prefix}/{wheel}"]
            self.drives.append(dict(name=key, pos=np.array(P.DRIVES[key], float), steer=s, wheel=w, steer_name=f"{prefix}/{steer}",
                                    wheel_name=f"{prefix}/{wheel}"))
        self.request = np.zeros(3)
        self.cmd = np.zeros(3)
        self.steer_ref = [0., 0.]
        self.wheel_ref = None
        self.odom = np.zeros(3)
        self.odom_twist = np.zeros(3)
        self.kp = None

    def set_twist(self, vx, vy, wz):
        lim = P.BASE_LIMITS
        v = math.hypot(vx, vy)
        for name, val in (("vx", vx), ("vy", vy), ("wz", wz)):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"base twist {name} must be finite")
        if v > lim["v"] + 1e-9 or abs(wz) > lim["w"] + 1e-9:
            raise ValueError(f"base twist exceeds the limits ({lim['v']} m/s, {lim['w']} rad/s)")
        self.request = np.array([vx, vy, wz], float)

    @property
    def moving(self):
        return bool(np.abs(self.cmd).max() > 1e-6 or np.abs(self.request).max() > 0)

    def step(self, q, qd, cur, vel, kp, kd):
        """Advance one tick: write steering and wheel servo targets into `cur` (joint coordinate layout) and their
        velocity targets into `vel` (DOF layout); `kp`, `kd` are the servo gains per DOF.

        Wheel torque limit: `wheel_cap[k]` is the ArgoDrive limit for this tick, 16 N m when the drive torque would
        accelerate the wheel in its direction of travel (or at standstill) and 28.4 N m when it brakes it. The world
        writes it into MuJoCo's joint-level actuator force range, which clamps the TOTAL of the position and velocity
        terms, so the velocity term cannot come on top of a capped position lead up to the 28.4 N m joint effort
        limit. The position lead is still bounded by the cap (anti-windup of the integrated reference)."""
        lim, dt = P.BASE_LIMITS, self.dt
        if self.wheel_ref is None:
            self.wheel_ref = [float(q[d["wheel"]["q"]]) for d in self.drives]
            self.steer_ref = [float(q[d["steer"]["q"]]) for d in self.drives]
        dv = self.request[:2] - self.cmd[:2]
        n = float(np.linalg.norm(dv))
        if n > lim["a"]*dt:
            dv *= lim["a"]*dt/n
        self.cmd[:2] += dv
        self.cmd[2] += float(np.clip(self.request[2] - self.cmd[2], -lim["alpha"]*dt, lim["alpha"]*dt))
        vx, vy, wz = self.cmd
        self.wheel_cap = []
        for k, d in enumerate(self.drives):
            x, y = d["pos"]
            v = np.array([vx - wz*y, vy + wz*x])
            speed = float(np.linalg.norm(v))
            sq, wq = d["steer"]["q"], d["wheel"]["q"]
            sd, wd = d["steer"]["dof"], d["wheel"]["dof"]
            if speed > 1e-5:
                ang = math.atan2(v[1], v[0])
                cur_ang = self.steer_ref[k]
                delta = wrap(ang - cur_ang)
                if abs(delta) > math.pi/2:                 # reverse the wheel instead of turning more than 90 deg
                    delta = wrap(delta - math.pi)
                    speed = -speed
                goal = cur_ang + delta
            else:
                goal, speed = self.steer_ref[k], 0.
            step = P.STEER_RATE*dt
            move = float(np.clip(goal - self.steer_ref[k], -step, step))
            self.steer_ref[k] += move
            cur[sq] = self.steer_ref[k]
            vel[sd] = move/dt
            err = wrap(float(q[sq]) - goal)
            omega = speed/P.WHEEL_R*max(0., math.cos(err))
            w_now = float(qd[wd])
            ref = self.wheel_ref[k] + omega*dt
            tau = kp[wd]*(ref - float(q[wq])) + kd[wd]*(omega - w_now)
            braking = tau*w_now < 0. and abs(w_now) > 1e-3
            cap = P.WHEEL_TORQUE[1] if braking else P.WHEEL_TORQUE[0]
            lead = float(np.clip(ref - float(q[wq]), -cap/kp[wd], cap/kp[wd]))
            ref = float(q[wq]) + lead
            self.wheel_ref[k] = ref
            cur[wq] = ref
            vel[wd] = omega
            self.wheel_cap.append(cap)
        self._odometry(q, qd)

    def _odometry(self, q, qd):
        """Least-squares base twist from the two drives' measured steering angles and wheel speeds."""
        A, b = [], []
        for d in self.drives:
            x, y = d["pos"]
            th = float(q[d["steer"]["q"]])
            u = float(qd[d["wheel"]["dof"]])*P.WHEEL_R
            A += [[1., 0., -y], [0., 1., x]]
            b += [u*math.cos(th), u*math.sin(th)]
        tw = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
        self.odom_twist = tw
        c, s = math.cos(self.odom[2]), math.sin(self.odom[2])
        self.odom += self.dt*np.array([c*tw[0] - s*tw[1], s*tw[0] + c*tw[1], tw[2]])

    def state(self):
        return dict(request=self.request.copy(), cmd=self.cmd.copy(), steer_ref=list(self.steer_ref),
                    wheel_ref=None if self.wheel_ref is None else list(self.wheel_ref), odom=self.odom.copy(),
                    odom_twist=self.odom_twist.copy())

    def restore(self, s):
        self.request, self.cmd = s["request"].copy(), s["cmd"].copy()
        self.steer_ref = list(s["steer_ref"])
        self.wheel_ref = None if s["wheel_ref"] is None else list(s["wheel_ref"])
        self.odom, self.odom_twist = s["odom"].copy(), s["odom_twist"].copy()
