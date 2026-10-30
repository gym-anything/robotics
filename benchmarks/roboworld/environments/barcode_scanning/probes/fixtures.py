"""ASSISTED fixtures (labelled; never reachable through the policy interface).

PhantomHold: a six-axis spring-damper on the scanner body that carries it along a smooth path to a target pose
and holds it there (gravity compensated) -- a "fixed mount on a motion stage". The reaction
goes to ground (the fixture is the apparatus), exactly as a test stand would.
PhantomTrigger: a fingertip force on the trigger blade, normal to its face, with its reaction on the scanner.
Both write body wrenches that the world adds before the physics step; every applied value is logged.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class PhantomHold:
    def __init__(self, world, body="scanner", kp=300., kr=1.5, zeta=1.0, extra=()):
        self.w = world
        self.body = world.scene.bodies[body]
        # bodies carried through a joint whose weight the hold also bears (the trigger; a book's front cover)
        self.extra = ([world.scene.bodies["scanner_trigger"]] if body == "scanner" else []) + [world.scene.bodies[e] for e in extra]
        m = float(world.model.body_mass.numpy()[self.body]) + sum(float(world.model.body_mass.numpy()[b]) for b in self.extra)
        Ib = np.asarray(world.model.body_inertia.numpy()[self.body], float)      # about the COM, body frame
        lam, self.V = np.linalg.eigh(Ib)
        I = float(np.max(lam))
        dt = world.config.rigid_dt
        # the hold's force is computed once per step from the state (an explicit spring): stable only while
        # omega*dt < 2 on every axis. Capped at omega*dt = 0.5 (k <= 0.25 m/dt^2, 0.25 I_i/dt^2 per principal axis):
        # at 2 N m/rad a 10 g lip balm (axial inertia 3e-7 kg m^2, omega*dt = 2.6) wobbled 60 deg in the hold and was
        # thrown at 631 m/s when the fingers closed on it
        kp = min(kp, .25*m/dt**2)
        self.kr_axes = np.minimum(kr, .25*lam/dt**2)
        self.m, self.kp, self.kd = m, kp, 2*zeta*math.sqrt(kp*m)
        self.kr, self.kdr = kr, 2*zeta*math.sqrt(kr*I*4)
        # rotational damping per principal axis, capped at 0.8 I_i/dt: the scalar damping sized on the largest axis is
        # beyond the explicit stability limit on a can's own (small-inertia) axis at 1 kHz -- the can spun back and
        # forth in the mount (63 px of blur at 0.1 m/s). The scanner's gains are below the cap on
        # every axis (unchanged from the scalar law)
        self.cr = np.minimum(self.kdr, .8*lam/world.config.rigid_dt)
        self.path = None
        self.prev = None
        self.log = []
        self.enabled = True

    def move(self, p_goal, q_goal, duration, profile="smooth", p_start=None):
        """Carry the body to p_goal/q_goal over `duration` s: smooth-step, or constant speed ("linear") from p_start."""
        p0, R0 = self.w.body_pose(self.body)
        if p_start is not None:
            p0 = np.asarray(p_start, float)
        self.path = dict(t0=self.w.sim_time, T=float(duration), p0=p0.copy(), p1=np.asarray(p_goal, float), profile=profile,
                         slerp=Slerp([0, 1], Rotation.from_matrix([R0, Rotation.from_quat(q_goal).as_matrix()])))

    def target(self, t):
        a = 1. if self.path is None else min(1., max(0., (t - self.path["t0"])/max(self.path["T"], 1e-6)))
        s = a if self.path and self.path.get("profile") == "linear" else a*a*(3 - 2*a)
        return (1 - s)*self.path["p0"] + s*self.path["p1"], self.path["slerp"](s)

    def wrench(self, wrench):
        if not self.enabled or self.path is None:
            return wrench
        w = self.w
        dt = w.config.rigid_dt
        p, R = w.body_pose(self.body)
        com = p + R @ w.model.body_com.numpy()[self.body]
        pt, Rt = self.target(w.sim_time)
        if self.prev is None:
            v, om = np.zeros(3), np.zeros(3)
        else:
            v = (p - self.prev[0])/dt
            om = (Rotation.from_matrix(R)*Rotation.from_matrix(self.prev[1]).inv()).as_rotvec()/dt
        self.prev = (p.copy(), R.copy())
        f = self.kp*(pt - p) - self.kd*v                              # the servo acts at the tracked point p ...
        g = np.array([0, 0, self.m*9.81])                             # ... gravity compensation at the centre of mass
        err = (Rt*Rotation.from_matrix(R).inv()).as_rotvec()
        D = R @ self.V @ np.diag(self.cr) @ self.V.T @ R.T
        # (with the gravity term at p too, 0.13 N m on the scanner, whose centre of mass sits ~6 cm from its origin,
        # held it 5.7 deg off its commanded look direction)
        Kr = R @ self.V @ np.diag(self.kr_axes) @ self.V.T @ R.T
        tau = Kr @ err - D @ om + np.cross(p - com, f)
        wrench[self.body, :3] += f + g
        wrench[self.body, 3:] += tau
        self.log.append(dict(t=w.sim_time, force_n=(f + g).tolist(), torque_nm=tau.tolist(), pos_err_mm=float(np.linalg.norm(pt - p)*1e3)))
        return wrench


class PhantomTrigger:
    """Force on the trigger blade's face centre, along its travel (into the handle); reaction on the scanner body. The
    push builds up at RAMP_N_PER_S, as a finger presses (3 N in 0.1 s), and stops at once when released: a 3 N step
    slammed the blade into its stop and over-ran it to 3.98 mm (stop 3.60 mm) on every pull."""
    RAMP_N_PER_S = 30.

    def __init__(self, world):
        self.w = world
        self.force = 0.
        self.applied = 0.
        self.log = []

    def wrench(self, wrench):
        if not self.force:
            self.applied = 0.
            return wrench
        self.applied = min(self.force, self.applied + self.RAMP_N_PER_S*self.w.config.rigid_dt)
        w = self.w
        sc = w.scene
        tb, sb = sc.bodies["scanner_trigger"], sc.bodies["scanner"]
        p, R = w.body_pose(tb)
        from barcode_scanning import params as P
        point = p + R @ np.array([.010, 0., -P.TRIGGER_ARM_M])     # blade front face centre (trigger frame: blade x .007 +/- .003)
        n = R @ np.array([-1., 0., 0.])                             # pressing direction: into the handle
        f = self.applied*n
        for body, sign in ((tb, 1.), (sb, -1.)):
            pb, Rb = w.body_pose(body)
            com = pb + Rb @ w.model.body_com.numpy()[body]
            wrench[body, :3] += sign*f
            wrench[body, 3:] += np.cross(point - com, sign*f)
        self.log.append(dict(t=w.sim_time, force_n=self.applied, travel_mm=w.trigger_travel_m()*1e3, switch=w.trigger.state))
        return wrench


def combine(world, *fixtures):
    def call(w):
        wr = np.zeros((w.model.body_count, 6), np.float32)
        for fx in fixtures:
            wr = fx.wrench(wr)
        return wr
    world.fixture = call


def phantom_summary(ph):
    """How hard one labelled phantom pushed over its log: steps, peak and mean force magnitude (N), peak torque (N m),
    peak tracking error (mm), largest trigger travel (mm), whichever the phantom logs."""
    out = dict(steps=len(ph.log))
    for key, name in (("force_n", "force_n"), ("torque_nm", "torque_nm")):
        v = [float(np.linalg.norm(r[key])) for r in ph.log if key in r]
        if v:
            out[f"peak_{name}"], out[f"mean_{name}"] = round(max(v), 3), round(float(np.mean(v)), 3)
    for key, name in (("pos_err_mm", "peak_tracking_error_mm"), ("travel_mm", "max_travel_mm")):
        v = [float(r[key]) for r in ph.log if key in r]
        if v:
            out[name] = round(max(v), 3)
    return out

