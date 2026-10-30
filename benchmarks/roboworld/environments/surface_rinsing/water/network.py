"""Water supply and outlets: pressure, line loss, valve openings read from joint angles, mixing, jets.

Each outlet solves, every tick, its steady branch balance from the static supply pressure:

    p_s = dp_line(Q) + dp_valve(f, Q) + dp_outlet(Q),   dp_line = f_D (L/D) rho v^2/2  (64/Re or 0.316 Re^-1/4),
    dp_valve = rho/2 (Q / (Cd A_v f))^2,                 dp_outlet = rho/2 (Q / (Cv A_o))^2 (orifice face)

solved by bisection; the result reaches the outlet through a first-order lag (2 ms, longer than the 1.4 ms line response
time). Spray valves (B-0107, EB-0107) open by their squeeze lever: f = smoothstep of the lever angle between crack and
full travel; the lever's return spring is an explicit preloaded torque tau = -(tau0 + k phi) on lever and valve body
(sized for 13 N at crack, 22 N full at the pad). Faucets: wrist-blade cartridges with a 5 deg dead band, linear to the
quarter turn; a laminar stream regulator caps the flow at 2.2 gpm. Hot and cold branches mix by flow: T = sum(Q_i T_i)/Q.
Every outlet emits one packet per orifice per tick (mass rho Q dt / n, velocity Cv sqrt(2 p_o / rho) along the orifice
axis plus the outlet's own velocity), and the momentum it emits is applied to the outlet's body as its reaction.
Nothing opens a valve except its joint angle; a closed valve emits exactly zero.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from .. import params as P
from .engine import NS

RHO = P.WATER["rho"]


def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x*x*(3.0 - 2.0*x)


def line_loss(Q, L, D, T=20.0):
    """Darcy-Weisbach pressure loss (Pa) of a smooth hose or pipe."""
    if Q <= 0 or L <= 0:
        return 0.0
    A = math.pi*D*D/4
    v = Q/A
    re = v*D/(P.viscosity(T)/RHO)
    f = 64.0/re if re < 2300 else 0.316*re**-0.25
    return f*(L/D)*RHO*v*v/2


def solve_flow(p_s, A_eff, L, D, T=20.0):
    """Flow Q (m3/s) through a line of length L, diameter D and an effective orifice area A_eff (Cv/Cd included)."""
    if A_eff <= 0 or p_s <= 0:
        return 0.0
    lo, hi = 0.0, A_eff*math.sqrt(2*p_s/RHO)
    for _ in range(50):
        Q = 0.5*(lo + hi)
        dp = line_loss(Q, L, D, T) + 0.5*RHO*(Q/A_eff)**2
        lo, hi = (Q, hi) if dp < p_s else (lo, Q)
    return 0.5*(lo + hi)


class Mixer:
    """Hot and cold wrist-blade handles feeding one node: per-branch opening fractions from the joint angles."""

    def __init__(self, world, fx, supply):
        self.w, self.supply = world, supply
        self.hot, self.cold = fx.get("hot"), fx.get("cold")
        F = P.FAUCET
        self.dead, self.turn = math.radians(F["dead_deg"]), math.radians(F["turn_deg"])

    def frac(self, blade):
        if blade is None:
            return 0.0
        q, _ = self.w.joint_value(blade["joint"])
        return min(1.0, max(0.0, (q - self.dead)/(self.turn - self.dead)))

    def state(self):
        fh, fc = self.frac(self.hot), self.frac(self.cold)
        s = self.supply
        tot = fh + fc
        T = (fh*s["T_hot"] + fc*s["T_cold"])/tot if tot > 0 else s["T_cold"]
        return fh, fc, T


class Outlet:
    """One water outlet: geometry on a body (or static), orifice set, flow law, lag, and emission."""

    def __init__(self, name, world, body, face, direction, kind, spec, mixer, *, gate=None, line=(1.0, 0.0127)):
        self.name, self.w, self.body, self.kind, self.spec, self.mixer = name, world, body, kind, spec, mixer
        self.face = np.asarray(face, float)
        self.dir = np.asarray(direction, float)/np.linalg.norm(direction)
        self.gate = gate                 # callable -> opening fraction of the outlet's own valve (lever, blade) or None
        self.line = line
        self.Q = 0.0                     # lagged flow, m3/s
        self.T = P.SUPPLY["T_cold"]
        self.v_jet = 0.0
        self.emitted_kg = 0.0
        self.reaction = np.zeros(3)
        if kind == "spray":
            n = int(spec["n_orifice"])
            v_r = spec["Cv"]*math.sqrt(2*spec["p_rated"]/RHO)
            self.A_eff = spec["Q_rated"]/v_r                       # total effective orifice area (Cv included)
            self.d_jet = math.sqrt(4*self.A_eff/(n*math.pi))
            cone = math.radians(spec["cone_deg"])
            # orifices: one centre jet and a ring, each axis tilted outward by up to the cone half-angle
            ang = 2*math.pi*np.arange(n - 1)/(n - 1)
            ring = spec["ring_r"]
            offs = [np.zeros(3)] + [np.array([ring*math.cos(a), ring*math.sin(a), 0.0]) for a in ang]
            dirs = [np.array([0., 0., 1.])] + [np.array([math.sin(cone)*math.cos(a), math.sin(cone)*math.sin(a), math.cos(cone)])
                                               for a in ang]
            self.offsets, self.dirs = np.array(offs), np.array(dirs)
        else:
            d = P.FAUCET["d_out"]
            self.A_eff = None
            self.d_jet = d
            self.offsets, self.dirs = np.zeros((1, 3)), np.array([[0., 0., 1.]])
        # local frame whose z is the outlet direction (orifice offsets and axes are given in it)
        z = self.dir
        x = np.cross((0., 1., 0.) if abs(z[1]) < 0.9 else (1., 0., 0.), z); x /= np.linalg.norm(x)
        self.Rf = np.stack([x, np.cross(z, x), z], 1)

    def target_flow(self, p_s):
        fh, fc, T = self.mixer.state() if self.mixer is not None else (0.0, 1.0, P.SUPPLY["T_cold"])
        g = self.gate() if self.gate is not None else 1.0
        if g <= 0.0 or fh + fc <= 0.0:
            return 0.0, T
        L, D = self.line
        if self.kind == "spray":
            # the handles' cartridges in series with the squeeze valve and the face orifices; a supply with no handles
            # (the hose reel's wall valve, left open) adds no cartridge. (First build: it counted as one faucet
            # cartridge fully open, Cv ~0.56, and the reel gave 2.64 gpm where its hose and valve pass 3.4 gpm.)
            A_v = self.A_eff*g
            if getattr(self.mixer, "cartridges", True):
                A_mix = P.FAUCET["Cd"]*P.FAUCET["A_max"]*(fh + fc)
                A = 1.0/math.sqrt(1.0/A_v**2 + 1.0/A_mix**2)
            else:
                A = A_v
            return solve_flow(p_s, A, L, D, T), T
        # faucet: cartridges (and the spout's own valve) then the laminar regulator
        A = P.FAUCET["Cd"]*P.FAUCET["A_max"]*(fh + fc)*g
        Q = solve_flow(p_s, A, L, D, T)
        return min(Q, P.FAUCET["Q_reg"]), T

    def pose(self):
        """World position of the outlet face, its rotation, and the face's velocity."""
        w = self.w
        if self.body is None or self.body < 0:
            return self.face.copy(), np.eye(3), np.zeros(3)
        q = w.state.body_q.numpy()[self.body]
        R = Rotation.from_quat(q[3:]).as_matrix()
        p = q[:3] + R @ self.face
        qd = w.state.body_qd.numpy()[self.body]
        com = q[:3] + R @ w.model.body_com.numpy()[self.body]
        v = qd[:3] + np.cross(qd[3:], p - com)
        return p, R, v

    def step(self, dt, p_s):
        Qt, T = self.target_flow(p_s)
        a = min(1.0, dt/P.SUPPLY["lag_s"])
        self.Q += a*(Qt - self.Q)
        if Qt == 0.0 and self.Q < 1e-9:
            self.Q = 0.0
        self.T = T
        self.reaction = np.zeros(3)
        if self.Q <= 0.0:
            self.v_jet = 0.0
            return None
        p, R, v = self.pose()
        Rw = R @ self.Rf
        n = len(self.offsets)
        if self.kind == "spray":
            v_jet = self.Q/self.A_eff                         # orifice velocity at this flow (Cv sqrt(2 p_o / rho))
            d = self.d_jet
        else:
            A0 = math.pi*self.d_jet**2/4
            v_jet = self.Q/A0
            d = self.d_jet
        self.v_jet = v_jet
        m = RHO*self.Q*dt/n
        x = p + (self.offsets @ Rw.T)
        dirs = self.dirs @ Rw.T
        vel = v + v_jet*dirs
        sp = np.zeros((NS, n))
        sp[3, :] = m/RHO*T                                    # heat: volume x temperature
        # reaction on the outlet's body: minus the momentum emitted per second (relative to the body)
        self.reaction = -RHO*self.Q*v_jet*dirs.mean(axis=0)
        self.emitted_kg += m*n
        return x, vel, np.full(n, m), np.full(n, d), np.zeros(n, np.int32), sp, p


class SprayValve:
    """Squeeze-lever valve on a body: explicit preloaded return spring on the lever, opening fraction from its angle."""

    def __init__(self, world, fx):
        self.w, self.fx = world, fx
        s = fx["spec"]
        arm = s["lever_arm"]
        self.phi_crack = s["crack_travel"]/arm
        self.phi_full = s["travel"]/arm
        t_crack, t_full = s["crack_n"]*arm, s["full_n"]*arm
        self.k = (t_full - t_crack)/(self.phi_full - self.phi_crack)
        self.tau0 = t_crack - self.k*self.phi_crack
        self.balancer = fx.get("balancer")

    def angle(self):
        return self.w.joint_value(self.fx["lever_joint"])[0]

    def opening(self):
        return smoothstep((self.angle() - self.phi_crack)/(self.phi_full - self.phi_crack))

    def spring(self, wrench):
        """Return-spring torque about the lever axis (world): -(tau0 + k phi) on the lever, the opposite on the body."""
        w = self.w
        phi = self.angle()
        tau = -(self.tau0 + self.k*max(phi, 0.0))
        q = w.state.body_q.numpy()[self.fx["valve"]]
        axis = Rotation.from_quat(q[3:]).apply([0., 1., 0.])
        wrench[self.fx["lever"], 3:] += tau*axis
        wrench[self.fx["valve"], 3:] -= tau*axis
        self.tau = tau
        return tau


class Balancer:
    """Spring balancer of a pre-rinse unit: an explicit spring force F = F0 + k s on its slide (s down from the arm),
    sized to hold the valve and hose at their built (parked) height, k 30 N/m (estimate: pulling the valve 0.3 m down
    to a plate takes 9 N more than its weight)."""

    def __init__(self, world, fx, k=30.0, s_park=0.0):
        self.w, self.fx = world, fx
        sc = world.scene
        m = sum(float(world.model.body_mass.numpy()[b]) for b in [fx["valve"], fx["lever"], fx["balancer"]] + list(fx["chain"]))
        self.k = k
        self.F0 = m*P.GRAVITY - k*s_park
        self.mass = m

    def force(self, wrench):
        s, _ = self.w.joint_value(self.fx["balancer_joint"])
        F = self.F0 + self.k*s
        wrench[self.fx["balancer"], 2] += F
        return F
