"""Super-elastic forklift tyres against drivable surfaces.

Contact: each tyre is sampled as 3 ribs across its width x an enveloping arc of N_ARC points on the lower circumference;
every sample is tested against the nearby drivable surfaces (oriented rectangles fixed to the world or to a moving body:
floor zones, ramps, leveler deck and lip, trailer floor). Per rib the deepest penetration p (< P_WINDOW) carries
    F_rib = (K/3) p^1.5 + k_s (p - s),  ds/dt = (k_s / C)(p - s)     (Traigo law with a Maxwell damper; K, k_s, C fitted
                                                                    to an 18x7-8 SE tyre measured by Oh et al. 2013)
Tangential forces are a relaxation (brush) model whose steady state is the SUPREM law (read):
    du/dt = -v_slip - |v_x| u / sigma,   sigma = max(SIGMA_MIN, |v_x| T),  T = k_d (v km/h)^-k_v
    F_y = F_z mu exp(-F_z/k_F1) tanh(atan(u_y/sigma)[deg] / (k_a + k_F2 F_z))
    F_x = F_z mu exp(-F_z/k_F1) rho tanh((u_x/sigma) / kappa*)      (rho = MU_LONG_RATIO, kappa* = KAPPA_STAR: one SE
                                                                    tyre's braking-slip curve, Pross 2019, digitised)
so the lateral force builds with SUPREM's first-order delay T = sigma/|v_x|, and at standstill u is a bristle spring that
holds a parked truck with a fixed elastic set instead of creeping. SUPREM's T is a lateral measurement (F_y,dyn + T dF_y,dyn/dt
= F_y,stat, Logistics Research 10:1 eq. 4); the longitudinal deflection u_x relaxes over its own length sigma_x =
max(SIGMA_MIN, a), the contact patch's half-length a = sqrt(2 r p): the brush model's relaxation length with a rigid
carcass (bristles enter the patch undeflected and the mean deflection a kappa is built over the patch, so the lumped
first-order state reaching u_x = kappa sigma_x has sigma_x = a; its standstill stiffness C_Fk / a is the patch's 2 a k_b).
It is a lower bound: the SE tyre's circumferential compliance adds C_Fk / C_c, which is not published (Pross & Bruns 2020
measured C_c on their drum but print no value). With u_x relaxing over the lateral sigma instead (0.3-0.44 m at
10-17 km/h), against the wheel's spin inertia that lag lets the braked front wheels oscillate at 5.8 Hz, their slip
swinging from -1.0 to +0.8 and the unladen truck's steer axle between 2 and 46 kN in a 300 N stop on dry concrete (E8:
the same at half the tick, gone with sigma_x = a and with 0.15 m). A bristle damper (LuGre sigma_1 form, damping ratio
BRISTLE_ZETA on the faster of the corner and wheel-spin modes) acts at low speed and fades out above BRISTLE_FADE_V. The
resultant is capped on Kamm's ellipse (semi-axes rho mu_eff F_z along, mu_eff F_z across); once the patch slides the cap
drops by the sliding ratio MU_SLIDE_RATIO (locked over peak on the same braking-slip curve), until the force the bristles
ask for falls back under it (Coulomb stick-slip: break away at the peak, slide at the locked value, re-adhere below it).
Turn slip: a torsional bristle about the contact normal, capped at the pivoting torque of the patch (mu F_z r_t); rolling,
it relaxes so that its steady state is the brush model's spin moment -C_Fk (b^2/3) w_rel / v_x. Pneumatic trail: the
lateral force of a rolling tyre acts (a/3)(...) behind the patch centre (brush model). The tread slips at the effective rolling radius
r_e = r - p/3 (Wong). Rolling resistance: torque -C_rr r_e F_z on the wheel joint (a resisting force C_rr F_z at the
ground). Patch shift on the outer tyre of a turn: moment F_y/k_M about the rolling direction (SUPREM eq. 6).
The normal and lateral forces act on the wheel at the contact point; the longitudinal force acts at the effective rolling
radius, where its slip is measured, so the wheel torque is F_x r_e and its power balances (energy-consistent). Each acts
equal and opposite on the surface body (a trailer is pushed by the truck's braking; a leveler deck is loaded by its
wheels). Wheel shapes never contact drivable surfaces natively (world.GROUPS); they still hit curbs, pallets and racks
through MuJoCo.
"""

import math
from dataclasses import dataclass

import numpy as np

from . import params as P
from .mathx import quat_to_R, cross, cross3

P_WINDOW = 0.10          # m: a deeper "penetration" is a surface far above the tyre, not contact
N_ARC = 9
ARC_HALF = math.radians(42.0)
BRISTLE_ZETA = P.TYRE_BRISTLE["zeta"]       # bristle damping ratio (authored: critically damped stick)
BRISTLE_FADE_V = P.TYRE_BRISTLE["fade_v"]     # m/s: bristle damper fades out above this speed (authored)
TURN_SLIP_ANGLE = P.TYRE_BRISTLE["turn_slip_angle"]   # rad of torsional deflection at the pivoting torque (estimate)
LONG_RELAX = "patch"     # u_x relaxes over the patch half-length (brush, rigid carcass); "lateral": the lateral sigma,
                         # kept only to show the wheel judder it causes (E8)


@dataclass
class Surface:
    name: str
    body: int            # -1 for the world
    origin: np.ndarray   # local (body frame) or world
    normal: np.ndarray
    u: np.ndarray
    half: tuple          # (half extent along u, along v)
    material: str
    priority: int = 0


@dataclass
class Wheel:
    name: str
    body: int
    dof: int
    radius: float
    width: float
    K: float
    suprem: dict
    driven: bool = False
    J: float = 5.0       # spin inertia incl. reflected drivetrain (kg m^2), for the bristle damper


class TyreSet:
    def __init__(self, wheels, surfaces):
        self.wheels = list(wheels)
        self.surfaces = list(surfaces)
        n = len(self.wheels)
        self.u = np.zeros((n, 2))
        self.phi = np.zeros(n)
        self.slid = np.zeros(n, bool)          # the patch slid last tick: capped at the sliding coefficient
        self.telemetry = {}
        self.energy = dict(slip=0.0, rolling=0.0)
        self._pack()
        phis = np.linspace(-ARC_HALF, ARC_HALF, N_ARC)
        self._cos, self._sin = np.cos(phis), np.sin(phis)

    def _pack(self):
        S = self.surfaces
        self.s_body = np.array([s.body for s in S], int)
        self.s_o = np.array([s.origin for s in S], float)
        self.s_n = np.array([np.asarray(s.normal, float) / np.linalg.norm(s.normal) for s in S], float)
        u = np.array([s.u for s in S], float)
        u = u - (u * self.s_n).sum(1, keepdims=True) * self.s_n
        self.s_u = u / np.linalg.norm(u, axis=1, keepdims=True)
        self.s_v = np.cross(self.s_n, self.s_u)
        self.s_half = np.array([s.half for s in S], float)
        self.s_rad = np.hypot(self.s_half[:, 0], self.s_half[:, 1]) + 1.0
        self.s_prio = np.array([s.priority for s in S], int)
        self.s_mu = np.array([P.MU[s.material] for s in S], float)
        self.s_crr = np.array([P.C_RR[s.material] for s in S], float)

    def set_material(self, name, material):
        """Change a surface's condition (wet patch, dust) between episodes."""
        for s in self.surfaces:
            if s.name == name:
                s.material = material
        self._pack()

    def surfaces_world(self, body_q):
        o, n, u, v = self.s_o.copy(), self.s_n.copy(), self.s_u.copy(), self.s_v.copy()
        moving = np.where(self.s_body >= 0)[0]
        if len(moving):
            from scipy.spatial.transform import Rotation
            q = body_q[self.s_body[moving]]
            R = Rotation.from_quat(q[:, 3:7]).as_matrix()
            o[moving] = q[:, :3] + np.einsum("kij,kj->ki", R, self.s_o[moving])
            n[moving] = np.einsum("kij,kj->ki", R, self.s_n[moving])
            u[moving] = np.einsum("kij,kj->ki", R, self.s_u[moving])
            v[moving] = np.einsum("kij,kj->ki", R, self.s_v[moving])
        return o, n, u, v

    def step(self, body_q, body_qd, com_w, chassis_body, dt, joint_qd=None):
        """Returns (wrench (body_count, 6): world force and torque about each body's COM; {wheel dof: joint torque})."""
        if not hasattr(self, "_wb"):
            self._wb = np.array([w.body for w in self.wheels])
            self._rad = np.array([w.radius for w in self.wheels])
            self._ribs = np.array([[-w.width / 3, 0.0, w.width / 3] for w in self.wheels])
            self._K3 = np.array([w.K / 3.0 for w in self.wheels])
        if not hasattr(self, "s_maxwell"):
            self.s_maxwell = np.zeros((len(self.wheels), 3))
        nb = len(body_q)
        wrench = np.zeros((nb, 6))
        tau_dof = {}
        o, n, su, sv = self.surfaces_world(body_q)
        W, S = len(self.wheels), len(o)
        wb = self._wb
        R = quat_to_R(body_q[wb, 3:7])
        c = body_q[wb, :3].astype(float)
        a = R[:, :, 1]
        down = np.array([0.0, 0.0, -1.0])[None] + a * a[:, 2:3]
        nd = np.linalg.norm(down, axis=1)
        down = down / np.maximum(nd, 1e-9)[:, None]
        fwd = cross(a, -down)
        arc = self._cos[None, :, None] * down[:, None, :] + self._sin[None, :, None] * fwd[:, None, :]       # (W,N,3)
        pts = (c[:, None, None, :] + self._ribs[:, :, None, None] * a[:, None, None, :]
               + self._rad[:, None, None, None] * arc[:, None, :, :])                                       # (W,3,N,3)
        near = np.linalg.norm(o[None] - c[:, None], axis=2) < (self.s_rad[None] + self._rad[:, None])       # (W,S)
        # per surface, the exact deepest point of each rib circle: the rib centre minus r along the surface normal
        # projected into the wheel plane. The arc alone is centred on the world vertical, so on a grade its nearest
        # sample (10.5 deg apart) sits off the true contact point and the normal force would get a lever arm about the
        # axle, a drive torque F_z r sin(theta) that cancels grades under ~9 % (E12: the laden truck crept up 4 % at idle)
        npl = n[None] - np.einsum("wk,sk->ws", a, n)[..., None] * a[:, None]                                # (W,S,3)
        npn = np.linalg.norm(npl, axis=2)
        deep = c[:, None, None, :] + self._ribs[:, :, None, None] * a[:, None, None, :] \
            - self._rad[:, None, None, None] * (npl / np.maximum(npn, 1e-9)[..., None])[:, None]             # (W,3,S,3)
        rel = np.concatenate([np.broadcast_to(pts[:, :, :, None, :], (W, 3, N_ARC, S, 3)), deep[:, :, None]], axis=2)
        rel = rel - o[None, None, None]                                                                     # (W,3,N+1,S,3)
        d = -np.einsum("wrnsk,sk->wrns", rel, n)
        du = np.einsum("wrnsk,sk->wrns", rel, su)
        dv = np.einsum("wrnsk,sk->wrns", rel, sv)
        d[:, :, N_ARC] = np.where((npn > 0.5)[:, None], d[:, :, N_ARC], -np.inf)                           # surface near edge-on
        inside = ((np.abs(du) <= self.s_half[:, 0]) & (np.abs(dv) <= self.s_half[:, 1]) & (d > 0) & (d < P_WINDOW)
                  & near[:, None, None, :] & (nd > 1e-6)[:, None, None, None])
        # per rib, the deepest penetration wins (a tyre bears on the highest surface under it); priority only breaks
        # ties between coplanar surfaces (a wet patch painted on a floor). If priority dominated (10 m per level), a
        # wheel rolling off the leveler deck onto the dock floor would keep the deck's rib through barely-touching trailing
        # arc points, sink ~6 cm and be kicked up at 58 kN when the floor took over (E15 load-trailer)
        score = np.where(inside, d + 1.0e-4 * self.s_prio, -np.inf).reshape(W, 3, -1)
        idx = np.argmax(score, axis=2)
        best = np.take_along_axis(score, idx[..., None], 2)[..., 0]
        has = np.isfinite(best)
        i_n, i_s = np.divmod(idx, S)
        wi, ri = np.meshgrid(np.arange(W), np.arange(3), indexing="ij")
        pen = np.where(has, d[wi, ri, i_n, i_s], 0.0)
        nrm = n[i_s]                                                                                        # (W,3,3)
        pc = rel[wi, ri, i_n, i_s] + o[i_s] + pen[..., None] * nrm
        vc, om = body_qd[wb, :3].astype(float), body_qd[wb, 3:].astype(float)
        comw = com_w[wb]
        vp = vc[:, None] + cross(om[:, None], pc - comw[:, None])
        sbody = self.s_body[i_s]
        vs = np.zeros_like(vp)
        mv = sbody >= 0
        if mv.any():
            bq = body_qd[sbody[mv]].astype(float)
            vs[mv] = bq[:, :3] + cross(bq[:, 3:], pc[mv] - com_w[sbody[mv]])
        pen_rate = -np.einsum("wrk,wrk->wr", vp - vs, nrm)
        # elastic law plus the rubber's damping as a Maxwell element (a dashpot C in series with a spring k_s), fitted to
        # the measured tyre's damping at 1 and 5 Hz; both grow with the wheel's load (a rib carrying F_el carries a third
        # of a wheel at 3 F_el). State: the dashpot's travel s per rib (relaxes with time constant C / k_s)
        F_el = self._K3[:, None] * np.maximum(pen, 0.0) ** P.TYRE_EXP
        scale = (3.0 * F_el / P.TYRE_DAMP_REF_LOAD) ** P.TYRE_DAMP_EXP / 3.0
        k_s, C = P.TYRE_MAXWELL_K * scale, P.TYRE_MAXWELL_C * scale
        s_ = np.where(has, self.s_maxwell, pen)
        F = F_el + k_s * (pen - s_)
        self.s_maxwell = np.where(has, s_ + dt * (P.TYRE_MAXWELL_K / P.TYRE_MAXWELL_C) * (pen - s_), 0.0)
        Fz_rib = np.where(has, np.maximum(F, 0.0), 0.0)
        Fz_w = Fz_rib.sum(1)
        tele = {}
        Rch = quat_to_R(body_q[chassis_body:chassis_body + 1, 3:7])[0]
        ey_truck = Rch[:, 1]
        pch = body_q[chassis_body, :3].astype(float)
        for k, w in enumerate(self.wheels):
            Fz = float(Fz_w[k])
            if Fz <= 1e-6:
                self._free(k, w, tele)
                continue
            wgt = Fz_rib[k] / Fz
            pck = wgt @ pc[k]
            nn = wgt @ nrm[k]; nn /= np.linalg.norm(nn)
            r_dom = int(np.argmax(Fz_rib[k]))
            sdom = int(i_s[k, r_dom])
            mu = float(self.s_mu[sdom]); crr = float(self.s_crr[sdom])
            ak = a[k]
            ex = cross3(ak, nn); ex /= np.linalg.norm(ex)
            ey = cross3(nn, ex)
            sb = int(self.s_body[sdom])
            if sb >= 0:
                bq = body_qd[sb].astype(float)
                vs_pt = bq[:3] + cross3(bq[3:], pck - com_w[sb])
                om_s = bq[3:]
            else:
                vs_pt = np.zeros(3); om_s = np.zeros(3)
            r_pc = pck - comw[k]
            # tread material velocity at the effective rolling radius r - pen/3 (a deflected tyre rolls on a radius
            # between its free and loaded radii; Wong, Theory of Ground Vehicles: r_e ~ r - delta/3)
            pen_c = float(Fz_rib[k] @ pen[k] / Fz)
            d_c = pck - c[k]
            r_e = (w.radius - pen_c / 3.0) / max(np.linalg.norm(d_c), 1e-9)
            slip = vc[k] + cross3(om[k], c[k] + d_c * r_e - comw[k]) - vs_pt
            vsx, vsy = float(slip @ ex), float(slip @ ey)
            vx = float((vc[k] + cross3(om[k], c[k] - comw[k]) - vs_pt) @ ex)
            sp = w.suprem
            mu_eff = mu * math.exp(-Fz / sp["kF1"])
            norm_deg = sp["ka"] + sp["kF2"] * Fz
            vkmh = max(abs(vx) * 3.6, 0.5)
            sigma = max(P.SIGMA_MIN, abs(vx) * P.SUPREM_KD * vkmh ** (-P.SUPREM_KV))
            ux, uy = float(self.u[k, 0]), float(self.u[k, 1])
            decay = abs(vx) * dt / sigma
            a_half = math.sqrt(max(2 * w.radius * pen_c, 1e-8))       # the patch's half-length (geometric chord)
            sigma_x = max(P.SIGMA_MIN, a_half) if LONG_RELAX == "patch" else sigma
            ux = (ux - vsx * dt) / (1.0 + abs(vx) * dt / sigma_x)
            uy = (uy - vsy * dt) / (1.0 + decay)
            # longitudinal grip: the lateral coefficient times MU_LONG_RATIO (the same SE tyre on the same drum peaks at
            # 0.94 braking and 0.766 sideways at 10 kN, Pross & Bruns 2019); the resultant is capped on Kamm's ellipse
            rho = P.MU_LONG_RATIO
            Fy = Fz * mu_eff * math.tanh(math.degrees(math.atan(uy / sigma)) / norm_deg)
            Fx = Fz * mu_eff * rho * math.tanh((ux / sigma_x) / P.KAPPA_STAR)
            fade = max(0.0, 1.0 - abs(vx) / BRISTLE_FADE_V)
            if fade > 0.0:
                m_corner = max(Fz / P.G, 1.0)
                m_spin = w.J / w.radius ** 2
                ky = Fz * mu_eff / norm_deg * 57.2958 / sigma
                kx = Fz * mu_eff * rho / P.KAPPA_STAR / sigma_x
                Fy -= fade * 2 * BRISTLE_ZETA * math.sqrt(ky * m_corner) * vsy
                Fx -= fade * 2 * BRISTLE_ZETA * math.sqrt(kx * min(m_corner, m_spin)) * vsx
            F_law = mu_eff * Fz                                          # the law's scale (static friction, lateral)
            Fmax = F_law * (P.MU_SLIDE_RATIO if self.slid[k] else 1.0)
            Ft = math.hypot(Fx / rho, Fy)                                # Kamm's ellipse: semi-axes rho F_law, F_law
            sliding = Ft > Fmax
            self.slid[k] = sliding
            if sliding:
                Fx *= Fmax / Ft; Fy *= Fmax / Ft
                sy = math.tan(math.radians(min(norm_deg * math.atanh(min(abs(Fy) / max(F_law, 1e-9), 0.995)), 85.0)))
                sx = P.KAPPA_STAR * math.atanh(min(abs(Fx) / max(rho * F_law, 1e-9), 0.995))
                uy = math.copysign(sy * sigma, uy) if abs(uy) > sy * sigma else uy
                ux = math.copysign(sx * sigma_x, ux) if abs(ux) > sx * sigma_x else ux
            self.u[k, 0], self.u[k, 1] = ux, uy
            r_t = P.TYRE_PIVOT_RADIUS * math.hypot(a_half, w.width / 2)
            Mmax = mu_eff * Fz * r_t
            w_rel = float((om[k] - om_s) @ nn)
            kphi = Mmax / TURN_SLIP_ANGLE
            # turn slip (spin). At standstill the torsional bristle holds the patch up to the pivoting torque. Rolling,
            # it relaxes over sigma_phi, set so that its steady state under a constant turn slip phi = w_rel / v_x is the
            # brush model's spin moment of a rolling tyre, M = -C_Fk (b^2 / 3) phi: on a curve the patch's outer edge
            # runs ahead of its inner edge, longitudinal slips -/+ phi y across the half-width b, with the longitudinal
            # slip stiffness C_Fk = F_z mu_x / KAPPA_STAR of the law above (derivation: E21). Relaxing over the lateral
            # sigma instead makes the rolling spin moment k_phi sigma phi, 12-26 x the brush value; on the rear-steered
            # knuckles that moment opposes the truck's yaw rate, i.e. pushes toward a larger steer angle, and the steering
            # unit's leakage lets the knuckles creep off centre (E21)
            b_half = w.width / 2
            sigma_phi = max(Fz * mu_eff * rho / P.KAPPA_STAR * b_half * b_half / 3.0 / kphi, 1e-4)
            ph = (float(self.phi[k]) - w_rel * dt) / (1.0 + abs(vx) * dt / sigma_phi)
            Mz = kphi * ph
            if fade > 0.0:
                I_corner = max(Fz / P.G, 1.0) * r_t ** 2 + 1.0
                Mz -= fade * 2 * BRISTLE_ZETA * math.sqrt(kphi * I_corner) * w_rel
            if abs(Mz) > Mmax:
                Mz = math.copysign(Mmax, Mz)
                ph = math.copysign(min(abs(ph), TURN_SLIP_ANGLE), ph)
            self.phi[k] = ph
            # pneumatic trail: a rolling tyre's lateral force acts behind the patch centre (brush model with the patch
            # half-length a: t = a/3 at small slip, falling to 0 at full sliding, t = (a/3)(1 - f)/(1 - psi + psi^2/3),
            # f = |F_y| / (mu F_z) = 1 - (1 - psi)^3; Pacejka, Tyre and Vehicle Dynamics, ch. 3). It builds with speed as
            # the bristle damper fades: at standstill the patch is shifted as a whole and the force acts at its centre
            f_y = min(abs(Fy) / max(F_law, 1e-9), 1.0)
            psi = 1.0 - (1.0 - f_y) ** (1.0 / 3.0)
            t_p = a_half / 3.0 * (1.0 - f_y) / max(1.0 - psi + psi * psi / 3.0, 1e-9)
            Mz_trail = -t_p * Fy * math.copysign(1.0, vx) * (1.0 - fade)
            # the normal and lateral forces act at the contact patch; the longitudinal force at the effective rolling
            # radius, where its slip is measured, so its power is F_x times the slip speed there (applied at the patch,
            # r - p from the axle, a driven tyre gained F_x omega 2p/3 of power from nowhere: 9 % more traction per unit
            # torque on a laden front tyre). Wheel torque = F_x r_e, as the drivetrain fit assumes
            Fn = Fz * nn + Fy * ey
            Ft = Fx * ex
            p_re = c[k] + d_c * r_e
            M = (Mz + Mz_trail) * nn
            yw = float((c[k] - pch) @ ey_truck)
            if yw * float(Fy * (ey @ ey_truck)) < 0:                   # outer tyre of the turn: patch shift inward
                M = M + (Fy / sp["kM"]) * ex
            wrench[w.body, :3] += Fn + Ft
            wrench[w.body, 3:] += cross3(r_pc, Fn) + cross3(p_re - comw[k], Ft) + M
            if sb >= 0:
                wrench[sb, :3] -= Fn + Ft
                wrench[sb, 3:] -= cross3(pck - com_w[sb], Fn) + cross3(p_re - com_w[sb], Ft) + M
            w_roll = float(joint_qd[w.dof]) if joint_qd is not None else float(om[k] @ ak)
            # rolling resistance C_rr F_z at the ground is the wheel torque C_rr F_z r_e (power C_rr F_z v)
            tau_rr = -crr * (w.radius - pen_c / 3.0) * Fz * math.tanh(w_roll * w.radius / 0.02)
            tau_dof[w.dof] = tau_dof.get(w.dof, 0.0) + tau_rr
            # signed: energy stored in the tyre's bristles is returned, not lost (E8: booking only the dissipative
            # ticks counts 16.6 kJ in an unladen 450 N stop whose signed total is 8.3 kJ)
            self.energy["slip"] += -(Fx * vsx + Fy * vsy) * dt
            self.energy["rolling"] += max(0.0, -tau_rr * w_roll) * dt
            tele[w.name] = dict(Fz=Fz, Fx=Fx, Fy=Fy, Mz=Mz, Mz_trail=Mz_trail, trail=t_p, mu=mu_eff,
                                surface=self.surfaces[sdom].name, contact=pck,
                                slip_angle_deg=math.degrees(math.atan2(-vsy, max(abs(vx), 1e-3))),
                                slip_ratio=-vsx / max(abs(vx), 0.1), sliding=sliding, sigma=sigma, sigma_x=sigma_x,
                                vx=vx, ribs=Fz_rib[k], pen=pen_c)
        self.telemetry = tele
        return wrench, tau_dof

    def _free(self, k, w, tele):
        self.u[k] = 0.0
        self.slid[k] = False
        self.phi[k] = 0.0
        tele[w.name] = dict(Fz=0.0, Fx=0.0, Fy=0.0, Mz=0.0, Mz_trail=0.0, trail=0.0, mu=0.0, surface=None, contact=None, slip_angle_deg=0.0,
                            slip_ratio=0.0, sliding=False, vx=0.0, ribs=None, pen=0.0)

    def _surface_velocity(self, s, p, body_qd, com_w):
        b = int(self.s_body[s])
        if b < 0:
            return np.zeros(3)
        v = body_qd[b]
        return v[:3] + cross3(v[3:], p - com_w[b])

    def _surface_omega(self, s, body_qd):
        b = int(self.s_body[s])
        return np.zeros(3) if b < 0 else body_qd[b, 3:]

    def checkpoint(self):
        return dict(u=self.u.copy(), phi=self.phi.copy(), energy=dict(self.energy),
                    s_maxwell=getattr(self, "s_maxwell", np.zeros((len(self.wheels), 3))).copy(), slid=self.slid.copy())

    def restore(self, c):
        self.u[:] = c["u"]; self.phi[:] = c["phi"]; self.energy = dict(c["energy"])
        self.s_maxwell = np.array(c.get("s_maxwell", np.zeros((len(self.wheels), 3))))
        self.slid = np.array(c.get("slid", np.zeros(len(self.wheels), bool)))
