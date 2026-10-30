"""Stable liquid-container coupling: the contained liquid's mass is lumped into each container's rigid inertia.

Explicit (staggered) coupling of a light container with a heavier liquid is unstable: a 0.5 kg bottle holding 0.75 kg of
wine, lifted by a phantom hand, blew up within 25 ms (the liquid's reaction to the bottle's last acceleration, larger
than the bottle's own inertia, amplifies every step: the added-mass instability of partitioned FSI, Causin, Gerbeau &
Nobile 2005). Standard remedy (added-mass partitioning): the rigid solver carries the liquid's mass at its centre of mass and its
inertia tensor (from the particles' second moments) inside the container, and the liquid wrench handed to the rigid
solver is corrected by the part the lumped mass already accounts for,

    F_extra = F_liquid + m_l (a_lag - g)
    tau_extra = tau_liquid + (c_l - c) x m_l (a_lag - g) + I_l alpha_lag                            (about c)

with a_lag, alpha_lag the acceleration of the liquid's centre of mass co-moving with the container and the container's
angular acceleration over the previous steps (filtered). For liquid moving with its container, F_liquid = m_l (g - a) and the correction cancels it; sloshing and
outflow remain in F_extra. The liquid itself still feels the container through its walls exactly as before.
"""

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from . import params as P
from .liquid import Walls, sdf_lookup

ACC_TAU_S = 0.004               # filter of the lagged container acceleration (4 rigid steps)
RESIDUAL_TAU_S = 0.008          # filter of the residual liquid wrench: the DFSPH wall force of a resting full bottle
                                # fluctuates +-2 N per 1 ms step about its 6.4 N mean (warm-started Jacobi stopped at its
                                # residual tolerance); acting unfiltered on the lumped body it kicked the bottle 3 m/s in
                                # one step. Sloshing and outflow forces vary over 50-300 ms; 8 ms passes them


@wp.kernel
def k_container_sums(x: wp.array(dtype=wp.vec3), lid: wp.array(dtype=int), rho0: wp.array(dtype=float), V: float,
                     bodies: wp.array(dtype=int), bpos: wp.array(dtype=wp.vec3), brot: wp.array(dtype=wp.quat),
                     mask: wp.array(dtype=wp.int8), m_off: wp.array(dtype=int), m_nr: wp.array(dtype=int),
                     m_nz: wp.array(dtype=int), m_res: wp.array(dtype=float), m_z0: wp.array(dtype=float),
                     mass: wp.array(dtype=float), mx: wp.array(dtype=wp.vec3), mxx: wp.array(dtype=float)):
    i = wp.tid()
    for k in range(bodies.shape[0]):
        b = bodies[k]
        pl = wp.quat_rotate_inv(brot[b], x[i] - bpos[b])
        r = wp.sqrt(pl[0]*pl[0] + pl[1]*pl[1])
        iu = int(r/m_res[k])
        iw = int((pl[2] - m_z0[k])/m_res[k])
        if iu < 0 or iw < 0 or iu >= m_nr[k] or iw >= m_nz[k]:
            continue
        if mask[m_off[k] + iu*m_nz[k] + iw] == wp.int8(0):
            continue
        m = rho0[lid[i]]*V
        xi = x[i]
        wp.atomic_add(mass, k, m)
        wp.atomic_add(mx, k, xi*m)
        wp.atomic_add(mxx, k*6 + 0, m*xi[0]*xi[0])
        wp.atomic_add(mxx, k*6 + 1, m*xi[1]*xi[1])
        wp.atomic_add(mxx, k*6 + 2, m*xi[2]*xi[2])
        wp.atomic_add(mxx, k*6 + 3, m*xi[0]*xi[1])
        wp.atomic_add(mxx, k*6 + 4, m*xi[0]*xi[2])
        wp.atomic_add(mxx, k*6 + 5, m*xi[1]*xi[2])
        return


@wp.kernel
def k_vertical_impulse(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), lid: wp.array(dtype=int),
                       rho0: wp.array(dtype=float), V: float, bodies: wp.array(dtype=int), bpos: wp.array(dtype=wp.vec3),
                       brot: wp.array(dtype=wp.quat), mask: wp.array(dtype=wp.int8), m_off: wp.array(dtype=int),
                       m_nr: wp.array(dtype=int), m_nz: wp.array(dtype=int), m_res: wp.array(dtype=float),
                       m_z0: wp.array(dtype=float), duz: wp.array(dtype=float), dz: wp.array(dtype=float),
                       x_out: wp.array(dtype=wp.vec3), momentum: wp.array(dtype=float)):
    """Impact impulse: liquid inside container k gets the vertical velocity change duz[k] and follows the container's
    vertical position correction dz[k] (see LumpedLiquid._impact_impulse)."""
    i = wp.tid()
    for k in range(bodies.shape[0]):
        if duz[k] == 0.0:
            continue
        b = bodies[k]
        pl = wp.quat_rotate_inv(brot[b], x[i] - bpos[b])
        r = wp.sqrt(pl[0]*pl[0] + pl[1]*pl[1])
        iu = int(r/m_res[k])
        iw = int((pl[2] - m_z0[k])/m_res[k])
        if iu < 0 or iw < 0 or iu >= m_nr[k] or iw >= m_nz[k]:
            continue
        if mask[m_off[k] + iu*m_nz[k] + iw] == wp.int8(0):
            continue
        vi = v[i]
        v[i] = wp.vec3(vi[0], vi[1], vi[2] + duz[k])
        xi = x[i]
        x_out[i] = wp.vec3(xi[0], xi[1], xi[2] + dz[k])
        wp.atomic_add(momentum, k, rho0[lid[i]]*V*duz[k])
        return


@wp.kernel
def k_follow_walls(x: wp.array(dtype=wp.vec3), W: Walls, inst: wp.array(dtype=int), flag: wp.array(dtype=int),
                   mask: wp.array(dtype=wp.int8), m_off: wp.array(dtype=int), m_nr: wp.array(dtype=int),
                   m_nz: wp.array(dtype=int), m_res: wp.array(dtype=float), m_z0: wp.array(dtype=float),
                   p_pred: wp.array(dtype=wp.vec3), q_pred: wp.array(dtype=wp.quat), c_pred: wp.array(dtype=wp.vec3),
                   c_act: wp.array(dtype=wp.vec3), dq: wp.array(dtype=wp.quat), h: float,
                   x_out: wp.array(dtype=wp.vec3), count: wp.array(dtype=int)):
    """Liquid in and against a container follows the container's position correction of the last tick (see
    LumpedLiquid._follow_walls): the particle is moved by the rigid map that takes the container's predicted pose to
    its actual pose."""
    i = wp.tid()
    xi = x[i]
    for k in range(flag.shape[0]):
        if flag[k] == 0:
            continue
        pl = wp.quat_rotate_inv(q_pred[k], xi - p_pred[k])
        r = wp.sqrt(pl[0]*pl[0] + pl[1]*pl[1])
        near = False
        iu = int(r/m_res[k])
        iw = int((pl[2] - m_z0[k])/m_res[k])
        if iu >= 0 and iw >= 0 and iu < m_nr[k] and iw < m_nz[k]:
            if mask[m_off[k] + iu*m_nz[k] + iw] != wp.int8(0):
                near = True
        if not near:
            c = inst[k]
            if r <= W.inst_rmax[c] and pl[2] >= W.inst_z0[c] and pl[2] <= W.inst_zmax[c]:
                if sdf_lookup(W, c, r, pl[2])[0] < h:
                    near = True
        if near:
            x_out[i] = c_act[k] + wp.quat_rotate(dq[k], xi - c_pred[k])
            wp.atomic_add(count, 0, 1)
            return


class LumpedLiquid:
    def __init__(self, world):
        from .inventory import interior_mask
        self.world = world
        L = world.liquid
        conts = sorted(world.scene.containers, key=lambda c: c[1].kind == "coaster")     # coasters last (bottle in a dish)
        self.conts = conts
        masks, offs, nrs, nzs, ress, z0s = [], [], [], [], [], []
        off = 0
        for body, geo, name in conts:
            m, res, z0, nr, nz = interior_mask(geo)
            masks.append(m.astype(np.int8).ravel()); offs.append(off); nrs.append(nr); nzs.append(nz); ress.append(res); z0s.append(z0)
            off += m.size
        with wp.ScopedDevice(L.device):
            self.bodies = wp.array(np.array([b for b, _, _ in conts], np.int32), dtype=int)
            self.mask = wp.array(np.concatenate(masks), dtype=wp.int8)
            self.m_off, self.m_nr, self.m_nz = (wp.array(np.array(a, np.int32), dtype=int) for a in (offs, nrs, nzs))
            self.m_res, self.m_z0 = wp.array(np.array(ress, np.float32), dtype=float), wp.array(np.array(z0s, np.float32), dtype=float)
            self.mass = wp.zeros(len(conts), dtype=float)
            self.mx = wp.zeros(len(conts), dtype=wp.vec3)
            self.mxx = wp.zeros(len(conts)*6, dtype=float)
        mjm = world.solver.mj_model
        nb = world.model.body_count
        m2n = np.asarray(world.solver.mjc_body_to_newton.numpy())[0]
        n2m = {int(n): j for j, n in enumerate(m2n) if n >= 0}
        self.mjb = [n2m[b] for b, _, _ in conts]
        self.glass = []
        for (b, geo, name), j in zip(conts, self.mjb):
            self.glass.append(dict(mass=float(mjm.body_mass[j]), ipos=np.array(mjm.body_ipos[j]), inertia=np.array(mjm.body_inertia[j]),
                                   iquat=np.array(mjm.body_iquat[j])))
        self.m_l = np.zeros(len(conts))
        self.c_l = np.zeros((len(conts), 3))           # liquid COM, world
        self.v_prev = None
        self.w_prev = None
        self.a_lag = np.zeros((len(conts), 3))
        self.al_lag = np.zeros((len(conts), 3))        # lagged angular acceleration
        self.I_l = np.zeros((len(conts), 3, 3))        # liquid inertia about its COM, world frame
        self.res = np.zeros((len(conts), 6))           # filtered residual wrench (force, torque about the combined COM)
        self.log = []
        with wp.ScopedDevice(L.device):
            self.duz = wp.zeros(len(conts), dtype=float)
            self.dz = wp.zeros(len(conts), dtype=float)
            self.momentum = wp.zeros(len(conts), dtype=float)
        self.vz_prev = None
        self.zc_prev = None
        self.impulse_wrench = np.zeros((world.model.body_count, 6))
        self.impulse_steps = 0
        # the liquid instance (liquid.W) of each container, and the last tick's pose and velocity (wall following)
        bodies_liq = [int(b) for b, _ in L.containers]
        self.inst = [bodies_liq.index(b) for b, _, _ in conts]
        with wp.ScopedDevice(L.device):
            self.f_inst = wp.array(np.array(self.inst, np.int32), dtype=int)
            self.f_flag = wp.zeros(len(conts), dtype=int)
            self.f_count = wp.zeros(1, dtype=int)
        self.q_prev = None
        self.qd_prev = None
        self.follow_steps = 0
        self.follow_particles = 0

    def update(self, dt):
        """Measure each container's liquid, update the MuJoCo body inertia, and refresh the lagged acceleration."""
        w, L = self.world, self.world.liquid
        self.mass.zero_(); self.mx.zero_(); self.mxx.zero_()
        if L.n:
            W = L.W
            wp.launch(k_container_sums, L.n, inputs=[L.x, L.lid, L.rho0, L.V, self.bodies, W.bpos, W.brot, self.mask,
                      self.m_off, self.m_nr, self.m_nz, self.m_res, self.m_z0], outputs=[self.mass, self.mx, self.mxx],
                      device=L.device)
        m = self.mass.numpy().astype(np.float64)
        mx = self.mx.numpy().astype(np.float64)
        mxx = self.mxx.numpy().astype(np.float64).reshape(-1, 6)
        q = w.state.body_q.numpy().astype(np.float64)
        qd = w.state.body_qd.numpy().astype(np.float64)
        mjm = w.solver.mj_model
        v_pts = np.zeros((len(self.conts), 3))
        for k, ((b, geo, name), j) in enumerate(zip(self.conts, self.mjb)):
            R = Rotation.from_quat(q[b, 3:]).as_matrix()
            c_g = q[b, :3] + R @ w.body_com[b]
            self.m_l[k] = m[k]
            self.c_l[k] = mx[k]/m[k] if m[k] > 0 else c_g
            v_pts[k] = qd[b, :3] + np.cross(qd[b, 3:], self.c_l[k] - c_g)
            if m[k] > 0:
                Sxx, Syy, Szz, Sxy, Sxz, Syz = mxx[k]
                S = np.array([[Sxx, Sxy, Sxz], [Sxy, Syy, Syz], [Sxz, Syz, Szz]]) - m[k]*np.outer(self.c_l[k], self.c_l[k])
                # an (almost) inviscid liquid does not spin with its container about the container's own axis: only the
                # transverse part is lumped. Lumping the axial moment too (0.8 of a bottle's own) and cancelling it with
                # the lagged I_l alpha correction spun a phantom-held bottle up to 5900 rad/s about its axis
                ax = R[:, 2]
                Pp = np.eye(3) - np.outer(ax, ax)
                self.I_l[k] = Pp @ (np.trace(S)*np.eye(3) - S) @ Pp
            else:
                self.I_l[k] = 0.
            gl = self.glass[k]
            M = gl["mass"] + m[k]
            r_l = R.T @ (self.c_l[k] - q[b, :3])                  # liquid COM in the body frame
            ipos = (gl["mass"]*gl["ipos"] + m[k]*r_l)/M
            Rg = Rotation.from_quat(np.r_[gl["iquat"][1:], gl["iquat"][0]]).as_matrix()
            I = Rg @ np.diag(gl["inertia"]) @ Rg.T
            for mass_, pos in ((gl["mass"], gl["ipos"]), (m[k], r_l)):
                d = pos - ipos
                I = I + mass_*(np.dot(d, d)*np.eye(3) - np.outer(d, d))
            I = I + R.T @ self.I_l[k] @ R                         # the liquid's own inertia about its COM
            # keep the compiled inertial frame (body_iquat): rewriting it made MuJoCo apply the new principal moments
            # along the old axes (126 kN contact forces in one step); the tensor's diagonal in that frame is used (the
            # off-axis liquid of a tilted bottle adds small off-diagonal terms that are dropped)
            Id = np.diag(Rg.T @ I @ Rg)
            mjm.body_mass[j] = M
            mjm.body_ipos[j] = ipos
            mjm.body_inertia[j] = np.maximum(Id, 1e-7)
        followed = self._follow_walls(dt, q, qd)
        self._impact_impulse(dt, qd, q, m, followed)
        w_now = np.array([qd[b, 3:] for b, _, _ in self.conts])
        if self.v_prev is not None:
            alpha = min(1., dt/ACC_TAU_S)
            self.a_lag = self.a_lag + alpha*((v_pts - self.v_prev)/dt - self.a_lag)
            self.al_lag = self.al_lag + alpha*((w_now - self.w_prev)/dt - self.al_lag)
        self.v_prev = v_pts
        self.w_prev = w_now

    def _follow_walls(self, dt, q, qd):
        """The liquid steps against walls extrapolated over the tick with the velocities the bodies had at its start;
        the rigid step then moves them with the velocities after contact. At an impact the two differ by dv dt (a glass
        landing on its bowl, its spin reversed from -9 to +3.6 rad/s: 2.5 mm), so the walls would jump into the liquid
        between ticks and the density solve would shoot it out at the 8 m/s cap (E6 tip). Liquid in a container
        or within a kernel radius of its wall therefore follows the container's correction (the rigid map from its
        predicted to its actual pose) when it exceeds WALL_FOLLOW_TOL_M at the container's extent. Counted."""
        w, L = self.world, self.world.liquid
        followed = np.zeros(len(self.conts), bool)
        if self.q_prev is not None and L.n:
            n = len(self.conts)
            p_pred = np.zeros((n, 3), np.float32); q_pred = np.zeros((n, 4), np.float32)
            c_pred = np.zeros((n, 3), np.float32); c_act = np.zeros((n, 3), np.float32); dq = np.zeros((n, 4), np.float32)
            for k, (b, geo, name) in enumerate(self.conts):
                com = np.asarray(w.body_com[b], float)
                R0 = Rotation.from_quat(self.q_prev[k, 3:])
                c0 = self.q_prev[k, :3] + R0.apply(com)
                om = self.qd_prev[k, 3:]
                R1 = Rotation.from_rotvec(om*dt)*R0
                c1 = c0 + self.qd_prev[k, :3]*dt
                Ra = Rotation.from_quat(q[b, 3:])
                ca = q[b, :3] + Ra.apply(com)
                dR = Ra*R1.inv()
                jump = float(np.linalg.norm(ca - c1) + dR.magnitude()*max(geo.rim_z, 0.1))
                if jump > P.WALL_FOLLOW_TOL_M:
                    followed[k] = True
                p_pred[k] = c1 - R1.apply(com); q_pred[k] = R1.as_quat(); c_pred[k] = c1; c_act[k] = ca; dq[k] = dR.as_quat()
            if followed.any():
                self.f_flag.assign(followed.astype(np.int32))
                self.f_count.zero_()
                wp.launch(k_follow_walls, L.n, inputs=[L.x, L.W, self.f_inst, self.f_flag, self.mask, self.m_off, self.m_nr,
                          self.m_nz, self.m_res, self.m_z0, wp.array(p_pred, dtype=wp.vec3, device=L.device),
                          wp.array(q_pred, dtype=wp.quat, device=L.device), wp.array(c_pred, dtype=wp.vec3, device=L.device),
                          wp.array(c_act, dtype=wp.vec3, device=L.device), wp.array(dq, dtype=wp.quat, device=L.device),
                          float(L.h)], outputs=[L.x, self.f_count], device=L.device)
                c = int(self.f_count.numpy()[0])
                self.follow_steps += 1
                self.follow_particles += c
                L.limiter_totals["wall_follow"] = L.limiter_totals.get("wall_follow", 0) + c
                L.limiter_totals["steps_wall_follow"] = L.limiter_totals.get("steps_wall_follow", 0) + 1
        self.q_prev = np.array([q[b] for b, _, _ in self.conts], np.float64)
        self.qd_prev = np.array([qd[b] for b, _, _ in self.conts], np.float64)
        return followed

    def _impact_impulse(self, dt, qd, q, m, followed=None):
        """Vertical impact impulse handed to the contained liquid directly. When a container is stopped hard (a bottle
        landing on its coaster: 1 m/s in about 4 ms, 25 g), the iterative pressure solve propagates the impulse through a
        25 cm column only a few particle layers per iteration: the column compressed and rebounded, and its top was
        thrown out of the mouth at the 8 m/s cap (14-29 mL in E6 drop). The exact incompressible response of liquid
        with a level free surface to a vertical velocity change of its container is that uniform change (pressure
        impulse P = rho dv (h - z) vanishes on the surface and meets the walls' normal velocity everywhere). It is
        applied to the vertical proper acceleration beyond IMPACT_ACCEL (ordinary handling, < 1-2 g, stays with the
        solver) of a container within IMPACT_MAX_TILT_DEG of upright (the only case where it is exact), with the equal
        and opposite momentum on the container. Counted in impulse_steps."""
        w = self.world
        L = w.liquid
        vz = np.array([qd[b, 2] for b, _, _ in self.conts])
        zc = np.array([q[b, 2] + (Rotation.from_quat(q[b, 3:]).as_matrix() @ w.body_com[b])[2] for b, _, _ in self.conts])
        self.impulse_wrench[:] = 0.
        if self.vz_prev is None or not L.n:
            self.vz_prev, self.zc_prev = vz, zc
            return
        ap = (vz - self.vz_prev)/dt + P.GRAVITY
        # the liquid's walls moved over the last tick at the velocity they had at its start; a container stopped
        # within the tick ends up to v dt (1 mm) above where its liquid was carried: that overlap, resolved by the
        # density solve in one substep, kicked the bottom layer at 3-4 m/s. The liquid follows the correction
        dzc = (zc - self.zc_prev) - self.vz_prev*dt
        self.vz_prev, self.zc_prev = vz, zc
        upright = np.array([Rotation.from_quat(q[b, 3:]).as_matrix()[2, 2] for b, _, _ in self.conts]) > \
            np.cos(np.radians(P.IMPACT_MAX_TILT_DEG))
        hit = (np.abs(ap) > P.IMPACT_ACCEL_M_S2) & (m > 0) & upright
        if not hit.any():
            return
        excess = np.where(hit, np.sign(ap)*(np.abs(ap) - P.IMPACT_ACCEL_M_S2), 0.)
        self.duz.assign((excess*dt).astype(np.float32))
        # the position correction is already applied to followed containers (_follow_walls)
        self.dz.assign(np.where(hit & ~(followed if followed is not None else np.zeros_like(hit)), dzc, 0.).astype(np.float32))
        self.momentum.zero_()
        W = L.W
        wp.launch(k_vertical_impulse, L.n, inputs=[L.x, L.v, L.lid, L.rho0, L.V, self.bodies, W.bpos, W.brot, self.mask,
                  self.m_off, self.m_nr, self.m_nz, self.m_res, self.m_z0, self.duz, self.dz], outputs=[L.x, self.momentum],
                  device=L.device)
        mom = self.momentum.numpy().astype(np.float64)
        for k, (b, geo, name) in enumerate(self.conts):
            if mom[k]:
                R = Rotation.from_quat(q[b, 3:]).as_matrix()
                c_g = q[b, :3] + R @ w.body_com[b]
                F = np.array([0., 0., -mom[k]/dt])
                self.impulse_wrench[b, :3] += F
                self.impulse_wrench[b, 3:] += np.cross(self.c_l[k] - c_g, F)
        self.impulse_steps += 1
        L.limiter_totals["impulse_steps"] = L.limiter_totals.get("impulse_steps", 0) + 1

    def correct(self, wrench):
        """External wrench on each container (liquid + any fixture; force at the glass COM, torque about it) -> the
        filtered residual wrench for the lumped body (torque about the combined COM), including the lumped mass's
        gravity cancellation. Called with the liquid wrench, before fixtures are added (world.step)."""
        w = self.world
        q = w.state.body_q.numpy().astype(np.float64)
        g = np.array([0., 0., -P.GRAVITY])
        out = np.array(wrench, dtype=np.float64)
        mjm = w.solver.mj_model
        for k, ((b, geo, name), j) in enumerate(zip(self.conts, self.mjb)):
            R = Rotation.from_quat(q[b, 3:]).as_matrix()
            c_g = q[b, :3] + R @ w.body_com[b]
            c_new = q[b, :3] + R @ np.asarray(mjm.body_ipos[j])
            F = out[b, :3].copy()
            tau = out[b, 3:] + np.cross(c_g - c_new, F)
            Fc = self.m_l[k]*(self.a_lag[k] - g)
            raw = np.r_[F + Fc, tau + np.cross(self.c_l[k] - c_new, Fc) + self.I_l[k] @ self.al_lag[k]]
            self.res[k] += min(1., w.config.rigid_dt/RESIDUAL_TAU_S)*(raw - self.res[k])
            out[b] = self.res[k]
        return out

    def body_inertia_state(self):
        return dict(m_l=self.m_l.copy(), a_lag=self.a_lag.copy(), al_lag=self.al_lag.copy(), res=self.res.copy(),
                    v_prev=None if self.v_prev is None else self.v_prev.copy(),
                    w_prev=None if self.w_prev is None else self.w_prev.copy(),
                    vz_prev=None if self.vz_prev is None else self.vz_prev.copy(),
                    zc_prev=None if self.zc_prev is None else self.zc_prev.copy(),
                    q_prev=None if self.q_prev is None else self.q_prev.copy(),
                    qd_prev=None if self.qd_prev is None else self.qd_prev.copy())
