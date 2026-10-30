"""Aerosol can: a finger-pressed valve, atomised parcels in an entraining jet, impaction.

Valve: the actuator button rides a 2.2 mm slide joint with hard stops at both ends against the valve spring (preload
10 N + 7.5 N/mm: full opening needs ~23.5 N, inside the 15-30 N finger range of aerosol valve guides); the opening
fraction ramps smoothly from 1.0 to 1.8 mm of travel. Only contact can press it -- a fingertip, the table, anything -- never a command.
Flow: m_dot = f(x) * 2.0 g/s (params.AEROSOL: a high-output valve) while the fill lasts; 60 % of the emitted mass is propellant and solvent that
flash in flight (estimate), the rest is the enamel (solids 0.25 after the flash).
Parcels: each carries paint volume Q dt / n with a droplet diameter from a log-normal (median 25 um, Goehler et al.
2014, sigma_g 1.7), launched at 25 m/s in a 12 deg half-angle cone from the button's orifice. Air: round free jet of the spray's
momentum flux J = m_dot U0, U_c(x) = 7 sqrt(J / rho_a) / x, Gaussian e-folding radius 0.113 x, turbulent dispersion
calibrated on the jet's scalar spread; within 0.15 H of the surface the
spray axis meets (H its distance), the normal component decays linearly and turns into a radial wall jet. Droplets
relax to the air velocity with the Schiller-Naumann drag time and fall under gravity (exact exponential update per
0.5 ms tick). A parcel that reaches a surface deposits on that face's film (or sink) as a Gaussian splat of radius
3 mm (it stands for ~2e4 droplets), never under the stencil sheet; the stencil's apertures let it through to the crate.
Parcels slow (< 0.2 m/s after 0.4 s) outside the plume, or still airborne after 3 s, are overspray mist ('air' sink).
"""

import math
import numpy as np

from . import params as P
from . import tools as T
from .film import _quat_rot

FLASH = P.AEROSOL["solvent_flash"]
TURB_DISPERSION = True     # droplets' turbulent dispersion in the jet


class Aerosol:
    kind = "aerosol"

    def __init__(self, system, name, can_body, button_body, joint):
        self.sys, self.name = system, name
        self.can, self.button, self.joint = int(can_body), int(button_body), int(joint)
        A = P.AEROSOL
        w = system.w
        self.qi = int(w.model.joint_q_start.numpy()[joint])
        # the valve spring: the button's slide-joint servo pulls toward q_ref = -preload/k (beyond the top stop at q = 0),
        # so the button rests on its stop under the preload. Newton initialises the target inside the joint range, so it
        # is set here explicitly (the world's servo loop leaves non-robot targets untouched)
        q_ref = -A["preload"]/A["k"]
        t = w.control.joint_target_q.numpy(); t[self.qi] = q_ref; w.control.joint_target_q.assign(t)
        w.targets[self.qi] = q_ref
        pt = system.paints[0]
        fill = A["fill_mass"]*(1 - FLASH)/pt.rho                      # enamel volume after flash
        Pn = len(system.paints)
        self.s = np.zeros(Pn); self.s[0] = fill*pt.phi0
        self.v = fill*(1 - pt.phi0)
        self.prop = A["fill_mass"]*FLASH                                # kg of propellant/solvent left
        self.pos = np.zeros((0, 3)); self.vel = np.zeros((0, 3)); self.d = np.zeros(0); self.ps = np.zeros((0, Pn))
        self.pv = np.zeros(0); self.age = np.zeros(0)
        self.rng = np.random.default_rng(7)
        self.emitted = 0.0
        self.open_frac = 0.0
        self.press = 0.0
        self.stats = dict(emitted_m3=0.0, deposited_m3=0.0, overspray_m3=0.0, max_open=0.0, parcels=0)
        self._carry = 0.0
        self.mass_lumped = None

    def load(self):
        s = self.s + (self.ps.sum(axis=0) if len(self.ps) else 0.0)
        return s, float(self.v + self.pv.sum())

    def forces(self, bq):
        pass

    def _nozzle(self, bq):
        A = P.AEROSOL
        q = bq[self.button]
        R = _quat_rot(q[3:7])
        p = np.asarray(q[:3], float)
        return p + R @ np.array([A["button_r"] + 0.0005, 0., 0.]), R @ np.array([1., 0., 0.])

    def transfer(self, bq, now):
        A = P.AEROSOL
        w = self.sys.w
        x = float(w.state.joint_q.numpy()[self.qi])                   # press depth (m)
        self.press = x
        lo, hi = A["x_open"]
        t = min(1.0, max(0.0, (x - lo)/(hi - lo)))
        f = t*t*(3 - 2*t)
        self.open_frac = f
        self.stats["max_open"] = max(self.stats["max_open"], f)
        if f > 0 and self.prop > 0 and self.s.sum() + self.v > 0:
            mdot = f*A["mdot"]
            pt = self.sys.paints[0]
            Q = mdot*(1 - FLASH)/pt.rho*P.DT                              # enamel volume this tick
            V = self.s.sum() + self.v
            Q = min(Q, V)
            self.prop = max(0.0, self.prop - mdot*FLASH*P.DT)
            fs, fv = self.s/V, self.v/V
            self.s = self.s - fs*Q; self.v -= fv*Q
            self._carry += A["parcels_per_s"]*f*P.DT
            npar = int(self._carry)
            self._carry -= npar
            if npar == 0:
                npar = 1
            o, ax = self._nozzle(bq)
            # cone directions: uniform over the spherical cap of half-angle alpha around the axis
            a = math.radians(A["half_angle_deg"])
            cz = 1 - self.rng.random(npar)*(1 - math.cos(a))
            ph = self.rng.random(npar)*2*math.pi
            sz = np.sqrt(1 - cz*cz)
            ref = np.array([0., 0., 1.]) if abs(ax[2]) < 0.9 else np.array([1., 0., 0.])
            e1 = np.cross(ax, ref); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
            dirs = cz[:, None]*ax + (sz*np.cos(ph))[:, None]*e1 + (sz*np.sin(ph))[:, None]*e2
            vbody = self._can_velocity(o)
            self.pos = np.vstack([self.pos, np.repeat(o[None], npar, 0)])
            self.vel = np.vstack([self.vel, dirs*A["U0"] + vbody])
            self.d = np.concatenate([self.d, A["d50"]*np.exp(math.log(A["sigma_g"])*self.rng.standard_normal(npar))])
            self.ps = np.vstack([self.ps, np.repeat((fs*Q/npar)[None], npar, 0)])
            self.pv = np.concatenate([self.pv, np.full(npar, fv*Q/npar)])
            self.age = np.concatenate([self.age, np.zeros(npar)])
            self.stats["emitted_m3"] += Q
            self.stats["emitted_solids_m3"] = self.stats.get("emitted_solids_m3", 0.0) + float(fs.sum()*Q)
            self.stats["parcels"] += npar
            # jet momentum flux for the air model
            self._J = mdot*A["U0"]
            self._axis = (o, ax)
        else:
            self._J = getattr(self, "_J", 0.0)*math.exp(-P.DT/0.05)          # the entrained jet decays once the valve shuts
        if len(self.pos):
            self._advect(bq)

    def _can_velocity(self, pt):
        import mujoco
        w = self.sys.w
        k = w._mj_body[self.can]
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(w.mjm, w.mjd, mujoco.mjtObj.mjOBJ_BODY, k, v6, 0)
        return v6[3:] + np.cross(v6[:3], pt - np.array(w.mjd.xpos[k]))

    def _air(self, X, bq):
        A = P.AEROSOL
        J = getattr(self, "_J", 0.0)
        if J <= 1e-9 or not hasattr(self, "_axis"):
            return np.zeros_like(X)
        o, ax = self._axis
        r = X - o
        xa = r @ ax
        rad = r - xa[:, None]*ax
        rr = np.linalg.norm(rad, axis=1)
        x0 = 0.01
        xe = np.maximum(xa, 0.0) + x0
        Uc = A["jet_K"]*math.sqrt(J/P.AIR["rho"])/xe
        s = A["jet_spread"]
        eta = rr/xe
        f = np.exp(-(eta/s)**2)
        # self-similar round jet: axial U_c f and, from continuity with U_c ~ 1/x, the radial velocity
        # U_c (eta f - s^2 (1 - f) / (2 eta)) -- outflow near the axis, entrainment toward the jet beyond
        vr = Uc*(eta*f - (s*s/(2*np.maximum(eta, 1e-9)))*(1 - f))
        er = np.where(rr[:, None] > 1e-12, rad/np.maximum(rr, 1e-12)[:, None], 0.0)
        u = (Uc*f)[:, None]*ax + vr[:, None]*er
        u = np.where(((xa > -x0) & (rr < 0.6*xe + 0.02))[:, None], u, 0.0)
        H, n_w = self._target(bq)
        if H is not None:
            zw = (X - (o + H*ax)) @ n_w                     # height above the wall plane (n_w points out of it)
            dl = A["stagnation"]*H
            hitp = o + H*ax
            rvall = X - hitp; rvall -= (rvall @ n_w)[:, None]*n_w
            # the impingement region only: within ~the jet's width of the hit point (an unbounded linear radial wall
            # jet flung parcels lying near the target's plane but metres away across the room, P5)
            inl = (zw < dl) & (zw > -0.01) & (np.linalg.norm(rvall, axis=1) < 0.3*H + 0.03)
            if np.any(inl):
                un = (u[inl] @ n_w)
                ut = u[inl] - un[:, None]*n_w
                hit = o + H*ax
                rv = X[inl] - hit; rv -= (rv @ n_w)[:, None]*n_w
                Ucw = A["jet_K"]*math.sqrt(J/P.AIR["rho"])/(H + x0)
                fac = np.clip(zw[inl]/dl, 0, 1)
                ur = (Ucw/(2*dl))*rv
                urm = np.linalg.norm(ur, axis=1, keepdims=True)
                ur = np.where(urm > Ucw, ur*Ucw/np.maximum(urm, 1e-12), ur)       # wall-jet speed <= the impinging jet's
                u[inl] = ut + (un*fac)[:, None]*n_w + ur
        return u

    def _target(self, bq):
        """Distance along the spray axis to the first surface it meets, and that surface's outward normal."""
        if not hasattr(self, "_axis"):
            return None, None
        key = (round(self.sys.w.tick/20),)
        if getattr(self, "_tkey", None) == key:
            return self._tval
        from .contact import nearest
        o, ax = self._axis
        s = np.linspace(0.02, 1.2, 60)
        pts = o + s[:, None]*ax
        d, n, idx = nearest(self.sys.prims, pts, bq)
        hit = np.nonzero(d < 0)[0]
        self._tval = (float(s[hit[0]]), n[hit[0]]) if len(hit) else (None, None)
        self._tkey = key
        return self._tval

    def _advect(self, bq):
        from .contact import nearest
        mu, rho_a = P.AIR["mu"], P.AIR["rho"]
        pt = self.sys.paints[0]
        dt = P.DT
        u = self._air(self.pos, bq)
        vr = self.vel - u
        Re = rho_a*np.linalg.norm(vr, axis=1)*self.d/mu
        tau = pt.rho*self.d**2/(18*mu*(1 + 0.15*Re**0.687))
        e = np.exp(-dt/tau)
        g = np.array([0., 0., -P.GRAVITY])
        vt = u + g*tau[:, None]
        newv = vt + (self.vel - vt)*e[:, None]
        newp = self.pos + 0.5*(self.vel + newv)*dt
        # turbulent dispersion: droplets that follow a jet's mean flow also follow its eddies -- an isotropic random walk
        # with eddy diffusivity D = kappa U_c x inside the plume (and at the target distance in the impingement layer,
        # where the wall-normal component deposits droplets the mean flow would carry past). kappa is calibrated
        # so the droplet flux spreads like the jet's passive scalar (half-width 0.11 x).
        J = getattr(self, "_J", 0.0)
        in_plume = np.zeros(len(newp), bool)
        if J > 1e-9 and hasattr(self, "_axis") and len(newp):
            o, ax = self._axis
            r = self.pos - o
            xr = r @ ax
            H, _ = self._target(bq)
            xa = np.maximum(xr, 0.0) + 0.01
            if H is not None:
                xa = np.minimum(xa, H + 0.01)
            rr = np.linalg.norm(r - xr[:, None]*ax, axis=1)
            in_plume = (xr > -0.01) & (rr < 0.35*xa + 0.02)
            if H is not None:
                in_plume |= (xr > 0.85*H - 0.02) & (rr < 0.3*H + 0.05)            # impingement layer
            if TURB_DISPERSION:
                Uc = P.AEROSOL["jet_K"]*math.sqrt(J/P.AIR["rho"])/xa
                Dt = np.where(in_plume, P.AEROSOL["jet_kappa"]*Uc*xa, 0.0)
                newp = newp + self.rng.standard_normal(newp.shape)*np.sqrt(2*Dt*dt)[:, None]
        self.age += dt
        c = newp.mean(axis=0)
        rad = float(np.linalg.norm(newp - c, axis=1).max()) + 0.02
        prims = self.sys.candidates(("spray", self.name), c, rad, bq)
        if prims:
            d, nrm, idx = nearest(prims, newp, bq)
            # swept test: a parcel moves up to a few mm per tick and the stencil is 0.3 mm thick; parcels that end
            # within one step of a surface are marched along their segment and stop at the first surface crossed
            step = np.linalg.norm(newp - self.pos, axis=1)
            near = np.nonzero(d < step + 1e-4)[0]
            if len(near):
                K = 8
                fr = (np.arange(1, K + 1)/K)
                seg = self.pos[near][:, None, :] + fr[None, :, None]*(newp[near] - self.pos[near])[:, None, :]
                ds, ns, ix = nearest(prims, seg.reshape(-1, 3), bq)
                ds, ns, ix = ds.reshape(len(near), K), ns.reshape(len(near), K, 3), ix.reshape(len(near), K)
                inside = ds < 0
                first = np.where(inside.any(axis=1), inside.argmax(axis=1), K - 1)
                r_ = np.arange(len(near))
                hitn = inside.any(axis=1)
                d[near] = np.where(hitn, ds[r_, first], np.maximum(d[near], 1e-6))
                nrm[near] = np.where(hitn[:, None], ns[r_, first], nrm[near])
                idx[near] = np.where(hitn, ix[r_, first], idx[near])
                newp[near] = np.where(hitn[:, None], seg[r_, first], newp[near])
        else:
            d, nrm, idx = np.full(len(newp), np.inf), np.zeros_like(newp), np.full(len(newp), -1)
        hit = d < 0
        if np.any(hit):
            # each parcel stands for ~2e4 droplets that the jet's turbulence has spread by the time they land: the splat
            # grows with the flight distance (4 % of it, 3-12 mm), so the fringe lands as a dusting rather than 3 mm
            # dots; splats never reach under a sheet lying on the chart
            r0 = P.AEROSOL["kernel_r"]
            if hasattr(self, "_axis"):
                dist = np.linalg.norm(newp - self._axis[0], axis=1)
                rk = np.clip(np.round(0.02*dist/r0), 1, 4)*r0
            else:
                rk = np.full(len(newp), r0)
            for k in np.unique(idx[hit]):
                for rr in np.unique(rk[hit & (idx == k)]):
                    m = np.nonzero(hit & (idx == k) & (rk == rr))[0]
                    pr = prims[k]
                    at = newp[m] - d[m][:, None]*nrm[m]
                    self.sys.deposit_at(pr, at, self.ps[m].T, self.pv[m], bq, texture=(0.15, P.FILM["tex_wl_spray"], 0.0),
                                        shear=True, now=self.sys.w.sim_time, normal=None, kernel=float(rr))
                    self.stats["deposited_m3"] += float(self.ps[m].sum() + self.pv[m].sum())
        # mist: parcels drifting slowly outside the plume and its impingement layer have left the spray (they settle as
        # dry dust far from the target or leave the box) and parcels still airborne after 3 s are overspray. Slow
        # parcels inside the plume stay, since its wings move at a few cm/s
        slow = (self.age > 0.4) & (np.linalg.norm(newv, axis=1) < 0.2) & ~in_plume
        old = ((self.age > P.AEROSOL["mist_age"]) | slow) & ~hit
        if np.any(old):
            self._dry_fall(newp[old], self.ps[old], self.pv[old], bq)
            self.stats["overspray_m3"] += float(self.ps[old].sum() + self.pv[old].sum())
        keep = ~hit & ~old
        self.pos, self.vel = newp[keep], newv[keep]
        self.d, self.ps, self.pv, self.age = self.d[keep], self.ps[keep], self.pv[keep], self.age[keep]

    def _dry_fall(self, pos, ps, pv, bq):
        """Overspray mist settles as dry dust: the parcels' solvent flashes (to the 'air' sink) and the dried droplets
        fall to the first surface below their position, drifting sideways with the room air (Gaussian, sigma
        AEROSOL['drift'] per metre of fall; estimate) -- the table, the cloth, the booth's floor. A policy is charged
        for the 37-45 % of the can that drifts out."""
        from .contact import nearest
        self.sys.to_sink("air", np.zeros(len(self.sys.paints)), float(pv.sum()))
        n = len(pos)
        zs = np.arange(0.0, 2.6, 0.008)
        fall = np.clip(pos[:, 2:3] - zs[None, :], -0.01, None)                      # (n, k) heights along the drop
        drift = self.rng.standard_normal((n, 2))*P.AEROSOL["drift"]
        pts = np.repeat(pos[:, None, :], len(zs), axis=1)
        pts[:, :, 2] = fall
        frac = np.clip(zs[None, :]/np.maximum(pos[:, 2:3], 0.05), 0.0, 1.0)
        pts[:, :, :2] += frac[:, :, None]*drift[:, None, :]*np.maximum(pos[:, 2:3], 0.05)[:, :, None]
        d, nrm, ix = nearest(self.sys.prims, pts.reshape(-1, 3), bq)
        d, ix = d.reshape(n, len(zs)), ix.reshape(n, len(zs))
        inside = d < 0
        found = inside.any(axis=1)
        first = inside.argmax(axis=1)
        r_ = np.arange(n)
        land, kprim = pts[r_, first], ix[r_, first]
        if np.any(~found):
            self.sys.to_sink("air", ps[~found].sum(axis=0), 0.0)
        for k in np.unique(kprim[found]):
            m = np.nonzero(found & (kprim == k))[0]
            # a mist parcel's dried droplets settle spread over centimetres, not as one 3 mm splat (which drew the
            # dust as dots): splat radius 12 mm
            self.sys.deposit_at(self.sys.prims[k], land[m], ps[m].T, np.zeros(len(m)), bq, texture=None, shear=False,
                                now=self.sys.w.sim_time, normal=None, kernel=4.0*P.AEROSOL["kernel_r"])
            self.stats["dry_fall_m3"] = self.stats.get("dry_fall_m3", 0.0) + float(ps[m].sum())

    def lump(self, bq):
        pt = self.sys.paints[0]
        m = float(self.s.sum()*pt.rho_solids + self.v*P.WATER_RHO + self.prop)
        if self.mass_lumped is None or abs(m - self.mass_lumped) > 5e-4:
            # the drawn-only fill part already carries the initial fill mass: lump the change relative to it
            self.sys.w.set_lumped_mass(self.can, m - P.AEROSOL["fill_mass"], np.array([0., 0., 0.065]))
            self.mass_lumped = m

    def snapshot(self):
        return dict(s=self.s.copy(), v=self.v, prop=self.prop, pos=self.pos.copy(), vel=self.vel.copy(), d=self.d.copy(),
                    ps=self.ps.copy(), pv=self.pv.copy(), age=self.age.copy(), stats=dict(self.stats), carry=self._carry,
                    J=getattr(self, "_J", 0.0))

    def restore(self, s):
        self.s, self.v, self.prop = s["s"].copy(), s["v"], s["prop"]
        self.pos, self.vel, self.d = s["pos"].copy(), s["vel"].copy(), s["d"].copy()
        self.ps, self.pv, self.age = s["ps"].copy(), s["pv"].copy(), s["age"].copy()
        self.stats, self._carry, self._J = dict(s["stats"]), s["carry"], s["J"]
        self.mass_lumped = None

    def state(self):
        return dict(press_mm=self.press*1e3, open_fraction=self.open_frac, parcels_airborne=int(len(self.pos)),
                    fill_left_mL=float((self.s.sum() + self.v)*1e6), **self.stats)
