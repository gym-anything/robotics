"""Brushes as pseudo-rigid-body tufts.

Each tuft (n_f filaments clamped together in the ferrule) is Howell's pseudo-rigid-body cantilever: a rigid link of
length gamma L pivoting at (1 - gamma) L from the clamp on a torsion spring K. K is set so the small-deflection tip
stiffness equals n_f times the exact tapered-filament cantilever stiffness (its integral), which also gives the
tuft's buckling load K / (gamma L) within 1 % of a Rayleigh estimate (sash: 3.75e-4 vs 3.71e-4 N per filament).
Tip contact: penalty k_c along the surface normal; tangentially a loaded (wet) tip feels the viscous drag of the film it
rides on, eta A U / h (no static friction), and a dry tip a stick-slip friction anchor (Coulomb mu); tip damping c;
quasi-static equilibrium per tick by damped Newton on the two tilt angles (warm-started in the brush frame). The
reactions act on the brush body. Rest directions carry the ferrule's splay and 1.5 deg of filament waviness, so an
axial push buckles and flares the tufts instead of stalling on a symmetric bifurcation.
Paint in the bundle (params.BRUSH_PAINT): each tuft's share of the free length's pore volume (from the filament
packing) is wetted by immersion up to the dip depth at k_up, drained by drips to 75 % after withdrawal (ASTM D5068
practice); a residual saturation of 0.25 of the wetted pores is never released (67 % of a drained load leaves at any
dip depth; 65-70 % measured after a D5068 half dip, eGate filament test). The bundle is one porous body: liquid redistributes between neighbouring tufts toward equal saturation (D_bundle), so the tufts
being wiped are fed by the rest (without it a brush-out stopped after 0.6 s with 73 % of the load still in the
non-touching tufts). Under a sliding tip the film is metered toward h_m = L_c sqrt(C eta U / w): a tuft lays
down liquid where the film is thinner and picks it up where thicker; lifting a tip leaves half of its free tip liquid
(film split, Walker and Fetsko 1955) -- stippling.
"""

import math
import numpy as np
from scipy.integrate import quad

from . import params as P
from . import tools as T
from .film import _quat_rot

BP = P.BRUSH_PAINT
K_UP = BP["k_up"]          # 1/s immersion uptake
HOLD = BP["hold"]          # held after draining / dipped pore volume
RESIDUAL = BP["residual"]  # unreleasable liquid per wetted pore volume (capillary hold-up)
K_DRIP = BP["k_drip"]      # 1/s drainage of the excess as drops
C_TIP = BP["c_tip"]        # N s/m tip damping per tuft
BUNDLE_EVERY = 4           # ticks between bundle redistribution steps
WET_FRAC = 0.02            # a tuft holding >= 2 % of its pore volume has a wet (lubricated) tip
H_FILM = 1.0e-4            # m: the film a loaded tip rides on (the brushed film build, 80-115 um wet: E7 and the
                           # Rhoplex SG-10M data sheet)
NEWTON_ITERS = 4            # damped Newton iterations of the quasi-static tuft solve per tick (refinement: E7 --newton 8)
BELLY_F = (0.45, 0.72, 1.0)  # contact samples along a round brush's tuft link (fraction of its length from the pivot)


def tapered_tip_stiffness(E, d0, d1, L):
    I = lambda s: math.pi*(d0 + (d1 - d0)*s/L)**4/64
    return 1.0/quad(lambda s: (L - s)**2/(E*I(s)), 0, L)[0]


class Brush:
    kind = "brush"

    def __init__(self, system, name, body, btype):
        self.sys, self.name, self.body, self.btype = system, name, int(body), btype
        B = P.BRUSHES[btype]
        g, kth = P.PRBM["gamma"], P.PRBM["k_theta"]
        clamp, d, Ls, nf = T.tuft_layout(btype)
        self.n = n = len(clamp)
        self.clamp, self.d, self.L, self.nf = clamp, d, Ls, nf
        self.ell = g*Ls
        self.pivot = clamp + (1 - g)*Ls[:, None]*d
        a = np.where(np.abs(d[:, 2]) < 0.9, 1.0, 0.0)
        ref = np.stack([a, 1 - a, np.zeros(n)], 1)
        b1 = np.cross(d, ref); b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
        self.b1, self.b2 = b1, np.cross(d, b1)
        k_fil = np.array([tapered_tip_stiffness(B["E"], B["d_root"], B["d_tip"], L) for L in Ls])
        self.K = nf*k_fil*self.ell**2
        self.kc, self.kt, self.mu = B["kc"], B["kc"], B["mu"]
        rnd = B.get("round", False)
        plug = 1.0 if rnd else 0.8
        pack = 0.65 if btype == "stencil" else 0.60
        A_env = (math.pi/4*B["width"]*B["thick"]) if rnd else B["width"]*B["thick"]
        self.b = math.sqrt(plug*A_env/n)                      # tuft width (contact strip)
        Lm = float(Ls.mean())
        dm = lambda s: B["d_root"] + (B["d_tip"] - B["d_root"])*s/Lm
        fil_vol_half = quad(lambda s: math.pi*dm(s)**2/4, Lm/2, Lm)[0]
        self.C_half = np.maximum(plug*A_env/n*Ls/2 - nf*fil_vol_half*Ls/Lm, 1e-12)
        self.metering = B["metering"]
        self.coh = float(B.get("cohesion", 0.0))
        self.ribbon = float(B.get("ribbon", 2.0))        # a pressed round tuft's band: b (1 + ribbon sin a)
        self.spread = bool(B.get("round", False)) and self.coh > 0      # a pressed soft tuft flattens into a ribbon
        self.belly = self.spread                                          # ... and lies on its belly
        # bundle graph: neighbouring tufts (8-neighbourhood of the clamp grid) exchange liquid toward equal saturation
        c2 = clamp[:, :2]
        dd = np.linalg.norm(c2[:, None] - c2[None], axis=2)
        sp = float(np.median(np.min(dd + np.eye(n)*1e9, axis=1)))
        I, J = np.nonzero(np.triu(dd < 1.6*sp, 1))
        self._bI, self._bJ = I, J
        self._bG = BP["D_bundle"]/dd[I, J]**2                     # 1/s per unit pore volume
        Pn = len(system.paints)
        self.s = np.zeros((Pn, n)); self.v = np.zeros(n)
        self._hm_tuft = np.full(n, 50e-6)                       # wet coat thickness on each tip (last metered; 50 um before)
        self.cap = np.zeros(n)                                 # pore volume wetted by the deepest dip since it drained
        self.theta = np.zeros((n, 2))
        self.anchor = np.zeros((n, 3)); self.contact = np.zeros(n, bool)
        self.X = np.zeros((n, 3)); self.Xprev = np.zeros((n, 3))
        self.fn = np.zeros(n); self.prim = np.full(n, -1); self.nrm = np.zeros((n, 3))
        self.last_cell = [None]*n
        self.wrench = np.zeros(6)
        self.stats = dict(deposited=0.0, picked=0.0, uptake=0.0, dripped=0.0, stipple=0.0, max_force=0.0, newton_resid=0.0)
        self.mass_lumped = 0.0
        self._cands = []

    # ------------------------------------------------------------------
    def _pose(self, bq):
        q = bq[self.body]
        return np.asarray(q[:3], float), _quat_rot(q[3:7])

    def load(self):
        return self.s.sum(axis=1), float(self.v.sum())

    def load_mass(self):
        return float(sum(self.s[k].sum()*p.rho_solids for k, p in enumerate(self.sys.paints)) + self.v.sum()*P.WATER_RHO)

    def tips(self, bq, theta=None):
        p, R = self._pose(bq)
        th = self.theta if theta is None else theta
        a = np.linalg.norm(th, axis=1)
        S = np.where(a > 1e-6, np.sin(a)/np.maximum(a, 1e-12), 1 - a*a/6)
        Cc = np.cos(a)
        w = th[:, :1]*self.b1 + th[:, 1:]*self.b2
        loc = self.pivot + self.ell[:, None]*(Cc[:, None]*self.d + S[:, None]*w)
        return p + loc @ R.T, (p + self.pivot @ R.T), R

    # ------------------------------------------------------------------ forces
    def forces(self, bq):
        B = P.BRUSHES[self.btype]
        p, R = self._pose(bq)
        centre = p + R @ np.array([0, 0, B["length"]*0.5])
        cands = self.sys.candidates(self.name, centre, B["length"]*0.6 + max(B["width"], B["thick"]), bq)
        self._cands = cands
        self.Xprev = self.X.copy()
        if not cands:
            # free: tufts spring back to rest (quasi-static, no inertia)
            self.theta[:] = 0.0
            self.X, _, _ = self.tips(bq)
            self.contact[:] = False; self.fn[:] = 0.0; self.wrench[:] = 0.0
            self._rest_pose, self._reactions = None, []
            return
        import mujoco
        w = self.sys.w
        # a brush lying still in the same pose exerts the same quasi-static wrench: reuse it
        if (getattr(self, "_rest_pose", None) is not None and np.abs(p - self._rest_pose[0]).max() < 2e-7
                and np.abs(R - self._rest_pose[1]).max() < 2e-6 and w.tick - self._rest_pose[2] < 400):
            w.body_wrench[self.body, :3] += self.wrench[:3]
            w.body_wrench[self.body, 3:] += self.wrench[3:]
            for b, fr, tr in getattr(self, "_reactions", ()):
                w.body_wrench[b, :3] += fr
                w.body_wrench[b, 3:] += tr
            return
        kb = w._mj_body[self.body]
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(w.mjm, w.mjd, mujoco.mjtObj.mjOBJ_BODY, kb, v6, 0)
        v_o, w_b = v6[3:], v6[:3]
        still = float(np.linalg.norm(v_o)) + 0.1*float(np.linalg.norm(w_b)) < 2e-4
        origin, com = np.array(w.mjd.xpos[kb]), np.array(w.mjd.xipos[kb])
        b1w, b2w, dw = self.b1 @ R.T, self.b2 @ R.T, self.d @ R.T
        th = self.theta.copy()
        n = self.n
        # wet tips are lubricated: a loaded tuft's tip rides on the paint film it meters, so its tangential force is the
        # film's viscous drag, eta(U/h) A U / h, and vanishes at rest; only a dry tip on a dry surface has Coulomb stick.
        # (A dry stick anchor at mu 0.35 on every tip, with the sash's 39 N/m lateral tip stiffness, would hold the tips
        # 5-14 mm off to either side depending on how they had touched down.) The drag is implicit: the tip is tied to
        # where it was last tick by c / dt, c = eta A / h, with h the brushed film (~100 um, E7) and A the tuft's
        # pressed patch.
        wet = (self.s.sum(axis=0) + self.v) > WET_FRAC*self.cap
        pt0 = self.sys.paints[0]
        for it in range(NEWTON_ITERS):
            X, Pw, _ = self.tips(bq, th)
            fsel = np.ones(n)
            if self.belly:
                # a soft round brush pressed over lies on its belly: each tuft touches at whichever of three points
                # along its link (45, 72, 100 % from the pivot) is deepest; the force acts there (lever f ell)
                pts = np.stack([Pw + f*(X - Pw) for f in BELLY_F])                  # (3, n, 3)
                best3 = np.full((len(BELLY_F), n), np.inf); nrm3 = np.zeros((len(BELLY_F), n, 3)); idx3 = np.full((len(BELLY_F), n), -1)
                for k, pr in enumerate(cands):
                    dd, nn = pr.sdf(pts.reshape(-1, 3), bq)
                    dd, nn = dd.reshape(len(BELLY_F), n), nn.reshape(len(BELLY_F), n, 3)
                    bb = dd < best3
                    best3 = np.where(bb, dd, best3); nrm3[bb] = nn[bb]; idx3[bb] = k
                ks = np.argmin(best3, axis=0)
                ar = np.arange(n)
                X = pts[ks, ar]; best = best3[ks, ar]; nrm = nrm3[ks, ar]; idx = idx3[ks, ar]
                fsel = np.asarray(BELLY_F)[ks]
            else:
                best = np.full(n, np.inf); nrm = np.zeros((n, 3)); idx = np.full(n, -1)
                for k, pr in enumerate(cands):
                    dd, nn = pr.sdf(X, bq)
                    bb = dd < best
                    best = np.where(bb, dd, best); nrm[bb] = nn[bb]; idx[bb] = k
            pen = np.maximum(-best, 0.0)
            inc = pen > 0
            # a fully bent tuft cannot push harder than about three times its buckling load K / (gamma L): beyond
            # that the ferrule meets the surface (its MuJoCo collider takes over)
            fnorm = np.minimum(self.kc*pen, 3.0*self.K/(fsel*self.ell))
            vtip = v_o + np.cross(w_b, X - origin)
            vn = (vtip*nrm).sum(axis=1)
            fnorm = np.where(inc, np.maximum(fnorm - C_TIP*vn, 0.0), 0.0)
            new = inc & ~self.contact
            anchor = np.where(new[:, None], X, self.anchor)
            vt_k = vtip - (vtip*nrm).sum(axis=1, keepdims=True)*nrm
            gd = np.maximum(np.linalg.norm(vt_k, axis=1), 1e-3)/H_FILM
            a_t = np.linalg.norm(th, axis=1)
            c_drag = pt0.eta(gd, 0.0)*self.b*self.b*(1 + 2*np.sin(np.minimum(a_t, 1.4)))/H_FILM
            kt = np.where(wet, c_drag/P.DT, self.kt)
            e = X - np.where(wet[:, None], self.Xprev, anchor)
            et = e - (e*nrm).sum(axis=1, keepdims=True)*nrm
            ft = -kt[:, None]*et
            ftm = np.linalg.norm(ft, axis=1)
            cap = np.where(wet, 2.0*ftm + 1.0, self.mu*fnorm)           # a wet tip never reaches a cap (no stick)
            slip = ftm > cap
            ft = np.where(slip[:, None], ft*(cap/np.maximum(ftm, 1e-12))[:, None], ft)
            ft = np.where(inc[:, None], ft, 0.0)
            F = fnorm[:, None]*nrm + ft
            n_over = np.zeros(n)
            if self.belly:
                # hair cannot share space: tufts lying on the surface push each other apart in its plane (penalty on
                # overlaps of their pressed width). This is what spreads a pressed round brush into a wider stroke;
                # without it cohesive tufts stacked on one line (E6: 2-3 mm at any press)
                ti = np.nonzero(inc)[0]
                if len(ti) > 1:
                    a_t = np.linalg.norm(th[ti], axis=1)
                    bw = self.b*(1 + self.ribbon*np.sin(np.minimum(a_t, 1.4)))
                    D = X[ti][:, None, :] - X[ti][None, :, :]
                    D = D - (D*nrm[ti][:, None, :]).sum(axis=2, keepdims=True)*nrm[ti][:, None, :]
                    dist = np.linalg.norm(D, axis=2) + np.eye(len(ti))*1e3
                    reach = 0.5*(bw[:, None] + bw[None, :])
                    ov = np.maximum(reach - dist, 0.0)
                    Fex = (self.kc*ov/np.maximum(dist, 1e-9))[:, :, None]*D
                    F[ti] = F[ti] + Fex.sum(axis=1)
                    n_over[ti] = (ov > 0).sum(axis=1)
            # Jacobian of the tip w.r.t. the tilt angles
            a = np.linalg.norm(th, axis=1)
            S = np.where(a > 1e-4, np.sin(a)/np.maximum(a, 1e-12), 1 - a*a/6)
            G = np.where(a > 1e-3, (np.cos(a) - S)/np.maximum(a*a, 1e-12), -1/3 + a*a/30)
            wv = th[:, :1]*b1w + th[:, 1:]*b2w
            ell = self.ell[:, None]
            J1 = ell*(-S[:, None]*th[:, :1]*dw + G[:, None]*th[:, :1]*wv + S[:, None]*b1w)
            J2 = ell*(-S[:, None]*th[:, 1:]*dw + G[:, None]*th[:, 1:]*wv + S[:, None]*b2w)
            J = np.stack([J1, J2], 2)*fsel[:, None, None]               # (n, 3, 2), at the contact point
            g = self.K[:, None]*th - np.einsum("nij,ni->nj", J, F)
            NN = nrm[:, :, None]*nrm[:, None, :]
            Kc = np.where(inc[:, None, None], self.kc*NN + np.where(slip, 0.0, kt)[:, None, None]*(np.eye(3) - NN), 0.0)
            Kc = Kc + (self.kc*n_over)[:, None, None]*(np.eye(3) - NN)
            H = self.K[:, None, None]*np.eye(2) + np.einsum("nij,njk,nkl->nil", J.transpose(0, 2, 1), Kc, J)
            if self.coh > 0:
                # wet-hair cohesion (round brushes): capillary bridges make neighbouring tufts bend together --
                # energy coh K / 2 sum |w_i - w_j|^2 over the bundle graph, w the tilt as a brush-frame rotation vector;
                # solved block-Jacobi inside the Newton iterations (warm-started across ticks)
                Wv = th[:, :1]*self.b1 + th[:, 1:]*self.b2
                diff = Wv[self._bI] - Wv[self._bJ]
                Kp = self.coh*0.5*(self.K[self._bI] + self.K[self._bJ])
                Gc = np.zeros((n, 3)); deg = np.zeros(n)
                np.add.at(Gc, self._bI, Kp[:, None]*diff); np.add.at(Gc, self._bJ, -Kp[:, None]*diff)
                np.add.at(deg, self._bI, Kp); np.add.at(deg, self._bJ, Kp)
                g = g + np.stack([(Gc*self.b1).sum(axis=1), (Gc*self.b2).sum(axis=1)], 1)
                H = H + deg[:, None, None]*np.eye(2)
            step = np.linalg.solve(H, g[:, :, None])[:, :, 0]
            sn = np.linalg.norm(step, axis=1, keepdims=True)
            step = np.where(sn > 0.25, step*0.25/np.maximum(sn, 1e-12), step)
            th = th - step
            an = np.linalg.norm(th, axis=1, keepdims=True)
            th = np.where(an > 1.45, th*1.45/np.maximum(an, 1e-12), th)
        self.stats["newton_resid"] = float(np.abs(g).max())
        self.theta = th
        X, Pw, _ = self.tips(bq, th)
        X = Pw + fsel[:, None]*(X - Pw)                 # the contact point (the tip, or a round brush's belly)
        self.X = X
        self.anchor = np.where((inc & ~wet)[:, None], np.where(slip[:, None], X - (ft/np.maximum(self.kt, 1e-9)) * -1.0, anchor), X)
        dn = float(np.linalg.norm(np.where((inc & wet)[:, None], ft, 0.0).sum(axis=0)))
        self.stats["drag_N"] = dn
        if (inc & wet).sum() >= 10 and np.linalg.norm(v_o) > 0.05:
            # the film drag while stroking (>= 10 wet tips touching, ferrule moving > 5 cm/s): E7 reports it
            self.stats["drag_stroke_sum"] = self.stats.get("drag_stroke_sum", 0.0) + dn
            self.stats["drag_stroke_n"] = self.stats.get("drag_stroke_n", 0) + 1
            self.stats["drag_stroke_max_N"] = max(self.stats.get("drag_stroke_max_N", 0.0), dn)
        self.contact = inc
        # kinematic sliding of each contacting tip with the brush: the quasi-static tilt solve chatters by millimetres
        # from tick to tick around stick/slip, so metering uses the brush's own motion at the tip instead (P4/P7).
        # Every touching tip counts as sliding: wet bristle tips on a wet film drag along (counting only the solver's
        # slip flags, which flicker, left the paint at the touch-down points)
        vt_ = vtip - (vtip*nrm).sum(axis=1, keepdims=True)*nrm
        self.slide = np.where(inc[:, None], vt_*P.DT, 0.0)
        self.fn = fnorm
        self.prim = np.where(inc, idx, -1)
        self.nrm = nrm
        self._cand_list = cands
        f = F.sum(axis=0)
        tau = np.cross(X - com, F).sum(axis=0)
        w.body_wrench[self.body, :3] += f
        w.body_wrench[self.body, 3:] += tau
        self._reactions = []
        for k in np.unique(idx[inc]):
            pr = cands[k]
            if pr.body >= 0:
                m = inc & (idx == k)
                kb2 = w._mj_body[pr.body]
                cb = np.array(w.mjd.xipos[kb2])
                fr, tr = -F[m].sum(axis=0), np.cross(X[m] - cb, -F[m]).sum(axis=0)
                w.body_wrench[pr.body, :3] += fr
                w.body_wrench[pr.body, 3:] += tr
                self._reactions.append((pr.body, fr, tr))
        self.wrench = np.concatenate([f, tau])
        self.stats["max_force"] = max(self.stats["max_force"], float(np.linalg.norm(f)))
        self._rest_pose = (p.copy(), R.copy(), w.tick) if (still and float(np.abs(step).max()) < 1e-6) else None

    # ------------------------------------------------------------------ paint
    def free(self):
        """Releasable liquid: what a tuft holds beyond the residual saturation of its wetted pores."""
        return np.maximum(self.s.sum(axis=0) + self.v - RESIDUAL*self.cap, 0.0)

    def _take_from_tufts(self, idx, vol):
        tot = self.s[:, idx].sum(axis=0) + self.v[idx]
        f = np.where(tot > 0, np.minimum(vol, tot)/np.maximum(tot, 1e-30), 0.0)
        ds = self.s[:, idx]*f; dv = self.v[idx]*f
        self.s[:, idx] -= ds; self.v[idx] -= dv
        return ds, dv

    def transfer(self, bq, now):
        sysm = self.sys
        B = P.BRUSHES[self.btype]
        exiting = np.zeros(self.n, bool)
        if not hasattr(self, "_age"):
            self._age = np.zeros(self.n, int)
        if hasattr(self, "_was"):
            # a lift-off stipples only after a held contact (>= 20 ms): tips flickering in and out of contact from one
            # quasi-static solve to the next dumped half their free paint on every flicker (blobs at touch-down, P4/P7)
            exiting = self._was & ~self.contact & (self._age >= 40)
        self._age = np.where(self.contact, self._age + 1, 0)
        self._was = self.contact.copy()
        if np.any(self.contact):
            self._meter(bq, now)
        if np.any(exiting):
            self._stipple(np.nonzero(exiting)[0], bq, now)
        self._uptake(bq)
        if self.sys.w.tick % BUNDLE_EVERY == 0:
            self._redistribute(BUNDLE_EVERY*P.DT)
        self._drip(bq)

    def _redistribute(self, dt):
        """Liquid moves between neighbouring tufts toward equal saturation V / C (C the free length's pore volume),
        with the donor's composition; positivity: no tuft gives more than half of what it holds in one step."""
        V = self.s.sum(axis=0) + self.v
        if float(V.sum()) <= 0.0:
            return
        C = 2.0*self.C_half
        sig = V/C
        I, J = self._bI, self._bJ
        Cm = 0.5*(C[I] + C[J])
        q = self._bG*Cm*(sig[J] - sig[I])*dt                       # volume J -> I (negative: I -> J)
        if float(np.abs(q).max()) < 1e-16:
            return
        donor = np.where(q > 0, J, I)
        recv = np.where(q > 0, I, J)
        vol = np.abs(q)
        out = np.bincount(donor, vol, minlength=self.n)
        scale = np.minimum(1.0, 0.5*V/np.maximum(out, 1e-30))
        vol = vol*scale[donor]
        tot = np.maximum(V[donor], 1e-30)
        fs = self.s[:, donor]/tot; fv = self.v[donor]/tot
        ds = fs*vol; dv = fv*vol
        for k in range(self.s.shape[0]):
            np.add.at(self.s[k], donor, -ds[k]); np.add.at(self.s[k], recv, ds[k])
        np.add.at(self.v, donor, -dv); np.add.at(self.v, recv, dv)
        # the wetted pore volume of a tuft is at least the liquid it now holds (a receiving tuft is wetted); the excess
        # over HOLD x cap still drains. (Raising cap to V / HOLD instead made every tuft hold what it had: no brush ever
        # drained after a dip -- E7 calibration run 1.)
        Vn = self.s.sum(axis=0) + self.v
        self.cap = np.maximum(self.cap, np.minimum(Vn, C))

    def _meter(self, bq, now):
        sysm = self.sys
        cands = self._cand_list
        ii = np.nonzero(self.contact)[0]
        nn = self.nrm[ii]
        dXt = getattr(self, "slide", np.zeros_like(self.X))[ii]
        dist = np.linalg.norm(dXt, axis=1)
        if float(dist.max()) < 1e-7:
            return                       # no tip slid this tick: nothing is metered (a pressed tip leaves paint on lift-off)
        U = dist/P.DT
        a = np.linalg.norm(self.theta[ii], axis=1)
        Lc = self.b*(1 + 2*np.sin(np.minimum(a, 1.4)))
        # the brush meters as one blade: the line load is the brush's total normal force over its contact width. Per-tuft
        # loads (a barely touching tuft "metered" a thick film, a hard-pressed one scraped it back) churned the film:
        # 1.6 mL laid and 1.5 mL picked back in one brush-out (P4)
        wline = np.full(len(ii), max(float(self.fn[ii].sum())/(len(ii)*self.b), 1e-3))
        pt = sysm.paints[0]
        eta = pt.eta(np.maximum(U, 1e-3)/50e-6, 0.0)
        # the metering law diverges as the load goes to zero; a barely touching tip lays no more than a brushed film
        # can hold (h_max: the thick end of brush-outs, Rhoplex SG-10M data sheet, with margin) -- uncapped, the first
        # touch of a lightly pressed brush dumps 60 um-dry blobs (assisted touch-up)
        h_m = np.minimum(Lc*np.sqrt(self.metering*eta*U/wline), BP["h_max"])
        self.stats["h_m_um"] = float(np.median(h_m)*1e6)
        self._hm_tuft[ii] = np.where(U > 0.005, h_m, self._hm_tuft[ii])     # the wet coat a sliding tip carries
        free = self.free()[ii]
        for k in np.unique(self.prim[ii]):
            m = self.prim[ii] == k
            pr = cands[k]
            sel = ii[m]
            # swept strip: points along the tip's path this tick
            nseg = np.maximum(1, np.ceil(dist[m]/1e-3).astype(int))
            pts, own, apt = [], [], []
            for t, cell in enumerate(sel):
                q = np.linspace(0.5/nseg[t], 1 - 0.5/nseg[t], nseg[t])
                line = (self.X[cell] - dXt[m][t])[None] + q[:, None]*dXt[m][t][None]
                # the tuft's contact band is b wide across its path: three lanes (tufts are 2.6 mm apart on 2 mm wall
                # cells; centre-line deposits alone left unpainted columns between tufts)
                tdir = dXt[m][t]/max(dist[m][t], 1e-12)
                lat = np.cross(nn[m][t], tdir)
                # a soft round-brush tuft pressed over flattens into a ribbon: its band widens with its tilt like its
                # contact length (b (1 + 2 sin a)); stiff nylon tufts keep b
                bw = self.b*(1 + self.ribbon*math.sin(min(float(a[m][t]), 1.4))) if self.spread else self.b
                for off in (-bw/3, 0.0, bw/3):
                    pts.append(line + off*lat[None])
                    own.append(np.full(nseg[t], t)); apt.append(np.full(nseg[t], bw*max(dist[m][t], 1e-12)/(3*nseg[t])))
            pts = np.concatenate(pts); own = np.concatenate(own); apt = np.concatenate(apt)
            moving = dist[m][own] > 1e-7
            ch, ci, cj, ok = sysm.chart_at(pr, pts, bq)
            if ch is None:
                # a surface without film (container wall, radiator, ferrule-free): wiping leaves free liquid there
                give = np.minimum(free[m], 0.3*free[m]*np.minimum(1.0, dist[m]/max(self.b, 1e-6)))
                if np.any(give > 0):
                    ds, dv = self._take_from_tufts(sel, give)
                    sysm.to_sink(pr.sink, ds.sum(axis=1), float(dv.sum()), points=pts)
                continue
            ok &= moving
            if not np.any(ok):
                continue
            pts, own, apt, ci, cj = pts[ok], own[ok], apt[ok], ci[ok], cj[ok]
            hs = ch.w[ci, cj] + ch.s[:, ci, cj].sum(axis=0)
            hm = h_m[m][own]
            dv = (hm - hs)*apt
            dep = dv > 0
            # deposit, limited by each tuft's free liquid
            if np.any(dep):
                want = np.zeros(len(sel)); np.add.at(want, own[dep], dv[dep])
                give = np.minimum(want, free[m])
                scale = np.where(want > 0, give/np.maximum(want, 1e-30), 0.0)
                vol = dv[dep]*scale[own[dep]]
                ds, dvv = self._take_from_tufts(sel, give)
                tot = ds.sum(axis=0) + dvv
                fs = np.where(tot > 0, ds/np.maximum(tot, 1e-30), 0.0); fv = np.where(tot > 0, dvv/np.maximum(tot, 1e-30), 0.0)
                o = own[dep]
                ang = math.atan2(float(np.mean(dXt[m] @ ch.frame(bq)[2])), float(np.mean(dXt[m] @ ch.frame(bq)[1])))
                ch.add(ci[dep], cj[dep], fs[:, o]*vol/ch.area, fv[o]*vol/ch.area,
                       texture=(0.5, P.FILM["tex_wl_brush"], ang), shear=True, now=now)
                self.stats["deposited"] += float(vol.sum())

            if np.any(~dep):
                room = np.maximum(self.C_half[sel]*2 - (self.s[:, sel].sum(axis=0) + self.v[sel]), 0.0)
                req = np.minimum(-dv[~dep], 1e9)
                want = np.zeros(len(sel)); np.add.at(want, own[~dep], req)
                scale = np.where(want > 0, np.minimum(1.0, room/np.maximum(want, 1e-30)), 0.0)
                req = req*scale[own[~dep]]
                gs, gv = ch.take(ci[~dep], cj[~dep], req/ch.area)
                o = own[~dep]
                for kk in range(len(sysm.paints)):
                    np.add.at(self.s[kk], sel[o], gs[kk]*ch.area)
                np.add.at(self.v, sel[o], gv*ch.area)
                ch.shear(ci[~dep], cj[~dep])
                self.stats["picked"] += float(gs.sum()*ch.area + gv.sum()*ch.area)
            for t, cell in enumerate(sel):
                self.last_cell[cell] = (pr, ch)

    def _stipple(self, ii, bq, now):
        """A tip leaving the surface splits the liquid bridge between it and the film (film split 0.5, Walker and Fetsko
        1955): it leaves half of the wet coat it carries over its contact patch, h_tip x L_c x b / 2, bounded by its free
        liquid."""
        sysm = self.sys
        a = np.linalg.norm(self.theta[ii], axis=1)
        Lc = self.b*(1 + 2*np.sin(np.minimum(a, 1.4)))
        give = np.minimum(0.5*self.free()[ii], 0.5*self._hm_tuft[ii]*Lc*self.b)
        give = np.where(give > 1e-15, give, 0.0)
        if not np.any(give > 0):
            return
        for t, cell in enumerate(ii):
            rec = self.last_cell[cell]
            if give[t] <= 0:
                continue
            ds, dv = self._take_from_tufts(np.array([cell]), np.array([give[t]]))
            pts = self.Xprev[cell][None]
            if rec is None or rec[1] is None:
                prim = None
                if 0 <= self.prim[cell] < len(self._cand_list):
                    prim = self._cand_list[self.prim[cell]]
                if prim is None:
                    sysm.drips.emit(pts, ds.T, dv)
                    continue
                sysm.deposit_at(prim, pts, ds, dv, bq, now=now)
                continue
            pr, ch = rec
            su, sv, hh = ch.local(pts, bq)
            ci, cj, ok = ch.cell_index(su, sv)
            # the dab covers the tuft footprint: spread over the cells within b/2
            r = max(self.b/2, ch.dx*0.5)
            offs = np.array([[0, 0]]) if r < ch.dx else np.array([[di, dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)])
            I = np.clip(ci[0] + offs[:, 0], 0, ch.nu - 1); J = np.clip(cj[0] + offs[:, 1], 0, ch.nv - 1)
            good = ch.mask[I, J]
            if not np.any(good) or not ok[0]:
                sysm.deposit_at(pr, pts, ds, dv, bq, now=now)
                continue
            I, J = I[good], J[good]
            k = len(I)
            ch.add(I, J, np.repeat(ds, k, axis=1)/k/ch.area, np.repeat(dv, k)/k/ch.area, texture=(0.6, 0.6e-3, 0.0), shear=True,
                   now=now)
            self.stats["stipple"] += float(give[t])

    def _uptake(self, bq):
        p, R = self._pose(bq)
        mid = p + (self.pivot + 0.5*self.ell[:, None]*self.d) @ R.T
        for res in self.sys.reservoirs.values():
            if res.volume <= 0:
                continue
            rp, _ = res.pose(bq)
            if np.linalg.norm(rp - p) > 0.35:
                continue
            imm_tip, depth = res.immersed(self.X, bq)
            if not np.any(imm_tip):
                continue
            # dipped share of the tuft length (tip upward), wetted pore volume grows with the dip up to the whole tuft
            frac = np.clip(depth/np.maximum(self.L, 1e-6), 0.0, 1.0)
            target = np.where(imm_tip, 2*self.C_half*frac, 0.0)
            self.cap = np.maximum(self.cap, target)
            V = self.s.sum(axis=0) + self.v
            want = np.where(imm_tip, np.maximum(target - V, 0.0)*min(1.0, K_UP*P.DT), 0.0)
            total = float(want.sum())
            if total <= 0:
                continue
            gs, gv = res.take(total)
            got = float(gs.sum() + gv)
            share = want/max(total, 1e-30)
            for k in range(len(self.sys.paints)):
                self.s[k] += share*gs[k]
            self.v += share*gv
            self.stats["uptake"] += got

    def _drip(self, bq):
        V = self.s.sum(axis=0) + self.v
        hold = HOLD*self.cap
        over = (~self.contact) & (V > hold + 1e-12) & (self.cap > 0)
        if self.sys.reservoirs:
            for res in self.sys.reservoirs.values():
                imm, _ = res.immersed(self.X, bq)
                over &= ~imm
        if not np.any(over):
            return
        ii = np.nonzero(over)[0]
        # drops form at the tips only when the tips hang below the heel: the drainage rate goes with the downward
        # component of each tuft's direction (bristles up, as when cutting in at a ceiling, the excess runs back into
        # the bundle)
        _, Pw, _ = self.tips(bq)
        dirn = self.X[ii] - Pw[ii]
        down = np.clip(-dirn[:, 2]/np.maximum(np.linalg.norm(dirn, axis=1), 1e-12), 0.0, 1.0)
        ex = (V[ii] - hold[ii])*K_DRIP*P.DT*down
        if not np.any(ex > 0):
            return
        ii, ex = ii[ex > 0], ex[ex > 0]
        ds, dv = self._take_from_tufts(ii, ex)
        self.sys.drips.emit(self.X[ii], ds.T, dv, accumulate=f"{self.name}")
        self.stats["dripped"] += float(ex.sum())
        # the wetted pore volume does not shrink as a tuft drains or is brushed out: its residual saturation stays in
        # the pores (relaxing cap to V / HOLD let a brush release two thirds of whatever it still held, forever)

    def lump(self, bq):
        m = self.load_mass()
        if abs(m - self.mass_lumped) > 2e-4:
            B = P.BRUSHES[self.btype]
            self.sys.w.set_lumped_mass(self.body, m, np.array([0., 0., B["length"]*0.4]))
            self.mass_lumped = m

    def snapshot(self):
        return dict(s=self.s.copy(), v=self.v.copy(), cap=self.cap.copy(), theta=self.theta.copy(), anchor=self.anchor.copy(),
                    contact=self.contact.copy(), X=self.X.copy(), stats=dict(self.stats), was=getattr(self, "_was", None),
                    age=getattr(self, "_age", None), hm=self._hm_tuft.copy())

    def restore(self, s):
        self.s[...] = s["s"]; self.v[...] = s["v"]; self.cap[...] = s["cap"]; self.theta[...] = s["theta"]
        self.anchor[...] = s["anchor"]; self.contact[...] = s["contact"]; self.X[...] = s["X"]; self.stats = dict(s["stats"])
        if s["was"] is not None:
            self._was = s["was"].copy()
        if s.get("age") is not None:
            self._age = s["age"].copy()
        if s.get("hm") is not None:
            self._hm_tuft[...] = s["hm"]
        self.mass_lumped = -1.0
        self._rest_pose, self._reactions = None, []

    def state(self):
        return dict(load_mL=float((self.s.sum() + self.v.sum())*1e6), tufts_in_contact=int(self.contact.sum()),
                    force_n=float(np.linalg.norm(self.wrench[:3])), max_tilt_deg=float(np.degrees(np.linalg.norm(self.theta, axis=1).max())),
                    **self.stats)
