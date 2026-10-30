"""9 in x 3/8 in roller cover: compliant nap contact, rolling by traction, nip film split, tray uptake.

Nap contact (not MuJoCo's single cylinder contact): n_around x n_along sample cells on the undeformed nap surface.
For each cell inside a surface by delta (signed distance of the cell's outer point), the nap pushes back with
    p = (E_n / H) delta (1 + beta (delta/H)^2) + c_n d(delta)/dt          (densification; hysteresis = rolling resistance)
along the surface normal, and drags with regularised Coulomb traction  -mu p A v_t / sqrt(|v_t|^2 + v_reg^2)  on the
cell's slip velocity (cover spin included), mu wet or dry. The summed wrench acts on the cover; the cover turns on the
cage axle only because this traction beats the bearing friction, and it skids when it does not. Beyond 85 % nap
compression the cover's rigid core collider (MuJoCo) stops it.
Paint (per cell, solids per paint + vehicle): compression in the nip expresses the share beta delta/H of the liquid a cell
holds (a drier or lighter-pressed cover gives less). When a cell leaves the nip that liquid and the film under it split
with s = 0.5
(forward roll coating at equal surface speeds: Walker and Fetsko 1955, Coyle et al. 1986); the substrate keeps s of
the sum, the nap takes the rest up to its pore volume. Cells below the tray's paint surface take up paint at k_up (C_sat - V) toward saturation
C_sat = sat C0 (BASF ~40 g on a 4 in cover, PCI Magazine 2018). Overloaded down-facing cells drip.
"""

import math
import numpy as np

from . import params as P
from . import tools as T
from .film import _quat_rot

SAT = P.ROLLER["sat"]      # uptake saturation of the pore volume (BASF 4-in cover: 40 g in 222 mL of nap) -- derived
BETA = P.ROLLER["express"]  # liquid expressed in the nip / liquid held, per unit nap strain (E8/E9 calibration)
NAP_D = 5e-6               # m2/s capillary redistribution in the pile (~1.4 s between 2.7 mm cells; estimate)


class Roller:
    kind = "roller"

    def __init__(self, system, name, frame_body, cover_body):
        self.sys, self.name = system, name
        self.frame, self.cover = int(frame_body), int(cover_body)
        R = P.ROLLER
        self.H = R["nap"]
        self.Rout = R["core_od"]/2 + R["nap"]
        na, nl = R["n_around"], R["n_along"]
        self.na, self.nl = na, nl
        phi = (np.arange(na) + 0.5)/na*2*math.pi
        ys = (np.arange(nl) + 0.5)/nl*R["cover_len"] - R["cover_len"]/2
        PH, YS = np.meshgrid(phi, ys, indexing="ij")
        self.radial = np.stack([np.cos(PH), np.zeros_like(PH), np.sin(PH)], -1).reshape(-1, 3)   # cover frame (axis y)
        self.local = self.radial*self.Rout + np.stack([np.zeros_like(YS), YS, np.zeros_like(YS)], -1).reshape(-1, 3)
        self.tangent = np.cross(np.array([0., 1., 0.]), self.radial)                                  # rolling direction
        self.A = (2*math.pi*self.Rout/na)*(R["cover_len"]/nl)
        self.arc, self.dy = 2*math.pi*self.Rout/na, R["cover_len"]/nl
        self.C0 = R["porosity"]*self.H*self.A
        n = na*nl
        Pn = len(system.paints)
        self.s = np.zeros((Pn, n)); self.v = np.zeros(n)
        self.delta = np.zeros(n); self.dmax = np.zeros(n)
        self.inside = np.zeros(n, bool)
        self.prim = np.full(n, -1)
        self.contact_pt = np.zeros((n, 3)); self.contact_n = np.zeros((n, 3))
        self.k_ref = R["E_nap"]/self.H
        self.wet_mu = False
        self.wrench = np.zeros(6)
        self.stats = dict(deposited=0.0, picked=0.0, uptake=0.0, dripped=0.0, contacts=0, skid=0.0)
        self._cands = []
        self._mjc = None
        self.mass_lumped = 0.0

    # ------------------------------------------------------------------ helpers
    def _pose(self, bq):
        q = bq[self.cover]
        return np.asarray(q[:3], float), _quat_rot(q[3:7])

    def _vel(self):
        import mujoco
        w = self.sys.w
        k = w._mj_body[self.cover]
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(w.mjm, w.mjd, mujoco.mjtObj.mjOBJ_BODY, k, v6, 0)
        return v6[3:], v6[:3], np.array(w.mjd.xpos[k]), np.array(w.mjd.xipos[k])

    def _surface_vel(self, prim, pts):
        if prim.body < 0:
            return np.zeros_like(pts)
        import mujoco
        w = self.sys.w
        k = w._mj_body[prim.body]
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(w.mjm, w.mjd, mujoco.mjtObj.mjOBJ_BODY, k, v6, 0)
        return v6[3:] + np.cross(v6[:3], pts - np.array(w.mjd.xpos[k]))

    def load(self):
        return self.s.sum(axis=1), float(self.v.sum())

    def load_mass(self):
        return float(sum(self.s[k].sum()*p.rho_solids for k, p in enumerate(self.sys.paints)) + self.v.sum()*P.WATER_RHO)

    # ------------------------------------------------------------------ contact forces
    def forces(self, bq):
        R_ = P.ROLLER
        c, Rc = self._pose(bq)
        cands = self.sys.candidates(self.name, c, R_["cover_len"]/2 + self.Rout + 0.01, bq)
        self._cands = cands
        n = len(self.local)
        if not cands:
            self.inside[:] = False
            self.delta[:] = 0.0
            self.wrench[:] = 0.0
            self._rest_pose, self._reactions = None, []
            return
        # a cover at rest in the same pose (lying on the cloth, parked in the tray) exerts the same wrench: reuse it
        if (getattr(self, "_rest_pose", None) is not None and np.abs(c - self._rest_pose[0]).max() < 2e-7
                and np.abs(Rc - self._rest_pose[1]).max() < 2e-6 and self.sys.w.tick - self._rest_pose[2] < 400):
            w = self.sys.w
            w.body_wrench[self.cover, :3] += self.wrench[:3]
            w.body_wrench[self.cover, 3:] += self.wrench[3:]
            for b, fr, tr in getattr(self, "_reactions", ()):
                w.body_wrench[b, :3] += fr
                w.body_wrench[b, 3:] += tr
            self._inside_now = self.inside.copy()
            return
        X = c + self.local @ Rc.T
        best = np.full(n, np.inf); nrm = np.zeros((n, 3)); idx = np.full(n, -1)
        # per primitive: distance of the cover axis to the surface (46 axial points) bounds which nap cells can touch:
        # a cell at angle phi reaches the surface only if R_out cos(angle to the surface normal) > d_axis
        nl = self.nl
        ax_pts = c + (self.local[:nl] - self.radial[:nl]*self.Rout) @ Rc.T          # axis points (first ring's axial stations)
        rad_w = self.radial @ Rc.T
        for k, pr in enumerate(cands):
            dax, nax = pr.sdf(ax_pts, bq)
            if float(dax.min()) > self.Rout:
                continue
            j = np.tile(np.arange(nl), self.na)                     # axial station of each sample (layout: around x along)
            reach = self.Rout*(-(rad_w*nax[j]).sum(axis=1))
            sel = np.nonzero(reach > dax[j] - 1e-4)[0]
            if not len(sel):
                continue
            d, nn = pr.sdf(X[sel], bq)
            b = d < best[sel]
            best[sel] = np.where(b, d, best[sel]); nrm[sel[b]] = nn[b]; idx[sel[b]] = k
        delta = np.clip(-best, 0.0, self.H)
        inside = delta > 0
        if not np.any(inside):
            self.inside[:] = False; self.delta[:] = 0.0; self.wrench[:] = 0.0
            self._rest_pose, self._reactions = None, []
            return
        v_o, w_c, origin, com = self._vel()
        ii = np.nonzero(inside)[0]
        Xi = X[ii]
        vel = v_o + np.cross(w_c, Xi - origin)
        prims_i = idx[ii]
        vs = np.zeros_like(Xi)
        for k in np.unique(prims_i):
            m = prims_i == k
            vs[m] = self._surface_vel(cands[k], Xi[m])
        vrel = vel - vs
        nn = nrm[ii]
        ddot = (delta[ii] - np.where(self.inside[ii], self.delta[ii], 0.0))/P.DT
        dl = delta[ii]/self.H
        p = self.k_ref*delta[ii]*(1 + R_["beta"]*dl*dl) + R_["c_nap"]*ddot
        p = np.maximum(p, 0.0)
        fn = p*self.A
        vt = vrel - (vrel*nn).sum(axis=1, keepdims=True)*nn
        # wet if the nap cell carries free liquid or the surface film under it is wet
        free = self._free(ii, delta[ii])
        mu = np.where(free > 1e-12, R_["mu_wet"], R_["mu_dry"])
        vmag = np.sqrt((vt*vt).sum(axis=1) + R_["v_reg"]**2)
        ft = -(mu*fn/vmag)[:, None]*vt
        # the cover's spin is a light degree of freedom (I ~ 1e-4 kg m2) and the regularised traction a stiff damper on
        # it (time constant below one tick): integrate the rolling-direction traction backward-Euler in the spin,
        # tau' = tau / (1 + D dt / I), D = sum(mu fn / |v| r^2), so the nap cannot over-spin the cover numerically
        tdir = np.cross(Rc[:, 1][None, :], Xi - c)
        tn = np.linalg.norm(tdir, axis=1, keepdims=True)
        tdir = tdir/np.maximum(tn, 1e-9)
        kc = self.sys.w._mj_body[self.cover]
        Xm = np.asarray(self.sys.w.mjd.ximat[kc]).reshape(3, 3)
        I_ax = float(Rc[:, 1] @ (Xm @ np.diag(np.asarray(self.sys.w.mjm.body_inertia[kc], float)) @ Xm.T) @ Rc[:, 1])
        D = float(((mu*fn/vmag)*(tn[:, 0]**2)).sum())
        fac = 1.0/(1.0 + D*P.DT/max(I_ax, 1e-9))
        ftt = (ft*tdir).sum(axis=1, keepdims=True)*tdir
        ft = ft - ftt + fac*ftt
        F = fn[:, None]*nn + ft
        f = F.sum(axis=0)
        tau = np.cross(Xi - com, F).sum(axis=0)
        w = self.sys.w
        w.body_wrench[self.cover, :3] += f
        w.body_wrench[self.cover, 3:] += tau
        # reaction on movable surfaces (tray)
        self._reactions = []
        for k in np.unique(prims_i):
            pr = cands[k]
            if pr.body >= 0:
                m = prims_i == k
                kb = w._mj_body[pr.body]
                cb = np.array(w.mjd.xipos[kb])
                fr, tr = -F[m].sum(axis=0), np.cross(Xi[m] - cb, -F[m]).sum(axis=0)
                w.body_wrench[pr.body, :3] += fr
                w.body_wrench[pr.body, 3:] += tr
                self._reactions.append((pr.body, fr, tr))
        self.wrench = np.concatenate([f, tau])
        self.stats["skid"] = float(np.sqrt((vt*vt).sum(axis=1)).max())
        # remember for the paint exchange
        self.delta[:] = 0.0
        self.delta[ii] = delta[ii]
        self.prim[:] = -1
        self.prim[ii] = prims_i
        self.contact_pt[ii] = Xi + best[ii][:, None]*nn         # projected onto the surface
        self.contact_n[ii] = nn
        self._inside_now = inside
        self._cand_list = cands
        self._fn = fn
        still = float(np.linalg.norm(v_o)) + float(np.linalg.norm(w_c))*self.Rout < 2e-4 and float(np.abs(ddot).max()) < 1e-3
        self._rest_pose = (c.copy(), Rc.copy(), self.sys.w.tick) if still else None

    def _free(self, ii, delta):
        """Liquid a nap cell offers to the nip: compression expresses a share of what it holds, proportional to the nap
        strain (BETA delta / H, capped at all of it); a drier or lightly pressed cover gives less."""
        V = self.s[:, ii].sum(axis=0) + self.v[ii]
        return V*np.minimum(1.0, BETA*delta/self.H)

    # ------------------------------------------------------------------ paint exchange
    def transfer(self, bq, now):
        sysm = self.sys
        inside = getattr(self, "_inside_now", np.zeros(len(self.local), bool))
        if not self._cands:
            inside = np.zeros(len(self.local), bool)
        entering = inside & ~self.inside
        staying = inside & self.inside
        leaving = ~inside & self.inside
        self.dmax = np.where(entering, self.delta, np.where(staying, np.maximum(self.dmax, self.delta), self.dmax))
        if np.any(entering):
            self.stats["contacts"] += int(entering.sum())
            self._remember(np.nonzero(entering)[0], bq, shear=True)
        if np.any(staying):
            self._remember(np.nonzero(staying)[0], bq, shear=False)
        if np.any(leaving):
            self._split(np.nonzero(leaving)[0], bq, now)
        self.inside = inside.copy()
        self._uptake(bq)
        self._drip(bq)
        if self.sys.w.tick % 20 == 0:
            self._wick(20*P.DT)
        self._inside_now = np.zeros(len(self.local), bool)

    def _wick(self, dt):
        """Capillary redistribution of the liquid inside the pile: wetter cells feed drier neighbours (around and along
        the cover), conservative diffusion of each paint's solids and the vehicle on the periodic (around) x open (along)
        cell grid, D = NAP_D. Without it a ring stripped on the ramp or by a stationary press stayed empty and, rolling
        the same path up and down, picked a line out of the film once per revolution (P3)."""
        D = NAP_D
        da = 2*math.pi*self.Rout/self.na
        dl = P.ROLLER["cover_len"]/self.nl
        ka, kl = D*dt/da**2, D*dt/dl**2
        for arr in list(self.s) + [self.v]:
            g = arr.reshape(self.na, self.nl)
            lap = ka*(np.roll(g, 1, 0) + np.roll(g, -1, 0) - 2*g)
            fl = np.zeros_like(g)
            fl[:, 1:] += kl*(g[:, :-1] - g[:, 1:])
            fl[:, :-1] += kl*(g[:, 1:] - g[:, :-1])
            g += lap + fl

    def _remember(self, ii, bq, shear):
        """Track the substrate cell under each contacting nap cell (the nap does not slip on it while rolling)."""
        if not hasattr(self, "_cell"):
            self._cell = [None]*len(self.local)
        cands = self._cand_list
        for k in np.unique(self.prim[ii]):
            m = ii[self.prim[ii] == k]
            pr = cands[k]
            ch, i, j, ok = self.sys.chart_at(pr, self.contact_pt[m], bq, normal=None)
            for t, cell in enumerate(m):
                if ch is not None and ok[t]:
                    self._cell[cell] = (pr, ch, int(i[t]), int(j[t]))
                else:
                    self._cell[cell] = (pr, None, -1, -1)
            if ch is not None and shear and np.any(ok):
                ch.shear(i[ok], j[ok])

    def _split(self, ii, bq, now):
        R_ = P.ROLLER
        sysm = self.sys
        s_ = R_["split"]
        free = self._free(ii, self.dmax[ii])
        tot = self.s[:, ii].sum(axis=0) + self.v[ii]
        frac_s = np.where(tot > 0, self.s[:, ii]/np.maximum(tot, 1e-30), 0.0)
        frac_v = np.where(tot > 0, self.v[ii]/np.maximum(tot, 1e-30), 0.0)
        groups = {}
        for t, cell in enumerate(ii):
            rec = self._cell[cell] if hasattr(self, "_cell") else None
            if rec is None:
                continue
            groups.setdefault(id(rec[1]) if rec[1] is not None else ("sink", rec[0].name), []).append((t, cell, rec))
        for key, items in groups.items():
            ts = np.array([x[0] for x in items]); cells = np.array([x[1] for x in items])
            pr, ch = items[0][2][0], items[0][2][1]
            if ch is None:
                # surface without a film chart (radiator, floor outside the cloth, tray well floor): the nap leaves s of its
                # free liquid there, into the surface's sink (or back into a container it belongs to)
                give = s_*free[ts]
                self._remove(cells, give, frac_s[:, ts], frac_v[ts])
                gs = (frac_s[:, ts]*give).sum(axis=1); gv = float((frac_v[ts]*give).sum())
                if pr.sink in sysm.reservoirs:
                    sysm.reservoirs[pr.sink].add(gs, gv)
                else:
                    sysm.to_sink(pr.sink, gs, gv, points=self.contact_pt[cells])
                continue
            # sub-points of each nap cell's footprint on the chart (_footprint_cells)
            I = np.array([x[2][2] for x in items]); J = np.array([x[2][3] for x in items])
            A_cell = ch.area
            # substrate liquid under the nap cell: its film thickness times the nap cell's footprint area
            h_s = ch.w[I, J] + ch.s[:, I, J].sum(axis=0)
            area = np.full(len(I), self.A)
            sub_take = h_s*area
            total = free[ts] + sub_take
            to_sub = s_*total
            delta_sub = to_sub - sub_take               # >0: deposit onto the substrate; <0: the nap picks up
            dep = delta_sub > 0
            pick = ~dep
            # deposit: nap free liquid (with the nap's composition) onto the footprint cells
            if np.any(dep):
                vol = delta_sub[dep]
                self._remove(cells[dep], vol, frac_s[:, ts[dep]], frac_v[ts[dep]])
                ang = self._roll_angle(ch, cells[dep], bq)
                fi, fj, fw, owner = self._footprint_cells(ch, cells[dep], I[dep], J[dep], bq)
                sol = frac_s[:, ts[dep]][:, owner]*vol[owner]*fw
                veh = frac_v[ts[dep]][owner]*vol[owner]*fw
                ok = ch.mask[fi, fj]
                ch.add(fi[ok], fj[ok], sol[:, ok]/A_cell, veh[ok]/A_cell, texture=(0.3, P.FILM["tex_wl_roller"], float(ang)),
                       shear=True, now=now)
                if np.any(~ok):
                    sysm.to_sink(f"{ch.name}_edge", sol[:, ~ok].sum(axis=1), float(veh[~ok].sum()))
                self.stats["deposited"] += float(vol.sum())
            if np.any(pick):
                want = -delta_sub[pick]
                room = np.maximum(self.C0 - (self.s[:, cells[pick]].sum(axis=0) + self.v[cells[pick]]), 0.0)
                want = np.minimum(want, room)
                fi, fj, fw, owner = self._footprint_cells(ch, cells[pick], I[pick], J[pick], bq)
                req = want[owner]*fw/A_cell
                gs, gv = ch.take(fi, fj, req)
                got_s = np.zeros((len(sysm.paints), pick.sum())); got_v = np.zeros(pick.sum())
                for k in range(len(sysm.paints)):
                    np.add.at(got_s[k], owner, gs[k]*A_cell)
                np.add.at(got_v, owner, gv*A_cell)
                self.s[:, cells[pick]] += got_s
                self.v[cells[pick]] += got_v
                self.stats["picked"] += float(got_s.sum() + got_v.sum())
                ch.shear(fi, fj)
                ang = self._roll_angle(ch, cells[pick], bq)
                hh = ch.w[fi, fj] + ch.s[:, fi, fj].sum(axis=0)
                ch.tex[fi, fj] = np.maximum(ch.tex[fi, fj], (0.3*hh).astype(np.float32))
                ch.tdir[fi, fj] = ang

    def _footprint_cells(self, ch, cells, I, J, bq):
        """Chart cells covered by each nap cell's footprint: sub-points spread evenly over its whole width along the axis
        (no more than half a chart cell apart) and two along the rolling direction, weights summing to 1 per nap cell.
        (Three sub-points at -1/3, 0, +1/3 of the 4.97 mm nap cell cover only its middle 3.3 mm, and on 2 mm wall cells
        the gaps beat into a 10.3 mm pinstripe of 19 and 38 um columns.)"""
        c, Rc = self._pose(bq)
        t_w = self.tangent[cells] @ Rc.T
        a_w = np.broadcast_to(Rc[:, 1], t_w.shape)
        base = self.contact_pt[cells]
        ka = max(3, int(math.ceil(2.0*self.dy/ch.dx)))
        offs = [(m + 0.5)/ka - 0.5 for m in range(ka)]
        pts, own = [], []
        for st in (-0.25, 0.25):
            for sa in offs:
                pts.append(base + st*self.arc*t_w + sa*self.dy*a_w)
                own.append(np.arange(len(cells)))
        pts = np.concatenate(pts); own = np.concatenate(own)
        su, sv, _ = ch.local(pts, bq)
        nsub = 2*ka
        # each sub-point is shared bilinearly among the four cells around it: assigned to its nearest cell, a sub-point
        # lattice incommensurate with the cells beat into stripes again (2.49 mm nap cells on 2 mm cells: 10 mm, 4.7 um)
        x, y = su/ch.dx - 0.5, sv/ch.dx - 0.5
        i0, j0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - i0, y - j0
        Is, Js, Ws, Os = [], [], [], []
        for di, dj, wgt in ((0, 0, (1 - fx)*(1 - fy)), (1, 0, fx*(1 - fy)), (0, 1, (1 - fx)*fy), (1, 1, fx*fy)):
            ii, jj = i0 + di, j0 + dj
            inside = (ii >= 0) & (ii < ch.nu) & (jj >= 0) & (jj < ch.nv)
            # corners off the chart fall back to the nap cell's own cell
            Is.append(np.where(inside, ii, np.repeat(I[None], nsub, 0).ravel()))
            Js.append(np.where(inside, jj, np.repeat(J[None], nsub, 0).ravel()))
            Ws.append(wgt/nsub); Os.append(own)
        return np.concatenate(Is), np.concatenate(Js), np.concatenate(Ws), np.concatenate(Os)

    def _roll_angle(self, ch, cells, bq):
        c, Rc = self._pose(bq)
        t = self.tangent[cells].mean(axis=0) @ Rc.T
        _, u, v, _ = ch.frame(bq)
        return math.atan2(float(t @ v), float(t @ u))

    def _remove(self, cells, vol, fs, fv):
        np.subtract.at(self.v, cells, fv*vol)
        for k in range(self.s.shape[0]):
            np.subtract.at(self.s[k], cells, fs[k]*vol)
        np.maximum(self.v, 0.0, out=self.v); np.maximum(self.s, 0.0, out=self.s)

    def _uptake(self, bq):
        R_ = P.ROLLER
        c, Rc = self._pose(bq)
        for res in self.sys.reservoirs.values():
            if res.volume <= 0:
                continue
            p, _ = res.pose(bq)
            if np.linalg.norm(p - c) > 0.45:
                continue
            X = c + self.local @ Rc.T
            # a nap cell is wetted when the nap around it (its outer point down to the core) dips below the surface
            imm, depth = res.immersed(X - (self.radial @ Rc.T)*self.H*0.5, bq)
            if not np.any(imm):
                continue
            ii = np.nonzero(imm)[0]
            V = self.s[:, ii].sum(axis=0) + self.v[ii]
            want = np.maximum(SAT*self.C0 - V, 0.0)*min(1.0, R_["uptake"]*P.DT)
            total = float(want.sum())
            if total <= 0:
                continue
            gs, gv = res.take(total)
            got = gs.sum() + gv
            f = want/max(total, 1e-30)*(got/max(total, 1e-30))*total
            for k in range(len(self.sys.paints)):
                self.s[k, ii] += f*(gs[k]/max(got, 1e-30))
            self.v[ii] += f*(gv/max(got, 1e-30))
            self.stats["uptake"] += float(got)

    def _drip(self, bq):
        """Down-facing nap cells holding more than the saturated load drain from their lowest point (k_drip 0.5 /s,
        estimate) as drops (painting.drips)."""
        c, Rc = self._pose(bq)
        nz = (self.radial @ Rc.T)[:, 2]
        V = self.s.sum(axis=0) + self.v
        over = (nz < -0.5) & ~self.inside & (V > 1.15*SAT*self.C0)
        if not np.any(over):
            return
        ii = np.nonzero(over)[0]
        ex = (V[ii] - 1.15*SAT*self.C0)*0.5*P.DT
        tot = V[ii]
        fs = self.s[:, ii]/np.maximum(tot, 1e-30); fv = self.v[ii]/np.maximum(tot, 1e-30)
        self._remove(ii, ex, fs, fv)
        X = c + self.local[ii] @ Rc.T
        self.sys.drips.emit(X, (fs*ex).T, fv*ex, accumulate=f"{self.name}:{self.cover}")
        self.stats["dripped"] += float(ex.sum())

    def lump(self, bq):
        m = self.load_mass()
        if abs(m - self.mass_lumped) > 5e-4:
            self.sys.w.set_lumped_mass(self.cover, m, np.zeros(3))
            self.mass_lumped = m

    def snapshot(self):
        return dict(s=self.s.copy(), v=self.v.copy(), inside=self.inside.copy(), dmax=self.dmax.copy(), stats=dict(self.stats),
                    cell=list(getattr(self, "_cell", [None]*len(self.local))))

    def restore(self, snap):
        self.s[...] = snap["s"]; self.v[...] = snap["v"]; self.inside[...] = snap["inside"]; self.dmax[...] = snap["dmax"]
        self.stats = dict(snap["stats"]); self._cell = list(snap["cell"])
        self.mass_lumped = -1.0

    def state(self):
        return dict(load_g=self.load_mass()*1e3, cells_in_contact=int(self.inside.sum()), wrench=self.wrench.tolist(), **self.stats)
