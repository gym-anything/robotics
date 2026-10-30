"""Scrub sponge: a compression slide, pore water, and contact exchange with surfaces.

Compression: the two bodies of the sponge (scene.sponge_item) slide together along the base's z; the cellulose's
nonlinear foam spring is an explicit force F = E A_eff (eps + 3 eps^3) on the slide (eps = q / H_c, the 12 mm
cellulose, held at its value at the 10.5 mm stop beyond it), applied equal and opposite to the two halves; the slide's
damping c v (pore water forced through the cells) is the joint's own damping, which MuJoCo integrates implicitly. (An
explicit damping that multiplies the slide speed by H_c is 83x too weak: the 2 g top rings on the spring at ~170 Hz
and a 60 N squeeze drives it to 20 m/s; P15, A5.) E_wet 50 kPa (range 20-200 kPa, a saturated non-reinforced
cellulose sponge measured 0.2 MPa, search) over A_eff 30 cm^2 (estimate: a pinch spreads over about half the 73 cm^2 face):
a 70 N two-finger pinch compresses it by ~35 %.

Pores: V_max(eps) = phi A H_c (1 - eps), phi 0.9 -> 79 mL free, the "20x its weight" claim of a 4 g cellulose block. The
pore water is a pool of the water system (well mixed, with its surfactant, soil, tracer and heat): water striking the
sponge's faces enters it; whatever exceeds V_max(eps) (a squeeze, or a saturated sponge under a tap) leaves -- onto the
surface the sponge presses on when it is in contact, otherwise as drops from its lower face. Immersed in a bowl it fills
to V_max and exchanges with the bowl (stirring speeds the exchange), so squeeze-and-release under water or under the tap
dilutes its soap (porous permeation).

Contact with a charted surface (every 10 ticks, from MuJoCo's contacts): the cells within 6 mm of the sponge's block
and facing it (the cellulose conforms: under a face, beside a side pushed against a wall, round a corner) share the
summed normal force as a uniform pressure p = F / A_touched (first build: a 2.5 cm disc around each of MuJoCo's corner
contact points, which left the middle of the face scrubbing nothing; second: the face's rectangle only, which could not
reach the corner of a pan). Those cells
  * give up film water into the pores when the sponge is under capacity (capillary uptake, tau_cap 0.1 s, estimate);
  * lose adhered soil by abrasion: dm/dt = k_abr p v_slip sigma_ref / sigma_c (Archard form; p the face pressure, v the
    slip speed; k_abr and sigma_ref 1 kPa in params.SPONGE: estimates), the removed solids entering the film; the scour
    face abrades 4x the cellulose (estimate).
A sponge that is not touching a surface removes nothing (hover control).
"""

import math
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from .. import params as P
from .engine import NS, f64

RHO = P.WATER["rho"]


@wp.kernel
def k_patch(idx: wp.array(dtype=int), w_abr: wp.array(dtype=float), uptake_frac: float, h: wp.array(dtype=f64),
            q: wp.array(dtype=wp.vec3), sp: wp.array2d(dtype=f64), area: wp.array(dtype=float), abr: wp.array(dtype=f64),
            soil_sig: wp.array(dtype=float), sig_ref: float, out: wp.array(dtype=f64)):
    """Abrasion into the cells' accumulator; capillary uptake of a fraction of their film (and species) into out[0:1+NS]."""
    i = wp.tid()
    c = idx[i]
    s = soil_sig[c]
    if s > 0.0:
        wp.atomic_add(abr, c, f64(w_abr[i]*sig_ref/wp.max(s, 1.0)))
    if uptake_frac > 0.0:
        A = f64(area[c])
        fr = f64(uptake_frac)
        dv = h[c]*fr*A
        h[c] = h[c] - dv/A
        q[c] = q[c]*(1.0 - uptake_frac)
        wp.atomic_add(out, 0, dv)
        for k in range(4):
            ds = sp[k, c]*fr
            sp[k, c] = sp[k, c] - ds
            wp.atomic_add(out, 1 + k, ds*A)


@wp.kernel
def k_deposit(idx: wp.array(dtype=int), vol: f64, spv: wp.array(dtype=f64), n: int, h: wp.array(dtype=f64),
              sp: wp.array2d(dtype=f64), area: wp.array(dtype=float)):
    """Expelled pore water spread evenly (per area) over the contact patch's cells."""
    i = wp.tid()
    c = idx[i]
    A = f64(area[c])
    share = f64(1.0)/f64(n)
    h[c] = h[c] + vol*share/A
    for k in range(4):
        sp[k, c] = sp[k, c] + spv[k]*share/A


class SpongeModel:
    def __init__(self, system, name):
        from .pools import Pool
        self.sys, self.w = system, system.w
        sc = self.w.scene
        it = sc.items[name]
        self.name, self.base, self.top, self.joint = name, it["body"], it["top"], it["joint"]
        d = P.ITEMS["sponge"]
        self.L, self.W, self.Hc = d["L"], d["W"], 0.012
        self.scour_t, self.conform = d["scour"], P.SPONGE["conform"]
        S = P.SPONGE
        self.E, self.A_eff, self.phi, self.c = S["E_wet"], S["A_eff"], d["phi"], S["damping"]
        self.v_full = self.phi*self.L*self.W*self.Hc
        self.q_stop = S["q_stop"]
        self.pool = Pool(f"sponge:{name}", "sponge", body=self.base)
        self.pool.cap = self.v_full
        self.pool.area = self.L*self.W
        self.shapes = set(it["shapes"])
        self.geoms = None
        self.contact_log = dict(abrasion_events=0, uptake_m3=0.0, deposit_m3=0.0, dripped_m3=0.0)
        self.eps = 0.0
        self.force = 0.0
        self.last_contact = []

    # ------------------------------------------------------------------ mechanics
    def strain(self):
        """Compressive strain and the slide speed (m/s)."""
        q, qd = self.w.joint_value(self.joint)
        return max(0.0, q)/self.Hc, qd

    def capacity(self):
        eps, _ = self.strain()
        return self.v_full*max(0.0, 1.0 - eps)

    def spring(self, wrench):
        eps, v = self.strain()
        self.eps = eps
        e = min(eps, self.q_stop/self.Hc)
        F = self.E*self.A_eff*(e + 3*e**3)
        R = Rotation.from_quat(self.w.state.body_q.numpy()[self.base, 3:]).as_matrix()
        z = R[:, 2]
        wrench[self.top, :3] += F*z                 # pushes the top half up (away from the base)
        wrench[self.base, :3] -= F*z
        self.force = F + self.c*v                    # spring plus the joint's (implicit) damping, for the logs

    # ------------------------------------------------------------------ water
    def _mj_geoms(self):
        if self.geoms is None:
            w = self.w
            self.geoms = {w._shape_geom[s] for s in self.shapes if s in w._shape_geom}
            self.scour = {w._shape_geom[s] for s in self.shapes if s in w._shape_geom and "scour" in w.scene.b.shape_label[s]}
            self.top_geoms = {w._shape_geom[s] for s in self.shapes if s in w._shape_geom and "cell_hi" in w.scene.b.shape_label[s]}
        return self.geoms

    def contacts(self):
        """Per charted item (or charted static surface) the sponge presses on: (item body, part, normal force N summed
        over MuJoCo's contact points, force-weighted slip speed m/s). part names the sponge part carrying most of the
        force ("scour" pad, "top" cellulose face, "side"), for the logs; the patch is geometric (patch)."""
        import mujoco
        w = self.w
        d, m = w.mjd, w.mjm
        g_sp = self._mj_geoms()
        acc = {}
        f6 = np.zeros(6)
        v6 = np.zeros(6)
        trees = self.sys.trees
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if g1 in g_sp:
                mine, other = g1, g2
            elif g2 in g_sp:
                mine, other = g2, g1
            else:
                continue
            ob = int(m.geom_bodyid[other])
            body = self.sys._mj_item.get(ob)
            if body is None or body not in trees:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            fn = abs(float(f6[0]))
            if fn <= 0.0:
                continue
            pt = np.array(c.pos)
            vels = []
            for bb in (int(m.geom_bodyid[mine]), ob):
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bb, v6, 0)
                vels.append(v6[3:] + np.cross(v6[:3], pt - d.xpos[bb]))
            n = np.asarray(c.frame[:3])
            dv = vels[0] - vels[1]
            slip = float(np.linalg.norm(dv - (dv @ n)*n))
            face = "scour" if mine in self.scour else ("top" if mine in self.top_geoms else "side")
            a = acc.setdefault(body, dict(fn=0.0, fs=0.0, faces={}))
            a["fn"] += fn
            a["fs"] += fn*slip
            a["faces"][face] = a["faces"].get(face, 0.0) + fn
        out = []
        for body, a in acc.items():
            face = max(a["faces"], key=a["faces"].get)
            out.append((body, face, a["fn"], a["fs"]/a["fn"]))
        return out

    def patch(self, body, bq):
        """Cells of the item's chart the pressed sponge touches: within 6 mm of the sponge's block (111 x 66 mm, from
        the scour face up to the top face at its current compression) and facing it. The cellulose is soft (it gives
        10 mm in thickness under a hand's load, and far more easily along its length), so a sponge pushed flat into the
        corner of a pan wraps the corner: its face scrubs the floor to the wall's foot and its side the foot of the
        wall. A rigid block could not: its straight edge bridges a round corner as a chord and its face cannot enter a
        concave corner at all (the first scrub demo left the floor's outer 15 mm and the wall's foot untouched).
        Returns (cell indices, their area, whether each is touched by the scour pad)."""
        Rf = Rotation.from_quat(bq[self.base, 3:]).as_matrix()
        c = bq[self.base, :3]
        ztop = float((bq[self.top, :3] - c) @ Rf[:, 2]) + 0.003
        Ri = Rotation.from_quat(bq[body, 3:]).as_matrix()
        tree, cells = self.sys.trees[body]
        mid = c + Rf[:, 2]*(0.5*ztop)
        near = tree.query_ball_point(Ri.T @ (mid - bq[body, :3]), 0.5*math.sqrt(self.L**2 + self.W**2 + ztop**2) + self.conform)
        if not near:
            return np.zeros(0, int), 0.0, np.zeros(0, bool)
        near = np.asarray(near, int)
        xw = bq[body, :3] + tree.data[near] @ Ri.T
        u = (xw - c) @ Rf
        lo, hi = np.array([-0.5*self.L, -0.5*self.W, 0.0]), np.array([0.5*self.L, 0.5*self.W, ztop])
        cl = np.clip(u, lo, hi)
        dv = u - cl
        d = np.linalg.norm(dv, axis=1)
        # a cell inside the block (rigid contact leaves ~0; the far side of a thin part can lie inside too): its
        # nearest face, and how deep
        gaps = np.stack([u[:, 0] - lo[0], hi[0] - u[:, 0], u[:, 1] - lo[1], hi[1] - u[:, 1], u[:, 2] - lo[2], hi[2] - u[:, 2]], axis=1)
        k = np.argmin(gaps, axis=1)
        depth = gaps[np.arange(len(u)), k]
        e_in = np.zeros_like(u)
        e_in[np.arange(len(u)), k//2] = np.where(k % 2 == 0, -1.0, 1.0)
        inside = d < 1e-9
        e = np.where(inside[:, None], e_in, dv/np.maximum(d, 1e-9)[:, None])
        dist = np.where(inside, depth, d)
        nrm = (self.sys.surf.nrm[cells[near]] @ Ri.T) @ Rf
        # ... and facing the block: the far side of a thin wall (a pan's 4 mm bottom) lies within 6 mm too but is not touched
        m = (dist < self.conform) & ((nrm*e).sum(axis=1) < -0.3)
        idx = cells[near[m]]
        scour = cl[m, 2] <= self.scour_t + 1e-4
        return idx, float(self.sys.surf.area[idx].sum()), scour

    def step(self, dt_contact):
        """Every 10 ticks: abrasion and uptake under the pressed face; pore water above capacity leaves."""
        sysm, E = self.sys, self.sys.eng
        pool = self.pool
        pool.vmax = self.capacity()
        hits = self.contacts()
        S = P.SPONGE
        bq = self.w.state.body_q.numpy()
        patch_cells = []
        A_face = self.L*self.W
        self.last_contact = []
        for body, face, fn, slip in hits:
            idx, A, scour = self.patch(body, bq)
            if not len(idx):
                continue
            # the normal force spreads over the touched cells (at least a tenth of the face: an edge contact)
            p = fn/max(A, 0.1*A_face)
            w_abr = (S["k_abr"]*np.where(scour, S["scour_factor"], 1.0)*p*slip*dt_contact).astype(np.float32)
            under = max(0.0, pool.vmax - pool.vol)
            frac = min(0.5, dt_contact/S["tau_cap"]) if under > 0 else 0.0
            out = wp.zeros(1 + NS, dtype=f64, device=E.device)
            ia = wp.array(idx.astype(np.int32), dtype=int, device=E.device)
            wp.launch(k_patch, dim=len(idx), inputs=[ia, wp.array(w_abr, dtype=float, device=E.device), float(frac), E.h, E.q, E.sp,
                                                     E.area, E.abr, E.soil_sig, S["sigma_ref"], out], device=E.device)
            o = out.numpy()
            if o[0] > 0:
                sp = np.zeros(NS); sp[:4] = o[1:5]
                take = min(o[0], under)
                if take < o[0]:
                    # the patch gave more than the pores hold: the rest goes straight back (no loss)
                    back = o[0] - take
                    spb = sp*back/o[0]
                    self._deposit(ia, len(idx), back, spb)
                    sp = sp - spb
                pool.add(take, sp)
                self.contact_log["uptake_m3"] += take
            self.contact_log["abrasion_events"] += 1
            self.last_contact.append(dict(face=face, force_N=round(fn, 3), slip_m_s=round(slip, 4), patch_cm2=round(A*1e4, 2),
                                          pressure_Pa=round(p, 1)))
            patch_cells.append(ia)
        # above capacity: onto the contacted surface, else as drops from the lower face
        if pool.vol > pool.vmax + 1e-12:
            v, sp = pool.take(pool.vol - pool.vmax)
            if patch_cells:
                for ia in patch_cells:
                    self._deposit(ia, ia.shape[0], v/len(patch_cells), sp/len(patch_cells))
                self.contact_log["deposit_m3"] += v
            else:
                self._drip(v, sp, bq)
                self.contact_log["dripped_m3"] += v

    def _deposit(self, ia, n, vol, sp):
        E = self.sys.eng
        wp.launch(k_deposit, dim=n, inputs=[ia, f64(vol), wp.array(np.asarray(sp[:4], np.float64), dtype=f64, device=E.device),
                                            n, E.h, E.sp, E.area], device=E.device)

    def _drip(self, vol, sp, bq):
        w = self.w
        R = Rotation.from_quat(bq[self.base, 3:]).as_matrix()
        corners = np.array([[sx*self.L*0.4, sy*self.W*0.4, 0.0] for sx in (-1, 1) for sy in (-1, 1)])
        X = bq[self.base, :3] + corners @ R.T
        low = X[np.argsort(X[:, 2])[:2]]
        qd = w.state.body_qd.numpy()[self.base]
        n = len(low)
        self.sys.pending.append((low - np.array([0, 0, 0.003]), np.tile(qd[:3], (n, 1)), np.full(n, vol*RHO/n), np.full(n, 0.004),
                                 np.tile(sp[:, None]/n, (1, n))))
