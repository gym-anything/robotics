"""Pools: well-mixed liquid in sink bowls (static boxes) and in the cavities of ware (glasses,
bowls, colander, can), with free surfaces normal to the effective gravity, drains, weirs, holes, flooding, buoyancy, drag.

Cavity level tables in any orientation: the cavity polygon is sampled with volume weights (ware.revolve_samples); for an
upward direction u (body frame, from g_eff) the volume below a plane u.x = s is the summed weight of samples with u.x < s,
so the free surface for a volume V is the s where the sorted cumulative weight reaches V (exact to one sample's volume). A cavity holds at most the volume below its lowest rim point; the
excess leaves over the rim as a weir, Q = Cw (2/3) sqrt(2g) b H^1.5 (b the wetted rim arc), capped for an inverted mouth by
glugging, Q <= 0.3 A_mouth sqrt(g D) (estimate: air must enter as water leaves). Sink bowls: level = (V + V_displaced)/A;
basket drains Q = Cd a f sqrt(2 g h) with the lever waste's opening f; colander holes Q = Cd a sqrt(2 g depth) per
submerged hole. A cup whose rim is below the level of the pool it is dipped in floods through its rim (weir), and drains
back when lifted. Buoyancy rho g V_sub acts at the submerged centroid of the body's material samples plus its air pocket
(cavity samples above its own pool and below the outer level); drag -1/2 rho Cd A |v| v on the submerged part
(A = V_sub^(2/3), estimate). For a body floating in a moving container's pool the opposite forces act on the container.
All amounts are float64 on the host; water and species only move between named accounts.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from .. import params as P
from .engine import NS

RHO = P.WATER["rho"]
G = P.GRAVITY


class Pool:
    def __init__(self, name, kind, body=-1, *, lo=None, hi=None, drain_d=0.0, drain_xy=None, ware=None, item=None):
        self.name, self.kind, self.body = name, kind, int(body)
        self.vol = 0.0
        self.sp = np.zeros(NS)                 # surfactant kg, suspended kg, tracer kg, heat m3 K, foam liquid m3, foam surf kg
        self.chlorine_ppm = 0.0
        self.drain_open = 0.0
        self.drained = 0.0                     # m3 to the building drain (accounted by the system)
        self.disp = 0.0                        # volume displaced by immersed bodies (m3)
        self.n = np.array([0., 0., 1.])
        self.d = -1e3
        self.T_default = 20.0
        if kind == "sponge":
            self.region = ((0., 0., 0.), (0., 0., 0.))
            self.holes = None
            self.cap = 0.0
            self.area = 1.0
        elif kind == "static":
            self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
            self.area = float((self.hi[0] - self.lo[0])*(self.hi[1] - self.lo[1]))
            self.drain_a = math.pi*drain_d**2/4
            self.drain_xy = drain_xy
            self.region = (tuple(self.lo), tuple(self.hi))
        else:
            cav = ware.cavity
            self.item = item
            self.cav = cav
            self.pts, self.vw = cav["samples"], cav["volumes"]
            self.cap = float(cav["capacity"])
            self.rim = cav["rim"]
            self.mouth_r = cav["mouth_r"]
            self.area = math.pi*self.mouth_r**2
            z0, z1, r0, r1 = cav["region"]
            self.region = ((z0, z1, 0.0), (r0, r1, 0.0))
            self.holes = cav.get("holes")
            self.hole_a = cav.get("hole_area", 0.0)
        self.overflow = []                     # pending emissions (x, v, m, d, sp) for the next tick

    # ------------------------------------------------------------------ concentrations
    def conc(self):
        """(NS,) concentrations per m3 of pool water (heat: temperature); [5] the surface foam's detergent per m3 of foam
        liquid (what an emerging wall lifted through the foam carries)."""
        c = np.zeros(NS)
        if self.vol > 1e-12:
            c[:4] = self.sp[:4]/self.vol
        else:
            c[3] = self.T_default
        if self.sp[4] > 1e-15:
            c[5] = max(self.sp[5], 0.0)/self.sp[4]
        return c

    @property
    def T(self):
        return self.sp[3]/self.vol if self.vol > 1e-12 else self.T_default

    def foam_thickness(self):
        return self.sp[4]/(1.0 - P.FOAM["phi"])/max(self.area, 1e-6)

    def add(self, vol, sp):
        self.vol += vol
        self.sp += sp

    def take(self, vol, with_foam=False):
        """Remove `vol` m3 of well-mixed pool liquid: returns its species (NS,). with_foam (water leaving over the rim):
        the surface foam floats on what leaves first, so it goes 3x faster than the liquid fraction (all of it once the
        pool is empty)."""
        vol = min(vol, self.vol)
        out = np.zeros(NS)
        if self.vol > 0 and vol > 0:
            f = vol/self.vol
            out[:4] = self.sp[:4]*f
            self.sp[:4] -= out[:4]
            if with_foam and self.sp[4] > 0:
                ff = 1.0 if vol >= self.vol - 1e-15 else min(1.0, 3.0*f)
                out[4:6] = self.sp[4:6]*ff
                self.sp[4:6] -= out[4:6]
            self.vol -= vol
        return vol, out

    def foam_decay(self, dt):
        """Surface foam on the pool collapses like foam on a film (params.FOAM): half-life, faster where the liquid under
        it is dilute; its liquid and surfactant rejoin the pool."""
        F = self.sp[4]
        if F <= 0:
            return
        c = self.sp[0]/self.vol if self.vol > 1e-12 else 0.0
        k = math.log(2.0)/P.FOAM["half_life"]
        cmc = P.SURFACTANT["cmc"]
        if c < 0.5*cmc:
            k += P.FOAM["k_dil"]*(1.0 - c/(0.5*cmc))
        fr = 1.0 - math.exp(-k*dt)
        dF, dS = F*fr, self.sp[5]*fr
        self.sp[4] -= dF
        self.sp[5] -= dS
        self.vol += dF
        self.sp[0] += dS

    # ------------------------------------------------------------------ geometry
    def plane(self, R=None, p=None, up_world=np.array([0., 0., 1.])):
        """Free-surface plane (n, d) in the pool's frame for its current volume; capacity below the lowest rim point."""
        if self.kind == "sponge":
            self.n, self.d, self.level_world = np.array([0., 0., 1.]), -1e3, None
            return self.n, self.d
        if self.kind == "static":
            z = self.lo[2] + (self.vol + self.disp)/self.area
            self.level = z
            self.n, self.d = np.array([0., 0., 1.]), z
            self.vmax = (self.hi[2] - self.lo[2])*self.area - self.disp
            return self.n, self.d
        u = R.T @ up_world
        cache = getattr(self, "_cache", None)
        if cache is None or float(np.abs(cache[0] - u).max()) > 2e-4:
            # sorted projections of the samples on the up direction (re-sorted only when the cavity turns > 0.01 deg)
            s = self.pts @ u
            order = np.argsort(s)
            self._cache = cache = (u.copy(), s[order], np.cumsum(self.vw[order]), float((self.rim @ u).min()))
        _, ss, cum, s_rim = cache
        self.vmax = float(cum[np.searchsorted(ss, s_rim) - 1]) if ss[0] < s_rim else 0.0
        self.s_rim = s_rim
        if self.vol <= 0:
            d = float(ss[0]) - 1e-4
        else:
            k = int(np.searchsorted(cum, min(self.vol, cum[-1])))
            d = float(ss[min(k, len(ss) - 1)])
        self.n, self.d, self.u = u, d, u
        self.level_world = None if p is None else float(up_world @ (p + R @ (u*d)))
        return u, d

    def wet_samples(self):
        """Cavity samples under the free surface (body frame) and their volumes."""
        m = self.pts @ self.n < self.d
        return self.pts[m], self.vw[m]


class PoolSet:
    """All pools of a scene and their per-tick host exchanges."""

    def __init__(self, system):
        self.sys = system
        self.pools = []
        self.index = {}

    def add(self, pool):
        self.index[pool.name] = len(self.pools)
        self.pools.append(pool)
        return pool

    def gpu_specs(self):
        out = []
        for p in self.pools:
            if p.kind == "sponge":
                # never captures packets by its plane: water reaches the pores through the sponge's faces (face_pool)
                out.append(dict(body=p.body, kind=0, n=(0., 0., 1.), d=-1e3, a=(0., 0., 0.), b=(0., 0., 0.)))
            elif p.kind == "static":
                out.append(dict(body=-1, kind=0, n=(0., 0., 1.), d=float(p.lo[2]), a=tuple(p.lo), b=tuple(p.hi)))
            else:
                (z0, z1, _), (r0, r1, _) = p.region
                out.append(dict(body=p.body, kind=1, n=(0., 0., 1.), d=z0, a=(z0, z1, 0.), b=(r0, r1, 0.)))
        return out

    # ------------------------------------------------------------------ per tick
    def update_planes(self, bq, up_of_body):
        n_list, d_list, conc, vol, foam = [], [], np.zeros((NS, len(self.pools))), [], []
        for k, p in enumerate(self.pools):
            if p.kind in ("static", "sponge"):
                n, d = p.plane()
            else:
                q = bq[p.body]
                R = Rotation.from_quat(q[3:]).as_matrix()
                n, d = p.plane(R, q[:3], up_of_body(p.body))
            n_list.append(n); d_list.append(d)
            conc[:, k] = p.conc()
            vol.append(p.vol)
            foam.append(p.foam_thickness())
        return n_list, d_list, conc, np.array(vol), np.array(foam)

    def outflows(self, dt, bq, bqd, com_local, up_of_body, emit):
        """Drains, holes, rim overflow; pending emissions go to emit(x, v, m, d, sp)."""
        for p in self.pools:
            p.foam_decay(dt)
        for p in self.pools:
            if p.vol <= 0 or p.kind == "sponge":
                continue
            if p.kind == "static":
                h = p.level - p.lo[2]
                if p.drain_open > 0 and h > 0:
                    Q = P.POOL["Cd"]*p.drain_a*p.drain_open*math.sqrt(2*G*h)
                    v, sp = p.take(Q*dt)
                    p.drained += v
                    self.sys.to_drain(v, sp)
                if p.vol + p.disp > p.vmax + 1e-12:
                    # a sink bowl filled past its rim overflows at the front: a sheet over the front rim onto the floor
                    v, sp = p.take(min(p.vol, p.vol + p.disp - p.vmax))
                    if v > 0:
                        x = np.array([0.5*(p.lo[0] + p.hi[0]), p.lo[1] - 0.03, p.hi[2] + 0.005])
                        emit(x[None], np.array([[0., -0.3, 0.]]), np.array([v*RHO]), np.array([0.004]), sp[:, None])
                continue
            q = bq[p.body]
            R = Rotation.from_quat(q[3:]).as_matrix()
            com = q[:3] + R @ com_local[p.body]
            vb, wb = bqd[p.body, :3], bqd[p.body, 3:]
            u = p.u
            # holes below the free surface (colander)
            if p.holes is not None and p.vol > 0:
                hp = p.holes[:, :3]
                depth = p.d - hp @ u
                sub = depth > 0
                if sub.any():
                    Qh = P.POOL["Cd"]*p.hole_a*np.sqrt(2*G*depth[sub])
                    Qt = float(Qh.sum())
                    v, sp = p.take(min(p.vol, Qt*dt))
                    if v > 0:
                        idx = np.flatnonzero(sub)
                        pick = idx[np.linspace(0, len(idx) - 1, min(12, len(idx))).astype(int)]
                        w_ = Qh[np.searchsorted(idx, pick)]
                        w_ = w_/w_.sum()
                        xs = q[:3] + hp[pick] @ R.T
                        nrm = p.holes[pick, 3:] @ R.T
                        vj = np.sqrt(2*G*np.maximum(depth[pick], 1e-4))
                        vel = vb + np.cross(wb, xs - com) + nrm*vj[:, None]
                        emit(xs + nrm*0.002, vel, v*RHO*w_, np.full(len(pick), P.ITEMS["colander"]["hole_d"]),
                             sp[:, None]*w_[None])
            # over the rim
            if p.vol > p.vmax + 1e-12:
                H = max(p.d - p.s_rim, 1e-4)
                rs = p.rim @ u
                wet = rs < p.d
                b = 2*math.pi*p.mouth_r*float(wet.mean())
                Q = P.POOL["Cw"]*(2/3)*math.sqrt(2*G)*b*H**1.5
                Q = min(Q, 0.3*p.area*math.sqrt(G*2*p.mouth_r) + (p.vol - p.vmax)/dt*0.0)
                v, sp = p.take(min(p.vol - p.vmax, max(Q*dt, 0.0)), with_foam=True)
                if v > 0:
                    order = np.argsort(rs)[:6]
                    xs = q[:3] + p.rim[order] @ R.T
                    outward = p.rim[order].copy(); outward[:, 2] = 0
                    outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-9)
                    vel = vb + np.cross(wb, xs - com) + (outward @ R.T)*math.sqrt(2*G*H)
                    n_ = len(order)
                    emit(xs + (outward @ R.T)*0.003, vel, np.full(n_, v*RHO/n_), np.full(n_, 0.004), np.tile(sp[:, None]/n_, (1, n_)))

    def flooding(self, dt, bq):
        """Cavities dipped in a sink bowl. Water stands inside to min(outer level, highest rim point): an upright or
        sideways glass floods (air leaves through the top of the mouth), a mouth-down glass keeps its air pocket. The
        cavity's volume relaxes toward that equilibrium at the weir/orifice rate of its submerged mouth, and a flooded,
        submerged cavity exchanges water with the bowl (stirring: k_x = 0.05 + 2 |v| per s, estimate) so its soap is
        diluted into the bowl. (First build: inflow capped at the volume below the lowest rim point left a sideways glass
        dry and its soapy inside untouched in the sanitizer; demo A1.)"""
        statics = [p for p in self.pools if p.kind == "static" and p.vol > 0]
        for p in self.pools:
            if p.kind == "static":
                continue
            if p.kind == "sponge":
                self._sponge_flooding(p, statics, dt, bq)
                continue
            q = bq[p.body]
            R = Rotation.from_quat(q[3:]).as_matrix()
            rim_w = q[:3] + p.rim @ R.T
            for s in statics:
                inside = (rim_w[:, 0] > s.lo[0]) & (rim_w[:, 0] < s.hi[0]) & (rim_w[:, 1] > s.lo[1]) & (rim_w[:, 1] < s.hi[1])
                if not inside.any():
                    continue
                sub = inside & (rim_w[:, 2] < s.level)
                if not sub.any():
                    continue
                # equilibrium: cavity samples below min(outer level, highest rim point), in world height
                pts_w = q[:3] + p.pts @ R.T
                cap_z = min(s.level, float(rim_w[:, 2].max()))
                v_eq = float(p.vw[pts_w[:, 2] < cap_z].sum())
                head = max(s.level - float(rim_w[sub, 2].min()), 1e-4)
                b = 2*math.pi*p.mouth_r*float(sub.mean())
                Q = P.POOL["Cw"]*(2/3)*math.sqrt(2*G)*b*head**1.5
                dv = v_eq - p.vol
                if dv > 1e-9:
                    v, sp = s.take(min(dv, Q*dt))
                    p.add(v, sp)
                elif dv < -1e-9:
                    v, sp = p.take(min(-dv, Q*dt))
                    s.add(v, sp)
                if sub.all() and p.vol > 0:
                    # exchange with the bowl through the submerged mouth (equal volumes swap, species follow)
                    vb = self.sys.eng.body_v.numpy()[p.body] if hasattr(self.sys, "eng") else np.zeros(3)
                    kx = 0.05 + 2.0*float(np.linalg.norm(vb))
                    vx = min(p.vol, s.vol)*min(1.0, kx*dt)
                    if vx > 0:
                        v1, sp1 = p.take(vx)
                        v2, sp2 = s.take(v1)
                        p.add(v2, sp2)
                        s.add(v1, sp1)

    def _sponge_flooding(self, p, statics, dt, bq):
        """A sponge under a bowl's water fills to its capacity and exchanges pore water with the bowl."""
        S = P.SPONGE
        x = bq[p.body, :3]
        for s in statics:
            if not (s.lo[0] < x[0] < s.hi[0] and s.lo[1] < x[1] < s.hi[1] and x[2] < s.level - 0.005):
                continue
            vmax = getattr(p, "vmax", p.cap)
            if p.vol < vmax:
                v, sp = s.take(min(vmax - p.vol, vmax*S["k_fill"]*dt))
                p.add(v, sp)
            vb = self.sys.eng.body_v.numpy()[p.body]
            kx = S["k_mix0"] + S["k_mix1"]*float(np.linalg.norm(vb))
            vx = min(p.vol, s.vol)*min(1.0, kx*dt)
            if vx > 0:
                v1, sp1 = p.take(vx)
                v2, sp2 = s.take(v1)
                p.add(v2, sp2)
                s.add(v1, sp1)
            return
