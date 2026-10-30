"""Headspace gas of the bottles: why a tilted bottle glugs instead of emptying freely.

When liquid fills a cross-section of the neck, the air above it inside the bottle is cut off from the room. As liquid
leaves, that trapped air expands adiabatically (glug period 8-21 ms << thermal diffusion ~5 s):
    p = p0 (V_g0 / V_g)^gamma,   gamma = 1.4,
the suction p - p_atm < 0 acts on the liquid's free surface facing the trapped air (holding the column back), and the
equal and opposite force acts on the bottle. When the neck clears a passage again the air vents: p -> p_atm (a bubble
enters within ~1 ms; not resolved). Letting 1 mL out of a sealed full bottle (19 mL headspace at 750 mL) drops the
pressure by gamma p dV/V = 7.5 kPa, three times the column's 2.4 kPa hydrostatic head: the trapped air
throttles the outflow at once, until the neck clears and a bubble enters.

Discretisation: per rigid step, one GPU pass counts the particles inside each bottle and in
2 mm slabs along its neck (body frame, exact interior polygon raster). Walking inward from the mouth, the first slab
holding at least SEAL_FILL of the particles a full slab holds seals the bottle if there is air (> 1 mL) beyond it.
The pressure difference enters as the pressure drop it causes along the neck liquid (from dp at the bottom of the neck
to 0 at the mouth): a uniform force dp/(rho L) per unit mass along the bottle axis on the neck liquid, total dp x neck
area; the reaction goes to the bottle. Air re-enters only as bubbles through the mouth (not resolved): while sealed
under suction the trapped air grows at the bubble-limited rate Q_air = beta A_mouth sqrt(g D) (Clanet & Searby 2004),
so a pouring bottle empties at that rate on average instead of draining like a vented orifice. The suction is bounded
by the column the headspace holds, rho g h (h = the highest liquid above the mouth), and the overpressure by
DP_LIMIT_HEADS rho g L_bottle.
"""

import math
import numpy as np
import warp as wp

from . import params as P

SLAB_M = 0.002
SEAL_FILL = 0.68               # of a full slab's count: ~0.80 of the geometric volume fits a 2 mm lattice in a 20 mm
                               # bore (dx/2 wall gap); a slab with a 20 %-area air channel counts ~0.64
GAS_MIN_ML = 1.0
FREE_SURFACE_S = 0.92
COUNT_TAU_S = 0.008            # the liquid volume inside a bottle is a particle count: ~100 particles flicker across the
                               # interior raster at the walls (0.8 mL, 4 % of a full bottle's 19 mL headspace = 6 kPa);
                               # an 8 ms exponential filter (below the 8-21 ms glug period) keeps that out of the pressure
DP_LIMIT_HEADS = 1.5           # overpressure <= 1.5 rho g L_bottle; the suction is bounded by min(1.5 rho g L_bottle,
                               # rho g h), h the height of the liquid held above the mouth: a sealed headspace cannot pull
                               # harder than the column it holds (count noise once reached 2.3 km/s through an unbounded
                               # suction)
BUBBLE_BETA = 0.80             # bubble-limited venting: while sealed under suction, air enters through the mouth at
                               # Q_air = beta A_mouth sqrt(g D_mouth). Clanet & Searby 2004 give U ~ 0.4-0.5 sqrt(gD) for
                               # an ideal (straight) inverted bottle; 0.80 is calibrated to Kenton et al. 2012's 0.77 L
                               # wine bottle at 80 deg (85-95 g/s; 0.45 gave 51 g/s, exactly this rate). The 40 deg
                               # case is the check (E3)


@wp.kernel
def k_gas_count(x: wp.array(dtype=wp.vec3), n: int, bodies: wp.array(dtype=int), bpos: wp.array(dtype=wp.vec3),
                brot: wp.array(dtype=wp.quat), mask: wp.array(dtype=wp.int8), m_off: wp.array(dtype=int),
                m_nr: wp.array(dtype=int), m_nz: wp.array(dtype=int), m_res: wp.array(dtype=float),
                m_z0: wp.array(dtype=float), z_neck: wp.array(dtype=float), nslab: int, slab: float,
                inside: wp.array(dtype=int), bins: wp.array(dtype=int), zmax: wp.array(dtype=float)):
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
        wp.atomic_add(inside, k, 1)
        wp.atomic_max(zmax, k, x[i][2])
        j = int((pl[2] - z_neck[k])/slab)
        if j >= 0 and j < nslab:
            wp.atomic_add(bins, k*nslab + j, 1)


@wp.kernel
def k_gas_force(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), lid: wp.array(dtype=int),
                rho0: wp.array(dtype=float), V: float, dt: float, bodies: wp.array(dtype=int),
                bpos: wp.array(dtype=wp.vec3), brot: wp.array(dtype=wp.quat), bcom: wp.array(dtype=wp.vec3),
                mask: wp.array(dtype=wp.int8), m_off: wp.array(dtype=int), m_nr: wp.array(dtype=int),
                m_nz: wp.array(dtype=int), m_res: wp.array(dtype=float), m_z0: wp.array(dtype=float),
                dp: wp.array(dtype=float), z_seal: wp.array(dtype=float), z_rim: wp.array(dtype=float),
                bforce: wp.array(dtype=wp.vec3), btorque: wp.array(dtype=wp.vec3)):
    """The headspace pressure difference as the pressure drop it causes along the liquid in the neck: kappa_offset
    falls linearly from dp/rho at the bottom of the neck to 0 at the mouth, i.e. a uniform force dp/(rho L) per unit
    mass along the bottle axis on the neck liquid (total dp x neck area). Applied at the gas surface instead, the same
    force on one particle layer needed tension the free-surface condition forbids: surface particles tore off at
    250 m/s; applied only between the sealing slab and the mouth, a stream filling the last slab got one particle layer
    of it and kept pouring at 330 g/s under full suction."""
    i = wp.tid()
    for k in range(bodies.shape[0]):
        if dp[k] == 0.0:
            continue
        b = bodies[k]
        q = brot[b]
        pl = wp.quat_rotate_inv(q, x[i] - bpos[b])
        if pl[2] < z_seal[k] or pl[2] > z_rim[k]:
            continue                                   # z_seal carries the bottom of the neck (GasModel.update)
        r = wp.sqrt(pl[0]*pl[0] + pl[1]*pl[1])
        iu = int(r/m_res[k])
        iw = int((pl[2] - m_z0[k])/m_res[k])
        if iu < 0 or iw < 0 or iu >= m_nr[k] or iw >= m_nz[k]:
            continue
        if mask[m_off[k] + iu*m_nz[k] + iw] == wp.int8(0):
            continue
        mi = rho0[lid[i]]*V
        L = wp.max(z_rim[k] - z_seal[k], 0.004)
        a = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))*(dp[k]/(rho0[lid[i]]*L))
        v[i] = v[i] + a*dt
        f = a*mi
        wp.atomic_add(bforce, b, -f)
        wp.atomic_add(btorque, b, wp.cross(x[i] - bcom[b], -f))


class GasModel:
    def __init__(self, world):
        from .inventory import interior_mask
        self.world = world
        L = world.liquid
        self.liquid = L
        self.bottles = [(body, geo, name) for body, geo, name in world.scene.containers if geo.kind.startswith("bottle")]
        self.enabled = bool(self.bottles) and L is not None
        if not self.enabled:
            return
        masks, offs, nrs, nzs, ress, z0s, znecks = [], [], [], [], [], [], []
        self.full, self.slab_z, self.v_int = [], [], []
        off = 0
        nslab = 0
        for body, geo, name in self.bottles:
            m, res, z0, nr, nz = interior_mask(geo)
            masks.append(m.astype(np.int8).ravel()); offs.append(off); nrs.append(nr); nzs.append(nz)
            ress.append(res); z0s.append(z0)
            off += m.size
            z_neck = self._neck_start(geo)
            znecks.append(z_neck)
            k = int(math.ceil((geo.rim_z - z_neck)/SLAB_M))
            nslab = max(nslab, k)
            edges = z_neck + SLAB_M*np.arange(k + 1)
            vol = np.diff([geo.volume_below(min(z, geo.rim_z)) for z in edges])
            self.full.append(vol/L.V)
            self.slab_z.append(edges)
            self.v_int.append(geo.capacity)
        self.nslab = nslab
        dev = L.device
        with wp.ScopedDevice(dev):
            self.bodies = wp.array(np.array([b for b, _, _ in self.bottles], np.int32), dtype=int)
            self.mask = wp.array(np.concatenate(masks), dtype=wp.int8)
            self.m_off, self.m_nr, self.m_nz = (wp.array(np.array(a, np.int32), dtype=int) for a in (offs, nrs, nzs))
            self.m_res, self.m_z0 = wp.array(np.array(ress, np.float32), dtype=float), wp.array(np.array(z0s, np.float32), dtype=float)
            self.z_neck = wp.array(np.array(znecks, np.float32), dtype=float)
            self.inside = wp.zeros(len(self.bottles), dtype=int)
            self.zmax = wp.zeros(len(self.bottles), dtype=float)
            self.bins = wp.zeros(len(self.bottles)*nslab, dtype=int)
            self.dp = wp.zeros(len(self.bottles), dtype=float)
            self.z_seal = wp.array(np.full(len(self.bottles), 1e3, np.float32), dtype=float)
            self.z_rim = wp.array(np.array([geo.rim_z for _, geo, _ in self.bottles], np.float32), dtype=float)
        nb = len(self.bottles)
        self.sealed = [False]*nb
        self.p = [P.P_ATM]*nb
        self.p0, self.vg0 = [P.P_ATM]*nb, [0.]*nb
        self.events = []
        self.history = []
        self.vl_f = [None]*nb
        self.n_atm = [0.]*nb                           # trapped air as its volume at p_atm (grows by bubble entry)
        self.z_neck_np = list(znecks)
        self.q_air = [BUBBLE_BETA*math.pi*geo.mouth_r**2*math.sqrt(P.GRAVITY*2*geo.mouth_r) for _, geo, _ in self.bottles]
        self.dp_limit = [DP_LIMIT_HEADS*max(P.LIQUIDS[n]["rho"] for n in P.LIQUIDS)*P.GRAVITY*geo.rim_z
                         for _, geo, _ in self.bottles]
        L.extra_forces.append(self.force)

    @staticmethod
    def _neck_start(geo):
        """Bottom of the neck: where the interior radius first falls below 1.5x the bore radius from below."""
        pts = geo.interior[geo.interior[:, 0] > 1e-6]
        pts = pts[np.argsort(pts[:, 1])]
        lim = 1.5*geo.mouth_r
        z = float(pts[-1, 1])
        for r, zz in pts[::-1]:
            if r > lim:
                break
            z = float(zz)
        return z

    def update(self, t):
        """Once per rigid step, before the liquid step: inventory, seal state, pressure."""
        if not self.enabled or not self.liquid.n:
            return
        L = self.liquid
        self.inside.zero_(); self.bins.zero_(); self.zmax.fill_(-1e6)
        W = L.W
        wp.launch(k_gas_count, L.n, inputs=[L.x, L.n, self.bodies, W.bpos, W.brot, self.mask, self.m_off, self.m_nr,
                  self.m_nz, self.m_res, self.m_z0, self.z_neck, self.nslab, SLAB_M],
                  outputs=[self.inside, self.bins, self.zmax], device=L.device)
        inside = self.inside.numpy()
        zmax = self.zmax.numpy()
        bins = self.bins.numpy().reshape(len(self.bottles), self.nslab)
        dps, zs = [], []
        from scipy.spatial.transform import Rotation
        q = self.world.state.body_q.numpy()
        qb = {body: Rotation.from_quat(q[body, 3:]).as_matrix() for body, _, _ in self.bottles}
        a = min(1., (t - getattr(self, "_t_last", t - 1e-3))/COUNT_TAU_S) if hasattr(self, "_t_last") else 1.
        self._t_last = t
        for k, (body, geo, name) in enumerate(self.bottles):
            raw = inside[k]*L.V
            self.vl_f[k] = raw if self.vl_f[k] is None else self.vl_f[k] + a*(raw - self.vl_f[k])
            vl = self.vl_f[k]
            vg = max(self.v_int[k] - vl, 1e-9)
            full = self.full[k]
            ns = len(full)
            seal_j = None
            for j in range(ns - 1, -1, -1):                       # from the mouth inward
                if bins[k, j] >= SEAL_FILL*full[j]:
                    seal_j = j
                    break
            gas_beyond = 0.
            if seal_j is not None:
                # air inside the bottle on the inner side of the sealing slab
                v_in = self.v_int[k] - (geo.capacity - geo.volume_below(self.slab_z[k][seal_j]))
                liq_in = vl - bins[k, seal_j:].sum()*L.V
                gas_beyond = v_in - liq_in
            # gas collects at the highest point of the interior: it is trapped only if that point lies on the inner
            # side of the sealing slab (an upright bottle's highest point is its mouth: never sealed; the particle
            # packing's ~1.5 % deficit at the walls otherwise read as 10 mL of 'gas' under a full neck)
            trapped = False
            if seal_j is not None:
                az = float(qb[body][2, 2])
                sz = math.sqrt(max(0., 1. - az*az))
                pts = geo.interior
                hgt = pts[:, 1]*az + pts[:, 0]*sz
                zs_ = self.slab_z[k][seal_j]
                inner, outer = hgt[pts[:, 1] < zs_], hgt[pts[:, 1] >= zs_]
                trapped = inner.size > 0 and outer.size > 0 and inner.max() > outer.max() + 0.002
            sealed = seal_j is not None and trapped and gas_beyond > GAS_MIN_ML*1e-6
            if sealed and not self.sealed[k]:
                self.p0[k], self.vg0[k] = self.p[k], vg
                self.n_atm[k] = vg*(self.p[k]/P.P_ATM)**(1./P.GAS_GAMMA)
                self.events.append(dict(t=round(t, 4), bottle=name, event="seal", gas_mL=round(vg*1e6, 2)))
            elif not sealed and self.sealed[k]:
                self.events.append(dict(t=round(t, 4), bottle=name, event="vent", dp_Pa=round(self.p[k] - P.P_ATM, 1)))
            self.sealed[k] = bool(sealed)
            if sealed and self.p[k] < P.P_ATM:
                self.n_atm[k] += self.q_air[k]*(t - self._t_prev if getattr(self, "_t_prev", None) is not None else 0.)
            p_new = P.P_ATM*(self.n_atm[k]/vg)**P.GAS_GAMMA if sealed else P.P_ATM
            # trapped air cannot hold back more than the column it supports: suction beyond rho g h (h = the liquid's
            # height above the mouth) draws air in through the mouth (the glug). The earlier authored bound,
            # 1.5 rho g L_bottle, let the suction exceed the column and set the 40-deg flux (E3: dp sat at the bound)
            mouth_z = float(q[body, 2] + qb[body][2, 2]*geo.rim_z)
            rho = max(P.LIQUIDS[n]["rho"] for n in P.LIQUIDS)
            lim = min(self.dp_limit[k], rho*P.GRAVITY*max(float(zmax[k]) - mouth_z, 0.))
            self.p[k] = P.P_ATM + float(np.clip(p_new - P.P_ATM, -lim, self.dp_limit[k]))
            dps.append(self.p[k] - P.P_ATM if sealed else 0.)
            zs.append(self.z_neck_np[k] if sealed else 1e3)
        self._t_prev = t
        self.dp.assign(np.array(dps, np.float32))
        self.z_seal.assign(np.array(zs, np.float32))
        self.history.append((round(t, 4), [round(d, 1) for d in dps]))
        if len(self.history) > 200000:
            self.history = self.history[-100000:]

    def force(self, L, dt):
        if not self.enabled or not any(self.sealed):
            return
        W = L.W
        wp.launch(k_gas_force, L.n, inputs=[L.x, L.v_star, L.lid, L.rho0, L.V, dt, self.bodies, W.bpos, W.brot, W.bcom,
                  self.mask, self.m_off, self.m_nr, self.m_nz, self.m_res, self.m_z0, self.dp, self.z_seal, self.z_rim],
                  outputs=[L.bforce, L.btorque], device=L.device)

    def state(self):
        if not self.enabled:
            return {}
        return {name: dict(sealed=bool(self.sealed[k]), dp_Pa=round(float(self.p[k] - P.P_ATM), 1))
                for k, (_, _, name) in enumerate(self.bottles)}
