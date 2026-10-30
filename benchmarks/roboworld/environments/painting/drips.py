"""Ballistic paint: drips from overloaded tools, spills over container rims.

Drops are packets (position, velocity, solids per paint, vehicle) falling under gravity (mm drops: drag neglected over
room heights, estimate) until they reach a surface of the contact primitives; they land as a round spot on that face's
film chart (spot thickness 0.4 mm, estimate from low-Reynolds spreading of a ~0.05 mL drop of a yield-stress paint) or
in the surface's sink. Small emissions from the same source accumulate into a drop of Tate volume
V_d = 0.6 * 2 pi r sigma / (rho g) (Harkins-Brown factor 0.6, r = 1.5 mm tip radius; estimate) before detaching.
"""

import math
import numpy as np

from . import params as P

SPOT_H = 0.4e-3


class Drips:
    def __init__(self, system):
        self.sys = system
        Pn = len(system.paints)
        self.pos = np.zeros((0, 3)); self.vel = np.zeros((0, 3)); self.s = np.zeros((0, Pn)); self.v = np.zeros(0)
        self.pending = {}
        pt = system.paints[0]
        self.v_drop = 0.6*2*math.pi*1.5e-3*pt.sigma/(pt.rho*P.GRAVITY)
        self.landed = 0
        self.max_speed = 0.0

    def load(self):
        s = self.s.sum(axis=0) if len(self.s) else np.zeros(len(self.sys.paints))
        for p in self.pending.values():
            s = s + p[1]
        v = float(self.v.sum()) + sum(p[2] for p in self.pending.values())
        return s, v

    def emit(self, pts, solids, vehicle, accumulate=None, vel=None):
        """Release liquid at points (solids (N, P), vehicle (N), m^3). With `accumulate` (a source key) the emissions
        collect into a pendant volume released as one drop when it reaches the Tate volume."""
        pts = np.atleast_2d(np.asarray(pts, float))
        solids = np.atleast_2d(np.asarray(solids, float)).reshape(len(pts), -1)
        vehicle = np.asarray(vehicle, float).ravel()
        if accumulate is not None:
            acc = self.pending.get(accumulate)
            tot = solids.sum() + vehicle.sum()
            if acc is None:
                acc = [np.zeros(3), np.zeros(solids.shape[1]), 0.0, 0.0]
            wsum = acc[3] + tot
            acc[0] = (acc[0]*acc[3] + (pts*(solids.sum(axis=1) + vehicle)[:, None]).sum(axis=0))/max(wsum, 1e-30)
            acc[1] = acc[1] + solids.sum(axis=0)
            acc[2] += float(vehicle.sum())
            acc[3] = wsum
            self.pending[accumulate] = acc
            if acc[3] >= self.v_drop:
                self._add(acc[0][None], acc[1][None], np.array([acc[2]]), None)
                del self.pending[accumulate]
            return
        self._add(pts, solids, vehicle, vel)

    def _add(self, pts, solids, vehicle, vel):
        self.pos = np.vstack([self.pos, pts])
        self.vel = np.vstack([self.vel, np.zeros_like(pts) if vel is None else vel])
        self.s = np.vstack([self.s, solids])
        self.v = np.concatenate([self.v, vehicle])

    def step(self, bq, dt):
        if not len(self.pos):
            return
        from .contact import nearest
        g = np.array([0., 0., -P.GRAVITY])
        new_v = self.vel + g*dt
        new_p = self.pos + new_v*dt
        c = new_p.mean(axis=0)
        rad = float(np.linalg.norm(new_p - c, axis=1).max()) + float(np.linalg.norm(new_v, axis=1).max())*dt + 0.01
        prims = self.sys.candidates(("drips", len(self.pos)), c, rad, bq)
        if not prims:
            self.pos, self.vel = new_p, new_v
            return
        d, nrm, idx = nearest(prims, new_p, bq)
        hit = d < 0
        self.max_speed = max(self.max_speed, float(np.linalg.norm(new_v, axis=1).max()))
        if np.any(hit):
            for k in np.unique(idx[hit]):
                m = hit & (idx == k)
                pr = prims[k]
                for t in np.nonzero(m)[0]:
                    self._land(pr, new_p[t] - d[t]*nrm[t], nrm[t], self.s[t], self.v[t], bq)
            keep = ~hit
            new_p, new_v = new_p[keep], new_v[keep]
            self.s, self.v = self.s[keep], self.v[keep]
        lost = new_p[:, 2] < -0.5
        if np.any(lost):
            raise FloatingPointError("a paint drop left the room through the floor")
        self.pos, self.vel = new_p, new_v

    def _land(self, prim, pt, n, solids, vehicle, bq):
        sysm = self.sys
        vol = float(solids.sum() + vehicle)
        r = math.sqrt(max(vol, 1e-15)/(math.pi*SPOT_H))
        for cn in prim.charts:
            ch = sysm.charts.get(cn)
            if ch is None:
                continue
            su, sv, hh = ch.local(pt[None], bq)
            if abs(hh[0]) > 0.006:
                continue
            k = max(1, int(math.ceil(r/ch.dx)))
            g = np.arange(-k, k + 1)
            DI, DJ = np.meshgrid(g, g, indexing="ij")
            disk = (DI**2 + DJ**2)*ch.dx**2 <= r*r + 1e-18
            i0, j0 = int(su[0]//ch.dx), int(sv[0]//ch.dx)
            I, J = i0 + DI[disk], j0 + DJ[disk]
            ok = (I >= 0) & (I < ch.nu) & (J >= 0) & (J < ch.nv)
            ok[ok] &= ch.mask[I[ok], J[ok]]
            if not np.any(ok):
                continue
            I, J = I[ok], J[ok]
            nI = len(I)
            ch.add(I, J, np.repeat(solids[:, None], nI, axis=1)/nI/ch.area, np.full(nI, vehicle/nI/ch.area), shear=True)
            ch.events.append(dict(kind="drop", volume_m3=vol, cells=nI))
            self.landed += 1
            return
        if prim.sink in sysm.reservoirs:
            sysm.reservoirs[prim.sink].add(solids, vehicle)
        else:
            sysm.to_sink(prim.sink, solids, vehicle, points=np.asarray(pt, float)[None])
        self.landed += 1

    def snapshot(self):
        return dict(pos=self.pos.copy(), vel=self.vel.copy(), s=self.s.copy(), v=self.v.copy(),
                    pending={k: [a.copy() if hasattr(a, "copy") else a for a in v] for k, v in self.pending.items()},
                    landed=self.landed)

    def restore(self, s):
        self.pos, self.vel, self.s, self.v = s["pos"].copy(), s["vel"].copy(), s["s"].copy(), s["v"].copy()
        self.pending = {k: [a.copy() if hasattr(a, "copy") else a for a in v] for k, v in s["pending"].items()}
        self.landed = s["landed"]
