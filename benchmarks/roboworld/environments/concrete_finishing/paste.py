"""The wet materials as layered 2.5-D surfaces.

A Region is one work surface (the pad, the floor patch, the block infill, the control joint, a wallboard patch, the
tub, the hawk, the mortar board) on a grid of square cells in its own frame: origin o, in-plane axes U (i) and V (j),
outward normal N of the rigid base. Per cell it holds the rigid base height b (along N; never changes), the bulk and
surface-layer thicknesses tb and ts (the free surface is h = b + tb + ts + bl, bl a blister), the bleed-water film w,
water trapped under a sealed skin q, and the surface states: aggregate exposure E, densification D, sealing S, water
worked into the top 6 mm dwc, tool work density W, texture amplitude a and direction th, the placement time t0, and
flags (paste may lie here; the cell is a finished neighbouring surface, where paste is smear; worked while wet; joint
wall contact for sealant).

What moves the material:
  * gravity (relax): depth-integrated viscoplastic flux of a two-layer Bingham sheet; with q = int_0^t zeta gdot dzeta
    and gdot = (G zeta - tau_y(zeta))+ / mu(zeta), G = rho |g_n grad h - g_par|, each layer gives
    (1/mu)[G (b^3 - a^3)/3 - tau (b^2 - a^2)/2] over its yielded depth [a, b] (one layer: Liu & Mei's
    G Y^2 (3h - Y)/(6 mu)); nothing moves where no depth is above its yield stress;
  * a tool sole (contact): what stands above the sole is cut into the tool's roll, the roll fills what lies below the
    sole where the sole bears, the roll's excess spills at the blade's ends; the paste's support of the
    sole is its bearing capacity, N tau per unit area of the cells that reach the sole (Prandtl/Green), with the
    thin-layer squeeze of a weak surface layer over a stronger bulk (coupling.py turns it into the support plate);
  * the material clock (slow_step): strength from age (materials.py), bleeding into the film or under a sealed skin,
    evaporation of the film, blisters where trapped water collects.

All fields are float32 numpy arrays (nu, nv); nothing here knows about the rigid world.
"""

import math
import numpy as np
from . import params as P
from .materials import evaporation

RHO_W = 1000.0
G = 9.81
SUB_TARGET = 40              # relax: substeps per call before the flux limiter takes over
CUT = 3.83                   # critical height of a vertical cut, H_c = CUT tau/(rho g) (Chen 1975 upper bound; memory)
FREE = 2.0                   # a slender feature with free faces on both sides (a fin, a column, taller than it is wide)
                             # stands while rho g H < FREE tau: Tresca's unconfined compressive strength 2 tau (derived)


def green(h):
    """Green's (1954) bearing factor of a strip under combined load, h = interface shear / yield stress (opened)."""
    h = np.clip(h, 0.0, 1.0)
    return 1.0 + math.pi/2 + np.arccos(h) + np.sqrt(1.0 - h*h)


class Region:
    """One work surface. Fields are set by layout.py; units SI."""

    FIELDS = ("b", "tb", "ts", "bl", "w", "q", "E", "D", "S", "dwc", "W", "a", "th", "t0", "seal_t", "wet_worked",
              "contact_l", "contact_r", "compact", "tau_w", "late", "worked", "reserve", "seal_age", "skin_torn")

    def __init__(self, name, material, origin, U, V, cell, shape, *, paste_mask=None, ref_mask=None, temp_c=21.1,
                 air="still", texture_kind=0):
        self.name, self.material = name, material
        self.o = np.asarray(origin, float)
        self.U = np.asarray(U, float)/np.linalg.norm(U)
        self.V = np.asarray(V, float)/np.linalg.norm(V)
        self.N = np.cross(self.U, self.V)
        self.R = np.column_stack([self.U, self.V, self.N])           # region -> world rotation
        self.cell = float(cell)
        self.nu, self.nv = shape
        for f in self.FIELDS:
            # thicknesses in float64: a float32 depth of 89 mm has a 7.5 nm step, and ten thousand small deposits on a
            # cell lose a measurable volume (B4: 62 mm3 of 99 L in 3.7 s); surface states are float32
            setattr(self, f, np.zeros(shape, np.float64 if f in ("b", "tb", "ts", "bl") else np.float32))
        self.th[:] = 0.0
        self.seal_t[:] = -1.0e9                                       # never worked
        self.seal_age[:] = -1.0e9
        self.active = np.ones(shape, bool)
        self.catches = {}
        self.paste_mask = np.ones(shape, bool) if paste_mask is None else paste_mask
        self.ref_mask = np.zeros(shape, bool) if ref_mask is None else ref_mask
        self.kind = np.full(shape, texture_kind, np.int8)
        self.temp, self.air = float(temp_c), air
        gvec = np.array([0.0, 0.0, -G])
        self.g_n = float(-gvec @ self.N)                              # gravity pulling toward the base
        gp = gvec - (gvec @ self.N)*self.N
        self.g_par = np.array([gp @ self.U, gp @ self.V])            # in-plane gravity (u, v)
        self.clock = 0.0                                             # s since the episode began (region clock)
        # the concrete at the site's temperature, the room's air at its own (20 C at a 21.1 C site, shifted with it)
        self.evap = evaporation(temp_c, P.AIR[air][0] + (temp_c - 21.1), *P.AIR[air][1:]) if air in P.AIR else 0.0   # kg/m2/h
        self.u = (np.arange(self.nu) + 0.5)*self.cell
        self.v = (np.arange(self.nv) + 0.5)*self.cell
        self.relax_substeps = 0
        self.relax_capped = 0
        self.gate_calls = 0
        self.dirty = True            # the surface changed since gravity last found it at rest (yield stresses only
                                     # grow with age, so a surface at rest stays at rest until something changes it)
        self.blisters = 0
        self.events = []                                             # (time, kind, details) of material faults/flags
        self.volume0 = None
        self._legacy_off_m3 = 0.0    # old snapshots only; new deposits are spatial catch tiles

    @property
    def off_m3(self):
        return self._legacy_off_m3 + sum(r.volume() for r in self.catches.values())

    @off_m3.setter
    def off_m3(self, value):
        self._legacy_off_m3 = float(value)

    # ------------------------------------------------------------------ geometry
    @property
    def t(self):
        return self.tb + self.ts

    @property
    def h(self):
        return self.b + self.tb + self.ts + self.bl

    def world(self, u, v, n):
        return self.o + np.multiply.outer(u, self.U) + np.multiply.outer(v, self.V) + np.multiply.outer(n, self.N)

    def to_region(self, p):
        d = np.asarray(p, float) - self.o
        return d @ self.R

    def extent(self):
        return self.nu*self.cell, self.nv*self.cell

    def age(self, sl=None):
        return self.clock - (self.t0 if sl is None else self.t0[sl])

    def h_at(self, sl):
        """Free surface on a slice only (h builds the whole field)."""
        return self.b[sl] + self.tb[sl] + self.ts[sl] + self.bl[sl]

    def support_height(self, sl=np.s_[:, :]):
        """Solid contact envelope differs from fibre volume in an open mesh.

        The blade bridges the 0.5 mm fibre tops while compound occupies the
        openings. b contains only the homogenized fibre volume (0.15 mm).
        """
        if hasattr(self, "mesh"):
            return np.where(self.mesh[sl], P.MESH_T, self.b[sl])
        return self.b[sl]

    def t_at(self, sl):
        return self.tb[sl] + self.ts[sl]

    def volume(self):
        # Blisters lift the envelope with trapped water/void; they do not create paste.
        return float(self.t.sum())*self.cell**2

    # ------------------------------------------------------------------ strength
    def tau_b(self, sl=np.s_[:, :]):
        return self.material.tau_bulk(self.age(sl))

    def tau_s(self, sl=np.s_[:, :]):
        fabric = self.compact[sl] if self.material.name == "mortar" else self.D[sl]
        return self.material.tau_surf(self.age(sl), fabric)

    def capacity(self, sl, width, length=None, depth=0.0):
        """Bearing pressure (Pa) of the cells in `sl` under a contact patch of narrow width B = `width` and length L
        (m), its sole `depth` D below the paste around it: failure through the bulk N tau_b with Skempton's factor
        N = N_c (1 + 0.2 B/L)(1 + 0.2 min(D/B, 2.5)) (5.14 for a strip on the surface, 9.25 for a square or round
        punch 2.5 widths deep: the pocket penetrometer's factor 9), or squeeze-out of a weak surface layer of thickness
        ts between the sole and the stronger bulk, tau_s (2 + B/(4 ts)) (Prandtl's thin layer),
        whichever is lower."""
        tb, ts = self.tau_b(sl), self.tau_s(sl)
        thin = np.maximum(self.ts[sl], 1e-7)          # Prandtl's squeeze grows without bound as the layer thins
        squeeze = ts*(2.0 + width/(4.0*thin))
        shape = 1.0 + 0.2*min(width/length, 1.0) if length else 1.0
        deep = P.NC*shape*(1.0 + 0.2*min(max(depth, 0.0)/max(width, 1e-6), 2.5))*tb
        pressure = np.minimum(deep, np.where(self.ts[sl] > 1e-5, squeeze, deep))
        if self.material.name == "sealant":
            pressure += P.NC*np.maximum(ts-tb, 0.)
        elif self.material.name == "mortar":
            # A consolidated surface contributes only over its finite depth.
            # Bulk strength remains age controlled; drag also reads tau_s.
            pressure += P.NC*np.maximum(ts-tb, 0.)*min(P.MORTAR_LAYER/max(width, 1e-6), 1.)
        return pressure

    # ------------------------------------------------------------------ the material clock
    def slow_step(self, dt):
        """Advance the region's clock by dt: bleeding, evaporation, trapped water and blisters, gravity flow."""
        self.clock += dt
        m = self.material
        if getattr(m, "bleeds", False):
            r = m.bleed_rate(self.age())*self.paste_mask                     # kg/m2/s
            sealed = np.clip(self.S, 0.0, 1.0)
            self.w += (dt*(r*(1.0 - sealed))/RHO_W).astype(np.float32)
            self.q += (dt*r*sealed/RHO_W).astype(np.float32)
            self.w -= np.float32(dt*self.evap/3600.0/RHO_W)
            np.maximum(self.w, 0.0, out=self.w)
            self._blisters()
        elif self.evap > 0:
            self.w -= np.float32(dt*self.evap/3600.0/RHO_W)
            np.maximum(self.w, 0.0, out=self.w)
        self.relax(dt)
        for r in self.catches.values():
            r.slow_step(dt)

    def _blisters(self):
        """Trapped bleed water raises a blister where it exceeds BLISTER_Q (CIP 13: 12-100 mm, a 3 mm skin):
        the trapped volume above the threshold lifts the surface, spread over ~15 mm (a phenomenological rule)."""
        over = self.q - np.float32(P.BLISTER_Q)
        if (over <= 0).all():
            return
        lift = np.where(over > 0, over*4.0, 0.0)
        k = max(1, int(round(0.0075/self.cell)))
        for ax in (0, 1):                                            # a box blur of +-7.5 mm
            acc = np.zeros_like(lift)
            for s in range(-k, k + 1):
                acc += np.roll(lift, s, axis=ax)
            lift = acc/(2*k + 1)
        new = np.minimum(np.maximum(self.bl, lift), 0.003)
        if float((new - self.bl).max()) > 1e-5:
            before = int((self.bl > 3e-4).sum())
            self.bl[:] = new
            self.dirty = True
            after = int((self.bl > 3e-4).sum())
            if after > before and before == 0:
                self.events.append((self.clock, "blister", dict(cells=after)))

    # ------------------------------------------------------------------ gravity flow
    def _bound(self, h, t, taus, taub):
        """An upper bound of every face's yield ratio (see _gate): when it is below 1 nothing can move. With l the width
        of a feature, |dh/l| min(t, l/CUT) <= dh/CUT and the surface layer's drop is clipped to its own thickness."""
        rho = self.material.rho
        on = t > 1e-5
        if not on.any():
            return 0.0
        hr = float(h[on].max() - h.min())
        gu = np.abs(np.diff(h, axis=0)).max(initial=0.0)/self.cell
        gv = np.abs(np.diff(h, axis=1)).max(initial=0.0)/self.cell
        slope = max(gu, gv)
        gp = float(np.hypot(*self.g_par))
        t_max, ts_max = float(t.max()), float(self.ts.max())
        bulk = rho*(min(self.g_n*hr/CUT, self.g_n*slope*t_max) + gp*t_max)/float(taub[on].min())
        surf = rho*(self.g_n/CUT + gp)*ts_max/float(taus[on].min())
        free = rho*self.g_n*min(hr, t_max)/(FREE*float(taub[on].min()))      # a free-standing feature (_free_standing)
        return max(bulk, surf, free)

    def _tiles(self, h, t, taus, taub, T=32):
        """Cell range (i0, i1), (j0, j1) of the TxT tiles whose neighbourhood could yield (the bound of _bound taken
        per tile: the relief of the tile and its eight neighbours over CUT for widths inside that neighbourhood, the
        whole region's relief over the neighbourhood's width for wider ones), or None."""
        from scipy.ndimage import maximum_filter, minimum_filter
        rho = self.material.rho
        nu, nv = h.shape
        pu, pv = (-nu) % T, (-nv) % T
        on = t > 1e-5
        def red(a, fill, f):
            a = np.pad(a, ((0, pu), (0, pv)), constant_values=fill)
            return f(a.reshape(a.shape[0]//T, T, a.shape[1]//T, T), axis=(1, 3))
        hmax = red(np.where(on, h, -np.inf), -np.inf, np.max)
        hmin = red(h, np.inf, np.min)
        tmax = red(t, 0.0, np.max)
        tsmax = red(self.ts.astype(float), 0.0, np.max)
        tbmin = red(np.where(on, taub, np.inf), np.inf, np.min)
        tsmin = red(np.where(on, taus, np.inf), np.inf, np.min)
        gu = np.abs(np.diff(h, axis=0, append=h[-1:, :]))/self.cell
        gv = np.abs(np.diff(h, axis=1, append=h[:, -1:]))/self.cell
        slope = red(np.maximum(gu, gv), 0.0, np.max)
        hr_n = maximum_filter(hmax, size=3, mode="nearest") - minimum_filter(hmin, size=3, mode="nearest")
        hr_n = np.where(np.isfinite(hr_n), np.maximum(hr_n, 0.0), 0.0)
        slope_n = maximum_filter(slope, size=3, mode="nearest")
        tmax_n = maximum_filter(tmax, size=3, mode="nearest")
        hr_all = float(h[on].max() - h.min()) if on.any() else 0.0
        gp = float(np.hypot(*self.g_par))
        g = self.g_n
        bulk = rho*(np.maximum(np.minimum(g*hr_n/CUT, g*slope_n*tmax_n), g*hr_all/(T*self.cell)*tmax_n) + gp*tmax_n)/tbmin
        surf = rho*(g/CUT + gp)*tsmax/tsmin
        # a free-standing feature narrower than a tile stands in the neighbourhood's relief; a wider one, slender only if
        # taller than a tile is wide, in the region's (_free_standing)
        wide = np.minimum(hr_all, tmax_n)
        free = rho*g*np.maximum(np.minimum(hr_n, tmax_n), np.where(wide > T*self.cell, wide, 0.0))/(FREE*tbmin)
        act = np.isfinite(bulk) & ((bulk >= 1.0) | (surf >= 1.0) | (free >= 1.0)) & (tmax > 1e-5)
        if not act.any():
            return None
        ti, tj = np.nonzero(act)
        return (int(ti.min())*T, min((int(ti.max()) + 1)*T, nu)), (int(tj.min())*T, min((int(tj.max()) + 1)*T, nv))

    def _gate(self, h, t, ts, taus, taub):
        """Faces where the paste yields under its own weight, per axis (face i between cells i and i+1).

        A surface feature of width l mobilises at most a depth l/CUT of the paste below it (CUT = 3.83: the critical
        height of a vertical cut in a cohesive material, H_c = 3.83 tau/(rho g)); for features much wider than the
        paste is deep this is the thin-layer criterion rho g |grad h| t > tau, for a step of height dh it is
        rho g dh > 3.83 tau. At widths l = (2k - 1) cell (k = 1, 2, 4, ...) the face's driving stress gradient is
        G_l = rho (g_n dh_l / l - g_par) and the face is open if |G_l| min(t_donor, l/CUT) exceeds the donor's bulk
        yield stress at any width. The surface layer is judged the same way with its own thickness and yield stress,
        its drop clipped to its thickness (a weak skin on top of a cliff does not pour over it unless the cliff fails).
        The flux through an open face is the thin-layer flux of the cell-scale gradient (relax): the gate decides
        whether the paste yields, the monotone thin-layer law how fast it moves."""
        rho = self.material.rho
        t_max = float(t.max())
        K = 1
        while (2*K - 1)*self.cell < CUT*t_max and K < 512:
            K *= 2
        out = []
        for ax in (0, 1):
            n = h.shape[ax]
            pad = [(0, 0), (0, 0)]
            P2 = 2*K
            pad[ax] = (P2, P2)
            hp = np.pad(h, pad, mode="edge")
            take = lambda a, lo: np.take(a, np.arange(lo, lo + n), axis=ax)
            t_n, ts_n = np.roll(t, -1, axis=ax), np.roll(ts, -1, axis=ax)
            tb_n, tS_n = np.roll(taub, -1, axis=ax), np.roll(taus, -1, axis=ax)
            open_ = np.zeros(h.shape, bool)
            k = 1
            while k <= K:
                m = 2*k - 1
                ell = m*self.cell
                # across the face: centred on it, and ending on either side of it (a narrow pillar is a cut seen
                # from its own flank)
                for hi, lo in ((P2 + k, P2 - k + 1), (P2 + 1, P2 + 1 - m), (P2 + m, P2)):
                    dh = take(hp, hi) - take(hp, lo)
                    Gn = rho*(self.g_n*dh/ell - self.g_par[ax])
                    up = Gn > 0                                      # h rises toward +axis: the donor is cell i+1
                    td, tsd = np.where(up, t_n, t), np.where(up, ts_n, ts)
                    tbd, tsy = np.where(up, tb_n, taub), np.where(up, tS_n, taus)
                    open_ |= np.abs(Gn)*np.minimum(td, ell/CUT) > tbd
                    Gs = rho*(self.g_n*np.clip(dh, -tsd, tsd)/ell - self.g_par[ax])
                    open_ |= np.abs(Gs)*np.minimum(tsd, ell/CUT) > tsy
                k *= 2
            # a narrow feature with free faces on both sides is not a cut into a half-space: it fails as a free-standing
            # wall (FREE) and its faces down to either side open. Judged only by the half-space cut, a 14 mm fin of
            # 5-minute concrete set down by a fingertip stood 189 mm proud of the pad, 1.3 times its free-standing
            # limit, and never relaxed
            free = self._free_standing(h, t, taub, ax)
            if free is not None:
                dh1 = np.roll(h, -1, axis=ax) - h
                open_ |= np.where(dh1 > 0, np.roll(free, -1, axis=ax), free) & (dh1 != 0)
            sl = [slice(None), slice(None)]
            sl[ax] = -1
            open_[tuple(sl)] = False
            out.append(open_)
        return out, K

    def _free_standing(self, h, t, taub, ax):
        """Cells of a slender feature along axis `ax` that stands above what lies on either side of it by more than a
        free-standing wall of the paste carries: narrower than L cells (L = 3, 7, 15, ...) and at least as tall as L
        cells are wide, with rho g_n min(top-hat, t) > FREE tau_b, the top-hat being h less its grey opening of length L
        (how far a feature narrower than L stands above its surroundings). Wider features are cuts into a half-space,
        judged by the gate's cut criterion. None when no column of the paste could fail so (the usual case: nothing to
        compute)."""
        from scipy.ndimage import maximum_filter1d, minimum_filter1d
        g, rho = self.g_n, self.material.rho
        if g <= 0:
            return None
        cand = (t > 1e-5) & (rho*g*t > FREE*taub)
        if not cand.any():
            return None
        n = h.shape[ax]
        widest = float(t[cand].max())                            # no feature is slender wider than the paste is deep
        out = np.zeros(h.shape, bool)
        L = 3
        while L <= n:
            opened = maximum_filter1d(minimum_filter1d(h, L, axis=ax, mode="nearest"), L, axis=ax, mode="nearest")
            top = h - opened
            fails = cand & (top >= L*self.cell) & (rho*g*np.minimum(top, t) > FREE*taub)
            if fails.any():
                # the feature fails through its base: all of it flows, its flanks with its crest (a peaked fin whose
                # crest alone counted only shuffled paste between its top cells)
                out |= (maximum_filter1d(fails.astype(np.uint8), L, axis=ax, mode="nearest") > 0) & (top > 1e-4) & (t > 1e-5)
            if L*self.cell >= widest:
                break
            L = 2*L + 1
        return out if out.any() else None

    def _layer_flux(self, Gm, t, ts, taus, taub):
        """Depth-integrated flux magnitude (m2/s) of a two-layer Bingham sheet under driving stress gradient Gm (Pa/m):
        with depth z from the free surface, a layer between depths [top, bottom] whose yield depth is a = max(tau/G,
        top) carries (1/mu)[G (b^3 - a^3)/3 - tau (b^2 - a^2)/2] where b > a (one layer: Liu and Mei)."""
        m = self.material
        G = np.maximum(Gm, 1e-12)
        out = np.zeros_like(Gm)
        for tau, top, bot, mu in ((taus, 0.0, ts, m.mu_surf), (taub, ts, t, m.mu_bulk)):
            a = np.minimum(np.maximum(tau/G, top), bot)
            out += np.where(bot > a, (G*(bot**3 - a**3)/3.0 - tau*(bot**2 - a**2)/2.0)/mu, 0.0)
        return np.maximum(out, 0.0)

    def relax(self, dt, max_sub=400):
        """Explicit conservative two-layer thin-layer flow over dt through the faces where the paste yields (_gate),
        with as many sub-steps as stability needs (capped), on the window of cells around the open faces."""
        rho = self.material.rho
        if not self.dirty:
            return
        t_all = self.t
        if not (t_all > 1e-5).any():
            self.dirty = False
            return
        taus_f, taub_f = self.tau_s().astype(float), self.tau_b().astype(float)
        if self._bound(self.h.astype(float), t_all.astype(float), taus_f, taub_f) < 1.0:
            self.dirty = False
            return
        h = (self.b + self.tb + self.ts + self.bl).astype(float)
        tiles = self._tiles(h, t_all.astype(float), taus_f, taub_f)
        if tiles is None:
            self.dirty = False
            return
        self.gate_calls += 1
        (a0, a1), (c0, c1) = tiles
        K = 1
        while (2*K - 1)*self.cell < CUT*float(t_all.max()) and K < 512:
            K *= 2
        m = 2*K + 2
        pre = (slice(max(a0 - m, 0), min(a1 + m, self.nu)), slice(max(c0 - m, 0), min(c1 + m, self.nv)))
        gates, _ = self._gate(h[pre], t_all[pre].astype(float), self.ts[pre].astype(float), taus_f[pre], taub_f[pre])
        hit = gates[0] | gates[1]
        if not hit.any():
            self.dirty = False
            return
        ii, jj = np.nonzero(hit)
        ii = ii + pre[0].start; jj = jj + pre[1].start
        m = K + 2
        win = (slice(max(int(ii.min()) - m, 0), min(int(ii.max()) + m + 1, self.nu)),
               slice(max(int(jj.min()) - m, 0), min(int(jj.max()) + m + 1, self.nv)))
        b, bl = self.b[win].astype(float), self.bl[win].astype(float)
        tb, ts = self.tb[win].astype(float), self.ts[win].astype(float)
        taus, taub = taus_f[win], taub_f[win]
        edge = [np.zeros(tb.shape, bool), np.zeros(tb.shape, bool)]   # faces on the window's far side carry nothing
        edge[0][-1, :] = True; edge[1][:, -1] = True
        done, sub = 0.0, 0
        while done < dt - 1e-12 and sub < max_sub:
            h = b + tb + ts + bl
            t = tb + ts
            gates, _ = self._gate(h, t, ts, taus, taub)
            if not (gates[0].any() or gates[1].any()):
                break
            fl, lim = {}, {}
            dts = [dt - done]
            for ax in (0, 1):
                other = 1 - ax
                grad = (np.roll(h, -1, axis=ax) - h)/self.cell
                gc = (np.roll(h, -1, axis=other) - np.roll(h, 1, axis=other))/(2*self.cell)
                gt = 0.5*(gc + np.roll(gc, -1, axis=ax))
                Gn = rho*(self.g_n*grad - self.g_par[ax])            # along the face normal
                Gt = rho*(self.g_n*gt - self.g_par[other])
                Gm = np.hypot(Gn, Gt)
                up = Gn > 0                                          # h rises toward +axis: paste flows from cell i+1
                td = np.where(up, np.roll(t, -1, axis=ax), t)
                tsd = np.where(up, np.roll(ts, -1, axis=ax), ts)
                tsu = np.where(up, np.roll(taus, -1, axis=ax), taus)
                tbu = np.where(up, np.roll(taub, -1, axis=ax), taub)
                qm = self._layer_flux(Gm, td, tsd, tsu, tbu)
                qm = np.minimum(qm, td*2.0*np.sqrt(G*td))            # inertia: no faster than a dam-break front (Ritter)
                active = self.active[win]
                qm = np.where(gates[ax] & ~edge[ax] & active & np.roll(active, -1, axis=ax), qm, 0.0)
                fl[ax] = (qm, Gn, Gm, td)
                act = qm > 0
                if act.any():
                    dts.append(0.25*self.cell*float(np.min(td[act]/qm[act])))
                    if self.g_n > 0:
                        Deff = qm[act]/np.maximum(Gm[act]/(rho*self.g_n), 1e-9)
                        dts.append(0.2*self.cell**2/float(Deff.max()))
            if len(dts) == 1:
                break                                                # open faces, but no cell-scale gradient drives them
            # a substep no shorter than dt/SUB_TARGET; where the viscous rate is faster than that resolves, each face
            # moves at most an eighth of its driving height difference and a quarter of its donor's depth per substep
            h_dt = min(dt - done, max(min(dts), dt/SUB_TARGET))
            div = np.zeros_like(h)
            fluxes = {}
            for ax in (0, 1):
                qm, Gn, Gm, td = fl[ax]
                cap = 0.25*td*self.cell/h_dt
                if self.g_n > 0:
                    cap = np.minimum(cap, 0.125*np.abs(Gn)/(rho*self.g_n)*self.cell*self.cell/h_dt)
                qm = np.minimum(qm, cap)
                qn = -qm*Gn/np.maximum(Gm, 1e-12)                    # flux through the face toward +axis
                fluxes[ax] = qn
                div += (qn - np.roll(qn, 1, axis=ax))/self.cell
            dtn = -h_dt*div                                          # thickness change
            if float(np.abs(dtn).max()) < 1e-9:
                break                                                # less than a nanometre a substep: at rest
            # Advect donor histories with the very same conservative face fluxes.
            # A receiving cell cannot confer its age or wet-work flags on a load.
            from .transfer import HISTORY
            total = np.maximum(t+dtn, 0.)
            for field in (*HISTORY, "t0", "ts", "reserve", "w", "q"):
                extensive = field in ("ts", "reserve", "w", "q")
                value = ts if field == "ts" else getattr(self, field)[win].astype(float)
                concentration = np.divide(value, t, out=np.zeros_like(t), where=t > 1e-15) if extensive else value
                amount = value.copy() if extensive else value*t
                for ax, qn in fluxes.items():
                    donor = np.where(qn >= 0, concentration, np.roll(concentration, -1, axis=ax))
                    face = qn*donor
                    amount -= h_dt*(face-np.roll(face, 1, axis=ax))/self.cell
                result = np.maximum(amount, 0.) if extensive else np.divide(amount, total, out=value.copy(), where=total > 1e-15)
                if field == "ts":
                    ts = np.minimum(result, total)
                else:
                    getattr(self, field)[win] = result
            tb = np.maximum(total-ts, 0.)
            done += h_dt
            sub += 1
        self.tb[win] = tb
        self.ts[win] = ts
        self.relax_substeps += sub
        if sub == 0 or (done < dt - 1e-12 and sub < max_sub):
            self.dirty = False                                       # the gate closed everywhere: at rest
        if done < dt - 1e-12 and sub >= max_sub:
            self.relax_capped += 1

    # ------------------------------------------------------------------ snapshots
    def snapshot(self):
        from .transfer import snapshot_catches
        return dict(clock=self.clock, fields={f: getattr(self, f).copy() for f in self.FIELDS},
                    kind=self.kind.copy(), blisters=self.blisters, events=list(self.events), off_m3=self._legacy_off_m3,
                    catches=snapshot_catches(self))

    def restore(self, snap):
        self.clock = snap["clock"]
        for f, v in snap["fields"].items():
            getattr(self, f)[:] = v
        self.kind[:] = snap["kind"]
        self.blisters = snap["blisters"]
        self.off_m3 = snap.get("off_m3", 0.0)
        self.events = list(snap["events"])
        from .transfer import restore_catches
        restore_catches(self, snap.get("catches", {}))
        self.dirty = True

    def summary(self):
        t = self.t
        on = self.paste_mask
        age = self.age()
        return dict(name=self.name, material=self.material.name, clock_s=round(self.clock, 3),
                    age_min=round(float(np.median(age[on]))/60.0, 2) if on.any() else None,
                    volume_l=round(self.volume()*1e3, 4), film_max_mm=round(float(self.w.max())*1e3, 4),
                    sheen_share=round(float((self.w[on] > P.SHEEN_W).mean()) if on.any() else 0.0, 4),
                    tau_bulk_kpa=round(float(np.median(self.tau_b()[on]))/1e3, 3) if on.any() else None,
                    blister_cells=int((self.bl > 3e-4).sum()), smear_cells=int(((t > 2e-4) & self.ref_mask).sum()))


# ------------------------------------------------------------------ tool soles
class Sole:
    """The working surface of a tool or fingertip in its own frame (x along the blade's length, y across it, z out of
    the sole toward the handle): the sole lies at z = f(x, y) over |x| <= hx, |y| <= hy (f = 0 for a flat blade)."""

    def __init__(self, kind, hx, hy, *, radius=0.0, lip=0.0, lip_side=1, lip_w=0.0015, long_axis=0, lift=0.003,
                 grid=None, res=0.0):
        self.kind, self.hx, self.hy = kind, float(hx), float(hy)
        self.radius, self.lip, self.lip_side, self.lip_w = float(radius), float(lip), lip_side, float(lip_w)
        self.long_axis = long_axis
        self.lift = float(lift)
        self.flex = 0.0
        self.grid, self.res = grid, float(res)          # "profile": the body's depth behind the face (nan: no body)

    def f(self, x, y):
        if self.kind == "profile":                                   # a tool's side or back as its parts shape it
            n, m = self.grid.shape
            i = np.clip(np.floor((np.asarray(x) + self.hx)/self.res).astype(int), 0, n - 1)
            j = np.clip(np.floor((np.asarray(y) + self.hy)/self.res).astype(int), 0, m - 1)
            return self.grid[i, j]
        if self.kind == "rod":                                       # a jointer's rod: radius along y across the joint
            r = self.radius
            yy = np.clip(np.abs(y), 0.0, r*0.999)
            return r - np.sqrt(r*r - yy*yy)
        if self.kind == "tip":                                       # rounded sealant tool tip (radius across y)
            r = self.radius
            yy = np.clip(np.abs(y), 0.0, r*0.999)
            return r - np.sqrt(r*r - yy*yy)
        if self.kind == "edger":
            # flat sole; toward the lip side (y -> +hy) a quarter-round of radius r goes down to the lip, which runs
            # down the form face to depth `lip`
            d = self.hy - self.lip_side*y                            # distance in from the lip's face
            r = self.radius
            z = np.zeros_like(x, dtype=float)
            inr = (d >= 0) & (d < r)
            z = np.where(inr, -(r - np.sqrt(np.maximum(r*r - (r - d)**2, 0.0))), z)
            z = np.where(d < self.lip_w, -self.lip, z)
            return z
        # Euler-Bernoulli cantilever, clamped at the handle (+y), loaded at -y.
        xi = np.clip((self.hy-y)/(2*self.hy), 0., 1.)
        return self.flex*xi*xi*(3.-xi)/2.


def footprint(region, sole, p_world, R_world, clip=True):
    """Cells under a sole at world pose (p, R): (slice into the region, mask of cells inside the footprint, sole height
    s along the region's normal at each cell (nan outside), local x and y). The sole's height at a cell is that of
    the sole point above the cell centre, to first order in the sole's tilt from the region's plane.

    clip=False: the whole footprint on the region's lattice, past its edges too (cells i0..i1, j0..j1 of the lattice the
    region is part of); the first item is then the index ranges ((i0, i1), (j0, j1)) and region.last_cover is left
    alone. What a tool lays down is spread over the whole footprint so that an edge cutting it lays no thicker."""
    Rr = region.R.T @ R_world                                        # tool -> region rotation
    pr = region.to_region(p_world)
    corners = np.array([[sx*sole.hx, sy*sole.hy, 0.0] for sx in (-1, 1) for sy in (-1, 1)])
    cr = pr + corners @ Rr.T
    c = region.cell
    if not clip:
        i0, i1 = int(math.floor(cr[:, 0].min()/c)) - 2, int(math.ceil(cr[:, 0].max()/c)) + 2
        j0, j1 = int(math.floor(cr[:, 1].min()/c)) - 2, int(math.ceil(cr[:, 1].max()/c)) + 2
        U, V = np.meshgrid((np.arange(i0, i1) + 0.5)*c, (np.arange(j0, j1) + 0.5)*c, indexing="ij")
        zt = Rr[:, 2]
        if zt[2] < 0.2:
            return None
        rel = np.stack([U - pr[0], V - pr[1], np.full_like(U, 0.0)], -1)
        nn = -(rel @ zt)/zt[2]
        loc = (rel + np.multiply.outer(nn, np.array([0.0, 0.0, 1.0]))) @ Rr
        x, y = loc[..., 0], loc[..., 1]
        cover = np.clip((sole.hx - np.abs(x))/c + 0.5, 0.0, 1.0)*np.clip((sole.hy - np.abs(y))/c + 0.5, 0.0, 1.0)
        inside = ((np.abs(x) <= sole.hx) & (np.abs(y) <= sole.hy)) | (cover > 0.25)
        s = np.where(inside, pr[2] + nn + sole.f(x, y)/zt[2], np.nan)
        return ((i0, i1), (j0, j1)), inside, s, x, y
    i0 = max(0, int(math.floor(cr[:, 0].min()/c)) - 2); i1 = min(region.nu, int(math.ceil(cr[:, 0].max()/c)) + 2)
    j0 = max(0, int(math.floor(cr[:, 1].min()/c)) - 2); j1 = min(region.nv, int(math.ceil(cr[:, 1].max()/c)) + 2)
    if i0 >= i1 or j0 >= j1:
        return None
    U, V = np.meshgrid(region.u[i0:i1], region.v[j0:j1], indexing="ij")
    # cell centre at the sole reference plane's height (n of the tool origin), expressed in the tool frame
    n0 = pr[2]
    rel = np.stack([U - pr[0], V - pr[1], np.full_like(U, 0.0)], -1)
    # move along the region normal to the tool's sole plane (z_tool = 0): nn solves (rel + nn N_r) . z_tool = 0
    zt = Rr[:, 2]                                                    # tool z axis in region coordinates
    if zt[2] < 0.2:
        return None                                                  # the sole is edge-on to this surface or faces away
    nn = -(rel @ zt)/zt[2]
    loc = (rel + np.multiply.outer(nn, np.array([0.0, 0.0, 1.0]))) @ Rr
    x, y = loc[..., 0], loc[..., 1]
    inside = (np.abs(x) <= sole.hx) & (np.abs(y) <= sole.hy)
    # the share of each cell under the sole (a cell-sized square about its centre against the sole's rectangle, in the
    # sole's frame): a small sole's bearing area is then right on average wherever it lands on the grid
    c2 = 0.5*c
    region.last_cover = (np.clip((sole.hx - np.abs(x))/c + 0.5, 0.0, 1.0)*np.clip((sole.hy - np.abs(y))/c + 0.5, 0.0, 1.0))
    inside = inside | (region.last_cover > 0.25)
    inside &= region.active[i0:i1, j0:j1]
    fz = sole.f(x, y)
    # the sole point (x, y, fz) in region coordinates, its normal coordinate
    s = n0 + nn + fz/zt[2]
    s = np.where(inside, s, np.nan)
    return (slice(i0, i1), slice(j0, j1)), inside, s, x, y
