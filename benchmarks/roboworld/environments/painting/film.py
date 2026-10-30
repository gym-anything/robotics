"""Paint films on surfaces: conserved liquid fields on planar charts, viscoplastic thin-film
transport, structure recovery, sub-grid texture levelling, drying, absorption, lap marks and an exact volume ledger.

A chart is a rectangle on a static surface or on a rigid body (origin, in-plane unit vectors u, v in the body frame;
the paint side is n = u x v). Per cell (i along u, j along v):
    s[k]  wet solids of paint k (m: volume per area)      w   wet water / vehicle (m)
    d[k]  set (dry) solids of paint k (m)                  a   vehicle absorbed into a porous substrate (m)
    lam   thixotropic structure (1 rested, 0 just sheared) tex sub-grid texture amplitude (m), twl its wavelength (m),
    tdir  stroke direction (rad, chart frame)              lap number of laps (fresh paint over a film past its wet edge)
    moved distance the local film has travelled (m; the run/sag tracker of the evaluator)
Wet thickness h = w + sum_k s[k], solids fraction phi = sum_k s[k] / h.

Transport (Balmforth-Craster depth-averaged Bingham flux, donor-cell mobility, CFL substeps, positivity limiter):
    p = rho g_n h - sigma lap(h + D)          (hydrostatic + capillary pressure; D = dry film under it)
    f = rho g_t - grad p                      (driving force per volume on each cell face)
    q = sgn(f) |f| (h_d - h_y)^2 (2 h_d + h_y) / (6 eta_d)  if h_d > h_y = tau_y / |f|, else 0
Drying: vehicle leaves at j (painting.params.evaporation_rate) until phi >= phi_set, when the film sets (its solids
join d, its remaining vehicle is counted as evaporated). Absorption (Washburn): a(t+dt) = sqrt(a^2 + S^2 dt), limited
by the vehicle present and the pore capacity. Texture: Orchard e-folding t_e = 3 eta (wl/2 pi)^4 / (sigma h^3) down to
Hester's residual amplitude A_res = tau_y wl^3 / (8 pi^3 sigma h) (his 4 pi^3 is for the peak-to-trough depth). Laps: paint arriving on a film with phi >= phi_we does not
merge; the old film sets under it (counted in `lap`).

Ledger: every transfer is between named reservoirs; Chart.totals() returns solids/vehicle volumes (m^3) held, absorbed,
evaporated and drained, so the system sum can be checked exactly (painting.ledger).
"""

import math
import numpy as np

from . import params as P

G_VEC = np.array([0.0, 0.0, -P.GRAVITY])


def _quat_rot(q):
    """Rotation matrix from (x, y, z, w)."""
    x, y, z, w = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


class Chart:
    def __init__(self, name, paints, *, origin, u, v, size, cell, body=-1, substrate_color="primer",
                 substrate="sealed", role="target", mask=None, sink_edges=(), sorptivity=None, capacity=None,
                 group=None, visible=True):
        self.name, self.paints, self.body, self.role = name, list(paints), int(body), role
        self.group = group or name
        self.origin = np.asarray(origin, float)
        self.u = np.asarray(u, float)/np.linalg.norm(u)
        self.v = np.asarray(v, float)/np.linalg.norm(v)
        self.n = np.cross(self.u, self.v)
        self.size = (float(size[0]), float(size[1]))
        self.dx = float(cell)
        self.nu = max(1, int(round(self.size[0]/self.dx)))
        self.nv = max(1, int(round(self.size[1]/self.dx)))
        self.area = self.dx*self.dx
        self.visible = visible
        Pn, sh = len(self.paints), (self.nu, self.nv)
        self.s = np.zeros((Pn,) + sh)
        self.w = np.zeros(sh)
        self.d = np.zeros((Pn,) + sh)
        self.a = np.zeros(sh)
        self.lam = np.ones(sh, np.float32)
        self.tex = np.zeros(sh, np.float32)
        self.twl = np.full(sh, P.FILM["tex_wl_roller"], np.float32)
        self.tdir = np.zeros(sh, np.float32)
        self.lap = np.zeros(sh, np.uint8)
        self.moved = np.zeros(sh, np.float32)
        self.touched = np.zeros(sh, np.uint8)        # wet film touched by a robot link or a tool's non-applicator part
        self.marred = np.zeros(sh, np.uint8)         # touched and not worked again since by an applicator (a mark left)
        self.flowed = np.zeros(sh, np.float32)       # net solids (m per cell area) carried into the cell by film flow
        self.mask = np.ones(sh, bool) if mask is None else np.asarray(mask, bool)
        self.flow_block = None          # cells a sheet lying on the chart covers: film cannot flow into them (stencil)
        sub = P.SUBSTRATE[substrate]
        self.substrate = substrate
        self.sorp = np.full(sh, sub["sorptivity"], np.float32) if sorptivity is None else np.asarray(sorptivity, np.float32)
        self.cap = np.full(sh, sub["capacity"], np.float32) if capacity is None else np.asarray(capacity, np.float32)
        self.porous = bool(np.any(self.sorp > 0))
        from .paint import srgb_to_linear
        sc = srgb_to_linear(P.COLORS[substrate_color]) if isinstance(substrate_color, str) else np.asarray(substrate_color, float)
        self.substrate_rgb = np.broadcast_to(sc, sh + (3,)).copy()          # linear RGB per cell
        self.sink_edges = tuple(sink_edges)          # subset of ("u0", "u1", "v0", "v1"): film flowing over drains
        self.evaporated = np.zeros(Pn + 1)           # (solids never evaporate; index -1 = vehicle) m^3
        self.drained = np.zeros(Pn + 1)              # m^3 left over sink edges (the caller collects it)
        self.laps_area = 0.0
        self.bbox = None                             # [i0, i1, j0, j1) of cells holding wet paint or absorbed vehicle
        self.version = 0                             # incremented when the appearance can have changed
        self.events = []
        self.d_base = None                          # dry coats present before the session (set by the scene's prepaint)
        self._sorp_scale = np.array([pt.rec["sorptivity_scale"] for pt in self.paints])
        self._j = np.array([pt.j_evap for pt in self.paints])

    # ------------------------------------------------------------------ geometry
    def frame(self, body_q=None):
        """World origin, u, v, n of the chart (body charts follow their body's pose)."""
        if self.body < 0 or body_q is None:
            return self.origin, self.u, self.v, self.n
        q = body_q[self.body]
        R = _quat_rot(q[3:7])
        p = np.asarray(q[:3], float)
        return p + R @ self.origin, R @ self.u, R @ self.v, R @ self.n

    def local(self, points, body_q=None):
        """(su, sv, height) of world points: metres along u and v from the corner and along n above the surface."""
        o, u, v, n = self.frame(body_q)
        r = np.asarray(points, float) - o
        return r @ u, r @ v, r @ n

    def world(self, su, sv, body_q=None):
        o, u, v, _ = self.frame(body_q)
        return o + np.multiply.outer(su, u) + np.multiply.outer(sv, v)

    def cell_index(self, su, sv):
        i = np.floor(np.asarray(su)/self.dx).astype(int)
        j = np.floor(np.asarray(sv)/self.dx).astype(int)
        inside = (i >= 0) & (i < self.nu) & (j >= 0) & (j < self.nv)
        ic, jc = np.clip(i, 0, self.nu - 1), np.clip(j, 0, self.nv - 1)
        return ic, jc, inside & self.mask[ic, jc]

    # ------------------------------------------------------------------ fields
    def h(self, sl=(slice(None), slice(None))):
        return self.w[sl] + self.s[(slice(None),) + sl].sum(axis=0)

    def dry_total(self, sl=(slice(None), slice(None))):
        return self.d[(slice(None),) + sl].sum(axis=0)

    def projected_dft(self):
        """Dry film thickness each cell will have once set: dry solids + wet solids (m)."""
        return self.d.sum(axis=0) + self.s.sum(axis=0)

    def totals(self):
        """Volumes (m^3): wet solids per paint, wet vehicle, dry solids per paint, absorbed vehicle, evaporated vehicle,
        drained solids per paint and vehicle."""
        A = self.area
        return dict(wet_solids=self.s.sum(axis=(1, 2))*A, wet_vehicle=float(self.w.sum())*A,
                    dry_solids=self.d.sum(axis=(1, 2))*A, absorbed=float(self.a.sum())*A,
                    evaporated=float(self.evaporated[-1]), drained_solids=self.drained[:-1].copy(),
                    drained_vehicle=float(self.drained[-1]))

    def _touch(self, i, j):
        i, j = np.asarray(i), np.asarray(j)
        if i.size == 0:
            return
        b = [int(i.min()), int(i.max()) + 1, int(j.min()), int(j.max()) + 1]
        if self.bbox is None:
            self.bbox = b
        else:
            self.bbox = [min(self.bbox[0], b[0]), max(self.bbox[1], b[1]), min(self.bbox[2], b[2]), max(self.bbox[3], b[3])]
        self.version += 1

    # ------------------------------------------------------------------ transfers (applicators, spray, drips)
    def add(self, i, j, solids, vehicle, *, texture=None, shear=True, now=0.0):
        """Deposit liquid paint: solids (P, N) and vehicle (N) volumes per cell AREA (m) at cells (i, j). Paint landing
        on a film past its wet edge does not merge: the old film sets first (a lap). texture: (amplitude fraction of the
        new wet thickness, wavelength m, direction rad). A call with a stroke texture is an applicator working the
        cells (brush, roller, spray; a skin's film split and a drop carry none): it clears their `marred` flag where it
        lays real paint (>= 5 um wet per cell; a brush with no free paint lays none)."""
        i, j = np.asarray(i, int).ravel(), np.asarray(j, int).ravel()
        solids = np.asarray(solids, float).reshape(len(self.paints), -1)
        vehicle = np.asarray(vehicle, float).ravel()
        if i.size == 0:
            return
        if np.any(solids < -1e-18) or np.any(vehicle < -1e-18):
            raise FloatingPointError(f"{self.name}: negative deposit")
        ok = self.mask[i, j]
        if not np.all(ok):
            raise ValueError(f"{self.name}: deposit on a masked cell (the caller must route it elsewhere)")
        # wet-edge rule on the cells as they are before the deposit
        h0 = self.w[i, j] + self.s[:, i, j].sum(axis=0)
        s0 = self.s[:, i, j].sum(axis=0)
        phi0 = np.where(h0 > 0, s0/np.maximum(h0, 1e-30), 0.0)
        phi_we = np.array([pt.rec["phi_we"] for pt in self.paints])
        dom = np.argmax(self.s[:, i, j], axis=0) if len(self.paints) > 1 else np.zeros(i.size, int)
        lapped = (h0 > P.FILM["h_min"]) & (phi0 >= phi_we[dom]) & ((solids.sum(axis=0) + vehicle) > 0)
        if np.any(lapped):
            li, lj = np.unique(np.stack([i[lapped], j[lapped]]), axis=1)
            self._set_cells(li, lj)                 # the old film sets under the new: no merge, whatever its thickness
            # a lap MARK is the doubled film over a set coat: counted where the old film carries at least
            # FILM['lap_visible'] of solids, so nanometre fringes at stroke edges do not count
            # (the coats that were on the wall before the session -- a deep colour's dry first coat -- are not the old
            # film of a lap: d_base; with them counted every fringe over the first coat would be a "visible lap")
            base = self.d_base[li, lj] if self.d_base is not None else 0.0
            vis = (self.s[:, li, lj].sum(axis=0) + self.d[:, li, lj].sum(axis=0) - base) >= P.FILM["lap_visible"]
            li, lj = li[vis], lj[vis]
            new = self.lap[li, lj] == 0
            self.laps_area += float(new.sum())*self.area
            self.lap[li, lj] = np.minimum(self.lap[li, lj].astype(int) + 1, 255).astype(np.uint8)
            if len(li):
                self.events.append(dict(kind="lap", t=float(now), cells=int(len(li))))
        for k in range(len(self.paints)):
            np.add.at(self.s[k], (i, j), solids[k])
        np.add.at(self.w, (i, j), vehicle)
        if shear:
            self.lam[i, j] = P.FILM["struct_after_shear"]
        if texture is not None:
            frac, wl, ang = texture
            hn = self.w[i, j] + self.s[:, i, j].sum(axis=0)
            self.tex[i, j] = np.maximum(self.tex[i, j], (frac*hn).astype(np.float32))
            self.twl[i, j] = wl
            self.tdir[i, j] = ang
            # a mark is repaired where the applicator laid real paint on it (>= 5 um wet), wet film or set
            real = (solids.sum(axis=0) + vehicle) >= 5e-6
            self.marred[i[real], j[real]] = 0
        self._touch(i, j)

    def take(self, i, j, volume):
        """Remove up to `volume` (m, per cell area) of wet liquid at each cell, with the cell's composition. Returns
        (solids (P, N), vehicle (N)) actually removed (per cell area)."""
        i, j = np.asarray(i, int).ravel(), np.asarray(j, int).ravel()
        volume = np.broadcast_to(np.asarray(volume, float), i.shape).copy()
        if i.size == 0:
            return np.zeros((len(self.paints), 0)), np.zeros(0)
        # duplicates: process unique cells with summed requests
        key = i*self.nv + j
        uk, inv = np.unique(key, return_inverse=True)
        req = np.zeros(len(uk)); np.add.at(req, inv, volume)
        ui, uj = uk//self.nv, uk % self.nv
        h = self.w[ui, uj] + self.s[:, ui, uj].sum(axis=0)
        f = np.where(h > 0, np.minimum(1.0, np.maximum(req, 0)/np.maximum(h, 1e-30)), 0.0)
        rs = self.s[:, ui, uj]*f
        rw = self.w[ui, uj]*f
        self.s[:, ui, uj] -= rs
        self.w[ui, uj] -= rw
        self._take_flowed(ui, uj, f)
        # give each request its share of its cell's removal
        share = np.where(req[inv] > 0, volume/np.maximum(req[inv], 1e-30), 0.0)
        self.version += 1
        return rs[:, inv]*share, rw[inv]*share

    def shear(self, i, j):
        """An applicator working the film without adding to it (a brush or roller taking up excess): a mark there is
        worked out (`marred` cleared)."""
        self.lam[np.asarray(i, int), np.asarray(j, int)] = P.FILM["struct_after_shear"]
        self.marred[np.asarray(i, int), np.asarray(j, int)] = 0

    def _set_cells(self, i, j):
        """Set (dry) the wet film at cells: solids join the dry stack, remaining vehicle evaporates."""
        self.d[:, i, j] += self.s[:, i, j]
        self.evaporated[-1] += float(self.w[i, j].sum())*self.area
        self.s[:, i, j] = 0.0
        self.w[i, j] = 0.0
        self.lam[i, j] = 1.0

    # ------------------------------------------------------------------ evolution
    def window(self, margin=3):
        if self.bbox is None:
            return None
        i0, i1, j0, j1 = self.bbox
        return (max(0, i0 - margin), min(self.nu, i1 + margin), max(0, j0 - margin), min(self.nv, j1 + margin))

    def _rheology(self, s, h, phi, lam, g_t):
        """Solids-weighted mixture of the paints' viscosity and yield stress at the cells. The viscosity is taken at the
        film's own wall shear rate rho g_t h / eta (two fixed-point passes from the zero-shear value)."""
        single = len(self.paints) == 1
        tot = np.maximum(s.sum(axis=0), 1e-30)
        eta = np.zeros(h.shape); ty = np.zeros(h.shape); rho = np.zeros(h.shape); sig = np.zeros(h.shape)
        lam = lam.astype(float)
        for k, pt in enumerate(self.paints):
            r = pt.rec
            c = 1.0 if single else s[k]/tot
            x = 1.0 - np.minimum(phi, r["phi_m"]*0.999)/r["phi_m"]
            kd = ((1.0 - r["phi0"]/r["phi_m"])/x)**2
            e0 = r["eta0_min"] + (r["eta0_max"] - r["eta0_min"])*lam
            if g_t > 0 and r["n"] > 0:
                base = e0
                for _ in range(2):          # gd -> eta(gd) increases monotonically toward the fixed point from below
                    gd = pt.rho*g_t*h/(kd*base)
                    base = r["eta_inf"] + (e0 - r["eta_inf"])/(1.0 + (gd/r["gamma_c"])**r["n"])
            else:
                base = e0
            eta += c*kd*base
            ty += c*kd*r["tau_y_max"]*lam
            rho += c*pt.rho
            sig += c*pt.sigma
        return eta, ty, rho, sig

    DRY_EVERY = 0.5          # s: drying/absorption/setting are integrated exactly over this interval (slow processes)
    FULL_EVERY = 5.0         # s: the whole wet window is flowed at least this often (catches slow creep of rested films)

    def mobility(self, sl, g_t, g_n):
        """Cells whose film can move: structure still recovering (lam < 0.9) or a load above the yield stress
        (gravity along the surface, or hydrostatic head on an up-facing film); rested films below yield are frozen."""
        s, w, lam = self.s[(slice(None),) + sl], self.w[sl], self.lam[sl]
        h = w + s.sum(axis=0)
        wet = h > P.FILM["h_min"]
        if not np.any(wet):
            return wet
        r = self.paints[0].rec
        phi = np.where(wet, s.sum(axis=0)/np.maximum(h, 1e-30), r["phi0"])
        x = 1.0 - np.minimum(phi, r["phi_m"]*0.999)/r["phi_m"]
        ty = ((1.0 - r["phi0"]/r["phi_m"])/x)**2*r["tau_y_max"]*lam
        load = self.paints[0].rho*(g_t + max(g_n, 0.0))*h
        return wet & ((lam < 0.9) | (load > 0.9*ty))

    def advance(self, dt, body_q=None, now=0.0):
        """Evolve the film over dt seconds: flow (CFL substeps) on the mobile part, drying/absorption every DRY_EVERY s,
        the whole wet window every FULL_EVERY s."""
        win = self.window()
        if win is None:
            return dict(substeps=0)
        _, u, v, n = self.frame(body_q)
        g_u, g_v, g_n = float(G_VEC @ u), float(G_VEC @ v), float(-(G_VEC @ n))
        g_t = math.hypot(g_u, g_v)
        i0, i1, j0, j1 = win
        sl = (slice(i0, i1), slice(j0, j1))
        self._full_acc = getattr(self, "_full_acc", 0.0) + dt
        self._dry_acc = getattr(self, "_dry_acc", 0.0) + dt
        info = dict(substeps=0, limited=0)
        if self._full_acc >= self.FULL_EVERY:
            info = self._flow(sl, dt, g_u, g_v, g_n)
            self._full_acc = 0.0
            self.mobile = True
        else:
            mob = self.mobility(sl, g_t, g_n)
            self.mobile = bool(np.any(mob))
            if self.mobile:
                ii, jj = np.nonzero(mob)
                m = 4
                fs = (slice(max(0, i0 + int(ii.min()) - m), min(self.nu, i0 + int(ii.max()) + 1 + m)),
                      slice(max(0, j0 + int(jj.min()) - m), min(self.nv, j0 + int(jj.max()) + 1 + m)))
                info = self._flow(fs, dt, g_u, g_v, g_n)
        if self._dry_acc >= self.DRY_EVERY:
            self._dry(sl, self._dry_acc)
            self._dry_acc = 0.0
            h = self.h(sl)
            wet = (h > P.FILM["h_min"]) | (self.a[sl] > 0)
            if np.any(wet):
                ii, jj = np.nonzero(wet)
                self.bbox = [i0 + int(ii.min()), i0 + int(ii.max()) + 1, j0 + int(jj.min()), j0 + int(jj.max()) + 1]
            else:
                self.bbox = None
        self.version += 1
        return info

    def corner_cells(self):
        """Deprecated: chart boundaries alone do not establish an adjoining face."""
        return np.zeros_like(self.mask)

    def bead_into_corner(self, i, j, reach_m=4e-3):
        """Compatibility no-op for old probes. No unmodelled meniscus transfer.

        The former per-deposit half-transfer had no elapsed time, adjoining-face
        geometry or material/rate basis. Brush contact and ordinary film flow remain;
        the historical bead-on E8 corner result is not current validation.
        """
        return 0.0

    def take_frac(self, i, j, frac):
        """Remove the fraction frac of the WET film (solids per paint and vehicle) at cells (i, j): what a body pressed
        into wet paint carries away (the dry stack stays). Returns the volumes (m^3): solids (P,), vehicle."""
        i, j = np.asarray(i, int).ravel(), np.asarray(j, int).ravel()
        if i.size == 0:
            return np.zeros(len(self.paints)), 0.0
        i, j = np.unique(np.stack([i, j]), axis=1)
        ds = self.s[:, i, j]*frac
        dv = self.w[i, j]*frac
        self.s[:, i, j] -= ds
        self.w[i, j] -= dv
        self._take_flowed(i, j, frac)
        self.version += 1
        return ds.sum(axis=1)*self.area, float(dv.sum())*self.area

    def _take_flowed(self, i, j, f):
        """Wet film carried away takes its share of the paint that flowed into the cell."""
        fl = self.flowed[i, j]
        self.flowed[i, j] = np.where(fl > 0, fl*(1.0 - f), fl)

    def wet_max(self):
        """Largest wet film (m) in the wet window (0 without one)."""
        win = self.window()
        if win is None:
            return 0.0
        sl = (slice(win[0], win[1]), slice(win[2], win[3]))
        return float(self.h(sl).max()) if self.h(sl).size else 0.0

    def solids_window(self):
        """(slices, wet + dry solids) over the wet window: invariant under drying and setting, changed only by flow."""
        win = self.window()
        if win is None:
            return None, None
        sl = (slice(win[0], win[1]), slice(win[2], win[3]))
        return sl, self.s[(slice(None),) + sl].sum(axis=0) + self.d[(slice(None),) + sl].sum(axis=0)

    def dry_only(self, dt):
        """Drying, absorption and setting over dt with no flow (the settle rule's fast path once a film has stopped
        moving: drying is integrated exactly for any dt, and a drying film's viscosity and yield stress only rise)."""
        win = self.window()
        if win is None:
            return
        i0, i1, j0, j1 = win
        sl = (slice(i0, i1), slice(j0, j1))
        self._dry(sl, dt)
        h = self.h(sl)
        wet = (h > P.FILM["h_min"]) | (self.a[sl] > 0)
        if np.any(wet):
            ii, jj = np.nonzero(wet)
            self.bbox = [i0 + int(ii.min()), i0 + int(ii.max()) + 1, j0 + int(jj.min()), j0 + int(jj.max()) + 1]
        else:
            self.bbox = None
        self.version += 1

    def _flow(self, sl, dt_total, g_u, g_v, g_n):
        s, w = self.s[(slice(None),) + sl], self.w[sl]
        lam, mask = self.lam[sl], self.mask[sl]
        if self.flow_block is not None:
            mask = mask & ~self.flow_block[sl]
        D = self.d[(slice(None),) + sl].sum(axis=0)
        dx = self.dx
        Pn = len(self.paints)
        ni, nj = w.shape
        g_t = math.hypot(g_u, g_v)
        # faces blocked by masked cells; chart edges are closed except sink edges
        open_x = mask[1:, :] & mask[:-1, :]
        open_y = mask[:, 1:] & mask[:, :-1]
        t, steps, limited = 0.0, 0, 0
        cfl, hmin = P.FILM["cfl"], P.FILM["h_min"]
        edge = dict(u0=sl[0].start == 0, u1=sl[0].stop == self.nu, v0=sl[1].start == 0, v1=sl[1].stop == self.nv)
        while t < dt_total - 1e-12:
            h = w + s.sum(axis=0)
            wet = h > hmin
            if not np.any(wet):
                break
            phi = np.where(wet, s.sum(axis=0)/np.maximum(h, 1e-30), 0.0)
            eta, ty, rho, sig = self._rheology(s, h, phi, lam, g_t)
            z = h + D
            zp = np.pad(z, 1, mode="edge")
            mp = np.pad(mask, 1, mode="constant", constant_values=False)
            c = zp[1:-1, 1:-1]
            nb = [np.where(mp[2:, 1:-1], zp[2:, 1:-1], c), np.where(mp[:-2, 1:-1], zp[:-2, 1:-1], c),
                  np.where(mp[1:-1, 2:], zp[1:-1, 2:], c), np.where(mp[1:-1, :-2], zp[1:-1, :-2], c)]
            lap = (nb[0] + nb[1] + nb[2] + nb[3] - 4*c)/(dx*dx)
            p = rho*g_n*h - sig*lap
            # x faces (between i and i+1)
            fx = 0.5*(rho[1:] + rho[:-1])*g_u - (p[1:] - p[:-1])/dx
            fy = 0.5*(rho[:, 1:] + rho[:, :-1])*g_v - (p[:, 1:] - p[:, :-1])/dx
            qx = self._flux(fx, h[:-1], h[1:], eta[:-1], eta[1:], ty[:-1], ty[1:]) * open_x
            qy = self._flux(fy, h[:, :-1], h[:, 1:], eta[:, :-1], eta[:, 1:], ty[:, :-1], ty[:, 1:]) * open_y
            # stable step: capillary (4th order), hydrostatic (2nd order) and gravity-advection rates
            mob = np.where(wet, h**3/(3*np.maximum(eta, 1e-6)), 0.0)
            rate = (32*np.max(sig*mob)/dx**4 + (4*np.max(rho*max(g_n, 0.0)*mob)/dx**2)
                    + 2*np.max(np.where(wet, rho*g_t*h*h/np.maximum(eta, 1e-6), 0.0))/dx)
            dt = min(dt_total - t, cfl/rate if rate > 0 else dt_total)
            if (dt_total - t)/max(dt, 1e-12) > P.FILM["max_substeps"] - steps:
                dt = (dt_total - t)/max(1, P.FILM["max_substeps"] - steps)
                limited += 1
            # sink edges: outflow only, driven by gravity along the edge normal (exterior dry, p = 0)
            out_edge = {}
            for e, (sel, gcomp, sign) in dict(u0=((0, slice(None)), g_u, -1), u1=((-1, slice(None)), g_u, +1),
                                              v0=((slice(None), 0), g_v, -1), v1=((slice(None), -1), g_v, +1)).items():
                if e in self.sink_edges and edge[e]:
                    # gravity along the outward edge normal plus the hydrostatic head against the dry exterior
                    f = sign*rho[sel]*gcomp + np.maximum(rho[sel]*g_n*h[sel], 0.0)/(0.5*dx)
                    hd, ed, td = h[sel], eta[sel], ty[sel]
                    hy = np.where(f > 0, td/np.maximum(f, 1e-30), np.inf)
                    out_edge[e] = np.where((f > 0) & (hd > hy), f*(hd - hy)**2*(2*hd + hy)/(6*np.maximum(ed, 1e-6)), 0.0) \
                        * mask[sel]
            # positivity limiter: a cell cannot lose more than it holds in one substep
            outflow = np.zeros_like(h)
            outflow[:-1] += np.maximum(qx, 0); outflow[1:] += np.maximum(-qx, 0)
            outflow[:, :-1] += np.maximum(qy, 0); outflow[:, 1:] += np.maximum(-qy, 0)
            for e, q in out_edge.items():
                outflow[dict(u0=(0, slice(None)), u1=(-1, slice(None)), v0=(slice(None), 0), v1=(slice(None), -1))[e]] += q
            outflow *= dt/dx
            scale = np.where(outflow > h, h/np.maximum(outflow, 1e-30), 1.0)
            if np.any(scale < 1.0):
                limited += 1
            # donor scaling and composition
            frac = np.concatenate([s, w[None]], axis=0)/np.maximum(h, 1e-30)      # (P+1, ni, nj)
            vx = qx*dt/dx
            vx = np.where(vx > 0, vx*scale[:-1], vx*scale[1:])
            vy = qy*dt/dx
            vy = np.where(vy > 0, vy*scale[:, :-1], vy*scale[:, 1:])
            mx = np.where(vx > 0, frac[:, :-1]*vx, frac[:, 1:]*vx)                # (P+1, ni-1, nj) volume moved +x
            my = np.where(vy > 0, frac[:, :, :-1]*vy, frac[:, :, 1:]*vy)
            delta = np.zeros_like(frac)
            delta[:, :-1] -= mx; delta[:, 1:] += mx
            delta[:, :, :-1] -= my; delta[:, :, 1:] += my
            for e, q in out_edge.items():
                sel = dict(u0=(0, slice(None)), u1=(-1, slice(None)), v0=(slice(None), 0), v1=(slice(None), -1))[e]
                vol = q*dt/dx*scale[sel]
                moved = frac[(slice(None),) + sel]*vol
                delta[(slice(None),) + sel] -= moved
                self.drained += moved.sum(axis=1)*self.area
            # run / sag tracker: fraction of the local film that left the cell this substep times the cell size
            leave = np.zeros_like(h)
            leave[:-1] += np.maximum(vx, 0); leave[1:] += np.maximum(-vx, 0)
            leave[:, :-1] += np.maximum(vy, 0); leave[:, 1:] += np.maximum(-vy, 0)
            self.moved[sl] += np.where(wet, leave/np.maximum(h, 1e-30)*dx, 0.0).astype(np.float32)
            self.flowed[sl] += delta[:Pn].sum(axis=0).astype(np.float32)
            s += delta[:Pn]
            w += delta[Pn]
            neg = (s < 0).any(axis=0) | (w < 0)
            if np.any(neg):
                worst = float(min(s.min(), w.min()))
                if worst < -1e-15:
                    raise FloatingPointError(f"{self.name}: negative film {worst:.3e} m after a transport substep")
                np.maximum(s, 0.0, out=s); np.maximum(w, 0.0, out=w)
            # structure recovers toward rest where wet
            lam[...] = np.where(wet, 1.0 - (1.0 - lam)*np.exp(-dt/self._t_r()), 1.0).astype(np.float32)
            t += dt
            steps += 1
        if steps:
            h = w + s.sum(axis=0)
            phi = np.where(h > hmin, s.sum(axis=0)/np.maximum(h, 1e-30), 0.0)
            _, ty, _, sig = self._rheology(s, h, phi, lam, 0.0)
            self._level_texture(sl, h, ty, sig, phi, dt_total)
        return dict(substeps=steps, limited=limited)

    def _t_r(self):
        return self.paints[0].rec["t_r"]

    @staticmethod
    def _flux(f, hl, hr, el, er, tl, tr):
        pos = f > 0
        hd = np.where(pos, hl, hr); ed = np.where(pos, el, er); td = np.where(pos, tl, tr)
        af = np.abs(f)
        hy = td/np.maximum(af, 1e-30)
        q = af*np.maximum(hd - hy, 0.0)**2*(2*hd + hy)/(6*np.maximum(ed, 1e-6))
        return np.where(hd > hy, np.sign(f)*q, 0.0)

    def _level_texture(self, sl, h, ty, sig, phi, dt):
        tex = self.tex[sl]
        wet = (h > 1e-7) & (tex > 0)
        if not np.any(wet):
            return
        wl = self.twl[sl][wet].astype(float)
        hh = h[wet]
        # levelling shear rates are tiny: zero-shear viscosity of the current structure
        lam = self.lam[sl][wet].astype(float)
        eta0 = np.zeros_like(hh)
        s = self.s[(slice(None),) + sl][:, wet]
        tot = np.maximum(s.sum(axis=0), 1e-30)
        for k, pt in enumerate(self.paints):
            eta0 += (s[k]/tot)*pt.eta(0.0, lam, phi[wet])
        te = 3*eta0*(wl/(2*math.pi))**4/(sig[wet]*hh**3)
        ares = ty[wet]*wl**3/(8*math.pi**3*sig[wet]*hh)      # amplitude = half Hester's depth
        A = tex[wet].astype(float)
        A = np.where(A > ares, ares + (A - ares)*np.exp(-dt/np.maximum(te, 1e-9)), A)
        tex[wet] = A.astype(np.float32)

    def _dry(self, sl, dt):
        s, w, a = self.s[(slice(None),) + sl], self.w[sl], self.a[sl]
        h = w + s.sum(axis=0)
        wet = h > P.FILM["h_min"]
        tot = s.sum(axis=0)
        if len(self.paints) == 1:
            j = np.full(h.shape, self._j[0])
        else:
            j = (s*self._j[:, None, None]).sum(axis=0)/np.maximum(tot, 1e-30)
        # evaporation from wet films; from damp porous substrate where no film is left
        ev = np.where(wet, np.minimum(w, j*dt), 0.0)
        w -= ev
        self.evaporated[-1] += float(ev.sum())*self.area
        if self.porous:
            S = self.sorp[sl].astype(float)
            if len(self.paints) == 1:
                S = S*self._sorp_scale[0]
            else:
                S = S*(s*self._sorp_scale[:, None, None]).sum(axis=0)/np.maximum(tot, 1e-30)
            cap = self.cap[sl].astype(float)
            room = np.maximum(cap - a, 0.0)
            take = np.where(wet & (S > 0), np.sqrt(a*a + S*S*dt) - a, 0.0)
            take = np.minimum(np.minimum(take, w), room)
            w -= take
            a += take
            dry_sub = (~wet) & (a > 0)
            if np.any(dry_sub):
                jj = self._j[0]
                ev2 = np.where(dry_sub, np.minimum(a, jj*dt), 0.0)
                a -= ev2
                self.evaporated[-1] += float(ev2.sum())*self.area
        # set: solids fraction reached (or a vanishing film)
        h = w + s.sum(axis=0)
        tot = s.sum(axis=0)
        phi = np.where(h > 0, tot/np.maximum(h, 1e-30), 0.0)
        phi_set = self.paints[0].rec["phi_set"]
        setm = (h > 0) & ((phi >= phi_set) | (h < P.FILM["set_thin"]))
        if np.any(setm):
            ii, jj = np.nonzero(setm)
            self._set_cells(ii + sl[0].start, jj + sl[1].start)

    # ------------------------------------------------------------------ checkpoint
    FIELDS = ("s", "w", "d", "a", "lam", "tex", "twl", "tdir", "lap", "moved", "touched", "marred", "flowed")

    def snapshot(self):
        return dict({k: getattr(self, k).copy() for k in self.FIELDS}, evaporated=self.evaporated.copy(),
                    drained=self.drained.copy(), laps_area=self.laps_area, bbox=None if self.bbox is None else list(self.bbox),
                    events=list(self.events))

    def restore(self, snap):
        for k in self.FIELDS:
            # snapshots made before `marred` existed: every touched cell counts as still marked; before `flowed`: none
            getattr(self, k)[...] = snap[k] if k in snap else (0 if k == "flowed" else snap["touched"])
        self.evaporated[:] = snap["evaporated"]
        self.drained[:] = snap["drained"]
        self.laps_area = snap["laps_area"]
        self.bbox = None if snap["bbox"] is None else list(snap["bbox"])
        self.events = list(snap["events"])
        self.version += 1

    # ------------------------------------------------------------------ appearance
    def reflectance(self, sl=(slice(None), slice(None))):
        from .paint import stack_reflectance
        return stack_reflectance(self.paints, self.d[(slice(None),) + sl], self.s[(slice(None),) + sl], self.w[sl],
                                 self.substrate_rgb[sl])

    def roughness(self):
        """Render roughness per cell: wet films glossy, set films at the product's sheen, bare substrate matte."""
        h = self.h()
        D = self.d.sum(axis=0)
        pt = self.paints[0].rec
        r = np.full(h.shape, 0.85, np.float32)
        r = np.where(D > 2e-6, pt["gloss_dry"], r)
        wetfrac = np.clip(h/20e-6, 0, 1)
        r = r*(1 - wetfrac) + pt["gloss_wet"]*wetfrac
        r = np.where(self.lap > 0, np.minimum(r + 0.08, 1.0), r)
        return r.astype(np.float32)
