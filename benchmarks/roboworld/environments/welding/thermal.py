"""Explicit finite-volume enthalpy heat conduction on a graded rectilinear voxel grid (Warp, GPU).

  dH/dt = div(k(T) grad T) + q,   H(T) = int rho c dT + rho L f_l(T)

* Cells carry a material id (0 = air: no conduction, no heat capacity) and a body id (which rigid part they belong to;
  WELD = solidified/liquid weld metal belongs to both). Faces between two metal cells conduct with the harmonic mean of
  the cells' conductances; across the faying surface of two *different, unfused* bodies the face conductance is the
  contact conductance h_cc in series with the two half cells. Once either cell has been liquid (peak liquid fraction
  >= 0.5) or one is weld metal, the face conducts fully (metallurgical continuity).
* Faces toward air or the grid boundary lose heat by convection and radiation; faces flagged in `sink` (e.g. the base
  plate's underside on the fixture table) conduct to an ambient-temperature sink through h_cc.
* Enthalpy -> temperature by monotone table inversion per material (64-entry search on a 5 C table); the liquid
  conductivity is multiplied by the pool-convection factor F_MARANGONI (params) above the liquidus, linearly through
  the mushy range.
* Sources: arrays of W per cell filled by the heat-source kernels each step (arc surface flux, droplet volume source).
* Ledger (float64): input, convective + radiative loss, sink loss, evaporation cap; stored = sum H V.

Index layout: flat arrays, index = (i*ny + j)*nz + k, x (i) the slowest axis, so one x slice is one contiguous block
(the pool model updates slices by contiguous copies).
"""

import math

import numpy as np
import warp as wp

from . import params as PR

AIR, STEEL, WELD = 0, 1, 2          # material ids (WELD: filler metal, steel properties)
NMAT = 8                            # table slots (3-7 unused)
BODY_NONE, BODY_WELD = 0, 255       # body ids: 0 air, 255 weld metal, others the scene's parts (1..)
T_TAB0, T_TAB_DT, T_TAB_N = 0., 5., 641   # table: 0..3200 C


# ------------------------------------------------------------------------------------------------ material tables
def _steel_cp(t):
    """params.STEEL_CP: EN 1993-1-2 eq. 3.2 (the formula is the parameter)."""
    assert PR.v(PR.STEEL_CP) == "EN1993"
    t = np.asarray(t, float)
    c = np.empty_like(t)
    a = t < 600.
    c[a] = 425 + .773*t[a] - 1.69e-3*t[a]**2 + 2.22e-6*t[a]**3
    b = (t >= 600.) & (t < 735.)
    c[b] = 666 + 13002/(738 - t[b])
    d = (t >= 735.) & (t < 900.)
    c[d] = 545 + 17820/(t[d] - 731)
    c[t >= 900.] = 650.
    return c


def _steel_k(t):
    """params.STEEL_K: EN 1993-1-2 eq. 3.3, 31 W/m K to the liquidus (the formula is the parameter)."""
    assert PR.v(PR.STEEL_K) == "EN1993"
    t = np.asarray(t, float)
    return np.where(t < 800., 54 - .0333*t, np.where(t < 1200., 27.3, 27.3 + (31. - 27.3)*np.clip((t - 1200.)/300., 0, 1)))


def material_tables():
    """(Hsens [NMAT, N] J/m3, K [NMAT, N] W/m K, solidus [NMAT], liquidus [NMAT], rho*L [NMAT]) on the uniform T table.
    The latent heat is added analytically (h_to_t): H = Hsens(T) + rho L f_l, f_l linear between solidus and liquidus,
    so a narrow (eutectic) melting range is exact whatever the table spacing."""
    T = T_TAB0 + T_TAB_DT*np.arange(T_TAB_N)
    H = np.zeros((NMAT, T_TAB_N))
    K = np.zeros((NMAT, T_TAB_N))
    sol = np.full(NMAT, 1e9)
    liq = np.full(NMAT, 1e9 + 1)
    lat = np.zeros(NMAT)
    fm = PR.v(PR.F_MARANGONI)

    def build(m, rho, cp_fn, k_fn, ts, tl, latent, cp_liq=None):
        cp = cp_fn(T)
        if cp_liq is not None:
            cp = np.where(T > tl, cp_liq, cp)
        fl = np.clip((T - ts)/(tl - ts), 0., 1.)
        h = np.concatenate([[0.], np.cumsum(.5*(cp[1:] + cp[:-1])*np.diff(T))])*rho      # sensible enthalpy only
        k = k_fn(T)*(1. + (fm - 1.)*fl)
        H[m], K[m], sol[m], liq[m], lat[m] = h, k, ts, tl, rho*latent

    rho = PR.v(PR.STEEL_RHO)
    for m in (STEEL, WELD):
        build(m, rho, _steel_cp, _steel_k, PR.v(PR.STEEL_SOLIDUS), PR.v(PR.STEEL_LIQUIDUS), PR.v(PR.STEEL_LATENT),
              cp_liq=PR.v(PR.STEEL_CP_LIQ))
    return H.astype(np.float32), K.astype(np.float32), sol.astype(np.float32), liq.astype(np.float32), lat


# ------------------------------------------------------------------------------------------------ kernels
@wp.func
def tab_lookup(tab: wp.array2d(dtype=wp.float32), m: int, t: float):
    x = (t - 0.0)/5.0
    x = wp.clamp(x, 0.0, 639.999)
    i = int(x)
    f = x - float(i)
    return tab[m, i]*(1.0 - f) + tab[m, i + 1]*f


@wp.func
def sens_to_t(tabH: wp.array2d(dtype=wp.float32), m: int, h: float):
    if h <= tabH[m, 0]:
        return 0.0
    if h >= tabH[m, 640]:
        return 3200.0
    lo = int(0)
    hi = int(640)
    for _ in range(10):
        mid = (lo + hi)//2
        if tabH[m, mid] <= h:
            lo = mid
        else:
            hi = mid
    d = tabH[m, hi] - tabH[m, lo]
    f = float(0.0)
    if d > 0.0:
        f = (h - tabH[m, lo])/d
    return (float(lo) + f)*5.0


@wp.func
def h_to_t(tabH: wp.array2d(dtype=wp.float32), m: int, h: float, ts: float, tl: float, rl: float):
    """Temperature from total enthalpy: sensible table below the solidus, linear through the latent interval,
    sensible table (minus the latent heat) above the liquidus."""
    hs = tab_lookup(tabH, m, ts)
    hl = tab_lookup(tabH, m, tl) + rl
    if h <= hs:
        return sens_to_t(tabH, m, h)
    if h >= hl:
        return sens_to_t(tabH, m, h - rl)
    return ts + (tl - ts)*(h - hs)/(hl - hs)


@wp.func
def face_g(m0: int, b0: int, f0: float, k0: float, m1: int, b1: int, f1: float, k1: float, d0: float, d1: float,
           area: float, hcc: float):
    """Conductance (W/K) of the face between two metal cells; half-widths d0, d1."""
    r = d0/k0 + d1/k1
    if b0 != b1 and b0 != 255 and b1 != 255 and f0 < 0.5 and f1 < 0.5:
        r = r + 1.0/hcc
    return area/r


@wp.func
def surf_loss(t: float, area: float, h: float, eps: float):
    tk = t + 273.15
    return area*(h*(t - 20.0) + eps*5.670374419e-8*(tk*tk*tk*tk - 293.15*293.15*293.15*293.15))


@wp.kernel
def k_step(nx: int, ny: int, nz: int,
           mat: wp.array(dtype=wp.uint8), body: wp.array(dtype=wp.uint8), sink: wp.array(dtype=wp.uint8),
           T: wp.array(dtype=wp.float32), H: wp.array(dtype=wp.float32), peakf: wp.array(dtype=wp.float32),
           src: wp.array(dtype=wp.float32),
           dx: wp.array(dtype=wp.float32), dy: wp.array(dtype=wp.float32), dz: wp.array(dtype=wp.float32),
           tabK: wp.array2d(dtype=wp.float32), dt: float, h_air: float, eps: float, hcc: float, hsink: float,
           Hout: wp.array(dtype=wp.float32), ledger: wp.array(dtype=wp.float64),
           Hlo: wp.array(dtype=wp.float32), Hlo_out: wp.array(dtype=wp.float32)):
    idx = wp.tid()
    m0 = int(mat[idx])
    if m0 == 0:
        Hout[idx] = 0.0
        Hlo_out[idx] = 0.0
        return
    k = idx % nz
    j = (idx//nz) % ny
    i = idx//(ny*nz)
    t0 = T[idx]
    b0 = int(body[idx])
    f0 = peakf[idx]
    k0 = tab_lookup(tabK, m0, t0)
    ax = dy[j]*dz[k]
    ay = dx[i]*dz[k]
    az = dx[i]*dy[j]
    q = float(0.0)
    loss = float(0.0)
    sk = float(0.0)
    sm = int(sink[idx])
    # six faces: (neighbour index or -1, area, own half width, neighbour half width, sink bit)
    for f in range(6):
        nb = int(-1)
        area = float(0.0)
        d0 = float(0.0)
        d1 = float(0.0)
        if f == 0:
            area = ax
            d0 = 0.5*dx[i]
            if i > 0:
                nb = idx - ny*nz
                d1 = 0.5*dx[i - 1]
        elif f == 1:
            area = ax
            d0 = 0.5*dx[i]
            if i < nx - 1:
                nb = idx + ny*nz
                d1 = 0.5*dx[i + 1]
        elif f == 2:
            area = ay
            d0 = 0.5*dy[j]
            if j > 0:
                nb = idx - nz
                d1 = 0.5*dy[j - 1]
        elif f == 3:
            area = ay
            d0 = 0.5*dy[j]
            if j < ny - 1:
                nb = idx + nz
                d1 = 0.5*dy[j + 1]
        elif f == 4:
            area = az
            d0 = 0.5*dz[k]
            if k > 0:
                nb = idx - 1
                d1 = 0.5*dz[k - 1]
        else:
            area = az
            d0 = 0.5*dz[k]
            if k < nz - 1:
                nb = idx + 1
                d1 = 0.5*dz[k + 1]
        m1 = int(0)
        if nb >= 0:
            m1 = int(mat[nb])
        if m1 != 0:
            t1 = T[nb]
            g = face_g(m0, b0, f0, k0, m1, int(body[nb]), peakf[nb], tab_lookup(tabK, m1, t1), d0, d1, area, hcc)
            q += g*(t1 - t0)
        elif ((sm >> f) & 1) == 1:
            g = area/(d0/k0 + 1.0/hsink)
            s = g*(t0 - 20.0)
            q -= s
            sk += s
        else:
            l = surf_loss(t0, area, h_air, eps)
            q -= l
            loss += l
    vol = dx[i]*dy[j]*dz[k]
    s = src[idx]
    # compensated update: H + Hlo carries the enthalpy to float64-like precision. In float32 alone a cooling cell's
    # change per substep near ambient (a few J/m3 on 1e8 J/m3) rounds systematically and the energy ledger drifts
    # 0.36 % in a 120 s soak (E21); with the remainder it stays at its starting value
    hf = wp.float64(H[idx]) + wp.float64(Hlo[idx]) + wp.float64(dt*(q + s)/vol)
    hn = wp.float32(hf)
    Hout[idx] = hn
    Hlo_out[idx] = wp.float32(hf - wp.float64(hn))
    if s != 0.0:
        wp.atomic_add(ledger, 0, wp.float64(dt*s))
    if loss != 0.0:
        wp.atomic_add(ledger, 1, wp.float64(dt*loss))
    if sk != 0.0:
        wp.atomic_add(ledger, 2, wp.float64(dt*sk))


@wp.kernel
def k_invert(mat: wp.array(dtype=wp.uint8), H: wp.array(dtype=wp.float32), tabH: wp.array2d(dtype=wp.float32),
             sol: wp.array(dtype=wp.float32), liq: wp.array(dtype=wp.float32), rlat: wp.array(dtype=wp.float32),
             tcap: float,
             dx: wp.array(dtype=wp.float32), dy: wp.array(dtype=wp.float32), dz: wp.array(dtype=wp.float32),
             ny: int, nz: int,
             T: wp.array(dtype=wp.float32), fl: wp.array(dtype=wp.float32), peakT: wp.array(dtype=wp.float32),
             peakf: wp.array(dtype=wp.float32), ledger: wp.array(dtype=wp.float64)):
    idx = wp.tid()
    m = int(mat[idx])
    if m == 0:
        T[idx] = 20.0
        fl[idx] = 0.0
        peakT[idx] = 20.0          # air has no thermal history (a removed cell must not stay 'fused')
        peakf[idx] = 0.0
        return
    h = H[idx]
    t = h_to_t(tabH, m, h, sol[m], liq[m], rlat[m])
    if t > tcap:
        # evaporation cap: enthalpy above the boiling point leaves as vapour (booked)
        hc = tab_lookup(tabH, m, tcap) + rlat[m]
        k = idx % nz
        j = (idx//nz) % ny
        i = idx//(ny*nz)
        wp.atomic_add(ledger, 3, wp.float64((h - hc)*dx[i]*dy[j]*dz[k]))
        H[idx] = hc
        t = tcap
    T[idx] = t
    hs = tab_lookup(tabH, m, sol[m])
    f = wp.clamp((h - hs)/(tab_lookup(tabH, m, liq[m]) + rlat[m] - hs), 0.0, 1.0)
    fl[idx] = f
    if t > peakT[idx]:
        peakT[idx] = t
    if f > peakf[idx]:
        peakf[idx] = f


@wp.kernel
def k_stored(mat: wp.array(dtype=wp.uint8), H: wp.array(dtype=wp.float32), dx: wp.array(dtype=wp.float32),
             dy: wp.array(dtype=wp.float32), dz: wp.array(dtype=wp.float32), ny: int, nz: int,
             out: wp.array(dtype=wp.float64), Hlo: wp.array(dtype=wp.float32)):
    idx = wp.tid()
    if int(mat[idx]) == 0:
        return
    k = idx % nz
    j = (idx//nz) % ny
    i = idx//(ny*nz)
    wp.atomic_add(out, 0, (wp.float64(H[idx]) + wp.float64(Hlo[idx]))*wp.float64(dx[i]*dy[j]*dz[k]))


@wp.kernel
def k_gauss_weights(mat: wp.array(dtype=wp.uint8), xc: wp.array(dtype=wp.float32), yc: wp.array(dtype=wp.float32),
                    zc: wp.array(dtype=wp.float32), dx: wp.array(dtype=wp.float32), dy: wp.array(dtype=wp.float32),
                    dz: wp.array(dtype=wp.float32), nx: int, ny: int, nz: int,
                    i0: int, j0: int, k0: int, bj: int, bk: int,
                    p: wp.vec3, axis: wp.vec3, sigma: float, w: wp.array(dtype=wp.float32),
                    wsum: wp.array(dtype=wp.float64)):
    """Surface-flux weights in a box of cells: for every metal cell face toward air whose outward normal faces the
    arc (n . -axis > 0), weight = exp(-r^2 / 2 sigma^2) * (n . -axis) * face area; summed per cell."""
    t = wp.tid()
    ii = t//(bj*bk)
    jj = (t//bk) % bj
    kk = t % bk
    i = i0 + ii
    j = j0 + jj
    k = k0 + kk
    if i < 0 or j < 0 or k < 0 or i >= nx or j >= ny or k >= nz:
        return
    idx = (i*ny + j)*nz + k
    if int(mat[idx]) == 0:
        return
    acc = float(0.0)
    for f in range(6):
        nb = int(-1)
        n = wp.vec3(0.0, 0.0, 0.0)
        c = wp.vec3(xc[i], yc[j], zc[k])
        area = float(0.0)
        if f == 0:
            n = wp.vec3(-1.0, 0.0, 0.0)
            area = dy[j]*dz[k]
            c = c + wp.vec3(-0.5*dx[i], 0.0, 0.0)
            if i > 0:
                nb = idx - ny*nz
        elif f == 1:
            n = wp.vec3(1.0, 0.0, 0.0)
            area = dy[j]*dz[k]
            c = c + wp.vec3(0.5*dx[i], 0.0, 0.0)
            if i < nx - 1:
                nb = idx + ny*nz
        elif f == 2:
            n = wp.vec3(0.0, -1.0, 0.0)
            area = dx[i]*dz[k]
            c = c + wp.vec3(0.0, -0.5*dy[j], 0.0)
            if j > 0:
                nb = idx - nz
        elif f == 3:
            n = wp.vec3(0.0, 1.0, 0.0)
            area = dx[i]*dz[k]
            c = c + wp.vec3(0.0, 0.5*dy[j], 0.0)
            if j < ny - 1:
                nb = idx + nz
        elif f == 4:
            n = wp.vec3(0.0, 0.0, -1.0)
            area = dx[i]*dy[j]
            c = c + wp.vec3(0.0, 0.0, -0.5*dz[k])
            if k > 0:
                nb = idx - 1
        else:
            n = wp.vec3(0.0, 0.0, 1.0)
            area = dx[i]*dy[j]
            c = c + wp.vec3(0.0, 0.0, 0.5*dz[k])
            if k < nz - 1:
                nb = idx + 1
        open_face = True
        if nb >= 0:
            if int(mat[nb]) != 0:
                open_face = False
        if open_face:
            facing = -wp.dot(n, axis)
            if facing > 0.0:
                r = c - p
                acc += wp.exp(-wp.dot(r, r)/(2.0*sigma*sigma))*facing*area
    if acc > 0.0:
        w[idx] = acc
        wp.atomic_add(wsum, 0, wp.float64(acc))


@wp.kernel
def k_cyl_weights(mat: wp.array(dtype=wp.uint8), xc: wp.array(dtype=wp.float32), yc: wp.array(dtype=wp.float32),
                  zc: wp.array(dtype=wp.float32), dx: wp.array(dtype=wp.float32), dy: wp.array(dtype=wp.float32),
                  dz: wp.array(dtype=wp.float32), nx: int, ny: int, nz: int,
                  i0: int, j0: int, k0: int, bj: int, bk: int,
                  p: wp.vec3, axis: wp.vec3, radius: float, depth: float, w: wp.array(dtype=wp.float32),
                  wsum: wp.array(dtype=wp.float64)):
    """Volume weights of metal cells inside a cylinder (start p, direction axis, radius, depth): the spray droplets'
    heat (Kumar & Bhaduri's cylindrical source)."""
    t = wp.tid()
    ii = t//(bj*bk)
    jj = (t//bk) % bj
    kk = t % bk
    i = i0 + ii
    j = j0 + jj
    k = k0 + kk
    if i < 0 or j < 0 or k < 0 or i >= nx or j >= ny or k >= nz:
        return
    idx = (i*ny + j)*nz + k
    if int(mat[idx]) == 0:
        return
    r = wp.vec3(xc[i], yc[j], zc[k]) - p
    s = wp.dot(r, axis)
    if s < 0.0 or s > depth:
        return
    rad = r - s*axis
    if wp.dot(rad, rad) > radius*radius:
        return
    vol = dx[i]*dy[j]*dz[k]
    w[idx] = vol
    wp.atomic_add(wsum, 0, wp.float64(vol))


@wp.kernel
def k_add_weighted(w: wp.array(dtype=wp.float32), scale: float, src: wp.array(dtype=wp.float32)):
    idx = wp.tid()
    if w[idx] > 0.0:
        src[idx] = src[idx] + w[idx]*scale
        w[idx] = 0.0


@wp.kernel
def k_add_scaled(w: wp.array(dtype=wp.float32), scale: float, src: wp.array(dtype=wp.float32)):
    idx = wp.tid()
    v = w[idx]
    if v > 0.0:
        src[idx] = src[idx] + v*scale


@wp.kernel
def k_slice_max(fl: wp.array(dtype=wp.float32), ny: int, nz: int, j0: int, j1: int, k0: int, k1: int,
                out: wp.array(dtype=wp.float32)):
    """Per x slice, the largest liquid fraction inside the y-z box [j0, j1) x [k0, k1)."""
    t = wp.tid()
    bj = j1 - j0
    bk = k1 - k0
    i = t//(bj*bk)
    j = j0 + (t//bk) % bj
    k = k0 + t % bk
    v = fl[(i*ny + j)*nz + k]
    if v > 0.0:
        wp.atomic_max(out, i, v)


@wp.kernel
def k_gather(src: wp.array(dtype=wp.float32), idx: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float32)):
    t = wp.tid()
    out[t] = src[idx[t]]


@wp.kernel
def k_gather_u8(src: wp.array(dtype=wp.uint8), idx: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.uint8)):
    t = wp.tid()
    out[t] = src[idx[t]]


@wp.kernel
def k_scatter(dst: wp.array(dtype=wp.float32), idx: wp.array(dtype=wp.int32), val: wp.array(dtype=wp.float32)):
    t = wp.tid()
    dst[idx[t]] = val[t]


@wp.kernel
def k_scatter_u8(dst: wp.array(dtype=wp.uint8), idx: wp.array(dtype=wp.int32), val: wp.array(dtype=wp.uint8)):
    t = wp.tid()
    dst[idx[t]] = val[t]


# ------------------------------------------------------------------------------------------------ grid
def graded(segments):
    """Cell widths from (length, width) segments with geometric transitions handled by the caller: a list of
    (length_m, cell_m) pieces laid out in order; each piece is divided into round(length/cell) equal cells."""
    out = []
    for length, cell in segments:
        n = max(1, int(round(length/cell)))
        out += [length/n]*n
    return np.asarray(out, np.float64)


def ramp(length, h0, h1, growth=1.2):
    """Cells growing geometrically from h0 toward h1 to fill `length` exactly (last cell adjusted)."""
    cells, s, h = [], 0., h0
    while s < length - 1e-12:
        h = min(h*growth, h1)
        if s + h > length:
            h = length - s
        cells.append(h)
        s += h
    if len(cells) > 1 and cells[-1] < .5*cells[-2]:
        cells[-2] += cells.pop()
    return np.asarray(cells)


class Grid:
    """A rectilinear grid: cell widths along each axis (m) and the world origin of the grid's corner (weld frame)."""

    def __init__(self, dx, dy, dz, origin=(0., 0., 0.)):
        self.dx, self.dy, self.dz = (np.asarray(a, np.float64) for a in (dx, dy, dz))
        self.origin = np.asarray(origin, float)
        self.nx, self.ny, self.nz = len(self.dx), len(self.dy), len(self.dz)
        self.n = self.nx*self.ny*self.nz
        self.xe = self.origin[0] + np.concatenate([[0.], np.cumsum(self.dx)])
        self.ye = self.origin[1] + np.concatenate([[0.], np.cumsum(self.dy)])
        self.ze = self.origin[2] + np.concatenate([[0.], np.cumsum(self.dz)])
        self.xc = .5*(self.xe[1:] + self.xe[:-1])
        self.yc = .5*(self.ye[1:] + self.ye[:-1])
        self.zc = .5*(self.ze[1:] + self.ze[:-1])

    def index(self, i, j, k):
        return (np.asarray(i)*self.ny + np.asarray(j))*self.nz + np.asarray(k)

    def locate(self, p):
        """Cell indices (i, j, k) containing point p (clipped to the grid)."""
        return tuple(int(np.clip(np.searchsorted(e, c) - 1, 0, len(e) - 2)) for e, c in
                     ((self.xe, p[0]), (self.ye, p[1]), (self.ze, p[2])))

    def range_of(self, lo, hi):
        """Index ranges (inclusive start, exclusive end) of cells overlapping the box [lo, hi]."""
        out = []
        for e, a, b in ((self.xe, lo[0], hi[0]), (self.ye, lo[1], hi[1]), (self.ze, lo[2], hi[2])):
            s = int(np.clip(np.searchsorted(e, a) - 1, 0, len(e) - 2))
            t = int(np.clip(np.searchsorted(e, b), 1, len(e) - 1))
            out.append((s, t))
        return out

    def volumes(self):
        return (self.dx[:, None, None]*self.dy[None, :, None]*self.dz[None, None, :]).ravel()


class ThermalSolver:
    """Owns the GPU fields of one grid. Build: set mat/body/sink on the host, then upload()."""

    def __init__(self, grid, device="cuda:0", t_amb=20.):
        self.g = g = grid
        self.device = device
        self.mat = np.zeros(g.n, np.uint8)
        self.body = np.zeros(g.n, np.uint8)
        self.sink = np.zeros(g.n, np.uint8)
        self.t_amb = t_amb
        self.tabH, self.tabK, self.sol, self.liq, self.lat = material_tables()
        self.time = 0.
        self.ledger_host = np.zeros(8)        # input, surface loss, sink loss, evaporation (J)
        # ledger slots: 0 input, 1 surface loss, 2 sink loss, 3 evaporation, 4 deposited metal, 5 removed metal (J),
        # 6 energy of cells taken out of the grid with a part that left its fit-up pose (J), 7 their volume (m3)
        self.uploaded = False

    def box(self, lo, hi, material, body):
        """Fill the cells whose centres lie inside the axis-aligned box [lo, hi] (weld frame, m)."""
        g = self.g
        I = np.nonzero((g.xc >= lo[0]) & (g.xc <= hi[0]))[0]
        J = np.nonzero((g.yc >= lo[1]) & (g.yc <= hi[1]))[0]
        K = np.nonzero((g.zc >= lo[2]) & (g.zc <= hi[2]))[0]
        if len(I) == 0 or len(J) == 0 or len(K) == 0:
            raise ValueError(f"box {lo}..{hi} contains no cell centre")
        idx = g.index(I[:, None, None], J[None, :, None], K[None, None, :]).ravel()
        self.mat[idx] = material
        self.body[idx] = body
        return idx

    def set_sink_face(self, idx, face):
        self.sink[idx] |= np.uint8(1 << face)

    def upload(self):
        g, d = self.g, self.device
        A = lambda a, t=wp.float32: wp.array(np.ascontiguousarray(a), dtype=t, device=d)
        self.w_dx, self.w_dy, self.w_dz = A(g.dx), A(g.dy), A(g.dz)
        self.w_xc, self.w_yc, self.w_zc = A(g.xc), A(g.yc), A(g.zc)
        self.w_mat, self.w_body, self.w_sink = A(self.mat, wp.uint8), A(self.body, wp.uint8), A(self.sink, wp.uint8)
        self.w_tabH = wp.array(self.tabH, dtype=wp.float32, device=d)
        self.w_tabK = wp.array(self.tabK, dtype=wp.float32, device=d)
        self.w_sol, self.w_liq, self.w_rlat = A(self.sol), A(self.liq), A(self.lat.astype(np.float32))
        H0 = np.zeros(g.n, np.float32)
        for m in np.unique(self.mat):
            if m:
                H0[self.mat == m] = np.interp(self.t_amb, T_TAB0 + T_TAB_DT*np.arange(T_TAB_N), self.tabH[m])
        self.w_H, self.w_H2 = A(H0), A(H0)
        self.w_Hlo, self.w_Hlo2 = A(np.zeros_like(H0)), A(np.zeros_like(H0))     # the enthalpy's float32 remainder
        self.w_T = A(np.full(g.n, self.t_amb, np.float32))
        self.w_fl = A(np.zeros(g.n, np.float32))
        self.w_peakT = A(np.full(g.n, self.t_amb, np.float32))
        self.w_peakf = A(np.zeros(g.n, np.float32))
        self.w_src = A(np.zeros(g.n, np.float32))
        self.w_wt = A(np.zeros(g.n, np.float32))
        self.w_wsum = wp.zeros(1, dtype=wp.float64, device=d)
        self.w_ledger = wp.zeros(8, dtype=wp.float64, device=d)
        self.w_store = wp.zeros(1, dtype=wp.float64, device=d)
        self.uploaded = True
        self._invert()
        self.stored0 = self.stored()

    # ---------------------------------------------------------------- stepping
    def stable_dt(self, safety=.9):
        """Explicit limit over the grid's smallest cells and the largest diffusivity any material can reach."""
        g = self.g
        alpha = 0.
        present = [m for m in np.unique(self.mat) if m]
        for m in present:
            T = T_TAB0 + T_TAB_DT*np.arange(T_TAB_N)
            dH = np.maximum(np.gradient(self.tabH[m].astype(float), T), 1.)
            alpha = max(alpha, float(np.max(self.tabK[m]/dH)))
        inv = 1/g.dx.min()**2 + 1/g.dy.min()**2 + 1/g.dz.min()**2
        return safety/(2*alpha*inv)

    def _invert(self):
        g = self.g
        wp.launch(k_invert, dim=g.n, inputs=[self.w_mat, self.w_H, self.w_tabH, self.w_sol, self.w_liq, self.w_rlat,
                                             PR.v(PR.STEEL_T_BOIL), self.w_dx, self.w_dy, self.w_dz, g.ny, g.nz,
                                             self.w_T, self.w_fl, self.w_peakT, self.w_peakf, self.w_ledger],
                  device=self.device)

    def step(self, dt, h_air=None, eps=None, hcc=None, hsink=None):
        g = self.g
        wp.launch(k_step, dim=g.n, inputs=[g.nx, g.ny, g.nz, self.w_mat, self.w_body, self.w_sink, self.w_T, self.w_H,
                                           self.w_peakf, self.w_src, self.w_dx, self.w_dy, self.w_dz, self.w_tabK,
                                           float(dt), float(PR.v(PR.H_AIR) if h_air is None else h_air),
                                           float(PR.v(PR.STEEL_EMISSIVITY) if eps is None else eps),
                                           float(PR.v(PR.H_CONTACT_STEEL) if hcc is None else hcc),
                                           float(PR.v(PR.H_CONTACT_STEEL) if hsink is None else hsink),
                                           self.w_H2, self.w_ledger, self.w_Hlo, self.w_Hlo2],
                  device=self.device)
        self.w_H, self.w_H2 = self.w_H2, self.w_H
        self.w_Hlo, self.w_Hlo2 = self.w_Hlo2, self.w_Hlo
        self._invert()
        self.w_src.zero_()
        self.time += dt

    # ---------------------------------------------------------------- sources
    def _box_launch(self, kernel, center, half, extra):
        g = self.g
        (i0, i1), (j0, j1), (k0, k1) = g.range_of(np.asarray(center) - half, np.asarray(center) + half)
        bi, bj, bk = i1 - i0, j1 - j0, k1 - k0
        if bi <= 0 or bj <= 0 or bk <= 0:
            return 0.
        self.w_wsum.zero_()
        wp.launch(kernel, dim=bi*bj*bk, inputs=[self.w_mat, self.w_xc, self.w_yc, self.w_zc, self.w_dx, self.w_dy,
                                               self.w_dz, g.nx, g.ny, g.nz, i0, j0, k0, bj, bk, *extra, self.w_wt,
                                               self.w_wsum], device=self.device)
        return float(self.w_wsum.numpy()[0])

    def add_surface_gaussian(self, power_w, p, axis, sigma):
        """Distribute power_w over the open metal faces facing the arc axis, Gaussian about p (weld frame)."""
        if power_w <= 0.:
            return 0.
        axis = np.asarray(axis, float)/np.linalg.norm(axis)
        W = self._box_launch(k_gauss_weights, p, 4.*sigma, [wp.vec3(*map(float, p)), wp.vec3(*map(float, axis)),
                                                            float(sigma)])
        if W <= 0.:
            return 0.
        wp.launch(k_add_weighted, dim=self.g.n, inputs=[self.w_wt, float(power_w/W), self.w_src], device=self.device)
        return power_w

    def add_cylinder(self, power_w, p, axis, radius, depth):
        """Distribute power_w uniformly over metal cells in a cylinder (droplet heat)."""
        if power_w <= 0.:
            return 0.
        axis = np.asarray(axis, float)/np.linalg.norm(axis)
        W = self._box_launch(k_cyl_weights, np.asarray(p) + .5*depth*axis, max(radius, depth) + 1e-3,
                             [wp.vec3(*map(float, p)), wp.vec3(*map(float, axis)), float(radius), float(depth)])
        if W <= 0.:
            return 0.
        wp.launch(k_add_weighted, dim=self.g.n, inputs=[self.w_wt, float(power_w/W), self.w_src], device=self.device)
        return power_w

    def weights(self, kind, center_or_p, axis, a, b=None):
        """Normalised source weights in a persistent array (sum 1): kind 'gauss' (surface flux; a = sigma) or 'cyl'
        (volume; a = radius, b = depth). Computed once per robot tick; apply() adds power x weights each substep."""
        name = "w_src_" + kind
        if not hasattr(self, name):
            setattr(self, name, wp.zeros(self.g.n, dtype=wp.float32, device=self.device))
        arr = getattr(self, name)
        arr.zero_()
        axis = np.asarray(axis, float)/np.linalg.norm(axis)
        p = np.asarray(center_or_p, float)
        if kind == "gauss":
            W = self._box_launch(k_gauss_weights, p, 4.*a, [wp.vec3(*map(float, p)), wp.vec3(*map(float, axis)),
                                                            float(a)])
        else:
            W = self._box_launch(k_cyl_weights, p + .5*b*axis, max(a, b) + 1e-3,
                                 [wp.vec3(*map(float, p)), wp.vec3(*map(float, axis)), float(a), float(b)])
        if W <= 0.:
            return 0.
        wp.launch(k_add_weighted, dim=self.g.n, inputs=[self.w_wt, float(1./W), arr], device=self.device)
        return W

    def apply(self, kind, power_w):
        if power_w <= 0.:
            return
        wp.launch(k_add_scaled, dim=self.g.n, inputs=[getattr(self, "w_src_" + kind), float(power_w), self.w_src],
                  device=self.device)

    # ---------------------------------------------------------------- checkpoints
    SNAP_FIELDS = ("H", "Hlo", "T", "fl", "peakT", "peakf")

    def snapshot(self):
        """Exact copy of the evolving state (host arrays): enthalpy and its derived fields, the melt history, the
        material and body maps, the ledger."""
        out = {n: self.field(n).copy() for n in self.SNAP_FIELDS}
        out.update(mat=self.mat.copy(), body=self.body.copy(), ledger=self.w_ledger.numpy().copy(), time=self.time)
        return out

    def restore(self, snap):
        for n in self.SNAP_FIELDS:
            if n == "Hlo" and n not in snap:              # a snapshot from before the remainder existed
                self.w_Hlo.zero_()
                continue
            getattr(self, "w_" + n).assign(snap[n])
        self.mat[:] = snap["mat"]
        self.body[:] = snap["body"]
        self.w_mat.assign(self.mat)
        self.w_body.assign(self.body)
        self.w_ledger.assign(snap["ledger"])
        self.w_H2.assign(snap["H"])
        self.w_Hlo2.assign(self.w_Hlo)
        self.w_src.zero_()
        self.time = snap["time"]

    # ---------------------------------------------------------------- host access
    def stored(self):
        self.w_store.zero_()
        g = self.g
        wp.launch(k_stored, dim=g.n, inputs=[self.w_mat, self.w_H, self.w_dx, self.w_dy, self.w_dz, g.ny, g.nz,
                                             self.w_store, self.w_Hlo], device=self.device)
        return float(self.w_store.numpy()[0])

    def ledger(self):
        led = self.w_ledger.numpy()
        stored = self.stored()
        inp, loss, sink, evap, dep, rem, det = (float(x) for x in led[:7])
        residual = (stored - self.stored0) - (inp + dep - rem - det - loss - sink - evap)
        return dict(input_j=inp, surface_loss_j=loss, sink_loss_j=sink, evaporation_j=evap,
                    deposited_metal_j=dep, removed_metal_j=rem, detached_j=det, detached_m3=float(led[7]),
                    stored_change_j=stored - self.stored0, residual_j=residual,
                    residual_rel=residual/max(inp + dep, 1.))   # relative to the energy put in (floor 1 J: a never-heated
                                                                 # coupon's float noise is not a 900 % error)

    def field(self, name):
        return getattr(self, "w_" + name).numpy()

    def slice_block(self, name, i0, i1):
        """Host copy of x slices [i0, i1) of a field (contiguous)."""
        g = self.g
        n = g.ny*g.nz
        arr = getattr(self, "w_" + name)
        out = wp.empty(n*(i1 - i0), dtype=arr.dtype, device="cpu")
        wp.copy(out, arr, 0, i0*n, n*(i1 - i0))
        return out.numpy().reshape(i1 - i0, g.ny, g.nz)

    def write_slice_block(self, name, i0, values):
        """Write host values (shape [k, ny, nz]) into x slices starting at i0."""
        g = self.g
        n = g.ny*g.nz
        arr = getattr(self, "w_" + name)
        npt = {wp.float32: np.float32, wp.uint8: np.uint8}[arr.dtype]
        v = np.ascontiguousarray(values, dtype=npt).ravel()
        src = wp.array(v, dtype=arr.dtype, device="cpu")
        wp.copy(arr, src, i0*n, 0, len(v))

    def slice_max_liquid(self, j0, j1, k0, k1):
        """Largest liquid fraction of each x slice inside the y-z box (GPU reduction; nx floats to the host)."""
        g = self.g
        out = wp.zeros(g.nx, dtype=wp.float32, device=self.device)
        wp.launch(k_slice_max, dim=g.nx*(j1 - j0)*(k1 - k0), inputs=[self.w_fl, g.ny, g.nz, j0, j1, k0, k1, out],
                  device=self.device)
        return out.numpy()

    def gather(self, name, idx):
        """Host values of a field at flat indices idx (int32 array)."""
        arr = getattr(self, "w_" + name)
        w_idx = wp.array(np.ascontiguousarray(idx, np.int32), dtype=wp.int32, device=self.device)
        if arr.dtype == wp.uint8:
            out = wp.empty(len(idx), dtype=wp.uint8, device=self.device)
            wp.launch(k_gather_u8, dim=len(idx), inputs=[arr, w_idx, out], device=self.device)
        else:
            out = wp.empty(len(idx), dtype=wp.float32, device=self.device)
            wp.launch(k_gather, dim=len(idx), inputs=[arr, w_idx, out], device=self.device)
            if name == "H":                              # the enthalpy with its remainder (float64)
                lo = wp.empty(len(idx), dtype=wp.float32, device=self.device)
                wp.launch(k_gather, dim=len(idx), inputs=[self.w_Hlo, w_idx, lo], device=self.device)
                return out.numpy().astype(np.float64) + lo.numpy()
        return out.numpy()

    def scatter(self, name, idx, values):
        """Write host values at flat indices into a field (and the host mirror for mat/body)."""
        if len(idx) == 0:
            return
        arr = getattr(self, "w_" + name)
        w_idx = wp.array(np.ascontiguousarray(idx, np.int32), dtype=wp.int32, device=self.device)
        if arr.dtype == wp.uint8:
            val = wp.array(np.ascontiguousarray(values, np.uint8), dtype=wp.uint8, device=self.device)
            wp.launch(k_scatter_u8, dim=len(idx), inputs=[arr, w_idx, val], device=self.device)
            if name in ("mat", "body"):
                getattr(self, name)[np.asarray(idx)] = values
        else:
            val = wp.array(np.ascontiguousarray(values, np.float32), dtype=wp.float32, device=self.device)
            wp.launch(k_scatter, dim=len(idx), inputs=[arr, w_idx, val], device=self.device)
            if name == "H":                              # a written enthalpy has no remainder
                z = wp.zeros(len(idx), dtype=wp.float32, device=self.device)
                wp.launch(k_scatter, dim=len(idx), inputs=[self.w_Hlo, w_idx, z], device=self.device)

    def set_ledger_extra(self, slot, joules):
        """Book host-side energy terms (e.g. droplet enthalpy carried by new weld-metal cells) into the ledger."""
        led = self.w_ledger.numpy()
        led[slot] += joules
        self.w_ledger.assign(led)

    def set_material(self, m, rho, cp, k, solidus=None, liquidus=None, latent=0.):
        """Replace one material's tables by constant properties (analytic verification probes only)."""
        T = T_TAB0 + T_TAB_DT*np.arange(T_TAB_N)
        self.tabH[m] = (rho*cp*T).astype(np.float32)
        self.tabK[m] = np.full(T_TAB_N, k, np.float32)
        self.sol[m], self.liq[m] = (1e9, 1e9 + 1) if solidus is None else (solidus, liquidus)
        self.lat[m] = rho*latent

    def h_of(self, material, t):
        """Total enthalpy (J/m3) of a material at temperature t (C), latent heat included above the solidus."""
        hs = float(np.interp(t, T_TAB0 + T_TAB_DT*np.arange(T_TAB_N), self.tabH[material]))
        ts, tl = float(self.sol[material]), float(self.liq[material])
        return hs + float(self.lat[material])*float(np.clip((t - ts)/(tl - ts), 0., 1.))
