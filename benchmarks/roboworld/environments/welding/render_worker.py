"""Isaac Sim render-only worker for the welding cell: draws recorded states, never runs a physics timeline.

Protocol and Kit setup follow wiping/render_worker.py (whiteboard.ipc, render_lifecycle parent watch, the minimal
render-only kit, capture_rgb readback); the state application is the welding frame (welding/render.py).
"""
import argparse
import base64
import faulthandler
import io
import json
import math
import os
import socket
import time
import traceback
from pathlib import Path

import numpy as np


def _dec(s, dtype, cols=None):
    a = np.frombuffer(base64.b64decode(s), dtype=dtype)
    return a.reshape(-1, cols) if cols else a


def capture(step_render, read_pixels, width, height, flush_attempts=1):
    """RGB readback (whiteboard.render_readback.capture_rgb's loop) that accepts an all-dark frame: the weld camera's
    process exposure renders the cell black while no arc burns, as the real camera does. A missing or wrongly
    shaped buffer still fails after 8 attempts."""
    last = None
    for attempt in range(8):
        step_render()
        raw = np.asarray(read_pixels())
        last = raw.shape
        if raw.ndim == 3 and raw.shape[:2] == (height, width) and raw.shape[2] >= 3 and attempt + 1 >= flush_attempts:
            return raw[:, :, :3].copy(), 0
    raise RuntimeError(f"No RGB readback after 8 render-only attempts; shape={last}")


# ------------------------------------------------------------------------------------------------ camera model
# The renderer's HdrColor is linear and exposure-scaled (render.RTX_K, measured); every camera's image is made here from
# it: exposure, lens veiling glare and the sensor's response (render.EXPOSURE). The standard cameras keep the curve the
# worker's `rgb` output applied before (ACES + sRGB, fitted).
_GLARE_KERNELS = {}


def _glare_kernel(h, w, r0):
    key = (h, w, round(r0, 3))
    if key not in _GLARE_KERNELS:
        H, W = 2*h, 2*w
        y = np.fft.fftfreq(H, 1./H)[:, None]
        x = np.fft.fftfreq(W, 1./W)[None, :]
        K = (x*x + y*y + r0*r0)**-1.5
        K /= K.sum()
        from scipy import fft as sfft
        _GLARE_KERNELS[key] = sfft.rfft2(K.astype(np.float32), workers=4)
    return _GLARE_KERNELS[key]


def _upsample(G, stride, out_h, out_w):
    """Bilinear upsampling of a grid G (hb x wb [x c]) sampled at the centres of stride x stride blocks to out_h x out_w
    (edges clamped). OpenCV's resize when available (the render worker's Python has it; 10x faster), numpy otherwise."""
    hb, wb = G.shape[:2]
    try:
        import cv2
        out = cv2.resize(np.ascontiguousarray(G, np.float32), (wb*stride, hb*stride), interpolation=cv2.INTER_LINEAR)
        if out.shape[0] >= out_h and out.shape[1] >= out_w:
            return out[:out_h, :out_w]
    except ImportError:
        pass
    gy = np.clip((np.arange(out_h) + .5)/stride - .5, 0., hb - 1.)
    gx = np.clip((np.arange(out_w) + .5)/stride - .5, 0., wb - 1.)
    y0 = np.floor(gy).astype(int); x0 = np.floor(gx).astype(int)
    y1 = np.minimum(y0 + 1, hb - 1); x1 = np.minimum(x0 + 1, wb - 1)
    fy = (gy - y0).astype(np.float32); fx = (gx - x0).astype(np.float32)
    if G.ndim == 3:
        fy, fx = fy[:, None, None], fx[None, :, None]
    else:
        fy, fx = fy[:, None], fx[None, :]
    top = G[y0][:, x0]*(1 - fx) + G[y0][:, x1]*fx
    bot = G[y1][:, x0]*(1 - fx) + G[y1][:, x1]*fx
    return top*(1 - fy) + bot*fy


def veiling_glare(img, fraction, r0_px, threshold=None, step=4):
    """Lens veiling glare: (1 - fraction) of the light where it is imaged, `fraction` spread with a kernel
    ~ (r^2 + r0^2)^-1.5 (zero-padded FFT convolution, energy conserving). The spread is computed on a `step`-times
    block-averaged image and interpolated back (the kernel is smooth beyond r0; for speed), and skipped when no pixel
    exceeds `threshold` (no source bright enough for the 2 % spread to show; for speed) [authored]."""
    if fraction <= 0 or (threshold is not None and float(img.max()) < threshold):
        return img
    h, w = img.shape[:2]
    hs, ws = -(-h//step), -(-w//step)
    pad = np.zeros((hs*step, ws*step, img.shape[2]), np.float32)
    pad[:h, :w] = img
    small = pad.reshape(hs, step, ws, step, -1).mean((1, 3))
    Kf = _glare_kernel(hs, ws, max(r0_px/step, .75))
    from scipy import fft as sfft
    small = small.astype(np.float32)
    spread = np.empty_like(small)
    for c in range(small.shape[2]):
        buf = np.zeros((2*hs, 2*ws), np.float32)
        buf[:hs, :ws] = small[:, :, c]
        spread[:, :, c] = sfft.irfft2(sfft.rfft2(buf, workers=4)*Kf, s=buf.shape, workers=4)[:hs, :ws]
    up = _upsample(np.maximum(spread, 0.).astype(np.float32), step, h, w)
    return (1 - fraction)*img + fraction*up


def aces_filmic(x):
    """The renderer's ACES tone curve as the worker's `rgb` output applied it (Narkowicz 2015 fit; matched the old images
    within 1-3 grey levels after sRGB encoding)."""
    x = np.maximum(x, 0.)
    return np.clip(x*(2.51*x + .03)/(x*(2.43*x + .59) + .14), 0., 1.)


def srgb_oetf(x):
    x = np.clip(x, 0., 1.)
    return np.where(x <= .0031308, 12.92*x, 1.055*np.power(x, 1/2.4) - .055)


def _edge(r_over_s, col):
    """The column's edge factor (render.arc_edge, duplicated): exp(-(r / (edge s))^p), 1 without one."""
    k = col.get("edge")
    if not k:
        return 1.
    return np.exp(-(np.asarray(r_over_s, float)/k)**col.get("edge_p", 6.))


_EDGE_LINE = {}


def _edge_line(x, col):
    """Line integral across an edge-cut unit Gaussian (render.ARC_EDGE) at impact distance x (in spreads), relative to
    the plain Gaussian's sqrt(2 pi) exp(-x^2/2): tabulated once per (edge, p) by quadrature, interpolated."""
    k = col.get("edge")
    if not k:
        return 1.
    key = (float(k), float(col.get("edge_p", 6.)))
    if key not in _EDGE_LINE:
        xs = np.linspace(0., 2*key[0] + 2., 801)
        w = np.linspace(0., 2*key[0] + 2., 2001)
        r = np.sqrt(xs[:, None]**2 + w[None, :]**2)
        trap = getattr(np, "trapezoid", None) or np.trapz            # the render runtime has numpy 1.x
        ex = 2*trap(np.exp(-.5*r*r)*np.exp(-(r/key[0])**key[1]), w, axis=1)
        _EDGE_LINE[key] = (xs, ex/(math.sqrt(2*math.pi)*np.exp(-.5*xs*xs)))
    xs, ratio = _EDGE_LINE[key]
    return np.interp(np.asarray(x, float), xs, ratio, right=0.)


def _profile(col, t, rho2):
    """Emission density j / j0 of the arc column (render.arc_profile, duplicated: the worker does not import the
    backend)."""
    up = np.where(t < 0, np.exp(-.5*(t/(col["up"]/3.))**2), 1.)
    f = np.clip(t/col["La"], 0., 1.)
    a = col.get("core_floor", 0.) + (1. - col.get("core_floor", 0.))*np.exp(-np.abs(t)/col["core"])
    sc = col["s_core"] + (col.get("s_core_root", col["s_core"]) - col["s_core"])*f
    r = np.sqrt(np.maximum(rho2, 0.))
    core = a*np.exp(-rho2/(2*sc*sc))*_edge(r/sc, col)
    s = col["s_wire"] + (col["s_root"] - col["s_wire"])*f
    env = col["env"]*np.exp(-rho2/(2*s*s))*_edge(r/s, col)
    down = np.where(t > col["La"], np.exp(-.5*((t - col["La"])/col["s_root"])**2), 1.)
    return np.where(t < -2*col["up"], 0., up*down*(core + env))


ARC_EXTENT = 4.5        # the composite covers 4.5 spreads about the axis: beyond, the envelope's tail is below 4e-5 of
                        # its peak, under the arc-lit metal around it (a 3-spread cut showed as a hard edge on the log
                        # camera; 6 spreads cost 1.8x the rays for no visible change)


def column_integral(O, D, col, s_hi=None):
    """Luminance (cd/m2) of the arc column seen along rays O + s D (unit D, N x 3), s in [0, s_hi]: the integral of the
    emission density j (render.arc_column) along each ray. Each Gaussian component is integrated across the column in
    closed form at the ray's closest approach to the axis (axial position t_c, distance d, crossing angle theta):
    exp(-d^2/2s^2) s sqrt(2 pi)/sin(theta) Phi((s_hi - s_c) sin(theta)/s) -- the axial profile is taken at t_c (it
    varies over the 2 s/sin(theta) crossing by 2 s cot(theta)); Phi cuts the integral at the first opaque surface.
    Rays within 17 deg of the axis are left out (none of the cameras looks along the wire). Optically thin."""
    from scipy.special import ndtr
    O = np.asarray(O, float)
    D = np.asarray(D, float)
    p0, ax = np.asarray(col["p0"], float), np.asarray(col["axis"], float)
    W = O - p0
    b = D @ ax
    sin2 = np.maximum(1. - b*b, 0.)
    ok = sin2 > .085
    sin = np.sqrt(np.where(ok, sin2, 1.))
    wa = W @ ax
    dw = np.einsum("ij,ij->i", D, W)
    s_ca = np.where(ok, (b*wa - dw)/np.where(ok, sin2, 1.), 0.)           # ray parameter of the closest approach
    t_c = wa + s_ca*b
    q = W + s_ca[:, None]*D - t_c[:, None]*ax
    d2 = np.einsum("ij,ij->i", q, q)
    t = np.clip(t_c, -2*col["up"], col["La"])
    up = np.where(t < 0, np.exp(-.5*(t/(col["up"]/3.))**2), 1.)
    # past the root plane: the fade over the root's spread (render.arc_profile); the work surface cuts it (s_hi)
    up = up*np.where(t_c > col["La"], np.exp(-.5*((t_c - col["La"])/col["s_root"])**2), 1.)
    inside = (t_c >= -2*col["up"]) & (t_c <= col["La"] + 4.5*col["s_root"])
    fa = np.clip(t/col["La"], 0., 1.)
    sd = col["s_wire"] + (col["s_root"] - col["s_wire"])*fa
    # the core: brightest below the wire, running the whole column (render.ARC_CORE_FLOOR), widening toward the root
    a_c = col.get("core_floor", 0.) + (1. - col.get("core_floor", 0.))*np.exp(-np.abs(t)/col["core"])
    s_c = col["s_core"] + (col.get("s_core_root", col["s_core"]) - col["s_core"])*fa
    out = np.zeros(len(O))
    hi = np.full(len(O), np.inf) if s_hi is None else np.asarray(s_hi, float)
    for amp, sg in ((a_c, s_c), (np.full(len(O), col["env"]), sd)):
        # the part of the crossing in [0, s_hi], from the closest approach s_ca, so a surface in front of the column
        # cuts it
        frac = ndtr((hi - s_ca)*sin/sg) - ndtr((0. - s_ca)*sin/sg)
        # the column's edge (render.ARC_EDGE): the line integral across the edge-cut Gaussian at the ray's impact
        # distance, as a factor on the Gaussian's closed form (_edge_line: a numerical table)
        out += amp*np.exp(-d2/(2*sg*sg))*sg*math.sqrt(2*math.pi)/sin*frac*_edge_line(np.sqrt(d2)/sg, col)
    return np.where(ok & inside, col["j0"]*up*out, 0.)


def _camera_rays(cam_m, hap, vap, focal, w, h, us, vs):
    """World origin and unit directions of rays through pixel coordinates (us, vs) of a USD camera (local -z forward,
    local-to-world matrix cam_m in USD row-vector convention)."""
    M = np.asarray(cam_m, float)
    R, o = M[:3, :3], M[3, :3]
    tx, ty = hap/2/focal, vap/2/focal
    dc = np.stack([(us - w/2)/(w/2)*tx, -(vs - h/2)/(h/2)*ty, -np.ones_like(us)], -1)
    d = dc @ R
    return o, d/np.linalg.norm(d, axis=-1, keepdims=True)


def _project(cam_m, hap, vap, focal, w, h, P):
    M = np.asarray(cam_m, float)
    R, o = M[:3, :3], M[3, :3]
    c = (np.asarray(P, float) - o) @ R.T
    front = c[:, 2] < -1e-6
    z = np.where(front, -c[:, 2], 1.)
    u = c[:, 0]/z/(hap/2/focal)*w/2 + w/2
    v = -c[:, 1]/z/(vap/2/focal)*h/2 + h/2
    return u, v, front


def _bbox(u, v, front, w, h, pad=1):
    if not front.any():
        return None
    u0, u1 = int(max(0, np.floor(u[front].min()) - pad)), int(min(w, np.ceil(u[front].max()) + pad + 1))
    v0, v1 = int(max(0, np.floor(v[front].min()) - pad)), int(min(h, np.ceil(v[front].max()) + pad + 1))
    return (u0, u1, v0, v1) if (u0 < u1 and v0 < v1) else None


def arc_composite(L, dist, cam_m, hap, vap, focal, arc, pool=None):
    """The arc as the camera sees it, added to the scene luminance L (H x W x 3, cd/m2) at its true luminance: the
    renderer's arc light is invisible to camera rays, and pixels this bright are clamped by the OptiX denoiser (a
    3.5e8 cd/m2 light came out at 5.2e5).
    Seen directly: per pixel (2 x 2, or 4 x 4 for a small arc) sub-rays, the emission integral of the column
    (render.arc_column) along the ray up to the depth AOV (distance_to_camera): the wire tip, the nozzle or the pool in
    front of or inside the column occlude it.
    In the pool's mirror: pool = (world points, unit normals, F0) of the pool surface; pixels whose depth-AOV hit lies on
    it (nearest pool vertex within 0.6 mm, within 0.25 mm of its tangent plane) reflect their ray about the normal
    interpolated from the 3 nearest vertices; the column integral along the reflected ray times the Fresnel reflectance
    (Schlick, F0 = 1 - emissivity of liquid iron) is added. Reflected rays are not tested for occlusion."""
    h, w = L.shape[:2]
    col = arc
    lum_rgb = np.asarray(col["color"], float)
    p0, p1 = np.asarray(col["p0"], float), np.asarray(col["p1"], float)
    ax = np.asarray(col["axis"], float)
    Rb = ARC_EXTENT*max(col["s_root"], col["s_wire"])
    R = np.asarray(cam_m, float)[:3, :3]
    ends = [p0 - ax*2*col["up"], p1 + ax*3*col["s_root"]]
    corners = np.array([e + sx*R[0]*Rb + sy*R[1]*Rb for e in ends for sx in (-1, 1) for sy in (-1, 1)])
    bb = _bbox(*_project(cam_m, hap, vap, focal, w, h, corners), w, h)
    if bb is not None:
        u0, u1, v0, v1 = bb
        # sub-rays per pixel: 4 x 4 for an arc a few pixels wide (far cameras), 1 when its core spans many pixels
        px_mm = (u1 - u0)/max(2*Rb*1e3, 1e-9)
        if px_mm > 12.:
            # a close-up (the weld camera): the column spans hundreds of pixels; one ray per 2 x 2 (3 x 3 beyond 20 px
            # per mm) block, the nearest surface in the block, interpolated -- occluding edges soften by 1-1.5 px
            st_ = 3 if px_mm > 20. else 2
            gu = np.arange(u0, u1, st_) + st_/2
            gv = np.arange(v0, v1, st_) + st_/2
            U, V = np.meshgrid(gu, gv)
            o, d = _camera_rays(cam_m, hap, vap, focal, w, h, U.ravel(), V.ravel())
            hb, wb = len(gv), len(gu)
            dzb = np.full((hb*st_, wb*st_), np.inf)
            blk = np.asarray(dist, float)[v0:v0 + hb*st_, u0:u0 + wb*st_]
            dzb[:blk.shape[0], :blk.shape[1]] = np.where(np.isfinite(blk) & (blk > 0), blk, np.inf)
            dzr = dzb.reshape(hb, st_, wb, st_)
            dz = dzr.min((1, 3)).ravel()
            Lc = column_integral(np.broadcast_to(o, d.shape), d, col, s_hi=dz).reshape(hb, wb).astype(np.float32)
            Lp = _upsample(Lc, st_, hb*st_, wb*st_)[:v1 - v0, :u1 - u0]
            # blocks straddling an occluding edge (depths more than 1 mm apart, e.g. the nozzle's silhouette): each pixel
            # there gets its own ray cut at its own depth -- interpolating blocks across the edge drew a regular zigzag of
            # leaked glow along the nozzle's outline in the weld camera
            edge = np.repeat(np.repeat((np.where(np.isfinite(dzr), dzr, 1e3).max((1, 3)) - dz.reshape(hb, wb)) > 1e-3,
                                       st_, 0), st_, 1)[:v1 - v0, :u1 - u0]
            for _ in range(1):                      # and their neighbours (the interpolation reaches one block)
                e2 = edge.copy()
                e2[st_:] |= edge[:-st_]
                e2[:-st_] |= edge[st_:]
                e2[:, st_:] |= edge[:, :-st_]
                e2[:, :-st_] |= edge[:, st_:]
                edge = e2
            if edge.any():
                ey, ex = np.nonzero(edge)
                oe, de = _camera_rays(cam_m, hap, vap, focal, w, h, ex + u0 + .5, ey + v0 + .5)
                de_ = np.asarray(dist, float)[ey + v0, ex + u0]
                de_ = np.where(np.isfinite(de_) & (de_ > 0), de_, np.inf)
                Lp[ey, ex] = column_integral(np.broadcast_to(oe, de.shape), de, col, s_hi=de_)
            L[v0:v1, u0:u1] += Lp[..., None]*lum_rgb[None, None, :]
            sub = 0
        else:
            sub = 4 if px_mm < 2. else (2 if px_mm < 6. else 1)
    if bb is not None and sub:
        off = (np.arange(sub) + .5)/sub
        uu = (np.arange(u0, u1)[:, None] + off[None, :]).ravel()
        vv = (np.arange(v0, v1)[:, None] + off[None, :]).ravel()
        U, V = np.meshgrid(uu, vv)
        o, d = _camera_rays(cam_m, hap, vap, focal, w, h, U.ravel(), V.ravel())
        dz = np.repeat(np.repeat(np.asarray(dist, float)[v0:v1, u0:u1], sub, 0), sub, 1).ravel()
        dz = np.where(np.isfinite(dz) & (dz > 0), dz, np.inf)
        Lc = column_integral(np.broadcast_to(o, d.shape), d, col, s_hi=dz)
        Lp = Lc.reshape(v1 - v0, sub, u1 - u0, sub).mean((1, 3))
        L[v0:v1, u0:u1] += Lp[..., None]*lum_rgb[None, None, :]
    if pool is not None and len(pool[0]):
        from scipy.spatial import cKDTree
        Pw, Nw, F0 = pool
        bb = _bbox(*_project(cam_m, hap, vap, focal, w, h, Pw), w, h)
        if bb is not None:
            u0, u1, v0, v1 = bb
            # one reflected ray per 2 x 2 block when the pool spans many pixels (the weld camera), interpolated
            st_ = 2 if (u1 - u0)*(v1 - v0) > 40000 else 1
            gu = np.arange(u0, u1, st_) + st_/2
            gv = np.arange(v0, v1, st_) + st_/2
            hb, wb = len(gv), len(gu)
            U, V = np.meshgrid(gu, gv)
            o, d = _camera_rays(cam_m, hap, vap, focal, w, h, U.ravel(), V.ravel())
            dz = np.asarray(dist, float)[np.minimum(V.astype(int), h - 1), np.minimum(U.astype(int), w - 1)].ravel()
            good = np.isfinite(dz) & (dz > 0)
            X = o + d*np.where(good, dz, 0.)[:, None]
            lo_, hi_ = Pw.min(0) - 6e-4, Pw.max(0) + 6e-4
            good &= ((X >= lo_) & (X <= hi_)).all(1)
            dk = np.full((len(X), 3), np.inf)
            ik = np.zeros((len(X), 3), int)
            if good.any():
                dk[good], ik[good] = cKDTree(Pw).query(X[good], k=3)
            n = (Nw[ik]*(1./np.maximum(dk, 1e-6))[..., None]).sum(1)
            n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
            plane = np.abs(np.einsum("ij,ij->i", X - Pw[ik[:, 0]], Nw[ik[:, 0]]))
            on = good & (dk[:, 0] < 6e-4) & (plane < 2.5e-4)
            if on.any():
                dd, nn, XX = d[on], n[on], X[on]
                cos = np.clip(-np.einsum("ij,ij->i", dd, nn), 0., 1.)
                r = dd + 2*cos[:, None]*nn
                Lr = column_integral(XX + nn*1e-5, r, col)
                Fr = F0 + (1 - F0)*(1 - cos)**5
                add = np.zeros(len(dz), np.float32)
                add[np.nonzero(on)[0]] = np.where(cos > 0, Lr*Fr, 0.)
                add = add.reshape(hb, wb)
                if st_ > 1:
                    add = _upsample(add, st_, hb*st_, wb*st_)
                add = add[:v1 - v0, :u1 - u0]
                L[v0:v1, u0:u1] += add[..., None]*lum_rgb[None, None, :]
    return L


def _hash01(i, j, k, seed):
    """Hash of integer lattice corners to [0, 1] (uint32 arithmetic, wrapping)."""
    with np.errstate(over="ignore"):
        h = (i.astype(np.uint32)*np.uint32(73856093)) ^ (j.astype(np.uint32)*np.uint32(19349663)) ^ \
            (k.astype(np.uint32)*np.uint32(83492791)) ^ np.uint32((seed*2654435761) & 0xFFFFFFFF)
        h = (h ^ (h >> np.uint32(13)))*np.uint32(1274126177)
        return ((h ^ (h >> np.uint32(16))) & np.uint32(0xFFFFFF)).astype(np.float32)*np.float32(1./0xFFFFFF)


NOISE_STD = .370        # standard deviation of _vnoise over space (measured on 1e6 random points: 0.370)
NOISE2_STD = .414       # of _vnoise(q) + 0.5 _vnoise(2.1 q + 31.4) (measured likewise: 0.414)


def _vnoise(P, seed):
    """Smooth value noise in [-1, 1] at points P (..., 3), lattice spacing 1 (trilinear between hashed corners,
    smoothstep weights); render.value_noise's lattice, vectorised for the worker."""
    q = np.asarray(P, np.float64)
    i0 = np.floor(q).astype(np.int32)
    f = (q - i0).astype(np.float32)
    f = f*f*(3 - 2*f)
    out = np.zeros(q.shape[:-1], np.float32)
    for dx in (0, 1):
        wx = f[..., 0] if dx else 1 - f[..., 0]
        for dy in (0, 1):
            wy = wx*(f[..., 1] if dy else 1 - f[..., 1])
            for dz in (0, 1):
                w = wy*(f[..., 2] if dz else 1 - f[..., 2])
                out += w*_hash01(i0[..., 0] + dx, i0[..., 1] + dy, i0[..., 2] + dz, seed)
    return 2*out - 1


def plume_turbulence(X, fume):
    """The plume's instantaneous structure at sample points X (..., 3), world (render.PLUME_* above): returns
    (axis offset dx, dy (m) from the arc's vertical, concentration factor or None). Deterministic in (X, fume t)."""
    b = np.asarray(fume["base"], float)
    tu = fume.get("turb")
    zr = X[..., 2] - b[2]
    dx = np.zeros(X.shape[:-1])
    dy = np.zeros(X.shape[:-1])
    dr = fume.get("drift")
    if dr:
        dx = np.interp(zr, dr["z"], dr["dx"])
        dy = np.interp(zr, dr["z"], dr["dy"])
    if not tu:
        return dx, dy, None
    zv, Wc, c = float(fume["z_v"]), float(fume["Wc"]), float(fume["bg_per_zeta"])
    e = float(tu["eddy"])
    seed = int(tu["seed"])
    t = float(fume["t"])
    lam0, lam1 = tu["laminar"]

    # the label along the axis: log height in eddy sizes (eddies grow with the plume, b_G ~ zeta), carried up at the
    # plume's own speed at the reference height (render.PLUME_ADV_Z): one rate for the whole plume, so the field never
    # shears into bands; below that height eddies are carried too slowly, above it too fast (both turn over within the
    # films' 100 ms frame interval below ~1 m, derived from b_G / w)
    zr_ = float(tu.get("adv_z", .3)) + zv
    om = Wc*zr_**(-1/3)/(e*c*zr_)

    def label(zeta):
        return np.log(zeta)/(e*c) - om*t
    # the axis's meander: a function of height only (large eddies), tabulated and interpolated
    zt = np.linspace(max(float(zr.min()), 0.), max(float(zr.max()), 1e-3), 256)
    zeta_t = zt + zv
    qt = label(zeta_t)
    mxt = _vnoise(np.stack([np.full_like(qt, .37), np.full_like(qt, .71), .5*qt], -1), seed + 11)
    myt = _vnoise(np.stack([np.full_like(qt, 50.87), np.full_like(qt, 51.21), .5*qt + 50.5], -1), seed + 12)
    ramp_t = np.clip((zt - lam0)/(lam1 - lam0), 0., 1.)
    mag_t = float(tu["meander"])*c*zeta_t*ramp_t/NOISE_STD
    dx = dx + np.interp(zr, zt, mag_t*mxt)
    dy = dy + np.interp(zr, zt, mag_t*myt)
    # concentration: lognormal factor of a two-octave noise in the plume's own frame (radial coordinates in eddy sizes
    # about the meandering axis, the parcel label along it), evaluated only where the turbulence has set in and the plume
    # is not negligible (2.5 Gaussian radii)
    fac = np.ones(X.shape[:-1], np.float32)
    zeta = np.maximum(zr + zv, 1e-3)
    bG = c*zeta
    rx = (X[..., 0] - b[0] - dx)/bG
    ry = (X[..., 1] - b[1] - dy)/bG
    r2b = rx*rx + ry*ry
    sel = (zr > lam0) & (r2b < 6.25)
    if sel.any():
        q = np.stack([rx[sel]/e, ry[sel]/e, label(zeta[sel])], -1)
        N = (_vnoise(q, seed) + .5*_vnoise(2.1*q + 31.4, seed + 1))/NOISE2_STD
        ramp = np.clip((zr[sel] - lam0)/(lam1 - lam0), 0., 1.)
        s = ramp*np.minimum(float(tu["sigma"]) + float(tu["edge"])*r2b[sel], 1.2)   # capped: the factor's mean stays
        fac[sel] = np.exp(s*N - .5*s*s)                                          # within 3 % of 1 (measured)
    return dx, dy, fac


def plume_composite(L, dist, cam_m, hap, vap, focal, fume, stride=8, n=None):
    """The fume plume (render.fume_plume) as an optically thin medium in single scattering, applied to the scene
    luminance L (H x W x 3, cd/m2) in place: L <- L exp(-tau) + L_s along each camera ray up to the depth AOV.
    kappa(r, zeta) = K zeta^(-5/3) exp(-(r / (c zeta))^2) about the vertical axis through the arc (zeta: height above
    the plume's virtual origin), inside zeta in the present plume's height range and r < reach c zeta;
    L_s = int T(s) albedo kappa [ I / (4 pi d^2) x unshadowed + L_amb ] ds (isotropic phase; I the arc column's
    luminous intensity from its centroid, d >= 3 mm; shadowed inside the gun's shadow cone, the cone from the arc
    through the nozzle's rim, the gun a cylinder along the torch axis; L_amb the shop's mean radiance). Computed on a grid of every `stride`-th pixel
    with the nearest surface in each block, n midpoint samples per ray, then interpolated to full resolution (the plume
    is smooth on that scale); each pixel cuts its interpolated ray at its own depth, so occluding edges stay sharp
    (a first version cut whole 8 x 8 blocks at their nearest surface: staircase edges along the clamps)."""
    h, w = L.shape[:2]
    # samples along each ray: the turbulent plume's eddies (render.PLUME_EDDY) need ~4 per eddy across the drawn width
    # (16 left horizontal streaks of aliased eddies in the side camera)
    n = n or (40 if fume.get("turb") else 16)
    b = np.asarray(fume["base"], float)
    zv, K, c = float(fume["z_v"]), float(fume["K"]), float(fume["bg_per_zeta"])
    reach = float(fume["reach"])*(1.3 if fume.get("turb") or fume.get("drift") else 1.)   # room for drift and meander
    z0w, z1w = b[2] + float(fume["z_lo"]), b[2] + float(fume["z_hi"])
    za = b[2] - zv                                             # the virtual origin (cone apex) height
    # bounding cone r <= q (z - za), q = reach c
    q = reach*c
    ring = np.array([[math.cos(a), math.sin(a), 0.] for a in np.linspace(0, 2*math.pi, 16, endpoint=False)])
    pts = np.concatenate([b + ring*q*(z - za) + [0, 0, z - b[2]] for z in (z0w, z1w)])
    u, v, front = _project(cam_m, hap, vap, focal, w, h, pts)
    if not front.any():
        return L
    if not front.all():                                        # the camera is inside / below the plume's top
        u0, u1, v0, v1 = 0, w, 0, h
    else:
        bb = _bbox(u, v, front, w, h, pad=stride)
        if bb is None:
            return L
        u0, u1, v0, v1 = bb
    gu = np.arange(u0, u1, stride) + stride/2
    gv = np.arange(v0, v1, stride) + stride/2
    U, V = np.meshgrid(gu, gv)
    o, d = _camera_rays(cam_m, hap, vap, focal, w, h, U.ravel(), V.ravel())
    # nearest surface in each stride block
    dz_full = np.where(np.isfinite(dist) & (dist > 0), dist, np.inf)
    hb, wb = len(gv), len(gu)
    pad = np.full((hb*stride, wb*stride), np.inf)
    blk = dz_full[v0:v0 + hb*stride, u0:u0 + wb*stride]
    pad[:blk.shape[0], :blk.shape[1]] = blk
    dz = pad.reshape(hb, stride, wb, stride).max((1, 3)).ravel()      # the farthest surface: the coarse ray covers
                                                                      # every pixel's depth; each pixel cuts it at its own
    def integrate(o, d, dz):
        """Optical depth and in-scatter accumulated along rays o + s d up to depth dz: (tau, L_s, sample ends), n per ray."""
        # ray / cone: (x - bx)^2 + (y - by)^2 = q^2 (z - za)^2
        ox, oy, oz = o[0] - b[0], o[1] - b[1], o[2] - za
        A2 = d[:, 0]**2 + d[:, 1]**2 - q*q*d[:, 2]**2
        B2 = 2*(ox*d[:, 0] + oy*d[:, 1] - q*q*oz*d[:, 2])
        C2 = ox*ox + oy*oy - q*q*oz*oz
        disc = B2*B2 - 4*A2*C2
        ok = disc > 0
        sq = np.sqrt(np.where(ok, disc, 0.))
        A2s = np.where(np.abs(A2) > 1e-12, A2, 1e-12)
        r1, r2 = (-B2 - sq)/(2*A2s), (-B2 + sq)/(2*A2s)
        rmin, rmax = np.minimum(r1, r2), np.maximum(r1, r2)
        outside = C2 > 0
        pos = A2 > 0
        # inside the cone: between the roots for rays shallower than the cone, beyond the far root for steeper ones (the
        # height slab below keeps the upper nappe)
        sa = np.where(outside, np.where(pos, rmin, rmax), 0.)
        sb = np.where(outside, np.where(pos, rmax, np.inf), np.where(pos, rmax, np.where(rmin > 0, rmin, np.inf)))
        ok = np.where(outside, ok, True)
        # height slab [z0w, z1w]
        with np.errstate(divide="ignore", invalid="ignore"):
            tz0 = (z0w - o[2])/d[:, 2]
            tz1 = (z1w - o[2])/d[:, 2]
        slo = np.where(np.abs(d[:, 2]) > 1e-12, np.minimum(tz0, tz1), np.where((o[2] >= z0w) & (o[2] <= z1w), -np.inf, np.inf))
        shi = np.where(np.abs(d[:, 2]) > 1e-12, np.maximum(tz0, tz1), np.where((o[2] >= z0w) & (o[2] <= z1w), np.inf, -np.inf))
        s0 = np.maximum(np.maximum(sa, slo), 0.)
        s1_free = np.minimum(sb, shi)                              # where the plume ends along the ray, no surface
        s1_free = np.where(ok & (s1_free > s0), s1_free, -np.inf)
        s1 = np.minimum(np.minimum(sb, shi), dz)
        ok &= s1 > s0
        idx = np.nonzero(ok)[0]
        tau_g = np.zeros((len(dz), n), np.float32)          # cumulative optical depth / in-scatter at the samples' ends
        Ls_g = np.zeros((len(dz), n), np.float32)
        S_g = np.full((len(dz), n), np.inf, np.float32)      # the samples' far ends along the ray
        if len(idx):
            f = (np.arange(n) + .5)/n
            S = s0[idx, None] + (s1 - s0)[idx, None]*f[None, :]
            ds = (s1 - s0)[idx]/n
            X = o[None, None, :] + S[..., None]*d[idx, None, :]
            zeta = np.maximum(X[..., 2] - za, 1e-4)
            ddx, ddy, turb = plume_turbulence(X, fume)
            r2 = (X[..., 0] - b[0] - ddx)**2 + (X[..., 1] - b[1] - ddy)**2
            kap = K*zeta**(-5/3)*np.exp(-r2/(c*zeta)**2)
            if turb is not None:
                kap = kap*turb
            kap = np.where((r2 <= (float(fume["reach"])*c*zeta)**2) & (X[..., 2] >= z0w) & (X[..., 2] <= z1w), kap, 0.)
            dtau = kap*ds[:, None]
            tau_before = np.cumsum(dtau, 1) - dtau/2
            src = np.full(kap.shape, float(fume["L_amb"]))
            so = fume.get("source")
            if so:
                Ps = np.asarray(so["pos"], float)
                dd2 = np.maximum(np.einsum("ijk,ijk->ij", X - Ps, X - Ps), 9e-6)
                lit = np.ones(kap.shape, bool)
                gun = fume.get("gun")
                if gun:
                    # the gun: a cylinder of radius R along the torch axis from the nozzle's end (distance d0 from the arc
                    # along the axis) back up the torch; seen from the arc it hides the cone through the nozzle's rim,
                    # perp / along < R / d0 beyond the nozzle
                    g0, g1 = np.asarray(gun["p0"], float), np.asarray(gun["p1"], float)
                    ga = (g1 - g0)/np.linalg.norm(g1 - g0)
                    d0 = max(float((g0 - Ps) @ ga), 1e-3)
                    v_ = X - Ps
                    along = v_ @ ga
                    perp2 = np.einsum("ijk,ijk->ij", v_, v_) - along*along
                    # penumbra: the arc is no point; across the rim's shadow edge the visible part of a source of radius
                    # r_s (the column's root spread) changes over an angle 2 r_s / d0 (linear ramp)
                    r_s = float(gun.get("source_r") or 1.8e-3)
                    ang = np.sqrt(np.maximum(perp2, 0.))/np.maximum(along, 1e-6) - float(gun["radius"])/d0
                    lit = np.where(along > d0, np.clip(.5 + ang/(2*r_s/d0), 0., 1.), 1.)
                src = src + float(so["intensity"])/(4*math.pi*dd2)*lit
            Ls_g[idx] = np.cumsum(np.exp(-tau_before)*float(fume["albedo"])*kap*src*ds[:, None], 1)
            tau_g[idx] = np.cumsum(dtau, 1)
            S_g[idx] = s0[idx, None] + (s1 - s0)[idx, None]*(np.arange(1, n + 1)/n)[None, :]
        return tau_g, Ls_g, S_g, s1_free

    tau_g, Ls_g, S_g, s1_free = integrate(o, d, dz)
    # per pixel: the bilinearly interpolated totals of the coarse rays (the plume's smooth field), and at occluding edges
    # each pixel's own ray (below)
    hh, ww = min(h - v0, hb*stride), min(w - u0, wb*stride)
    # rays that miss the plume hold zeros at sample depth 0 (always "in front"), so values blend to zero smoothly across
    # the plume's silhouette (a far-away placeholder depth drew the silhouette as a staircase)
    S_g = np.where(np.isfinite(S_g), S_g, 0.).astype(np.float32)
    up1 = lambda G: _upsample(G.reshape(hb, wb), stride, hb*stride, wb*stride)[:hh, :ww]
    dpx = dz_full[v0:v0 + hh, u0:u0 + ww]
    tau_i = up1(tau_g[:, -1])                                      # the whole column in front of the pixel's surface
    Ls_i = up1(Ls_g[:, -1])
    # Elsewhere each coarse ray ends at its block's farthest surface, within 2 % of every pixel's own depth there. Blocks
    # straddling an occluding edge (depths more than 2 % apart, e.g. the nozzle's outline in front of the plate) and
    # their neighbours (the interpolation reaches one block): each pixel is integrated along its own ray to its own depth
    # (interpolating coarse rays across the edge draws a zigzag of the block period along the nozzle's outline in the
    # weld camera)
    bd = pad.reshape(hb, stride, wb, stride)
    bmin = bd.min((1, 3))
    bmax = np.where(np.isfinite(bd), bd, 1e3).max((1, 3))
    # only where the plume reaches past the block's nearest surface (the plume's end along the coarse ray beyond it):
    # depth steps wholly behind the plume (the table's holes under it) leave every pixel the same column
    eb = (bmax > 1.02*bmin) & np.isfinite(bmin) & (s1_free.reshape(hb, wb) > bmin)
    from scipy.ndimage import binary_dilation
    eb2 = binary_dilation(eb, np.ones((3, 3), bool))
    em = np.repeat(np.repeat(eb2, stride, 0), stride, 1)[:hh, :ww] & np.isfinite(dpx)
    if em.any():
        py, px = np.nonzero(em)
        oe, de = _camera_rays(cam_m, hap, vap, focal, w, h, px + u0 + .5, py + v0 + .5)
        tg, lg, _, _ = integrate(oe, de, dpx[py, px])
        tau_i[py, px] = tg[:, -1]
        Ls_i[py, px] = lg[:, -1]
    col = np.asarray(fume["color"], float)
    reg = L[v0:v0 + hh, u0:u0 + ww]
    reg *= np.exp(-tau_i)[..., None]
    reg += Ls_i[..., None]*col[None, None, :]
    return L


def camera_image(hdr, ex, k_render, rtx_k, arc_fn=None):
    """8-bit RGB of one camera from the renderer's HdrColor (H x W x >= 3) rendered at k_render HdrColor per cd/m2.
    ex: render.EXPOSURE entry. 'filmic' / 'srgb': H = rtx_k t S / N^2 x L, glare, (ACES curve,) sRGB OETF. 'hdr_log': HDR
    process camera, saturation L_sat at the frame's brightest pixel (auto exposure), per channel
    y = ln(1 + L / L_f) / ln(1 + D), D = 10^(dr_db / 20), L_f = L_sat / D, black level at the 0.5 percentile."""
    L = np.nan_to_num(np.asarray(hdr, np.float32)[:, :, :3], nan=0., posinf=65504.)*np.float32(1./k_render)
    L = np.maximum(L, np.float32(0.))
    if arc_fn is not None:
        L = arc_fn(L)
    h, w = L.shape[:2]
    resp = ex.get("response", "filmic")
    if resp == "hdr_log":
        # the camera's exposure follows the arc (the Xiris weld-arc photodetector; render.HDR_PROCESS): the brightest
        # pixel just reaches saturation, never below L_sat_min
        Ylum = L @ np.array([.2126, .7152, .0722])
        Ls = max(float(np.max(Ylum)), float(ex.get("L_sat_min", 0.)), 1e-3)
        # glare matters when a source outshines the scene (2 % spread): skip it unless the brightest pixel exceeds
        # 1000x the median [authored, for speed]
        thr = None if Ls > 1e3*max(float(np.median(Ylum)), 1e-6) else np.inf
    else:
        thr = 20./(rtx_k*float(ex["exposureTime"])*float(ex["filmIso"])/float(ex["fNumber"])**2*float(ex.get("filter_T", 1.)))
    if thr is None or np.isfinite(thr):
        L = veiling_glare(L, float(ex.get("glare", 0.)), 3.*w/1280., threshold=thr)
    if resp == "hdr_log":
        D = 10**(float(ex["dr_db"])/20.)
        y = np.log1p(np.minimum(L, Ls)/(Ls/D))/math.log1p(D)          # per channel: each photosite is logarithmic
        Y = y @ np.array([.2126, .7152, .0722])
        lo = float(np.percentile(Y, .5))                                 # black level: the darkest 0.5 % of the frame
        out = np.clip((y - lo)/max(1. - lo, .05), 0., 1.)**float(ex.get("display_gamma", 1.))
    else:
        k = rtx_k*float(ex["exposureTime"])*float(ex["filmIso"])/float(ex["fNumber"])**2*float(ex.get("filter_T", 1.))
        out = srgb_oetf(aces_filmic(L*np.float32(k)) if resp == "filmic" else L*np.float32(k))
    return np.round(out*255.).astype(np.uint8)


RENDER_FAILURES = ("Rendering failed", "vkAllocateMemory failed", "Unable to allocate buffer")


def render_failure(kit_log, offset):
    """Raise if Kit logged a failed render while this image was captured: the readback still returns a correctly
    shaped (stale or blank) buffer then, which a film would keep as a frame (seen on a GPU shared with other renderers).
    The message says 'out of memory' so a job runner retries the job."""
    try:
        with open(kit_log, "rb") as f:
            f.seek(offset)
            new = f.read().decode("utf-8", "replace")
    except OSError:
        return
    hit = next((m for m in RENDER_FAILURES if m in new), None)
    if hit:
        raise RuntimeError(f"render failed ({hit}): GPU out of memory; the image is not a render of this state")


def _apply_mesh(stage, b):
    """Set a per-vertex weld_surface mesh (the band or the pool) from its request entry."""
    from pxr import Sdf, UsdGeom, Vt
    m = UsdGeom.Mesh(stage.GetPrimAtPath(b["path"]))
    if not m:
        raise ValueError("Missing mesh " + b["path"])
    P = _dec(b["points"], np.float32, 3)
    F = _dec(b["tris"], np.int32, 3)
    N = _dec(b["normals"], np.float32, 3)
    C = _dec(b["color"], np.float32, 3)
    G = _dec(b["glow"], np.float32, 3)
    if not (np.isfinite(P).all() and np.isfinite(G).all()) or (len(F) and (F.min() < 0 or F.max() >= len(P))):
        raise ValueError("Invalid surface " + b["path"])
    pv = UsdGeom.PrimvarsAPI(m)
    m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P))
    m.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
    m.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(F.ravel()))
    m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(N))
    m.SetNormalsInterpolation("vertex")
    m.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([P.min(0), P.max(0)], np.float32)))
    dc = pv.GetPrimvar("displayColor") or pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                                                           UsdGeom.Tokens.vertex)
    dc.Set(Vt.Vec3fArray.FromNumpy(C))
    dc.SetInterpolation(UsdGeom.Tokens.vertex)
    gl = pv.GetPrimvar("glow") or pv.CreatePrimvar("glow", Sdf.ValueTypeNames.Float3Array, UsdGeom.Tokens.vertex)
    gl.Set(Vt.Vec3fArray.FromNumpy(G))
    gl.SetInterpolation(UsdGeom.Tokens.vertex)
    for key in ("weldfrac", "roughness", "metalness"):
        if key in b:
            W_ = _dec(b[key], np.float32)
            if len(W_) != len(P) or not np.isfinite(W_).all():
                raise ValueError("Invalid primvar " + key + " of " + b["path"])
            wf = pv.GetPrimvar(key) or pv.CreatePrimvar(key, Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
            wf.Set(Vt.FloatArray.FromNumpy(W_))
            wf.SetInterpolation(UsdGeom.Tokens.vertex)


def _pool_world(stage, pool):
    """The pool surface's vertices and unit normals in world coordinates and its reflectance F0 (1 - its glow's
    emissivity is sent as the per-vertex base colour)."""
    from pxr import Usd, UsdGeom
    P = _dec(pool["points"], np.float32, 3).astype(float)
    N = _dec(pool["normals"], np.float32, 3).astype(float)
    C = _dec(pool["color"], np.float32, 3)
    M = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(pool["path"])).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()))
    Pw = P @ M[:3, :3] + M[3, :3]
    Nw = N @ M[:3, :3]
    Nw /= np.maximum(np.linalg.norm(Nw, axis=1), 1e-12)[:, None]
    return Pw, Nw, float(np.mean(C)) if len(C) else .6


def apply_state(stage, request):
    from pxr import Gf, Sdf, UsdGeom, Vt
    layer = stage.GetRootLayer()
    updates = []
    for path, q in request["body_q"].items():
        q = np.asarray(q, float)
        if not np.isfinite(q).all():
            raise ValueError("Non-finite body pose")
        updates.append((path + ".xformOp:translate", Gf.Vec3d(*q[:3])))
        updates.append((path + ".xformOp:orient", Gf.Quatf(float(q[6]), Gf.Vec3f(*map(float, q[3:6])))))
    with Sdf.ChangeBlock():
        for attr, value in updates:
            spec = layer.GetAttributeAtPath(attr)
            if not spec:
                raise ValueError("Missing pose attribute " + attr)
            spec.default = value
    b = request.get("band")
    if b is not None:
        _apply_mesh(stage, b)
    pool = request.get("pool")
    pm = stage.GetPrimAtPath(str(Path(request["band"]["path"]).parent/"pool")) if request.get("band") else None
    if pool is not None:
        _apply_mesh(stage, pool)
        UsdGeom.Imageable(stage.GetPrimAtPath(pool["path"])).MakeVisible()
    elif pm:
        UsdGeom.Imageable(pm).MakeInvisible()
    bb = request.get("bracket_band")
    if bb:
        prim = stage.GetPrimAtPath(bb["path"])
        if not prim:
            raise ValueError("Missing bracket band piece " + bb["path"])
        (UsdGeom.Imageable(prim).MakeVisible if bb["visible"] else UsdGeom.Imageable(prim).MakeInvisible)()
    for lk in request.get("looks") or ():
        m = stage.GetPrimAtPath(lk["path"])
        if not m:
            raise ValueError("Missing coupon piece " + lk["path"])
        pv = UsdGeom.PrimvarsAPI(m)
        C = _dec(lk["color"], np.float32, 3)
        R_ = _dec(lk["roughness"], np.float32)
        M_ = _dec(lk["metalness"], np.float32)
        if not (len(C) == len(R_) == len(M_) == int(lk["n"])) or not np.isfinite(C).all():
            raise ValueError("Invalid coupon look " + lk["path"])
        pv.GetPrimvar("displayColor").Set(Vt.Vec3fArray.FromNumpy(C))
        pv.GetPrimvar("roughness").Set(Vt.FloatArray.FromNumpy(R_))
        pv.GetPrimvar("metalness").Set(Vt.FloatArray.FromNumpy(M_))
    arc = request.get("arc")
    from pxr import UsdLux
    lp = stage.GetPrimAtPath("/World/Arc/light")
    if lp and lp.IsA(UsdLux.SphereLight):
        light = UsdLux.SphereLight(lp)
        if arc and arc.get("light"):
            lt = arc["light"]
            vals = [*lt["pos"], lt["radius"], lt["intensity"]]
            if not np.isfinite(vals).all():
                raise ValueError("Non-finite arc light")
            ops = UsdGeom.Xformable(lp).GetOrderedXformOps()
            if not ops:
                raise ValueError("The scene was exported by an older render.py (no posable arc light): re-export it")
            ops[0].Set(Gf.Vec3d(*map(float, lt["pos"])))
            light.GetRadiusAttr().Set(float(lt["radius"]))
            light.GetIntensityAttr().Set(float(lt["intensity"]))
            UsdGeom.Imageable(lp).MakeVisible()
        else:
            light.GetIntensityAttr().Set(0.)
            UsdGeom.Imageable(lp).MakeInvisible()
    # the column's reflections (render.arc_light spec_chain): cylinder lights along the column (their local x axis)
    chain = ((arc or {}).get("light") or {}).get("spec_chain") or []
    k = 0
    while True:
        sp = stage.GetPrimAtPath(f"/World/Arc/spec_{k}")
        if not sp or not sp.IsA(UsdLux.CylinderLight):
            break
        cl = UsdLux.CylinderLight(sp)
        spec = chain[k] if k < len(chain) else None
        if spec:
            vals = [*spec["center"], *spec["axis"], spec["radius"], spec["length"], spec["intensity"]]
            if not np.isfinite(vals).all():
                raise ValueError("Non-finite arc reflection light")
            a_ = np.asarray(spec["axis"], float)
            a_ /= np.linalg.norm(a_)
            ex = np.array([1., 0., 0.])
            v = np.cross(ex, a_)
            s_, c_ = float(np.linalg.norm(v)), float(ex @ a_)
            if s_ < 1e-9:
                q_ = Gf.Quatf(1., Gf.Vec3f(0., 0., 0.)) if c_ > 0 else Gf.Quatf(0., Gf.Vec3f(0., 0., 1.))
            else:
                ang = math.atan2(s_, c_)
                v /= s_
                q_ = Gf.Quatf(float(math.cos(ang/2)), Gf.Vec3f(*map(float, v*math.sin(ang/2))))
            ops = UsdGeom.Xformable(sp).GetOrderedXformOps()
            ops[0].Set(Gf.Vec3d(*map(float, spec["center"])))
            ops[1].Set(q_)
            cl.GetRadiusAttr().Set(float(spec["radius"]))
            cl.GetLengthAttr().Set(float(spec["length"]))
            cl.GetIntensityAttr().Set(float(spec["intensity"]))
            UsdGeom.Imageable(sp).MakeVisible()
        else:
            cl.GetIntensityAttr().Set(0.)
            UsdGeom.Imageable(sp).MakeInvisible()
        k += 1
    if chain and k == 0:
        raise ValueError("The scene was exported by an older render.py (no arc reflection chain): re-export it")
    fol = request.get("follow")
    if fol:
        UsdGeom.Xformable(stage.GetPrimAtPath("/World/Follow")).GetOrderedXformOps()[0].Set(Gf.Vec3d(*map(float, fol)))
    # the tracking camera, aimed per frame by a film (render.TRACK; e.g. the E20 ram film on each ram's contact)
    tr = request.get("track")
    tp = stage.GetPrimAtPath("/Cameras/track")
    if tr and tp:
        vals = [*tr["eye"], *tr["target"]]
        if not np.isfinite(vals).all():
            raise ValueError("Non-finite tracking camera")
        m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, tr["eye"])), Gf.Vec3d(*map(float, tr["target"])),
                                    Gf.Vec3d(*map(float, tr.get("up", (0., 0., 1.)))))
        UsdGeom.Xformable(tp).GetOrderedXformOps()[0].Set(m.GetInverse())
        if tr.get("hfov"):
            cam = UsdGeom.Camera(tp)
            half = math.tan(math.radians(float(tr["hfov"]))/2)
            cam.GetFocalLengthAttr().Set(float(cam.GetHorizontalApertureAttr().Get())/2/half)
    for path, pts in (request.get("curves") or {}).items():
        H = np.asarray(pts, np.float32)
        if not np.isfinite(H).all():
            raise ValueError("Non-finite curve " + path)
        c = UsdGeom.BasisCurves(stage.GetPrimAtPath(path))
        if not c:
            raise ValueError("Missing curve " + path)
        c.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(H))
        c.GetCurveVertexCountsAttr().Set([len(H)])
    wire = request.get("wire")
    wx = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Wire"))
    if wire and wire.get("length_m", 0.) > 1e-4:
        tip, axis, L = np.asarray(wire["tip"], float), np.asarray(wire["axis"], float), float(wire["length_m"])
        mid = tip + axis*L/2
        z = np.array([0., 0., 1.])
        v = np.cross(z, axis)
        s, c = np.linalg.norm(v), float(np.dot(z, axis))
        if s < 1e-9:
            quat = Gf.Quatf(1., Gf.Vec3f(0, 0, 0))
        else:
            ang = np.arctan2(s, c)
            v /= s
            quat = Gf.Quatf(float(np.cos(ang/2)), Gf.Vec3f(*map(float, v*np.sin(ang/2))))
        ops = wx.GetOrderedXformOps()
        ops[0].Set(Gf.Vec3d(*map(float, mid)))
        ops[1].Set(quat)
        if len(ops) < 3:
            wx.AddScaleOp().Set(Gf.Vec3f(1., 1., L))
        else:
            ops[2].Set(Gf.Vec3f(1., 1., L))
        UsdGeom.Imageable(wx).MakeVisible()
    else:
        UsdGeom.Imageable(wx).MakeInvisible()


def main():
    from gym_anything_robotics.render_lifecycle import watch_parent
    from whiteboard.ipc import send, receive
    watch_parent()
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--samples", type=int, required=True)
    a = p.parse_args()
    connection = socket.create_connection(("127.0.0.1", int(os.environ["WIPE_RENDER_PORT"])))
    send(connection, {"secret": os.environ.pop("WIPE_RENDER_SECRET")})
    kit = Path(a.scene).parent/"kit"
    kit_log = kit/"worker.log"
    kit.mkdir(exist_ok=True)
    from isaacsim import SimulationApp
    import wiping
    app = SimulationApp(dict(headless=True, open_usd=a.scene, create_new_stage=False, renderer="PathTracing",
                             width=a.width, height=a.height, samples_per_pixel_per_frame=a.samples, max_bounces=8,
                             multi_gpu=False, active_gpu=0, disable_viewport_updates=True,
                             extra_args=["--portable", f"--portable-root={kit}", "--/app/enableCrashReporter=false",
                                         "--/exts/omni.kit.viewport.window/startup/disableWindowOnLoad=true",
                                         "--/app/vulkan=true", "--/rtx/materialDb/syncLoads=true",
                                         "--/rtx/hydra/materialSyncLoads=true",
                                         "--/rtx-transient/resourcemanager/texturestreaming/async=false",
                                         f"--/log/file={kit/'worker.log'}"]),
                        experience=str(Path(wiping.__file__).with_name("render_only.kit")))
    try:
        import carb
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import UsdGeom, UsdPhysics, UsdRender
        omni.usd.add_hydra_engine("rtx", omni.usd.get_context())
        settings = carb.settings.get_settings()
        # one fixed internal exposure for HdrColor (render.RENDER_EXPOSURE, sent with each request); each camera's own
        # exposure, glare and response are applied to it in camera_image. No auto-exposure.
        for key, value in {"/rtx/post/histogram/enabled": False, "/rtx/pathtracing/spp": a.samples,
                           "/rtx/pathtracing/totalSpp": a.samples, "/rtx/pathtracing/optixDenoiser/enabled": True,
                           # the firefly filter stays on (default clamps): off, the pool's caustics of the arc came
                           # out as denoised white smears that read as fume; on, it costs ~15-20 % of the arc's
                           # indirect light in the cell camera (measured)
                           "/rtx/pathtracing/fireflyFilter/enabled": True,
                           "/rtx/post/tonemap/op": 1, "/rtx/post/tonemap/filmIso": 100.,
                           "/rtx/post/tonemap/fNumber": 16., "/rtx/post/tonemap/exposureTime": 1/120.}.items():
            settings.set(key, value)
        dump_hdr = os.environ.get("WELD_RENDER_DUMP_HDR")
        for _ in range(12):
            app.update()
        omni.timeline.get_timeline_interface().pause()
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                cam = UsdGeom.Camera(prim)
                cam.GetVerticalApertureAttr().Set(cam.GetHorizontalApertureAttr().Get()*a.height/a.width)
        if any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.IsA(UsdPhysics.Scene) for p in stage.Traverse()):
            raise RuntimeError("Renderer must not contain a dynamics scene")
        rep.orchestrator.set_capture_on_play(False)
        product, annotator, active = None, None, None
        send(connection, {"phase": "ready"})
        while True:
            request = receive(connection)
            if request["op"] == "close":
                break
            try:
                if request["op"] != "observe":
                    raise ValueError("Unknown render operation")
                faulthandler.dump_traceback_later(120, repeat=False)
                t0 = time.perf_counter()
                apply_state(stage, request)
                # diagnostic switches (for test renders): the denoiser
                dbg = request.get("debug") or {}
                settings.set("/rtx/pathtracing/optixDenoiser/enabled", bool(dbg.get("denoiser", True)))
                settings.set("/rtx/pathtracing/fireflyFilter/enabled", bool(dbg.get("firefly", True)))
                timings = {"usd_update_s": time.perf_counter() - t0}
                images = {}
                pool_world = None
                for name in request["cameras"]:
                    path = "/Cameras/" + name
                    alias = stage.GetPrimAtPath(path)
                    if alias and alias.GetCustomDataByKey("mounted_camera_path"):
                        path = alias.GetCustomDataByKey("mounted_camera_path")
                    if not UsdGeom.Camera(stage.GetPrimAtPath(path)):
                        raise ValueError("Unknown camera " + name)
                    if product is None:
                        product = rep.create.render_product(path, (a.width, a.height))
                        annotator = rep.AnnotatorRegistry.get_annotator("HdrColor")
                        annotator.attach([product])
                        depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
                        depth.attach([product])
                        product.hydra_texture.set_updates_enabled(True)
                        rep.orchestrator.step(rt_subframes=8, delta_time=0.)
                    elif path != active:
                        UsdRender.Product(stage.GetPrimAtPath(product.path)).GetCameraRel().SetTargets([path])
                        product.hydra_texture.set_camera_path(path)
                    active = path
                    ex = request["exposure"][name]
                    rx = request["render_exposure"]
                    settings.set("/rtx/post/tonemap/exposureTime", float(rx["exposureTime"]))
                    settings.set("/rtx/post/tonemap/fNumber", float(rx["fNumber"]))
                    settings.set("/rtx/post/tonemap/filmIso", float(rx["filmIso"]))
                    k_render = float(request["rtx_k"])*float(rx["exposureTime"])*float(rx["filmIso"])/float(rx["fNumber"])**2
                    t1 = time.perf_counter()
                    log0 = kit_log.stat().st_size if kit_log.exists() else 0
                    hdr, _ = capture(lambda: rep.orchestrator.step(rt_subframes=2, delta_time=0.),
                                     annotator.get_data, a.width, a.height, flush_attempts=1)
                    render_failure(kit_log, log0)
                    timings[name + "_capture_s"] = time.perf_counter() - t1
                    dist = np.asarray(depth.get_data(), np.float32).reshape(a.height, a.width)
                    if dump_hdr:                     # debugging: scene luminance (cd/m2) and the depth AOV
                        Path(dump_hdr).mkdir(parents=True, exist_ok=True)
                        np.save(Path(dump_hdr)/f"{request['tick']}_{name}.npy",
                                (np.asarray(hdr, np.float32)/k_render).astype(np.float32))
                        np.save(Path(dump_hdr)/f"{request['tick']}_{name}_dist.npy", dist)
                    arc_fn = None
                    if (request.get("arc") and request["arc"].get("j0")) or request.get("fume"):
                        from pxr import Usd
                        cprim = stage.GetPrimAtPath(path)
                        cam_m = np.array(UsdGeom.Xformable(cprim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                        gc = UsdGeom.Camera(cprim)
                        # the column's mirror image in the pool is the path tracer's reflection of the specular
                        # cylinder light (render.arc_light); the analytic mirror composite would count it twice
                        geom = (cam_m, gc.GetHorizontalApertureAttr().Get(), gc.GetVerticalApertureAttr().Get(),
                                gc.GetFocalLengthAttr().Get())

                        def arc_fn(L_, d_=dist, g_=geom, pw=pool_world):
                            if request.get("arc") and request["arc"].get("j0"):
                                L_ = arc_composite(L_, d_, *g_, request["arc"], pw)
                            if request.get("fume"):
                                L_ = plume_composite(L_, d_, *g_, request["fume"])
                            return L_
                    if dump_hdr:                     # debugging: the luminance after the arc and fume composites
                        tag = (request.get("debug") or {}).get("tag", "")
                        Lc = np.maximum(np.nan_to_num(np.asarray(hdr, np.float32)[:, :, :3]/k_render, nan=0.), 0.)
                        if arc_fn is not None:
                            Lc = arc_fn(Lc)
                        np.save(Path(dump_hdr)/f"{request['tick']}_{name}{tag}_final.npy", Lc.astype(np.float32))
                        np.save(Path(dump_hdr)/f"{request['tick']}_{name}{tag}_dist.npy", dist)
                        if arc_fn is not None:
                            (Path(dump_hdr)/f"{request['tick']}_{name}{tag}_geom.json").write_text(json.dumps(dict(
                                cam_m=np.asarray(geom[0]).tolist(), hap=geom[1], vap=geom[2], focal=geom[3],
                                arc=request.get("arc"), fume=request.get("fume"), exposure=ex,
                                k_render=k_render, rtx_k=float(request["rtx_k"]))))
                        np.save(Path(dump_hdr)/f"{request['tick']}_{name}{tag}.npy",
                                (np.asarray(hdr, np.float32)/k_render).astype(np.float32))
                    t2 = time.perf_counter()
                    pixels = camera_image(hdr, ex, k_render, float(request["rtx_k"]), arc_fn)
                    timings[name + "_camera_model_s"] = time.perf_counter() - t2
                    timings[name + "_render_s"] = time.perf_counter() - t1
                    out = io.BytesIO()
                    Image.fromarray(pixels).save(out, format="PNG", compress_level=1)
                    images[name] = base64.b64encode(out.getvalue()).decode()
                send(connection, dict(ok=True, tick=request["tick"], sequence=request["sequence"], physics_steps=0,
                                      images=images, timings=timings))
            except Exception as exc:
                traceback.print_exc()
                send(connection, dict(ok=False, error=str(exc)))
            finally:
                faulthandler.cancel_dump_traceback_later()
        if product:
            annotator.detach()
            depth.detach()
            product.destroy()
    finally:
        connection.close()
        app.close()


if __name__ == "__main__":
    main()
