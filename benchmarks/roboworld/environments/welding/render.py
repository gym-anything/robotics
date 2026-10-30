"""Render-only view of the welding cell (Isaac Sim 5.1 RTX in a separate worker; welding/render_worker.py).

The backend process records a FRAME of physical state -- every rigid body's pose, the joint region's surface rebuilt
from the thermal solver's cells, its temperatures, the pool model's liquid profiles, the slices' porosity and air under
the arc, the arc (root, current, wire end) and the wire stick-out -- and the worker draws it. Nothing in the image is
animated or authored to look like welding: the bead is the cells' metal surface, the pool the pool model's free surface,
the glow each surface's blackbody radiance at its temperature, the arc a luminous column whose luminance follows the
arc current (measured arc luminances), pores and oxide the slices' frozen porosity and air fraction. Rendering never
steps physics.

Visual model (renderer units measured):
  joint band  y in [-3, 14] mm, z in [-3, 14] mm of the weld frame (the fillet, both toes, the heat-tinted margins):
              where there is weld metal, a fan of rays across each 0.5 mm slice drawn from the pool model's own
              profiles (bead_grid: each slice's free-surface polyline, pool.SliceState.pts or its height function;
              an overhanging profile by its outermost crossing; the cells' surface only where they hold metal no
              profile describes), smoothed along the weld below the model's own sharing length; elsewhere the exposed
              faces of metal cells, smoothed (cell size 0.25 mm). Colour by material and peak temperature (heat tint,
              grown as the metal cools in air, not under the torch's gas), emission = emissivity x blackbody luminance
              at the temperature beneath (CIE 1931 colour matching, Wyman et al. 2013 fit); oxide and soot from the air
              under the arc; open pores from the frozen porosity (Delesse rule).
  pool        the liquid part of that surface (the metal beneath at liquid fraction >= 0.5: its outline is the thermal
              field's fusion and solidification fronts): a liquid-iron surface (reflectance 1 - emissivity of liquid
              iron) emitting at its temperature, glossy while the arc's droplets keep it moving.
  coupon      outside the band, the plate and bracket are boxes (the band carves them out); a loose bracket whose cells
              have left the grid is drawn whole on its rigid pose.
  arc         a luminous column from the wire's end to the arc root (an emission density with an edge where the plasma
              stops conducting, composited at its measured luminance, ARC_LB, up to the first surface in front of it);
              it lights the scene through a diffuse-only sphere light of its luminous intensity inside a specular-only
              cylinder light along it that carries its reflections; flicker not drawn.
  fume        a buoyant point-source plume in single scattering, turbulent and drawn toward the canopy (worker).
  cameras     HdrColor -> each camera's exposure, lens veiling glare and response (sRGB standard cameras; HDR
              logarithmic process cameras), render_worker.camera_image; a tracking camera a film aims per frame.
"""
from gym_anything_robotics.paths import worker_env

import base64
import json
import math
import os
import secrets
import select
import socket
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import layout as LY
from . import torch as TO

ROOT = Path(__file__).resolve().parent
BAND = dict(y=(-3e-3, 14e-3), z=(-3e-3, 14e-3))        # weld frame (m)
# ---- the arc's light. Renderer units, measured: a UsdLux light's `intensity` is its surface
# luminance in cd/m2 (sphere, disk, rect and cylinder lights within 1-7 % of the photometric expectation); OmniSurface
# emission_nt is cd/m2; OmniPBR's emissive_intensity is an exitance (luminance x pi).
# Arc luminance from the effective blue-light radiance L_B of GMAW arcs on mild steel vs current (Nakashima, Takahashi,
# Fujii & Okuno 2024, Ind. Health 62(2):79-89, doi 10.2486/indhealth.2023-0038: 1.2 mm solid wire, 80 % Ar / 20 % CO2,
# steady current, spectroradiometer at 2 m, a 3.3 mm field centred on the arc). (current A, L_B W/cm2/sr): 100 A from the
# abstract, 200-300 A back-calculated from their Table 3 exposure times, 350 A the abstract maximum [derived].
ARC_LB = ((100., 5.0), (200., 25.), (250., 39.), (300., 72.), (350., 118.))
# luminance per unit blue-light radiance of this arc's spectrum: the 250 A spectrum of their Fig. 3b weighted by V(lambda)
# gives 2.0e8 cd/m2 for 39 W/cm2/sr (and by the blue-light function 39.8, a check of the digitisation) [derived,
# +-30 %]: 513 cd/W, applied at every current (the Fe-line spectrum's shape is taken as fixed) [estimate].
ARC_LV_PER_LB = 513.
# The column's light: an optically thin plasma drawn as an emission density j (luminance per unit path, cd/m3) about the
# axis from the wire's end (t = 0) to the arc root (t = L_a), and ARC_UP up the wire's tapered tip (spray: the arc
# envelops the conical wire end, NISTIR 3976):
#   j(t, rho) = j0 [ a(t) exp(-rho^2 / (2 s_c(t)^2))                                the metal-vapour core
#                    + ARC_ENV exp(-rho^2 / (2 s(t)^2)) ],                          the column's argon envelope,
#   a(t) = ARC_CORE_FLOOR + (1 - ARC_CORE_FLOOR) exp(-|t| / ARC_CORE): the core runs the whole column from the wire's end
#   to the work, brightest just below the wire (a core decaying over 1.2 mm draws the arc as a blob at the wire's end
#   with no column to the work). Basis: Hertel, Fuessel & Schnick 2009 (GMA arc model with
#   metal vapour, TU Dresden; their Fig. 4, 250 A): the iron-vapour-dominated core reaches from the wire tip to the
#   workpiece, widening toward it, and it is the vapour's "high radiation emission" that cools it; their temperature
#   maximum lies just under the wire [read: figure and text]. The floor and widths are estimates.
#   s_c(t) = ARC_S_CORE at the wire end -> ARC_CORE_ROOT x sigma(I) at the root; s(t) = ARC_S_WIRE at the wire end ->
#   sigma(I) at the root (params.ARC_PRESSURE_SIGMA, the arc's current-density spread, Kim 2003 fit to Lin & Eagar,
#   read: 1.80 mm at 251 A); both fade above the wire end as a Gaussian of ARC_UP / 3 (ARC_UP: where they are down to 1 %).
# Published images of spray arcs show "a very bright arc" with a compact core at the wire end and a bell-shaped glow.
# j0 is set so that the mean apparent luminance over a 3.3 mm disc centred on the column, seen
# side-on, equals the measured luminance (arc_luminance): the measurement's own geometry. The apparent luminance of a pixel
# is the integral of j along its ray up to the first opaque surface (render_worker.arc_composite), so the wire tip inside
# the column and anything in front of it occlude it physically.
ARC_S_WIRE = .7e-3              # m, the envelope's spread at the wire end [estimate: wire radius 0.45 mm + anode sheath]
ARC_S_CORE = .45e-3             # m, the core's spread at the wire end: the wire's radius [estimate]
ARC_CORE_ROOT = .5              # the core's spread at the root as a fraction of the current spread sigma(I) [estimate:
                                # Hertel et al. 2009 Fig. 4, the vapour core widens toward the work]
ARC_CORE = 1.2e-3               # m, decay length of the core's excess brightness below the wire tip [estimate]
ARC_CORE_FLOOR = .5             # the core's brightness along the column relative to just below the wire [estimate]
ARC_ENV = .10                   # the envelope's emission density relative to the core's peak [estimate]
ARC_UP = 1.0e-3                 # m, how far the luminous region climbs the tapered wire tip [estimate]
ARC_LIGHT_R = .5e-3             # m: the scene is lit by ONE sphere light of this radius at the column's luminous centroid,
                                # carrying the column's whole luminous intensity (int j dV, isotropic for a thin source),
                                # invisible to the cameras (the worker composites the seen arc). Inside the reflection
                                # chain below: a light hit by a reflected ray ends it (measured: a light enclosed by
                                # another is not seen in reflections), so at 1.5 mm this diffuse-only sphere would hide
                                # the column's middle from every mirror; its diffuse light does
                                # not depend on the radius at the distances lit (> 3 mm) [authored]
# The column's edge. The plasma emits (and conducts) only where it is hot: argon's visible and near-infrared lines come
# from levels ~13 eV above the ground state, so their emission falls as exp(-13 eV / kT), by ~4 orders of magnitude
# between the column's ~15 000 K and 8 000 K, the temperature below which argon stops conducting [recalled textbook
# values, not re-checked]. The luminous column therefore ends where the current does, not in a Gaussian tail: each
# component is cut by exp(-(r / (ARC_EDGE s))^ARC_EDGE_P), ARC_EDGE spreads out (a Gaussian current holds 95 % within
# 2.45 sigma) [estimate]. (Gaussian tails, 1e-3 of the axis at 3.7 spreads, draw a ~14 mm glow around the root on the
# log weld camera; the 3.3 mm field of the measured anchor is within 1 spread of the axis and is unaffected.)
ARC_EDGE = 2.5
ARC_EDGE_P = 6.
# The column's reflections: ARC_SPEC_N cylinder lights end to end from the wire end to the root, each carrying its
# segment's luminous intensity (core and envelope, broadside L = dI / (2 R dt)), specular only, of radius
# ARC_SPEC_R_SPREAD x the segment's light-weighted spread (sqrt of the intensity-weighted mean of the core's and the
# envelope's s^2). One cylinder (N = 1): chains of short ones lose light in this renderer (measured:
# six end-to-end cylinders gave 75 % of one long one on a mirror plate, and in the weld scene six 1 mm segments 23 % of
# one 6 mm cylinder; nested cylinders hide the inner one from reflections). The diffuse sphere light lies inside it
# (ARC_LIGHT_R): a 1.5 mm sphere around a 1 mm cylinder hides the column's middle from every mirror.
ARC_SPEC_N = 1
ARC_SPEC_R_SPREAD = 1.5         # [authored: a Gaussian is down to a third of its peak at 1.5 spreads]
ARC_COLOR = (.86, .91, 1.)      # linear RGB, luminance-normalised below: the arc's white-blue [estimate; the measured
                                # spectrum is blue-violet heavy]
EMISSIVITY = .7                 # steel (params.STEEL_EMISSIVITY)
EMISSIVITY_LIQUID = .39         # normal spectral emissivity of liquid iron at 807 nm, 0.394 +- 0.012 over 1600-1950 K
                                # [read via a search abstract; the visible value is similar (650-800 nm measurements
                                # exist, Watanabe et al. 2003): estimate for the visible]; reflectance 1 - eps (opaque)
# While the arc burns the pool's surface is in motion: the droplet stream (the arc model's melt rate in drops of about the
# wire's diameter: ~350 per s at 250 A, derived) and the arc pressure raise capillary waves (~2 mm long in liquid steel
# they run at ~0.9 m/s and decay over ~60 ms: they cross the 20 mm pool many times, derived from gamma 1.8 N/m, rho 7000
# kg/m3, mu 6 mPa s); a camera sees the liquid as a glossy surface, its reflection of the column a soft bright lobe (the
# Xiris GMAW frame: a uniformly bright pool), most disturbed near the arc root. With the arc off the waves die out and the
# liquid is the mirror LIQUID_ROUGHNESS [estimates: roughness values and radius]. (A calm mirror pool shows the column's
# specular stand-in as a sharp bright rectangle.)
POOL_RIPPLE_ROUGHNESS = .35     # liquid under a burning arc
POOL_AGITATED_ROUGHNESS = .45   # at the arc root
POOL_AGITATED_R = 3.0e-3        # m, Gaussian radius about the arc root
LIQUID_ROUGHNESS = .04          # the pool's free surface: a liquid metal mirror rippled by the arc [estimate]
POOL_EDGE_FL = (.45, .55)       # liquid fraction over which the drawn surface turns from solid to liquid [authored]
MILL_SCALE_SRGB = (62, 64, 68)  # hot-rolled A36 mill scale: dark blue-grey [estimate from photographs]
WELD_SRGB = (100, 98, 95)       # as-welded ER70S-6 bead under 90/10 gas: grey with a thin oxide film [estimate]
GROUND_SRGB = (150, 152, 156)   # mill scale ground off with a flap disc: bright steel [estimate]
# the fitter grinds the mill scale off the joint before tacking, about an inch either side (common shop practice for
# GMAW on hot-rolled plate) [estimate]. Appearance only: the physics does not distinguish scale from bare steel.
GROUND_W = .025
# temper (oxide interference) colours of bare carbon steel by the peak temperature it reached: straw, brown, purple,
# blue, grey. Tables give them for minutes at temperature (straw 220-240 C ... blue 300-320 C); a weld's heat-affected
# metal is hot for seconds, so the bands sit higher; shifted here by about 50 C [estimate]. On mill scale only a
# darkening shows. On a ground surface the films are thin and patchy, so the colours are drawn at TEMPER_SATURATION of
# the chart's saturation [estimate: a first render at full chart colour read as an oil-slick ring]. (peak C, sRGB)
TEMPER_SATURATION = .55
# The films grow while the metal is hot in air; a point still heating (ahead of and beside the arc, under the gas) has
# not grown its film yet. The colour of its peak temperature is drawn in full once the metal has cooled TINT_DEV_K below
# that peak, and in proportion before (colours drawn on reaching the peak make a thin rainbow ring run ahead of the arc
# in the weld camera; real heat tint shows as bands beside and behind the bead)
# [estimate: a proxy for the oxide's growth over the cooling time, which the solver does not record].
TINT_DEV_K = 100.
# ... and only in air. Under the torch's shielding gas the hot metal does not oxidise; its films form once the gas has
# moved on and the metal, still hot, meets the air (films drawn on metal under the gas around the pool, where the arc's
# reflection lights them, show as rainbow rings in the weld camera). The gas covers the work within SHIELD_R[0] of the
# wire's axis fully and fades to none at SHIELD_R[1] (the nozzle's bore radius and diameter, params.NOZZLE_ID) while the
# nozzle's end is within SHIELD_REACH of the surface, diluted by the process's own air fraction at the pool (x_air: a
# draft, a long stand-off, the valve shut) [estimates]. A film already grown on cold metal stays (metal below ~150-250 C
# grows no film in seconds).
SHIELD_R = (7.95e-3, 15.9e-3)
SHIELD_REACH = (25e-3, 35e-3)   # m along the wire axis from the contact tip: full cover, none [estimate]
FILM_T = (150., 250.)           # C: below the first, the shield does not matter (no growth); above the second, it gates
TEMPER = ((260., GROUND_SRGB), (290., (206, 186, 128)), (330., (188, 140, 72)), (370., (142, 84, 64)),
          (400., (112, 64, 118)), (430., (64, 84, 160)), (480., (104, 132, 168)), (600., (98, 102, 108)),
          (900., (62, 60, 60)))
SILICATE_SRGB = (92, 66, 40)    # glassy silicate islands left on an ER70S-6 bead by its Si/Mn deoxidisers [estimate]
SPATTER_STICK = .02             # fraction of the spatter volume (arc.py ledger) drawn as balls stuck on the coupon
                                # [estimate: most spatter flies clear; render-only]
SPATTER_R = (.15e-3, .5e-3)     # stuck spatter ball radii, log-uniform [estimate]
# ---- porosity and oxidation, drawn from the pool's slice state (process.pool.slices[i]: porosity, x_air_max).
# Pores: a slice's frozen pore volume fraction phi is drawn as open pores covering the area fraction phi of its bead
# surface, and of its weld metal in the macro sections (E4 figure). Rule: Delesse's principle (in a
# random plane section the area fraction of a phase equals its volume fraction) [derived]; the free surface is taken as
# such a section, a lower bound for surface-breaking pores, which float up [estimate]. Positions and sizes are a fixed
# hash of (slice, pore), so pores never move between frames; they appear when the slice freezes (porosity is set then).
PORE_D = (.5e-3, 1.6e-3)        # pore diameters, log-uniform [estimate: visible surface gas pores of GMAW beads]
# Oxidation: the air under the burning arc while a slice was liquid (x_air_max) oxidises its bead: none up to 1 % air
# (TWI's onset of porosity, the same shielding failure; read), complete with no shielding at all (air fraction 1;
# 0.21 in the physics before the gas-valve change), log-interpolated between [estimate]. Appearance of an unshielded
# solid-wire bead: dull black scale with a brown-grey sooty deposit beside it [estimate from descriptions].
OX_AIR = (.01, 1.)
OXIDE_SRGB = (34, 31, 29)       # black magnetite / wustite scale [estimate]
SOOT_SRGB = (60, 52, 45)        # brown-grey iron-oxide fume deposit beside the bead [estimate]
SOOT_W = .010                   # e-folding distance of the deposit from the joint, beyond the bead [estimate]




# ------------------------------------------------------------------------------------------------ colour science
def _cmf(lam_nm):
    """CIE 1931 2 deg colour matching functions, multi-lobe Gaussian fit (Wyman, Sloan & Shirley 2013)."""
    l = np.asarray(lam_nm, float)

    def g(x, mu, s1, s2):
        s = np.where(x < mu, s1, s2)
        return np.exp(-.5*((x - mu)/s)**2)
    x = 1.056*g(l, 599.8, 37.9, 31.0) + .362*g(l, 442.0, 16.0, 26.7) - .065*g(l, 501.1, 20.4, 26.2)
    y = .821*g(l, 568.8, 46.9, 40.5) + .286*g(l, 530.9, 16.3, 31.1)
    z = 1.217*g(l, 437.0, 11.8, 36.0) + .681*g(l, 459.0, 26.0, 13.8)
    return x, y, z


@lru_cache(maxsize=1)
def blackbody_table():
    """(T_C grid, luminance cd/m2 of a blackbody, linear sRGB chromaticity normalised to luminance 1)."""
    h, c, kB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    lam = np.arange(380., 781., 2.)
    x, y, z = _cmf(lam)
    Tc = np.arange(400., 3201., 10.)
    L, rgb = [], []
    M = np.array([[3.2406, -1.5372, -.4986], [-.9689, 1.8758, .0415], [.0557, -.2040, 1.0570]])
    for t in Tc + 273.15:
        lm = lam*1e-9
        B = 2*h*c*c/lm**5/(np.exp(h*c/(lm*kB*t)) - 1.)            # W / (m2 sr m)
        X, Y, Z = (np.trapezoid(B*f, lm) for f in (x, y, z))
        L.append(683.*Y)
        v = M @ np.array([X, Y, Z])/max(Y, 1e-30)
        rgb.append(np.clip(v, 0, None))
    return Tc, np.array(L), np.array(rgb)


def glow(T, emissivity=None):
    """Per-vertex emission (linear RGB x luminance cd/m2) of steel at temperatures T (C)."""
    Tc, L, rgb = blackbody_table()
    T = np.asarray(T, float)
    lum = (EMISSIVITY if emissivity is None else emissivity)*np.interp(T, Tc, L, left=0.)
    col = np.stack([np.interp(T, Tc, rgb[:, k], left=rgb[0, k]) for k in range(3)], 1)
    return col*lum[:, None]


def srgb_to_linear(c):
    c = np.asarray(c, float)/255.
    return np.where(c <= .04045, c/12.92, ((c + .055)/1.055)**2.4)


def _hash01(i, j, k, seed):
    h = (i*73856093) ^ (j*19349663) ^ (k*83492791) ^ (seed*2654435761)
    h = (h ^ (h >> 13))*1274126177
    return ((h ^ (h >> 16)) & 0xFFFFFF)/float(0xFFFFFF)


def value_noise(P, cell, seed):
    """Smooth value noise in [-1, 1] at points P (m) on a lattice of `cell` m (trilinear between hashed corners)."""
    q = np.asarray(P, float)/cell
    i0 = np.floor(q).astype(np.int64)
    f = q - i0
    f = f*f*(3 - 2*f)
    out = np.zeros(len(q))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0])*(f[:, 1] if dy else 1 - f[:, 1])*(f[:, 2] if dz else 1 - f[:, 2])
                out += w*_hash01(i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz, seed)
    return 2*out - 1


def _temper(peakT):
    T = np.asarray(peakT, float)
    Tc = np.array([t for t, _ in TEMPER])
    rgb = np.array([srgb_to_linear(c) for _, c in TEMPER])
    lum = rgb @ np.array([.2126, .7152, .0722])
    rgb = lum[:, None] + TEMPER_SATURATION*(rgb - lum[:, None])
    return np.stack([np.interp(T, Tc, rgb[:, k]) for k in range(3)], 1)


def film_air(P, T, shield):
    """Per point: the share of air the hot metal's surface meets now (1 outside the gas cover, x_air under it), the
    weight of the temper films' growth (TINT_DEV_K, SHIELD_R); 1 everywhere without a shield (shield None: no torch
    information). shield: dict(tip, axis (weld frame), x_air) from world_frame."""
    P = np.asarray(P, float)
    if shield is None:
        return np.ones(len(P))
    tip, ax = np.asarray(shield["tip"], float), np.asarray(shield["axis"], float)
    v = P - tip
    along = v @ ax
    rad = np.linalg.norm(v - along[:, None]*ax[None], axis=1)
    cover = (1. - np.clip((rad - SHIELD_R[0])/(SHIELD_R[1] - SHIELD_R[0]), 0., 1.))
    cover = cover*(1. - np.clip((along - SHIELD_REACH[0])/(SHIELD_REACH[1] - SHIELD_REACH[0]), 0., 1.))*(along > -5e-3)
    air = 1. - cover*(1. - float(np.clip(shield.get("x_air", 0.), 0., 1.)))
    if T is None:
        return air
    hot = np.clip((np.asarray(T, float) - FILM_T[0])/(FILM_T[1] - FILM_T[0]), 0., 1.)
    return 1. - hot*(1. - air)


def surface_look(P, peakT, weld, ox=None, soot=None, T=None, shield=None):
    """Base colour (linear), roughness and metalness of the coupon's surface at weld-frame points P (m): mill scale with
    a mottled oxide, the ground strip along the joint, the temper colours of the peak temperature reached (grown as the
    metal cools in air: TINT_DEV_K, film_air), and the bead (weld metal, `weld` in [0, 1]) with its silicate islands.
    `ox` (per point, 0-1): the oxidation of a bead laid with air under the arc (slice_states); `soot` (0-1): the oxide
    fume deposit on the parent metal beside such a bead. Render-only appearance [estimates above]."""
    col, rough, metal = _surface_look(P, peakT, weld, T, film_air(P, T, shield) if shield is not None else None)
    if ox is None and soot is None:
        return col, rough, metal
    w = np.clip(np.asarray(weld, float), 0, 1)
    a = np.clip(np.asarray(ox if ox is not None else np.zeros(len(col)), float), 0, 1)*w
    s = np.clip(np.asarray(soot if soot is not None else np.zeros(len(col)), float), 0, 1)*(1 - w)
    mott = (1 + .25*value_noise(np.asarray(P, float), .0012, 7))[:, None]          # a crusty, uneven scale
    col = col*(1 - a)[:, None] + (srgb_to_linear(OXIDE_SRGB)*mott)*a[:, None]
    col = col*(1 - s)[:, None] + srgb_to_linear(SOOT_SRGB)*s[:, None]
    rough = rough*(1 - a - s) + .85*(a + s)
    metal = metal*(1 - a - s) + .12*(a + s)
    return col.astype(np.float32), rough.astype(np.float32), metal.astype(np.float32)


def slice_states(proc, x):
    """Per point at weld-frame x (m): (oxidation 0-1 of the bead, soot 0-1 along x, porosity) from the pool's slices.
    Oxidation: log-interpolated air fraction under the burning arc while the slice was liquid (OX_AIR); the soot follows
    the same value smoothed 3 mm along the weld (the fume drifts) [estimates above]."""
    g = proc.g
    sl = proc.pool.slices
    xam = np.array([float(getattr(s, "x_air_max", 0.)) for s in sl])
    phi = np.array([float(getattr(s, "porosity", 0.)) for s in sl])
    lo, hi = OX_AIR
    ox = np.clip(np.log(np.maximum(xam, 1e-12)/lo)/math.log(hi/lo), 0, 1)
    if ox.any():
        from scipy.ndimage import gaussian_filter1d
        soot = np.clip(gaussian_filter1d(ox, 3e-3/float(np.median(g.dx))), 0, 1)
    else:
        soot = ox
    i = np.clip(np.searchsorted(g.xe, np.asarray(x, float)) - 1, 0, g.nx - 1)
    return ox[i], soot[i], phi[i]


def soot_weight(P):
    """Fraction of the full soot deposit on the parent metal at weld-frame points: 1 within 7 mm of the root (under and
    at the bead's toes), falling off with SOOT_W beyond, on the weld side only [estimate]."""
    P = np.asarray(P, float)
    d = np.maximum(P[:, 1], P[:, 2])
    side = (P[:, 1] > -1e-3) & (P[:, 2] > -1e-3)
    return np.exp(-np.maximum(d - 7e-3, 0.)/SOOT_W)*side


def _surface_look(P, peakT, weld, T=None, air=None):
    P = np.asarray(P, float)
    pk = np.asarray(peakT, float)
    # temper colours as far as the film has grown (TINT_DEV_K, in air: film_air): the colour of the peak reached, faded
    # in from the untinted ground colour by the growth. (Sweeping the drawn peak itself from 260 C up to the peak would
    # draw metal heated far above the colour bands, e.g. 1400 C at the pool's edge, straw, purple and blue on its way,
    # a rainbow ring round the hot zone. A film grown hot passes those colours in milliseconds.)
    grow = np.ones(len(pk)) if T is None else np.clip((pk - np.asarray(T, float))/TINT_DEV_K, 0., 1.)
    if air is not None:
        grow = grow*np.asarray(air, float)
    w = np.clip(np.asarray(weld, float), 0, 1)[:, None]
    y, z = P[:, 1], P[:, 2]
    edge = GROUND_W + .003*value_noise(P*[1, 0, 0], .012, 5)
    g = np.clip((edge - np.maximum(y, z))/.004, 0, 1)*((y > -1e-3) & (z > -1e-3))
    g = g[:, None]
    mott = 1 + .14*value_noise(P, .018, 1) + .07*value_noise(P, .005, 2)
    mill = srgb_to_linear(MILL_SCALE_SRGB)*mott[:, None]
    mill = mill*(1 - .45*np.clip((pk - 500.)/600., 0, 1)*grow)[:, None]     # scale darkens through the HAZ
    bare = _temper(np.minimum(pk, 260.))
    ground = (bare + (_temper(pk) - bare)*grow[:, None])*(1 + .05*value_noise(P*[.2, 1, 1], .002, 3))[:, None]
    base = mill*(1 - g) + ground*g
    isl = np.clip((value_noise(P, .0035, 4) - .55)/.12, 0, 1)[:, None]      # silicate islands on the bead
    bead = srgb_to_linear(WELD_SRGB)*(1 + .08*value_noise(P, .0015, 6))[:, None]
    bead = bead*(1 - isl) + srgb_to_linear(SILICATE_SRGB)*isl
    col = base*(1 - w) + bead*w
    # the films grown above ~500 C are an oxide scale (grey to black), dull rather than a thin interference film on
    # bright metal: the ground strip's gloss fades with the drawn peak from 480 to 900 C (a glossy grey halo inside the
    # temper bands reads as glass around the tacks) [estimate]
    ox_t = (np.clip((pk - 480.)/420., 0., 1.)*grow)[:, None]
    g_rough = .38*(1 - ox_t) + .7*ox_t
    g_metal = .85*(1 - ox_t) + .35*ox_t
    rough = (.66*(1 - g) + g_rough*g)[:, 0]*(1 - w[:, 0]) + (.52*(1 - isl[:, 0]) + .2*isl[:, 0])*w[:, 0]
    metal = (.25*(1 - g) + g_metal*g)[:, 0]*(1 - w[:, 0]) + (.7*(1 - isl[:, 0]) + .1*isl[:, 0])*w[:, 0]
    return col.astype(np.float32), rough.astype(np.float32), metal.astype(np.float32)


def heat_tint(peakT):
    """Surface colour (linear) of mill-scaled parent steel by its peak temperature (kept for callers outside the band:
    the look of surface_look away from the ground strip)."""
    P = np.zeros((len(np.atleast_1d(peakT)), 3)) + [0., -1., -1.]
    return surface_look(P, peakT, np.zeros(len(P)))[0]


def peak_at(proc, P, N, field="peakT"):
    """Peak temperature (or another cell field) of the cell under each surface point (weld frame P, outward normal N)."""
    g = proc.g
    q = np.asarray(P, float) - np.asarray(N, float)*1e-4
    i = np.clip(np.searchsorted(g.xe, q[:, 0]) - 1, 0, g.nx - 1)
    j = np.clip(np.searchsorted(g.ye, q[:, 1]) - 1, 0, g.ny - 1)
    k = np.clip(np.searchsorted(g.ze, q[:, 2]) - 1, 0, g.nz - 1)
    return proc.s.field(field).reshape(g.nx, g.ny, g.nz)[i, j, k]


def spatter_balls(proc, extent_x):
    """Stuck spatter on the coupon (render-only): SPATTER_STICK of the ledger's spatter volume as balls whose places
    follow a fixed random sequence (so balls never jump between frames), shown once the weld has reached their x.
    Returns (points, tris) of flattened low-poly balls in the weld frame, or None."""
    V = float(proc.src.st.spatter_m3)*SPATTER_STICK
    r0, r1 = SPATTER_R
    mean_v = 4/3*math.pi*(r1**3 - r0**3)/(3*math.log(r1/r0))
    n = int(V/mean_v)
    if n <= 0 or extent_x is None:
        return None
    rng = np.random.default_rng(20261008)
    M = 4000
    x = rng.uniform(-.005, proc.c.length + .005, M)
    side = rng.random(M) < .25
    # distance from the root: exponential from 2 mm (balls inside the bead are hidden by it); starting at 7.5 mm put
    # the nearest balls on one line along each toe, a dotted line in the renders [estimate]
    d = .002 + rng.exponential(.009, M)
    r = np.exp(rng.uniform(math.log(r0), math.log(r1), M))
    lo, hi = extent_x
    ok = (x >= lo - .01) & (x <= hi + .01)
    sel = np.nonzero(ok)[0][:n]
    if not len(sel):
        return None
    # an octahedron-based ball, flattened by a third where it splashed
    U = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    T = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    pts, tris = [], []
    for m_, k in enumerate(sel):
        c = np.array([x[k], 0., d[k]]) if side[k] else np.array([x[k], d[k], 0.])
        sq = np.array([1, .67, 1]) if side[k] else np.array([1, 1, .67])
        pts.append(c + U*r[k]*sq)
        tris.append(T + 6*m_)
    return np.concatenate(pts), np.concatenate(tris)


# ------------------------------------------------------------------------------------------------ spatter in flight
# The arc model's ledger books (1 - deposition efficiency) of the melted wire as spatter (arc.py: st.spatter_m3). Part
# of that loss is fume (fume_rate below: NIOSH's 3.5 mg per g of wire in spray, ~3.6 mg/s at 251 A; read). The rest
# leaves the arc as droplets, drawn in flight at their instantaneous positions (no motion blur): sizes, speeds and
# directions [estimates below], glowing at the droplet temperature (2673 K, params.DROPLET_T, read) less radiative
# cooling in flight (~2000 K/s for a 0.5 mm droplet: eps sigma T^4 A / (m c) at 2600 K, derived). A droplet leaves the
# picture when it reaches the plate or the bracket (the stuck balls are spatter_balls) or after FLIGHT_MAX_S.
FLY_D = (.2e-3, 1.0e-3)         # droplet diameters, log-uniform [estimate: spray-transfer spatter is fine]
FLY_V = (1., 4.)                # m/s, log-uniform [estimate]
FLY_SPREAD = 60.                # deg about the joint's open bisector [estimate]
FLIGHT_MAX_S = .15
FLY_COOL = 2000.                # K/s [derived above]


def spatter_in_flight(proc):
    """Droplets in flight now (weld frame): (points, tris, normals, glow) of low-poly balls, or None. Droplet n leaves the
    arc root at t_n = n / rate (rate from the arc's melt rate, the ledger's loss share less fume_rate); its size, speed and
    direction come from a generator seeded by n, so each droplet keeps its path between frames."""
    from . import params as PR
    src, st = proc.src, proc.src.st
    if not st.arc or proc.root is None:
        return None
    I = float(st.current)
    melt = src.a*I + src.b*float(st.l_s)*I*I                       # m/s of wire
    eff = PR.v(PR.DEPOSITION_EFF)                                   # by transfer mode (arc.py)
    loss = melt*src.aw*(1 - eff.get(str(st.mode or "spray"), eff["spray"]))   # m3/s
    rho = PR.v(PR.STEEL_RHO)
    drops = max(loss - fume_rate(proc)/rho, 0.)
    d0, d1 = FLY_D
    mean_v = math.pi/6*(d1**3 - d0**3)/(3*math.log(d1/d0))
    rate = drops/mean_v                                             # droplets per second
    if rate <= 0:
        return None
    t = float(proc.time)
    n_now = int(math.floor(t*rate))
    root = np.asarray(proc.root, float)
    bis = np.array([0., 1., 1.])/math.sqrt(2.)
    e1 = np.array([1., 0., 0.])
    e2 = np.cross(bis, e1)
    g = np.array([0., 0., -9.81])
    U = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    Tt = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    pts, tris, nrm, gl = [], [], [], []
    T0 = PR.v(PR.DROPLET_T)                                        # C (params stores 2673 K as 2400 C)
    for n in range(n_now, max(n_now - int(FLIGHT_MAX_S*rate) - 1, -1), -1):
        age = t - n/rate
        if age < 0 or age > FLIGHT_MAX_S:
            continue
        rng = np.random.default_rng([20261010, n])
        d = math.exp(rng.uniform(math.log(d0), math.log(d1)))
        v = math.exp(rng.uniform(*np.log(FLY_V)))
        th = math.radians(FLY_SPREAD)*math.sqrt(rng.random())
        ph = 2*math.pi*rng.random()
        u = math.cos(th)*bis + math.sin(th)*(math.cos(ph)*e1 + math.sin(ph)*e2)
        c = root + u*v*age + .5*g*age*age
        if (c[2] < 0 and c[1] > 0) or (c[1] < 0 and c[2] > 0) or (c[1] < 0 and c[2] < 0):
            continue                                                # reached the plate or the bracket
        k = len(pts)
        pts.append(c + U*d/2)
        nrm.append(U)
        tris.append(Tt + 6*k)
        gl.append(np.tile(glow(np.array([T0 - FLY_COOL*age]), EMISSIVITY_LIQUID)[0], (6, 1)))
    if not pts:
        return None
    return np.concatenate(pts), np.concatenate(tris), np.concatenate(nrm), np.concatenate(gl)


# ------------------------------------------------------------------------------------------------ the fume plume
# The fume rises from the arc as a buoyant plume into the extraction canopy. It is drawn by the worker's camera model
# (render_worker.plume_composite) as an optically thin participating medium in single scattering: along each camera
# ray up to the first opaque surface (depth AOV), the background is attenuated by exp(-tau) and light is scattered in
# from the arc (a point source of the column's luminous intensity, isotropic phase, shadowed by the gun) and from the
# shop's ambient light. The path tracer's own medium (an index-matched closed mesh with MDL volume coefficients) was
# tried first and renders Beer-Lambert correctly, but its boundary blocks the light sampling of the arc
# inside it: the cell camera's arc-lit crop fell from 91.6 to 43.1 grey and the depth AOV stopped at the plume's
# surface, hiding the arc (measured).
# Source: fume formation per gram of wire melted, NIOSH chamber (Keane, Siert, Chen & Stone 2014, Ann. Occup. Hyg.
# 58(4):403, Table 2, ER70S-3 1.14 mm): axial spray in Ar/CO2 90/10 3.5 mg/g (212 mg/min = 3.5 mg/s at 230-260 A);
# globular 4.3 mg/g and short circuit 2.3 mg/g, both in 75/25 [read], applied to the arc model's
# own melt rate (0.9 mm wire at 251 A: ~3.7 mg/s), by the transfer mode the arc model reports (with the gas valve shut
# the arc burns in air and the model's mode is globular). An arc in air: FUME_PER_G_AIR, 6.4 mg/g [estimate; proxy: the
# most oxidising shield measured, globular in 100 % CO2, Keane 2014 Table 2; no measurement for GMAW without gas was
# found], approached linearly with the air fraction at the pool (process x_air: 0 under full shielding, 1 with the valve
# shut).
FUME_PER_G = {"spray": 3.5e-3, "globular": 4.3e-3, "short_circuit": 2.3e-3}
FUME_PER_G_AIR = 6.4e-3
# Plume: Morton, Taylor & Turner's point-source plume, top-hat form, entrainment alpha = 0.1: radius b = 6 alpha z / 5,
# velocity w = (75 F / (144 pi alpha^2))^(1/3) z^(-1/3), buoyancy flux F = g Q / (rho c_p T) [recalled textbook result,
# derived here; not re-checked against a source]; Q the arc's heat not entering the work, (1 - eta) V I, half of it
# convected [estimate]; a virtual origin below the arc so the plume starts as wide as the gas nozzle's bore (15.9 mm,
# params.NOZZLE_ID: the fume leaves the shielding envelope over its footprint) [estimate]. The Gaussian profile of the
# top-hat plume (b_G = b / sqrt 2, centre concentration twice the top-hat value: equal fluxes) is drawn.
# Optics: mass extinction coefficient 2000 m2/kg and single-scattering albedo 0.55 (small iron-oxide agglomerates
# in the Rayleigh limit with a refractive index 2.42 + 0.5 i: real part of magnetite read, imaginary part estimate;
# ~150 nm aged agglomerates for scattering) [derived, factor ~2].
FUME_K_EXT = 2.0e3              # m2/kg
FUME_ALBEDO = .55
FUME_COLOR = (1., .97, .92)     # relative scattering colour (linear): a faintly brown grey, iron oxides [estimate]
FUME_L_AMB = 150.               # cd/m2: mean radiance of the shop around the plume (its lamps give ~1000-2000 lx of
                                # scalar illuminance at the bench; 1/(4 pi) of it) [estimate]
PLUME_ALPHA = .1
PLUME_Q_CONV = .5               # convected share of the arc's lost heat [estimate]
PLUME_REACH = 3.5               # the plume is drawn out to 3.5 Gaussian radii (exp(-12.25) = 5e-6 of the centre;
                                # a cut at 2.5 radii showed as a jagged edge where the arc lights the fume)
# Turbulence and drift (the mean plume drawn alone reads as a smooth bright column). The
# plume's Reynolds number w 2b / nu = 1.1e4 zeta^(2/3) (zeta in m, the MTT velocity and width above, nu of ambient air)
# passes ~2000 within a few centimetres of the arc (derived; hotter gas there is more viscous): a laminar column at the
# nozzle, turbulent above. Its instantaneous concentration is the mean times a lognormal factor exp(s n - s^2/2) of a
# smooth two-octave noise n of unit spread, and its axis meanders by a fraction of its radius; both live in the plume's
# own frame: radial coordinates in plume radii, and along the axis log height in eddy sizes (self-similar: eddies grow
# with the plume, b_G ~ zeta) carried upward with time at the plume's speed at PLUME_ADV_Z. Seeded and a function of the
# simulation time only: the same state renders the same plume. Its eddies turn over in b_G / w, 5-50 ms below 1 m
# (derived), faster than the films' 100 ms frames: consecutive frames show different eddies, as a real plume would.
PLUME_EDDY = 1.5                # eddy size in Gaussian radii b_G [estimate: energy-containing eddies of a plume ~ its width]
PLUME_TURB_SIGMA = .7           # lognormal spread of the instantaneous concentration on the axis [estimate: plume
                                # concentration fluctuation intensities of order 0.3-1 on the axis rising toward the
                                # intermittent edges]
PLUME_TURB_EDGE = .5            # extra spread per (r / b_G)^2 toward the edges (intermittency) [estimate]
PLUME_MEANDER = .35             # axis meander amplitude in Gaussian radii [estimate]
PLUME_ADV_Z = .3                # m above the arc: the height whose plume speed carries the eddy field [authored]
PLUME_LAMINAR = (.015, .08)     # m above the arc: turbulence ramps in between these heights [derived Re above; estimate]
# Drift toward the extraction: the canopy (shop.CANOPY, mouth 2.6 m, over the cell) as a sink of PLUME_HOOD_Q in a
# half-space, u = Q / (2 pi d^2) toward its mouth's centre (capped at the mouth's face velocity); the plume's axis is
# carried by its horizontal part while it rises at the MTT velocity (integrated from the arc up). The arc sits ~0.1 m
# off the canopy's centre: the axis leans a few centimetres toward it near the mouth; a cross draft is not drawn.
PLUME_HOOD_Q = 1.0              # m3/s extracted by the canopy [estimate: a fabrication-shop canopy hood, ~2000 cfm]
GUN_SHADOW_R = .015             # m: the gun (nozzle, body, neck) as a cylinder of this radius along the torch axis,
GUN_SHADOW_L = .25              # this long behind the nozzle, shadows the arc's light on the fume [estimate from
                                # welding/torch.py: nozzle 9.5-13.5 mm, body 17 mm radius]


def _plume_constants(proc):
    from . import params as PR
    st = proc.src.st
    eta = PR.v(PR.ARC_ETA)
    Q = max((1. - eta)*float(st.current)*float(st.v_term)*PLUME_Q_CONV, 1.)
    F = 9.81*Q/(1.2*1005.*293.)
    Wc = (75.*F/(144.*math.pi*PLUME_ALPHA**2))**(1/3)
    z_v = PR.v(PR.NOZZLE_ID)/2/(1.2*PLUME_ALPHA)
    return Q, Wc, z_v


def fume_rate(proc):
    """kg/s of fume now: the arc model's wire melt rate x FUME_PER_G of its transfer mode, moved toward FUME_PER_G_AIR
    with the air fraction at the pool."""
    from . import params as PR
    src, st = proc.src, proc.src.st
    if not st.arc:
        return 0.
    I = float(st.current)
    melt = src.a*I + src.b*float(st.l_s)*I*I
    mode = str(getattr(st, "mode", "") or "spray")
    per_g = FUME_PER_G.get(mode, FUME_PER_G["spray"])
    x_air = float(np.clip(getattr(proc, "x_air", 0.) or 0., 0., 1.))
    per_g = per_g + (FUME_PER_G_AIR - per_g)*x_air
    return PR.v(PR.STEEL_RHO)*src.aw*melt*per_g


def _arc_times(world):
    """(t_on, t_end) of the latest arc: from the world's tool_io log (arc_on commands) and the source's state; None if
    no arc has burned since the start state."""
    t_on = t_off = None
    dt = float(world.config.rigid_dt)
    for ev in getattr(world, "events", ()):
        io = ev.get("tool_io") if isinstance(ev, dict) else None
        if not io or "arc_on" not in io:
            continue
        if io["arc_on"]:
            t_on, t_off = ev["tick"]*dt, None
        elif t_on is not None:
            t_off = ev["tick"]*dt
    st = world.process.src.st
    now = float(world.time)
    if t_on is None:
        return None
    if st.arc:
        return t_on, now
    return (t_on, t_off) if t_off is not None else None


def fume_plume(world):
    """The plume now as nested frusta in world coordinates: dict(base, segments=[dict(z0, z1, radii0[b], radii1[b],
    extinction[b])], rate_kg_s), or None. Fume emitted while the latest arc burned (t_on..t_end) has risen to heights
    whose travel time from the arc matches its age; the plume's velocity and width follow the arc's present (or last
    burning) heat."""
    p = world.process
    if p is None:
        return None
    times = _arc_times(world)
    if times is None:
        return None
    t_on, t_end = times
    now = float(world.time)
    st = p.src.st
    # the rate and heat of the arc while it burned: the present values, or for a cooling plume the nominal fraction of
    # the last schedule (the source no longer reports a current) [estimate]
    stray = getattr(world, "stray_arc", None)
    if st.arc and (p.root is not None or stray is not None):
        rate = fume_rate(p)
        Q, Wc, z_v = _plume_constants(p)
        if p.root is not None:
            o, R = world.weld_frame()
            base = o + R @ np.asarray(p.root, float)
        else:
            # a stray arc on a grounded body off the coupon (world._process_tick): its fume rises from where it burns
            base = np.asarray(stray["root_w"], float)
    else:
        last = getattr(p, "_render_last_plume", None)
        if last is None:
            return None
        rate, Q, Wc, z_v, base = last
    p._render_last_plume = (rate, Q, Wc, z_v, base)     # render-side memo on the process object (not physics state)
    if rate <= 0:
        return None
    from . import shop as SH
    z_top = SH.CANOPY["bottom"] - float(base[2])
    # travel time from the arc to height z: int dz / w = 3 / (4 Wc) (zeta^(4/3) - z_v^(4/3)), zeta = z + z_v
    Tinv = lambda tt: max((tt*4*Wc/3 + z_v**(4/3))**.75 - z_v, 0.)
    z_lo = Tinv(max(now - t_end, 0.))
    z_hi = min(Tinv(max(now - t_on, 0.)), z_top)
    if z_hi <= z_lo + 1e-3:
        return None
    # centre extinction kappa(zeta) = K zeta^(-5/3): k_ext x Gaussian centre concentration 2 m_dot / (pi b^2 w), top-hat
    # b = 1.2 alpha zeta, w = Wc zeta^(-1/3); Gaussian radius b_G = 1.2 alpha zeta / sqrt 2
    K = FUME_K_EXT*2*rate/(math.pi*(1.2*PLUME_ALPHA)**2*Wc)
    return dict(base=np.asarray(base, float).tolist(), z_lo=z_lo, z_hi=z_hi, z_v=z_v, K=K,
                bg_per_zeta=1.2*PLUME_ALPHA/math.sqrt(2.), reach=PLUME_REACH, albedo=FUME_ALBEDO,
                color=list(FUME_COLOR), L_amb=FUME_L_AMB, rate_kg_s=rate, heat_w=Q, Wc=Wc,
                t=now, drift=plume_drift(base, z_v, Wc, z_top),
                turb=dict(eddy=PLUME_EDDY, sigma=PLUME_TURB_SIGMA, edge=PLUME_TURB_EDGE, meander=PLUME_MEANDER,
                          adv_z=PLUME_ADV_Z, laminar=list(PLUME_LAMINAR), seed=20261010))


def plume_drift(base, z_v, Wc, z_top, n=48):
    """The plume axis's horizontal offset (m) from the arc at heights z above it (n heights to z_top): carried by the
    canopy sink's horizontal velocity (PLUME_HOOD_Q, half-space sink at the canopy mouth's centre, capped at the mouth's
    face velocity) while it rises at the MTT velocity w = Wc zeta^(-1/3) plus the sink's vertical pull. Returns
    dict(z [n], dx [n], dy [n])."""
    from . import shop as SH
    C = SH.CANOPY
    S = np.array([C["center"][0], C["center"][1], C["bottom"]], float)
    v_face = PLUME_HOOD_Q/(C["size"][0]*C["size"][1])
    b = np.asarray(base, float)
    zs = np.linspace(0., max(z_top, 1e-3), n)
    xy = np.zeros(2)
    out = [xy.copy()]
    for k in range(1, n):
        z0, z1 = zs[k - 1], zs[k]
        zm = .5*(z0 + z1)
        P = np.array([b[0] + xy[0], b[1] + xy[1], b[2] + zm])
        d = S - P
        r2 = max(float(d @ d), 1e-4)
        u = PLUME_HOOD_Q/(2*math.pi*r2)*d/math.sqrt(r2)
        sp = float(np.linalg.norm(u))
        if sp > v_face:
            u *= v_face/sp
        w = Wc*(zm + z_v)**(-1/3) + max(float(u[2]), 0.)
        xy = xy + u[:2]/w*(z1 - z0)
        out.append(xy.copy())
    out = np.array(out)
    return dict(z=zs.tolist(), dx=out[:, 0].tolist(), dy=out[:, 1].tolist())


# ------------------------------------------------------------------------------------------------ pores
def _slice_pts(st):
    """A slice's free-surface polyline (weld-frame y-z points from toe A to toe B), or None if the slice has no profile.
    pool.SliceState.pts when the pool model provides it (a profile that may overhang its toes: the points may run back
    past a toe; neither y nor the chord coordinate is assumed monotone along it); else the height function over the chord
    of the earlier model, A + u t + h n."""
    pts = getattr(st, "pts", None)
    if pts is not None:
        pts = np.asarray(pts, float)
        return pts if pts.ndim == 2 and len(pts) >= 2 and np.isfinite(pts).all() else None
    if getattr(st, "h", None) is None or getattr(st, "u", None) is None or st.A is None or st.chord_t is None \
            or st.chord_n is None:
        return None
    t, n = np.asarray(st.chord_t, float), np.asarray(st.chord_n, float)
    u, h = np.asarray(st.u, float), np.asarray(st.h, float)
    return np.asarray(st.A, float)[None] + u[:, None]*t[None] + h[:, None]*n[None]


def _profile_points(st):
    """A slice's pool profile (_slice_pts) as y-z points and outward unit normals from the polyline itself (the local
    tangent turned the way the chord's tangent turns into the chord's outward normal, so an overhanging profile keeps its
    outside), or (None, None)."""
    yz = _slice_pts(st)
    if yz is None:
        return None, None
    tg = np.gradient(yz, axis=0) if len(yz) > 2 else np.repeat((yz[-1] - yz[0])[None], len(yz), 0)
    tg /= np.maximum(np.linalg.norm(tg, axis=1), 1e-15)[:, None]
    ct = np.asarray(st.chord_t, float) if st.chord_t is not None else (yz[-1] - yz[0])
    cn = np.asarray(st.chord_n, float) if st.chord_n is not None else np.array([-ct[1], ct[0]])
    sgn = 1. if (-ct[1]*cn[0] + ct[0]*cn[1]) >= 0 else -1.          # +90 deg or -90 deg turns t into n
    nn = sgn*np.stack([-tg[:, 1], tg[:, 0]], 1)
    return yz, nn/np.maximum(np.linalg.norm(nn, axis=1), 1e-12)[:, None]


def pore_sites(proc, P, N, bead):
    """Surface pores of the frozen bead: [(centre, outward normal, radius)] in the weld frame. Per slice with porosity
    phi the pores cover phi of the slice's bead surface (Delesse rule, PORE_D sizes); each pore's place along the slice
    (x within the slice, arclength along its frozen profile) and its size are drawn from a generator seeded by the slice
    index, then snapped to the nearest bead vertex of the rendered surface (P, N, bead mask)."""
    g = proc.g
    sl = proc.pool.slices
    phi = np.array([float(getattr(s, "porosity", 0.)) for s in sl])
    todo = np.nonzero(phi > 0)[0]
    if not len(todo) or not bead.any():
        return []
    from scipy.spatial import cKDTree
    bi = np.nonzero(bead)[0]
    tree = cKDTree(P[bi])
    d0, d1 = PORE_D
    mean_area = math.pi/4*(d1*d1 - d0*d0)/(2*math.log(d1/d0))         # E[pi d^2 / 4], d log-uniform
    out = []
    for i in todo:
        st = sl[i]
        rng = np.random.default_rng([20261009, int(i)])
        yz = _slice_pts(st)
        if yz is None:                              # no profile (stacked metal): the bead vertices of the slice
            m = (P[bi, 0] >= g.xe[i]) & (P[bi, 0] < g.xe[i + 1])
            if m.sum() < 3:
                continue
            q = P[bi][m]
            yz = q[np.argsort(np.arctan2(q[:, 2], q[:, 1])), 1:]
        seg = np.linalg.norm(np.diff(yz, axis=0), axis=1)
        cum = np.concatenate([[0.], np.cumsum(seg)])
        if cum[-1] <= 0:
            continue
        n_exp = phi[i]*cum[-1]*g.dx[i]/mean_area
        n = int(n_exp) + int(rng.random() < n_exp - int(n_exp))
        for _ in range(n):
            xt = g.xe[i] + rng.random()*g.dx[i]
            s = (.04 + .92*rng.random())*cum[-1]
            d = math.exp(rng.uniform(math.log(d0), math.log(d1)))
            k = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1))
            f = (s - cum[k])/max(seg[k], 1e-12)
            p = np.array([xt, *(yz[k] + (yz[k + 1] - yz[k])*f)])
            _, j = tree.query(p)
            out.append((P[bi[j]].copy(), N[bi[j]].copy(), d/2))
    return out


def pore_cup(c, n, r, depth, rings=4, seg=14):
    """The inside of an open gas pore: a spherical cap of rim radius r whose rim lies on the surface at c (outward normal
    n), `depth` deep. (points, tris, normals); faces and normals look into the opening."""
    n = np.asarray(n, float)/np.linalg.norm(n)
    a = np.cross(n, [1., 0., 0.] if abs(n[0]) < .9 else [0., 1., 0.])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    R = (r*r + depth*depth)/(2*depth)
    o = np.asarray(c, float) + n*(R - depth)                # sphere centre
    ph_max = math.asin(min(r/R, 1.)) if depth <= R else math.pi - math.asin(min(r/R, 1.))
    pts = [o - n*R]
    for k in range(1, rings + 1):
        ph = ph_max*k/rings
        for j in range(seg):
            ps = 2*math.pi*j/seg
            pts.append(o + R*(math.sin(ph)*(math.cos(ps)*a + math.sin(ps)*b) - math.cos(ph)*n))
    pts = np.array(pts)
    tris = []
    for j in range(seg):
        tris.append([0, 1 + j, 1 + (j + 1) % seg])
    for k in range(rings - 1):
        r0, r1 = 1 + k*seg, 1 + (k + 1)*seg
        for j in range(seg):
            j1 = (j + 1) % seg
            tris += [[r0 + j, r1 + j, r1 + j1], [r0 + j, r1 + j1, r0 + j1]]
    tris = np.array(tris)
    nrm = (o - pts)/R
    fn = np.cross(pts[tris[:, 1]] - pts[tris[:, 0]], pts[tris[:, 2]] - pts[tris[:, 0]])
    flip = np.einsum("ij,ij->i", fn, nrm[tris].mean(1)) < 0
    tris[flip] = tris[flip][:, ::-1]
    return pts, tris, nrm


# ------------------------------------------------------------------------------------------------ the liquid pool
POOL_SKIRT = .3e-3      # the pool surface's edges continue this far into the metal, closing the seam with the cells'
                        # surface around it (render geometry only)


def pool_surface(proc):
    """The weld pool's free surface from the pool model's own profiles (pool.SliceState of every liquid slice: toes A-B,
    chord t / n, heights h over u): a specular liquid-metal surface across consecutive liquid slices, emitting
    EMISSIVITY_LIQUID x blackbody luminance at the temperature of the cell under each node and reflecting 1 - that.
    Returns dict(points, tris, normals, color, glow, roughness, metalness, x_range) in the weld frame, or None."""
    g, s = proc.g, proc.s
    sl = proc.pool.slices
    idx = [i for i, st in enumerate(sl) if st.liquid and _slice_pts(st) is not None]
    if not idx:
        return None
    Tf = s.field("T").reshape(g.nx, g.ny, g.nz)
    cols = {}
    for i in idx:
        yz, nn = _profile_points(sl[i])
        n = np.asarray(sl[i].chord_n, float)
        yz = np.concatenate([yz[:1] - n*POOL_SKIRT, yz, yz[-1:] - n*POOL_SKIRT])
        nn = np.concatenate([nn[:1], nn, nn[-1:]])
        cols[i] = (yz, nn)
    m = len(next(iter(cols.values()))[0])
    runs, cur = [], [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1 and len(cols[i][0]) == m:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    P, Nn, F = [], [], []
    for run in runs:
        xs = [g.xc[i] for i in run]
        cs = [cols[i] for i in run]
        # end caps: the first and last column again half a slice beyond, sunk into the metal (a skirt)
        x_lo, x_hi = g.xe[run[0]], g.xe[run[-1] + 1]
        xs = [x_lo] + xs + [x_hi]
        cs = [(cs[0][0] - cs[0][1]*POOL_SKIRT, cs[0][1])] + cs + [(cs[-1][0] - cs[-1][1]*POOL_SKIRT, cs[-1][1])]
        base = sum(len(p) for p in P)
        for x, (yz, nn) in zip(xs, cs):
            P.append(np.column_stack([np.full(m, x), yz]))
            Nn.append(np.column_stack([np.zeros(m), nn]))
        for c in range(len(xs) - 1):
            a0, a1 = base + c*m, base + (c + 1)*m
            for k in range(m - 1):
                F += [[a0 + k, a0 + k + 1, a1 + k + 1], [a0 + k, a1 + k + 1, a1 + k]]
    P, Nn, F = np.concatenate(P), np.concatenate(Nn), np.array(F, np.int64)
    # orient faces along the outward normals
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    flip = np.einsum("ij,ij->i", fn, Nn[F].mean(1)) < 0
    F[flip] = F[flip][:, ::-1]
    q = P - Nn*1.5e-4
    ii = np.clip(np.searchsorted(g.xe, q[:, 0]) - 1, 0, g.nx - 1)
    jj = np.clip(np.searchsorted(g.ye, q[:, 1]) - 1, 0, g.ny - 1)
    kk = np.clip(np.searchsorted(g.ze, q[:, 2]) - 1, 0, g.nz - 1)
    T = Tf[ii, jj, kk]
    gl = glow(T, EMISSIVITY_LIQUID)
    n_ = len(P)
    return dict(points=P.astype(np.float32), tris=F.astype(np.int32), normals=Nn.astype(np.float32),
                color=np.full((n_, 3), 1. - EMISSIVITY_LIQUID, np.float32), glow=gl.astype(np.float32),
                weldfrac=np.ones(n_, np.float32), roughness=np.full(n_, LIQUID_ROUGHNESS, np.float32),
                metalness=np.ones(n_, np.float32), slices=[int(i) for i in idx], tmax=float(T.max()))


# ------------------------------------------------------------------------------------------------ the arc column
def arc_luminance(current):
    """Mean luminance (cd/m2) of the arc over the measured 3.3 mm field at `current` (A): ARC_LB interpolated log-log
    (the end slopes beyond the table) times ARC_LV_PER_LB."""
    a = np.log([c for c, _ in ARC_LB])
    b = np.log([v for _, v in ARC_LB])
    x = math.log(max(float(current), 1.))
    if x <= a[0]:
        y = b[0] + (b[1] - b[0])/(a[1] - a[0])*(x - a[0])
    elif x >= a[-1]:
        y = b[-1] + (b[-1] - b[-2])/(a[-1] - a[-2])*(x - a[-1])
    else:
        y = float(np.interp(x, a, b))
    return math.exp(y)*1e4*ARC_LV_PER_LB              # W/cm2/sr -> W/m2/sr -> cd/m2


def arc_column(wire_end, root, current):
    """The arc column's emission model (see ARC_* above): dict(p0 wire end, p1 root, axis, La, s_wire, s_root, a_bg,
    core, up, j0 cd/m3, intensity_cd, centroid, color), with j0 normalised to arc_luminance(current) over the 3.3 mm
    disc."""
    from . import params as PR
    p0, p1 = np.asarray(wire_end, float), np.asarray(root, float)
    ax = p1 - p0
    La = float(np.linalg.norm(ax))
    if La < 1e-5:
        return None
    ax /= La
    sig = PR.v(PR.ARC_PRESSURE_SIGMA)
    s_root = sig[0] + sig[1]*float(current)
    col = dict(p0=p0.tolist(), p1=p1.tolist(), axis=ax.tolist(), La=La, s_wire=ARC_S_WIRE, s_root=s_root,
               s_core=ARC_S_CORE, s_core_root=max(ARC_CORE_ROOT*s_root, ARC_S_CORE), core=ARC_CORE,
               core_floor=ARC_CORE_FLOOR, env=ARC_ENV, up=ARC_UP, edge=ARC_EDGE, edge_p=ARC_EDGE_P)
    # side-on apparent luminance per unit j0 (the integral across the line of sight), its mean over a 3.3 mm disc
    # centred at the column's mid-length
    R = 1.65e-3
    g = np.linspace(-R, R, 67)
    w = np.linspace(-4*s_root, 4*s_root, 81)
    U, V, Wd = np.meshgrid(g, g, w, indexing="ij")
    inside = (U[..., 0]**2 + V[..., 0]**2) <= R*R
    app = np.trapezoid(arc_profile(col, La/2 + U, V*V + Wd*Wd), w, axis=2)
    mean1 = float(app[inside].mean())
    j0 = arc_luminance(current)/max(mean1, 1e-12)
    # luminous intensity: int j dV (isotropic for an optically thin source); per unit length the radial integral of each
    # edge-cut Gaussian is 2 pi s^2 x _edge_area()
    t = np.linspace(-2*ARC_UP, La, 600)
    up = np.where(t < 0, np.exp(-.5*(t/(ARC_UP/3.))**2), 1.)
    sd = ARC_S_WIRE + (s_root - ARC_S_WIRE)*np.clip(t/La, 0., 1.)
    sc = col["s_core"] + (col["s_core_root"] - col["s_core"])*np.clip(t/La, 0., 1.)
    ac = ARC_CORE_FLOOR + (1. - ARC_CORE_FLOOR)*np.exp(-np.abs(t)/ARC_CORE)
    ea = _edge_area(ARC_EDGE, ARC_EDGE_P)
    dIc = j0*up*2*math.pi*ac*sc*sc*ea
    dIe = j0*up*2*math.pi*ARC_ENV*sd*sd*ea
    dI = dIc + dIe
    I_v = float(np.trapezoid(dI, t))
    tc = float(np.trapezoid(dI*t, t)/max(I_v, 1e-30))
    lc = np.asarray(ARC_COLOR, float)
    col.update(j0=j0, intensity_cd=I_v, centroid=(p0 + ax*tc).tolist(), t_centroid=tc,
               color=(lc/float(lc @ np.array([.2126, .7152, .0722]))).tolist(), luminance=arc_luminance(current))
    # the reflection chain (arc_light): per station, the segment's core and envelope luminous intensities (the light
    # above the wire end, t < 0, joins the first segment)
    tb = np.linspace(0., La, ARC_SPEC_N + 1)
    seg = []
    for k in range(ARC_SPEC_N):
        m = (t >= (tb[k] if k else -1.)) & (t <= tb[k + 1])
        tm = .5*(tb[k] + tb[k + 1])
        f = tm/La
        sc_ = float(col["s_core"] + (col["s_core_root"] - col["s_core"])*f)
        se_ = float(ARC_S_WIRE + (s_root - ARC_S_WIRE)*f)
        Ic_, Ie_ = float(np.trapezoid(dIc[m], t[m])), float(np.trapezoid(dIe[m], t[m]))
        seg.append(dict(t0=float(tb[k]), t1=float(tb[k + 1]), s_core=sc_, s_env=se_, I_core=Ic_, I_env=Ie_,
                        s_eff=math.sqrt((Ic_*sc_*sc_ + Ie_*se_*se_)/max(Ic_ + Ie_, 1e-30))))
    col["spec_segments"] = seg
    return col


@lru_cache(maxsize=8)
def _edge_area(k, p):
    """int_0^inf x exp(-x^2/2) exp(-(x/k)^p) dx: the radial integral of an edge-cut unit Gaussian relative to 2 pi s^2."""
    x = np.linspace(0., 8., 4001)
    return float(np.trapezoid(x*np.exp(-.5*x*x)*np.exp(-(x/k)**p), x))


def arc_edge(r_over_s, col):
    """The column's edge factor exp(-(r / (edge s))^p) (ARC_EDGE); 1 for a column dict without one (older requests)."""
    k = col.get("edge")
    if not k:
        return 1.
    return np.exp(-(np.asarray(r_over_s, float)/k)**col.get("edge_p", ARC_EDGE_P))


def arc_profile(col, t, rho2):
    """Emission density j / j0 of the column (arc_column dict) at axial positions t (m from the wire end toward the
    root) and squared distances rho2 (m2) from the axis."""
    t = np.asarray(t, float)
    up = np.where(t < 0, np.exp(-.5*(t/(col["up"]/3.))**2), 1.)
    f = np.clip(t/col["La"], 0., 1.)
    a = col.get("core_floor", 0.) + (1. - col.get("core_floor", 0.))*np.exp(-np.abs(t)/col["core"])
    sc = col["s_core"] + (col.get("s_core_root", col["s_core"]) - col["s_core"])*f
    r = np.sqrt(np.maximum(rho2, 0.))
    core = a*np.exp(-rho2/(2*sc*sc))*arc_edge(r/sc, col)
    s = col["s_wire"] + (col["s_root"] - col["s_wire"])*f
    env = col["env"]*np.exp(-rho2/(2*s*s))*arc_edge(r/s, col)
    # past the root plane the plasma ends on the work surface (the depth AOV cuts it there); where the surface falls away
    # beside the root it fades over the root's spread instead of ending on the plane (a plane cut draws a straight edge
    # across the weld camera's frame)
    down = np.where(t > col["La"], np.exp(-.5*((t - col["La"])/col["s_root"])**2), 1.)
    return np.where(t < -2*col["up"], 0., up*down*(core + env))


def arc_light(col):
    """The scene's lights for the column (UsdLux intensity = luminance, measured): a sphere light of radius ARC_LIGHT_R at
    the luminous centroid with the column's luminous intensity, L = I / (pi R^2), for the diffuse light it sheds; and for
    its reflections in glossy and liquid surfaces a chain of ARC_SPEC_N short cylinder lights end to end along the column
    (arc_column's spec_segments), each L = dI / (2 R dt) broadside, specular only. The
    split is the lights' diffuse / specular multipliers (measured: each light then contributes only its
    part; mirror images come out at the light's luminance x reflectance within 1-8 %)."""
    p0 = np.asarray(col["p0"], float)
    ax = np.asarray(col["axis"], float)
    chain = []
    for sg in col.get("spec_segments", ()):
        L = max(sg["t1"] - sg["t0"], 1e-5)
        R = ARC_SPEC_R_SPREAD*sg["s_eff"]
        chain.append(dict(center=(p0 + ax*.5*(sg["t0"] + sg["t1"])).tolist(), axis=list(col["axis"]), length=L,
                          radius=R, intensity=(sg["I_core"] + sg["I_env"])/(2*R*L)))
    return dict(pos=col["centroid"], radius=ARC_LIGHT_R, intensity=col["intensity_cd"]/(math.pi*ARC_LIGHT_R**2),
                spec_chain=chain)


# ------------------------------------------------------------------------------------------------ joint band surface
def band_edges(g):
    """Cell indices (j0, j1, k0, k1) and weld-frame edges of the joint band on the thermal grid."""
    j0, j1 = (int(np.searchsorted(g.ye, v)) for v in BAND["y"])
    k0, k1 = (int(np.searchsorted(g.ze, v)) for v in BAND["z"])
    return (j0, j1, k0, k1), dict(y=(float(g.ye[j0]), float(g.ye[j1])), z=(float(g.ze[k0]), float(g.ze[k1])))


SURFACE_BLUR_M = .375e-3      # isotropic occupancy blur for the band's surface: below the cells' 0.25 / 0.5 mm, so the
                              # rendered surface is the cells' shape without their staircase (a 0.75-cell blur in index
                              # space leaves 0.25 mm terraces running along the bead) [authored]


def _metal_interp(F, solid, xc, yc, zc, P):
    """Field F (on cell centres xc, yc, zc) interpolated trilinearly at points P from the solid cells only (weights of
    air cells dropped, the rest renormalised); NaN where no solid cell is among the 8 neighbours."""
    from scipy.ndimage import map_coordinates

    def frac(c, v):
        i = np.clip(np.searchsorted(c, v) - 1, 0, len(c) - 2)
        return i + np.clip((v - c[i])/(c[i + 1] - c[i]), 0., 1.)
    coords = np.stack([frac(xc, P[:, 0]), frac(yc, P[:, 1]), frac(zc, P[:, 2])])
    m = solid.astype(np.float64)
    num = map_coordinates(np.where(solid, F, 0.).astype(np.float64), coords, order=1, mode="nearest")
    den = map_coordinates(m, coords, order=1, mode="nearest")
    return np.where(den > 1e-6, num/np.maximum(den, 1e-6), np.nan)


# ------------------------------------------------------------------------------------------------ the bead from the pool model
# The weld metal's surface where the pool model holds it as a continuous free surface (pool.SliceState: toes A-B, chord
# t / n, heights h over u; kept by every slice that has been liquid, the frozen ones at the profile they froze in) is
# drawn from those profiles, not from the cells that realise them on the 0.25 mm grid (the evaluator keeps reading the
# cells). Each x slice's surface is taken along a fan of rays in the y-z plane from a point inside the metal under the
# root (BEAD_C_DEPTH along the bisector of the joint's faces) through the joint's original surface (BEAD_DS apart along
# it): the distance to the profile where the ray crosses the chord's span, to the original surface outside it, and to
# the cells' surface (their occupancy blurred across the slice by SURFACE_BLUR_M) where the cells hold more metal there
# than the profile (droplets stacked with no pool, metal spilled beyond the toes: the model's own metal without a
# profile) or where the slice has no profile at all. The distances are then smoothed along the weld with a Gaussian of
# BEAD_SIGMA_X: the slices are 2-D sections 0.5 mm apart whose toes step with the 0.25 mm cells; the pool model itself
# shares its metal along x with a 1.0 mm kernel because surface tension evens out features shorter than that
# (pool.LONG_SIGMA), so a surface smoothed below that length draws nothing the model resolves. Its ends taper into the
# original surface over the same length (the smoothing conserves the deposit's cross-section integral along x).
# Liquid or solid: per vertex, the liquid fraction of the metal under it (thermal cells, trilinear over metal cells):
# the pool's outline is the solidification front of the thermal field, not a slice boundary.
BEAD_SIGMA_X = .5e-3          # m, along-weld smoothing of the drawn bead surface [authored: one slice, half the pool
                              # model's own LONG_SIGMA 1.0 mm]
BEAD_DS = .1e-3               # m, ray spacing along the original surface where there is weld metal [authored]
BEAD_DS_FAR = .4e-3           # m, beyond 2 mm from any weld metal (flat parent metal: carries only its colours)
BEAD_C_DEPTH = 4.25e-3        # m, the rays' centre below the root along the faces' bisector (3 mm under each face of a
                              # tee): every ray meets the original surface once (star-shaped) [authored geometry]
BEAD_TOL = .30e-3             # m: the cells' surface is drawn where it lies this far outside the profile (1.2 cells)
BEAD_X_SUB = 2                # vertex columns per slice along x
BEAD_OVERHANG = .5e-3         # m, the fan continues this far past the joint's ends (the band edges at 14 mm)
BEAD_SKIRT = .5e-3            # m, the fan's end columns continue this far into the metal [authored geometry]
BEAD_SIGMA_CELLS = .3e-3      # m, across-slice smoothing of the surface taken from the cells (droplets stacked without a
                              # pool profile): about one cell, removing the blurred cells' terraces [authored]
BEAD_MARCH = (-2.5e-3, 7.5e-3, .25e-3)      # ray march for the cells' surface: from / to (about the original surface),
                                            # step (m; the blurred occupancy is linear enough over one cell)


def _joint_point(J, s):
    """Points of the joint's original surface at arclengths s, continued straight beyond its ends (the fan reaches past
    the band's edge, under the coupon pieces outside it: no gap at the seam)."""
    s = np.asarray(s, float)
    p, _ = J.point(s)
    lo, hi = s < 0, s > J.length
    p[lo] = J.surface[0] + J.tangent[0]*s[lo, None]
    p[hi] = J.surface[-1] + J.tangent[-1]*(s[hi, None] - J.length)
    return p


def _bead_rays(joint, s):
    """Rays of the bead fan at original-surface arclengths s: centre c (y, z), unit directions D, distance r0 to the
    original surface and angle phi (rad) of each."""
    J = joint
    root, _ = J.point(np.array([float(J.root_s)]))
    b = J.normal.mean(0)
    b = b/np.linalg.norm(b)
    c = root[0] - b*BEAD_C_DEPTH
    p = _joint_point(J, s)
    V = p - c
    r0 = np.linalg.norm(V, axis=1)
    D = V/r0[:, None]
    return c, D, r0, np.arctan2(D[:, 1], D[:, 0])


def _cross2(u, v):
    return u[..., 0]*v[..., 1] - u[..., 1]*v[..., 0]


def _ray_profile(c, D, phi, Q):
    """Distances along the fan's rays to a slice's profile polyline Q (K x 2, y-z) and the mask of rays that meet it.
    A profile seen from c as single-valued (each ray meets it once) is intersected per ray directly; one that folds over
    itself as seen from c (a profile overhanging its toe, pool.SliceState.pts) gives each ray its outermost crossing: the
    fan draws the overhang's top and outer face; the undercut beneath it is not drawn (a fan cannot)."""
    V = Q - c
    ang = np.arctan2(V[:, 1], V[:, 0])
    if ang[0] > ang[-1]:
        Q, V, ang = Q[::-1], V[::-1], ang[::-1]
    if np.any(np.diff(ang) <= 0):
        E = Q[1:] - Q[:-1]                                              # segments
        den = _cross2(D[:, None, :], E[None, :, :])                     # rays x segments
        ok = np.abs(den) > 1e-15
        den = np.where(ok, den, 1.)
        r = _cross2(V[None, :-1, :], E[None, :, :])/den
        u = _cross2(V[None, :-1, :], D[:, None, :])/den
        hit = ok & (u >= -1e-9) & (u <= 1. + 1e-9) & (r > 0.)
        rr = np.where(hit, r, -np.inf).max(1)
        inside = np.isfinite(rr)
        return np.where(inside, rr, np.nan), inside
    inside = (phi >= ang[0]) & (phi <= ang[-1])
    k = np.clip(np.searchsorted(ang, phi) - 1, 0, len(Q) - 2)
    e = Q[k + 1] - Q[k]
    den = _cross2(D, e)
    r = _cross2(V[k], e)/np.where(np.abs(den) > 1e-15, den, 1e-15)
    return np.where(inside, r, np.nan), inside


def _frac_index(c, v):
    i = np.clip(np.searchsorted(c, v) - 1, 0, len(c) - 2)
    return i + np.clip((v - c[i])/(c[i + 1] - c[i]), -.5, 1.5)


def _ray_cells(f3, yc, zc, c, D, r0):
    """Distances along each ray (rows of D) to the first exit from the metal of the occupancy f3 (slices x y x z,
    blurred across the slice; level 0.5) in every slice, marching from r0 + BEAD_MARCH[0] outward; linear between
    samples. Returns (slices x rays)."""
    from scipy.ndimage import map_coordinates
    a, b_, st = BEAD_MARCH
    off = np.arange(a, b_ + 1e-12, st)
    R = r0[:, None] + off[None, :]
    fy = _frac_index(yc, (c[0] + D[:, 0:1]*R).ravel())
    fz = _frac_index(zc, (c[1] + D[:, 1:2]*R).ravel())
    ns = f3.shape[0]
    co = np.stack([np.repeat(np.arange(ns, dtype=float), len(fy)), np.tile(fy, ns), np.tile(fz, ns)])
    F = map_coordinates(f3, co, order=1, mode="nearest").reshape(ns, len(r0), len(off))
    inm = F > .5
    started = np.maximum.accumulate(inm, axis=2)
    ex = ~inm & started
    has = ex.any(2)
    m = np.where(has, ex.argmax(2), len(off) - 1)
    m0 = np.maximum(m - 1, 0)
    f0 = np.take_along_axis(F, m0[..., None], 2)[..., 0]
    f1 = np.take_along_axis(F, m[..., None], 2)[..., 0]
    tt = np.clip((f0 - .5)/np.where(np.abs(f0 - f1) > 1e-9, f0 - f1, 1.), 0., 1.)
    r = r0[None, :] + off[m0] + tt*st
    return np.where(inm.any(2), r, r0[None, :] + a)


def bead_grid(proc, f2, f2o, yc, zc, i0, i1):
    """The bead fan surface over slices i0..i1 (inclusive; the weld frame's grid). f2: the band occupancy (x slices
    i0..i1 by the band's y-z cells yc, zc) blurred across the slice only; f2o: the same for one section of the original
    metal. Returns dict(x (columns), s (ray arclengths),
    P (ncol x nray x 3), r, r0, src (per ray and slice: 0 original, 1 profile, 2 cells)) or None."""
    from scipy.ndimage import gaussian_filter1d
    g = proc.g
    J = proc.pool.joint
    sl = proc.pool.slices
    n = i1 - i0 + 1
    M = proc.s.mat.reshape(g.nx, g.ny, g.nz)
    # weld metal and profiles in the range: where the fine rays go (arclengths along the original surface)
    s_fine = np.arange(-BEAD_OVERHANG, J.length + BEAD_OVERHANG + 1e-9, BEAD_DS)
    c, D, r0, phi = _bead_rays(J, s_fine)
    has_prof = np.array([_slice_pts(sl[i]) is not None for i in range(i0, i1 + 1)])
    R = np.repeat(r0[None, :], n, 0)
    src = np.zeros((n, len(r0)), np.int8)
    rc = np.full((n, len(r0)), np.nan)
    for k in range(n):
        i = i0 + k
        st = sl[i]
        if has_prof[k]:
            yz, _ = _profile_points(st)
            rp = _ray_profile(c, D, phi, yz)
            if rp is not None:
                r_, ins = rp
                R[k, ins] = r_[ins]
                src[k, ins] = 1
            else:
                has_prof[k] = False
    # the cells' surface, only along rays within 1.5 mm (in angle at 5 mm) of the weld metal's or the profiles' extent
    Wm = (M[i0:i1 + 1] == 2)
    (j0_, _, k0_, _), _ = band_edges(g)
    jy = np.searchsorted(g.ye, yc[0] + 1e-9) - 1
    kz = np.searchsorted(g.ze, zc[0] + 1e-9) - 1
    Wm = Wm[:, jy:jy + len(yc), kz:kz + len(zc)]
    wk = np.nonzero(Wm.any(0))
    ang = list(np.arctan2(zc[wk[1]] - c[1], yc[wk[0]] - c[0])) if len(wk[0]) else []
    ang += list(phi[(src == 1).any(0)])
    near = np.zeros(len(phi), bool)
    if ang:
        near = (phi >= min(ang) - .3) & (phi <= max(ang) + .3)
    if near.any():
        rcn = _ray_cells(f2, yc, zc, c, D[near], r0[near])
        # the blur rounds the joint's own concave root by ~0.2 mm; the same march over the original metal alone (f2o: one
        # section, the joint's cross-section is uniform along the bracket) measures that rounding, and only the
        # difference is metal
        rcn = r0[near][None, :] + (rcn - _ray_cells(f2o[None], yc, zc, c, D[near], r0[near]))
        rc[:, near] = rcn
    rc = np.where(np.isfinite(rc), rc, r0[None, :])
    rcs = gaussian_filter1d(rc, max(BEAD_SIGMA_CELLS/BEAD_DS, .5), axis=1, mode="nearest")
    use_c = (rcs > R + BEAD_TOL) | (~has_prof[:, None] & (np.abs(rcs - r0[None, :]) > .5*BEAD_TOL))
    R = np.where(use_c, rcs, R)
    src[use_c] = 2
    # along the weld
    sig = BEAD_SIGMA_X/float(np.median(g.dx[i0:i1 + 1]))
    Rs = gaussian_filter1d(R, sig, axis=0, mode="nearest") if n > 1 else R
    # coarser rays far from any deposit (flat parent metal)
    dep = np.abs(Rs - r0[None, :]) > 2e-5
    near = np.zeros(len(r0), bool)
    if dep.any():
        jj = np.nonzero(dep.any(0))[0]
        near[max(jj.min() - int(2e-3/BEAD_DS), 0):jj.max() + int(2e-3/BEAD_DS) + 1] = True
    step = max(int(round(BEAD_DS_FAR/BEAD_DS)), 1)
    keep = near | (np.arange(len(r0)) % step == 0)
    keep[[0, -1]] = True
    Rs, r0k, Dk, sk = Rs[:, keep], r0[keep], D[keep], s_fine[keep]
    src = src[:, keep]
    # columns: slice edges and BEAD_X_SUB per slice, linear between the smoothed slice centres
    xe = g.xe[i0:i1 + 2]
    xs = np.unique(np.concatenate([xe] + [g.xe[i0:i1 + 1] + (q + .5)/BEAD_X_SUB*g.dx[i0:i1 + 1]
                                          for q in range(BEAD_X_SUB)]))
    xcs = g.xc[i0:i1 + 1]
    Rc = np.stack([np.interp(xs, xcs, Rs[:, j]) for j in range(Rs.shape[1])], 1)
    # skirts: at each end the fan's last column continues BEAD_SKIRT into the metal (a short wall at the same x), closing
    # any gap to the cells' surface beyond (whose blur rounds the bracket's corners and a stacked lump's ends)
    xs = np.concatenate([[xs[0]], xs, [xs[-1]]])
    Rc = np.concatenate([Rc[:1] - BEAD_SKIRT, Rc, Rc[-1:] - BEAD_SKIRT])
    YZ = c[None, None, :] + Rc[..., None]*Dk[None, :, :]
    P = np.concatenate([np.broadcast_to(xs[:, None, None], YZ.shape[:2] + (1,)), YZ], 2)
    return dict(x=xs, s=sk, P=P, r=Rc, r0=r0k, src=src, slices=(i0, i1), center=c)


def _grid_mesh(P):
    """Triangles and smooth vertex normals of a structured grid of points P (ncol x nray x 3); faces oriented so the
    normal points out of the metal (away from the fan's centre, i.e. along increasing r: checked per face below)."""
    nc, nr = P.shape[:2]
    V = P.reshape(-1, 3)
    idx = np.arange(nc*nr).reshape(nc, nr)
    a, b, c_, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    F = np.concatenate([np.stack([a, b, c_], 1), np.stack([a, c_, d], 1)])
    return V, F


def _bead_ranges(proc):
    """Slice ranges [(i0, i1)] the bead fan replaces: runs of slices holding weld metal or a pool profile, each widened
    by 4 BEAD_SIGMA_X (runs closer than that merge), inside the joint's length (the tee's bracket: beyond its ends the
    joint is a bare plate and the cells draw it)."""
    g = proc.g
    (j0, j1, k0, k1), _ = band_edges(g)
    M = proc.s.mat.reshape(g.nx, g.ny, g.nz)
    weld = (M[:, j0:j1, k0:k1] == 2).any(axis=(1, 2))
    prof = np.array([_slice_pts(st) is not None for st in proc.pool.slices])
    on = np.nonzero(weld | prof)[0]
    if not len(on):
        return []
    c = proc.c
    if c.kind == "tee":
        lo, hi = 0., float(c.length)
    else:
        lo, hi = float(c.base_x0), float(c.base_x0 + c.base[0])
    ok = np.nonzero((g.xe[:-1] >= lo - 1e-9) & (g.xe[1:] <= hi + 1e-9))[0]
    if not len(ok):
        return []
    m = int(math.ceil(4*BEAD_SIGMA_X/float(np.median(g.dx))))
    runs, a, b = [], int(on[0]), int(on[0])
    for i in on[1:]:
        if i - b > 2*m + 1:
            runs.append((a, b))
            a = int(i)
        b = int(i)
    runs.append((a, b))
    out = []
    for a, b in runs:
        i0, i1 = max(a - m, int(ok.min())), min(b + m, int(ok.max()))
        if i1 > i0:
            out.append((i0, i1))
    return out


def band_surface(proc, smooth_iters=4, margin=3, shield=None):
    """The joint band's metal surface (weld frame) by surface nets on the cells' occupancy, blurred isotropically by
    SURFACE_BLUR_M: one vertex per dual cube straddling the 0.5 level, at the mean of its edges' interpolated crossings;
    a quad across every cell-to-cell edge that crosses. Flat faces land exactly on cell boundaries (where the coupon's
    boxes meet them). Returns dict(points, tris, normals, color, glow, tmax) or None."""
    from scipy.ndimage import gaussian_filter
    g, s = proc.g, proc.s
    (j0, j1, k0, k1), _ = band_edges(g)
    ja, jb = max(j0 - margin, 0), min(j1 + margin, g.ny)
    ka, kb = max(k0 - margin, 0), min(k1 + margin, g.nz)
    M = s.mat.reshape(g.nx, g.ny, g.nz)
    occ = (M[:, ja:jb, ka:kb] != 0).astype(np.float32)
    occ = np.pad(occ, ((1, 1), (0, 0), (0, 0)))                     # air beyond the coupon's x ends
    sig = (SURFACE_BLUR_M/float(np.median(g.dx[g.dx < 1e-3])), SURFACE_BLUR_M/float(np.median(g.dy[ja:jb])),
           SURFACE_BLUR_M/float(np.median(g.dz[ka:kb])))
    f = gaussian_filter(occ, sig, mode="nearest")
    T = np.pad(s.field("T").reshape(g.nx, g.ny, g.nz)[:, ja:jb, ka:kb], ((1, 1), (0, 0), (0, 0)), mode="edge")
    pk = np.pad(s.field("peakT").reshape(g.nx, g.ny, g.nz)[:, ja:jb, ka:kb], ((1, 1), (0, 0), (0, 0)), mode="edge")
    wl = np.pad((M[:, ja:jb, ka:kb] == 2).astype(np.float32), ((1, 1), (0, 0), (0, 0)))
    fl = np.pad(s.field("fl").reshape(g.nx, g.ny, g.nz)[:, ja:jb, ka:kb], ((1, 1), (0, 0), (0, 0)), mode="edge")
    xc = np.concatenate([[g.xe[0] - g.dx[0]/2], g.xc, [g.xe[-1] + g.dx[-1]/2]])
    yc, zc = g.yc[ja:jb], g.zc[ka:kb]
    nx, ny, nz = f.shape
    inside = f > .5
    # dual cubes
    c = [inside[dx_:nx - 1 + dx_, dy_:ny - 1 + dy_, dz_:nz - 1 + dz_] for dx_ in (0, 1) for dy_ in (0, 1) for dz_ in (0, 1)]
    anyin = np.logical_or.reduce(c)
    allin = np.logical_and.reduce(c)
    active = anyin & ~allin
    I, J, K = np.nonzero(active)
    if not len(I):
        return None
    vid = -np.ones(active.shape, np.int64)
    vid[I, J, K] = np.arange(len(I))
    # vertex = mean of the crossing points on the cube's 12 edges
    acc = np.zeros((len(I), 3))
    cnt = np.zeros(len(I))
    X = np.stack(np.meshgrid(xc, yc, zc, indexing="ij"), -1)
    for ax in range(3):
        for o1 in (0, 1):
            for o2 in (0, 1):
                off = [0, 0, 0]
                others = [d for d in range(3) if d != ax]
                off[others[0]], off[others[1]] = o1, o2
                a_ = (I + off[0], J + off[1], K + off[2])
                bo = list(off)
                bo[ax] += 1
                b_ = (I + bo[0], J + bo[1], K + bo[2])
                fa, fb = f[a_], f[b_]
                cross = (fa > .5) != (fb > .5)
                t = np.where(cross, (.5 - fa)/np.where(cross, fb - fa, 1.), 0.)
                p = X[a_] + t[:, None]*(X[b_] - X[a_])
                acc[cross] += p[cross]
                cnt[cross] += 1
    P = acc/np.maximum(cnt, 1)[:, None]
    # quads: each crossing edge between cell centres (i,j,k)-(i+e) is shared by 4 dual cubes
    quads = []
    for ax in range(3):
        sl_a = [slice(None)]*3
        sl_b = [slice(None)]*3
        sl_a[ax] = slice(0, -1)
        sl_b[ax] = slice(1, None)
        ia, ib = inside[tuple(sl_a)], inside[tuple(sl_b)]
        e = ia != ib
        u, v = [d for d in range(3) if d != ax]
        # edges on the outer layer have fewer than 4 cubes
        valid = np.ones_like(e)
        for d_ in (u, v):
            for end in (0, -1):
                sl = [slice(None)]*3
                sl[d_] = end
                valid[tuple(sl)] = False
        E = np.argwhere(e & valid)
        if not len(E):
            continue
        def cube(du, dv):
            q = E.copy()
            q[:, u] -= du
            q[:, v] -= dv
            return vid[q[:, 0], q[:, 1], q[:, 2]]
        q = np.stack([cube(1, 1), cube(0, 1), cube(0, 0), cube(1, 0)], 1)
        out = ia[tuple(E.T)]                            # inside at the low end: normal along +ax
        flip = ~out if (u, v) in ((1, 2), (2, 0), (0, 1)) else out
        q[flip] = q[flip][:, ::-1]
        quads.append(q[(q >= 0).all(1)])
    Q = np.concatenate(quads)
    tris = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    nV = len(P)
    # Taubin smoothing (few iterations: the blur already rounded the field)
    E2 = np.unique(np.sort(np.concatenate([Q[:, [0, 1]], Q[:, [1, 2]], Q[:, [2, 3]], Q[:, [3, 0]]]), 1), axis=0)
    from scipy.sparse import coo_matrix
    A = coo_matrix((np.ones(2*len(E2)), (np.r_[E2[:, 0], E2[:, 1]], np.r_[E2[:, 1], E2[:, 0]])), shape=(nV, nV)).tocsr()
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    for _ in range(smooth_iters):
        for lam in (.5, -.53):
            P = P + lam*(A @ P/deg[:, None] - P)
    # the bead fan (bead_grid, below) replaces the cells' faces wholly inside its slice ranges: they go before the
    # per-vertex work (those straddling a range's ends stay: the fan's end columns lie on the same original surface, so
    # the two overlap there instead of leaving a gap)
    ranges, parts = [], []
    try:
        ranges = _bead_ranges(proc)
        if ranges:
            # the original metal's occupancy in one section of the bead's first range, blurred the same way
            o_ = proc.pool.orig.reshape(g.nx, g.ny, g.nz)[ranges[0][0], ja:jb, ka:kb].astype(np.float32)
            f2o = gaussian_filter(o_, (sig[1], sig[2]), mode="nearest")
            parts = [bead_grid(proc, gaussian_filter(occ[i0 + 1:i1 + 2], (0, sig[1], sig[2]), mode="nearest"), f2o, yc,
                               zc, i0, i1) for i0, i1 in ranges]
    except Exception as exc:        # a pool model whose state the fan cannot read: the cells are drawn, as before
        import warnings
        warnings.warn(f"bead fan not drawn ({type(exc).__name__}: {exc}); drawing the cells' surface")
        ranges, parts = [], []
    if ranges:
        tx = P[tris, 0]
        fnx = np.cross(P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]])
        fnx = np.abs(fnx[:, 0])/np.maximum(np.linalg.norm(fnx, axis=1), 1e-30)
        cut = np.zeros(len(tris), bool)
        for i0, i1 in ranges:
            xlo, xhi = float(g.xe[i0]), float(g.xe[i1 + 1])
            in_rng = ((tx > xlo + 1e-7) & (tx < xhi - 1e-7)).all(1)
            # faces turned along the weld within a slice of a range's end stay: the coupon's end faces there (the
            # bracket's end, whose vertices the smoothing pulls just inside the range: cut, they leave a slot open into
            # the bracket at its corner)
            end_face = (fnx > .7) & ((np.abs(tx - xlo) < 7.5e-4) | (np.abs(tx - xhi) < 7.5e-4)).all(1)
            cut |= in_rng & ~end_face
        tris = tris[~cut]
        used = np.unique(tris)
        remap = -np.ones(len(P), np.int64)
        remap[used] = np.arange(len(used))
        P, tris = P[used], remap[tris]
        nV = len(P)
    fn = np.cross(P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]])
    N = np.zeros_like(P)
    for k_ in range(3):
        np.add.at(N, tris[:, k_], fn)
    N /= np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]
    # per-vertex fields: the solid cell nearest each vertex (temperature, peak temperature, weld metal)
    ii = np.clip(np.searchsorted(xc, P[:, 0]), 1, nx - 1)
    jj = np.clip(np.searchsorted(yc, P[:, 1]), 1, ny - 1)
    kk = np.clip(np.searchsorted(zc, P[:, 2]), 1, nz - 1)
    best = np.zeros(nV, bool)
    vT, vpk, vw, vfl = np.zeros(nV), np.zeros(nV), np.zeros(nV), np.zeros(nV)
    for di in (-1, 0):
        for dj in (-1, 0):
            for dk in (-1, 0):
                a_ = (ii + di, jj + dj, kk + dk)
                ok = inside[a_] & ~best
                vT[ok], vpk[ok], vw[ok], vfl[ok] = T[a_][ok], pk[a_][ok], wl[a_][ok], fl[a_][ok]
                best |= ok
    # the glow follows the temperature interpolated between the solid cells around each vertex (trilinear over metal
    # cells only): the nearest cell's value drew the hot region's edge as 0.25-0.5 mm steps, the blackbody luminance
    # rising 13x from 1000 to 1200 C
    vTi = _metal_interp(T, occ > .5, xc, yc, zc, P)
    vT = np.where(np.isfinite(vTi), vTi, vT)
    # oxidation of beads laid with air under the arc, and its soot beside them (the pool's slice state)
    ox, soot, _ = slice_states(proc, P[:, 0])
    col, rough, metal = surface_look(P, vpk, vw, ox, soot*soot_weight(P), vT, shield)
    gl = glow(vT).astype(np.float32)
    # the bead drawn from the pool model's profiles (bead_grid) replaces the cells' surface over the slices holding weld
    # metal or a profile; its liquid part is the pool's surface (returned as "pool")
    pool_out = None
    if ranges:
        meshes = [_grid_mesh(G_["P"]) for G_ in parts]
        offs = np.cumsum([0] + [len(m_[0]) for m_ in meshes])
        gP = np.concatenate([m_[0] for m_ in meshes])
        gF = np.concatenate([m_[1] + o for m_, o in zip(meshes, offs)])
        G = dict(center=parts[0]["center"], r=np.concatenate([G_["r"].reshape(-1) for G_ in parts]),
                 r0=np.concatenate([np.broadcast_to(G_["r0"][None, :], G_["r"].shape).reshape(-1) for G_ in parts]),
                 src=np.concatenate([G_["src"].reshape(-1) for G_ in parts]))
        # outward normals: faces turned so their normal points away from the fan's centre line
        rad = gP.copy()
        rad[:, 0] = 0.
        rad[:, 1:] -= G["center"][None, :]
        fn_ = np.cross(gP[gF[:, 1]] - gP[gF[:, 0]], gP[gF[:, 2]] - gP[gF[:, 0]])
        flip = np.einsum("ij,ij->i", fn_, rad[gF].mean(1)) < 0
        gF[flip] = gF[flip][:, ::-1]
        fn_ = np.cross(gP[gF[:, 1]] - gP[gF[:, 0]], gP[gF[:, 2]] - gP[gF[:, 0]])
        gN = np.zeros_like(gP)
        for k_ in range(3):
            np.add.at(gN, gF[:, k_], fn_)
        gN /= np.maximum(np.linalg.norm(gN, axis=1), 1e-12)[:, None]
        q_ = gP - gN*1.5e-4                                   # just inside the metal
        solid_ = occ > .5
        gT = _metal_interp(T, solid_, xc, yc, zc, q_)
        gpk = _metal_interp(pk, solid_, xc, yc, zc, q_)
        gfl = _metal_interp(fl, solid_, xc, yc, zc, q_)
        gT = np.where(np.isfinite(gT), gT, 20.)
        gpk = np.where(np.isfinite(gpk), gpk, 20.)
        gfl = np.where(np.isfinite(gfl), gfl, 0.)
        from . import params as PR_
        dep = G["r"] - G["r0"]
        gw = np.maximum(np.clip(dep/1e-4, 0., 1.), np.clip((gpk - PR_.v(PR_.STEEL_SOLIDUS))/40., 0., 1.)*(dep > -3e-3))
        # liquid weight: the pool's edge is drawn at fl = 0.5, the pool model's own liquid criterion (a slice is liquid
        # at fl >= 0.5, a toe fused at peak fl >= 0.5), blended over POOL_EDGE_FL. (A blend over fl 0.3-0.7 would draw
        # the solver's flat mushy tail, fl 0.2-0.6 over 5 mm behind a 250 A pool, as a 5 mm gradient at the pool's
        # trailing edge instead of the crisp solidification front seen in process-camera frames)
        wl_ = np.clip((gfl - POOL_EDGE_FL[0])/(POOL_EDGE_FL[1] - POOL_EDGE_FL[0]), 0., 1.)
        wl_ = wl_*wl_*(3 - 2*wl_)
        gox, gsoot, _ = slice_states(proc, gP[:, 0])
        gcol, grough, gmetal = surface_look(gP, gpk, gw, gox, gsoot*soot_weight(gP), gT, shield)
        gcol = gcol*(1 - wl_)[:, None] + (1. - EMISSIVITY_LIQUID)*wl_[:, None]
        lr = np.full(len(gP), LIQUID_ROUGHNESS)
        if proc.src.st.arc and proc.root is not None:
            d2 = np.sum((gP - np.asarray(proc.root, float)[None, :])**2, 1)
            lr = POOL_RIPPLE_ROUGHNESS + (POOL_AGITATED_ROUGHNESS - POOL_RIPPLE_ROUGHNESS)*np.exp(-.5*d2/POOL_AGITATED_R**2)
        grough = grough*(1 - wl_) + lr*wl_
        gmetal = gmetal*(1 - wl_) + wl_
        ggl = glow(gT)*(1 - wl_)[:, None] + glow(gT, EMISSIVITY_LIQUID)*wl_[:, None]
        liq_t = wl_[gF].max(1) > .02
        if liq_t.any():
            used = np.unique(gF[liq_t])
            remap = -np.ones(len(gP), np.int64)
            remap[used] = np.arange(len(used))
            pool_out = dict(points=gP[used].astype(np.float32), tris=remap[gF[liq_t]].astype(np.int32),
                            normals=gN[used].astype(np.float32), color=gcol[used].astype(np.float32),
                            glow=ggl[used].astype(np.float32), weldfrac=np.ones(len(used), np.float32),
                            roughness=grough[used].astype(np.float32), metalness=gmetal[used].astype(np.float32),
                            slices=[int(i) for i, st in enumerate(proc.pool.slices) if st.liquid],
                            tmax=float(gT[used].max()), liquid_weight=wl_[used].astype(np.float32))
        n0 = len(P)
        P = np.concatenate([P, gP])
        N = np.concatenate([N, gN])
        tris = np.concatenate([tris, gF[~liq_t] + n0])
        col = np.concatenate([col, gcol]).astype(np.float32)
        rough = np.concatenate([rough, grough]).astype(np.float32)
        metal = np.concatenate([metal, gmetal]).astype(np.float32)
        gl = np.concatenate([gl, ggl.astype(np.float32)])
        vw = np.concatenate([vw, gw*(wl_ < .5)])
        vT = np.concatenate([vT, gT])
        src = G["src"]
        bead_info = dict(ranges=[[int(a_), int(b_)] for a_, b_ in ranges], vertices=int(len(gP)),
                         profile_frac=float((src == 1).mean()), cells_frac=float((src == 2).mean()))
    else:
        bead_info = None
    # open pores in the frozen bead (pore_sites: the slices' porosity): a hole in the surface and the pore's cup
    sites = pore_sites(proc, P, N, vw > .5)
    if sites:
        from scipy.spatial import cKDTree
        cen = P[tris].mean(1)
        bead_t = (vw[tris] > .5).all(1)
        drop = np.zeros(len(tris), bool)
        tree = cKDTree(cen)
        for c_, n_, r_ in sites:
            for t_ in tree.query_ball_point(c_, r_):
                drop[t_] |= bead_t[t_]
        tris = tris[~drop]
        cP, cT, cN, cC, cG, cR, cM, cW = [], [], [], [], [], [], [], []
        n0 = len(P)
        near = cKDTree(P).query(np.array([c_ for c_, _, _ in sites]))[1]
        for (c_, n_, r_), v_ in zip(sites, near):
            pts, tr, nr = pore_cup(c_ - n_*4e-5, n_, r_*1.3, r_*1.1)
            cT.append(tr + n0 + sum(len(q) for q in cP))
            cP.append(pts)
            cN.append(nr)
            # a gas pore's wall formed inside the metal, out of the air: smooth bright weld metal [estimate]
            cC.append(np.tile(srgb_to_linear(WELD_SRGB), (len(pts), 1)))
            cG.append(np.tile(gl[v_], (len(pts), 1)))
            cR.append(np.full(len(pts), .25))
            cM.append(np.full(len(pts), .85))
            cW.append(np.ones(len(pts)))
        P = np.concatenate([P] + cP)
        N = np.concatenate([N] + cN)
        tris = np.concatenate([tris] + cT)
        col = np.concatenate([col] + cC).astype(np.float32)
        gl = np.concatenate([gl] + cG).astype(np.float32)
        rough = np.concatenate([rough] + cR).astype(np.float32)
        metal = np.concatenate([metal] + cM).astype(np.float32)
        vw = np.concatenate([vw] + cW)
    wx = xc[np.nonzero(wl.any(axis=(1, 2)))[0]]
    sp = spatter_balls(proc, (float(wx.min()), float(wx.max())) if len(wx) else None)
    if sp is not None:
        sP, sT = sp
        n0 = len(P)
        sN = np.zeros_like(sP)
        sGl = np.zeros((len(sP), 3), np.float32)
        Tfull = s.field("T").reshape(g.nx, g.ny, g.nz)
        for k_ in range(0, len(sP), 6):
            c_ = sP[k_:k_ + 6].mean(0)
            v_ = sP[k_:k_ + 6] - c_
            sN[k_:k_ + 6] = v_/np.maximum(np.linalg.norm(v_, axis=1), 1e-12)[:, None]
        # a stuck ball is at the temperature of the metal it sits on (it froze onto it): its glow is that metal's
        # (drawn black on red-hot metal it shows as black specks in the weld camera)
        cb = sP.reshape(-1, 6, 3).mean(1)
        cb = cb - np.where(np.abs(cb[:, 1:2]) < np.abs(cb[:, 2:3]), [[0, 1e-4, 0]], [[0, 0, 1e-4]])
        bi_ = np.clip(np.searchsorted(g.xe, cb[:, 0]) - 1, 0, g.nx - 1)
        bj_ = np.clip(np.searchsorted(g.ye, cb[:, 1]) - 1, 0, g.ny - 1)
        bk_ = np.clip(np.searchsorted(g.ze, cb[:, 2]) - 1, 0, g.nz - 1)
        sGl = np.repeat(glow(Tfull[bi_, bj_, bk_]), 6, axis=0).astype(np.float32)
        P = np.concatenate([P, sP])
        N = np.concatenate([N, sN])
        tris = np.concatenate([tris, sT + n0])
        col = np.concatenate([col, np.tile(srgb_to_linear((46, 46, 50)), (len(sP), 1)).astype(np.float32)])
        rough = np.concatenate([rough, np.full(len(sP), .35, np.float32)])
        metal = np.concatenate([metal, np.full(len(sP), .6, np.float32)])
        gl = np.concatenate([gl, sGl])
        vw = np.concatenate([vw, np.ones(len(sP))])
    return dict(points=P.astype(np.float32), tris=tris.astype(np.int32), normals=N.astype(np.float32),
                color=col.astype(np.float32), glow=gl, weldfrac=vw.astype(np.float32), roughness=rough,
                metalness=metal, tmax=float(vT.max()), pores=len(sites), pool=pool_out, bead=bead_info)


# ------------------------------------------------------------------------------------------------ robot visuals
def _origin(el):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    if el is not None:
        T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in el.get("rpy", "0 0 0").split()]).as_matrix()
        T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    return T


# robot body labels '<instance>/<URDF robot name>/<link>' -> catalog model; '<model>_bare' draws an arm whose Franka Hand
# is not on it (a torch arm: robots/backends/newton.py builds it without the hand)
ROBOT_LABEL_MODELS = {"ur10e": "ur10e", "fr3": "fr3", "panda": "panda", "g1_29dof_with_hand_rev_1_0": "g1"}


@lru_cache(maxsize=8)
def _urdf(model):
    from robots.assets import resolve
    if model == "panda_bare":
        from robots.backends.newton import _panda_without_hand
        p = Path(resolve("panda"))
        return p, ET.fromstring(_panda_without_hand(p))
    p = Path(resolve(model.removesuffix("_bare")))
    if model == "fr3":
        p = p.with_name("fr3_franka_hand.urdf")
    return p, ET.parse(p).getroot()


def _mesh_path(urdf_path, fn):
    if fn.startswith("package://"):
        rel = fn[len("package://"):].split("/", 1)[1]
        return urdf_path.parent.parent/rel
    return (urdf_path.parent/fn).resolve()


def _load_mesh(path, T):
    import trimesh
    sc = trimesh.load(str(path), force="scene")
    parts = []
    for node in sc.graph.nodes_geometry:
        M, gname = sc.graph[node]
        g = sc.geometry[gname]
        V = np.asarray(g.vertices, float) @ M[:3, :3].T + M[:3, 3]
        V = V @ T[:3, :3].T + T[:3, 3]
        col = (200, 200, 200)
        mat = getattr(g.visual, "material", None)
        for attr in ("baseColorFactor", "main_color", "diffuse"):
            cc = getattr(mat, attr, None) if mat is not None else None
            if cc is not None:
                col = tuple(int(v) for v in np.asarray(cc)[:3])
                break
        parts.append((V.astype(np.float32), np.asarray(g.faces, np.int32), col))
    return parts


@lru_cache(maxsize=64)
def merged_link_visuals(model, link):
    """Visual parts [(V, F, srgb)] of a URDF link and every link fixed to it (Newton collapses fixed joints), in the
    link's frame."""
    path, root = _urdf(model)
    links = {l.get("name"): l for l in root.findall("link")}
    children = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j)
    out = []

    def walk(name, T):
        el = links[name]
        for vis in el.findall("visual"):
            m = vis.find("geometry/mesh")
            if m is None:
                continue
            out.extend(_load_mesh(_mesh_path(path, m.get("filename")), T @ _origin(vis.find("origin"))))
        for j in children.get(name, []):
            if j.get("type") == "fixed":
                walk(j.find("child").get("link"), T @ _origin(j.find("origin")))
    walk(link, np.eye(4))
    return out


def robot_roots():
    """(model, root link) of each robot model: drawn at its mount pose (Newton merges a fixed root into the world)."""
    return {"ur10e": "base_link", "fr3": "fr3_link0", "fr3_bare": "fr3_link0", "panda": "panda_link0",
            "panda_bare": "panda_link0"}        # the G1 floats: its pelvis is a body, drawn with the others


# ------------------------------------------------------------------------------------------------ frames
def world_frame(world, label="UNASSISTED"):
    """A recorded frame of the world's physical state for the renderer."""
    w = world
    q = w.state.body_q.numpy().astype(float)
    labels = list(w.model.body_label)
    frame = dict(tick=int(w.tick), time_s=float(w.time), label=label, bodies=labels, body_q=q.tolist())
    p = w.process
    if p is not None:
        o, R = w.weld_frame()
        st = p.src.st
        tip, axis, tcp = w.torch_pose()
        # the shielding gas over the work (film_air): the wire's axis from the contact tip, while the gas flows
        shield = dict(tip=(R.T @ (np.asarray(tip, float) - o)).tolist(), axis=(R.T @ np.asarray(axis, float)).tolist(),
                      x_air=float(p.x_air) if st.gas_on else 1.)
        surf = band_surface(p, shield=shield)
        frame["band"] = surf
        frame["coupon_look"] = coupon_look(p, coupon_meshes(w.coupon, band_edges(p.g)[1]), shield)
        arc = None
        stray = getattr(w, "stray_arc", None)       # an arc burning on a grounded body off the coupon (world.py)
        if st.arc and (p.root is not None or stray):
            root_w = o + R @ np.asarray(p.root, float) if p.root is not None else np.asarray(stray["root_w"], float)
            # the column runs from the wire's end (the stick-out along the torch axis) to the arc root
            wire_end = np.asarray(tip, float) + np.asarray(axis, float)*float(st.l_s)
            arc = dict(root=root_w.tolist(), current_a=float(st.current), voltage_v=float(st.v_term),
                       power_w=float(st.current*st.v_term), length_m=float(st.l_a), wire_end=wire_end.tolist(),
                       mode=str(getattr(st, "mode", "") or ""))
        frame["arc"] = arc
        frame["wire"] = dict(tip=tip.tolist(), axis=axis.tolist(), length_m=float(st.l_s) if st.on_cmd or st.l_s else .0)
        # the pool's free surface: the liquid part of the bead drawn from the pool model's profiles (band_surface); the
        # liquid slices' own profile mesh (pool_surface) if the band has none
        frame["pool"] = surf.pop("pool", None) if surf is not None else None
        if frame["pool"] is None and surf is not None and surf.get("bead") is None:
            frame["pool"] = pool_surface(p)
        fly = spatter_in_flight(p)
        if fly is not None and surf is not None:
            fP, fT, fN, fG = fly
            n0 = len(surf["points"])
            surf["points"] = np.concatenate([surf["points"], fP.astype(np.float32)])
            surf["normals"] = np.concatenate([surf["normals"], fN.astype(np.float32)])
            surf["tris"] = np.concatenate([surf["tris"], (fT + n0).astype(np.int32)])
            surf["color"] = np.concatenate([surf["color"], np.full((len(fP), 3), 1. - EMISSIVITY_LIQUID, np.float32)])
            surf["glow"] = np.concatenate([surf["glow"], fG.astype(np.float32)])
            surf["weldfrac"] = np.concatenate([surf["weldfrac"], np.ones(len(fP), np.float32)])
            surf["roughness"] = np.concatenate([surf["roughness"], np.full(len(fP), LIQUID_ROUGHNESS, np.float32)])
            surf["metalness"] = np.concatenate([surf["metalness"], np.ones(len(fP), np.float32)])
            surf["flying"] = int(len(fP)//6)
    # the loose bracket's cells leave the thermal grid when it leaves its fit-up pose (world.bracket_in_grid, the lead's
    # physics change): the whole rigid box is then drawn on the bracket's pose, without the band's carve-out
    frame["bracket_in_grid"] = bool(getattr(w, "bracket_in_grid", True))
    frame["fume"] = fume_plume(w) if p is not None else None
    pb, Rb = w.body_pose(w.scene.gun_body)
    pt, Rt = w.scene.tool0_in_body
    rear = pb + Rb @ (pt + Rt @ HOSE_ATTACH_TOOL)
    # the follow camera's point: the torch work point projected on the joint's root line (weld frame x)
    o, R = w.weld_frame()
    tcp = pb + Rb @ (pt + Rt @ w.scene.gun_spec.tcp)
    xw = float(np.clip((R.T @ (tcp - o))[0], -.02, w.coupon.length + .02))
    frame["follow"] = (o + R @ np.array([xw, 0., 0.])).tolist()
    from . import shop as SH
    frame["curves"] = {"/World/Hose/curve": SH.hose_points(w, rear).tolist(),
                       "/World/Cables/ground": SH.work_cable_points(w).tolist()}
    frame["curves"].update(assist_arrows(w))
    return frame


ASSIST_ARROW_M_PER_N = 5e-5     # the drawn phantom-force arrow: 50 mm per kN [authored]


def assist_arrows(w):
    """ASSISTED evidence made visible: each phantom force (world.assist_force) drawn as a magenta arrow ending at its
    point of application, length proportional to the force; parked under the floor when none acts."""
    items = w.assist_force if isinstance(w.assist_force, list) else ([w.assist_force] if w.assist_force else [])
    out = {}
    for k in range(2):
        P = np.array([[0., 0., -5.], [0., 0., -5.01], [0., 0., -5.02], [0., 0., -5.03]])
        if k < len(items):
            body, pt, fn = items[k]
            F = np.asarray(fn(w.time), float)
            f = float(np.linalg.norm(F))
            if f > 1.:
                u = F/f
                pt = np.asarray(pt, float)
                tail = pt - u*max(.02, f*ASSIST_ARROW_M_PER_N)
                side = np.cross(u, [0, 0, 1.]) if abs(u[2]) < .9 else np.cross(u, [1., 0, 0])
                side /= np.linalg.norm(side)
                P = np.array([tail, pt, pt - u*.012 + side*.006, pt])
        out[f"/World/Assist/arrow_{k}"] = P.tolist()
    return out


# ------------------------------------------------------------------------------------------------ USD export
def weld_camera_in_tool0(gun):
    """(eye, target, up) of the gun's process camera in the flange frame: LY.WELD_CAMERA's place relative to the work
    point at the procedure's angles, carried rigidly by the gun; it looks at the work point, the weld frame's z up."""
    from scipy.spatial.transform import Rotation
    from .kinematics import torch_orientation, torch_tcp_in_tool0
    Rw = Rotation.from_quat(torch_orientation(*LY.WELD_CAMERA["procedure_angles"])).as_matrix()
    T = torch_tcp_in_tool0(gun)
    e = T[:3, :3] @ (Rw.T @ np.asarray(LY.WELD_CAMERA["eye_work"])) + T[:3, 3]
    return e, T[:3, 3].copy(), T[:3, :3] @ (Rw.T @ np.array([0., 0., 1.]))


def _camera(stage, path, eye, target, hfov, role, up=(0., 0., 1.)):
    from pxr import Gf, UsdGeom
    c = UsdGeom.Camera.Define(stage, path)
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(*map(float, up)))
    c.AddTransformOp().Set(m.GetInverse())
    c.CreateHorizontalApertureAttr(36.)
    c.CreateVerticalApertureAttr(20.25)
    c.CreateFocalLengthAttr(18./math.tan(math.radians(hfov)/2))
    c.CreateClippingRangeAttr(Gf.Vec2f(.004, 60.))
    c.GetPrim().SetCustomDataByKey("observation_role", role)
    c.GetPrim().SetCustomDataByKey("horizontal_fov_deg", float(hfov))
    return c


class _Author:
    """Small USD authoring helper (materials cached by name)."""

    def __init__(self, stage, folder):
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
        self.stage, self.folder = stage, Path(folder)
        self.Gf, self.Sdf, self.UsdGeom, self.UsdShade, self.Vt = Gf, Sdf, UsdGeom, UsdShade, Vt
        self.mats = {}
        UsdGeom.Scope.Define(stage, "/Looks")

    def pbr(self, name, srgb, rough=.5, metal=0., diff=None, rough_tex=None, normal=None, tile_m=1., opacity=None,
            emissive=None, emissive_nits=0.):
        if name in self.mats:
            return self.mats[name]
        Sdf, Gf, UsdShade = self.Sdf, self.Gf, self.UsdShade
        m = UsdShade.Material.Define(self.stage, "/Looks/" + name)
        sh = UsdShade.Shader.Define(self.stage, str(m.GetPath()) + "/Shader")
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        inp = lambda k, t, v: sh.CreateInput(k, t).Set(v)
        inp("diffuse_color_constant", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*map(float, srgb_to_linear(srgb))))
        inp("reflection_roughness_constant", Sdf.ValueTypeNames.Float, float(rough))
        inp("metallic_constant", Sdf.ValueTypeNames.Float, float(metal))
        inp("texture_scale", Sdf.ValueTypeNames.Float2, Gf.Vec2f(1./tile_m, 1./tile_m))
        if diff is not None:
            inp("diffuse_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(str(Path(diff).resolve())))
        if rough_tex is not None:
            inp("reflectionroughness_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(str(Path(rough_tex).resolve())))
            inp("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float, 1.)
        if normal is not None:
            inp("normalmap_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(str(Path(normal).resolve())))
        if opacity is not None:
            inp("enable_opacity", Sdf.ValueTypeNames.Bool, True)
            inp("opacity_constant", Sdf.ValueTypeNames.Float, float(opacity))
        if emissive is not None:
            inp("enable_emission", Sdf.ValueTypeNames.Bool, True)
            inp("emissive_color", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*map(float, emissive)))
            inp("emissive_intensity", Sdf.ValueTypeNames.Float, float(emissive_nits))
        self.mats[name] = m
        return m

    def glass(self, name, srgb, ior=1.45, thin=True):
        if name in self.mats:
            return self.mats[name]
        Sdf, Gf, UsdShade = self.Sdf, self.Gf, self.UsdShade
        m = UsdShade.Material.Define(self.stage, "/Looks/" + name)
        sh = UsdShade.Shader.Define(self.stage, str(m.GetPath()) + "/Shader")
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
        m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, srgb_to_linear(srgb))))
        sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(float(ior))
        sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(bool(thin))
        self.mats[name] = m
        return m

    def surface(self, name, fallback_srgb=None):
        """welding/weld_surface.mdl: per-vertex base colour and blackbody emission (primvars); meshes without the
        primvars (the coupon's boxes) use `fallback_srgb`, so the band and the boxes shade identically."""
        if name in self.mats:
            return self.mats[name]
        Sdf, UsdShade, Gf = self.Sdf, self.UsdShade, self.Gf
        m = UsdShade.Material.Define(self.stage, "/Looks/" + name)
        sh = UsdShade.Shader.Define(self.stage, str(m.GetPath()) + "/Shader")
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(Sdf.AssetPath(str((ROOT/"weld_surface.mdl").resolve())), "mdl")
        sh.SetSourceAssetSubIdentifier("weld_surface", "mdl")
        m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        if fallback_srgb is not None:
            sh.CreateInput("fallback", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, srgb_to_linear(fallback_srgb))))
        self.mats[name] = m
        return m

    def bind(self, prim, mat):
        from pxr import UsdShade
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim() if hasattr(prim, "GetPrim") else prim).Bind(mat)

    def xform(self, path, pos=(0, 0, 0), quat_xyzw=None):
        x = self.UsdGeom.Xform.Define(self.stage, path)
        x.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, pos)))
        q = (0., 0., 0., 1.) if quat_xyzw is None else quat_xyzw
        x.AddOrientOp().Set(self.Gf.Quatf(float(q[3]), self.Gf.Vec3f(*map(float, q[:3]))))
        return x

    def box(self, path, center, half, mat, quat_xyzw=None, uv=False):
        from pxr import UsdGeom
        c = np.asarray(center, float)
        h = np.asarray(half, float)
        sgn = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
        V = sgn*h
        if quat_xyzw is not None:
            from scipy.spatial.transform import Rotation
            V = Rotation.from_quat(quat_xyzw).apply(V)
        V = V + c
        F = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]
        # per-face vertices so normals are flat
        P, idx, N, ST = [], [], [], []
        for f in F:
            a, b, cc = V[f[0]], V[f[1]], V[f[2]]
            n = np.cross(b - a, cc - a)
            n /= np.linalg.norm(n)
            for k in f:
                idx.append(len(P))
                P.append(V[k])
                N.append(n)
                ax = np.argmax(np.abs(n))
                u, v = [d for d in range(3) if d != ax]
                ST.append((V[k][u], V[k][v]))
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr(self.Vt.Vec3fArray.FromNumpy(np.asarray(P, np.float32)))
        m.CreateFaceVertexCountsAttr([4]*6)
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateNormalsAttr(self.Vt.Vec3fArray.FromNumpy(np.asarray(N, np.float32)))
        m.SetNormalsInterpolation("faceVarying")
        m.CreateSubdivisionSchemeAttr("none")
        if uv:
            pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", self.Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            pv.Set(self.Vt.Vec2fArray.FromNumpy(np.asarray(ST, np.float32)[idx]))
        self.bind(m, mat)
        return m

    def cyl(self, path, p0, p1, r, mat, sides=24, r1=None):
        """A cylinder from p0 to p1 of radius r; a truncated cone if r1 (the radius at p1) is given."""
        r1 = r if r1 is None else r1
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        L = np.linalg.norm(d)
        z = d/L
        x = np.cross(z, [1, 0, 0] if abs(z[0]) < .9 else [0, 1, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        a = np.linspace(0, 2*math.pi, sides, endpoint=False)
        ring = np.cos(a)[:, None]*x + np.sin(a)[:, None]*y
        V = np.concatenate([p0 + r*ring, p1 + r1*ring, [p0, p1]])
        F = []
        for k in range(sides):
            k1 = (k + 1) % sides
            F.append([k, k1, sides + k1, sides + k])
            F.append([2*sides, k1, k])
            F.append([2*sides + 1, sides + k, sides + k1])
        return self.mesh(path, V, F, mat, smooth=True)

    def mesh(self, path, V, F, mat, smooth=False, color=None):
        from pxr import UsdGeom
        m = UsdGeom.Mesh.Define(self.stage, path)
        V = np.asarray(V, np.float32)
        m.CreatePointsAttr(self.Vt.Vec3fArray.FromNumpy(V))
        m.CreateFaceVertexCountsAttr([len(f) for f in F])
        m.CreateFaceVertexIndicesAttr([int(i) for f in F for i in f])
        m.CreateSubdivisionSchemeAttr("none")
        if smooth:
            N = np.zeros_like(V)
            for f in F:
                n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
                for i in f:
                    N[i] += n
            N /= np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]
            m.CreateNormalsAttr(self.Vt.Vec3fArray.FromNumpy(N))
            m.SetNormalsInterpolation("vertex")
        self.bind(m, mat)
        return m

    def tri_mesh(self, path, V, F, mat):
        from pxr import UsdGeom
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr(self.Vt.Vec3fArray.FromNumpy(np.asarray(V, np.float32)))
        m.CreateFaceVertexCountsAttr(self.Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(self.Vt.IntArray.FromNumpy(np.asarray(F, np.int32).ravel()))
        m.CreateSubdivisionSchemeAttr("none")
        self.bind(m, mat)
        return m

    def reference(self, path, usd, pos, yaw_deg=0., scale=1., ground=True):
        """Reference a pinned Poly Haven model, rebind its materials to OmniPBR with the same textures (RTX cannot
        compile their MaterialX normal-map nodes: 'ND_normalmap_float' errors and flat red surfaces; the
        label_application recipe, as filleting uses it) and set its lowest point on z = pos z."""
        from pxr import Usd
        from label_application.dressing import Dresser as LAD
        x = self.UsdGeom.Xform.Define(self.stage, path)
        x.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, pos)))
        x.AddRotateZOp().Set(float(yaw_deg))
        if scale != 1.:
            x.AddScaleOp().Set(self.Gf.Vec3f(scale, scale, scale))
        inner = self.UsdGeom.Xform.Define(self.stage, path + "/asset")
        inner.GetPrim().GetReferences().AddReference(str(Path(usd).resolve()))
        LAD._rebind_omnipbr(self, path, Path(usd).stem.replace("_1k", ""), Path(usd))
        if ground:
            r = self.UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
                self.stage.GetPrimAtPath(path)).ComputeAlignedRange()
            if not r.IsEmpty():
                inner.AddTranslateOp().Set(self.Gf.Vec3d(0., 0., float(pos[2] - r.GetMin()[2])))
        return x


COUPON_STEP = 2.5e-3          # vertex spacing of the coupon pieces' meshes (carries the mottle and the heat tint)
_PIECE_CACHE = {}


def coupon_meshes(coupon, band):
    """The coupon pieces outside the band as meshes in the weld frame: [(body, name, V, F, N)], each face a grid of
    COUPON_STEP so the surface look (surface_look) can vary over it."""
    key = (coupon.kind, tuple(coupon.base), tuple(coupon.web or ()), coupon.base_x0, tuple(band["y"]), tuple(band["z"]))
    if key in _PIECE_CACHE:
        return _PIECE_CACHE[key]
    out = []
    for body, name, c, h in _coupon_pieces(coupon, band):
        c, h = np.asarray(c, float), np.asarray(h, float)
        Vs, Fs, Ns, n0 = [], [], [], 0
        for ax in range(3):
            u, v = [d for d in range(3) if d != ax]
            for sgn in (-1, 1):
                nu = max(1, int(math.ceil(2*h[u]/COUPON_STEP)))
                nv = max(1, int(math.ceil(2*h[v]/COUPON_STEP)))
                a = np.linspace(-h[u], h[u], nu + 1)
                b = np.linspace(-h[v], h[v], nv + 1)
                A_, B_ = np.meshgrid(a, b, indexing="ij")
                V = np.zeros((A_.size, 3))
                V[:, u], V[:, v], V[:, ax] = A_.ravel(), B_.ravel(), sgn*h[ax]
                idx = np.arange(A_.size).reshape(nu + 1, nv + 1)
                q = np.stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()], 1)
                # winding so the normal points out along sgn*ax
                n_test = np.cross(np.eye(3)[u], np.eye(3)[v])[ax]
                if n_test*sgn < 0:
                    q = q[:, ::-1]
                F = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]) + n0
                N = np.zeros_like(V)
                N[:, ax] = sgn
                Vs.append(V + c), Fs.append(F), Ns.append(N)
                n0 += len(V)
        out.append((body, name, np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Ns)))
    _PIECE_CACHE[key] = out
    return out


def coupon_look(proc, pieces, shield=None):
    """Per-vertex (color, roughness, metalness) of every coupon piece from the solver's peak temperatures."""
    out = {}
    for body, name, V, F, N in pieces:
        pk = peak_at(proc, V, N) if proc is not None else np.zeros(len(V))
        Tn = peak_at(proc, V, N, "T") if proc is not None else None
        soot = slice_states(proc, V[:, 0])[1]*soot_weight(V) if proc is not None else None
        out[f"{body}/{name}"] = surface_look(V, pk, np.zeros(len(V)), None, soot, Tn, shield)
    return out


def _coupon_pieces(coupon, band):
    """Visual boxes of the coupon outside the joint band (band: its weld-frame edges on the grid), in the weld frame:
    (body, name, center, half)."""
    bx, by, bz = coupon.base
    x0, x1 = coupon.base_x0, coupon.base_x0 + bx
    y0, y1 = coupon.base_y
    ya, yb = band["y"]
    za, zb = band["z"]
    out = []
    bx_c = (x0 + x1)/2
    out.append(("base_plate", "base_lower", (bx_c, (y0 + y1)/2, (-bz + za)/2), (bx/2, (y1 - y0)/2, (za + bz)/2)))
    out.append(("base_plate", "base_top_back", (bx_c, (y0 + ya)/2, za/2), (bx/2, (ya - y0)/2, -za/2)))
    out.append(("base_plate", "base_top_front", (bx_c, (yb + y1)/2, za/2), (bx/2, (y1 - yb)/2, -za/2)))
    if coupon.kind == "tee":
        wx, wt, wh = coupon.web
        out.append(("bracket", "bracket_upper", (wx/2, -wt/2, (zb + wh)/2), (wx/2, wt/2, (wh - zb)/2)))
        out.append(("bracket", "bracket_lower", (wx/2, (-wt + ya)/2, zb/2), (wx/2, (ya + wt)/2, zb/2)))
        # the bracket's part inside the band (y in [ya, 0], z in [0, zb]): drawn only while the bracket's cells are out
        # of the thermal grid (frame bracket_in_grid False); otherwise the band surface draws it from the cells
        out.append(("bracket", BRACKET_BAND_PIECE, (wx/2, ya/2, zb/2), (wx/2, -ya/2, zb/2)))
    return out


BRACKET_BAND_PIECE = "bracket_band"


# ---- cameras. The worker renders HdrColor at one fixed internal exposure (RENDER_EXPOSURE) and applies each camera's
# model in numpy (render_worker.camera_image): scene luminance L = HdrColor / (RTX_K t S / N^2); the camera's exposure
# H = RTX_K t S / N^2 x L for its own t, S, N; lens veiling glare; the sensor's response (the arc's light must pass
# through the glare at its true luminance, before any curve clips it).
RTX_K = .01152          # HdrColor per (cd/m2 x s x ISO / f-number^2) in this renderer [measured]; ISO 12232's
                        # saturation-based relation, q/78 with q = 0.65, gives 0.0083 [read]
RENDER_EXPOSURE = dict(exposureTime=1/120., fNumber=16., filmIso=100.)   # HdrColor 3.75e-5 per cd/m2: below float16's
                                                                         # 65504 up to 1.7e9 cd/m2 (an arc at ~420 A)
GLARE = .02             # veiling glare: fraction of the light a lens spreads over the image, kernel ~ (r^2 + r0^2)^-1.5
                        # [estimate: veiling glare indices of camera lenses are a few per cent]
# standard cameras (the shop, cell and policy cameras): fixed exposure (the shop's authored values), the renderer's own
# filmic curve as the old pipeline applied it (fitted: ACES in Narkowicz's fit, then sRGB encoding; the
# worker's earlier `rgb` output matched it within 1-3 grey levels) after the lens glare. The arc saturates them and its
# light floods the cell, as it does a real camera without a filter.
# The weld camera (and the films' follow camera) is an HDR process camera of the Xiris XVC-1100 class: > 140 dB dynamic
# range, "simultaneous visibility of high-intensity arc regions and low-light background features in a single image",
# tone mapping among its image controls (Xiris datasheet, read). Model: response linear near the floor and logarithmic
# above it (Xiris: "a logarithmic response ... suppressing the response in the area of the arc intensity and leaving it as
# a near-linear response in the darker region", read), y = ln(1 + L/L_f) / ln(1 + 10^(DR/20)), L_f = L_sat / 10^(DR/20);
# per colour channel (each photosite responds logarithmically, so bright colours desaturate toward white as on the
# sensor). Exposure: the camera carries a "weld arc photodetector" (Xiris datasheet, read) to set its exposure when the
# arc burns; modelled as auto exposure that puts the frame's brightest pixel at saturation L_sat (at least L_sat_min),
# so the arc core is the only white and its falloff, the pool and the cold metal stay inside the 140 dB window; black
# level at the frame's 0.5 percentile, then a display contrast (power 2.2 on the log signal) so that, as in the published
# GMAW frames, the parent metal reads dark, the pool and the arc-lit metal bright and the arc core white (Xiris XVC-750
# fillet frame) (the camera's tone-mapping algorithm is not published) [estimates]. No sensor
# noise is drawn.
HDR_PROCESS = dict(response="hdr_log", L_sat_min=50., dr_db=140., glare=GLARE, display_gamma=2.2)
EXPOSURE = {"default": dict(exposureTime=1/120., fNumber=2.8, filmIso=400., response="filmic", glare=GLARE),
            "weld_camera": HDR_PROCESS,
            "follow": HDR_PROCESS}


def export_scene(folder, frame, world_meta, extra_cameras=None):
    """Static render stage posed at `frame`: shop, table, robots, gun, coupon, clamps, cameras and the dynamic prims
    (band surface, arc, wire). world_meta: dict from scene_meta(world). Writes folder/scene.usdc and export.json."""
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade, Sdf, Vt
    from . import visual_assets as VA
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(folder/"scene.usdc"))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    A = _Author(stage, folder)
    # ---- the building, welding equipment, screens, camera mounts, canopy and dressing (welding/shop.py)
    from . import shop as SH
    SH.author(A, VA, stage)
    # ---- fixture table (nitrided steel, 16 mm holes on a 50 mm grid drawn as a dark ring pattern is not modelled)
    T = LY.TABLE
    top = A.pbr("table_nitrided", (255, 255, 255), rough=.42, metal=.85, diff=table_texture(), tile_m=.5)
    legs = A.pbr("table_legs", (52, 54, 58), rough=.5, metal=.6)
    sx, sy = T["size"]
    A.box("/World/Table/top", (0., 0., T["top_z"] - T["top_thickness"]/2), (sx/2, sy/2, T["top_thickness"]/2), top,
          uv=True)
    for i, x in enumerate((-sx/2 + .06, sx/2 - .06)):
        for j, y in enumerate((-sy/2 + .06, sy/2 - .06)):
            A.box(f"/World/Table/leg_{i}{j}", (x, y, (T["top_z"] - .012)/2), (.04, .04, (T["top_z"] - .012)/2), legs)
    A.box("/World/Table/shelf", (0., 0., .18), (sx/2 - .06, sy/2 - .06, .006), legs)
    # ---- sky outside the windows (the panes are emissive; the dome only lights through the openings) [authored]
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/hall")
    dome.CreateTextureFileAttr(Sdf.AssetPath(str(VA.hdri("machine_shop_02").resolve())))
    dome.CreateIntensityAttr(30.)
    # ---- rigid bodies -> /World/Bodies/b_<i> (posed per frame)
    labels = frame["bodies"]
    q = np.asarray(frame["body_q"], float)
    steel = A.surface("mill_scale", MILL_SCALE_SRGB)
    neoprene = A.pbr("neoprene", (22, 22, 22), rough=.8)
    clamp_red = A.pbr("clamp_paint", (178, 32, 28), rough=.45, metal=.2)
    copper = A.pbr("copper", (205, 120, 80), rough=.35, metal=1.)
    black = A.pbr("gun_jacket", (20, 20, 22), rough=.6)
    gun_grey = A.pbr("gun_body", (70, 72, 76), rough=.35, metal=.7)
    nozzle_mat = A.surface("nozzle", NOZZLE_COPPER_SRGB)   # per-vertex colour, roughness, metalness (nozzle_meshes)
    robot_mats = {}
    body_paths = {}
    for i, lab in enumerate(labels):
        path = f"/World/Bodies/b_{i}"
        A.xform(path, q[i, :3], q[i, 3:7]).GetPrim().SetCustomDataByKey("gar_name", lab)
        body_paths[lab] = path
        bits = lab.split("/")
        if len(bits) >= 3 and bits[-2] in ROBOT_LABEL_MODELS:
            model, link = ROBOT_LABEL_MODELS[bits[-2]], bits[-1]
            model += "_bare" if bits[0] in world_meta.get("bare_arms", ()) else ""
            for k, (V, F, col) in enumerate(merged_link_visuals(model, link)):
                key = f"{model}_{col[0]}_{col[1]}_{col[2]}"
                if key not in robot_mats:
                    metal = .2 if model == "ur10e" else .1
                    robot_mats[key] = A.pbr("robot_" + key, col, rough=.35, metal=metal)
                A.tri_mesh(f"{path}/vis_{k}", V, F, robot_mats[key])
    # robot roots at their mounts
    for arm, info in world_meta["arms"].items():
        model = info["model"] + ("_bare" if arm in world_meta.get("bare_arms", ()) else "")
        root = robot_roots().get(model)
        if root is None:
            continue
        path = f"/World/Robots/{arm}_base"
        A.xform(path, info["mount_pos"], LY.yaw_quat(info["mount_yaw_deg"]))
        for k, (V, F, col) in enumerate(merged_link_visuals(model, root)):
            key = f"{model}_{col[0]}_{col[1]}_{col[2]}"
            if key not in robot_mats:
                robot_mats[key] = A.pbr("robot_" + key, col, rough=.35, metal=.2)
            A.tri_mesh(f"{path}/vis_{k}", V, F, robot_mats[key])
    # the gun on the torch arm's last link
    gb = body_paths[world_meta["gun_body_label"]]
    pt, Rt = np.asarray(world_meta["tool0_pos"]), np.asarray(world_meta["tool0_R"])
    gun = TO.for_model(world_meta["gun_model"])
    for name, kind, c, Rp, dims, mass in gun.parts():
        axis = Rt @ (Rp @ np.array([0, 0, 1.]))
        center = pt + Rt @ c
        if name.startswith("gun_nozzle"):
            # drawn as the one smooth tapered nozzle its collision segments approximate (the stacked cylinders read as
            # copper bands in the films), hollow: its 5/8 in bore open at the end with the contact tip recessed in it
            # (a closed cone whose cap shares smoothed normals with its side reads as a glass bell)
            if name == "gun_nozzle":
                for V_, F_, part, look in nozzle_meshes(pt + Rt @ gun.nozzle_start, pt + Rt @ gun.nozzle_end,
                                                        gun.nozzle[0], gun.nozzle[1]):
                    _set_look(A.mesh(f"{gb}/gun_nozzle_{part}", V_, F_, nozzle_mat, smooth=True), *look)
            continue
        mat = {"gun_mount": gun_grey, "gun_body": black, "gun_neck": black}[name]
        A.cyl(f"{gb}/{name}", center - axis*dims[1], center + axis*dims[1], dims[0], mat)
    tip_end = pt + Rt @ gun.contact_tip
    A.cyl(f"{gb}/contact_tip", tip_end - Rt @ gun.axis*.012, tip_end, .003, copper)
    clamp = TO.MOUNTS.get(world_meta["gun_model"])
    if clamp is not None:                            # the torch's clamp on a hand (the G1's): its collider box
        A.box(f"{gb}/gun_clamp", clamp.clamp_center, clamp.clamp_half, gun_grey)
    # coupon pieces outside the band (weld frame -> body frame), fine meshes whose look follows the peak temperature
    looks = frame.get("coupon_look") or {}
    piece_paths = {}
    for body, name, V, F, N in coupon_meshes(world_meta["coupon"], world_meta["band"]):
        off = np.asarray(world_meta["weld_origin_in"][body])
        path = f"{body_paths[body]}/{name}"
        m = A.tri_mesh(path, V - off, F, steel)
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(N.astype(np.float32)))
        m.SetNormalsInterpolation("vertex")
        col, rough, metal = looks.get(f"{body}/{name}") or surface_look(V, np.zeros(len(V)), np.zeros(len(V)))
        _set_look(m, col, rough, metal)
        piece_paths[f"{body}/{name}"] = path
        if name == BRACKET_BAND_PIECE and frame.get("bracket_in_grid", True):
            UsdGeom.Imageable(m).MakeInvisible()
    # clamps: DESTACO 225-U class vertical hold-down (zinc-plated steel, red vinyl grip, neoprene cap), static except
    # the sprung pad body (scene.py)
    zinc = A.pbr("zinc", (168, 170, 172), rough=.35, metal=.9)
    grip = A.pbr("clamp_grip", (186, 30, 26), rough=.55)
    for C_ in LY.CLAMPS:
        x, y = C_["pad_xy"]
        zp = LY.WELD_ORIGIN[2]
        zt = LY.TABLE["top_z"]
        p = f"/World/Clamps/{C_['name']}"
        A.box(p + "/flange", (x, y - .085, zt + .003), (.032, .022, .003), zinc)
        A.box(p + "/base_l", (x - .011, y - .085, zt + .03), (.003, .018, .027), zinc)
        A.box(p + "/base_r", (x + .011, y - .085, zt + .03), (.003, .018, .027), zinc)
        A.box(p + "/bar_l", (x - .006, y - .045, zp + .038), (.003, .052, .007), zinc)
        A.box(p + "/bar_r", (x + .006, y - .045, zp + .038), (.003, .052, .007), zinc)
        A.cyl(p + "/spindle", (x, y, zp + .008), (x, y, zp + .052), .004, zinc)
        A.cyl(p + "/nut", (x, y, zp + .044), (x, y, zp + .050), .0065, zinc)
        A.cyl(p + "/lever", (x, y - .085, zt + .055), (x, y - .095, zt + .125), .005, zinc)
        A.cyl(p + "/grip", (x, y - .095, zt + .125), (x, y - .103, zt + .19), .011, grip)
        pad = body_paths.get(f"{C_['name']}_pad")
        if pad:
            A.cyl(f"{pad}/cap", (0, 0, -LY.CLAMP_PAD["height"]/2), (0, 0, LY.CLAMP_PAD["height"]/2),
                  LY.CLAMP_PAD["radius"], neoprene)
    wc = body_paths.get("work_clamp")
    if wc:
        sx_, sy_, sz_ = LY.WORK_CLAMP["size"]
        housing = A.pbr("magnet_housing", (150, 28, 22), rough=.5, metal=.2)
        A.box(f"{wc}/housing", (0, 0, -.004), (sx_/2, sy_/2, sz_/2 - .004), housing)
        A.box(f"{wc}/pole", (0, 0, -sz_/2 + .002), (sx_/2 - .003, sy_/2 - .003, .002), zinc)
        A.cyl(f"{wc}/knob", (-.02, 0, sz_/2 - .008), (-.02, 0, sz_/2 + .006), .009, A.pbr("knob", (25, 25, 25), rough=.6))
        A.box(f"{wc}/lug", (.022, 0, sz_/2 - .002), (.016, .009, .004), copper)
    # work cable (source stud -> ground clamp) and torch hose (feeder -> along the torch arm -> gun): posed per frame
    cable = A.pbr("cable_rubber", (18, 18, 19), rough=.7)
    _curve(stage, "/World/Cables/ground", np.asarray(frame["curves"]["/World/Cables/ground"]), .012, cable)
    _curve(stage, "/World/Hose/curve", np.asarray(frame["curves"]["/World/Hose/curve"]), .024, cable)
    # phantom-force arrows (ASSISTED probes only; parked under the floor otherwise)
    assist = A.pbr("assist_arrow", (230, 40, 200), rough=.4, emissive=(.9, .15, .8), emissive_nits=400.)
    for k in range(2):
        c_ = _curve(stage, f"/World/Assist/arrow_{k}", np.asarray(frame["curves"][f"/World/Assist/arrow_{k}"]), .006,
                    assist)
        c_.CreateTypeAttr().Set("linear")
        c_.CreateWrapAttr().Set("nonperiodic")             # 'pinned' applies to cubic curves only
    # ---- dynamic: band surface (weld frame, under the base plate), arc, wire
    band_parent = body_paths["base_plate"] + "/band"
    off = np.asarray(world_meta["weld_origin_in"]["base_plate"])
    A.xform(band_parent, -off)
    bm = UsdGeom.Mesh.Define(stage, band_parent + "/surface")
    bm.CreateSubdivisionSchemeAttr("none")
    A.bind(bm, A.surface("weld_band", MILL_SCALE_SRGB))
    # the pool's free surface (pool_surface; same per-vertex material as the band, liquid-metal values)
    pm = UsdGeom.Mesh.Define(stage, band_parent + "/pool")
    pm.CreateSubdivisionSchemeAttr("none")
    A.bind(pm, A.surface("weld_band", MILL_SCALE_SRGB))
    UsdGeom.Imageable(pm).MakeInvisible()
    # the arc's light: one sphere light carrying the column's luminous intensity (arc_light; luminance units measured),
    # invisible to camera rays (the worker composites the seen arc and its mirror image in the pool at true luminance)
    UsdGeom.Xform.Define(stage, "/World/Arc")
    lc = np.asarray(ARC_COLOR, float)
    lc = lc/float(lc @ np.array([.2126, .7152, .0722]))
    L_ = UsdLux.SphereLight.Define(stage, "/World/Arc/light")
    L_.AddTranslateOp().Set(Gf.Vec3d(0., 0., -10.))
    L_.CreateRadiusAttr(ARC_LIGHT_R)
    L_.CreateIntensityAttr(0.)
    L_.CreateColorAttr(Gf.Vec3f(*map(float, lc)))
    L_.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
    L_.CreateDiffuseAttr(1.)
    L_.CreateSpecularAttr(0.)
    UsdGeom.Imageable(L_).MakeInvisible()
    # the column's reflections (arc_light spec_chain): specular-only cylinder lights along the column, posed per frame
    for k in range(ARC_SPEC_N):
        Ls = UsdLux.CylinderLight.Define(stage, f"/World/Arc/spec_{k}")
        Ls.AddTranslateOp().Set(Gf.Vec3d(0., 0., -10.))
        Ls.AddOrientOp().Set(Gf.Quatf(1., Gf.Vec3f(0., 0., 0.)))
        Ls.CreateRadiusAttr(1e-3)
        Ls.CreateLengthAttr(1e-3)
        Ls.CreateIntensityAttr(0.)
        Ls.CreateColorAttr(Gf.Vec3f(*map(float, lc)))
        Ls.CreateDiffuseAttr(0.)
        Ls.CreateSpecularAttr(1.)
        Ls.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
        UsdGeom.Imageable(Ls).MakeInvisible()
    wire_m = A.pbr("wire_copper_coat", (196, 130, 90), rough=.25, metal=1.)
    wx = A.xform("/World/Wire", (0, 0, -10.))
    wcyl = UsdGeom.Cylinder.Define(stage, "/World/Wire/rod")
    wcyl.CreateRadiusAttr(TO_WIRE_R)
    wcyl.CreateHeightAttr(1.)
    wcyl.CreateAxisAttr("Z")
    A.bind(wcyl, wire_m)
    # ---- a diagnostic camera that follows the arc along the seam (films): its rig is moved per frame
    A.xform("/World/Follow", (0., 0., 0.))
    _camera(stage, "/World/Follow/camera", FOLLOW["eye"], (0., 0., 0.), FOLLOW["hfov"], "diagnostic")
    alias_f = UsdGeom.Xform.Define(stage, "/Cameras/follow")
    alias_f.GetPrim().SetCustomDataByKey("mounted_camera_path", "/World/Follow/camera")
    # ---- a diagnostic camera a film aims per frame (frame["track"]: eye, target, hfov), e.g. on the contact of a
    # stress probe's ram (no fixed camera sees the rams); until aimed it shows TRACK
    _camera(stage, "/Cameras/track", TRACK["eye"], TRACK["target"], TRACK["hfov"], "diagnostic")
    # ---- cameras: fixed views and the weld camera on the gun
    for name, c in {**LY.CAMERAS, **(extra_cameras or {})}.items():
        _camera(stage, "/Cameras/" + name, c["eye"], c["target"], c["hfov"], c.get("role", "diagnostic"))
    e_tool, t_tool, up_tool = weld_camera_in_tool0(gun)
    wc_eye, wc_tgt, up = pt + Rt @ e_tool, pt + Rt @ t_tool, Rt @ up_tool
    cam = _camera(stage, gb + "/weld_camera", wc_eye, wc_tgt, LY.WELD_CAMERA["hfov"], "policy", up=tuple(up))
    alias = UsdGeom.Xform.Define(stage, "/Cameras/weld_camera")
    alias.GetPrim().SetCustomDataByKey("mounted_camera_path", str(cam.GetPath()))
    stage.GetRootLayer().Save()
    meta = dict(bodies=labels, body_paths=body_paths, band=band_parent + "/surface", pool=band_parent + "/pool",
                pieces=piece_paths, cameras=list(LY.CAMERAS) + list(extra_cameras or {}) + ["weld_camera", "follow",
                                                                                              "track"],
                exposure=EXPOSURE, render_exposure=RENDER_EXPOSURE, rtx_k=RTX_K,
                bracket_band=piece_paths.get("bracket/" + BRACKET_BAND_PIECE))
    (folder/"export.json").write_text(json.dumps(meta, indent=1, default=str))
    return folder/"scene.usdc"


NOZZLE_TINT = .30              # front fraction of the nozzle's length drawn heat-tinted (a used nozzle's oxidised end)
NOZZLE_TINT_SRGB = (112, 72, 54)   # oxidised copper: dull brown [estimate]
NOZZLE_COPPER_SRGB = (205, 120, 80)
NOZZLE_RIM_SRGB = (52, 44, 40)     # the rim and bore: spatter and soot on oxidised copper [estimate]


def nozzle_meshes(p0, p1, r0, r1, sides=48, rings=24):
    """The gas nozzle as a hollow tapered tube from p0 (radius r0) to its open end p1 (radius r1): outer side, the rim
    annulus down to the bore (params.NOZZLE_ID), the bore's inside and the back cap; [(V, F, part, (colour, roughness,
    metalness) per vertex)], each part with its own vertices (normals are smoothed within a part only). The outer side
    is copper, its front NOZZLE_TINT of the length graded (smoothstep) to oxidised copper; rim and bore soiled."""
    from . import params as PR
    rb = PR.v(PR.NOZZLE_ID)/2
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    z = d/L
    x = np.cross(z, [1, 0, 0] if abs(z[0]) < .9 else [0, 1, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    a = np.linspace(0, 2*math.pi, sides, endpoint=False)
    ring = np.cos(a)[:, None]*x + np.sin(a)[:, None]*y

    def band(fs, rs):
        V = np.concatenate([p0 + d*f + r*ring for f, r in zip(fs, rs)])
        F = []
        for k in range(len(fs) - 1):
            for j in range(sides):
                j1 = (j + 1) % sides
                F.append([k*sides + j, k*sides + j1, (k + 1)*sides + j1, (k + 1)*sides + j])
        return V, F

    def annulus(f, ro, ri):
        V = np.concatenate([p0 + d*f + ro*ring, p0 + d*f + ri*ring])
        return V, [[j, sides + j, sides + (j + 1) % sides, (j + 1) % sides] for j in range(sides)]

    def look(col, rough, metal, n):
        return (np.tile(srgb_to_linear(col), (n, 1)).astype(np.float32), np.full(n, rough, np.float32),
                np.full(n, metal, np.float32))
    fs = np.linspace(0., 1., rings + 1)
    V, F = band(fs, r0 + (r1 - r0)*fs)
    w = np.clip((fs - (1. - NOZZLE_TINT))/NOZZLE_TINT, 0., 1.)
    w = np.repeat(w*w*(3 - 2*w), sides)
    cu, ox = srgb_to_linear(NOZZLE_COPPER_SRGB), srgb_to_linear(NOZZLE_TINT_SRGB)
    side_look = ((cu[None]*(1 - w[:, None]) + ox[None]*w[:, None]).astype(np.float32),
                 (.35 + .2*w).astype(np.float32), (1. - .3*w).astype(np.float32))
    out = [(V, F, "side", side_look)]
    V, F = annulus(1., r1, rb)
    out.append((V, F, "rim", look(NOZZLE_RIM_SRGB, .7, .4, len(V))))
    V, F = band(np.linspace(.08, 1., 4), np.full(4, rb))
    out.append((V, F, "bore", look(NOZZLE_RIM_SRGB, .7, .4, len(V))))
    V, F = annulus(0., r0, 1e-4)
    out.append((V, F, "back", look(NOZZLE_COPPER_SRGB, .35, 1., len(V))))
    # faces wound so their normals point out of the copper: checked against the radial / axial direction
    fixed = []
    for V, F, part, lk in out:
        F = np.asarray(F)
        Vc = V[F].mean(1)
        n_ = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        radial = (Vc - p0) - np.outer((Vc - p0) @ z, z)
        ref = {"side": radial, "bore": -radial, "rim": np.tile(z, (len(F), 1)), "back": np.tile(-z, (len(F), 1))}[part]
        flip = np.einsum("ij,ij->i", n_, ref) < 0
        F[flip] = F[flip][:, ::-1]
        fixed.append((V, F.tolist(), part, lk))
    return fixed


def _set_look(mesh, col, rough, metal):
    from pxr import Sdf, UsdGeom, Vt
    pv = UsdGeom.PrimvarsAPI(mesh)
    pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec3fArray.FromNumpy(np.asarray(col, np.float32)))
    pv.CreatePrimvar("roughness", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(
        Vt.FloatArray.FromNumpy(np.asarray(rough, np.float32)))
    pv.CreatePrimvar("metalness", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(
        Vt.FloatArray.FromNumpy(np.asarray(metal, np.float32)))


TO_WIRE_R = .889e-3/2
# relative to the followed point (world axes): trailing low on the plate side, where the nozzle does not hide the arc
# and the fresh bead (the line-of-sight search of the weld camera) [authored]
FOLLOW = dict(eye=(-.118, .043, .034), hfov=34.)
TRACK = dict(eye=(1.2, -.95, 1.7), target=(-.05, .1, 1.0), hfov=55.)      # the tracking camera before a film aims it
                                                                          # (the cell view 'robots') [authored]
HOSE_ATTACH_TOOL = np.array([.0, -.03, .13])     # where the hose package enters the gun body (tool0 frame) [authored]


def _sag(a, b, sag, n=16):
    """Points of a hanging cable from a to b sagging `sag` m below the chord's midpoint (parabola)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.linspace(0., 1., n)[:, None]
    return a*(1 - t) + b*t - np.array([0, 0, 1.])*sag*4*t*(1 - t)


def _curve(stage, path, P, width, mat):
    from pxr import UsdGeom, Vt, UsdShade
    c = UsdGeom.BasisCurves.Define(stage, path)
    c.CreateTypeAttr("cubic")
    c.CreateBasisAttr("catmullRom")
    c.CreateWrapAttr("pinned")
    c.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(P, np.float32)))
    c.CreateCurveVertexCountsAttr([len(P)])
    c.CreateWidthsAttr([float(width)])
    c.SetWidthsInterpolation("constant")
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    return c


def table_texture():
    """Diffuse map of the fixture table top: nitrided steel with 16 mm holes on a 50 mm grid (one 0.5 m tile,
    generated deterministically). Visual only: the physics table top is a flat plate."""
    from PIL import Image, ImageDraw
    out = ROOT/".cache"/"textures"/"table_holes.png"
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    px = 1024
    mm = px/500.
    rng = np.random.default_rng(7)
    base = (np.array([58, 60, 63], float)[None, None, :] + rng.normal(0, 3.5, (px, px, 1))).clip(0, 255)
    im = Image.fromarray(base.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    for i in range(10):
        for j in range(10):
            cx, cy = (25 + 50*i)*mm, (25 + 50*j)*mm
            r1, r0 = 9.5*mm, 8.*mm                        # 1 mm chamfer ring, 16 mm bore
            d.ellipse([cx - r1, cy - r1, cx + r1, cy + r1], fill=(92, 94, 98))
            d.ellipse([cx - r0, cy - r0, cx + r0, cy + r0], fill=(8, 8, 9))
    im.save(out)
    return out


def scene_meta(world):
    """What export_scene needs from the world (static facts)."""
    sc = world.scene
    pt, Rt = sc.tool0_in_body
    arms = {}
    from . import layout as L
    for arm, hw in sc.arm_hw.items():
        if arm not in L.ROBOT_BASES:                 # the G1 stands on the floor (welding/humanoid.py)
            arms[arm] = dict(model=hw["model"], mount_pos=[*hw["mount"][0], hw["base_z"]],
                             mount_yaw_deg=math.degrees(hw["mount"][1]))
            continue
        base = L.ROBOT_BASES[arm]
        arms[arm] = dict(model=hw["model"], mount_pos=list(base["pos"]), mount_yaw_deg=base["yaw_deg"])
    band = band_edges(world.process.g)[1] if world.process is not None else dict(BAND)
    return dict(arms=arms, band=band, gun_body_label=world.model.body_label[sc.gun_body], tool0_pos=pt.tolist(),
                gun_model=sc.arm_hw[getattr(sc, "torch_instance", sc.torch_arm)]["model"],
                bare_arms=[a for a, hw in sc.arm_hw.items() if hw["model"] in ("fr3", "panda") and not hw.get("hand")],
                tool0_R=np.asarray(Rt).tolist(), coupon=world.coupon,
                weld_origin_in={"base_plate": (-sc.weld_origin_in_base).tolist(),
                                "bracket": (-sc.weld_origin_in_bracket).tolist() if hasattr(sc, "weld_origin_in_bracket") else [0, 0, 0]})


# ------------------------------------------------------------------------------------------------ worker client
def _enc(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype).tobytes()).decode()


def _mesh_request(path, b):
    return dict(path=path, n=int(len(b["points"])), m=int(len(b["tris"])),
                points=_enc(b["points"], np.float32), tris=_enc(b["tris"], np.int32),
                normals=_enc(b["normals"], np.float32), color=_enc(b["color"], np.float32),
                glow=_enc(b["glow"], np.float32), weldfrac=_enc(b["weldfrac"], np.float32),
                roughness=_enc(b["roughness"], np.float32), metalness=_enc(b["metalness"], np.float32))


def build_request(frame, meta, cameras, sequence):
    expo = meta["exposure"]
    req = dict(op="observe", sequence=sequence, tick=int(frame["tick"]), cameras=list(cameras),
               exposure={c: expo.get(c, expo["default"]) for c in cameras},
               render_exposure=meta.get("render_exposure", RENDER_EXPOSURE), rtx_k=meta.get("rtx_k", RTX_K),
               body_q={meta["body_paths"][lab]: list(map(float, q)) for lab, q in zip(frame["bodies"], frame["body_q"])})
    b = frame.get("band")
    if b is not None:
        req["band"] = _mesh_request(meta["band"], b)
    pool = frame.get("pool")
    req["pool"] = None if pool is None or "pool" not in meta else _mesh_request(meta["pool"], pool)
    looks = frame.get("coupon_look")
    if looks:
        req["looks"] = [dict(path=meta["pieces"][k], n=int(len(c)), color=_enc(c, np.float32), roughness=_enc(r, np.float32),
                             metalness=_enc(m, np.float32)) for k, (c, r, m) in looks.items() if k in meta.get("pieces", {})]
    arc = frame.get("arc")
    col = None if arc is None else arc_column(arc.get("wire_end", arc["root"]), arc["root"], arc["current_a"])
    # the column's emission model (arc_column) for the worker's composites of the seen arc and its mirror image in the
    # pool (the OptiX denoiser clamps such pixels), and the one light that carries its luminous intensity
    req["arc"] = None if col is None else dict(col, current_a=arc["current_a"], light=arc_light(col))
    fume = frame.get("fume")
    if fume is not None:
        wire = frame.get("wire") or {}
        fume = dict(fume)
        if col is not None:
            fume["source"] = dict(pos=col["centroid"], intensity=col["intensity_cd"])
        if wire.get("tip") is not None:
            ax = np.asarray(wire["axis"], float)
            tip = np.asarray(wire["tip"], float) + ax*.0032          # the nozzle's end, 3.2 mm beyond the contact tip
            fume["gun"] = dict(p0=tip.tolist(), p1=(tip - ax*GUN_SHADOW_L).tolist(), radius=GUN_SHADOW_R,
                               source_r=None if col is None else float(col["s_root"]))
    req["fume"] = fume
    if meta.get("bracket_band"):
        req["bracket_band"] = dict(path=meta["bracket_band"], visible=not frame.get("bracket_in_grid", True))
    wire = frame.get("wire")
    req["wire"] = wire
    req["curves"] = frame.get("curves")
    req["follow"] = frame.get("follow")
    req["track"] = frame.get("track")
    return req


class Renderer:
    """Owns one render-only Isaac worker (welding.render_worker)."""

    def __init__(self, folder, scene, *, width=1280, height=720, samples=16):
        from gym_anything_robotics.render_lifecycle import close_worker, watch_worker
        from whiteboard.ipc import receive
        self.folder = Path(folder)
        self.closed, self.lock, self.sequence = False, threading.Lock(), 0
        self.scene = Path(scene)
        self.meta = json.loads((self.scene.parent/"export.json").read_text())
        executable = os.environ.get("ISAAC_RENDER_PYTHON")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("ISAAC_RENDER_PYTHON is not configured: set it to the Isaac Sim python interpreter")
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(300)
        secret = secrets.token_urlsafe(32)
        env = dict(os.environ, WIPE_RENDER_PORT=str(listener.getsockname()[1]), WIPE_RENDER_SECRET=secret,
                   WIPE_MINIMAL_RENDER="1")
        self.log = (self.folder/"worker.log").open("wb")
        self.process = None
        self.timings = []
        try:
            self.process = subprocess.Popen([executable, "-m", "welding.render_worker", "--scene", str(self.scene.resolve()),
                                             "--width", str(width), "--height", str(height), "--samples", str(samples)],
                                            cwd=ROOT.parent, env=worker_env(env), stdout=self.log, stderr=subprocess.STDOUT)
            self.connection, _ = listener.accept()
            self.connection.settimeout(600)
            if not secrets.compare_digest(receive(self.connection).get("secret", ""), secret):
                raise PermissionError("Renderer authentication failed")
            if receive(self.connection).get("phase") != "ready":
                raise RuntimeError("Renderer startup failed")
        except BaseException:
            close_worker(self, graceful=False)
            raise
        finally:
            listener.close()
        watch_worker(self)

    @property
    def pid(self):
        return self.process.pid if self.process else None

    def observe(self, frame, cameras=("bench",)):
        from whiteboard.ipc import send, receive
        with self.lock:
            if self.closed:
                raise RuntimeError("Renderer is closed")
            self.sequence += 1
            t0 = time.perf_counter()
            req = build_request(frame, self.meta, cameras, self.sequence)
            send(self.connection, req)
            deadline = time.monotonic() + 600
            while not select.select([self.connection], [], [], 1.)[0]:
                if self.process.poll() is not None:
                    raise RuntimeError("Renderer exited before replying; see " + str(self.folder/"worker.log"))
                if time.monotonic() > deadline:
                    raise TimeoutError("Renderer timed out")
            result = receive(self.connection)
            if not result.get("ok") or result.get("physics_steps") != 0:
                raise RuntimeError("Invalid render response: " + str(result)[:600])
            self.timings.append(dict(tick=int(frame["tick"]), round_trip_s=time.perf_counter() - t0,
                                     worker=result.get("timings", {})))
            return {k: base64.b64decode(v) for k, v in result["images"].items()}

    def close(self):
        from gym_anything_robotics.render_lifecycle import close_worker
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True
