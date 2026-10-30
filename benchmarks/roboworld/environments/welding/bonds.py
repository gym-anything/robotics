"""Metallurgical bond between bracket and base plate, from the thermal solver's cells.

Per x slice of the joint, the parts are bonded where solid weld metal (or parent metal melted and refrozen) connects
the bracket's parent metal to the base plate's: the slice's load path is the weld's effective throat, the shortest
distance through solid joint material from the root to the face (the minimum cut of a fillet section). Its strength is
the fillet weld's ultimate shear strength on its throat, 0.852 F_EXX (AWS D1.1:2000 Commentary C2.14.4, from tests;
F_EXX the filler's tensile strength; the codes' 0.6 F_EXX of AISC 360 J2.4 is a design value with margin), reduced by
the steel's strength factor k_y(T) at the section's hottest solid joint metal (EN 1993-1-2 Table 3.1). A section with
liquid metal at its root carries nothing.

    capacity = sum over bonded slices of  0.852 F_EXX * throat_i * dx_i * k_y(T_i)

The mechanical bond is one MuJoCo weld equality between the two bodies (scene.bond), switched on at the parts' pose
when the capacity first becomes positive (no bond exists without fusion) and removed when its constraint load exceeds
what the weld group carries (the joint breaks; the fracture follows the throats).

Load check, per direction (a check on norms, |F|/C + |M|/(C r), would break the weld by pushing the bracket into it at
the load that pulls it off). The bond's load on the weld, in the weld frame at the anchor on the root
line (x along the weld, y out of the bracket face toward the weld side, z up), is set against the weld group, each
bonded section i carrying a_i = throat_i dx_i k_y(T_i) of throat area at the strength tau = 0.852 F_EXX:
  F_x (along the weld)                longitudinal shear                     C   = tau sum a_i
  F_y (across the weld)               transverse shear                       f C (f = TRANSVERSE_FACTOR, below: 1, no
                                                                             directional increase on a single-sided
                                                                             fillet)
  F_z > 0 (pulling the bracket off)   transverse tension                     f C; F_z < 0 bears on the plate
  M_x > 0 (top away from the weld)    peel: the bracket pivots on its far    f C (t_b + throat / sqrt 2): the weld's
                                      bottom edge (y = -t_b), the weld pulls tension on the lever from that edge to the
                                                                             throat's middle; M_x < 0 turns the bracket
                                                                             into the weld, which bears it in
                                                                             compression (not a weld failure)
  M_y (lifting one end)               the bracket pivots on the far end's    f tau sum a_i d_i, d_i from that end
                                      bottom corner; the sections pull
  M_z (twisting in plan)              transverse shear about the group's     f tau sum a_i r_i, r_i the
                                      centroid                               section's mean distance from x_c
                                                                             (|x_i - x_c|; dx/4 + d^2/dx when the
                                                                             section spans x_c)
(plastic weld-group capacities, every section at its strength), combined linearly: the bond breaks when
|F_x|/C + |F_y|/fC + max(F_z, 0)/fC + max(M_x, 0)/M_peel + |M_y|/M_y,cap + |M_z|/M_z,cap > 1.
The rule is a hand calculation of the weld group, not fitted to anything; E7 compares the break with it in closed form.
"""

import math

import numpy as np

from . import params as PR
from . import thermal as TH
from .sections import FUSED

TRANSVERSE_FACTOR = PR.v(PR.WELD_TRANSVERSE_FACTOR)   # no directional increase on this single-sided weld (ledger)


def k_y(t):
    tab = np.asarray(PR.v(PR.K_Y_STEEL), float)
    return float(np.interp(t, tab[:, 0], tab[:, 1]))


def slice_bonds(proc, i0=None, i1=None, temps=None):
    """Per-slice bond sections over [i0, i1): list of (i, throat m, k_y, solid?). `temps`: a dict filled with each
    section's hottest joint temperature (C)."""
    g, s, pool = proc.g, proc.s, proc.pool
    if i0 is None:
        live = [i for i, st in enumerate(pool.slices) if st.h is not None or st.frozen]
        if not live:
            return []
        i0, i1 = min(live), max(live) + 1
    j0, j1, k0, k1 = pool.j0, pool.j1, pool.k0, pool.k1
    mat = s.mat.reshape(g.nx, g.ny, g.nz)[i0:i1, j0:j1, k0:k1]
    orig = proc.original_metal.reshape(g.nx, g.ny, g.nz)[i0:i1, j0:j1, k0:k1]
    pf = s.slice_block("peakf", i0, i1)[:, j0:j1, k0:k1]
    fl = s.slice_block("fl", i0, i1)[:, j0:j1, k0:k1]
    T = s.slice_block("T", i0, i1)[:, j0:j1, k0:k1]
    yc, zc = g.yc[j0:j1], g.zc[k0:k1]
    dy, dz = g.dy[j0:j1], g.dz[k0:k1]
    jb, kb = int(np.searchsorted(yc, 0.)), int(np.searchsorted(zc, 0.))
    Y, Z = np.meshgrid(yc, zc, indexing="ij")
    half = .5*np.minimum(dy[:, None], dz[None, :])
    out = []
    for n in range(i1 - i0):
        m = mat[n]
        melted = orig[n] & (((pf[n] >= FUSED) & (m != TH.AIR)) | (m == TH.WELD))     # parent metal fused (or refilled)
        joint = (m == TH.WELD) | melted
        # bonded only if weld metal fills the root and has fused into BOTH parts (weld metal lying on an unfused part
        # is a cold lap and carries nothing)
        fused_bracket = bool((melted & (Y < 0) & (Z >= 0)).any())
        fused_base = bool((melted & (Z < 0)).any())
        if not (joint[jb, kb] and fused_bracket and fused_base):
            continue
        if fl[n][jb - 2:jb + 2, kb - 2:kb + 2].max() >= FUSED:
            out.append((i0 + n, 0., 0., False))               # liquid at the root: no strength yet
            continue
        weld = m == TH.WELD
        face = (m == TH.AIR) & (Y > 0) & (Z > 0)
        nb = np.zeros_like(weld)
        nb[1:, :] |= weld[:-1, :]
        nb[:-1, :] |= weld[1:, :]
        nb[:, 1:] |= weld[:, :-1]
        nb[:, :-1] |= weld[:, 1:]
        face &= nb
        if not face.any():
            continue
        throat = float((np.sqrt(Y**2 + Z**2) - half)[face].min())
        tj = float(T[n][joint].max()) if joint.any() else 20.
        if temps is not None:
            temps[i0 + n] = tj
        out.append((i0 + n, max(throat, 0.), k_y(tj), True))
    return out


FIT_MM, FIT_DEG = 1.0, 2.0     # the cells describe the joint at fit-up: a bracket farther than this from its fit-up pose
                               # is not where the grid's weld metal is, and no bond forms to it (4 cells) [authored]


class Bond:
    """State of the bracket-to-plate bond and its MuJoCo equality.

    Sections that were bonded when the bond broke are cracked: they carry nothing until they melt again (a crack heals
    only by re-fusion). New fusion elsewhere bonds the parts again, at their current pose, while the bracket is still
    where the thermal grid has it (FIT_MM, FIT_DEG). (A bond made at a 0.7 N first sliver of fusion would break under
    the bracket's own weight on the next tick and, removed for good, leave a whole tack fixture unbonded.)"""

    def __init__(self, world):
        self.w = world
        self.active = False
        self.broken = False
        self.cracked = set()
        self.capacity_n = 0.
        self.length_m = 0.
        self.x_range = None
        self.load = dict(force_n=0., moment_nm=0., ratio=0.)
        self.group = []
        self.events = []
        self.shrink_target_rad = 0.
        self.tau = PR.v(PR.WELD_SHEAR_RATIO)*PR.v(PR.WELD_UTS)

    def update(self):
        """Recompute capacity from the cells (called every BOND_EVERY ticks); switch the equality on at first fusion."""
        proc = self.w.process
        temps = {}
        rows = slice_bonds(proc, temps=temps)
        g = proc.g
        for i, throat, ky, solid in rows:
            if not solid and i in self.cracked:
                self.cracked.discard(i)             # liquid at the root again: the crack is re-fused
        cap, xs, length = 0., [], 0.
        self.sections = []
        self.group = []                             # (x m, throat area x k_y m2, throat m) per bonded section
        for i, throat, ky, solid in rows:
            if solid and throat > 0 and i not in self.cracked:
                cap += self.tau*throat*g.dx[i]*ky
                xs.append(g.xc[i])
                length += g.dx[i]
                self.sections.append(i)
                self.group.append((float(g.xc[i]), float(throat*g.dx[i]*ky), float(throat)))
        self.capacity_n = cap
        self.length_m = float(length)
        # shrinkage distortion: the bracket turns toward the weld as the bonded weld metal cools (SHRINK_ANGLE_REF)
        ang, a_ref, l_ref = PR.v(PR.SHRINK_ANGLE_REF)
        t_s = PR.v(PR.STEEL_SOLIDUS)
        vol = 0.
        for i, throat, ky, solid in rows:
            if solid and throat > 0 and i not in self.cracked:
                cooled = float(np.clip((t_s - temps.get(i, 20.))/(t_s - 20.), 0., 1.))
                vol += throat*throat*g.dx[i]*cooled
        self.shrink_target_rad = math.radians(ang)*vol/(a_ref*l_ref)
        self.x_range = (min(xs), max(xs)) if xs else None
        if cap > 0 and not self.active and self._at_fit_up():
            self.w.set_bond(True)
            self.active = True
            self.events.append(dict(tick=self.w.tick, event="rebonded" if self.broken else "bonded", capacity_n=cap))
        elif self.active:
            self.w.update_bond_anchor()             # the bonded length grows as the weld runs
        return cap

    def _at_fit_up(self, scale=1.):
        """The bracket within `scale` x the fit-up tolerance (FIT_MM, FIT_DEG) of the pose the thermal grid has it."""
        w = self.w
        if "bracket" not in w.scene.bodies:
            return False
        pb, Rb = w.body_pose(w.scene.bodies["base_plate"])
        pk, Rk = w.body_pose(w.scene.bodies["bracket"])
        ang = math.degrees(math.acos(np.clip((Rb.T @ Rk)[2, 2], -1, 1)))
        off = float(np.linalg.norm(Rb.T @ (pk - pb) - w.scene.bracket_rel_nominal))
        return ang <= FIT_DEG*scale and off <= FIT_MM*1e-3*scale

    def capacities(self, x_anchor):
        """The weld group's capacities (module docstring) for an anchor at x_anchor on the root line (weld frame)."""
        grp = self.group or []
        tau = self.tau
        A = sum(a for _, a, _ in grp)
        C = tau*A
        L = self.w.coupon.length
        t_b = self.w.coupon.web[1] if self.w.coupon.web is not None else 0.
        throat = float(np.median([t for _, _, t in grp])) if grp else 0.
        x_c = sum(x*a for x, a, _ in grp)/A if A > 0 else x_anchor
        g = self.w.process.g if self.w.process is not None else None

        def arm(x):
            # mean distance from x_c over the section's own length dx (uniform shear flow along it): |x - x_c| when the
            # section lies to one side of x_c, dx/4 + d^2/dx when it spans it. With the bare |x - x_c| a single bonded
            # section has no twist capacity at all, and the first fused section of a tack breaks under 5 nN m
            d = abs(x - x_c)
            if g is None:
                return d
            dx = float(g.dx[min(max(int(np.searchsorted(g.xe, x)) - 1, 0), g.nx - 1)])
            return d if d >= dx/2 else dx/4 + d*d/dx
        f = TRANSVERSE_FACTOR
        return dict(C=C, C_T=f*C, M_peel=f*C*(t_b + throat/math.sqrt(2.)),
                    # M_y > 0 lifts the low-x end: the bracket pivots on its x = L end, and the reverse
                    M_y_pos=f*tau*sum(a*(L - x) for x, a, _ in grp), M_y_neg=f*tau*sum(a*x for x, a, _ in grp),
                    M_z=f*tau*sum(a*arm(x) for x, a, _ in grp), lever_peel_m=t_b + throat/math.sqrt(2.))

    def check(self):
        """Every tick while active: break the bond if its load exceeds what the weld group carries (module docstring).
        bond_wrench reports the load the bond carries from the bracket (force in the world frame, moment about the
        anchor in the base plate's frame)."""
        if not self.active:
            return
        F, M = self.w.bond_wrench()
        o, R = self.w.weld_frame()
        Fw = R.T @ np.asarray(F, float)
        Mw = np.asarray(M, float)
        xr = self.x_range
        c = self.capacities(.5*(xr[0] + xr[1]) if xr else self.w.coupon.length/2)
        if c["C"] <= 0:
            ratio = float("inf")
        else:
            ratio = float(abs(Fw[0])/c["C"] + abs(Fw[1])/c["C_T"] + max(Fw[2], 0.)/c["C_T"]
                          + max(Mw[0], 0.)/max(c["M_peel"], 1e-12)
                          + abs(Mw[1])/max(c["M_y_pos"] if Mw[1] > 0 else c["M_y_neg"], 1e-12)
                          + abs(Mw[2])/max(c["M_z"], 1e-12))
        self.load = dict(force_n=float(np.linalg.norm(F)), moment_nm=float(np.linalg.norm(M)), ratio=ratio,
                         force_weld_n=Fw.round(3).tolist(), moment_weld_nm=Mw.round(4).tolist())
        if ratio > 1.:
            self.w.set_bond(False)
            self.active = False
            self.broken = True
            self.cracked |= set(getattr(self, "sections", []))
            self.events.append(dict(tick=self.w.tick, event="broken", capacity_n=self.capacity_n,
                                    length_m=self.length_m, cracked_sections=len(self.cracked), **self.load))
