"""Preset C: one untrimmed pork tenderloin (M. psoas major with the "chain", M. psoas minor, attached along one
side on a fat-and-fascia seam; silver skin over part of the upper surface; loose flake fat at the head end).

Workpiece frame: lying flat on z = 0, head (thick) end at -x, tail at +x, chain along the +y side, bounding box
centred on x = y = 0. Local loft frame: s = x (station), v = z (up), w = y - y_c(s) (lateral from the curved
centre line).
"""

import numpy as np

from filleting import params as PR
from filleting.anatomy.common import (Workpiece, Loft, Profile, smin, smax, smoothstep, prov, grid_integrate,
                                      value_noise, save_png, srgb, marching_tets)

L = PR.TENDERLOIN.value["length"]
FAT_DENSITY = prov(920., "kg/m3", "estimate", "adipose tissue ~0.9-0.95 g/cm3 (general knowledge)")

# width (m) along t = (x - head)/L; head 6.5 cm (design), tail ~2 cm (design), taper shape estimated from the
# retail descriptions ("one end tapered", cooksinfo) - no published profile found
WIDTH = [(0., .006), (.008, .030), (.02, .046), (.06, .058), (.12, .065), (.22, .063), (.40, .057), (.55, .050),
         (.70, .041), (.82, .031), (.92, .023), (.97, .018), (.99, .011), (1., .004)]
HEIGHT_RATIO = [(0., .62), (.30, .66), (.70, .72), (1., .80)]

PROV = {
    "length": prov(L, "m", PR.TENDERLOIN.label, PR.TENDERLOIN.basis),
    "head_width": prov(PR.TENDERLOIN.value["diameter"], "m", PR.TENDERLOIN.label, PR.TENDERLOIN.basis),
    "width_profile": prov(WIDTH, "m", "estimate", "design head 6.5 cm, tail ~2 cm; cooksinfo 'about 30 cm long, "
                          "5-7.5 cm wide ... one end tapered'"),
    "height_ratio": prov(HEIGHT_RATIO, "-", "estimate", "flattened lying on the board ('4 distinct sides', "
                         "cooksinfo); no measured section found"),
    "section": prov(dict(egg=-.12, n_top=2.3, n_bottom=3.0), "-", "estimate", "flat underside on the board"),
    "curvature": prov(.008, "m", "estimate", "gentle bow of the centre line in plan view"),
    "fibre": prov("along the centre line", "-", "read", "pig psoas major is a parallel, fusiform muscle (Park et al. "
                  "2022, PMC9478981)"),
    "chain": prov(dict(width=.030, height=.016, t_range=(.08, .80), overlap=.35), "m", "estimate",
                  "psoas minor ('side chain', Nebraska porcine myology) along one side; no measured size found, "
                  "the section is sized so the chain falls in the design's 50-80 g (a 24 x 13 mm first guess gave "
                  "37 g)"),
    "silver_coverage": prov(.60, "fraction of the main muscle's upper surface", "design(estimate)", ""),
    "silver_band": prov(dict(t_range=(.10, .88)), "fraction of length", "estimate",
                        "silver skin seen as a longitudinal band along the top of the thick part in untrimmed-"
                        "tenderloin photos (thestickymenuplan.com trimming guide, viewed); band width calibrated to "
                        "the 60 % coverage"),
    "silver_thickness": prov(PR.SILVER_THICKNESS.value, "m", PR.SILVER_THICKNESS.label, PR.SILVER_THICKNESS.basis),
    "silver_reference_layer": prov(1e-3, "m", "authored", "tissue under the silver skin an ideal denuding cut takes"),
    "flake_fat": prov(dict(n=5, semi=((.012, .020), (.008, .014), (.004, .006))), "m", "estimate",
                      "loose flake fat lumps at the head end; thicker than the 3 mm IMPS limit "
                      "so denuding is required"),
    "fat_density": FAT_DENSITY,
    "medallion": prov(dict(thickness=.025, min_width=.040, head_trim=.035), "m", "read+authored",
                      "25 mm +- 5 (IMPS 400 section 6.3); medallions from the thick centre cut: start 35 mm behind "
                      "the head end, stop where the width falls below 40 mm (authored)"),
    "variation": prov(dict(length=.05, fat="lump positions/sizes"), "-", "authored", ""),
}

REGION_NAMES = ["main", "chain", "fat"]
REFERENCE_NAMES = ["main", "chain", "silver", "fat"]


class Tenderloin:
    def __init__(self, seed=0):
        self.seed = seed
        rng = np.random.default_rng(8000 + seed)
        fL = 1.0 if seed == 0 else 1 + rng.uniform(-.05, .05)
        self.fL = fL
        self.L = L*fL
        self.x0 = -.5*self.L
        t = np.array([p[0] for p in WIDTH])
        W = np.array([p[1] for p in WIDTH])*fL
        H = W*np.interp(t, *np.array(HEIGHT_RATIO).T)
        s = self.x0 + t*self.L
        sec = PROV["section"]["value"]
        self.main = Loft(s, H, np.zeros_like(H), .5*W, np.full_like(s, sec["egg"]),
                         (np.full_like(s, sec["n_top"]), np.full_like(s, sec["n_bottom"])), (s[0], s[-1]), 1e-3)
        self.Wp = Profile(s, W)
        self.Hp = Profile(s, H)
        # chain loft along the +y flank
        ch = PROV["chain"]["value"]
        tc = np.linspace(ch["t_range"][0], ch["t_range"][1], 20)
        env = np.sin(np.pi*(tc - tc[0])/(tc[-1] - tc[0]))**.5
        env[0] = env[-1] = .05
        sc = self.x0 + tc*self.L
        Wc, Hc = ch["width"]*fL*env, ch["height"]*fL*env
        self.chain = Loft(sc, Hc, np.zeros_like(Hc), .5*Wc, np.full_like(sc, -.05), np.full_like(sc, 2.2),
                          (sc[0], sc[-1]), 1e-3)
        self.Wc = Profile(sc, Wc)
        self.ch_overlap = ch["overlap"]
        # flake fat lumps at the head end (upper surface)
        ff = PROV["flake_fat"]["value"]
        self.lumps = []
        for q in range(ff["n"]):
            tq = rng.uniform(.015, .16)
            sq = self.x0 + tq*self.L
            wq = rng.uniform(-.35, .35)*float(self.Wp(sq))
            a = np.array([rng.uniform(*ff["semi"][0]), rng.uniform(*ff["semi"][1]), rng.uniform(*ff["semi"][2])])*fL
            zt = float(self.surface_top(np.array([sq]), np.array([wq]))[0])
            self.lumps.append((np.array([sq, self.y_c(sq) + wq, zt + .2*a[2]]), a))
        self.variation = dict(seed=seed, length_scale=fL, fat_lumps=[(c.tolist(), a.tolist()) for c, a in self.lumps],
                              pose=dict(dx=float(rng.uniform(-.015, .015)) if seed else 0.,
                                        dy=float(rng.uniform(-.015, .015)) if seed else 0.,
                                        yaw_deg=float(rng.uniform(-8, 8)) if seed else 0.))
        self.silver_t, self.silver_frac = None, 1.0

    def y_c(self, x):
        t = (x - self.x0)/self.L
        return -PROV["curvature"]["value"]*self.fL*np.sin(np.pi*np.clip(t, 0, 1))

    def surface_top(self, s, w):
        """Height of the main muscle's upper surface above (s, lateral w) - by bisection on v."""
        lo, hi = np.zeros_like(s), np.full_like(s, .1)
        for _ in range(40):
            m = .5*(lo + hi)
            inside = self.main.field(s, m, w) < 0
            lo, hi = np.where(inside, m, lo), np.where(inside, hi, m)
        return .5*(lo + hi)

    def d_main(self, x, y, z):
        return self.main.sdf(x, z, y - self.y_c(x))

    def chain_y(self, x):
        return self.y_c(x) + .5*self.Wp(x) + (.5 - self.ch_overlap)*self.Wc(x)

    def d_chain(self, x, y, z):
        return self.chain.sdf(x, z, y - self.chain_y(x))

    def d_fat(self, x, y, z):
        d = np.full(len(x), 1.)
        for c, a in self.lumps:
            q = np.sqrt(((x - c[0])/a[0])**2 + ((y - c[1])/a[1])**2 + ((z - c[2])/a[2])**2)
            d = np.minimum(d, (q - 1)*a.min())
        return d

    def tissue(self, x, y, z):
        d = smin(self.d_main(x, y, z), self.d_chain(x, y, z), 1.5e-3)
        d = smin(d, self.d_fat(x, y, z), 1.0e-3)
        return smax(d, -z, .5e-3)

    def region(self, x, y, z):
        dm, dc, df = self.d_main(x, y, z), self.d_chain(x, y, z), self.d_fat(x, y, z)
        lab = np.zeros(len(x), int)
        lab[dc < dm] = 1
        lab[(df < 0) & (df < dm)] = 2
        return lab

    def upper_normal_z(self, x, y, z, eps=2e-5):
        f = self.tissue
        g = np.stack([(f(x + eps, y, z) - f(x - eps, y, z)), (f(x, y + eps, z) - f(x, y - eps, z)),
                      (f(x, y, z + eps) - f(x, y, z - eps))], 1)
        return g[:, 2]/np.maximum(np.linalg.norm(g, axis=1), 1e-12)

    def silver_patch(self, x, y, z):
        """Silver skin: a longitudinal band along the top of the main muscle (as in the untrimmed-tenderloin
        photos) over stations silver_t, band half-width = silver_frac x the local half-width, with silver_frac
        calibrated so the band covers the stated 60 % of the main muscle's upper surface; not on chain or fat."""
        t = (x - self.x0)/self.L
        ta, tb = self.silver_t
        on = (t > ta) & (t < tb)
        on &= np.abs(y - self.y_c(x)) < self.silver_frac*.5*self.Wp(x)
        on &= self.region(x, y, z) == 0
        return on

    def reference(self, x, y, z):
        lab = self.region(x, y, z)
        ref = np.where(lab == 1, 1, 0)
        ref[lab == 2] = 3
        # silver layer: main tissue within 1 mm below the silver-skin patch on the upper surface
        dm = self.d_main(x, y, z)
        near = (dm > -PROV["silver_reference_layer"]["value"]) & (lab == 0)
        if near.any():
            idx = np.nonzero(near)[0]
            up = self.upper_normal_z(x[idx], y[idx], z[idx]) > .3
            sp = self.silver_patch(x[idx], y[idx], z[idx])
            ref[idx[up & sp]] = 2
        return ref


def _calibrate_silver(tl):
    """Choose the station range [ta, tb] (centred on the thick part) whose silver patch covers the stated fraction
    of the main muscle's upper surface area (surface from marching tetrahedra at 1 mm)."""
    lo = np.array([tl.x0 - 3e-3, -.06, -1e-3])
    hi = np.array([tl.x0 + tl.L + 3e-3, .06, .06])
    V, F = marching_tets(lambda P: tl.tissue(P[:, 0], P[:, 1], P[:, 2]), lo, hi, 1e-3)
    c = V[F].mean(1)
    area = .5*np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    up = tl.upper_normal_z(c[:, 0], c[:, 1], c[:, 2]) > .3
    main = tl.region(c[:, 0], c[:, 1], c[:, 2]) == 0
    base = up & main
    A = area[base].sum()
    t = (c[:, 0] - tl.x0)/tl.L
    target = PROV["silver_coverage"]["value"]
    ta, tb = PROV["silver_band"]["value"]["t_range"]
    rel = np.abs(c[:, 1] - tl.y_c(c[:, 0]))/np.maximum(.5*tl.Wp(c[:, 0]), 1e-6)
    best = None
    for frac in np.linspace(.2, 1.2, 201):
        cov = area[base & (t > ta) & (t < tb) & (rel < frac)].sum()/A
        if best is None or abs(cov - target) < abs(best[3] - target):
            best = (ta, tb, frac, cov)
    return best, A


def _pork_texture(tl, res=(512, 2048)):
    """Surface texture on the cylindrical unwrap (u = t along, v = angle/2pi, 0 = top): raw pork red-pink with
    fibre striations along the length; silver skin as a pearly fibrous sheet on the silver patch."""
    H, W = res
    rng = np.random.default_rng(9000 + tl.seed)
    u = (np.arange(W) + .5)/W
    v = (np.arange(H) + .5)/H
    V, U = np.meshgrid(v, u, indexing="ij")
    meat = srgb("#a8463f")          # raw pork tenderloin, deeper red (untrimmed-tenderloin photos)
    col = np.tile(meat, (H, W, 1))
    n1 = value_noise((H, W), 30, rng)
    stri = value_noise((H, W // 16), 6, rng, octaves=2)
    stri = np.repeat(stri, 16, axis=1)[:, :W]
    col *= (.88 + .12*n1[..., None])*(.94 + .08*stri[..., None])
    ta, tb = tl.silver_t
    top = (np.minimum(V, 1 - V) < .22)
    patch = top & (U > ta) & (U < tb)
    sil = srgb("#dfe2e6")
    fib = value_noise((H, W), 4, rng, octaves=2)
    a = (.55 + .25*fib)[..., None]
    col = np.where(patch[..., None], col*(1 - a) + sil*a, col)
    return np.clip(col, 0, 1)


def _pork_flesh(tl, x, y, z):
    meat, chain_c, fat, fascia = srgb("#b4524a"), srgb("#a44a44"), srgb("#f2eadb"), srgb("#eee6dc")
    lab = tl.region(x, y, z)
    col = np.tile(meat, (len(x), 1))
    col[lab == 1] = chain_c
    col[lab == 2] = fat
    # the seam between chain and main: thin whitish fascia/fat line
    seam = np.abs(tl.d_chain(x, y, z) - tl.d_main(x, y, z)) < .6e-3
    col[seam & (lab < 2)] = fascia
    # sparse intramuscular fat flecks (fixed pseudo-random field in rest space)
    h = np.sin(x*9137.1 + y*7331.7 + z*5113.3)*43758.5453
    fl = (h - np.floor(h)) > .992
    col[fl & (lab < 2)] = .5*col[fl & (lab < 2)] + .5*fat
    return np.clip(col, 0, 1)


def build(seed=0, texture_dir=None, texture_res=(512, 2048)):
    tl = Tenderloin(seed)
    tl.silver_t, tl.silver_frac = PROV["silver_band"]["value"]["t_range"], 1.0
    (ta, tb, frac, cov), A_up = _calibrate_silver(tl)
    tl.silver_t, tl.silver_frac = (ta, tb), frac

    def W(fn):
        return lambda p: fn(*np.atleast_2d(p).T)

    xs = np.linspace(tl.x0, tl.x0 + tl.L, 400)
    y_lo = float(np.min(tl.y_c(xs) - .5*tl.Wp(xs))) - 2e-3
    y_hi = float(np.max(tl.chain_y(xs) + .5*tl.Wc(xs)))
    y_hi = max(y_hi, float(np.max(tl.y_c(xs) + .5*tl.Wp(xs)))) + 2e-3
    lo = np.array([tl.x0 - 2e-3, y_lo, -1e-3])
    hi = np.array([tl.x0 + tl.L + 2e-3, y_hi, float(tl.Hp.v.max()) + .012])

    def fibre(p):
        x = np.atleast_2d(p)[:, 0]
        eps = 1e-4
        dy = (tl.y_c(x + eps) - tl.y_c(x - eps))/(2*eps)
        g = np.column_stack([np.ones_like(x), dy, np.zeros_like(x)])
        return g/np.linalg.norm(g, axis=1)[:, None]

    tol = .75e-3

    def skin_mask(p, tol=tol):
        x, y, z = np.atleast_2d(p).T
        on = np.abs(tl.tissue(x, y, z)) < tol
        up = tl.upper_normal_z(x, y, z) > .3
        return on & up & tl.silver_patch(x, y, z)

    interfaces = {
        "chain_seam": dict(sdf=lambda p: (lambda x, y, z: .5*(tl.d_chain(x, y, z) - tl.d_main(x, y, z)))(*np.atleast_2d(p).T),
                           extent=lambda p: (lambda x, y, z: (tl.tissue(x, y, z) < 0) & (tl.d_chain(x, y, z) < 6e-3)
                                             & (tl.d_main(x, y, z) < 6e-3))(*np.atleast_2d(p).T),
                           params=("CHAIN_SEAM_G", "CHAIN_SEAM_SIGMA"), note="bisector of the chain and main-muscle bodies inside the "
                                                          "union (fat and fascia seam)"),
        "silver_skin": dict(sdf=W(tl.tissue), extent=skin_mask, params=("SILVER_BOND_G", "SILVER_BOND_SIGMA"),
                            note="silver skin to muscle bond on the silver patch"),
        "flake_fat": dict(sdf=W(lambda x, y, z: .5*(tl.d_fat(x, y, z) - tl.d_main(x, y, z))),
                          extent=lambda p: (lambda x, y, z: (tl.tissue(x, y, z) < 0) & (tl.d_fat(x, y, z) < 3e-3))(*np.atleast_2d(p).T),
                          params=("CHAIN_SEAM_G", "CHAIN_SEAM_SIGMA"), note="loose flake fat attached by thin fascia; no data, the seam "
                                                         "value is a placeholder to be swept"),
    }

    def uv(p):
        x, y, z = np.atleast_2d(p).T
        th = tl.main.theta_of(x, z, y - tl.y_c(x))
        return np.column_stack([(x - tl.x0)/tl.L, th/(2*np.pi)])

    tex_path = None
    if texture_dir is not None:
        import os
        os.makedirs(texture_dir, exist_ok=True)
        tex_path = os.path.join(texture_dir, f"pork_surface_seed{seed}.png")
        save_png(tex_path, _pork_texture(tl, texture_res))
    appearance = dict(skin_uv=uv, skin_texture=dict(path=tex_path, make=lambda res=texture_res: _pork_texture(tl, res),
                                                    provenance="generated (authored procedural)"),
                      flesh_color=W(lambda x, y, z: _pork_flesh(tl, x, y, z)),
                      flesh_provenance="authored procedural: red-pink lean, whitish fat and seam fascia")
    targets = dict(mass=dict(value=PR.TENDERLOIN.value["kg"], label=PR.TENDERLOIN.label, basis=PR.TENDERLOIN.basis),
                   chain_mass=dict(value=(.050, .080), label="estimate", basis=""),
                   silver_coverage=dict(value=.60, label="estimate", basis=""),
                   literature=dict(psoas_major_dissected_g=(404, 455), pigs_live_kg=(94, 101),
                                   source="PMC4994323",
                                   imps415_kg=(.88, .89), note="IMPS #415 incl. psoas minor, 107 kg HCW (Bohrer 2023)"))
    wp = Workpiece(
        name="pork_tenderloin", sdf=W(tl.tissue), bounds=(lo, hi), region=W(tl.region), region_names=REGION_NAMES,
        sheet_normal=fibre, fibre=fibre, rigid_bodies=[], rods=[], interfaces=interfaces,
        skin=dict(skin_mask=skin_mask, thickness=PR.SILVER_THICKNESS.value, params="SILVER_THICKNESS",
                  modulus_param="SILVER_E", strength_param="SILVER_STRENGTH", toughness_param="SILVER_R",
                  tolerance=tol, coverage=cov, upper_area=A_up, station_range=(ta, tb), band_fraction=frac,
                  note="silver skin (epimysium) on the upper surface of the main muscle"),
        fins=[], reference=W(tl.reference), reference_names=REFERENCE_NAMES, appearance=appearance,
        targets=targets, measured={}, provenance=PROV, variation=tl.variation,
        extra=dict(model=tl, densities=dict(muscle=PR.PORK_DENSITY.value, fat=FAT_DENSITY["value"])))
    wp.measured = measure(wp)
    wp.extra["medallion_planes"] = wp.measured["medallions"]
    return wp


def measure(wp, h=.5e-3):
    tl = wp.extra["model"]
    rho_m, rho_f = PR.PORK_DENSITY.value, FAT_DENSITY["value"]
    lo, hi = wp.bounds

    def inside(P):
        return wp.sdf(P) < 0

    def lab(P):
        return wp.reference(P)*10 + wp.region(P)
    vol, _, _ = grid_integrate(inside, lab, lo, hi, h)
    by_reg = {n: 0. for n in REGION_NAMES}
    by_ref = {n: 0. for n in REFERENCE_NAMES}
    for key, V in vol.items():
        rho = rho_f if key % 10 == 2 else rho_m
        by_reg[REGION_NAMES[key % 10]] += V*rho
        by_ref[REFERENCE_NAMES[key//10]] += V*rho
    total = sum(by_reg.values())
    # medallions: planes normal to x every 25 mm from head_trim while the width >= min_width
    md = PROV["medallion"]["value"]
    x = tl.x0 + md["head_trim"]*tl.fL
    planes = [x]
    while True:
        xn = x + md["thickness"]
        if float(tl.Wp(xn)) < md["min_width"]:
            break
        planes.append(xn)
        x = xn
    ys = np.arange(lo[1], hi[1], h) + .5*h
    zs = np.arange(lo[2], hi[2], h) + .5*h
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    Y, Z = Y.ravel(), Z.ravel()
    med = []
    for a, b in zip(planes[:-1], planes[1:]):
        m = 0.
        for xx in np.arange(a, b, h) + .5*h:
            P = np.column_stack([np.full_like(Y, xx), Y, Z])
            ins = (wp.sdf(P) < 0) & (wp.region(P) == 0)
            m += ins.sum()*h**3*rho_m
        med.append(dict(x0=a, x1=b, mass=m))
    return dict(grid_h=h, total_mass=total, region_mass=by_reg, reference_mass=by_ref,
                silver_coverage=wp.skin["coverage"], medallion_planes=planes, medallions=med)
