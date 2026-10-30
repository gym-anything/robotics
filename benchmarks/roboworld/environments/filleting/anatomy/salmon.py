"""Preset B: front half of a skin-on Atlantic salmon (Salmo salar) side in Trim A state:
backbone and belly bones off, pin bones in, belly membrane and belly fat on, collarbone (cleithrum) still on
(Seaborn trim guide: Trim B is the level that takes the collarbone off).

Workpiece frame: skin down on z = 0, head end at -x, back (dorsal edge, loin) toward -y, belly flap toward +y,
bounding box centred on x = y = 0.

Anatomical helper coordinates: u = across-width fraction (0 dorsal edge, 1 belly edge), mapped linearly onto the
fish height; the lateral coordinate of the fish (distance from its median plane) is T(x, u) - z, because the
fillet's upper face is the median cut (loin) or the cavity lining (belly flap).

Numbers and provenance: PROV below.
"""

import numpy as np

from filleting import params as PR
from filleting.anatomy.common import (Workpiece, Profile, SegmentSet, smin, smax, smoothstep, prov,
                                      capsule_mass_props, combine_mass_props, rod_volume, grid_integrate,
                                      value_noise, scatter_points, save_png, srgb)
from filleting.anatomy.trout import CAVITY_ARCH

L_HALF = .32
X0, X1 = -.16, .16

# Across-width thickness profile (fraction of the local loin thickness) vs u. Built from the trout MRI steak
# half-section (Frontiers Genet. 2021 Fig. 1E: width profile of the upper body, cavity arch below the backbone,
# belly wall 0.30 of the half-width), transferred to salmon (estimate: same family, same body plan).
_U = np.array([0., .03, .06, .10, .14, .18, .22, .26, .30, .35, .40, .45, .47, .50, .54, .58, .62, .66, .70, .74,
               .78, .82, .90, .95, .985, 1.])
_P_upper = np.array([.17, .35, .48, .63, .75, .83, .90, .94, .96, .97, .98, .98, .975])


def _arch_profile(u):
    xi = np.clip((u - .47)/(.9*.53), 0, None)
    A = np.minimum(1., (xi/CAVITY_ARCH["xi_full"])**CAVITY_ARCH["exponent"])
    return 1 - (1 - CAVITY_ARCH["wall_over_halfwidth"])*A


_P = np.concatenate([_P_upper, _arch_profile(_U[13:])])
_P[-3:] = [.30, .27, .18]        # belly edge rounding (edge ~5 mm, Wang & Tang for 2.3 kg fish: 'close to the edge')
CROSS_PROFILE = np.column_stack([_U, _P])

PROV = {
    "source_fish": prov(dict(gutted_kg=4.5, fork_length_m=.72), "-", "read+derived",
                        "harvest 4-5 kg gutted, mean 4.5 (Mowi 2024); FL from FishBase-type "
                        "L-W for salmonids ~0.72 m (estimate)"),
    "length": prov(L_HALF, "m", "design", "front half 32 cm of a 55-65 cm side"),
    "width_max": prov(.17, "m", "design(estimate)", "Baader envelope, side width"),
    "width_profile": prov("trout photo depth profile 0.21-0.565 TL, normalised", "-", "read(photo, trout)+transfer",
                          "RFEIMG-0305 depth profile over the fillet's span; salmonid body plan assumed similar"),
    "loin_thickness": prov(.030, "m", "read",
                           "4 kg salmon fillets, thickness above the lateral line at 7 stations head->tail: "
                           "3.0 3.0 3.0 2.8 2.5 2.2 2.1 cm (right), SD 0.2-0.3 (Sigurgisladottir et al. 1999, "
                           "J Food Sci 64:99)"),
    "front_edge_thickness": prov(.024, "m", "estimate", "collar-cut face ramps to the full loin over 50 mm"),
    "cross_profile": prov(CROSS_PROFILE.tolist(), "fraction of loin thickness", "read(image, trout)+transfer",
                          "trout MRI steak half-section (Frontiers Genet. 2021 Fig. 1E)"),
    "belly_flap_thickness": prov(.30*.030, "m", "derived", "0.30 x loin (MRI belly wall / half-width); estimate "
                                 "10-15 mm"),
    "edge_thickness": prov(.005, "m", "read(other size)", "'5 mm was close to the thickness of the edge' for 2.34 kg "
                           "salmon fillets (Wang & Tang, WSU)"),
    "backbone_line": prov(.42, "fraction of width from the dorsal edge", "estimate",
                          "lateral line / horizontal septum at ~39 % depth in the trout photo; groove where the "
                          "centra were cut off"),
    "backbone_groove": prov(dict(depth=3e-3, halfwidth=6e-3), "m", "estimate",
                            "salmon centrum diameter 10-14 mm (estimate); machine cut leaves a shallow channel"),
    "pin_count": prov(29, "per side", "read", "Schroeder et al. 2019 (micro-CT, 3-5 kg fish); PARAMS PIN_COUNT_SALMON"),
    "pin_lengths": prov(dict(range=(14, 50), median=(38, 40), min_4to5kg=18), "mm", "read",
                        "Schroeder 2019: salmon 20-50 mm, shortest 14 (3-4 kg) / 18 (4-5 kg), medians 38 and 40"),
    "pin_length_profile": prov("longest in the front third, shortest at the tail end", "-", "read(direction)+estimate",
                               "SINTEF: 82 % of missed bones in the rear half 'where the smallest pin bones are'; "
                               "Schroeder numbers bones 1 (neck) .. 29 (tail)"),
    "pin_position": prov("epaxial, one per segment from segment 1, proximal end just dorsal of the backbone line, "
                         "0.5-2 mm under the cut face", "-", "read+estimate",
                         "epineurals 'at segment 1 ... caudally until segment 30' (Sankar et al. 2025); 'embedded in "
                         "the epaxial muscles ... within the loin' (Skjelvareid 2017); ends 'often well hidden into "
                         "the flesh' (SINTEF)"),
    "pin_direction": prov(dict(down_deg=30, dorsal_deg=10, jitter_deg=8), "-", "read(qualitative)+estimate",
                          "'extend essentially perpendicularly from the dorsal spine and otherwise point backwards "
                          "towards the tail', ends not all the same way (EP0959686B1); angles not published"),
    "segment_length": prov(9.2e-3, "m", "derived", "59 vertebrae (mode 58-59, Sankar 2025) over ~0.75 FL of a 0.72 m "
                           "fish"),
    "pin_radius": prov((.40e-3, .10e-3), "m", "derived+estimate",
                       "mean volume 9.38 mm3 over ~38 mm (Schroeder) -> 0.56 mm mean diameter; tip section 0.21 mm "
                       "carries the 3.91 N mean break force at 110 MPa"),
    "belly_fat_strip": prov(.025, "m", "search+estimate", "belly-flap strips sold 1-3 cm and 3+ cm (Bakkafrost); Trim D "
                            "removes the belly fat keeping 'a narrow, even belly edge' (Fish of Norway)"),
    "membrane": prov(dict(u_from=.47, thickness=1e-4), "-,m", "estimate",
                     "peritoneum lining the cavity face of the belly flap (cavity apex at 47 % of the depth)"),
    "melanin_spot": prov(dict(diameter=(.015, .030), depth=(.003, .008), region="cranioventral belly flap"), "m",
                         "read+design", "spots 1-3 cm, 2-6 myotomes, 'frequently located in the cranioventral and "
                         "craniodorsal regions of the abdominal wall' (Bjorgen et al. 2015); depth 3-8 mm"),
    "collarbone": prov(dict(radius=2.5e-3), "m", "estimate", "cleithrum along the front edge (Trim A keeps it)"),
    "dark_muscle": prov(dict(half_width=.06, depth=.12), "fraction of width / of thickness", "estimate",
                        "lateral red muscle wedge under the skin at the lateral line (as trout, Stickland ratio)"),
    "portion_mass": prov(.150, "kg", "design", "150 +- 7 g portions"),
    "variation": prov(dict(length=.05, pins=2, spot="position/size/depth"), "-", "authored", ""),
}

REGION_NAMES = ["white_muscle", "dark_muscle", "belly_flap", "belly_fat", "melanin_spot"]
REFERENCE_NAMES = ["loin", "trim_belly"]


class Salmon:
    def __init__(self, seed=0):
        self.seed = seed
        rng = np.random.default_rng(5000 + seed)
        if seed == 0:
            fL, dpin = 1.0, 0
        else:
            fL, dpin = 1 + rng.uniform(-.05, .05), int(rng.integers(-2, 3))
        self.fL = fL
        self.x0, self.x1 = X0*fL, X1*fL
        self.n_pin = PROV["pin_count"]["value"] + dpin
        # width profile: trout photo depth over 0.21..0.565 TL (the fillet's span on the fish), normalised
        from filleting.anatomy.trout import OUTLINE
        s = OUTLINE[:, 0]
        dep = OUTLINE[:, 1] - OUTLINE[:, 2]
        span = np.linspace(.21, .565, 30)
        d = np.interp(span, s, dep)
        d = d/d.max()
        xs = np.linspace(self.x0, self.x1, 30)
        W = PROV["width_max"]["value"]*fL*d
        # dorsal edge nearly straight, belly edge convex (the flap)
        self.W = Profile(xs, W)
        self.xs = xs
        self.T_loin = PROV["loin_thickness"]["value"]*fL
        self.P = Profile(CROSS_PROFILE[:, 0], CROSS_PROFILE[:, 1])
        self.y_off = 0.
        xx = np.linspace(self.x0, self.x1, 400)
        self.y_off = .5*(float(np.min(self.y_dorsal(xx))) + float(np.max(self.y_belly(xx))))   # centre bbox on y=0
        self.u_bb = PROV["backbone_line"]["value"]
        self.lam = PROV["segment_length"]["value"]*fL
        self.variation = dict(seed=seed, length_scale=fL, pin_bones=self.n_pin,
                              pose=dict(dx=float(rng.uniform(-.015, .015)) if seed else 0.,
                                        dy=float(rng.uniform(-.015, .015)) if seed else 0.,
                                        yaw_deg=float(rng.uniform(-8, 8)) if seed else 0.))
        # melanin spot (seeded; always present, the B3 task needs one)
        spot = PROV["melanin_spot"]["value"]
        self.spot_d = float(rng.uniform(*spot["diameter"]))*fL
        self.spot_depth = float(rng.uniform(*spot["depth"]))*fL
        self.spot_x = self.x0 + (.10 + .30*rng.random())*(self.x1 - self.x0)
        self.spot_u = float(rng.uniform(.64, .86))
        self.variation["melanin_spot"] = dict(diameter=self.spot_d, depth=self.spot_depth, x=self.spot_x,
                                              u=self.spot_u)
        self._build_pins(np.random.default_rng(5500 + seed))
        self._build_collarbone()

    # ------------------------------------------------------------ outline and thickness
    def y_dorsal(self, x):
        """Dorsal edge: nearly straight; the belly edge (y_dorsal + W) carries the convexity."""
        W = self.W(x)
        return -.5*W + .018*W*(1 - ((x - self.x0)/(self.x1 - self.x0) - .4)**2/.36) - self.y_off

    def y_belly(self, x):
        return self.y_dorsal(x) + self.W(x)

    def x_front(self, y):
        """Collar-cut front edge (convex forward, following the cleithrum), estimate."""
        W = self.W(self.x0)
        u = np.clip((y - self.y_dorsal(self.x0))/W, 0, 1)
        return self.x0 + .022*self.fL*u**1.5

    def u_of(self, x, y):
        return (y - self.y_dorsal(x))/np.maximum(self.W(x), 1e-6)

    def T_long(self, x):
        t = (x - self.x0)/(self.x1 - self.x0)
        f = np.interp(t, [0, .16, .6, 1.], [PROV["front_edge_thickness"]["value"]/.030, 1., 1., 28/30.])
        return self.T_loin*f

    def T(self, x, y):
        u = np.clip(self.u_of(x, y), 0, 1)
        t = self.T_long(x)*self.P(u)
        g = PROV["backbone_groove"]["value"]
        du = (u - self.u_bb)*self.W(x)
        t = t - g["depth"]*self.fL*np.exp(-(du/(g["halfwidth"]*self.fL))**2)
        return np.maximum(t, PROV["edge_thickness"]["value"]*self.fL*.8)

    def outline2d(self, x, y):
        d = smax(self.y_dorsal(x) - y, y - self.y_belly(x), 1.5e-3)
        d = smax(d, self.x_front(y) - x, 1.5e-3)
        return smax(d, x - self.x1, 0.)

    def tissue(self, x, y, z):
        eps = 2e-5
        T = self.T(x, y)
        Tx = (self.T(x + eps, y) - self.T(x - eps, y))/(2*eps)
        Ty = (self.T(x, y + eps) - self.T(x, y - eps))/(2*eps)
        top = (z - T)/np.sqrt(1 + Tx*Tx + Ty*Ty)
        d = smax(self.outline2d(x, y), top, 2e-3)
        return smax(d, -z, 1e-3)

    # ------------------------------------------------------------ anatomy
    def _cone_axes(self, x):
        # fish height v in [-1, 1] (+1 dorsal) from u; lateral w = T - z; cones mid-epaxial / mid-hypaxial
        v_bb = 1 - 2*self.u_bb
        v_e = .5*(1 + v_bb)
        v_h = .5*(v_bb - 1)
        return v_e, v_h, .5*self.T_long(x)/(.5*self.W(x)), np.ones_like(x)

    def myo_coords(self, x, y, z):
        u = self.u_of(x, y)
        v = 1 - 2*u                                 # normalised fish height
        w = (self.T(x, y) - z)/(.5*self.W(x))       # lateral, normalised by the half-depth
        return v, w

    def _phase_phys(self, x, y, z):
        """W-myomere phase (see common.Myomeres): one myoseptum per segment (lam), nested cones about the
        mid-epaxial and mid-hypaxial axes, cone length 15 segments per unit of normalised radius (as trout)."""
        v, w = self.myo_coords(x, y, z)
        v_e, v_h, w_o, _ = self._cone_axes(x)
        v_bb = 1 - 2*self.u_bb
        # each cone normalised by its own muscle height (x2 so that r spans ~0..1 as in the trout model):
        # the W's forward cusp lies on the horizontal septum
        r_e = np.hypot(v - v_e, np.abs(w) - w_o)/(1 - v_bb)
        r_h = np.hypot(v - v_h, np.abs(w) - w_o)/(1 + v_bb)
        r = smin(r_e, r_h, .08)
        return (x - self.x0 + 1.5*self.lam + 15*self.lam*r)/self.lam

    def sheet(self, x, y, z, eps=5e-5):
        f = self._phase_phys
        g = np.stack([(f(x + eps, y, z) - f(x - eps, y, z))/(2*eps), (f(x, y + eps, z) - f(x, y - eps, z))/(2*eps),
                      (f(x, y, z + eps) - f(x, y, z - eps))/(2*eps)], 1)
        gn = np.linalg.norm(g, axis=1)
        return g/np.maximum(gn, 1e-12)[:, None], 1/np.maximum(gn, 1e-12)

    def region(self, x, y, z):
        u = self.u_of(x, y)
        lab = np.zeros(len(x), int)
        lab[u > .47] = 2
        # dark muscle wedge on the skin side along the lateral line
        dm = PROV["dark_muscle"]["value"]
        prof = np.clip(1 - np.abs(u - self.u_bb)/dm["half_width"], 0, 1)
        lab[z < dm["depth"]*self.T_long(x)*prof] = 1
        lab[self.belly_fat(x, y)] = 3
        lab[self.spot_sdf(x, y, z) < 0] = 4
        return lab

    def belly_fat(self, x, y):
        return self.y_belly(x) - y < PROV["belly_fat_strip"]["value"]*self.fL

    def spot_sdf(self, x, y, z):
        """Seeded melanin spot: an ellipsoid set into the cavity face of the belly flap."""
        yc = self.y_dorsal(self.spot_x) + self.spot_u*self.W(self.spot_x)
        zc = float(self.T(np.array([self.spot_x]), np.array([yc]))[0])
        a = np.array([.5*self.spot_d, .4*self.spot_d, self.spot_depth])
        q = np.sqrt(((x - self.spot_x)/a[0])**2 + ((y - yc)/a[1])**2 + ((z - zc)/a[2])**2)
        return (q - 1)*a.min()

    def reference(self, x, y, z):
        ref = np.zeros(len(x), int)
        ref[self.belly_fat(x, y)] = 1
        return ref

    # ------------------------------------------------------------ pins and collarbone
    def _build_pins(self, rng):
        fL = self.fL
        rods = []
        Lr = PROV["pin_lengths"]["value"]
        lo, hi = Lr["range"]
        r0, r1 = PROV["pin_radius"]["value"]
        dd = PROV["pin_direction"]["value"]
        for j in range(1, self.n_pin + 1):
            xj = self.x0 + (j - .3)*self.lam
            t = (j - 1)/max(self.n_pin - 1, 1)
            prof = np.interp(t, [0, .2, .45, .75, 1.], [.62, 1.0, .88, .55, .15])
            L = (lo + (hi - lo)*prof)*(1 + .07*rng.standard_normal())*1e-3*fL
            L = float(np.clip(L, lo*1e-3*fL, hi*1e-3*fL))
            u0 = self.u_bb - .035
            y0 = self.y_dorsal(xj) + u0*self.W(xj)
            z_top = float(self.T(np.array([xj]), np.array([y0]))[0])
            depth0 = float(rng.uniform(.5e-3, 2e-3))
            p0 = np.array([xj, y0, z_top - depth0])
            dn = np.radians(dd["down_deg"] + dd["jitter_deg"]*rng.uniform(-1, 1))
            dy = np.radians(dd["dorsal_deg"] + dd["jitter_deg"]*rng.uniform(-1, 1))
            d = np.array([np.cos(dn)*np.cos(dy), -np.cos(dn)*np.sin(dy), -np.sin(dn)])
            # keep the tip at least 3 mm above the skin: flatten the dive if the bone is long
            max_drop = p0[2] - 3e-3
            if L*np.sin(dn) > max_drop:
                dn = np.arcsin(max(max_drop/L, .05))
                d = np.array([np.cos(dn)*np.cos(dy), -np.cos(dn)*np.sin(dy), -np.sin(dn)])
            n = max(int(np.ceil(L/5e-3)) + 1, 4)
            pts = p0 + np.linspace(0, L, n)[:, None]*d
            rods.append(dict(kind="pin", name=f"pin_{j}", number_from_neck=j, points=pts,
                             radius=np.linspace(r0, r1, n)*fL, root=None, length=L,
                             note="root bond at the proximal end (PIN_ROOT_SHARE), shaft bond along it"))
        self.rods = rods

    def _build_collarbone(self):
        fL = self.fL
        r = PROV["collarbone"]["value"]["radius"]*fL
        us = [.30, .55, .78, .95]
        caps = []
        pts = []
        for u in us:
            y = self.y_dorsal(self.x0) + u*self.W(self.x0)
            x = float(self.x_front(np.array([y]))[0]) + r + 1.5e-3
            T = float(self.T(np.array([x]), np.array([y]))[0])
            pts.append(np.array([x, y, max(T - r - 1.2e-3, r + 1e-3)]))
        for a, b in zip(pts[:-1], pts[1:]):
            caps.append((a, b, r))
        self.collar_caps = caps


def _salmon_texture(sm, res=(1024, 2048)):
    """Skin texture on the planar skin UV (u = (x - x0)/L, v = (y - y_dorsal)/W): steel-blue back, silver flanks,
    small black X-shaped spots above the lateral line, white belly (USGS fact sheet; Animal Diversity Web; NOAA)."""
    H, W = res
    rng = np.random.default_rng(6000 + sm.seed)
    v = (np.arange(H) + .5)/H
    u = (np.arange(W) + .5)/W
    V, U = np.meshgrid(v, u, indexing="ij")
    back, flank, belly = srgb("#46586a"), srgb("#b9c2c8"), srgb("#eef0ee")
    t1 = smoothstep(.30, .02, V)
    t3 = smoothstep(.55, .85, V)
    col = np.empty((H, W, 3))
    for c in range(3):
        col[..., c] = flank[c]*(1 - t1) + back[c]*t1
        col[..., c] = col[..., c]*(1 - t3) + belly[c]*t3
    n1 = value_noise((H, W), 60, rng)
    col *= (.92 + .1*n1[..., None])
    # scales (~2 mm on a 4.5 kg salmon)
    sx, sy = 2.2e-3/(sm.x1 - sm.x0)*W, 2.2e-3/.17*H
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    row = np.floor(Y/sy)
    fx = ((X/sx + .5*(row % 2)) % 1) - .5
    fy = (Y/sy) % 1
    rim = np.clip(1 - np.abs(np.hypot(fx, fy - .1) - .45)/.12, 0, 1)
    col *= 1 - .06*rim[..., None]
    pts = scatter_points(420, rng, weight=lambda a, b: float(np.clip((.42 - b)/.35, 0, 1)), min_dist=.012)
    for a, b in pts:
        cx, cy = a*W, b*H
        size = rng.uniform(1.8e-3, 3.2e-3)/(sm.x1 - sm.x0)*W
        for ang in (rng.uniform(0, np.pi), None):
            ang = ang if ang is not None else ang_prev + np.pi/2 + rng.uniform(-.3, .3)
            ang_prev = ang
            L = np.linspace(-size, size, 40)
            for q in L:
                xx, yy = int(cx + q*np.cos(ang)), int(cy + q*np.sin(ang)*H/W*(sm.x1 - sm.x0)/.17)
                if 0 <= xx < W - 2 and 0 <= yy < H - 2:
                    col[yy:yy + 2, xx:xx + 2] = col[yy:yy + 2, xx:xx + 2]*.25 + srgb("#1a1c1f")*.75
    ll = np.abs(V - sm.u_bb) < .002
    col[ll] *= .8
    return np.clip(col, 0, 1)


def _salmon_flesh(sm, x, y, z):
    orange = srgb("#f2855d")      # raw farmed salmon (SalmoFan ~27-28, estimate)
    fat = srgb("#fbe3d4")
    dark = srgb("#7b5a52")
    black = srgb("#2a2624")
    col = np.tile(orange, (len(x), 1))
    lab = sm.region(x, y, z)
    u = sm.u_of(x, y)
    # fattier (paler) toward the belly
    pale = np.clip((u - .45)/.5, 0, 1)[:, None]
    col = col*(1 - .35*pale) + fat*.35*pale
    col[lab == 1] = dark
    col[lab == 3] = .35*orange + .65*fat
    # myosepta: white fat lines, wider in the belly (fattier)
    ph = sm._phase_phys(x, y, z)
    _, sp = sm.sheet(x, y, z)
    dist = np.abs(ph - np.round(ph))*sp
    width = .35e-3 + .5e-3*pale[:, 0]
    a = np.clip(1 - dist/width, 0, 1)[:, None]
    col = col*(1 - .85*a) + fat*.85*a
    # melanin spot: diffuse grey-black, mottled, fading at the rim (Bjorgen 2015 describes focal dark
    # discolouration; the pattern here is authored)
    ds = sm.spot_sdf(x, y, z)
    near = ds < 2e-3
    if near.any():
        h = np.sin(x[near]*5711.3 + y[near]*3313.7 + z[near]*9133.1)*43758.5453
        mott = .55 + .45*(h - np.floor(h))
        a = .8*np.clip(1 - ds[near]/2.5e-3, 0, 1)**1.2*np.clip(mott + .2, 0, 1)
        col[near] = col[near]*(1 - a[:, None]) + (srgb("#3b302c")*(.7 + .5*mott)[:, None])*a[:, None]
    return np.clip(col, 0, 1)


def build(seed=0, texture_dir=None, texture_res=(1024, 2048)):
    sm = Salmon(seed)
    fL = sm.fL

    def to_local(p):
        p = np.atleast_2d(p)
        return p[:, 0], p[:, 1], p[:, 2]

    def W(fn):
        return lambda p: fn(*to_local(p))

    xx = np.linspace(sm.x0, sm.x1, 400)
    y_lo, y_hi = float(np.min(sm.y_dorsal(xx))), float(np.max(sm.y_belly(xx)))
    lo = np.array([sm.x0 - 2e-3, y_lo - 2e-3, -2e-3])
    hi = np.array([sm.x1 + 2e-3, y_hi + 2e-3, sm.T_loin + 2e-3])
    rho_b = PR.BONE_DENSITY.value
    parts = [capsule_mass_props(a, b, r, rho_b) for a, b, r in sm.collar_caps]
    M, c, I = combine_mass_props(parts)
    rigid = [dict(name="collarbone", mass=M, com=c, inertia=I, capsules=sm.collar_caps, parent=None, joint=None,
                  bond=True, provenance="cleithrum left on in Trim A (Trim B removes it); capsules x BONE_DENSITY")]
    rods = [dict(r) for r in sm.rods]
    for r in rods:
        r["mass"] = rod_volume(r["points"], r["radius"])*rho_b

    def sheet_normal(p):
        n, _ = sm.sheet(*to_local(p))
        return n

    def fibre(p):
        x = to_local(p)[0]
        return np.column_stack([np.ones_like(x), np.zeros_like(x), np.zeros_like(x)])

    tol = .75e-3

    def skin_mask(p, tol=tol):
        x, y, z = to_local(p)
        return (z < tol) & (sm.outline2d(x, y) < -tol)

    def membrane_mask(p, tol=tol):
        x, y, z = to_local(p)
        u = sm.u_of(x, y)
        return (np.abs(z - sm.T(x, y)) < tol) & (u > PROV["membrane"]["value"]["u_from"]) & (sm.outline2d(x, y) < -tol)

    interfaces = {
        "belly_fat_line": dict(
            sdf=lambda p: (lambda x, y, z: sm.y_belly(x) - PROV["belly_fat_strip"]["value"]*fL - y)(*to_local(p)),
            extent=lambda p: sm.tissue(*to_local(p)) < 0, params=(), insert=False,
            note="boundary between the belly-fat strip and the lean flap (Trim D line): a line a trimmer judges by eye "
                 "through continuous tissue, not a fascial plane, so it is NOT pre-split in the world (a reference "
                 "line for the demonstration's plan and the evaluator's reference trim)"),
        "collarbone": dict(sdf=lambda p: SegmentSet([(a, b, r, r, 0) for a, b, r in sm.collar_caps]).distance(p)[0],
                           extent=lambda p: sm.tissue(*to_local(p)) < 0, params=("FLESH_BONE_G", "FLESH_BONE_SIGMA")),
    }

    def uv(p):
        x, y, z = to_local(p)
        return np.column_stack([(x - sm.x0)/(sm.x1 - sm.x0), np.clip(sm.u_of(x, y), 0, 1)])

    tex_path = None
    if texture_dir is not None:
        import os
        os.makedirs(texture_dir, exist_ok=True)
        tex_path = os.path.join(texture_dir, f"salmon_skin_seed{seed}.png")
        save_png(tex_path, _salmon_texture(sm, texture_res))

    appearance = dict(skin_uv=uv, skin_texture=dict(path=tex_path, make=lambda res=texture_res: _salmon_texture(sm, res),
                                                    layout="u = (x - x0)/L (columns), v = across-width fraction "
                                                           "(rows, 0 dorsal edge)",
                                                    provenance="generated (authored procedural); USGS/ADW/NOAA "
                                                               "descriptions"),
                      flesh_color=lambda p: _salmon_flesh(sm, *to_local(p)),
                      flesh_provenance="authored procedural: orange flesh, white fat lines on the myosepta (wider "
                                       "toward the belly), grey-brown lateral dark muscle, black melanin spot")

    # portion planes (150 g), from the head end, on the as-is side and on the belly-trimmed side
    targets = dict(mass=dict(value=PR.SALMON_SIDE.value["kg"], label=PR.SALMON_SIDE.label, basis=PR.SALMON_SIDE.basis),
                   pin_count=dict(value=PR.PIN_COUNT_SALMON.value, pm=2, label=PR.PIN_COUNT_SALMON.label),
                   pin_length_range_mm=dict(value=(14, 50), label="read", basis="Schroeder 2019"),
                   pin_length_median_mm=dict(value=(38, 40), label="read", basis="Schroeder 2019 (3-4 / 4-5 kg)"),
                   loin_thickness=dict(value=.030, label="read", basis="Sigurgisladottir 1999"),
                   belly_flap_thickness=dict(value=(.010, .015), label="estimate", basis=""),
                   melanin_spot=dict(diameter=(.015, .030), depth=(.003, .008), label="read+design"))
    wp = Workpiece(
        name="salmon_half_side", sdf=W(sm.tissue), bounds=(lo, hi), region=W(sm.region), region_names=REGION_NAMES,
        sheet_normal=sheet_normal, fibre=fibre, rigid_bodies=rigid, rods=rods, interfaces=interfaces,
        skin=dict(skin_mask=skin_mask, thickness=PR.SKIN_THICKNESS_SALMON.value, params="SKIN_THICKNESS_SALMON",
                  membrane_mask=membrane_mask, membrane_thickness=PROV["membrane"]["value"]["thickness"],
                  tolerance=tol, note="skin on the z = 0 face; belly membrane (peritoneum) on the cavity face of the "
                                      "flap, u > 0.47"),
        fins=[], reference=W(sm.reference), reference_names=REFERENCE_NAMES, appearance=appearance,
        targets=targets, measured={}, provenance=PROV, variation=sm.variation,
        extra=dict(model=sm, to_local=lambda p: np.atleast_2d(p), thickness=lambda x, y: sm.T(x, y),
                   spot_sdf=W(sm.spot_sdf), densities=dict(tissue=PR.FISH_DENSITY.value, bone=rho_b)))
    wp.measured = measure(wp)
    wp.extra["portion_planes"] = wp.measured["portion_planes"]
    return wp


def measure(wp, h=.6e-3):
    sm = wp.extra["model"]
    rho = PR.FISH_DENSITY.value
    lo, hi = wp.bounds

    def inside(P):
        return wp.sdf(P) < 0

    def lab(P):
        return wp.reference(P)*10 + wp.region(P)
    vol, m1, _ = grid_integrate(inside, lab, lo, hi, h)
    by_ref = {n: 0. for n in REFERENCE_NAMES}
    by_reg = {n: 0. for n in REGION_NAMES}
    for key, V in vol.items():
        by_ref[REFERENCE_NAMES[key//10]] += V*rho
        by_reg[REGION_NAMES[key % 10]] += V*rho
    tissue = sum(by_ref.values())
    bones = sum(rb["mass"] for rb in wp.rigid_bodies) + sum(r["mass"]*(1 - rho/PR.BONE_DENSITY.value) for r in wp.rods)
    # mass per x slab for portion planes
    xs = np.arange(lo[0], hi[0], h) + .5*h
    ys = np.arange(lo[1], hi[1], h) + .5*h
    zs = np.arange(lo[2], hi[2], h) + .5*h
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    Y, Z = Y.ravel(), Z.ravel()
    slab_all, slab_trim = [], []
    for x in xs:
        P = np.column_stack([np.full_like(Y, x), Y, Z])
        m = wp.sdf(P) < 0
        slab_all.append(m.sum()*h**3*rho)
        slab_trim.append((m & (wp.reference(P) == 0)).sum()*h**3*rho)

    def planes(slab):
        cum = np.cumsum(slab)
        target = PROV["portion_mass"]["value"]
        out, k = [], 1
        while k*target < cum[-1]:
            i = np.searchsorted(cum, k*target)
            out.append(float(xs[i] + .5*h))
            k += 1
        return out
    pins = [r for r in wp.rods if r["kind"] == "pin"]
    lens = np.array([r["length"] for r in pins])*1e3
    return dict(grid_h=h, total_mass=tissue + bones, tissue_mass=tissue, bone_mass=bones, reference_mass=by_ref,
                region_mass=by_reg, trim_fraction=by_ref["trim_belly"]/tissue,
                portion_planes=dict(as_is=planes(slab_all), trimmed=planes(slab_trim)),
                pin_count=len(pins), pin_length_mm=dict(min=float(lens.min()), median=float(np.median(lens)),
                                                        max=float(lens.max()), all=lens.round(1).tolist()),
                spot=dict(diameter=sm.spot_d, depth=sm.spot_depth, volume=by_reg["melanin_spot"]/rho))
