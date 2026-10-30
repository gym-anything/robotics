"""Preset A: gutted, head-on portion rainbow trout (Oncorhynchus mykiss), gills removed, belly cavity open and
rinsed.

Anatomical frame (internal): s = distance from the snout tip along the body axis, v = dorsal-positive height
above the body axis, w = lateral (positive = the upper flank when the fish lies on the board).
Workpiece frame: x = s - TL/2 (head at -x), y = -v + y_c (back toward -y), z = w + half-width
(the fish lies on its lower flank, lowest point at z = 0).

Shape source. The dorsal and ventral outline is DIGITISED from a scaled photograph of a gutted farmed rainbow
trout of 317.0 mm total length lying on its side: FDA Regulatory Fish Encyclopedia image RFEIMG-0305 (public
domain; https://commons.wikimedia.org/wiki/File:Oncorhynchus_mykiss_(RFEIMG-0305).jpg; 10 cm scale bar
measured at 86.0 px/cm on the 3840 px rendition). Fins were masked out of the silhouette and interpolated
across. Landmarks (eye, gill-cover margin, fin positions, lateral line) were read from the same photograph and
cross-checked against the farmed-trout morphometric means of Sci. Rep. 2025 (PMC12589511, Table 6).
Counts: vertebrae 62, 36 pre-haemal (rib-bearing) vertebrae, 13 dorsal and 12 anal pterygiophores, 18
supraneurals (Boglione et al. 2014, PLoS ONE 9:e96983, Table 4, read); 31 pin bones per fillet, which are
epineurals in rainbow trout (Schroeder et al. 2019, PMC6582037; BMC Biology 2024, PMC11689531).
Every number below carries a provenance label in PROV.
"""

import numpy as np

from filleting import params as PR
from filleting.anatomy.common import (Workpiece, Loft, Profile, SegmentSet, Myomeres, smin, smax, smoothstep,
                                      prov, capsule_mass_props, combine_mass_props, rod_volume, grid_integrate,
                                      inertia_from_moments, value_noise, scatter_points, save_png, srgb)

PHOTO_TL = 316.98          # mm, digitised total length of RFEIMG-0305 (read: photo, 86.0 px/cm scale bar)

# Outline digitised from RFEIMG-0305: (s/TL, v_top/TL, v_bottom/TL) about the snout-to-peduncle axis, fins removed.
# Rows past 0.8266 (fin base, where the silhouette is caudal-fin rays) are an estimate of the tissue end.
OUTLINE = np.array([
    (0.0000, +0.0012, -0.0124), (0.0126, +0.0137, -0.0324), (0.0252, +0.0229, -0.0452), (0.0379, +0.0304, -0.0536),
    (0.0505, +0.0370, -0.0582), (0.0631, +0.0425, -0.0600), (0.0757, +0.0473, -0.0665), (0.0883, +0.0518, -0.0734),
    (0.1010, +0.0577, -0.0859), (0.1136, +0.0633, -0.0965), (0.1262, +0.0681, -0.1026), (0.1388, +0.0724, -0.1069),
    (0.1514, +0.0769, -0.1100), (0.1640, +0.0819, -0.1125), (0.1767, +0.0863, -0.1140), (0.1893, +0.0902, -0.1148),
    (0.2019, +0.0934, -0.1187), (0.2145, +0.0962, -0.1227), (0.2271, +0.0989, -0.1235), (0.2398, +0.1010, -0.1230),
    (0.2524, +0.1030, -0.1232), (0.2650, +0.1044, -0.1237), (0.2776, +0.1059, -0.1248), (0.2902, +0.1071, -0.1258),
    (0.3029, +0.1082, -0.1266), (0.3155, +0.1091, -0.1269), (0.3281, +0.1099, -0.1272), (0.3407, +0.1105, -0.1272),
    (0.3533, +0.1113, -0.1261), (0.3660, +0.1121, -0.1241), (0.3786, +0.1124, -0.1214), (0.3912, +0.1127, -0.1207),
    (0.4038, +0.1131, -0.1219), (0.4164, +0.1129, -0.1230), (0.4291, +0.1112, -0.1236), (0.4417, +0.1091, -0.1243),
    (0.4543, +0.1071, -0.1251), (0.4669, +0.1051, -0.1251), (0.4795, +0.1031, -0.1233), (0.4921, +0.1010, -0.1212),
    (0.5048, +0.0990, -0.1190), (0.5174, +0.0970, -0.1169), (0.5300, +0.0950, -0.1148), (0.5426, +0.0931, -0.1127),
    (0.5552, +0.0918, -0.1105), (0.5679, +0.0901, -0.1085), (0.5805, +0.0886, -0.1069), (0.5931, +0.0875, -0.1048),
    (0.6057, +0.0857, -0.1021), (0.6183, +0.0840, -0.0958), (0.6310, +0.0824, -0.0889), (0.6436, +0.0807, -0.0841),
    (0.6562, +0.0786, -0.0793), (0.6688, +0.0764, -0.0746), (0.6814, +0.0744, -0.0698), (0.6941, +0.0727, -0.0651),
    (0.7067, +0.0709, -0.0603), (0.7193, +0.0676, -0.0556), (0.7319, +0.0637, -0.0508), (0.7445, +0.0597, -0.0461),
    (0.7572, +0.0558, -0.0417), (0.7698, +0.0519, -0.0394), (0.7824, +0.0486, -0.0380), (0.7950, +0.0460, -0.0374),
    (0.8076, +0.0438, -0.0390), (0.8202, +0.0445, -0.0415),
    (0.8392, +0.0470, -0.0440), (0.8518, +0.0485, -0.0455), (0.8644, +0.0495, -0.0465), (0.8770, +0.0500, -0.0470),
])
OUTLINE_READ_END = 0.8266   # beyond this the silhouette is fin rays; rows are an estimate

# Cross-section shape. Measured on the three MRI trout steaks of Frontiers in Genetics 2021 (doi
# 10.3389/fgene.2021.639223, Fig. 1E, CC BY; ~2 kg farmed trout, steak cut in front of the dorsal fin):
# width/depth 0.437, 0.469, 0.468 (mean 0.46); the width profile over the height fits the egg-superellipse with
# e = -0.03, n_top = 2.23, n_bottom = 3.5 at 3.5 % rms of the width (n_bottom capped at 3.5: the free fit ran to
# 6 because the steak's gutted belly flaps splay). Head and peduncle values are estimates (no section found).
WIDTH_RATIO = [(0.0, .80), (.04, .72), (.10, .60), (.18, .52), (.28, .46), (.58, .46), (.70, .44), (.80, .42),
               (.877, .40)]
EGG = [(0.0, .0), (.15, -.03), (.60, -.03), (.75, .0), (.877, .0)]
POWER_TOP = [(0.0, 2.0), (.15, 2.23), (.60, 2.23), (.75, 2.0), (.877, 2.0)]
POWER_BOT = [(0.0, 2.0), (.10, 2.5), (.18, 3.5), (.60, 3.5), (.68, 2.3), (.75, 2.0), (.877, 2.0)]

# Cavity arch, measured on the same MRI steaks: apex just below the backbone (47 % of the depth from the top),
# cavity half-width / max = (xi/0.7)^0.55 (xi = fraction of the apex-to-bottom height), belly wall = 0.15 of the
# steak width = 0.30 of the half-width on the lower cavity.
CAVITY_ARCH = dict(xi_full=.70, exponent=.55, wall_over_halfwidth=.30)

# Lateral line (= horizontal septum = backbone height) read off RFEIMG-0305 (image rows converted to mm above
# the snout-peduncle axis): (s mm, v mm) in photo units.
LATERAL_LINE = [(40.0, 8.6), (46.5, 8.5), (100.0, 7.5), (140.0, 6.4), (180.0, 4.7), (220.0, 4.2), (250.0, 2.1),
                (274.0, 1.0), (285.0, 1.0)]

# ------------------------------------------------------------------------------------------ provenance
N_SINGLE = 5            # anterior vertebrae with a backbone joint each (_build_skeleton)

PROV = {
    "total_length": prov(.315, "m", "design+read(photo)",
                         "31.5 cm; reference photo RFEIMG-0305 fish measures 317.0 mm"),
    "outline": prov("OUTLINE table", "fraction of TL", "read(photo)",
                    "digitised silhouette of RFEIMG-0305 (gutted farmed trout, 10 cm scale bar), fins masked"),
    "max_depth": prov(.0753*.315/.31698, "m", "read(photo)", "75.3 mm at 34 % TL in RFEIMG-0305 (design 7.5 cm)"),
    "width_ratio": prov(WIDTH_RATIO, "-", "read(image)+estimate",
                        "trunk 0.46 = mean width/depth of 3 MRI steaks (Frontiers Genet. 2021 Fig. 1E, ~2 kg trout); "
                        "juvenile regression 0.51 depth + 0.76 mm (figshare 664126) would give 0.52 and the design "
                        "3.6/7.5 = 0.48; head and peduncle ratios estimated"),
    "section_shape": prov(dict(egg=EGG, power_top=POWER_TOP, power_bottom=POWER_BOT), "-", "read(image)+estimate",
                          "egg-superellipse fitted to the MRI steak width profile (rms 3.5 % of width) on the trunk; "
                          "head/peduncle estimated"),
    "cavity_arch": prov(CAVITY_ARCH, "-", "read(image)", "MRI steak cavity profile (Frontiers Genet. 2021 Fig. 1E)"),
    "standard_length": prov(.865, "fraction of TL", "read",
                            "SL/TL 0.847-0.880 (Sci. Rep. 2025 farmed-trout means, PMC12589511 Table 6)"),
    "head_length": prov(57./PHOTO_TL, "fraction of TL", "read(photo)",
                        "gill-cover margin at 57 mm in RFEIMG-0305 (18.0 % TL); farm means 20.4-21.4 % TL "
                        "(PMC12589511) - this fish has a smaller head than those farms' means"),
    "eye": prov(dict(s=16., v=1.5, r=6.6), "photo mm", "read(photo)+read",
                "eye centre from RFEIMG-0305; diameter 4.0-4.3 % TL (PMC12589511)"),
    "lateral_line": prov(LATERAL_LINE, "photo mm", "read(photo)", "pored-scale line traced in RFEIMG-0305"),
    "backbone_height": prov("= lateral line", "-", "estimate",
                            "vertebral column placed on the horizontal septum, which meets the skin at the lateral "
                            "line; no published position found"),
    "vertebrae": prov(62, "-", "read", "median 62-63, range 57-65 (Boglione et al. 2014 Table 4); FishBase 60-66"),
    "rib_pairs": prov(36, "-", "read+derived",
                      "pre-haemal vertebrae (35-37) carry pleural ribs (Boglione 2014 definition); ribs on V3.."),
    "pin_bones": prov(31, "per fillet", "read", "31 in rainbow trout, micro-CT of 3-5 kg fish (Schroeder 2019); "
                      "they are epineurals (PMC11689531); rooted on V1..V31"),
    "pin_length": prov((.44*15, .44*55), "mm", "read+derived",
                       "trout 15-55 mm in 3-5 kg fish (Schroeder 2019) x 0.44 = body-length ratio 31.7/72 cm "
                       "(FishBase W = 0.00933 L^3.03 at 4 kg); profile along the row (longest in the front third, "
                       "shortest at the tail end, SINTEF) is an estimate"),
    "pin_radius": prov((.18e-3, .06e-3), "m", "derived+estimate",
                       "salmon mean diameter 0.56 mm from Schroeder's 9.38 mm3 over 38 mm, scaled by 0.44"),
    "dorsal_pterygiophores": prov(13, "-", "read", "Boglione 2014 Table 4 median 13 (11-15)"),
    "anal_pterygiophores": prov(12, "-", "read", "Boglione 2014 Table 4 median 12-13"),
    "supraneurals": prov(18, "-", "read", "Boglione 2014 Table 4"),
    "fins": prov(dict(dorsal=(133, 170), adipose=(228, 240), pelvic=(150, 157), anal=(200, 228),
                      pectoral=(60, 66)), "photo mm", "read(photo)+read",
                 "fin bases from RFEIMG-0305; dorsal origin 42 % TL (farm means 41.1-43.2 %), anal base 8.8 % TL "
                 "(8.7-9.3 %), pectoral length 12.1-13.3 % TL (PMC12589511)"),
    "vent": prov(196., "photo mm", "estimate", "4 mm ahead of the anal-fin origin (read from photo, 200 mm)"),
    "centrum_radius": prov(1.8e-3, "m", "estimate", "centrum diameter ~3.5 mm; vertebra length 3.7 mm here"),
    "spine_radius": prov(.25e-3, "m", "estimate", "neural/haemal spine; no source"),
    "spine_reach": prov(.62, "fraction of backbone-to-outline height", "estimate", "no source"),
    "spine_incline": prov((35., 55.), "deg from vertical (front, tail)", "estimate", "no source"),
    "rib_radius": prov((.3e-3, .15e-3), "m", "estimate", "rib diameter 0.5-0.8 mm; tapered to the tip"),
    "rib_incline": prov(25., "deg (posterior slant)", "estimate", "no source"),
    "rib_reach": prov(.72, "fraction of the cavity height below the roof", "estimate",
                      "ribs seen reaching most of the belly wall in the Commons dissection photo "
                      "(Abdominal_cavity_of_rainbow_trout, CC BY-SA 4.0)"),
    "rib_depth": prov(.8e-3, "m", "estimate", "rib centre below the cavity lining (lining ~0.45 mm + radius)"),
    "body_wall": prov(.30, "fraction of the local max half-width", "read(image)",
                      "belly wall 0.15 of steak width on the lower cavity (Frontiers Genet. 2021 Fig. 1E); the "
                      "cavity volume is compared with the viscera of 300-370 g trout, 17.0 % of 334 g (Souza et al. "
                      "2015, Acta Sci. Anim. Sci. Table 2, read)"),
    "slit_gap": prov(6e-3, "m", "estimate", "belly flaps of a gutted fish at rest gape a few mm (photo shows the "
                     "pink cut edges); must exceed the 3.5 mm mesh size to exist in the mesh"),
    "gill_void": prov(dict(centre=(42., -9., 7.), semi=(13., 12., 5.)), "photo mm", "estimate",
                      "gills removed: one ellipsoidal chamber per side under the gill cover (gill cover 25-57 mm "
                      "in the photo), ~2 mm inside the skin and below the skull; gill mass not sourced"),
    "head_density": prov(1100., "kg/m3", "estimate", "soft tissue 1050 + ~5 % bone by volume at 2000"),
    "head_boundary": prov(dict(top=42., backbone=47., bottom=60.), "photo mm", "read(photo)+estimate",
                          "nape scale line at 40-42 mm, occiput above the gill-cover corner (48 mm), cleithrum/"
                          "pectoral base at 60 mm"),
    "collar_line": prov(dict(top=50., backbone=56., pectoral=68., bottom=70.), "photo mm", "estimate",
                        "behind the pectoral fin base (60-66 mm), Seafish 'cut behind it'"),
    "belly_edge_strip": prov(7e-3, "m", "authored", "ventral margin strip counted as trim loss ('belly-edge "
                             "losses'); no published width"),
    "bone_clearance": prov(.5e-3, "m", "authored", ""),
    "dark_muscle": prov(dict(half_height=.105, depth=.09), "fraction of local depth / half-width",
                        "derived+estimate", "red:white cross-section 1:25 in rainbow trout (Stickland 1983 as "
                        "quoted by Karahmet 2014) -> ~4 % of the muscle section"),
    "myomere_cone": prov(15., "segment lengths per unit normalised radius", "estimate",
                         "W amplitude on the skin ~3 segments; rings ~2.5 mm apart on a transverse cut"),
    "fin_thickness": prov(.25e-3, "m", "estimate", "membrane with rays; render only"),
    "variation": prov(dict(length=.05, ribs=1, pins=2), "-", "authored", "seeded variation"),
}

BELLY_EDGE_STRIP = PROV["belly_edge_strip"]["value"]
CLEAR = PROV["bone_clearance"]["value"]

REGION_NAMES = ["white_muscle", "dark_muscle", "belly_flap", "collar"]
REFERENCE_NAMES = ["fillet_upper", "fillet_lower", "frame", "head", "belly_edge"]


class Trout:
    """All geometry in the anatomical frame (s, v, w), metres."""

    def __init__(self, seed=0, texture_dir=None):
        self.seed = seed
        rng = np.random.default_rng(1000 + seed)
        if seed == 0:
            fL, drib, dpin = 1.0, 0, 0
        else:
            fL = 1.0 + rng.uniform(-.05, .05)
            drib = int(rng.integers(-1, 2))
            dpin = int(rng.integers(-2, 3))
        self.fL = fL
        self.TL = .315*fL
        self.k = self.TL/PHOTO_TL            # photo mm -> m
        k = self.k
        self.n_vert = PROV["vertebrae"]["value"]
        self.n_rib = PROV["rib_pairs"]["value"] + drib
        self.n_pin = PROV["pin_bones"]["value"] + dpin
        self.variation = dict(seed=seed, length_scale=fL, total_length=self.TL, rib_pairs=self.n_rib,
                              pin_bones=self.n_pin,
                              pose=dict(dx=float(rng.uniform(-.015, .015)) if seed else 0.,
                                        dy=float(rng.uniform(-.015, .015)) if seed else 0.,
                                        yaw_deg=float(rng.uniform(-8, 8)) if seed else 0.,
                                        note="board pose for the scene; the geometry is unchanged"),
                              mass_scaling="all lengths x length_scale, masses x length_scale^3")
        # ------------------------------------------------------------ outer loft
        sT = OUTLINE[:, 0]*self.TL
        top, bot = OUTLINE[:, 1]*self.TL, OUTLINE[:, 2]*self.TL
        depth = top - bot
        wr = np.interp(OUTLINE[:, 0], *np.array(WIDTH_RATIO).T)
        hw = .5*wr*depth
        egg = np.interp(OUTLINE[:, 0], *np.array(EGG).T)
        pw = (np.interp(OUTLINE[:, 0], *np.array(POWER_TOP).T), np.interp(OUTLINE[:, 0], *np.array(POWER_BOT).T))
        self.SL = PROV["standard_length"]["value"]*self.TL
        self.s_end = sT[-1]                                   # tissue end (fin base) = SL + ~3.5 mm
        self.loft = Loft(sT, top, bot, hw, egg, pw, (0., self.s_end), cap_blend=1e-3)
        self.hw_max = float(np.max(hw))
        # lateral line / backbone
        ll = np.array(LATERAL_LINE)
        self.bb = Profile(ll[:, 0]*k, ll[:, 1]*k)
        self.r_c = PROV["centrum_radius"]["value"]*fL
        self.s_occ = 46.5*k
        self.lam = (self.SL - self.s_occ)/self.n_vert
        self.s_vent = 196.*k
        # head boundary s_hb(v) and collar line s_col(v): quadratic/pchip in v through read/estimated points
        hb = PROV["head_boundary"]["value"]
        v_top42 = float(self.loft.top(42*k))
        v_bot60 = float(self.loft.bot(60*k))
        self._hb = np.polyfit([v_top42, 8.5*k, v_bot60], [hb["top"]*k, hb["backbone"]*k, hb["bottom"]*k], 2)
        cl = PROV["collar_line"]["value"]
        self._col = Profile([v_bot60 - .02, v_bot60, -21.7*k, 8.5*k, v_top42, v_top42 + .02],
                            [cl["bottom"]*k, cl["bottom"]*k, cl["pectoral"]*k, cl["backbone"]*k, cl["top"]*k,
                             cl["top"]*k])
        # cavity
        self.wall_frac = PROV["body_wall"]["value"]
        self.depth_max = float(np.max(depth))
        self.g_slit = PROV["slit_gap"]["value"]
        # myomeres (one per vertebra)
        self.myo = Myomeres(self.s_occ, self.lam, PROV["myomere_cone"]["value"]*self.lam, self._cone_axes)
        # skeleton, rods, fins
        self._build_skeleton()
        self._build_fins()
        self.texture_dir = texture_dir

    # ---------------------------------------------------------------- basic fields (anatomical frame)
    def s_hb(self, v):
        return np.polyval(self._hb, v)

    def s_col(self, v):
        return self._col(v)

    def outer(self, s, v, w):
        return self.loft.sdf(s, v, w)

    def roof(self, s):
        """Cavity apex height: at the centrum equator - kidney removed and rinsed, the ventral half of the centra
        is bare in the cavity roof (and no thin tissue wedge is left between the centrum and the roof)."""
        return self.bb(s)

    def wall(self, s):
        return self.wall_frac*self.loft.hw(s)

    def closure(self, s):
        """1 over the open cavity, falling to 0 at the vent (the cavity tapers into the vent)."""
        return 1. - smoothstep(self.s_vent - 25e-3*self.fL, self.s_vent + 1e-3, s)

    def cavity(self, s, v, w, outer=None):
        """Signed distance-like field of the open, rinsed body cavity (negative inside): the MRI arch
        (half-width grows as xi^0.55 below the apex) limited by a belly wall of constant thickness."""
        d_out = self.outer(s, v, w) if outer is None else outer
        va, b = self.roof(s), self.loft.bot(s)
        xi = np.clip((va - v)/(.9*np.maximum(va - b, 1e-3)), 0, None)
        A = np.minimum(1., (xi/CAVITY_ARCH["xi_full"])**CAVITY_ARCH["exponent"])
        cmax = (self.loft.hw(s) - self.wall(s))*self.closure(s)
        # the apex is at least as wide as the centra (the rinsed kidney groove), so no tissue cusp is left
        # between the centrum and the arch walls
        half = smax(A*cmax, 1.15*self.r_c, .5e-3)
        c = smax(d_out + self.wall(s), np.abs(w) - half, .8e-3)
        # the cavity ends (toward the vent) where its full half-width would drop below 2 mm: a clean end face
        # instead of a sub-resolution crack
        c = smax(c, 2e-3 - cmax, .5e-3)
        c = smax(c, v - va, .8e-3)
        c = smax(c, s - (self.s_vent + 1e-3), .8e-3)
        return c

    def slit(self, s, v, w):
        g = .5*self.g_slit
        d = np.abs(w) - g
        d = smax(d, v - (self.loft.bot(s) + self.wall(s) + 3e-3*self.fL), .5e-3)
        d = smax(d, 55*self.k - s, .5e-3)
        d = smax(d, s - self.s_vent, 1.5e-3)
        return d

    def channel(self, s, v, w):
        slope = self.bb.d(s)
        d = np.hypot(v - self.bb(s), w)/np.sqrt(1 + slope*slope) - self.r_c
        d = smax(d, self.s_occ - s, .3e-3)
        return smax(d, s - self.SL, .3e-3)

    def head_cut(self, s, v):
        slope = np.polyval(np.polyder(self._hb), v)
        return (self.s_hb(v) - s)/np.sqrt(1 + slope*slope)

    def tissue(self, s, v, w):
        d_out = self.outer(s, v, w)
        d = smax(d_out, self.head_cut(s, v), 0.)
        d = smax(d, -self.cavity(s, v, w, d_out), .8e-3)
        d = smax(d, -self.slit(s, v, w), .5e-3)
        d = smax(d, -self.channel(s, v, w), .3e-3)
        return d

    def gill_void(self, s, v, w):
        c = np.array(PROV["gill_void"]["value"]["centre"])*self.k
        a = np.array(PROV["gill_void"]["value"]["semi"])*self.k
        q = np.sqrt(((s - c[0])/a[0])**2 + ((v - c[1])/a[1])**2 + ((np.abs(w) - c[2])/a[2])**2)
        return (q - 1)*a.min()

    def head_solid(self, s, v, w):
        d = smax(self.outer(s, v, w), -self.head_cut(s, v), 0.)
        d = smax(d, -self.gill_void(s, v, w), .5e-3)
        # throat opened ventrally where the gills were cut out (pink area 30-57 mm in the photo)
        throat = smax(np.abs(w) - .5*self.g_slit, v - (self.loft.bot(s) + 6e-3), .5e-3)
        throat = smax(throat, 30*self.k - s, .5e-3)
        return smax(d, -throat, .5e-3)

    # ---------------------------------------------------------------- myomere cone axes
    def _cone_axes(self, s):
        vb = self.bb(s)
        t, b = self.loft.top(s), self.loft.bot(s)
        v_e = vb + .5*(t - vb)
        v_h = vb - .5*(vb - b)
        w_o = .5*self.loft.hw(s)
        return v_e, v_h, w_o, np.maximum(t - vb, 1e-3), np.maximum(vb - b, 1e-3)

    # ---------------------------------------------------------------- skeleton
    def vert_s(self, i):
        """Centre station of vertebra i (1-based)."""
        return self.s_occ + (i - .5)*self.lam

    def _spine(self, s0, up, incline_deg, reach, r):
        """Neural (up=True) or haemal spine from the centrum at station s0 toward the dorsal/ventral outline."""
        vb = float(self.bb(s0))
        edge = float(self.loft.top(s0)) if up else float(self.loft.bot(s0))
        h = abs(edge - vb)*reach
        a = np.radians(incline_deg)
        L = max(h - self.r_c, .5e-3)/np.cos(a)
        sgn = 1. if up else -1.
        p0 = np.array([s0, vb + sgn*self.r_c*.9, 0.])
        p1 = p0 + L*np.array([np.sin(a), sgn*np.cos(a), 0.])
        return p0, p1, r

    def _build_skeleton(self):
        k, fL = self.k, self.fL
        nv = self.n_vert
        # segment sizes: the first N_SINGLE vertebrae one per segment (a heading cut finds a joint every vertebra, 3.7
        # mm, as on a real backbone), the rest lumped 4-5 per segment (12 segments of V6-V62)
        lumped = [nv//13 + (1 if i < nv % 13 else 0) for i in range(13)]       # 13 segments of 4-5 vertebrae
        if N_SINGLE:
            if N_SINGLE != lumped[0]:
                raise ValueError(f"N_SINGLE must split the first lumped segment ({lumped[0]} vertebrae)")
            sizes = [1]*N_SINGLE + lumped[1:]
        else:
            sizes = lumped
        nseg = len(sizes)
        bounds = np.cumsum([0] + sizes)
        self.segments = []
        r_sp = PROV["spine_radius"]["value"]*fL
        inc0, inc1 = PROV["spine_incline"]["value"]
        reach = PROV["spine_reach"]["value"]
        first_haemal = 3 + self.n_rib               # V39 with 36 rib pairs
        last_spined = nv - 4                        # ural vertebrae (last 4) carry the hypural fan
        for j in range(nseg):
            i0, i1 = bounds[j] + 1, bounds[j + 1]   # vertebra indices (1-based, inclusive)
            sa = self.s_occ + (i0 - 1)*self.lam
            sb = self.s_occ + i1*self.lam
            caps = []
            # centra: one capsule per vertebra (short cylinders with rounded ends), axis on the backbone curve
            for i in range(i0, i1 + 1):
                sc = self.vert_s(i)
                half = max(.5*self.lam - self.r_c*.55, .15e-3)
                p0 = np.array([sc - half, float(self.bb(sc - half)), 0.])
                p1 = np.array([sc + half, float(self.bb(sc + half)), 0.])
                caps.append((p0, p1, self.r_c, "centrum", i))
                frac = (sc - self.s_occ)/(self.SL - self.s_occ)
                inc = inc0 + (inc1 - inc0)*frac
                if i <= last_spined:
                    q0, q1, r = self._spine(sc, True, inc, reach, r_sp)
                    caps.append((q0, q1, r, "neural_spine", i))
                if first_haemal <= i <= last_spined:
                    q0, q1, r = self._spine(sc, False, inc, reach, r_sp)
                    caps.append((q0, q1, r, "haemal_spine", i))
            if j == nseg - 1:
                # hypural fan: the ural centra spread into the caudal-fin base (inside the tissue end)
                sh = self.SL
                vb = float(self.bb(sh))
                ph = float(self.loft.top(sh) - self.loft.bot(sh))
                for dv in (-.28, 0., .28):
                    caps.append((np.array([sh - 2.5*self.lam, vb, 0.]),
                                 np.array([self.s_end - .8e-3, vb + dv*ph, 0.]), .45e-3*fL, "hypural", nv))
            name = f"segment_{j:02d}"
            parent = "head" if j == 0 else f"segment_{j - 1:02d}"
            anchor = np.array([sa, float(self.bb(sa)), 0.])
            self.segments.append(dict(name=name, vertebrae=(int(i0), int(i1)), s_range=(sa, sb), capsules=caps,
                                      parent=parent,
                                      joint=dict(anchor=anchor, kind="ball", gap=.6e-3*fL, radius=self.r_c)))
        self.seg_of_vert = {i: s["name"] for s in self.segments for i in range(s["vertebrae"][0],
                                                                              s["vertebrae"][1] + 1)}
        self._build_rods()

    def _cavity_wall_w(self, s, v, lo=0., hi=None):
        """Lateral position of the cavity wall (w > 0 side) at stations (s, v) by bisection."""
        hi = np.full_like(s, .05) if hi is None else hi
        lo = np.full_like(s, lo)
        for _ in range(40):
            m = .5*(lo + hi)
            inside = self.cavity(s, v, m) < 0
            lo = np.where(inside, m, lo)
            hi = np.where(inside, hi, m)
        return .5*(lo + hi)

    def _rib_path(self, s0, v_roof, v_cb, v_tip, beta, depth_rib, spacing=4e-3):
        """Rib centreline: from the centrum flank, along the cavity roof and down the lateral wall, offset
        `depth_rib` outward from the cavity lining; the rib slants posteriorly by beta as it descends."""
        vc = v_roof - .4*(v_roof - v_cb)
        phi = np.radians(np.linspace(3., 178., 176))
        dv, dw = np.cos(phi), np.sin(phi)
        ss = np.full_like(phi, s0)
        for _ in range(2):                       # iterate the posterior slant (s depends on the height)
            lo, hi = np.zeros_like(phi), np.full_like(phi, .06)
            for _ in range(40):
                m = .5*(lo + hi)
                inside = self.cavity(ss, vc + m*dv, m*dw) < 0
                lo, hi = np.where(inside, m, lo), np.where(inside, hi, m)
            t = .5*(lo + hi) + depth_rib
            pts = np.column_stack([ss, vc + t*dv, t*dw])
            ss = s0 + np.clip(v_roof - pts[:, 1], 0, None)*np.tan(beta)
        keep = (pts[:, 2] > self.r_c + .3e-3) & (pts[:, 1] >= v_tip)
        if keep.sum() < 3:
            return None
        pts = pts[keep]
        root = np.array([[s0, float(self.bb(s0)) + .5e-3, self.r_c + .45e-3]])
        pts = np.vstack([root, pts])
        # resample at ~spacing
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        n = max(int(np.ceil(cum[-1]/spacing)) + 1, 3)
        u = np.linspace(0, cum[-1], n)
        return np.column_stack([np.interp(u, cum, pts[:, q]) for q in range(3)])

    def _build_rods(self):
        k, fL = self.k, self.fL
        rods = []
        # ---- pleural ribs: V3 .. V(2+n_rib), both sides, along the cavity wall just under the lining
        r0, r1 = PROV["rib_radius"]["value"]
        beta = np.radians(PROV["rib_incline"]["value"])
        depth_rib = PROV["rib_depth"]["value"]*fL
        reach = PROV["rib_reach"]["value"]
        self.rib_tips = []
        for i in range(3, 3 + self.n_rib):
            s0 = self.vert_s(i)
            v0 = float(self.roof(s0))
            # cavity bottom at this station (lowest cavity point on the midline)
            vv = np.linspace(float(self.loft.bot(s0)), v0, 400)
            inside = self.cavity(np.full_like(vv, s0), vv, np.zeros_like(vv)) < 0
            if not inside.any():
                continue
            v_cb = vv[np.argmax(inside)]
            v_tip = v0 - reach*(v0 - v_cb)
            path = self._rib_path(s0, v0, v_cb, v_tip, beta, depth_rib)
            if path is None:
                continue
            for side in (1., -1.):
                pts = path.copy()
                pts[:, 2] *= side
                n = len(pts)
                rad = np.linspace(r0, r1, n)*fL
                rods.append(dict(kind="rib", name=f"rib_V{i}_{'upper' if side > 0 else 'lower'}",
                                 points=pts, radius=rad, root=self.seg_of_vert[i], vertebra=i,
                                 side="upper" if side > 0 else "lower"))
            self.rib_tips.append((float(path[-1, 0]), float(path[-1, 1])))
        # ---- pin bones = epineurals V1..V(n_pin), both sides
        Lmin, Lmax = PROV["pin_length"]["value"]
        pr0, pr1 = PROV["pin_radius"]["value"]
        rng = np.random.default_rng(2000 + self.seed)
        for j in range(1, self.n_pin + 1):
            i = j
            s0 = self.vert_s(i)
            x = (j - 1)/max(self.n_pin - 1, 1)
            prof = np.interp(x, [0, .25, .55, 1.], [.62, 1.0, .85, .30])     # estimate (see PROV)
            L = (Lmin + (Lmax - Lmin)*prof)*1e-3*self.TL/.315*(1 + .06*rng.standard_normal())
            L = float(np.clip(L, Lmin*1e-3*fL, Lmax*1e-3*fL))
            for side in (1., -1.):
                d = np.array([np.cos(np.radians(35.)), np.sin(np.radians(10.)), side*np.sin(np.radians(35.))])
                d /= np.linalg.norm(d)
                p0 = np.array([s0, float(self.bb(s0)) + self.r_c + .4e-3, side*.6e-3])
                n = max(int(np.ceil(L/4e-3)) + 1, 3)
                pts = p0 + np.linspace(0, L, n)[:, None]*d
                rods.append(dict(kind="pin", name=f"pin_{j}_{'upper' if side > 0 else 'lower'}", points=pts,
                                 radius=np.linspace(pr0, pr1, n)*fL, root=self.seg_of_vert[i], vertebra=i,
                                 number_from_neck=j, side="upper" if side > 0 else "lower"))
        # ---- dorsal and anal pterygiophores (median plane)
        def ptery(sa, sb, count, up, name):
            out = []
            for q, s0 in enumerate(np.linspace(sa, sb, count)):
                vb = float(self.bb(s0))
                edge = float(self.loft.top(s0) if up else self.loft.bot(s0))
                sgn = -1. if up else 1.
                p0 = np.array([s0, edge + sgn*1.8e-3, 0.])
                L = .36*abs(edge - vb)
                d = np.array([-np.sin(np.radians(12.)), sgn*np.cos(np.radians(12.)), 0.])
                n = max(int(np.ceil(L/4e-3)) + 1, 3)
                pts = p0 + np.linspace(0, L, n)[:, None]*d
                out.append(dict(kind="fin_support", name=f"{name}_{q}", points=pts,
                                radius=np.linspace(.3e-3, .15e-3, n)*fL, root=None, fin=name))
            return out
        rods += ptery(135*k, 167*k, PROV["dorsal_pterygiophores"]["value"], True, "dorsal_pterygiophore")
        rods += ptery(202*k, 226*k, PROV["anal_pterygiophores"]["value"], False, "anal_pterygiophore")
        # ---- supraneurals: short median rods above the neural spines between the occiput and the dorsal fin
        for q, s0 in enumerate(np.linspace(50*k, 128*k, PROV["supraneurals"]["value"])):
            vb, t = float(self.bb(s0)), float(self.loft.top(s0))
            p0 = np.array([s0, vb + .68*(t - vb), 0.])
            p1 = np.array([s0 + 2e-3*fL, t - 2.5e-3*fL, 0.])
            rods.append(dict(kind="supraneural", name=f"supraneural_{q}", points=np.stack([p0, .5*(p0 + p1), p1]),
                             radius=np.full(3, .2e-3*fL), root=None))
        # ---- pelvic girdle (basipterygia) in the ventral wall beside the slit
        for side in (1., -1.):
            ss = np.linspace(145*k, 158*k, 4)
            vv = self.loft.bot(ss) + 2.6e-3*fL
            pts = np.column_stack([ss, vv, side*np.full(4, .5*self.g_slit + 2.4e-3*fL)])
            rods.append(dict(kind="fin_support", name=f"pelvic_girdle_{'upper' if side > 0 else 'lower'}",
                             points=pts, radius=np.full(4, .4e-3*fL), root=None, fin="pelvic"))
        self.rods = rods

    # ---------------------------------------------------------------- fins (render membranes)
    def _build_fins(self):
        k = self.k
        top, bot, hw = self.loft.top, self.loft.bot, self.loft.hw
        fins = []

        def median(name, base, pts_rel, up=True):
            s0, s1 = base[0]*k, base[1]*k
            edge = top if up else bot
            sgn = 1. if up else -1.
            outline = [(s0, float(edge(s0)))]
            for ds, h in pts_rel:
                outline.append((s0 + ds*k, float(edge(s0 + ds*k)) + sgn*h*k))
            outline.append((s1, float(edge(s1))))
            o = np.array([(a, b, 0.) for a, b in outline])
            nb = 8
            attach = np.array([(s, float(edge(s)) - sgn*.3e-3, 0.) for s in np.linspace(s0, s1, nb)])
            fins.append(dict(name=name, outline=o, attach=attach, thickness=PROV["fin_thickness"]["value"],
                             plane="median", n_rays=0))
        # dorsal fin: leading ray ~31 mm (0.10 TL, estimate), raked back; adipose small fleshy lobe
        median("dorsal", (133, 170), [(4, 20), (12, 30), (24, 22), (40, 9)], True)
        fins[-1]["n_rays"] = 16
        median("adipose", (228, 240), [(6, 5.5), (15, 6.5), (16, 4)], True)
        median("anal", (200, 228), [(4, 17), (10, 24), (24, 15), (33, 6)], False)
        fins[-1]["n_rays"] = 15
        # caudal fin: forked, from the tissue end to the lobe tips (photo: upper tip 316/+32, fork 300/+3,
        # lower tip 317/-23)
        se = self.s_end
        vt, vb = float(top(se)), float(bot(se))
        cau = np.array([(se, vt - .5e-3, 0.), (292*k, 26*k, 0.), (316*k, 32*k, 0.), (300*k, 3*k, 0.),
                        (317*k, -23*k, 0.), (293*k, -20*k, 0.), (se, vb + .5e-3, 0.)])
        fins.append(dict(name="caudal", outline=cau, attach=np.array([(se, v, 0.) for v in np.linspace(vb, vt, 8)]),
                         thickness=PROV["fin_thickness"]["value"], plane="median", n_rays=19))

        # paired fins lie folded back against the flanks (a dead fish on its side)
        def paired(name, s_base, v_base, length, width, droop, n_rays):
            out = []
            for side in (1., -1.):
                ss = np.linspace(0, 1, 7)
                spine_s = s_base*k + ss*length*k
                spine_v = v_base*k - droop*k*ss
                pts = []
                for sgn in (1, -1):
                    for q, (a, b) in enumerate(zip(spine_s, spine_v)):
                        half = .5*width*k*np.sin(np.pi*np.clip(ss[q]*.9 + .1, 0, 1))
                        pts.append((a, b + sgn*half))
                pts = pts[:7] + pts[7:][::-1]
                P = np.array(pts)
                # lay onto the flank surface (+0.4 mm)
                wv = side*(self.loft.half_width_at(P[:, 0], P[:, 1]) + .4e-3)
                o = np.column_stack([P, wv])
                out.append(dict(name=f"{name}_{'upper' if side > 0 else 'lower'}", outline=o,
                                attach=o[:2], thickness=PROV["fin_thickness"]["value"], plane="flank",
                                n_rays=n_rays))
            return out
        fins += paired("pectoral", 61, -21.7, 40, 12, 6, 13)
        fins += paired("pelvic", 151, -34, 28, 9, 2, 9)
        self.fins = fins

    # ---------------------------------------------------------------- bones as capsule sets
    def bone_segments(self, include_pins=False):
        segs = []
        for seg in self.segments:
            for p0, p1, r, kind, i in seg["capsules"]:
                segs.append((p0, p1, r, r, kind))
        for rd in self.rods:
            if rd["kind"] == "pin" and not include_pins:
                continue
            P, R = rd["points"], rd["radius"]
            for q in range(len(P) - 1):
                segs.append((P[q], P[q + 1], R[q], R[q + 1], rd["kind"]))
        return segs

    # ---------------------------------------------------------------- regions and references
    def dark_band(self, s, v, w, d_out):
        vb = self.bb(s)
        dep = self.loft.top(s) - self.loft.bot(s)
        hh = PROV["dark_muscle"]["value"]["half_height"]*dep
        tm = PROV["dark_muscle"]["value"]["depth"]*self.loft.hw(s)
        prof = np.clip(1 - np.abs(v - vb)/np.maximum(hh, 1e-6), 0, 1)
        return (-d_out < tm*prof) & (s > self.s_hb(v) + 3e-3) & (s < self.SL)

    def region(self, s, v, w):
        d_out = self.outer(s, v, w)
        lab = np.zeros(len(s), int)
        belly = (s < self.s_vent + 2e-3) & (v < self.bb(s))
        lab[belly] = 2
        lab[self.dark_band(s, v, w, d_out)] = 1
        lab[s < self.s_col(v)] = 3
        return lab

    def reference(self, s, v, w, bones=None):
        """0 fillet_upper, 1 fillet_lower, 2 frame, 3 head, 4 belly_edge (tissue points; head points too)."""
        bones = self._bones if bones is None else bones
        ref = np.where(w >= 0, 0, 1)
        d_out = self.outer(s, v, w)
        # belly edge strip (authored width) along the slit, from the collar to just past the vent
        be = (v - self.loft.bot(s) < BELLY_EDGE_STRIP) & (s < self.s_vent + 4e-3) & (v < self.roof(s))
        ref[be] = 4
        # frame: bone clearance (pins stay in the fillet), inside the rib cage, the tail stub past SL
        P = np.column_stack([s, v, w])
        db, _ = bones.distance(P, upper=CLEAR + 1e-3)
        frame = db < CLEAR
        cav = self.cavity(s, v, w, d_out)
        rib_zone = (s > self.rib_s0) & (s < self.rib_s1) & (v > self._rib_tip_v(s))
        frame |= rib_zone & (cav < PROV["rib_depth"]["value"]*self.fL + PROV["rib_radius"]["value"][0] + CLEAR)
        frame |= s > self.SL
        ref[frame] = 2
        ref[s < self.s_col(v)] = 3
        near = s < self.s_hb(v) + 1e-3
        if near.any():
            hs = self.head_solid(s[near], v[near], w[near]) < 0
            idx = np.nonzero(near)[0][hs]
            ref[idx] = 3
        return ref

    def _rib_tip_v(self, s):
        t = np.array(self.rib_tips)
        return np.interp(s, t[:, 0], t[:, 1])

    def finalize(self):
        self._bones = SegmentSet(self.bone_segments(include_pins=False))
        self._all_bones = SegmentSet(self.bone_segments(include_pins=True))
        ribs = [r for r in self.rods if r["kind"] == "rib"]
        self.rib_s0 = min(r["points"][0, 0] for r in ribs) - 1e-3
        self.rib_s1 = max(r["points"][:, 0].max() for r in ribs) + 1e-3


# ============================================================================== workpiece assembly
def _frame_maps(tr):
    """Anatomical (s, v, w) <-> workpiece (x, y, z)."""
    xc = .5*tr.TL                                          # bbox snout .. caudal tip
    vv = np.concatenate([tr.loft.top.v, tr.loft.bot.v])
    yc = .5*(vv.max() + vv.min())
    z0 = tr.hw_max

    def to_local(p):
        p = np.atleast_2d(p)
        return p[:, 0] + xc, -(p[:, 1] - yc), p[:, 2] - z0

    def to_world(s, v, w):
        return np.column_stack([np.asarray(s) - xc, -np.asarray(v) + yc, np.asarray(w) + z0])

    def vec_to_world(g):
        return np.column_stack([g[:, 0], -g[:, 1], g[:, 2]])
    return to_local, to_world, vec_to_world, (xc, yc, z0)


def _skin_texture(tr, res=(1024, 4096)):
    """Generated rainbow-trout skin, unwrapped: rows = angle theta around the section (0 dorsal, pi/2 upper
    flank, pi ventral, 3pi/2 lower flank), columns = s/TL from the snout (0) to the tissue end (1).
    Palette read off the reference photos: RFEIMG-0305 (silvery farmed fish, dark grey-olive back),
    the Commons market photo Rainbow-trout-in-market.jpg (gutted farmed trout: fine black spots, faint pink band)
    and Rainbow_trout_white_background.jpg (olive back, dense spots, pink-red lateral band, rosy gill cover)."""
    H, W = res
    rng = np.random.default_rng(3000 + tr.seed)
    th = (np.arange(H) + .5)/H*2*np.pi
    u = (np.arange(W) + .5)/W
    S = u*tr.s_end
    TH, SS = np.meshgrid(th, S, indexing="ij")
    # height of each texel on the body (normalised: +1 dorsal edge, -1 ventral edge) and the LL height
    v, w = tr.loft.surface(SS, TH)
    top, bot = tr.loft.top(SS), tr.loft.bot(SS)
    eta = (v - .5*(top + bot))/np.maximum(.5*(top - bot), 1e-6)
    vb = tr.bb(SS)
    eta_ll = (vb - .5*(top + bot))/np.maximum(.5*(top - bot), 1e-6)
    back, flank = srgb("#3d4a3c"), srgb("#8f9a9e")      # dark olive-steel back, silver-grey flank (RFEIMG-0305)
    band, lower, belly = srgb("#c7908c"), srgb("#d9dcda"), srgb("#f1efea")
    col = np.empty((H, W, 3))
    t1 = smoothstep(.15, .72, eta)            # 1 on the back
    t2 = np.exp(-((eta - eta_ll + .02)/.24)**2)  # lateral band (broad, soft, iridescent pink)
    t3 = smoothstep(-.2, -.75, eta)            # 1 on the belly
    for c in range(3):
        base = flank[c]*(1 - t1) + back[c]*t1
        base = base*(1 - .45*t2) + band[c]*.45*t2
        base = base*(1 - t3) + (lower[c]*(1 - smoothstep(-.6, -.9, eta)) + belly[c]*smoothstep(-.6, -.9, eta))*t3
        col[..., c] = base
    # iridescent sheen and mottling
    n1 = value_noise((H, W), 40, rng)
    n2 = value_noise((H, W), 9, rng, octaves=3)
    col *= (.9 + .12*n1[..., None])
    col[..., 2] += .03*(n2 - .5)*(1 - t1)
    # scale pattern: small overlapping scales (rainbow trout scales are tiny, ~1 mm on a 32 cm fish)
    circ_s = .5*np.pi*(tr.loft.top(S) - tr.loft.bot(S) + 2*tr.loft.hw(S))     # section perimeter (approx)
    circ = circ_s[None, :]
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    arc = TH/(2*np.pi)*circ                   # physical arc length around the section (m)
    row = np.floor(arc/1.1e-3)
    fx = ((SS/1.1e-3 + .5*(row % 2)) % 1.0) - .5
    fy = ((arc/1.1e-3) % 1.0)
    rim = np.clip(1 - np.abs(np.hypot(fx, fy - .1) - .45)/.12, 0, 1)
    col *= (1 - .07*rim[..., None]*(1 - .6*t3[..., None]))
    # black spots: dense above the lateral line and on the back, sparse below, none on the belly
    def weight(uu, tt):
        i, j = min(int(tt*H), H - 1), min(int(uu*W), W - 1)
        e = eta[i, j]
        return float(np.clip((e - eta_ll[i, j] + .35)/.9, 0, 1)**1.3)*(1 if uu > .03 else 0)
    # spots: rainbow trout carry small black spots (~1-3 mm on a portion fish) on the back, the upper flank and
    # the head top; rotation and radii are applied in physical millimetres (texels are anisotropic)
    n_sp = int(650*tr.s_end/.277)
    pts = scatter_points(n_sp, rng, weight=weight, min_dist=.0025)
    dark = srgb("#1c1d19")
    for uu, tt in pts:
        r_a = rng.uniform(.45e-3, 1.1e-3)
        r_b = r_a*rng.uniform(.55, .9)
        cs = circ_s[min(int(uu*W), W - 1)]
        mx, my = tr.s_end/W, cs/H                  # metres per texel along / around the body
        cx, cy = uu*W, tt*H
        ext = 1.6*r_a
        x0, x1 = int(max(cx - ext/mx - 2, 0)), int(min(cx + ext/mx + 2, W))
        y0, y1 = int(max(cy - ext/my - 2, 0)), int(min(cy + ext/my + 2, H))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        ang = rng.uniform(0, np.pi)
        dx, dy = (xx - cx)*mx, (yy - cy)*my
        a = (dx*np.cos(ang) + dy*np.sin(ang))/r_a
        b = (-dx*np.sin(ang) + dy*np.cos(ang))/r_b
        m = np.clip((1.15 - np.hypot(a, b))*4, 0, 1)*rng.uniform(.6, .9)
        col[y0:y1, x0:x1] = col[y0:y1, x0:x1]*(1 - m[..., None]) + dark*m[..., None]
    # lateral line pores (thin dark dashes)
    ll = np.abs(eta - eta_ll)*(.5*(top - bot)) < .25e-3
    dash = (np.floor(X/(0.9e-3/tr.s_end*W)) % 2) == 0
    col[ll & dash & (SS > 50*tr.k)] *= .78
    # head: darker crown, rosy gill cover, eye, mouth line
    head = SS < tr.s_hb(v)
    # rosy gill cover: soft ellipse over the operculum (photos: Rainbow_trout_white_background.jpg)
    q = np.hypot((SS - 44*tr.k)/(13*tr.k), (v - (-6)*tr.k)/(16*tr.k))
    op = np.clip(1.2 - q, 0, 1)[..., None]*.55
    col = col*(1 - op) + srgb("#b7766c")*op
    crown = (smoothstep(.25, .6, eta)*(1 - smoothstep(tr.s_hb(v) - 8e-3, tr.s_hb(v) + 4e-3, SS)))[..., None]
    col *= 1 - .15*crown
    e = PROV["eye"]["value"]
    for sgn in (1, -1):
        de = np.hypot(SS - e["s"]*tr.k, v - e["v"]*tr.k)
        side = (w*sgn) > 0
        iris = side & (de < e["r"]*tr.k)
        pupil = side & (de < .55*e["r"]*tr.k)
        col[iris] = srgb("#8f7f55")*(.75 + .25*np.clip(de[iris]/(e["r"]*tr.k), 0, 1))[:, None]
        col[pupil] = srgb("#0d0d0f")
    mouth = (SS < 20*tr.k) & (np.abs(eta + .15 - .05*(SS/(20*tr.k))) < .03)
    col[mouth] *= .55
    return np.clip(col, 0, 1)


def build(seed=0, texture_dir=None, texture_res=(1024, 4096)):
    """Build preset A for a seed (seed 0 = the nominal fish). If texture_dir is given the generated skin
    texture is written there as PNG; otherwise appearance['skin_texture']['make'] generates it on demand."""
    tr = Trout(seed, texture_dir)
    tr.finalize()
    to_local, to_world, vec_to_world, (xc, yc, z0) = _frame_maps(tr)
    k, fL = tr.k, tr.fL

    def W(fn):
        def f(p):
            s, v, w = to_local(p)
            return fn(s, v, w)
        return f

    sdf = W(tr.tissue)
    vs = np.concatenate([tr.loft.top.v, tr.loft.bot.v])
    lo = np.array([float(tr.s_hb(vs.max())) - 2e-3 - xc, -vs.max() + yc - 2e-3, -2e-3])
    hi = np.array([tr.s_end + 2e-3 - xc, -vs.min() + yc + 2e-3, 2*tr.hw_max + 2e-3])

    # ------------------------------------------------ rigid bodies
    # head: integrated from its solid on a 0.6 mm grid
    hlo = np.array([0., float(vs.min()) - 1e-3, -tr.hw_max - 1e-3])
    hhi = np.array([float(tr.s_hb(vs.min())) + 1e-3, float(vs.max()) + 1e-3, tr.hw_max + 1e-3])
    rho_h = PROV["head_density"]["value"]
    vol, m1, m2 = grid_integrate(lambda P: tr.head_solid(P[:, 0], P[:, 1], P[:, 2]) < 0, None, hlo, hhi, .6e-3)
    Mh, ch, Ih = inertia_from_moments(vol[0], m1[0], m2[0], rho_h)
    head_caps = _fit_head_capsules(tr)
    Rw = np.diag([1., -1., 1.])
    rigid = [dict(name="head", mass=Mh, com=to_world(*ch[:, None])[0], inertia=Rw @ Ih @ Rw,
                  capsules=[(to_world(*p0[:, None])[0], to_world(*p1[:, None])[0], r) for p0, p1, r, _ in head_caps],
                  capsule_names=[n for *_, n in head_caps], parent=None, joint=None, bond=True,
                  sdf=W(tr.head_solid), volume=vol[0], density=rho_h,
                  provenance="mass = head-solid volume (loft ahead of the head boundary, gills removed) x "
                             "head bulk density (estimate)")]
    rho_b = PR.BONE_DENSITY.value
    for seg in tr.segments:
        parts = [capsule_mass_props(p0, p1, r, rho_b) for p0, p1, r, _, _ in seg["capsules"]]
        M, c, I = combine_mass_props(parts)
        rigid.append(dict(name=seg["name"], mass=M, com=to_world(*c[:, None])[0], inertia=Rw @ I @ Rw,
                          capsules=[(to_world(*p0[:, None])[0], to_world(*p1[:, None])[0], r)
                                    for p0, p1, r, _, _ in seg["capsules"]],
                          capsule_kinds=[kd for *_, kd, _ in seg["capsules"]],
                          vertebrae=seg["vertebrae"], parent=seg["parent"],
                          joint=dict(anchor=to_world(*seg["joint"]["anchor"][:, None])[0], kind="ball",
                                     gap=seg["joint"]["gap"], radius=seg["joint"]["radius"]),
                          bond=True, provenance="capsule volumes x BONE_DENSITY (overlaps not removed)"))
    rods = []
    for rd in tr.rods:
        r2 = dict(rd)
        r2["points"] = to_world(rd["points"][:, 0], rd["points"][:, 1], rd["points"][:, 2])
        r2["mass"] = rod_volume(rd["points"], rd["radius"])*rho_b
        rods.append(r2)
    fins = []
    for f in tr.fins:
        f2 = dict(f)
        f2["outline"] = to_world(f["outline"][:, 0], f["outline"][:, 1], f["outline"][:, 2])
        f2["attach"] = to_world(f["attach"][:, 0], f["attach"][:, 1], f["attach"][:, 2])
        P = f["outline"]
        # polygon area in its own plane (Newell normal)
        nrm = np.zeros(3)
        for a, b in zip(P, np.roll(P, -1, 0)):
            nrm += np.cross(a, b)
        f2["area"] = .5*float(np.linalg.norm(nrm))
        f2["mass"] = f2["area"]*f["thickness"]*PR.SKIN_DENSITY.value
        fins.append(f2)

    # ------------------------------------------------ fields
    def sheet_normal(p):
        s, v, w = to_local(p)
        n, _ = tr.myo.normal_and_spacing(s, v, w)
        return vec_to_world(n)

    def fibre(p):
        s, v, w = to_local(p)
        slope = tr.bb.d(s)
        g = np.column_stack([np.ones_like(s), slope, np.zeros_like(s)])
        g /= np.linalg.norm(g, axis=1)[:, None]
        return vec_to_world(g)

    tol_skin = .75e-3

    def skin_mask(p, tol=tol_skin):
        s, v, w = to_local(p)
        d_out = tr.outer(s, v, w)
        m = np.abs(d_out) < tol
        m &= tr.cavity(s, v, w, d_out) > tol
        m &= tr.slit(s, v, w) > .5*tol
        m &= tr.head_cut(s, v) < -tol
        m &= tr.channel(s, v, w) > tol
        return m

    interfaces = {
        "median_septum": dict(
            sdf=lambda p: to_local(p)[2],
            extent=lambda p: _median_extent(tr, *to_local(p)),
            params=("FLESH_BONE_G", "FLESH_BONE_SIGMA"),
            note="median plane above the centra (between the neural spines, pterygiophores and supraneurals) "
                 "and below them behind the vent (between the haemal spines)"),
        "rib_cage": dict(
            sdf=lambda p: _rib_cage_sdf(tr, *to_local(p)),
            extent=lambda p: _rib_cage_extent(tr, *to_local(p)),
            params=("FLESH_BONE_G", "FLESH_BONE_SIGMA"),
            note="outer surface of the pleural ribs (rib centre depth + radius below the cavity lining)"),
        "head_joint": dict(
            sdf=lambda p: -tr.head_cut(*to_local(p)[:2]),
            extent=lambda p: (tr.outer(*to_local(p)) < 0),
            params=("FLESH_BONE_G", "FLESH_BONE_SIGMA"),
            note="flesh bonded to the posterior face of the head body (skull, cleithrum); every backbone joint is "
                 "cuttable (JOINT_R), one per vertebra over the first five (anatomy/trout.py N_SINGLE)"),
    }

    def uv(p):
        s, v, w = to_local(p)
        th = tr.loft.theta_of(s, v, w)
        return np.column_stack([np.clip(s/tr.s_end, 0, 1), th/(2*np.pi)])

    tex_path = None
    if texture_dir is not None:
        import os
        os.makedirs(texture_dir, exist_ok=True)
        tex_path = os.path.join(texture_dir, f"trout_skin_seed{seed}.png")
        save_png(tex_path, _skin_texture(tr, texture_res))

    def flesh_color(p):
        s, v, w = to_local(p)
        return _trout_flesh(tr, s, v, w)

    appearance = dict(
        skin_uv=uv,
        skin_texture=dict(path=tex_path, make=lambda res=texture_res: _skin_texture(tr, res),
                          layout="rows: theta/(2 pi) (0 dorsal, .25 upper flank, .5 ventral, .75 lower flank); "
                                 "columns: s/s_end (snout to tissue end)",
                          provenance="generated (authored procedural); palette and spot density read off "
                                     "RFEIMG-0305, Rainbow-trout-in-market.jpg, Rainbow_trout_white_background.jpg"),
        flesh_color=flesh_color,
        flesh_provenance="authored procedural: raw farmed rainbow trout flesh is pale orange-pink (RFEIMG-0306 "
                         "fillets, market photo); dark lateral muscle brown-red; myosepta thin white lines from "
                         "the W-myomere field")

    targets = dict(
        gutted_mass=dict(value=PR.TROUT.value["gutted_kg"], unit="kg", label=PR.TROUT.label, basis=PR.TROUT.basis),
        fillet_pair_fraction=dict(value=PR.TROUT_FILLET_YIELD.value, label=PR.TROUT_FILLET_YIELD.label,
                                  basis=PR.TROUT_FILLET_YIELD.basis),
        head_fraction=dict(value=(.15, .20), label="estimate", basis=""),
        literature_portion_trout=dict(
            source="Souza et al. 2015, Acta Scientiarum Animal Sciences 37: Table 2, 300-370 g farmed trout "
                   "(read from the PDF)",
            eviscerated_of_whole=.8300, head_of_whole=.1327, skin_on_fillet_of_whole=.4529,
            abdominal_muscle_of_whole=.0211, skin_of_whole=.0638,
            head_of_eviscerated=.1327/.83, fillet_of_eviscerated=.4529/.83,
            fillet_plus_abdominal_of_eviscerated=(.4529 + .0211)/.83))

    wp = Workpiece(
        name="trout", sdf=sdf, bounds=(lo, hi), region=W(tr.region), region_names=REGION_NAMES,
        sheet_normal=sheet_normal, fibre=fibre, rigid_bodies=rigid, rods=rods, interfaces=interfaces,
        skin=dict(skin_mask=skin_mask, thickness=PR.SKIN_THICKNESS_TROUT.value, params="SKIN_THICKNESS_TROUT",
                  tolerance=tol_skin,
                  note="skin on the whole outer surface of the tissue; none on the cavity lining, slit faces, "
                       "head joint face or backbone channel"),
        fins=fins, reference=W(tr.reference), reference_names=REFERENCE_NAMES, appearance=appearance,
        targets=targets, measured={}, provenance=PROV, variation=tr.variation,
        extra=dict(model=tr, to_local=lambda p: np.column_stack(to_local(p)), to_world=to_world,
                   outer_sdf=W(tr.outer), cavity_sdf=W(lambda s, v, w: tr.cavity(s, v, w)),
                   slit_sdf=W(tr.slit), head_sdf=W(tr.head_solid), frame_offsets=(xc, yc, z0),
                   densities=dict(tissue=PR.FISH_DENSITY.value, bone=rho_b, head=rho_h,
                                  skin=PR.SKIN_DENSITY.value)))
    wp.measured = measure(wp)
    return wp


def _median_extent(tr, s, v, w):
    inside = tr.tissue(s, v, w) < 0
    vb = tr.bb(s)
    up = v > vb + tr.r_c
    down = (v < vb - tr.r_c) & (s > tr.s_vent)
    return inside & (up | down) & (s < tr.SL)


def _rib_cage_sdf(tr, s, v, w):
    off = PROV["rib_depth"]["value"]*tr.fL + PROV["rib_radius"]["value"][0]*tr.fL
    return tr.cavity(s, v, w) - off


def _rib_cage_extent(tr, s, v, w):
    return (s > tr.rib_s0) & (s < tr.rib_s1) & (v > tr._rib_tip_v(s) - 1e-3) & (v < tr.roof(s) + 1e-3) & \
           (tr.tissue(s, v, w) < 0)


def _fit_head_capsules(tr):
    """A few fat capsules inside the head solid for collision: skull, lower jaw/branchial floor, two gill
    covers and two cleithra. Radii are the largest that keep each capsule 0.3 mm inside the head surface."""
    k = tr.k
    bb = lambda s: float(tr.bb(s))
    top, bot, hw = tr.loft.top, tr.loft.bot, tr.loft.hw
    axes = [
        ("skull", (10*k, .45*float(top(10*k))), (44*k, bb(44*k) + 1*k), 0.),
        ("snout", (2.5*k, -1.5*k), (14*k, 0.), 0.),
        ("lower_jaw", (5*k, float(bot(5*k)) + 3*k), (28*k, float(bot(28*k)) + 5*k), 0.),
    ]
    caps = []

    def fit(p0, p1, name):
        t = np.linspace(0, 1, 41)
        P = p0 + t[:, None]*(p1 - p0)
        d = tr.head_solid(P[:, 0], P[:, 1], P[:, 2])
        r = float(-d.max()) - .3e-3
        if r > .4e-3:
            caps.append((p0, p1, r, name))
    for name, a, b, w0 in axes:
        # a chain of three capsules per axis so each one takes the radius its own stretch allows
        nodes = np.linspace(np.array([a[0], a[1], w0]), np.array([b[0], b[1], w0]), 4)
        for q in range(3):
            fit(nodes[q], nodes[q + 1], f"{name}_{q}")
    hwa = tr.loft.half_width_at
    for side in (1., -1.):
        tag = "upper" if side > 0 else "lower"
        # gill covers: thin plates over the (emptied) gill chamber, two capsules each just under the skin
        for q, (sa, va, sb, vb_) in enumerate([(26, 4, 50, 4), (28, -10, 52, -16)]):
            pa = np.array([sa*k, va*k, 0.])
            pb = np.array([sb*k, vb_*k, 0.])
            pa[2] = side*(float(hwa(np.array([pa[0]]), np.array([pa[1]]))[0]) - 1.3e-3)
            pb[2] = side*(float(hwa(np.array([pb[0]]), np.array([pb[1]]))[0]) - 1.3e-3)
            fit(pa, pb, f"gill_cover_{tag}_{q}")
        # cleithrum: the shoulder-girdle bar behind the gill opening
        pa = np.array([45.5*k, 7*k, 0.])
        pb = np.array([54.5*k, -26*k, 0.])
        pa[2] = side*(float(hwa(np.array([pa[0]]), np.array([pa[1]]))[0]) - 2.2e-3)
        pb[2] = side*(float(hwa(np.array([pb[0]]), np.array([pb[1]]))[0]) - 2.2e-3)
        fit(pa, pb, f"cleithrum_{tag}")
    return caps


def _trout_flesh(tr, s, v, w):
    """sRGB of raw cut flesh at rest positions (anatomical frame)."""
    white = srgb("#eea08a")        # raw pigmented portion-trout white muscle (pale orange-pink)
    dark = srgb("#8c4f45")         # lateral red muscle
    sept = srgb("#f6ece6")         # myosepta (collagen + fat, whitish)
    fat = srgb("#f3d9c9")
    d_out = tr.outer(s, v, w)
    col = np.tile(white, (len(s), 1))
    # slightly lighter toward the belly (fattier), deeper colour in the loin core
    vb = tr.bb(s)
    lt = np.clip((vb - v)/.03, 0, 1)[:, None]
    col = col*(1 - .25*lt) + fat*.25*lt
    dm = tr.dark_band(s, v, w, d_out)
    col[dm] = dark
    # subcutaneous fat line under the skin
    sub = (-d_out < .5e-3)[:, None]
    col = np.where(sub, .5*col + .5*fat, col)
    # myosepta: thin white lines (anti-aliased, ~0.3 mm)
    dist = tr.myo.septum_distance(s, v, w)
    a = np.clip(1 - dist/.18e-3, 0, 1)[:, None]
    col = col*(1 - .8*a) + sept*.8*a
    # horizontal septum (between epaxial and hypaxial) as a faint white line
    hs = np.clip(1 - np.abs(v - vb)/.25e-3, 0, 1)[:, None]
    col = col*(1 - .6*hs) + sept*.6*hs
    return np.clip(col, 0, 1)


# ============================================================================== measured partition
def measure(wp, h=.5e-3):
    """Mass partition from the generator's own geometry (grid integration at h), before meshing."""
    tr = wp.extra["model"]
    rho = PR.FISH_DENSITY.value
    to_local = wp.extra["to_local"]
    lo, hi = wp.bounds

    def inside(P):
        return wp.sdf(P) < 0

    def lab(P):
        return wp.reference(P)*10 + wp.region(P)
    vol, _, _ = grid_integrate(inside, lab, lo, hi, h)
    by_ref = {n: 0. for n in REFERENCE_NAMES}
    by_reg = {n: 0. for n in REGION_NAMES}
    for key, V in vol.items():
        by_ref[REFERENCE_NAMES[key//10]] += V*rho
        by_reg[REGION_NAMES[key % 10]] += V*rho
    tissue_mass = sum(by_ref.values())
    head = wp.rigid_bodies[0]["mass"]
    segs = sum(rb["mass"] for rb in wp.rigid_bodies[1:])
    rho_b = PR.BONE_DENSITY.value
    # rods: embedded in the tissue volume -> add only their excess density
    rod_excess = {kd: 0. for kd in ("rib", "pin", "fin_support", "supraneural")}
    for rd in wp.rods:
        rod_excess[rd["kind"]] += rod_volume(rd["points"], rd["radius"])*(rho_b - rho)
    fins = sum(f["mass"] for f in wp.fins)
    total = tissue_mass + head + segs + sum(rod_excess.values()) + fins
    # skin area on the fillets (for the skin mass inside the fillet)
    fillet_pair = by_ref["fillet_upper"] + by_ref["fillet_lower"] + rod_excess["pin"]
    head_product = head + by_ref["head"]
    frame = segs + by_ref["frame"] + rod_excess["rib"] + rod_excess["fin_support"] + rod_excess["supraneural"] + fins
    out = dict(
        grid_h=h, total_mass=total, tissue_mass=tissue_mass, head_rigid_mass=head, backbone_mass=segs,
        rod_excess_mass=rod_excess, fins_mass=fins,
        reference_tissue_mass=by_ref, region_mass=by_reg,
        fillet_pair_mass=fillet_pair, fillet_pair_fraction=fillet_pair/total,
        fillet_pair_with_belly_edge_fraction=(fillet_pair + by_ref["belly_edge"])/total,
        head_product_mass=head_product, head_product_fraction=head_product/total,
        head_rigid_fraction=head/total, collar_mass=by_ref["head"],
        frame_product_mass=frame, frame_fraction=frame/total,
        belly_edge_mass=by_ref["belly_edge"], belly_edge_fraction=by_ref["belly_edge"]/total,
        cavity_volume=_cavity_volume(tr),
        note="masses in kg; 'head_product' = head rigid body + collar tissue in front of the collar line")
    return out


def _cavity_volume(tr, h=.6e-3):
    lo = np.array([0., float(tr.loft.bot.v.min()), -tr.hw_max])
    hi = np.array([tr.s_vent + 3e-3, float(tr.loft.top.v.max()), tr.hw_max])

    def inside(P):
        s, v, w = P[:, 0], P[:, 1], P[:, 2]
        return (tr.outer(s, v, w) < 0) & (tr.cavity(s, v, w) < 0) & (s > tr.s_hb(v))
    vol, _, _ = grid_integrate(inside, None, lo, hi, h)
    return vol.get(0, 0.)


# ------------------------------------------------------------------------------------------ task A4's workpiece
FILLET_REFERENCE_NAMES = ["fillet", "rib_strip"]
FILLET_TRIM_M = 3e-3             # A4 fillet: edges thinner than this above the backbone cut are trimmed (authored)


def build_fillet_rib_strip(seed=0, texture_dir=None, texture_res=(1024, 4096)):
    """Task A4's workpiece ('a skin-on trout fillet with its rib strip, skin-down'): the upper-side
    fillet a cook takes by cutting flat along the backbone through the rib roots, so the upper ribs stay in its belly
    flap. Derived from the whole trout of the same seed: the tissue above a plane 0.3 mm over the centra
    (w >= centrum radius + 0.3 mm), trimmed where the fish rises less than 3 mm above that plane (the thin fin-base
    edges), turned 180 deg about the fish's long axis and lifted so the skin rests at z = 0.
    The upper ribs stay bonded to the flesh along the rib-cage interface (their root ends, below the plane, are cut
    off); no skeleton; pin bones, supraneurals and fin supports reach the plane and are dropped; fins are trimmed.
    Reference: 'rib_strip' = the whole trout's 'frame' label on this side (the thin layer around the ribs), 'fillet'
    = the rest."""
    base = build(seed, texture_dir, texture_res)
    tr = base.extra["model"]
    xc, yc, z0 = base.extra["frame_offsets"]
    w_cut = float(tr.r_c) + .3e-3
    z_cut = z0 + w_cut
    z_top = z0 + float(tr.hw_max) + 1e-3

    def unflip(P):
        P = np.atleast_2d(np.asarray(P, float))
        return np.column_stack([P[:, 0], -P[:, 1], z_top - P[:, 2]])

    def flip(P):
        return unflip(P)                                   # the map is its own inverse

    def flipvec(V):
        V = np.atleast_2d(np.asarray(V, float))
        return np.column_stack([V[:, 0], -V[:, 1], -V[:, 2]])

    outer = base.extra["outer_sdf"]
    trim = FILLET_TRIM_M

    def sdf(P):
        Q = unflip(P)
        d = np.maximum(np.asarray(base.sdf(Q)), z_cut - Q[:, 2])
        # a trimmed fillet: keep only where the fish rises at least `trim` above the cut plane (the thin fin-base
        # edges, thinner than one tet, are trimmed off as a cook trims them; without this the plane met them at a
        # shallow angle and the mesh edge came out saw-toothed)
        Qt = Q.copy()
        Qt[:, 2] = z_cut + trim
        return np.maximum(d, np.asarray(outer(Qt)))

    lo0, hi0 = (np.asarray(b, float) for b in base.bounds)
    lo = np.array([lo0[0], -hi0[1], 0.]) - 1e-3
    hi = np.array([hi0[0], -lo0[1], z_top - z_cut]) + 1e-3
    ref0 = base.reference_names.index("frame")
    rods = []
    for rd in base.rods:
        if rd["kind"] != "rib" or rd.get("side") != "upper":
            continue
        pts = np.asarray(rd["points"], float)
        keep = pts[:, 2] >= z_cut + .2e-3
        if keep.sum() < 2:
            continue
        rods.append(dict(rd, points=flip(pts[keep]), radius=np.asarray(rd["radius"], float)[keep], root=None,
                         name=rd["name"] + "_fillet"))
    rib = base.interfaces["rib_cage"]
    interfaces = {"rib_cage": dict(rib, sdf=lambda P: rib["sdf"](unflip(P)),
                                   extent=lambda P: np.asarray(rib["extent"](unflip(P)), bool) & (unflip(P)[:, 2] > z_cut),
                                   note="outer surface of the upper ribs (whole trout's rib cage interface, flipped)")}
    skin = dict(base.skin, skin_mask=lambda P: np.asarray(base.skin["skin_mask"](unflip(P)), bool)
                & (unflip(P)[:, 2] > z_cut + .5e-3),
                note="the upper side's skin (now underneath); none on the backbone cut face or the rib strip")
    appearance = dict(base.appearance, skin_uv=lambda P: base.appearance["skin_uv"](unflip(P)))
    wp = Workpiece(
        name="trout_fillet_rib_strip", sdf=sdf, bounds=(lo, hi), region=lambda P: base.region(unflip(P)),
        region_names=base.region_names, sheet_normal=lambda P: flipvec(base.sheet_normal(unflip(P))),
        fibre=lambda P: flipvec(base.fibre(unflip(P))), rigid_bodies=[], rods=rods, interfaces=interfaces, skin=skin,
        fins=[], reference=lambda P: (np.asarray(base.reference(unflip(P))) == ref0).astype(int),
        reference_names=FILLET_REFERENCE_NAMES, appearance=appearance,
        targets=dict(base.targets, note="whole-trout targets; A4 criteria are relative to this fillet's own mass"),
        measured={}, provenance=base.provenance, variation=dict(base.variation, cut_plane_w_mm=w_cut*1e3),
        extra=dict(base.extra, to_local=lambda P: base.extra["to_local"](unflip(P)), flipped=True,
                   z_cut_before_flip=z_cut, z_top=z_top, base_reference_names=base.reference_names))
    return wp
