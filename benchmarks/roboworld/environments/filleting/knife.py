"""Knife geometry shared by the tissue solver (wedge colliders + edge samples) and the rigid world.

Blade local frame (per blade segment and for the rigid blade body):
  x along the blade from heel (x=0) to tip (x=L), y through the thickness, z from the cutting edge
  (z = 0) to the spine (z = H). The cutting direction n points from the spine to the edge: -z.
Half-thickness profile: flat grind from the spine to a 0.25 mm shoulder, then the 18 deg bevel
(Victorinox 5.3703.18: 1.2 mm spine, 25 mm height, 18 deg per side).
"""

import math
import numpy as np
from . import params as PR
from .tissue import WEDGE

EDGE_HALF_SHOULDER = .125e-3       # m, half thickness where the bevel meets the flat grind [estimate]
SAMPLE_SPACING = 1e-3              # m, edge law sample spacing


class BladeSpec:
    def __init__(self, kind="fillet", segments=5, sharp=True):
        d = PR.v(PR.FILLET_BLADE if kind == "fillet" else PR.BONING_BLADE)
        self.kind = kind
        self.L, self.H, self.t = d["length"], d["height"], d["spine"]
        self.mass = d["mass"]
        self.bevel = math.radians(d["bevel_deg"])
        self.segments = int(segments)
        self.bsi = PR.v(PR.BSI_HONED) if sharp else PR.v(PR.BSI_WORN)
        self.seg_len = self.L/self.segments
        # flat-wise EI at the heel between the full-wedge and rectangle bounds; the spine tapers to
        # ~50 % at the tip (fillet knives are distal-tapered) [derived + estimate]
        E = PR.v(PR.STEEL_E)
        I_rect = self.H*self.t**3/12.
        I_wedge = self.t**3*self.H/48.
        self.EI_heel = E*math.sqrt(I_rect*I_wedge)     # geometric mean of the bounds
        self.taper_tip = .5 if kind == "fillet" else .8

    def spine_at(self, xi):
        """Spine thickness at fraction xi in [0,1] from heel to tip."""
        return self.t*(1. - (1. - self.taper_tip)*xi)

    def segment_shapes(self, mu_tissue, mu_membrane, mu_rod):
        """Wedge collider parameters for each blade segment (local origin at the segment's edge centre)."""
        out = []
        for s in range(self.segments):
            xi = (s + .5)/self.segments
            tt = self.spine_at(xi)
            out.append(dict(type=WEDGE, half=(self.seg_len/2, tt/2, self.H/2),
                            param=(math.tan(self.bevel), EDGE_HALF_SHOULDER),
                            mu=(mu_tissue, mu_membrane, mu_rod), com_local=(0., 0., self.H/2),
                            name=f"blade_{s}"))
        return out

    def samples_per_segment(self):
        return max(2, int(round(self.seg_len/SAMPLE_SPACING)))

    def edge_layout(self, shape0):
        """Edge samples as points of the blade segment shapes (shape index shape0 + segment): returns dict of
        shape ids, local points (on the edge, z = 0), edge length per sample and blade thickness there."""
        m = self.samples_per_segment()
        xs = (np.arange(m) + .5)/m*self.seg_len - self.seg_len/2
        shape, loc, ell, thick = [], [], [], []
        for s in range(self.segments):
            for x in xs:
                shape.append(shape0 + s)
                loc.append((x, 0., 0.))
                ell.append(self.seg_len/m)
                thick.append(self.spine_at((s + .5)/self.segments))
        return dict(shape=np.array(shape, np.int32), loc=np.array(loc), ell=np.array(ell), thick=np.array(thick))

    def edge_samples(self, seg_poses):
        """World positions, tangents and cutting directions of the edge samples, given each segment's
        pose as (R 3x3, p 3) of its local frame (origin at its edge centre). Returns dict of arrays."""
        m = self.samples_per_segment()
        xs = (np.arange(m) + .5)/m*self.seg_len - self.seg_len/2
        pos, tan, nor, seg, ell, thick = [], [], [], [], [], []
        for s, (R, p) in enumerate(seg_poses):
            for x in xs:
                pos.append(p + R[:, 0]*x)
                tan.append(R[:, 0])
                nor.append(-R[:, 2])
                seg.append(s)
                ell.append(self.seg_len/m)
                thick.append(self.spine_at((s + .5)/self.segments))
        return dict(pos=np.array(pos), t=np.array(tan), n=np.array(nor), shape=np.array(seg, np.int32),
                    ell=np.array(ell), thick=np.array(thick))
