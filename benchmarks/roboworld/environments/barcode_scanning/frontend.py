"""Sub-pixel, model-based 1-D barcode front end (runs after decoder.read_all's six zxing passes fail).

A DS2208-class imager reads 1-D symbols at about 0.7-1.0 pixels per module through an optical blur of
sigma ~0.3-0.7 module; zxing-cpp alone needs ~1.2-1.5 ppm.  This module closes that
gap the way commercial decoders do, without ever emitting text on its own:

1. Regions: structure-tensor energy x coherence finds oriented bar fields (fields of one symbol split at wide,
   blurred spaces are merged); the bar extent along the bars is measured from the image.
2. Profile: the bar-normal angle is refined by minimising the within-bin scatter; every pixel of the band is
   projected onto the bar normal and binned at 0.25 px (0.5 px for wide fields), so a small rotation yields a
   super-resolved profile (the slanted-edge principle).  Per-pixel noise variance var = a + b*DN is measured
   from the within-bin scatter of locally flat bins, independent of any symbol model.
3. Forward model: level(u) = sum of element edges, each edge response = unit-width pixel box convolved with a
   Gaussian PSF of sigma s (px); edges at u(x) = u0 + q x + c x^2 (x in modules), ink spread d, wide:narrow
   ratio R, gain/offset A, B and a linear illumination slope.  Samples are bin means, so each sample's PSF
   gets the bin's measured positional variance added.
4. Hypotheses: candidate symbol extents are dark clusters whose outside white runs are the widest (quiet zone
   test); their ends are converted to bar edges with the modelled crossing offset of each symbology's end
   elements.  Every direction x extent x symbology x length (x R for two-width codes) with 0.5-3.4 ppm is
   scored by a DP over the first and last two characters; model-free bounds prune lengths (a flat run is one
   element; every significant dip needs its own bar).
5. Decoding, in scan order: a Viterbi over *characters* of the full symbology tables (EAN/UPC L/G/R digits,
   all 103 Code 128 values + 3 starts, 43 Code 39 characters, 20 Codabar characters, all 100 ITF digit
   pairs) alternates with Levenberg-Marquardt on the continuous parameters.  The model is linear in the space
   indicator, so adjacent characters have exact least-squares pair costs.  Then a model-based bar-angle
   correction (shear fit on strips), a fresh profile, and a final Viterbi carrying the mandatory checksums
   (EAN-13/UPC-A, EAN-8, UPC-E, Code 128) in its state; min-marginals give the best *different* sequence.
6. The reconstructed element sequence is rendered as a clean image (3 px/module, quiet zones) and handed to
   zxing-cpp, which does the symbology decode and validation.  Only zxing-validated texts are candidates.
7. No-ghost gate: accepted only if the refitted model's chi-square against the measured noise (plus a 1 %
   of contrast model floor) is inside the chi-square bound with false-rejection probability ALPHA, the fitted
   blur is < 1 module, and the standard quiet zones are observed inside the frame.  The competitor sequence
   is refitted and gated the same way; if it also passes with a different text -- or two hypotheses at one
   location give different gated texts -- that location yields nothing.

read_1d(frame, formats) -> list of dict(format, text, quad_px, passes='subpixel...').
"""

import time

import numpy as np
from scipy import ndimage
from scipy.special import ndtr
from scipy.stats import chi2 as _chi2

# ----------------------------------------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------------------------------------
ALPHA = 1e-4            # gate: probability that a correct, correctly-modelled read is rejected by the chi2 bound
MODEL_ERR = 0.01        # model-error floor added to the noise, as a fraction of the fitted contrast B
QUIET = {"EAN13": 7.0, "EAN8": 7.0, "UPCE": 7.0}  # quiet zone (modules) required and fitted on each side
QUIET_DEFAULT = 10.0    # Code 128, Code 39, Codabar, ITF (ISO/IEC 15417, 16388, 16390, Codabar spec: 10X)
S_MOD_MAX = 1.0         # hypotheses needing an optical blur sigma >= 1 module are refused (modulation < 1 %)
PPM_RANGE = (0.5, 3.4)   # pixels per module considered
BLUR_INIT = 0.45        # initial PSF sigma guess, modules
MAX_REGIONS = 4
TOPK = 12               # hypotheses refined per profile (in scan order)
EARLY_EXIT = 80.0       # abandon a hypothesis whose reduced chi-square after an EM round exceeds this

_SQ2PI = 1.0 / np.sqrt(2.0 * np.pi)


# ----------------------------------------------------------------------------------------------------------
# symbology tables (full tables; elements are (narrow_units, wide_units); colour 0 = bar, 1 = space)
# ----------------------------------------------------------------------------------------------------------
_EAN_L = ["0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111", "0001011"]
_EAN_R = ["".join("1" if c == "0" else "0" for c in s) for s in _EAN_L]
_EAN_G = [s[::-1] for s in _EAN_R]
_EAN13_FIRST = ["LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG", "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL"]
_UPCE_PAR0 = ["EEEOOO", "EEOEOO", "EEOOEO", "EEOOOE", "EOEEOO", "EOOEEO", "EOOOEE", "EOEOEO", "EOEOOE", "EOOEOE"]
_UPCE_W = [[1, 3, 3, 1, 3, 1], [1, 3, 1, 1, 3, 0], [1, 3, 1, 3, 3, 0], [1, 3, 1, 3, 1, 3]]

_C128 = ("212222 222122 222221 121223 121322 131222 122213 122312 132212 221213 221312 231212 112232 122132 "
         "122231 113222 123122 123221 223211 221132 221231 213212 223112 312131 311222 321122 321221 312212 "
         "322112 322211 212123 212321 232121 111323 131123 131321 112313 132113 132311 211313 231113 231311 "
         "112133 112331 132131 113123 113321 133121 313121 211331 231131 213113 213311 213131 311123 311321 "
         "331121 312113 312311 332111 314111 221411 431111 111224 111422 121124 121421 141122 141221 112214 "
         "112412 122114 122411 142112 142211 241211 221114 413111 241112 134111 111242 121142 121241 114212 "
         "124112 124211 411212 421112 421211 212141 214121 412121 111143 111341 131141 114113 114311 411113 "
         "411311 113141 114131 311141 411131 211412 211214 211232").split()
_C128_STOP = "2331112"
_C39_ALPHA = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%"
_C39_ENC = [0x034, 0x121, 0x061, 0x160, 0x031, 0x130, 0x070, 0x025, 0x124, 0x064, 0x109, 0x049, 0x148, 0x019,
            0x118, 0x058, 0x00D, 0x10C, 0x04C, 0x01C, 0x103, 0x043, 0x142, 0x013, 0x112, 0x052, 0x007, 0x106,
            0x046, 0x016, 0x181, 0x0C1, 0x1C0, 0x091, 0x190, 0x0D0, 0x085, 0x184, 0x0C4, 0x0A8, 0x0A2, 0x08A,
            0x02A]
_C39_STAR = 0x094
_CBAR_ALPHA = "0123456789-$:/.+ABCD"
_CBAR_ENC = [0x003, 0x006, 0x009, 0x060, 0x012, 0x042, 0x021, 0x024, 0x030, 0x048, 0x00C, 0x018, 0x045, 0x051,
             0x054, 0x015, 0x01A, 0x029, 0x00B, 0x00E]
_ITF_D = ["00110", "10001", "01001", "11000", "00101", "10100", "01100", "00011", "10010", "01010"]


def _bits_elements(bits):
    """'0'/'1' module string (1 = bar) -> [(na, nb)], first colour."""
    out, cur, n = [], bits[0], 0
    for ch in bits:
        if ch == cur:
            n += 1
        else:
            out.append((n, 0)); cur, n = ch, 1
    out.append((n, 0))
    return out, (0 if bits[0] == "1" else 1)


def _width_elements(ws):
    return [(int(w), 0) for w in ws]


def _nw_elements(code, k):
    """k-element narrow/wide pattern from an integer bit mask (MSB first, 1 = wide)."""
    return [((0, 1) if (code >> (k - 1 - i)) & 1 else (1, 0)) for i in range(k)]


class SlotType:
    """A set of alternative element patterns occupying one character position."""

    def __init__(self, name, pats, first, values=None, cls=None):
        self.name, self.first = name, first
        self.pats = pats
        self.C = len(pats)
        self.values = list(range(self.C)) if values is None else list(values)
        self.width = np.array([[sum(e[0] for e in p), sum(e[1] for e in p)] for p in pats], float)  # (C, 2)
        self.cls = np.zeros(self.C, int) if cls is None else np.asarray(cls, int)
        keys, rows = {}, []
        for c, els in enumerate(pats):
            a = b = 0
            col = first
            for (na, nb) in els:
                if col == 1:
                    k1 = keys.setdefault((a, b, 0), len(keys))
                    k2 = keys.setdefault((a + na, b + nb, 1), len(keys))
                    rows.append((c, k1, 1.0)); rows.append((c, k2, -1.0))
                a += na; b += nb; col ^= 1
        self.bnd = np.array(list(keys), float).reshape(-1, 3)  # (P, 3): a, b, kind (0 space start, 1 space end)
        self.inc = np.zeros((self.C, len(keys)))
        for c, k, v in rows:
            self.inc[c, k] += v


def _mk_tables():
    T = {}
    T["guard3"] = SlotType("guard3", [_bits_elements("101")[0]], 0)
    T["guard5"] = SlotType("guard5", [_bits_elements("01010")[0]], 1)
    T["guard6"] = SlotType("guard6", [_bits_elements("010101")[0]], 1)
    lg = [_bits_elements(s)[0] for s in _EAN_L] + [_bits_elements(s)[0] for s in _EAN_G]
    T["eanLG"] = SlotType("eanLG", lg, 1, values=list(range(20)))
    T["eanL"] = SlotType("eanL", lg[:10], 1)
    T["eanR"] = SlotType("eanR", [_bits_elements(s)[0] for s in _EAN_R], 0)
    T["c128start"] = SlotType("c128start", [_width_elements(p) for p in _C128[103:106]], 0, values=[103, 104, 105])
    T["c128data"] = SlotType("c128data", [_width_elements(p) for p in _C128[:103]], 0)
    T["c128stop"] = SlotType("c128stop", [_width_elements(_C128_STOP)], 0)
    gap = [(1, 0)]
    T["c39star_gap"] = SlotType("c39star_gap", [_nw_elements(_C39_STAR, 9) + gap], 0)
    T["c39star"] = SlotType("c39star", [_nw_elements(_C39_STAR, 9)], 0)
    T["c39data"] = SlotType("c39data", [_nw_elements(e, 9) + gap for e in _C39_ENC], 0)
    T["cbar_ss_gap"] = SlotType("cbar_ss_gap", [_nw_elements(e, 7) + gap for e in _CBAR_ENC[16:]], 0, values=range(16, 20))
    T["cbar_ss"] = SlotType("cbar_ss", [_nw_elements(e, 7) for e in _CBAR_ENC[16:]], 0, values=range(16, 20))
    T["cbar_data"] = SlotType("cbar_data", [_nw_elements(e, 7) + gap for e in _CBAR_ENC[:16]], 0,
                              cls=[1 if bin(e).count("1") == 3 else 0 for e in _CBAR_ENC[:16]])
    T["itf_start"] = SlotType("itf_start", [[(1, 0)] * 4], 0)
    T["itf_stop"] = SlotType("itf_stop", [[(0, 1), (1, 0), (1, 0)]], 0)
    pairs = []
    for d1 in range(10):
        for d2 in range(10):
            els = []
            for i in range(5):
                els.append((0, 1) if _ITF_D[d1][i] == "1" else (1, 0))
                els.append((0, 1) if _ITF_D[d2][i] == "1" else (1, 0))
            pairs.append(els)
    T["itf_pair"] = SlotType("itf_pair", pairs, 0)
    return T


TABLES = _mk_tables()


class Hyp:
    """One symbol hypothesis: family, length, wide ratio and slot layout.

    slots: list of (SlotType, start_ab(t) -> (a, b), variable?, tset) where tset is the list of Codabar
    position states (number of preceding 3-wide data characters) allowed at that slot."""

    def __init__(self, family, n=0, R=2.5, T=0):
        self.family, self.n, self.R, self.T = family, n, float(R), T
        self.two_width = family in ("Code39", "Codabar", "ITF")
        S = TABLES
        sl = []
        if family == "EAN13":
            sl.append((S["guard3"], (0, 0)))
            for k in range(6):
                sl.append((S["eanLG"], (3 + 7 * k, 0)))
            sl.append((S["guard5"], (45, 0)))
            for k in range(6):
                sl.append((S["eanR"], (50 + 7 * k, 0)))
            sl.append((S["guard3"], (92, 0)))
            self.N = (95, 0)
        elif family == "EAN8":
            sl.append((S["guard3"], (0, 0)))
            for k in range(4):
                sl.append((S["eanL"], (3 + 7 * k, 0)))
            sl.append((S["guard5"], (31, 0)))
            for k in range(4):
                sl.append((S["eanR"], (36 + 7 * k, 0)))
            sl.append((S["guard3"], (64, 0)))
            self.N = (67, 0)
        elif family == "UPCE":
            sl.append((S["guard3"], (0, 0)))
            for k in range(6):
                sl.append((S["eanLG"], (3 + 7 * k, 0)))
            sl.append((S["guard6"], (45, 0)))
            self.N = (51, 0)
        elif family == "Code128":
            sl.append((S["c128start"], (0, 0)))
            for k in range(n + 1):  # n data + check
                sl.append((S["c128data"], (11 * (k + 1), 0)))
            sl.append((S["c128stop"], (11 * (n + 2), 0)))
            self.N = (11 * n + 35, 0)
        elif family == "Code39":
            sl.append((S["c39star_gap"], (0, 0)))
            for k in range(n):
                sl.append((S["c39data"], (7 * (k + 1), 3 * (k + 1))))
            sl.append((S["c39star"], (7 * (n + 1), 3 * (n + 1))))
            self.N = (7 * n + 13, 3 * n + 6)
        elif family == "ITF":
            sl.append((S["itf_start"], (0, 0)))
            for k in range(n):
                sl.append((S["itf_pair"], (4 + 6 * k, 4 * k)))
            sl.append((S["itf_stop"], (4 + 6 * n, 4 * n)))
            self.N = (6 * n + 6, 4 * n + 1)
        elif family == "Codabar":
            sl.append((S["cbar_ss_gap"], (0, 0)))
            for k in range(n):
                sl.append((S["cbar_data"], (5 + 6 * k, 3 + 2 * k)))
            sl.append((S["cbar_ss"], (5 + 6 * n, 3 + 2 * n)))
            self.N = (9 + 6 * n - T, 6 + 2 * n + T)
        else:
            raise ValueError(family)
        self.slots = []
        for j, (st, ab) in enumerate(sl):
            if family == "Codabar":
                if j == 0:
                    ts = [0]
                elif j == len(sl) - 1:
                    ts = [T]
                else:
                    k = j - 1
                    ts = list(range(max(0, T - (n - k)), min(k, T) + 1))
            else:
                ts = [0]
            self.slots.append(dict(st=st, ab=ab, var=st.C > 1, ts=ts))
        self.var_idx = [j for j, s in enumerate(self.slots) if s["var"]]

    def xN(self, R=None):
        R = self.R if R is None else R
        return self.N[0] + self.N[1] * R

    def start_x(self, j, t, R):
        a, b = self.slots[j]["ab"]
        return (a - t) + (b + t) * R  # Codabar: each preceding 3-wide data char shifts by (-1, +1)

    def key(self):
        return (self.family, self.n, self.T, round(self.R, 2))


# ----------------------------------------------------------------------------------------------------------
# edge response of (unit pixel box) x (Gaussian sigma s), all in px
# ----------------------------------------------------------------------------------------------------------
def _psi_exact(z):
    return z * ndtr(z) + _SQ2PI * np.exp(-0.5 * z * z)


# Psi(z) = z Phi(z) + phi(z) tabulated for speed (linear interpolation error < 2e-7 on the unit-step scale)
_PSI_Z0, _PSI_DZ = -9.0, 0.002
_PSI_T = _psi_exact(np.arange(_PSI_Z0, 9.0 + 2 * _PSI_DZ, _PSI_DZ))


def _psi(z):
    t = (np.clip(z, _PSI_Z0, 9.0) - _PSI_Z0) * (1.0 / _PSI_DZ)
    i = t.astype(np.intp)
    f = t - i
    v = _PSI_T[i] + f * (_PSI_T[i + 1] - _PSI_T[i])
    return np.where(z > 9.0, z, v)


def edge_resp(x, s):
    a = (x + 0.5) / s; b = (x - 0.5) / s
    return s * (_psi(a) - _psi(b))


def _edge_all(x, s):
    """E, dE/dx, dE/ds."""
    a = (x + 0.5) / s; b = (x - 0.5) / s
    Pa, Pb = ndtr(a), ndtr(b)
    pa, pb = _SQ2PI * np.exp(-0.5 * a * a), _SQ2PI * np.exp(-0.5 * b * b)
    E = s * (a * Pa + pa - b * Pb - pb)
    return E, Pa - Pb, pa - pb


# ----------------------------------------------------------------------------------------------------------
# profile extraction
# ----------------------------------------------------------------------------------------------------------
class Profile:
    pass


def _structure(img):
    gx = ndimage.sobel(img, 1)
    gy = ndimage.sobel(img, 0)
    return gx * gx, gy * gy, gx * gy


def find_regions(img, max_regions=MAX_REGIONS):
    """Oriented bar fields: (centre, unit normal across the bars, u-extent, v-extent)."""
    f = img.astype(np.float32)
    Jxx, Jyy, Jxy = _structure(f)
    k = 9
    Jxx = ndimage.uniform_filter(Jxx, k); Jyy = ndimage.uniform_filter(Jyy, k); Jxy = ndimage.uniform_filter(Jxy, k)
    E = Jxx + Jyy
    D = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    coh = D / (E + 1e-6)
    noiseE = float(np.median(E)) + 1e-3
    mask = (coh > 0.75) & (E > 25 * noiseE) & (E > 2000.0)
    mask = ndimage.binary_opening(mask, np.ones((3, 3), bool))
    lab, nlab = ndimage.label(mask)
    if nlab == 0:
        return []
    objs = ndimage.find_objects(lab)
    regs = []
    for i, sl in enumerate(objs):
        sub = lab[sl] == i + 1
        area = int(sub.sum())
        if area < 120:
            continue
        ys, xs = np.nonzero(sub)
        ys = ys + sl[0].start; xs = xs + sl[1].start
        jxx, jyy, jxy = Jxx[ys, xs].sum(), Jyy[ys, xs].sum(), Jxy[ys, xs].sum()
        th = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
        n = np.array([np.cos(th), np.sin(th)])
        t = np.array([-n[1], n[0]])
        cx, cy = xs.mean(), ys.mean()
        u = (xs - cx) * n[0] + (ys - cy) * n[1]
        v = (xs - cx) * t[0] + (ys - cy) * t[1]
        u0, u1 = np.percentile(u, [0.5, 99.5]); v0, v1 = np.percentile(v, [1, 99])
        if u1 - u0 < 14 or v1 - v0 < 7:
            continue
        cohm = float(D[ys, xs].sum() / (E[ys, xs].sum() + 1e-6))
        c = np.array([cx, cy]) + n * (u0 + u1) / 2 + t * (v0 + v1) / 2
        regs.append(dict(c=c, n=n, t=t, uh=(u1 - u0) / 2, vh=(v1 - v0) / 2, score=area * cohm, coh=cohm))
    regs = _merge_regions(regs)
    regs.sort(key=lambda r: -r["score"])
    return regs[:max_regions]


def _merge_regions(regs):
    """Merge bar fields of one symbol that the energy mask split at wide (blurred) spaces: same bar angle
    (< 2 deg), bands aligned along the bars, gap along the bar normal < 1/4 of the bar length."""
    regs = list(regs)
    changed = True
    while changed:
        changed = False
        for i in range(len(regs)):
            for j in range(i + 1, len(regs)):
                a, b = regs[i], regs[j]
                if abs(float(np.dot(a["n"], b["n"]))) < np.cos(np.radians(2.0)):
                    continue
                n, t = a["n"], a["t"]
                d = b["c"] - a["c"]
                du, dv = float(np.dot(d, n)), float(np.dot(d, t))
                vh = max(a["vh"], b["vh"])
                if abs(dv) > 0.3 * vh or min(a["vh"], b["vh"]) < 0.6 * vh:
                    continue
                gap = abs(du) - a["uh"] - b["uh"]
                if gap > 0.5 * vh:
                    continue
                lo = min(-a["uh"], du - b["uh"]); hi = max(a["uh"], du + b["uh"])
                c = a["c"] + n * (lo + hi) / 2 + t * dv / 2
                regs[i] = dict(c=c, n=a["n"], t=a["t"], uh=(hi - lo) / 2, vh=vh, score=a["score"] + b["score"],
                               coh=max(a["coh"], b["coh"]))
                del regs[j]
                changed = True
                break
            if changed:
                break
    return regs


def _band_pixels(img, c, n, t, u_lo, u_hi, v_lo, v_hi):
    H, W = img.shape
    corners = [c + n * uu + t * vv for uu in (u_lo, u_hi) for vv in (v_lo, v_hi)]
    xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
    x0, x1 = max(0, int(np.floor(min(xs)))), min(W - 1, int(np.ceil(max(xs))))
    y0, y1 = max(0, int(np.floor(min(ys)))), min(H - 1, int(np.ceil(max(ys))))
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    dx = xx - c[0]; dy = yy - c[1]
    u = dx * n[0] + dy * n[1]
    v = dx * t[0] + dy * t[1]
    m = (u >= u_lo) & (u <= u_hi) & (v >= v_lo) & (v <= v_hi)
    return u[m].astype(float), v[m].astype(float), img[y0:y1 + 1, x0:x1 + 1][m].astype(float)


def _orient_resid(dx, dy, val, th, bw):
    """Mean squared scatter of pixels about the binned profile along direction th: minimal when the
    projection direction is exactly normal to the bars (all pixels at one bar-normal coordinate agree)."""
    u = dx * np.cos(th) + dy * np.sin(th)
    idx = np.floor((u - u.min()) / bw).astype(int)
    cnt = np.bincount(idx)
    ok = cnt > 0
    um = np.bincount(idx, weights=u)[ok] / cnt[ok]
    pm = np.bincount(idx, weights=val)[ok] / cnt[ok]
    r = val - np.interp(u, um, pm)
    return float(np.mean(r * r))


def _refine_angle(dx, dy, val, th0):
    best = th0
    for bw, step, k in ((0.5, np.radians(0.15), 5), (0.25, np.radians(0.04), 3)):
        ths = best + step * np.arange(-k, k + 1)
        sc = np.array([_orient_resid(dx, dy, val, th, bw) for th in ths])
        i = int(np.argmin(sc))
        best = ths[i]
        if 0 < i < len(ths) - 1:
            d = sc[i - 1] - 2 * sc[i] + sc[i + 1]
            if d > 0:
                best = ths[i] + np.clip(0.5 * (sc[i - 1] - sc[i + 1]) / d, -1, 1) * step
    return best


def _bar_vrange(img, reg):
    """Extent of the bars along the bar direction, measured from the image: rows (1 px slices along v) whose
    values vary strongly along u belong to the bars.  The structure-tensor mask can extend beyond the bar ends
    (label edges, text next to the symbol), and rows past the bar ends would corrupt the profile."""
    c, n, t = reg["c"], reg["n"], reg["t"]
    uh, vh = reg["uh"], reg["vh"]
    bp = _band_pixels(img, c, n, t, -0.8 * uh, 0.8 * uh, -vh - 4.0, vh + 4.0)
    trim = min(6.0, 0.25 * vh)
    fallback = (-vh + trim, vh - trim)
    if bp is None or bp[0].size < 50:
        return fallback
    u, v, val = bp
    k = np.floor(v - v.min()).astype(int)
    cnt = np.bincount(k).astype(float)
    ok = cnt >= 5
    m1 = np.bincount(k, weights=val); m2 = np.bincount(k, weights=val * val)
    sd = np.sqrt(np.maximum(m2 / np.maximum(cnt, 1) - (m1 / np.maximum(cnt, 1)) ** 2, 0.0))
    sd[~ok] = 0.0
    thr = 0.5 * np.percentile(sd[ok], 90) if ok.any() else 0.0
    act = sd > thr
    edges = np.flatnonzero(np.diff(np.concatenate([[0], act.astype(int), [0]])))
    if edges.size < 2:
        return fallback
    v0 = v.min()
    runs = [(a, b) for a, b in zip(edges[::2], edges[1::2])]
    a, b = max(runs, key=lambda r: r[1] - r[0])
    lo, hi = v0 + a + 2.0, v0 + b - 2.0            # 2 px inside the (blurred) bar ends
    if hi - lo < 4:
        return fallback
    return lo, hi


def _bin(u, v, val, u_lo, bw):
    """Bin pixels along u.  Per bin: mean u, mean value, count, u-variance, mean v, and the residual sum of
    squares about a within-bin linear trend in u with its degrees of freedom (for the noise estimate)."""
    idx = np.floor((u - u_lo) / bw).astype(int)
    nb = int(idx.max()) + 1
    B = lambda wts: np.bincount(idx, weights=wts, minlength=nb)
    cnt = np.bincount(idx, minlength=nb).astype(float)
    ok = cnt > 0
    n = cnt[ok]
    su, sp, sv = B(u)[ok], B(val)[ok], B(v)[ok]
    suu, sup, spp = B(u * u)[ok], B(u * val)[ok], B(val * val)[ok]
    um, pm, vm = su / n, sp / n, sv / n
    Suu = np.maximum(suu - n * um * um, 0.0)
    Sup = sup - n * um * pm
    Spp = np.maximum(spp - n * pm * pm, 0.0)
    has = Suu > 1e-8
    slope = np.where(has, Sup / np.where(has, Suu, 1.0), 0.0)
    SSR = np.maximum(Spp - slope * Sup, 0.0)
    dof = n - np.where(has, 2, 1)
    return um, pm, n, Suu / n, vm, SSR, dof


def _noise_model(um, pm, SSR, dof, bw):
    """Pixel noise variance law var(DN) = a + b*DN (read + quantisation, shot), from the within-bin scatter of
    bins where the profile is locally flat (so that the scatter is noise, not profile curvature)."""
    grid = np.arange(um[0], um[-1] + 1e-9, 0.25)
    if grid.size < 5:
        return 1.0, 0.03
    pg = np.interp(grid, um, pm)
    curv = np.zeros_like(pg)
    curv[2:-2] = np.abs(pg[4:] - 2 * pg[2:-2] + pg[:-4]) / 0.25     # |P''| over +-0.5 px
    curv[:2] = curv[2]; curv[-2:] = curv[-3]
    cb = np.interp(um, grid, ndimage.maximum_filter1d(curv, 3))
    lim = 100.0 * (0.25 / bw) ** 2
    m = (cb < lim) & (dof >= 2)
    if m.sum() < 8:
        m = dof >= 2
    if m.sum() < 3:
        return None
    L, V, Dg = pm[m], SSR[m] / dof[m], dof[m]
    qs = np.unique(np.quantile(L, np.linspace(0, 1, 6)))
    Lg, Vg, Wg = [], [], []
    for i in range(len(qs) - 1):
        k = (L >= qs[i]) & (L <= qs[i + 1])
        if k.sum() < 3:
            continue
        vk, dk = V[k], Dg[k]
        keep = vk <= 5 * np.median(vk)          # <=: a saturated or flat rendered band has zero scatter everywhere
        if not keep.any() or np.sum(dk[keep]) <= 0:
            continue
        Vg.append(float(np.sum(vk[keep] * dk[keep]) / np.sum(dk[keep])))
        Lg.append(float(np.sum(L[k][keep] * dk[keep]) / np.sum(dk[keep]))); Wg.append(float(np.sum(dk[keep])))
    Lg, Vg, Wg = np.array(Lg), np.array(Vg), np.array(Wg)
    q = 1.0 / 12.0
    if Lg.size == 0:
        return 1.0, 0.03
    if Lg.size < 2 or np.ptp(Lg) < 40:
        lm, vm = float(np.average(Lg, weights=Wg)), float(np.average(Vg, weights=Wg))
        return q, max(vm - q, 0.0) / max(lm, 1.0)
    wt = Wg / np.maximum(Vg, 0.05) ** 2
    A = np.stack([np.ones_like(Lg), Lg], 1)
    if not (np.isfinite(A).all() and np.isfinite(Vg).all() and np.isfinite(wt).all()):
        return 1.0, 0.03
    a, b = np.linalg.lstsq(A * np.sqrt(wt)[:, None], Vg * np.sqrt(wt), rcond=None)[0]
    if a < q:
        a = q
        b = float(np.sum(wt * Lg * (Vg - a)) / np.sum(wt * Lg * Lg))
    if b < 0:
        b = 0.0
        a = max(float(np.average(Vg, weights=Wg)), q)
    return float(a), float(b)


def _make_profile(um, pm, n, uvar, vm, noise):
    prof = Profile()
    prof.u, prof.p, prof.cnt, prof.uvar, prof.v = um, pm, n, uvar, vm
    prof.noise = noise
    prof.var = (noise[0] + noise[1] * np.maximum(pm, 0)) / n   # variance of each bin mean (DN^2)
    return prof


def extract_profile(img, reg, v_frac=(0.0, 1.0), refine=True, theta=None, strips=0):
    """Super-resolved bar-normal profile of a region (optionally a sub-band of its bars).  With strips=K the
    band is split into K strips along the bars, each binned separately (samples carry their strip's v)."""
    H, W = img.shape
    c, n = reg["c"], reg["n"]
    uh, vh = reg["uh"], reg["vh"]
    margin = max(14.0, 0.35 * uh * 2)
    u_lo, u_hi = -uh - margin, uh + margin
    if "vbar" not in reg:
        reg["vbar"] = _bar_vrange(img, reg)
    va, vb = reg["vbar"]
    v_lo, v_hi = va + (vb - va) * v_frac[0], va + (vb - va) * v_frac[1]
    if v_hi - v_lo < 3:
        return None
    th0 = np.arctan2(n[1], n[0]) if theta is None else theta

    def get(th):
        nn = np.array([np.cos(th), np.sin(th)]); tt = np.array([-nn[1], nn[0]])
        return _band_pixels(img, c, nn, tt, u_lo, u_hi, v_lo, v_hi), nn, tt

    if refine and theta is None:
        # bar-normal angle: minimise the within-bin scatter over a fixed pixel set (the symbol core)
        bp, nn, tt = get(th0)
        if bp is None or bp[0].size < 50:
            return None
        u, v, val = bp
        core = (u > -uh) & (u < uh)
        if core.sum() > 50:
            uu, vv, val = u[core], v[core], val[core]
            dx = uu * nn[0] + vv * tt[0]; dy = uu * nn[1] + vv * tt[1]
            th0 = _refine_angle(dx, dy, val, th0)
    bp, nn, tt = get(th0)
    if bp is None or bp[0].size < 50:
        return None
    u, v, val = bp
    bw = 0.25 if (u_hi - u_lo) < 260 else 0.5
    um, pm, cn, uvar, vm, SSR, dof = _bin(u, v, val, u_lo, bw)
    noise = _noise_model(um, pm, SSR, dof, bw)
    for bwn in (2 * bw, 4 * bw, 8 * bw):
        if noise is not None:
            break
        b = _bin(u, v, val, u_lo, bwn)
        noise = _noise_model(b[0], b[1], b[5], b[6], bwn)
    if noise is None:
        return None
    if strips:
        edges = np.linspace(v_lo, v_hi, strips + 1)
        parts = []
        for k in range(strips):
            mk = (v >= edges[k]) & (v < edges[k + 1]) if k < strips - 1 else (v >= edges[k])
            if mk.sum() < 20:
                continue
            parts.append(_bin(u[mk], v[mk], val[mk], u_lo, bw)[:5])
        um, pm, cn, uvar, vm = [np.concatenate([p[i] for p in parts]) for i in range(5)]
        o = np.argsort(um, kind="stable")
        um, pm, cn, uvar, vm = um[o], pm[o], cn[o], uvar[o], vm[o]
    prof = _make_profile(um, pm, cn, uvar, vm, noise)
    prof.theta = th0
    prof.c, prof.n, prof.t = c, nn, tt
    prof.v_lo, prof.v_hi = v_lo, v_hi
    prof.u_lo, prof.u_hi = u_lo, u_hi
    prof.u_frame_lo, prof.u_frame_hi = _frame_limits(c, nn, tt, v_lo, v_hi, W, H)
    prof.bw = bw
    prof.reversed = False
    return prof


def _frame_limits(c, n, t, v_lo, v_hi, W, H):
    """Range of u along the bar normal over which the whole band is inside the frame."""
    lo, hi = -np.inf, np.inf
    for vv in (v_lo, v_hi):
        p0 = c + t * vv
        for k, lim in ((0, W - 1), (1, H - 1)):
            if abs(n[k]) < 1e-9:
                continue
            a = (0 - p0[k]) / n[k]; b = (lim - p0[k]) / n[k]
            lo = max(lo, min(a, b)); hi = min(hi, max(a, b))
    return lo, hi


def _decimate(prof, f):
    """Merge groups of f consecutive bins (for the coarse hypothesis scan at high ppm)."""
    if f <= 1:
        return prof
    cache = prof.__dict__.setdefault("_dec", {})
    if f in cache:
        return cache[f]
    k = np.arange(prof.u.size) // f
    n = np.bincount(k, weights=prof.cnt)
    B = lambda a: np.bincount(k, weights=prof.cnt * a) / n
    u = B(prof.u)
    q = Profile()
    q.__dict__.update({kk: vv for kk, vv in prof.__dict__.items() if kk not in ("_dec", "_lev")})
    q.u, q.p, q.v, q.cnt = u, B(prof.p), B(prof.v), n
    q.uvar = B(prof.uvar + prof.u ** 2) - u ** 2
    q.uvar = np.maximum(q.uvar, 0.0)
    q.var = (prof.noise[0] + prof.noise[1] * np.maximum(q.p, 0)) / n
    cache[f] = q
    return q


def _reverse(prof):
    q = Profile()
    q.__dict__.update(prof.__dict__)
    q.u = -prof.u[::-1]; q.p = prof.p[::-1]; q.cnt = prof.cnt[::-1]; q.uvar = prof.uvar[::-1]; q.var = prof.var[::-1]
    q.v = prof.v[::-1]
    q._dec = {}
    q._lev = None
    q.u_frame_lo, q.u_frame_hi = -prof.u_frame_hi, -prof.u_frame_lo
    q.u_lo, q.u_hi = -prof.u_hi, -prof.u_lo
    q.reversed = True
    return q


# ----------------------------------------------------------------------------------------------------------
# geometry, fixed-part level and the full model
# ----------------------------------------------------------------------------------------------------------
def _x_of_u(u, g):
    q, c, u0 = g["q"], g["c"], g["u0"]
    d = u - u0
    if abs(c) < 1e-12:
        return d / q
    disc = np.maximum(q * q + 4 * c * d, 1e-12)
    return (-q + np.sqrt(disc)) / (2 * c)


def _u_of_x(x, g):
    return g["u0"] + g["q"] * x + g["c"] * x * x


def _edges_level(u, e, D, s_i, jac=False):
    """level(u) = 1 + sum_k D_k E(u - e_k); edges e sorted.  s_i: per-sample PSF sigma."""
    S = u.size
    h = 0.5 + 5.0 * np.max(s_i)
    lo = np.searchsorted(e, u - h)
    hi = np.searchsorted(e, u + h)
    cumD = np.concatenate([[0.0], np.cumsum(D)])
    base = 1.0 + cumD[lo]
    cnt = hi - lo
    tot = int(cnt.sum())
    if tot == 0:
        return (base, None) if jac else base
    ii = np.repeat(np.arange(S), cnt)
    kk = lo[ii] + (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    x = u[ii] - e[kk]
    if not jac:
        Ev = edge_resp(x, s_i[ii])
        return base + np.bincount(ii, weights=D[kk] * Ev, minlength=S)
    Ev, dEx, dEs = _edge_all(x, s_i[ii])
    lev = base + np.bincount(ii, weights=D[kk] * Ev, minlength=S)
    return lev, (ii, kk, D[kk] * dEx, D[kk] * dEs)


def _slot_edges(st, pat, x0):
    """Edges of one placed pattern: (a, b, kind) offsets -> list of (a, b, sign, kind)."""
    out = []
    a = b = 0
    col = st.first
    for (na, nb) in st.pats[pat]:
        if col == 1:
            out.append((a, b, 1.0, 0)); out.append((a + na, b + nb, -1.0, 1))
        a += na; b += nb; col ^= 1
    return out


def full_edges(hyp, path_pats, path_ts):
    """All edges of the symbol for a complete pattern assignment -> arrays (a, b, sign, kind)."""
    A, Bw, SG, K = [0.0], [0.0], [-1.0], [1]       # left quiet zone ends at x=0 (space end)
    for j, sl in enumerate(hyp.slots):
        st = sl["st"]
        pat = path_pats[j]
        t = path_ts[j]
        a0, b0 = sl["ab"]
        a0, b0 = a0 - t, b0 + t
        for (a, b, sg, kd) in _slot_edges(st, pat, 0):
            A.append(a0 + a); Bw.append(b0 + b); SG.append(sg); K.append(kd)
    A.append(hyp.N[0]); Bw.append(hyp.N[1]); SG.append(1.0); K.append(0)  # right quiet zone starts at x=N
    return np.array(A), np.array(Bw), np.array(SG), np.array(K)


PARAMS = ["u0", "q", "c", "s", "A", "B", "g", "d", "R"]


def model(prof, sel, g, E4, jac=False):
    """Prediction (and Jacobian) at the selected samples for full edge set E4=(a, b, sign, kind)."""
    a, b, sg, kd = E4
    R, d, q, c = g["R"], g["d"], g["q"], g["c"]
    x = a + b * R + np.where(kd == 0, d / 2, -d / 2)
    e = g["u0"] + q * x + c * x * x
    order = np.argsort(e, kind="stable")
    x, e, sg, kd, b_ = x[order], e[order], sg[order], kd[order], b[order]
    u = prof.u[sel]
    phi = g.get("phi", 0.0)
    if phi != 0.0:
        u = u - prof.v[sel] * phi
    s_i = np.sqrt(g["s"] ** 2 + prof.uvar[sel])
    G = 1.0 + g["g"] * (u - g["uc"])
    if not jac:
        lev = _edges_level(u, e, sg, s_i)
        return G * (g["A"] + g["B"] * lev), None
    lev, (ii, kk, dx, ds) = _edges_level(u, e, sg, s_i, jac=True)
    pred = G * (g["A"] + g["B"] * lev)
    S = u.size
    GB = G * g["B"]
    dL_de = -dx  # d level / d e_k  (per pair)
    dedx = q + 2 * c * x
    cols = {}
    cols["u0"] = GB * np.bincount(ii, weights=dL_de, minlength=S)
    cols["q"] = GB * np.bincount(ii, weights=dL_de * x[kk], minlength=S)
    cols["c"] = GB * np.bincount(ii, weights=dL_de * x[kk] ** 2, minlength=S)
    cols["s"] = GB * np.bincount(ii, weights=ds, minlength=S) * (g["s"] / s_i)
    cols["A"] = G
    cols["B"] = G * lev
    cols["g"] = (u - g["uc"]) * (g["A"] + g["B"] * lev)
    kk_sign = np.where(kd == 0, 0.5, -0.5)
    cols["d"] = GB * np.bincount(ii, weights=dL_de * dedx[kk] * kk_sign[kk], minlength=S)
    cols["R"] = GB * np.bincount(ii, weights=dL_de * dedx[kk] * b_[kk], minlength=S)
    cols["phi"] = prof.v[sel] * cols["u0"]
    return pred, cols


def _weights(prof, sel, g):
    return 1.0 / (prof.var[sel] + (MODEL_ERR * g["B"]) ** 2)


def _qz(hyp):
    return QUIET.get(hyp.family, QUIET_DEFAULT)


def _sel(prof, g, xN, qfit=QUIET_DEFAULT):
    ulo = _u_of_x(-qfit, g); uhi = _u_of_x(xN + qfit, g)
    return np.nonzero((prof.u >= ulo) & (prof.u <= uhi))[0]


_BOUNDS = dict(s=(0.05, 6.0), d=(-0.45, 0.45), R=(1.8, 3.4), g=(-0.004, 0.004), phi=(-0.02, 0.02))


def refine(prof, hyp, E4, g, free, iters=12):
    """Levenberg-Marquardt on the continuous parameters with the element sequence fixed."""
    g = dict(g)
    xN = hyp.N[0] + hyp.N[1] * g["R"]
    lam = 1e-3
    sel = _sel(prof, g, xN, _qz(hyp))
    if sel.size < 10:
        return g, np.inf, 0, sel
    w = _weights(prof, sel, g)
    y = prof.p[sel]
    pred, cols = model(prof, sel, g, E4, jac=True)
    r = y - pred
    chi = float(np.sum(w * r * r))
    for it in range(iters):
        J = np.stack([cols[k] for k in free], 1)
        Jw = J * w[:, None]
        H = J.T @ Jw
        gv = Jw.T @ r
        dH = np.diag(H).copy()
        dH[dH <= 0] = 1e-12
        improved = False
        for _ in range(6):
            try:
                step = np.linalg.solve(H + lam * np.diag(dH), gv)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            g2 = dict(g)
            for k, v in zip(free, step):
                g2[k] = g[k] + v
            for k, (lo, hi) in _BOUNDS.items():
                if k in g2:
                    g2[k] = min(max(g2[k], lo), hi)
            if g2["q"] <= 0.05 or g2["B"] <= 1:
                lam *= 10; continue
            xN2 = hyp.N[0] + hyp.N[1] * g2["R"]
            g2["s"] = min(g2["s"], S_MOD_MAX * g2["q"])
            if abs(g2["c"]) * xN2 > 0.15 * g2["q"]:
                g2["c"] = np.sign(g2["c"]) * 0.15 * g2["q"] / xN2
            sel2 = _sel(prof, g2, xN2, _qz(hyp))
            if sel2.size < 10:
                lam *= 10; continue
            w2 = _weights(prof, sel2, g2)
            pred2, cols2 = model(prof, sel2, g2, E4, jac=True)
            r2 = prof.p[sel2] - pred2
            chi2v = float(np.sum(w2 * r2 * r2))
            # compare per-sample normalised to tolerate the (small) change of the selected sample set
            if chi2v / sel2.size < chi / sel.size:
                rel = (chi / sel.size - chi2v / sel2.size) / (chi / sel.size + 1e-12)
                g, sel, w, r, cols, chi = g2, sel2, w2, r2, cols2, chi2v
                lam = max(lam / 4, 1e-7)
                improved = True
                break
            lam *= 6
        if not improved or rel < 1e-5:
            break
    return g, chi, sel.size, sel


# ----------------------------------------------------------------------------------------------------------
# character-level dynamic programme
# ----------------------------------------------------------------------------------------------------------
def _fixed_level(prof, hyp, g, xs):
    """Level contributed by the quiet zones and the fixed (single-pattern) slots, at samples with module coords xs."""
    A, Bw, SG, K = [0.0], [0.0], [-1.0], [1]
    for j, sl in enumerate(hyp.slots):
        if sl["var"]:
            continue
        t = sl["ts"][0]
        a0, b0 = sl["ab"]
        a0, b0 = a0 - t, b0 + t
        for (a, b, sg, kd) in _slot_edges(sl["st"], 0, 0):
            A.append(a0 + a); Bw.append(b0 + b); SG.append(sg); K.append(kd)
    A.append(hyp.N[0]); Bw.append(hyp.N[1]); SG.append(1.0); K.append(0)
    A, Bw, SG, K = map(np.asarray, (A, Bw, SG, K))
    x = A + Bw * g["R"] + np.where(K == 0, g["d"] / 2, -g["d"] / 2)
    e = _u_of_x(x, g)
    o = np.argsort(e, kind="stable")
    # fixed contributions as spaces: level_fixed = 1 + sum SG E  -- but the variable slots' spaces are added
    # separately, so the fixed part must not include the implicit "1" of the right quiet zone twice.
    return e[o], SG[o]


def _footprints(st, x0, g, u, s_i, cls_mask=None):
    """(C, S) level footprints of every pattern of slot type st placed at module x0, at samples u."""
    bnd = st.bnd
    xb = x0 + bnd[:, 0] + bnd[:, 1] * g["R"] + np.where(bnd[:, 2] == 0, g["d"] / 2, -g["d"] / 2)
    ub = _u_of_x(xb, g)
    Em = edge_resp(u[None, :] - ub[:, None], s_i[None, :])
    inc = st.inc if cls_mask is None else st.inc[cls_mask]
    return inc @ Em


class DPProblem:
    """Pairwise cost matrices over expanded states (t, pattern) of the variable slots, for fixed geometry."""

    def __init__(self, prof, hyp, g):
        self.hyp, self.g = hyp, g
        R = g["R"]
        xN = hyp.xN(R)
        sel = _sel(prof, g, xN, _qz(hyp))
        self.sel = sel
        u = prof.u[sel]
        self.ok = sel.size >= 10
        if not self.ok:
            return
        s_i = np.sqrt(g["s"] ** 2 + prof.uvar[sel])
        x = _x_of_u(u, g)
        G = 1.0 + g["g"] * (u - g["uc"])
        w = _weights(prof, sel, g) * (g["B"] * G) ** 2      # weights in level units
        e_f, sg_f = _fixed_level(prof, hyp, g, x)
        lev_fixed = _edges_level(u, e_f, sg_f, s_i)       # quiet zones + fixed slots (+1 baseline)
        # the variable slots' space footprints add to lev_fixed; observed level:
        y = (prof.p[sel] / G - g["A"]) / g["B"] - lev_fixed
        self.w, self.y = w, y
        V = hyp.var_idx
        self.V = V
        self.states = []    # per variable slot: (ts list, C)
        for j in V:
            sl = hyp.slots[j]
            self.states.append((sl["ts"], sl["st"].C))

        def mids(j, t):
            st = hyp.slots[j]["st"]
            x0 = hyp.start_x(j, t, R)
            widths = st.width[:, 0] + st.width[:, 1] * R
            return x0, widths

        def rng_idx(xa, xb):
            return np.searchsorted(x, xa), np.searchsorted(x, xb)

        const = 0.0
        # unary first: samples before mid of first variable slot
        j0 = V[0]
        ts0, C0 = self.states[0]
        U0 = np.full(len(ts0) * C0, np.inf)
        for ti, t in enumerate(ts0):
            x0, wd = mids(j0, t)
            st = hyp.slots[j0]["st"]
            for cl in np.unique(st.cls):
                cm = st.cls == cl
                i0, i1 = rng_idx(-np.inf, x0 + wd[cm][0] / 2)
                if i1 > i0:
                    F = _footprints(st, x0, g, u[i0:i1], s_i[i0:i1], cm)
                    cost = ((y[i0:i1][None, :] - F) ** 2 * w[i0:i1][None, :]).sum(1)
                else:
                    cost = np.zeros(cm.sum())
                U0[ti * C0 + np.nonzero(cm)[0]] = cost
        self.U0 = U0
        # pairs
        self.M = []
        for pi in range(len(V) - 1):
            ja, jb = V[pi], V[pi + 1]
            tsa, Ca = self.states[pi]
            tsb, Cb = self.states[pi + 1]
            sta, stb = hyp.slots[ja]["st"], hyp.slots[jb]["st"]
            Mfull = np.full((len(tsa) * Ca, len(tsb) * Cb), np.inf)
            for tai, t in enumerate(tsa):
                xa0, wda = mids(ja, t)
                for cla in np.unique(sta.cls):
                    cma = sta.cls == cla
                    inc = cla if (hyp.family == "Codabar" and sta.name == "cbar_data") else 0
                    t2 = t + inc
                    if t2 not in tsb:
                        continue
                    tbi = tsb.index(t2)
                    xb0, wdb = mids(jb, t2)
                    for clb in np.unique(stb.cls):
                        cmb = stb.cls == clb
                        i0, i1 = rng_idx(xa0 + wda[cma][0] / 2, xb0 + wdb[cmb][0] / 2)
                        rows = tai * Ca + np.nonzero(cma)[0]
                        colsb = tbi * Cb + np.nonzero(cmb)[0]
                        if i1 <= i0:
                            Mfull[np.ix_(rows, colsb)] = 0.0
                            continue
                        uu, ss, yy, ww = u[i0:i1], s_i[i0:i1], y[i0:i1], w[i0:i1]
                        F = _footprints(sta, xa0, g, uu, ss, cma)
                        Gm = _footprints(stb, xb0, g, uu, ss, cmb)
                        wy = ww * yy
                        c0 = float(np.sum(ww * yy * yy))
                        fa = -2 * F @ wy + (F * F) @ ww
                        gb = -2 * Gm @ wy + (Gm * Gm) @ ww
                        cross = 2 * (F * ww[None, :]) @ Gm.T
                        Mfull[np.ix_(rows, colsb)] = c0 + fa[:, None] + gb[None, :] + cross
            self.M.append(Mfull)
        # unary last
        jl = V[-1]
        tsl, Cl = self.states[-1]
        UL = np.full(len(tsl) * Cl, np.inf)
        stl = hyp.slots[jl]["st"]
        for ti, t in enumerate(tsl):
            x0, wd = mids(jl, t)
            for cl in np.unique(stl.cls):
                cm = stl.cls == cl
                i0, i1 = rng_idx(x0 + wd[cm][0] / 2, np.inf)
                if i1 > i0:
                    F = _footprints(stl, x0, g, u[i0:i1], s_i[i0:i1], cm)
                    cost = ((y[i0:i1][None, :] - F) ** 2 * w[i0:i1][None, :]).sum(1)
                else:
                    cost = np.zeros(cm.sum())
                UL[ti * Cl + np.nonzero(cm)[0]] = cost
        if len(V) == 1:  # single variable slot: both unaries cover disjoint halves
            pass
        self.UL = UL
        self.const = const

    def decode_state(self, pi, s):
        ts, C = self.states[pi]
        return ts[s // C], s % C


class FastDP:
    """DPProblem for fixed slot positions (every family except Codabar with 3-wide data characters),
    batched over all adjacent slot pairs: one edge-response evaluation and one batched matmul."""

    def __init__(self, prof, hyp, g, scan_k=None):
        self.hyp, self.g = hyp, g
        R = g["R"]
        xN = hyp.xN(R)
        sel = _sel(prof, g, xN, _qz(hyp))
        self.sel = sel
        self.ok = sel.size >= 10
        if not self.ok:
            return
        fs = _fast_struct(hyp)
        J = fs["J"]
        X0 = fs["a0"] + fs["b0"] * R
        mids = X0 + (fs["wa"] + fs["wb"] * R) / 2
        K = scan_k
        split = K is not None and J > 2 * K + 1
        if split:  # only the samples of the first and last K characters' windows are needed
            ua, ub = _u_of_x(mids[K - 1], g), _u_of_x(mids[J - K], g)
            su = prof.u[sel]
            sel = sel[(su < ua) | (su >= ub)]
        u = prof.u[sel]
        s_i = np.sqrt(g["s"] ** 2 + prof.uvar[sel])
        x = _x_of_u(u, g)
        G = 1.0 + g["g"] * (u - g["uc"])
        w = _weights(prof, sel, g) * (g["B"] * G) ** 2
        e_f, sg_f = _fixed_level(prof, hyp, g, x)
        y = (prof.p[sel] / G - g["A"]) / g["B"] - _edges_level(u, e_f, sg_f, s_i)
        self.V = hyp.var_idx
        self.states = fs["states"]
        Cs = fs["Cs"]
        if g["d"] == 0.0:   # no ink spread: space-start and space-end boundaries at one position coincide
            UB = _u_of_x(X0[:, None] + fs["BA2"] + fs["BB2"] * R, g)
            INC = fs["INC2"]
        else:
            d2 = g["d"] / 2
            UB = _u_of_x(X0[:, None] + fs["BA"] + fs["BB"] * R + np.where(fs["BK"] == 0, d2, -d2), g)  # (J, P)
            INC = fs["INC"]
        cuts = np.searchsorted(x, mids)
        lo = np.concatenate([[0], cuts]); hi = np.concatenate([cuts, [x.size]])
        nwin = hi - lo
        Smax = max(int(nwin.max()), 1)
        ar = np.arange(Smax)
        valid = ar[None, :] < nwin[:, None]
        idx = np.where(valid, lo[:, None] + ar[None, :], 0)
        Uw, Sw, Yw = u[idx], s_i[idx], y[idx]
        Ww = np.where(valid, w[idx], 0.0)

        def foot(slot_ids, wins):
            Em = edge_resp(Uw[wins][:, None, :] - UB[slot_ids][:, :, None], Sw[wins][:, None, :])
            return np.matmul(INC[slot_ids], Em)            # (K, C, S)
        inf = np.inf
        F0 = foot(np.array([0]), np.array([0]))[0]
        U0 = ((Yw[0][None, :] - F0) ** 2 * Ww[0][None, :]).sum(1)
        FL = foot(np.array([J - 1]), np.array([J]))[0]
        UL = ((Yw[J][None, :] - FL) ** 2 * Ww[J][None, :]).sum(1)
        ban = fs["ban"]
        self.U0 = np.where(ban[0], inf, U0[:Cs[0]])
        self.UL = np.where(ban[-1], inf, UL[:Cs[-1]])
        self.M = []
        pk = np.concatenate([np.arange(0, K - 1), np.arange(J - K, J - 1)]) if split else np.arange(J - 1)
        if J > 1:
            wins = pk + 1
            F = foot(pk, wins)                                   # (npairs, Cmax, S)
            Gm = foot(pk + 1, wins)
            Wp = Ww[wins][:, :, None]
            wy = Wp * Yw[wins][:, :, None]
            c0 = (Ww[wins] * Yw[wins] ** 2).sum(1)
            fa = (np.matmul(F * F, Wp) - 2 * np.matmul(F, wy))[:, :, 0]
            gb = (np.matmul(Gm * Gm, Wp) - 2 * np.matmul(Gm, wy))[:, :, 0]
            cross = 2 * np.matmul(F * np.transpose(Wp, (0, 2, 1)), np.transpose(Gm, (0, 2, 1)))
            Mall = c0[:, None, None] + fa[:, :, None] + gb[:, None, :] + cross
            for i, k in enumerate(pk):
                Mk = Mall[i, :Cs[k], :Cs[k + 1]]
                if ban[k].any() or ban[k + 1].any():
                    Mk = Mk.copy(); Mk[ban[k], :] = inf; Mk[:, ban[k + 1]] = inf
                self.M.append(Mk)
        if K is not None:
            # scan score: best cost of the first K and last K characters (the whole chain if short)
            if split:
                ca = _vit_min(self.U0, self.M[:K - 1], np.zeros(Cs[K - 1]))
                cb = _vit_min(np.zeros(Cs[J - K]), self.M[K - 1:], self.UL)
                used = int(nwin[:K].sum() + nwin[J - K + 1:].sum())
                self.scan = (ca + cb, used)
            else:
                self.scan = (_vit_min(self.U0, self.M, self.UL), int(nwin.sum()))

    def decode_state(self, pi, s):
        return self.states[pi][0][0], s


_FS_CACHE = {}


def _fast_struct(hyp):
    """Geometry-independent arrays of a hypothesis (cached by family, length, T)."""
    fs = getattr(hyp, "_fs", None)
    if fs is not None:
        return fs
    key = (hyp.family, hyp.n, hyp.T)
    if key in _FS_CACHE:
        hyp._fs = _FS_CACHE[key]
        return hyp._fs
    V = hyp.var_idx
    sts = [hyp.slots[j]["st"] for j in V]
    J = len(V)
    Cs = np.array([st.C for st in sts])
    Cmax, Pmax = int(Cs.max()), max(st.bnd.shape[0] for st in sts)
    BA = np.zeros((J, Pmax)); BB = np.zeros((J, Pmax)); BK = np.zeros((J, Pmax)); INC = np.zeros((J, Cmax, Pmax))
    a0 = np.zeros(J); b0 = np.zeros(J)
    for k, (j, st) in enumerate(zip(V, sts)):
        t = hyp.slots[j]["ts"][0]
        a, b = hyp.slots[j]["ab"]
        a0[k], b0[k] = a - t, b + t
        nb = st.bnd.shape[0]
        BA[k, :nb], BB[k, :nb], BK[k, :nb] = st.bnd[:, 0], st.bnd[:, 1], st.bnd[:, 2]
        INC[k, :st.C, :nb] = st.inc
    ban = [np.zeros(st.C, bool) if not (hyp.family == "Codabar" and st.name == "cbar_data")
           else st.cls.astype(bool) for st in sts]
    # boundaries merged over kind (valid when the ink spread d = 0)
    P2 = 1
    merged = []
    for st in sts:
        keys = {}
        cols = []
        for (a, b, kd) in st.bnd:
            cols.append(keys.setdefault((a, b), len(keys)))
        inc2 = np.zeros((st.C, len(keys)))
        for pcol, k2 in enumerate(cols):
            inc2[:, k2] += st.inc[:, pcol]
        merged.append((np.array(list(keys), float).reshape(-1, 2), inc2))
        P2 = max(P2, len(keys))
    BA2 = np.zeros((J, P2)); BB2 = np.zeros((J, P2)); INC2 = np.zeros((J, Cmax, P2))
    for k, (ab, inc2) in enumerate(merged):
        BA2[k, :len(ab)], BB2[k, :len(ab)] = ab[:, 0], ab[:, 1]
        INC2[k, :inc2.shape[0], :inc2.shape[1]] = inc2
    fs = dict(J=J, Cs=Cs, BA=BA, BB=BB, BK=BK, INC=INC, a0=a0, b0=b0, BA2=BA2, BB2=BB2, INC2=INC2,
              wa=np.array([st.width[0, 0] for st in sts]), wb=np.array([st.width[0, 1] for st in sts]),
              states=[([hyp.slots[j]["ts"][0]], st.C) for j, st in zip(V, sts)], ban=ban)
    hyp._fs = fs
    _FS_CACHE[key] = fs
    return fs


def make_dp(prof, hyp, g):
    if hyp.family == "Codabar" and hyp.T > 0:
        return DPProblem(prof, hyp, g)
    return FastDP(prof, hyp, g)


def _vit_min(U0, Ms, UL):
    V = U0
    for M in Ms:
        V = np.min(V[:, None] + M, 0)
    return float(np.min(V + UL))


def _vit_path(U0, Ms, UL):
    V = U0
    bps = []
    for M in Ms:
        t = V[:, None] + M
        a = np.argmin(t, 0)
        bps.append(a)
        V = t[a, np.arange(t.shape[1])]
    fin = V + UL
    s = int(np.argmin(fin))
    cost = float(fin[s])
    if not np.isfinite(cost):
        return np.inf, None
    path = [s]
    for a in reversed(bps):
        s = int(a[s]); path.append(s)
    return cost, path[::-1]


def viterbi(U0, Ms, UL, aux=None, second=False):
    """Min-sum Viterbi over expanded states with an optional deterministic auxiliary (checksum) state.

    aux = (init (Z,) costs, trans list [(Z, S_j) int, -1 = disallowed], accept (Z,) bool).
    Returns (cost, path) and, if second, the best path that differs from it in at least one slot."""
    if aux is None and not second:
        return _vit_path(U0, Ms, UL)
    nS = [U0.size] + [M.shape[1] for M in Ms]
    J = len(nS)
    if aux is None:
        Z = 1
        init = np.zeros(1)
        trans = [np.zeros((1, n), int) for n in nS]
        accept = np.ones(1, bool)
    else:
        init, trans, accept = aux
        Z = init.size
    # forward
    Vs, BPs = [], []
    V = np.full((nS[0], Z), np.inf)
    zi = np.nonzero(np.isfinite(init))[0]
    for z in zi:
        zz = trans[0][z]
        ok = zz >= 0
        sidx = np.nonzero(ok)[0]
        cur = V[sidx, zz[ok]]
        V[sidx, zz[ok]] = np.minimum(cur, U0[sidx] + init[z])
    Vs.append(V)
    for jj in range(1, J):
        M = Ms[jj - 1]
        tot = V.T[None, :, :] + M.T[:, None, :]        # (Sb, Z, Sa): reduce over the contiguous last axis
        am = np.argmin(tot, -1)                         # (Sb, Z)
        mv = np.take_along_axis(tot, am[..., None], -1)[..., 0]
        Vn = np.full((nS[jj], Z), np.inf)
        bp = np.full((nS[jj], Z, 2), -1, int)
        tr = trans[jj]                                   # (Z, Sb)
        zz, bb = np.nonzero(tr >= 0)
        z2 = tr[zz, bb]
        vals = mv[bb, zz]
        better = vals < Vn[bb, z2]
        Vn[bb[better], z2[better]] = vals[better]
        bp[bb[better], z2[better], 0] = am[bb[better], zz[better]]
        bp[bb[better], z2[better], 1] = zz[better]
        V = Vn
        Vs.append(V); BPs.append(bp)
    fin = V + UL[:, None]
    fin[:, ~accept] = np.inf
    k = int(np.argmin(fin))
    best = float(fin.flat[k])
    if not np.isfinite(best):
        return (np.inf, None) if not second else (np.inf, None, np.inf, None)
    s, z = divmod(k, Z)
    path = [0] * J
    zs = [0] * J
    path[-1], zs[-1] = s, z
    for jj in range(J - 1, 0, -1):
        s, z = BPs[jj - 1][s, z]
        path[jj - 1], zs[jj - 1] = int(s), int(z)
    if not second:
        return best, path
    # backward pass for min-marginals
    Bk = np.full((nS[-1], Z), np.inf)
    Bk[:, accept] = UL[:, None]
    Bks = [None] * J
    FPs = [None] * J
    Bks[-1] = Bk
    for jj in range(J - 1, 0, -1):
        M = Ms[jj - 1]
        tr = trans[jj]                                  # (Z, Sb)
        Gz = np.full((Z, nS[jj]), np.inf)
        zz, bb = np.nonzero(tr >= 0)
        Gz[zz, bb] = Bk[bb, tr[zz, bb]]
        tot = M[:, None, :] + Gz[None, :, :]            # (Sa, Z, Sb)
        am = np.argmin(tot, -1)                         # (Sa, Z)
        Bk = np.take_along_axis(tot, am[..., None], -1)[..., 0]
        Bks[jj - 1] = Bk
        FPs[jj - 1] = am
    best2, arg2 = np.inf, None
    for jj in range(J):
        MM = Vs[jj] + Bks[jj]
        MM[path[jj], :] = np.inf
        k = int(np.argmin(MM))
        if MM.flat[k] < best2:
            best2 = float(MM.flat[k]); arg2 = (jj, *divmod(k, Z))
    if arg2 is None or not np.isfinite(best2):
        return best, path, np.inf, None
    jj, s, z = arg2
    p2 = [0] * J
    p2[jj] = s
    cs, cz = s, z
    for j2 in range(jj, 0, -1):
        cs, cz = BPs[j2 - 1][cs, cz]
        p2[j2 - 1] = int(cs)
    cs, cz = s, z
    for j2 in range(jj, J - 1):
        nb = int(FPs[j2][cs, cz])
        cz = int(trans[j2 + 1][cz, nb])
        cs = nb
        p2[j2 + 1] = cs
    return best, path, best2, p2


# ----------------------------------------------------------------------------------------------------------
# checksum auxiliary states for the final DP
# ----------------------------------------------------------------------------------------------------------
def _aux(hyp, dp, formats):
    fam = hyp.family
    nvar = len(dp.V)
    if fam == "EAN13":
        firsts = range(10) if "EAN13" in formats else [0]
        Z = 100
        init = np.full(Z, np.inf)
        for p in firsts:
            init[p * 10 + p % 10] = 0.0
        trans = []
        for k in range(12):
            nS = 20 if k < 6 else 10
            tr = np.full((Z, nS), -1, int)
            wgt = 3 if k % 2 == 0 else 1
            for z in range(Z):
                p, s = divmod(z, 10)
                for c in range(nS):
                    d = c % 10
                    if k < 6:
                        par = "G" if c >= 10 else "L"
                        if _EAN13_FIRST[p][k] != par:
                            continue
                    tr[z, c] = p * 10 + (s + wgt * d) % 10
            trans.append(tr)
        accept = np.array([z % 10 == 0 for z in range(Z)])
        return init, trans, accept
    if fam == "EAN8":
        Z = 10
        init = np.full(Z, np.inf); init[0] = 0
        trans = []
        for k in range(8):
            wgt = 3 if k % 2 == 0 else 1
            tr = np.full((Z, 10), -1, int)
            for z in range(Z):
                for c in range(10):
                    tr[z, c] = (z + wgt * c) % 10
            trans.append(tr)
        accept = np.arange(Z) == 0
        return init, trans, accept
    if fam == "UPCE":
        Z = 20 * 4 * 10
        init = np.full(Z, np.inf)
        for p in range(20):
            ns = p // 10
            for kap in range(4):
                init[(p * 4 + kap) * 10 + (3 * ns) % 10] = 0.0
        trans = []
        for k in range(6):
            tr = np.full((Z, 20), -1, int)
            for z in range(Z):
                pk, s = divmod(z, 10)
                p, kap = divmod(pk, 4)
                ns, chk = divmod(p, 10)
                pat = _UPCE_PAR0[chk]
                for c in range(20):
                    d = c % 10
                    par_even = c >= 10
                    want_even = (pat[k] == "E") ^ (ns == 1)
                    if par_even != want_even:
                        continue
                    if k == 5:
                        cls = 0 if d <= 2 else (1 if d == 3 else (2 if d == 4 else 3))
                        if cls != kap:
                            continue
                    tr[z, c] = (p * 4 + kap) * 10 + (s + _UPCE_W[kap][k] * d) % 10
            trans.append(tr)
        accept = np.zeros(Z, bool)
        for z in range(Z):
            pk, s = divmod(z, 10)
            p, kap = divmod(pk, 4)
            accept[z] = (10 - s) % 10 == p % 10
        return init, trans, accept
    if fam == "Code128":
        Z = 103
        init = np.full(Z, np.inf); init[0] = 0
        trans = []
        tr = np.full((Z, 3), -1, int)
        tr[0, :] = [103 % 103, 104 % 103, 105 % 103]
        trans.append(tr)
        ndata = nvar - 2
        vals = np.arange(103)
        for k in range(ndata):
            tr = (np.arange(Z)[:, None] + (k + 1) * vals[None, :]) % 103
            trans.append(tr.astype(int))
        tr = np.full((Z, 103), -1, int)
        tr[np.arange(103), np.arange(103)] = 0
        trans.append(tr)
        accept = np.ones(Z, bool)
        return init, trans, accept
    return None


# ----------------------------------------------------------------------------------------------------------
# candidate rendering and zxing validation
# ----------------------------------------------------------------------------------------------------------
def _formats(names):
    import zxingcpp
    # a tuple, as decoder._formats (zxing-cpp deprecates combining formats with |)
    return tuple(getattr(zxingcpp.BarcodeFormat, n) for n in names)


def render_clean(hyp, pats, ts, R, px_per_module=3):
    """Clean synthetic image of an element sequence (for zxing)."""
    Rr = float(np.clip(np.round(R * 2) / 2, 2.0, 3.0)) if hyp.two_width else 1.0
    widths, cols = [], []
    for j, sl in enumerate(hyp.slots):
        st = sl["st"]
        col = st.first
        for (na, nb) in st.pats[pats[j]]:
            widths.append(na + nb * Rr); cols.append(col); col ^= 1
    q = 12
    row = [np.full(int(round(q * px_per_module)), 255, np.uint8)]
    for w, c in zip(widths, cols):
        row.append(np.full(max(1, int(round(w * px_per_module))), 255 if c == 1 else 0, np.uint8))
    row.append(np.full(int(round(q * px_per_module)), 255, np.uint8))
    row = np.concatenate(row)
    img = np.tile(row, (24, 1))
    return np.pad(img, ((8, 8), (0, 0)), constant_values=255)


def zxing_validate(img, fmts):
    import zxingcpp
    res = zxingcpp.read_barcodes(img, formats=fmts, try_rotate=False, try_downscale=False, try_invert=False)
    out = []
    for r in res:
        if not r.valid:
            continue
        name = r.format.name if hasattr(r.format, "name") else str(r.format)
        out.append((str(name).replace("-", "").replace(" ", ""), r.text))
    return out


def norm_text(fmt, text):
    if fmt in ("EAN13", "UPCA", "UPCE", "EAN8"):
        return "".join(ch for ch in text if ch.isdigit()).zfill(14)
    if fmt == "Codabar" and len(text) > 2 and text[0] in "ABCD" and text[-1] in "ABCD":
        return text[1:-1]
    return text


# ----------------------------------------------------------------------------------------------------------
# hypothesis enumeration and the per-profile search
# ----------------------------------------------------------------------------------------------------------
_FAMILY_OF = {"EAN13": "EAN13", "UPCA": "EAN13", "EAN8": "EAN8", "UPCE": "UPCE", "Code128": "Code128",
              "Code39": "Code39", "Codabar": "Codabar", "ITF": "ITF"}


def _levels(prof):
    if getattr(prof, "_lev", None) is not None:
        return prof._lev
    prof._lev = _levels0(prof)
    return prof._lev


def _levels0(prof):
    p = prof.p
    ps = ndimage.uniform_filter1d(p, 3)
    white = float(np.percentile(ps, 97))
    dark = float(np.percentile(ps, 1))
    return white, dark


def _spans(prof):
    """Candidate [left, right] outer extents of a symbol inside the profile (u units)."""
    white, dark = _levels(prof)
    if white - dark < 12:
        return []
    lvl = (prof.p - dark) / (white - dark)
    darkm = lvl < 0.72
    if darkm.sum() < 3:
        return []
    u = prof.u
    du = u[darkm]
    gaps = np.diff(du)
    centre = 0.0
    out = []
    for G in (2.5, 4.0, 6.0, 9.0, 14.0, 22.0):
        cut = np.nonzero(gaps > G)[0]
        starts = np.concatenate([[0], cut + 1]); ends = np.concatenate([cut, [du.size - 1]])
        best, bl = None, -1
        for a, b in zip(starts, ends):
            L = du[b] - du[a]
            if du[a] - 2 <= centre <= du[b] + 2 or L > bl:
                if du[a] - 2 <= centre <= du[b] + 2:
                    best = (du[a], du[b]); break
                best, bl = (du[a], du[b]), L
        if best is not None and best[1] - best[0] > 10:
            if not any(abs(best[0] - o[0]) < 1.0 and abs(best[1] - o[1]) < 1.0 for o in out):
                out.append(best)
    # a symbol's quiet zones (>= 7 modules) are wider than any space inside it (<= 4 modules, ratio >= 1.75); blur only
    # narrows white runs, so a span whose outside white runs are not clearly the widest is not a symbol
    keep = []
    for a, b in out:
        inside = du[(du >= a) & (du <= b)]
        gint = float(np.max(np.diff(inside))) if inside.size > 1 else 0.0
        left = du[du < a]; right = du[du > b]
        gl = a - (left.max() if left.size else u[0])
        gr = (right.min() if right.size else u[-1]) - b
        if min(gl, gr) >= max(1.6 * gint, 3.0):
            keep.append((_crossing(u, lvl, a, -1), _crossing(u, lvl, b, +1)))
    return keep


def _crossing(u, lvl, ue, side, thr=0.72):
    """Sub-sample position of the thr crossing next to the outermost dark sample at ue (side -1 = left end)."""
    i = int(np.searchsorted(u, ue))
    j = i + side
    if j < 0 or j >= u.size or lvl[j] <= thr:
        return ue
    return u[i] + (u[j] - u[i]) * (thr - lvl[i]) / (lvl[j] - lvl[i])


def _span_bounds(prof, span):
    """Model-free facts about the symbol inside `span` used to prune length hypotheses:
    flat: longest run (px) that stays flat at full dark or full white -- it can only be ONE element (any space
          inside a bar, or bar inside a space, of >= 1 module would raise a bump far above the noise), so
          q >= flat / (widest element of the symbology);
    dips: number of significant local minima -- each needs its own bar, so bars(n) >= dips."""
    from scipy.signal import find_peaks
    white, dark = _levels(prof)
    m = (prof.u >= span[0]) & (prof.u <= span[1])
    u, p = prof.u[m], prof.p[m]
    if u.size < 5:
        return 0.0, 0
    sd = np.sqrt(prof.var[m])
    contrast = max(white - dark, 1.0)
    tol = 4 * sd + 0.02 * contrast
    pmin = float(np.min(ndimage.uniform_filter1d(p, 3)))
    flat = 0.0
    for fl in (p < pmin + tol, p > white - tol):
        edges = np.flatnonzero(np.diff(np.concatenate([[0], fl.astype(int), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= 2:
                flat = max(flat, float(u[b - 1] - u[a]))
    # dips on a count-weighted +-0.35 px box-smoothed profile (single sparse bins are too noisy)
    Cn = np.concatenate([[0.0], np.cumsum(prof.cnt[m])]); Cp = np.concatenate([[0.0], np.cumsum(prof.cnt[m] * p)])
    lo = np.searchsorted(u, u - 0.35); hi = np.searchsorted(u, u + 0.35, side="right")
    nn = Cn[hi] - Cn[lo]
    ps = (Cp[hi] - Cp[lo]) / nn
    sds = np.sqrt((prof.noise[0] + prof.noise[1] * np.maximum(ps, 0)) / nn)
    prom = 6 * float(np.median(sds)) + 0.04 * contrast
    pk, _ = find_peaks(-ps, prominence=prom)
    return flat, int(pk.size)


def _bars(fam, n):
    return {"EAN13": 30, "EAN8": 22, "UPCE": 17}.get(fam) or {
        "Code128": 3 * n + 10, "Code39": 5 * (n + 2), "ITF": 5 * n + 4, "Codabar": 4 * (n + 2)}[fam]


def _enumerate(fam, L, formats, flat=0.0, dips=0):
    """Length hypotheses whose implied ppm = L/x(N) lies in PPM_RANGE (L = outer span in px), consistent with
    the flat-run and dip-count bounds of _span_bounds."""
    lo, hi = PPM_RANGE
    out = []

    def ok(N, wmax, n):
        q = L / N
        return lo <= q <= hi and q >= flat / wmax and _bars(fam, n) >= dips - max(2, 0.1 * dips)
    if fam in ("EAN13", "EAN8", "UPCE"):
        h = Hyp(fam)
        if ok(h.xN(), 4, 0):
            out.append(h)
        return out
    if fam == "Code128":
        for n in range(1, 60):
            if ok(11 * n + 35, 4, n):
                out.append(Hyp(fam, n))
        return out
    for R in (2.0, 2.5, 3.0):
        for n in range(1, 60):
            if fam == "Code39":
                N = (7 * n + 13) + (3 * n + 6) * R
            elif fam == "ITF":
                N = (6 * n + 6) + (4 * n + 1) * R
            else:  # Codabar; 3-wide data characters (T>0) are tried at refinement
                N = (9 + 6 * n) + (6 + 2 * n) * R
            if ok(N, R, n):
                out.append(Hyp(fam, n, R))
    return out


_END_CACHE = {}


def _end_edges(hyp):
    """Module coordinates and signs of the edges within 8 modules of each end (pattern 0 for variable slots)."""
    key = (hyp.family, hyp.n, hyp.T, round(hyp.R, 3))
    if key not in _END_CACHE:
        a, b, sg, kd = full_edges(hyp, [0] * len(hyp.slots), [sl["ts"][0] for sl in hyp.slots])
        x = a + b * hyp.R
        xN = hyp.xN()
        o = np.argsort(x, kind="stable")
        x, sg = x[o], sg[o]
        _END_CACHE[key] = (x, sg, xN)
    return _END_CACHE[key]


_OFF_CACHE = {}


def _crossing_offsets(hyp, q, s, lstar):
    """Where the modelled level first/last drops below lstar, relative to the outer bar edges (px, positive =
    inside the symbol), for the given module size q and PSF sigma s.  The ends of a symbology do not depend on
    its length, so the result (in modules) is cached on a 0.5 % grid of q and s/q."""
    lq = round(np.log(q) / 0.005); lsm = round(s / q / 0.01); ll = round(lstar / 0.005)
    key = (hyp.family, round(hyp.R, 3), lq, lsm, ll)
    if key not in _OFF_CACHE:
        qq = float(np.exp(lq * 0.005)); ss = lsm * 0.01 * qq
        rep = Hyp(hyp.family, 6 if hyp.family != "Code128" else 4, hyp.R, 0)
        ol, orr = _crossing_offsets0(rep, qq, ss, ll * 0.005)
        _OFF_CACHE[key] = (ol / qq, orr / qq)
    ol, orr = _OFF_CACHE[key]
    return ol * q, orr * q


def _crossing_offsets0(hyp, q, s, lstar):
    x, sg, xN = _end_edges(hyp)
    grid_l = np.linspace(-4, 8, 97); grid_r = np.linspace(xN - 8, xN + 4, 97)
    si = np.full(grid_l.size, s)
    out = []
    for grid, left in ((grid_l, True), (grid_r, False)):
        m = (x > grid[0] - 6) & (x < grid[-1] + 6)
        e = x[m] * q
        base = 1.0 + (np.sum(sg[x <= grid[0] - 6]) if left else np.sum(sg[x <= grid[0] - 6]))
        lev = base + (sg[m][None, :] * edge_resp(grid[:, None] * q - e[None, :], si[:, None])).sum(1)
        below = np.nonzero(lev < lstar)[0]
        if below.size == 0:
            out.append(0.0)
            continue
        if left:
            i = below[0]
            xc = grid[i - 1] + (grid[i] - grid[i - 1]) * (lev[i - 1] - lstar) / (lev[i - 1] - lev[i]) if i > 0 else grid[i]
            out.append(xc * q)
        else:
            i = below[-1]
            xc = grid[i] + (grid[i + 1] - grid[i]) * (lstar - lev[i]) / (lev[i + 1] - lev[i]) if i < grid.size - 1 else grid[i]
            out.append((xN - xc) * q)
    return out


def _init_geom(prof, hyp, span, s_mod=BLUR_INIT):
    """Geometry implied by a span: the span ends are where the level crosses 72 % of the observed range, so the
    outer bar edges are placed using the modelled crossing offsets of this symbology's end elements."""
    white, dark = _levels(prof)
    A = dark - 0.15 * (white - dark)
    B = white - A
    lstar = (dark + 0.72 * (white - dark) - A) / B
    L = span[1] - span[0]
    xN = hyp.xN()
    q = L / xN
    offl = offr = 0.0
    for _ in range(3):
        offl, offr = _crossing_offsets(hyp, q, s_mod * q, lstar)
        q = (L + offl + offr) / xN
    u0 = span[0] - offl
    return dict(u0=u0, q=q, c=0.0, s=s_mod * q, A=A, B=B, g=0.0, d=0.0, R=hyp.R, uc=0.5 * (span[0] + span[1]))


def _path_to_pats(hyp, dp, path):
    pats = [0] * len(hyp.slots)
    ts = [sl["ts"][0] for sl in hyp.slots]
    for pi, j in enumerate(dp.V):
        t, c = dp.decode_state(pi, path[pi])
        pats[j], ts[j] = c, t
    return pats, ts


def _quiet_ok(prof, g, xN, qz=QUIET_DEFAULT):
    """Observed quiet zone of qz modules on both sides, inside the frame."""
    ul = _u_of_x(-qz, g); ur = _u_of_x(xN + qz, g)
    if ul < prof.u_frame_lo or ur > prof.u_frame_hi:
        return False
    return prof.u[0] <= ul and prof.u[-1] >= ur


def _gate(chi, nsel, nfree):
    nu = max(nsel - nfree, 1)
    return chi <= _chi2.isf(ALPHA, nu), nu


def _free_params(hyp):
    f = ["u0", "q", "s", "A", "B", "g", "d", "c"]
    if hyp.two_width:
        f.append("R")
    return f


def _solve(prof, hyp, g, n_alt=3, multistart=False):
    """Alternate DP (sequence) and LM (geometry).  Returns (g, pats, ts, chi, nsel), or dict(early=chi2/nu) if
    the fit is hopeless.  With multistart, a failed first round is retried from the best (by DP cost) of the
    end positions shifted by +-0.35 module (the span-derived ends can be off by a fraction of a module)."""
    res = _solve1(prof, hyp, g, n_alt)
    if not multistart or not isinstance(res, dict):
        return res
    xN = hyp.xN(g["R"])
    starts = []
    for d0 in (-0.35, 0.0, 0.35):
        for d1 in (-0.7, -0.35, 0.0, 0.35, 0.7):      # the far end carries the scale error
            if d0 == 0.0 and d1 == 0.0:
                continue
            g2 = dict(g)
            g2["u0"] = g["u0"] + d0 * g["q"]
            g2["q"] = g["q"] * (1.0 + (d1 - d0) / xN)
            dp = make_dp(prof, hyp, g2)
            if not dp.ok:
                continue
            cost, path = viterbi(dp.U0, dp.M, dp.UL)
            if path is not None:
                starts.append((cost / dp.sel.size, g2))
    for _, g2 in sorted(starts, key=lambda z: z[0])[:2]:
        res2 = _solve1(prof, hyp, g2, n_alt)
        if not isinstance(res2, dict):
            return res2
        if res2["early"] < res["early"]:
            res = res2
    return res


def _solve1(prof, hyp, g, n_alt=3):
    free0 = ["u0", "q", "s", "A", "B"]
    pats = ts = None
    last = None
    chi, nsel = np.inf, 0
    for it in range(n_alt):
        dp = make_dp(prof, hyp, g)
        if not dp.ok:
            return None
        cost, path = viterbi(dp.U0, dp.M, dp.UL)
        if path is None:
            return None
        pats, ts = _path_to_pats(hyp, dp, path)
        if last is not None and pats == last:
            break
        last = pats
        E4 = full_edges(hyp, pats, ts)
        free = free0 if it == 0 else _free_params(hyp)
        g, chi, nsel, _ = refine(prof, hyp, E4, g, free, iters=8 if it < n_alt - 1 else 12)
        if not np.isfinite(chi):
            return None
        if chi / max(nsel - len(free), 1) > EARLY_EXIT:
            return dict(early=chi / max(nsel - len(free), 1))
    return g, pats, ts, chi, nsel


def _fit_angle(img, reg, prof, hyp, pats, ts, g, rev):
    """Model-based bar-angle correction: fit a shear u -> u - v*phi on strip profiles with the sequence fixed.
    Returns the corrected bar-normal angle (radians) of the un-reversed frame."""
    K = 4 if (prof.v_hi - prof.v_lo) >= 16 else 2
    sp = extract_profile(img, reg, v_frac=getattr(prof, "v_frac", (0.0, 1.0)), theta=prof.theta, strips=K)
    if sp is None:
        return prof.theta
    if rev:
        sp = _reverse(sp)
    E4 = full_edges(hyp, pats, ts)
    g0 = dict(g); g0["phi"] = 0.0
    g1, chi, n1, _ = refine(sp, hyp, E4, g0, _free_params(hyp) + ["phi"], iters=10)
    phi = g1.get("phi", 0.0) if np.isfinite(chi) else 0.0
    if rev:
        phi = -phi
    return prof.theta - phi


def _finish(img, reg, prof0, rev, hyp, g, formats, fmts_zx, debug, multistart=False):
    """Angle correction, final checksum-constrained DP with competitor, gate, zxing.  Returns a candidate
    dict or None."""
    P = _reverse(prof0) if rev else prof0
    res = _solve(P, hyp, g, multistart=multistart)
    if res is None:
        return None
    if isinstance(res, dict):
        return dict(reject="em", chi2nu=res["early"])
    g, pats, ts, chi, nsel = res
    free = _free_params(hyp)
    nu0 = max(nsel - len(free), 1)
    if chi / nu0 > 400:
        return None                     # hopeless hypothesis: skip the expensive steps
    # bar angle from the model, then a fresh profile at that angle (skipped when the fit is already at noise)
    th = prof0.theta if chi / nu0 <= 1.0 else _fit_angle(img, reg, prof0, hyp, pats, ts, g, rev)
    if abs(th - prof0.theta) > np.radians(0.005):
        prof1 = extract_profile(img, reg, v_frac=getattr(prof0, "v_frac", (0.0, 1.0)), theta=th)
        if prof1 is not None:
            prof1.v_frac = getattr(prof0, "v_frac", (0.0, 1.0))
            prof0 = prof1
            P = _reverse(prof0) if rev else prof0
            E4 = full_edges(hyp, pats, ts)
            g, chi, nsel, _ = refine(P, hyp, E4, g, free, iters=12)
    dp = make_dp(P, hyp, g)
    if not dp.ok:
        return None
    aux = _aux(hyp, dp, formats)
    c1, path1, c2, path2 = viterbi(dp.U0, dp.M, dp.UL, aux=aux, second=True)
    if path1 is None:
        return None
    pats1, ts1 = _path_to_pats(hyp, dp, path1)
    E4 = full_edges(hyp, pats1, ts1)
    g1, chi1, n1, _ = refine(P, hyp, E4, g, free, iters=12)
    xN = hyp.xN(g1["R"])
    passed, nu = _gate(chi1, n1, len(free))
    rec = dict(family=hyp.family, n=hyp.n, T=hyp.T, rev=rev, chi2=round(chi1, 2), nu=nu, gate=bool(passed),
               R=round(g1["R"], 3), ppm=round(g1["q"], 3), s_mod=round(g1["s"] / g1["q"], 3),
               quiet=bool(_quiet_ok(P, g1, xN, _qz(hyp))), theta=round(float(np.degrees(prof0.theta)), 3),
               noise=tuple(round(x, 3) for x in prof0.noise))
    if debug is not None:
        debug.setdefault("tried", []).append(rec)
    rec["blur_ok"] = bool(g1["s"] < 0.98 * S_MOD_MAX * g1["q"])
    if not (passed and rec["quiet"] and rec["blur_ok"]):
        return dict(reject="gate", chi2nu=chi1 / nu)
    zx = zxing_validate(render_clean(hyp, pats1, ts1, g1["R"]), fmts_zx)
    if not zx:
        rec["zxing"] = "invalid"
        return dict(reject="zxing", chi2nu=chi1 / nu)
    fmt, text = zx[0]
    rec.update(fmt=fmt, text=text)
    amb = None
    if path2 is not None:
        pats2, ts2 = _path_to_pats(hyp, dp, path2)
        g2, chi2v, n2, _ = refine(P, hyp, full_edges(hyp, pats2, ts2), g1, free, iters=12)
        pass2, _ = _gate(chi2v, n2, len(free))
        rec["chi2_competitor"] = round(chi2v, 2)
        if pass2:
            zx2 = zxing_validate(render_clean(hyp, pats2, ts2, g2["R"]), fmts_zx)
            if zx2 and norm_text(*zx2[0]) != norm_text(fmt, text):
                amb = zx2[0]
    rec["ambiguous"] = amb is not None
    ua, ub = _u_of_x(0.0, g1), _u_of_x(xN, g1)
    if rev:
        ua, ub = -ub, -ua
    if amb is not None:
        return dict(ambiguous=True, fmt=fmt, text=text, other=amb, span=(ua, ub))
    c, n, t = prof0.c, prof0.n, prof0.t
    quad = [c + n * ua + t * prof0.v_lo, c + n * ub + t * prof0.v_lo, c + n * ub + t * prof0.v_hi,
            c + n * ua + t * prof0.v_hi]
    return dict(ambiguous=False, fmt=fmt, text=text, quad=np.array(quad).tolist(), chi2=chi1, nu=nu,
                ppm=g1["q"], s_mod=g1["s"] / g1["q"], family=hyp.family, span=(ua, ub))


SCAN_K = 2              # characters at each end used by the hypothesis scan
FINISH_RATIO = 30.0     # refine hypotheses whose scan score is within this factor of the best


def scan_hypotheses(prof, formats):
    """Score every (direction, span, family, length, R) hypothesis with the DP on the first and last SCAN_K
    characters at the span-implied geometry.  Returns a list sorted by score (mean weighted sq. residual)."""
    fams = []
    for f in formats:
        fam = _FAMILY_OF.get(f)
        if fam and fam not in fams:
            fams.append(fam)
    scored = []
    spans = _spans(prof)
    for rev in (False, True):
        P = _reverse(prof) if rev else prof
        for span0 in spans:
            span = (-span0[1], -span0[0]) if rev else span0
            L = span[1] - span[0]
            flat, dips = _span_bounds(P, span)
            for fam in fams:
                for hyp in _enumerate(fam, L, formats, flat, dips):
                    g = _init_geom(P, hyp, span)
                    f = max(1, int(g["q"] / (2.5 * P.bw)))
                    dp = FastDP(_decimate(P, f), hyp, g, scan_k=SCAN_K)
                    if not dp.ok or not np.isfinite(dp.scan[0]):
                        continue
                    scored.append((dp.scan[0] / max(dp.scan[1], 1), rev, span, hyp, g))
    scored.sort(key=lambda z: z[0])
    return scored


def decode_profile(img, reg, prof, formats, fmts_zx, deadline=None, debug=None, scored=None):
    """Gated candidates of one profile (both directions).  `scored` (from scan_hypotheses) may be passed in
    to reuse the scan of another band of the same region."""
    if scored is None:
        scored = scan_hypotheses(prof, formats)
    if debug is not None:
        debug["scored"] = [(round(z[0], 2), z[1], z[3].key()) for z in scored[:12]]
        debug["nscored"] = len(scored)
    if not scored:
        return [], scored
    best = scored[0][0]
    cands = []
    ok_score = None
    cbars = []
    for rank, z in enumerate(scored[:TOPK]):
        sc, rev, span, hyp, g = z
        if sc > FINISH_RATIO * best or (ok_score is not None and sc > 3.0 * ok_score):
            break
        if deadline is not None and time.perf_counter() > deadline:
            break
        cd = _finish(img, reg, prof, rev, hyp, g, formats, fmts_zx, debug, multistart=rank < 3)
        if hyp.family == "Codabar" and cd is not None:
            cbars.append((cd.get("chi2nu", 0.0), z))
        if cd is not None and cd.get("reject") is None:
            cands.append(cd)
            ok_score = sc if ok_score is None else ok_score
    # bases for the 3-wide fallback: the first two Codabar hypotheses in scan order and the two with the lowest
    # refined chi-square (neither ranking alone is reliable when the length is wrong)
    bases = [z for _, z in cbars[:2]] + [z for _, z in sorted(cbars, key=lambda c: c[0])[:2]]
    cbars = []
    for z in bases:
        if all(z is not y for y in cbars):
            cbars.append(z)
    if cbars and ok_score is None:
        # Codabar 3-wide data characters (: / . +) lengthen the symbol by T*(R-1) modules, which the T=0 scan
        # cannot represent: try the (n', T <= 3) whose total length is within 10 % of the best Codabar scan
        # lengths, nearest first
        tried = set()
        per_base = []
        for sc, rev, span, hyp, g in cbars:
            N0 = hyp.xN()
            mine = []
            for n2 in range(max(1, hyp.n - 3), hyp.n + 2):
                for T in (1, 2, 3):
                    if T <= n2:
                        h2 = Hyp("Codabar", n2, hyp.R, T)
                        dN = abs(h2.xN() - N0)
                        if dN <= 0.1 * N0:
                            mine.append((dN, rev, span, h2))
            per_base.append(sorted(mine, key=lambda a: a[0])[:4])     # each base's 4 nearest lengths
        alts = []
        for i in range(4):                                           # round-robin over the bases
            for mine in per_base:
                if i < len(mine):
                    dN, rev, span, h2 = mine[i]
                    key = (h2.n, h2.T, h2.R, rev)
                    if key not in tried:
                        tried.add(key); alts.append(mine[i])
        for dN, rev, span, h2 in alts:
            if deadline is not None and time.perf_counter() > deadline:
                break
            cd = _finish(img, reg, prof, rev, h2, _init_geom(_reverse(prof) if rev else prof, h2, span),
                         formats, fmts_zx, debug)
            if cd is not None and cd.get("reject") is None:
                cands.append(cd)
                break
    return cands, scored


def warm_up(max_chars=60):
    """Precompute the length-dependent character structures (about a second, once per process).  Optional:
    without it the first frames of a process pay this cost lazily."""
    for fam in ("EAN13", "EAN8", "UPCE"):
        _fast_struct(Hyp(fam))
    for n in range(1, max_chars):
        for fam in ("Code128", "Code39", "ITF", "Codabar"):
            _fast_struct(Hyp(fam, n))


def read_1d(frame, formats, budget_s=None, debug=None):
    """frame: 2-D uint8 mono image.  formats: names from ('EAN13','UPCA','UPCE','EAN8','Code128','Code39',
    'Codabar','ITF').  Returns list of dict(format, text, quad_px, passes='subpixel...') -- texts decoded and
    validated by zxing-cpp from the reconstructed element sequence; gated against the measured noise."""
    t0 = time.perf_counter()
    deadline = None if budget_s is None else t0 + budget_s
    formats = tuple(f for f in formats if f in _FAMILY_OF)
    if not formats:
        return []
    img = np.ascontiguousarray(frame)
    if img.ndim == 3:
        img = img.mean(2)
    fmts_zx = _formats(formats)
    regs = find_regions(img)
    if debug is not None:
        debug["regions"] = [dict(c=r["c"].tolist(), n=r["n"].tolist(), uh=r["uh"], vh=r["vh"], coh=r["coh"]) for r in regs]
        debug["regions_detail"] = []
    out = []
    for reg in regs:
        if deadline is not None and time.perf_counter() > deadline:
            break
        allc = []
        scored = None
        for vf in ((0.0, 1.0), (0.0, 0.5), (0.5, 1.0)):
            prof = extract_profile(img, reg, vf)
            if prof is None:
                continue
            prof.v_frac = vf
            dbg = {} if debug is not None else None
            cands, sc = decode_profile(img, reg, prof, formats, fmts_zx, deadline, dbg, scored)
            if scored is None:
                scored = sc            # sub-bands reuse the full band's hypothesis ranking
            if debug is not None:
                debug["regions_detail"].append(dict(vf=vf, **dbg))
            allc += cands
            if cands:
                break  # the full band (or first sub-band) settled this region
        # one symbol per location: candidates whose extents along the bar normal overlap compete; if they
        # disagree (or any of them had a passing competitor) the location yields nothing
        groups = []
        for cd in allc:
            for gr in groups:
                if cd["span"][0] < gr["span"][1] and gr["span"][0] < cd["span"][1]:
                    gr["c"].append(cd); gr["span"] = (min(gr["span"][0], cd["span"][0]), max(gr["span"][1], cd["span"][1]))
                    break
            else:
                groups.append(dict(span=tuple(cd["span"]), c=[cd]))
        for gr in groups:
            texts = {norm_text(cd["fmt"], cd["text"]) for cd in gr["c"]}
            if any(cd["ambiguous"] for cd in gr["c"]) or len(texts) != 1:
                continue
            cd = next(c for c in gr["c"] if not c["ambiguous"])
            out.append(dict(format=cd["fmt"], text=cd["text"], quad_px=cd["quad"],
                            passes=f"subpixel:{cd['family']}:ppm{cd['ppm']:.2f}:blur{cd['s_mod']:.2f}"))
    # de-duplicate identical texts from overlapping regions
    uniq = {}
    for o in out:
        uniq.setdefault((o["format"], o["text"]), o)
    return list(uniq.values())
