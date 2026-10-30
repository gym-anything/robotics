"""The scanner's decoder: zxing-cpp on each 8-bit frame, as a site-configured imager would run it.

Strategy (measured): zxing's three binarizers (LocalAverage, GlobalHistogram,
FixedThreshold), each on the native frame and on a 2x bicubic upsample, stopping at the first pass that
yields any symbol. The extra passes removed non-monotonic failures (a decode that fails at a closer
distance than one that succeeds) found with the default binarizer alone.

Choice among several symbols in one frame (picklist disabled, the DS2208 default): the decoded
symbol whose quadrilateral centre is nearest the image centre. With picklist enabled only a symbol whose
quadrilateral contains the aim point's image position is reported. The real firmware's order is
unpublished; this rule is authored and disclosed.
"""

import numpy as np

FORMATS = {
    "retail": ("EAN13", "UPCA", "UPCE", "EAN8"),
    "inventory": ("EAN13", "UPCA", "UPCE", "EAN8", "Code128", "ITF", "DataMatrix", "QRCode"),
    "library": ("EAN13", "UPCA", "Codabar", "Code128"),
}


def _formats(names):
    import zxingcpp
    return tuple(getattr(zxingcpp.BarcodeFormat, n) for n in names)


def gtin14(text):
    """Normalise an EAN/UPC reading (UPC-A as 12 or 13 digits, EAN-13, UPC-E expanded by zxing) to GTIN-14."""
    t = "".join(c for c in text if c.isdigit())
    return t.zfill(14)


def normalise(fmt, text):
    fmt = fmt.replace("-", "").replace(" ", "")
    if fmt == "Codabar" and len(text) > 2 and text[0] in "ABCD" and text[-1] in "ABCD":
        return text[1:-1]
    if fmt in ("EAN13", "UPCA", "UPCE", "EAN8", "ITF"):
        return gtin14(text) if fmt != "ITF" or len(text) == 14 else text
    return text


def expected(kind, data):
    """The normalised text a correct decode of a printed symbol (artwork.Symbol kind, data) produces."""
    if kind == "UPCE":
        from .artwork import gtin, upce_expand
        return gtin14(gtin(upce_expand(data[:7])))
    return normalise(kind, data)


def _quad(position):
    p = position
    return np.array([[p.top_left.x, p.top_left.y], [p.top_right.x, p.top_right.y],
                     [p.bottom_right.x, p.bottom_right.y], [p.bottom_left.x, p.bottom_left.y]], float)


def _inside(q, pt):
    """Point in convex quadrilateral (any winding), with a 2 px tolerance for 1-D symbols (zero-height quads)."""
    x, y = pt
    xs, ys = q[:, 0], q[:, 1]
    if ys.max() - ys.min() < 4 or xs.max() - xs.min() < 4:          # degenerate (1-D line quad): distance test
        a, b = q[0], q[1] if np.linalg.norm(q[1]-q[0]) > 1 else q[2]
        ab = b - a
        t = np.clip(np.dot(np.asarray(pt) - a, ab)/max(np.dot(ab, ab), 1e-9), 0, 1)
        return np.linalg.norm(a + t*ab - np.asarray(pt)) < 6.
    sign = None
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        c = (b[0]-a[0])*(y-a[1]) - (b[1]-a[1])*(x-a[0])
        if abs(c) < 1e-9:
            continue
        s = c > 0
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


FAMILY = {"EAN13": "UPC/EAN", "UPCA": "UPC/EAN", "UPCE": "UPC/EAN", "EAN8": "UPC/EAN"}
# The flat-field last stage of read_all: an imager binarizes against the local white level, so a skewed card, lit unevenly
# by the scanner's own LEDs, still reads (-60/+55 deg for Code 39 20 mil and +-50 deg for UPC-A, against +-25/+-30
# deg without it; the DS2208: +-65). On the decode-range bench it misread UPC-A twice beyond its far limit; with the misread check
# below those two are gone (0 misreads, the bench's UPC-A row to 203 mm). BS_DECODER_FLAT_FIELD=0 turns it off.
import os as _os
FLAT_FIELD_STAGE = _os.environ.get("BS_DECODER_FLAT_FIELD", "1") != "0"
# The DS2208's default Security Level 1 "eliminates most misdecodes"; the six-pass consensus alone does not for
# UPC/EAN near the decoder's limit: on the front end's synthetic bench it returned a checksum-valid wrong UPC-E or
# EAN-13 in 11 frames at 1.1-1.5 px per module, every pass agreeing. Each 1-D read is
# therefore checked against a physical edge model of its neighbourhood (veto_1d). BS_DECODER_VETO=0 turns the check
# off for comparisons.
VETO_1D = _os.environ.get("BS_DECODER_VETO", "1") != "0"
MATRIX = ("DataMatrix", "QRCode", "PDF417")
# symbologies whose read by a single pass is reported (their Reed-Solomon check taken as conclusive); every other
# symbology needs two passes to agree (_read). PDF417 is not among them: zxing's PDF417 decoder, which uses erasures,
# returned a wrong 22-character text from one pass on the decode-range bench (107 mm, one frame of at least 1 695), and requiring
# two passes removes it at a cost of 14 mm of far range (115 -> 101 mm; datasheet 145 mm). Requiring two
# for Data Matrix rejected valid reads at close range, and neither Data Matrix nor QR has misread
ONE_PASS = ("DataMatrix", "QRCode")
# the DS2208's default symbol lengths (PRG ch. 12, "Set Lengths for ...": the number of human-readable characters
# including check digits, for Codabar without its start and stop characters): a read of another length is not
# reported. Without it, a blurred Code 128 5 mil card rolled 25-125 deg decoded as the Codabar "C82D" (two data
# characters) in 17 frames of a roll sweep, agreed by two passes
LENGTHS = {"Code128": (1, 55), "Code39": (1, 55), "Codabar": (4, 55), "ITF": (6, 55)}
QUIET_MODULES = {"UPCA": 9, "EAN13": 7, "UPCE": 7, "EAN8": 7, "Code128": 10, "Code39": 10, "Codabar": 10, "ITF": 10}
_MODULES = {"UPCA": 95, "EAN13": 95, "UPCE": 51, "EAN8": 67}


def _modules(fmt, text):
    """Bar-pattern length in modules of a decoded 1-D symbol (re-encoded from its text; Code 39 / Codabar / ITF at
    wide:narrow 2.5-3, so the module width estimate for those is good to about +-15 %)."""
    if fmt in _MODULES:
        return _MODULES[fmt]
    try:
        from .artwork import Symbol
        kind = fmt if fmt in ("Code128", "Code39", "ITF") else "Codabar" if fmt == "Codabar" else None
        if kind is None:
            return None
        data = text if kind != "Codabar" or text[:1] in "ABCD" else f"A{text}B"
        return Symbol(kind, data, 1.).width_modules
    except Exception:
        return None


def _quiet_zones_in_frame(fmt, text, quad, width, height):
    """A 1-D symbol is only delimited by its quiet zones: both must lie inside the frame (a real imager cannot tell
    a symbol's end from the frame edge; the datasheet's near limits are 'limited by width of bar code in field of
    view'). The module width comes from the decoded symbol's own module count."""
    q = np.asarray(quad, float)
    left, right = (q[0] + q[3])/2, (q[1] + q[2])/2
    axis = right - left
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return False
    mods = _modules(fmt, text)
    m = QUIET_MODULES.get(fmt, 10)*(n/mods if mods else .06*n/10)
    for pt in (left - m*axis/n, right + m*axis/n):
        if not (0 <= pt[0] <= width - 1 and 0 <= pt[1] <= height - 1):
            return False
    return True


def _length_ok(fmt, text):
    lo_hi = LENGTHS.get(fmt)
    if lo_hi is None:
        return True
    if fmt == "Codabar" and len(text) >= 2 and text[0] in "ABCDabcd" and text[-1] in "ABCDabcd":
        text = text[1:-1]
    return lo_hi[0] <= len(text) <= lo_hi[1]


# Decode security for 1-D symbols, the second half of this environment's model of the DS2208's default Security
# Level 1: zxing accepts a 1-D symbol once two scan lines decode it to the same text and reports the first and last
# agreeing lines as its quad's edges; a consensus read is reported only if those lines lie at least SCAN_SEP_PX apart
# across the bars in one of its agreeing passes (line_sep_px), unless the sub-pixel front end, which fits the whole bar
# height, reads the same text on the symbol (veto_1d). With two adjacent rows, the consensus read the marinara jar's
# UPC-A in the checkout run as the checksum-valid EAN-13 8012871001583 from rows 196 and 197 alone, which graze the tops
# of its bars (they stand at uneven heights on the curved label). Measured on every recorded task frame.
# BS_DECODER_LINES=0 turns the check off for comparisons.
SCAN_LINES = _os.environ.get("BS_DECODER_LINES", "1") != "0"
SCAN_SEP_PX = 4.
# The sub-pixel front end as the last 1-D stage of a frame in which the consensus reported no 1-D symbol: zxing needs
# about 1.2-1.5 pixels per module and reaches 0.28-0.65 of the DS2208's published 1-D far limits on the decode-range bench; the
# front end reads to about 0.7 ppm, as commercial imager decoders do (frontend.py).
# BS_DECODER_FRONTEND=0 turns it off for comparisons.
FRONTEND_STAGE = _os.environ.get("BS_DECODER_FRONTEND", "1") != "0"


def read_all(frame, formats):
    """Symbols in the frame, each dict(format, text, quad_px, passes, votes); with `roll_deg` where they were read
    from the frame turned to level their bars: the consensus decoder (_consensus); each 1-D read checked against the
    sub-pixel front end's edge model of its neighbourhood (veto_1d; VETO_1D) and, unless the front end confirmed it,
    for the separation of the scan lines that agreed on it (SCAN_LINES); then, if no 1-D read is left, the front end on
    the whole frame (FRONTEND_STAGE)."""
    found = _consensus(frame, formats)
    if VETO_1D:
        found = veto_1d(found, frame, formats)
    if SCAN_LINES:
        found = [f for f in found if f["format"] in MATRIX or f.get("confirmed") or f.get("line_sep_px", 0.) >= SCAN_SEP_PX]
    fm = tuple(x for x in formats if x not in MATRIX)
    if FRONTEND_STAGE and fm and not any(f["format"] not in MATRIX for f in found):
        from . import frontend as FE
        found = found + [dict(g, votes=None, frontend=True) for g in FE.read_1d(np.ascontiguousarray(frame), fm)
                         if _length_ok(g["format"], g["text"])]
    return found


def _consensus(frame, formats):
    """Symbols in the frame by the six-pass consensus, levelled where the bars are rolled (read_all without the veto).

    An area imager decodes 1-D symbols at any roll (DS2208: roll 0-360 deg). zxing's 1-D readers scan rows (and
    columns, try_rotate), so a symbol rolled by more than its bar height allows is missed: measured on recorded
    frames, a 16-character Codabar library label read only within 5 deg of level, an EAN-13 within 5 deg, a Code 128
    within 10 deg. The level passes run first; only when they report nothing, the bars' orientation is found
    from the frame's gradient structure (bar_angles) and the same passes run on the frame turned to level them, as an
    imager locates a symbol and samples along its axis. Positions are mapped back to the frame, so the quiet-zone
    rule and the choice among symbols see the true frame."""
    found = _read(frame, formats, 0.)
    if found:
        if any(f["format"] not in MATRIX for f in found):
            # a row scan across bars near 45 deg can decode a checksum-valid wrong UPC/EAN, agreed by several passes
            # (a clean UPC-A and EAN-13 at 45 and 135 deg): where the bars lie 30 deg or more from level, the
            # levelled frame is decoded too, and its read of the same family at the same place replaces a different one
            for ang in [a for a in bar_angles(frame) if abs(a) >= 30.]:
                found = [_prefer_levelled(f, _read(frame, formats, ang), ang) for f in found]
        return found
    for ang in bar_angles(frame):
        found = _read(frame, formats, ang)
        if found:
            return [dict(f, roll_deg=round(float(ang), 1)) for f in found]
    if not FLAT_FIELD_STAGE:
        return []
    # uneven illumination: a card skewed 30 deg to the window is lit about 1.7 times as brightly on its near side as on its
    # far side, with the dark room beyond its edge in the same rows, and zxing thresholds each 1-D row as a whole: a
    # skew sweep reads only to +-25-30 deg (DS2208: +-65 deg). An imager binarizes against the local white
    # level; this stage divides the frame by it (the 15 px running maximum, smoothed over 20 px) and decodes again
    flat = flat_field(frame)
    found = _read(flat, formats, 0.)
    if found:
        return [dict(f, flat_field=True) for f in found]
    for ang in bar_angles(flat):
        found = _read(flat, formats, ang)
        if found:
            return [dict(f, roll_deg=round(float(ang), 1), flat_field=True) for f in found]
    return []


def flat_field(frame, size=15, sigma=20., floor=30.):
    """The frame divided by its local white level (running maximum over `size` px, Gaussian-smoothed over `sigma` px;
    the level is floored at `floor` counts so that dark surroundings are not amplified into noise), scaled to 200."""
    from scipy import ndimage
    f = np.asarray(frame, np.float32)
    white = ndimage.gaussian_filter(ndimage.maximum_filter(f, size=size), sigma)
    return np.clip(f/np.maximum(white, floor)*200., 0, 255).astype(np.uint8)


def _prefer_levelled(f, levelled, ang):
    if f["format"] in MATRIX:
        return f
    q = np.asarray(f["quad_px"]); c = q.mean(0)
    size = max(np.ptp(q[:, 0]), np.ptp(q[:, 1]), 24.)
    fam = FAMILY.get(f["format"], f["format"])
    for g in levelled:
        qg = np.asarray(g["quad_px"])
        if FAMILY.get(g["format"], g["format"]) == fam and np.linalg.norm(qg.mean(0) - c) < max(.5*max(size, np.ptp(qg[:, 0]), np.ptp(qg[:, 1])), 12.) \
                and normalise(g["format"], g["text"]) != normalise(f["format"], f["text"]):
            return dict(g, roll_deg=round(float(ang), 1), replaced=f["text"])
    return f


def bar_angles(frame, block=16, max_peaks=2):
    """Candidate roll angles (deg, in (-45, 45], level +-3 deg excluded) of bar patterns in the frame: the dominant
    gradient direction of every 16 x 16 block (structure tensor of Sobel gradients after a 1.5 px Gaussian; plain
    central differences on sharp bar edges read a 10 deg roll as 12.1 deg and a 30 deg one as 34.1 deg, too far off for
    a Codabar that decodes only within a few degrees of level), kept where the gradients are strong (rms above 5
    counts per pixel: a library label's narrow modules keep little gradient after the smoothing) and coherent (a bar pattern's gradients all point across the bars), histogrammed at 1 deg modulo
    90 deg (the 90 deg turns are zxing's own try_rotate); peaks of their own (local maxima, a tenth of the weight, at
    least 3 blocks) refined by their weighted mean."""
    from scipy import ndimage
    g = ndimage.gaussian_filter(np.asarray(frame, np.float32), 1.5)
    gx, gy = ndimage.sobel(g, axis=1)/8, ndimage.sobel(g, axis=0)/8
    h, w = g.shape[0]//block*block, g.shape[1]//block*block
    blk = lambda a: a[:h, :w].reshape(h//block, block, w//block, block).mean(axis=(1, 3)).ravel()
    jxx, jyy, jxy = blk(gx*gx), blk(gy*gy), blk(gx*gy)
    energy = jxx + jyy
    coh = np.sqrt((jxx - jyy)**2 + 4*jxy**2)/np.maximum(energy, 1e-6)
    keep = (energy > 5.**2) & (coh > .75)
    if keep.sum() < 3:
        return []
    phi = np.degrees(.5*np.arctan2(2*jxy[keep], (jxx - jyy)[keep]))       # gradient direction, (-90, 90]
    phi = (phi + 45.) % 90. - 45.                                          # modulo the 90 deg turns
    wgt = energy[keep]
    hist = np.zeros(90)
    np.add.at(hist, np.floor(phi + 45.).astype(int) % 90, wgt)
    count = np.zeros(90)
    np.add.at(count, np.floor(phi + 45.).astype(int) % 90, 1)
    # a candidate is a peak of its own: a local maximum of the histogram smoothed over +-2 deg (circular), holding at
    # least a tenth of the coherent weight. The shoulders of the level peak are not candidates: a level card's edges
    # spread +-5 deg at the far limit, and trying their +-4 deg tails decoded a 5-character Codabar out of noise
    smooth = sum(np.roll(hist, k)*w_ for k, w_ in ((-2, 1.), (-1, 2.), (0, 3.), (1, 2.), (2, 1.)))
    local = np.array([smooth[k] >= smooth[[(k + j) % 90 for j in (-3, -2, -1, 1, 2, 3)]].max() for k in range(90)])
    share = np.array([hist[[(k + j) % 90 for j in range(-3, 4)]].sum() for k in range(90)])/max(hist.sum(), 1e-9)
    out = []
    for k in np.argsort(-smooth):
        if len(out) >= max_peaks or smooth[k] <= 0:
            break
        if not local[k] or share[k] < .1:
            continue
        near = np.abs(((phi - (k - 45 + .5)) + 45.) % 90. - 45.) <= 3.
        if count[[(k + j) % 90 for j in range(-3, 4)]].sum() < 3 or not near.any():
            continue
        # weighted mean on the 4x angle circle (period 90 deg)
        a = np.radians(4*phi[near])
        ang = np.degrees(np.arctan2((wgt[near]*np.sin(a)).sum(), (wgt[near]*np.cos(a)).sum()))/4
        if abs(ang) < 3. or any(abs(((ang - o) + 45.) % 90. - 45.) < 6. for o in out):
            continue
        out.append(float(ang))
    return out


def _turned(img, deg):
    """The image with its content turned by -deg (so a gradient at deg lies along x), and the map from the turned
    image's pixel coordinates back to the image's."""
    from PIL import Image
    t = np.radians(deg)
    c, s_ = np.cos(t), np.sin(t)
    H, W = img.shape
    W2, H2 = int(np.ceil(abs(W*c) + abs(H*s_))), int(np.ceil(abs(W*s_) + abs(H*c)))
    ci, co = np.array([(W - 1)/2, (H - 1)/2]), np.array([(W2 - 1)/2, (H2 - 1)/2])
    R = np.array([[c, -s_], [s_, c]])                                     # turned -> image: p = ci + R (p' - co)
    off = ci - R @ co
    fill = int(np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])))
    out = Image.fromarray(img).transform((W2, H2), Image.AFFINE, (R[0, 0], R[0, 1], off[0], R[1, 0], R[1, 1], off[1]),
                                          resample=Image.BICUBIC, fillcolor=fill)
    return np.asarray(out), (lambda q: (np.asarray(q, float) @ R.T) + off)


def _read(frame, formats, roll_deg):
    """The six passes on the frame, turned by roll_deg first unless it is 0.

    All six passes run (zxing's LocalAverage, GlobalHistogram and FixedThreshold binarizers on the native frame and
    on a 2x bicubic upsample). Results are grouped by symbology family and location (quad centres closer than
    half the longer quad's length, at least 12 px); within a group a text is reported only if at least two passes
    produced it (one for Data Matrix and QR, ONE_PASS) and it has more than twice the votes of any competing text
    there. This is the stand-in for the
    imager's redundancy / security levels (the default "eliminates most misdecodes"): the decode-range bench found stock zxing
    returning a checksum-valid wrong UPC-A and Code 128 on blurred, pitched symbols in the same frame as the
    correct read."""
    import zxingcpp
    from PIL import Image
    fmts = _formats(formats)
    img = np.ascontiguousarray(frame)
    # zxing-cpp misses readable symbols near the frame border (a Code 128 with 30 modules of quiet zone, 68 px
    # from the left edge, read only once the frame was padded or cropped): pad by edge replication and map the
    # coordinates back; the quiet-zone rule below still uses the true frame bounds, so padding cannot supply a
    # quiet zone that lies outside the frame
    PAD = 48
    padded = np.pad(img, PAD, mode="edge")
    back = lambda q: q
    if roll_deg:
        padded, back = _turned(padded, roll_deg)
    hits = []
    for scale in (1, 2):
        src = padded if scale == 1 else np.asarray(Image.fromarray(padded).resize((padded.shape[1]*2, padded.shape[0]*2), Image.BICUBIC))
        for bname in ("LocalAverage", "GlobalHistogram", "FixedThreshold"):
            res = zxingcpp.read_barcodes(src, formats=fmts, binarizer=getattr(zxingcpp.Binarizer, bname),
                                         try_rotate=True, try_downscale=True, try_invert=False)
            for r in res:
                if not r.valid:
                    continue
                name = r.format.name if hasattr(r.format, "name") else str(r.format)
                fmt = str(name).replace("-", "").replace(" ", "")
                quad = back(_quad(r.position)/scale) - PAD
                if fmt not in MATRIX and not _quiet_zones_in_frame(fmt, r.text, quad, img.shape[1], img.shape[0]):
                    continue
                if not _length_ok(fmt, r.text):
                    continue
                hits.append(dict(format=fmt, text=r.text, quad_px=quad.tolist(), passes=f"{bname}x{scale}"))
    groups = []
    for h in hits:
        q = np.asarray(h["quad_px"]); c = q.mean(0)
        size = max(np.ptp(q[:, 0]), np.ptp(q[:, 1]), 24.)
        fam = FAMILY.get(h["format"], h["format"])
        for g in groups:
            if g["family"] == fam and np.linalg.norm(g["centre"] - c) < max(.5*max(g["size"], size), 12.):
                g["hits"].append(h); break
        else:
            groups.append(dict(family=fam, centre=c, size=size, hits=[h]))
    out = []
    for g in groups:
        votes = {}
        for h in g["hits"]:
            votes.setdefault(normalise(h["format"], h["text"]), []).append(h)
        ranked = sorted(votes.items(), key=lambda kv: -len({x["passes"] for x in kv[1]}))
        best, hs = ranked[0]
        n_best = len({x["passes"] for x in hs})
        n_next = len({x["passes"] for x in ranked[1][1]}) if len(ranked) > 1 else 0
        # Data Matrix and QR: one valid decode (requiring two passes rejected valid Data Matrix reads at close range);
        # 1-D symbols' checksums are weak, and zxing's PDF417 decoder misread from one pass: two passes
        need = 1 if g["family"] in ONE_PASS else 2
        if n_best >= need and n_best > 2*n_next:
            first = hs[0]
            out.append(dict(first, votes=n_best, rivals=[k for k, _ in ranked[1:]],
                            line_sep_px=round(max(_line_sep(h["quad_px"]) for h in hs), 2)))
    return out


def _line_sep(quad):
    """Distance (px) between the two scan lines that bound a 1-D read: zxing reports a linear symbol by the first and
    last rows that decoded it to the same text (its own two-line agreement), as the quad's top and bottom edges."""
    q = np.asarray(quad, float)
    a, b = q[0], q[1]
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-6:
        return 0.
    nrm = np.array([-d[1], d[0]])/n
    return float(max(abs(np.dot(q[3] - a, nrm)), abs(np.dot(q[2] - a, nrm))))


def veto_1d(found, frame, formats):
    """Model-consistency check of the consensus reads' 1-D symbols (with the consensus, this environment's model of the
    DS2208's default Security Level 1): the sub-pixel front end (frontend.read_1d: a physical edge model fitted to
    the frame, chi-square gated against the measured noise) re-reads the neighbourhood of each 1-D read (the read's
    extent and 0.75 of its longer side plus 16 px beyond it on every side, which holds the quiet zones), and the whole
    frame when it finds no symbol of the read's family on the same symbol there; the read is dropped when the front end
    reads a different text of the same family on the same symbol. Where the front end finds nothing the read stands (it
    never adds reads, so it cannot extend the range). The whole-frame look: a UPC-E (51 modules) misread off the left
    part of an EAN-13 at 1.1 px per module left the EAN-13's far end at the neighbourhood's edge, the front end read
    nothing there and the misread stood (six frames of the front end's bench with the flat-field stage on; a
    wider neighbourhood caught five of them). Measured on the decode-range bench (consensus+veto, every range as without it)
    and on the front end's synthetic bench."""
    one_d = [f for f in found if f["format"] not in MATRIX]
    if not one_d:
        return found
    from . import frontend as FE
    fm = tuple(f for f in formats if f not in MATRIX)
    img = np.asarray(frame)
    H, W = img.shape[:2]
    keep = []
    for f in found:
        if f["format"] in MATRIX:
            keep.append(f); continue
        q = np.asarray(f["quad_px"], float); c = q.mean(0); size = max(np.ptp(q[:, 0]), np.ptp(q[:, 1]), 24.)
        m = .75*size + 16.
        x0, x1 = int(max(0, q[:, 0].min() - m)), int(min(W, q[:, 0].max() + m + 1))
        y0, y1 = int(max(0, q[:, 1].min() - m)), int(min(H, q[:, 1].max() + m + 1))
        fam = FAMILY.get(f["format"], f["format"])
        on_symbol = lambda fe, off: [g for g in fe if FAMILY.get(g["format"], g["format"]) == fam
                                     and _same_symbol(np.asarray(g["quad_px"], float) + off, c, size)]
        seen = on_symbol(FE.read_1d(np.ascontiguousarray(img[y0:y1, x0:x1]), fm), (x0, y0))
        if not seen and (x0, y0, x1, y1) != (0, 0, W, H):
            seen = on_symbol(FE.read_1d(np.ascontiguousarray(img), fm), (0, 0))
        clash = [g for g in seen if normalise(g["format"], g["text"]) != normalise(f["format"], f["text"])]
        if not clash:
            keep.append(dict(f, confirmed=True) if seen else f)
    return keep


def _same_symbol(fe_quad, c, size):
    """Does a consensus read centred at c (scan-line length `size`) lie on the symbol the front end located (fe_quad,
    covering its bars)? zxing reports a 1-D symbol by one scan line, and the consensus can report one symbol from two
    rows 41 px apart (a UPC-E at 1.1 px per module on the front end's bench, 56 px long): the read must lie inside the
    front end's quadrilateral grown by a quarter about its centre, or within half its own length of that centre."""
    m = fe_quad.mean(0)
    return _inside(m + 1.25*(fe_quad - m), c) or np.linalg.norm(m - c) < max(.5*size, 12.)


def choose(found, width, height, aim_px=None, picklist=False):
    """The one symbol a DS2208-class imager reports from a frame (see module docstring)."""
    if not found:
        return None
    centre = np.array([width/2, height/2]) if aim_px is None else np.asarray(aim_px, float)
    if picklist:
        found = [f for f in found if _inside(np.asarray(f["quad_px"]), centre)]
        if not found:
            return None
    return min(found, key=lambda f: float(np.linalg.norm(np.asarray(f["quad_px"]).mean(0) - centre)))
