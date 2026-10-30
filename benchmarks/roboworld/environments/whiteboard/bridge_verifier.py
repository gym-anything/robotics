"""Bridge-copy verifier and progress reward, revision bridge-copy-v1.

The job is to copy the supplied diagram (Task.reference_view: the whole writing
surface as seen from the front) where it appears and at the size shown. Every
quantity comes from the final board pigment alone: no render, no VLM, and no
trajectory, stroke order, tool handling or intermediate milestone. The numbers
are documented scoring tolerances; the instruction does not
repeat them.

1. Visible ink. In each texel the pigment with the largest optical density
   (mean density x |absorption row| of the board colour law, ink.py) is the
   colour a viewer sees. Its weight rises linearly from 0 at density 0.02 to 1
   at 0.15.
2. Where the drawing is (placement fit). One similarity transform of the whole
   diagram (shift of its centre anywhere on the board, scale 0.85-1.15,
   rotation up to 5 deg) maximising blurred colour-matched overlap of all
   elements at once. A fit away from the placement shown must explain 1 % more
   of the weighted diagram per unit of normalised displacement. A board that
   fails there is also judged at the placement shown, and the judgement with
   the higher progress is kept, so one misplaced part cannot drag the others.
3. What is drawn (content fit). Fidelity and stray marks are read at the
   transform within 3.5 mm, 1.5 % and 0.75 deg of the placement fit that
   maximises fidelity, so parts drawn a few millimetres apart (one marker each)
   are judged against one common placement. Which samples count is fixed by
   the diagram; a sample that this fit moves onto another element's line keeps
   the better of its two readings.
4. Fidelity F. Each element's centreline is sampled every 1 mm. Along each
   sample's normal (never along the line) the ink-weighted centre of its own
   colour within 6 mm is found; credit is the visibility of the ink within 1 mm
   of that centre times 1 within 3.5 mm, falling linearly to 0 at 6 mm.
   Stretches where another element of the same colour meets this one, or a
   darker colour crosses it, belong to neither and are not sampled; along the
   line, such a stretch is passed over where visible ink of its colour or a
   darker one crosses it (at the placement fit) and counts as missing
   elsewhere. A skip of length g (sample count x spacing) between inked
   stretches of lengths a and b is filled with clip(min(a, b) / g - 1, 0, 1) x
   clip((6 mm - g) / 2 mm, 0, 1) of its neighbours' credit (a stretch running
   to the stroke's end does not limit it): short skips in a mostly inked line
   are filled, a row of dots or dashes is not. An element needs 90 % coverage
   and no stroke (hangers exempt) with an uncovered run, measured along the
   line, longer than 10 mm. Its credit is
   min(min(1, coverage / 0.9) x wave, 1 - gap excess / element length), so
   missing ink is counted once; wave is 0.5-1 for a water line whose drawn
   offset, regressed (with an intercept) on the shown wave at the placement
   fit, has slope above 0.7 (a straight line gives 1). Parts are the colour
   key's four; elements are equal within a part.
5. Cleanliness C. Visible ink of a colour farther than 3.5 mm plus the line's
   half-width (at least 2.5 mm) from that colour's lines counts as stray, with
   weight rising linearly to full over a further 3 mm; unused colours count in
   full. C = 1 - max(0, stray - 1 cm^2) / A, clipped to [0, 1].
6. Placement P (at the placement fit). Excess movement = shift beyond 30 mm +
   largest scale-caused movement beyond that of 5 % + largest rotation-caused
   movement beyond that of 2 deg; P = 1 / (1 + excess / 60 mm).

progress = F x C x P. success = every element check, stray <= 1 cm^2 and no
excess movement, which is exactly progress == 1. score = 1 on success, else
0.9 x progress, so every failure stays below every success.
"""

from functools import lru_cache
import math

import numpy as np
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw

from .config import COLORS
from .tasks import bridge_strokes

REVISION = "bridge-copy-v1"
PARTS = {"towers": "orange", "cables and hangers": "red", "roadway": "black", "water": "blue"}
LINE_TOLERANCE_M = 0.0035
PROXIMITY_LIMIT_M = 0.006
CENTRE_INK_M = 0.001  # ink must lie within this of the found centre
COVERAGE_REQUIRED = 0.90
SKIP_BRIDGED_M, SKIP_LIMIT_M = 0.004, 0.006  # skips filled in full up to 4 mm, none from 6 mm
GAP_ALLOWED_M = 0.010
GAP_INK_MIN = 0.05  # a centreline sample with less credit than this counts as part of a gap
WAVE_FOLLOW_MAX = 0.7  # slope of the drawn line's offset on the shown wave, with intercept; a straight line gives 1
VISIBLE_DENSITY, DRAWN_DENSITY = 0.02, 0.15
MARKER_HALF_WIDTH_M = 0.0025
LINE_WIDTH_M = {"thick": 0.006, "normal": 0.003, "thin": 0.0024}
JUNCTION_MARGIN_M = 0.0005
STRAY_ALLOWANCE_M2 = 1.0e-4
STRAY_RAMP_M = 0.003
SHIFT_ACCEPT_M, SCALE_ACCEPT, ROTATION_ACCEPT_DEG = 0.030, 0.05, 2.0
PLACEMENT_HALF_M = 0.060
SEARCH_SCALE, SEARCH_ROTATION_DEG, LOCAL_WINDOW_M = 0.15, 5.0, 0.150
CONTENT_SHIFT_M, CONTENT_SCALE, CONTENT_ROTATION_DEG = 0.0035, 0.015, 0.75
FAIL_SCORE_SCALE = 0.9
SAMPLE_SPACING_M = 0.001
OFFSET_STEP_M = 0.0005
COARSE_CELL_M = 0.003
COARSE_SCALES = (0.88, 0.94, 1.0, 1.06, 1.12)
COARSE_ROTATIONS_DEG = (-4.0, -2.0, 0.0, 2.0, 4.0)
REFINE_ROUNDS = 40
FIT_PREFERENCE = 0.01  # overlap a fit must gain per unit of normalised displacement from the placement shown

_CHANNEL = {c: i for i, c in enumerate(COLORS)}
_ABSORPTION = -np.log(np.maximum(np.asarray(list(COLORS.values())), 0.012))  # ink.py rgb()
_STRENGTH = np.linalg.norm(_ABSORPTION, axis=1)
_PART = {c: p for p, c in PARTS.items()}
_THETA = math.radians(SEARCH_ROTATION_DEG)


def _segment_distance(points, path):
    a, b = path[:-1], path[1:]
    d = b - a
    t = np.clip(((points[:, None, :] - a[None]) * d[None]).sum(-1) / np.maximum((d * d).sum(-1), 1e-18), 0, 1)
    return np.hypot(*(points[:, None, :] - (a[None] + t[..., None] * d[None])).transpose(2, 0, 1)).min(axis=1)


@lru_cache(maxsize=16)
def _geometry(width_m):
    """Centreline samples (metres from the drawing centre), normals, junction exclusion and weights."""
    strokes = bridge_strokes()
    elements = list(dict.fromkeys(s[0] for s in strokes))
    colour_of = {s[0]: s[1] for s in strokes}
    paths, rows = [np.asarray(s[3], float) * width_m for s in strokes], []
    for i, ((name, colour, weight, _), path) in enumerate(zip(strokes, paths)):
        cum = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
        arc = np.linspace(0, cum[-1], max(2, math.ceil(cum[-1] / SAMPLE_SPACING_M) + 1))
        x, z = np.interp(arc, cum, path[:, 0]), np.interp(arc, cum, path[:, 1])
        tx, tz = np.gradient(x), np.gradient(z)
        norm = np.hypot(tx, tz)
        dev = (z - z.mean()) * (tx / norm) if colour == "blue" else np.zeros_like(z)  # wave offset along the normal
        rows.append(dict(element=elements.index(name), colour=colour, weight=weight, points=np.c_[x, z],
                         normals=np.c_[-tz / norm, tx / norm], dev=dev, spacing=arc[1] - arc[0],
                         gap_rule=name != "hangers", length=cum[-1]))
    for i, row in enumerate(rows):
        keep = np.ones(len(row["points"]), bool)
        for j, (other, path) in enumerate(zip(rows, paths)):
            if j == i:
                continue
            half = LINE_WIDTH_M[other["weight"]] / 2
            if other["colour"] == row["colour"]:
                radius = max(half, MARKER_HALF_WIDTH_M) + JUNCTION_MARGIN_M
            elif _STRENGTH[_CHANNEL[other["colour"]]] > _STRENGTH[_CHANNEL[row["colour"]]]:
                radius = half + JUNCTION_MARGIN_M
            else:
                continue
            keep &= _segment_distance(row["points"], path) > radius
        row["included"] = keep
    counts = np.bincount(np.concatenate([[r["element"]] * len(r["points"]) for r in rows]))
    per_part = {p: sum(colour_of[e] == c for e in elements) for p, c in PARTS.items()}
    for row in rows:
        row["weights"] = np.full(len(row["points"]), 1 / (len(PARTS) * per_part[_PART[row["colour"]]]
                                                           * counts[row["element"]]))
        row["band"] = LINE_TOLERANCE_M + max(LINE_WIDTH_M[row["weight"]] / 2, MARKER_HALF_WIDTH_M)
    cat = lambda key: np.concatenate([r[key] for r in rows])  # noqa: E731
    included_length = np.zeros(len(elements))
    for row in rows:
        included_length[row["element"]] += row["included"].sum() * row["spacing"]
    points = cat("points")
    return dict(elements=tuple(elements), colour_of=colour_of, rows=rows, points=points, normals=cat("normals"),
                channel=np.asarray([_CHANNEL[r["colour"]] for r in rows for _ in r["points"]]),
                element=np.asarray([r["element"] for r in rows for _ in r["points"]]), included=cat("included"),
                weights=cat("weights"), band=np.concatenate([[r["band"]] * len(r["points"]) for r in rows]),
                included_length=included_length,
                element_length=np.bincount([r["element"] for r in rows], [r["length"] for r in rows], len(elements)),
                reach=float(np.hypot(*points.T).max()),
                area_m2=sum(r["length"] * LINE_WIDTH_M[r["weight"]] for r in rows))


def _line(raw, hit, transparent, spacing):
    """Graded skip fill and longest uncovered run (metres, along the line) of one stroke's readings.

    hit: the sample shows its own ink. transparent: a junction stretch the crossing line visibly covers; it neither
    ends an uncovered run nor adds to it. Every other sample is missing.
    """
    n = len(raw)
    missing = ~hit & ~transparent
    d = np.diff(np.r_[0, (~hit).astype(np.int8), 0])
    longest = max([missing[s:e].sum() * spacing for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))],
                  default=0.0)
    d = np.diff(np.r_[0, missing.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    value = raw.copy()
    for k, (s, e) in enumerate(zip(starts, ends)):
        if s == 0 or e == n:
            continue  # a stroke drawn short is not a skip
        g = (e - s) * spacing  # measured like the gap rule's runs
        lo, hi = (ends[k - 1] if k else 0), (starts[k + 1] if k + 1 < len(starts) else n)
        # The inked stretch on each side; one that runs to the stroke's end is not limited by a skip pattern.
        a = math.inf if lo == 0 else (s - lo) * spacing
        b = math.inf if hi == n else (hi - e) * spacing
        f = (np.clip(min(a, b) / g - 1, 0, 1)
             * np.clip((SKIP_LIMIT_M - g) / (SKIP_LIMIT_M - SKIP_BRIDGED_M), 0, 1))
        left, right = raw[lo:s][hit[lo:s]], raw[e:hi][hit[e:hi]]
        sides = ([left[-1]] if len(left) else []) + ([right[0]] if len(right) else [])
        if f > 0 and sides:
            value[s:e] = np.maximum(raw[s:e], f * min(sides))
    return value, longest

def _place(rel, p, task):
    tx, tz, s, theta = p
    c, sn = math.cos(theta), math.sin(theta)
    return (task.center_x + tx + s * (c * rel[..., 0] - sn * rel[..., 1]),
            task.center_z + tz + s * (sn * rel[..., 0] + c * rel[..., 1]))


def _uv(ink, x, z):
    b = ink.board
    return (x / b.writing_width + 0.5) * ink.width, (0.5 - (z - b.center_z) / b.writing_height) * ink.height


def _bilinear(a, fy, fx):
    y0, x0 = np.floor(fy).astype(int), np.floor(fx).astype(int)
    wy, wx = fy - y0, fx - x0
    total = np.zeros(fy.shape)
    for dy, wa in ((0, 1 - wy), (1, wy)):
        for dx, wb in ((0, 1 - wx), (1, wx)):
            yy, xx = y0 + dy, x0 + dx
            ok = (yy >= 0) & (yy < a.shape[0]) & (xx >= 0) & (xx < a.shape[1])
            total[ok] += (wa * wb)[ok] * a[yy[ok], xx[ok]]
    return total


def _tiles(ink):
    """Visible colour and weight per allocated tile."""
    out = {}
    for (tx, ty), (pigment, _) in ink.tiles.items():
        h, w = min(128, ink.height - ty * 128), min(128, ink.width - tx * 128)
        density = pigment[:h, :w].mean(axis=2)
        dominant = np.argmax(density * _STRENGTH, axis=2)
        value = np.take_along_axis(density, dominant[..., None], axis=2)[..., 0]
        weight = np.clip((value - VISIBLE_DENSITY) / (DRAWN_DENSITY - VISIBLE_DENSITY), 0, 1).astype(np.float32)
        if weight.any():
            out[tx, ty] = (dominant.astype(np.int8), weight)
    return out


def _crop(tiles, crop):
    u0, v0, u1, v1 = crop
    colour = np.full((v1 - v0, u1 - u0), -1, np.int8)
    weight = np.zeros((v1 - v0, u1 - u0), np.float32)
    for (tx, ty), (dominant, value) in tiles.items():
        ya, yb = max(v0, ty * 128), min(v1, ty * 128 + value.shape[0])
        xa, xb = max(u0, tx * 128), min(u1, tx * 128 + value.shape[1])
        if ya < yb and xa < xb:
            colour[ya - v0:yb - v0, xa - u0:xb - u0] = dominant[ya - ty * 128:yb - ty * 128, xa - tx * 128:xb - tx * 128]
            weight[ya - v0:yb - v0, xa - u0:xb - u0] = value[ya - ty * 128:yb - ty * 128, xa - tx * 128:xb - tx * 128]
    return colour, weight


def evaluate_bridge(task, ink):
    geo = _geometry(task.width_m)
    rel, weights, channel = geo["points"], geo["weights"], geo["channel"]
    px_x, px_z = ink.board.writing_width / ink.width, ink.board.writing_height / ink.height
    texel = px_x * px_z
    tiles = _tiles(ink)
    totals = np.zeros(len(COLORS))
    for dominant, value in tiles.values():
        totals += np.bincount(dominant.ravel().astype(int), value.ravel(), len(COLORS))
    needed = sorted(_CHANNEL[c] for c in PARTS.values())
    f = max(1, round(COARSE_CELL_M / px_x))

    def penalty(p):
        return FIT_PREFERENCE * (np.hypot(p[0], p[1]) / LOCAL_WINDOW_M + abs(p[2] - 1) / SEARCH_SCALE
                                 + abs(p[3]) / _THETA)

    # Board-wide coarse search at the size shown, so a copy drawn far away is still found.
    centres = [(0.0, 0.0)]
    if tiles:
        ny, nx = -(-ink.height // f), -(-ink.width // f)
        sigma = LINE_TOLERANCE_M / 2 / (f * px_x)
        u, v = _uv(ink, *_place(rel, (0, 0, 1, 0), task))
        iu, iv = (u // f).astype(int), (v // f).astype(int)
        ok = (iu >= 0) & (iu < nx) & (iv >= 0) & (iv < ny)
        shape = (next_fast_len(2 * ny), next_fast_len(2 * nx))
        acc = 0
        for c in needed:
            grid = np.zeros(ny * nx)
            for (tx, ty), (dominant, value) in tiles.items():
                rr, cc = np.nonzero((dominant == c) & (value > 0))
                np.add.at(grid, ((ty * 128 + rr) // f) * nx + (tx * 128 + cc) // f, value[rr, cc])
            grid = gaussian_filter(grid.reshape(ny, nx) / f ** 2, sigma, mode="constant")
            samples = np.zeros((ny, nx))
            k = ok & (channel == c)
            np.add.at(samples, (iv[k], iu[k]), weights[k])
            acc = acc + rfft2(grid, shape) * np.conj(rfft2(samples, shape))
        corr = irfft2(acc, shape)
        dv, du = np.r_[-iv[ok].min():ny - iv[ok].max()], np.r_[-iu[ok].min():nx - iu[ok].max()]
        corr = corr[np.ix_(dv % shape[0], du % shape[1])]
        dx, dz = du[None, :] * f * px_x, -dv[:, None] * f * px_z
        key = corr - FIT_PREFERENCE * np.hypot(dx, dz) / LOCAL_WINDOW_M
        local = np.where(np.hypot(dx, dz) <= LOCAL_WINDOW_M, key, -np.inf)
        i, j = np.unravel_index(np.argmax(key), key.shape)
        if math.hypot(dx[0, j], dz[i, 0]) > 0.8 * LOCAL_WINDOW_M and key[i, j] > local.max() + 1e-9:
            centres.append((float(dx[0, j]), float(dz[i, 0])))

    fits, crops = [], {}
    for centre in centres:
        reach = []
        for s in (1 - SEARCH_SCALE, 1 + SEARCH_SCALE):
            for theta in (-_THETA, 0.0, _THETA):
                x, z = _place(rel, (*centre, s, theta), task)
                reach += [(x.min(), z.min()), (x.max(), z.max())]
        reach = np.asarray(reach)
        margin = LOCAL_WINDOW_M + PROXIMITY_LIMIT_M + MARKER_HALF_WIDTH_M + LINE_TOLERANCE_M + STRAY_RAMP_M + 0.01
        (ua, vb), (ub, va) = (_uv(ink, reach[:, 0].min() - margin, reach[:, 1].min() - margin),
                              _uv(ink, reach[:, 0].max() + margin, reach[:, 1].max() + margin))
        box = (max(0, math.floor(ua)), max(0, math.floor(va)), min(ink.width, math.ceil(ub)), min(ink.height, math.ceil(vb)))
        colour, weight = _crop(tiles, box)
        crops[centre] = (box, colour, weight)
        u0, v0 = box[:2]
        sigma = (LINE_TOLERANCE_M / 2 / px_z, LINE_TOLERANCE_M / 2 / px_x)
        pad = int(math.ceil(3 * max(sigma))) + 1
        maps = {}
        for c in needed:
            own = np.where(colour == c, weight, 0)
            rr, cc = np.nonzero(own)
            if len(rr):
                y0, x0 = max(0, rr.min() - pad), max(0, cc.min() - pad)
                maps[c] = (y0, x0, gaussian_filter(own[y0:rr.max() + pad + 1, x0:cc.max() + pad + 1], sigma,
                                                   mode="constant", truncate=3))
        groups = {c: (rel[channel == c], weights[channel == c]) for c in maps}

        def overlap(p):
            total = 0.0
            for c, (y0, x0, blurred) in maps.items():
                u, v = _uv(ink, *_place(groups[c][0], p, task))
                total += float(groups[c][1] @ _bilinear(blurred, v - v0 - y0 - 0.5, u - u0 - x0 - 0.5))
            return total - penalty(p)

        if centre == (0.0, 0.0):
            fits.append((overlap((0.0, 0.0, 1.0, 0.0)) if maps else 0.0, (0.0, 0.0, 1.0, 0.0), centre))
        if not maps:
            continue
        ny, nx = -(-colour.shape[0] // f), -(-colour.shape[1] // f)
        ru, rv = math.ceil(LOCAL_WINDOW_M / (f * px_x)), math.ceil(LOCAL_WINDOW_M / (f * px_z))
        shape = (next_fast_len(ny + rv + 1), next_fast_len(nx + ru + 1))
        spectra = {}
        for c, (y0, x0, blurred) in maps.items():
            full = np.zeros((ny * f, nx * f), np.float32)
            full[y0:y0 + blurred.shape[0], x0:x0 + blurred.shape[1]] = blurred[:ny * f - y0, :nx * f - x0]
            spectra[c] = rfft2(full.reshape(ny, f, nx, f).mean(axis=(1, 3)), shape)
        dv, du = np.r_[-rv:rv + 1], np.r_[-ru:ru + 1]
        dx, dz = centre[0] + du[None, :] * f * px_x, centre[1] - dv[:, None] * f * px_z
        inside = np.hypot(dx - centre[0], dz - centre[1]) <= LOCAL_WINDOW_M
        candidates = []
        for s in COARSE_SCALES:
            for theta in map(math.radians, COARSE_ROTATIONS_DEG):
                u, v = _uv(ink, *_place(rel, (*centre, s, theta), task))
                iu, iv = ((u - u0) // f).astype(int), ((v - v0) // f).astype(int)
                acc = 0
                for c, spectrum in spectra.items():
                    k = (channel == c) & (iu >= 0) & (iu < nx) & (iv >= 0) & (iv < ny)
                    samples = np.zeros((ny, nx))
                    np.add.at(samples, (iv[k], iu[k]), weights[k])
                    acc = acc + spectrum * np.conj(rfft2(samples, shape))
                corr = irfft2(acc, shape)[np.ix_(dv % shape[0], du % shape[1])]
                key = np.where(inside, corr - penalty((dx, dz, s, theta)), -np.inf)
                i, j = np.unravel_index(np.argmax(key), key.shape)
                candidates.append((key[i, j], (float(dx[0, j]), float(dz[i, 0]), s, theta)))
        for _, start in sorted(candidates, key=lambda item: -item[0])[:3]:
            p, value = np.asarray(start, float), overlap(start)
            step = np.asarray([f * px_x, f * px_z, 0.01, math.radians(0.5)])
            for _ in range(REFINE_ROUNDS):
                if step[0] <= 1e-4:
                    break
                moved = False
                for d in range(4):
                    for sign in (1, -1):
                        q = p.copy()
                        q[d] += sign * step[d]
                        off = q[:2] - centre
                        if math.hypot(*off) > LOCAL_WINDOW_M:
                            q[:2] = np.asarray(centre) + off * LOCAL_WINDOW_M / math.hypot(*off)
                        q[2] = np.clip(q[2], 1 - SEARCH_SCALE, 1 + SEARCH_SCALE)
                        q[3] = np.clip(q[3], -_THETA, _THETA)
                        candidate = overlap(q)
                        if candidate > value + 1e-12:
                            p, value, moved = q, candidate, True
                if not moved:
                    step /= 2
            fits.append((value, tuple(map(float, p)), centre))
    _, p0, centre = max(fits, key=lambda item: item[0])
    offsets = np.arange(-PROXIMITY_LIMIT_M, PROXIMITY_LIMIT_M + 1e-9, OFFSET_STEP_M)
    n_el = len(geo["elements"])
    included = geo["included"]
    mean_px = (px_x + px_z) / 2

    def judge(p0, centre):
        box, colour, weight = crops[centre]
        u0, v0 = box[:2]
        # Each ink texel belongs to the element whose centreline (same colour) is nearest at the placement fit.
        # Fixed here, so the small content-fit moves cannot hand one element's ink to another.
        owner = np.full(colour.shape, -1, np.int16)
        rr, cc = np.nonzero(weight > 0)
        if len(rr):
            px0, pz0 = _place(rel, p0, task)
            wx = ((cc + u0 + 0.5) / ink.width - 0.5) * ink.board.writing_width
            wz = ink.board.center_z + (0.5 - (rr + v0 + 0.5) / ink.height) * ink.board.writing_height
            for c in needed:
                k, lines = colour[rr, cc] == c, channel == c
                if k.any():
                    index = cKDTree(np.c_[px0[lines], pz0[lines]]).query(np.c_[wx[k], wz[k]])[1]
                    owner[rr[k], cc[k]] = geo["element"][lines][index]
        # Junction zones, fixed on the board at the placement fit: for each element, the bands around other
        # elements' lines where it is not sampled (same rule as the geometry's exclusion). A sample that the
        # content fit moves into a zone keeps the better of its two readings, and only its placement-fit reading
        # can show ink present: the fit may move where ink is looked for, never which samples count.
        zones = np.zeros((n_el, *colour.shape), bool)
        for e, name in enumerate(geo["elements"]):
            mine = geo["colour_of"][name]
            image = Image.new("1", (colour.shape[1], colour.shape[0]), 0)
            draw = ImageDraw.Draw(image)
            for row, (other, _, weight_class, path) in zip(geo["rows"], bridge_strokes()):
                if row["element"] == e:
                    continue
                half = LINE_WIDTH_M[weight_class] / 2
                if row["colour"] == mine:
                    radius = max(half, MARKER_HALF_WIDTH_M) + JUNCTION_MARGIN_M
                elif _STRENGTH[_CHANNEL[row["colour"]]] > _STRENGTH[_CHANNEL[mine]]:
                    radius = half + JUNCTION_MARGIN_M
                else:
                    continue
                su, sv = _uv(ink, *_place(np.asarray(path, float) * task.width_m, p0, task))
                pts = list(zip((su - u0).tolist(), (sv - v0).tolist()))
                r = radius / mean_px
                draw.line(pts, fill=1, width=max(1, round(2 * r)), joint="curve")
                for a, b in (pts[0], pts[-1]):
                    draw.ellipse((a - r, b - r, a + r, b + r), fill=1)
            zones[e] = np.asarray(image, bool)

        def read(p, with_cover=False):
            """Per-sample credit, ink centre offset along the normal and ink mass under one transform."""
            x, z = _place(rel, p, task)
            c, sn = math.cos(p[3]), math.sin(p[3])
            nx_ = c * geo["normals"][:, 0] - sn * geo["normals"][:, 1]
            nz_ = sn * geo["normals"][:, 0] + c * geo["normals"][:, 1]
            u, v = _uv(ink, x[:, None] + offsets * nx_[:, None], z[:, None] + offsets * nz_[:, None])
            iu, iv = np.floor(u).astype(int) - u0, np.floor(v).astype(int) - v0
            ok = (iu >= 0) & (iu < colour.shape[1]) & (iv >= 0) & (iv < colour.shape[0])
            iu, iv = np.where(ok, iu, 0), np.where(ok, iv, 0)
            w = np.where(ok & (colour[iv, iu] == channel[:, None]) & (owner[iv, iu] == geo["element"][:, None]),
                         weight[iv, iu], 0.0)
            mass = w.sum(axis=1)
            centre_offset = np.where(mass > 0, (w * offsets).sum(axis=1) / np.maximum(mass, 1e-12), 0.0)
            # The line is where its ink's centre is, and only ink at that centre is a line (two lines either
            # side of the centreline have their centre on it but no ink there).
            at_centre = np.where(np.abs(offsets[None, :] - centre_offset[:, None]) <= CENTRE_INK_M, w, 0.0).max(axis=1)
            raw = at_centre * np.clip((PROXIMITY_LIMIT_M - np.abs(centre_offset))
                                      / (PROXIMITY_LIMIT_M - LINE_TOLERANCE_M), 0, 1)
            su, sv = _uv(ink, x, z)
            ju, jv = np.floor(su).astype(int) - u0, np.floor(sv).astype(int) - v0
            inside = (ju >= 0) & (ju < colour.shape[1]) & (jv >= 0) & (jv < colour.shape[0])
            in_zone = np.where(inside, zones[geo["element"], np.where(inside, jv, 0), np.where(inside, ju, 0)], False)
            if not with_cover:
                return raw, centre_offset, mass, in_zone
            # Visible ink across the line within the reading window, of its own colour (either line's) or darker.
            col = np.where(ok, colour[iv, iu], -1)
            other = ok & (col >= 0) & (weight[iv, iu] >= GAP_INK_MIN)
            other &= (col == channel[:, None]) | (_STRENGTH[np.maximum(col, 0)] > _STRENGTH[channel][:, None])
            return raw, centre_offset, mass, in_zone, other.any(axis=1)

        raw0, offset0, mass0, _, cover0 = read(p0, with_cover=True)
        # A junction stretch's own reading is ambiguous; it is transparent where ink covers the line (fixed at
        # the placement fit, like ownership and the zones) and missing elsewhere.
        transparent = ~included & cover0
        # Wave following, measured once at the placement fit (the content fit cannot change it).
        follow = np.zeros(n_el)
        start = 0
        for row in geo["rows"]:
            n = len(row["points"])
            if row["colour"] == "blue":
                keep = included[start:start + n]
                found = keep & (mass0[start:start + n] > 0)
                if found.sum() >= 0.5 * keep.sum():
                    o, dev = offset0[start:start + n][found], row["dev"][found] * p0[2]
                    dev = dev - dev.mean()
                    if (dev ** 2).sum() > 0:
                        follow[row["element"]] = -((o - o.mean()) * dev).sum() / (dev ** 2).sum()
            start += n
        wave = np.where(follow > WAVE_FOLLOW_MAX,
                        np.maximum(0.5, 1 - 0.5 * (follow - WAVE_FOLLOW_MAX) / (1 - WAVE_FOLLOW_MAX)), 1.0)

        def content(p):
            """Per-element coverage, credit and checks under one transform."""
            raw, _, _, in_zone = read(p)
            # A sample moved onto another element's line may read that line's ink, so there only the placement-fit
            # reading can show that ink is present; its credit is the better of the two.
            hit = included & (np.where(in_zone, raw0, raw) >= GAP_INK_MIN)  # one reading per sample, never both
            raw = np.where(in_zone, np.maximum(raw, raw0), raw)
            covered, gap_excess, length = np.zeros(n_el), np.zeros(n_el), np.zeros(n_el)
            start = 0
            for row in geo["rows"]:
                n = len(row["points"])
                keep = included[start:start + n]
                value, gap = _line(raw[start:start + n], hit[start:start + n], transparent[start:start + n],
                                   row["spacing"])
                covered[row["element"]] += value[keep].sum() * row["spacing"]
                length[row["element"]] += keep.sum() * row["spacing"]
                if row["gap_rule"]:
                    gap_excess[row["element"]] += max(0.0, gap - GAP_ALLOWED_M)
                start += n
            coverage = covered / np.maximum(length, 1e-12)
            credit = np.clip(np.minimum(np.minimum(1, coverage / COVERAGE_REQUIRED) * wave,
                                        1 - gap_excess / geo["element_length"]), 0, 1)
            parts = {}
            for part, colour_name in PARTS.items():
                members = [i for i, e in enumerate(geo["elements"]) if geo["colour_of"][e] == colour_name]
                parts[part] = {"colour": colour_name, "credit": float(np.mean(credit[members])),
                               "coverage": {geo["elements"][i]: round(float(coverage[i]), 4) for i in members},
                               "gap_excess_mm": {geo["elements"][i]: round(float(gap_excess[i]) * 1000, 1)
                                                 for i in members if gap_excess[i] > 0},
                               "wave_follow": {geo["elements"][i]: round(float(follow[i]), 3)
                                               for i in members if colour_name == "blue"}}
            fidelity = float(np.mean([part["credit"] for part in parts.values()]))
            complete = bool((coverage >= COVERAGE_REQUIRED).all() and (gap_excess == 0).all()
                            and (follow <= WAVE_FOLLOW_MAX).all())
            return dict(parts=parts, fidelity=fidelity, complete=complete, raw=float(np.mean(coverage)))

        def objective(p, result):
            distance = (math.hypot(p[0] - p0[0], p[1] - p0[1]) / CONTENT_SHIFT_M + abs(p[2] / p0[2] - 1) / CONTENT_SCALE
                        + abs(p[3] - p0[3]) / math.radians(CONTENT_ROTATION_DEG))
            return result["fidelity"] + 0.01 * result["raw"] - 1e-4 * distance

        def clamp(q):
            q = np.asarray(q, float).copy()
            off = q[:2] - p0[:2]
            if math.hypot(*off) > CONTENT_SHIFT_M:
                q[:2] = np.asarray(p0[:2]) + off * CONTENT_SHIFT_M / math.hypot(*off)
            q[2] = np.clip(q[2], p0[2] * (1 - CONTENT_SCALE), p0[2] * (1 + CONTENT_SCALE))
            q[3] = np.clip(q[3], p0[3] - math.radians(CONTENT_ROTATION_DEG), p0[3] + math.radians(CONTENT_ROTATION_DEG))
            return q

        best_p = np.asarray(p0, float)
        best = content(best_p)
        best_value = objective(best_p, best)
        if tiles and not best["complete"]:
            h = CONTENT_SHIFT_M / 2
            seeds = [clamp((0.0, 0.0, 1.0, 0.0))] + [clamp((p0[0] + a, p0[1] + b, p0[2], p0[3]))
                                                      for a in (-2 * h, -h, 0, h, 2 * h) for b in (-2 * h, -h, 0, h, 2 * h)
                                                      if math.hypot(a, b) <= CONTENT_SHIFT_M + 1e-12]
            for q in seeds:
                result = content(q)
                if objective(q, result) > best_value:
                    best_p, best, best_value = q, result, objective(q, result)
            step = np.asarray([h / 2, h / 2, CONTENT_SCALE / 3, math.radians(CONTENT_ROTATION_DEG) / 3])
            for _ in range(20):
                if best["complete"] or step[0] < 2e-4:
                    break
                moved = False
                for d in range(4):
                    for sign in (1, -1):
                        q = best_p.copy()
                        q[d] += sign * step[d]
                        q = clamp(q)
                        result = content(q)
                        if objective(q, result) > best_value + 1e-12:
                            best_p, best, best_value, moved = q, result, objective(q, result), True
                if not moved:
                    step /= 2
        p = tuple(map(float, best_p))

        # Cleanliness: visible ink by its distance from the nearest line of its own colour.
        by_colour = totals.copy()
        x, z = _place(rel, p, task)
        rr, cc = np.nonzero(weight > 0)
        for c in needed:
            k = colour[rr, cc] == c
            if not k.any():
                continue
            lines = channel == c
            wx = ((cc[k] + u0 + 0.5) / ink.width - 0.5) * ink.board.writing_width
            wz = ink.board.center_z + (0.5 - (rr[k] + v0 + 0.5) / ink.height) * ink.board.writing_height
            distance, index = cKDTree(np.c_[x[lines], z[lines]]).query(np.c_[wx, wz])
            ramp = np.clip((distance - geo["band"][lines][index]) / STRAY_RAMP_M, 0, 1)
            by_colour[c] -= (weight[rr[k], cc[k]] * (1 - ramp)).sum()
        stray_m2 = float(by_colour.sum() * texel)

        shift = math.hypot(p0[0], p0[1])
        scale_move = geo["reach"] * abs(p0[2] - 1)
        turn_move = 2 * geo["reach"] * math.sin(abs(p0[3]) / 2)
        excess = (max(0.0, shift - SHIFT_ACCEPT_M) + max(0.0, scale_move - geo["reach"] * SCALE_ACCEPT)
                  + max(0.0, turn_move - 2 * geo["reach"] * math.sin(math.radians(ROTATION_ACCEPT_DEG) / 2)))
        factors = {"fidelity": best["fidelity"],
                   "cleanliness": float(np.clip(1 - max(0.0, stray_m2 - STRAY_ALLOWANCE_M2) / geo["area_m2"], 0, 1)),
                   "placement": 1 / (1 + excess / PLACEMENT_HALF_M)}
        checks = {"every_element_drawn": best["complete"], "rest_of_board_clean": stray_m2 <= STRAY_ALLOWANCE_M2,
                  "placed_where_shown": excess == 0}
        success = all(checks.values())
        product = factors["fidelity"] * factors["cleanliness"] * factors["placement"]
        # Each factor is exactly 1.0 when its check holds, so success gives progress 1.0;
        # the cap only keeps a rounding-level miss from reading as complete.
        progress = product if success else min(product, math.nextafter(1.0, 0.0))
        return {
            "score": 1.0 if success else FAIL_SCORE_SCALE * progress,
            "success": success,
            "progress": progress,
            "score_rule": "1 if success else 0.9 x progress",
            "revision": REVISION,
            "metric": f"{REVISION}: registered copy of the supplied bridge diagram (per-element cross-line coverage, "
                      "longest gaps, visible stray marks, placement); deterministic, not a semantic judge",
            "checks": checks,
            "factors": factors,
            "parts": best["parts"],
            "registration": {
                "shift_mm": [round(p0[0] * 1000, 2), round(p0[1] * 1000, 2)],
                "shift_norm_mm": round(shift * 1000, 2),
                "scale": round(p0[2], 4),
                "rotation_deg": round(math.degrees(p0[3]), 3),
                "excess_movement_mm": round(excess * 1000, 2),
                "content_fit_offset_mm": [round((p[0] - p0[0]) * 1000, 2), round((p[1] - p0[1]) * 1000, 2)],
                "content_fit_scale": round(p[2], 4),
                "content_fit_rotation_deg": round(math.degrees(p[3]), 3),
                "axes": "shift along world +x (left as seen from the front) and +z (up)",
            },
            "stray": {
                "area_cm2": round(stray_m2 * 1e4, 3),
                "allowance_cm2": STRAY_ALLOWANCE_M2 * 1e4,
                "by_colour_cm2": {c: round(float(a) * texel * 1e4, 3) for c, a in zip(COLORS, by_colour) if a > 1e-9},
                "reference_ink_area_cm2": round(geo["area_m2"] * 1e4, 2),
            },
        }

    shown = (0.0, 0.0, 1.0, 0.0)
    result = judge(p0, centre)
    result["registration"]["judged_at"] = "placement fit"
    if not result["success"] and tiles and tuple(p0) != shown:
        # One misplaced part can pull the overlap fit off the parts that are where shown; judge those too.
        other = judge(shown, (0.0, 0.0))
        if other["progress"] > result["progress"]:
            other["registration"]["judged_at"] = "placement shown"
            other["registration"]["placement_fit_shift_mm"] = result["registration"]["shift_mm"]
            result = other
    return result
