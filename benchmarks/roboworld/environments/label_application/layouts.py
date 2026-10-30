"""Seeded task layouts: everything a seed changes besides the carton pose drawn in scene.parcel.

Pure functions of (task, seed), shared by the scene, the label artwork and the evaluator, so the
printed label, the item it belongs to and the verifier always agree. Each draw uses its own
generator (numpy SeedSequence of (seed, tag)), so adding one never shifts another, and the carton
pose drawn by Scene.rng is unchanged. Seed 0 keeps fixed reference layouts (payloads, tube roll,
canister position).
"""
import math
import numpy as np

IN = .0254

# seed-0 payloads (the reference layouts); other seeds derive payloads of the same length and format.
# gs1-case-side: the SSCC's GS1 mod-10 check digit is 4.
PAYLOADS = {"carrier-top": "961201950123456789", "cover-old-label": "961201950198765432",
            "press-down": "961201950155501234", "gs1-case-side": "00106141410000004154",
            "tube-wrap": "S2609230742", "part-id-hdpe": "NG-3310-0007-000912",
            "part-id-match": "NG-7715-0042-000318"}
OLD_LABEL_PAYLOAD = "961201950987654321"


def _rng(seed, tag):
    return np.random.default_rng([int(seed), int(tag)])


def gs1_check_digit(digits):
    """GS1 mod-10 check digit of a data string (weights 3, 1, 3, ... from the right)."""
    s = sum(int(c)*(3 if k % 2 == 0 else 1) for k, c in enumerate(reversed(digits)))
    return str((10 - s % 10) % 10)


def payload(task, seed):
    """The barcode payload of the label printed for (task, seed). The ladder tasks print the same
    format as the task they share a bench with: a carrier tracking number, an accession number."""
    if task in ("smooth-corner", "finish-securing"):
        return "9612019" + "".join(str(d) for d in _rng(seed, 11 if task == "smooth-corner" else 12).integers(0, 10, 11))
    if task == "tube-label":
        return "S260923" + "".join(str(d) for d in _rng(seed, 13).integers(0, 10, 4))
    if task == "part-id-match":
        p = plates(seed)
        return f"NG-{p['part_numbers'][p['target']]}-{_serial(seed, 7):06d}"
    if seed == 0:
        return PAYLOADS[task]
    r = _rng(seed, 1)
    if task in ("carrier-top", "cover-old-label", "press-down"):
        return "9612019" + "".join(str(d) for d in r.integers(0, 10, 11))           # 18-digit tracking number
    if task == "gs1-case-side":
        data = "1" + "0614141" + "".join(str(d) for d in r.integers(0, 10, 9))      # extension + GS1 prefix + serial
        return "00" + data + gs1_check_digit(data)                                    # AI (00) + SSCC-18
    if task == "tube-wrap":
        return "S260923" + "".join(str(d) for d in r.integers(0, 10, 4))           # accession number
    if task == "part-id-hdpe":
        return f"NG-3310-0007-{_serial(seed, 7):06d}"
    raise ValueError(task)


def old_label_payload(seed):
    if seed == 0:
        return OLD_LABEL_PAYLOAD
    return "9612019" + "".join(str(d) for d in _rng(seed, 2).integers(0, 10, 11))


def _serial(seed, tag):
    return 318 if seed == 0 and tag == 7 else int(_rng(seed, tag).integers(100, 999999))


# ------------------------------------------------------------------ cover-old-label
def old_label(seed, carton_W=12*IN, tape_half=.024, margin=5e-3):
    """The previous shipment's 4 x 3 in address label on the carton top, in carton coordinates (u along the
    carton's 16 in length, v across). It lies where a previous shipper would have put it under the same carrier
    rules (entirely on one flap half, off the seam, >= 5 mm from the edges), with its 4 in side along the carton,
    35-100 mm either side of the carton's middle, on either flap half. Its centre across the flap half is kept where
    the new 4 x 6 in label can cover it under the carrier rules with at least ~10 mm of room across (v) and 25 mm
    along (u). The >= 35 mm offset keeps a centred carrier-top placement from covering it by chance."""
    r = _rng(seed, 3)
    side = 1 if r.random() < .5 else -1           # +1: the flap half toward the robots (+y), -1: the far half
    lu, lv = 4*IN, 3*IN
    new_w = 4*IN                                  # the new label's 4 in side lies across the flap half
    lo = tape_half + new_w/2                       # the new label's centre range across the flap half
    hi = carton_W/2 - margin - new_w/2
    slack = (new_w - lv)/2                         # 12.7 mm: how far the new label may sit off the old one
    room = 10e-3
    vc = r.uniform(lo - slack + room, hi + slack - room)
    uc = (1 if r.random() < .5 else -1)*r.uniform(.035, .10)
    v0, v1 = side*vc - lv/2, side*vc + lv/2
    return dict(u=(uc - lu/2, uc + lu/2), v=(min(v0, v1), max(v0, v1)), side=side, format="4x3 in", centre=(uc, side*vc))


# ------------------------------------------------------------------ press-down
def laid_label(seed, carton_W=12*IN, tape_half=.024, margin=5e-3, label_w=4*IN):
    """Where the co-worker laid the 4 x 6 in label on the carton top for press-down (carton coordinates): on the
    flap half toward the robots, inside the carrier zone, centre within +-30 mm along the carton and +-5 mm across
    the zone's middle, turned up to 2 deg."""
    r = _rng(seed, 4)
    zone = (tape_half + carton_W/2 - margin)/2
    return dict(centre=(float(r.uniform(-.03, .03)), float(zone + r.uniform(-.005, .005))),
                yaw=float(math.radians(r.uniform(-2., 2.))))


# ------------------------------------------------------------------ task ladder: partly applied labels
LABEL_UV = (6*IN, 4*IN)               # the 4 x 6 in shipping label, material (u along the 6 in side, v across)
# The printed tracking symbol with its quiet zones in label material coordinates (artwork.shipping_label draws it at
# barcode_px (175, 620, 637, 820) of 812 x 1218 px for every 18-digit tracking number: u 49.8-74.8 mm from the trailing
# edge, v 21.9-79.7 mm). Lifted areas stay KEEPOUT clear of it, so a curl never sits across the symbol.
SYMBOL_UV = ((.0498, .0748), (.0219, .0797))
KEEPOUT = 4e-3
REGION_GAP = 8e-3                     # m between two lifted areas: each is its own flap, hinged on bonded label
# 1/m: the curl of a lifted area is at most 100/m (10 mm radius), below the paper's bending yield curvature (119/m,
# materials.PAPER_DT: 2 x 0.5 % / 83.8 um), so pressing it flat is elastic and leaves no crease
MAX_CURL = 100.


def _flap(kind, a, b, c, tip_deg):
    """One lifted area: the part of the label beyond a straight bond front (a half-plane of the label's material
    rectangle). kind corner: the front cuts the two edges at a material corner; legs a (along u) and b (along v).
    kind end: the front crosses the label near a 4 in end (u = 0 or 6 in), depth a at v = 0 and b at v = 4 in.
    kind side: the front runs along a 6 in side (v = 0 or 4 in), depth a at u = 0 and b at u = 6 in.
    c: which corner (cu, cv), end cu or side cv (0 or the far edge). The flap curls up from the front at constant
    curvature so its farthest point stands tip_deg off the carton (ladder.lay_partly_applied)."""
    L, W = LABEL_UV
    if kind == "corner":
        cu, cv = c
        su, sv = (1 if cu == 0 else -1), (1 if cv == 0 else -1)
        p1, p2, far = np.array([cu + su*a, cv]), np.array([cu, cv + sv*b]), [np.array([cu, cv])]
    elif kind == "end":
        su = 1 if c == 0 else -1
        p1, p2, far = np.array([c + su*a, 0.]), np.array([c + su*b, W]), [np.array([c, 0.]), np.array([c, W])]
    else:
        sv = 1 if c == 0 else -1
        p1, p2, far = np.array([0., c + sv*a]), np.array([L, c + sv*b]), [np.array([0., c]), np.array([L, c])]
    t = (p2 - p1)/np.linalg.norm(p2 - p1)
    nrm = np.array([-t[1], t[0]])
    if (far[0] - p1) @ nrm < 0:
        nrm = -nrm
    depth = float(max((f - p1) @ nrm for f in far))
    tip = min(math.radians(tip_deg), MAX_CURL*depth)
    names = {(0, 0): "bottom-left", (0, W): "bottom-right", (L, 0): "top-left", (L, W): "top-right"}
    name = {"corner": f"{names[tuple(c)]} corner" if kind == "corner" else None,
            "end": "bottom end" if c == 0 else "top end", "side": "left side" if c == 0 else "right side"}[kind]
    return dict(kind=kind, name=name, point=tuple(map(float, p1)), normal=tuple(map(float, nrm)), depth=depth,
                tip_deg=float(math.degrees(tip)), curvature=float(tip/depth))


def flap_mask(X, flap):
    """Label material points (N x 2) inside a lifted area, and their distance from its bond front (m)."""
    d = (np.asarray(X, float)[:, :2] - np.asarray(flap["point"])) @ np.asarray(flap["normal"])
    return d > 0, d


def _grid(step=1e-3):
    L, W = LABEL_UV
    u, v = np.meshgrid(np.arange(step/2, L, step), np.arange(step/2, W, step), indexing="ij")
    return np.stack([u.ravel(), v.ravel()], 1)


def _symbol_distance(P):
    (u0, u1), (v0, v1) = SYMBOL_UV
    du = np.maximum(np.maximum(u0 - P[:, 0], P[:, 0] - u1), 0.)
    dv = np.maximum(np.maximum(v0 - P[:, 1], P[:, 1] - v1), 0.)
    return np.hypot(du, dv)


def _flaps_ok(flaps, area_range=None):
    """Lifted areas clear of the symbol, at least REGION_GAP apart, each >= 300 mm2; total area fraction in range."""
    G = _grid(1e-3)
    masks = [flap_mask(G, f)[0] for f in flaps]
    for m in masks:
        if m.sum()*1e-6 < 300e-6 or (_symbol_distance(G[m]) < KEEPOUT).any():
            return False
    coarse = _grid(2e-3)
    cm = [flap_mask(coarse, f)[0] for f in flaps]
    for i in range(len(flaps)):
        for j in range(i + 1, len(flaps)):
            a, b = coarse[cm[i]], coarse[cm[j]]
            if np.sqrt(((a[:, None, :] - b[None, :, :])**2).sum(-1)).min() < REGION_GAP:
                return False
    if area_range is not None:
        frac = np.any(masks, 0).mean()
        if not area_range[0] <= frac <= area_range[1]:
            return False
    return True


def edge_regions():
    """The label's whole perimeter as the eight largest areas the repair rungs can leave lifted, the generator's own
    shapes at their largest draws: the four corners (legs 45 mm along u and 40 mm along v), the two 4 in ends (46 mm
    deep: 40 mm plus the 6 mm out-of-square) and the two 6 in sides (19 mm deep: 17 mm plus 2 mm). Every lifted area
    partly_applied draws lies inside the region of its kind and place. task.Evaluator holds each of them to the
    standard the lifted areas are held to, so a part of the label that comes loose during the work is judged as a
    lifted one is."""
    L, W = LABEL_UV
    out = [_flap("corner", .045, .040, c, 90.) for c in ((0., 0.), (0., W), (L, 0.), (L, W))]
    out += [_flap("end", .046, .046, c, 90.) for c in (0., L)]
    out += [_flap("side", .019, .019, c, 90.) for c in (0., W)]
    return out


def partly_applied(task, seed, carton_W=12*IN, tape_half=.024, margin=5e-3):
    """Where a co-worker stuck the 4 x 6 in label on the carton top and which parts of it are still lifted, for the
    two repair rungs of the task ladder (carton coordinates; regions in label material coordinates).
    smooth-corner: on the flap half toward the robots, like press-down's laid label (centre +-30 mm along, +-5 mm
    across the zone's middle, turned up to 2 deg), either way round; one lifted corner (any of the four), legs 28-38
    mm along the label and 22-32 mm across, its tip curled 65-90 deg off the carton.
    finish-securing: on either flap half (centre +-30 mm along, +-5 mm across, turned up to 3 deg), either way round;
    two to four lifted areas among the four corners (legs 28-45 / 24-40 mm), the two ends (14-40 mm deep, the front
    up to 12 mm out of square) and the two long sides (12-17 mm deep), curled 60-90 deg (at most 100/m), apart from
    each other and clear of the barcode, together 12-35 % of the label."""
    L, W = LABEL_UV
    r = _rng(seed, 9 if task == "smooth-corner" else 10)
    side = 1 if task == "smooth-corner" or r.random() < .5 else -1
    zone = (tape_half + carton_W/2 - margin)/2
    across = .005
    yaw_deg = 2. if task == "smooth-corner" else 3.
    centre = (float(r.uniform(-.03, .03)), float(side*(zone + r.uniform(-across, across))))
    yaw = float(math.radians(r.uniform(-yaw_deg, yaw_deg)))
    flip = bool(r.random() < .5)
    corners = [(0., 0.), (0., W), (L, 0.), (L, W)]
    if task == "smooth-corner":
        c = corners[int(r.integers(0, 4))]
        flaps = [_flap("corner", r.uniform(.028, .038), r.uniform(.022, .032), c, r.uniform(65., 90.))]
    elif task == "finish-securing":
        for _ in range(1000):
            k = int(r.choice([2, 2, 3, 3, 4]))
            pool = [("corner", c) for c in corners] + [("end", 0.), ("end", L), ("side", 0.), ("side", W)]
            pick = [pool[i] for i in r.permutation(len(pool))[:k]]
            flaps = []
            for kind, c in pick:
                tip = r.uniform(60., 90.)
                if kind == "corner":
                    flaps.append(_flap(kind, r.uniform(.028, .045), r.uniform(.024, .040), c, tip))
                elif kind == "end":
                    w, skew = r.uniform(.014, .040), r.uniform(-.006, .006)
                    flaps.append(_flap(kind, w + skew, w - skew, c, tip))
                else:
                    w, skew = r.uniform(.012, .017), r.uniform(-.002, .002)
                    flaps.append(_flap(kind, w + skew, w - skew, c, tip))
            if _flaps_ok(flaps, (.12, .35)):
                break
        else:                                   # pragma: no cover - the draw always succeeds within a few tries
            raise RuntimeError(f"no lifted-area layout for seed {seed}")
    else:
        raise ValueError(task)
    if task == "smooth-corner" and not _flaps_ok(flaps):
        raise RuntimeError(f"smooth-corner seed {seed}: corner layout invalid")
    return dict(centre=centre, yaw=yaw, flip=flip, side=side, regions=flaps)


# ------------------------------------------------------------------ tube-wrap
def tube_roll(seed):
    """Roll of the tube about its own axis in the cradle (rad). Seed 0: manufacturer label facing up; other
    seeds: any roll, as a tube is laid in a cradle by hand."""
    return 0. if seed == 0 else float(_rng(seed, 5).uniform(-math.pi, math.pi))


# ------------------------------------------------------------------ part-id-hdpe
def canister_offset(seed):
    """Canister position offset on the bench (m). Seed 0: none."""
    return (0., 0.) if seed == 0 else tuple(float(v) for v in _rng(seed, 6).uniform(-.02, .02, 2))


# ------------------------------------------------------------------ part-id-match
PLATE_SIZE = (.16, .12, .006)          # the stainless plate of the retired part-id-steel task, unchanged
PLATE_SLOTS = (-.03, .17, .37)         # x of the three plates on the receiving bench (left of the tool tray)
PLATE_Y = -.03
MARKING_SIZE = (.064, .014)            # laser-marked "P/N ...." block on the plate face (u, v extents)


def plates(seed):
    """Three stainless mounting plates from one delivery, each laser-marked with its part number; the printed ID
    label belongs to one of them. Part numbers are confusable (same family, one digit or one place apart), as
    left/right-hand and revision variants are. Returns slot poses (x, y, yaw), part numbers per slot, marking
    rectangles in plate coordinates and the target slot."""
    r = _rng(seed, 8)
    if seed == 0:
        a, b = 7715, 42
    else:
        a, b = int(r.integers(1100, 9800)), int(r.integers(10, 980))
    family = [f"{a:04d}-{b:04d}", f"{a:04d}-{b + 1:04d}", f"{a + 10:04d}-{b:04d}"]
    order = [0, 1, 2] if seed == 0 else [int(k) for k in r.permutation(3)]
    numbers = [family[k] for k in order]
    target = order.index(0)                       # the label is printed for family[0]
    poses, markings = [], []
    for k, x in enumerate(PLATE_SLOTS):
        if seed == 0:
            dx, dy, yaw = 0., 0., 0.
        else:
            dx, dy = (float(v) for v in r.uniform(-.008, .008, 2))
            yaw = float(math.radians(r.uniform(-4., 4.)))
        poses.append((x + dx, PLATE_Y + dy, yaw))
        su = 1 if (seed == 0 and k != 1) or (seed and r.random() < .5) else -1
        sv = 1 if (seed == 0) or r.random() < .5 else -1
        mu = su*.040
        mv = sv*(PLATE_SIZE[1]/2 - .009 - MARKING_SIZE[1]/2)
        markings.append(dict(u=(mu - MARKING_SIZE[0]/2, mu + MARKING_SIZE[0]/2),
                             v=(mv - MARKING_SIZE[1]/2, mv + MARKING_SIZE[1]/2)))
    return dict(poses=poses, part_numbers=numbers, markings=markings, target=target)
