"""Seeded task layouts: everything a seed changes besides the carton pose drawn in scene.parcel.

Pure functions of (task, seed), shared by the scene, the label artwork and the evaluator, so the
printed label, the item it belongs to and the verifier always agree. Each draw uses its own
generator (numpy SeedSequence of (seed, tag)), so adding one never shifts another, and the carton
pose drawn by Scene.rng is unchanged. Seed 0 keeps fixed reference layouts (payloads, tube roll,
canister position). A benchmark task's own draws are in its setup.py (tasks/<task>/: payload, and press-down's
laid label, the repair rungs' lifted areas, part-id-hdpe's canister offset, part-id-match's plates).
"""
import math
import numpy as np

IN = .0254

# seed-0 payloads (the reference layouts) of the tasks outside the benchmark (a benchmark task's: its setup.py
# PAYLOAD); other seeds derive payloads of the same length and format.
PAYLOADS = {"cover-old-label": "961201950198765432", "tube-wrap": "S2609230742"}
OLD_LABEL_PAYLOAD = "961201950987654321"


def _rng(seed, tag):
    return np.random.default_rng([int(seed), int(tag)])


def gs1_check_digit(digits):
    """GS1 mod-10 check digit of a data string (weights 3, 1, 3, ... from the right)."""
    s = sum(int(c)*(3 if k % 2 == 0 else 1) for k, c in enumerate(reversed(digits)))
    return str((10 - s % 10) % 10)


def payload(task, seed):
    """The barcode payload of the label printed for (task, seed): a benchmark task's setup.py payload(seed), else
    the draws below."""
    from .config import task_setup
    folder = task_setup(task)
    if folder is not None:
        return folder.payload(seed)
    if seed == 0:
        return PAYLOADS[task]
    r = _rng(seed, 1)
    if task == "cover-old-label":
        return "9612019" + "".join(str(d) for d in r.integers(0, 10, 11))           # 18-digit tracking number
    if task == "tube-wrap":
        return "S260923" + "".join(str(d) for d in r.integers(0, 10, 4))           # accession number
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
    two repair rungs of the task ladder (carton coordinates; regions in label material coordinates): centre +-30 mm
    along the carton and +-5 mm across the middle of the carrier zone of a flap half, turned, either way round. The
    rung's setup.py holds its own draws (tasks/smooth-corner/, tasks/finish-securing/): the generator's tag
    (LAYOUT_TAG), the flap half (label_side), the turn (YAW_DEG) and the lifted areas (lifted_areas)."""
    from .config import task_setup
    rung = task_setup(task)
    if not hasattr(rung, "lifted_areas"):
        raise ValueError(task)
    L, W = LABEL_UV
    r = _rng(seed, rung.LAYOUT_TAG)
    side = rung.label_side(r)
    zone = (tape_half + carton_W/2 - margin)/2
    across = .005
    yaw_deg = rung.YAW_DEG
    centre = (float(r.uniform(-.03, .03)), float(side*(zone + r.uniform(-across, across))))
    yaw = float(math.radians(r.uniform(-yaw_deg, yaw_deg)))
    flip = bool(r.random() < .5)
    corners = [(0., 0.), (0., W), (L, 0.), (L, W)]
    flaps = rung.lifted_areas(r, corners, L, W, seed)
    return dict(centre=centre, yaw=yaw, flip=flip, side=side, regions=flaps)


# ------------------------------------------------------------------ tube-wrap
def tube_roll(seed):
    """Roll of the tube about its own axis in the cradle (rad). Seed 0: manufacturer label facing up; other
    seeds: any roll, as a tube is laid in a cradle by hand."""
    return 0. if seed == 0 else float(_rng(seed, 5).uniform(-math.pi, math.pi))
