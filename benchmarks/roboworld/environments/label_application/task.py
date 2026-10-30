"""Task instructions and the physical/optical evaluator.

Success is judged from simulator state (bond field, gaps, trapped air, permanent set, membrane stress,
label pose on the item, robot contacts, item rest) and a zxing-cpp decode of rendered reader views
(optical.py). Thresholds are authored and cite the standard they implement where one exists.

Instructions describe the real job, not the verifier's checklist. Ordinary finished-label
workmanship is implicit; numerical acceptance tolerances and optical-reader details live in
the evaluator.

Task ladder (smooth-corner, finish-securing, tube-label): success is judged as for the task each shares its bench
with (press-down, tube-wrap), plus, on the repair rungs, every area that started lifted must itself reach the wet-out
standard, as an area and along its own 3 mm label edge; the repair rungs accept the label on either flap half
(carrier-top's placement rules; press-down alone says "where it lies"). A barcode creased beyond the paper's yield
zeroes progress as a tear does, and tube-label's registration ramps end at physical limits.

Every task is scored by the continuous progress reward (REWARD_CONTRACTS, Evaluator.progress). The seven other tasks
(contract progress-v2) each take their zone and registration factors from their own placement rules; the mean
of the checks is reported beside it (reward.mean_of_checks).

tube-label: the instruction is set at the point of collection and states which end of the label goes to the cap;
success adds two placement checks its instruction states (the label over the manufacturer's label, _place_tube_label;
the patient's name toward the cap); the score is tube-wrap's take-and-apply progress rule with the viewing window read
from where the label is going. The skew factor reads which end of the label is heading for the cap, a label lost off
the bench zeroes integrity, and a label still bonded to anything is never counted lost.
"""

import math
import numpy as np

from .config import LADDER_TASKS, REPAIR_TASKS, START_CREDIT_TASKS, base_task

INSTRUCTIONS = {
    "carrier-top": (
        "You are at a shipping pack bench. The desktop label printer has just printed and presented the shipping "
        "label for the sealed carton on the bench. Take the label from the printer and apply it to the carton's top, "
        "clear of the sealing tape along the seam and clear of the top's edges."),
    "gs1-case-side": (
        "You are at a shipping pack bench. The desktop label printer has just printed and presented the GS1 logistic "
        "label for the sealed carton on the bench. Take the label from the printer and apply it to one of the "
        "carton's long sides where GS1 places it on a case: the bottom of the SSCC barcode 32 mm above the carton's "
        "base, and the barcode, including its quiet zones, at least 19 mm from either vertical edge of that side."),
    "cover-old-label": (
        "You are at a shipping pack bench. The sealed carton on the bench is being reused for a new shipment and "
        "still carries the previous shipment's address label on its top. The desktop label printer has just printed "
        "and presented the new shipping label. Take it from the printer and apply it over the old label so that the "
        "old label is completely covered, keeping the new label clear of the sealing tape along the seam and "
        "the top's edges."),
    "press-down": (
        "You are at a shipping pack bench. A co-worker has laid the shipping label in position on the sealed "
        "carton's top and left it only lightly tacked. Finish applying it: press it down where it lies."),
    "tube-wrap": (
        "You are at a clinical specimen accessioning bench. A blood collection tube lies in the cradle on the bench, "
        "and the printer has just printed and presented its patient label. Take the label from the printer and apply "
        "it lengthwise and straight along the tube, directly over the tube manufacturer's label: its end nearest the "
        "cap lined up with the manufacturer's label's edge nearest the cap, not above it, and a clear "
        "viewing strip left open along the tube. Leave the tube in the cradle."),
    "part-id-hdpe": (
        "You are at a receiving bench. The thermal-transfer printer has just printed and presented the polyester ID "
        "label for the round HDPE canister standing on the bench. Take the label from the printer and apply it to "
        "the canister's side, running level around it, about half-way up."),
    "part-id-match": (
        "You are at a receiving bench. Three stainless-steel mounting plates from the same delivery lie on the "
        "bench, each laser-marked with its part number. The thermal-transfer printer has just printed and presented "
        "the polyester ID label for one of them. Take the label from the printer and apply it to the plate whose "
        "marked part number matches the one printed on the label, on its marked face, without covering the "
        "marking."),
    # task ladder (config.LADDER_TASKS)
    "smooth-corner": (
        "You are at a shipping pack bench, beside a sealed carton with its shipping label on top. Smooth down the "
        "lifted corner of this shipping label."),
    "finish-securing": (
        "You are at a shipping pack bench, beside a sealed carton with its shipping label on top. Finish securing "
        "this partially applied shipping label."),
    "tube-label": (
        "You are at a blood collection station. A blood collection tube, just drawn from the patient, lies in its "
        "cradle, and the printer has just printed and presented the patient's label. Take the printed patient label "
        "and apply it correctly to the specimen tube. Here that means lengthwise and straight over the tube "
        "manufacturer's label, with the patient's name at the end toward the cap and that end lined up with the "
        "manufacturer's label's edge nearest the cap (no closer to the cap), a clear viewing strip left open along "
        "the tube, and the tube left in its cradle."),
}

# Reward contracts. Every task: score = 1 on success, else 0.9 x progress, where progress (Evaluator.progress)
# measures the work product in the final state -- label area lying adhesive-down on the right surface where the task
# allows it, and how far that area is wetted out, times the task's registration factors -- whatever route produced
# it. Success itself is the AND of the checks. "progress-v1": the ladder. "progress-v2": the seven
# other tasks. Every evaluation also reports the fraction of passed checks as reward.mean_of_checks.
REWARD_CONTRACTS = {t: ("progress-v1" if t in LADDER_TASKS else "progress-v2") for t in INSTRUCTIONS}
# tube-label is scored by tube-wrap's take-and-apply rule (take credit, wet-out against the pass line), with its own
# registration (Evaluator._tube_registration: the viewing window read from where the label is going, 1 only over the
# manufacturer's label); the repair rungs keep the ladder's contract. Its skew factor also reads which end of the label
# points to the cap (a label heading for the reversed placement gets 0), and a label lost off the bench, held by
# nothing, zeroes integrity (LOST_LABEL_ZEROES). A label still bonded to anything (the bench, the printer) is never
# "lost", however much of it hangs off the bench
REWARD_CONTRACTS["tube-label"] = "progress-v3"
FAIL_SCORE_SCALE = .9          # a failed state scores 0.9 x progress: every failure stays below every success
TUBE_BOTTOM_CLEAR = 9e-3       # m: the tube-wrap placement rule clear_of_tube_bottom>=9mm, as a zone
# The paper facestock's bending yield as a set curvature (the no_crease_in_barcode limit every task uses): a
# barcode set beyond it is a permanent crease through the symbol. Avery: "If a wrinkle runs through the barcode, it
# can interfere with how the barcode scanner reads it" (web-sourced); that such a label is reprinted, as a torn one
# is, is our inference, not a quoted rule. In this world the set is a rest shape that bending the other way pulls back
# down (on the 13 mm tube the wrap itself undoes a crease bent against it), so the ladder tasks read the largest set
# the barcode has carried in the episode (World.set_peak), as paper keeps the broken fibre of a crease bent back.
BARCODE_CREASE_YIELD = 119.    # 1/m
# tube-label registration ramps (Evaluator.registration): each factor is 1 inside tube-wrap's acceptance band and
# falls linearly to 0 where the placement stops being a labelled tube. Cap end: the cap (scene.specimen: the cap
# collider, half-length 12 mm centred 10 mm beyond the tube end) covers the tube's top 2.0 mm, so a label edge there
# is on the cap. Bottom: the label's far end at the 9 mm bottom clearance. Skew: 45 deg is a diagonal label. The
# far ends are authored task choices anchored to that geometry, not measurements.
TUBE_CAP_SKIRT = 2e-3          # m from the cap end
# Repair rungs: the label edge judged locally. Each square window EDGE_WINDOW on a side, set on a label edge every
# EDGE_WINDOW_STEP (and flush at each edge's ends), must reach the wet-out standard as an area-weighted average. One average over a large edge region would let a loose tongue at the middle of an end (12 x 30 mm)
# pass; 25 mm windows separate every correct finish on record (lowest window >= 0.93, the starts' own tack nodes
# included) from every loose-patch state (<= 0.53), and 20 mm would fail a correct repair at finish-securing seed 658.
EDGE_WINDOW = 25e-3            # m
EDGE_WINDOW_STEP = 3e-3        # m
TUBE_SKEW_ZERO_DEG = 45.
# Registration factors of the seven other tasks (progress-v2, Evaluator.registration), read from the label
# part lying in the task's zone and extrapolated over the label (as tube-label's): each is 1 where the task's own
# placement check passes and falls linearly to 0 where the placement stops being a label in that place. The bands
# repeat the success checks' tolerances (_place_*); the far ends are the item's own geometry or a diagonal label.
LABEL_SKEW_ZERO_DEG = TUBE_SKEW_ZERO_DEG   # bars / label edge 45 deg off: a diagonal label (gs1, canister)
GS1_BAR_BOTTOM_BAND = (.026, .038)  # m, barcode_bottom_32+-6mm_above_base; 0 where the label reaches the base / top
GS1_EDGE_CLEARANCE = .019           # m, barcode_>=19mm_from_vertical_edges; 0 with the symbol at the vertical edge
CANISTER_MID_TOL = .030             # m, middle_within_30mm_of_mid_height; 0 where the label reaches the wall's end
CANISTER_END_MARGIN = .020          # m, the same check's 20 mm from the top and the bottom; 0 at the wall's end
# The six take-and-apply tasks credit the take, the first clause of each instruction ("Take the label from the
# printer ..."), with weight TAKE_CREDIT: progress = integrity x R x (TAKE_CREDIT x s + (1 - TAKE_CREDIT) x credited
# area), s the share of the label area that lay on the printer at the start and is now off it without being stuck in
# a wrong place (Evaluator.take_share). Every finished application has s = 1, by any route; a held, undamaged label
# earns TAKE_CREDIT and no more. An authored weight, not a measurement. tube-label ("Take the printed patient label
# and apply it ...") has it too.
TAKE_CREDIT = .1
# Wet-out is credited against the pass line (WET_OUT_FRACTION of the standard, capped at 1) on every task, so no credit
# is given for pressing beyond what success asks.

# Wet-out reference: the contact fraction a standard firm application reaches on the same surface.
# The standard application pressure is 3M's published PSA guidance, "approximately 100 kPa (15 psi)"
# of roller or platen pressure (web-sourced: 3M VHB Tape 5952 technical data sheet, application
# section; the same figure appears across 3M PSA bulletins). Independent check (not the anchor): the
# ASTM D3330 2 kg roller pass in the virtual kraft peel bench left phi = 0.894, which inverts to
# ~88.8 kPa through phi = 1 - exp(-(p + p_tack)/p_r) with p_r = 40 kPa -- within 11 % of the guidance.
# The 100 kPa is a generic firm-application pressure TRANSFERRED from VHB tape guidance, not a measured
# pressure-to-contact law for this label on kraft; the 90 % pass fraction is an authored task choice. A label
# counts as wetted out when it reaches WET_OUT_FRACTION of the reference on the surface it is bonded to.
P_STANDARD_APPLICATION = 100e3                                 # Pa, 3M guidance (web-sourced)
P_ROLLER_BENCH_CHECK = -40e3*np.log(1 - .894) - 1e3            # Pa, model-derived consistency check
WET_OUT_FRACTION = .90
# The label solver's adhesive contact band (labelsim: bond formation reach, `half + 5e-5`): a node closer
# than this to a surface is in contact by the solver's own criterion. Trapped air and lifted edges are
# judged against it (engine-derived, not authored).
CONTACT_BAND = 5e-5
TABLE = 0.91                      # bench top height (scene.TABLE; repeated so the evaluator imports no scene)
BENCH_HALF = (.915, .38)          # bench top half extents (scene.room)
REST_TOL = 3e-3                   # m: an item resting on the bench has its base within 3 mm of the top
UPRIGHT_DEG = 5.
# Task acceptance geometry, not physics. The render's carton edge bevel is 1.5 mm;
# there is no 5 mm physical fold radius in this scene.
CARTON_TOP_EDGE_CLEARANCE = 1.5e-3
# Authored visual-alignment tolerance: allow a sub-mm registration error capward,
# while retaining the existing 2 mm allowance below the manufacturer's edge.
TUBE_CAPWARD_TOL = .5e-3
TUBE_BELOW_TOL = 2e-3
# tube-label "over the tube manufacturer's label": the 25.4 mm patient label covers about 222 deg of the 13 mm tube
# and the manufacturer's label 240 deg, so a centred label has about 1 mm to spare each side. Authored misregistration
# tolerance round the tube, the same 2 mm the axial rule allows below the manufacturer's edge: at every station along
# the label, at most this much of the manufacturer label's bare viewing strip may lie under the patient label. Not a
# cited measurement. tube-wrap checks only the window (>= 5 mm).
TUBE_STRIP_TOL = 2e-3
# What the 2 mm decides: the check reads the strip at the window measure's nine stations, so a skewed label's corners
# can lie up to about 0.4 mm further onto the strip (2.40 mm at 5 deg skew); and it is stricter than any window rule
# found: analyzers need 6-6.5 mm left open (Roche cobas p 671, "c = min. 6.5 mm"; Waikato, 6 mm after a lab label),
# which a label 2 mm onto the strip leaves many times over. The 2 mm enforces the instruction's "over the tube
# manufacturer's label" (Roche: the manufacturer marking "should be covered with the label when possible"), not the
# window: a label turned 27-45 deg round the tube, leaving 9-11.5 mm of window, fails it. A label that has left the
# bench with nothing holding it (Evaluator._label_lost: no node bonded to anything -- a label still stuck to the bench
# or the printer is not "held by nothing" -- and no node within reach of a robot or tool collider; the solver's pinch
# record and contact loads are not read, they go stale on a label moved away) cannot be made good -- the arms cannot
# reach the floor, and the floor has no label collider, so it keeps falling -- and zeroes integrity as a torn one
# does, instead of keeping its take credit. The cause (take_share counts every unbonded node as taken) is shared with
# the other take-and-apply tasks; only tube-label applies this rule.
LOST_LABEL_ZEROES = ("tube-label",)

# labelled item per task; part-id-match reads its target from the scene (seeded plate)
TARGET = {"carrier-top": "carton", "gs1-case-side": "carton", "cover-old-label": "carton", "press-down": "carton",
          "tube-wrap": "tube", "part-id-hdpe": "part", "part-id-match": None,
          "smooth-corner": "carton", "finish-securing": "carton", "tube-label": "tube"}


def vertex_normals(x, tris):
    fn = np.cross(x[tris[:, 1]] - x[tris[:, 0]], x[tris[:, 2]] - x[tris[:, 0]])
    vn = np.zeros_like(x)
    for k in range(3):
        np.add.at(vn, tris[:, k], fn)
    return vn/np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-15)


def material_points(lab, uv):
    """World positions and facestock normals of label MATERIAL points uv (N x 2: u = MD, v = CD), interpolated
    in the triangle of the rest mesh that contains each point (barycentric; exact for the piecewise-linear
    sheet). The mapping is cached on the label per point set."""
    uv = np.asarray(uv, float)
    key = uv.tobytes()
    caches = getattr(lab, "_material_point_cache", None)
    if not isinstance(caches, dict):
        caches = lab._material_point_cache = {}
    cache = caches.get(key)
    if cache is None or cache[0] is not lab.tris:
        X, T = lab.X, lab.tris
        a, b, c = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
        den = (b[:, 1] - c[:, 1])*(a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0])*(a[:, 1] - c[:, 1])
        tri = np.zeros(len(uv), int); bary = np.zeros((len(uv), 3))
        for k, p in enumerate(uv):
            l1 = ((b[:, 1] - c[:, 1])*(p[0] - c[:, 0]) + (c[:, 0] - b[:, 0])*(p[1] - c[:, 1]))/den
            l2 = ((c[:, 1] - a[:, 1])*(p[0] - c[:, 0]) + (a[:, 0] - c[:, 0])*(p[1] - c[:, 1]))/den
            l3 = 1. - l1 - l2
            worst = np.minimum(np.minimum(l1, l2), l3)
            j = int(np.argmax(worst))                       # containing triangle (or the nearest for edge points)
            tri[k], bary[k] = j, (l1[j], l2[j], l3[j])
        cache = caches[key] = (lab.tris, key, tri, bary)
    tri, bary = cache[2], cache[3]
    T = lab.tris[tri]
    x = lab.x
    pts = (bary[:, :, None]*x[T]).sum(1)
    fn = np.cross(x[T[:, 1]] - x[T[:, 0]], x[T[:, 2]] - x[T[:, 0]])
    return pts, fn/np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-15)


def symbol_rect(lab, art):
    """The printed symbol including its quiet zones, as a material rectangle ((u0, u1), (v0, v1))."""
    W, H = art["image_px"]
    x0, y0, x1, y1 = art["barcode_px"]
    # image x -> v (CD), image y top -> u = MD (leading edge)
    return ((1 - y1/H)*lab.length, (1 - y0/H)*lab.length), (x0/W*lab.width, x1/W*lab.width)


def symbol_samples(lab, art, n_u=9, n_v=17):
    """Material sample points over the symbol rectangle, boundary included (u varies slowest)."""
    (u0, u1), (v0, v1) = symbol_rect(lab, art)
    U, V = np.meshgrid(np.linspace(u0, u1, n_u), np.linspace(v0, v1, n_v), indexing="ij")
    return np.stack([U.ravel(), V.ravel()], 1)


def _clip_polygon(poly, axis, bound, keep_above=True):
    """Clip a convex polygon (with interpolated extra columns) to a half-plane."""
    if not len(poly):
        return poly
    result = []
    prev = poly[-1]
    dp = (prev[axis] - bound)*(1 if keep_above else -1)
    for point in poly:
        dc = (point[axis] - bound)*(1 if keep_above else -1)
        if (dp >= 0) != (dc >= 0):
            result.append(prev + dp/(dp - dc)*(point - prev))
        if dc >= 0:
            result.append(point)
        prev, dp = point, dc
    return np.asarray(result, float).reshape(-1, poly.shape[1])


def registration_factors(top, skew_deg, window, tube, label_long, bare_strip=None):
    """tube-wrap and tube-label registration factors (Evaluator.registration) for a label whose cap-side edge is `top`
    m from the cap end, turned skew_deg on the unrolled tube, leaving `window` m of the viewing strip; `tube` the
    scene's tube geometry, label_long the label's long side (applied lengthwise). Each is 1 inside the task's
    acceptance band and falls linearly to 0 where the placement stops being a labelled tube (TUBE_CAP_SKIRT,
    TUBE_SKEW_ZERO_DEG, no window left). Window: tube-wrap, window / 5 mm (its viewing_window check); tube-label
    (`bare_strip`, the manufacturer label's whole bare strip, given), 1 while at most TUBE_STRIP_TOL of the strip is
    covered (its over_mfr_label check), falling linearly to 0 with the whole strip covered."""
    want = tube["mfr_label_top_from_cap_end"]
    lo, hi = want - TUBE_CAPWARD_TOL, want + TUBE_BELOW_TOL
    on_cap = TUBE_CAP_SKIRT                                             # cap-side edge on the cap
    at_bottom = tube["L"] - TUBE_BOTTOM_CLEAR - label_long             # far end at the 9 mm bottom clearance
    r_cap = 1. if lo <= top <= hi else float(np.clip((top - on_cap)/(lo - on_cap) if top < lo
                                                     else (at_bottom - top)/(at_bottom - hi), 0., 1.))
    r_skew = float(np.clip((TUBE_SKEW_ZERO_DEG - skew_deg)/(TUBE_SKEW_ZERO_DEG - 5.), 0., 1.))
    knee = 5e-3 if bare_strip is None else bare_strip - TUBE_STRIP_TOL
    r_win = float(np.clip(window/knee, 0., 1.))
    return r_cap, r_skew, r_win, dict(zero_on_cap=on_cap, band=(lo, hi), zero_at_bottom_clearance=at_bottom)


def _level_factor(tilt_deg):
    """1 up to the 5 deg the placement checks allow, 0 at a diagonal label (LABEL_SKEW_ZERO_DEG)."""
    return float(np.clip((LABEL_SKEW_ZERO_DEG - tilt_deg)/(LABEL_SKEW_ZERO_DEG - 5.), 0., 1.))


def gs1_registration_factors(bottom, clearance, tilt_deg, label_low, label_high, H):
    """gs1-case-side registration factors for a label whose SSCC symbol's printed bottom edge is `bottom` m above the
    carton's base, the symbol with its quiet zones `clearance` m from the nearer vertical edge, its bars tilt_deg off
    vertical, and the label itself spanning label_low..label_high m above the base on a side H m tall. Each is 1
    where the placement check passes: the barcode height falls linearly to 0 where the label would reach the base (or
    the top edge) by moving the same way, the edge clearance to 0 with the symbol at the edge, the bars to 0 at 45 deg."""
    lo, hi = GS1_BAR_BOTTOM_BAND
    if lo <= bottom <= hi:
        r_h = 1.
    elif bottom < lo:
        r_h = float(np.clip(label_low/(lo - bottom + label_low), 0., 1.)) if label_low > 0 else 0.
    else:
        room = H - label_high
        r_h = float(np.clip(room/(room + bottom - hi), 0., 1.)) if room > 0 else 0.
    r_e = float(np.clip(clearance/GS1_EDGE_CLEARANCE, 0., 1.))
    return r_h, r_e, _level_factor(tilt_deg)


def canister_registration_factors(mean_z, top_margin, bottom_margin, tilt_deg):
    """part-id-hdpe registration factors for a label whose nodes' mean height is mean_z m from the canister's
    mid-height, leaving top_margin / bottom_margin m of side wall above / below it, its long side tilt_deg off level.
    Each is 1 where the placement check passes: mid-height falls linearly to 0 where the label would reach the end of
    the wall by moving the same way, the margin to 0 at the wall's end, level to 0 at 45 deg."""
    r_margin = float(np.clip(min(top_margin, bottom_margin)/CANISTER_END_MARGIN, 0., 1.))
    off = abs(mean_z) - CANISTER_MID_TOL
    room = top_margin if mean_z > 0 else bottom_margin
    r_mid = 1. if off <= 0 else (float(np.clip(room/(room + off), 0., 1.)) if room > 0 else 0.)
    return r_mid, r_margin, _level_factor(tilt_deg)


def lifted_material_area(X, tris, gap, band=CONTACT_BAND, rect=None):
    """Area where the piecewise-linear nodal gap exceeds band, optionally inside
    a material-space rectangle ((u0,u1),(v0,v1)). Integrate clipped triangles,
    not lumped areas of vertices selected by a Boolean predicate. This removes
    node-sized area jumps; it cannot recover bubbles absent from the mesh."""
    total = 0.
    selected = tris[np.max(gap[tris], axis=1) > band]
    for ids in selected:
        poly = _clip_polygon(np.column_stack((X[ids], gap[ids])), 2, band)
        if rect is not None:
            for axis, (lo, hi) in enumerate(rect):
                poly = _clip_polygon(poly, axis, lo)
                poly = _clip_polygon(poly, axis, hi, keep_above=False)
        if len(poly) >= 3:
            uv = poly[:, :2] - poly[0, :2]
            total += .5*abs(np.sum(uv[:, 0]*np.roll(uv[:, 1], -1)
                                  - uv[:, 1]*np.roll(uv[:, 0], -1)))
    return float(total)


class Evaluator:
    def __init__(self, world, art):
        self.world, self.art = world, art
        lab = world.label
        self.boundary_band = self._band(3e-3)
        # barcode region in label material coordinates (u = MD from trailing edge, v = CD)
        self.bar_u, self.bar_v = symbol_rect(lab, art)
        X = lab.X
        self.in_barcode = (X[:, 1] >= self.bar_v[0]) & (X[:, 1] <= self.bar_v[1]) & \
                          (X[:, 0] >= self.bar_u[0]) & (X[:, 0] <= self.bar_u[1])
        self.initial_items = {n: world.body_pose(n)[0].copy() for n in world.scene.items}
        # the label area on the printer's presentation (its liner web and peel bar) when the evaluator is made -- in
        # LabelEnv after the settle, i.e. the episode's start (env.reset restores it); the take share is measured from it
        cols = getattr(lab, "colliders", None)
        self.start_presented = (np.isin(lab.bond, [i for i, c in enumerate(cols) if c.role in ("liner", "plate")])
                                if cols is not None and hasattr(lab, "bond") else None)
        task = world.config.task
        self.target = getattr(world.scene, "target", None) or TARGET[task]
        self.last_optical = None

    def _band(self, width):
        lab = self.world.label
        X = lab.X
        return (X[:, 0] < width) | (X[:, 0] > lab.length - width) | (X[:, 1] < width) | (X[:, 1] > lab.width - width)

    def _target_colliders(self):
        lab = self.world.label
        return [i for i, c in enumerate(lab.colliders) if c.name == self.target]

    def _target_distance(self, x):
        """Signed distance and outward normal of the nearest target collider surface."""
        lab = self.world.label
        best_d, best_n = None, None
        for i in self._target_colliders():
            d, n, _ = lab.colliders[i].sdf(x)
            if best_d is None:
                best_d, best_n = d, n
            else:
                m = d < best_d
                best_d, best_n = np.where(m, d, best_d), np.where(m[:, None], n, best_n)
        return best_d, best_n

    def evaluate(self, optical=None):
        w = self.world
        lab = w.label
        task = w.config.task
        target_name = self.target
        cols = [c.name for c in lab.colliders]
        tidx = self._target_colliders()
        A = lab.vertex_area
        on_target = np.isin(lab.bond, tidx)
        # per-vertex reference wet-out of the surface the vertex is bonded to (target base otherwise)
        from . import materials as M
        pr = np.full(lab.n, lab.colliders[tidx[0]].surface.p_rough)
        for i in np.flatnonzero(on_target):
            pr[i] = lab.colliders[lab.bond[i]].surface_list()[lab.bond_surface[i]].p_rough
        phi_ref = 1. - np.exp(-(P_STANDARD_APPLICATION + M.P_TACK)/pr)
        rel = np.minimum(lab.phi/phi_ref, 1.)*on_target
        frac = float((rel*A).sum()/A.sum())
        band = self.boundary_band
        band_frac = float((rel[band]*A[band]).sum()/A[band].sum())
        # edges stuck down: a boundary node is down if bonded to the item, or lying within the solver's contact
        # band with its adhesive face toward the item (released in contact). A node standing
        # off, or folded back over the label (adhesive facing away), is lifted. One node (3 mm) is tolerated.
        half = .5*lab.stock.thickness
        d, nrm = self._target_distance(lab.x)
        gap = d - half
        facing = (vertex_normals(lab.x, lab.tris)*nrm).sum(1) > 0.
        down = on_target | ((gap <= CONTACT_BAND) & facing)
        lifted = lab.boundary & ~down
        lifted_len = float(lifted.sum()*lab.h)
        air = self._trapped_air()
        # permanent set (creases) inside the barcode region
        H = lab.hinge
        in_bar_h = self.in_barcode[H[:, 0]] & self.in_barcode[H[:, 1]]
        crease = float(np.abs(lab.theta_rest[in_bar_h]).max()/lab.hinge_dual[in_bar_h].mean()) if in_bar_h.any() else 0.
        peak = getattr(w, "set_peak", None)
        crease_peak = max(crease, float(peak[in_bar_h].max()/lab.hinge_dual[in_bar_h].mean())
                          if peak is not None and in_bar_h.any() else crease)
        other = sorted({cols[b] for b in np.unique(lab.bond[lab.bond >= 0]) if cols[b] != target_name})
        speed = float(np.linalg.norm(lab.v, axis=1).max())                       # instantaneous (reported)
        hist = getattr(w, "label_history", None)
        if hist is not None and len(hist) >= 2:
            span = (len(hist) - 1)*w.config.label_dt
            drift = float((np.linalg.norm(lab.x - hist[0], axis=1)).max()/span)  # net motion over <= 0.1 s
        else:
            drift = speed
        moved = {n: float(np.linalg.norm(w.body_pose(n)[0] - p0)) for n, p0 in self.initial_items.items()}
        resting = self.resting(target_name)
        hands = self.hands_off(target_name)
        placement = (self._place_repair_rung if task in REPAIR_TASKS else self._place_tube_label if task == "tube-label"
                     else getattr(self, "_place_" + base_task(task).replace("-", "_")))(on_target)
        checks = {
            "wet_out>=90%_of_standard_application": frac >= WET_OUT_FRACTION,
            "edge_band_wet_out>=90%_of_standard_application": band_frac >= WET_OUT_FRACTION,
            "all_edges_stuck_down": lifted_len <= 3e-3,
            "no_trapped_air_over_20mm2": air["largest_m2"] <= 20e-6,
            "no_trapped_air_over_3mm2_in_barcode": air["in_barcode_m2"] <= 3e-6,
            **({"no_crease_in_barcode(set<119/m_at_any_time)": crease_peak < BARCODE_CREASE_YIELD} if task in LADDER_TASKS
               else {"no_crease_in_barcode(curvature<119/m set)": crease < BARCODE_CREASE_YIELD}),
            "label_not_torn": not lab.torn,
            "label_bonded_only_to_target": not other,
            "label_settled(<1mm/s)": drift < 1e-3,
            "item_resting_on_bench": resting["ok"],
            **({"hands_and_tools_off_label_and_item": hands["ok"]} if hands is not None else {}),
            **{f"placement:{k}": bool(v) for k, v in placement["checks"].items()},
        }
        lifted_areas = None
        if task in REPAIR_TASKS:
            # every area that started lifted must itself reach the standard the whole label is held to: the label
            # average alone passes with a corner only tacked down (a lifted corner is 2-4 % of the label). Its own
            # 3 mm label edge is held to the same standard as the label's edge band: the area average alone passes
            # a corner pressed down except its tip
            lifted_areas = self.lifted_area_wet_out(rel*on_target)
            checks["each_lifted_area_wet_out>=90%_of_standard"] = bool(
                all(r["wet_out_relative"] >= WET_OUT_FRACTION for r in lifted_areas))
            checks["each_lifted_area_edge_band_wet_out>=90%_of_standard"] = bool(
                all(r["edge_band_wet_out_relative"] >= WET_OUT_FRACTION for r in lifted_areas))
            # the same two standards over the label's whole perimeter (its four corners, two ends and two sides at the
            # largest size the task can leave lifted, layouts.edge_regions): a part that comes loose during the work
            # and lies back down at touch-down contact is the same defect as a lifted area left unpressed
            from . import layouts
            edge_regions = self.area_wet_out(layouts.edge_regions(), rel*on_target)
            checks["every_label_edge_region_wet_out>=90%_of_standard"] = bool(
                all(r["wet_out_relative"] >= WET_OUT_FRACTION for r in edge_regions))
            checks["every_label_edge_region_edge_band_wet_out>=90%_of_standard"] = bool(
                all(r["edge_band_wet_out_relative"] >= WET_OUT_FRACTION for r in edge_regions))
            edge_windows = self.edge_window_wet_out(rel*on_target)
            checks["every_25mm_label_edge_window_wet_out>=90%_of_standard"] = bool(
                edge_windows["lowest"] >= WET_OUT_FRACTION)
        if optical is not None:
            checks["optical_decode_matches_payload"] = bool(optical.get("match"))
        success = bool(all(checks.values()))
        # progress: every task (REWARD_CONTRACTS). The crease that zeroes it is the one the task's own crease check
        # reads: the episode's largest on the ladder, the present set on the seven other tasks (on the tube the wrap
        # can bend a crease back below the line, and tube-wrap's success accepts that label)
        reward = self.progress(dict(on_target=on_target, rel=rel, gap=gap, facing=facing, resting=resting,
                                    hands=hands, other=other, crease=crease_peak if task in LADDER_TASKS else crease))
        score = 1. if success else FAIL_SCORE_SCALE*reward["progress"]
        reward.update(contract=REWARD_CONTRACTS[task], score_rule="1 if success else 0.9 x progress")
        if task not in LADDER_TASKS:
            # the fraction of passed checks (mean-of-checks), for comparison only
            reward["mean_of_checks"] = float(np.mean(list(checks.values())))
        out = dict(success=success, score=score, checks=checks,
                    metrics=dict(wet_out_relative=frac, edge_band_wet_out_relative=band_frac,
                                 bonded_area_fraction=float((A*on_target).sum()/A.sum()), lifted_edge_m=lifted_len,
                                 trapped_air=air, largest_unbonded_enclosed_m2=air["largest_enclosed_m2"],
                                 barcode_set_curvature_1_m=crease, barcode_peak_set_curvature_1_m=crease_peak,
                                 max_label_speed_m_s=speed,
                                 max_label_net_speed_0p1s_m_s=drift, target=target_name,
                                 bonded_elsewhere=other, item_displacement_m=moved, item_rest=resting,
                                 robot_contacts=hands, torn=lab.torn,
                                 placement=placement["metrics"], optical=optical,
                                 max_membrane_force_n_m=getattr(lab, "max_membrane_force", 0.)),
                    optical_evaluated=optical is not None)
        if lifted_areas is not None:
            out["metrics"]["lifted_areas"] = lifted_areas
            out["metrics"]["edge_regions"] = edge_regions
            out["metrics"]["edge_windows"] = edge_windows
        out["progress"] = reward["progress"]
        out["reward"] = reward
        return out

    # ------------------------------------------------------------------ progress reward (every task; REWARD_CONTRACTS)
    def _contact_state(self):
        """Per label node: bonded to the target, wet-out relative to the standard on the bonded surface, gap to the
        target surface and whether the adhesive face is toward it (the quantities evaluate() uses)."""
        lab = self.world.label
        tidx = self._target_colliders()
        on_target = np.isin(lab.bond, tidx)
        from . import materials as M
        pr = np.full(lab.n, lab.colliders[tidx[0]].surface.p_rough)
        for i in np.flatnonzero(on_target):
            pr[i] = lab.colliders[lab.bond[i]].surface_list()[lab.bond_surface[i]].p_rough
        rel = np.minimum(lab.phi/(1. - np.exp(-(P_STANDARD_APPLICATION + M.P_TACK)/pr)), 1.)*on_target
        d, nrm = self._target_distance(lab.x)
        gap = d - .5*lab.stock.thickness
        facing = (vertex_normals(lab.x, lab.tris)*nrm).sum(1) > 0.
        return on_target, rel, gap, facing

    def zone(self, on_target=None):
        """Label nodes that lie where the task lets the label go (item frame, per node). The carton-top tasks: on the
        carton top, on the right flap half, off the seam tape and clear of the top's edge bevel (the carrier placement
        rules, node by node); the flap half is the one holding the label (flap_half) on the repair rungs and
        carrier-top (either half is accepted), the one the label was laid on for press-down ("where it lies") and the
        one carrying the old label for cover-old-label. tube-wrap and tube-label: on the tube wall clear of its bottom
        9 mm (clear_of_tube_bottom). gs1-case-side: on the long side face holding the label (gs1_face). part-id-hdpe:
        on the canister's side wall (not its lid or base). part-id-match: on the target plate's upper face, inside
        its footprint. The remaining placement rules enter the reward through registration()."""
        task = self.world.config.task
        base = base_task(task)
        if task in REPAIR_TASKS or base in ("carrier-top", "press-down", "cover-old-label"):
            g = self.world.scene.task_geometry["carton"]
            q = self._item_local("carton")
            if task in REPAIR_TASKS or base == "carrier-top":
                side = self.flap_half(q, on_target)
            elif base == "press-down":
                side = float(np.sign(self.world.scene.task_geometry["laid_label"]["centre"][1]))
            else:
                side = float(self.world.scene.task_geometry["old_label"]["side"])
            c = CARTON_TOP_EDGE_CLEARANCE - 1e-12
            return ((q[:, 2] > g["H"]/2 - 2e-3) & (np.abs(q[:, 0]) <= g["L"]/2 - c)
                    & (side*q[:, 1] >= g["tape_half"]) & (side*q[:, 1] <= g["W"]/2 - c))
        if base == "tube-wrap":
            g = self.world.scene.task_geometry["tube"]
            q = self._item_local("tube")
            return q[:, 2] >= -g["L"]/2 + TUBE_BOTTOM_CLEAR
        if base == "gs1-case-side":
            g = self.world.scene.task_geometry["carton"]
            q = self._item_local("carton")
            side = self.gs1_face(q, on_target)
            # the on_a_long_side_face rule node by node: beyond that face's plane less 2 mm, inside the face
            return ((side*q[:, 1] > g["W"]/2 - 2e-3) & (np.abs(q[:, 0]) <= g["L"]/2)
                    & (np.abs(q[:, 2]) <= g["H"]/2))
        if base == "part-id-hdpe":
            it = self.world.scene.items["part"]
            q = self._item_local("part")
            # the on_side rule node by node (radius > r - 1 mm), between the lid and the base
            return (np.hypot(q[:, 0], q[:, 1]) > it["radius"] - 1e-3) & (np.abs(q[:, 2]) < it["height"]/2)
        if base == "part-id-match":
            sx, sy, sz = self.world.scene.items[self.target]["size"]
            q = self._item_local(self.target)
            # the on_target_plate_upper_face rule node by node
            return (q[:, 2] > sz/2 - 1e-3) & (np.abs(q[:, 0]) <= sx/2) & (np.abs(q[:, 1]) <= sy/2)
        raise ValueError(f"{task} has no progress zone")

    def flap_half(self, q=None, on_target=None):
        """Repair rungs and carrier-top: the carton-top flap half (+1 / -1, the sign of carton-local y) holding the
        label, i.e. the half with more label area bonded to the carton. Either half is accepted (one_flap_half). The
        repair rungs' settled start is bonded on the half it was laid on, so its start credit is unchanged; with nothing
        bonded, that half (carrier-top, which has no laid label: +1, and nothing is credited then)."""
        lab = self.world.label
        q = self._item_local("carton") if q is None else q
        if on_target is None:
            on_target = self._contact_state()[0]
        s = float((lab.vertex_area*on_target*np.sign(q[:, 1])).sum())
        if s != 0.:
            return float(np.sign(s))
        lay = self.world.scene.task_geometry.get("laid_label")
        return float(np.sign(lay["centre"][1])) if lay is not None else 1.

    def gs1_face(self, q=None, on_target=None):
        """gs1-case-side: the long side face (-1: front, -y; +1: back, +y) holding the label, i.e. the one with more
        label area bonded to the carton within 2 mm of its plane; with nothing bonded there, the front (nothing is
        credited then). Either long side is accepted (on_a_long_side_face)."""
        lab = self.world.label
        g = self.world.scene.task_geometry["carton"]
        q = self._item_local("carton") if q is None else q
        if on_target is None:
            on_target = self._contact_state()[0]
        a = {s: float((lab.vertex_area*on_target*(s*q[:, 1] > g["W"]/2 - 2e-3)).sum()) for s in (-1., 1.)}
        return 1. if a[1.] > a[-1.] else -1.

    def node_credit(self, state=None):
        """Per label node, in [0, 1]: one half for lying adhesive-down on the target surface inside the zone
        (bonded to it, or within the solver's adhesive contact band with the adhesive face toward it), one half
        times the node's wet-out once it is bonded there, relative to the pass line (WET_OUT_FRACTION of the standard
        application) and capped at 1.
        A torn-away node counts nothing. It reads the final state only; how or in which order the label got there does
        not enter."""
        lab = self.world.label
        on_target, rel, gap, facing = state if state is not None else self._contact_state()
        # wet-out credited against the task's own pass line, so label pressed to what success accepts is full
        # credit: on the repair rungs pressed pre-applied label is not counted as damage; elsewhere no credit is given
        # for pressing beyond the line
        rel = np.minimum(rel/WET_OUT_FRACTION, 1.)
        alive = np.asarray(getattr(lab, "vertex_alive", np.ones(lab.n, bool)), bool)
        zone = self.zone(on_target) & alive
        placed = (on_target | ((gap <= CONTACT_BAND) & facing)) & zone
        return .5*placed + .5*(on_target & zone)*rel

    def lifted_area_wet_out(self, rel_on_target):
        """Repair rungs: area-weighted wet-out (relative to the standard) over each area that started lifted
        (material regions, layouts.partly_applied), and over that area's own 3 mm label-edge band (its nodes in
        boundary_band, the band edge_band_wet_out judges label-wide)."""
        return self.area_wet_out(self.world.scene.task_geometry["laid_label"]["regions"], rel_on_target)

    def area_wet_out(self, regions, rel_on_target):
        """Per material region (layouts flap dicts): area-weighted wet-out relative to the standard, over the region
        and over its own 3 mm label-edge band."""
        from . import layouts
        lab = self.world.label
        A = lab.vertex_area
        out = []
        for f in regions:
            m = layouts.flap_mask(lab.X, f)[0]
            b = m & self.boundary_band
            out.append(dict(area=f["name"], area_mm2=float(A[m].sum()*1e6),
                            wet_out_relative=float((rel_on_target[m]*A[m]).sum()/A[m].sum()),
                            edge_band_nodes=int(b.sum()),
                            edge_band_wet_out_relative=float((rel_on_target[b]*A[b]).sum()/A[b].sum()) if b.any() else 1.))
        return out

    def _edge_window_weights(self):
        """(names, W): the repair rungs' edge windows (EDGE_WINDOW squares on the label edge, every EDGE_WINDOW_STEP along
        each of the four edges and flush at its ends: the label nodes within EDGE_WINDOW of that edge and within
        EDGE_WINDOW/2 of the window's centre along it), W the windows x nodes matrix of vertex areas inside each."""
        cache = getattr(self, "_edge_windows_cache", None)
        lab = self.world.label
        if cache is not None and cache[0] is lab.vertex_area:
            return cache[1], cache[2]
        X, L, W_, S = lab.X, lab.length, lab.width, EDGE_WINDOW
        names, rows = [], []
        for name, depth, along, span in (("u=0 end", X[:, 0], X[:, 1], W_), ("u=L end", L - X[:, 0], X[:, 1], W_),
                                         ("v=0 side", X[:, 1], X[:, 0], L), ("v=W side", W_ - X[:, 1], X[:, 0], L)):
            for c in np.unique(np.round(np.r_[np.arange(S/2, span - S/2 + 1e-9, EDGE_WINDOW_STEP), span - S/2], 9)):
                m = (depth < S) & (np.abs(along - c) <= S/2)
                names.append(f"{name} at {c*1e3:.0f} mm")
                rows.append(lab.vertex_area*m)
        Wm = np.asarray(rows)
        self._edge_windows_cache = (lab.vertex_area, names, Wm)
        return names, Wm

    def edge_window_wet_out(self, rel_on_target):
        """Repair rungs: the lowest area-weighted wet-out (relative to the standard) over the edge windows, where it
        is, and how many windows fall below the standard."""
        names, Wm = self._edge_window_weights()
        v = (Wm @ rel_on_target)/Wm.sum(1)
        k = int(np.argmin(v))
        return dict(lowest=float(v[k]), where=names[k], windows=len(names),
                    below_standard=int((v < WET_OUT_FRACTION).sum()), side_m=EDGE_WINDOW)

    def stage(self, on_target, robot_cols, gap=None, creased=False):
        """Diagnostic only, never in the score: where the label is (bonded to the item / to the robot or a tool /
        still on the printer's liner / free), with bonded node counts per surface. Damage is in the flags.
        `pre_contact` names the state the label is in, damage first, by the reward's own damage test: "torn"; "barcode
        creased" (the task's own crease measure at or above the paper's yield, `creased`); "on the item"; "still on the printer's
        liner"; "taken off the liner, stuck elsewhere" (bonded to a surface that is neither the target, the robot, a
        tool nor the printer's presentation: the bench, the printer's housing, another item); "taken off the liner
        undamaged, held" (by a robot surface or tool); "taken off the liner undamaged, loose". It also gives the
        closest gap to the target surface and the item the label is nearest, and whether that is the target (for
        part-id-match, the plate the label was brought to; while the label is still on the liner this is only the item
        nearest the printer). It reads the final state only."""
        lab = self.world.label
        names = [c.name for c in lab.colliders]
        b = lab.bond[lab.bond >= 0]
        counts = {n: int(k) for n, k in zip(*np.unique([names[i] for i in b], return_counts=True))} if len(b) else {}
        liner = [i for i, c in enumerate(lab.colliders) if c.role == "liner"]
        where = ("on the item" if on_target.any() else "on the robot or a tool" if np.isin(lab.bond, robot_cols).any()
                 else "on the printer's liner" if np.isin(lab.bond, liner).any() else "free")
        # the printer's presentation: its liner web and peel bar (role "plate"); a label stuck to the printer's housing
        # has been taken off the liner
        presented = [i for i, c in enumerate(lab.colliders) if c.role in ("liner", "plate")]
        on_liner = int(np.isin(lab.bond, presented).sum())
        held = bool(np.isin(lab.bond, robot_cols).any())
        tidx = self._target_colliders()
        elsewhere = sorted({names[i] for i in np.unique(b) if i not in tidx and i not in robot_cols and i not in presented})
        alive = np.asarray(getattr(lab, "vertex_alive", np.ones(lab.n, bool)), bool)
        items = self.world.scene.items
        near = {}
        for i, c in enumerate(lab.colliders):
            if c.name in items and alive.any():
                d = c.sdf(lab.x[alive])[0] - .5*lab.stock.thickness
                near[c.name] = min(near.get(c.name, np.inf), float(d.min()))
        nearest = min(near, key=near.get) if near else None
        phase = ("torn" if lab.torn else "barcode creased" if creased else "on the item" if on_target.any()
                 else "still on the printer's liner" if on_liner else "taken off the liner, stuck elsewhere" if elsewhere
                 else "taken off the liner undamaged, held" if held else "taken off the liner undamaged, loose")
        return dict(where=where, bonded_nodes=counts,
                    pre_contact=dict(phase=phase, intact=not (lab.torn or creased), taken_off_liner=not on_liner,
                                     nodes_on_liner=on_liner, held_by_robot_or_tool=held, stuck_elsewhere=elsewhere,
                                     closest_gap_to_target_m=(float(np.asarray(gap)[alive].min())
                                                              if gap is not None and alive.any() else None),
                                     nearest_item=nearest, nearest_item_gap_m=near.get(nearest),
                                     nearest_item_is_target=nearest == self.target if nearest else None))

    def take_share(self, on_target, zone):
        """The take-and-apply tasks (the six, and tube-label): the share of the label area that lay on the printer's
        presentation (liner web, peel bar) at the start (start_presented) and is now off every printer part without
        being stuck in a wrong place: unbonded, bonded only to a robot surface or tool, or on the target inside the
        task's zone. Label stuck to the bench, the printer's housing, another item or the target outside the zone does
        not count. 0 untouched; 1 for every finished application, whatever the route (by hand, by tool, or applied
        straight from the peel bar onto an item brought to the printer). Damage is not judged here: integrity zeroes a
        torn or creased label. It reads the final state only."""
        lab = self.world.label
        m0 = self.start_presented
        if m0 is None or not m0.any():
            return 0.
        cols = lab.colliders
        printer = np.array([c.role in ("liner", "plate") or c.name.startswith("printer") for c in cols])
        robot = np.array([c.role in ("pad", "robot", "tool") for c in cols])
        b = lab.bond
        bi = np.maximum(b, 0)
        alive = np.asarray(getattr(lab, "vertex_alive", np.ones(lab.n, bool)), bool)
        ok = (b < 0) | ((b >= 0) & robot[bi]) | (on_target & zone)
        taken = m0 & alive & ~((b >= 0) & printer[bi]) & ok
        A = lab.vertex_area
        return float((A*taken).sum()/(A*m0).sum())

    @staticmethod
    def _off_bench(p):
        return bool(abs(p[0]) > BENCH_HALF[0] or abs(p[1]) > BENCH_HALF[1] or p[2] < TABLE - .02)

    def _item_off_bench(self):
        """The item has left the bench top (knocked to the floor or out over the edge): the work cannot be finished."""
        return self._off_bench(self.world.body_pose(self.target)[0])

    def _label_lost(self, on_target, robot_cols):
        """The label has left the bench top with nothing holding it: no node on the target, no alive node bonded to
        anything (a label still stuck to the bench, the printer or a robot is held by it, so the bonded part keeps it
        from being "lost" whatever share of it hangs over the edge), no node within reach of a robot or tool collider
        (holding by pinch or contact needs the pads at the label; the solver's pinch record and contact loads are not
        read, as they are refreshed only while the label is near a collider and go stale on a label moved away), and
        its area centroid outside the bench top by _item_off_bench's test. A label
        lying on the bench, stuck to it, or held out past its edge, is not lost."""
        lab = self.world.label
        alive = np.asarray(getattr(lab, "vertex_alive", np.ones(lab.n, bool)), bool)
        if on_target.any() or (np.asarray(lab.bond)[alive] >= 0).any():
            return False
        x = lab.x[alive]
        for i in robot_cols:                       # bounding sphere of the collider, plus the solver's 1.5 mm reach
            c = lab.colliders[i]
            if (np.linalg.norm(x - np.asarray(c.p, float), axis=1) <= float(np.linalg.norm(c.size)) + 1.5e-3).any():
                return False
        A = lab.vertex_area*alive
        return self._off_bench((A[:, None]*lab.x).sum(0)/max(float(A.sum()), 1e-30))

    def registration(self, placed, on_target=None):
        """How well the label part lying in the task's zone (`placed` nodes) is registered, as a product of factors in
        [0, 1], each 1 where the task's own placement check passes: tube-wrap and tube-label, _tube_registration;
        gs1-case-side, _gs1_registration; cover-old-label, _cover_registration; part-id-hdpe, _canister_registration;
        part-id-match, _plate_registration. carrier-top's and press-down's zone already holds every placement rule
        (factor 1). Fewer than 3 placed nodes, or collinear ones, leave nothing to register (factor 1; progress is
        then about 0 anyway)."""
        base = base_task(self.world.config.task)
        fn = dict(zip(("tube-wrap", "gs1-case-side", "cover-old-label", "part-id-hdpe", "part-id-match"),
                      (self._tube_registration, self._gs1_registration, self._cover_registration,
                       self._canister_registration, self._plate_registration))).get(base)
        if fn is None:
            return dict(factor=1., note="the zone holds every placement rule")
        return fn(placed) if base == "tube-wrap" else fn(placed, on_target)

    def _fit(self, coords, placed):
        """Affine map from label material coordinates (u, v) to item-frame coordinates `coords` (one row per placed
        node), least squares over the placed nodes: (3 x k) coefficients, rows d/du, d/dv, offset. None when fewer than
        3 nodes are placed or they are collinear."""
        lab = self.world.label
        if placed.sum() < 3:
            return None
        coef, _, rank, _ = np.linalg.lstsq(np.c_[lab.X[placed], np.ones(int(placed.sum()))], coords, rcond=None)
        return coef if rank >= 3 else None

    def _fitted_cover(self, coef, pts):
        """Fraction of the item-plane points pts (N x 2) inside the label rectangle mapped by the affine fit coef (the
        label part lying on the face extrapolated over the whole label)."""
        lab = self.world.label
        Ainv = np.linalg.pinv(coef[:2])
        uv = (np.asarray(pts, float) - coef[2]) @ Ainv
        e = 1e-9
        inside = (uv[:, 0] >= -e) & (uv[:, 0] <= lab.length + e) & (uv[:, 1] >= -e) & (uv[:, 1] <= lab.width + e)
        return float(inside.mean())

    def _gs1_registration(self, placed, on_target=None):
        """gs1-case-side: the label part on the long side face, fitted over the label (_fit on the face's along / up
        coordinates), gives the SSCC symbol's printed bottom-edge height above the base, its clearance (quiet zones
        included) from the nearer vertical edge and the bars' tilt from vertical, measured as _place_gs1_case_side
        measures them, and the label's own height span; gs1_registration_factors turns them into factors."""
        g = self.world.scene.task_geometry["carton"]
        lab = self.world.label
        q = self._item_local("carton")
        coef = self._fit(q[placed][:, [0, 2]], placed)
        if coef is None:
            return dict(factor=1., note="fewer than 3 label nodes (or collinear ones) on the side face: nothing to register")
        (u0, u1), (v0, v1) = self.bar_u, self.bar_v
        n = 25
        bottom_uv = np.stack([np.full(n, u0), np.linspace(v0, v1, n)], 1)
        edge_uv = np.concatenate([bottom_uv, np.stack([np.full(n, u1), np.linspace(v0, v1, n)], 1),
                                  np.stack([np.linspace(u0, u1, n), np.full(n, v0)], 1),
                                  np.stack([np.linspace(u0, u1, n), np.full(n, v1)], 1)])
        P = lambda uv: np.c_[uv, np.ones(len(uv))] @ coef                  # (along, up) in the carton frame
        bottom = float((P(bottom_uv)[:, 1] + g["H"]/2).mean())
        clearance = float(g["L"]/2 - np.abs(P(edge_uv)[:, 0]).max())
        d = coef[0]                                                      # the label's MD (bars) direction on the face
        tilt = float(np.degrees(np.arccos(min(1., abs(d[1])/max(np.linalg.norm(d), 1e-12)))))
        span = P(np.array([[0., 0.], [0., lab.width], [lab.length, 0.], [lab.length, lab.width]]))[:, 1] + g["H"]/2
        r_h, r_e, r_t = gs1_registration_factors(bottom, clearance, tilt, float(span.min()), float(span.max()), g["H"])
        return dict(factor=r_h*r_e*r_t, barcode_height=r_h, edge_clearance=r_e, bars_vertical=r_t,
                    face="front(-y)" if self.gs1_face(q, on_target) < 0 else "back(+y)",
                    predicted_barcode_bottom_above_base_m=bottom, predicted_barcode_edge_clearance_m=clearance,
                    fitted_bar_tilt_deg=tilt, predicted_label_span_above_base_m=(float(span.min()), float(span.max())))

    def _cover_registration(self, placed, on_target=None):
        """cover-old-label: the share of the old label (the old_label_fully_covered check's 17 x 13 points) that the
        label lying on the carton top covers, extrapolated over the label (_fit, _fitted_cover): a label laid from one
        end is judged by where it is going; one lying clear of the old label scores 0, as it does not do the job."""
        o = self.world.scene.task_geometry["old_label"]
        q = self._item_local("carton")
        coef = self._fit(q[placed][:, :2], placed)
        if coef is None:
            return dict(factor=1., note="fewer than 3 label nodes (or collinear ones) on the carton top: nothing to register")
        uu, vv = np.meshgrid(np.linspace(*o["u"], 17), np.linspace(*o["v"], 13))
        c = self._fitted_cover(coef, np.stack([uu.ravel(), vv.ravel()], 1))
        return dict(factor=c, old_label_covered=c, predicted_old_label_covered_fraction=c)

    def _canister_registration(self, placed, on_target=None):
        """part-id-hdpe: the label part on the side wall, unrolled round the canister (_unrolled_along_label) and fitted
        over the label, gives its mean height, its margins to the wall's top and bottom and its long side's tilt from
        level (as _place_part_id_hdpe measures them); canister_registration_factors turns them into factors."""
        it = self.world.scene.items["part"]
        lab = self.world.label
        q = self._item_local("part")
        if placed.sum() < 3:
            return dict(factor=1., note="fewer than 3 label nodes on the side wall: nothing to register")
        sz = self._unrolled_along_label(q, placed, it["radius"] + .5*lab.stock.thickness)
        coef = self._fit(sz, placed)
        if coef is None:
            return dict(factor=1., note="label nodes on the side wall are collinear: no registration fit")
        z = np.c_[lab.X, np.ones(lab.n)] @ coef[:, 1]
        top, bottom = float(it["height"]/2 - z.max()), float(z.min() + it["height"]/2)
        d = coef[1]                                                      # the label's CD (long side) round the wall
        tilt = float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]))))
        r_mid, r_margin, r_level = canister_registration_factors(float(z.mean()), top, bottom, tilt)
        return dict(factor=r_mid*r_margin*r_level, mid_height=r_mid, end_margin=r_margin, level=r_level,
                    predicted_mean_height_m=float(z.mean()), predicted_margins_m=(top, bottom), fitted_tilt_deg=tilt)

    def _plate_registration(self, placed, on_target=None):
        """part-id-match: 1 - the share of the target plate's marking (the marking_not_covered check's 13 x 5 points)
        that the label lying on its face covers, extrapolated over the label (_fit, _fitted_cover)."""
        it = self.world.scene.items[self.target]
        q = self._item_local(self.target)
        coef = self._fit(q[placed][:, :2], placed)
        if coef is None:
            return dict(factor=1., note="fewer than 3 label nodes (or collinear ones) on the plate face: nothing to register")
        mk = it["marking"]
        uu, vv = np.meshgrid(np.linspace(*mk["u"], 13), np.linspace(*mk["v"], 5))
        c = self._fitted_cover(coef, np.stack([uu.ravel(), vv.ravel()], 1))
        return dict(factor=1. - c, marking_clear=1. - c, predicted_marking_covered_fraction=c)

    def _tube_registration(self, placed):
        """tube-wrap and tube-label: how well the label lying on the tube is registered, from the label part on the tube
        (`placed` nodes) extrapolated over the whole label through an affine fit of their unrolled tube position on
        their material coordinates (a partly laid label is judged by where it is going). Each factor is 1 inside
        the task's own acceptance band and falls linearly to 0 where the placement stops being a labelled tube:
        cap-end alignment (band 0.5 mm capward to 2 mm below the manufacturer label's cap-side edge; 0 with the
        label's cap-side edge on the cap, TUBE_CAP_SKIRT, or its far end at the 9 mm bottom clearance), skew (1 up to
        5 deg, 0 at 45 deg, a diagonal label), viewing window. tube-wrap: window / 5 mm, from the placed triangles
        only. tube-label: the window the whole label will leave where the fit says it is going
        (_predicted_tube_window), so the order in which its long edges went down does not enter; 1 while at most TUBE_STRIP_TOL of the bare strip is covered (its over_mfr_label check),
        falling linearly to 0 with the whole strip covered. The patient name's end (tube-label's
        patient_name_end_toward_cap): the skew ramp reads the signed angle of the label's name axis from the tube's +z,
        0-180 deg, so a label heading for the reversed placement gets 0 at every stage and a label turned past 90 deg in
        its plane does not climb back toward full credit; no new constant."""
        g = self.world.scene.task_geometry["tube"]
        lab = self.world.label
        q = self._item_local("tube")
        if placed.sum() < 3:
            return dict(factor=1., note="fewer than 3 label nodes on the tube: nothing to register")
        r = g["r"] + .5*lab.stock.thickness
        sz = self._unrolled_along_label(q, placed, r)
        M_ = np.c_[lab.X[placed], np.ones(int(placed.sum()))]
        coef, _, rank, _ = np.linalg.lstsq(M_, sz, rcond=None)
        if rank < 3:
            return dict(factor=1., note="label nodes on the tube are collinear: no registration fit")
        z_all = np.c_[lab.X, np.ones(lab.n)] @ coef[:, 1]
        top = float(g["L"]/2 - z_all.max())
        dv = coef[1]
        skew = float(np.degrees(np.arctan2(abs(dv[0]), abs(dv[1]))))
        extra = {}
        skew_read = skew
        if self.world.config.task == "tube-label":
            window, bare = self._predicted_tube_window(q, g, placed, coef, r)
            # the skew ramp reads the SIGNED angle between the label's name axis (+v, the end where the printed name
            # begins; see _place_tube_label) and the tube's +z toward the cap, 0-180 deg: the same as tube-wrap's
            # reading for a label heading name end capward, 0 for one heading for the reversed placement
            skew_read = float(np.degrees(np.arctan2(abs(dv[0]), dv[1])))
            extra = dict(window_from="predicted extent of the whole label", bare_strip_m=bare,
                         window_full_credit_from_m=bare - TUBE_STRIP_TOL, name_axis_from_cap_deg=skew_read,
                         name_end_heading="cap" if dv[1] > 0 else "tube bottom")
        else:
            T = self._alive_tris()
            T = T[placed[T].all(1)]
            window, bare = self._tube_window(q, g, tris=T)[0], None
        r_cap, r_skew, r_win, ramp = registration_factors(top, skew_read, window, g, max(lab.length, lab.width),
                                                          bare_strip=bare)
        return dict(factor=r_cap*r_skew*r_win, cap_end=r_cap, skew=r_skew, window=r_win,
                    predicted_top_edge_from_cap_end_m=top, fitted_skew_deg=skew, window_m=window, cap_end_ramp_m=ramp,
                    **extra)

    def _predicted_tube_window(self, q, g, placed, coef, r):
        """tube-label: (window, bare strip) the label would leave lying where the affine fit of its placed nodes'
        unrolled tube position on their material coordinates puts every node (the fit _tube_registration makes for cap
        end and skew), measured by _tube_window over all of its triangles. For a finished label it equals the measured
        window to within about 0.02 mm; for a partly laid one it is where that label is going (read from the placed
        triangles only, a label laid edge-first would be scored by which long edge went down first, and finishing a
        wrap onto the strip would lower its score)."""
        lab = self.world.label
        ang = np.arctan2(q[placed, 1], q[placed, 0])
        gm = math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean()))   # _unrolled_along_label's origin
        pred = np.c_[lab.X, np.ones(lab.n)] @ coef
        a = gm + pred[:, 0]/r
        window, bare, _ = self._tube_window(np.c_[r*np.cos(a), r*np.sin(a), pred[:, 1]], g)
        return float(window), float(bare)

    def _unrolled_along_label(self, q, nodes, r):
        """(arc length, axial z) of the label nodes `nodes` on the tube, the angle unwrapped edge by edge along the label
        mesh from one node of each connected patch, so a label that winds more than once round the tube (any skew
        beyond about 19 deg for the 2 x 1 in label on this 13 mm tube) keeps a continuous arc length. Where the label
        stays within half a turn of its mean angle this equals _tube_unrolled (tube-wrap's checks use that one)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import breadth_first_order, connected_components
        lab = self.world.label
        idx = np.flatnonzero(nodes)
        loc = -np.ones(lab.n, int)
        loc[idx] = np.arange(len(idx))
        T = self._alive_tris()
        E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
        E = E[nodes[E].all(1)]
        G = coo_matrix((np.ones(len(E)), (loc[E[:, 0]], loc[E[:, 1]])), shape=(len(idx), len(idx))).tocsr()
        G = G + G.T
        ang = np.arctan2(q[idx, 1], q[idx, 0])
        wrap = lambda a: (a + np.pi) % (2*np.pi) - np.pi
        gm = math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean()))
        un = np.zeros(len(idx))
        ncomp, comp = connected_components(G, directed=False)
        for k in range(ncomp):
            members = np.flatnonzero(comp == k)
            base = gm + wrap(math.atan2(float(np.sin(ang[members]).mean()), float(np.cos(ang[members]).mean())) - gm)
            root = members[np.argmin(np.abs(wrap(ang[members] - base)))]
            order, pred = breadth_first_order(G, root, directed=False, return_predecessors=True)
            un[root] = base + wrap(ang[root] - base)
            for i in order[1:]:
                un[i] = un[pred[i]] + wrap(ang[i] - ang[pred[i]])
        return np.stack([r*(un - gm), q[idx, 2]], 1)

    def progress(self, st):
        """The progress reward (contracts progress-v1, -v2 and -v3), from the final state only.
        Full application: the seven take-and-apply tasks (the six and tube-label), progress = integrity x registration
        x (TAKE_CREDIT x s + (1 - TAKE_CREDIT) x sum_i A_i c_i / sum_i A_i), c_i the node credit (node_credit, in the
        task's zone), registration the task's own placement factors (registration()), s the take share (take_share).
        Repair rungs and press-down (the label starts on the item): the share of what was left to do that is now done,
        net of damage: progress = clip(sum_i A_i (c_i - c0_i) / sum_i A_i (1 - c0_i), 0, 1) x integrity, c0_i the node's credit in the settled start state (pre-applied area starts at 1 and
        can only lose credit; press-down's laid label starts at its tack; staying idle scores 0).
        Integrity is 0 when the label can no longer be made good and has to be replaced, or the work cannot be
        finished: the label is torn (the world freezes it), its barcode carries a permanent crease beyond the paper's
        yield (the task's own no_crease_in_barcode measure: on the ladder the largest set of the episode,
        World.set_peak; on the seven other tasks the set left now; that a creased symbol is reprinted, as a torn
        label is, is our inference), or the item has left the bench; on tube-label (LOST_LABEL_ZEROES) also when the
        label itself has left the bench held by nothing (_label_lost). Bumps and other recoverable losses are weighed
        against the work left at the start (repair rungs, press-down) or the whole label, never zeroed. Flags and the
        stage report state without adding to the score."""
        w = self.world
        lab = w.label
        task = w.config.task
        A = lab.vertex_area
        c = self.node_credit((st["on_target"], st["rel"], st["gap"], st["facing"]))
        torn, off = bool(lab.torn), self._item_off_bench()
        creased = bool(st.get("crease", 0.) >= BARCODE_CREASE_YIELD)
        robot_cols = [i for i, cc in enumerate(lab.colliders) if cc.role in ("pad", "robot", "tool")]
        bonded_robot = int(np.isin(lab.bond, robot_cols).sum())
        lost = task in LOST_LABEL_ZEROES and self._label_lost(st["on_target"], robot_cols)
        integrity = 0. if (torn or off or creased or lost) else 1.
        flags = dict(label_torn=torn, barcode_creased=creased, item_off_bench=off,
                     **({"label_lost_off_bench": lost} if task in LOST_LABEL_ZEROES else {}),
                     item_not_resting=not st["resting"]["ok"],
                     robot_or_tool_touching=bool(st["hands"] is not None and not st["hands"]["ok"]),
                     label_nodes_bonded_to_robot_or_tool=bonded_robot, bonded_elsewhere=st["other"])
        flags["stage"] = self.stage(st["on_target"], robot_cols, st["gap"], creased)
        if task in REPAIR_TASKS:
            c0 = w.scene.task_geometry.get("start_credit")
            if c0 is None:          # geometry harnesses: lifted areas start at 0, the rest at 1
                from . import ladder
                c0 = 1. - np.any(ladder.lifted_masks(lab, w.scene.task_geometry["laid_label"]), 0)
            c0 = np.asarray(c0, float)
            todo = float((A*(1. - c0)).sum())
            done = float((A*(c - c0)).sum())
            raw = float(np.clip(done/todo, 0., 1.)) if todo > 0 else 1.
            lost = float((A*np.minimum(c - c0, 0.)).sum())
            from . import ladder, layouts
            area_credit = []                      # diagnostic only: the share of each lifted area's work done
            for f, m in zip(w.scene.task_geometry["laid_label"]["regions"],
                            ladder.lifted_masks(lab, w.scene.task_geometry["laid_label"])):
                t_ = float((A[m]*(1. - c0[m])).sum())
                area_credit.append(dict(area=f["name"], work_left_at_start_mm2=t_*1e6,
                                        done=float(np.clip((A[m]*(c[m] - c0[m])).sum()/t_, 0., 1.)) if t_ > 0 else 1.))
            # diagnostic only: where credit was lost, by perimeter region (a node in two regions counts in both) and
            # in the label's middle
            loss = A*np.minimum(c - c0, 0.)
            ms = [layouts.flap_mask(lab.X, f)[0] for f in layouts.edge_regions()]
            lost_by = {f["name"]: float(-loss[m].sum()*1e6) for f, m in zip(layouts.edge_regions(), ms)}
            lost_by["middle"] = float(-loss[~np.any(ms, 0)].sum()*1e6)
            return dict(progress=raw*integrity, work_left_at_start_mm2=todo*1e6, net_work_done_mm2=done*1e6,
                        credit_lost_mm2=-lost*1e6, credit_lost_by_region_mm2=lost_by, integrity=integrity,
                        lifted_areas=area_credit, flags=flags)
        on_target, rel, gap, facing = st["on_target"], st["rel"], st["gap"], st["facing"]
        alive = np.asarray(getattr(lab, "vertex_alive", np.ones(lab.n, bool)), bool)
        if task in START_CREDIT_TASKS:          # press-down: the co-worker's label starts laid in place, tacked
            c0 = w.scene.task_geometry.get("start_credit")
            if c0 is None:
                # worlds built without LabelEnv (constructed-state harnesses): the start as constructed, every node
                # lying in the zone at tack (env.lay_label_on_carton; the settle tacks it)
                from . import materials as M
                pr = lab.colliders[self._target_colliders()[0]].surface.p_rough
                rel_tack = (1. - np.exp(-M.P_TACK/pr))/(1. - np.exp(-(P_STANDARD_APPLICATION + M.P_TACK)/pr))
                c0 = np.full(lab.n, .5 + .5*rel_tack)
            c0 = np.asarray(c0, float)
            todo = float((A*(1. - c0)).sum())
            done = float((A*(c - c0)).sum())
            raw = float(np.clip(done/todo, 0., 1.)) if todo > 0 else 1.
            placed = (on_target | ((gap <= CONTACT_BAND) & facing)) & self.zone(on_target) & alive
            return dict(progress=raw*integrity, work_left_at_start_mm2=todo*1e6, net_work_done_mm2=done*1e6,
                        credit_lost_mm2=-float((A*np.minimum(c - c0, 0.)).sum())*1e6,
                        credited_area_fraction=float((A*c).sum()/A.sum()),
                        placed_area_fraction=float((A*placed).sum()/A.sum()), integrity=integrity, flags=flags)
        zone = self.zone(on_target)
        placed = (on_target | ((gap <= CONTACT_BAND) & facing)) & zone & alive
        reg = self.registration(placed, on_target)
        frac = float((A*c).sum()/A.sum())
        take = self.take_share(on_target, zone)
        return dict(progress=integrity*reg["factor"]*(TAKE_CREDIT*take + (1. - TAKE_CREDIT)*frac),
                    credited_area_fraction=frac, take_share=take, take_credit_weight=TAKE_CREDIT,
                    placed_area_fraction=float((A*placed).sum()/A.sum()), registration=reg, integrity=integrity,
                    flags=flags)

    # ------------------------------------------------------------------ shared physical checks
    def _trapped_air(self):
        """Measure each existing pocket's lifted area without changing the pocket/air physics.
        Gap is sampled at that pocket's vertices and interpolated over incident triangles,
        closing to zero at vertices outside the pocket (bonded boundary or other region).
        Clip against the contact band and the actual barcode rectangle in material space.
        Enclosed area remains the solver's lumped-area diagnostic, not the pass measure."""
        lab = self.world.label
        A = lab.vertex_area
        half = .5*lab.stock.thickness
        largest = in_bar = enclosed = 0.
        rows = []
        for pk in getattr(lab, "pockets", []):
            v = np.asarray(pk["vertices"])
            col = next((c for c in lab.colliders if c.name == pk["collider"]), None)
            if col is None or not len(v):
                continue
            d, _, _ = col.sdf(lab.x[v])
            gap = d - half
            field = np.zeros(lab.n)
            field[v] = np.maximum(gap, 0.)
            tris = self._alive_tris()
            area = lifted_material_area(lab.X, tris, field)
            bar = lifted_material_area(lab.X, tris, field, rect=(self.bar_u, self.bar_v))
            largest, in_bar = max(largest, area), max(in_bar, bar)
            enclosed = max(enclosed, float(A[v].sum()))
            rows.append(dict(enclosed_m2=float(A[v].sum()), standing_off_m2=area, standing_off_in_barcode_m2=bar,
                             max_gap_m=float(gap.max()), overpressure_pa=pk.get("overpressure_pa"), collider=pk["collider"]))
        return dict(largest_m2=largest, in_barcode_m2=in_bar, largest_enclosed_m2=enclosed, pockets=rows,
                    rule=f"clipped piecewise-linear material area of pocket gap > {CONTACT_BAND*1e6:.0f} um")

    def _rest_pose_ok(self, name):
        """Is the item resting on the bench the way the task leaves it?"""
        w = self.world
        it = w.scene.items[name]
        p, R = w.body_pose(name)
        ez = np.array([0., 0., 1.])
        on_bench = bool(abs(p[0]) <= BENCH_HALF[0] and abs(p[1]) <= BENCH_HALF[1])
        if name == "tube":
            # lying in its cradle: axis level and along the cradle, centre between the rails at its resting height
            g = w.scene.task_geometry["tube"]
            cx, cy, cz = g.get("cradle", (p[0], p[1], p[2]))
            axis = R[:, 2]
            level = float(np.degrees(np.arcsin(min(1., abs(axis[2])))))
            along = float(np.degrees(np.arccos(min(1., abs(axis[0])))))
            off = (float(p[0] - cx), float(p[1] - cy), float(p[2] - cz))
            ok = level <= 10. and along <= 10. and abs(off[0]) <= .025 and abs(off[1]) <= .004 and abs(off[2]) <= REST_TOL
            return dict(ok=bool(ok), kind="in_cradle", axis_tilt_deg=level, axis_off_cradle_deg=along, offset_m=off)
        if name == "carton" or it.get("kind") == "steel_plate":
            if name == "carton":
                g = w.scene.task_geometry["carton"]
                size = (g["L"], g["W"], g["H"])
            else:
                size = it["size"]
            half_ext = .5*np.asarray(size, float)
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])*half_ext
            bottom = float((p + corners @ R.T)[:, 2].min())
            cosz = float(R[:, 2] @ ez)
            tilt = float(np.degrees(np.arccos(min(1., cosz if name == "carton" else abs(cosz)))))
        else:                                      # upright cylinder (canister)
            axis = R[:, 2]
            tilt = float(np.degrees(np.arccos(min(1., float(axis @ ez)))))
            rr, hh = it["radius"], it["height"]
            s = math.sqrt(max(0., 1 - (axis @ ez)**2))
            bottom = float(p[2] - .5*hh*abs(axis @ ez) - rr*s)
        ok = on_bench and tilt <= UPRIGHT_DEG and abs(bottom - TABLE) <= REST_TOL
        return dict(ok=bool(ok), kind="lying_flat_on_bench" if it.get("kind") == "steel_plate" else "upright_on_bench",
                    tilt_deg=tilt, base_above_bench_top_m=bottom - TABLE, on_bench_top=on_bench)

    def resting(self, name):
        return self._rest_pose_ok(name)

    def hands_off(self, name):
        """No robot surface or tool touches the label (label-solver contact load or bond: a finger or a squeegee left
        on it would hold a lifting edge down), and no robot surface or tool touches the item (MuJoCo contact with a
        normal force: the instructions say "take your hands and any tool off the label and the <item>"; a card left
        lying on the carton fails). None when the world has no robot (geometry
        harnesses)."""
        w = self.world
        lab = w.label
        robot_cols = [i for i, c in enumerate(lab.colliders) if c.role in ("pad", "robot", "tool")]
        if not robot_cols or not hasattr(w, "solver"):
            return None
        lam = getattr(lab, "_pair_lam", None)
        label_n = float(lam[:, robot_cols].sum()) if lam is not None and lam.size else 0.
        label_bonds = int(np.isin(lab.bond, robot_cols).sum())
        item_n = tool_item_n = 0.
        try:
            m, d = w.solver.mj_model, w.solver.mj_data
            body = w.scene.bodies[name]
            robot = set(w.scene.robot_bodies)
            tools = {c.body for c in lab.colliders if c.role == "tool" and c.body >= 0}
            f6 = np.zeros(6)
            gs = w._geom_shape
            for i in range(d.ncon):
                con = d.contact[i]
                bodies = [int(w.shape_body[int(gs[g])]) if 0 <= g < len(gs) and gs[g] >= 0 else -1 for g in con.geom]
                if body in bodies and any(b in robot or b in tools for b in bodies):
                    w._mujoco.mj_contactForce(m, d, i, f6)
                    if any(b in robot for b in bodies):
                        item_n += abs(float(f6[0]))
                    else:
                        tool_item_n += abs(float(f6[0]))
        except Exception as exc:                  # pragma: no cover - reported, never silently passed
            return dict(ok=False, error=f"{type(exc).__name__}: {exc}")
        ok = label_n <= 1e-3 and label_bonds == 0 and item_n <= 1e-2 and tool_item_n <= 1e-2
        return dict(ok=bool(ok), robot_or_tool_on_label_n=label_n, label_nodes_bonded_to_robot_or_tool=label_bonds,
                    robot_on_item_n=item_n, tool_on_item_n=tool_item_n)

    # ------------------------------------------------------------------ placement rules
    def _item_local(self, name):
        p, R = self.world.body_pose(name)
        return (self.world.label.x - p) @ R

    def _local(self, name, x):
        p, R = self.world.body_pose(name)
        return (x - p) @ R

    def _place_carrier_top(self, on_target, old=None):
        g = self.world.scene.task_geometry["carton"]
        q = self._item_local("carton")
        top = q[:, 2] > g["H"]/2 - 2e-3
        on_top = bool(top.all())
        over_tape = bool((np.abs(q[:, 1]) < g["tape_half"]).any())
        one_half = bool((q[:, 1] > 0).all() or (q[:, 1] < 0).all())
        margin = float(min(g["L"]/2 - np.abs(q[:, 0]).max(), g["W"]/2 - np.abs(q[:, 1]).max()))
        return dict(checks={"on_top_face": on_top, "not_over_tape_seam": not over_tape, "one_flap_half": one_half,
                            "clear_of_top_edge_bevel": margin >= CARTON_TOP_EDGE_CLEARANCE - 1e-12},
                    metrics=dict(edge_margin_m=margin))

    def _place_press_down(self, on_target):
        """The laid label must end where the carrier rules allow, on the flap half it was laid on ("entirely on that
        flap half")."""
        base = self._place_carrier_top(on_target)
        side = np.sign(self.world.scene.task_geometry["laid_label"]["centre"][1])
        q = self._item_local("carton")
        checks = {k: v for k, v in base["checks"].items() if k != "one_flap_half"}
        checks["on_the_flap_half_it_was_laid_on"] = bool((side*q[:, 1] > 0).all())
        return dict(checks=checks, metrics=base["metrics"])

    def _place_repair_rung(self, on_target):
        """smooth-corner, finish-securing: carrier-top's rules (on the top, off the seam tape, on one flap half, clear
        of the edge bevel). Either flap half is accepted: these instructions ask to smooth down or finish securing
        the label, not to keep it where it lies, which only press-down says."""
        return self._place_carrier_top(on_target)

    def _place_cover_old_label(self, on_target):
        g = self.world.scene.task_geometry
        base = self._place_carrier_top(on_target)
        q = self._item_local("carton")
        (u0, u1), (v0, v1) = g["old_label"]["u"], g["old_label"]["v"]
        uu, vv = np.meshgrid(np.linspace(u0, u1, 17), np.linspace(v0, v1, 13))
        pts = np.stack([uu.ravel(), vv.ravel()], 1)
        covered = self._covered(q[:, :2], self._alive_tris(), pts)
        return dict(checks={**base["checks"], "old_label_fully_covered": covered >= 1.},
                    metrics=dict(**base["metrics"], old_label_covered_fraction=covered,
                                 old_label_uv_m=dict(u=list(g["old_label"]["u"]), v=list(g["old_label"]["v"]))))

    def _alive_tris(self):
        lab = self.world.label
        alive = getattr(lab, "alive", None)
        return lab.tris if alive is None else lab.tris[alive]

    @staticmethod
    def _covered(uv, T, pts):
        """Fraction of the points pts (N x 2) inside any triangle T of the 2-D vertex positions uv."""
        if not len(T):
            return 0.
        a, b, c = uv[T[:, 0]], uv[T[:, 1]], uv[T[:, 2]]
        lo, hi = np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)
        hit = np.zeros(len(pts), bool)
        for k, p in enumerate(pts):
            near = np.flatnonzero((lo[:, 0] <= p[0] + 1e-9) & (hi[:, 0] >= p[0] - 1e-9) &
                                  (lo[:, 1] <= p[1] + 1e-9) & (hi[:, 1] >= p[1] - 1e-9))
            if not len(near):
                continue
            A_, B_, C_ = a[near], b[near], c[near]
            d1 = (B_[:, 0]-A_[:, 0])*(p[1]-A_[:, 1]) - (B_[:, 1]-A_[:, 1])*(p[0]-A_[:, 0])
            d2 = (C_[:, 0]-B_[:, 0])*(p[1]-B_[:, 1]) - (C_[:, 1]-B_[:, 1])*(p[0]-B_[:, 0])
            d3 = (A_[:, 0]-C_[:, 0])*(p[1]-C_[:, 1]) - (A_[:, 1]-C_[:, 1])*(p[0]-C_[:, 0])
            hit[k] = (((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))).any()
        return float(hit.mean())

    def _place_gs1_case_side(self, on_target):
        g = self.world.scene.task_geometry["carton"]
        lab = self.world.label
        q = self._item_local("carton")
        front = bool((q[:, 1] < -g["W"]/2 + 2e-3).all())
        back = bool((q[:, 1] > g["W"]/2 - 2e-3).all())
        X = lab.X
        # the printed symbol's own edges, interpolated through the mesh (outermost vertices inside the symbol sit up to
        # one mesh cell inside its printed edge)
        (u0, u1), (v0, v1) = self.bar_u, self.bar_v
        n = 25
        bottom_uv = np.stack([np.full(n, u0), np.linspace(v0, v1, n)], 1)
        edge_uv = np.concatenate([bottom_uv, np.stack([np.full(n, u1), np.linspace(v0, v1, n)], 1),
                                  np.stack([np.linspace(u0, u1, n), np.full(n, v0)], 1),
                                  np.stack([np.linspace(u0, u1, n), np.full(n, v1)], 1)])
        pts, _ = material_points(lab, np.concatenate([bottom_uv, edge_uv]))
        ql = self._local("carton", pts)
        base_height = float((ql[:n, 2] + g["H"]/2).mean())
        edge = float(g["L"]/2 - np.abs(ql[n:, 0]).max())
        # bars vertical: label MD axis vs carton vertical
        md = q[X[:, 0] > lab.length - lab.h*.5].mean(0) - q[X[:, 0] < lab.h*.5].mean(0)
        tilt = float(np.degrees(np.arccos(abs(md[2])/max(np.linalg.norm(md), 1e-12))))
        return dict(checks={"on_a_long_side_face": front or back, "barcode_bottom_32+-6mm_above_base": abs(base_height - .032) <= .006,
                            "bars_vertical_within_5deg": tilt <= 5., "barcode_>=19mm_from_vertical_edges": edge >= .019},
                    metrics=dict(face="front(-y)" if front else "back(+y)" if back else None,
                                 barcode_bottom_above_base_m=base_height, md_tilt_from_vertical_deg=tilt,
                                 barcode_edge_clearance_m=edge))

    def _tube_unrolled(self, q, r):
        """Label vertices on the tube surface, unrolled: (arc length around the tube from the label's mean angle,
        axial z) in the tube frame."""
        ang = np.arctan2(q[:, 1], q[:, 0])
        mean = math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean()))
        rel = (ang - mean + np.pi) % (2*np.pi) - np.pi
        return np.stack([r*rel, q[:, 2]], 1), ang

    def _tube_window(self, q, g, tris=None):
        """Width of the manufacturer label's bare strip that the patient label leaves uncovered, at the narrowest
        axial station along the patient label (where both labels overlap axially). The bare strip is the tube wall
        outside +-half_angle of the manufacturer label, between its top and bottom edges. `tris`: only these label
        triangles (default: all alive ones)."""
        lab = self.world.label
        r, L = g["r"], g["L"]
        a_m = g.get("mfr_label_half_angle", 2.1)
        z_top, z_bot = L/2 - g["mfr_label_top_from_cap_end"], L/2 - g.get("mfr_label_bottom_from_cap_end", .075)
        T = self._alive_tris() if tris is None else tris
        ang = np.arctan2(q[:, 1], q[:, 0])
        z = q[:, 2]
        zs_ = z if tris is None else z[np.unique(T)]
        bins = 3600
        width = 2*np.pi/bins
        grid = (np.arange(bins) + .5)*width - np.pi               # bin centres in (-pi, pi]
        bare = np.abs(grid) > a_m
        full = float(bare.sum()*width*r)
        if not len(zs_):
            return full, full, 0
        lo, hi = max(zs_.min(), z_bot), min(zs_.max(), z_top)
        if hi <= lo:
            return full, full, 0
        stations = np.linspace(lo, hi, 9)
        # each triangle unrolled about its first vertex (no seam jump inside a triangle)
        A0 = ang[T[:, 0]]
        th = A0[:, None] + ((ang[T] - A0[:, None] + np.pi) % (2*np.pi) - np.pi)
        zt = z[T]
        worst = full
        for zs in stations:
            cover = np.zeros(bins + 1, int)
            m = (zt.min(1) <= zs) & (zt.max(1) >= zs)
            for t_, zz in zip(th[m], zt[m]):
                ts = []
                for i, j in ((0, 1), (1, 2), (2, 0)):
                    if (zz[i] - zs)*(zz[j] - zs) <= 0 and zz[i] != zz[j]:
                        s = (zs - zz[i])/(zz[j] - zz[i])
                        ts.append(t_[i] + s*(t_[j] - t_[i]))
                    elif zz[i] == zs:
                        ts.append(t_[i])
                if not ts:
                    continue
                a0, a1 = min(ts), max(ts)
                k0, k1 = int(np.floor((a0 + np.pi)/width)), int(np.ceil((a1 + np.pi)/width))
                for kk0, kk1 in self._bin_spans(k0, k1, bins):
                    cover[kk0] += 1; cover[kk1] -= 1
            covered = np.cumsum(cover)[:bins] > 0
            free = bare & ~covered
            worst = min(worst, self._longest_circular_run(free)*width*r)
        return worst, full, len(stations)

    @staticmethod
    def _bin_spans(k0, k1, bins):
        """Bin index spans [a, b) on a circle of `bins` bins covering the unwrapped span [k0, k1)."""
        if k1 <= k0:
            return ()
        if k1 - k0 >= bins:
            return ((0, bins),)
        a, b = k0 % bins, k1 % bins
        if b == 0:
            b = bins
        return ((a, b),) if a < b else ((a, bins), (0, b))

    @staticmethod
    def _longest_circular_run(flags):
        """Length of the longest run of True in a circular boolean array (the bare strip straddles +-pi)."""
        if flags.all():
            return len(flags)
        if not flags.any():
            return 0
        rolled = np.roll(flags, -int(np.argmin(flags)))          # start at a False: no run wraps the end
        d = np.diff(np.r_[0, rolled.astype(np.int8), 0])
        return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())

    def _place_tube_wrap(self, on_target):
        g = self.world.scene.task_geometry["tube"]
        lab = self.world.label
        q = self._item_local("tube")          # tube local z: +z toward the cap end
        cap_end = g["L"]/2
        top_from_cap = float(cap_end - q[:, 2].max())
        want = g["mfr_label_top_from_cap_end"]
        # skew on the tube surface: unrolled (arc, z) positions fitted as an affine map of the material coordinates;
        # the label's CD axis (along the tube when straight) is d(arc, z)/dv
        sz, _ = self._tube_unrolled(q, g["r"] + .5*lab.stock.thickness)
        M_ = np.c_[lab.X, np.ones(lab.n)]
        coef, *_ = np.linalg.lstsq(M_, sz, rcond=None)
        dv = coef[1]
        skew = float(np.degrees(np.arctan2(abs(dv[0]), abs(dv[1]))))
        window, bare, stations = self._tube_window(q, g)
        bottom_clear = float(q[:, 2].min() + g["L"]/2)
        return dict(checks={"cap_end_aligned_with_mfr_label": want - TUBE_CAPWARD_TOL - 1e-12 <= top_from_cap <= want + TUBE_BELOW_TOL + 1e-12,
                            "skew<=5deg": skew <= 5., "viewing_window>=5mm_of_bare_strip_uncovered": window >= 5e-3,
                            "clear_of_tube_bottom>=9mm": bottom_clear >= 9e-3},
                    metrics=dict(top_edge_from_cap_end_m=top_from_cap, skew_deg=skew, window_m=window,
                                 bare_strip_m=bare, window_stations=stations, bottom_clearance_m=bottom_clear))

    def _place_tube_label(self, on_target):
        """tube-label: tube-wrap's placement rules (unchanged, _place_tube_wrap) plus the two requirements its
        instruction states that those rules do not check:
        - over the tube manufacturer's label: at the narrowest station along the label, at most
          TUBE_STRIP_TOL of the manufacturer label's bare strip lies under it (bare strip less the window that
          _tube_window measures). tube-wrap's window >= 5 mm alone would let a label turned 80 deg, a third of it on bare
          PET over 59 % of the strip, pass;
        - the patient's name at the end toward the cap. The name is printed from the label art's
          left edge (artwork.specimen_label), which the renderer maps to the label's v = width end (render.export:
          s = 1 - v/width), so that end is at the cap when the label's CD axis runs toward the cap: dz/dv > 0 in the
          tube frame (+z toward the cap), from an affine fit of the nodes' axial position on their material
          coordinates. symbol_rect maps image x to v the other way; the symbol is centred across
          the label, so its rectangle is the same either way."""
        base = self._place_tube_wrap(on_target)
        lab = self.world.label
        q = self._item_local("tube")
        m = base["metrics"]
        intrusion = float(m["bare_strip_m"] - m["window_m"])
        dz_dv = float(np.linalg.lstsq(np.c_[lab.X, np.ones(lab.n)], q[:, 2], rcond=None)[0][1])
        checks = dict(base["checks"])
        checks["over_mfr_label(<=2mm_onto_viewing_strip)"] = intrusion <= TUBE_STRIP_TOL + 1e-12
        checks["patient_name_end_toward_cap"] = dz_dv > 0.
        return dict(checks=checks, metrics=dict(m, viewing_strip_covered_m=intrusion,
                                                strip_tolerance_m=TUBE_STRIP_TOL, name_end_dz_dv=dz_dv,
                                                name_end="cap" if dz_dv > 0. else "tube bottom"))

    def _place_part_id_hdpe(self, on_target):
        g = self.world.scene.items["part"]
        lab = self.world.label
        q = self._item_local("part")
        side = bool((np.hypot(q[:, 0], q[:, 1]) > g["radius"] - 1e-3).all())
        z = q[:, 2]
        centred = bool(abs(z.mean()) < .03 and z.max() < g["height"]/2 - .02 and z.min() > -g["height"]/2 + .02)
        cd = q[lab.X[:, 1] > lab.width - lab.h*.5].mean(0) - q[lab.X[:, 1] < lab.h*.5].mean(0)
        tilt = float(np.degrees(np.arcsin(min(1., abs(cd[2])/max(np.linalg.norm(cd), 1e-12)))))
        return dict(checks={"on_side": side, "middle_within_30mm_of_mid_height": centred, "horizontal_within_5deg": tilt <= 5.},
                    metrics=dict(mean_height_m=float(z.mean()), tilt_deg=tilt))

    def _place_part_id_match(self, on_target):
        name = self.target
        it = self.world.scene.items[name]
        sx, sy, sz = it["size"]
        q = self._item_local(name)
        on_face = bool((q[:, 2] > sz/2 - 1e-3).all() and (np.abs(q[:, 0]) <= sx/2).all() and (np.abs(q[:, 1]) <= sy/2).all())
        mk = it["marking"]
        uu, vv = np.meshgrid(np.linspace(*mk["u"], 13), np.linspace(*mk["v"], 5))
        covered = self._covered(q[:, :2], self._alive_tris(), np.stack([uu.ravel(), vv.ravel()], 1))
        lab = self.world.label
        bonded = {}
        if hasattr(lab, "colliders"):                    # (absent in the geometry harness)
            cols = [c.name for c in lab.colliders]
            for b in lab.bond[lab.bond >= 0]:
                bonded[cols[b]] = bonded.get(cols[b], 0) + 1
        return dict(checks={"on_target_plate_upper_face": on_face, "marking_not_covered": covered == 0.},
                    metrics=dict(target_plate=name, target_part_number=it["part_number"], marking_covered_fraction=covered,
                                 nodes_bonded_per_surface=bonded))
