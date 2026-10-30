"""Tasks and the verifier.

The verifier reads physical state only (owner side): tissue fragment connectivity (whole tets; node splitting),
each tet's REST reference label (where every gram came from: anatomy reference(), never the robot's path), thin-bone
pieces (active links) and what they are bonded to, rigid bones and their joints, the knife's pose in the grasp,
bodies' positions against the tray / bowl footprints, and histories recorded every tick by env.py (blade proximity
to the holding hand, per-tet worst compression, recent motion). Thresholds are authored and anchored to published
practice where one exists; each check returns value, unit, threshold and pass.

Returns gar.evaluation.v1: boolean success, finite score (fraction of the task's criteria met), checks, metrics.
Solver faults are reported as faults, not task failures (unified contract, invariant 7).
"""

import math

import numpy as np

from . import layout as LY
from . import params as PR

INSTRUCTIONS = {
    "A0": "Take the fillet knife from the block and hold it edge-down above the board.",
    "A1": "Remove the head with one cut behind the gill cover and pectoral fin, through a joint of the backbone.",
    "A2": "Fillet the upper side in one piece, skin on, leaving the ribs on the frame, and lay it skin-down in the "
          "perforated fillet tray.",
    "A3": "Fillet both sides with no rib bones left in the fillets, and put the fillets skin-down in the perforated "
          "fillet tray and the frame and head in the empty pan.",
    "A4": "Remove the rib bones from the belly by sliding the blade under them, taking as little flesh as possible.",
    "B1": "Remove all pin bones with the tweezers and drop them in the dish.",
    "B2": "Trim the belly fat strip and the belly membrane off, following the fat line, leaving a smooth edge.",
    "B3": "Cut away the dark spot, taking as little good flesh as possible.",
    "B4": "Remove the skin in one piece.",
    "B5": "Cut the loin into 150 g portions with straight cuts and weigh them on the scale.",
    "C1": "Remove the chain (side muscle) along its seam.",
    "C2": "Remove the silver skin, taking as little meat as possible.",
    "C3": "Cut 25 mm medallions from the thick part.",
}
PRESET_OF = {t: t[0] for t in INSTRUCTIONS}
TIME_LIMIT_S = {"A0": 30., "A1": 90., "A2": 180., "A3": 360., "A4": 120., "B1": 300., "B2": 180., "B3": 120.,
                "B4": 180., "B5": 180., "C1": 180., "C2": 240., "C3": 180.}
WORKPIECE_OF = {"A": "trout", "B": "salmon", "C": "tenderloin"}
WORKPIECE_OF_TASK = {"A4": "trout_fillet_rib_strip"}      # tasks whose start state is a derived workpiece
REF = dict(fillet_upper=0, fillet_lower=1, frame=2, head=3, belly_edge=4)
# tasks scored by a continuous progress measure (1.0 only on success) instead of the fraction of checks passed
PROGRESS_SCORED = ("A3",)
# product receptacles per task (layout names): a piece descending into one is not thrown tissue (env.py)
RECEPTACLES_OF_TASK = {"A3": ("FILLET_TRAY", "FRAME_PAN")}
# tasks whose episode ends at TIME_LIMIT_S: env.act runs no tick past it (an action crossing it is cut at the limit)
STOP_AT_TIME_LIMIT = ("A3",)


NOTHING_MADE = "nothing made yet: a quality check of a product passes only when the product exists"


def check(name, value, threshold, unit, passed, note=""):
    v = float(value) if value is not None and np.isfinite(value) else None
    return name, dict(value=v, threshold=threshold, unit=unit, passed=bool(passed), note=note)


# fillet-whole-fish (A3) result-path helpers, shared by the evaluator and env.py
TOUCH_M = 2e-3                       # m: a node this close to a robot hand box is held by it
# m (one mesh length): compression this close to the knife's edge is the cut, not a bruise. E2 coupon press and slice
# on the 3.5 mm mesh: no tet below J = 0.6 in the record kept beyond 2.5 mm
EDGE_REACH_M = 3.5e-3


def near_boxes(world, box_ids, pts, margin=TOUCH_M):
    """Per point: within `margin` of one of the tissue solver's posed boxes `box_ids` (robot fingers, pads, inserts,
    hand housings)."""
    from .workpiece import _R
    ts = world.tissue
    hit = np.zeros(len(pts), bool)
    if not len(box_ids) or not len(pts):
        return hit
    p0, q0 = ts.s_p0.numpy(), ts.s_q0.numpy()
    for s in box_ids:
        d = np.abs((pts - p0[s]) @ _R(q0[s])) - np.asarray(world.tissue_shapes[s]["half"])
        hit |= np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(1), 0) < margin
    return hit


def edge_distance(world, pts):
    """Per point: distance (m) to the knife's edge, the line along the bottom of each blade segment (edge-centre
    frames, workpiece.blade_segment_poses; x along the edge)."""
    from .workpiece import blade_segment_poses
    d = np.full(len(pts), np.inf)
    for k, (R, p) in enumerate(blade_segment_poses(world)):
        hx = world.tissue.shapes[world.blade_shape0 + k]["half"][0]
        s = np.clip((pts - p) @ R[:, 0], -hx, hx)
        d = np.minimum(d, np.linalg.norm(pts - (p + s[:, None]*R[:, 0]), axis=1))
    return d


class Evaluator:
    def __init__(self, env):
        self.env = env

    # ------------------------------------------------------------------ state digests
    def fragments(self):
        """Per active tet: fragment label; per fragment: mass (kg) and mass by reference label. Tissue is connected
        through un-cut tet faces AND through intact interface bonds (a bonded interface is intact tissue; only a bond
        cut by the blade or torn to full damage separates)."""
        w = self.env.world
        ts, cut = w.tissue, w.cut
        lab = cut._fragments()
        nt = len(cut.orig)
        if getattr(ts, "n_bonds", 0):
            dmg = ts.bonds["damage"].numpy()
            a = ts.bonds["a"].numpy()[dmg < 1.]
            b = ts.bonds["b"].numpy()[dmg < 1.]
            if len(a):
                act_ = np.nonzero(ts.hactive[:nt])[0]
                node_tet = np.full(ts.n, -1)
                node_tet[ts.htets[act_].ravel()] = np.repeat(act_, 4)
                ta, tb = node_tet[a], node_tet[b]
                ok = (ta >= 0) & (tb >= 0)
                from scipy.sparse import coo_matrix
                from scipy.sparse.csgraph import connected_components
                L = lab.copy()
                n_lab = int(L.max()) + 1
                g = coo_matrix((np.ones(ok.sum()), (L[ta[ok]], L[tb[ok]])), shape=(n_lab, n_lab))
                _, merge = connected_components(g, directed=False)
                lab = np.where(L >= 0, merge[np.maximum(L, 0)], -1)
        act = np.nonzero(ts.hactive[:nt])[0]
        m = ts.htmass[:nt]
        ref = w.assembly["ref_tet"] if w.assembly is not None else np.zeros(nt, np.int32)
        frags = {}
        for c in np.unique(lab[act]):
            sel = act[lab[act] == c]
            by = np.bincount(ref[sel], weights=m[sel], minlength=8)
            frags[int(c)] = dict(tets=sel, mass=float(m[sel].sum()), by_ref=by)
        return lab, frags

    def fragment_nodes(self, frag):
        ts = self.env.world.tissue
        return np.unique(ts.htets[frag["tets"]])

    def bone_attachment(self, lab):
        """Which fragment each anatomical rigid body is bonded to (intact flesh-bone ties): body index -> fragment
        -> number of intact ties."""
        w = self.env.world
        ts = w.tissue
        if not ts.n_bties:
            return {}
        node = ts.bties["node"].numpy()
        body = ts.bties["body"].numpy()
        dmg = ts.bties["damage"].numpy()
        node_frag = self._node_fragment(lab)
        out = {}
        for n_, b_, d_ in zip(node, body, dmg):
            if d_ >= 1.:
                continue
            f_ = node_frag.get(int(n_))
            if f_ is None:
                continue
            out.setdefault(int(b_), {}).setdefault(f_, 0)
            out[int(b_)][f_] += 1
        return out

    def _node_fragment(self, lab):
        w = self.env.world
        ts = w.tissue
        nt = len(w.cut.orig)
        act = np.nonzero(ts.hactive[:nt])[0]
        rows = ts.htets[act]
        nf = {}
        for t, row in zip(act, rows):
            for n_ in row:
                nf[int(n_)] = int(lab[t])
        return nf

    def rod_pieces(self, lab):
        """Thin-bone pieces: (rod name, kind, length m, fragment it is bonded to (majority of intact shaft bonds) or
        None if loose)."""
        w = self.env.world
        ts = w.tissue
        if not getattr(ts, "n_rods", 0):
            return []
        r = ts.rods
        lon = r["lon"].numpy()
        tt = r["tie_tet"].numpy()
        dmg = r["tie_dmg"].numpy()
        x = ts.x.numpy()
        out = []
        for rd in w.assembly["rods"]:
            nodes = rd["nodes"]
            piece = [nodes[0]]
            pieces = []
            for a, b in zip(nodes[:-1], nodes[1:]):
                if lon[a]:
                    piece.append(b)
                else:
                    pieces.append(piece); piece = [b]
            pieces.append(piece)
            for pc in pieces:
                L = float(np.linalg.norm(np.diff(x[pc], axis=0), axis=1).sum()) if len(pc) > 1 else 0.
                votes = {}
                for n_ in pc:
                    if tt[n_] >= 0 and dmg[n_] < 1. and ts.hactive[tt[n_]]:
                        f_ = int(lab[tt[n_]])
                        votes[f_] = votes.get(f_, 0) + 1
                frag = max(votes, key=votes.get) if votes else None
                out.append(dict(name=rd["name"], kind=rd["kind"], length=L, fragment=frag, nodes=pc))
        return out

    # ------------------------------------------------------------------ helpers for placement and products
    def frame_fragment(self, lab, frags, attach):
        """The frame: the fragment bonded to the most backbone segments."""
        w = self.env.world
        nk = w.assembly["n_knife_bodies"]
        names = [rb["name"] for rb in w.workpiece.rigid_bodies]
        votes = {}
        for j, nm in enumerate(names):
            if nm == "head":
                continue
            for f_, n_ in attach.get(nk + j, {}).items():
                votes[f_] = votes.get(f_, 0) + n_
        return max(votes, key=votes.get) if votes else None

    def head_fragment(self, attach):
        w = self.env.world
        nk = w.assembly["n_knife_bodies"]
        a = attach.get(nk + 0, {})
        return max(a, key=a.get) if a else None

    def in_footprint(self, pts, spec, margin=0.):
        cx, cy = spec["center"]
        sx, sy = spec["size"][:2]
        return (np.abs(pts[:, 0] - cx) <= sx/2 - margin) & (np.abs(pts[:, 1] - cy) <= sy/2 - margin)

    def in_bowl(self, pts):
        cx, cy = LY.BOWL["center"]
        return np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) <= LY.BOWL["diameter"]/2

    def skin_down(self, frag):
        """Mean outward skin normal of the fragment's skin triangles (current), against world down."""
        w = self.env.world
        ts = w.tissue
        if not ts.n_tris:
            return None
        tset = set(frag["tets"].tolist())
        keep = np.array([t in tset for t in ts.mem_tet])
        if not keep.any():
            return None
        idx = ts._membrane_nodes()[keep]
        x = ts.x.numpy()
        n = np.cross(x[idx[:, 1]] - x[idx[:, 0]], x[idx[:, 2]] - x[idx[:, 0]])
        # orient each normal outward: away from its tet's centroid
        tc = x[ts.htets[ts.mem_tet[keep]]].mean(1)
        fc = x[idx].mean(1)
        s = np.sign(np.einsum("ij,ij->i", n, fc - tc))
        nm = (n*s[:, None]).sum(0)
        nm /= max(np.linalg.norm(nm), 1e-12)
        return math.degrees(math.acos(np.clip(-nm[2], -1, 1)))

    def settled(self, nodes, window_s=1.):
        h = self.env.motion_history
        if len(h) < 2:
            return None
        t_now, x_now = h[-1]
        old = [(t, x) for t, x in h if t_now - t >= window_s - 1e-6]
        if not old:
            return None
        t0, x0 = old[-1]
        return float(np.linalg.norm(x_now[nodes] - x0[nodes], axis=1).max())

    # ------------------------------------------------------------------ evaluation
    def evaluate(self):
        env = self.env
        w = env.world
        task = w.config.task
        if w.fault:
            return dict(schema="gar.evaluation.v1", success=False, score=0., checks={}, metrics={}, fault=w.fault)
        fn = getattr(self, f"_eval_{task}", None)
        checks, metrics = ({}, {}) if fn is None else fn()
        metrics["crack_faces"] = self.crack_faces()
        passed = [c["passed"] for c in checks.values()]
        success = bool(passed) and all(passed)
        if task in PROGRESS_SCORED:
            # 1.0 for every completion that passes; otherwise the progress toward the stated outcome, kept below it
            score = 1. if success else self.PROGRESS_CAP*metrics["progress"]["value"]
        else:
            score = float(np.mean(passed)) if passed else 0.
        return dict(schema="gar.evaluation.v1", task=task, success=success, score=score,
                    checks=checks, metrics=metrics, time_s=env.sim_time,
                    note=None if fn else "no evaluator implemented for this task")

    def crack_faces(self):
        """Crack faces by origin: interfaces split at build, torn (tearing.py), the rest made by the knife (an upper
        bound: a knife batch that later splits a torn region's nodes is not told apart)."""
        w = self.env.world
        allf = set(w.cut.cut_faces)
        pre = w.assembly.get("interface_faces", set()) if w.assembly is not None else set()
        iface = allf & pre
        torn = (set(getattr(w.cut, "torn", {})) & allf) - iface
        return dict(total=len(allf), interface_at_build=len(iface), torn=len(torn),
                    knife_upper_bound=len(allf) - len(iface) - len(torn))

    def _common_knife(self, checks):
        env = self.env
        off, ang = env.world.knife_in_grip()
        dropped = float(np.linalg.norm(off)) > .015 or ang > 30.
        note = ("ASSISTED run: the knife is on the phantom rig, not in the robot's hand (not a policy result)"
                if env.world.config.phantom_rig else "")
        checks.update([check("knife_not_dropped", float(np.linalg.norm(off))*1e3, 15., "mm (handle offset in grasp)",
                             not dropped, note)])
        n = env.blade_left_hand_contacts
        checks.update([check("no_blade_contact_with_holding_hand", n, 0, "ticks with contact", n == 0)])

    # ------------------------------------------------------------------ debris and products left on the knife
    DEBRIS_G = 1.0                   # g: a loose piece under this is debris (a knife makes two faces, not crumbs)
    DEBRIS_BUDGET = .005             # of the workpiece's mass, and DEBRIS_COUNT pieces at most (authored)
    DEBRIS_COUNT = 10

    def _blade_touch(self, pts, margin=2e-3):
        """Per point: within `margin` of a blade segment's slab (resting on or hanging from the knife)."""
        from .workpiece import blade_segment_poses
        w = self.env.world
        hit = np.zeros(len(pts), bool)
        for k, (R, p) in enumerate(blade_segment_poses(w)):
            hx, hy, hz = w.tissue.shapes[w.blade_shape0 + k]["half"]
            loc = (pts - p) @ R
            hit |= (np.abs(loc[:, 0]) <= hx + margin) & (np.abs(loc[:, 1]) <= hy + margin) & \
                (loc[:, 2] >= -margin) & (loc[:, 2] <= 2*hz + margin)
        return hit

    def _resting(self, frag):
        """A piece rests on a support (the board, the bench, the fillet tray, the bowl or the scale) and does not
        touch the knife: its lowest node within 3 mm of the board top (over the board) or of the bench top, or its
        nodes over the tray, bowl or scale; and no node within 2 mm of the blade."""
        x = self.env.world.tissue.x.numpy()
        p = x[self.fragment_nodes(frag)]
        low = float(p[:, 2].min())
        c = p.mean(0)[None]
        over_board = bool(self.in_footprint(c, LY.BOARD)[0])
        on = (over_board and abs(low - LY.BOARD_TOP_Z) < 3e-3) or (not over_board and abs(low - LY.TABLE["top_z"]) < 3e-3)
        on |= bool(self.in_footprint(c, LY.FILLET_TRAY)[0]) or bool(self.in_bowl(c)[0]) or \
            bool(self.in_footprint(c, LY.SCALE)[0])
        return on and not self._blade_touch(p).any()

    THROWN_MPS = 1.                  # m/s: no tissue piece faster than this in a demonstration (gate 5's abuse limit)

    def _common_thrown(self, checks):
        """No tissue thrown: the fastest piece over the episode (a fragment's centroid, sampled every 10 ms) under
        THROWN_MPS, and no tissue fallen below the bench top (on the floor). The fastest node is reported but not
        judged: in every cutting run it is a node of an attached sliver ringing on a cut face, not a thrown piece."""
        w = self.env.world
        ts = w.tissue
        sp, at, kind = (tuple(getattr(self.env, "tissue_speed_max", (0., None, None))) + (None,))[:3]
        ps, pat, pn = (tuple(getattr(self.env, "piece_speed_max", (0., None, None))) + (None,))[:3]
        kd = ts.kind.numpy()[:ts.n]
        z = ts.x.numpy()[:ts.n, 2][(kd == 0) | (kd == 2)]
        fallen = int((z < LY.TABLE["top_z"] - .02).sum())
        checks.update([check("no_tissue_thrown", ps, self.THROWN_MPS, f"m/s fastest piece ({pn} nodes, at {pat} s; the "
                             f"fastest single node {sp:.3g} m/s, a {kind} node at {at} s); {fallen} nodes below the bench "
                             f"top", ps <= self.THROWN_MPS and fallen == 0)])
        return dict(tissue_speed_max_mps=sp, tissue_speed_max_at_s=at, tissue_speed_max_kind=kind,
                    piece_speed_max_mps=ps, piece_speed_max_at_s=pat, piece_speed_max_nodes=pn,
                    tissue_nodes_below_bench=fallen)

    def _common_cut(self, checks, frags, products=()):
        """Debris (every loose piece under DEBRIS_G) within the budget, and every product piece resting on a support,
        not carried by the knife."""
        total = sum(f["mass"] for f in frags.values())
        deb = [f["mass"] for f in frags.values() if f["mass"] < self.DEBRIS_G*1e-3]
        budget = self.DEBRIS_BUDGET*total
        checks.update([check("debris", sum(deb)*1e3, budget*1e3, f"g in pieces under {self.DEBRIS_G:g} g "
                             f"(at most {self.DEBRIS_COUNT} pieces)", sum(deb) <= budget and len(deb) <= self.DEBRIS_COUNT,
                             f"{len(deb)} pieces")])
        thrown = self._common_thrown(checks)
        if products:
            pm = [frags[c]["mass"] for c in products]
            ok = [self._resting(frags[c]) for c in products]
            frac = float(sum(m for m, o in zip(pm, ok) if o)/max(sum(pm), 1e-12))
            checks.update([check("products_resting", frac, 1., "fraction of the removed pieces' mass on a support, "
                                 "off the knife", all(ok))])
        return dict(debris_pieces=len(deb), debris_g=sum(deb)*1e3, **thrown)

    def _eval_A2(self, side="upper"):
        env, w = self.env, self.env.world
        gutted = PR.v(PR.TROUT)["gutted_kg"]
        lab, frags = self.fragments()
        attach = self.bone_attachment(lab)
        fr = self.frame_fragment(lab, frags, attach)
        hd = self.head_fragment(attach)
        key = REF["fillet_" + side]
        removed = {c: f for c, f in frags.items() if c not in (fr, hd) and f["mass"] >= 1e-3}
        side_removed = sum(f["by_ref"][key] for f in removed.values())
        fillet_id = max(removed, key=lambda c: removed[c]["by_ref"][key]) if removed else None
        checks, metrics = {}, {}
        if fillet_id is None:
            note = "nothing has been removed from the frame yet"
            checks.update([check("one_piece", 0., .97, "fraction", False, note),
                           check("yield", 0., .33*gutted, "kg", False, note),
                           check("no_rib_pieces_over_2mm", None, 0, "pieces", False, note),
                           check("no_backbone_or_fin_support_fragments", None, 0, "pieces", False, note),
                           check("workmanship_score", None, 85., "points", False, note),
                           check("placed_in_tray", 0., 1., "fraction", False, note),
                           check("skin_down", None, 45., "deg", False, note),
                           check("settled", None, 1e-3, "m", False, note)])
            self._common_knife(checks)
            # debris and thrown tissue are scored whatever else happened
            dm = self._common_cut(checks, frags)
            checks.update([check("time", env.sim_time, TIME_LIMIT_S["A2"], "s", env.sim_time <= TIME_LIMIT_S["A2"])])
            return checks, dict(fragments=len(frags), **dm)
        fil = frags[fillet_id]
        one = fil["by_ref"][key]/max(side_removed, 1e-12)
        checks.update([check("one_piece", one, .97, "fraction of the side's removed muscle in the largest piece",
                             one >= .97)])
        skin_mass = self._skin_mass(fil)
        y = fil["mass"] + skin_mass
        checks.update([check("yield", y, .33*gutted, "kg (fillet incl. skin)", y >= .33*gutted,
                             "0.33 x gutted = half of the published 66 % skilled lower bound")])
        pieces = self.rod_pieces(lab)
        ribs_in = [p for p in pieces if p["fragment"] == fillet_id and p["kind"] == "rib" and p["length"] > 2e-3]
        # median supports only (dorsal/anal pterygiophores, supraneurals): lying in the median plane, they reach a
        # fillet only when the knife went below it. The pelvic girdle sits in the lateral belly wall beside the slit
        # (anatomy/trout.py) and normally comes away with a skin-on fillet's belly flap; A2 does not ask for fin
        # trimming, so it is reported as a metric, not scored.
        fins_in = [p for p in pieces if p["fragment"] == fillet_id and p["kind"] in ("fin_support", "supraneural")
                   and not p["name"].startswith("pelvic_girdle")]
        pelvic_in = [p["name"] for p in pieces if p["fragment"] == fillet_id and p["name"].startswith("pelvic_girdle")]
        bones_in = [b for b, a in attach.items() if fillet_id in a and b >= w.assembly["n_knife_bodies"] + 1
                    and max(a, key=a.get) == fillet_id]
        checks.update([check("no_rib_pieces_over_2mm", len(ribs_in), 0, "pieces", not ribs_in),
                       check("no_backbone_or_fin_support_fragments", len(fins_in) + len(bones_in), 0, "pieces",
                             not fins_in and not bones_in)])
        ws, wm = self.workmanship(fil, fillet_id, lab, key)
        checks.update([check("workmanship_score", ws, 85., "points (NOAA method on state)", ws >= 85.)])
        metrics.update(workmanship=wm)
        nodes = self.fragment_nodes(fil)
        x = w.tissue.x.numpy()
        pts = x[nodes]
        in_tray = float(self.in_footprint(pts, LY.FILLET_TRAY).mean())
        down = self.skin_down(fil)
        st = self.settled(nodes)
        checks.update([check("placed_in_tray", in_tray, 1., "fraction of fillet nodes over the tray", in_tray >= .999),
                       check("skin_down", down if down is not None else 180., 45., "deg from down",
                             down is not None and down <= 45.),
                       check("settled", st if st is not None else 1e3, 1e-3, "m net motion over 1 s",
                             st is not None and st < 1e-3)])
        self._common_knife(checks)
        metrics.update(self._common_cut(checks, frags, (fillet_id,)))
        checks.update([check("time", env.sim_time, TIME_LIMIT_S["A2"], "s", env.sim_time <= TIME_LIMIT_S["A2"])])
        metrics.update(fillet_mass_kg=y, fillet_fragment=fillet_id, frame_fragment=fr, head_fragment=hd,
                       pelvic_girdle_pieces_in_fillet=pelvic_in, pin_bone_pieces_in_fillet=sum(
                           1 for p in pieces if p["fragment"] == fillet_id and p["kind"] == "pin"),
                       fragments=len(frags), side_removed_kg=float(side_removed),
                       frame_meat_kg=float(frags[fr]["by_ref"][REF["fillet_upper"]] + frags[fr]["by_ref"][REF["fillet_lower"]])
                       if fr is not None else None)
        return checks, metrics

    # ------------------------------------------------------------------ A3 fillet-whole-fish
    PROGRESS_CAP = .95               # an attempt that is not a success scores below every success (evaluate)
    TOUCH_M = TOUCH_M                # m: a node this close to a robot hand box is held, to the blade is on the knife
    SUNK_MAX = .25                   # share of a fillet's nodes inside the other fillet's footprint at the same height
    A3_W = dict(upper=.4, lower=.4, frame=.2, cut=.6, place=.4, sever=.6, quality=.4)    # authored progress weights

    def _eval_A3(self):
        """A3 fillet-whole-fish. Success: every check, each once.

        Per side (upper_/lower_: rest-material labels, so they follow the flesh however the fish is turned). The two
        fillets are two DIFFERENT removed pieces, each holding the most of its own side's muscle (_assign_fillets); a
        removed piece holds no rigid bone (a piece holding the head or a vertebra by the majority of its intact ties is
        frame or head, so a fillet that took a vertebra is not a fillet):
          one_piece   >= 97 % of the side's removed muscle in its fillet;
          yield       the fillet's mass with skin, less any of the other side's muscle, >= 0.33 x gutted (half the
                      published 66 % skilled lower bound);
          no_rib_pieces_over_2mm, no_backbone_or_fin_support_fragments (median fin supports, supraneurals);
          workmanship_score >= 85 (workmanship_A3: NOAA points per 25.4 mm square; compression recorded while the
                      knife's edge was within EDGE_REACH_M of a tet is the cut, not a bruise; a knife score is an
                      incision left in the fillet, while meat left on the bone is priced by yield alone);
          in_tray     resting in the perforated fillet tray: every node inside its walls, the lowest below its rim, no
                      node within TOUCH_M of a robot hand box or of the blade, net motion under 1 mm over 1 s (no
                      centring), and not sunk into the other fillet (more than SUNK_MAX of its nodes inside the
                      other's footprint at the same height: the tissue solver has no tissue-tissue contact, so a
                      fillet laid on the other passes through it; flat seed-0 fillets side by side pass with edges
                      overlapping up to about 40 % of a fillet's width); in A3 the tray stands on
                      the holding-arm side (layout.FILLET_TRAY_CENTER_OF_TASK), where both fillets fit side by side;
          skin_down   mean outward skin normal within 45 deg of down.
        frame_and_head_in_pan: every rigid bone (head, vertebrae) inside the empty pan's rim and below it
          (layout.FRAME_PAN), every piece holding one resting as above (head on the frame or cut off: both pass).
        knife_not_dropped: held by either hand (or in the right hand's grasp), or put down: blade and handle over the
          bench above its top, outside the tray, the pan and the ice pan, not on the fish or a product (in the block or
          on the board passes; on the floor, on the fish, in a receptacle fails).
        no_blade_contact_with_holding_hand; debris (pieces under 1 g within A2's allowance for each of the two sides:
          1 % of the tissue's mass, at most 20 pieces); no_tissue_thrown (no free piece - none a hand touches -
          faster than 1 m/s outside the tray and the pan, nothing on the floor); time <= 360 s (env.act runs no tick
          past it: STOP_AT_TIME_LIMIT). The rest tests compare only nodes present in both motion snapshots, so the
          evaluation works right after a cut or tear splits nodes.

        Progress (the score when not a success, evaluate): the delivered fraction of the stated outcome, a function of
        the state alone; no method, order or intermediate step is credited.
          P = 0.4 c_upper + 0.4 c_lower + 0.2 c_frame;   c_s = 0.6 cut_s + 0.4 place_s (0 if its fillet is on the floor)
          cut_s = 0.6 sever_s + 0.4 y_s q_s;   place_s = y_s r_s k_s (1/2 + o_s/2);   c_frame = b
          sever_s = free_s + attached_s (1 - R_s/R0_s): free_s / attached_s are the shares of the side's muscle in its
            fillet / in bone-holding pieces (muscle in other loose pieces is lost, so mincing earns nothing); R_s is the
            fewest intact connections (uncut tet faces, interface bonds, flesh-bone ties; tears count as cut near the
            knife's cut) still joining the side's meat core (CORE_M from frame or head, no ties)
            to any rigid bone, a minimum cut (_remaining_connection), so cuts that free nothing (a trim, a chop
            across) leave it as it was; R0_s the same before any cut. A minimum cut reads the narrowest remaining
            connection, so the same opened area reads differently by where it was opened: at half the separating
            surface opened, four stroke orders read 0.25 to 0.61. Tears are counted only near
            the knife's cut, so peeling a fillet off by pulling earns nothing until it comes free. The
            quality term exists only for a removed fillet, so sever_s steps by about 0.1 when a fillet comes free.
          y_s = min(1, yield / threshold); q_s = min(1, one-piece share / 0.97) x 1/(1 + bone pieces in the fillet) x
            min(1, max(0, workmanship) / 85); r_s = the fraction of the fillet's nodes inside the tray's walls if it rests
            there, else 0; k_s = 1 if no more than SUNK_MAX of it is sunk into the other fillet (the in-tray test), else
            1 - sunk_s; o_s = clip((90 deg - skin angle) / 45 deg, 0, 1).
          b = sum over rigid bones of w_j [inside and below the pan's rim] [its piece released and still] gate_p g_pan,
            with w = 1/2 for the head and 1/2 shared by the vertebrae (the head body is 87 % of the rigid bones' mass); a
            bone with no intact tie belongs to the piece holding a bone it is still jointed to, and a chain free of all
            tissue is released if no hand box touches it.
            gate_p = clip(1 - (fillet meat on that piece - t) / (m - t), 0, 1), m half the fish's fillet meat and
            t = 0.3 m (about the meat a passing pair may leave on the bone): a frame still carrying a fillet earns
            nothing, a head cut off first earns its share. g_pan, the same expression of the fillet meat lying inside
            the pan's rim on any piece, withholds the bones' credit while the fish or a fillet lies in the pan (a whole
            fish dropped in, its head torn off by the impact, earns nothing).
        The weights (A3_W) are authored: two fillets and one disposal; cutting before placing."""
        env, w = self.env, self.env.world
        ts = w.tissue
        W = self.A3_W
        top = LY.TABLE["top_z"]
        Y = .33*PR.v(PR.TROUT)["gutted_kg"]
        lab, frags = self.fragments()
        attach = self.bone_attachment(lab)
        nk = w.assembly["n_knife_bodies"]
        holders = {max(a, key=a.get) for a in attach.values() if a}
        cands = {c: f for c, f in frags.items() if c not in holders and f["mass"] >= 1e-3}
        assign = self._assign_fillets(cands)
        pieces = self.rod_pieces(lab)
        x = ts.x.numpy()
        T = LY.fillet_tray(w.config.task)
        tray = (tuple(T["center"]), (T["size"][0]/2 - T["wall"], T["size"][1]/2 - T["wall"]), top + T["wall"],
                top + T["size"][2])
        blade_tets = self._blade_compressed()
        nb = w.cut._nbr[blade_tets].ravel()
        knife_zone = blade_tets.copy()
        knife_zone[nb[nb >= 0]] = True              # tears within two faces of the knife's cut count as severed
        fnodes = {s: self.fragment_nodes(cands[assign[s]]) for s in ("upper", "lower") if assign[s] is not None}
        sunk = dict(upper=0., lower=0.)
        if len(fnodes) == 2:
            sunk["upper"], sunk["lower"] = self._interpenetration(fnodes["upper"], fnodes["lower"], x)
        checks, metrics, prog = {}, {}, {}
        for s, k, ko in (("upper", REF["fillet_upper"], REF["fillet_lower"]),
                         ("lower", REF["fillet_lower"], REF["fillet_upper"])):
            total = max(sum(f["by_ref"][k] for f in frags.values()), 1e-12)
            removed = sum(f["by_ref"][k] for f in cands.values())
            attached = sum(frags[c]["by_ref"][k] for c in holders)
            R, R0 = self._remaining_connection(s, knife_zone=knife_zone), self._remaining_connection(s, intact=True)
            fid = assign[s]
            f = cands.get(fid)
            p = dict(fillet_fragment=fid, side_muscle_g=total*1e3, removed_g=removed*1e3, attached_g=attached*1e3,
                     remaining_connections=R, remaining_connections_at_start=R0)
            free = quality = place = 0.
            lost = False
            if f is None:
                note = "no piece of this side's muscle has been removed from the bones yet"
                checks.update([check(f"{s}_one_piece", 0., .97, "fraction", False, note),
                               check(f"{s}_yield", 0., Y, "kg", False, note),
                               check(f"{s}_no_rib_pieces_over_2mm", None, 0, "pieces", False, note),
                               check(f"{s}_no_backbone_or_fin_support_fragments", None, 0, "pieces", False, note),
                               check(f"{s}_workmanship_score", None, 85., "points", False, note),
                               check(f"{s}_in_tray", 0., 1., "fraction", False, note),
                               check(f"{s}_skin_down", None, 45., "deg", False, note)])
            else:
                one = f["by_ref"][k]/max(removed, 1e-12)
                ym = f["mass"] - f["by_ref"][ko] + self._skin_mass(f)
                ribs = [q for q in pieces if q["fragment"] == fid and q["kind"] == "rib" and q["length"] > 2e-3]
                fins = [q for q in pieces if q["fragment"] == fid and q["kind"] in ("fin_support", "supraneural")
                        and not q["name"].startswith("pelvic_girdle")]
                ws, wm = self.workmanship_A3(f, lab, k)
                nodes = fnodes[s]
                inside, rest, rm = self._rests_in(nodes, tray, x)
                ok_tray = rest and inside >= .999 and sunk[s] <= self.SUNK_MAX
                down = self.skin_down(f)
                lost = bool((x[nodes, 2] < top - .02).any())
                why = [n for n, b in (("above the rim", rm["lowest_above_floor_mm"] >= 1e3*(tray[3] - tray[2])),
                                      ("touched by a hand", rm["held_by_hand"]), ("on the knife", rm["on_knife"]),
                                      ("moving", not rm["still"]),
                                      ("sunk into the other fillet", sunk[s] > self.SUNK_MAX)) if b]
                checks.update([
                    check(f"{s}_one_piece", one, .97, "fraction of the side's removed muscle in its fillet", one >= .97),
                    check(f"{s}_yield", ym, Y, "kg (fillet incl. skin, less the other side's muscle)", ym >= Y,
                          "0.33 x gutted = half of the published 66 % skilled lower bound"),
                    check(f"{s}_no_rib_pieces_over_2mm", len(ribs), 0, "pieces", not ribs),
                    check(f"{s}_no_backbone_or_fin_support_fragments", len(fins), 0, "pieces", not fins,
                          "a piece holding a vertebra is frame, not a fillet"),
                    check(f"{s}_workmanship_score", ws, 85., "points (NOAA per square on state)", ws >= 85.),
                    check(f"{s}_in_tray", inside if rest else 0., 1., "fraction of fillet nodes inside the tray's "
                          "walls, resting (below the rim, released, off the knife, still, not sunk into the other "
                          "fillet)", ok_tray, "" if ok_tray else "not resting in the tray: " + ", ".join(why)),
                    check(f"{s}_skin_down", down if down is not None else 180., 45., "deg from down",
                          down is not None and down <= 45.)])
                free = f["by_ref"][k]/total
                y = min(1., ym/Y)
                q_ = min(1., one/.97)/(1. + len(ribs) + len(fins))*min(1., max(0., ws)/85.)
                quality = y*q_
                o = 0. if down is None else float(np.clip((90. - down)/45., 0., 1.))
                # placement credit: released and still below the rim (a knife lying on it fails knife_not_dropped and
                # the in-tray check, not the placement), less the share sunk into the other fillet beyond what the
                # in-tray test accepts
                laid = rm["lowest_above_floor_mm"] < 1e3*(tray[3] - tray[2]) and not rm["held_by_hand"] and rm["still"]
                place = y*(inside if laid else 0.)*(1. if sunk[s] <= self.SUNK_MAX else 1. - sunk[s])*(.5 + .5*o)
                p.update(fillet_g=f["mass"]*1e3, yield_g=ym*1e3, y=y, one_piece=one, bone_pieces=len(ribs) + len(fins),
                         workmanship=ws, workmanship_detail=wm, q=q_, tray=rm, sunk_share=sunk[s], skin_deg=down, o=o,
                         on_floor=lost)
            sever = min(1., max(0., free + attached/total*(1. - R/R0))) if R0 > 0 else free
            cut_ = W["sever"]*sever + W["quality"]*quality
            c_s = 0. if lost else W["cut"]*cut_ + W["place"]*place
            p.update(free=free, attached=attached/total, sever=sever, quality=quality, cut=cut_, place=place, credit=c_s)
            prog[s] = p
        # frame and head: every rigid bone inside and below the pan's rim, every piece holding one resting there
        st = ts.body_states()
        bones = st["p"][nk:]
        (pcx, pcy), (phx, phy), _, prim = pan = LY.frame_pan_inner()
        in_pan = (np.abs(bones[:, 0] - pcx) <= phx) & (np.abs(bones[:, 1] - pcy) <= phy) & (bones[:, 2] < prim)
        bone_still = np.linalg.norm(st["v"][nk:], axis=1) < .02
        holder_of = [max(attach[nk + j], key=attach[nk + j].get) if attach.get(nk + j) else None
                     for j in range(len(bones))]
        # a bare bone (no intact tie) goes with the piece holding a bone it is still jointed to (intervertebral joints):
        # a hand holding the frame holds its bare tail vertebrae
        ja, jb, j_on = ts.joints["a"].numpy(), ts.joints["b"].numpy(), ts.joints["on"].numpy()
        bone_joints = [(a_ - nk, b_ - nk) for a_, b_, o_ in zip(ja, jb, j_on) if o_ and a_ >= nk and b_ >= nk]
        for _ in range(len(bones)):
            todo = [(i, k) for i, k in bone_joints + [(k, i) for i, k in bone_joints]
                    if holder_of[i] is None and holder_of[k] is not None]
            if not todo:
                break
            for i, k in todo:
                holder_of[i] = holder_of[k]
        meat = REF["fillet_upper"], REF["fillet_lower"]
        on = {c: f["by_ref"][meat[0]] + f["by_ref"][meat[1]] for c, f in frags.items()}
        m_side = max(.5*sum(on.values()), 1e-9)
        t_ = .3*m_side          # seed 0: ideal yields 115.7 + 107.6 g against 2 x 95.5 g leave about 32 g

        def meat_gate(m):
            return float(np.clip(1. - max(0., m - t_)/(m_side - t_), 0., 1.))

        def in_pan_share(c):
            q = x[self.fragment_nodes(frags[c])]
            return float(((np.abs(q[:, 0] - pcx) <= phx) & (np.abs(q[:, 1] - pcy) <= phy) & (q[:, 2] < prim)).mean())
        # the bones' credit is withheld while fillet meat lies in the pan, on any piece: a whole fish dropped in (its
        # head torn off by the impact) earns nothing
        m_pan = sum(m*in_pan_share(c) for c, m in on.items() if m > 1e-4)
        g_pan = meat_gate(m_pan)
        piece_rest, piece_gate, pans = {}, {}, {}
        for c in holders:
            _, r_, rm_ = self._rests_in(self.fragment_nodes(frags[c]), pan, x)
            piece_rest[c], pans[str(c)] = r_, rm_
            # each bone-holding piece is credited by the fillet meat on it: a frame still carrying a fillet earns
            # nothing, a head cut off first earns its share, lifting one piece leaves another's
            piece_gate[c] = meat_gate(on[c])*g_pan
        # bones free of all tissue (a loose chain) are released if no robot hand box is within TOUCH_M of the capsules
        # of any bone jointed to them
        held_bare = np.zeros(len(bones), bool)
        sb, boxes = getattr(ts, "h_s_body", None), getattr(env, "_robot_boxes", [])
        if sb is not None and len(boxes):
            from .workpiece import _R
            p0, q0 = ts.s_p0.numpy(), ts.s_q0.numpy()
            for j in (j for j, h in enumerate(holder_of) if h is None):
                for s in np.nonzero(sb == nk + j)[0]:
                    hl, r = ts.shapes[s]["half"][:2]
                    axis = p0[s] + np.outer(np.linspace(-1., 1., 5)*hl, _R(q0[s])[:, 0])
                    held_bare[j] |= bool(near_boxes(w, boxes, axis, r + self.TOUCH_M).any())
            for _ in range(len(bones)):
                todo = [i for i, k in bone_joints + [(k, i) for i, k in bone_joints] if held_bare[k] and not held_bare[i]]
                if not todo:
                    break
                held_bare[todo] = True
        rest_j = np.array([piece_rest.get(h, True) for h in holder_of]) & bone_still & ~held_bare
        gate_j = np.array([piece_gate.get(h, g_pan) for h in holder_of])
        w_j = np.full(len(bones), .5/max(len(bones) - 1, 1)); w_j[0] = .5   # head and frame half each (head body: 87 %
        b = float((w_j*in_pan*rest_j*gate_j).sum())                          # of the rigid bones' mass, seed 0)
        ok_frame = bool(in_pan.all() and rest_j.all())
        checks.update([check("frame_and_head_in_pan", float(in_pan.mean()), 1., "fraction of the rigid bones (head, "
                             "vertebrae) inside the empty pan's rim; the pieces holding them resting there", ok_frame)])
        prog["frame"] = dict(bones_in_pan=int(in_pan.sum()), bones=int(len(in_pan)), resting=bool(rest_j.all()),
                             bare_bones_held=int(held_bare.sum()), fillet_meat_in_pan_g=m_pan*1e3, g_pan=g_pan,
                             pieces=pans, gates={str(c): v for c, v in piece_gate.items()}, b=b, credit=b)
        # common: knife, holding hand, debris, thrown tissue, time. The knife is judged against the products (the two
        # fillets and the bone-holding pieces), not trimmings
        prods = [c for c in (assign["upper"], assign["lower"]) if c is not None] + sorted(holders)
        ok, where = self._knife_put_down(prods, frags, x)
        checks.update([check("knife_not_dropped", float(ok), 1., "held, or put down clear of the products and the "
                             "receptacles", ok, where)])
        n = env.blade_left_hand_contacts
        checks.update([check("no_blade_contact_with_holding_hand", n, 0, "ticks with contact", n == 0)])
        total_m = sum(f["mass"] for f in frags.values())
        deb = [f["mass"] for f in frags.values() if f["mass"] < self.DEBRIS_G*1e-3]
        budget, count = 2*self.DEBRIS_BUDGET*total_m, 2*self.DEBRIS_COUNT
        checks.update([check("debris", sum(deb)*1e3, budget*1e3, f"g in pieces under {self.DEBRIS_G:g} g (at most "
                             f"{count} pieces; A2's allowance for each of the two sides)",
                             sum(deb) <= budget and len(deb) <= count, f"{len(deb)} pieces")])
        ps, pat, pn = (tuple(getattr(env, "piece_speed_max_outside", (0., None, None))) + (None,))[:3]
        kd = ts.kind.numpy()[:ts.n]
        fallen = int((x[:ts.n, 2][(kd == 0) | (kd == 2)] < top - .02).sum())
        checks.update([check("no_tissue_thrown", ps, self.THROWN_MPS, f"m/s fastest free piece outside the tray and "
                             f"the pan ({pn} nodes, at {pat} s; a piece a hand touches is carried); {fallen} nodes below "
                             f"the bench top", ps <= self.THROWN_MPS and fallen == 0)])
        checks["time"] = check("time", env.sim_time, TIME_LIMIT_S["A3"], "s", env.sim_time <= TIME_LIMIT_S["A3"])[1]
        P = W["upper"]*prog["upper"]["credit"] + W["lower"]*prog["lower"]["credit"] + W["frame"]*prog["frame"]["credit"]
        metrics.update(progress=dict(value=float(P), weights=dict(W), **prog), fragments=len(frags),
                       debris_pieces=len(deb), debris_g=sum(deb)*1e3, tissue_nodes_below_bench=fallen,
                       piece_speed_max_all_mps=float(getattr(env, "piece_speed_max", (0.,))[0]))
        return checks, metrics

    @staticmethod
    def _assign_fillets(cands):
        """{"upper": fragment or None, "lower": ...}: two DIFFERENT removed pieces chosen to hold the most of their own
        side's muscle together (a piece holding none of a side's muscle is never that side's fillet; a butterflied
        pair is one piece and serves one side)."""
        U, L = REF["fillet_upper"], REF["fillet_lower"]

        def top2(k):
            return sorted((c for c in cands if cands[c]["by_ref"][k] > 0), key=lambda c: -cands[c]["by_ref"][k])[:2]
        best, best_v = (None, None), -1.
        for u in top2(U) + [None]:
            for l_ in top2(L) + [None]:
                if u is not None and u == l_:
                    continue
                v = (cands[u]["by_ref"][U] if u is not None else 0.) + (cands[l_]["by_ref"][L] if l_ is not None else 0.)
                if v > best_v:
                    best, best_v = (u, l_), v
        return dict(upper=best[0], lower=best[1])

    def _robot_touch(self, pts):
        """Per point: within TOUCH_M of a robot finger, pad, insert or hand box (the tissue solver's posed copies)."""
        from .workpiece import _R
        ts = self.env.world.tissue
        hit = np.zeros(len(pts), bool)
        boxes = getattr(self.env, "_robot_boxes", [])
        if not boxes or not len(pts):
            return hit
        p0, q0 = ts.s_p0.numpy(), ts.s_q0.numpy()
        for s in boxes:
            d = np.abs((pts - p0[s]) @ _R(q0[s])) - np.asarray(self.env.world.tissue_shapes[s]["half"])
            hit |= np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(1), 0) < self.TOUCH_M
        return hit

    def _rests_in(self, nodes, rec, x):
        """A piece (its nodes) in a receptacle rec = (centre xy, inner half extents xy at the rim, floor top z, rim z):
        (fraction of its nodes inside the walls in plan, whether it rests there, details). Resting: its lowest node
        below the rim, no node within TOUCH_M of a robot hand box (released) or of the blade (off the knife), net motion
        under 1 mm over 1 s. A held piece is not put down; a released piece at rest below the rim inside the walls is
        supported by the receptacle or by what lies in it."""
        p = x[nodes]
        (cx, cy), (hx, hy), floor, rim = rec
        inside = float(((np.abs(p[:, 0] - cx) <= hx) & (np.abs(p[:, 1] - cy) <= hy)).mean())
        low = float(p[:, 2].min())
        held = bool(self._robot_touch(p).any())
        on_knife = bool(self._blade_touch(p, self.TOUCH_M).any())
        st = self._settled_common(nodes)
        still = st is not None and st < 1e-3
        rest = low < rim and not held and not on_knife and still
        return inside, bool(rest), dict(inside=inside, lowest_above_floor_mm=(low - floor)*1e3, held_by_hand=held,
                                        on_knife=on_knife, still=bool(still),
                                        net_motion_1s_mm=None if st is None else st*1e3)

    def _settled_common(self, nodes, window_s=1.):
        """settled() over the nodes that exist in both motion snapshots: cutting and tearing append split nodes after a
        snapshot is taken, so settled() alone would index past its end. A split node coincides with its twin when
        created."""
        h = self.env.motion_history
        if len(h) < 2:
            return None
        t_now, x_now = h[-1]
        old = [(t, x_) for t, x_ in h if t_now - t >= window_s - 1e-6]
        if not old:
            return None
        x0 = old[-1][1]
        nodes = np.asarray(nodes)
        nodes = nodes[nodes < min(len(x_now), len(x0))]
        return float(np.linalg.norm(x_now[nodes] - x0[nodes], axis=1).max()) if len(nodes) else None

    @staticmethod
    def _interpenetration(na, nb, x, cell=3e-3, tol=2e-3):
        """(share of piece a's nodes, share of piece b's) lying in 3 mm plan cells that the other piece occupies over an
        overlapping height range deeper than `tol`: one fillet sunk into the other. The tissue solver has no
        tissue-to-tissue contact, so a fillet laid on the other passes through it to the tray floor (in one check
        65.6 % of its nodes inside the lower piece). Fillets lying side by side and touching share only an
        edge line of cells."""
        out = []
        cells = []
        for nd in (na, nb):
            pt = x[nd]
            key = np.floor(pt[:, 0]/cell).astype(np.int64)*1000003 + np.floor(pt[:, 1]/cell).astype(np.int64)
            u, inv = np.unique(key, return_inverse=True)
            lo = np.full(len(u), np.inf); hi = np.full(len(u), -np.inf)
            np.minimum.at(lo, inv, pt[:, 2]); np.maximum.at(hi, inv, pt[:, 2])
            cells.append((u, inv, lo, hi))
        (ua, ia, loa, hia), (ub, ib, lob, hib) = cells
        _, ca, cb = np.intersect1d(ua, ub, return_indices=True)
        deep = np.minimum(hia[ca], hib[cb]) - np.maximum(loa[ca], lob[cb]) > tol
        for u, inv, c_ in ((ua, ia, ca), (ub, ib, cb)):
            bad = np.zeros(len(u), bool); bad[c_[deep]] = True
            out.append(float(bad[inv].mean()))
        return tuple(out)

    def prime(self):
        """The progress measure's starting connection counts (R0, intact topology), from the state the environment
        starts in, so they do not depend on when the first evaluation happens."""
        for s in ("upper", "lower"):
            self._remaining_connection(s, intact=True)

    def _knife_put_down(self, pieces, frags, x):
        """(passed, where) for A3's knife_not_dropped: held by either hand (both fingers pressing the handle), or in the
        right hand's grasp (handle within 15 mm and 30 deg), or put down: every blade segment and the handle over the
        bench above its top, none inside the fillet tray, the frame pan or the ice pan, and no node of a piece of 1 g or
        more within TOUCH_M of the blade. Standing in the block and lying on the board pass; on the floor, on the fish
        or a product, or in a receptacle fail."""
        from .world import KNIFE_PINCH_N
        from .workpiece import blade_segment_poses
        w = self.env.world
        held = sorted(a for a, f_ in w.pinch_forces().items() if min(f_) > KNIFE_PINCH_N)
        off, ang = w.knife_in_grip()
        if held or (float(np.linalg.norm(off)) <= .015 and ang <= 30.):
            return True, "held by the " + (" and ".join(held) if held else "right") + " hand"
        pts = np.array([p for _, p in blade_segment_poses(w)] + [w.tissue.body_states()["p"][w.knife_body0]])
        T, top = LY.TABLE, LY.TABLE["top_z"]
        if not ((np.abs(pts[:, 0] - T["center"][0]) <= T["size"][0]/2) &
                (np.abs(pts[:, 1] - T["center"][1]) <= T["size"][1]/2) & (pts[:, 2] > top - .005)).all():
            return False, "not on the bench: dropped"
        for name, spec, c in (("fillet tray", LY.FILLET_TRAY, LY.fillet_tray(w.config.task)["center"]),
                              ("frame pan", LY.FRAME_PAN, LY.FRAME_PAN["center"]),
                              ("ice pan", LY.ICE_PAN, LY.ice_pan_center(w.config.task))):
            if ((np.abs(pts[:, 0] - c[0]) <= spec["size"][0]/2) & (np.abs(pts[:, 1] - c[1]) <= spec["size"][1]/2) &
                    (pts[:, 2] < top + spec["size"][2] + .01)).any():
                return False, f"in the {name}"
        nodes = np.unique(np.concatenate([self.fragment_nodes(frags[c]) for c in pieces])) if pieces else np.zeros(0, int)
        if len(nodes) and self._blade_touch(x[nodes], self.TOUCH_M).any():
            return False, "lying on the fish or a product"
        return True, "put down on the bench, the board or in the block"

    def _blade_compressed(self):
        """Per original tet: where the knife's edge compresses the tissue it cuts (workmanship_A3): within one face of a
        tet that owns a knife-made cut face (neither an interface split at build nor torn) or holds a node of an
        interface bond the blade cut."""
        w = self.env.world
        ts = w.tissue
        nt = len(w.cut.orig)
        near = np.zeros(nt, bool)
        pre = w.assembly.get("interface_faces", set())
        torn = set(getattr(w.cut, "torn", {}))
        kf = [f for f in w.cut.cut_faces if f not in pre and f not in torn]
        if kf:
            near[np.asarray(kf, np.int64).ravel()] = True
        cutb = self.env.bond_cut_by_blade()
        if cutb.any():
            bn = np.unique(np.concatenate([ts.bonds["a"].numpy()[cutb], ts.bonds["b"].numpy()[cutb]]))
            act = np.nonzero(ts.hactive[:nt])[0]
            near[act[np.isin(ts.htets[act], bn).any(1)]] = True
        nb = w.cut._nbr[near].ravel()
        out = near.copy()
        out[nb[nb >= 0]] = True
        return out

    def workmanship_A3(self, frag, lab, key):
        """NOAA workmanship on state as the standard grades it, per 25.4 mm square of the fillet's flesh side (rest
        coordinates projected on the fish's side plane): a square touched by a defect costs 1/2 point for a knife score
        (an incision left in the fillet: a knife-made cut face with both its tets in the fillet), 1/2 for a tear (a torn
        interface bond the blade did not cut) and 2 for a bruise (a tet compressed below J = 0.6 at any time), each at
        most once per square; normalised per 454 g of fillet. The knife's own cutting compression is not a bruise: the
        edge law crushes the tissue it cuts below J = 0.5 (tissue.py), so the bruise reads env.tet_jmin_far, each tet's
        worst J recorded only while the knife's edge was at least EDGE_REACH_M away. In the environment's E2 coupon cuts
        on the trout's 3.5 mm mesh (press and slice), no tet went below 0.6 in that record; a crush by a pad, the board
        or the blade's flat is recorded. Handling: every recorded robot grasp that lifted a fillet crushed it below 0.6
        (inverted tets under the fingers), and the position-mode closes that crushed nothing did not lift it; the
        threshold is kept, since those are real crushes in this world, and the grasp finding is reported as an
        environment issue. Bones have their own checks. (A2 uses workmanship(), which deducts per tet and per bond.)"""
        w = self.env.world
        ts = w.tissue
        asm = w.assembly
        tets = frag["tets"]
        rest_l = asm["to_local"](ts.hX[ts.htets[tets]].mean(1))
        cell = 25.4e-3
        gi = np.floor(rest_l[:, :2]/cell).astype(int)
        cells = {tuple(g) for g in gi}
        knife, tear, bruise = set(), set(), set()
        # a knife score: a knife-made cut face with both its tets in this fillet, an incision that did not separate
        # anything. Meat left on the bone is priced by yield alone
        own = set(tets.tolist())
        pre = asm.get("interface_faces", set())
        torn_f = set(getattr(w.cut, "torn", {}))
        inc = [t for f in w.cut.cut_faces if f[0] in own and f[1] in own and f not in pre and f not in torn_f for t in f]
        if inc:
            knife = {tuple(g) for g in gi[np.isin(tets, np.asarray(inc, np.int64))]}
        if getattr(ts, "n_bonds", 0):
            dmg = ts.bonds["damage"].numpy()
            a, b = ts.bonds["a"].numpy(), ts.bonds["b"].numpy()
            x = ts.x.numpy()
            nodes = np.unique(ts.htets[tets])
            torn = (dmg > .999) & (np.isin(a, nodes) | np.isin(b, nodes)) & \
                (np.linalg.norm(x[a] - x[b], axis=1) > 1e-3) & ~self.env.bond_cut_by_blade()
            if torn.any():
                pl = asm["to_local"](ts.hX[a[torn]])
                tear = {g for g in map(tuple, np.floor(pl[:, :2]/cell).astype(int)) if g in cells}
        jmin = getattr(self.env, "tet_jmin_far", None)
        jall = self.env.tet_jmin_history
        nb_counted = nb_edge = 0
        if jmin is not None:
            counted = jmin[tets] < .6
            nb_counted = int(counted.sum())
            nb_edge = int(((jall[tets] < .6) & ~counted).sum()) if jall is not None else 0
            bruise = {tuple(g) for g in gi[counted]}
        ded = .5*len(knife) + .5*len(tear) + 2.*len(bruise)
        score = 100. - ded*(.454/max(frag["mass"], 1e-6))
        return float(score), dict(cells=len(cells), knife_score_squares=len(knife), tear_squares=len(tear),
                                  bruise_squares=len(bruise), deductions=float(ded), bruised_tets_counted=nb_counted,
                                  compressed_tets_at_the_edge=nb_edge)

    def _mincut_graph(self):
        """Static data for _remaining_connection, built once: the original tet face graph and each tet's side (its own
        label for muscle; for everything else - frame, head, belly edge - the side of the nearest muscle tet, so one
        side's number neither routes through nor changes with the other side's half of the fish)."""
        if getattr(self, "_mc", None) is not None:
            return self._mc
        from scipy.spatial import cKDTree
        w = self.env.world
        ts = w.tissue
        nt = len(w.cut.orig)
        ref = np.asarray(w.assembly["ref_tet"][:nt])
        U, L = REF["fillet_upper"], REF["fillet_lower"]
        side = np.where(ref == L, 1, 0)
        mus, oth = np.nonzero((ref == U) | (ref == L))[0], np.nonzero((ref != U) & (ref != L))[0]
        cen = ts.hX[w.cut.orig].mean(1)
        side[oth] = side[mus][cKDTree(cen[mus]).query(cen[oth])[1]]
        t = np.repeat(np.arange(nt), 4)
        u = w.cut._nbr[:nt].ravel()
        m = u > t
        t, u = t[m], u[m]
        keys = t.astype(np.int64)*nt + u
        pre = w.assembly.get("interface_faces", set())
        prek = np.fromiter((a*nt + b for a, b in pre), np.int64, len(pre))
        # the source of each side's minimum cut is its meat core: muscle more than CORE_M from frame or head tissue and
        # holding no flesh-bone tie, so a cut that leaves the near-bone layer on the frame (which the task allows; the
        # yield term prices it) severs the side as fully as one along the label boundary (459 of
        # the 789 starting connections are ties straight from fillet-labelled meat)
        fh = np.nonzero((ref == REF["frame"]) | (ref == REF["head"]))[0]
        dist = cKDTree(cen[fh]).query(cen[mus])[0] if len(fh) else np.full(len(mus), np.inf)
        tied = np.isin(w.cut.orig, np.unique(ts.bties["node"].numpy())).any(1) if ts.n_bties else np.zeros(nt, bool)
        core = np.zeros(nt, bool)
        core[mus[dist > self.CORE_M]] = True
        core &= ~tied
        self._mc = dict(nt=nt, ref=ref, side=side, t=t, u=u, keys=keys, iface=np.isin(keys, prek), R0={}, core=core)
        return self._mc

    CORE_M = 5.25e-3                 # m (1.5 mesh lengths): the meat core starts this far from frame or head tissue

    def _remaining_connection(self, side, intact=False, knife_zone=None):
        """The fewest intact connections - uncut tet faces, interface bonds not fully damaged, flesh-bone ties not
        fully damaged, one each - still joining the side's muscle (rest label fillet_<side>) to any rigid bone (head or
        vertebra): a maximum flow / minimum cut over the tets of that side (_mincut_graph). It never rises as tissue is
        cut, is unchanged by cuts that free nothing, and is ~0 once the side's meat core is off the bone. Its source is
        the side's meat core (_mincut_graph, CORE_M). A torn face, a torn (not blade-cut) bond or a fully damaged tie
        counts as severed only inside `knife_zone` (per tet; None: nowhere): tearing that runs from the knife's cut is
        part of cutting, tearing elsewhere is not. A creator check dropped the uncut trout 20 mm into the frame pan:
        ~465 faces torn, 250 ties fully damaged, no knife; counting tears that had opened by 1 mm
        credited that drop 0.05 of progress, so peeling is not credited until the fillet comes free.
        intact=True: the same before any cut (only the interfaces split at build are cut faces, every bond and tie
        intact), cached."""
        g = self._mincut_graph()
        if intact and side in g["R0"]:
            return g["R0"][side]
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import maximum_flow
        w = self.env.world
        ts = w.tissue
        nt = g["nt"]
        s_ = 0 if side == "upper" else 1
        act = ts.hactive[:nt] > 0
        alive = (g["side"] == s_) & (act | intact)
        t, u = g["t"], g["u"]
        keep = alive[t] & alive[u]
        if intact:
            keep &= ~g["iface"]
        zone = np.zeros(nt, bool) if knife_zone is None else np.asarray(knife_zone, bool)[:nt]
        if not intact and w.cut.cut_faces:
            torn = set(getattr(w.cut, "torn", {}))
            sev = [f for f in w.cut.cut_faces if f not in torn or zone[f[0]] or zone[f[1]]]
            ck = np.fromiter((a*nt + b for a, b in sev), np.int64, len(sev))
            keep &= ~np.isin(g["keys"], ck)
        S, K = nt, nt + 1
        rows, cols = [t[keep], u[keep]], [u[keep], t[keep]]
        ai = np.nonzero(act | intact)[0]
        node_tet = np.full(ts.n, -1, np.int64)
        node_tet[ts.htets[ai].ravel()] = np.repeat(ai, 4)
        if getattr(ts, "n_bonds", 0):
            ta, tb = node_tet[ts.bonds["a"].numpy()], node_tet[ts.bonds["b"].numpy()]
            e = (ta >= 0) & (tb >= 0) & (ta != tb)
            if not intact:
                broken = ts.bonds["damage"].numpy() >= 1.
                cut_b = self.env.bond_cut_by_blade()
                e &= ~broken | (~cut_b & ~zone[np.maximum(ta, 0)] & ~zone[np.maximum(tb, 0)])
            e[e] &= alive[ta[e]] & alive[tb[e]]
            rows += [ta[e], tb[e]]
            cols += [tb[e], ta[e]]
        if ts.n_bties:
            tt = node_tet[ts.bties["node"].numpy()]
            ok = (ts.bties["body"].numpy() >= w.assembly["n_knife_bodies"]) & (tt >= 0)
            if not intact:
                ok &= (ts.bties["damage"].numpy() < 1.) | ~zone[np.maximum(tt, 0)]
            tt = tt[ok]
            tt = tt[alive[tt]]
            rows.append(tt)
            cols.append(np.full(len(tt), K))
        src = np.nonzero(alive & g["core"] & (g["ref"] == (REF["fillet_upper"], REF["fillet_lower"])[s_]))[0]
        if not len(src):
            return 0
        rows.append(np.full(len(src), S))
        cols.append(src)
        r, c = np.concatenate(rows).astype(np.int64), np.concatenate(cols).astype(np.int64)
        cap = np.ones(len(r), np.int64)
        cap[-len(src):] = 10**6
        G = coo_matrix((cap, (r, c)), shape=(nt + 2, nt + 2)).tocsr()
        G.sum_duplicates()
        G.data = np.minimum(G.data, 2**31 - 1).astype(np.int32)
        G.setdiag(0)
        G.eliminate_zeros()
        R = int(maximum_flow(G, S, K).flow_value)
        if intact:
            g["R0"][side] = R
        return R

    def _eval_A0(self):
        """A0 take-knife: held by the right hand's inserts alone for 3 s, seated in the channel (axis
        within 10 deg, lateral offset within the channel), slip under 2 mm over that time, edge down above the board,
        the blade never touched a robot link."""
        env, w = self.env, self.env.world
        from .scene import INSERT, HANDLE_H
        pinch = w.pinch_forces()
        held_for = 0. if env.knife_held_since is None else env.sim_time - env.knife_held_since
        right = min(pinch["right"]) > .5
        left_touch = max(pinch.get("left", [0., 0.])) > 0. or env.blade_left_hand_contacts > 0
        checks = dict([check("held_by_right_hand_s", held_for, 3., "s of continuous right-hand pinch",
                             right and held_for >= 3.)])
        checks.update([check("not_touched_by_left_hand", float(left_touch), 0., "bool", not left_touch)])
        off, ang = w.knife_in_grip()
        lateral = float(np.hypot(off[1], off[2]))
        # the channel admits the handle's 32 mm height with 0.5 mm clearance each side: a seated handle's centre line
        # lies within the lips (lateral <= lip height)
        seat_tol = INSERT["lip_height"]
        checks.update([check("axis_in_channel_deg", ang, 10., "deg", right and ang <= 10.),
                       check("seated_lateral_offset_mm", lateral*1e3, seat_tol*1e3, "mm", right and lateral <= seat_tol)])
        hist = [h for h in env.grip_history if h[0] >= env.sim_time - 3. - 1e-9]
        slip = max((float(np.linalg.norm(h[1] - hist[-1][1])) for h in hist), default=None) if hist else None
        checks.update([check("slip_mm_over_3s", None if slip is None else slip*1e3, 2., "mm",
                             slip is not None and held_for >= 3. and slip <= 2e-3)])
        # edge down above the board: the knife frame's edge->spine axis (handle z) points up within 30 deg, and every
        # blade segment is over the board and above its surface
        from .workpiece import blade_segment_poses
        _, Rk = w.body_pose("knife_handle")
        tilt = math.degrees(math.acos(float(np.clip(Rk[2, 2], -1, 1))))
        segs = [p for _, p in blade_segment_poses(w)]
        over = all(abs(p[0] - LY.BOARD["center"][0]) <= LY.BOARD["size"][0]/2 and
                   abs(p[1] - LY.BOARD["center"][1]) <= LY.BOARD["size"][1]/2 and p[2] > LY.BOARD_TOP_Z + .01 for p in segs)
        checks.update([check("edge_down_deg", tilt, 30., "deg from vertical", tilt <= 30.),
                       check("blade_over_board", float(over), 1., "bool", over),
                       check("blade_never_touched_robot", env.blade_robot_contacts, 0, "ticks with contact",
                             env.blade_robot_contacts == 0)])
        checks["time"] = check("time", env.sim_time, TIME_LIMIT_S["A0"], "s", env.sim_time <= TIME_LIMIT_S["A0"])[1]
        return checks, dict(pinch_n=pinch, grasp_offset_mm=(np.asarray(off)*1e3).round(2).tolist(),
                            knife_mode=w.knife_mode, knife_start=w.knife_start)

    def _eval_A4(self):
        """A4 remove-belly-bones: no rib piece over 2 mm left in the fillet, no skin hole over 25 mm^2,
        flesh removed with the bones at most 6 % of the fillet's mass (authored, not anchored: no published number)."""
        env = self.env
        lab, frags, main = self._parts()
        total = sum(f["mass"] for f in frags.values())
        removed = total - frags[main]["mass"] if main is not None else 0.
        pieces = self.rod_pieces(lab)
        left = [p for p in pieces if p["kind"] == "rib" and p["fragment"] == main and p["length"] > 2e-3]
        skin_all = sum(f["skin_m2"] for f in frags.values())
        skin_off = skin_all - frags[main]["skin_m2"] if main is not None else 0.
        # a quality check of a product counts only when there is a product: some rib taken out of the fillet
        ribs_out = any(p["kind"] == "rib" and p["fragment"] != main for p in pieces)
        nop = "" if ribs_out else NOTHING_MADE
        checks = dict([check("no_rib_pieces_over_2mm", len(left), 0, "pieces left in the fillet", not left),
                       check("skin_hole_mm2", skin_off*1e6, 25., "mm2 of skin gone from the fillet",
                             ribs_out and skin_off*1e6 <= 25., nop),
                       check("flesh_removed", removed/max(total, 1e-12), .06, "fraction of the fillet's mass",
                             ribs_out and removed <= .06*total, nop),
                       check("time", env.sim_time, TIME_LIMIT_S["A4"], "s", env.sim_time <= TIME_LIMIT_S["A4"])])
        dm = self._common_cut(checks, frags)
        return checks, dict(**dm, fillet_kg=total, removed_kg=removed, rib_pieces_left=len(left),
                            rib_pieces_total=sum(1 for p in pieces if p["kind"] == "rib"))

    def _eval_A1(self):
        env, w = self.env, self.env.world
        lab, frags = self.fragments()
        attach = self.bone_attachment(lab)
        hd = self.head_fragment(attach)
        fr = self.frame_fragment(lab, frags, attach)
        ts = w.tissue
        # the backbone between the head and the frame is cut at some joint: the joints head->segment_00,
        # segment_00->segment_01, ... up to the frame's first segment (the anterior vertebrae have a joint each, so the
        # heading cut may take one or more of them with the head)
        on = ts.joints["on"].numpy()
        jname = {jd.get("name", ""): j for j, jd in enumerate(ts.joint_desc)}
        nk = w.assembly["n_knife_bodies"]
        names = [rb["name"] for rb in w.workpiece.rigid_bodies]
        segs = sorted(n for n in names if n.startswith("segment_"))
        def tied_to(n):
            a_ = attach.get(nk + names.index(n), {})
            return max(a_, key=a_.get) if a_ else None
        first = next((k for k, n in enumerate(segs) if fr is not None and tied_to(n) == fr), len(segs) - 1)
        chain = ["head->segment_00"] + [f"{segs[k - 1]}->{segs[k]}" for k in range(1, first + 1)]
        cut_joints = [c for c in chain if c in jname and not on[jname[c]]]
        head_joint_on = not cut_joints
        separated = (not head_joint_on) and (hd is None or hd != fr)
        meat = 0.
        if hd is not None and hd != fr:
            meat = float(frags[hd]["by_ref"][REF["fillet_upper"]] + frags[hd]["by_ref"][REF["fillet_lower"]])
        on_board = self.in_footprint(ts.x.numpy()[self.fragment_nodes(frags[fr])], LY.BOARD).mean() if fr is not None else 0.
        checks = dict([check("head_separated", float(separated), 1., "bool", separated,
                             f"backbone cut at {', '.join(cut_joints) or 'no joint'}; no flesh path from the head to "
                             "the body" if hd != fr else "the head's flesh is joined to the body"),
                       check("meat_with_head", meat*1e3, 8., "g of fillet muscle behind the collar line",
                             separated and meat <= 8e-3, "" if separated else NOTHING_MADE),
                       check("fish_on_board", on_board, .95, "fraction of body nodes over the board", on_board >= .95)])
        self._common_knife(checks)
        dm = self._common_cut(checks, frags)
        checks["time"] = check("time", env.sim_time, TIME_LIMIT_S["A1"], "s", env.sim_time <= TIME_LIMIT_S["A1"])[1]
        return checks, dict(**dm, head_fragment=hd, frame_fragment=fr, fragments=len(frags), meat_with_head_kg=meat)

    # ------------------------------------------------------------------ presets B and C
    def _parts(self):
        """Fragments with their mass, mass by reference label and by region label, skin area (m^2) and the main
        fragment (the largest)."""
        w = self.env.world
        ts = w.tissue
        lab, frags = self.fragments()
        nt = len(w.cut.orig)
        reg = w.assembly["region_tet"][:nt]
        m = ts.htmass[:nt]
        tri_frag = lab[ts.mem_tet] if getattr(ts, "n_tris", 0) else np.zeros(0, int)
        area = getattr(ts, "mem_area0", np.zeros(len(tri_frag)))
        for c, f in frags.items():
            t = f["tets"]
            f["by_reg"] = np.bincount(reg[t], weights=m[t], minlength=8)
            f["skin_m2"] = float(area[tri_frag == c].sum()) if len(tri_frag) else 0.
        main = max(frags, key=lambda c: frags[c]["mass"]) if frags else None
        return lab, frags, main

    def _cut_faces_between(self, lab, a, b=None, inside=False):
        """Rest-space centroids of cut faces between fragment a and b (b None: a and any other fragment); inside=True:
        cut faces with both sides in fragment a (a crack that has not separated anything)."""
        w = self.env.world
        ts = w.tissue
        cen = ts.hX[ts.htets[:len(w.cut.orig)]].mean(1)
        out = []
        pre = w.assembly.get("interface_faces", set())
        for u, v in w.cut.cut_faces:
            la, lb = lab[u], lab[v]
            if inside:
                if (min(u, v), max(u, v)) in pre:
                    continue            # a bonded interface split at build is not a crack
                if la == lb == a:
                    out.append(.5*(cen[u] + cen[v]))
                continue
            if la == lb or a not in (la, lb):
                continue
            if b is not None and b not in (la, lb):
                continue
            out.append(.5*(cen[u] + cen[v]))
        return np.array(out).reshape(-1, 3)

    def _sweep_normals(self):
        """Unit normals of the recorded blade surfaces (rest space), one per triangle, and their areas."""
        tri = np.array(self.env.world.cut.surface).reshape(-1, 3, 3)
        if not len(tri):
            return np.zeros((0, 3)), np.zeros(0)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        a = .5*np.linalg.norm(n, axis=1)
        return n/np.maximum(2*a[:, None], 1e-15), a

    def _region_id(self, name):
        return self.env.world.workpiece.region_names.index(name)

    def _ref_id(self, name):
        return self.env.world.workpiece.reference_names.index(name)

    def _eval_B2(self):
        """Trim the belly fat strip along the fat line: >= 90 % of the reference trim removed from the side, loin loss
        <= 3 % of the side's mass, trimmed edge raggedness (RMS deviation of the side's cut edge from its own 25 mm
        moving average, plan view) <= 2 mm. The belly membrane ('and 90 % of the membrane area') is not a
        separate layer in this model and is not scored."""
        env = self.env
        lab, frags, main = self._parts()
        rt, rl = self._ref_id("trim_belly"), self._ref_id("loin")
        total = sum(f["mass"] for f in frags.values())
        trim_all = sum(f["by_ref"][rt] for f in frags.values())
        trim_off = sum(f["by_ref"][rt] for c, f in frags.items() if c != main)
        loin_off = sum(f["by_ref"][rl] for c, f in frags.items() if c != main)
        pts = self._cut_faces_between(lab, main)
        rag = None
        if len(pts) >= 8:
            o = np.argsort(pts[:, 0]); x, y = pts[o, 0], pts[o, 1]
            sm = np.array([y[np.abs(x - xi) <= .0125].mean() for xi in x])
            rag = float(np.sqrt(np.mean((y - sm)**2)))
        checks = dict([check("trim_removed", trim_off/max(trim_all, 1e-12), .90, "fraction of the reference trim",
                             trim_off >= .90*trim_all),
                       check("loin_loss", loin_off*1e3, .03*total*1e3, "g (3 % of the side)",
                             trim_off > 0 and loin_off <= .03*total, "" if trim_off > 0 else NOTHING_MADE),
                       check("edge_raggedness_mm", None if rag is None else rag*1e3, 2., "mm RMS",
                             rag is not None and rag <= 2e-3, "no trimmed edge yet" if rag is None else ""),
                       check("time", env.sim_time, TIME_LIMIT_S["B2"], "s", env.sim_time <= TIME_LIMIT_S["B2"])])
        dm = self._common_cut(checks, frags, [c for c, f in frags.items() if c != main and f["mass"] >= 1e-3])
        return checks, dict(**dm, side_kg=total, trim_kg=trim_all, trim_removed_kg=trim_off, loin_removed_kg=loin_off,
                            fragments=len(frags))

    def _eval_B3(self):
        """Cut away the melanin spot: >= 99 % of the spot removed from the side, good flesh removed <= 2 x the spot's
        mass, no skin removed from the side (no skin hole)."""
        env = self.env
        lab, frags, main = self._parts()
        rs = self._region_id("melanin_spot")
        spot = sum(f["by_reg"][rs] for f in frags.values())
        spot_off = sum(f["by_reg"][rs] for c, f in frags.items() if c != main)
        good_off = sum(f["mass"] - f["by_reg"][rs] for c, f in frags.items() if c != main)
        skin_all = sum(f["skin_m2"] for f in frags.values())
        skin_off = skin_all - frags[main]["skin_m2"] if main is not None else 0.
        checks = dict([check("spot_removed", spot_off/max(spot, 1e-12), .99, "fraction of the spot's mass", spot_off >= .99*spot),
                       check("good_flesh_removed", good_off*1e3, 2*spot*1e3, "g (2 x the spot)",
                             spot_off > 0 and good_off <= 2*spot, "" if spot_off > 0 else NOTHING_MADE),
                       check("skin_hole_mm2", skin_off*1e6, 0., "mm2 of skin gone from the side",
                             spot_off > 0 and skin_off*1e6 < 1., "" if spot_off > 0 else NOTHING_MADE),
                       check("time", env.sim_time, TIME_LIMIT_S["B3"], "s", env.sim_time <= TIME_LIMIT_S["B3"])])
        dm = self._common_cut(checks, frags)
        return checks, dict(**dm, spot_kg=spot, spot_removed_kg=spot_off, good_removed_kg=good_off)

    def _eval_B5(self):
        """Portion the side: every full portion (all pieces but the two end pieces along the side) 150 +- 7 g (IMPS
        tolerance below 170 g); blade surfaces within 10 deg of perpendicular to the board (recorded cut surface,
        area-weighted, rest space); no loose piece of 1 to 5 g (a tag end cut free); every full portion weighed: the
        bench scale showed a steady reading within 2 g of its mass at some time."""
        env = self.env
        lab, frags, main = self._parts()
        pieces = sorted(((c, f) for c, f in frags.items() if f["mass"] >= 5e-3),
                        key=lambda cf: float(self.env.world.tissue.hX[self.env.world.tissue.htets[cf[1]["tets"]]].mean((0, 1))[0]))
        full = pieces[1:-1] if len(pieces) >= 3 else []
        n, a = self._sweep_normals()
        tilt = float(np.degrees(np.arcsin(np.clip(np.sum(np.abs(n[:, 2])*a)/max(a.sum(), 1e-15), 0, 1)))) if len(a) else None
        tags = [c for c, f in frags.items() if 1e-3 <= f["mass"] < 5e-3]
        log = list(env.scale_log)
        weighed = []
        for c, f in full:
            ok = False
            for i in range(2, len(log)):
                win = [r for _, r in log[i - 2:i + 1]]
                if max(win) - min(win) < 1. and abs(np.mean(win) - f["mass"]*1e3) <= 2.:
                    ok = True; break
            weighed.append(ok)
        masses = [f["mass"]*1e3 for _, f in full]
        in_tol = bool(masses) and all(143. <= m_ <= 157. for m_ in masses)
        checks = dict([check("full_portions", len(full), 1, "count", len(full) >= 1),
                       check("portions_150_7g", float(max((abs(m_ - 150.) for m_ in masses), default=np.nan)), 7.,
                             "g worst deviation", in_tol),
                       check("cut_tilt_deg", tilt, 10., "deg from vertical (area-weighted)",
                             bool(full) and tilt is not None and tilt <= 10., "" if full else NOTHING_MADE),
                       check("no_tag_ends", len(tags), 0, "pieces of 1-5 g", not tags),
                       check("weighed", float(np.mean(weighed)) if weighed else 0., 1., "fraction of full portions",
                             bool(weighed) and all(weighed)),
                       check("time", env.sim_time, TIME_LIMIT_S["B5"], "s", env.sim_time <= TIME_LIMIT_S["B5"])])
        dm = self._common_cut(checks, frags, [c for c, _ in pieces])
        return checks, dict(**dm, portion_g=masses, end_pieces_g=[f["mass"]*1e3 for _, f in (pieces[:1] + pieces[-1:])],
                            scale_last_g=log[-1][1] if log else None)

    def _eval_C1(self):
        """Remove the chain along its seam: >= 95 % of the chain's mass separated from the main muscle, main-muscle
        loss <= 3 %, no crack running into the main piece further than 25 mm (the 'tail split', approximated by the
        longest un-separating cut inside the main piece)."""
        env = self.env
        lab, frags, main_c = self._parts()
        rm, rc = self._region_id("main"), self._region_id("chain")
        # the main piece: the fragment holding most main-muscle mass
        main_c = max(frags, key=lambda c: frags[c]["by_reg"][rm]) if frags else None
        chain = sum(f["by_reg"][rc] for f in frags.values())
        chain_off = sum(f["by_reg"][rc] for c, f in frags.items() if c != main_c)
        main_all = sum(f["by_reg"][rm] for f in frags.values())
        main_off = main_all - frags[main_c]["by_reg"][rm] if main_c is not None else 0.
        inner = self._cut_faces_between(lab, main_c, inside=True) if main_c is not None else np.zeros((0, 3))
        split = float(np.ptp(inner[:, 0])) if len(inner) else 0.
        checks = dict([check("chain_separated", chain_off/max(chain, 1e-12), .95, "fraction of the chain's mass",
                             chain_off >= .95*chain),
                       check("main_loss", main_off/max(main_all, 1e-12), .03, "fraction of the main muscle",
                             chain_off > 0 and main_off <= .03*main_all, "" if chain_off > 0 else NOTHING_MADE),
                       check("tail_split_mm", split*1e3, 25., "mm", split <= .025),
                       check("time", env.sim_time, TIME_LIMIT_S["C1"], "s", env.sim_time <= TIME_LIMIT_S["C1"])])
        dm = self._common_cut(checks, frags, [c for c, f in frags.items() if c != main_c and f["mass"] >= 1e-3])
        return checks, dict(**dm, chain_kg=chain, chain_separated_kg=chain_off, main_lost_kg=main_off)

    def _eval_C2(self):
        """Denude: >= 90 % of the silver skin's area removed from the main muscle, main-muscle loss <= 5 %. The fat
        depth criterion (IMPS: no remaining fat deeper than 3 mm) is reported as the fat mass left on the main piece."""
        env = self.env
        lab, frags, _ = self._parts()
        rm, rf = self._region_id("main"), self._region_id("fat")
        main_c = max(frags, key=lambda c: frags[c]["by_reg"][rm]) if frags else None
        silver_all = sum(f["skin_m2"] for f in frags.values())
        silver_left = frags[main_c]["skin_m2"] if main_c is not None else 0.
        main_all = sum(f["by_reg"][rm] for f in frags.values())
        main_off = main_all - frags[main_c]["by_reg"][rm] if main_c is not None else 0.
        checks = dict([check("silver_removed", 1 - silver_left/max(silver_all, 1e-12), .90, "fraction of silver area",
                             silver_left <= .10*silver_all),
                       check("main_loss", main_off/max(main_all, 1e-12), .05, "fraction of the main muscle",
                             silver_left < silver_all and main_off <= .05*main_all,
                             "" if silver_left < silver_all else NOTHING_MADE),
                       check("time", env.sim_time, TIME_LIMIT_S["C2"], "s", env.sim_time <= TIME_LIMIT_S["C2"])])
        dm = self._common_cut(checks, frags, [main_c] if main_c is not None else [])
        return checks, dict(**dm, silver_m2=silver_all, silver_left_m2=silver_left,
                            fat_left_kg=float(frags[main_c]["by_reg"][rf]) if main_c is not None else None)

    def _eval_C3(self):
        """Medallions 25 +- 5 mm thick from the thick part: pieces of 30 g or more with both ends cut; thickness along
        the tenderloin's axis (rest space) measured on their tets more than 6 mm from the piece's outline, thickest
        minus thinnest <= 5 mm, blade surfaces within 10 deg of perpendicular to the axis; no loose piece of 1 to 5 g.
        (The IMPS suspension check is not modelled.)"""
        env, w = self.env, self.env.world
        ts = w.tissue
        lab, frags, main = self._parts()
        cen = ts.hX[ts.htets[:len(w.cut.orig)]].mean(1)
        rows = []
        for c, f in frags.items():
            if f["mass"] < .03:
                continue
            p = cen[f["tets"]]
            r = np.hypot(p[:, 1] - p[:, 1].mean(), p[:, 2] - p[:, 2].mean())
            core = p[r <= max(r.max() - .006, 0.)]
            if len(core) < 4:
                continue
            ends = self._cut_faces_between(lab, c)
            both = len(ends) and (ends[:, 0].min() < p[:, 0].mean() < ends[:, 0].max())
            th = float(np.ptp(core[:, 0])) + w.config.mesh_h/2      # tet centroids sit ~h/4 inside each face
            rows.append(dict(fragment=c, mass_g=f["mass"]*1e3, thickness_mm=th*1e3, both_ends_cut=bool(both)))
        # a medallion is a slice: both ends cut and at most twice the nominal thickness
        meds = [r for r in rows if r["both_ends_cut"] and r["thickness_mm"] <= 50.]
        n, a = self._sweep_normals()
        tilt = float(np.degrees(np.arccos(np.clip(np.sum(np.abs(n[:, 0])*a)/max(a.sum(), 1e-15), 0, 1)))) if len(a) else None
        ths = [r["thickness_mm"] for r in meds]
        tags = [c for c, f in frags.items() if 1e-3 <= f["mass"] < 5e-3]
        checks = dict([check("medallions", len(meds), 1, "count", len(meds) >= 1),
                       check("thickness_25_5mm", float(max((abs(t - 25.) for t in ths), default=np.nan)), 5., "mm worst",
                             bool(ths) and all(abs(t - 25.) <= 5. for t in ths)),
                       check("thickness_spread_mm", float(np.ptp(ths)) if ths else None, 5., "mm", bool(ths) and np.ptp(ths) <= 5.),
                       check("cut_tilt_deg", tilt, 10., "deg from perpendicular",
                             bool(meds) and tilt is not None and tilt <= 10., "" if meds else NOTHING_MADE),
                       check("no_tag_ends", len(tags), 0, "pieces of 1-5 g", not tags),
                       check("time", env.sim_time, TIME_LIMIT_S["C3"], "s", env.sim_time <= TIME_LIMIT_S["C3"])])
        dm = self._common_cut(checks, frags, [c for c, f in frags.items() if f["mass"] >= .03])
        return checks, dict(**dm, pieces=rows)

    # ------------------------------------------------------------------ workmanship (NOAA method translated to state)
    def workmanship(self, frag, fid, lab, key):
        """25.4 mm grid over the fillet's flesh side (rest coordinates, projected on the fish's side plane); per
        touched square: 1/2 point for a knife score (cut flesh more than 2 mm inside the ideal fillet), 1/2 for a tear
        (a torn interface or sheet bond), 2 for a bruise (a tet compressed below J = 0.6 at any time), 3 for bones
        normally removed (rib or fin-support piece). Normalised per 454 g of fillet as the standard does."""
        w = self.env.world
        ts = w.tissue
        asm = w.assembly
        tets = frag["tets"]
        rest_l = asm["to_local"](ts.hX[ts.htets[tets]].mean(1))
        ref = asm["ref_tet"][tets]
        cell = 25.4e-3
        gi = np.floor(rest_l[:, :2]/cell).astype(int)
        cells = {tuple(g) for g in gi}
        ded = {}
        # knife scores: tets of this fragment whose cut faces lie >2 mm inside the ideal fillet: fragment tets with
        # the fillet label that have a cut face and whose rest centroid is > 2 mm from any non-fillet tet
        cutf = w.cut.cut_faces
        cut_tets = {a for a, b in cutf} | {b for a, b in cutf}
        from scipy.spatial import cKDTree
        allc = asm["to_local"](ts.hX[ts.htets[:len(w.cut.orig)]].mean(1))
        non = np.nonzero(asm["ref_tet"] != key)[0]
        tree = cKDTree(allc[non]) if len(non) else None
        for t, g, r in zip(tets, gi, ref):
            if r == key and int(t) in cut_tets and tree is not None:
                d, _ = tree.query(allc[t])
                if d > 2e-3 + .5*w.config.mesh_h:
                    ded[tuple(g)] = max(ded.get(tuple(g), 0.), .5)
        # tears: damaged interface bonds with a node in the fragment
        if getattr(ts, "n_bonds", 0):
            dmg = ts.bonds["damage"].numpy()
            a = ts.bonds["a"].numpy(); b = ts.bonds["b"].numpy()
            x = ts.x.numpy()
            nodes = set(np.unique(ts.htets[tets]).tolist())
            torn = (dmg > .999) & np.array([int(u) in nodes or int(v) in nodes for u, v in zip(a, b)])
            opening = np.linalg.norm(x[a] - x[b], axis=1) > 1e-3
            for u in np.nonzero(torn & opening & ~self.env.bond_cut_by_blade())[0]:
                p = asm["to_local"](ts.hX[a[u]][None])[0]
                g = tuple(np.floor(p[:2]/cell).astype(int))
                if g in cells:
                    ded[g] = ded.get(g, 0.) + .5
        # bruises: worst compression history
        jmin = self.env.tet_jmin_history
        if jmin is not None:
            for t, g in zip(tets, gi):
                if jmin[t] < .6:
                    ded[tuple(g)] = ded.get(tuple(g), 0.) + 2.
        total = sum(ded.values())
        mass = max(frag["mass"], 1e-6)
        score = 100. - total*(.454/mass)
        return float(score), dict(cells=len(cells), deductions=float(total), deducted_cells=len(ded))

    def _skin_mass(self, frag):
        w = self.env.world
        ts = w.tissue
        if not ts.n_tris:
            return 0.
        tset = set(frag["tets"].tolist())
        keep = np.array([t in tset for t in ts.mem_tet])
        return float(ts.mem_corner_m[keep].sum())
