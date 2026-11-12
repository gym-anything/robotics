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
# A benchmark task's evaluation is in its verifier.py (tasks/<name>/, layout.task_verifier), which may also define what
# env.py reads: RECEPTACLES (layout names of its product receptacles: a piece descending into one is not thrown tissue),
# STOP_AT_TIME_LIMIT (env.act runs no tick past TIME_LIMIT_S: an action crossing it is cut at the limit) and
# prime(evaluator) (its progress measure's start, from the state the environment starts in).


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


def has_evaluator(task):
    """Whether the released evaluation of a task has checks: a benchmark task's verifier.py with released_checks, or an
    Evaluator._eval_<task> below."""
    folder = LY.task_verifier(task)
    if folder is not None:
        return hasattr(folder, "released_checks")
    return hasattr(Evaluator, f"_eval_{task}")


class Evaluator:
    contract = None                  # the released checks (task_set_evaluator.py and -v2 name their contract)

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
        """The episode's outcome: a benchmark task's verifier.py (tasks/<name>/) decides; any other task's checks are
        its _eval_<task> below (evaluate_checks)."""
        task = self.env.world.config.task
        folder = LY.task_verifier(task)
        if folder is not None:
            return folder.verify(self)
        return self.evaluate_checks(getattr(type(self), f"_eval_{task}", None))

    def evaluate_checks(self, fn, *, progress_cap=None):
        """The released evaluation: fn(evaluator) -> (checks, metrics), None for a task without an evaluator (reported
        in the note). Success needs every check; the score is the fraction of checks passed, or with progress_cap (a
        task scored by its progress measure) 1 on success, else progress_cap x the progress in its metrics."""
        env = self.env
        w = env.world
        task = w.config.task
        if w.fault:
            return dict(schema="gar.evaluation.v1", success=False, score=0., checks={}, metrics={}, fault=w.fault)
        checks, metrics = ({}, {}) if fn is None else fn(self)
        metrics["crack_faces"] = self.crack_faces()
        passed = [c["passed"] for c in checks.values()]
        success = bool(passed) and all(passed)
        if progress_cap is not None:
            # 1.0 for every completion that passes; otherwise the progress toward the stated outcome, kept below it
            score = 1. if success else progress_cap*metrics["progress"]["value"]
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

    # ------------------------------------------------------------------ robot hands and the knife put down
    TOUCH_M = TOUCH_M                # m: a node this close to a robot hand box is held, to the blade is on the knife

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
