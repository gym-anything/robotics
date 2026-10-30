"""Task instructions and the physical-state evaluator.

Success is judged only from simulator state (sheet vertices, rigid poses, crease angles, tape bond damage,
robot contact) after a hands-off window: every "at rest" criterion is evaluated over the last HANDS_OFF_S of
the episode, during which no robot link may touch any packaging item. Thresholds are authored [A]. A numerical
fault is a backend fault, never a task outcome.

bag-purchase differs from open-bag, which shares _eval_bag: the board must be inside the bag (on the bottom panel's
inner side), the candle must rest on it with its print up, the bag must stand mouth up, the stack under the taken bag
is reported but not judged, the score is 1 on success else FAILED_SCORE_FACTOR x a continuous progress (_bag_progress), and the
explicit evaluation completes the hands-off window itself (SETTLE_TASKS; env.GiftBagsEnv.final_evaluation).
"Open" is the mouth plus the least front-to-back face gap at any height (the gusset checks are reported only); the
window counts only contact with the job's items, directly or through one piece of stock (_job_contact); "at rest" is
read over the window's last 2 s.
"""

import math
from collections import deque

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree

HANDS_OFF_S = 3.0

INSTRUCTIONS = {
    "open-bag": (
        "You are at the cash-wrap counter of a boutique. Take the top gift bag from the stack of flat bags, open "
        "it, square its block bottom, put one of the bottom boards inside it flat on the bottom, and leave the bag "
        "standing open on the dark rubber counter mat. Do not tear the bag. Release everything when you are done."),
    "bag-purchase": (
        "You are at the cash-wrap counter of a boutique, bagging a customer's boxed candle. Take the top gift bag from "
        "the stack, open it, square its bottom and put one of the bottom boards in it. Leave the bag standing open on "
        "the dark rubber counter mat with the boxed candle inside, upright on the board. Release everything when you "
        "are done."),
    "lift-lid": (
        "A gift box stands on the dining table. Take the lid off the box and put it down on the table. Do not lift "
        "or tip the box's base while doing it, and leave the photo frame inside the base. Release the lid."),
    "unwrap-gift": (
        "A wrapped present lies on the dining table. Remove the wrapping paper from the gift box (tearing it is "
        "fine) and leave the box closed, upright on the table. Release everything when you are done."),
    "open-present": (
        "A wrapped present lies on the dining table. Unwrap it (tearing the paper is fine, and you may pick the "
        "present up and turn it over), take the lid off the box, take out the photo frame and put it down on the table "
        "next to the box. Leave the lid on the table and the box's base standing upright on the table. Release "
        "everything when you are done."),
    "open-carton": (
        "You are at the receiving table in a back room. A supply carton sealed with brown packing tape stands on the "
        "stainless table. Open it: there is no knife, so peel the seam tape off where the two top flaps meet, starting "
        "at the tape's folded-back tab on one end of the carton, and fold all four top flaps open so that they stay "
        "open. Leave the contents inside and the carton upright where it is. Release everything when you are done."),
    "pinch": "Development scene: pinch the gift wrap overhanging the block and lift it.",
}

# authored thresholds [A]
MOUTH_RATIO_MIN = .80          # convex hull of the rim over W x G
BOTTOM_FLAT_FRACTION = .85     # bottom panel area within BOTTOM_FLAT_MM of the support plane
BOTTOM_FLAT_MM = 3.
TILT_MAX_DEG = 10.
GUSSET_OPEN_DEG = 120.         # gusset centre crease opening (180 flat, 0 folded) ...
GUSSET_LENGTH_FRACTION = .80   # ... over this fraction of its length
BASE_DRIFT_MM = 2.             # bag base drift over the hands-off window
BOARD_FRACTION = .90           # board area within BOARD_MM of the bottom panel
BOARD_MM = 4.
STACK_MOVE_MM = 20.            # bystanders
CANDLE_TILT_DEG = 15.
LID_BASE_LIFT_MM = 5.          # lift-lid: rise of the HIGHEST point of the base's bottom face (any end of it), whole
                               # episode (the lowest point misses a base raised at one end: one end
                               # lifted ~3 mm while the lowest point read 0.25 mm)
LID_BASE_TILT_DEG = 10.
COVERAGE_MAX = .05             # unwrap: box outer area with paper within COVER_MM
COVER_MM = 3.
LID_CLOSED_OVERLAP_MM = 20.
TEAR_TOL_MM = 5.               # open-bag / bag-purchase: new crack length allowed in the bag (below one 8-10 mm element:
                               # any split or fully torn triangle of the tote fails "Do not tear the bag") [A]
# bag-purchase only; open-bag keeps the rules above
CANDLE_REST_MM = 5.            # "upright on the board": the carton's lowest corner at most this far above the top face of
                               # a board that is in the bag, its centre over that board (a carton resting on it: ~0.03 mm) [A]
REST_MM = 5.                   # progress: the bag stands ON the mat when its lowest point is this close to the mat top [A]
TEAR_ZERO_MM = 25.             # progress: integrity 1 up to TEAR_TOL_MM, falling linearly to 0 here (about 2.5 of the
                               # tote's 10 mm cells, a rip anyone sees: a torn tote is not handed to a customer) [A]
FAILED_SCORE_FACTOR = .9       # score: 1 on success, else this x progress (label_application's convention) [A]
RIM_GAP_MIN_FRAC = .5          # "open" is the mouth (hull ratio >= MOUTH_RATIO_MIN) and not pinched: the least
RIM_PAIR_HALF_X = .080         # front-to-back gap of the rim across its middle 160 mm >= this fraction of the gusset
                               # depth G (a mouth pinched shut at its middle reads hull 1.0; recorded totes 76-102 mm) [A]
JOB_MARGIN_MM = 20.            # the release window counts a robot link that touches something only if it is
                               # within this of the tote, its ropes or knots, the candle or a board taken from its pile
BOARD_TAKEN_MM = 20.           # a board more than this from its start pose has left the pile
SETTLE_TASKS = ("bag-purchase",)  # the explicit evaluation completes the hands-off window itself (env.final_evaluation)
# open-carton [A]
FLAP_OPEN_DEG = 90.            # every top flap at or past vertical (90 deg from closed) through the whole window
SEAM_JOINED_MAX = .10          # seam length still joined (tape bonded to both majors) <= 10 %: removed from >= 90 %
CARTON_MOVE_MM = 30.           # carton body displacement from the start
CARTON_TILT_DEG = 10.          # carton upright
CONTENTS_MOVE_MM = 10.         # each item of the contents, displacement from the start


def _tilt_deg(R, axis=2):
    return math.degrees(math.acos(min(1., abs(float(R[2, axis])))))


class History:
    """Per-step compact record for window and whole-episode criteria."""

    def __init__(self, dt):
        self.n = int(round(HANDS_OFF_S/dt))
        self.rows = deque(maxlen=self.n)
        self.peak = {}

    def add(self, row, peaks):
        self.rows.append(row)
        for k, v in peaks.items():
            self.peak[k] = max(self.peak.get(k, -np.inf), v)


class Evaluator:
    def __init__(self, env):
        self.env = env
        self.world = env.world
        self.task = env.config.task
        sc = self.world.scene
        self.geo = sc.geometry
        ipc = self.world.ipc
        self.body = {b.name: i for i, b in enumerate(ipc.bodies)}
        self.initial = {n: ipc.body_pose(i)[0].copy() for n, i in self.body.items() if ipc.bodies[i].kind == "dynamic"}
        self.initial_R = {n: ipc.body_pose(i)[1].copy() for n, i in self.body.items() if ipc.bodies[i].kind == "dynamic"}
        self.history = History(env.config.dt)
        # torn length of every tearable sheet at the start (IPCWorld.crack_length: split crack faces + crack band)
        self.crack0 = {k: sum(ipc.crack_length(k)) for k in range(len(ipc.sheets))} if hasattr(ipc, "crack_length") else {}
        if sc.preset == "counter":
            self._bag_setup()
            self.mouth0 = self._mouth(self._bag_x())[0]      # the start state's mouth ratio: bag-purchase progress's zero
            self.rim_gap0 = self._rim_gap(self._bag_x())     # and its least face gap (the flat tote: ~0.2 mm)

    # ------------------------------------------------------------------ geometry helpers
    def _bag_setup(self):
        from . import bagnet as B
        ipc = self.world.ipc
        self.bag = ipc.sheets[0]
        X = np.asarray(self.bag.X2)
        self.W, self.G, self.H = B.W_BAG, B.G_BAG, B.H_BAG
        self.rim = np.where(np.abs(X[:, 2] - self.H) < 1e-7)[0]
        kinds = self.bag.render["kinds"]
        self.bottom_tris = np.where(kinds == "bottom")[0]
        self.bottom_v = np.unique(np.asarray(self.bag.tris)[self.bottom_tris])
        crease = getattr(self.bag, "crease_names", {})
        self.gusset_edges = {s: [e for e, nm in crease.items() if nm == "gusset_centre" and X[e[0], 0]*s > 0]
                             for s in (-1., 1.)}
        T = np.asarray(self.bag.tris)
        edge_tris = {}
        for t, tri in enumerate(T):
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge_tris.setdefault((min(a, b), max(a, b)), []).append(t)
        self.edge_tris = edge_tris
        # front and back face vertices paired by rest (x, z) across the middle of the width, at every height of the
        # faces (pinch guard)
        face = (np.abs(np.abs(X[:, 0]) - self.W/2) > 1e-9) & (np.abs(X[:, 0]) <= RIM_PAIR_HALF_X) & (X[:, 2] > 1e-9)
        front = np.where(face & (np.abs(X[:, 1] - self.G/2) < 1e-9))[0]
        back = np.where(face & (np.abs(X[:, 1] + self.G/2) < 1e-9))[0]
        key = {(round(float(X[b, 0]), 6), round(float(X[b, 2]), 6)): b for b in back}
        pairs = [(f, key.get((round(float(X[f, 0]), 6), round(float(X[f, 2]), 6)))) for f in front]
        self.rim_pairs = np.array([(f, b) for f, b in pairs if b is not None], int).reshape(-1, 2)

    def _rim_gap(self, x):
        """Least distance between paired front and back face vertices across the middle 160 mm of the width, at any
        height of the faces, rim included (m); None if the mesh has no such pairs."""
        if not len(self.rim_pairs):
            return None
        return float(np.linalg.norm(x[self.rim_pairs[:, 0]] - x[self.rim_pairs[:, 1]], axis=1).min())

    def _job_contact(self):
        """bag-purchase's release window: is a robot link that touches something within
        JOB_MARGIN_MM of an item of the job, i.e. the tote (sheet 0) with its ropes and knots, the candle or a board that
        has left its pile? A link farther than that from all of them touches only the stock (the lower totes, the unused
        boards), which the job does not ask to let go of. Distances to a link's convex hull (plane test: never more than
        the true distance, so near counts as touching); for rigid items also the link's vertices to the item's hull. The
        margin exceeds the tote's 10 mm cell, so a link touching paper between vertices is still near one."""
        ipc = self.world.ipc
        links = [b for b, (F, _) in ipc.reactions.items() if ipc.bodies[b].name.startswith("robot:") and np.linalg.norm(F) > 1e-3]
        if not links:
            return False
        world = lambda i: (lambda p, R: np.asarray(ipc.bodies[i].verts) @ R.T + p)(*ipc.body_pose(i))
        x = ipc.x_free()
        pts = np.vstack([x[ipc.sheet_slices[0]]] + [x[sl] for sl in getattr(ipc, "rod_slices", [])])
        rigid = [world(i) for n, i in self.body.items() if n.startswith("knot_") or n == "candle" or (
            n.startswith("board_") and np.linalg.norm(ipc.body_pose(i)[0] - self.initial[n]) > BOARD_TAKEN_MM*1e-3)]
        m = JOB_MARGIN_MM*1e-3

        def near(points, hull_pts):
            try:
                E = ConvexHull(hull_pts).equations
            except Exception:
                return True                                   # degenerate hull: count it (conservative)
            return bool(((points @ E[:, :3].T + E[:, 3]).max(1) <= m).any())
        job_near = lambda V: near(pts, V) or any(near(Rv, V) or near(V, Rv) for Rv in rigid)
        stock = None
        for b in links:
            V = world(b)
            if job_near(V):
                return True
            # one hop: a link touching a stock body (a lower tote, a board still in its
            # pile) that is itself within the margin of an item of the job pushes on the job through it
            if stock is None:
                stock = [world(i) for n, i in self.body.items() if n.startswith("stack_bag_") or (
                    n.startswith("board_") and np.linalg.norm(ipc.body_pose(i)[0] - self.initial[n]) <= BOARD_TAKEN_MM*1e-3)]
            if any((near(Sv, V) or near(V, Sv)) and job_near(Sv) for Sv in stock):
                return True
        return False

    def _bag_x(self):
        ipc = self.world.ipc
        return ipc.x_free()[ipc.sheet_slices[0]]

    def _mouth(self, x):
        """(area of the rim's convex hull in the plane normal to the bag axis over W x G, the unit axis bottom
        centroid -> rim centroid)."""
        rim, base = x[self.rim], x[self.bottom_v]
        axis = rim.mean(0) - base.mean(0)
        axis /= max(np.linalg.norm(axis), 1e-12)
        e1 = np.cross(axis, [1., 0., 0.]) if abs(axis[0]) < .9 else np.cross(axis, [0., 1., 0.])
        e1 /= np.linalg.norm(e1); e2 = np.cross(axis, e1)
        P = np.c_[rim @ e1, rim @ e2]
        try:
            area = ConvexHull(P).volume
        except Exception:
            area = 0.
        return area/(self.W*self.G), axis

    def _above_bottom(self, x, M, c0, pe1, pe2, n_in):
        """For each point of M: does the tote's bottom panel lie beneath it, all of it below the point? Beneath means
        along n_in, the panel's normal toward the mouth: a bottom triangle whose projection on the panel's plane (pe1,
        pe2) holds the point's. False where no bottom paper is beneath the point or some of it is above."""
        P = np.asarray(x)[np.asarray(self.bag.tris)[self.bottom_tris]] - c0      # (m, 3 corners, 3)
        A, h = np.stack([P @ pe1, P @ pe2], -1), P @ n_in                       # (m, 3, 2), (m, 3)
        q, hq = np.c_[(M - c0) @ pe1, (M - c0) @ pe2], (M - c0) @ n_in
        v0, v1 = A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]
        den = v0[:, 0]*v1[:, 1] - v1[:, 0]*v0[:, 1]
        good = np.abs(den) > 1e-12                       # a triangle standing on edge in projection covers nothing
        den = np.where(good, den, 1.)
        w = q[:, None, :] - A[None, :, 0]                # (k, m, 2)
        s = (w[..., 0]*v1[:, 1] - v1[:, 0]*w[..., 1])/den
        t = (v0[:, 0]*w[..., 1] - w[..., 0]*v0[:, 1])/den
        under = good & (s >= -1e-9) & (t >= -1e-9) & (s + t <= 1. + 1e-9)
        top = np.where(under, h[:, 0] + s*(h[:, 1] - h[:, 0]) + t*(h[:, 2] - h[:, 0]), -np.inf).max(1)
        return np.isfinite(top) & (hq > top)

    def _robot_contact(self):
        ipc = self.world.ipc
        return any(np.linalg.norm(F) > 1e-3 for b, (F, _) in ipc.reactions.items()
                   if ipc.bodies[b].name.startswith("robot:"))

    # ------------------------------------------------------------------ per-step record
    def record(self):
        ipc = self.world.ipc
        row = dict(t=self.world.sim_time, contact=self._robot_contact())
        if self.task == "bag-purchase" and row["contact"]:
            row["contact_any"], row["contact"] = True, self._job_contact()
        peaks = {}
        if self.world.scene.preset == "counter":
            x = self._bag_x()
            row["base"] = x[self.bottom_v].mean(0)
        if "gift_base" in self.body:
            pb, Rb = ipc.body_pose(self.body["gift_base"])
            low = self._lowest_z(self.body["gift_base"])
            peaks["base_lowest_lift_mm"] = 1e3*(low - self.geo_table_z())
            peaks["base_lift_mm"] = 1e3*(self._bottom_high_z(self.body["gift_base"]) - self.geo_table_z())
            peaks["base_tilt_deg"] = _tilt_deg(Rb)
            row["base"] = pb
        if self.world.scene.preset == "receiving":
            from .carton import flap_angles
            row["flaps"] = flap_angles(ipc)
        self.history.add(row, peaks)

    def geo_table_z(self):
        return self.world.scene.surface_z

    def _bottom_high_z(self, b_i):
        """Highest point of a body's bottom face (its lowest vertices in the body frame), in the world."""
        ipc = self.world.ipc
        sl = ipc.body_vslice[b_i]
        if not hasattr(self, "_bottom_cache"):
            self._bottom_cache = {}
        if b_i not in self._bottom_cache:
            r = np.asarray(ipc.bodies[b_i].verts)
            self._bottom_cache[b_i] = np.flatnonzero(r[:, 2] <= r[:, 2].min() + 1e-6)
        xc = (ipc.P @ ipc.q + ipc.cvec).reshape(-1, 3)[sl]
        return float(xc[self._bottom_cache[b_i], 2].max()) - ipc.caliper["deflate"]

    def _lowest_z(self, b_i):
        ipc = self.world.ipc
        sl = ipc.body_vslice[b_i]
        xc = (ipc.P @ ipc.q + ipc.cvec).reshape(-1, 3)[sl]
        return float(xc[:, 2].min()) - ipc.caliper["deflate"]

    # ------------------------------------------------------------------ evaluation
    def evaluate(self):
        checks, metrics = {}, {}
        rows = list(self.history.rows)
        complete = len(rows) >= self.history.n
        checks["hands_off_window"] = complete and not any(r["contact"] for r in rows)
        metrics["hands_off_window_s"] = len(rows)*self.env.config.dt
        if self.task in ("open-bag", "bag-purchase"):
            self._eval_bag(checks, metrics, rows)
            self._eval_tear(checks, metrics, 0, "bag", TEAR_TOL_MM)
        elif self.task == "lift-lid":
            self._eval_lid(checks, metrics)
        elif self.task in ("unwrap-gift", "open-present"):
            self._eval_unwrap(checks, metrics)
            if self.task == "open-present":
                self._eval_lid(checks, metrics, frame_out=True)
        elif self.task == "open-carton":
            self._eval_carton(checks, metrics, rows)
        checks = {k: bool(v) for k, v in checks.items()}
        success = bool(checks) and all(checks.values())
        score = float(np.mean([bool(v) for v in checks.values()])) if checks else 0.
        if self.task == "bag-purchase":
            # 1 on success, else FAILED_SCORE_FACTOR x progress: every failure below every success, every success 1.
            # The mean of the checks is also reported.
            metrics["mean_of_checks"] = score
            progress = self._bag_progress(checks, metrics)
            score = 1. if success else FAILED_SCORE_FACTOR*progress
        return dict(success=success, score=score, checks=checks, metrics=metrics)

    def _bag_progress(self, checks, metrics):
        """bag-purchase progress in [0, 1], read from the judged state only: how much of the instructed end state
        exists. A third each for the bag (open; standing on the mat, counted as far as it is open; bottom squared on
        what it rests on), the board (inside; flat on the bottom) and the candle (inside; upright; resting on a board in
        the bag), times the tote's integrity. Each part is continuous in a quantity the checks measure, from its
        start-state value to its check's threshold, so a state that passes every check of an object gets all of its
        third; the bag's openness counts whichever way the bag lies, everything else only with the bag's mouth up. No
        term reads time, order, path, grasp, arm or which board, and none pays for something not having gone wrong (the
        stack, the absence of contact). Equal weights are an authored choice [A]. "Opened" is the success rule's
        one "open" (mouth hull and least rim gap), not the gusset creases."""
        ramp = lambda v, a, b: float(min(1., max(0., (v - a)/(b - a))))
        up = ramp(90. - metrics["bag_tilt_signed_deg"], 0., 90. - TILT_MAX_DEG)      # 0 on its side or upside down
        # opened (one "open" with success): the mouth hull ratio and the least rim gap, each from its
        # start-state value to its limit, the smaller counts (a mouth folded or pinched shut is not open)
        m0 = min(float(getattr(self, "mouth0", 0.)), MOUTH_RATIO_MIN/2)
        g_lim = RIM_GAP_MIN_FRAC*self.G
        g0 = min(float(getattr(self, "rim_gap0", None) or 0.), g_lim/2)
        gap = metrics.get("face_gap_min_mm")
        opened = min(ramp(metrics["mouth_ratio"], m0, MOUTH_RATIO_MIN), 1. if gap is None else ramp(1e-3*gap, g0, g_lim))
        placed = up*float(checks["on_mat"] and metrics["bag_lowest_above_mat_mm"] <= REST_MM)
        # squared: mouth up and resting on what it stands on (mat or counter), bottom flat against it
        rests = up*float(metrics["bag_lowest_above_support_mm"] <= REST_MM)
        squared = rests*ramp(metrics["bottom_flat_fraction_on_support"], 0., BOTTOM_FLAT_FRACTION)*ramp(
            90. - metrics["bottom_tilt_deg"], 0., 90. - TILT_MAX_DEG)
        # standing on the mat counts as far as the bag is open: a closed bag set on its end is not yet "standing
        # open"
        bag = (opened + opened*placed + squared)/3.
        board = up*metrics["board_credit"]
        c_in = metrics["candle_corners_inside"]/8.
        c_up = ramp(90. - metrics["candle_tilt_deg"], 0., 90. - CANDLE_TILT_DEG)
        c_on = metrics["candle_on_board_credit"]          # = 1 exactly when the check passes
        candle = up*c_in*(1. + c_up + c_up*c_on)/3.
        crack = metrics.get("bag_new_crack_length_mm")
        integrity = 1. if crack is None else 1. - ramp(crack, TEAR_TOL_MM, TEAR_ZERO_MM)
        progress = integrity*(bag + board + candle)/3.
        metrics["progress"] = progress
        metrics["reward"] = dict(bag=bag, opened=opened, standing_on_mat=placed, resting_upright=rests, squared=squared, board=board,
                                 candle=candle, candle_inside=c_in, candle_upright=c_up, candle_on_board=c_on,
                                 mouth_up=up, integrity=integrity)
        return progress

    def _eval_bag(self, checks, metrics, rows):
        ipc = self.world.ipc
        x = self._bag_x()
        purchase = self.task == "bag-purchase"
        base = x[self.bottom_v]
        metrics["mouth_ratio"], axis = self._mouth(x)
        checks["mouth_open"] = metrics["mouth_ratio"] >= MOUTH_RATIO_MIN
        metrics["bag_tilt_deg"] = math.degrees(math.acos(min(1., abs(axis[2]))))
        if purchase:
            # one definition of "open": the mouth, and not pinched shut at its middle (the
            # rim's convex hull spans its four corners whatever the faces do between them)
            gap = self._rim_gap(x)
            metrics["face_gap_min_mm"] = None if gap is None else 1e3*gap
            checks["mouth_open"] = checks["mouth_open"] and (gap is None or gap >= RIM_GAP_MIN_FRAC*self.G)
            # signed: an upside-down bag is not standing
            metrics["bag_tilt_signed_deg"] = math.degrees(math.acos(max(-1., min(1., float(axis[2])))))
            metrics["bag_lowest_above_mat_mm"] = 1e3*(float(x[:, 2].min()) - self.geo["mat"]["center"][2])
        checks["standing"] = metrics["bag_tilt_signed_deg" if purchase else "bag_tilt_deg"] <= TILT_MAX_DEG
        mat = self.geo["mat"]
        zc = mat["center"][2] + ipc.caliper["rest_sheet_rigid"]
        T = np.asarray(self.bag.tris)[self.bottom_tris]
        cen = x[T].mean(1)
        n = np.cross(x[T[:, 1]] - x[T[:, 0]], x[T[:, 2]] - x[T[:, 0]])
        a = .5*np.linalg.norm(n, axis=1)
        flat = np.abs(cen[:, 2] - zc) <= BOTTOM_FLAT_MM*1e-3
        metrics["bottom_flat_fraction"] = float(a[flat].sum()/a.sum())
        nm = n.sum(0); nm /= max(np.linalg.norm(nm), 1e-12)
        metrics["bottom_tilt_deg"] = math.degrees(math.acos(min(1., abs(nm[2]))))
        checks["bottom_squared"] = metrics["bottom_flat_fraction"] >= BOTTOM_FLAT_FRACTION and metrics["bottom_tilt_deg"] <= TILT_MAX_DEG
        bc = base.mean(0)
        hx, hy = mat["half"]
        checks["on_mat"] = abs(bc[0] - mat["center"][0]) <= hx and abs(bc[1] - mat["center"][1]) <= hy
        if purchase:
            # progress only: the bottom read against what the bag stands on, the mat or,
            # off it, the counter, and whether it rests there
            sz = mat["center"][2] if checks["on_mat"] else self.world.scene.surface_z
            ok = np.abs(cen[:, 2] - sz - ipc.caliper["rest_sheet_rigid"]) <= BOTTOM_FLAT_MM*1e-3
            metrics["bottom_flat_fraction_on_support"] = float(a[ok].sum()/a.sum())
            metrics["bag_lowest_above_support_mm"] = 1e3*(float(x[:, 2].min()) - sz)
        from .ipcworld import hinge_angles
        for s, edges in self.gusset_edges.items():
            opening, length = [], []
            for e in edges:
                ts = self.edge_tris.get(e, [])
                if len(ts) != 2:
                    continue
                t1, t2 = np.asarray(self.bag.tris)[ts]
                n1 = np.cross(x[t1[1]] - x[t1[0]], x[t1[2]] - x[t1[0]]); n2 = np.cross(x[t2[1]] - x[t2[0]], x[t2[2]] - x[t2[0]])
                c = float(n1 @ n2/max(np.linalg.norm(n1)*np.linalg.norm(n2), 1e-30))
                opening.append(180. - math.degrees(math.acos(max(-1., min(1., c)))))
                length.append(np.linalg.norm(x[e[0]] - x[e[1]]))
            opening, length = np.asarray(opening), np.asarray(length)
            frac = float(length[opening >= GUSSET_OPEN_DEG].sum()/max(length.sum(), 1e-12)) if len(length) else 0.
            side = "left" if s < 0 else "right"
            metrics[f"gusset_{side}_open_fraction"] = frac
            metrics[f"gusset_{side}_median_opening_deg"] = float(np.median(opening)) if len(opening) else 0.
            if not purchase:
                # bag-purchase reports the gussets only (a second, disagreeing definition of "open"; with the
                # carton seated the body is forced open where it stands, the mouth judges the rest)
                checks[f"gusset_{side}_deployed"] = frac >= GUSSET_LENGTH_FRACTION
        if rows and all("base" in r for r in rows):
            b = np.asarray([r["base"] for r in rows])
            metrics["base_drift_window_mm"] = 1e3*float(np.linalg.norm(b[-1] - b[0]))
            checks["base_at_rest"] = metrics["base_drift_window_mm"] <= BASE_DRIFT_MM
            if purchase:
                # "left standing" is judged on the window's settled part, its last 2 s, at the same speed limit (2 mm
                # per 3 s), so a bag that shifts once as the carton lands at the start of a window the evaluation
                # completed from the release, then stands still, passes as it does when the agent waits; a bag still
                # sliding fails
                k = max(1, len(b) - int(round(2./max(float(self.env.config.dt), 1e-9))))
                metrics["base_drift_settled_mm"] = 1e3*float(np.linalg.norm(b[-1] - b[min(k, len(b) - 1)]))
                checks["base_at_rest"] = metrics["base_drift_settled_mm"] <= BASE_DRIFT_MM*2./HANDS_OFF_S
        else:
            checks["base_at_rest"] = False
        # board flat on the bottom panel: sampled board underside within BOARD_MM of the panel's plane and over it.
        # bag-purchase also asks that the board be INSIDE, on the panel's inner side (the unsigned distance and the 2D
        # footprint cannot tell the sides apart: a board lying on the mat under the bag would pass): at every counted
        # sample the bottom panel's paper is beneath the board's mid-plane, none of it above.
        # The board is 0.5 mm thick and the contact caliper keeps paper 0.06-0.09 mm off its faces, so the mid-plane is
        # >= 0.3 mm above the paper when the board lies on it and >= 0.3 mm below it when the bag stands on the board.
        boards = [n for n in self.body if n.startswith("board_")]
        hx_, hy_, hz_ = self.geo["board"]["half"]
        u, v = np.meshgrid(np.linspace(-hx_, hx_, 9), np.linspace(-hy_, hy_, 5))
        grid = np.c_[u.ravel(), v.ravel()]
        frac, best, n_in = {}, 0., None
        if len(base) >= 3:
            c0 = base.mean(0)
            _, _, Vt = np.linalg.svd(base - c0)
            npl = Vt[2]
            pe1, pe2 = Vt[0], Vt[1]
            hull = Delaunay(np.c_[(base - c0) @ pe1, (base - c0) @ pe2])
            n_in = npl if npl @ axis >= 0. else -npl
            for nb in boards:
                p, R = ipc.body_pose(self.body[nb])
                S = p + (np.c_[grid, np.full(len(grid), -hz_)] @ R.T)
                d = np.abs((S - c0) @ npl)
                inside = hull.find_simplex(np.c_[(S - c0) @ pe1, (S - c0) @ pe2]) >= 0
                ok = (d <= BOARD_MM*1e-3) & inside
                if purchase:
                    ok &= self._above_bottom(x, p + np.c_[grid, np.zeros(len(grid))] @ R.T, c0, pe1, pe2, n_in)
                frac[nb] = float(ok.mean())
                best = max(best, frac[nb])
        metrics["board_on_bottom_fraction"] = best
        checks["board_in_bag"] = best >= BOARD_FRACTION
        moved = [1e3*np.linalg.norm(ipc.body_pose(self.body[n])[0] - self.initial[n]) for n in self.body if n.startswith("stack_bag_")]
        metrics["stack_max_move_mm"] = float(max(moved)) if moved else 0.
        if not purchase:
            # bag-purchase reports the move only (neither its instruction nor the job limits how far taking the top
            # bag may shift the stock under it)
            checks["stack_undisturbed"] = metrics["stack_max_move_mm"] <= STACK_MOVE_MM
        if purchase:
            tote = Delaunay(x)
            p, R = ipc.body_pose(self.body["candle"])
            hc = np.asarray(self.geo["candle"]["half"], float)
            corners = np.array([[sx*hc[0], sy*hc[1], sz*hc[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) @ R.T + p
            inside = tote.find_simplex(corners) >= 0
            metrics["candle_corners_inside"] = int(inside.sum())
            # signed: the carton's +z is the up of its print (scene.counter starts it with
            # R = I; render.py prints it upright on that axis), so a carton standing on its lid reads 180 deg
            metrics["candle_tilt_deg"] = math.degrees(math.acos(max(-1., min(1., float(R[2, 2])))))
            checks["candle_in_bag"] = bool(inside.all())
            checks["candle_upright"] = metrics["candle_tilt_deg"] <= CANDLE_TILT_DEG
            # "upright on the board": its lowest
            # corner at most CANDLE_REST_MM above the top face of a board that passes the board rule, its centre over
            # that board. Not centred: anywhere on the board counts, and any board in the bag (two stacked: the top one)
            gap = None
            for nb in boards:
                if frac.get(nb, 0.) < BOARD_FRACTION:
                    continue
                pb, Rb = ipc.body_pose(self.body[nb])
                nt = Rb[:, 2] if Rb[:, 2] @ n_in >= 0. else -Rb[:, 2]
                rel = Rb.T @ (p - pb)
                if abs(rel[0]) > hx_ or abs(rel[1]) > hy_:
                    continue
                g = float(((corners - pb) @ nt).min()) - hz_
                gap = g if gap is None or abs(g) < abs(gap) else gap
            metrics["candle_above_board_mm"] = None if gap is None else 1e3*gap
            checks["candle_on_board"] = gap is not None and -1e-3 <= gap <= CANDLE_REST_MM*1e-3
            # progress credit of each board: 1 when it passes the board rule, else half for its share inside the tote
            # (its mid-plane samples in the tote's convex hull) and half for its share counted by the board rule
            bc = {}
            for nb in boards:
                if frac.get(nb, 0.) >= BOARD_FRACTION:
                    bc[nb] = 1.
                    continue
                pb, Rb = ipc.body_pose(self.body[nb])
                share = float((tote.find_simplex(pb + np.c_[grid, np.zeros(len(grid))] @ Rb.T) >= 0).mean())
                bc[nb] = .5*share + .5*frac.get(nb, 0.)/BOARD_FRACTION
            metrics["board_credit"] = max(bc.values(), default=0.)
            # progress's "on the board": resting (the check's band) on ANY board, weighted by
            # that board's credit, so a board a little short of the board rule is not charged twice
            on = 0.
            for nb in boards:
                pb, Rb = ipc.body_pose(self.body[nb])
                rel = Rb.T @ (p - pb)
                if n_in is None or abs(rel[0]) > hx_ or abs(rel[1]) > hy_:
                    continue
                nt = Rb[:, 2] if Rb[:, 2] @ n_in >= 0. else -Rb[:, 2]
                g = float(((corners - pb) @ nt).min()) - hz_
                if -1e-3 <= g <= CANDLE_REST_MM*1e-3:
                    on = max(on, bc[nb])
            metrics["candle_on_board_credit"] = on

    def _eval_lid(self, checks, metrics, frame_out=False):
        ipc = self.world.ipc
        g = self.geo["box"]
        pb, Rb = ipc.body_pose(self.body["gift_base"])
        pl, Rl = ipc.body_pose(self.body["gift_lid"])
        base_org = pb - Rb @ ipc.bodies[self.body["gift_base"]].render["com_offset"]
        lid_org = pl - Rl @ ipc.bodies[self.body["gift_lid"]].render["com_offset"]
        overlap = g["base_h"] - float((Rb.T @ (lid_org - base_org))[2])
        metrics["lid_overlap_mm"] = 1e3*overlap
        rel = Rb.T @ (lid_org - base_org)
        over_base = abs(rel[0]) < g["hx"] + .005 and abs(rel[1]) < g["hy"] + .005
        checks["lid_off"] = overlap <= 0. or not over_base
        lid_low = self._lowest_z(self.body["gift_lid"]) - self.world.scene.surface_z
        metrics["lid_lowest_above_table_mm"] = 1e3*lid_low
        checks["lid_on_table"] = lid_low <= .005
        metrics["episode_base_lift_mm"] = float(self.history.peak.get("base_lift_mm", 0.))
        metrics["episode_base_lowest_lift_mm"] = float(self.history.peak.get("base_lowest_lift_mm", 0.))
        metrics["episode_base_tilt_deg"] = float(self.history.peak.get("base_tilt_deg", 0.))
        if frame_out:
            # open-present: the present may be picked up and turned while it is unwrapped (people do; the seam tape is
            # underneath), so the base is judged where it ends, not over the episode (the instruction does not
            # ask for the episode rule here). The episode peaks stay in the metrics.
            metrics["base_tilt_deg"] = _tilt_deg(Rb)
            metrics["base_lowest_above_table_mm"] = 1e3*(self._lowest_z(self.body["gift_base"]) - self.world.scene.surface_z)
            checks["base_upright_on_table"] = (metrics["base_tilt_deg"] <= LID_BASE_TILT_DEG
                                               and metrics["base_lowest_above_table_mm"] <= LID_BASE_LIFT_MM)
        else:
            checks["base_not_lifted"] = metrics["episode_base_lift_mm"] <= LID_BASE_LIFT_MM
            checks["base_not_tipped"] = metrics["episode_base_tilt_deg"] <= LID_BASE_TILT_DEG
        pf, Rf = ipc.body_pose(self.body["photo_frame"])
        relf = Rb.T @ (pf - base_org)
        in_base = abs(relf[0]) < g["hx"] and abs(relf[1]) < g["hy"] and 0. < relf[2] < g["base_h"]
        if frame_out:
            fl = self._lowest_z(self.body["photo_frame"]) - self.world.scene.surface_z
            metrics["frame_lowest_above_table_mm"] = 1e3*fl
            checks["frame_out_on_table"] = (not in_base) and fl <= .005
        else:
            checks["frame_in_base"] = bool(in_base)

    def _eval_unwrap(self, checks, metrics):
        ipc = self.world.ipc
        xs = ipc.x_free()
        pts = []
        for k, sh in enumerate(ipc.sheets):
            T = np.asarray(sh.tris)
            x = xs[ipc.sheet_slices[k]]
            tri_off = int(np.searchsorted(ipc.sheet_of_tri, k))
            alive = ipc.damage[tri_off:tri_off + len(T)] < 1. if len(ipc.damage) else np.ones(len(T), bool)
            A, B, C = x[T[alive, 0]], x[T[alive, 1]], x[T[alive, 2]]
            for u, v in ((1/3, 1/3), (2/3, 1/6), (1/6, 2/3), (1/6, 1/6)):
                pts.append(A + u*(B - A) + v*(C - A))
        paper = cKDTree(np.vstack(pts)) if pts else None
        samples = []
        for name in ("gift_lid", "gift_base"):
            b = ipc.bodies[self.body[name]]
            p, R = ipc.body_pose(self.body[name])
            for half, c in b.render["parts"]:
                hx_, hy_, hz_ = half
                u, v = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7))
                for ax in range(3):
                    for sgn in (-1, 1):
                        P = np.zeros((u.size, 3))
                        others = [i for i in range(3) if i != ax]
                        P[:, ax] = sgn*half[ax]
                        P[:, others[0]] = u.ravel()*half[others[0]]
                        P[:, others[1]] = v.ravel()*half[others[1]]
                        samples.append((P + np.asarray(c) - b.render["com_offset"]) @ R.T + p)
        S = np.vstack(samples)
        # keep the box's OUTER surface: points at least 1 mm outside the convex hull shrunk inward
        allv = S
        hull = Delaunay(allv)
        cen = allv.mean(0)
        outer = hull.find_simplex(cen + (S - cen)*1.01) < 0
        S = S[outer]
        if paper is not None and len(S):
            d, _ = paper.query(S)
            metrics["box_coverage"] = float((d <= COVER_MM*1e-3).mean())
        else:
            metrics["box_coverage"] = 0.
        checks["unwrapped"] = metrics["box_coverage"] <= COVERAGE_MAX
        if self.task == "unwrap-gift":
            g = self.geo["box"]
            pb, Rb = ipc.body_pose(self.body["gift_base"])
            pl, Rl = ipc.body_pose(self.body["gift_lid"])
            base_org = pb - Rb @ ipc.bodies[self.body["gift_base"]].render["com_offset"]
            lid_org = pl - Rl @ ipc.bodies[self.body["gift_lid"]].render["com_offset"]
            overlap = g["base_h"] - float((Rb.T @ (lid_org - base_org))[2])
            metrics["lid_overlap_mm"] = 1e3*overlap
            checks["box_closed"] = overlap*1e3 >= LID_CLOSED_OVERLAP_MM
            metrics["box_tilt_deg"] = _tilt_deg(Rb)
            checks["box_upright"] = metrics["box_tilt_deg"] <= 15.
        # fragments: connected components of intact wrap triangles; large torn-off pieces left on the box fail
        metrics["damaged_triangles"] = int((ipc.damage >= 1.).sum()) if len(ipc.damage) else 0
        self._eval_tear(checks, metrics, 0, "wrap", None)                # tearing the wrap is allowed: reported only

    def _eval_tear(self, checks, metrics, k, name, tol_mm):
        """New crack length of sheet k since the episode start (rest length, mm): split crack edges (vertex splitting),
        each weighted by the dissipated fraction of its cohesive bond's fracture energy (a split whose bond never opened
        is intact paper and counts 0; IPCWorld.crack_length), plus the crack band's fully torn triangles. With `tol_mm`,
        the check `<name>_not_torn` fails above it."""
        ipc = self.world.ipc
        if not hasattr(ipc, "crack_length") or k >= len(ipc.sheets):
            return
        split, band = ipc.crack_length(k)
        new = 1e3*(split + band - self.crack0.get(k, 0.))
        metrics[f"{name}_new_crack_length_mm"] = new
        metrics[f"{name}_split_crack_mm"], metrics[f"{name}_band_crack_mm"] = 1e3*split, 1e3*band
        if ipc.stats.get("tear_pool_exhausted"):
            metrics["tear_pool_exhausted"] = int(ipc.stats["tear_pool_exhausted"])
        if tol_mm is not None:
            checks[f"{name}_not_torn"] = new <= tol_mm

    def _eval_carton(self, checks, metrics, rows):
        """open-carton, from simulated state only: tape bond damage along the seam, flap angles from
        the rigid poses over the hands-off window, carton and contents displacement and containment."""
        from . import carton as CT
        ipc = self.world.ipc
        st = CT.tape_status(ipc)
        metrics.update({k: v for k, v in st.items()})
        checks["seam_open"] = st.get("seam_joined_fraction", 1.) <= SEAM_JOINED_MAX
        for f in CT.FLAPS:
            a = [r["flaps"][f] for r in rows if "flaps" in r]
            metrics[f"{f}_min_window_deg"] = float(min(a)) if a else None
            metrics[f"{f}_final_deg"] = float(a[-1]) if a else None
            checks[f"{f}_open"] = bool(a) and len(a) == len(rows) and min(a) >= FLAP_OPEN_DEG
        pc, Rc = ipc.body_pose(self.body["carton_body"])
        metrics["carton_move_mm"] = 1e3*float(np.linalg.norm(pc - self.initial["carton_body"]))
        metrics["carton_tilt_deg"] = _tilt_deg(Rc)
        checks["carton_in_place"] = metrics["carton_move_mm"] <= CARTON_MOVE_MM
        checks["carton_upright"] = metrics["carton_tilt_deg"] <= CARTON_TILT_DEG
        oc = pc - Rc @ ipc.bodies[self.body["carton_body"]].render["com_offset"]
        moved, outside = {}, []
        for n in CT.CONTENTS:
            p, _ = ipc.body_pose(self.body[n])
            moved[n] = 1e3*float(np.linalg.norm(p - self.initial[n]))
            r = Rc.T @ (p - oc)
            if not (abs(r[0]) < CT.L_IN/2 and abs(r[1]) < CT.W_IN/2 and 2*CT.T_B < r[2] < CT.Z_END):
                outside.append(n)
        metrics["contents_max_move_mm"] = max(moved.values())
        metrics["contents_outside"] = outside
        checks["contents_in_place"] = metrics["contents_max_move_mm"] <= CONTENTS_MOVE_MM and not outside
