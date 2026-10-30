"""Task instructions and the outcome evaluator.

The opt-in classroom-tasks-v1 contract dispatches to task_set.py. The checks in this file
and the group-work evaluator stay the default, unchanged.

The evaluator reads only the simulated world: furniture body poses, glide heights above the floor, contacts, the
1 s pose history of the watched items (rest window) and the floor occupancy. It never reads actions, the robot's
commands or any task flag. Thresholds are authored (params.EVAL) with the practice they come from:
  on_marks    every glide within 30 mm of its floor dot (a 19 mm dot hides under a 28.6 mm glide up to ~24 mm), yaw
              within 3 deg of the marked orientation;
  tucked      seat front >= 0.30 m under the desk top (within ~0.12 m of the end stop), lateral offset <= 0.08 m, yaw within 10 deg of the desk's;
  in_use      chair centre 0.35-0.55 m behind its desk centre, lateral <= 0.08 m, facing the desk within 15 deg;
  ready       seat front 0.20-0.35 m from the table edge it faces, facing it within 15 deg (2 on the room side,
              1 at each end);
  up_on       chair inverted (tilt >= 175 deg) on its own desk, its lowest point at most 0.40 m below the desk top, centre of mass >=
              0.05 m inside the top's edges, the desk upright on its dots;
  pod         facing desks' front edges <= 10 mm apart and not overlapping, lateral offset <= 30 mm;
  exam        every pair of the row's chair centres >= 1.25 m apart (JCQ 11.10);
  route       a 0.915 m wide path (ADA 2010 Standards 403.5.1) on a 2 cm floor-occupancy grid from the door to the back;
and for every checked item: upright (tilt <= 3 deg, all glides on the floor within 1 mm), at rest (moved < 2 mm in
the last 1 s), nothing in the door swing, no desk or chair on its side or upside down unless that is its target.

group-work scores places, not named furniture (Evaluator.by_place; params.EVAL_GROUP_WORK): the room's 24 identical student desks and its identical chairs are matched one-to-one to the places
that need one, so a finished room passes whichever desks were turned and whichever chairs went where:
  slot        a pod place holds a desk: every glide within 30 mm of the place's dot, yaw within 3 deg of its facing;
  seat        a chair at the desk in that place, on the outside of the pod, facing it within 15 deg, lateral <= 0.08 m,
              its seat front from 0.15 m under the top edge to 0.35 m beyond it, and no other desk it stands behind and
              faces is nearer;
  pod         the closest approach of the two facing front edges -2..+10 mm (contact anywhere along the edges), their
              widest separation <= 41.9 mm (10 mm + the desk width x sin 3 deg), centres within 60 mm sideways (the
              two desks' 30 mm dot tolerances);
  as_found    every other place holds its chair within 0.10 m and 10 deg, its desk, table or teacher's desk within
              0.06 m and 5 deg, upright; the tote and wastebasket within 0.20 m, tilted <= 10 deg;
and score = 1 on success, else 0.9 x (progress - deduction) x penalties (by_place's docstring), not the fraction of
criteria met.
"""

import math
import numpy as np
from . import params as P
from . import layout as L
from . import furniture as F
from .scene import chair_up_pose

E = P.EVAL
FOOT = {"desk": (P.DESK_TOP[0]/2, P.DESK_TOP[1]/2), "chair": (P.CHAIR_OVERALL[0]/2, 0.25),
        "table": (P.TABLE_TOP[0]/2, P.TABLE_TOP[1]/2), "teacher_desk": (P.TEACHER_TOP[0]/2, P.TEACHER_TOP[1]/2),
        "tote": (P.TOTE[0]/2, P.TOTE[1]/2), "basket": (P.BASKET[0], P.BASKET[0])}


def instruction(config, spec):
    from .catalogue import CONTRACT, description
    text = description(config.task, CONTRACT) if getattr(config, "contract", "v1") == CONTRACT else spec["instruction"]
    return text + (" The coloured floor dots mark where each desk's feet go." if spec["dots"] else "")


def _ang(a):
    return abs((a + math.pi) % (2*math.pi) - math.pi)


def to_local(pose, x, y):
    px, py, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = x - px, y - py
    return c*dx + s*dy, -s*dx + c*dy


# group-work's distance scale: each pod desk's move from its row onto its pod dots, 0.471 m (no credit beyond it)
REACH = (L.ROWS_Y[1] - L.ROWS_Y[0])/2 - P.DESK_TOP[1]/2


def front_edge(pose):
    """The two ends of a student desk's front (closed, board-facing at yaw 0) top edge, sorted by x."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = P.DESK_TOP[0]/2, P.DESK_TOP[1]/2
    return sorted((x + c*a + s*hy, y + s*a - c*hy) for a in (-hx, hx))


def edge_span(front, back):
    """(closest, widest) separation along y of the front edges of a pod's board-side desk (pose `front`, turned to face
    +y) and the desk behind it (`back`, facing the board), over the stretch of x both edges span (the edges are straight,
    so both extremes lie at the ends of that stretch). Negative where they overlap; None if they share no stretch of x."""
    F, B = front_edge(front), front_edge(back)
    lo, hi = max(F[0][0], B[0][0]), min(F[1][0], B[1][0])
    if hi - lo < 1e-6:
        return None

    def at(e, x):
        (x0, y0), (x1, y1) = e
        return y0 + (y1 - y0)*(x - x0)/(x1 - x0)
    ends = [at(B, x) - at(F, x) for x in (lo, hi)]
    return min(ends), max(ends)


def edge_gap(front, back):
    """The closest approach of edge_span: contact anywhere along the edges counts."""
    span = edge_span(front, back)
    return None if span is None else span[0]


class Evaluator:
    def __init__(self, env):
        self.env = env
        self.spec = env.spec
        w = env.world
        self._glide_geoms = {}
        for n, it in w.scene.items.items():
            gl = it["glides"]
            if gl:
                self._glide_geoms[n] = np.array([w._shape_geom[s] for s in gl], int)
        self._inverted_start = {n for n, it in self.spec["items"].items() if it.get("inverted_on")}
        from .catalogue import CONTRACT
        if getattr(env.config, "contract", "v1") == CONTRACT:
            from .task_set import prepare_start
            prepare_start(self)

    # ------------------------------------------------------------------ geometry helpers
    def pose(self, n):
        return self.env.world.pose2d(n)

    def glides_xyz(self, n):
        return self.env.world.mjd.geom_xpos[self._glide_geoms[n]].copy()

    def upright(self, n):
        (_, _, _), tilt = self.pose(n)
        g = self._glide_geoms.get(n)
        on_floor = True
        if g is not None:
            half = self.env.world.mjm.geom_size[g, 1]
            on_floor = bool((self.glides_xyz(n)[:, 2] - half <= 0.001).all())
        return tilt <= E["upright_deg"] and on_floor, dict(tilt_deg=round(tilt, 2), glides_on_floor=on_floor)

    # ------------------------------------------------------------------ criteria
    def on_marks(self, n):
        target = self.spec["targets"][n]
        dots = self.spec["dots"][n]
        g = self.glides_xyz(n)[:, :2]
        err = [float(np.hypot(*(g[k] - np.asarray(dots[k])))) for k in range(len(dots))]
        (x, y, yaw), _ = self.pose(n)
        dyaw = math.degrees(_ang(yaw - target[2]))
        ok = max(err) <= E["glide_tol"] and dyaw <= E["yaw_tol_deg"]
        return ok, dict(worst_glide_mm=round(max(err)*1e3, 1), yaw_error_deg=round(dyaw, 2))

    def chair_rel(self, chair, desk):
        (cx, cy, cyaw), _ = self.pose(chair)
        (dx, dy, dyaw), _ = self.pose(desk)
        lx, ly = to_local((dx, dy, dyaw), cx, cy)
        return lx, ly, math.degrees(_ang(cyaw - dyaw))

    def tucked(self, chair, desk):
        lx, ly, dyaw = self.chair_rel(chair, desk)
        depth = F.DESK_EDGE_Y - (ly + F.CHAIR_SEAT_FRONT_Y)
        ok = depth >= E["tuck_depth"] and abs(lx) <= E["tuck_lateral"] and dyaw <= E["tuck_yaw_deg"]
        return ok, dict(seat_under_top_m=round(depth, 3), lateral_m=round(lx, 3), yaw_error_deg=round(dyaw, 2))

    def in_use(self, chair, desk):
        lx, ly, dyaw = self.chair_rel(chair, desk)
        lo, hi = E["use_range"]
        ok = lo <= ly <= hi and abs(lx) <= E["tuck_lateral"] and dyaw <= E["facing_deg"]
        return ok, dict(behind_m=round(ly, 3), lateral_m=round(lx, 3), yaw_error_deg=round(dyaw, 2))

    def behind(self, chair, desk, rng):
        lx, ly, _ = self.chair_rel(chair, desk)
        (cx, cy, cyaw), _ = self.pose(chair)
        (dx, dy, dyaw), _ = self.pose(desk)
        facing = math.degrees(_ang(cyaw - dyaw - math.pi))       # sitter faces the desk's -y side ... from behind
        facing = min(facing, math.degrees(_ang(cyaw - dyaw)))
        ok = rng[0] <= -ly <= rng[1] and abs(lx) <= 0.40 and facing <= E["facing_deg"]
        return ok, dict(behind_m=round(-ly, 3), lateral_m=round(lx, 3), yaw_error_deg=round(facing, 2))

    def ready(self, chair, table):
        (tx, ty, tyaw), _ = self.pose(table)
        (cx, cy, cyaw), _ = self.pose(chair)
        hx, hy = P.TABLE_TOP[0]/2, P.TABLE_TOP[1]/2
        sit = (math.sin(cyaw), -math.cos(cyaw))           # the sitter's facing direction (chair -y)
        sf = (cx + sit[0]*-F.CHAIR_SEAT_FRONT_Y, cy + sit[1]*-F.CHAIR_SEAT_FRONT_Y)
        lx, ly = to_local((tx, ty, tyaw), *sf)
        lsx, lsy = math.cos(tyaw)*sit[0] + math.sin(tyaw)*sit[1], -math.sin(tyaw)*sit[0] + math.cos(tyaw)*sit[1]
        sides = {"room": (-ly - hy, (0., 1.), abs(lx) <= hx - 0.2), "back": (ly - hy, (0., -1.), abs(lx) <= hx - 0.2),
                 "left_end": (-lx - hx, (1., 0.), abs(ly) <= hy - 0.15), "right_end": (lx - hx, (-1., 0.), abs(ly) <= hy - 0.15)}
        best = None
        for side, (dist, want, along) in sides.items():
            face = math.degrees(math.acos(max(-1., min(1., lsx*want[0] + lsy*want[1]))))
            ok = E["ready_range"][0] <= dist <= E["ready_range"][1] and face <= E["facing_deg"] and along
            cand = (ok, side, round(dist, 3), round(face, 2))
            if best is None or (ok and not best[0]) or (ok == best[0] and abs(dist) < abs(best[2])):
                best = cand
        return best[0], dict(side=best[1], seat_front_from_edge_m=best[2], facing_error_deg=best[3])

    def up_on(self, chair, desk):
        w = self.env.world
        (_, _, _), tilt = self.pose(chair)
        body = w.scene.bodies[chair]
        com = w.state.body_q.numpy()[body, :3] + 0.  # body origin; COM offset below
        p, R = w.body_pose(chair)
        c = p + R @ np.asarray(w.model.body_com.numpy()[body], float)
        (dx, dy, dyaw), dtilt = self.pose(desk)
        lx, ly = to_local((dx, dy, dyaw), c[0], c[1])
        inside = min(P.DESK_TOP[0]/2 - abs(lx), P.DESK_TOP[1]/2 - abs(ly))
        lowest = self._lowest(chair)
        # backrest over the student side (the chair's +y axis along the desk's +y), as the instruction says
        dside = np.array([-math.sin(dyaw), math.cos(dyaw)])
        cy = R[:2, 1]/max(1e-9, float(np.linalg.norm(R[:2, 1])))
        side_deg = math.degrees(math.acos(float(np.clip(cy @ dside, -1., 1.))))
        ok = tilt >= E["inverted_deg"] and inside >= E["inside_edge"] and lowest >= P.DESK_HEIGHT - 0.40 \
            and side_deg <= E["up_side_deg"]
        floor = self._touches_floor(chair)
        ok = ok and not floor
        return ok, dict(tilt_deg=round(tilt, 1), com_inside_edge_m=round(inside, 3), touches_floor=floor,
                        backrest_side_error_deg=round(side_deg, 1))

    def _lowest(self, n):
        w = self.env.world
        g = [w._shape_geom[s] for s in w.scene.items[n]["shapes"]]
        return float(w.mjd.geom_xpos[g, 2].min())

    def _touches_floor(self, n):
        w = self.env.world
        g = set(w._shape_geom[s] for s in w.scene.items[n]["shapes"])
        floor = w._shape_geom[0]
        d = w.mjd
        for i in range(d.ncon):
            a, b = d.contact[i].geom
            if (a in g and b == floor) or (b in g and a == floor):
                return True
        return False

    def footprint(self, n):
        (x, y, yaw), _ = self.pose(n)
        hx, hy = FOOT[self.env.spec["items"][n]["kind"]]
        c, s = math.cos(yaw), math.sin(yaw)
        return [(x + c*a - s*b, y + s*a + c*b) for a, b in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]

    def route(self, start, goal, width=None):
        """Accessible route: 4-connected components on a 2 cm grid of the floor, a cell free when it is at least half the route width
        (0.4575 m, less half a cell) from every item's plan footprint, the walls, bookcase and cubbies; True if the
        goal is reachable from the start. `width` (default the 0.915 m AISLE_MIN) tests another width the same way."""
        res = 0.02
        half = (P.AISLE_MIN if width is None else width)/2 - res/2
        nx, ny = int(P.ROOM_X/res), int(P.ROOM_Y/res)
        xs = -P.ROOM_X/2 + (np.arange(nx) + .5)*res
        ys = (np.arange(ny) + .5)*res
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        free = (np.abs(X) <= P.ROOM_X/2 - half) & (Y >= half) & (Y <= P.ROOM_Y - half)

        def block(cx, cy, c, s, hx, hy):
            # each footprint (plus the half-route margin) only touches the grid cells in its bounding window
            r = math.hypot(hx, hy) + half
            i0, i1 = max(0, int((cx - r + P.ROOM_X/2)/res)), min(nx, int((cx + r + P.ROOM_X/2)/res) + 2)
            j0, j1 = max(0, int((cy - r)/res)), min(ny, int((cy + r)/res) + 2)
            if i0 >= i1 or j0 >= j1:
                return
            Xw, Yw = X[i0:i1, j0:j1], Y[i0:i1, j0:j1]
            lx, ly = c*(Xw - cx) + s*(Yw - cy), -s*(Xw - cx) + c*(Yw - cy)
            ex, ey = np.maximum(np.abs(lx) - hx, 0), np.maximum(np.abs(ly) - hy, 0)
            free[i0:i1, j0:j1] &= np.hypot(ex, ey) > half
        for n, it in self.env.spec["items"].items():
            if it["kind"] not in FOOT or n.startswith("book"):
                continue
            (x, y, yaw), _ = self.pose(n)
            hx, hy = FOOT[it["kind"]]
            block(x, y, math.cos(yaw), math.sin(yaw), hx, hy)
        for name, cc, h, _ in L.STATIC:
            if name in ("bookcase", "cubbies"):
                block(cc[0], cc[1], 1., 0., h[0], h[1])
        si = (int((start[0] + P.ROOM_X/2)/res), int(start[1]/res))
        gi = (int((goal[0] + P.ROOM_X/2)/res), int(goal[1]/res))
        if not free[si] or not free[gi]:
            return False
        # 4-connected components (the same neighbourhood as a breadth-first search, ~10x faster)
        from scipy import ndimage
        lab, _ = ndimage.label(free)
        return bool(lab[si] == lab[gi])

    def route_width(self, start, goal, floor, steps=5):
        """The widest route width in [floor, AISLE_MIN] that route() still finds (bisection, (AISLE_MIN - floor)/2^steps
        resolution, rounded down), or `floor` if even that width is closed. Only called when the AISLE_MIN route fails."""
        if not self.route(start, goal, floor):
            return floor
        lo, hi = floor, P.AISLE_MIN
        for _ in range(steps):
            mid = (lo + hi)/2
            lo, hi = (mid, hi) if self.route(start, goal, mid) else (lo, mid)
        return lo

    def in_door_zone(self, n):
        (xr, yr) = L.DOOR_ZONE
        return any(xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1] for x, y in self.footprint(n))

    # ------------------------------------------------------------------ evaluation
    def evaluate(self):
        spec = self.spec
        motion, window = self.env.window_motion()
        from .catalogue import CONTRACT
        if getattr(self.env.config, "contract", "v1") == CONTRACT:
            from .task_set import CatalogueEvaluation
            return CatalogueEvaluation(self, motion, window).evaluate()
        if any(isinstance(c, tuple) and c[0] == "slot" for c in spec["check"].values()):
            return self.by_place(motion, window)
        criteria = {}
        for n, chk in spec["check"].items():
            if n == "route" or n.startswith("pod") or n == "exam":
                continue
            kind = chk if isinstance(chk, str) else chk[0]
            if kind == "on_marks":
                ok, m = self.on_marks(n)
            elif kind == "unmoved":
                (x, y, yaw), _ = self.pose(n)
                x0, y0, yaw0 = spec["items"][n]["pose"]
                d = math.hypot(x - x0, y - y0)
                ok, m = d <= 0.05 and math.degrees(_ang(yaw - yaw0)) <= 5, dict(moved_m=round(d, 3))
            elif kind == "tucked":
                ok, m = self.tucked(n, chk[1])
            elif kind == "in_use":
                ok, m = self.in_use(n, chk[1])
            elif kind == "ready":
                ok, m = self.ready(n, chk[1])
            elif kind == "behind":
                ok, m = self.behind(n, chk[1], chk[2])
            elif kind == "up_on":
                ok, m = self.up_on(n, chk[1])
            elif kind == "in_zone":
                (x, y, _), tilt = self.pose(n)
                (xr, yr) = chk[1]
                ok, m = xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1] and tilt <= 10, dict(x=round(x, 3), y=round(y, 3))
            elif kind == "near":
                (x, y, _), tilt = self.pose(n)
                d = math.hypot(x - chk[1][0], y - chk[1][1])
                ok, m = d <= chk[2] and tilt <= 10, dict(distance_m=round(d, 3))
            else:
                raise ValueError(kind)
            if kind not in ("up_on", "in_zone", "near"):
                up, um = self.upright(n)
                m.update(um)
                ok = ok and up
            rest = motion.get(n, math.inf) <= E["rest_motion"] and window
            m["at_rest"] = bool(rest)
            criteria[n] = dict(check=kind, ok=bool(ok and rest), **m)
        if "sides" not in spec and any(c["check"] == "ready" for c in criteria.values()):
            sides = [c["side"] for c in criteria.values() if c["check"] == "ready" and c["ok"]]
            ok = sides.count("room") == 2 and sides.count("left_end") == 1 and sides.count("right_end") == 1
            criteria["seating_plan"] = dict(check="seating_plan", ok=ok, sides=sorted(sides))
        for n, chk in spec["check"].items():
            if n.startswith("pod"):
                criteria[n] = self.pod(chk)
            elif n == "exam":
                pts = [self.pose(c)[0][:2] for c in chk]
                dmin = min(math.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(pts) for b in pts[i + 1:])
                criteria[n] = dict(check="exam_spacing", ok=dmin >= E["exam_pitch"] - 1e-9, min_chair_spacing_m=round(dmin, 3))
            elif n == "route":
                criteria[n] = dict(check="route", ok=self.route(*chk))
        criteria["no_toppled_or_blocking"] = self.safety()
        oks = [c["ok"] for c in criteria.values()]
        return {"succeeded": all(oks), "score": round(sum(oks)/len(oks), 4), "criteria": criteria,
                "window_complete": bool(window), "note": "world state only (poses, glide heights, contacts, 1 s rest window)"}

    def safety(self):
        """Never success: furniture on its side / upside down (other than chairs-up targets), anything in the door swing."""
        spec = self.spec
        bad = []
        for n, it in spec["items"].items():
            if it["kind"] in ("desk", "chair", "table", "teacher_desk"):
                (_, _, _), tilt = self.pose(n)
                target_up = isinstance(spec["check"].get(n), tuple) and spec["check"][n][0] == "up_on"
                if tilt > 30 and not target_up and not (n in self._inverted_start and tilt > 150):
                    bad.append(f"{n} tipped ({tilt:.0f} deg)")
            if it["kind"] in FOOT and self.in_door_zone(n):
                bad.append(f"{n} in the door swing")
        return dict(check="safety", ok=not bad, problems=bad[:10])

    # ------------------------------------------------------------------ group-work: places, not named furniture
    def by_place(self, motion, window):
        """group-work. Each place that needs a desk (12 pod slots, plus the 12 places of rows 3-4) and each place that
        needs a chair (12 pod seats, plus where every other chair stood) is matched one-to-one to whichever identical
        desk or chair stands there: the assignment with the most credit (scipy linear_sum_assignment), desks first, then
        chairs at the desks matched to the slots. Success needs every criterion. The score is 1 on success, else
        fail_scale x max(0, progress - deduction) x penalties (constants in params.EVAL_GROUP_WORK), where
          progress  = 0.60 x mean desk-place credit + 0.15 x mean front-edge-meeting credit (6) + 0.25 x mean chair
                      credit. A desk's position credit rises linearly as its worst glide approaches the place's dot,
                      from 0 at REACH (0.471 m: an untouched row desk, or a desk on the dots facing the wrong way) to 1
                      within 30 mm. At the six board-side places, whose facing differs from every desk's start facing,
                      the place credit is 0.3 x a turn credit (the desk faces the place's way: 0 at 90 deg or more off,
                      1 within 3 deg, wherever the desk stands, upright) + 0.7 x (0.7 x the position credit + 0.3 x
                      the turn credit x an approach credit falling linearly from 1 at 30 mm to 0 at 2.5 m between the
                      desk's centre and the place's); elsewhere it is the position credit. A meeting's credit is the
                      smaller of its two desks' position credits times a
                      fall-off over REACH of how far the edges are outside the pod's bounds (closest gap -2..+10 mm,
                      widest gap, sideways offset). A chair's is its desk's position credit times a fall-off over half
                      the chair's depth outside the seat window (and over 75 deg beyond the 15 deg facing), 0 unless
                      that desk is the nearest one the chair stands behind and faces (not a chair still pulled out at
                      another desk in between). Near credit needs no rest window (it reads the poses); criteria do.
          deduction = 0.05 (one desk place's share of progress) per whole place of the rest of the room not as
                      found, each place charged from 0 at its tolerance to 1 at 0.30 m / 45 deg (tote, wastebasket:
                      0.20 m to 0.50 m; knocked over or empty: 1), the same at any stage of the work;
          penalties = route: x1 at 0.915 m falling linearly to x0.5 at 0.815 m, the widest route width that still
                      connects the door to the back (only measured when 0.915 m fails); x0.5 for anything toppled or in
                      the door swing:
        the untouched room scores 0, and a constraint already met at the start earns nothing."""
        from scipy.optimize import linear_sum_assignment
        spec, G = self.spec, P.EVAL_GROUP_WORK
        chk, items, targets = spec["check"], spec["items"], spec["targets"]
        slots = [n for n, c in chk.items() if isinstance(c, tuple) and c[0] == "slot"]
        seats = [n for n, c in chk.items() if isinstance(c, tuple) and c[0] == "seat"]
        kind = {n: it["kind"] for n, it in items.items()}
        desks = sorted(n for n in items if kind[n] == "desk")
        chairs = sorted(n for n in items if kind[n] == "chair")
        homes = [n for n in items if n not in targets]             # every place the pods do not use, as found
        desk_homes = [n for n in homes if kind[n] == "desk"]
        chair_homes = [n for n in homes if kind[n] == "chair"]
        loose = [n for n in homes if kind[n] not in ("desk", "chair")]
        pose, tilt, up, rest = {}, {}, {}, {}
        for n in desks + chairs + loose:
            pose[n], tilt[n] = self.pose(n)
            up[n] = self.upright(n)[0]
            rest[n] = motion.get(n, math.inf) <= E["rest_motion"] and window
        standing = {n: tilt[n] <= G["toppled_deg"] for n in pose}
        tol = E["glide_tol"]

        def as_found(n, home):
            """1 when the item stands within the place's tolerance, upright; falling linearly to 0 at as_found_full."""
            (x, y, yaw), (x0, y0, yaw0) = pose[n], items[home]["pose"]
            d, dyaw = math.hypot(x - x0, y - y0), math.degrees(_ang(yaw - yaw0))
            (d0, a0) = G["as_found"] if kind[home] == "chair" else G["as_found_furniture"]
            (d1, a1) = G["as_found_full"]
            return up[n]*max(0., 1. - max((d - d0)/(d1 - d0), (dyaw - a0)/(a1 - a0), 0.))

        # desks -> 12 slots + 12 desk homes
        g = np.stack([self.glides_xyz(n)[:, :2] for n in desks])                         # (24, 4, 2)
        dots = np.array([spec["dots"][chk[s][1]] for s in slots])                       # (12, 4, 2)
        err = np.linalg.norm(g[:, None] - dots[None], axis=3).max(axis=2)               # worst glide, (24, 12)
        dyaw = np.array([[math.degrees(_ang(pose[n][2] - targets[chk[s][1]][2])) for s in slots] for n in desks])
        up_ = np.array([standing[n] for n in desks], float)[:, None]
        near = np.clip((REACH - err)/(REACH - tol), 0., 1.)*up_                         # position credit
        # board-side places: their facing differs from the start facing of the desk whose target they are (all desks
        # start facing the board), so a desk facing their way earns turn credit wherever it stands
        turned = np.array([_ang(targets[chk[s][1]][2] - items[chk[s][1]]["pose"][2]) > math.radians(E["yaw_tol_deg"])
                           for s in slots])
        turn = np.clip((G["turn_zero_deg"] - dyaw)/(G["turn_zero_deg"] - E["yaw_tol_deg"]), 0., 1.)*up_
        # a desk so turned also earns approach credit as its centre comes within approach_radius of the place
        centre = np.array([pose[n][:2] for n in desks])
        place = np.array([targets[chk[s][1]][:2] for s in slots])
        dist = np.linalg.norm(centre[:, None] - place[None], axis=2)
        approach = turn*np.clip((G["approach_radius"] - dist)/(G["approach_radius"] - tol), 0., 1.)
        a, b = G["turn_share"], G["approach_share"]
        credit = np.where(turned[None, :], a*turn + (1. - a)*((1. - b)*near + b*approach), near)
        V = np.hstack([credit, [[as_found(n, h) for h in desk_homes] for n in desks]])
        rows, cols = linear_sum_assignment(V, maximize=True)
        at_place = {c: r for r, c in zip(rows, cols)}
        criteria, slot_desk, slot_credit, slot_pos, out = {}, {}, {}, {}, {}
        for k, s in enumerate(slots):
            i = at_place.get(k)
            c = float(V[i, k]) if i is not None else 0.
            n = desks[i] if c > 0 else None
            slot_desk[s], slot_credit[s], slot_pos[s] = n, c, float(near[i, k]) if n else 0.
            if n is None:
                j = int(np.argmin(err[:, k]))
                criteria[s] = dict(check="slot", ok=False, desk=None, credit=0., closest_desk=desks[j],
                                   closest_worst_glide_mm=round(float(err[j, k])*1e3, 1))
                continue
            ok = err[i, k] <= tol and dyaw[i, k] <= E["yaw_tol_deg"] and up[n] and rest[n]
            criteria[s] = dict(check="slot", ok=bool(ok), desk=n, worst_glide_mm=round(float(err[i, k])*1e3, 1),
                               yaw_error_deg=round(float(dyaw[i, k]), 2), tilt_deg=round(tilt[n], 2), upright=up[n],
                               at_rest=bool(rest[n]), credit=round(c, 4), position_credit=round(slot_pos[s], 4),
                               turn_credit=round(float(turn[i, k]), 4) if turned[k] else None)
        for k, h in enumerate(desk_homes):
            i = at_place.get(len(slots) + k)
            charge = 1. - (float(V[i, len(slots) + k]) if i is not None else 0.)
            if charge > 0:
                out[f"{h}'s place"] = round(charge, 4)

        # chairs -> 12 seats (at the desks now in the slots) + every other chair's place
        lo = F.DESK_EDGE_Y - F.CHAIR_SEAT_FRONT_Y + G["seat_front"][0]                 # chair centre behind the desk
        hi = F.DESK_EDGE_Y - F.CHAIR_SEAT_FRONT_Y + G["seat_front"][1]
        nearest = {}                     # the nearest desk the chair stands behind (student side) and faces
        for j in chairs:
            rel = []
            for d in desks:
                lx, ly = to_local(pose[d], *pose[j][:2])
                if ly > 0 and _ang(pose[j][2] - pose[d][2]) < math.pi/2:
                    rel.append((math.hypot(lx, ly), d))
            nearest[j] = min(rel)[1] if rel else None
        fit = {}
        for j in chairs:
            for s in seats:
                d = slot_desk[chk[s][1]]
                if d is None:
                    fit[j, s] = (0., None)
                    continue
                lx, ly = to_local(pose[d], *pose[j][:2])
                face = math.degrees(_ang(pose[j][2] - pose[d][2]))
                off = math.hypot(max(lo - ly, ly - hi, 0.), max(abs(lx) - E["tuck_lateral"], 0.))
                over = max(face - E["facing_deg"], 0.)
                w = (max(0., 1. - off/G["chair_reach"])*max(0., 1. - over/(90. - E["facing_deg"]))*standing[j]
                     *(nearest[j] == d)*slot_pos[chk[s][1]])
                fit[j, s] = (w, dict(desk=d, behind_m=round(ly, 3), lateral_m=round(lx, 3), yaw_error_deg=round(face, 2),
                                     seat_front_beyond_edge_m=round(ly - (F.DESK_EDGE_Y - F.CHAIR_SEAT_FRONT_Y), 3),
                                     nearest_desk=nearest[j], window=off == 0. and over == 0.))
        W = np.hstack([[[fit[j, s][0] for s in seats] for j in chairs],
                       [[as_found(j, h) for h in chair_homes] for j in chairs]])
        rows, cols = linear_sum_assignment(W, maximize=True)
        at_place = {c: r for r, c in zip(rows, cols)}
        seat_credit = {}
        for k, s in enumerate(seats):
            i = at_place.get(k)
            w, m = fit[chairs[i], s] if i is not None else (0., None)
            seat_credit[s] = w
            if m is None or (w == 0. and not m["window"]):
                criteria[s] = dict(check="seat", ok=False, chair=None, desk=slot_desk[chk[s][1]], credit=0.)
                continue
            j = chairs[i]
            ok = m["window"] and m["nearest_desk"] == m["desk"] and up[j] and rest[j]
            criteria[s] = dict(check="seat", ok=bool(ok), chair=j, **m, tilt_deg=round(tilt[j], 2), upright=up[j],
                               at_rest=bool(rest[j]), credit=round(w, 4))
        for k, h in enumerate(chair_homes):
            i = at_place.get(len(seats) + k)
            charge = 1. - (float(W[i, len(seats) + k]) if i is not None else 0.)
            if charge > 0:
                out[f"{h}'s place"] = round(charge, 4)

        # pods: each board-side slot and the slot behind it meet front edge to front edge: the closest approach of the
        # edges -2..+10 mm (contact anywhere), the widest no more than pod_widest, sideways within pod_lateral
        meet = []
        for p, places in ((p, v) for p, v in chk.items() if p.startswith("pod")):
            pairs, ok = {}, True
            for f, b in ((places[0], places[2]), (places[1], places[3])):
                df, db = slot_desk[f], slot_desk[b]
                span = edge_span(pose[df], pose[db]) if df and db else None
                if span is None:
                    gap = wide = lat = None
                    good, c = False, 0.
                else:
                    (gap, wide), lat = span, abs(pose[db][0] - pose[df][0])
                    good = -0.002 <= gap <= E["pod_gap"] and wide <= G["pod_widest"] and lat <= G["pod_lateral"]
                    excess = (max(0., gap - E["pod_gap"], -0.002 - gap) + max(0., wide - G["pod_widest"])
                              + max(0., lat - G["pod_lateral"]))
                    c = min(slot_pos[f], slot_pos[b])*max(0., 1. - excess/REACH)
                meet.append(c)
                ok &= good
                pairs[f"{f}|{b}"] = dict(desks=[df, db], gap_mm=None if gap is None else round(gap*1e3, 1),
                                         widest_mm=None if wide is None else round(wide*1e3, 1),
                                         lateral_mm=None if lat is None else round(lat*1e3, 1), credit=round(c, 4))
            criteria[p] = dict(check="pod", ok=bool(ok), pairs=pairs)

        # the rest of the room as found; the route; nothing toppled or in the door swing
        for n in loose:
            if kind[n] in ("tote", "basket"):
                (x, y, _), (x0, y0, _) = pose[n], items[n]["pose"]
                d0, d1 = G["loose_as_found"][0], G["loose_full"]
                d = math.hypot(x - x0, y - y0)
                value = (tilt[n] <= G["loose_as_found"][1])*max(0., 1. - max((d - d0)/(d1 - d0), 0.))
            else:
                value = as_found(n, n)
            if value < 1:
                out[n] = round(1. - value, 4)
        criteria["rest_of_room"] = dict(check="as_found", ok=not out, out_of_place=sorted(out), charges=out,
                                        places=len(desk_homes) + len(chair_homes) + len(loose))
        criteria["route"] = dict(check="route", ok=self.route(*chk["route"]))
        if not criteria["route"]["ok"]:
            criteria["route"]["widest_route_m"] = round(self.route_width(*chk["route"], G["route_floor"]), 4)
        criteria["no_toppled_or_blocking"] = self.safety()

        w = G["weights"]
        progress = (w["desks"]*float(np.mean(list(slot_credit.values()))) + w["abutment"]*float(np.mean(meet))
                    + w["chairs"]*float(np.mean(list(seat_credit.values()))))
        pen = G["penalty"]
        narrow = 0. if criteria["route"]["ok"] else \
            min(1., (P.AISLE_MIN - criteria["route"]["widest_route_m"])/(P.AISLE_MIN - G["route_floor"]))
        penalty = ((1. - (1. - pen["route"])*narrow)
                   *(pen["safety"] if not criteria["no_toppled_or_blocking"]["ok"] else 1.))
        deduction = G["place_charge"]*sum(out.values())
        succeeded = all(c["ok"] for c in criteria.values())
        score = 1. if succeeded else round(G["fail_scale"]*max(0., progress - deduction)*penalty, 4)
        return {"succeeded": succeeded, "score": score, "progress": round(progress, 4),
                "reward": dict(rule="1 if success else fail_scale x max(0, progress - rest_of_room_deduction) x penalty",
                               fail_scale=G["fail_scale"], rest_of_room_deduction=round(deduction, 4),
                               weights=dict(w), desk_places=round(float(np.mean(list(slot_credit.values()))), 4),
                               front_edge_meetings=round(float(np.mean(meet)), 4),
                               chairs=round(float(np.mean(list(seat_credit.values()))), 4), penalty=round(penalty, 4),
                               penalty_parts=dict(route=round(1. - (1. - pen["route"])*narrow, 4),
                                                  safety=pen["safety"] if not criteria["no_toppled_or_blocking"]["ok"] else 1.)),
                "criteria": criteria, "window_complete": bool(window),
                "note": "world state only (poses, glide heights, contacts, 1 s rest window); desks and chairs matched to "
                        "places, not by name"}

    def pod(self, desks):
        a1, a2, b1, b2 = desks                 # row-1 pair, row-2 pair (same columns)
        out, ok = {}, True
        for f, b in ((a1, b1), (a2, b2)):
            (fx, fy, _), _ = self.pose(f)
            (bx, by, _), _ = self.pose(b)
            gap = (by - P.DESK_TOP[1]/2) - (fy + P.DESK_TOP[1]/2)
            lat = abs(bx - fx)
            good = -0.002 <= gap <= E["pod_gap"] and lat <= E["pod_lateral"]
            ok &= good
            out[f"{f}|{b}"] = dict(gap_mm=round(gap*1e3, 1), lateral_mm=round(lat*1e3, 1))
        return dict(check="pod", ok=bool(ok), pairs=out)
