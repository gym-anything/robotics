"""Task instructions and evaluators.

Every criterion reads physical state, the host app's record, or the episode history the environment keeps
(env.history: the scanner's longest steady hold out of its cradle, each free body's highest rise), never commanded
targets:
  record       the host app's record equals ground truth (counts under the counted location / location / sale /
               check-in);
  attribution  each physical unit's own symbol read exactly once, no inner-pack or ISBN reads, no read of another
               slot's location label (scan-location), and no read that no printed label explains (privileged ray
               attribution of each decode, env._attribute);
  stock        the counted slot and the rest of the shelf by count-by-scanning's stock rules below (count-key-quantity
               too); nothing on the floor (any free body -- stock, containers, cases, the scanner --
               whose lowest point is within 15 mm of it);
  containers   bagged / returned items inside the tote / bin volume (between its floor and 50 mm above its rim), the
               container upright on its counter;
  tool         scanner seated in its cradle (seat sensor) and not held (no finger in contact), no key pressed; for
               hold-scanner a steady hold (< 20 mm/s, < 0.25 rad/s) out of the cradle for 5 s; for checkout-bulky
               the case never lifted (rise < 10 mm).
Thresholds are authored. A wrong outcome is a valid result; a non-finite state
is a backend fault (world.fault), never an evaluation.

Every task scores 1 when every check passes, else 0.9 x progress, and reports its scoring contract
(evaluation["contract"], CONTRACTS). The seven tasks other than count-by-scanning take their progress from scoring.py.
count-by-scanning has its own checks and score:
  bookings     only reads the host booked into a count are judged (rep["booked_location"]): a read the host refused
               ("scan location first") and a location-label read book nothing. The slot's own count is judged, not
               which location label was read last. A misread fails only if the record holds data from it: a booked
               misread, or a misread location label (rep["sets_location"]) whose code a booking was made under before
               any correct read of that code in the same stretch (consecutive location reads setting one code). A
               stray misread of the code already correctly active changes nothing and is not judged;
  stock        each counted unit in the slot (+-20 mm), resting on the deck (lowest point within 5 mm of the deck top:
               not held above it, not stacked), upright (tilt <= 15 deg) and faced: front within +-30 deg of the
               aisle, a can or bottle on its label front like a box on its front panel; each facing lane fronted:
               some unit of the lane's own product, standing in the lane's found x extent, has its front face at most
               40 mm behind where the lane's front face was found (rows and identical units may be swapped). Every
               other shelf unit and shipping case as found: in its own slot
               (+-20 mm), its lowest point within 5 mm of where it was found, upright (<= 15 deg), turned at most 30 deg
               from its found heading or facing within 30 deg of the aisle; and every other slot's facing lanes fronted
               by the same rule;
  score        success = every check. Otherwise score = 0.9 x progress, progress = max(0, hits - errors - 0.5 x
               disturbed) / N, x 0.9 while the scanner is not seated and released or a key is down; N = the slot's
               units; hits = counted units whose own symbol was booked under the counted location, decoded correctly;
               errors = every other booking (a unit booked again, another label, a misread, anything booked under
               another location), plus one per unbacked misread stretch of the counted slot's code; disturbed = units
               and cases failing the stock rules, other slots' units in the counted slot, free bodies on the floor
               (an unfronted lane counts once unless one of its units already does). The location read, aiming,
               lifting or turning a unit earn nothing by themselves, and no order of units, hand, trigger pull or
               hands-free presentation is preferred. Reading the score along an episode: each unit off its place
               costs 0.5/N of progress until it is back, so an honest trace dips at every pick-up and can sit at 0
               with units already booked (the floor at 0); metrics hits / errors / disturbed report the parts.
               Mid-episode the scanner factor scores a handheld partial count 10 % below the same count presented to
               the docked scanner; final scores agree.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation
from . import params as P
from . import scoring as S
from .catalog import SKUS
from .scoring import DISTURBED_COST, NOT_RETURNED, FAILED     # count-by-scanning's score constants (module docstring)

INSTRUCTIONS = {
    "hold-scanner": "Take the barcode scanner out of its cradle on the cart tray, hold it steady for 5 seconds, "
                    "then put it back in the cradle.",
    "scan-location": "Take the scanner from its cradle, scan the shelf location label for slot {slot} (the label "
                     "text is printed above its barcode on the shelf beam), then return the scanner to its cradle.",
    "count-by-scanning": "Cycle count slot {slot} (a slot runs from its location label to the next label on the "
                         "right): scan the slot's location label, then scan every unit in the slot exactly once (the "
                         "barcodes are on the back or side of the units). Leave the stock faced on the shelf as you "
                         "found it and return the scanner to its cradle.",
    "count-key-quantity": "Cycle count slot {slot} (a slot runs from its location label to the next label on the "
                          "right): scan the slot's location label, then for each different item scan one unit and key "
                          "the number of units of that item in the slot on the keypad, then ENTER. Leave the stock "
                          "faced on the shelf as you found it and return the scanner to its cradle.",
    "checkout-present": "Ring up the customer's basket: present each item's barcode to the scanner standing in its "
                        "cradle so each is scanned once, and put each scanned item in the bagging tote.",
    "checkout-bulky": "Ring up the basket as usual, and scan the case of water at the end of the counter with the "
                      "handheld scanner without lifting the case; return the scanner to its cradle.",
    "checkout-counter": "Ring up the customer's items set out on the counter: present each item's barcode to the "
                        "scanner standing in its cradle so each is scanned once, and put each scanned item in the "
                        "bagging tote.",
    "library-checkin": "Check in the returned books: scan each book's library barcode (the library's own label, "
                       "not the ISBN; it is on the back cover or inside the front cover) at the desk scanner and put "
                       "each book in the returns bin.",
}


def instruction(config, meta):
    return INSTRUCTIONS[config.task].format(slot=(meta or {}).get("target_slot") or "")


# count-by-scanning: authored tolerances and score constants (module docstring)
SLOT_TOL_M = .02        # a unit's centre within its slot +-20 mm
REST_TOL_M = .005       # its lowest point within 5 mm of the deck it stands on (resting, not held above it or stacked)
UPRIGHT_DEG = 15.
FACED_DEG = 30.         # front panel / label front within 30 deg of the aisle (cans spawn within about 26 deg, 3 sigma)
FRONTED_TOL_M = .04     # a facing lane's front-most unit at most 40 mm behind where the lane's front face was found
TURNED_DEG = 30.        # stock outside the counted slot turned at most 30 deg from its found heading (or faced, FACED_DEG)
ISBN_FORMATS = ("EAN13", "UPCA")    # library-checkin: a book's ISBN symbologies (its library label is Codabar)

# The scoring contract each task's evaluation reports (evaluation["contract"]).
CONTRACTS = {
    "count-by-scanning": "count-by-scanning-v1",
    "hold-scanner": "hold-scanner-v1",
    "scan-location": "scan-location-v1",
    "count-key-quantity": "count-key-quantity-v1",
    "checkout-present": "checkout-present-v1",
    "checkout-bulky": "checkout-bulky-v1",
    "checkout-counter": "checkout-counter-v1",
    "library-checkin": "library-checkin-v1",
}


class Evaluator:
    def __init__(self, env):
        self.env = env
        self.world = env.world
        self.meta = getattr(self.world.scene, "task_meta", {}) or {}
        q = self.world.state.body_q.numpy()
        self.initial = {n: q[it["body"]].copy() for n, it in self.world.scene.items.items()}
        # each free body's collision shapes as oriented boxes in its own frame (lowest point, container volumes)
        m = self.world.model
        sb, st, ss, sx = (a.numpy() for a in (m.shape_body, m.shape_type, m.shape_scale, m.shape_transform))
        bodies = {it["body"] for it in self.world.scene.items.values()} | {self.world.scene.bodies["scanner"]}
        self.corners = {}
        import newton
        for i in range(m.shape_count):
            b = int(sb[i])
            if b not in bodies:
                continue
            if int(st[i]) == int(newton.GeoType.BOX):
                h = np.asarray(ss[i], float)
                loc = np.array([[a*h[0], c*h[1], e*h[2]] for a in (-1, 1) for c in (-1, 1) for e in (-1, 1)])
            else:
                src = m.shape_source[i]
                if src is None or not hasattr(src, "vertices"):
                    continue
                loc = np.asarray(src.vertices, float)*np.asarray(ss[i], float)
            tf = sx[i]
            pts = Rotation.from_quat(tf[3:7]).apply(loc) + tf[:3]
            self.corners.setdefault(b, []).append(pts)
        self.corners = {b: np.vstack(v) for b, v in self.corners.items()}
        robots = set(self.world.scene.robot_bodies)
        labels = list(m.body_label)
        self.fingers = [b for b in robots if any(k in labels[b] for k in ("finger", "hand"))]

    def _lowest_z(self, qrow, body):
        pts = self.corners.get(body)
        if pts is None:
            return float(qrow[2])
        return float((Rotation.from_quat(qrow[3:]).apply(pts)[:, 2] + qrow[2]).min())

    def _in_container(self, q, cname, body):
        """Body inside the container's volume: within its walls, its lowest point above the container floor's
        underside, its centre at most 50 mm above the rim; the container upright (tilt <= 20 deg)."""
        c = self.world.scene.items.get(cname)
        if not c:
            return False
        cq = q[c["body"]]
        L, W, H = c["container_size"]
        Rc = Rotation.from_quat(cq[3:])
        if self._tilt_deg(cq) > 20.:
            return False
        rel = Rc.inv().apply(q[body, :3] - cq[:3])
        return bool(abs(rel[0]) < L/2 and abs(rel[1]) < W/2 and rel[2] < H/2 + .05
                    and self._lowest_z(q[body], body) >= cq[2] - H/2 - .002)

    def _container_on_counter(self, q, cname):
        c = self.world.scene.items.get(cname)
        top = self.world.scene.fixtures.get("counter", {}).get("top")
        if not c or top is None:
            return True
        return abs(self._lowest_z(q[c["body"]], c["body"]) - top) < .02

    def _scanner_held(self):
        """A finger or hand body presses the scanner or its trigger with more than 0.2 N (one pair)."""
        sc = self.world.scene
        forces = self.world.pair_contact_forces(self.fingers, (sc.bodies["scanner"], sc.bodies["scanner_trigger"]))
        return any(f > .2 for f in forces.values())

    def _tilt_deg(self, qrow):
        z = Rotation.from_quat(qrow[3:]).apply((0, 0, 1))
        return math.degrees(math.acos(max(-1., min(1., z[2]))))

    def _front_dev_deg(self, qrow, sku):
        """Deviation of the unit's front normal from the aisle direction (-y); cylinders use their label front."""
        f = Rotation.from_quat(qrow[3:]).apply((0, -1, 0))
        f[2] = 0
        if np.linalg.norm(f) < 1e-6:
            return 180.
        f /= np.linalg.norm(f)
        return math.degrees(math.acos(max(-1., min(1., -f[1]))))

    @staticmethod
    def _turn_deg(q0, q1):
        """Angle between two poses' horizontal front directions (body -y)."""
        f = [Rotation.from_quat(r[3:]).apply((0, -1, 0))[:2] for r in (q0, q1)]
        n = [float(np.linalg.norm(v)) for v in f]
        if min(n) < 1e-6:
            return 180.
        return math.degrees(math.acos(max(-1., min(1., float(f[0] @ f[1])/(n[0]*n[1])))))

    @staticmethod
    def _half_extent(qrow, sku, axis):
        """Half extent of a unit's hull along world axis `axis` (0: x, 1: y) in its pose."""
        R = Rotation.from_quat(qrow[3:]).as_matrix()
        if sku.shape == "box":
            return float(np.abs(R[axis]) @ (np.asarray(sku.dims)/2))
        r, h = sku.dims
        a = float(R[axis, 2])
        return abs(a)*h/2 + r*math.sqrt(max(0., 1 - a*a))

    def _lanes(self, units):
        """One slot's facing lanes as found: each facing's product, the x extent of its units and its front face's y
        (from the reset poses, so computed once)."""
        cache = self.__dict__.setdefault("_lane_cache", {})
        if tuple(units) in cache:
            return cache[tuple(units)]
        sc = self.world.scene
        lanes = {}
        for n in units:
            it, sku, row = sc.items[n], SKUS[sc.items[n]["sku"]], self.initial[n]
            hx = self._half_extent(row, sku, 0)
            front = float(row[1]) - self._half_extent(row, sku, 1)
            lane = lanes.setdefault(f"{it['slot']}/{it['sku']}/{it.get('facing', 0)}",
                                    dict(sku=it["sku"], x=[np.inf, -np.inf], front_y=np.inf, units=[]))
            lane["x"] = [min(lane["x"][0], float(row[0]) - hx), max(lane["x"][1], float(row[0]) + hx)]
            lane["front_y"] = min(lane["front_y"], front)
            lane["units"].append(n)
        return cache.setdefault(tuple(units), lanes)

    def _unfronted(self, q, units):
        """Facing lanes of one slot's `units` left unfronted: no unit of the lane's own product standing in the lane's
        found x extent has its front face within FRONTED_TOL_M of where the lane's front face was found."""
        sc = self.world.scene
        out = []
        for name, lane in self._lanes(units).items():
            fronts = [float(q[b, 1]) - self._half_extent(q[b], SKUS[lane["sku"]], 1)
                      for n in units if sc.items[n]["sku"] == lane["sku"] for b in (sc.items[n]["body"],)
                      if lane["x"][0] <= q[b, 0] <= lane["x"][1]]
            front = min(fronts) if fronts else None
            if front is None or front > lane["front_y"] + FRONTED_TOL_M:
                out.append(dict(lane=name, units=lane["units"], found_front_y_m=round(lane["front_y"], 4),
                                front_y_m=None if front is None else round(front, 4)))
        return out

    def _count_by_scanning(self, q, rec, reads, checks, metrics):
        """count-by-scanning (module docstring): the count booked under the slot's location from each unit's own
        symbol, the shelf left as found, the scanner back; score = 1 on success, else FAILED x progress."""
        from .catalog import sku_gtin14
        sc = self.world.scene
        target = self.meta.get("target_slot")
        want = {sku_gtin14(SKUS[k]): n for k, n in self.meta.get("truth", {}).items()}
        got = dict((rec.get("location_counts") or {}).get(target, {}))
        metrics["record"], metrics["truth"] = got, want
        checks["record_matches"] = got == want
        units = [n for n, it in sc.items.items() if it.get("slot") == target and it.get("sku")]
        # bookings only: a read the host refused and a location label book nothing (device.HostApp.on_scan)
        per_unit, foreign, hits, errors = {}, [], set(), []
        for r in reads:
            loc = r.get("booked_location")
            if not loc:
                continue
            a = r.get("attribution") or {}
            mine = a.get("kind") == "product" and a.get("item") in units
            if mine:
                per_unit[a["item"]] = per_unit.get(a["item"], 0) + 1
            else:
                foreign.append(a.get("label"))
            if mine and loc == target and not a.get("misread") and a["item"] not in hits:
                hits.add(a["item"])
            else:
                errors.append(dict(label=a.get("label"), location=loc, misread=bool(a.get("misread"))))
        metrics["unit_reads"], metrics["foreign_reads"] = per_unit, foreign
        checks["each_unit_once"] = all(per_unit.get(u, 0) == 1 for u in units) and not foreign
        # misreads the record holds data from: a booked misread, or a misread location label whose code a booking was
        # made under before any correct read of that code in the same stretch (consecutive location reads setting one
        # code). A refused read, a stray misread of the code already correctly active, or a misread label re-scanned
        # before anything was booked leaves nothing in the record. An unbacked stretch of the counted slot's code costs
        # one error (its bookings would otherwise be hits); bookings under another code are errors already.
        in_record, stretch = {i for i, r in enumerate(reads) if r.get("booked_location")
                              and (r.get("attribution") or {}).get("misread")}, None
        for i, r in enumerate(reads):
            code = r.get("sets_location")
            if code:
                if stretch is None or stretch["code"] != code:
                    stretch = dict(code=code, backed=False, misread=[], charged=False)
                if (r.get("attribution") or {}).get("misread"):
                    stretch["misread"].append(i)
                else:
                    stretch["backed"] = True
            elif r.get("booked_location") and stretch is not None and not stretch["backed"]:
                in_record.update(stretch["misread"])                 # unbacked: every read in the stretch misread
                if stretch["code"] == target and not stretch["charged"]:
                    stretch["charged"] = True
                    errors.append(dict(label=(reads[stretch["misread"][0]].get("attribution") or {}).get("label"),
                                       location=target, misread=True))
        metrics["misreads_in_record"] = len(in_record)
        checks["no_misreads"] = not in_record
        disturbed = self._shelf(q, target, units, checks, metrics)
        # progress: the count booked, less wrong bookings and what is left disturbed, by any route
        returned = checks["scanner_seated"] and checks["no_key_pressed"]
        progress = max(0., len(hits) - len(errors) - DISTURBED_COST*len(disturbed))/max(1, len(units)) \
            * (1. if returned else NOT_RETURNED)
        metrics.update(units=len(units), hits=sorted(hits), errors=errors, disturbed=disturbed, scanner_returned=bool(returned))
        ok = all(checks.values())
        return dict(success=bool(ok), score=1. if ok else FAILED*progress, progress=float(progress),
                    contract=CONTRACTS["count-by-scanning"], checks={k: bool(v) for k, v in checks.items()},
                    metrics=metrics)

    def _shelf(self, q, target, units, checks, metrics):
        """The shelf against count-by-scanning's stock rules (module docstring), for the counted slot `target` holding
        `units`: sets stock_faced_in_slot, no_foreign_units_in_slot and neighbouring_stock_undisturbed (with their
        metrics) and returns what is left disturbed: units and cases off their place, other slots' units in the counted
        slot, free bodies on the floor, and unfronted facing lanes (a lane counted once unless one of its units is)."""
        sc = self.world.scene
        # the counted slot: each unit in place, upright and faced (cans on their label front), each facing lane fronted
        s = sc.slots[target]
        problems = {}
        for n in units:
            it, sku = sc.items[n], SKUS[sc.items[n]["sku"]]
            row = q[it["body"]]
            resting = abs(self._lowest_z(row, it["body"]) - s["z"]) <= REST_TOL_M         # on the deck, not held above it
            inside = s["x"][0] - SLOT_TOL_M <= row[0] <= s["x"][1] + SLOT_TOL_M \
                and s["y"][0] - SLOT_TOL_M <= row[1] <= s["y"][1] + SLOT_TOL_M and resting
            upright = self._tilt_deg(row) <= UPRIGHT_DEG
            faced = self._front_dev_deg(row, sku) <= FACED_DEG
            if not (inside and upright and faced):
                problems[n] = dict(unit=n, inside=bool(inside), resting=resting, upright=upright, faced=faced)
        unfronted = self._unfronted(q, units)
        metrics["stock_problems"], metrics["unfronted_lanes"] = list(problems.values()), unfronted
        checks["stock_faced_in_slot"] = not problems and not unfronted
        foreign_in = [n for n, it in sc.items.items() if it.get("sku") and it.get("slot") != target
                      and s["x"][0] <= q[it["body"], 0] <= s["x"][1] and s["y"][0] <= q[it["body"], 1] <= s["y"][1]
                      and s["z"] - .01 <= q[it["body"], 2] <= s["z"] + .35]
        metrics["foreign_units_in_slot"] = foreign_in
        checks["no_foreign_units_in_slot"] = not foreign_in
        # every other shelf unit and shipping case as found: in its own slot, standing, not turned away (within 30 deg
        # of its found heading, or of the aisle), and each other slot's facing lanes fronted by their own product
        others, by_slot = [], {}
        for n, it in sc.items.items():
            if not it.get("slot") or n in units:
                continue
            if it.get("sku"):
                by_slot.setdefault(it["slot"], []).append(n)
            row, row0, so, b = q[it["body"]], self.initial[n], sc.slots[it["slot"]], it["body"]
            inside = so["x"][0] - SLOT_TOL_M <= row[0] <= so["x"][1] + SLOT_TOL_M \
                and so["y"][0] - SLOT_TOL_M <= row[1] <= so["y"][1] + SLOT_TOL_M \
                and abs(self._lowest_z(row, b) - self._lowest_z(row0, b)) <= REST_TOL_M           # resting where found
            upright = self._tilt_deg(row) <= UPRIGHT_DEG
            as_faced = self._turn_deg(row0, row) <= TURNED_DEG or self._front_dev_deg(row, None) <= FACED_DEG
            if not (inside and upright and as_faced):
                others.append(dict(unit=n, slot=it["slot"], inside=bool(inside), upright=upright, as_faced=as_faced))
        others_unfronted = [lane for slot_units in by_slot.values() for lane in self._unfronted(q, slot_units)]
        metrics["neighbour_problems"], metrics["neighbour_unfronted_lanes"] = others, others_unfronted
        checks["neighbouring_stock_undisturbed"] = not others and not others_unfronted
        bad = set(problems) | {o["unit"] for o in others}
        disturbed = sorted(bad | set(foreign_in) | set(metrics["on_floor"]))
        disturbed += [f"lane {u['lane']}" for u in unfronted + others_unfronted if not set(u["units"]) & bad]
        return disturbed

    def _count_key_quantity(self, q, rec, reads, checks, metrics, returned):
        """count-key-quantity: the quantities the host holds under the slot's location equal the truth,
        nothing keyed under any other location, the shelf by count-by-scanning's stock rules, the scanner back."""
        from .catalog import sku_gtin14
        sc = self.world.scene
        target = self.meta.get("target_slot")
        want = {sku_gtin14(SKUS[k]): n for k, n in self.meta.get("truth", {}).items()}
        counts = rec.get("location_counts") or {}
        got = dict(counts.get(target, {}))
        metrics["record"], metrics["truth"] = got, want
        checks["record_matches"] = got == want
        # quantities keyed under another location are wrong data in the record (the counted slot's own quantities are
        # judged, not which location label was read last, as in count-by-scanning)
        elsewhere = {loc: dict(c) for loc, c in counts.items() if loc != target and c}
        metrics["counts_under_other_locations"] = elsewhere
        checks["no_counts_under_other_locations"] = not elsewhere
        units = [n for n, it in sc.items.items() if it.get("slot") == target and it.get("sku")]
        disturbed = self._shelf(q, target, units, checks, metrics)
        # which scan each saved quantity was keyed for: the host's entries replayed beside the reads' attribution
        scans, booked = S.key_bookings(rec["entries"], [r.get("attribution") or {} for r in reads], sc.slots.keys())
        if {loc: {c: v[0] for c, v in d.items()} for loc, d in booked.items()} != counts:
            raise RuntimeError(f"count-key-quantity: the replayed entries {booked} disagree with the host record "
                               f"{counts}")
        scanned = {code for loc, code, a in scans if loc == target and not a.get("misread")}
        errors = [dict(location=loc, code=code, misread=bool(a.get("misread"))) for loc, d in booked.items()
                  for code, (_, a) in d.items() if loc != target or code not in want or a.get("misread")]
        credits = [S.quantity_credit(g in scanned, got.get(g, 0), n) for g, n in want.items()]
        metrics.update(products=len(want), scanned=sorted(scanned), errors=errors, units=len(units),
                       disturbed=disturbed)
        return self._done(checks, metrics, S.item_progress(credits, len(errors), len(disturbed), returned))

    def evaluate(self):
        env, w = self.env, self.world
        c = env.config
        q = w.state.body_q.numpy()
        rec = env.host.record
        checks, metrics = {}, {}
        reads = env.firmware.reads
        metrics["reads"] = len(reads)
        metrics["frames"] = len(env.frames)
        metrics["time_s"] = w.sim_time
        target = self.meta.get("target_slot")
        free = {n: it["body"] for n, it in w.scene.items.items()}
        free["scanner"] = w.scene.bodies["scanner"]
        on_floor = [n for n, b in free.items() if self._lowest_z(q[b], b) < .015]
        metrics["on_floor"] = on_floor
        checks["nothing_on_floor"] = not on_floor
        held = self._scanner_held()
        metrics["scanner_held"] = held
        checks["scanner_seated"] = bool(w.seated) and not held
        checks["no_key_pressed"] = not any(s.state for s in w.key_switch.values())
        ghosts = [r for r in reads if (r.get("attribution") or {}).get("ghost")]
        metrics["ghost_reads"] = len(ghosts)
        metrics["misreads"] = sum(1 for r in reads if (r.get("attribution") or {}).get("misread"))
        checks["no_unexplained_reads"] = not ghosts
        # a record holding a misread did not get its data from the label it claims, even if the number happens to be
        # the one the task wanted
        checks["no_misreads"] = metrics["misreads"] == 0
        hist = getattr(env, "history", {}) or {}
        if c.task == "count-by-scanning":
            return self._count_by_scanning(q, rec, reads, checks, metrics)
        # the other seven: progress from scoring.py, by any route
        returned = checks["scanner_seated"] and checks["no_key_pressed"]
        metrics["scanner_returned"] = bool(returned)
        att = [dict(r.get("attribution") or {}, format=r["format"]) for r in reads]
        if c.task == "hold-scanner":
            steady = float(hist.get("max_steady_out_s", 0.))
            checks["no_reads"] = len(reads) == 0
            metrics["steady_hold_s"] = round(steady, 3)
            checks["held_steady_5s"] = steady >= S.HOLD_S
            return self._done(checks, metrics, S.hold_progress(steady, returned, len(reads), len(on_floor)))
        if c.task == "scan-location":
            checks["location_recorded"] = rec["location"] == target
            # a read of another slot's location label, or one that made another slot's code the location (a
            # second read of the right label changes nothing in the record and is not a workmanship error)
            label = f"loc_{target}"
            wrong = [a.get("label") for r, a in zip(reads, att) if r.get("sets_location") not in (None, target)
                     or (a.get("kind") == "location" and a.get("label") != label)]
            metrics["wrong_location_reads"] = wrong
            checks["no_wrong_location_read"] = not wrong
            hit = checks["location_recorded"] and any(a.get("label") == label and not a.get("misread") for a in att)
            metrics["location_hit"] = bool(hit)
            return self._done(checks, metrics, S.location_progress(hit, returned, len(wrong), len(on_floor)))
        if c.task == "count-key-quantity":
            return self._count_key_quantity(q, rec, reads, checks, metrics, returned)
        if c.task in ("checkout-present", "checkout-bulky", "checkout-counter"):
            from .catalog import sku_gtin14
            items = {n: it for n, it in w.scene.items.items() if it.get("sku")}
            want = sorted(sku_gtin14(SKUS[it["sku"]]) for it in items.values())
            metrics["sale"], metrics["truth"] = sorted(rec["sale"]), want
            checks["sale_matches"] = sorted(rec["sale"]) == want
            per_unit = {}
            for r in reads:
                a = r.get("attribution") or {}
                if a.get("kind") == "product":
                    per_unit[a["item"]] = per_unit.get(a["item"], 0) + 1
            metrics["unit_reads"] = per_unit
            checks["each_item_once"] = all(per_unit.get(n, 0) == 1 for n in items) and not any(
                (r.get("attribution") or {}).get("kind") == "inner" for r in reads)
            basket = [n for n in items if n != "water_case"]
            bagged = set()
            if w.scene.items.get("bag_tote"):
                inside = {n for n in basket if self._in_container(q, "bag_tote", items[n]["body"])}
                tote_ok = self._container_on_counter(q, "bag_tote")
                metrics["not_bagged"] = [n for n in basket if n not in inside]
                checks["items_bagged"] = not metrics["not_bagged"] and tote_ok
                bagged = inside if tote_ok else set()
            # progress: each item rung up (its own symbol read correctly) and bagged; one error per other read (a
            # repeat, an inner unit, a misread, a label that is no item's) and half an item per item bagged unpaid
            rung, errors = S.tally(att, items, "product", lambda a: a.get("kind") == "inner")
            unpaid = sorted(bagged - rung)
            credits = [S.put_away_credit(n in rung, n in bagged) for n in basket]
            if c.task == "checkout-bulky" and "water_case" in w.scene.items:
                rise = float((hist.get("max_rise_m") or {}).get(int(w.scene.items["water_case"]["body"]), 0.))
                metrics["water_case_max_rise_mm"] = round(rise*1e3, 2)
                checks["case_not_lifted"] = rise < S.LIFT_FREE_M
                credits.append(S.lift_credit(rise) if "water_case" in rung else 0.)
            metrics.update(rung=sorted(rung), read_errors=errors, unpaid=unpaid)
            return self._done(checks, metrics, S.item_progress(credits, len(errors) + S.UNPAID_COST*len(unpaid),
                                                               len(on_floor), returned))
        if c.task == "library-checkin":
            books = {n: it for n, it in w.scene.items.items() if it.get("library_id")}
            want = sorted(it["library_id"] for it in books.values())
            metrics["checked_in"], metrics["truth"] = sorted(rec["checked_in"]), want
            checks["checkin_matches"] = sorted(rec["checked_in"]) == want
            checks["no_isbn_reads"] = not any(r["format"] in ISBN_FORMATS for r in reads)
            in_bin = set()
            if w.scene.items.get("returns_bin"):
                inside = {n for n, it in books.items() if self._in_container(q, "returns_bin", it["body"])}
                bin_ok = self._container_on_counter(q, "returns_bin")
                metrics["not_returned"] = [n for n in books if n not in inside]
                checks["books_in_bin"] = not metrics["not_returned"] and bin_ok
                in_bin = inside if bin_ok else set()
            # progress: each book checked in (its library label read correctly) and binned; one error per other read
            # (an ISBN, a misread, a repeat check-in, a label that is no book's library label)
            checked, errors = S.tally(att, books, "library", lambda a: a.get("format") in ISBN_FORMATS)
            metrics.update(checked=sorted(checked), read_errors=errors)
            credits = [S.put_away_credit(b in checked, b in in_bin) for b in books]
            return self._done(checks, metrics, S.item_progress(credits, len(errors), len(on_floor), returned))
        raise ValueError(f"No evaluator for task {c.task!r}")

    def _done(self, checks, metrics, progress):
        """score = 1 if every check passes, else FAILED x progress (scoring.py)."""
        ok = all(checks.values())
        return dict(success=bool(ok), score=S.score(ok, progress), progress=float(progress),
                    contract=CONTRACTS[self.env.config.task], checks={k: bool(v) for k, v in checks.items()},
                    metrics=metrics)
