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
(evaluation["contract"], CONTRACT in its verifier.py). Evaluator.evaluate checks what every task shares and calls the
task's verify (tasks/<task>/verifier.py) for the rest. The seven tasks other than count-by-scanning take their progress
from scoring.py and their verifier.py. count-by-scanning has its own checks and score (its verifier.py), and its stock
rules are count-key-quantity's too (Evaluator._shelf):
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
from .config import task_verifier

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


# count-by-scanning's stock rules, which count-key-quantity shares: authored tolerances (module docstring)
SLOT_TOL_M = .02        # a unit's centre within its slot +-20 mm
REST_TOL_M = .005       # its lowest point within 5 mm of the deck it stands on (resting, not held above it or stacked)
UPRIGHT_DEG = 15.
FACED_DEG = 30.         # front panel / label front within 30 deg of the aisle (cans spawn within about 26 deg, 3 sigma)
FRONTED_TOL_M = .04     # a facing lane's front-most unit at most 40 mm behind where the lane's front face was found
TURNED_DEG = 30.        # stock outside the counted slot turned at most 30 deg from its found heading (or faced, FACED_DEG)

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

    def evaluate(self):
        """The checks every task shares (nothing on the floor, the scanner seated and released, no key down, no read
        unexplained or misread), then the task's verify (tasks/<task>/verifier.py): its own checks and its score."""
        env, w = self.env, self.world
        c = env.config
        q = w.state.body_q.numpy()
        rec = env.host.record
        checks, metrics = {}, {}
        reads = env.firmware.reads
        metrics["reads"] = len(reads)
        metrics["frames"] = len(env.frames)
        metrics["time_s"] = w.sim_time
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
        folder = task_verifier(c.task)
        if folder is None:
            raise ValueError(f"No evaluator for task {c.task!r}")
        return folder.verify(self, q, rec, reads, checks, metrics)

    def _returned(self, checks, metrics):
        """The seven tasks other than count-by-scanning (progress from scoring.py, by any route): the scanner back,
        seated and released with no key down; recorded as metrics["scanner_returned"]."""
        returned = checks["scanner_seated"] and checks["no_key_pressed"]
        metrics["scanner_returned"] = bool(returned)
        return returned

    @staticmethod
    def _attributions(reads):
        """Each read's attribution with its format (the reads scoring.tally sorts)."""
        return [dict(r.get("attribution") or {}, format=r["format"]) for r in reads]

    def _checkout(self, q, rec, reads, checks, metrics, returned, case=None):
        """checkout-present, checkout-bulky, checkout-counter: the sale equals the items, each item's own symbol read
        once and no inner unit's, every basket item in the bagging tote. case(e, rung, checks, metrics, credits), from
        checkout-bulky's verifier.py, judges the case at the counter's end and appends its credit."""
        w = self.world
        on_floor = metrics["on_floor"]
        att = self._attributions(reads)
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
        if case is not None:
            case(self, rung, checks, metrics, credits)
        metrics.update(rung=sorted(rung), read_errors=errors, unpaid=unpaid)
        return self._done(checks, metrics, S.item_progress(credits, len(errors) + S.UNPAID_COST*len(unpaid),
                                                           len(on_floor), returned))

    def _done(self, checks, metrics, progress):
        """score = 1 if every check passes, else FAILED x progress (scoring.py); the contract is the task's (CONTRACT in
        its verifier.py)."""
        ok = all(checks.values())
        return dict(success=bool(ok), score=S.score(ok, progress), progress=float(progress),
                    contract=task_verifier(self.env.config.task).CONTRACT,
                    checks={k: bool(v) for k, v in checks.items()}, metrics=metrics)
