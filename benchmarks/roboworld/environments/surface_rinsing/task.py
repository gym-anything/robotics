"""Task evaluator: success is read from the physical state of the water, the ware and the fixtures.

Settle rule (painting's, adapted): `evaluate()` branches the state (owner checkpoint), keeps the robot holding its pose
with its hands as they are, advances the whole world for at least CRITERIA.settle_min_s (2 s) and then until it is quiet
(no outlet flowing, no water in the air, no item moving, nothing still draining) or CRITERIA.settle_s (30 s) has passed
(racks drain, films thin, streams stop once their valves are released), measures, and restores; the live episode is not
advanced. The verdict is cached per (tick, generation). `progress()` is the cheap per-step readout of the same measures
without the settle.

Criteria (per task: Evaluator.criteria): soap-free (surfactant in solution, adsorbed and in foam over all of an item's
faces <= 10 ug, no 1 x 1 cm gauge above 1 mg/m2, no foam), sanitized (>= 30 s of full immersion in the sanitizer bowl,
having carried <= 10 ug into it), visibly clean (every 1 cm gauge of the target faces <= 1 ug/cm2 of soil), drained
(free water inside an item -- its pool plus the film above what clings -- below its limit), placed (racks, bowls, colander, can, valves parked), contained (soil, tracer
and water off the clean surfaces), intact (no glass or china impact above the 1.0 m/s gate; nothing on the floor), and
no simulation fault (finite state, closed ledger). A task failure is distinct from a simulation fault.

A benchmark task's own parts are in its folder (tasks/<task>/verifier.py), which Evaluator reads: its gates (criteria),
what the task-set contracts add (task_set/evaluator.py, task_set_v2_evaluator.py) and its progress formulas, and for
scrub-rinse-pan its sanitizer clock (PanTracker) and score contract (REWARD_CONTRACT). evaluate() calls the folder's
verify. rinse-dishware, outside the benchmark, keeps its gates here.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from . import layout as L
from . import ware as W
from .layout import task_verifier

C = P.CRITERIA
GAUGE = 0.01

# ------------------------------------------------------------------ sanitizer trackers and score contracts
# (scrub-rinse-pan's PanTracker and REWARD_CONTRACT, rinse-glassware's GlassTracker: their tasks/<task>/verifier.py)
FAIL_SCORE_SCALE = 0.9
# share of the pan's inner area under liquid for sanitizer time to count: a flooded pan reads 0.91-0.99 (the inner rim
# corner, ~7 % of the cells, never registers), a mouth-down pan 0.0
PAN_INSIDE_WET = 0.85
# water a running outlet's stream may add to a sanitized pan outside every bowl before the pan counts as rinsed again
# (Food Code 4-904.14: sanitized utensils are not rinsed before air-drying); stray drops and splashes stay below it
PAN_STREAM_RINSE_ML = 5.0
# the dropped-item line of intact_and_not_dropped: a pan whose origin goes below it has touched the floor
PAN_FLOOR_Z = 0.25


def ledger_closed(L, tol=1e-9):
    """Every conserved quantity closes: water, and the detergent and soil it carries (relative tol, or within 1 ng when
    there is almost none). Gated by no_simulation_fault and used by every probe's pass criterion. (A water-only check
    passed a splatter race that created 7 ug of detergent in a can rinse.)"""
    return bool(L["relative"] < tol and all(L[r] < tol or abs(L[a]) < 1e-12 for r, a in (("relative_surf", "surfactant_kg"),
                                                                                      ("relative_solids", "solids_kg"))))


def gauges(pos, area, value, nrm=None):
    """Areal density per 1 x 1 cm gauge: cells binned by their (body-frame) position in 1 cm cubes. With the cells'
    normals, each cell is binned half a gauge out along its normal, so the two faces of a thin wall (a plate is 5 mm
    thick) never share a gauge (a shared gauge would read a soiled spot on a plate's face at half its density, diluted
    by the clean back)."""
    if not len(pos):
        return np.zeros(0)
    if nrm is not None:
        pos = pos + 0.5*GAUGE*nrm
    key = np.floor(pos/GAUGE).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    m = np.bincount(inv, weights=value*area)
    a = np.bincount(inv, weights=area)
    return m/np.maximum(a, 1e-12)


class Impacts:
    """Per-item maximum normal approach speed at new contacts against anything but a finger pad (world recorder)."""

    def __init__(self, world, items):
        self.w = world
        self.items = {world._mj_body[b]: n for n, b in items.items() if world._mj_body.get(b, -1) >= 0}
        self.max = {n: 0.0 for n in items.values() if False}
        self.max = {n: 0.0 for n in items}
        self.prev = set()
        self.pads = set(int(g) for g in world.pad_geoms)
        self._v = np.zeros(6)

    def __call__(self, world):
        import mujoco
        d, m = world.mjd, world.mjm
        now = set()
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if (b1 not in self.items and b2 not in self.items) or g1 in self.pads or g2 in self.pads:
                continue
            key = (min(g1, g2), max(g1, g2))
            now.add(key)
            if key in self.prev:
                continue
            n = np.asarray(c.frame[:3])
            v = []
            for b in (b1, b2):
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, self._v, 0)
                w_, lin = self._v[:3], self._v[3:]
                v.append(lin + np.cross(w_, np.asarray(c.pos) - d.xpos[b]))
            vn = abs(float(np.dot(v[1] - v[0], n)))
            for b in (b1, b2):
                if b in self.items:
                    name = self.items[b]
                    self.max[name] = max(self.max[name], vn)
        self.prev = now


class Evaluator:
    def __init__(self, env):
        self.env = env
        w = env.world
        self.task = env.spec["task"]
        self.items = {n: it["body"] for n, it in w.scene.items.items()}
        # items that stand on the floor by design (the garbage can): exempt from the dropped-on-the-floor gate (else
        # rinse-garbage-can fails at its own start state)
        self.floor_items = {n for n, b in self.items.items() if w.body_pose(b)[0][2] < 0.25}
        self.impacts = Impacts(w, self.items)
        w.recorders.append(self.impacts)
        self.sanitize = {n: 0.0 for n in self.items}
        self.surf_into_sanitizer = {n: 0.0 for n in self.items}
        self._in_san = set()
        tracker = getattr(task_verifier(self.task), "PanTracker", None)       # scrub-rinse-pan's (its verifier.py)
        self.pan = tracker(env) if tracker is not None and env.water is not None else None
        w.recorders.append(self._track)
        self._cache = None

    # ------------------------------------------------------------------ per-tick tracking
    def _track(self, world):
        ws = self.env.water
        if ws is None or world.tick % 20:
            return
        if self.pan is not None:
            self.pan.sample(self)
            return
        san = ws.pools.index.get("sanitize")
        if san is None:
            return
        dt = 20*P.DT
        now = set()
        for name, (pool, frac) in ws.immersed.items():
            if pool == "sanitize" and frac > 0.999:
                now.add(name)
                self.sanitize[name] = self.sanitize.get(name, 0.0) + dt
        for name in now - self._in_san:
            self.surf_into_sanitizer[name] = max(self.surf_into_sanitizer.get(name, 0.0), self.item_surfactant(name))
        self._in_san = now

    def snapshot(self):
        return dict(sanitize=dict(self.sanitize), into=dict(self.surf_into_sanitizer), in_san=set(self._in_san),
                    impacts=dict(self.impacts.max), prev=set(self.impacts.prev),
                    pan=self.pan.snapshot() if self.pan is not None else None)

    def restore(self, s):
        self.sanitize, self.surf_into_sanitizer, self._in_san = dict(s["sanitize"]), dict(s["into"]), set(s["in_san"])
        self.impacts.max, self.impacts.prev = dict(s["impacts"]), set(s["prev"])
        if self.pan is not None and s.get("pan") is not None:
            self.pan.restore(s["pan"])
        self._cache = None

    # ------------------------------------------------------------------ measures
    def _fields(self):
        E = self.env.water.eng
        sp = E.sp.numpy()
        return dict(h=E.h.numpy(), s=sp[0] + sp[5], foam=sp[4], gamma=E.gamma.numpy(), soil=E.soil.numpy(),
                    tracer=sp[2], sus=sp[1])

    def item_surfactant(self, name, f=None):
        ws = self.env.water
        f = f or self._fields()
        cells = ws.item_cells.get(name)
        if cells is None:
            return 0.0
        A = ws.surf.area[cells]
        tot = float(((f["s"][cells] + f["gamma"][cells])*A).sum())
        k = ws.pools.index.get(f"cavity:{name}")
        if k is not None:
            p = ws.pools.pools[k]
            tot += float(p.sp[0] + p.sp[5])
        return tot

    def item_regions(self, name, f=None):
        """Per chart region (inside, rim, outside, foot / face, back ...): surfactant (ug: solution + adsorbed + foam),
        film water (mL) and soil (mg)."""
        ws = self.env.water
        f = f or self._fields()
        w_ = self.env.world.scene.ware.get(name)
        ch = ws.item_charts.get(name)
        if w_ is None or ch is None or w_.kind not in w_.regions:
            return {}
        cells = ws.item_cells[name]
        A = ws.surf.area[cells]
        out = {}
        for reg, segs in w_.regions[w_.kind].items():
            m = np.isin(ch.cell_seg, segs)
            c = cells[m]
            out[reg] = dict(surf_ug=round(float(((f["s"][c] + f["gamma"][c])*A[m]).sum())*1e9, 3),
                            adsorbed_ug=round(float((f["gamma"][c]*A[m]).sum())*1e9, 3),
                            film_mL=round(float((f["h"][c]*A[m]).sum())*1e6, 4),
                            soil_mg=round(float((f["soil"][c]*A[m]).sum())*1e6, 4))
        return out

    def item_report(self, name, f):
        ws, w = self.env.water, self.env.world
        cells = ws.item_cells.get(name)
        out = dict()
        p, R = w.body_pose(self.items[name])
        out["position"] = [round(float(x), 4) for x in p]
        out["tilt_deg"] = round(math.degrees(math.acos(max(-1., min(1., float(R[2, 2]))))), 2)
        out["max_impact_m_s"] = round(self.impacts.max.get(name, 0.0), 3)
        if cells is not None:
            A = ws.surf.area[cells]
            pos = ws.surf.pos[cells]
            surf_area = f["s"][cells] + f["gamma"][cells]
            out["surfactant_ug"] = round(self.item_surfactant(name, f)*1e9, 3)
            nrm = ws.surf.nrm[cells]
            g = gauges(pos, A, surf_area, nrm)
            out["surfactant_gauge_max_mg_m2"] = round(float(g.max())*1e6, 4) if len(g) else 0.0
            out["foam_mL"] = round(float((f["foam"][cells]*A).sum())*1e6, 4)
            gs = gauges(pos, A, f["soil"][cells], nrm)
            out["soil_gauge_max_ug_cm2"] = round(float(gs.max())*1e5, 4) if len(gs) else 0.0
            out["soil_mg"] = round(float((f["soil"][cells]*A).sum())*1e6, 4)
            out["film_mL"] = round(float((f["h"][cells]*A).sum())*1e6, 4)
        k = ws.pools.index.get(f"cavity:{name}")
        if k is not None:
            out["pooled_mL"] = round(ws.pools.pools[k].vol*1e6, 3)
            out["free_water_mL"] = round(out["pooled_mL"] + self.free_film(name, f, R)*1e6, 3)
        out["sanitized_s"] = round(self.sanitize.get(name, 0.0), 2)
        out["surfactant_into_sanitizer_ug"] = round(self.surf_into_sanitizer.get(name, 0.0)*1e9, 3)
        return out

    def free_film(self, name, f, R):
        """Water inside an item's cavity that would still run if it were tipped (m3): on its inner faces, the film above
        what clings -- the engine's own pinned-bead depth for that substrate and slope (h_break / sqrt(sin slope), at
        most 5 h_break), and never less than 50 um (a glass draining for 5-60 s carries 15-45 um, Jeffreys). A shallow
        puddle on the bottom of an emptied container is film to the engine, not pool: the drained gates count both."""
        ws = self.env.water
        w_ = self.env.world.scene.ware.get(name)
        ch = ws.item_charts.get(name)
        if w_ is None or ch is None or "inner" not in w_.regions.get(w_.kind, {}):
            return 0.0
        if not hasattr(self, "_h_break"):
            self._h_break = ws.eng.h_break.numpy().astype(float)
        cells = ws.item_cells[name][np.isin(ch.cell_seg, w_.regions[w_.kind]["inner"])]
        nz = (ws.surf.nrm[cells] @ np.asarray(R).T)[:, 2]
        sin_a = np.sqrt(np.clip(1.0 - nz*nz, 0.0, 1.0))
        h_ret = np.maximum(self._h_break[cells]/np.sqrt(np.maximum(sin_a, 0.04)), 50e-6)
        return float((np.maximum(f["h"][cells] - h_ret, 0.0)*ws.surf.area[cells]).sum())

    def _in_rack_cell(self, name, rack):
        """Glass centred in a rack compartment (rack frame), mouth down."""
        w = self.env.world
        if rack not in self.items:
            return False
        pg, Rg = w.body_pose(self.items[name])
        pr, Rr = w.body_pose(self.items[rack])
        xl = Rr.T @ (pg - pr)
        wr = W.build("glass_rack")
        near = min(math.hypot(xl[0] - cx, xl[1] - cy) for cx, cy in wr.cells)
        # mouth down on the rack floor: the glass's foot (its body origin) is its own height above the floor (a fixed
        # 0.13 m ceiling would refuse the 150 mm mixing glasses standing correctly in their compartments)
        H = float(np.asarray(w.scene.ware[name].solid)[:, 1].max())
        return near < 0.025 and abs(xl[0]) < 0.25 and abs(xl[1]) < 0.25 and -0.015 < xl[2] - (wr.floor_z + H) < 0.02

    def _in_peg_rack(self, name, rack):
        w = self.env.world
        if rack not in self.items:
            return False
        pg, Rg = w.body_pose(self.items[name])
        pr, Rr = w.body_pose(self.items[rack])
        xl = Rr.T @ (pg - pr)
        axis = Rr.T @ Rg[:, 2]                         # plate face normal in the rack frame
        # a plate standing in a 35 mm slot between 60 mm posts leans up to ~21 deg either way (D1: the spray pushed the
        # front plate from 6 deg back to 21 deg forward, still in its slot); beyond rack_lean_deg it has left the slot
        upright = abs(axis[2]) < math.sin(math.radians(C["rack_lean_deg"]))
        return upright and abs(xl[0]) < 0.24 and abs(xl[1]) < 0.12 and 0.0 < xl[2] < 0.20

    def _inside_cavity(self, name, container):
        """Body origin of `name` inside the cavity region of `container`."""
        w = self.env.world
        wc = self.env.world.scene.ware[container]
        pg, _ = w.body_pose(self.items[name])
        pc, Rc = w.body_pose(self.items[container])
        xl = Rc.T @ (pg - pc)
        z0, z1, r0, r1 = wc.cavity["region"]
        rr = r0 + (r1 - r0)*min(1, max(0, (xl[2] - z0)/(z1 - z0)))
        return z0 - 0.01 < xl[2] < z1 + 0.02 and math.hypot(xl[0], xl[1]) < rr + 0.01

    def _on_clean_board(self, name):
        w = self.env.world
        p, _ = w.body_pose(self.items[name])
        x0, x1 = L.DRAINBOARDS["clean"]
        return x0 < p[0] < x1 and L.SINK_Y0 < p[1] < L.Y and L.RIM - 0.01 < p[2] < L.RIM + 0.10

    def _on_shelf(self, name):
        """On the wall shelf over the clean drainboard: the item's centre over the board (the shelf runs 0.30 m past the
        board's end, over bare floor, so a pan entirely past the board does not pass)."""
        w = self.env.world
        p, _ = w.body_pose(self.items[name])
        x0, x1 = L.DRAINBOARDS["clean"]
        return x0 < p[0] < x1 and L.Y - 0.32 < p[1] < L.Y and 1.50 < p[2] < 1.65

    def _on_ledge(self, name, compartments=False):
        """On the back ledge. compartments: only the deck behind the three compartments (scrub-rinse-pan; behind the
        drainboards the boards run to the backsplash, and a box over the whole sink took their back strips for the
        ledge, the clean board's included). Its height ceiling admits the sponge resting on the deck in any
        orientation: its origin (the scour face's centre) is 49.5 mm up when it stands on its short end; holding is the
        hands gate's business."""
        w = self.env.world
        p, _ = w.body_pose(self.items[name])
        x0, x1 = (L.BOWLS["wash"][0], L.BOWLS["sanitize"][1]) if compartments else (L.SINK_X0, L.SINK_X1)
        return x0 < p[0] < x1 and L.BOWL_Y1 < p[1] < L.Y and L.RIM - 0.005 < p[2] < L.RIM + (0.055 if compartments else 0.03)

    def sponge_report(self):
        ws = self.env.water
        sm = next(iter(ws.sponges.values()), None)
        if sm is None:
            return {}
        p = sm.pool
        return dict(pore_mL=round(p.vol*1e6, 3), fill=round(p.vol/sm.v_full, 4), c_g_L=round(p.sp[0]/p.vol, 5) if p.vol > 1e-9 else 0.0,
                    strain=round(sm.eps, 4), squeeze_force_N=round(sm.force, 3), contacts=dict(sm.contact_log))

    def _fixtures_closed(self):
        ws, w = self.env.water, self.env.world
        flows = {n: o.Q/P.GPM for n, o in ws.outlets.items()}
        open_ = {n: round(q, 4) for n, q in flows.items() if q > 1e-4}
        parked = {}
        for n, f in w.scene.fixtures.items():
            if f["kind"] == "spray_valve":
                p, _ = w.body_pose(f["valve"])
                parked[n] = round(float(np.linalg.norm(p - np.asarray(f["park"]))), 3)
        holding = bool(w.holding)
        if self.pan is not None:
            holding = bool(self.pan.held(w))        # pads on a movable thing only (PanTracker.held)
        return open_, parked, holding

    def _contamination(self, f):
        """Soil, tracer and water on the designated clean surfaces of the station."""
        ws = self.env.water
        out = {}
        S = ws.surf
        clean = [ch for ch in S.charts if ch.role == "clean"]
        for name in ("cutting_board",):
            if name in ws.item_charts:
                clean.append(ws.item_charts[name])
        for ch in clean:
            cells = np.arange(ch.first, ch.first + ch.count)
            A = S.area[cells]
            dirt = f["soil"][cells] + f["sus"][cells] + f["tracer"][cells]
            g = gauges(S.pos[cells], A, dirt, S.nrm[cells])
            out[ch.name] = dict(gauge_max_ug_cm2=round(float(g.max())*1e5, 4) if len(g) else 0.0,
                                water_mL=round(float((f["h"][cells]*A).sum())*1e6, 3))
        if "clean_bowl" in ws.item_charts:
            name = "clean_bowl"
            cells = ws.item_cells[name]
            A = S.area[cells]
            g = gauges(S.pos[cells], A, f["soil"][cells] + f["tracer"][cells], S.nrm[cells])
            out[name] = dict(gauge_max_ug_cm2=round(float(g.max())*1e5, 4))
        sinks = ws.totals()["sinks"]
        if "shelf" in sinks:
            out["dunnage_shelf"] = dict(water_g=round(sinks["shelf"]*1e3, 3))
        floor = sinks.get("floor", 0.0)
        for ch in S.charts:
            if ch.role == "floor" and "drain" not in getattr(ch, "extra", {}):
                cells = np.arange(ch.first, ch.first + ch.count)
                floor += float((f["h"][cells]*S.area[cells]).sum())*P.WATER["rho"]
        out["floor_water_mL"] = round(floor/P.WATER["rho"]*1e6, 3)
        return out

    def measure(self):
        env = self.env
        ws, w = env.water, env.world
        f = self._fields()
        items = {n: self.item_report(n, f) for n in self.items}
        open_, parked, holding = self._fixtures_closed()
        T = ws.eng.totals()
        cl = T["clamped"]
        led = dict(ws.ledger_residual(), lost_water_kg=float(T["lost"]),
                   lost_water_by_cause_kg=dict(zip(("aged_out", "left_the_room", "overwritten"),
                                                   [float(x) for x in T["lost_by"]])),
                   created_by_clamps_kg=dict(water=float(cl[0])*P.WATER["rho"], detergent=float(cl[1] + cl[6]),
                                             soil=float(cl[2])),
                   lifted_from_under_floor_kg=float(T["lifted"]), kept_by_floor_plane_unstruck_kg=float(T["floor_plane_caught"]))
        out = dict(items=items, open_outlets=open_, valves_from_park_m=parked, holding=holding, sponge=self.sponge_report(),
                   contamination=self._contamination(f), ledger=led, fault=w.fault,
                   pools={p.name: dict(volume_L=round(p.vol*1e3, 4), T_C=round(p.T, 2)) for p in ws.pools.pools
                          if p.kind == "static"},
                   reflexes=len(w.reflexes))
        if self.pan is not None:
            out["pan"] = self.pan.measure(self, f, held=self.pan.held(w), any_pad_contact=bool(w.holding))
        return out

    # ------------------------------------------------------------------ verdicts
    def criteria(self, m):
        t, items = self.task, m["items"]
        crit = {}

        def gate(name, ok, detail):
            crit[name] = dict(passed=bool(ok), detail=detail)

        L = m["ledger"]
        gate("no_simulation_fault", m["fault"] is None and ledger_closed(L),
             dict(fault=m["fault"], ledger_relative=L["relative"], ledger_relative_surfactant=L["relative_surf"],
                  ledger_relative_solids=L["relative_solids"], lost_water_kg=L["lost_water_kg"],
                  lost_water_by_cause_kg=L["lost_water_by_cause_kg"], created_by_clamps_kg=L["created_by_clamps_kg"],
                  lifted_from_under_floor_kg=L["lifted_from_under_floor_kg"],
                  kept_by_floor_plane_unstruck_kg=L["kept_by_floor_plane_unstruck_kg"]))
        released = not m["open_outlets"] and not m["holding"]
        hands = dict(open_outlets=m["open_outlets"], holding=m["holding"])
        if "pan" in m:
            hands.update(held=m["pan"]["held"], any_pad_contact=m["pan"]["any_pad_contact"])
        gate("water_off_and_hands_free", released, hands)
        kinds = self.env.world.scene.items
        fragile = [n for n in items if kinds.get(n, {}).get("kind") in P.BREAKABLE]
        bad = {n: items[n]["max_impact_m_s"] for n in fragile if items[n]["max_impact_m_s"] > P.IMPACT_GATE}
        on_floor = [n for n, r in items.items() if r["position"][2] < 0.25 and n not in self.floor_items]
        gate("intact_and_not_dropped", not bad and not on_floor, dict(impacts_over_gate=bad, on_floor=on_floor))
        own = getattr(task_verifier(t), "criteria", None)
        if own is not None:                         # a benchmark task's own gates (its verifier.py)
            own(self, m, gate)
        elif t == "rinse-dishware":
            def place(n):
                return (self._in_peg_rack(n, "peg_rack") if n.startswith("plate") else
                        self._on_clean_board(n) and items[n]["tilt_deg"] > 180 - C["place_tilt_deg"])
            self.ware_criteria(m, gate, place)
        return crit

    def ware_criteria(self, m, gate, place):
        """The gates of washed ware to rinse, sanitize and set down to drain (rinse-glassware, rinse-dishware):
        soap_free, sanitized, drained, placed (place(name): the item stands where its task puts it) and contained."""
        items, cont = m["items"], m["contamination"]
        ware = [n for n in items if n.startswith(("glass_", "plate_", "bowl_")) and n != "glass_rack"]
        soap = {n: (items[n]["surfactant_ug"], items[n]["surfactant_gauge_max_mg_m2"], items[n]["foam_mL"]) for n in ware}
        gate("soap_free", all(s <= C["surf_per_item"]*1e9 and g <= C["surf_gauge"]*1e6 and fm < 1e-3
                              for s, g, fm in soap.values()), soap)
        san = {n: (items[n]["sanitized_s"], items[n]["surfactant_into_sanitizer_ug"]) for n in ware}
        gate("sanitized", all(s >= C["sanitize_s"] and u <= C["surf_per_item"]*1e9 for s, u in san.values()), san)
        lim = {"glass": C["drained_glass"], "bowl": C["drained_bowl"]}
        dr = {n: items[n]["free_water_mL"] for n in ware if "free_water_mL" in items[n]}
        gate("drained", all(v <= lim["glass" if n.startswith("glass") else "bowl"]*1e6 for n, v in dr.items()), dr)
        placed = {n: place(n) for n in ware}
        gate("placed", all(placed.values()), placed)
        gate("contained", cont.get("drainboard_clean", {}).get("gauge_max_ug_cm2", 0) <= P.VRL*1e5 and
             cont["floor_water_mL"] <= C["floor_water"]*1e6, cont)

    def _score(self, crit, m):
        """(score, reward record). Tasks whose verifier.py names no REWARD_CONTRACT: the fraction of passed gates
        (record None). scrub-rinse-pan (progress-v1): 1 if success else FAIL_SCORE_SCALE x progress, progress = (C +
        C R + C R Z + C R Z T)/4 (its PanTracker), with the fraction of passed gates beside it as mean_of_checks. A
        state with a simulation fault or an open water, detergent or soil ledger is not a task result: its score is None
        (withheld) and the record says verifiable false, per step as in the verdict."""
        frac = round(sum(c["passed"] for c in crit.values())/max(1, len(crit)), 4)
        contract = getattr(task_verifier(self.task), "REWARD_CONTRACT", None)
        if contract is None or "pan" not in m:
            return frac, None
        rw = self.pan.progress(m, crit)
        ok = all(c["passed"] for c in crit.values())
        verifiable = bool(m["fault"] is None and ledger_closed(m["ledger"]))
        rw.update(contract=contract, score_rule="1 if success else 0.9 x progress; None if unverifiable",
                  mean_of_checks=frac, verifiable=verifiable)
        if not verifiable:
            led = m["ledger"]
            rw["unverifiable_reason"] = (f"simulation fault: {m['fault']}" if m["fault"] is not None else
                                         f"open ledger: residuals water {led['relative']:.2e}, detergent "
                                         f"{led['relative_surf']:.2e}, soil {led['relative_solids']:.2e}")
            return None, rw
        return (1.0 if ok else round(FAIL_SCORE_SCALE*rw["progress"], 4)), rw

    def progress(self):
        if self.env.water is None:
            return dict(score=0.0, succeeded=False)
        m = self.measure()
        crit = self.criteria(m)
        score, rw = self._score(crit, m)
        out = dict(score=score, succeeded=all(c["passed"] for c in crit.values()),
                   criteria={k: v["passed"] for k, v in crit.items()}, settled=False)
        if rw is not None:
            out["reward"] = rw
            out["verifiable"] = rw["verifiable"]
        return out

    def _quiet(self, prev):
        """Has the world stopped changing in the ways the gates read? Returns (quiet, state for the next comparison,
        what is still moving)."""
        env = self.env
        ws, w = env.water, env.world
        flows = {n: o.Q/P.GPM for n, o in ws.outlets.items() if o.Q/P.GPM > 1e-4}
        T = ws.totals()
        qd = w.state.body_qd.numpy()
        bodies = np.array(list(self.items.values()), int)
        v = float(np.linalg.norm(qd[bodies, :3], axis=1).max()) if len(bodies) else 0.0
        f = self._fields()
        free = {}
        for n, b in self.items.items():
            if ws.pools.index.get(f"cavity:{n}") is not None:
                _, R = w.body_pose(b)
                free[n] = ws.pools.pools[ws.pools.index[f"cavity:{n}"]].vol + self.free_film(n, f, R)
        # water still running off surfaces into a drain, a bowl or onto the floor (1 g in the last second)
        free["__films_kg"] = T["film_kg"]
        free["__away_kg"] = T["drain_kg"] + float(sum(T["sinks"].values())) + T["pools_kg"]
        moving = []
        if flows:
            moving.append("water running")
        if T["packets_kg"] > 1e-3:
            moving.append("water in the air")
        if v > 2e-3:
            moving.append("an item moving")
        if prev is not None and any(abs(free[n] - prev.get(n, 0.0)) > max(2e-7, 0.02*free[n]) for n in free if not n.startswith("__")):
            moving.append("water still draining from an item")
        if prev is not None and (abs(free["__films_kg"] - prev["__films_kg"]) > 1e-3 or abs(free["__away_kg"] - prev["__away_kg"]) > 1e-3):
            moving.append("water still running off surfaces")
        return (prev is not None and not moving), free, moving

    def evaluate(self, settle_s=None):
        """The episode's verdict: a benchmark task's verifier.py (tasks/<task>/) gives it with its verify (which settles
        with evaluate_settled); a task outside the benchmark is evaluate_settled's."""
        own = task_verifier(self.task)
        if own is not None:
            return own.verify(self, settle_s)
        return self.evaluate_settled(settle_s)

    def evaluate_settled(self, settle_s=None):
        """Settle branch (robot holds; world advances), measure, restore. The branch runs for at least
        CRITERIA.settle_min_s, then second by second until the world is quiet -- no outlet flowing, less than 1 g of
        water in the air, no item faster than 2 mm/s, no item's free water changing by more than 0.2 mL or 2 % per
        second, less than 1 g/s running off the surfaces into drains, bowls or onto the floor -- or `settle_s` (default
        CRITERIA.settle_s) has passed. A finished, hands-off state is therefore
        verified in a few simulated seconds; a state still dripping, pouring or rolling gets the full time."""
        env = self.env
        key = (int(env.world.tick), env.generation)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        t_max = C["settle_s"] if settle_s is None else settle_s
        t_min = min(C["settle_min_s"], t_max)
        used, why = 0.0, "state faulted: not advanced"
        if env.world.fault is not None:
            # a faulted world cannot be advanced: the verdict is read from the state as it is (the fault gate fails)
            m = self.measure()
            crit = self.criteria(m)
        else:
            snap = env.checkpoint()
            settle_fault = None
            if self.pan is not None:
                self.pan.frozen = True               # the settle branch neither adds nor resets sanitizer time
            try:
                try:
                    env.step_free(int(round(t_min/P.DT)))
                    used = t_min
                    quiet, prev, moving = self._quiet(None)
                    why = "time limit"
                    while used < t_max - 1e-9:
                        dt = min(1.0, t_max - used)
                        env.step_free(int(round(dt/P.DT)))
                        used += dt
                        quiet, prev, moving = self._quiet(prev)
                        if quiet:
                            why = "quiet"
                            break
                    if why == "time limit" and moving:
                        why = "time limit (" + ", ".join(moving) + ")"
                except (RuntimeError, FloatingPointError) as exc:
                    settle_fault = f"during the settle branch: {exc}"
                if settle_fault is None:
                    m = self.measure()
                    crit = self.criteria(m)
            finally:
                env.restore(snap)
                if self.pan is not None:
                    self.pan.frozen = False
            if settle_fault is not None:
                # the branch faulted: the live episode is restored (not faulted), but this state cannot be verified
                m = self.measure()
                m["fault"] = settle_fault
                crit = self.criteria(m)
                why = settle_fault
        score, rw = self._score(crit, m)
        out = dict(task=self.task, succeeded=all(c["passed"] for c in crit.values()), score=score,
                   criteria=crit, measures=m, settle_s=round(used, 3), settle_limit_s=t_max, settle_stopped=why, tick=key[0])
        if rw is not None:
            # a faulted settle branch or an open water, detergent or soil ledger is not a task result (the backend refuses
            # it instead of returning a score)
            out["reward"] = rw
            out["verifiable"] = rw["verifiable"]
        self._cache = (key, out)
        return out
