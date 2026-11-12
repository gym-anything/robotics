"""Outcome of scrub-rinse-pan: the pan visibly clean of egg, free of detergent, sanitized for at least 30 s after a
qualifying entry, drained and set upside down on the wall shelf over the clean drainboard; the sponge back on the ledge
behind the compartments, no water running and nothing in the hands.

criteria(e, m, gate) adds the task's gates to task.Evaluator.criteria: e is the evaluator, m its measures
(task.Evaluator.measure, with the PanTracker record as m["pan"]), gate(name, ok, detail) records one. task.Evaluator
builds its PanTracker (e.pan) from this module and scores with REWARD_CONTRACT. verify(e, settle_s) gives the episode's
verdict, the dict of task.Evaluator.evaluate_settled (the settle branch of at most settle_s seconds, None: the default).

scrub-rinse-pan: its sanitizer time is the qualifying clock of PanTracker, not every full immersion. Time counts only while the pan's metal is
fully under the sanitizer and its inside is flooded (PAN_INSIDE_WET of the inner area under liquid; a mouth-down pan keeps
its air), only after an entry into the sanitizer at which the pan was visibly clean and carried at most its soap limit
(its own faces' detergent, read before any cell touched the sanitizer, plus the detergent of the water standing in it
when that water is stronger than rinse strength: the soap limit in the drained limit of water, 3.3 mg/L for this pan),
and it restarts when the pan is wetted again by the wash or rinse compartment, by more than PAN_STREAM_RINSE_ML from a
running outlet while outside every bowl, or by touching the floor. An entry that does not qualify (and a floor touch)
leaves a rinse owed, and no entry qualifies until the pan is rinsed again outside the sanitizer. The settle branch neither adds
nor resets that time. Its placement box is the shelf over the clean drainboard (the pan's centre over the board), the
sponge's ledge is the deck behind the three compartments, and "nothing in the hands" is no finger pad on the pan, the
sponge or the spray valve (a pad resting on the sink, the shelf or a closed faucet blade is not holding). Its score is
the progress contract (REWARD_CONTRACT): 1 if success else 0.9 x progress, progress = (C + C R + C R Z + C R Z T)/4
of the pan's own state, withheld (None) when the state is unverifiable; the fraction of its gates is reported beside
it as reward.mean_of_checks.
"""
import numpy as np

from surface_rinsing import layout as L, params as P
from surface_rinsing.task import GAUGE, PAN_FLOOR_Z, PAN_INSIDE_WET, PAN_STREAM_RINSE_ML

C = P.CRITERIA
# score contract (task.Evaluator._score): "progress-v1" (each stage credited on a pan that keeps the earlier ones, a
# continuous rinse term, no lock-out, None when unverifiable, and the stored stage also needs soap_free) = 1 if success
# else FAIL_SCORE_SCALE x progress (label_application's rule), progress read from the pan's own state; a task without
# one scores the fraction of its gates
REWARD_CONTRACT = "progress-v1"
# detergent a lifted and poured pan carries on its own faces (withdrawal film and suds: 13.47-13.74 mg measured at
# 0.75 g/L; an instant withdrawal carries 8.71 mg): the reference of the rinse progress term, scaled by the seed's wash
# strength
PAN_SURF_REF = 13.5e-6


class PanTracker:
    """scrub-rinse-pan: the sanitizer clock and carry-in, what "in the hands" means,
    and the progress terms, all read from the pan's own state. Sampled every 20 ticks (10 ms) by Evaluator._track;
    `frozen` (the evaluator's settle branch) stops it.

    Sanitizer clock (Evaluator.sanitize["pan"]; the gate needs C["sanitize_s"]):
      * counts while the pan's metal is fully under the sanitizer (frac > 0.999) AND at least PAN_INSIDE_WET
        of its inner area is under liquid (its cavity flooded through the submerged mouth, so the sanitizer reaches the
        food-contact face; a pan held mouth-down keeps its air and counts nothing);
      * only after a qualifying entry into the sanitizer: the pan visibly clean (every 1 cm gauge <= VRL), carrying at
        most its soap limit, and no rinse owed (below). Entry = the first sample at which any outer cell is under the
        sanitizer or its volume samples are; what it carried is the largest of the sample before any cell touched the
        sanitizer, the samples while its centre was over the sanitizer bowl, and the entry sample (cells that go under
        give their film to the bowl at once). Carried = detergent in solution, foam and
        adsorbed on its faces, plus the water standing in it when that water is stronger than rinse strength (c_rinse:
        the soap limit in the drained limit of water, 3.3 mg/L; a pan carried level full of rinse water is rinsed);
      * restarts (0) when the pan is wetted by the wash or rinse compartment, when more than PAN_STREAM_RINSE_ML from a
        running outlet lands on it while it is outside every bowl (a potable rinse after sanitizing), when it touches
        the floor (its origin below PAN_FLOOR_Z, the dropped-item line), or at an entry that does not qualify.
        Re-entering the sanitizer after a qualifying entry continues it, so it is the time in the sanitizer since the pan
        was last washed, rinsed or dropped.
    The order rule: an entry that does not qualify, and a touch of the floor, leave a rinse OWED. No later entry qualifies until rinse water has
    reached the pan's food-contact face outside the sanitizer: the pan's metal fully under the rinse compartment's water
    with at least PAN_INSIDE_WET of its inner area under liquid (the sanitizer clock's own immersion test; water the
    pan carries in its cavity does not make a graze a rinse), or more than
    PAN_STREAM_RINSE_ML of a running outlet's water gained on its inside (inner film plus cavity) while it is outside
    every bowl. No duration: a submerged cell gives its film to the bowl at once, so a full dunk is a rinse. A touch of
    the rinse water by the underside is not. A premature dip is therefore recoverable by
    the stated order (rinse, then sanitize), and using the sanitizer as the rinse (soapy in, out, back in) never counts.
    The report's `counting` is whether the clock runs at this sample; `session_qualified` whether the current entry
    qualified.
    Evaluator.surf_into_sanitizer["pan"] is the carry-in of the entry the current clock depends on (the largest over a
    qualifying session's re-entries; a non-qualifying entry's own figure, with the clock at 0)."""

    def __init__(self, env):
        self.env = env
        w, ws = env.world, env.water
        cells = ws.item_cells["pan"]
        self.sl = slice(int(cells[0]), int(cells[-1]) + 1)
        ch, ware = ws.item_charts["pan"], w.scene.ware["pan"]
        self.inner = np.isin(ch.cell_seg, ware.regions[ware.kind]["inner"])
        self.A = ws.surf.area[cells].astype(float)
        self.area = float(self.A.sum())
        pos = ws.surf.pos[cells] + 0.5*GAUGE*ws.surf.nrm[cells]         # the binning of gauges() (body frame)
        _, inv = np.unique(np.floor(pos/GAUGE).astype(np.int64), axis=0, return_inverse=True)
        self.gid = inv.ravel()
        self.garea = np.bincount(self.gid, weights=self.A)
        self.g0 = self.gauges(ws.eng.soil[self.sl].numpy())             # egg per gauge at the start
        self.soiled = self.g0 > P.VRL
        self.surf_lim = C["surf_per_item"]/0.085*self.area               # kg: the soap limit (16.5 ug on this pan)
        self.c_rinse = self.surf_lim/C["drained_bowl"]                   # kg/m3: rinse strength (3.3 mg/L)
        self.surf_ref = PAN_SURF_REF*env.spec.get("c_wash", P.SURFACTANT["c_wash"])/P.SURFACTANT["c_wash"]
        self.v_withdraw = PAN_SURF_REF/P.SURFACTANT["c_wash"]           # m3: the water that reference is (18 mL)
        ix = ws.pools.index
        self.k_san, self.k_cav = ix["sanitize"], ix["cavity:pan"]
        self.k_other = {ix["wash"]: "wash", ix["rinse"]: "rinse"}
        self.k_static = np.array([ix[n] for n in L.BOWLS])
        self.body = w.scene.items["pan"]["body"]
        self.over_san = (*L.BOWLS["sanitize"], L.BOWL_Y0, L.BOWL_Y1)
        self.shelf_x = [round(v, 3) for v in L.DRAINBOARDS["clean"]]
        # what a hand can hold here: the pan, the sponge (both bodies), the spray valve (body and lever). Static parts,
        # the faucet's blades and the hose are not held by a pad resting on them.
        mov = {}
        for n, it in w.scene.items.items():
            for k in ("body", "top"):
                if it.get(k) is not None and w._mj_body.get(it[k], -1) >= 0:
                    mov[int(w._mj_body[it[k]])] = n
        for n, f in w.scene.fixtures.items():
            if f["kind"] == "spray_valve":
                for k in ("valve", "lever"):
                    if f.get(k) is not None and w._mj_body.get(f[k], -1) >= 0:
                        mov[int(w._mj_body[f[k]])] = n
        self.movable = mov
        self.frozen = False
        self.state = dict(ok=False, live=False, owed=None, in_san=False, prev=None, window=None, outside_prev=False,
                          water_prev=None, water_in_prev=None, stream_mL=0.0, stream_in_mL=0.0, n_entries=0, entries=[],
                          n_resets=0, resets=[], rinses_after_owed=[])

    def snapshot(self):
        return {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v) for k, v in self.state.items()}

    def restore(self, s):
        self.state = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v) for k, v in s.items()}

    def gauges(self, soil):
        return np.bincount(self.gid, weights=soil*self.A)/np.maximum(self.garea, 1e-12)

    def carried(self, s, gamma, cav):
        """Detergent the pan carries (kg): its faces' solution, foam and adsorbed, plus its standing water's when that
        water is stronger than rinse strength."""
        tot = float(((s + gamma)*self.A).sum())
        cs = float(cav.sp[0] + cav.sp[5])
        if cav.vol <= 1e-12 or cs/cav.vol > self.c_rinse:
            tot += cs
        return tot

    def held(self, w):
        """Movable things a finger pad touches now (names)."""
        d, m = w.mjd, w.mjm
        out = set()
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom[0]), int(d.contact[i].geom[1])
            for a, b in ((g1, g2), (g2, g1)):
                if a in w._pad_geoms:
                    n = self.movable.get(int(m.geom_bodyid[b]))
                    if n is not None:
                        out.add(n)
        return sorted(out)

    def _reset(self, ev, t, why):
        st = self.state
        st["n_resets"] += 1
        st["resets"] = (st["resets"] + [dict(t=t, sanitized_s_before=round(ev.sanitize["pan"], 2), why=why)])[-5:]
        ev.sanitize["pan"] = 0.0
        st["ok"] = False

    def sample(self, ev):
        if self.frozen:
            return
        st, ws, w = self.state, self.env.water, self.env.world
        E, sl = ws.eng, self.sl
        sub = E.sub[sl].numpy()
        sp = E.sp[:, sl].numpy()
        s, gamma, soil, h = sp[0] + sp[5], E.gamma[sl].numpy(), E.soil[sl].numpy(), E.h[sl].numpy()
        cav = ws.pools.pools[self.k_cav]
        pool, frac = ws.immersed.get("pan", (None, 0.0))
        so = sub[~self.inner]
        t = round(float(w.sim_time), 3)
        touch = pool == "sanitize" or bool((so == self.k_san).any())
        other = pool if pool in ("wash", "rinse") else next((n for k, n in self.k_other.items() if (so == k).any()), None)
        in_bowl = pool is not None or bool(np.isin(so, self.k_static).any())
        now = dict(t=t, carried=self.carried(s, gamma, cav), gauge=float(self.gauges(soil).max()))
        p, _ = w.body_pose(self.body)
        Ai = self.A[self.inner]
        wet_in = float((Ai*(sub[self.inner] >= 0)).sum()/Ai.sum())       # share of the inside under liquid
        rinsed_now = None                                   # how the inside was rinsed outside the sanitizer, this sample
        if other is not None:
            if st["ok"] or ev.sanitize["pan"] > 0:
                self._reset(ev, t, f"wetted by the {other} compartment's water")
            if other == "rinse" and pool == "rinse" and frac > 0.999 and wet_in >= PAN_INSIDE_WET:
                rinsed_now = dict(by="the rinse compartment's water, the inside flooded", inside_under=round(wet_in, 4))
        if p[2] < PAN_FLOOR_Z:
            if st["ok"] or ev.sanitize["pan"] > 0:
                self._reset(ev, t, "touched the floor")
            if st["owed"] is None:
                st["owed"] = dict(t=t, why="touched the floor")
        if touch and not st["in_san"]:
            cands = [now] + [c for c in (st["prev"], st["window"]) if c is not None]
            into, g_in = max(c["carried"] for c in cands), max(c["gauge"] for c in cands)
            clean, within = g_in <= P.VRL, into <= self.surf_lim
            qualifies = clean and within and st["owed"] is None
            st["n_entries"] += 1
            st["entries"] = (st["entries"] + [dict(t=t, carried_ug=round(into*1e9, 3), soil_gauge_max_ug_cm2=round(g_in*1e5, 3),
                                                   visibly_clean=bool(clean), within_soap_limit=bool(within),
                                                   rinse_owed=st["owed"] is not None, qualifies=bool(qualifies))])[-5:]
            if qualifies:
                prior = ev.surf_into_sanitizer.get("pan", 0.0) if ev.sanitize["pan"] > 0 else 0.0
                ev.surf_into_sanitizer["pan"] = max(prior, into)
                if other is None:
                    st["ok"] = True
            else:
                why = ("not visibly clean" if not clean else "carrying detergent" if not within else
                       f"before a rinse owed since t={st['owed']['t']} ({st['owed']['why']})")
                ev.surf_into_sanitizer["pan"] = into
                self._reset(ev, t, "entered the sanitizer " + why)
                if st["owed"] is None:
                    st["owed"] = dict(t=t, why="entered the sanitizer " + why)
            st["window"], st["stream_mL"], st["stream_in_mL"] = None, 0.0, 0.0
        st["in_san"] = touch
        # a running outlet's water landing on the pan outside every bowl: any 5 mL on the pan restarts a sanitized pan's
        # clock (a potable rinse after sanitizing); 5 mL gained on its INSIDE (inner film + cavity) is a rinse of the
        # food-contact face, which clears an owed rinse (a graze of the rinse water by the underside does not)
        water_in = float((h[self.inner]*self.A[self.inner]).sum()) + cav.vol
        water = float((h*self.A).sum()) + cav.vol
        outside = not touch and not in_bowl
        if outside and st["outside_prev"] and st["water_prev"] is not None and \
                any(o.Q/P.GPM > 1e-4 for o in ws.outlets.values()):
            st["stream_mL"] += max(0.0, water - st["water_prev"])*1e6
            wp_in = st.get("water_in_prev")
            st["stream_in_mL"] = st.get("stream_in_mL", 0.0) + max(0.0, water_in - (water_in if wp_in is None else wp_in))*1e6
            if st["stream_mL"] > PAN_STREAM_RINSE_ML:
                if st["ok"] or ev.sanitize["pan"] > 0:
                    self._reset(ev, t, f"rinsed by a running outlet after sanitizing ({st['stream_mL']:.1f} mL)")
                st["stream_mL"] = 0.0
            if st["stream_in_mL"] > PAN_STREAM_RINSE_ML:
                rinsed_now = dict(by="a running outlet's water on the inside", inside_gain_mL=round(st["stream_in_mL"], 2))
                st["stream_in_mL"] = 0.0
        st["outside_prev"], st["water_prev"], st["water_in_prev"] = outside, water, water_in
        if rinsed_now is not None and st["owed"] is not None and p[2] >= PAN_FLOOR_Z:
            st["rinses_after_owed"] = (st["rinses_after_owed"] + [dict(t=t, owed=st["owed"], **rinsed_now)])[-5:]
            st["owed"] = None
        st["live"] = bool(st["ok"] and pool == "sanitize" and frac > 0.999 and wet_in >= PAN_INSIDE_WET)
        if st["live"]:
            ev.sanitize["pan"] += 20*P.DT
        if not touch:
            st["prev"] = now
            x0, x1, y0, y1 = self.over_san
            if x0 < p[0] < x1 and y0 < p[1] < y1:
                wd = st["window"]
                st["window"] = now if wd is None else dict(t=t, carried=max(wd["carried"], now["carried"]),
                                                           gauge=max(wd["gauge"], now["gauge"]))
            else:
                st["window"] = None

    def measure(self, ev, f, held, any_pad_contact):
        """The pan's progress quantities and the sanitizer record (Evaluator.measure)."""
        ws, st, cells = self.env.water, self.state, self.sl
        # C: area-weighted over the gauges soiled at the start, the share of the way (in log) from the start load to VRL
        if self.soiled.any():
            g0, g = self.g0[self.soiled], np.maximum(self.gauges(f["soil"][cells])[self.soiled], 1e-30)
            lp = np.clip(np.log(g0/g)/np.log(g0/P.VRL), 0.0, 1.0)
            clean = float((lp*self.garea[self.soiled]).sum()/self.garea[self.soiled].sum())
        else:
            clean = 1.0
        # R: the same, from what a poured pan carries out of the wash (S_ref) to its soap limit. S = the faces' detergent
        # plus the detergent of the water the pan would take away: the bowl's strength times the reference film volume
        # while it is immersed, its own standing water (up to that volume) when it is not. Continuous in both; 0 in the
        # wash by construction
        cav = ws.pools.pools[self.k_cav]
        faces = float(((f["s"][cells] + f["gamma"][cells])*self.A).sum())
        c_cav = float(cav.sp[0] + cav.sp[5])/cav.vol if cav.vol > 1e-12 else 0.0
        water_kg = c_cav*min(cav.vol, self.v_withdraw)
        pool = ws.immersed.get("pan", (None, 0.0))[0]
        if pool is not None:
            pb = ws.pools.pools[ws.pools.index[pool]]
            c_bowl = float(pb.sp[0] + pb.sp[5])/pb.vol if pb.vol > 1e-12 else 0.0
            water_kg = max(water_kg, c_bowl*self.v_withdraw)
        S = faces + water_kg
        rinsed = float(np.clip(np.log(self.surf_ref/max(S, 1e-15))/np.log(self.surf_ref/self.surf_lim), 0.0, 1.0))
        return dict(held=held, any_pad_contact=any_pad_contact, clean=round(clean, 4), rinsed=round(rinsed, 4),
                    rinse_term_detergent_ug=round(S*1e9, 3),
                    carried_ug=round(self.carried(f["s"][cells], f["gamma"][cells], cav)*1e9, 3),
                    sanitizer=dict(counting=bool(st.get("live", False)), session_qualified=bool(st["ok"]),
                                   rinse_owed=st["owed"],
                                   rinse_strength_mg_L=round(self.c_rinse*1e3, 3), entries=list(st["entries"]),
                                   n_entries=st["n_entries"], resets=list(st["resets"]), n_resets=st["n_resets"],
                                   rinses_after_owed=list(st["rinses_after_owed"])))

    def progress(self, m, crit):
        """progress = (C + C R + C R Z + C R Z T)/4: C clean, R rinsed, Z sanitized (qualifying time / 30 s), T stored
        (on the shelf upside down, drained and passing soap_free). Each stage is credited only on a pan that still has
        every earlier outcome (a pan sanitized and then soaped again is not nearly finished).
        T reads soap_free so that progress 1 is exactly "every pan gate passes": R is a mass reading, and a pan with a
        residual patch above the gauge cap could otherwise read 1 on every term. No term reads
        how, in what order or with which tool or hand the pan got there; the sponge, the water, the hands and the floor
        are success conditions only."""
        p = m["pan"]
        cl, ri = p["clean"], p["rinsed"]
        z = min(m["items"]["pan"]["sanitized_s"]/C["sanitize_s"], 1.0)
        tt = 1.0 if all(crit[k]["passed"] for k in ("placed", "drained", "soap_free")) else 0.0
        return dict(progress=round((cl + cl*ri + cl*ri*z + cl*ri*z*tt)/4, 4),
                    terms=dict(clean=cl, rinsed=ri, sanitized=round(z, 4), stored=tt))


def criteria(e, m, gate):
    items = m["items"]
    r = items["pan"]
    ws = e.env.water
    A = float(ws.surf.area[ws.item_cells["pan"]].sum())
    gate("visibly_clean", r["soil_gauge_max_ug_cm2"] <= P.VRL*1e5, dict(gauge_max=r["soil_gauge_max_ug_cm2"]))
    gate("soap_free", r["surfactant_ug"]*1e-9/A <= C["surf_per_item"]/0.085 and
         r["surfactant_gauge_max_mg_m2"] <= C["surf_gauge"]*1e6 and r["foam_mL"] < 1e-3,
         dict(mean_mg_m2=round(r["surfactant_ug"]*1e-3/A, 5), gauge_max=r["surfactant_gauge_max_mg_m2"]))
    # sanitized_s: PanTracker's qualifying clock; into_ug: the carry-in of the entry that clock depends on
    gate("sanitized", r["sanitized_s"] >= C["sanitize_s"] and r["surfactant_into_sanitizer_ug"]*1e-9/A <= C["surf_per_item"]/0.085,
         dict(sanitized_s=r["sanitized_s"], into_ug=r["surfactant_into_sanitizer_ug"],
              limit_ug=round(C["surf_per_item"]/0.085*A*1e9, 3), **m["pan"]["sanitizer"]))
    gate("drained", r.get("free_water_mL", 0.0) <= C["drained_bowl"]*1e6,
         dict(free_water_mL=r.get("free_water_mL", 0.0), pooled_mL=r.get("pooled_mL", 0.0)))
    gate("placed", e._on_shelf("pan") and r["tilt_deg"] > 180 - C["place_tilt_deg"],
         dict(position=r["position"], tilt_deg=r["tilt_deg"], centre_x_between_m=e.pan.shelf_x))
    gate("sponge_on_ledge", e._on_ledge("sponge", compartments=True), dict(position=items["sponge"]["position"]))


def verify(e, settle_s=None):
    return e.evaluate_settled(settle_s)
