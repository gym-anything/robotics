"""Glass qualifying-entry tracker, adapted from task.PanTracker.

The original PanTracker stays byte-identical. This copy uses each glass's name,
10 ug carry-in limit and 2 mL drained limit. A premature dip is recoverable only
by rinsing outside sanitizer; immersion must flood the food-contact inside.
"""
import numpy as np
from .. import params as P, layout as L
from ..task import GAUGE, PAN_INSIDE_WET, PAN_STREAM_RINSE_ML, PAN_FLOOR_Z


class GlassTracker:
    def __init__(self, env, name):
        self.env, self.name = env, name
        w, ws = env.world, env.water
        cells = ws.item_cells[name]
        self.sl = slice(int(cells[0]), int(cells[-1]) + 1)
        ch, ware = ws.item_charts[name], w.scene.ware[name]
        self.inner = np.isin(ch.cell_seg, ware.regions[ware.kind]["inner"])
        self.A = ws.surf.area[cells].astype(float)
        pos = ws.surf.pos[cells] + 0.5*GAUGE*ws.surf.nrm[cells]
        _, inv = np.unique(np.floor(pos/GAUGE).astype(np.int64), axis=0, return_inverse=True)
        self.gid = inv.ravel()
        self.garea = np.bincount(self.gid, weights=self.A)
        self.surf_lim = P.CRITERIA["surf_per_item"]
        self.c_rinse = self.surf_lim/P.CRITERIA["drained_glass"]
        ix = ws.pools.index
        self.k_san, self.k_cav = ix["sanitize"], ix[f"cavity:{name}"]
        self.k_other = {ix["wash"]: "wash", ix["rinse"]: "rinse"}
        self.k_static = np.array([ix[n] for n in L.BOWLS])
        self.body = w.scene.items[name]["body"]
        self.over_san = (*L.BOWLS["sanitize"], L.BOWL_Y0, L.BOWL_Y1)
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
        """Detergent the item carries (kg): its faces' solution, foam and adsorbed, plus its standing water's when that
        water is stronger than rinse strength."""
        tot = float(((s + gamma)*self.A).sum())
        cs = float(cav.sp[0] + cav.sp[5])
        if cav.vol <= 1e-12 or cs/cav.vol > self.c_rinse:
            tot += cs
        return tot

    def _reset(self, ev, t, why):
        st = self.state
        st["n_resets"] += 1
        st["resets"] = (st["resets"] + [dict(t=t, sanitized_s_before=round(ev.sanitize[self.name], 2), why=why)])[-5:]
        ev.sanitize[self.name] = 0.0
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
        pool, frac = ws.immersed.get(self.name, (None, 0.0))
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
            if st["ok"] or ev.sanitize[self.name] > 0:
                self._reset(ev, t, f"wetted by the {other} compartment's water")
            if other == "rinse" and pool == "rinse" and frac > 0.999 and wet_in >= PAN_INSIDE_WET:
                rinsed_now = dict(by="the rinse compartment's water, the inside flooded", inside_under=round(wet_in, 4))
        if p[2] < PAN_FLOOR_Z:
            if st["ok"] or ev.sanitize[self.name] > 0:
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
                prior = ev.surf_into_sanitizer.get(self.name, 0.0) if ev.sanitize[self.name] > 0 else 0.0
                ev.surf_into_sanitizer[self.name] = max(prior, into)
                if other is None:
                    st["ok"] = True
            else:
                why = ("not visibly clean" if not clean else "carrying detergent" if not within else
                       f"before a rinse owed since t={st['owed']['t']} ({st['owed']['why']})")
                ev.surf_into_sanitizer[self.name] = into
                self._reset(ev, t, "entered the sanitizer " + why)
                if st["owed"] is None:
                    st["owed"] = dict(t=t, why="entered the sanitizer " + why)
            st["window"], st["stream_mL"], st["stream_in_mL"] = None, 0.0, 0.0
        st["in_san"] = touch
        # a running outlet's water landing on the item outside every bowl: any 5 mL on the item restarts a sanitized item's
        # clock (a potable rinse after sanitizing); 5 mL gained on its INSIDE (inner film + cavity) is a rinse of the
        # food-contact face, which clears an owed rinse (a graze of the rinse water on the underside does not)
        water_in = float((h[self.inner]*self.A[self.inner]).sum()) + cav.vol
        water = float((h*self.A).sum()) + cav.vol
        outside = not touch and not in_bowl
        if outside and st["outside_prev"] and st["water_prev"] is not None and \
                any(o.Q/P.GPM > 1e-4 for o in ws.outlets.values()):
            st["stream_mL"] += max(0.0, water - st["water_prev"])*1e6
            wp_in = st.get("water_in_prev")
            st["stream_in_mL"] = st.get("stream_in_mL", 0.0) + max(0.0, water_in - (water_in if wp_in is None else wp_in))*1e6
            if st["stream_mL"] > PAN_STREAM_RINSE_ML:
                if st["ok"] or ev.sanitize[self.name] > 0:
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
            ev.sanitize[self.name] += 20*P.DT
        if not touch:
            st["prev"] = now
            x0, x1, y0, y1 = self.over_san
            if x0 < p[0] < x1 and y0 < p[1] < y1:
                wd = st["window"]
                st["window"] = now if wd is None else dict(t=t, carried=max(wd["carried"], now["carried"]),
                                                           gauge=max(wd["gauge"], now["gauge"]))
            else:
                st["window"] = None
