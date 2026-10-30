"""Task evaluator: physically meaningful criteria read from the world, with a settle rule.

`progress()` (every state, cheap): the criteria on the film as it is now (wet paint counted by the dry film it will
leave, DFT = dry solids + wet solids).
`evaluate()` (verification): the film fields are copied and only the film equations (flow, levelling, drying,
absorption) are advanced until every wet cell has set or 30 min of film time have passed -- the deterministic
physical future of the actual film (sag, curtains and laps appear if they will) -- then measured, and the live film is
restored untouched. The rigid world is not advanced.

Criteria (thresholds: painting.params.EVAL):
  hide        the dry stack within dE*ab 1.0 of the paint's own colour on >= 98 % of the target (bands 95 %); artist
              acrylic and aerosol coats use their own DFT ranges (aerosol: 5 mm gauge readings)
  runs        walls: progress() fails a wet film thicker than 250 um; verify fails a film that travelled > 1 mm during
              the settle (the settle shows whether a thick film will sag; level faces: the wet thickness at declaration)
  laps        visible lap marks (fresh paint over a set film of >= 4 um solids) <= 1 cm2 inside the target
  uniformity  dry appearance of 50 mm patches within dE*ab 1.0 of the target's median
  protected   paint on ceiling, trim, outlet plate, radiator (cells with DFT > 1 um, plus the 4 mm voxels of sink
              surfaces holding >= 1 um) <= 0.5 cm2
  untouched   wet target film touched by a robot link or a tool's non-applicator part <= 0.5 cm2
  tools       roll-strip: the roller rests in the tray, released
behind-radiator judges the wall it leaves (BEHIND_RADIATOR below),
all read from the settled film, never from the order, route or timing of the work:
  hide        the band but the pocket the brush held by the robot's hand at its handle's end cannot reach (the bottom
              of the middle, x 0.525-0.580 m below z 0.35 m) hidden on >= 98 %, with no connected unhidden patch
              (8-connected cells) larger than 20 cm2: up to ~77 cm2 of the scored band may stay at the first coat, in
              patches of up to 20 cm2
  runs        a sag, curtain or drip left in the wall: band cells into which the film's flow carried more than 50 um of
              dry paint (`flowed`, net of what an applicator took back up), on <= 1 cm2. Coats laid over set paint
              stack without flowing and are not runs, however many. The band's bottom three 2 mm cell rows (centres
              z 0.083-0.087 m, ~5 mm of wall above the baseboard top), where creeping film stops at the chart's
              closed edge, are not read. progress() also fails a wet film over run_h
  marks       a mark left in the coat by a link or a tool part: cells a skin touched while wet on which no applicator
              has since laid paint (>= 5 um wet per cell) or picked up wet film (`no_marks`), <= 0.5 cm2 (one 4 mm
              contact marks 0.52 cm2); a mark brushed out or re-coated, wet or after the coat has set, is not one;
              an empty brush passing over it does not repair it
  protected   also the trim faces without a chart (apron underside, stool nosing, baseboard ends): radiator and trim;
              drops already falling when the episode ends are landed first (a brush's pendant volume is not)
  laps        reported, not part of success (the instruction asks for full hide behind a radiator, not lap-free sheen)
  score       1 on success, else 0.9 x progress: 1 - the area still to paint before hide passes (relative to the start),
              less the cells marked or run and 10 x the paint on the radiator or trim (as wall area; see _reward)
A non-finite state or a ledger fault is not a task result: the backend reports it as a fault.

The explicit painting-tasks-v1 selection uses scoring.TaskProgress and the corrected
checks marked in Evaluator.measure. Bare selections retain the legacy predicates
above; behind-radiator always retains its own progress-v1 contract.
"""

import math
import numpy as np

from . import params as P
from . import layout as L
from .paint import delta_e

E = P.EVAL

# behind-radiator's verdict and reward contract. Coverage is full hide of the
# band the task's own tool reaches: the 355 mm dog-leg brush held at its handle's end by the FR3 hand reaches the band
# from both ends and over the radiator top except the bottom of the middle -- hand-clear, pushed from the ends along the
# bottom row it hides the wall to x 0.531 m from the left and from 0.573 m from the right, and pushed down over the
# radiator top at x 0.55 it lays paint down to z 0.341 m (phantom-held, the hand's volume checked along the actual
# trajectory). That pocket, with a
# few mm of margin (`pocket`, 145 cm2 of the 4,006 cm2 band), is not scored; the rest must hide on >= 98 % (the strip
# task's full hide) with no unhidden patch over 20 cm2 (a brushed coat's stroke-edge near-misses: 31 cm2 in 15 patches,
# largest 3.9 cm2, in the accepted assisted episode). A threshold of 90 % of the whole band would pass a 400 cm2 holiday
# -- the whole middle-bottom, or the bottom 60 mm -- as full hide.
# Runs and marks are read from the settled film (cumulative creep would fail an even coat laid off ten times, and a
# mark flag cleared by an empty brush would pass a mark left in the coat). A run is paint the film's flow
# piled up: more than run_flow (50 um dry) carried into a cell by flow. A thickness test (new paint over the 105 um a
# wet film at the 250 um sag limit leaves) would fail coats stacked over set paint -- a second coat at 125 um, a
# touch-up after the coat set -- and pass a sag lip on the thinner coat below a 280 um patch.
# run_flow is calibrated on the environment's own sag criterion (a front travelling > 1 mm, EVAL sag_move): a 250 um
# patch in a 100 um coat, travel 0.995 mm, carries 42.9 um in; a 280 um patch, travel 1.32 mm, 60.6 um. Honest work
# stays below: laying off ten times <= 0.9 um, a sag brushed out 32 um, stacked coats and touch-ups <= 12 um, a 100 um
# coat brushed back and forth 30 s 40.3 um at its lower edge.
# A mark is a touched spot no applicator has laid real paint on (>= 5 um wet; an empty brush's 0.1 um is not) or picked
# wet film up from since; a thinner-than-the-surrounding-coat test would also fail a mark re-coated after the coat
# set.
# Reward: score 1 on success, else fail_scale x progress (label_application's rule), progress a function of the final
# film only (no milestones, order or route): 1 - the area still to paint before the hide check passes (the larger of
# the unhidden area beyond the 2 % allowance and the patches' excess over 20 cm2) relative to that area at the start,
# less the hidden cells marked or run beyond their tolerances and damage_weight x the paint on the radiator or trim
# beyond its tolerance, all as wall area. damage_weight is a judgement: wiping latex off a cast-iron radiator's columns
# or white trim (or, dried, scraping and repainting it) is far more work than brushing the same area of wall; 0.2 mL on
# the radiator (20 cm2) costs 5 % of the job, a 4 mL dump all of it.
BEHIND_RADIATOR = dict(hide_frac=0.98, holiday_cm2=20.0, pocket=dict(x0=0.525, x1=0.580, z1=0.350), run_area=1e-4,
                       run_edge=0.004, run_flow=50e-6, damage_weight=10.0, contract="progress-v1", fail_scale=0.9)
TRIM_SINKS = frozenset(n for n, _, _, mat in L.static_boxes() if mat == "trim")


def _rect_cells(ch, x0, x1, z0, z1, axis_origin=None):
    """Mask of chart cells whose centres lie in the world rectangle (wall charts: u = x, v = z)."""
    xs = ch.origin[0] + (np.arange(ch.nu) + 0.5)*ch.dx
    zs = ch.origin[2] + (np.arange(ch.nv) + 0.5)*ch.dx
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    return (X >= x0) & (X <= x1) & (Z >= z0) & (Z <= z1) & ch.mask


def _paper_xy(ch):
    xs = (np.arange(ch.nu) + 0.5)*ch.dx - ch.size[0]/2
    ys = (np.arange(ch.nv) + 0.5)*ch.dx - ch.size[1]/2
    return np.meshgrid(xs, ys, indexing="ij")


def _gauge(dft, mask, dx):
    """Dry film as a DFT gauge reads it: the mean over its probe's footprint (5 mm, E['gauge']) inside the coated area.
    Spray coats are judged by gauge readings (SSPC-PA 2 spot readings) -- a single 1 mm cell carries the spray
    parcels' discreteness, not the coat (max/min over cells can be 99 on an even coat)."""
    from scipy.ndimage import uniform_filter
    k = max(1, int(round(E["gauge"]/dx)))
    m = mask.astype(float)
    num = uniform_filter(dft*m, k, mode="constant")
    den = uniform_filter(m, k, mode="constant")
    return np.where(den > 1e-9, num/np.maximum(den, 1e-9), 0.0)


S_CURVE = lambda t: np.stack([-0.10 + 0.20*t, 0.035*np.sin(2*math.pi*t)], -1)        # paper frame, m


def stroke_width(t):
    """Template width along the S stroke: 2 -> 6 -> 2 mm (half-sine swell): a #8 round's thick-and-thin stroke, from a
    hairline on the tip to the belly pressed to ~6 mm (its 6-8 mm spread)."""
    return 2e-3 + 4e-3*np.sin(math.pi*t)


class Evaluator:
    def __init__(self, env):
        self.env = env
        self.spec = env.spec
        self.task = env.spec["task"]
        self.paint = env.paint
        self.targets = self._targets()
        self._decl_h = {}
        self._drops0 = 0
        from .catalogue import CONTRACT, selected_contract
        self.contract = selected_contract(self.task, getattr(getattr(env, "config", None), "contract", None))
        self._task_progress = None
        if self.contract == CONTRACT:
            from .scoring import TaskProgress
            self._task_progress = TaskProgress(self)
        if self.task == "behind-radiator":
            # the band the brush reaches (BEHIND_RADIATOR: the bottom of the middle is not scored), the band's bottom
            # row on the baseboard (not read for runs), and the area still to paint at the start: the reward's unit
            C, w = BEHIND_RADIATOR, self.paint.charts["wall"]
            pk, sl = C["pocket"], self._bbox("wall", self.targets["wall"])
            self._pocket = _rect_cells(w, pk["x0"], pk["x1"], L.BEHIND_RAD["z0"], pk["z1"]) & self.targets["wall"]
            self._edge = _rect_cells(w, -1.0, 9.0, L.BEHIND_RAD["z0"], L.BEHIND_RAD["z0"] + C["run_edge"])[sl]
            hid, m = self._hidden(False)
            ms = m & ~self._pocket[sl]
            self._remaining0 = max(self._remaining(hid, ms, w.area)[0], w.area)

    # ------------------------------------------------------------------ geometry of the targets
    def _targets(self):
        t, ch = self.task, self.paint.charts
        out = {}
        if "wall" in ch:
            w = ch["wall"]
            if t == "roll-strip":
                s = L.STRIP
                out["wall"] = _rect_cells(w, s["x0"], s["x1"], s["z0"], s["z1"])
            elif t == "cut-in-ceiling":
                b = L.CEILING_BAND
                out["wall"] = _rect_cells(w, b["x0"], b["x1"], b["z0"], b["z1"])
                out["corner"] = _rect_cells(w, b["x0"], b["x1"], L.Z - 2*w.dx, L.Z)
            elif t == "cut-in-trim":
                b, o = L.CASING_BAND, L.OUTLET_BAND
                out["wall"] = _rect_cells(w, b["x0"], b["x1"], b["z0"], b["z1"]) | _rect_cells(w, o["x0"], o["x1"], o["z0"], o["z1"])
            elif t == "behind-radiator":
                b = L.BEHIND_RAD
                out["wall"] = _rect_cells(w, b["x0"], b["x1"], b["z0"], b["z1"])
            elif t == "touch-up":
                m = np.zeros_like(w.mask)
                xs = w.origin[0] + (np.arange(w.nu) + 0.5)*w.dx; zs = (np.arange(w.nv) + 0.5)*w.dx
                X, Z = np.meshgrid(xs, zs, indexing="ij")
                for (sx, sz) in self.spec.get("scuffs", []):
                    m |= (((X - sx)/0.035)**2 + ((Z - sz)/0.012)**2 < 1)
                out["wall"] = m & w.mask
                out["halo"] = np.zeros_like(m)
                for (sx, sz) in self.spec.get("scuffs", []):
                    out["halo"] |= (((X - sx)/0.08)**2 + ((Z - sz)/0.05)**2 < 1)
                out["halo"] &= w.mask & ~out["wall"]
        if "paper" in ch:
            X, Y = _paper_xy(ch["paper"])
            if t == "fill-circle":
                r = np.hypot(X, Y)
                out["paper"] = r <= 0.030
                out["outside"] = r > 0.030 + E["overrun"]
            elif t == "swatch-card":
                out["paper"] = (np.abs(X) <= 0.025) & (np.abs(Y) <= 0.025)
                out["outside"] = (np.abs(X) > 0.025 + E["overrun"]) | (np.abs(Y) > 0.025 + E["overrun"])
            elif t == "brush-stroke":
                ts = np.linspace(0, 1, 400)
                c = S_CURVE(ts)
                d = np.full(X.shape, np.inf); half = np.zeros(X.shape)
                for t_, (cx, cy) in zip(ts, c):
                    dd = np.hypot(X - cx, Y - cy)
                    closer = dd < d
                    d = np.where(closer, dd, d)
                    half = np.where(closer, stroke_width(t_)/2, half)
                # "within 1 mm of the template width": the painted width may fall short by 1 mm (0.5 mm a side), so
                # coverage is required on the template's core; paint beyond 1 mm outside its edge is overrun (the first
                # build required its exact 1 mm-cell edge, which no stroke reproduces cell for cell)
                out["paper"] = d <= np.maximum(half - 0.5e-3, 0.5e-3)
                out["outside"] = d > half + E["overrun"]
        if "crate" in ch:
            out["crate"] = self._aperture_on_crate()
        if "print" in ch:
            out["print"] = ch["print"].mask.copy()
        return out

    def _aperture_on_crate(self):
        """Crate cells under the stencil's apertures at the start pose (the stencil is not to be moved)."""
        w = self.env.world
        bq = w.state.body_q.numpy()
        cr, st = self.paint.charts["crate"], self.paint.charts["stencil"]
        xs = (np.arange(cr.nu) + 0.5)*cr.dx; ys = (np.arange(cr.nv) + 0.5)*cr.dx
        U, V = np.meshgrid(xs, ys, indexing="ij")
        pts = cr.world(U.ravel(), V.ravel(), bq)
        su, sv, _ = st.local(pts, bq)
        hole = self.paint._stencil_holes(np.stack([su - st.size[0]/2, sv - st.size[1]/2], 1))
        return hole.reshape(cr.nu, cr.nv)

    # ------------------------------------------------------------------ measurements
    def _protected(self, trim_sinks=False):
        """Painted area on protected charts (DFT > 1 um) and in protected sinks (m2). trim_sinks: also the trim faces
        that have no chart and send paint to their primitive's sink (apron underside, stool nosing and underside,
        baseboard ends)."""
        area = 0.0
        detail = {}
        cache = self.__dict__.setdefault("_prot_cache", {})
        for n, ch in self.paint.charts.items():
            if ch.role != "protected":
                continue
            hit = cache.get(n)
            if hit is not None and hit[0] == ch.version:
                a = hit[1]
            else:
                a = float(((ch.projected_dft() > 1e-6) & ch.mask).sum())*ch.area if (ch.bbox is not None or ch.d.any()) else 0.0
                cache[n] = (ch.version, a)
            if a > 0:
                detail[n] = a
            area += a
        for n, s in self.paint.sinks.items():
            if n in ("radiator", "rad_pipe", "glazing", "ceiling") or n.startswith("casing") or n == "outlet_plate" or \
                    (trim_sinks and n in TRIM_SINKS):
                vol = float(s["solids"].sum() + s["vehicle"])
                a = max(s["area"], vol/100e-6)
                detail[f"sink:{n}"] = a
                area += a
        return area, detail

    def _bbox(self, key, mask):
        cache = self.__dict__.setdefault("_bb", {})
        if key not in cache:
            ii, jj = np.nonzero(mask)
            cache[key] = (slice(int(ii.min()), int(ii.max()) + 1), slice(int(jj.min()), int(jj.max()) + 1)) if len(ii) else \
                (slice(0, 1), slice(0, 1))
        return cache[key]

    def _appearance(self, ch, mask):
        """Dry-film appearance (linear RGB of the stack with wet solids counted as dry) of 50 mm patches in the target."""
        from .paint import stack_reflectance
        sl = self._bbox(f"app_{ch.name}", mask)
        d = (ch.d + ch.s)[(slice(None),) + sl]
        R = stack_reflectance(ch.paints, d, np.zeros_like(d), np.zeros_like(ch.w[sl]), ch.substrate_rgb[sl])
        mask = mask[sl]
        k = max(1, int(round(0.05/ch.dx)))
        ii, jj = np.nonzero(mask)
        if not len(ii):
            return None
        pi, pj = ii//k, jj//k
        keys = pi*100000 + pj
        uk, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
        col = np.zeros((len(uk), 3))
        np.add.at(col, inv, R[ii, jj])
        col /= cnt[:, None]
        full = cnt >= 0.8*k*k
        if not np.any(full):
            return None
        col = col[full]
        med = np.median(col, axis=0)
        return float(delta_e(col, med[None]).max())

    def _hidden(self, settled):
        """(hidden cells, target) over the wall target's bounding box. Hiding is a colour criterion (ASTM D2805's
        contrast ratio 0.98 is a just-noticeable difference): a cell hides when its dry stack (wet solids counted dry)
        is within dE*ab 1.0 of the paint's own infinitely thick colour, over whatever old colour is under it -- a dark
        paint hides beige at a fraction of the white base's ~30 um, so no single DFT threshold fits every colour."""
        from .paint import stack_reflectance
        w = self.paint.charts["wall"]
        sl = self._bbox("wall", self.targets["wall"])
        hc = self.__dict__.setdefault("_hide_cache", {})
        if hc.get("key") != (w.version, settled):
            Rh = stack_reflectance(w.paints, (w.d + w.s)[(slice(None),) + sl], np.zeros_like(w.s[(slice(None),) + sl]),
                                   np.zeros_like(w.w[sl]), w.substrate_rgb[sl])
            hc.update(key=(w.version, settled), hid=delta_e(Rh, w.paints[0].rinf[None, None]) <= E["hide_dE"])
        return hc["hid"], self.targets["wall"][sl]

    def measure(self, settled):
        t, ch = self.task, self.paint.charts
        out = dict(task=t, settled=settled, contract=self.contract)
        checks = {}
        diag = {}
        br = t == "behind-radiator"
        C = BEHIND_RADIATOR
        if "wall" in self.targets:
            w = ch["wall"]
            m_full = self.targets["wall"]
            sl = self._bbox("wall", m_full)
            dft = (w.d[(slice(None),) + sl] + w.s[(slice(None),) + sl]).sum(axis=0)
            hid, m = self._hidden(settled)
            ms = (m & ~self._pocket[sl]) if br else m             # the scored cells (behind-radiator: not the pocket)
            frac = float(hid[ms].mean()) if ms.any() else 0.0
            need = C["hide_frac"] if br else dict(**{"roll-strip": E["hide_frac"]}).get(t, 0.95)
            if t == "touch-up":
                halo = self.targets["halo"][sl]
                from .paint import stack_reflectance
                R = stack_reflectance(w.paints, (w.d + w.s)[(slice(None),) + sl], np.zeros_like(w.s[(slice(None),) + sl]),
                                      np.zeros_like(w.w[sl]), w.substrate_rgb[sl])
                ref = R[halo].mean(axis=0) if halo.any() else R[m].mean(axis=0)
                de = float(delta_e(R[m], ref[None]).max()) if m.any() else 99.0
                added = float((dft[m | halo] - 45e-6 - float(self.spec.get("first_coat") or 0.0)).max()) if (m | halo).any() else 0.0
                checks["scuffs_hidden"] = dict(value=round(de, 3), limit=E["dE"], ok=de <= E["dE"])
                checks["no_ridge"] = dict(value=round(added*1e6, 2), limit_um=25.0, ok=added <= 25e-6)
            else:
                checks["hide"] = dict(value=round(frac, 4), limit=need, dE_limit=E["hide_dE"],
                                      dft_um_p5_median=[round(float(np.percentile(dft[m], 5))*1e6, 1), round(float(np.median(dft[m]))*1e6, 1)] if m.any() else None,
                                      ok=frac >= need)
            if br:
                # a contiguous patch left at the old coat is a holiday whatever the overall share (8-connected cells)
                remaining, big = self._remaining(hid, ms, w.area)
                checks["hide"].update(largest_unhidden_cm2=round(big*1e4, 2), limit_holiday_cm2=C["holiday_cm2"],
                                      whole_band_value=round(float(hid[m].mean()), 4),
                                      unscored_pocket_cm2=round(float(self._pocket.sum())*w.area*1e4, 1),
                                      still_to_paint_cm2=round(remaining*1e4, 2),
                                      ok=frac >= need and big*1e4 <= C["holiday_cm2"])
                # the paint this task added (the coats on the wall before the session are d_base)
                new = dft - (w.d_base[sl] if w.d_base is not None else 0.0)
            hmax = float(w.h(sl)[m].max()) if m.any() else 0.0
            moved = float(w.moved[sl][m].max()) if m.any() else 0.0
            if br:
                # a run left in the wall: paint the film's flow piled up -- more than run_flow of dry paint carried into
                # a cell by flow (net of what an applicator took back up), outside the bottom rows. Coats laid over set
                # paint stack without flowing and are not runs
                fl = w.flowed[sl]
                run = (fl > C["run_flow"]) & m & ~self._edge
                run_a = float(run.sum())*w.area
                me = m & ~self._edge
                checks["no_runs"] = dict(wet_max_um=round(hmax*1e6, 1), run_cm2=round(run_a*1e4, 3), limit_cm2=C["run_area"]*1e4,
                                         new_paint_max_um=round(float(new[me].max())*1e6, 1) if me.any() else 0.0,
                                         flowed_in_max_um=round(float(fl[me].max())*1e6, 1) if me.any() else 0.0,
                                         flowed_limit_um=round(C["run_flow"]*1e6, 1),
                                         settle_travel_mm=round(moved*1e3, 3),
                                         ok=(hmax <= E["run_h"] or settled) and run_a <= C["run_area"])
            elif self._task_progress is not None:
                # behind-radiator's run definition: it judges the film left behind, not
                # cumulative travel during a repair or another coat's settle.
                run_a = float(((w.flowed[sl] > C["run_flow"]) & m).sum())*w.area
                checks["no_runs"] = dict(wet_max_um=round(hmax*1e6, 1), run_cm2=run_a*1e4,
                                         limit_cm2=C["run_area"]*1e4, flowed_limit_um=C["run_flow"]*1e6,
                                         ok=(hmax <= E["run_h"] or settled) and run_a <= C["run_area"])
            else:
                checks["no_runs"] = dict(wet_max_um=round(hmax*1e6, 1), travel_mm=round(moved*1e3, 3),
                                         ok=(hmax <= E["run_h"] or settled) and (moved <= E["sag_move"] if settled else True))
            lap = float((w.lap[sl][m] > 0).sum())*w.area
            (diag if br else checks)["no_laps"] = dict(area_cm2=round(lap*1e4, 3), limit_cm2=E["lap_area"]*1e4,
                                                        ok=lap <= E["lap_area"])
            if t == "roll-strip":
                de = self._appearance(w, m_full) if settled else None
                checks["uniform"] = dict(dE=None if de is None else round(de, 3), limit=E["dE"],
                                         ok=(de is not None and de <= E["dE"]) if settled else False,
                                         note=None if settled else "measured by verify (settle rule)")
            if t == "cut-in-ceiling" and "corner" in self.targets:
                c = self.targets["corner"][sl]
                cf = float(hid[c].mean()) if c.any() else 0.0
                checks["reaches_corner"] = dict(value=round(cf, 4), limit=0.90, ok=cf >= 0.90)
        if "paper" in self.targets:
            pch = ch["paper"]
            dft = pch.projected_dft()
            m, outside = self.targets["paper"], self.targets["outside"]
            # filling (circle, swatch) asks for a filled, near-opaque coat (15 um dry); the S-stroke is judged on its
            # shape, so a visibly painted cell counts (5 um dry, E['stroke_dft'])
            thr = E["stroke_dft"] if t == "brush-stroke" else E["stencil_hide_dft"]
            if self._task_progress is not None:
                from .scoring import paper_colour, PIGMENT_SHARE
                wanted, share = paper_colour(pch, t)
                cov = float(((wanted[m] >= thr) & (share[m] >= PIGMENT_SHARE)).mean())
            else:
                cov = float((dft[m] >= thr).mean())
            over = float(((dft > 3e-6) & outside).sum())*pch.area
            checks["coverage"] = dict(value=round(cov, 4), limit=0.95, ok=cov >= 0.95)
            if self._task_progress is not None:
                checks["coverage"].update(paint="cadmium_red" if t == "brush-stroke" else "ultramarine",
                                          min_pigment_share=PIGMENT_SHARE)
            checks["within_line"] = dict(area_cm2=round(over*1e4, 3), limit_cm2=0.5, ok=over <= 0.5e-4)
        if "crate" in self.targets:
            cr = ch["crate"]
            dft = cr.projected_dft()
            m = self.targets["crate"]
            lo, hi = E["spray_dft"]
            g = _gauge(dft, m, cr.dx)
            inband = float(((g[m] >= lo) & (g[m] <= hi)).mean()) if m.any() else 0.0
            from scipy.ndimage import binary_dilation
            grow = binary_dilation(m, iterations=max(1, int(round(E["bleed"]/cr.dx))))
            bleed = float(((dft > 3e-6) & ~grow).sum())/max(1, m.sum())
            checks["coat"] = dict(value=round(inband, 4), limit=E["stencil_frac"], ok=inband >= E["stencil_frac"])
            checks["sharp_edges"] = dict(outside_fraction=round(bleed, 4), limit=0.02, ok=bleed <= 0.02,
                                        requested_margin_mm=E["bleed"]*1e3,
                                        effective_margin_mm=max(1, int(round(E["bleed"]/cr.dx)))*cr.dx*1e3)
            hmax = float(cr.h()[m].max()) if m.any() else 0.0
            if settled:
                hmax = self._decl_h.get("crate", hmax)
            moved = float(cr.moved[m].max()) if m.any() else 0.0
            # the crate side lies flat: a heavy coat pools rather than runs (a front creeping 1 mm on a level face is
            # levelling, not a run), so the criterion is the wet thickness at declaration
            checks["no_runs"] = dict(wet_max_um=round(hmax*1e6, 1), travel_mm=round(moved*1e3, 3), limit_wet_um=E["run_h"]*1e6,
                                     ok=hmax <= E["run_h"])
            checks["overspray"] = self._overspray()
        if "print" in self.targets:
            pr = ch["print"]
            dft = _gauge(pr.projected_dft(), self.targets["print"], pr.dx)[self.targets["print"]]
            lo, hi = E["spray_dft"]
            inband = float(((dft >= lo) & (dft <= hi)).mean())
            ratio = float(dft.max()/max(dft.min(), 1e-9)) if dft.max() > 0 else 99.0
            checks["coat"] = dict(value=round(inband, 4), limit=0.95, ok=inband >= 0.95)
            checks["even"] = dict(max_over_min=round(min(ratio, 99.0), 3), limit=2.0, ok=ratio <= 2.0)
            moved = float(pr.moved.max())
            hmax = float(pr.h()[self.targets["print"]].max())
            if settled:
                hmax = self._decl_h.get("print", hmax)
            checks["no_runs"] = dict(wet_max_um=round(hmax*1e6, 1), travel_mm=round(moved*1e3, 3), limit_wet_um=E["run_h"]*1e6,
                                     ok=hmax <= E["run_h"])
            checks["overspray"] = self._overspray()
        if t == "load-roller":
            checks.update(self._roller_loaded())
        if self.spec["preset"] == "wall" and t != "load-roller":
            a, det = self._protected(trim_sinks=br or self._task_progress is not None)
            checks["protected"] = dict(area_cm2=round(a*1e4, 3), limit_cm2=E["protected_area"]*1e4, detail_cm2={k: round(v*1e4, 3) for k, v in det.items()},
                                       ok=a <= E["protected_area"])
        if br:
            # a mark the robot leaves in the wet coat (a link or the ferrule/handle touched it): touched cells on
            # which no applicator has laid real paint (or picked up wet film) since the contact, wet or set
            marred = w.marred[sl].astype(bool) & m
            mark_a = float(marred.sum())*w.area
            checks["no_marks"] = dict(area_cm2=round(mark_a*1e4, 3), limit_cm2=E["protected_area"]*1e4,
                                      touched_cm2=round(float((w.touched[sl].astype(bool) & m).sum())*w.area*1e4, 3),
                                      ok=mark_a <= E["protected_area"])
        elif t != "load-roller":
            # wet coat marred by a robot link or a tool's non-applicator part (frame, ferrule, handle, can): the film it
            # touched (legacy), or the unrepaired marks it leaves (eight-task contract)
            area = 0.0
            for n in ("wall", "paper", "crate", "print"):
                if n in self.targets and n in ch:
                    marks = ch[n].marred if self._task_progress is not None else ch[n].touched
                    area += float((marks.astype(bool) & self.targets[n]).sum())*ch[n].area
            checks["untouched"] = dict(area_cm2=round(area*1e4, 3), limit_cm2=E["protected_area"]*1e4,
                                       ok=area <= E["protected_area"])
            if self._task_progress is not None:
                checks["untouched"]["basis"] = "unrepaired marks left in the film (marred), as in behind-radiator"
        if t == "roll-strip":
            checks["roller_in_tray"] = self._roller_in_tray()
        out["checks"] = checks
        oks = [c["ok"] for c in checks.values()]
        out["succeeded"] = bool(oks) and all(oks)
        out["score"] = float(np.mean(oks)) if oks else 0.0
        if br:
            out["diagnostics"] = diag
            out["reward"] = self._reward(hid, ms, w.area, need, marred, run, a)
            out["reward"]["mean_of_checks"] = out["score"]
            out["score"] = 1.0 if out["succeeded"] else C["fail_scale"]*out["reward"]["progress"]
        elif self._task_progress is not None:
            from .scoring import FAILED
            out["reward"] = self._task_progress.reward(checks, settled)
            out["score"] = 1.0 if out["succeeded"] else FAILED*out["reward"]["progress"]
        return out

    def _remaining(self, hid, ms, cell):
        """(area still to paint before the hide check can pass, largest unhidden patch) in the scored cells ms (m2):
        the larger of the unhidden area beyond the allowance and the unhidden 8-connected patches' excess over the cap.
        It is 0 exactly when the hide check passes, and painting a cell never raises it."""
        from scipy.ndimage import label
        C = BEHIND_RADIATOR
        un = ms & ~hid
        lab, n_lab = label(un, structure=np.ones((3, 3), int))
        sizes = np.bincount(lab.ravel())[1:]*cell if n_lab else np.zeros(1)
        excess = float(np.maximum(sizes - C["holiday_cm2"]*1e-4, 0.0).sum())
        beyond = float(un.sum())*cell - (1.0 - C["hide_frac"])*float(ms.sum())*cell
        return max(beyond, excess, 0.0), float(sizes.max())

    def _reward(self, hid, m, cell, need, marred, run, protected):
        """behind-radiator's progress (BEHIND_RADIATOR): one quantity, wall area, from the final film only.
        cover    = 1 - (area still to paint before hide passes) / (that area at the start), clipped to [0, 1]
                   (_remaining: it is 0 exactly when the hide check passes);
        defects  = hidden cells marked beyond the marks' 0.5 cm2 + hidden run cells beyond the runs' 1 cm2
                   + damage_weight x paint on the radiator or trim beyond its 0.5 cm2 (m2);
        progress = clip(cover - defects / (area to paint at the start), 0, 1).
        Inside every tolerance the defects are 0, so progress is 1 when the verdict succeeds; progress 1 without
        success (a wet film over 250 um before the settle) still scores 0.9, below every success. The band here is the
        scored band (the pocket left out); marks, runs and damage count anywhere."""
        C = BEHIND_RADIATOR
        rem, _ = self._remaining(hid, m, cell)
        r0 = self._remaining0
        cov = min(max(1.0 - rem/r0, 0.0), 1.0)
        marks = max(0.0, float((marred & hid).sum())*cell - E["protected_area"])
        runs = max(0.0, float((run & hid & ~marred).sum())*cell - C["run_area"])
        damage = C["damage_weight"]*max(0.0, protected - E["protected_area"])
        prog = min(max(cov - (marks + runs + damage)/r0, 0.0), 1.0)
        return dict(contract=C["contract"], score_rule=f"1 if success else {C['fail_scale']} x progress", progress=round(prog, 6),
                    coverage=round(cov, 6), still_to_paint_cm2=round(rem*1e4, 1), start_to_paint_cm2=round(r0*1e4, 1),
                    hidden_scored_cm2=round(float(hid[m].sum())*cell*1e4, 1),
                    defects_cm2=dict(marks=round(marks*1e4, 3), runs=round(runs*1e4, 3), damage=round(damage*1e4, 3)),
                    net_after_defects_cm2=round((r0 - rem - marks - runs - damage)*1e4, 1),
                    damage_weight=C["damage_weight"])

    def _overspray(self):
        """Containment on conserved nonvolatile solids, never on residual solvent.

        The 5% tolerance is the share of emitted solids outside the booth. Airborne
        or unlocated solids and paint carried on robot skins count as uncontained;
        flashed/evaporated solvent does not. Drying alone cannot change this ratio.
        """
        a = self.paint.applicators.get("aerosol")
        if a is None:
            return dict(ok=True)
        paints = getattr(self.paint, "paints", None) or next(iter(self.paint.charts.values())).paints
        # A single-product can checkpoint without the cumulative solids counter
        # falls back to the emitted volume x the paint's solids fraction. The can cannot be refilled.
        emitted = a.stats.get("emitted_solids_m3", a.stats["emitted_m3"]*paints[0].phi0)
        contained = ("spray_box", "box_", "crate", "stencil", "print")
        parts = {}
        for n, sink in self.paint.sinks.items():
            if not n.startswith(contained):
                parts["sink:" + n] = float(sink["solids"].sum())
        for n, ch in self.paint.charts.items():
            t = ch.totals()
            if not n.startswith(contained):
                parts["chart:" + n] = float(t["wet_solids"].sum() + t["dry_solids"].sum())
            # Runoff leaving a chart has no recorded landing inside the box.
            parts["runoff:" + n] = float(t["drained_solids"].sum())
        for n, skin in getattr(self.paint, "skins", {}).items():
            parts["skin:" + str(n)] = float(skin["solids"].sum())
        parts["airborne"] = float(getattr(a, "ps", np.zeros(0)).sum())
        outside = sum(parts.values())
        frac = outside/emitted if emitted > 0 else (0.0 if outside == 0 else 1.0)
        return dict(fraction=round(frac, 6), basis="nonvolatile solids volume",
                    outside_solids_mL=outside*1e6, emitted_solids_mL=emitted*1e6,
                    detail_solids_mL={n: v*1e6 for n, v in parts.items() if v},
                    limit=0.05, ok=frac <= 0.05)

    def _roller_loaded(self):
        from .roller import SAT
        r = self.paint.applicators.get("roller")
        load = r.load_mass()
        V = r.s.sum(axis=0) + r.v
        spread = float((V > 0.2*SAT*r.C0).mean())
        w = self.env.world
        bq = w.state.body_q.numpy()
        tray = w.scene.bodies.get("tray")
        above = False
        if tray is not None:
            pc = bq[r.cover][:3]; pt = bq[tray][:3]
            from .film import _quat_rot
            loc = _quat_rot(bq[tray][3:7]).T @ (pc - pt)
            g = P.TRAY["size"]
            above = abs(loc[0]) < g[0]/2 and abs(loc[1]) < g[1]/2 and 0.10 < loc[2] < 0.6
        quiet = (w.tick - getattr(self, "_last_drip_tick", -10**9))*P.DT >= 10.0
        return dict(load=dict(grams=round(load*1e3, 2), limit_g=50.0, ok=load >= 0.050),
                    spread=dict(fraction=round(spread, 3), limit=0.8, ok=spread >= 0.8),
                    held_above_tray=dict(ok=bool(above)),
                    no_drip_10s=dict(seconds_since_last_drop=round((w.tick - getattr(self, "_last_drip_tick", -10**9))*P.DT, 2)
                                     if hasattr(self, "_last_drip_tick") else None, ok=bool(quiet)))

    def _roller_in_tray(self):
        w = self.env.world
        bq = w.state.body_q.numpy()
        r = self.paint.applicators.get("roller")
        tray = w.scene.bodies.get("tray")
        if r is None or tray is None:
            return dict(ok=False)
        from .film import _quat_rot
        loc = _quat_rot(bq[tray][3:7]).T @ (bq[r.cover][:3] - bq[tray][:3])
        g = P.TRAY["size"]
        inside = abs(loc[0]) < g[0]/2 and abs(loc[1]) < g[1]/2 and loc[2] < 0.12
        released = (self._task_progress is not None or all(w.grip_force[a] == 0 for a in w.arms)) and not self._held(r)
        return dict(cover_in_tray=bool(inside), released=bool(released), ok=bool(inside and released))

    def _held(self, app):
        w = self.env.world
        fingers = {i for i in w.scene.robot_bodies if "finger" in w.model.body_label[i]}
        bodies = {app.frame, app.cover} if hasattr(app, "frame") else {app.body}
        return any((a in fingers and b in bodies) or (b in fingers and a in bodies) for a, b, *_ in w.contact_table())

    # ------------------------------------------------------------------ API
    def checkpoint_state(self):
        """Only the opt-in contract adds task state to checkpoints; no physics state."""
        if self._task_progress is None:
            return None
        return dict(drops0=self._drops0, last_drip_tick=getattr(self, "_last_drip_tick", None))

    def restore_state(self, state):
        if self._task_progress is None:
            return
        self._drops0 = state["drops0"] if state else self.paint.drips.landed
        last = state.get("last_drip_tick") if state else self.env.world.tick
        if last is None:
            self.__dict__.pop("_last_drip_tick", None)
        else:
            self._last_drip_tick = last
        for key in ("_verdict_cache", "_hide_cache", "_prot_cache"):
            self.__dict__.pop(key, None)
        self._task_progress.declaration.clear()
        self._decl_h.clear()
        self.env._eval_key = None

    def note_drips(self):
        d = self.paint.drips
        if d.landed != self._drops0 or len(d.pos):
            self._last_drip_tick = self.env.world.tick
            self._drops0 = d.landed

    def progress(self):
        self.note_drips()
        m = self.measure(settled=False)
        m["complete"] = False
        return m

    def evaluate(self, max_s=None, fast=None):
        """Settle rule: advance only the films until set (or 30 min), measure, restore the live film.

        Fast path (E['settle_fast']): catch and tool charts are only dried; a target or protected chart whose wet + dry
        solids changed by less than 0.1 nm per 2 s step for three steps has stopped flowing (its viscosity and yield
        stress only rise as it dries), so it is only dried from then on, in 30 s steps once every chart is quiet. P17 checks the fast verdicts against the full settle. The verdict
        is cached per world state (tick and every chart's version): the GAR clients' evaluation route returns it again
        without re-settling while nothing has changed."""
        import copy
        import time as _time
        paint = self.paint
        fast = E["settle_fast"] if fast is None else bool(fast)
        key = (self.env.world.tick, tuple(c.version for c in paint.charts.values()), max_s, fast,
               tuple((n, float(s["vehicle"])) for n, s in sorted(paint.sinks.items())))
        hit = getattr(self, "_verdict_cache", None)
        if hit is not None and hit[0] == key:
            out = copy.deepcopy(hit[1])
            out["cached"] = True
            return out
        t_wall = _time.perf_counter()
        if self._task_progress is not None:
            self._task_progress.capture_declaration()
        snap = {n: c.snapshot() for n, c in paint.charts.items()}
        bq = self.env.world.state.body_q.numpy()
        psnap, t_drop = None, 0.0
        if (self.task == "behind-radiator" or (self._task_progress is not None and "wall" in self.targets)) and len(paint.drips.pos):
            # paint already falling when the episode ends lands where it falls before the radiator and trim are judged
            # the drops in flight fall with the rigid world frozen. A brush's pendant
            # volume below one drop, and drips a still-loaded brush would shed later, are not released.
            # The explicit eight-task contract uses this rule for its other painted-wall tasks too.
            psnap = paint.snapshot()
            while len(paint.drips.pos) and t_drop < 2.0:
                paint.drips.step(bq, P.DT)
                t_drop += P.DT
        t = 0.0
        limit = float(max_s if max_s is not None else E["settle_max_s"])
        for c in paint.charts.values():
            c.moved[...] = 0.0
        # level faces are judged on the wet thickness at declaration (pooling): read it before the settle sets the film
        # (measured after it, it is always 0 and a 320 um puddle would pass)
        self._decl_h = {n: float(paint.charts[n].h()[self.targets[n]].max()) for n in ("crate", "print")
                        if n in self.targets and n in paint.charts and self.targets[n].any()}
        # catch and tool charts (drop cloth, booth walls and floor, the stencil, the tray) are only read for how much
        # paint they hold, which flow does not change: they are dried, not flowed (the stencil's 1 mm film kept the
        # settle of a finished spray-stencil episode flowing for 1,044 s of film time, 659 s of wall time). P17 checks
        # the verdicts against the full settle.
        quiet = {n for n, c in paint.charts.items() if c.role in ("catch", "tool")} if fast else set()
        calm = {}
        steps = dict(flow=0, dry=0)
        try:
            while t < limit:
                live = [c for c in paint.charts.values() if c.bbox is not None and c.wet_max() > P.FILM["h_min"]]
                if not live:
                    break
                dt = min(30.0 if (fast and all(c.name in quiet for c in live)) else 2.0, limit - t)
                for c in paint.charts.values():
                    if c.name in quiet:
                        c.dry_only(dt)
                        steps["dry"] += 1
                        continue
                    sl, S0 = c.solids_window() if fast else (None, None)
                    c.advance(dt, bq, self.env.world.sim_time + t)
                    steps["flow"] += 1
                    if fast and sl is not None:
                        dS = float(np.abs(c.s[(slice(None),) + sl].sum(axis=0) + c.d[(slice(None),) + sl].sum(axis=0) - S0).max())
                        calm[c.name] = calm.get(c.name, 0) + 1 if dS < 1e-10 else 0
                        if calm[c.name] >= 3:
                            quiet.add(c.name)
                t += dt
            out = self.measure(settled=True)
            out["settle_s"] = t
            out["complete"] = True
            out["settle"] = dict(fast=fast, wall_s=round(_time.perf_counter() - t_wall, 2), quiet_charts=sorted(quiet), **steps)
            if psnap is not None:
                out["settle"]["drops_landed_s"] = round(t_drop, 4)
        finally:
            for n, c in paint.charts.items():
                c.restore(snap[n])
            if psnap is not None:
                paint.restore(psnap)
        key_after = (key[0], tuple(c.version for c in paint.charts.values()), *key[2:])
        self._verdict_cache = (key_after, copy.deepcopy(out))
        return out
