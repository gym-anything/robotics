"""Outcome of behind-radiator: the wall behind the radiator, between the baseboard and the window apron, painted to
full hide without runs or marks left in the coat and without paint on the radiator or the trim. The task has its own
verdict and reward contract (progress-v1, task.BEHIND_RADIATOR). It judges the wall it leaves, all read from the
settled film, never from the order, route or timing of the work:
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

verify(e, settled) receives the episode's task.Evaluator and whether the films have settled, and returns the verdict
with its checks, score and reward (progress-v1). targets(e) gives the target cells (the band, BEHIND_RAD), prepare(e)
what the verdict reads from the start (the unscored pocket, the bottom rows not read for runs, the area to paint at
the start: the reward's unit) and LAND_DROPS = True has task.Evaluator.evaluate land the drops already falling before
the settle."""
import numpy as np

from painting import params as P
from painting.layout import BB, RAD, WIN
from painting.task import BEHIND_RADIATOR, _rect_cells

E = P.EVAL
LAND_DROPS = True
# the band behind the radiator on the wall chart (x0, x1, z0, z1) in world metres
BEHIND_RAD = dict(x0=WIN["x"] - RAD["len"]/2, x1=WIN["x"] + RAD["len"]/2, z0=BB["h"], z1=WIN["sill"] - WIN["stool_t"] - WIN["apron"])


def targets(e):
    ch = e.paint.charts
    out = {}
    if "wall" in ch:
        w = ch["wall"]
        b = BEHIND_RAD
        out["wall"] = _rect_cells(w, b["x0"], b["x1"], b["z0"], b["z1"])
    return out


def prepare(e):
    # the band the brush reaches (BEHIND_RADIATOR: the bottom of the middle is not scored), the band's bottom
    # row on the baseboard (not read for runs), and the area still to paint at the start: the reward's unit
    C, w = BEHIND_RADIATOR, e.paint.charts["wall"]
    pk, sl = C["pocket"], e._bbox("wall", e.targets["wall"])
    e._pocket = _rect_cells(w, pk["x0"], pk["x1"], BEHIND_RAD["z0"], pk["z1"]) & e.targets["wall"]
    e._edge = _rect_cells(w, -1.0, 9.0, BEHIND_RAD["z0"], BEHIND_RAD["z0"] + C["run_edge"])[sl]
    hid, m = e._hidden(False)
    ms = m & ~e._pocket[sl]
    e._remaining0 = max(_remaining(hid, ms, w.area)[0], w.area)


def verify(e, settled):
    t, ch = e.task, e.paint.charts
    out = dict(task=t, settled=settled, contract=e.contract)
    checks = {}
    diag = {}
    C = BEHIND_RADIATOR
    if "wall" in e.targets:
        w = ch["wall"]
        m_full = e.targets["wall"]
        sl = e._bbox("wall", m_full)
        dft = (w.d[(slice(None),) + sl] + w.s[(slice(None),) + sl]).sum(axis=0)
        hid, m = e._hidden(settled)
        ms = m & ~e._pocket[sl]                               # the scored cells (not the pocket)
        frac = float(hid[ms].mean()) if ms.any() else 0.0
        need = C["hide_frac"]
        checks["hide"] = dict(value=round(frac, 4), limit=need, dE_limit=E["hide_dE"],
                              dft_um_p5_median=[round(float(np.percentile(dft[m], 5))*1e6, 1), round(float(np.median(dft[m]))*1e6, 1)] if m.any() else None,
                              ok=frac >= need)
        # a contiguous patch left at the old coat is a holiday whatever the overall share (8-connected cells)
        remaining, big = _remaining(hid, ms, w.area)
        checks["hide"].update(largest_unhidden_cm2=round(big*1e4, 2), limit_holiday_cm2=C["holiday_cm2"],
                              whole_band_value=round(float(hid[m].mean()), 4),
                              unscored_pocket_cm2=round(float(e._pocket.sum())*w.area*1e4, 1),
                              still_to_paint_cm2=round(remaining*1e4, 2),
                              ok=frac >= need and big*1e4 <= C["holiday_cm2"])
        # the paint this task added (the coats on the wall before the session are d_base)
        new = dft - (w.d_base[sl] if w.d_base is not None else 0.0)
        hmax = float(w.h(sl)[m].max()) if m.any() else 0.0
        moved = float(w.moved[sl][m].max()) if m.any() else 0.0
        # a run left in the wall: paint the film's flow piled up -- more than run_flow of dry paint carried into
        # a cell by flow (net of what an applicator took back up), outside the bottom rows. Coats laid over set
        # paint stack without flowing and are not runs
        fl = w.flowed[sl]
        run = (fl > C["run_flow"]) & m & ~e._edge
        run_a = float(run.sum())*w.area
        me = m & ~e._edge
        checks["no_runs"] = dict(wet_max_um=round(hmax*1e6, 1), run_cm2=round(run_a*1e4, 3), limit_cm2=C["run_area"]*1e4,
                                 new_paint_max_um=round(float(new[me].max())*1e6, 1) if me.any() else 0.0,
                                 flowed_in_max_um=round(float(fl[me].max())*1e6, 1) if me.any() else 0.0,
                                 flowed_limit_um=round(C["run_flow"]*1e6, 1),
                                 settle_travel_mm=round(moved*1e3, 3),
                                 ok=(hmax <= E["run_h"] or settled) and run_a <= C["run_area"])
        lap = float((w.lap[sl][m] > 0).sum())*w.area
        diag["no_laps"] = dict(area_cm2=round(lap*1e4, 3), limit_cm2=E["lap_area"]*1e4,
                               ok=lap <= E["lap_area"])
    if e.spec["preset"] == "wall":
        a, det = e._protected(trim_sinks=True)
        checks["protected"] = dict(area_cm2=round(a*1e4, 3), limit_cm2=E["protected_area"]*1e4, detail_cm2={k: round(v*1e4, 3) for k, v in det.items()},
                                   ok=a <= E["protected_area"])
    # a mark the robot leaves in the wet coat (a link or the ferrule/handle touched it): touched cells on
    # which no applicator has laid real paint (or picked up wet film) since the contact, wet or set
    marred = w.marred[sl].astype(bool) & m
    mark_a = float(marred.sum())*w.area
    checks["no_marks"] = dict(area_cm2=round(mark_a*1e4, 3), limit_cm2=E["protected_area"]*1e4,
                              touched_cm2=round(float((w.touched[sl].astype(bool) & m).sum())*w.area*1e4, 3),
                              ok=mark_a <= E["protected_area"])
    out["checks"] = checks
    oks = [c["ok"] for c in checks.values()]
    out["succeeded"] = bool(oks) and all(oks)
    out["score"] = float(np.mean(oks)) if oks else 0.0
    out["diagnostics"] = diag
    out["reward"] = _reward(e, hid, ms, w.area, need, marred, run, a)
    out["reward"]["mean_of_checks"] = out["score"]
    out["score"] = 1.0 if out["succeeded"] else C["fail_scale"]*out["reward"]["progress"]
    return out


def _remaining(hid, ms, cell):
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


def _reward(e, hid, m, cell, need, marred, run, protected):
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
    rem, _ = _remaining(hid, m, cell)
    r0 = e._remaining0
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
