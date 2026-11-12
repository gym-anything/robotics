"""Outcome of melody: exactly the tune's notes in order on the xylophone, struck with the mallets, their onsets on the
least-squares beat grid of the written rhythm within onset_s at the stated tempo within tempo_rel, each stroke clean,
its fundamental strong, no neighbouring bar excited, each note ringing until its window closes, no stroke on another
instrument (params.MELODY_EVAL); nothing dropped and the robot still at the end.

verify(e) evaluates an episode; e is its evaluator, task.Evaluator, whose verdict adds the common checks and calls
judge(e, g, checks, meta, ...) with the goal g (layout["goal"]): judge fills in the task's checks and meta and returns
its progress. verify returns the evaluation (success, score, checks, meta): 1 if every check passes, else 0.9 x progress
(melody's own contract: params.MELODY_EVAL fail_scale). melody(ev, g) gives the checks, the meta and the progress
(melody_progress).

tick(e) is called every tick (task.Evaluator.tick, from the environment's act): it logs each struck bar's ring
against its free ring (e.ring_log), which ring_damping reads back."""
import math

import numpy as np

from percussion import params as P


def verify(e):
    return e.verdict(judge, fail_scale=P.MELODY_EVAL["fail_scale"])


def judge(e, g, checks, meta, *, ramp, TP, rim_factor):
    mc, mm, progress = melody(e, g)
    checks.update(mc)
    meta.update(mm)
    return progress


def melody(ev, g):
    """Melody checks, diagnostics and progress (params.MELODY_EVAL). The notes are the
    xylophone strikes approaching at >= 0.05 m/s, every one in the episode. Each is judged on the stroke's own
    excitation, split from a bar still ringing by the linear modal model (world._contact_split): a mallet head
    struck it; one short contact, no bounce (clean); its fundamental at least tone_min_db against the fundamental
    an impulse of the same size gives at the bar's centre (tone); no other bar given neighbour_max of the stroke's
    own energy (no_neighbours); and when the note's ring window closes - half its written length, at most an eighth
    note (ring_beats) at the stated tempo, sooner if the next note is struck - its bar stands less than
    ring_max_damping_db below the free continuation of its state when the stroke ended (rings, tick: later
    damping, and staccato within half a note's length, are articulation). The tune: exactly its notes in order
    (notes), onsets within onset_s of the least-squares beat grid of its written rhythm, at the stated tempo within
    tempo_rel (timing); no stroke on another instrument (only_xylophone)."""
    E = P.MELODY_EVAL
    fs = P.AUDIO_FS
    want = [n for n, _ in g["notes"]]
    lengths = sorted({b for _, b in g["notes"]})
    T0 = 60.0/g["bpm"]
    allst = ev.strikes(min_speed=P.STRIKE_MIN_SPEED_M_S)
    st = [s for s in allst if s["surface"].startswith("bar_")]
    other = [s for s in allst if not s["surface"].startswith("bar_")]
    ramp = lambda x, full, zero: float(np.clip((zero - x)/(zero - full), 0.0, 1.0))
    per = []
    for k, s in enumerate(st):
        m = s["segments"][0]
        fund, ref = m.get("fundamental_J"), m.get("fundamental_centre_J")
        tone_db = 10*math.log10(max(fund, 1e-30)/max(ref, 1e-30)) if fund is not None and ref else None
        e = s.get("bars_stroke_J") or {}
        neigh = max([v for n, v in e.items() if n != s["note"]] + [0.])/max(e.get(s["note"], 0.), 1e-30) if e else None
        nxt = st[k + 1]["onset_s"] if k + 1 < len(st) else math.inf
        damp = {L: ring_damping(ev, s, min(nxt, s["onset_s"] + min(L/2, E["ring_beats"])*T0)) for L in lengths}
        mallet = all(x["striker"] in E["strikers"] for x in s["segments"])
        q_stroke = float(mallet)*ramp(s["contact_ms"], P.CONTACT_MAX_S["bar"]*1e3, E["contact_zero_ms"]) * \
            (0.0 if tone_db is None else ramp(-tone_db, -E["tone_min_db"], -E["tone_zero_db"])) * \
            (0.0 if neigh is None else ramp(neigh, E["neighbour_max"], E["neighbour_zero"]))
        L_k = g["notes"][k][1] if k < len(want) else max(lengths)
        per.append(dict(clean=s["contact_ms"] <= P.CONTACT_MAX_S["bar"]*1e3,
                        tone=tone_db is not None and tone_db >= E["tone_min_db"],
                        neighbours=neigh is not None and neigh < E["neighbour_max"], mallet=mallet,
                        rings=damp[L_k] < E["ring_max_damping_db"], tone_db=tone_db, neigh=neigh, damp=damp[L_k],
                        q={L: q_stroke*ramp(damp[L], E["ring_max_damping_db"], E["ring_zero_db"]) for L in lengths}))
    notes = [s["note"] for s in st]
    checks = dict(notes=notes == want)
    meta = dict(strikes=len(st), notes=notes, onsets_s=[round(s["onset_s"], 4) for s in st])
    if notes == want:
        beats = np.cumsum([0.] + [b for _, b in g["notes"][:-1]])
        on = np.array([s["onset_s"] for s in st])
        (T, t0), *_ = np.linalg.lstsq(np.vstack([beats, np.ones_like(beats)]).T, on, rcond=None)
        err = np.abs(on - (beats*T + t0))
        checks["timing"] = float(err.max()) <= E["onset_s"] and abs(T0/T - 1) <= E["tempo_rel"]
        meta.update(fitted_bpm=round(60/T, 2), max_timing_error_s=round(float(err.max()), 4))
    checks["clean"] = bool(st) and all(p["clean"] for p in per) and ev._restrike(st) == 0
    checks["tone"] = bool(st) and all(p["tone"] for p in per)
    checks["rings"] = bool(st) and all(p["rings"] for p in per)
    checks["mallets"] = bool(st) and all(p["mallet"] for p in per)
    checks["no_neighbours"] = bool(st) and all(p["neighbours"] for p in per)
    checks["only_xylophone"] = not other
    events, last, idx = [], {}, {id(s): k for k, s in enumerate(st)}
    for s in sorted(allst, key=lambda x: x["start"]):
        k = last.get(s["surface"])
        if k is not None and s["onset_s"] - events[k]["end"] <= E["regroup_s"]:
            events[k].update(q={L: 0.0 for L in lengths}, end=s["end"]/fs)
            continue
        p = per[idx[id(s)]] if id(s) in idx else None
        last[s["surface"]] = len(events)
        events.append(dict(note=s["note"] if p else None, onset=s["onset_s"], end=s["end"]/fs,
                           q=p["q"] if p else {L: 0.0 for L in lengths}))
    progress, info = melody_progress(events, g["notes"], g["bpm"], E)
    nn = [p["neigh"] for p in per if p["neigh"] is not None]
    meta.update(progress=progress, note_events=len(events), **info,
                tone_db=[None if p["tone_db"] is None else round(p["tone_db"], 2) for p in per],
                ring_damping_db=[round(p["damp"], 2) for p in per],
                max_neighbour_energy_ratio=max(nn) if nn else None,
                other_strikers=sorted({x["striker"] for s in st for x in s["segments"]} - set(E["strikers"])),
                other_instrument_strikes=len(other))
    return checks, meta, progress


def tick(e):
    """Every tick (melody): for each xylophone bar struck within its longest ring window (MELODY_EVAL ring_beats at
    the stated tempo), the bar's vibration energy (static deflection removed) against the free continuation of its
    state when the strike ended (world.ring_ref), in dB, keyed '<surface>@<end sample>'. The ring is judged on this
    trace where the note's window closes (not at the end of whichever touch began inside it)."""
    w = e.env.world
    span = P.MELODY_EVAL["ring_beats"]*60.0/e.env.layout["goal"]["bpm"] + 0.002
    t = w.sim_time
    for surf, (end, zref) in w.ring_ref.items():
        if not surf.startswith("bar_") or t - end/P.AUDIO_FS > span:
            continue
        f = next(x for x in w.surfaces if x["name"] == surf)
        st, sl = w.structs[f["struct"]], f["slice"]
        free = float(st.dof_energy(zref*np.exp(st.lam*(t - end/P.AUDIO_FS)))[sl].sum())
        now = float(st.dof_energy(w._dyn_z(f["struct"]))[sl].sum())
        e.ring_log.setdefault(f"{surf}@{end}", []).append(
            (round(t, 6), round(10.0*math.log10(max(free, 1e-30)/max(now, 1e-30)), 4)))


def ring_damping(e, strike, w_end):
    """dB below its free ring at which the strike's bar stands when its ring window closes at w_end (or at the last
    sample, if the episode ended first): 0 if nothing was logged."""
    keys = [k for k in e.ring_log if k.split("@")[0] == strike["surface"]
            and strike["start"] <= int(k.split("@")[1]) <= strike["end"]]
    if not keys:
        return 0.0
    vals = [db for t, db in e.ring_log[max(keys, key=lambda k: int(k.split("@")[1]))] if t < w_end - 1e-9]
    return max(0.0, vals[-1]) if vals else 0.0


def melody_progress(events, notes, bpm, E=P.MELODY_EVAL, hypotheses=6):
    """How much of the tune was heard, and how well (0..1): a note-level match of the performed events to the tune as
    a weighted Jaccard index, sum(c) / (tune notes + events - matched). events: dicts (note or None for a stroke on
    another instrument, which never matches; onset s; q: the stroke's graded quality per written length) in onset
    order. The matching keeps order and pitch and pairs as many notes as it can; among equally long matchings it
    takes the one most consistent with a steady beat (a tie-break only). Per matched note
        c = 1/2 + 1/2 (r + theta r + q)/3
    - 1/2 for the right note in its place (not onset-gated, unlike MIREX's +-50 ms note matching: the order of the
      pitches is the tune's first dimension);
    - r, rhythm: the mean credit of the spans from the note to every matched note within rhythm_span_beats written
      beats (one 4/4 measure; at least its neighbours), each span against its written length at the performer's
      tempo: 1 within 2 onset_s (two onsets), 0 at twice that, the tolerance shrinking in proportion to the beat once
      the reading is faster than the band `timing` accepts (so speed earns no rhythm credit). Spans make a sustained
      tempo drift or a rushed run cost, while a single hesitation stays local. The performer's tempo is the reading
      - least squares, median interval, or any one interval's own tempo - that gives the largest rhythm-and-tempo
      credit; a performance that passes `timing` gets every span and the tempo in full (each span's error is at most
      2 onset_s against the least-squares reading);
    - theta r, tempo: theta 1 within tempo_rel of the stated tempo, 0 at tempo_zero_rel; weighted by the note's
      rhythm, since an arrhythmic sequence has no tempo;
    - q, stroke quality: the verifier's own per-note quantities graded (MELODY_EVAL zero points; 1 at its thresholds).
    A single matched note has r = theta = 1. A correct k-note beginning scores k/len(notes); a performance meeting
    every melody check scores 1, whoever played it, with either hand, from any start time."""
    N, M = len(notes), len(events)
    none = dict(paired_notes=0, progress_tempo_bpm=None)
    if not M:
        return 0.0, none
    q = np.array([n for n, _ in notes])
    L = [b for _, b in notes]
    beta = np.cumsum([0.] + L[:-1])
    T0 = 60.0/bpm
    pitch = np.array([e["note"] or "" for e in events])
    t = np.array([e["onset"] for e in events], float)
    same = q[:, None] == pitch[None, :]
    if not same.any():
        return 0.0, none
    ramp = lambda x, full, zero: np.clip((zero - x)/(zero - full), 0.0, 1.0)

    def match(T, t0):
        tau = ramp(np.abs(t[None, :] - beta[:, None]*T - t0), E["onset_s"], E["onset_zero_s"])
        W = np.where(same, 1.0 + 0.01*tau, -np.inf)
        D = np.zeros((N + 1, M + 1))
        for i in range(N):
            D[i + 1, 1:] = np.maximum.accumulate(np.maximum(D[i, 1:], D[i, :-1] + W[i]))
        pairs, i, j = [], N, M
        while i and j:
            if D[i, j] == D[i, j - 1]:
                j -= 1
            elif D[i, j] == D[i - 1, j]:
                i -= 1
            else:
                pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        return pairs[::-1]

    def fit(pairs):
        ii, jj = (list(x) for x in zip(*pairs))
        if len(pairs) >= 2:
            (T, t0), *_ = np.linalg.lstsq(np.vstack([beta[ii], np.ones(len(ii))]).T, t[jj], rcond=None)
            if T > 0:
                return T, t0
        return T0, t[jj[0]] - beta[ii[0]]*T0

    def credit(pairs):
        ii, jj = (np.array(x) for x in zip(*pairs))
        tempo = lambda T_: float(ramp(abs(T0/T_ - 1), E["tempo_rel"], E["tempo_zero_rel"]))
        r, T = np.ones(len(pairs)), T0
        if len(pairs) >= 2:
            dt, db = np.diff(t[jj]), np.diff(beta[ii])
            pos, tt = beta[ii], t[jj]
            span = np.abs(pos[None, :] - pos[:, None])
            k = np.arange(len(pairs))
            near = ((span > 0) & (span <= E["rhythm_span_beats"])) | (np.abs(k[None, :] - k[:, None]) == 1)
            readings = [x for x in (fit(pairs)[0], float(np.median(dt/db)), *(dt/db)) if x > 0]
            best = None
            for T_ in readings:
                tol = 2*E["onset_s"]*min(1.0, T_/((1 - E["tempo_rel"])*T0))
                rho = ramp(np.abs((tt[None, :] - tt[:, None]) - (pos[None, :] - pos[:, None])*T_), tol, 2*tol)
                r_ = (rho*near).sum(axis=1)/near.sum(axis=1)
                if best is None or r_.sum()*(1 + tempo(T_)) > best + 1e-12:
                    best, r, T = r_.sum()*(1 + tempo(T_)), r_, T_
        theta = tempo(T)
        qs = np.array([events[j]["q"][L[i]] for i, j in zip(ii, jj)])
        c = 0.5 + 0.5*(r + theta*r + qs)/3
        return float(c.sum()/(N + M - len(pairs))), T

    off = np.sort((t[None, :] - beta[:, None]*T0)[same])
    cand = np.unique(np.round(off, 3))
    z = E["onset_zero_s"]
    votes = np.searchsorted(off, cand + z, "right") - np.searchsorted(off, cand - z, "left")
    best = (-1.0, [], T0)
    for t0 in cand[np.lexsort((cand, -votes))[:hypotheses]]:
        T = T0
        for _ in range(3):
            pairs = match(T, t0)
            p_, T_read = credit(pairs)
            if p_ > best[0] + 1e-12:
                best = (p_, pairs, T_read)
            T1, t1 = fit(pairs)
            if abs(T1 - T) < 1e-9 and abs(t1 - t0) < 1e-9:
                break
            T, t0 = T1, t1
    p_, pairs, T = best
    return p_, dict(paired_notes=len(pairs), progress_tempo_bpm=round(60.0/T, 2))
