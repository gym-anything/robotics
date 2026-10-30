"""Task instructions and the environment-owned evaluator.

The evaluator reads only simulated quantities: the vibro-contact log (every contact segment between a striker and a
vibrating surface: start/end sample, peak force, impulse, approach speed, struck point, energy given to the struck bar
and its modes), rigid contacts of tools with the drum's hoop (rim strokes), the client-ear microphone signal (IEC 61672
L_AF, A-weighted with fast time weighting), and object poses. It never changes the simulation.

Strikes: contact segments on one surface closer than 10 ms (drum: the rim-reflection re-contacts belong to the stroke)
or 3 ms (bars) form one strike; its onset is the first contact. A contact still in progress when the episode is judged
counts as a segment ending now (world.open_segments; otherwise a strike that stayed on the bar would pass
strike-and-stop). Reference levels are physical anchors, not hand-set numbers: the same instrument model,
contact law and acoustic chain struck by the named striker at reference speeds - Dahl's measured stick speeds on the
drum, estimates around Rucz's 1 m/s mallet on the bars, a measured fingertip mass at a keystroke speed for finger-tap
(params.REFERENCE_SPEEDS, params.FINGERTIP_TAP; reference_level).
"""

import copy
import math
from fractions import Fraction

import numpy as np

from . import params as P

INSTRUCTIONS = {
    # the instruction states the job; the ready zone (tips 3-15 cm above the head, inside it) stays in the verifier.
    # The sticks go back on the trap tables (the scene has no rack).
    "take-sticks": "Take both drumsticks from the trap tables, hold them ready a few centimetres above the drumhead for "
                   "{hold_s:.0f} s without playing, then put them back on the trap tables.",
    "melody": "Play '{tune_title}' on the xylophone with the mallets at {bpm} beats per minute. The notes, each with its "
              "length in beats: {notes}.",
    "bordun": "Play the {key} bordun ({n0} and {n1} together) {count} times at {bpm} BPM on the xylophone, both mallets "
              "striking together (within 15 ms) and equally loud.",
    "roll": "Play a single-stroke roll on the {bar} resonator bar with the two mallets: at least {rate:.0f} strokes per "
            "second, alternating hands, for {dur:.0f} s, with even strokes (equal mallet speeds).",
    "finger-tap": "Tap the drum {count} times with a fingertip (no tool) at {bpm} BPM, softly (mezzo-piano), letting the "
                  "head ring between taps.",
    "steady-beat": "Keep a steady beat on the drum with the drumsticks at the client's walking cadence of {bpm} beats "
                   "per minute for {beats} beats, then {faster_pct}% faster ({bpm2:.0f} BPM) for {faster_beats} beats.",
    # the order within the bar (the accent is its fourth stroke, as judged) and what the accent is louder than (the soft
    # strokes) are stated; the contrast is the task's own number, so it is stated too
    "accent-ostinato": "Play an ostinato on the drum with the drumsticks at {bpm} BPM for {bars} bars, each bar three soft "
                       "strokes and then an accent at least {accent_db:.0f} dB louder than the soft strokes, every stroke "
                       "evenly spaced.",
    "single-tone": "Play one clear mezzo-forte tone on the {bar} resonator bar (one stroke between the pins, no second "
                   "bounce) for the client's listening test.",
    "dynamics": "Play three single strokes on the drum with a drumstick: piano, then mezzo-forte, then forte, each "
                "clearly louder than the one before.",
    "echo": "Listen: the loudspeaker plays a short rhythm on a woodblock starting at {start:.1f} s. Echo the same rhythm "
            "on the drum with a drumstick, starting within two beats after it ends.",
    # the instruction states the job; the damping threshold and judging window are the verifier's
    # (params, evaluation thresholds), as in the melody instruction.
    "strike-and-stop": "Play a short tone: strike the {bar} resonator bar once, then stop its ringing by touching the bar "
                       "when the cue light on the mast turns on ({cue_delay_s:.2f} s after your stroke), without striking it "
                       "again.",
    "pentatonic-setup": "Set the xylophone up for pentatonic play: lift the F4, B4 and F5 bars off their pins and lay "
                        "them flat in the accessory tray; leave every other bar seated on its pins.",
}
TUNE_TITLES = {"hot-cross-buns": "Hot Cross Buns", "mary-had-a-little-lamb": "Mary Had a Little Lamb"}


def instruction(config):
    from .layout import preset_layout
    lay = preset_layout(config)
    g = lay["goal"]
    t = config.task
    fmt = dict(g)
    if t == "melody":
        fmt.update(tune_title=TUNE_TITLES.get(g["tune"], g["tune"]),
                   notes=", ".join(f"{n} {Fraction(b).limit_denominator(8)}" for n, b in g["notes"]))
    if t == "bordun":
        fmt.update(n0=g["notes"][0], n1=g["notes"][1])
    if t == "roll":
        fmt.update(rate=g["rate_hz"], dur=g["duration_s"])
    if t == "steady-beat":
        fmt.update(bpm2=g["bpm"]*(1 + g["faster_pct"]/100))
    if t == "echo":
        fmt.update(start=g["start_s"])
    if t == "strike-and-stop":
        fmt.update(judge_end=g["judge_after_s"] + g["judge_window_s"])

    return INSTRUCTIONS[t].format(**fmt)


def echo_stimulus(goal, fs=P.AUDIO_FS):
    """Woodblock clicks for the echo task: a 1.4 kHz decaying burst (tau 18 ms) per note, 94 dB SPL peak at 1 m
    (a struck woodblock; level an estimate). Returns (signal p_1m(t) in Pa, onset times relative to the start)."""
    beat = 60.0/goal["bpm"]
    onsets, t = [], 0.0
    for d in goal["rhythm"]:
        onsets.append(t)
        t += d*beat
    n = int((t + 0.3)*fs)
    sig = np.zeros(n)
    k = np.arange(int(0.12*fs))/fs
    burst = np.sin(2*np.pi*1400*k)*np.exp(-k/0.018)*np.sin(np.pi*np.clip(k/0.0008, 0, 0.5))
    amp = P.P_REF*10**(94/20)*math.sqrt(2)
    for o in onsets:
        i = int(round(o*fs))
        sig[i:i + len(burst)] += amp*burst[:max(0, min(len(burst), n - i))]
    return sig, onsets


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


# ------------------------------------------------- eight-task set: progress of the three added tasks
def _ramp(x, full, zero):
    """1 at or below `full`, 0 at or beyond `zero`, linear between (the five-task set's grading)."""
    return float(np.clip((zero - x)/(zero - full), 0.0, 1.0))


def take_sticks_progress(both_s, single_s, hold_s, returned_share, strikes):
    """take-sticks progress (0..1): hold x (1/2 + 1/2 x share of the sticks back on a trap table) / (1 + strikes), with
    hold = min(1, max(both_s/hold_s, 1/2 x mean_i min(1, single_s[i]/hold_s))) - both_s the longest time both sticks
    were held ready together, single_s[i] each stick's own longest. A stick held ready alone earns at most a quarter of
    the hold, both together all of it; the sticks lying where they started (nothing held) score 0, and putting them
    back earns the other half only after a hold."""
    one = 0.5*float(np.mean([min(1.0, s/hold_s) for s in single_s])) if len(single_s) else 0.0
    hold = min(1.0, max(both_s/hold_s, one))
    return hold*(0.5 + 0.5*returned_share)/(1.0 + strikes)


def rests_on_support(mid, axis, supports, kind="trap_table", R=P.STICK_RETURN):
    """Whether a tool with midpoint `mid` and unit axis `axis` lies on one of the `supports` of `kind`
    (layout.tool_rack: axis-aligned boxes, centre/half/top_z): its midpoint over the support's top, no lower than the
    top and at most max_above_top_m above it, its axis within max_tilt_deg of horizontal (params.STICK_RETURN)."""
    if abs(float(axis[2])) > math.sin(math.radians(R["max_tilt_deg"])):
        return False
    for s in supports:
        (cx, cy), (hx, hy) = s["center"], s["half"][:2]
        if s["kind"] == kind and abs(mid[0] - cx) <= hx and abs(mid[1] - cy) <= hy and \
                s["top_z"] <= mid[2] <= s["top_z"] + R["max_above_top_m"]:
            return True
    return False


def still_over(track, now, window_s, max_m):
    """Whether the sampled positions `track` [(t, xyz)] cover the last window_s before `now` and none of those lies
    farther than max_m from the latest."""
    if not track or track[0][0] > now - window_s + 1e-9:
        return False
    last = np.asarray(track[-1][1], float)
    return all(float(np.linalg.norm(np.asarray(p, float) - last)) <= max_m for t, p in track if t >= now - window_s - 1e-9)


def steady_beat_progress(onsets, levels, clean, goal, TP=P.TASK_PROGRESS):
    """steady-beat progress (0..1, before the rim factor). The strokes are split into the two sections at the stroke k
    that gives the most credit - with the right count and the change in its place k = beats, and a missed beat does not
    push every later interval into the wrong section:
        count_k x mean(tempo_1, tempo_2, steady_1, steady_2, level, clean)
    count_k = min(k, beats)/max(k, beats) x min(n - k, faster_beats)/max(n - k, faster_beats): strokes per section
    against those asked, so extra or missing strokes and a change in the wrong place cost. Per section, over its
    intervals (the one across the change excluded, as the check does): tempo 1 within 1 % of its stated interval, 0 at
    beat_tempo_zero_rel; steady 1 at CV <= 3 %, 0 at steadiness_zero_cv; a section with fewer than two intervals scores
    0 on both. level: the SD of the stroke levels, 1 at <= 2 dB, 0 at level_sd_zero_db, times the share of strokes
    with a level; clean: the share of strokes within the clean contact time. Returns (progress, details)."""
    n, n1, n2 = len(onsets), goal["beats"], goal["faster_beats"]
    if n < 2:
        return 0.0, dict(progress_split_stroke=None)
    T = (60/goal["bpm"], 60/(goal["bpm"]*(1 + goal["faster_pct"]/100)))
    d = np.diff(np.asarray(onsets, float))
    c1, c2 = np.r_[0.0, np.cumsum(d)], np.r_[0.0, np.cumsum(d*d)]

    def section(i0, i1, Ts):                       # the intervals d[i0:i1], against the section's interval Ts
        m = i1 - i0
        mu = (c1[i1] - c1[i0])/m if m >= 2 else 0.0
        if mu <= 0:
            return 0.0, 0.0
        cv = math.sqrt(max((c2[i1] - c2[i0])/m - mu*mu, 0.0))/mu
        return _ramp(abs(mu/Ts - 1), 0.01, TP["beat_tempo_zero_rel"]), _ramp(cv, 0.03, TP["steadiness_zero_cv"])
    lv = [v for v in levels if v is not None]
    level = _ramp(float(np.std(lv)), 2.0, TP["level_sd_zero_db"])*len(lv)/n if lv else 0.0
    clean_share = float(np.mean(clean))
    best = (-1.0, None, None)
    for k in sorted(range(1, n), key=lambda k: (abs(k - n1), k)):
        count = min(k, n1)/max(k, n1)*min(n - k, n2)/max(n - k, n2)
        (t1, s1), (t2, s2) = section(0, k - 1, T[0]), section(k, n - 1, T[1])
        p = count*float(np.mean([t1, t2, s1, s2, level, clean_share]))
        if p > best[0] + 1e-12:
            best = (p, k, dict(count=count, tempo=[t1, t2], steady=[s1, s2], level=level, clean=clean_share))
    p, k, parts = best
    return p, dict(progress_split_stroke=k,
                   progress_parts={a: [round(x, 4) for x in b] if isinstance(b, list) else round(b, 4)
                                   for a, b in parts.items()})


def accent_ostinato_progress(onsets, levels, clean, goal, TP=P.TASK_PROGRESS):
    """accent-ostinato progress (0..1, before the rim factor):
        count x (accent_share x accent + (1 - accent_share) x mean(tempo, steady, clean))
    count = min(n, N)/max(n, N) for the N = bars x 4 strokes asked; tempo: the mean interval 1 within 2 % of the beat,
    0 at tempo_zero_rel; steady: the intervals' CV 1 at <= 4 %, 0 at steadiness_zero_cv (fewer than three strokes: 0);
    clean: the share of strokes within the clean contact time; accent: over the accent places of the pattern among the
    first min(n, N) strokes (counted from the first stroke, as the check does), each accent's level above the mean of
    the soft strokes there, 1 at >= accent_db and 0 at no contrast (a missing level 0). Returns (progress, details)."""
    pat = goal["pattern"]
    N, n = goal["bars"]*len(pat), len(onsets)
    if not n:
        return 0.0, dict(progress_parts=None)
    count = min(n, N)/max(n, N)
    d = np.diff(np.asarray(onsets, float))
    tempo = _ramp(abs(d.mean()/(60/goal["bpm"]) - 1), 0.02, TP["tempo_zero_rel"]) if n >= 2 else 0.0
    steady = _ramp(d.std()/d.mean(), 0.04, TP["steadiness_zero_cv"]) if n >= 3 else 0.0
    k = min(n, N)
    soft = [levels[i] for i in range(k) if not pat[i % len(pat)] and levels[i] is not None]
    acc = [levels[i] for i in range(k) if pat[i % len(pat)]]
    accent = float(np.mean([0.0 if v is None else float(np.clip((v - np.mean(soft))/goal["accent_db"], 0.0, 1.0))
                            for v in acc])) if acc and soft else 0.0
    clean_share = float(np.mean(clean))
    a = TP["accent_share"]
    p = count*(a*accent + (1 - a)*float(np.mean([tempo, steady, clean_share])))
    return p, dict(progress_parts=dict(count=round(count, 4), accent=round(accent, 4), tempo=round(tempo, 4),
                                       steady=round(steady, 4), clean=round(clean_share, 4)))


# ====================================================================================================== evaluator
class Evaluator:
    def __init__(self, env):
        self.env = env
        self.meta = {}
        self._L = np.zeros(0, np.float32)          # L_AF of the client microphone, per sample
        self._zi = None
        self._done = 0
        self._refs = {}
        self.hold = dict(best_s=0.0, since=None)
        self.hold_single = {}                      # take-sticks: per stick, the same timer for that stick alone
        self.stick_track = {}                      # take-sticks: per stick, (t, midpoint) every 10 ms over the last 0.5 s
        self.rim_events = []
        self.floor_tools = set()
        self.energy_trace = []                     # (t, J) of the strike-and-stop bar, every 10 ms (monitor)
        self._ke = []
        self.stop_log = []                         # strike-and-stop, every tick: (t, KE, free KE from the stroke, from the cue)
        self._stop_ref = {}                        # the bar's modal state just after the stroke and at the cue
        self.ring_log = {}

    def snapshot(self):
        return copy.deepcopy(dict(L=self._L, zi=self._zi, done=self._done, hold=self.hold, rim=self.rim_events,
                                  floor=self.floor_tools, refs=self._refs, energy=self.energy_trace, ke=self._ke,
                                  stop_log=self.stop_log, stop_ref=self._stop_ref, ring_log=self.ring_log,
                                  hold_single=self.hold_single, stick_track=self.stick_track))

    def load(self, s):
        s = copy.deepcopy(s)
        self._L, self._zi, self._done, self.hold = s["L"], s["zi"], s["done"], s["hold"]
        self.hold_single, self.stick_track = s.get("hold_single", {}), s.get("stick_track", {})
        self.rim_events, self.floor_tools, self._refs = s["rim"], s["floor"], s["refs"]
        self.energy_trace = s.get("energy", [])
        self._ke = s.get("ke", [])
        self.stop_log, self._stop_ref = s.get("stop_log", []), s.get("stop_ref", {})
        self.ring_log = s.get("ring_log", {})

    def sample_ring(self):
        """Every tick (melody): for each xylophone bar struck within its longest ring window (MELODY_EVAL ring_beats at
        the stated tempo), the bar's vibration energy (static deflection removed) against the free continuation of its
        state when the strike ended (world.ring_ref), in dB, keyed '<surface>@<end sample>'. The ring is judged on this
        trace where the note's window closes (not at the end of whichever touch began inside it)."""
        w = self.env.world
        span = P.MELODY_EVAL["ring_beats"]*60.0/self.env.layout["goal"]["bpm"] + 0.002
        t = w.sim_time
        for surf, (end, zref) in w.ring_ref.items():
            if not surf.startswith("bar_") or t - end/P.AUDIO_FS > span:
                continue
            f = next(x for x in w.surfaces if x["name"] == surf)
            st, sl = w.structs[f["struct"]], f["slice"]
            free = float(st.dof_energy(zref*np.exp(st.lam*(t - end/P.AUDIO_FS)))[sl].sum())
            now = float(st.dof_energy(w._dyn_z(f["struct"]))[sl].sum())
            self.ring_log.setdefault(f"{surf}@{end}", []).append(
                (round(t, 6), round(10.0*math.log10(max(free, 1e-30)/max(now, 1e-30)), 4)))

    def ring_damping(self, strike, w_end):
        """dB below its free ring at which the strike's bar stands when its ring window closes at w_end (or at the last
        sample, if the episode ended first): 0 if nothing was logged."""
        keys = [k for k in self.ring_log if k.split("@")[0] == strike["surface"]
                and strike["start"] <= int(k.split("@")[1]) <= strike["end"]]
        if not keys:
            return 0.0
        vals = [db for t, db in self.ring_log[max(keys, key=lambda k: int(k.split("@")[1]))] if t < w_end - 1e-9]
        return max(0.0, vals[-1]) if vals else 0.0

    # -------------------------------------------------------------- levels
    def _update_levels(self):
        from scipy.signal import sosfilt, lfilter
        from .acoustics import a_weighting_sos
        A = self.env.world.acoustics
        if A is None:
            return
        n = A.n
        if n <= self._done:
            return
        x = A.signal("client_mic", self._done, n)
        if self._zi is None:
            self._sos = a_weighting_sos()
            self._zi = (np.zeros((self._sos.shape[0], 2)), np.zeros(1))
        y, z1 = sosfilt(self._sos, x, zi=self._zi[0])
        a = 1 - math.exp(-1/(0.125*P.AUDIO_FS))
        e, z2 = lfilter([a], [1, -(1 - a)], y**2, zi=self._zi[1])
        self._zi = (z1, z2)
        self._L = np.r_[self._L, (10*np.log10(np.maximum(e, 1e-30)/P.P_REF**2)).astype(np.float32)]
        self._done = n

    def level_max(self, t0, t1):
        i0, i1 = int(t0*P.AUDIO_FS), int(t1*P.AUDIO_FS)
        seg = self._L[max(0, i0):max(0, min(len(self._L), i1))]
        return float(seg.max()) if len(seg) else None

    def leq(self, t0, t1):
        """A-weighted equivalent level (dB) of the client microphone over [t0, t1] - no 125 ms meter smoothing (an
        L_AF meter falls at most 35 dB/s, too slow to see a stopped bar). Fake environments: the L_AF trace's max."""
        A = self.env.world.acoustics
        if A is None:
            return self.level_max(t0, t1)
        from scipy.signal import sosfilt
        from .acoustics import a_weighting_sos
        i0, i1 = int(t0*P.AUDIO_FS), int(t1*P.AUDIO_FS)
        pre = int(0.2*P.AUDIO_FS)
        x = A.signal("client_mic", max(0, i0 - pre), min(A.n, i1))
        if len(x) <= i0 - max(0, i0 - pre):
            return None
        y = sosfilt(a_weighting_sos(), x)[i0 - max(0, i0 - pre):]
        return float(10*np.log10(max(np.mean(y**2), 1e-30)/P.P_REF**2))

    def level_at(self, t):
        i = int(t*P.AUDIO_FS)
        return float(self._L[i]) if 0 <= i < len(self._L) else None

    # -------------------------------------------------------------- strikes
    def strikes(self, surfaces=None, min_speed=0.0):
        """Contact segments clustered into strikes per surface."""
        opened = getattr(self.env.world, "open_segments", None)
        log = list(self.env.world.contact_log) + (list(opened()) if callable(opened) else [])
        log = [s for s in log if surfaces is None or s["surface"] in surfaces]
        log = sorted(log, key=lambda s: s["start"])
        out, open_ = [], {}
        for s in log:
            gap = P.STRIKE_CLUSTER_S["drum" if s["kind"] == "head" else "bar"]*P.AUDIO_FS
            cur = open_.get(s["surface"])
            if cur is not None and s["start"] - cur["end"] <= gap:
                cur["segments"].append(s)
                cur["end"] = max(cur["end"], s["end"])
                cur["peak_N"] = max(cur["peak_N"], s["peak_N"])
                cur["impulse_Ns"] += s["impulse_Ns"]
                cur["E_after"] = s.get("E_after", cur["E_after"])
                cur["E_peak"] = max(cur["E_peak"], s.get("E_max_dyn", 0.0))
                cur["tools"].add(s["tool"] or s["striker"])
                for k in ("mode1_share", "bar_energy_J", "all_bars_energy_J"):
                    if k in s:
                        cur[k] = s[k]
                for key in ("bars_added_J", "bars_stroke_J"):
                    if key in s:
                        acc = cur.setdefault(key, {})
                        for n, v in s[key].items():
                            acc[n] = acc.get(n, 0.0) + v
                continue
            cur = dict(surface=s["surface"], kind=s["kind"], note=s.get("note"), start=s["start"], end=s["end"],
                       peak_N=s["peak_N"], impulse_Ns=s["impulse_Ns"], v_in=s["v_in"], segments=[s],
                       E_before=s.get("E_before", 0.), E_after=s.get("E_after", 0.), tools={s["tool"] or s["striker"]},
                       striker=s["striker"], E_peak=s.get("E_max_dyn", 0.0))
            for k in ("mode1_share", "bar_energy_J", "all_bars_energy_J"):
                if k in s:
                    cur[k] = s[k]
            for key in ("bars_added_J", "bars_stroke_J"):
                if key in s:
                    cur[key] = dict(s[key])
            open_[s["surface"]] = cur
            out.append(cur)
        for st in out:
            st["onset_s"] = st["start"]/P.AUDIO_FS
            st["contact_ms"] = (st["end"] - st["start"])/P.AUDIO_FS*1e3
            st["first_contact_ms"] = (st["segments"][0]["end"] - st["segments"][0]["start"])/P.AUDIO_FS*1e3
        return [s for s in out if abs(s["v_in"]) >= min_speed]

    def _restrike(self, strikes):
        lo, hi = P.RESTRIKE_WINDOW_S
        bad = 0
        for a, b in zip(strikes, strikes[1:]):
            if a["surface"] == b["surface"] and lo <= b["onset_s"] - a["end"]/P.AUDIO_FS <= hi and \
                    (b["tools"] & a["tools"]):
                bad += 1
        return bad

    def strike_level(self, st):
        return self.level_max(st["onset_s"], st["onset_s"] + P.LEVEL_WINDOW_S)

    def natural_decay_db(self, struct_name, seconds):
        """dB the struck structure's energy-carrying mode decays in `seconds` on its own (60 dB per T60): the complex
        mode with the largest share of the current modal amplitude |z|^2 |lambda|^2 (the G4 resonator bar's energy sits
        in its mode-1 pair, T60 0.2 s - not in the slower 855 Hz mode a 'slowest of the three lowest' rule picked)."""
        w = self.env.world
        st = getattr(w, "structs", {}).get(struct_name)
        if st is None:
            return 20.0*seconds/0.3          # fake environments (evaluator tests): 20 dB per 0.3 s
        wgt = np.abs(st.z)**2*np.abs(st.lam)**2
        j = int(np.argmax(wgt)) if wgt.max() > 0 else int(np.argmin(np.abs(st.lam.imag) + 1e9*(st.lam.imag <= 0)))
        t60 = float(6.91/(-st.lam[j].real))
        return 60.0*seconds/t60

    def energy_at(self, t):
        """The task structure's vibration energy (J) at time t, log-interpolated in the monitor's 10 ms trace."""
        tr = getattr(self.env, "energy_trace", None) or self.energy_trace
        if not tr or t < tr[0][0] or t > tr[-1][0]:
            return None
        ts = np.array([a for a, _ in tr])
        Es = np.log(np.maximum(np.array([b for _, b in tr]), 1e-30))
        return float(np.exp(np.interp(t, ts, Es)))

    # -------------------------------------------------------------- physical anchors
    def reference_level(self, surface_name, dynamic, striker="stick"):
        """L_AF,max at the client's ear of a reference stroke: the named striker (stick tip: its free-pivot effective
        mass 8.8 g; mallet head: the free mallet's effective mass at the head; fingertip: a tap,
        params.FINGERTIP_TAP - 21.5 g at 0.5 m/s, measured finger impedance and keystroke speeds) at the reference speed
        of `dynamic` (params.REFERENCE_SPEEDS: Dahl's measured stick speeds on the drum; estimates around Rucz's 1 m/s
        on the bars), struck at the bar's centre or 40 mm from the head's centre, through the live instrument model,
        contact law and acoustic chain."""
        key = (surface_name, dynamic, striker)
        if key in self._refs:
            return self._refs[key]
        from . import contact as CT
        from . import modal as MD
        from .acoustics import Acoustics, level_fast
        w = self.env.world
        f = next(x for x in w.surfaces if x["name"] == surface_name)
        st = copy.deepcopy(w.structs[f["struct"]])
        st.z[:] = 0
        kind = "drum" if f["kind"] == "head" else "bar"
        if striker == "stick":
            from . import modes
            mat, m_eff, R = "hickory_tip", 0.0088, float(modes.stick()["tip_r"])
            v = P.REFERENCE_SPEEDS[kind][dynamic]
        elif striker == "mallet":
            mat, R = "rubber_mallet", P.MALLET["head_r"]
            mb = w._newton_to_mj[w.scene.bodies["mallet_left"]]
            M, I = float(w.mjm.body_mass[mb]), float(np.max(w.mjm.body_inertia[mb]))
            xh = float(w.scene.strikers[[s["name"] for s in w.scene.strikers].index("mallet_left_head")]["p"][0])
            r = xh - float(w.mjm.body_ipos[mb][0])
            m_eff = 1.0/(1.0/M + r*r/I)
            v = P.REFERENCE_SPEEDS[kind][dynamic]
        else:
            tip = next(s for s in w.strikers if s.get("fingertip"))
            mat, R = "rubber_pad", float(tip["r"])
            m_eff, v = P.FINGERTIP_TAP["m_eff_kg"], P.FINGERTIP_TAP["v_m_s"]
        if f["kind"] == "bar":
            b = np.zeros(st.n)
            b[f["slice"]] = MD.bar_shape(f["bar"], 0.0, 0.0, P.BAR_THICK_M/2, np.array([0., 0., 1.]))
            surf_mat, c_res = "rosewood", 0.0
        else:
            patch = CT.patch_radius(mat, "mylar", R, st.membrane["tension"])
            b = MD.membrane_shape(st, 0.04, 0.0, patch)
            c_res = MD.membrane_residual(st, 0.04, patch, b)
            surf_mat = "mylar"
        K, a, lam, _ = P.CONTACT[mat][surf_mat]
        dt = st.dt
        c, wv = st.out_row(b), st.in_col(b)
        S = st.compliance(c, wv, half=False) + dt**2/(2*m_eff) + c_res
        sk = CT.FreeStriker(m_eff, 1e-5, -v)
        A = Acoustics(dict(self.env.world.scene.mics), seed=self.env.config.seed)
        d0 = -1e-5
        n_ticks = int(0.30/P.RIGID_DT_S)
        for tk in range(n_ticks):
            Z = np.empty((st.nc, P.SAMPLES_PER_TICK), complex)
            for k in range(P.SAMPLES_PER_TICK):
                uf = 2*np.real(np.dot(c, st.E*st.z))
                F, _ = CT.contact_force(d0, uf - sk.free_end(dt), S, K, a, lam, dt)
                Z[:, k] = st.step_radiating(wv*(-F) if F else None)
                sk.step(F, dt)
                d0 = 2*np.real(np.dot(c, st.z)) - c_res*F - sk.y
            if st.radiators is not None:
                A.add_block(st.radiators, Z, tk*P.SAMPLES_PER_TICK)
            A.advance(P.SAMPLES_PER_TICK)
            if tk > 20 and sk.v > 0 and d0 < -1e-3:
                # the striker has left: ring freely for the rest of the window
                for tk2 in range(tk + 1, n_ticks):
                    act, Zf = st.free_block()
                    A.add_block(st.radiators, Zf, tk2*P.SAMPLES_PER_TICK, act)
                    A.advance(P.SAMPLES_PER_TICK)
                break
        L = float(level_fast(A.signal("client_mic")).max())
        self._refs[key] = L
        return L

    # -------------------------------------------------------------- per-tick monitor (rim strokes, holding, drops)
    def rim_check(self):
        """Every tick: rigid contacts of a tool with the drum's hoop/shell (a rim stroke; MuJoCo contact, no vibro)."""
        w = self.env.world
        sc = w.scene
        if "drum" not in sc.bodies or not w.mjd.ncon:
            return
        if not hasattr(self, "_tool_bodies"):
            self._tool_bodies = {sc.items[t["name"]]["body"]: t["name"] for t in sc.layout["tools"]}
        f = w.pair_contact_forces(list(self._tool_bodies), [sc.bodies["drum"]])
        for (tb, _), F in f.items():
            if F > 0.5 and (not self.rim_events or w.sim_time - self.rim_events[-1]["t"] > 0.02):
                self.rim_events.append(dict(t=round(w.sim_time, 4), tool=self._tool_bodies[tb], force_N=round(F, 2)))

    def monitor(self):
        env = self.env
        w = env.world
        sc = w.scene
        if env.config.task == "strike-and-stop" and getattr(self, "_ke", None):
            # vibration energy = 2 x the kinetic energy averaged over the last 10 ticks (2.6 periods of the C4 bar's
            # mode-1 pair): the total modal energy also counts the static strain energy of the bar bent under a
            # resting fingertip (~1e-6-1e-4 J at 0.5-5 N), which is no ringing
            self.energy_trace.append((round(w.sim_time, 6), 2.0*float(np.mean(self._ke))))
            self._ke = []
        for t in sc.layout["tools"]:
            p, _ = w.body_pose(t["name"])
            if p[2] < 0.3:
                self.floor_tools.add(t["name"])
        if env.config.task == "take-sticks":
            names = env.layout["goal"]["sticks"]
            ready = {n: self._stick_ready(n) for n in names}
            for n in names:
                self._hold_timer(self.hold_single.setdefault(n, dict(best_s=0.0, since=None)), ready[n], w.sim_time)
            self._hold_timer(self.hold, all(ready.values()), w.sim_time)
            # each stick's midpoint, for "still" in the return check (params.STICK_RETURN)
            keep = w.sim_time - P.STICK_RETURN["still_s"] - 0.02
            for n in names:
                p, R = w.body_pose(n)
                tr = self.stick_track.setdefault(n, [])
                tr.append((round(w.sim_time, 6), (p + R[:, 0]*sc.items[n]["length"]/2).tolist()))
                while tr[0][0] < keep:
                    tr.pop(0)

    @staticmethod
    def _hold_timer(h, ok, t):
        """A continuous-hold timer: `since` the hold began (None when not holding), `best_s` its longest run."""
        if ok:
            if h["since"] is None:
                h["since"] = t
            h["best_s"] = max(h["best_s"], t - h["since"])
        else:
            h["since"] = None

    def sample_vibration(self):
        """Every tick (strike-and-stop only): the struck bar's modal kinetic energy (bar and box air), and - once the
        stroke has ended - the kinetic energy its free ring would have now: the exact free continuation exp(lam t) z of
        the bar's modal state just after the stroke and at the cue (privileged; the linear modal model makes the
        counterfactual exact, so the judgement does not extrapolate a measured decay through the bar-box beat)."""
        env = self.env
        g = env.layout["goal"]
        st = env.world.structs.get(f"resbar_{g['bar']}")
        if st is None:
            return
        _, v = st.physical()
        ke = 0.5*float(np.sum(v**2))
        self._ke = getattr(self, "_ke", None) or []
        self._ke.append(ke)
        cue = env.cue.get("at")
        if cue is None:
            return
        t = env.world.sim_time
        ref = self._stop_ref
        if "z_stroke" not in ref:
            ref.update(t_stroke=t, z_stroke=st.z.copy())
        if "z_cue" not in ref and t >= cue - 1e-9:
            ref.update(t_cue=t, z_cue=st.z.copy())
        if t > cue + g["judge_after_s"] + g["judge_window_s"] + 0.005:
            return

        def free(z0, t0):
            _, vf = st.physical(z0*np.exp(st.lam*(t - t0)))
            return 0.5*float(np.sum(vf**2))
        self.stop_log.append((round(t, 6), ke, free(ref["z_stroke"], ref["t_stroke"]),
                              free(ref["z_cue"], ref["t_cue"]) if "z_cue" in ref else None))

    def _held_by(self, tool):
        """Arm whose two fingers both touch the tool (MuJoCo contacts), or None."""
        w = self.env.world
        body = w.scene.items[tool]["body"]
        for arm, fingers in w.scene.finger_bodies.items():
            f = w.pair_contact_forces(fingers, [body])
            touching = {fb for (fb, _), F in f.items() if F > 0.2}
            if len(touching) >= 2:
                return arm
        return None

    def _stick_ready(self, name):
        """One stick held ready: pinched by two fingers of one hand, its tip over the head (at least 2 cm in from its
        edge) and 3-15 cm above it."""
        w = self.env.world
        if "drum" not in w.scene.bodies or self._held_by(name) is None:
            return False
        pd, Rd = w.body_pose("drum")
        ps, Rs = w.body_pose(name)
        tip = ps + Rs @ np.array([P.STICK["L"] - P.STICK["tip"]/2, 0, 0])
        loc = Rd.T @ (tip - pd)
        return math.hypot(loc[0], loc[1]) <= P.DRUM_RADIUS_M - 0.02 and 0.03 <= loc[2] <= 0.15

    # -------------------------------------------------------------- evaluation
    def evaluate(self):
        env = self.env
        w = env.world
        task = env.config.task
        g = env.layout["goal"]
        self._update_levels()
        checks, meta = {}, {}
        progress = None
        TP = P.TASK_PROGRESS
        ramp = lambda x, full, zero: float(np.clip((zero - x)/(zero - full), 0.0, 1.0))
        rim_factor = 1.0/(1.0 + len(self.rim_events))
        if task == "take-sticks":
            checks["held_ready"] = self.hold["best_s"] >= g["hold_s"]
            strikes = self.strikes(min_speed=0.05)
            checks["no_strike"] = not strikes
            # back on either trap table, resting and still (params.STICK_RETURN), not within 30 mm of where it started
            # (_in_rack)
            back = {n: self._on_trap_table(n) for n in g["sticks"]}
            checks["sticks_returned"] = all(back.values())
            single = [self.hold_single.get(n, {}).get("best_s", 0.0) for n in g["sticks"]]
            progress = take_sticks_progress(self.hold["best_s"], single, g["hold_s"], float(np.mean(list(back.values()))),
                                            len(strikes))
            meta.update(hold_best_s=round(self.hold["best_s"], 3), hold_single_best_s=[round(s, 3) for s in single],
                        returned=back, strikes=len(strikes))
        elif task in ("single-tone", "strike-and-stop", "roll"):
            bar = "rbar_" + g["bar"]
            st = self.strikes(min_speed=0.05)
            on = [s for s in st if s["surface"] == bar]
            other = [s for s in st if s["surface"] != bar]
            meta.update(strikes_on_bar=len(on), strikes_elsewhere=len(other))
            if task == "single-tone":
                checks["one_stroke"] = len(on) == 1 and not other
                if on:
                    s = on[0]
                    ref = self.reference_level(bar, "mf", "mallet")
                    L = self.strike_level(s)
                    checks["clean"] = s["contact_ms"] <= P.CONTACT_MAX_S["bar"]*1e3 and self._restrike(st) == 0
                    checks["tone"] = s.get("mode1_share", 0) >= 0.5
                    checks["level"] = L is not None and abs(L - ref) <= P.LEVEL_BAND_DB
                    meta.update(level_dBA=L, reference_dBA=round(ref, 2), mode1_share=s.get("mode1_share"),
                                contact_ms=round(s["contact_ms"], 3))
            elif task == "strike-and-stop":
                cue = env.cue.get("at")
                if cue is None:                     # never struck: no cue
                    cue = float("inf")
                # only contacts that add vibration energy are strikes; a damping touch removes it (so a touch
                # landing just before the cue - an anticipated stop - is not a second stroke). A contact
                # whose striker stays on the bar - closed or still open at the evaluation - is also a strike when the
                # largest vibration energy seen while it lasted (E_peak, per tick) at least doubles the bar's energy at
                # its start (+3 dB: a 1.9 m/s felt adds x15; a touch set down on the ringing bar, which can lift it by
                # a few per cent as it lands, does not count)
                adds = lambda s: (s["E_after"] > 1.05*s["E_before"] + 1e-7
                                  or s.get("E_peak", 0.0) > 2.0*s["E_before"] + 1e-6)
                before = [s for s in on if s["onset_s"] < cue and adds(s)]
                after = [s for s in st if s["onset_s"] >= cue and adds(s)]
                checks["struck_once"] = len(before) == 1
                checks["no_strike_after_cue"] = not after
                # judged on the struck bar's own vibration energy (privileged modal state, kinetic energy of the bar and
                # its box air, so the static bending under a resting fingertip does not count) against its exact free
                # ring from the same state (sample_vibration) - not on the level at the client's ear, which the room's
                # reverberant tail keeps up after any stop, and not against a decay rate extrapolated through the
                # bar-box beat (such a rule flips with the beat phase).
                # stopped: one exchange period after the cue the bar rings >= min_damping_db below its free ring over
                # judge_window_s; rang_until_cue: in the 50 ms before the cue it was not yet damped that much.
                fin = math.isfinite(cue)
                log = self.stop_log or list(getattr(self.env, "stop_log", None) or [])
                ref_t = self._stop_ref.get("t_cue", cue) if fin else None
                a0 = ref_t + g["judge_after_s"] if fin else None
                a1 = a0 + g["judge_window_s"] if fin else None
                win = [(k, fc) for t_, k, fs, fc in log if fin and fc is not None and a0 - 1e-9 <= t_ < a1 - 1e-9]
                pre = [(k, fs) for t_, k, fs, fc in log if fin and ref_t - 0.05 - 1e-9 <= t_ < ref_t - 1e-9]
                complete = len(win) >= int(0.9*g["judge_window_s"]/P.RIGID_DT_S)
                damp = 10*math.log10(max(np.mean([f for _, f in win]), 1e-30)/max(np.mean([k for k, _ in win]), 1e-30)) \
                    if complete else None
                early = 10*math.log10(max(np.mean([f for _, f in pre]), 1e-30)/max(np.mean([k for k, _ in pre]), 1e-30)) \
                    if pre else None
                checks["stopped"] = complete and damp >= g["min_damping_db"]
                checks["rang_until_cue"] = bool(before) and early is not None and early < g["min_damping_db"]
                # progress: one stroke (1/n for n strokes before the cue), the ring kept until the cue (half) and the
                # damping reached in the judging window against the required dB (half), less for strokes after the cue
                stroke = 1.0/len(before) if before else 0.0
                stop = 0.0 if damp is None else float(np.clip(damp/g["min_damping_db"], 0.0, 1.0))
                rang = 1.0 if checks["rang_until_cue"] else 0.0
                progress = stroke*(0.5*rang + 0.5*stop)/(1.0 + len(after))
                E_stroke = before[0]["E_after"] if before else None
                Lc = self.leq(cue - 0.03, cue) if fin else None
                La = self.leq(a0, a1) if fin else None
                meta.update(damping_db=None if damp is None else round(damp, 2), required_damping_db=g["min_damping_db"],
                            damping_before_cue_db=None if early is None else round(early, 2),
                            judge_window_s=None if not fin else [round(a0, 4), round(a1, 4)], judge_complete=complete,
                            energy_at_cue_J=self.energy_at(cue) if fin else None, energy_after_stroke_J=E_stroke,
                            free_ring_drop_to_window_db=round(10*math.log10(max(log[0][2], 1e-30)/max(win[-1][1], 1e-30)), 2)
                            if win and log else None,
                            level_at_cue_dBA=Lc, level_in_window_dBA=La, cue_s=cue if fin else None,
                            judged_on="struck bar vibration energy vs its exact free ring")
            else:
                ons = [s["onset_s"] for s in on]
                best = 0
                for i in range(len(ons)):
                    j = i
                    while j + 1 < len(ons) and ons[j + 1] - ons[i] <= g["duration_s"]:
                        j += 1
                    best = max(best, j - i + 1)
                checks["rate"] = best >= int(g["rate_hz"]*g["duration_s"])
                tools = [sorted(s["tools"])[0] for s in on]
                checks["alternating"] = len(tools) >= 2 and all(a != b for a, b in zip(tools, tools[1:]))
                # evenness is judged on what the player controls: each stroke's approach speed, in dB (20 log10 v:
                # the striker's incoming kinetic energy). The energy each stroke adds also depends, on a ringing bar,
                # on the bar's phase at impact (a phantom roll with strokes phase-locked to the bar reads 0.13 dB);
                # that spread is reported only
                spd = [20*math.log10(max(abs(s["v_in"]), 1e-6)) for s in on]
                inc = [10*math.log10(max(s["E_after"] - s["E_before"], 1e-12)) for s in on]
                checks["even"] = len(spd) >= 2 and float(np.std(spd)) <= g["level_sd_db"]
                meta.update(max_strokes_in_window=best,
                            approach_speed_sd_db=round(float(np.std(spd)), 2) if spd else None,
                            energy_increment_sd_db=round(float(np.std(inc)), 2) if inc else None)
        elif task in ("dynamics", "steady-beat", "accent-ostinato", "finger-tap", "echo"):
            st = [s for s in self.strikes(["drumhead"], min_speed=0.05)]
            lv = [self.strike_level(s) for s in st]
            meta.update(strikes=len(st), onsets_s=[round(s["onset_s"], 4) for s in st],
                        levels_dBA=[None if v is None else round(v, 2) for v in lv])
            clean = all(s["contact_ms"] <= P.CONTACT_MAX_S["drum"]*1e3 for s in st) and self._restrike(st) == 0
            checks["clean"] = bool(st) and clean
            checks["no_rim"] = not self.rim_events
            ioi = np.diff([s["onset_s"] for s in st])
            if task == "dynamics":
                refs = [self.reference_level("drumhead", d, "stick") for d in g["sequence"]]
                checks["three_strokes"] = len(st) == 3
                if len(st) == 3:
                    checks["levels"] = all(v is not None and abs(v - r) <= P.LEVEL_BAND_DB for v, r in zip(lv, refs))
                    checks["ordered"] = all(lv[i + 1] - lv[i] >= g["min_step_db"] for i in range(2))
                meta.update(reference_dBA=[round(r, 2) for r in refs])
            elif task == "steady-beat":
                n1, n2 = g["beats"], g["faster_beats"]
                T1, T2 = 60/g["bpm"], 60/(g["bpm"]*(1 + g["faster_pct"]/100))
                checks["count"] = len(st) == n1 + n2
                if len(st) == n1 + n2:
                    a, b = ioi[:n1 - 1], ioi[n1:]
                    checks["tempo"] = abs(a.mean()/T1 - 1) <= 0.01 and abs(b.mean()/T2 - 1) <= 0.01
                    checks["steady"] = a.std()/a.mean() <= 0.03 and b.std()/b.mean() <= 0.03
                    checks["even_level"] = all(v is not None for v in lv) and float(np.std(lv)) <= 2.0
                    meta.update(ioi_mean_s=[round(float(a.mean()), 4), round(float(b.mean()), 4)],
                                ioi_cv=[round(float(a.std()/a.mean()), 4), round(float(b.std()/b.mean()), 4)],
                                level_sd_db=round(float(np.std(lv)), 2) if all(v is not None for v in lv) else None)
                # progress (eight-task set): steady_beat_progress, divided by (1 + rim hits)
                progress, info = steady_beat_progress([s["onset_s"] for s in st], lv,
                                                      [s["contact_ms"] <= P.CONTACT_MAX_S["drum"]*1e3 for s in st], g, TP)
                progress *= rim_factor
                meta.update(info)
            elif task == "accent-ostinato":
                n = 4*g["bars"]
                checks["count"] = len(st) == n
                if len(st) == n:
                    # judged whenever the count is right; a stroke without a level fails `accents` (so the task cannot
                    # pass without them when no level can be read)
                    checks["accents"] = False
                    checks["steady"] = ioi.std()/ioi.mean() <= 0.04 and abs(ioi.mean()/(60/g["bpm"]) - 1) <= 0.02
                    meta.update(ioi_cv=round(float(ioi.std()/ioi.mean()), 4))
                    if all(v is not None for v in lv):
                        acc = np.array(lv[3::4]); un = np.array([v for i, v in enumerate(lv) if i % 4 != 3])
                        checks["accents"] = float(acc.min() - un.mean()) >= g["accent_db"]
                        meta.update(accent_minus_unaccented_db=round(float(acc.min() - un.mean()), 2))
                # progress (eight-task set): accent_ostinato_progress, divided by (1 + rim hits)
                progress, info = accent_ostinato_progress([s["onset_s"] for s in st], lv,
                                                          [s["contact_ms"] <= P.CONTACT_MAX_S["drum"]*1e3 for s in st], g, TP)
                progress *= rim_factor
                meta.update(info)
            elif task == "finger-tap":
                checks["count"] = len(st) == g["count"]
                hand = {x["name"] for x in self.env.world.strikers if x.get("fingertip")}   # pads / Dex3 tips only
                # (the palm, the hand housing and the other finger links are strikers too: a slap is heard, not a tap)
                checks["fingertip"] = bool(st) and all(s["tools"] <= hand for s in st)
                if len(st) >= 2:
                    T = 60/g["bpm"]
                    checks["tempo"] = abs(ioi.mean()/T - 1) <= 0.02 and ioi.std()/ioi.mean() <= 0.04
                ref = self.reference_level("drumhead", "mp", "fingertip")
                checks["level"] = bool(lv) and all(v is not None and abs(v - ref) <= 4.0 for v in lv)
                meta.update(reference_dBA=round(ref, 2))
                # progress: taps heard against taps asked (min/max, so extra taps cost like missing ones) times the mean
                # of fingertip share, tempo, steadiness, level and clean-contact share; rim hits divide it
                n, want_n = len(st), g["count"]
                count_credit = min(n, want_n)/max(n, want_n) if n else 0.0
                tip = float(np.mean([s["tools"] <= hand for s in st])) if st else 0.0
                tempo = ramp(abs(ioi.mean()/(60/g["bpm"]) - 1), 0.02, TP["tempo_zero_rel"]) if len(st) >= 2 else 0.0
                steady = ramp(ioi.std()/ioi.mean(), 0.04, TP["steadiness_zero_cv"]) if len(st) >= 3 else tempo
                level = float(np.mean([0.0 if v is None else ramp(abs(v - ref), 4.0, TP["level_zero_db"]) for v in lv])) if lv else 0.0
                clean_share = float(np.mean([s["contact_ms"] <= P.CONTACT_MAX_S["drum"]*1e3 for s in st])) if st else 0.0
                progress = count_credit*float(np.mean([tip, tempo, steady, level, clean_share]))*rim_factor
            else:
                stim = env.stimulus or {}
                so = np.array(stim.get("onsets_s", []))
                beat = 60/g["bpm"]
                end = so[-1] + g["rhythm"][-1]*beat if len(so) else 0.
                resp = [s for s in st if s["onset_s"] > end - 0.05]
                ro = np.array([s["onset_s"] for s in resp])
                checks["count"] = len(ro) == len(so)
                checks["started"] = len(ro) > 0 and ro[0] - end <= 2*beat
                if len(ro) == len(so) and len(so) >= 2:
                    ratio = np.diff(ro)/np.diff(so)
                    checks["rhythm"] = bool(np.all(np.abs(ratio/ratio.mean() - 1) <= g["ratio_tol"]))
                    meta.update(ioi_ratio=[round(float(r), 3) for r in ratio])
                # progress: strokes against the stimulus (min/max) times the mean of the start (full within 2 beats,
                # none at start_zero_beats), the rhythm (each of the first min(n) intervals' ratio to the stimulus, graded
                # against ratio_tol, over the stimulus' interval count) and the clean-contact share; rim hits divide it
                k = min(len(ro), len(so))
                count_credit = k/max(len(ro), len(so)) if k else 0.0
                start = ramp(max(0.0, (ro[0] - end)/beat), 2.0, TP["start_zero_beats"]) if len(ro) else 0.0
                rhythm = 0.0
                if k >= 2 and len(so) >= 2:
                    r = np.diff(ro[:k])/np.diff(so[:k])
                    rhythm = float(np.sum([ramp(abs(x/r.mean() - 1), g["ratio_tol"], TP["ratio_zero_factor"]*g["ratio_tol"]) for x in r]))/(len(so) - 1)
                clean_share = float(np.mean([s["contact_ms"] <= P.CONTACT_MAX_S["drum"]*1e3 for s in resp])) if resp else 0.0
                progress = count_credit*float(np.mean([start, rhythm, clean_share]))*rim_factor
                meta.update(stimulus_onsets_s=[round(float(v), 3) for v in so], response_onsets_s=[round(float(v), 3) for v in ro])
        elif task == "melody":
            mc, mm, progress = self._melody(g)
            checks.update(mc)
            meta.update(mm)
        elif task == "bordun":
            st = [s for s in self.strikes(min_speed=0.05) if s["surface"].startswith("bar_")]
            notes = [s["note"] for s in st]
            meta.update(strikes=len(st), notes=notes, onsets_s=[round(s["onset_s"], 4) for s in st])
            tone = [s.get("mode1_share", 0) >= 0.5 for s in st]
            checks["clean"] = bool(st) and all(s["contact_ms"] <= P.CONTACT_MAX_S["bar"]*1e3 for s in st) and \
                self._restrike(st) == 0
            checks["tone"] = bool(st) and all(tone)
            pairs = []
            i = 0
            while i + 1 < len(st):
                a, b = st[i], st[i + 1]
                if {a["note"], b["note"]} == set(g["notes"]) and abs(a["onset_s"] - b["onset_s"]) <= 0.05:
                    pairs.append((a, b)); i += 2
                else:
                    i += 1
            checks["count"] = len(pairs) == g["count"] and len(st) == 2*g["count"]
            if pairs:
                asyn = [abs(a["onset_s"] - b["onset_s"]) for a, b in pairs]
                bal = [abs(10*math.log10(max(a.get("bar_energy_J", 1e-12), 1e-12)/max(b.get("bar_energy_J", 1e-12), 1e-12)))
                       for a, b in pairs]
                on = [min(a["onset_s"], b["onset_s"]) for a, b in pairs]
                checks["together"] = max(asyn) <= g["max_async_s"]
                checks["balanced"] = max(bal) <= 3.0
                if len(on) >= 2:
                    d = np.diff(on)
                    checks["tempo"] = abs(d.mean()/(60/g["bpm"]) - 1) <= 0.02 and d.std()/d.mean() <= 0.03
                meta.update(asynchrony_ms=[round(1e3*v, 2) for v in asyn], balance_db=[round(v, 2) for v in bal])
        elif task == "pentatonic-setup":
            sc = w.scene
            tray = sc.items["bar_tray"]
            pt, Rt = w.body_pose(tray["body"])
            hx, hy, hz = tray["half"]
            res = {}
            for n in g["remove"]:
                p, R = w.body_pose(f"bar_{n}")
                loc = Rt.T @ (p - pt)
                res[n] = abs(loc[0]) <= hx and abs(loc[1]) <= hy + 0.05 and loc[2] <= 2*hz + 0.02 and R[2, 2] >= math.cos(math.radians(15))
            checks["removed_to_tray"] = all(res.values())
            seated = self._seated_bars()
            checks["others_seated"] = all(v for k, v in seated.items() if k not in g["remove"])
            # progress: each bar to remove - 1 lying flat in the tray, graded down to 0 at tilt_zero_deg when in the
            # tray but tilted, off_pins_credit when off its pins but elsewhere - times the share of the other bars still
            # seated
            credit = {}
            for n in g["remove"]:
                p, R = w.body_pose(f"bar_{n}")
                loc = Rt.T @ (p - pt)
                inside = abs(loc[0]) <= hx and abs(loc[1]) <= hy + 0.05 and loc[2] <= 2*hz + 0.02
                tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1.0, 1.0))))
                credit[n] = ramp(tilt, 15.0, TP["tilt_zero_deg"]) if inside else (TP["off_pins_credit"] if not seated.get(n, True) else 0.0)
            others = [v for k, v in seated.items() if k not in g["remove"]]
            progress = float(np.mean(list(credit.values())))*(float(np.mean(others)) if others else 1.0)
            meta.update(in_tray=res, seated={k: v for k, v in seated.items()}, bar_credit=credit)
        checks["nothing_dropped"] = not self.floor_tools
        checks["robot_still"] = env.robot_still_s() >= P.SETTLE_STILL_S
        if w.fault:
            checks = {k: False for k in checks}
        meta.update(rim_events=len(self.rim_events), floor_tools=sorted(self.floor_tools))
        self.meta = meta
        success = bool(checks) and all(checks.values())
        score = float(np.mean([bool(v) for v in checks.values()])) if checks else 0.
        if task == "melody" and not w.fault:
            score = 1.0 if success else P.MELODY_EVAL["fail_scale"]*progress
        elif progress is not None and not w.fault:      # the other seven tasks of the eight-task set
            score = 1.0 if success else TP["fail_scale"]*progress
        if progress is not None:
            meta.update(progress=progress, score_rule="1 if every check passes, else 0.9 x progress (params.TASK_PROGRESS)")
        return dict(success=success, score=score, checks={k: bool(v) for k, v in checks.items()},
                    meta={k: (round(v, 4) if isinstance(v, float) else v) for k, v in meta.items()})

    def _melody(self, g):
        """Melody checks, diagnostics and progress (params.MELODY_EVAL). The notes are the
        xylophone strikes approaching at >= 0.05 m/s, every one in the episode. Each is judged on the stroke's own
        excitation, split from a bar still ringing by the linear modal model (world._contact_split): a mallet head
        struck it; one short contact, no bounce (clean); its fundamental at least tone_min_db against the fundamental
        an impulse of the same size gives at the bar's centre (tone); no other bar given neighbour_max of the stroke's
        own energy (no_neighbours); and when the note's ring window closes - half its written length, at most an eighth
        note (ring_beats) at the stated tempo, sooner if the next note is struck - its bar stands less than
        ring_max_damping_db below the free continuation of its state when the stroke ended (rings, sample_ring: later
        damping, and staccato within half a note's length, are articulation). The tune: exactly its notes in order
        (notes), onsets within onset_s of the least-squares beat grid of its written rhythm, at the stated tempo within
        tempo_rel (timing); no stroke on another instrument (only_xylophone)."""
        E = P.MELODY_EVAL
        fs = P.AUDIO_FS
        want = [n for n, _ in g["notes"]]
        lengths = sorted({b for _, b in g["notes"]})
        T0 = 60.0/g["bpm"]
        allst = self.strikes(min_speed=P.STRIKE_MIN_SPEED_M_S)
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
            damp = {L: self.ring_damping(s, min(nxt, s["onset_s"] + min(L/2, E["ring_beats"])*T0)) for L in lengths}
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
        checks["clean"] = bool(st) and all(p["clean"] for p in per) and self._restrike(st) == 0
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

    def _in_rack(self, tool):
        w = self.env.world
        t = next(t for t in w.scene.layout["tools"] if t["name"] == tool)
        p, R = w.body_pose(tool)
        return float(np.linalg.norm(p - np.asarray(t["butt"]))) <= 0.03 and abs(float(R[:, 0] @ np.asarray(t["dir"]))) >= 0.98 \
            and self._held_by(tool) is None

    def _on_trap_table(self, tool):
        """take-sticks' return (params.STICK_RETURN): the tool rests on either trap table (rests_on_support), no hand
        holds it, and its midpoint has stayed within still_m over the last still_s (monitor's stick_track)."""
        w = self.env.world
        S = P.STICK_RETURN
        p, R = w.body_pose(tool)
        mid = p + R[:, 0]*w.scene.items[tool]["length"]/2
        return rests_on_support(mid, R[:, 0], w.scene.layout["tool_supports"]) and self._held_by(tool) is None and \
            still_over(self.stick_track.get(tool, []), w.sim_time, S["still_s"], S["still_m"])

    def _seated_bars(self):
        """A bar is seated when it sits where it started relative to the xylophone frame (within 3 mm): on its pins,
        wherever the instrument itself has been nudged to."""
        from scipy.spatial.transform import Rotation
        w = self.env.world
        sc = w.scene
        q0 = self.env._initial["rigid"]["body_q"]
        fb = sc.bodies["xylophone_frame"]
        pf, Rf = w.body_pose(fb)
        pf0, Rf0 = q0[fb, :3].astype(float), Rotation.from_quat(q0[fb, 3:].astype(float)).as_matrix()
        out = {}
        for bb in sc.xylo_bars:
            p, _ = w.body_pose(bb["body"])
            out[bb["note"]] = float(np.linalg.norm(Rf.T @ (p - pf) - Rf0.T @ (q0[bb["body"], :3] - pf0))) <= 0.003
        return out
