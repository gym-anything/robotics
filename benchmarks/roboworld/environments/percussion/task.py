"""Task instructions and the environment-owned evaluator.

A benchmark task is judged by its folder (tasks/<task>/verifier.py, loaded by layout.task_verifier): evaluate
calls its verify(e) with e the evaluator, which hands its own checks to verdict and reads the shared
measurements below (strikes, levels, rim strokes, the state the monitor keeps); the other tasks are judged
in verdict itself.

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
from .layout import task_verifier

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


# ------------------------------------------------- eight-task set: the progress ramp of the task folders
def _ramp(x, full, zero):
    """1 at or below `full`, 0 at or beyond `zero`, linear between (the five-task set's grading)."""
    return float(np.clip((zero - x)/(zero - full), 0.0, 1.0))


# ====================================================================================================== evaluator
class Evaluator:
    def __init__(self, env):
        self.env = env
        self.verifier = task_verifier(env.config.task)   # the benchmark task's folder, or None
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

    def tick(self):
        """Every tick (the environment's act): the task folder's tick(e), if it has one."""
        tick = getattr(self.verifier, "tick", None)
        if tick is not None:
            tick(self)

    def monitor(self):
        """Every 10 ms (the environment's act): tools on the floor, then the task folder's monitor(e), if it has
        one."""
        env = self.env
        w = env.world
        sc = w.scene
        for t in sc.layout["tools"]:
            p, _ = w.body_pose(t["name"])
            if p[2] < 0.3:
                self.floor_tools.add(t["name"])
        monitor = getattr(self.verifier, "monitor", None)
        if monitor is not None:
            monitor(self)

    # -------------------------------------------------------------- evaluation
    def evaluate(self):
        """The episode's evaluation (success, score, checks, meta): the benchmark task's verify(e), else verdict."""
        if self.verifier is not None:
            return self.verifier.verify(self)
        return self.verdict()

    def verdict(self, judge=None, *, strokes=None, fail_scale=None):
        """The evaluation with the checks every task shares (nothing dropped, the robot still; all false on a
        fault). A task folder's verify passes judge(e, g, checks, meta, **parts), which fills in the task's checks
        and meta and returns its progress; strokes names the strikes it is judged on, counted here first: "bar" (st
        the strikes, on those on the goal's bar) or "drum" (st the drumhead's strikes, lv their levels, ioi their
        intervals; the clean and no_rim checks); parts also has ramp, TP (params.TASK_PROGRESS) and rim_factor. The
        score is 1 if every check passes, else fail_scale (default TP's) x progress; without a progress the share
        of checks passed. Without judge: the tasks outside the benchmark."""
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
        if judge is not None and strokes is None:
            progress = judge(self, g, checks, meta, ramp=ramp, TP=TP, rim_factor=rim_factor)
        elif strokes == "bar" or task in ("single-tone", "roll"):
            bar = "rbar_" + g["bar"]
            st = self.strikes(min_speed=0.05)
            on = [s for s in st if s["surface"] == bar]
            other = [s for s in st if s["surface"] != bar]
            meta.update(strikes_on_bar=len(on), strikes_elsewhere=len(other))
            if judge is not None:
                progress = judge(self, g, checks, meta, st=st, on=on, ramp=ramp, TP=TP, rim_factor=rim_factor)
            elif task == "single-tone":
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
        elif strokes == "drum" or task == "dynamics":
            st = [s for s in self.strikes(["drumhead"], min_speed=0.05)]
            lv = [self.strike_level(s) for s in st]
            meta.update(strikes=len(st), onsets_s=[round(s["onset_s"], 4) for s in st],
                        levels_dBA=[None if v is None else round(v, 2) for v in lv])
            clean = all(s["contact_ms"] <= P.CONTACT_MAX_S["drum"]*1e3 for s in st) and self._restrike(st) == 0
            checks["clean"] = bool(st) and clean
            checks["no_rim"] = not self.rim_events
            ioi = np.diff([s["onset_s"] for s in st])
            if judge is not None:
                progress = judge(self, g, checks, meta, st=st, lv=lv, ioi=ioi, ramp=ramp, TP=TP,
                                 rim_factor=rim_factor)
            elif task == "dynamics":
                refs = [self.reference_level("drumhead", d, "stick") for d in g["sequence"]]
                checks["three_strokes"] = len(st) == 3
                if len(st) == 3:
                    checks["levels"] = all(v is not None and abs(v - r) <= P.LEVEL_BAND_DB for v, r in zip(lv, refs))
                    checks["ordered"] = all(lv[i + 1] - lv[i] >= g["min_step_db"] for i in range(2))
                meta.update(reference_dBA=[round(r, 2) for r in refs])
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
        checks["nothing_dropped"] = not self.floor_tools
        checks["robot_still"] = env.robot_still_s() >= P.SETTLE_STILL_S
        if w.fault:
            checks = {k: False for k in checks}
        meta.update(rim_events=len(self.rim_events), floor_tools=sorted(self.floor_tools))
        self.meta = meta
        success = bool(checks) and all(checks.values())
        score = float(np.mean([bool(v) for v in checks.values()])) if checks else 0.
        if fail_scale is not None and not w.fault:      # a task's own contract (melody: params.MELODY_EVAL)
            score = 1.0 if success else fail_scale*progress
        elif progress is not None and not w.fault:      # the other seven tasks of the eight-task set
            score = 1.0 if success else TP["fail_scale"]*progress
        if progress is not None:
            meta.update(progress=progress, score_rule="1 if every check passes, else 0.9 x progress (params.TASK_PROGRESS)")
        return dict(success=success, score=score, checks={k: bool(v) for k, v in checks.items()},
                    meta={k: (round(v, 4) if isinstance(v, float) else v) for k, v in meta.items()})
