"""Outcome of strike-and-stop: the C4 resonator bar struck exactly once before the cue and not again after it, still
ringing in the 50 ms before the cue (less than min_damping_db below its free ring), and one bar-box exchange period
after the cue (judge_after_s) at least min_damping_db (10 dB) below its free ring over judge_window_s (0.1 s); nothing
dropped and the robot still at the end.

verify(e) evaluates an episode; e is its evaluator, task.Evaluator, whose verdict counts the strikes (strokes="bar": the
strikes st and those on the goal's bar, on) and calls judge(e, g, checks, meta, ...) with the goal g (layout["goal"]):
judge fills in the task's checks and meta and returns its progress. verify returns the evaluation (success, score,
checks, meta): 1 if every check passes, else 0.9 x progress (params.TASK_PROGRESS fail_scale). The progress: one stroke
(1/n for n before the cue) x (half for the ring kept until the cue + half for the damping reached), divided by 1 +
strokes after the cue.

tick(e) is called every tick (task.Evaluator.tick, from the environment's act): it samples the struck bar's energy
and its free ring (stop_log). monitor(e) is called every 10 ms (task.Evaluator.monitor): it adds the bar's vibration
energy to energy_trace. energy_at(e, t) and leq(e, t0, t1) read them back for the evaluation's meta."""
import math

import numpy as np

from percussion import params as P


def verify(e):
    return e.verdict(judge, strokes="bar")


def judge(e, g, checks, meta, *, st, on, ramp, TP, rim_factor):
    env = e.env
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
    # ring from the same state (tick) - not on the level at the client's ear, which the room's
    # reverberant tail keeps up after any stop, and not against a decay rate extrapolated through the
    # bar-box beat (such a rule flips with the beat phase).
    # stopped: one exchange period after the cue the bar rings >= min_damping_db below its free ring over
    # judge_window_s; rang_until_cue: in the 50 ms before the cue it was not yet damped that much.
    fin = math.isfinite(cue)
    log = e.stop_log or list(getattr(e.env, "stop_log", None) or [])
    ref_t = e._stop_ref.get("t_cue", cue) if fin else None
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
    Lc = leq(e, cue - 0.03, cue) if fin else None
    La = leq(e, a0, a1) if fin else None
    meta.update(damping_db=None if damp is None else round(damp, 2), required_damping_db=g["min_damping_db"],
                damping_before_cue_db=None if early is None else round(early, 2),
                judge_window_s=None if not fin else [round(a0, 4), round(a1, 4)], judge_complete=complete,
                energy_at_cue_J=energy_at(e, cue) if fin else None, energy_after_stroke_J=E_stroke,
                free_ring_drop_to_window_db=round(10*math.log10(max(log[0][2], 1e-30)/max(win[-1][1], 1e-30)), 2)
                if win and log else None,
                level_at_cue_dBA=Lc, level_in_window_dBA=La, cue_s=cue if fin else None,
                judged_on="struck bar vibration energy vs its exact free ring")
    return progress


def tick(e):
    """Every tick (strike-and-stop only): the struck bar's modal kinetic energy (bar and box air), and - once the
    stroke has ended - the kinetic energy its free ring would have now: the exact free continuation exp(lam t) z of
    the bar's modal state just after the stroke and at the cue (privileged; the linear modal model makes the
    counterfactual exact, so the judgement does not extrapolate a measured decay through the bar-box beat)."""
    env = e.env
    g = env.layout["goal"]
    st = env.world.structs.get(f"resbar_{g['bar']}")
    if st is None:
        return
    _, v = st.physical()
    ke = 0.5*float(np.sum(v**2))
    e._ke = getattr(e, "_ke", None) or []
    e._ke.append(ke)
    cue = env.cue.get("at")
    if cue is None:
        return
    t = env.world.sim_time
    ref = e._stop_ref
    if "z_stroke" not in ref:
        ref.update(t_stroke=t, z_stroke=st.z.copy())
    if "z_cue" not in ref and t >= cue - 1e-9:
        ref.update(t_cue=t, z_cue=st.z.copy())
    if t > cue + g["judge_after_s"] + g["judge_window_s"] + 0.005:
        return

    def free(z0, t0):
        _, vf = st.physical(z0*np.exp(st.lam*(t - t0)))
        return 0.5*float(np.sum(vf**2))
    e.stop_log.append((round(t, 6), ke, free(ref["z_stroke"], ref["t_stroke"]),
                       free(ref["z_cue"], ref["t_cue"]) if "z_cue" in ref else None))


def monitor(e):
    """Every 10 ms (task.Evaluator.monitor): the struck bar's vibration energy, from the kinetic energies that tick
    samples."""
    w = e.env.world
    if getattr(e, "_ke", None):
        # vibration energy = 2 x the kinetic energy averaged over the last 10 ticks (2.6 periods of the C4 bar's
        # mode-1 pair): the total modal energy also counts the static strain energy of the bar bent under a
        # resting fingertip (~1e-6-1e-4 J at 0.5-5 N), which is no ringing
        e.energy_trace.append((round(w.sim_time, 6), 2.0*float(np.mean(e._ke))))
        e._ke = []


def energy_at(e, t):
    """The task structure's vibration energy (J) at time t, log-interpolated in the monitor's 10 ms trace."""
    tr = getattr(e.env, "energy_trace", None) or e.energy_trace
    if not tr or t < tr[0][0] or t > tr[-1][0]:
        return None
    ts = np.array([a for a, _ in tr])
    Es = np.log(np.maximum(np.array([b for _, b in tr]), 1e-30))
    return float(np.exp(np.interp(t, ts, Es)))


def leq(e, t0, t1):
    """A-weighted equivalent level (dB) of the client microphone over [t0, t1] - no 125 ms meter smoothing (an
    L_AF meter falls at most 35 dB/s, too slow to see a stopped bar). Fake environments: the L_AF trace's max."""
    A = e.env.world.acoustics
    if A is None:
        return e.level_max(t0, t1)
    from scipy.signal import sosfilt
    from percussion.acoustics import a_weighting_sos
    i0, i1 = int(t0*P.AUDIO_FS), int(t1*P.AUDIO_FS)
    pre = int(0.2*P.AUDIO_FS)
    x = A.signal("client_mic", max(0, i0 - pre), min(A.n, i1))
    if len(x) <= i0 - max(0, i0 - pre):
        return None
    y = sosfilt(a_weighting_sos(), x)[i0 - max(0, i0 - pre):]
    return float(10*np.log10(max(np.mean(y**2), 1e-30)/P.P_REF**2))
