"""Start of echo: the assessment station (two resonator bars on the table, the drum on its stand, the loudspeaker at the
back of the table); the loudspeaker plays a seeded rhythm on a woodblock from 1.0 s at 90 BPM, to be echoed on the drum.

setup(cfg, rng, lay) is called by layout.preset_layout (through layout.goal) when it lays out the start: cfg the
episode's Config, rng the layout's generator (seeded by task and seed; the instruments' placements are drawn from it
first), lay the layout so far; it returns the task's goal (layout["goal"]): the rhythm (note lengths in beats, one of
ECHO_RHYTHMS), its tempo, its start (start_s) and the rhythm tolerance. schedule(env, g) is called by the environment
when it builds the episode (PercussionEnv._schedule_task_events): with acoustics on it adds the stimulus
(echo_stimulus) to the room at the loudspeaker and sets env.stimulus (its onsets)."""
import math

import numpy as np

from percussion import params as P

ECHO_RHYTHMS = ([1, 1, 1, 1], [1, 1, 2], [.5, .5, 1, 2], [1, .5, .5, 2], [2, 1, 1], [.5, .5, .5, .5, 2])


def setup(cfg, rng, lay):
    k = int(rng.integers(0, len(ECHO_RHYTHMS)))
    return dict(rhythm=list(ECHO_RHYTHMS[k]), bpm=90, start_s=1.0, instrument="drum", ratio_tol=0.10)


def schedule(env, g):
    if env.world.acoustics is not None:
        sig, onsets = echo_stimulus(g)
        sp = env.world.scene.fixtures["speaker"]["position"]
        start = int(round(g["start_s"]*P.AUDIO_FS))
        env.world.acoustics.add_point_source(np.asarray(sp, float), sig, start)
        env.stimulus = dict(onsets_s=[g["start_s"] + t for t in onsets], start_s=g["start_s"])


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
