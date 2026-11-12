"""Start of strike-and-stop: the assessment station (the C4 and G4 resonator bars on the table, the drum on its stand,
the loudspeaker); the C4 bar is to be struck once and stopped by a touch when the cue light comes on, a seeded 0.20-0.35
s after the stroke.

setup(cfg, rng, lay) is called by layout.preset_layout (through layout.goal) when it lays out the start: cfg the
episode's Config, rng the layout's generator (seeded by task and seed; the instruments' placements are drawn from it
first), lay the layout so far; it returns the task's goal (layout["goal"]): the bar, the cue's delay after the stroke
(cue_delay_s) and the judgement's timing and threshold (judge_after_s, judge_window_s, min_damping_db). schedule(env,
g) is called by the environment when it builds the episode (PercussionEnv._schedule_task_events): it sets the cue
(env.cue), which the environment lights cue_delay_s after the first stroke on the bar."""


def setup(cfg, rng, lay):
    rng.integers(0, 2)                      # (kept: the seed stream of the other draws stays as before)
    # A short tone of a stated length, as a therapist shortens a stimulus: strike the C4 resonator bar, then stop it
    # by touching the bar when the cue light comes on, a seeded 0.20-0.35 s after the stroke (stated in the
    # instruction, so the stop is anticipated - a planned touch lands within ~50 ms of a known moment: sensorimotor
    # synchronisation - not a reaction to the light). The C4 bar only: its own box leaves it T60 ~0.6 s;
    # the G4 bar's (T60 ~0.3 s) is nearly silent by the cue. Judgement (verifier.py): one bar-box energy-exchange
    # period after the cue (judge_after_s, the time a touch landing at the cue needs to have drained the bar
    # whichever way the energy then lay), the bar's vibration energy over the next 0.1 s must be min_damping_db
    # below what its free ring would leave (the exact free continuation of its modal state at the cue): 10 dB, half
    # as loud (loudness doubles per 10 phon, ISO 532-1). A 15 dB authored margin over the extrapolated decay within
    # 0.3 s would need a touch within ~0-100 ms and flip with the exchange phase. The exchange period follows from
    # the authored box opening (sensitivity study: 0.30-5.3 s over plausible slots - a weakly coupled box does
    # not beat with the bar); capped at 0.35 s so that a box without
    # a beat is not judged seconds late (the shipped box: 0.327 s, the cap does not act)
    from percussion.modal import resonator_bar_beat_s
    return dict(bar="C4", cue_delay_s=float(rng.uniform(0.20, 0.35)),
                judge_after_s=round(min(resonator_bar_beat_s("C4"), 0.35), 3), judge_window_s=0.1, min_damping_db=10.0)


def schedule(env, g):
    env.cue = dict(on=False, at=None, delay=float(g["cue_delay_s"]), surface=f"rbar_{g['bar']}")
