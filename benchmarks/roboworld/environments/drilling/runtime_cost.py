"""Cost model behind the shortest policy command, min_command_s (the report's "Runtime (E15)" and "Policy cost
model"). Kept out of the physics sources (and out of params.py) so recalibrating it does not change the physics;
each constant's value, unit, provenance label and basis are in LEDGER at the end of this file, in the parameter
ledger's format, and are explained below.

The wall time of one policy command - the POST /v1/action round trip with the policy key, the action-history records,
the per-tick physics trace, the state and observation JSON, the camera frames and the client's image downloads - is
modelled at its worst measured by

    wall(T) = OVERHEAD_S + IK_S + CONTENTION * sum_i frame_s(w_i, h_i, spp) + PHYSICS_RATIO_MAX * T
    frame_s(w, h, spp) = FRAME_S + w*h*(PIXEL_S + PIXEL_SAMPLE_S*spp)

for a command of T simulated seconds whose observation holds cameras of w_i x h_i pixels rendered at spp samples per
pixel (each camera's own size: 960 x 540 station and overview, 640 x 480 wrists, or renderer_options width/height).
min_command_s is the shortest whole-millisecond T with wall(T) <= DESIGN_RATIO * T, at least one 1 ms tick; zero
cameras still pay OVERHEAD_S and IK_S, so they get a floor too. The floor is computed at the design ratio (150x), not
at the limit (200x): the quarter between them is what is left for a stall longer than the longest one measured. The
floor is one number for every command kind, so every kind is given the inverse-kinematics term although only
Cartesian and keyboard-move commands solve anything.

What the fixed term is (measured with a timer around each stage inside the session server, 5,000 zero-camera commands
at 5 ms): 84 % of the server's time for such a command is the shared action history writing its record to disk three
times per command (gym_anything_robotics/action_history.py: before the motion, with the trace handle, after the
motion; each a full-sync SQLite commit), 8 % is the physics with its per-tick trace, 5 % the inverse-kinematics solve
of the Cartesian and keyboard commands (once per command; 31 ms median, 0.14 s worst for the 4 mm targets of that run:
see IK_S below for far targets), 1 % the two state snapshots. One commit takes 33 ms at the median and 0.34 s at the
99th percentile, 1.21 s at worst, and the slow ones come in bursts of two to six consecutive commands while other jobs
write to the same disk and the container sits at its CPU quota. In the 36 commands over 1.0 s the commits were 96.5 %
of the server's time. The history file is the platform's, not drilling's, so the stall is not removed here; the floor
is sized to it.

Measured basis: a shared machine (RTX 5090 and a 16-CPU quota, both shared with other jobs; 1-minute load 8-26), real
`gar run drilling` servers driven over HTTP with the policy key, recording on, images downloaded, 10 commands per
configuration at 10 ms (1 ms with no camera) on wall/anchor-holes and bench/press-drill (the MEASURED table below),
and in-process runs while another job held the GPU at 100 % utilisation and 18 GB.
  FRAME_S, PIXEL_S, PIXEL_SAMPLE_S (measured) - least-squares fit to the median command times minus the zero-camera
    median (68 ms) over eight configurations: 1 to 4 native cameras, one camera at 1920 x 1080, four at 1280 x 720,
    one at 32 spp, bench and wall; residuals within 15 %. At 16 spp: 38 ms per frame plus 0.40 us per pixel (0.25 s
    for the 960 x 540 station camera, 0.82 s for all four native cameras). Doubling the samples from 16 to 32 added
    0.14 s to one 960 x 540 frame (median 0.339 -> 0.483 s per command): PIXEL_SAMPLE_S.
  CONTENTION (measured) - the slowest observation measured, relative to frame_s: 3.85 for one camera under another
    job's GPU load (0.962 s at 10 ms against a 0.247 s model frame), 3.34 for four cameras there (2.785 s at 50 ms);
    without that load at most 2.6 (one camera at 32 spp, 1.454 s) and 2.47 (four cameras, 2.06 s). 3.9 bounds them.
  OVERHEAD_S (measured) - the slowest non-camera, non-physics part of a command (its wall time less the physics
    of its ticks), in every kept measurement with SEVERAL SESSIONS RUNNING SIDE BY SIDE, which is how a benchmark uses
    the machine; the rule: the largest, rounded up to 0.1 s: 2.7. The kept measurements (rows of MEASURED):
      2.57 s - seven camera-less sessions at once (six with five command kinds, a seventh with Cartesian commands;
        load 7.4-15.3). The seventh session's slowest command took 2.61 s at 15 ms (a 4 mm cartesian_delta; less 15
        ticks of physics at 2.3x: 2.57 s); of the six sessions' 3,600 commands 66 took over 1.0 s and 11 over 2.0 s,
        the slowest 2.36 s, and the slowest of all six fell in the same three moments, each with 0.74-1.02 s of the
        container's disk stall.
      0.86 s and 1.52 s - six camera-less sessions (3,537 commands of six kinds at 17 ms, load 4.4-27.8, slowest
        1.01 s: 0.86 s; one command over 1.0 s, none over 2.0 s) and four sessions with two cameras each (466 commands
        at 31 ms, load 15-35, slowest 3.62 s; the largest remainder after the model's camera term and the physics was
        1.52 s, of a 3.57 s command).
      2.70 s and 1.35 s - the same cases at floors of 2.6 s: six camera-less sessions (3,537 commands at 20 ms, load
        8.0-14.3): slowest 2.79 s (a Cartesian command to a far pose; less 20 ticks of physics at that session's
        4.75x: 2.70 s), 60 commands over 1.0 s and 12 over 2.0 s, in five moments shared by the sessions; four
        sessions with two cameras each (466 commands at 35 ms, load 15-21): slowest 3.52 s, largest remainder
        1.35 s. None of these 4,003 commands ran over 200x or over the 150x design ratio.
    The same six-session case found 0.86 s in one run and 2.70 s in another 45 minutes later, with the machine's
    backup job running through both: the stall is the shared disk's, it differs from hour to hour, and a sample of
    four thousand commands does not find its largest. So 2.7 is the largest stall seen, not a bound, and a later run
    may well see a larger one. E15 repeats the several-sessions cases and says in fixed_term whether the value
    covered that run.
    One session at a time, over every zero-camera command timed at the floor through gar run + HTTP with the policy
    key at the rate the same sessions ran their 0.5 s commands (2.1-2.6 wall s per simulated s with the trace on;
    9 ms for 5 ticks by the timer inside the server; 7,161 commands): 1.99 s (a 2.02 s Cartesian command of 13 ms
    among 1,200 of the four kinds, 1-minute load 16-24), 1.75 s (400 commands of the four kinds at 5 ms, load
    14.5-17.1), 1.68 s (5,000 commands of the four kinds at 5 ms over 14 minutes, load 12.8-44.6; wall time median
    0.132 s, 99th percentile 0.844 s, 99.9th 1.471 s) and 1.38 s (561 hold and move commands at 5 ms, load 7-10).
    The sample's maximum grows with the sample, with the machine's load and with the number of sessions (1.38, 1.68,
    1.75, 1.99, 2.57, 2.70 s), and in the 5,000-command stream the number of commands over a given time fell about
    threefold for every 0.2 s beyond 1.2 s (24 over 1.2 s, 10 over 1.4 s, 3 over 1.6 s, none over 1.8 s) - a slope
    that belonged to that stream: later samples produced 1.99, 2.57 and 2.70 s. No extrapolation is offered; E15
    counts what actually happens (over_limit, over_design, over_model, for one session at a time and for several at
    once). At the 21 ms floor that 2.7 and IK_S give, a zero-camera command reaches 200x when its non-physics part
    reaches 3.91 s.
  IK_S (measured) - the slowest ACCEPTED inverse-kinematics solve of a Cartesian command (backend.ik.solve, in
    process) over 900 targets up to 0.6 m and 1.5 rad from the hands of three scenes (E15 ik_block), rounded up to
    0.1 s: 0.071 s alone (load 2.5-5.5 and again at load 6.2-12.7), 0.068 s and 0.071 s beside six camera-less
    sessions (load 4.9-6.4 and 8.2-8.5) and 0.071 s under other jobs' load (7.4-15.4); a refused target took at most
    0.067 s in four of the five passes and 0.194 s once (a stall of the machine during that solve: such stalls are
    what OVERHEAD_S holds), and advances no time. With the shared solver's 2-point numerical Jacobian (eight
    forward-kinematics calls per evaluation; drilling/kinematics.py uses an analytic one) the same targets take
    0.115 s at the median, 0.55 s at worst and 1.79 s under a load of 17; both accept the same 782 of the 900. The
    solve is capped at 300 evaluations, a count and not a time, so IK_S is also the largest seen, not a bound: it
    grows with the machine's load. It is added to OVERHEAD_S rather than taken as part of it because the slowest
    solve and the slowest stall need not have met in one command of any sample (the 2.61 s command held a 4 mm
    solve).
  PHYSICS_RATIO_MAX (measured) - wall/sim of the busiest physics with the trace recording, the rotary hammer drilling
    the wall under a phantom holder for 3 s, in process: 13.96x (the container at its CPU quota, load 18), 11.4x
    (load 5), 10.74x and 8.62x. 14 is the largest, rounded up. The start scenes' physics runs at 2.6-3.7x through
    HTTP (the server's span of 0.5 s commands at wall/anchor-holes and bench/press-drill) and at 3.5-5.3x in process
    for the costliest three. At the floor the physics term is under 10 % of the budget, so the floors hardly move
    with it: 12 or 14 changes one of the five native floors by 1 ms.
  DESIGN_RATIO (authored): 25 % under the 200x limit, the margin for spikes beyond the worst measured.
  COLD_START_S (measured) - a render worker starting and drawing its first frame: 10.5-11.6 s (one camera; after a
    resume, after an idle release, and with release after every operation), and 13.4-17.3 s inside resume requests
    while other jobs used the GPU. 18 is the largest, rounded up to 1 s. No policy command can contain it: a live
    session keeps its worker, and a paused session that gave it back restarts it inside the owner's resume request
    (the backend's module docstring). The value is what the capabilities tell the owner a resume may take; it is not
    in the floor.
  WORKER_WATCH_S (authored) - how often a live session with cameras looks at its render worker's process; when it
    has exited (or a frame error has cost the session its worker) a new one is started at once and draws its first
    frame as soon as no command holds the world. Measured on a real session: worker killed, restart begun within
    2 s, and 30 s later the first policy command took 0.84 s (31x) where it took 10.4 s (359x) when the restart
    waited for that command.
  IDLE_RELEASE_S (authored) - how long a paused session keeps its render worker before it gives the GPU back
    (renderer_options idle_release_s overrides it; 0 at the pause, null never): long enough that the usual pause
    between two `gar benchmark` runs costs the owner no restart, short enough that a forgotten paused session frees
    its 4.2-4.6 GB of GPU memory (measured with the two default cameras: 4,191 and 4,630 MiB) within ten minutes.

Validation: E15 times every configuration at these floors with six command kinds (Cartesian commands to far poses
among them), one session at a time and several sessions at once, and reports per case the commands over the limit,
over the design ratio and over this model (over_model, model_margin_s, needed_overhead_s); its claims block says what
each "none over" covers and its fixed_term block whether OVERHEAD_S and IK_S covered that run.
"""
import math

LIMIT = 200.0
DESIGN_RATIO = 150.0
OVERHEAD_S = 2.7
IK_S = 0.1
FRAME_S = 0.038
PIXEL_S = 0.079e-6
PIXEL_SAMPLE_S = 0.324e-6/16
CONTENTION = 3.9
PHYSICS_RATIO_MAX = 14.0
COLD_START_S = 18.0
IDLE_RELEASE_S = 600.0
WORKER_WATCH_S = 1.0
TICK_S = 0.001

# The calibration table: the slowest command of each configuration (wall time including image downloads; physics_s is
# that command's server-side physics-and-trace span, or the median for the GPU-contention rows). In the one-session
# rows of 5 to 13 ms physics_s is the physics alone at its usual rate (9-10 ms for 5 ticks by the timer inside the
# server; taken as 20-30 ms for the 12 and 13 ms commands): the history span of a stalled command contains one of the
# three commits, which does not shrink or grow with the command's duration. OVERHEAD_S is the largest
# worst_total_s - physics_s of the zero-camera rows, rounded up to 0.1 s.
MEASURED = (
    dict(case="0 cameras, 15 ms, seven sessions at once, 411 Cartesian commands beside 3,600 of five kinds "
              "(load 7.4-15.3)", resolutions=[], duration_s=0.015,
         worst_total_s=2.6088, physics_s=0.0345),                      # no span for that row: 15 ticks at 2.3x
    dict(case="0 cameras, 15 ms, six sessions at once, 3,600 commands of five kinds "
              "(load 7.4-15.3)", resolutions=[], duration_s=0.015, worst_total_s=2.3562, physics_s=0.0345),
    dict(case="0 cameras, 20 ms, six sessions at once, 3,537 commands of six kinds (load "
              "8.0-14.3)", resolutions=[], duration_s=0.020, worst_total_s=2.7945, physics_s=0.0950),   # 20 ticks at 4.75x
    dict(case="2 cameras, 35 ms, four sessions at once, 466 commands of six kinds (load "
              "15-21)", resolutions=[(960, 540)]*2, duration_s=0.035, worst_total_s=3.3948, physics_s=0.1171),
    dict(case="0 cameras, 17 ms, six sessions at once, 3,537 commands of six kinds (load 4.4-27.8)",
         resolutions=[], duration_s=0.017, worst_total_s=1.0122, physics_s=0.1543),   # 17 ticks at that session's rate
    dict(case="2 cameras, 31 ms, four sessions at once, 466 commands of six kinds (load 15-35)",
         resolutions=[(960, 540)]*2, duration_s=0.031, worst_total_s=3.5666, physics_s=0.1167),   # largest remainder
    dict(case="2 cameras, 29 ms, four sessions at once, 480 commands of five kinds "
              "(load 7.9-14.9)", resolutions=[(960, 540)]*2, duration_s=0.029, worst_total_s=2.0574, physics_s=0.1011),
    dict(case="0 cameras, 13 ms, 1200 commands of four kinds (load 16-24)", resolutions=[],
         duration_s=0.013, worst_total_s=2.0205, physics_s=0.03),     # no timer in that run: 13 ticks at 2.3x
    dict(case="0 cameras, 5 ms, 400 commands of four kinds (load 14.5-17.1)", resolutions=[],
         duration_s=0.005, worst_total_s=1.759, physics_s=0.01),
    dict(case="0 cameras, 5 ms, 5000 commands of four kinds (load 12.8-44.6)", resolutions=[],
         duration_s=0.005, worst_total_s=1.6881, physics_s=0.01),
    dict(case="0 cameras, 5 ms, 561 hold and move commands", resolutions=[],
         duration_s=0.005, worst_total_s=1.3886, physics_s=0.01),
    dict(case="1 camera, 12 ms, 120 move commands", resolutions=[(960, 540)],
         duration_s=0.012, worst_total_s=1.5696, physics_s=0.02),
    dict(case="0 cameras, 1 ms", resolutions=[], duration_s=0.001, worst_total_s=0.2226, physics_s=0.0228),
    dict(case="0 cameras, 1 ms (second run)", resolutions=[], duration_s=0.001, worst_total_s=0.2742,
         physics_s=0.02),
    dict(case="station 960x540", resolutions=[(960, 540)], duration_s=0.01, worst_total_s=0.6452,
         physics_s=0.0452),
    dict(case="station 960x540, bench", resolutions=[(960, 540)], duration_s=0.01, worst_total_s=0.3585,
         physics_s=0.0637),
    dict(case="station 960x540 at 32 spp", resolutions=[(960, 540)], samples=32, duration_s=0.01,
         worst_total_s=1.4537, physics_s=0.386),
    dict(case="station + overview", resolutions=[(960, 540)]*2, duration_s=0.01, worst_total_s=0.6134,
         physics_s=0.0596),
    dict(case="station + overview + wrist", resolutions=[(960, 540)]*2 + [(640, 480)], duration_s=0.01,
         worst_total_s=0.9716, physics_s=0.1347),
    dict(case="four native cameras", resolutions=[(960, 540)]*2 + [(640, 480)]*2, duration_s=0.01,
         worst_total_s=1.7708, physics_s=0.3486),
    dict(case="four native cameras (second run)", resolutions=[(960, 540)]*2 + [(640, 480)]*2,
         duration_s=0.01, worst_total_s=2.0595, physics_s=0.04),
    dict(case="one camera 1920x1080", resolutions=[(1920, 1080)], duration_s=0.01, worst_total_s=1.30,
         physics_s=0.0367),
    dict(case="four cameras 1280x720", resolutions=[(1280, 720)]*4, duration_s=0.01, worst_total_s=1.87,
         physics_s=0.2692),
    dict(case="station, GPU contention", resolutions=[(960, 540)], duration_s=0.01, worst_total_s=0.962,
         physics_s=0.0124),
    dict(case="four native cameras, GPU contention", resolutions=[(960, 540)]*2 + [(640, 480)]*2,
         duration_s=0.05, worst_total_s=2.785, physics_s=0.0547),
)


def frame_s(width, height, samples=16):
    """Median cost of one camera frame on a warm worker (s): render, readback, PNG, JSON, download."""
    if not (width > 0 and height > 0 and samples >= 0):
        raise ValueError("camera width and height must be positive and samples non-negative")
    return FRAME_S + width*height*(PIXEL_S + PIXEL_SAMPLE_S*samples)


def camera_term_s(resolutions, samples=16):
    """The model's camera term (s): every camera's frame at the worst contention measured."""
    return CONTENTION*sum(frame_s(w, h, samples) for w, h in resolutions)


def worst_command_s(resolutions, samples=16, duration_s=0.0):
    """Modelled worst wall time (s) of one policy command of duration_s returning cameras of these (width, height).
    Every command kind is given the inverse-kinematics term: the floor is one number for all kinds."""
    return OVERHEAD_S + IK_S + camera_term_s(resolutions, samples) + PHYSICS_RATIO_MAX*duration_s


def min_command_s(resolutions, samples=16):
    """Shortest policy command (s, whole ticks) with worst_command_s(T) <= DESIGN_RATIO * T."""
    fixed = worst_command_s(resolutions, samples)
    ticks = max(1, math.ceil(round(fixed/(DESIGN_RATIO - PHYSICS_RATIO_MAX)/TICK_S, 9)))
    return round(ticks*TICK_S, 6)


NATIVE = {0: [], 1: [(960, 540)], 2: [(960, 540)]*2, 3: [(960, 540)]*2 + [(640, 480)],
          4: [(960, 540)]*2 + [(640, 480)]*2}
FLOORS_MS = {n: int(round(min_command_s(res)*1e3)) for n, res in NATIVE.items()}


def describe():
    """The model's constants and the floors it gives."""
    return dict(model="wall(T) = OVERHEAD_S + IK_S + CONTENTION * sum frame_s(w, h, spp) + PHYSICS_RATIO_MAX * T; "
                      "frame_s = FRAME_S + w*h*(PIXEL_S + PIXEL_SAMPLE_S*spp); floor: wall(T) <= DESIGN_RATIO * T",
                limit=LIMIT, design_ratio=DESIGN_RATIO, overhead_s=OVERHEAD_S, ik_s=IK_S, frame_s=FRAME_S, pixel_s=PIXEL_S,
                pixel_sample_s=PIXEL_SAMPLE_S, contention=CONTENTION, physics_ratio_max=PHYSICS_RATIO_MAX,
                cold_start_s=COLD_START_S, idle_release_s=IDLE_RELEASE_S, worker_watch_s=WORKER_WATCH_S,
                ledger=[dict(e) for e in LEDGER],
                floors_ms_native_cameras=FLOORS_MS,
                floors_ms_custom={"1 camera 1920x1080": int(round(min_command_s([(1920, 1080)])*1e3)),
                                  "4 cameras 1280x720": int(round(min_command_s([(1280, 720)]*4)*1e3)),
                                  "1 camera 960x540 at 32 spp": int(round(min_command_s([(960, 540)], 32)*1e3))})


def _p(name, value, unit, label, basis):
    return dict(name=name, value=value, unit=unit, label=label, basis=basis)


# The cost model's parameters in the parameter ledger's format (drilling/params.py: name, value, unit, provenance
# label, basis). They are not in params.py because these do not touch physics.
LEDGER = (
    _p("LIMIT", LIMIT, "wall s per simulated s", "authored", "the platform's 200x wall-clock limit for a policy command"),
    _p("DESIGN_RATIO", DESIGN_RATIO, "wall s per simulated s", "authored",
       "25 % under LIMIT: the room left for a stall longer than the longest measured"),
    _p("OVERHEAD_S", OVERHEAD_S, "s", "measured",
       "largest non-camera, non-physics part of a policy command at the floor with several sessions side by side, "
       "rounded up to 0.1 s: 2.70 s with six camera-less sessions; 2.57 s with seven; 0.86 s in the same six-session "
       "case in another run; one session at a time 1.99 s. "
       "84 % of a camera-less command is the shared action history's three full-sync commits (33 ms median, 1.21 s "
       "worst each); the largest stall seen, not a bound"),
    _p("IK_S", IK_S, "s", "measured",
       "slowest accepted inverse-kinematics solve of a Cartesian command over 900 targets up to 0.6 m and 1.5 rad "
       "away (E15 ik_block): 0.071 s alone, 0.068 s beside six sessions, 0.071 s at load 7-15; rounded up "
       "to 0.1 s; capped at 300 evaluations, so it grows with the machine's load; the largest seen, not a bound"),
    _p("FRAME_S", FRAME_S, "s", "measured", "per-frame constant of the least-squares fit to median command times"),
    _p("PIXEL_S", PIXEL_S, "s per pixel", "measured", "per-pixel term of the same fit"),
    _p("PIXEL_SAMPLE_S", PIXEL_SAMPLE_S, "s per pixel per sample", "measured",
       "16 -> 32 samples added 0.14 s to one 960 x 540 frame"),
    _p("CONTENTION", CONTENTION, "-", "measured",
       "slowest observation over the model frame: 3.85 under another job's GPU load"),
    _p("PHYSICS_RATIO_MAX", PHYSICS_RATIO_MAX, "wall s per simulated s", "measured",
       "the rotary hammer cutting with the trace on, in process: 13.96x at the CPU quota, 11.4x and 10.74x "
       "elsewhere; the largest, rounded up"),
    _p("COLD_START_S", COLD_START_S, "s", "measured",
       "a render worker's start and first frame: 10.5-11.6 s over five measurements, 13.4-17.3 s with other jobs on "
       "the GPU; the largest, rounded up to 1 s; the owner's resume may take this long; not part of the floor"),
    _p("IDLE_RELEASE_S", IDLE_RELEASE_S, "s", "authored",
       "how long a paused session keeps its render worker (4.2-4.6 GB of GPU memory, measured) before giving it back; "
       "renderer_options idle_release_s overrides it"),
    _p("WORKER_WATCH_S", WORKER_WATCH_S, "s", "authored",
       "how often a live session looks whether its render worker's process is still there, to restart it at once "
       "after a crash instead of inside the next camera command; a poll of one "
       "process's exit status, so its cost is nil"),
    _p("TICK_S", TICK_S, "s", "repo", "the rigid step, drilling/config.py rigid_dt"),
)
