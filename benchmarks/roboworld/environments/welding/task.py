"""Tasks of the robotic GMAW cell (preset 'steel-cell') and their evaluators.

The evaluators read physical state only: the thermal solver's cells and the pool's free surface (sections.py,
bonds.py), the rigid bodies' poses and contacts, and the source's recorded history. Thresholds: [read] from AWS
D1.1:2000 Table 6.1 / Fig 5.4 / 5.18.2 (the public copy at law.resource.org) or the cited trade source; [authored] where
the build chose the number.

Weld frame (all tasks): x along the joint, y out of the bracket's near (weld) face, z up from the base plate; the
joint's root line is the x axis, the bracket spans x in [0, 150] mm.
"""

import math
from types import SimpleNamespace

import numpy as np

from . import params as PR
from . import sections as SC
from .bonds import slice_bonds, FIT_MM

PRESETS = {"steel-cell": "Robotic GMAW cell: UR10e with a cobot MIG gun, FR3 with a Franka Hand, fixture table"}
TASKS = {
    "position-torch": "A0",
    "tack": "A1",
    "fillet-2f": "A2",
    # A3-A6 are reserved for steel-cell tasks that are not built: pipe on a positioner, vertical-up, sheet seam and
    # stick electrode (welding/REPORT.pdf §18)
    "set-up-gun": "A7",
    "fit-up-bracket": "A8",
    "half-fillet": "A9",
    "stitch-weld-ends": "A10",
    "tack-and-weld": "A11",
    "fillet-from-both-ends": "A12",
    "hold-and-present": "A13",
    "set-up-and-weld": "A14",
}
# the eight-task set, easiest first. tack, stitch-weld-ends and fillet-from-both-ends stay in the catalogue outside the
# set because none can be shown to succeed as built (no standard is relaxed to make one pass):
#   tack: no demonstration passes tack's D1.1 tack checks; tack-and-weld judges the tacks in the final fillet
#   stitch-weld-ends: with the only start on record (aim_tacks) a 2 in window is 87.8 % full size against D1.1's 90 %
#   fillet-from-both-ends: no start at the far end within the 0.5 mm start-zone lap is on record (the one recorded:
#     3.5 mm)
EIGHT_TASKS = ("position-torch", "set-up-gun", "fit-up-bracket", "half-fillet", "hold-and-present", "fillet-2f",
               "set-up-and-weld", "tack-and-weld")
_SPRAY = ("The spray-transfer schedule for 0.035 inch ER70S-6 wire with 90/10 gas is about 500 in/min (12.7 m/min) wire "
          "feed at 29 V.")
INSTRUCTIONS = {
    "position-torch": (
        "Bring the welding torch to the start of the near-side seam (the end on the robots' right, which is the right "
        "end in the bench camera's image) "
        "ready to weld: wire pointing at the root of the joint, 45 degree work angle, 5 to 10 degree push angle, "
        "3/4 inch (19 mm) contact tip to work. Do not strike an arc; the nozzle must not touch the parts (touching "
        "them with the wire to find the seam is fine)."),
    "tack": (
        "Tack the bracket square to the base plate: one tack 1/2 to 1 inch long at each end of the near side. "
        "Hold the bracket in place while you tack, then let go."),
    "fillet-2f": (
        "Weld a continuous 1/4 inch fillet along the near side of the tacked bracket in the horizontal position, "
        "from end to end, and fill the crater. The spray-transfer schedule for 0.035 inch ER70S-6 wire with "
        "90/10 gas is about 500 in/min (12.7 m/min) wire feed at 29 V."),
    "set-up-gun": (
        "Set the cell up for the spray procedure before welding: set the power source to the procedure's 500 in/min "
        "(12.7 m/min) wire feed and 29 V, and leave 1/2 inch of wire sticking out of the contact tip. Do not strike an "
        "arc, keep the gun clear of the parts and leave the shielding gas off."),
    "fit-up-bracket": (
        "The bracket has been set down on the base plate off its layout. Fit it up for tacking: stand it upright and "
        "square to the plate, centred on the plate along and across it with its faces parallel to the plate's long "
        "edges, to within 1 mm, then let go of it."),
    "half-fillet": (
        "Weld a 1/4 inch fillet along the right half of the near side of the tacked bracket in the horizontal "
        "position: from the right end of the joint (the end on the robots' right, the right end in the bench camera's "
        "image) to its middle, 3 inches, and fill the crater where you stop. Leave the left half of the joint "
        "unwelded. " + _SPRAY),
    "stitch-weld-ends": (
        "Stitch-weld the near side of the tacked bracket in the horizontal position: a 1/4 inch fillet 2 inches long "
        "at each end of the joint, leaving the middle of the joint unwelded. " + _SPRAY),
    "tack-and-weld": (
        "The bracket stands loose on its layout on the clamped base plate. Tack it square to the plate at both ends "
        "of the near side, holding it in place while you tack, then let go and weld a continuous 1/4 inch fillet "
        "along the near side in the horizontal position, from end to end, and fill the crater. " + _SPRAY),
    "fillet-from-both-ends": (
        "Weld a continuous 1/4 inch fillet along the near side of the tacked bracket in the horizontal position in "
        "two runs that start at the two ends of the joint and meet near its middle, the second run tied into the "
        "first, so the joint carries one continuous fillet from end to end; fill the crater. " + _SPRAY),
    "hold-and-present": (
        "Get the joint ready for the first tack. With the hand, grip the bracket by its top edge and hold it upright "
        "and square on its layout; with the welding torch, come to the start of the near-side seam (the end on the "
        "robots' right, which is the right end in the bench camera's image) ready to weld: wire pointing at the root "
        "of the joint, 45 degree work angle, 5 to 10 degree push angle, 3/4 inch (19 mm) contact tip to work. Do not "
        "strike an arc; the nozzle must not touch the parts (touching them with the wire to find the seam is fine). "
        "Finish with the hand still holding the bracket and the torch in place."),
    "set-up-and-weld": (
        "The power source was left on another job's settings. Set it to this joint's spray procedure, 500 in/min "
        "(12.7 m/min) wire feed and 29 V for 0.035 inch ER70S-6 wire with 90/10 gas, then weld a continuous 1/4 inch "
        "fillet along the near side of the tacked bracket in the horizontal position, from end to end, and fill the "
        "crater."),
}
TIME_LIMIT_S = {"position-torch": 60., "tack": 120., "fillet-2f": 180., "set-up-gun": 60., "fit-up-bracket": 120.,
                "half-fillet": 120., "stitch-weld-ends": 180., "tack-and-weld": 300., "fillet-from-both-ends": 240.,
                "hold-and-present": 90., "set-up-and-weld": 180.}
# tasks that need the FR3's hand (torch-only rigs and the G1 are refused)
NEEDS_HAND = {"tack", "fit-up-bracket", "tack-and-weld", "hold-and-present"}
START_STATE = {"position-torch": "tacked", "tack": "loose", "fillet-2f": "tacked", "set-up-gun": "tacked",
               "fit-up-bracket": "loose", "half-fillet": "tacked", "stitch-weld-ends": "tacked",
               "tack-and-weld": "loose", "fillet-from-both-ends": "tacked", "hold-and-present": "loose",
               "set-up-and-weld": "tacked"}
# a task's own initial condition, set on its built start state before tick 0 by welding/presets.py (disclosed in the
# assisted log), as welding/variations.py apply_initial sets a variation's wire stick-out
START_PRESET = {"set-up-gun": "source left on the last job's short-circuit setting, 1/4 in of wire",
                "fit-up-bracket": "bracket stood off its layout",
                "set-up-and-weld": "source left on the last job's short-circuit setting"}
# tasks judged on the weld's sections (the on-demand evaluation freezes a still-liquid pool hands-off, as fillet-2f)
WELD_TASKS = ("fillet-2f", "half-fillet", "stitch-weld-ends", "tack-and-weld", "fillet-from-both-ends",
              "set-up-and-weld")

L_BRACKET = .150
NOMINAL_LEG = 6.35e-3                      # 1/4 in [read: the work order]
UNDERSIZE = 2.4e-3                         # 3/32 in allowed under on <= 10 % of the length [read: D1.1:2000 Table
                                           # 6.1(6)]
# undercut: "shall not exceed 1/32 in. (1 mm), except that a maximum 1/16 in. (2 mm) is permitted for an accumulated
# length of 2 in. (50 mm) in any 12 in. (300 mm)" (D1.1:2000 Table 6.1(7)(A), material under 25 mm) [read]; any 300 mm
# window holds the whole 150 mm joint, so 50 mm over the joint
UNDERCUT_MAX = 2e-3
UNDERCUT_OVER_1MM_LENGTH = 50e-3
END_ALLOWANCE = 5e-3                       # weld may stop this short of each end [authored]; fillet-2f accepts a start's
                                           # cold lap there up to START_OVERLAP_MAX (_eval_fillet_2f)
TACK_MERGE_GAP = 10e-3                     # fused stretches closer than this belong to one tack [authored]
# conditional checks: the weld's quality checks need a weld over this share of the inspected seam (the same 90 % the
# leg-size rule asks to be full size) [authored]; the torch's presentation checks need the wire tip this close to the
# root at the seam's start [authored]
COVERAGE_FOR_QUALITY = .9
OVERLAP_RESOLUTION = 2.5e-4                # m: one cell; AWS D1.1 Fig 5.4 accepts no overlap [read], this is what the
                                           # cell sections can resolve [authored]
PRESENTED_M = 5e-3
# the crater: "All craters shall be filled to provide the specified weld size, except for the ends of intermittent
# fillet welds outside of their effective length" (D1.1:2000 Table 6.1(3), Crater Cross Section [read]).
# It is judged over CRATER_LEN centred on every place a pool froze last (Evaluator._craters) [authored: about half the
# liquid pool the model leaves at arc-off (39-46 slices, 20-23 mm, in the fillet demos), where the last liquid's
# solidification shrinkage leaves the depression; the last-frozen section lies at the arc's last position after a dwell
# (aim_tacks: 146.25-146.75 mm, the arc at 147) and 3-4 mm behind it without one (the fixture's tacks, runs stopped at
# 75 and 76 mm), and the sizes fall between it and the pass's front]. Toward the nearer end of
# the stop's own pass, within FRONT_REACH, the window runs on to that end, whether a bead end or older metal [authored:
# a pool's length at arc-off; the measured fronts lie 4.25 mm and 10.5 mm (a second run, against the first run's metal)
# past their stops]. The bead's own end face (CRATER_END_FACE, two 0.5 mm slices) is left out only at a
# bead end [authored]. A place froze last if no section of its own pass within STOP_RADIUS froze later. A pass (freeze
# episode) is the metal one moving pool left: adjacent sections inside the joint that froze within EPISODE_GAP_S of each
# other; a section beyond a bracket end never splits a pass [authored: inside the joint the largest measured step within
# one pass is 0.79 s (at the first tack's start); the steps of 1.19-2.53 s lie beyond the bracket ends (a strike's
# or a front's last thin section); a strike's first section freezes 3.7-3.8 s after the strike, so a restart
# struck right at an earlier front meets it about 4 s apart]. A later pass that stops within
# STOP_RADIUS of an earlier crater without remelting it, or restarts more than EPISODE_GAP_S later, therefore leaves that
# crater judged, and a crater judges only its own pass's metal. STOP_RADIUS is half the crater
# [authored: a step inside a pass read as a stop at 2.0 mm or less, at aim_tacks' aim zone (6.25 mm)].
CRATER_LEN = 10e-3
CRATER_END_FACE = 1e-3
STOP_RADIUS = CRATER_LEN/2
EPISODE_GAP_S = 2.
FRONT_REACH = 2*CRATER_LEN
# tacks: "Tack welds shall be subject to the same quality requirements as the final welds" (D1.1:2000 5.18.2 [read];
# its two exceptions, preheat and leaving undercut / unfilled craters / porosity, apply only before submerged arc
# welding); a tack is incorporated in a 1/4 in fillet here, and D1.1's tack welder test is a fillet tack of 1/4 in
# maximum size [read, secondary]. A weld run's end sections may be unfused for one section each (the 0.5 mm slicing)
# [authored]
TACK_MAX_SIZE = NOMINAL_LEG
TACK_END_SECTIONS = 1

# fillet-2f. Acceptance basis:
# AWS D1.1:2000 visual and macro-section criteria for a statically loaded nontubular fillet (Table 6.1, Fig 5.4, Annex
# B), the cell's own procedure sheet ("Inspection visual, AWS D1.1"). The two numbers that are not D1.1's are stated as
# such: the porosity limit (D1.1 Table 6.1(8) limits *visible piping* porosity, which the model cannot show; the model's
# porosity is the gas volume fraction of each 0.5 mm slice, held here to ISO 5817:2023 level C's 1.5 %, a projected-area
# figure over the weld: an authored, stricter use of that number) and one cell of overlap (the resolution). The legs'
# difference is not judged: D1.1 accepts unequal legs when the smaller meets the size, which the size (the inscribed
# isosceles triangle) already requires; it is reported in the evaluation's diagnostics (ASME QW-184, a
# qualification-coupon rule, holds it to 3.2 mm). No upper size bound (D1.1 accepts an oversized fillet).
X_READ = (-2e-3, L_BRACKET + 2e-3)         # the sections read: the joint and 2 mm past each end
# penetration, undercut and overlap are read on the 0.25 mm cells as grid edges, which carry float residue (one cell into
# the base plate read 0.2499999999999933 mm and failed ">= 0.25 mm"; eight cells of bracket undercut read
# 2.0000000000000157 mm and failed "<= 2 mm"); they are compared with this tolerance
CELL_TOL = 1e-9
PENETRATION_MIN = 2.5e-4                   # one cell into each member (the etched macro section) [authored resolution]
UNDERCUT_LIMIT = 1e-3                      # D1.1:2000 Table 6.1(7)(A) [read]; UNDERCUT_MAX on UNDERCUT_OVER_1MM_LENGTH
POROSITY_MAX = .015                        # [authored use of ISO 5817 level C, above]
# a start's cold lap within END_ALLOWANCE of an end (outside every crater) is accepted up to two cells, 0.5 mm: what the
# two start techniques shown to clear the tacks' starts leave there (aim_tacks, travel_5p0_x0_1: 0.5 mm on 0.25-4.75 mm);
# every other saved start laps 1.5-12.25 mm there, and E26's nine techniques at least 0.75 mm, a start geometry
# not sized against reality [authored leniency]. It is keyed to place, not to a start: a start away from a
# joint end, as a middle-outward weld's, is held to one cell, and so is a lap running on past 5.0 mm; the demonstrated
# start's lap (aim_tacks) ends one slice inside the zone. Keyed to each pass's metal edge it would take in the film
# beyond the bracket's end and fail aim_tacks; whether to forgive every restart is left open, with the unsized cold
# start
START_OVERLAP_MAX = 2*OVERLAP_RESOLUTION
# the score (reward): 1 for an accepted weld, else FAIL_SCORE_SCALE x progress, progress the share of the required fillet
# that exists as sound, fused weld measured on the final weld (fillet_progress), net of the tacked start; every failure
# stays below every success (the label_application convention, label_application/task.py). The mean of the checks is
# reported beside it (reward.mean_of_checks)
REWARD_CONTRACT = "progress-v1"
FAIL_SCORE_SCALE = .9
# a weld is inspected solid. An on-demand evaluation (GET /v1/evaluation, the benchmark's final one) of a weld whose pool
# is still freezing with the arc off judges the weld as it freezes hands-off: a copy of the weld process is stepped with
# nothing commanded until no slice is liquid, at most SETTLE_MAX_S (the replayed aim_tacks weld: 25 sections liquid at
# the end of its 0.5 s post-flow, where "solidified" failed; its sections solid by 1.5 s after arc off, its pool's last
# slices frozen 1.8 s after). The per-state evaluation (backend option state_evaluation) judges the present
# state, a liquid section as no weld yet
SETTLE_MAX_S = 10.

# ---------------------------------------------------------------- the tasks added after the first three
# Every task but fillet-2f (whose rule is above) is scored the same way: 1 on success, else FAIL_SCORE_SCALE x its
# progress, progress 0 at the untouched start state and continuous in the end state, whatever the route (the fraction
# of checks passed is reported as reward.mean_of_checks)
# set-up-gun: the procedure (params.SCHEDULES spray_0p9_90-10: 12.7 m/min, 29 V; electrical stick-out = contact tip to
# work - 1/4 in = 19 - 6.35 = 12.7 mm, Lincoln L-56 [read], the basis of the variations' stick-out range); tolerances
# [authored]: about a dial's resolution on the feed and the voltage, 1/16 in on the stick-out
SETUP_PROCEDURE = "spray_0p9_90-10"
SETUP_WFS_TOL = .3                         # m/min
SETUP_VOLTS_TOL = .5                       # V
SETUP_STICKOUT = 12.7e-3
SETUP_STICKOUT_TOL = 1.5e-3
# its start (welding/presets.py): the source left on the cell's documented short-circuit job (params.SCHEDULES
# sc_0p9_C25_175A, 6.4 m/min at 20 V, 1/2 in contact tip to work) and that job's 1/4 in of wire (the lower end of the
# variations' range, the same basis); a catalogue variation's own stick-out replaces the wire (variations.apply_initial)
SETUP_LEFT_JOB = "sc_0p9_C25_175A"
SETUP_LEFT_STICKOUT = 6.35e-3
# position-torch's progress: each presentation term is 1 inside its band and falls to 0 at twice the band's half-width
# outside it; gun contact or an arc during the episode halves it (failures of the job, not of the position) [authored]
POSITION_SEAM_START = (0., 10e-3)          # the stretch of the root line the wire tip must reach (at_seam_start)
FAULT_FACTOR = .5
# fit-up-bracket: the bracket stood off its layout by a disclosed preset (welding/presets.py): FITUP_SHIFT toward the
# weld side (+y, onto the open plate between the bracket and the robots; the clamps hold the other edge) and yawed about
# its own vertical axis, seed 0 the design's 25 mm / 6 deg, other seeds drawn in the ranges [authored]. Success: both
# ends of the bracket's root edge (the joint's line) within FIT_MM of the layout (the fit-up the thermal grid describes,
# the tack task's position limit; a centre offset alone misses a yaw), square within 1 deg (the tack task's), released,
# at rest (E0's scale: 1 mm/s), the clamped plate not moved over FITUP_PLATE_MM, no arm latched in a protective stop
FITUP_SHIFT_MM = (15., 30.)
FITUP_YAW_DEG = (3., 8.)
FITUP_SEED0 = (25., 6.)
FITUP_SQUARE_DEG = 1.
FITUP_SQUARE_ZERO_DEG = 10.                # a bracket tipped this far earns no position credit [authored]
FITUP_REST_M_S = 1e-3
FITUP_PLATE_MM = .5
# half-fillet: the right half of the joint, END_ALLOWANCE in from its end to its middle, judged as fillet-2f judges its
# span (size, throat, per-section soundness, fusion, the start zone's lap bound), every crater of the weld laid in the
# episode or touching the half filled to size, and the left half left unwelded up to the second tack (its metal starts
# at 125.75 mm in the tacked start). 70 mm of span hold the aim_tacks start's 5.5 mm of undersize (5.25-10.25 mm)
# inside D1.1's 10 %; a 2 in stitch's 45 mm do not (below)
HALF_SPAN = (END_ALLOWANCE, L_BRACKET/2)
HALF_FUSION_X = (6e-3, .25*L_BRACKET, 70e-3)
HALF_LEFT_FREE = (100e-3, 120e-3)
# stitch-weld-ends (outside the eight-task set): 2 in (50.8 mm) at each end, judged
# END_ALLOWANCE in from the joint's end (the design's windows), the
# middle left bare with 15 mm of run-out and run-in beside it; outside the windows only undercut (base-metal damage) is
# judged, and a crater only where it lies in a window ("except for the ends of intermittent fillet welds outside of
# their effective length", D1.1:2000 Table 6.1(3) [read]); the windows' undersize allowance is D1.1's 10 % of each,
# 4.5 mm, which the aim_tacks start's 5.5 mm exceeds: a stitch needs a start that comes to size sooner
STITCH_WINDOWS = ((END_ALLOWANCE, 50e-3), (100e-3, L_BRACKET - END_ALLOWANCE))
STITCH_FUSION_X = ((6e-3, 27.5e-3, 48e-3), (102e-3, 122.5e-3, 144e-3))
STITCH_MIDDLE_FREE = (65e-3, 85e-3)
FREE_ZONE_PENALTY = .5                     # progress lost per unit share of a zone to be left bare that holds metal
# fillet-from-both-ends (outside the eight-task set: no far-end start within the 0.5 mm
# start-zone lap is on record): fillet-2f's checks, and the weld's passes (freeze episodes, Evaluator._episodes) read
# from the order its sections froze, as an inspector reads a bead's ripples: one run started at each end (within
# END_ALLOWANCE) and ran toward the middle, the two meeting in the joint's middle third, no third run of RUN_MIN_LEN or
# more on the span. A run's direction: which of its two RUN_END_LEN end stretches froze first (its start); shorter
# episodes have none [authored]
BOTH_MEET = (L_BRACKET/3, 2*L_BRACKET/3)
RUN_MIN_LEN = 5e-3
RUN_END_LEN = 5e-3
# hold-and-present: position-torch's checks, unchanged, with the loose bracket held: both Franka fingers on it (the tack
# demonstrations grip its top edge across its 8 mm thickness, 30 N per finger) and the bracket square within 1 deg and
# within FIT_MM of its fit-up, the tack task's own limits (E24: the grip left it 0.058 deg and 0.068 mm off).
# Progress: HOLD_PRESENT_TORCH_SHARE x position-torch's progress + the rest x the hold (held x the alignment term)
# [authored]
HOLD_PRESENT_TORCH_SHARE = .5
# set-up-and-weld: fillet-2f's checks, unchanged, on a tacked start whose source the preset left on the cell's short-
# circuit job (SETUP_LEFT_JOB; the wire as the tacked start leaves it), and the source at the procedure at the end
# (SETUP_PROCEDURE within SETUP_WFS_TOL / SETUP_VOLTS_TOL, as set-up-gun judges it)


def _overlap(r):
    return max(r["overlap_base_m"], r["overlap_bracket_m"])


def _undercut(r):
    return max(r["undercut_base_m"], r["undercut_bracket_m"])


def _convexity_limit(r):
    """D1.1 Fig 5.4: 1/16 in for a face up to 5/16 in wide, else 1/8 in [read]."""
    return 3.2e-3 if r["face_width_m"] > 7.9e-3 else 1.6e-3


def _fused_section(r):
    """An etched macro section shows this section fused: the root melted and both members entered by a cell."""
    return bool(r["root_fused"] and r["penetration_base_m"] >= PENETRATION_MIN - CELL_TOL
                and r["penetration_bracket_m"] >= PENETRATION_MIN - CELL_TOL)


def _key(r):
    return round(r["x_m"]*1e6)


def _span(r):
    return END_ALLOWANCE - 1e-9 <= r["x_m"] <= L_BRACKET - END_ALLOWANCE + 1e-9


def _undercut_beyond_allowance(rows):
    """Keys of the joint's sections whose undercut over 1 mm (up to 2 mm) lies beyond the 50 mm D1.1:2000 Table
    6.1(7)(A) allows: the shallowest are kept within the allowance and the deepest charged first [authored order]."""
    mid = sorted((r for r in rows if 0. <= r["x_m"] <= L_BRACKET
                  and UNDERCUT_LIMIT + CELL_TOL < _undercut(r) <= UNDERCUT_MAX + CELL_TOL),
                 key=lambda r: (_undercut(r), r["x_m"]))
    used, out = 0., set()
    for r in mid:
        if used + r["dx_m"] <= UNDERCUT_OVER_1MM_LENGTH + CELL_TOL:
            used += r["dx_m"]
        else:
            out.add(_key(r))
    return out


def _section_defects(r, undercut_beyond=frozenset(), overlap_limit=OVERLAP_RESOLUTION):
    """Conditions that reject this one section (D1.1 Fig 5.4, Table 6.1(7)(A); porosity as above): the per-section terms
    of fillet_progress. Undercut rejects a section over 2 mm, or over 1 mm where it lies beyond the 50 mm the code allows
    (_undercut_beyond_allowance), as success judges it (charging every section over 1 mm would give an accepted weld
    progress 0.59). The undersize success allows on 10 % of the length
    is not an allowance of progress: a section earns its size fraction."""
    out = []
    if _overlap(r) > overlap_limit + CELL_TOL:
        out.append("overlap")
    if _undercut(r) > UNDERCUT_MAX + CELL_TOL or _key(r) in undercut_beyond:
        out.append("undercut")
    if r["porosity"] > POROSITY_MAX:
        out.append("porosity")
    if r["convexity_m"] > _convexity_limit(r) + CELL_TOL:
        out.append("convexity")
    return out


def _required(rows):
    """The sections of the span the weld must cover: the joint less the END_ALLOWANCE it may stop short of each end."""
    return [r for r in rows if END_ALLOWANCE - 1e-9 <= r["x_m"] <= L_BRACKET - END_ALLOWANCE + 1e-9]


def section_credit(r, beyond=frozenset(), overlap_limit=OVERLAP_RESOLUTION):
    """One section's credit toward the required fillet (fillet_progress, and the added tasks that judge part
    of the joint): (min(1, size / 6.35 mm) if solid, fused and free of a rejecting condition, else 0; why it earns
    nothing, or None; the rejecting conditions)."""
    if r.get("liquid"):
        return 0., "liquid", ()
    if r["weld_area_m2"] <= 0 or r["size_m"] <= 0:
        return 0., "no_fillet", ()
    if not _fused_section(r):
        return 0., "unfused", ()
    bad = _section_defects(r, beyond, overlap_limit)
    if bad:
        return 0., "defective", bad
    return min(1., r["size_m"]/NOMINAL_LEG), None, ()


def fillet_progress(rows, start_raw=None, craters=None, eligible=None):
    """The fillet-2f progress reward (REWARD_CONTRACT) from one state's sections, whatever route produced them: the
    length-weighted share of the required 1/4 in fillet that exists as sound, fused weld over the span the weld must
    cover. Each 0.5 mm section of the span earns min(1, size / 6.35 mm) (a fillet's capacity per length is proportional
    to its effective throat, 0.707 x size; oversize is capped) if it is solid, fused (root melted, both members entered by
    a cell) and free of a rejecting condition (_section_defects), else nothing: a defect costs its length. What success
    judges outside the span costs progress the same way, and sound metal there earns nothing more: a crater section
    there (Evaluator._craters, e.g. the stop at a bracket end) is charged its size deficit, or all of it if rejected; a
    start zone's weld metal is charged whole if undercut, porosity, convexity or overlap beyond START_OVERLAP_MAX rejects
    it. Liquid metal is not weld yet.
    raw = (sum(dx credit) - sum(dx charge outside)) / sum(dx) over the span; progress = (raw - start_raw) / (1 -
    start_raw) clipped to [0, 1], start_raw the tacked start state's raw (its tacks' sound, undersized metal), so standing
    still earns 0; net_progress is the same unclipped (damage below the start state reads negative). No travel
    direction, run count, order, schedule or technique enters it. `eligible` (fillet-from-both-ends only; None for
    fillet-2f) withholds the credit of a span section it rejects (lost as "wrong_direction")."""
    if craters is None:
        craters = Evaluator._craters(rows)
    beyond = _undercut_beyond_allowance(rows)
    in_crater = {_key(r) for c in craters for r in c["rows"]}

    def credit(r, overlap_limit=OVERLAP_RESOLUTION):
        return section_credit(r, beyond, overlap_limit)
    req = _required(rows)
    L = sum(r["dx_m"] for r in req)
    total, outside = 0., 0.
    lost, by = dict(liquid=0., no_fillet=0., unfused=0., defective=0., undersize_equivalent=0.), {}
    if eligible is not None:
        lost["wrong_direction"] = 0.
    for r in req:
        c, why, bad = credit(r)
        if eligible is not None and why is None and not eligible(r):
            c, why = 0., "wrong_direction"
        total += r["dx_m"]*c
        if why:
            lost[why] += r["dx_m"]
        else:
            lost["undersize_equivalent"] += r["dx_m"]*(1. - c)
        for d in bad:
            by[d] = by.get(d, 0.) + r["dx_m"]
    for r in rows:
        if not 0. <= r["x_m"] <= L_BRACKET or _span(r) or r["weld_area_m2"] <= 0:
            continue
        if _key(r) in in_crater:
            outside += r["dx_m"]*(1. - credit(r)[0])
        elif _section_defects(r, beyond, START_OVERLAP_MAX):
            outside += r["dx_m"]
    raw = (total - outside)/L if L > 0 else 0.
    net = (raw - start_raw)/(1. - start_raw) if start_raw is not None and start_raw < 1. else raw
    progress = float(np.clip(net, 0., 1.)) if start_raw is None or start_raw < 1. else float(raw >= 1.)
    return dict(progress=progress, net_progress=float(net), raw=raw, start_raw=start_raw, required_mm=round(L*1e3, 3),
                credited_mm=round(total*1e3, 3), charged_outside_span_mm=round(outside*1e3, 3),
                lost_mm={k: round(v*1e3, 3) for k, v in lost.items()},
                defective_by_condition_mm={k: round(v*1e3, 3) for k, v in sorted(by.items())})


def freeze_hands_off(process, tip, axis, dt, max_s=SETTLE_MAX_S):
    """A copy of the weld process (the world's is not touched) stepped with nothing commanded -- the torch held where it
    is, the arc off -- until no slice is liquid, at most max_s simulated: the weld as it freezes if left alone. The
    copy keeps the world's pool obstacle (the torch's and fixtures' geometry, which hold still). Returns (process copy,
    simulated s)."""
    from .process import WeldProcess
    p = process
    q = WeldProcess(p.c, device=p.s.device, draft_m_s=p.draft, gravity=tuple(np.asarray(p.gravity, float)),
                    pool_every=p.pool_every)
    q.pool.obstacle = p.pool.obstacle
    q.restore(p.snapshot())
    t0, every = q.time, max(1, int(round(.05/dt)))
    while q.time - t0 < max_s - 1e-9 and any(sl.liquid for sl in q.pool.slices):
        for _ in range(every):
            q.tick(dt, tip, axis, (0., 0., 0.), circuit_closed=p.circuit_closed)
    return q, q.time - t0


# ---------------------------------------------------------------- progress terms and judgements of the added tasks
TASK_REWARD_CONTRACT = "progress-v2"


def graded(err, band, fall):
    """1 while err <= band, falling linearly to 0 at band + fall (an unreadable err earns 0)."""
    if err is None or not math.isfinite(err):
        return 0.
    return float(1. - np.clip((err - band)/fall, 0., 1.))


def net_of_start(err, err0, band):
    """1 inside the band, 0 at the start state's error err0 or worse, linear between; None when the start was already
    inside the band (the term then earns nothing either way)."""
    if err0 is None or not math.isfinite(err0) or err0 <= band + 1e-12:
        return None
    if err is None or not math.isfinite(err):
        return 0.
    return float(1. - np.clip((err - band)/(err0 - band), 0., 1.))


def _mean(values):
    return float(np.mean(values)) if values else 0.


def _in(r, lo, hi):
    return lo - 1e-9 <= r["x_m"] <= hi + 1e-9


def seam_start_distance(wire_tip):
    """Distance (m) of the wire tip (weld frame) from the root line's stretch POSITION_SEAM_START (position-torch)."""
    x0, x1 = POSITION_SEAM_START
    dx = max(x0 - float(wire_tip[0]), 0., float(wire_tip[0]) - x1)
    return float(math.sqrt(dx*dx + float(wire_tip[1])**2 + float(wire_tip[2])**2))


def position_progress(d, d0, work_deg, travel_deg, ctwd_m, settled_mm, faulted):
    """position-torch's progress: reach x pose. Reach is the approach of the wire tip to the seam's start net of the
    start state (1 within 1 mm, the wire_on_root limit); pose the mean of the four presentation terms (work angle
    40-50 deg, push 0-15 deg, contact tip to work 17-21 mm, TCP motion under 0.5 mm in the last second), each falling to
    0 at twice its band's half-width outside the band; halved for gun contact or an arc during the episode."""
    reach = net_of_start(d, d0, 1e-3)
    if reach is None:
        reach = float(d <= 1e-3 + 1e-12)
    pose = dict(work_angle=graded(abs(work_deg - 45.), 5., 10.), travel_angle=graded(abs(travel_deg - 7.5), 7.5, 15.),
                ctwd=graded(abs(ctwd_m - 19e-3) if ctwd_m is not None else None, 2e-3, 4e-3),
                settled=graded(settled_mm, .5, .5))
    p = reach*_mean(list(pose.values()))*(FAULT_FACTOR if faulted else 1.)
    return dict(progress=float(p), reach=reach, pose_terms=pose, seam_distance_mm=d*1e3, start_seam_distance_mm=d0*1e3,
                fault_factor=FAULT_FACTOR if faulted else 1.)


def setup_progress(wfs_m_min, volts, stickout_m, start, faulted):
    """set-up-gun's progress: the mean of the wire feed, voltage and stick-out terms, each net of the start state
    (net_of_start; a term already in its band at the start is left out); halved for gun contact or an arc."""
    target = PR.v(PR.SCHEDULES)[SETUP_PROCEDURE]
    terms = dict(
        wire_feed=net_of_start(abs(wfs_m_min - target["wfs_m_min"]), abs(start["wfs_m_min"] - target["wfs_m_min"]),
                               SETUP_WFS_TOL),
        voltage=net_of_start(abs(volts - target["voltage_v"]), abs(start["voltage_v"] - target["voltage_v"]),
                             SETUP_VOLTS_TOL),
        stickout=net_of_start(abs(stickout_m - SETUP_STICKOUT), abs(start["stickout_m"] - SETUP_STICKOUT),
                              SETUP_STICKOUT_TOL))
    p = _mean([v for v in terms.values() if v is not None])*(FAULT_FACTOR if faulted else 1.)
    return dict(progress=float(p), terms=terms, fault_factor=FAULT_FACTOR if faulted else 1.)


def fitup_displacement(seed):
    """(shift m toward +y, yaw deg) of fit-up-bracket's start (welding/presets.py) for a world seed (Config.seed: the
    legacy seed or a variation's config_seed); deterministic, seed 0 the design's 25 mm / 6 deg."""
    if int(seed) == 0:
        return FITUP_SEED0[0]*1e-3, FITUP_SEED0[1]
    rng = np.random.default_rng(271828 + int(seed))
    shift = float(rng.uniform(*FITUP_SHIFT_MM))*1e-3
    yaw = float(rng.uniform(*FITUP_YAW_DEG))*(1. if rng.uniform() < .5 else -1.)
    return shift, yaw


def fitup_errors(pb, Rb, pk, Rk, origin_in_base, origin_in_bracket, nominal_rel, length=L_BRACKET):
    """The bracket's fit-up errors against its layout, in the base plate's frame: the larger displacement of the two
    ends of its root edge (the joint's line: weld-frame x = 0 and L on the bracket) from where the layout has them,
    the centre's offset (env.bracket_alignment's), its tilt from upright (squareness) and its yaw."""
    pb, Rb, pk, Rk = (np.asarray(a, float) for a in (pb, Rb, pk, Rk))
    Rrel = Rb.T @ Rk
    ends = []
    for x in (0., length):
        e = np.array([x, 0., 0.])
        ends.append(float(np.linalg.norm(Rb.T @ (pk + Rk @ (np.asarray(origin_in_bracket, float) + e) - pb)
                                         - (np.asarray(origin_in_base, float) + e))))
    return dict(root_edge_mm=max(ends)*1e3, root_ends_mm=[round(v*1e3, 4) for v in ends],
                centre_mm=float(np.linalg.norm(Rb.T @ (pk - pb) - np.asarray(nominal_rel, float)))*1e3,
                tilt_deg=math.degrees(math.acos(float(np.clip(Rrel[2, 2], -1., 1.)))),
                yaw_deg=math.degrees(math.atan2(Rrel[1, 0], Rrel[0, 0])))


def fitup_progress(err, start_err, held):
    """fit-up-bracket's progress: the root edge's approach to its layout net of the start (1 within FIT_MM), times a
    squareness term (1 within 1 deg, 0 tipped FITUP_SQUARE_ZERO_DEG), and 0 while the hand still touches the bracket
    (held is not placed)."""
    pos = net_of_start(err["root_edge_mm"], start_err["root_edge_mm"], FIT_MM)
    if pos is None:
        pos = float(err["root_edge_mm"] <= FIT_MM + 1e-9)
    sq = graded(err["tilt_deg"], FITUP_SQUARE_DEG, FITUP_SQUARE_ZERO_DEG - FITUP_SQUARE_DEG)
    return dict(progress=0. if held else float(pos*sq), position=pos, squareness=sq, held=bool(held))


def hold_present_progress(torch_progress, held, square_deg, offset_mm):
    """hold-and-present's progress: HOLD_PRESENT_TORCH_SHARE x position-torch's progress (position_progress, net of the
    start) + the rest x the hold: 1 while both fingers hold the bracket, times its alignment (squareness and position
    each 1 within the tack task's limits, 0 at twice: the smaller); 0 while not held."""
    align = min(graded(abs(square_deg), 1., 1.), graded(offset_mm, FIT_MM, FIT_MM))
    hold = float(align) if held else 0.
    return dict(progress=float(HOLD_PRESENT_TORCH_SHARE*torch_progress + (1. - HOLD_PRESENT_TORCH_SHARE)*hold),
                torch_progress=float(torch_progress), hold=hold, alignment=float(align), held=bool(held))


def tack_progress(tacks, bonded, square_deg, offset_mm, touching, quality_share):
    """tack's progress (the task kept outside the eight-task set): 0.5 x B + 0.5 x Q, both 0 without a bond. B: each of
    the two places (a tack starting within 40 mm of the near end, one ending past 110 mm) earns half its longest fused
    tack / 12.7 mm, capped at 1; Q: the mean of squareness and position (1 within their limits, 0 at twice), release, and
    the share of the weld runs' sections that are fused and free of a rejecting condition."""
    if not bonded:
        return dict(progress=0., bonded_tacks=0., quality=0., bonded=False)
    near = max((t["fused_mm"] for t in tacks if t["x0_mm"] < 40.), default=0.)
    far = max((t["fused_mm"] for t in tacks if t["x1_mm"] > 110.), default=0.)
    b = (min(1., near/12.7) + min(1., far/12.7))/2
    q = _mean([graded(abs(square_deg), 1., 1.), graded(offset_mm, FIT_MM, FIT_MM), 0. if touching else 1.,
               quality_share])
    return dict(progress=float(.5*b + .5*q), bonded_tacks=float(b), quality=float(q), bonded=True)


def episode_metal(start_rows):
    """A predicate on sections: True where the section holds weld metal the episode laid or melted again, False where it
    is bare or holds the start state's metal as it froze then (the fixture's tacks, untouched)."""
    old = {_key(r): r.get("freeze_t") for r in (start_rows or []) if r["weld_area_m2"] > 0}

    def new(r):
        if r["weld_area_m2"] <= 0:
            return False
        k = _key(r)
        if k not in old:
            return True
        t, t0 = r.get("freeze_t"), old[k]
        return t is None or t0 is None or abs(t - t0) > 1e-9
    return new


def window_credit(rows, lo, hi, beyond=None):
    """The length-weighted share of the 1/4 in fillet over [lo, hi] that exists as sound, fused weld (section_credit)."""
    beyond = _undercut_beyond_allowance(rows) if beyond is None else beyond
    sec = [r for r in rows if _in(r, lo, hi)]
    L = sum(r["dx_m"] for r in sec)
    return sum(r["dx_m"]*section_credit(r, beyond)[0] for r in sec)/L if L > 0 else 0.


def _window_checks(rows, lo, hi, fusion_x, beyond, suffix=""):
    """fillet-2f's span checks over one window [lo, hi]: continuous fused metal, size (>= 6.35 mm on 90 % of the
    window, >= 3.95 mm everywhere: D1.1:2000 Table 6.1(6)), median throat, every section sound (_section_defects, overlap
    at one cell) and macro sections fused at fusion_x."""
    sec = [r for r in rows if _in(r, lo, hi)]
    dxs = np.array([r["dx_m"] for r in sec]) if sec else np.ones(1)
    L = float(dxs.sum())
    size = np.array([r["size_m"] for r in sec]) if sec else np.zeros(1)
    frac = float(np.sum(dxs*(size >= NOMINAL_LEG - 1e-9))/L)
    covered = float(np.sum(dxs*(size > 0))/L)
    unfused = [round(r["x_m"]*1e3, 2) for r in sec if not (r["weld_area_m2"] > 0 and r["root_fused"])]
    bad = [(round(r["x_m"]*1e3, 2), _section_defects(r, beyond)) for r in sec
           if r["weld_area_m2"] > 0 and _section_defects(r, beyond)]
    throat = float(np.median([r["throat_m"] for r in sec])) if sec else 0.
    fus = []
    for xq in fusion_x:
        r = min(rows, key=lambda r_: abs(r_["x_m"] - xq)) if rows else None
        fus.append(dict(x_mm=round(xq*1e3, 3), ok=bool(r and _fused_section(r))))
    return {
        "continuous" + suffix: dict(window_mm=[lo*1e3, hi*1e3], unfused_or_bare_mm=unfused[:40],
                                    ok=_ok(sec and not unfused)),
        "leg_size" + suffix: dict(fraction_size_ge_6_35mm=frac, min_size_mm=float(size.min()*1e3), sections=len(sec),
                                  ok=_ok(frac >= .9 - 1e-9 and size.min() >= NOMINAL_LEG - UNDERSIZE)),
        "throat" + suffix: dict(median_mm=throat*1e3, limit_mm=.707*NOMINAL_LEG*1e3,
                                ok=_ok(sec and throat >= .707*NOMINAL_LEG)),
        "sound" + suffix: dict(rejected_sections=bad[:40], weld_coverage=covered,
                               ok=_ok(covered >= COVERAGE_FOR_QUALITY and not bad)),
        "fusion_sections" + suffix: dict(sections=fus, ok=_ok(all(f["ok"] for f in fus))),
    }


def _joint_checks(rows, free=None, free_name="left_unwelded"):
    """What every partial-length task judges on the whole joint: undercut (D1.1:2000 Table 6.1(7)(A), base-metal damage
    wherever it lies), a zone left bare where the work order leaves it unwelded, and the weld solid and frozen."""
    joint = [r for r in rows if 0. <= r["x_m"] <= L_BRACKET]
    uc = [_undercut(r) for r in joint]
    uc_over = sum(r["dx_m"] for r, u in zip(joint, uc) if u > UNDERCUT_LIMIT + CELL_TOL)
    liquid = sum(1 for r in rows if r["liquid"])
    freezing = sum(1 for r in rows if r["weld_area_m2"] > 0 and r.get("freeze_t") is None)
    out = {"undercut": dict(max_mm=max(uc, default=0.)*1e3, over_1mm_length_mm=uc_over*1e3,
                            ok=_ok(max(uc, default=0.) <= UNDERCUT_MAX + CELL_TOL
                                   and uc_over <= UNDERCUT_OVER_1MM_LENGTH + CELL_TOL))}
    if free is not None:
        share = free_share(rows, *free)
        out[free_name] = dict(zone_mm=[free[0]*1e3, free[1]*1e3], metal_share=share, ok=_ok(share == 0.))
    out["solidified"] = dict(liquid_sections=liquid, freezing_sections=freezing, ok=_ok(liquid == 0 and freezing == 0))
    return out


def free_share(rows, lo, hi):
    """The share of [lo, hi] holding weld metal."""
    sec = [r for r in rows if _in(r, lo, hi)]
    L = sum(r["dx_m"] for r in sec)
    return float(sum(r["dx_m"] for r in sec if r["weld_area_m2"] > 0)/L) if L > 0 else 0.


def weld_runs(rows):
    """The weld's passes on the joint, read from the order its sections froze (as an inspector reads a bead's ripples):
    each freeze episode of each bead (Evaluator._episodes) clipped to the joint, its extent, its sections' keys and the
    way it ran, +1 toward +x or -1 toward -x (its start is the end stretch of RUN_END_LEN that froze first), 0 where it
    is shorter than RUN_MIN_LEN (a remnant, not a run)."""
    out = []
    for bead in Evaluator._beads(rows):
        for a, b in Evaluator._episodes(bead):
            ep = [r for r in bead[a:b] if 0. <= r["x_m"] <= L_BRACKET]
            if not ep:
                continue
            x0, x1 = ep[0]["x_m"] - ep[0]["dx_m"]/2, ep[-1]["x_m"] + ep[-1]["dx_m"]/2
            way = 0
            if x1 - x0 >= RUN_MIN_LEN - 1e-9:
                t_lo = np.mean([r["freeze_t"] for r in ep if r["x_m"] <= x0 + RUN_END_LEN + 1e-9])
                t_hi = np.mean([r["freeze_t"] for r in ep if r["x_m"] >= x1 - RUN_END_LEN - 1e-9])
                way = 1 if t_hi > t_lo else -1 if t_lo > t_hi else 0
            out.append(dict(x0_mm=round(x0*1e3, 3), x1_mm=round(x1*1e3, 3), direction=way,
                            keys=frozenset(_key(r) for r in ep)))
    return out


def toward_middle(runs):
    """fillet-from-both-ends' eligibility: a section's metal counts where the run that laid it moved toward the middle
    from its own side: a +x run up to the far edge of the meeting third, a -x run down to its near edge."""
    way = {k: r["direction"] for r in runs for k in r["keys"]}

    def eligible(r):
        d = way.get(_key(r), 0)
        return (d > 0 and r["x_m"] <= BOTH_MEET[1] + 1e-9) or (d < 0 and r["x_m"] >= BOTH_MEET[0] - 1e-9)
    return eligible


def both_ends_sequence(runs):
    """The two runs fillet-from-both-ends asks for: one +x run starting within END_ALLOWANCE of the near end and one
    -x run starting within END_ALLOWANCE of the far end, adjacent (the second tied into the first), meeting in the middle
    third, and no other run of RUN_MIN_LEN or more on the span."""
    long_ = [r for r in runs if r["x1_mm"] - r["x0_mm"] >= RUN_MIN_LEN*1e3 - 1e-6]
    lo_end, hi_end = END_ALLOWANCE*1e3 + 1e-6, (L_BRACKET - END_ALLOWANCE)*1e3 - 1e-6
    low = [r for r in long_ if r["direction"] > 0 and r["x0_mm"] <= lo_end]
    high = [r for r in long_ if r["direction"] < 0 and r["x1_mm"] >= hi_end]
    others = [r for r in long_ if r not in low and r not in high
              and r["x1_mm"] > lo_end and r["x0_mm"] < hi_end]
    meet = None
    if len(low) == 1 and len(high) == 1 and abs(low[0]["x1_mm"] - high[0]["x0_mm"]) <= 1e-6:
        meet = low[0]["x1_mm"]
    ok = bool(meet is not None and BOTH_MEET[0]*1e3 - 1e-6 <= meet <= BOTH_MEET[1]*1e3 + 1e-6 and not others)
    show = lambda rs: [{k: v for k, v in r.items() if k != "keys"} for r in rs]
    return dict(runs=show(runs), from_near_end=show(low), from_far_end=show(high), other_runs=show(others),
                meet_mm=meet, meet_window_mm=[BOTH_MEET[0]*1e3, BOTH_MEET[1]*1e3], ok=ok)


def _ok(v):
    return bool(v)


class Evaluator:
    def __init__(self, env):
        self.env = env
        # fillet-2f's progress is net of the start state: the environment builds its evaluator on the start state it
        # hands over (WeldingEnv.__init__, after the tacked start is placed), and every reset restores that state
        self.start_raw_progress = None
        # the other weld tasks' progress is net of their start state's sections too (kept: start_rows)
        self.start_rows = None
        self._start_snap = self._start = None
        w = getattr(env, "world", None)
        if getattr(getattr(env, "config", None), "task", None) in WELD_TASKS and getattr(w, "process", None) is not None:
            self.start_rows = SC.weld_report(w.process, X_READ)
            self.start_raw_progress = fillet_progress(self.start_rows)["raw"]

    def evaluate(self, settle=False):
        """success (every check), score and checks. fillet-2f: score = 1 on success, else 0.9 x progress (REWARD_CONTRACT);
        settle=True (the on-demand evaluation) judges a weld still freezing with the arc off as it freezes hands-off.
        Every other task: 1 on success, else 0.9 x its progress (TASK_REWARD_CONTRACT); the weld tasks
        judge a freezing weld as fillet-2f does."""
        task = self.env.config.task
        if task == "fillet-2f":
            return self._evaluate_fillet_2f(settle)
        if task in START_PRESET and getattr(self.env, "start_preset", None) is None:
            raise RuntimeError(f"task {task!r} starts from its preset ({START_PRESET[task]}), which was not applied to "
                               f"this environment: build it through the backend, or call welding.presets."
                               f"apply_start_preset(env) before tick 0")
        if task in WELD_TASKS:
            rows, judged, flags = self._final_rows(settle)
            if task == "tack-and-weld":
                return self.judge_tack_and_weld(rows, judged, flags, **self._assembly())
            if task == "set-up-and-weld":
                s = self.env.world.process.src.s
                return self.judge_set_up_and_weld(rows, judged, flags, wfs_m_min=float(s.wfs_m_min),
                                                  voltage_v=float(s.voltage_v))
            return getattr(self, "judge_" + task.replace("-", "_"))(rows, judged, flags)
        fn = getattr(self, "_eval_" + task.replace("-", "_"))
        checks = {k: {("pass" if kk == "ok" else kk): vv for kk, vv in c.items()} for k, c in fn().items()}
        passed = [c["pass"] for c in checks.values()]
        reward = getattr(self, "_progress_" + task.replace("-", "_"))()
        return self._scored(bool(passed) and all(passed), checks, reward)

    @staticmethod
    def _scored(success, checks, reward, **extra):
        """1 on success, else FAIL_SCORE_SCALE x progress (TASK_REWARD_CONTRACT); the mean of the checks is reported."""
        passed = [c["pass"] for c in checks.values()]
        reward = dict(reward, contract=TASK_REWARD_CONTRACT, score_rule="1 if success else 0.9 x progress",
                      mean_of_checks=_mean(passed))
        return dict(success=bool(success), score=1. if success else FAIL_SCORE_SCALE*reward["progress"], checks=checks,
                    progress=reward["progress"], reward=reward, **extra)

    def _start_measures(self):
        """What the progress of position-torch, set-up-gun and fit-up-bracket is net of, read from the environment's
        initial checkpoint (the state a reset restores; read again when a catalogue variation's initial condition or a
        task preset replaces it): the world's own torch_pose / weld_frame on the checkpoint's body poses."""
        snap = self.env.initial
        if self._start_snap is snap:
            return self._start
        from scipy.spatial.transform import Rotation
        from .world import World
        sc = self.env.world.scene
        bq = np.asarray(snap["rigid"]["body_q"], float)

        def body_pose(body):
            return bq[body, :3], Rotation.from_quat(bq[body, 3:]).as_matrix()
        view = SimpleNamespace(scene=sc, body_pose=body_pose)
        st, s = snap["process"]["st"], snap["process"]["settings"]
        tip, axis, _ = World.torch_pose(view)
        o, R = World.weld_frame(view)
        m = dict(seam_distance_m=seam_start_distance(R.T @ (tip - o) + (R.T @ axis)*st.l_s),
                 wfs_m_min=float(s.wfs_m_min), voltage_v=float(s.voltage_v), stickout_m=float(st.l_s))
        if "bracket" in sc.bodies:
            m.update(fitup=self._fitup_errors(body_pose), plate_m=np.array(body_pose(sc.bodies["base_plate"])[0]))
        self._start_snap, self._start = snap, m
        return m

    def _fitup_errors(self, body_pose):
        sc = self.env.world.scene
        pb, Rb = body_pose(sc.bodies["base_plate"])
        pk, Rk = body_pose(sc.bodies["bracket"])
        return fitup_errors(pb, Rb, pk, Rk, sc.weld_origin_in_base, sc.weld_origin_in_bracket, sc.bracket_rel_nominal)

    def _progress_position_torch(self):
        m = self._measured
        return position_progress(seam_start_distance(m["wire_tip"]), self._start_measures()["seam_distance_m"],
                                 m["work"], m["travel"], m["ctwd"], m["settled"], m["faulted"])

    # ------------------------------------------------------------------ set-up-gun (A7)
    def _eval_set_up_gun(self):
        """The source's set points as its state holds them (what the tool_io outputs set, not a command log), the wire's
        stick-out, no arc, the gun's hardware clear of everything (the wire may touch: touch sensing measures the
        stick-out against the plate), the gas valve closed at the end (a purge during the episode is allowed)."""
        env, w = self.env, self.env.world
        st, s = w.process.src.st, w.process.src.s
        target = PR.v(PR.SCHEDULES)[SETUP_PROCEDURE]
        contacts = self._gun_contacts()
        h = env.history
        self._measured = dict(wfs=float(s.wfs_m_min), volts=float(s.voltage_v), stickout=float(st.l_s),
                              faulted=bool(h["gun_contact_ticks"] or h["arc_s"] or contacts))
        return {
            "wire_feed": dict(value_m_min=float(s.wfs_m_min), target_m_min=target["wfs_m_min"],
                              tolerance_m_min=SETUP_WFS_TOL,
                              ok=_ok(abs(s.wfs_m_min - target["wfs_m_min"]) <= SETUP_WFS_TOL + 1e-9)),
            "voltage": dict(value_v=float(s.voltage_v), target_v=target["voltage_v"], tolerance_v=SETUP_VOLTS_TOL,
                            ok=_ok(abs(s.voltage_v - target["voltage_v"]) <= SETUP_VOLTS_TOL + 1e-9)),
            "stickout": dict(value_mm=float(st.l_s*1e3), target_mm=SETUP_STICKOUT*1e3,
                             tolerance_mm=SETUP_STICKOUT_TOL*1e3,
                             ok=_ok(abs(st.l_s - SETUP_STICKOUT) <= SETUP_STICKOUT_TOL + 1e-9)),
            "no_arc": dict(arc_s=h["arc_s"], ok=_ok(h["arc_s"] == 0.)),
            "no_contact": dict(contacts=contacts, wire_touched=bool(h["wire_touch"]),
                               gun_contact_ticks=h["gun_contact_ticks"],
                               ok=_ok(not contacts and h["gun_contact_ticks"] == 0)),
            "gas_off": dict(gas_on=bool(st.gas_on), ok=_ok(not st.gas_on)),
        }

    def _progress_set_up_gun(self):
        m = self._measured
        return setup_progress(m["wfs"], m["volts"], m["stickout"], self._start_measures(), m["faulted"])

    # ------------------------------------------------------------------ fit-up-bracket (A8)
    def _eval_fit_up_bracket(self):
        """The bracket on its layout (both ends of its root edge within FIT_MM), square, released, at rest; the clamped
        plate not moved; no arm latched in a protective stop."""
        env, w = self.env, self.env.world
        sc = w.scene
        err = self._fitup_errors(w.body_pose)
        held = env.hand_touches_bracket()
        speed = float(np.linalg.norm(w.state.body_qd.numpy()[sc.bodies["bracket"], :3]))
        moved = float(np.linalg.norm(w.body_pose(sc.bodies["base_plate"])[0] - self._start_measures()["plate_m"]))*1e3
        stopped = sorted(a for a, until in w._reflex_until.items() if w.time < until)
        self._measured = dict(err=err, held=held)
        return {
            "on_layout": dict(root_edge_mm=err["root_edge_mm"], root_ends_mm=err["root_ends_mm"],
                              centre_offset_mm=err["centre_mm"], yaw_deg=err["yaw_deg"], limit_mm=FIT_MM,
                              ok=_ok(err["root_edge_mm"] <= FIT_MM + 1e-9)),
            "square": dict(angle_error_deg=err["tilt_deg"], limit_deg=FITUP_SQUARE_DEG,
                           ok=_ok(err["tilt_deg"] <= FITUP_SQUARE_DEG + 1e-9)),
            "released": dict(touching=bool(held), ok=_ok(not held)),
            "at_rest": dict(speed_mm_s=speed*1e3, limit_mm_s=FITUP_REST_M_S*1e3, ok=_ok(speed < FITUP_REST_M_S)),
            "plate_unmoved": dict(moved_mm=moved, limit_mm=FITUP_PLATE_MM, ok=_ok(moved <= FITUP_PLATE_MM + 1e-9)),
            "no_protective_stop": dict(stopped_arms=stopped, ok=_ok(not stopped)),
        }

    def _progress_fit_up_bracket(self):
        m = self._measured
        return fitup_progress(m["err"], self._start_measures()["fitup"], m["held"])

    # ------------------------------------------------------------------ hold-and-present (A13)
    def _fingers_on_bracket(self):
        """Whether each finger of the handling hand touches the bracket (MuJoCo contacts with load): a grip across its
        thickness has both; a hand resting on it or one finger pushing it does not."""
        w = self.env.world
        if "bracket" not in w.scene.bodies or not w.hand_arms:
            return []
        lab = list(w.model.body_label)
        fingers = [b for b in w.hand_bodies() if "finger" in lab[b].split("/")[-1]]
        return [bool(w.bodies_touch(w.scene.bodies["bracket"], {b})) for b in fingers]

    def _eval_hold_and_present(self):
        """position-torch's checks, unchanged, and the hold: both fingers on the bracket, the bracket square within 1 deg
        and within FIT_MM of its fit-up (env.bracket_alignment, the tack task's measures), no arm latched in a protective
        stop."""
        env, w = self.env, self.env.world
        checks = self._eval_position_torch()
        fingers = self._fingers_on_bracket()
        held = len(fingers) >= 2 and all(fingers)
        sq, off = env.bracket_alignment()
        stopped = sorted(a for a, until in w._reflex_until.items() if w.time < until)
        self._measured.update(held=held, square=float(sq), offset=float(off))
        checks.update({
            "holding": dict(fingers_on_bracket=fingers, ok=_ok(held)),
            "square": dict(angle_error_deg=float(sq), limit_deg=1., ok=_ok(held and abs(sq) <= 1.)),
            "position": dict(offset_mm=float(off), limit_mm=FIT_MM, ok=_ok(held and off <= FIT_MM)),
            "no_protective_stop": dict(stopped_arms=stopped, ok=_ok(not stopped)),
        })
        return checks

    def _progress_hold_and_present(self):
        m = self._measured
        return hold_present_progress(self._progress_position_torch()["progress"], m["held"], m["square"], m["offset"])

    # ------------------------------------------------------------------ set-up-and-weld (A14)
    def judge_set_up_and_weld(self, rows, judged=None, flags=None, wfs_m_min=0., voltage_v=0.):
        """set-up-and-weld (A14): every fillet-2f check on the weld, unchanged, and the source at the procedure at the
        end (12.7 +- 0.3 m/min, 29 +- 0.5 V: set-up-gun's tolerances). Progress: fillet_progress net of the tacked start,
        as fillet-2f's (the weld is the job; a weld laid at the short-circuit job's settings is judged as what it is)."""
        target = PR.v(PR.SCHEDULES)[SETUP_PROCEDURE]
        checks = self._eval_fillet_2f(rows)
        diagnostics = dict(self.diagnostics)
        checks["source_at_procedure"] = dict(
            wire_feed_m_min=wfs_m_min, voltage_v=voltage_v, target=dict(wire_feed_m_min=target["wfs_m_min"],
                                                                        voltage_v=target["voltage_v"]),
            tolerance=dict(wire_feed_m_min=SETUP_WFS_TOL, voltage_v=SETUP_VOLTS_TOL),
            ok=_ok(abs(wfs_m_min - target["wfs_m_min"]) <= SETUP_WFS_TOL + 1e-9
                   and abs(voltage_v - target["voltage_v"]) <= SETUP_VOLTS_TOL + 1e-9))
        checks = self._named(checks)
        reward = fillet_progress(rows, self.start_raw_progress, self._last_craters)
        return self._scored(all(c["pass"] for c in checks.values()), checks, reward, judged=judged or {},
                            flags=flags or {}, diagnostics=diagnostics)

    # ------------------------------------------------------------------ shared measurements
    def _wire(self):
        """Contact tip, wire tip (end of the current stick-out) and axis in the weld frame."""
        w = self.env.world
        tip, axis, _ = w.torch_pose()
        o, R = w.weld_frame()
        tip_w, ax_w = R.T @ (tip - o), R.T @ axis
        ls = w.process.src.st.l_s
        return tip_w, tip_w + ax_w*ls, ax_w

    def _gun_contacts(self):
        """Gun shapes touching anything that is not the robot (MuJoCo contacts with load)."""
        w = self.env.world
        return w.gun_contacts()

    # ------------------------------------------------------------------ A0
    def _eval_position_torch(self):
        env, w = self.env, self.env.world
        tip, wire_tip, ax = self._wire()
        # distance of the wire tip from the root line (x axis) and its position along the seam
        root_dist = float(math.hypot(wire_tip[1], wire_tip[2]))
        work = math.degrees(math.atan2(-ax[2], -ax[1]))           # from the base plate, in the y-z plane
        travel = math.degrees(math.asin(np.clip(ax[0], -1, 1)))    # + push toward +x (the weld runs +x)
        from .arc import attach
        _, ctwd, _ = attach(w.process.g, w.process.s.mat, w.process.s.body, tip, ax)
        settled = env.tcp_motion_mm(1.0)
        # the way the torch was presented (no contact, no arc, held still) counts only once the wire is at the seam's
        # start (otherwise a torch left at its home pose passes these three)
        presented = root_dist <= PRESENTED_M and -PRESENTED_M <= wire_tip[0] <= 10e-3 + PRESENTED_M
        checks = {
            "wire_on_root": dict(value_mm=root_dist*1e3, limit_mm=1.0, ok=_ok(root_dist <= 1e-3)),
            "at_seam_start": dict(value_mm=float(wire_tip[0]*1e3), range_mm=[0., 10.],
                                  ok=_ok(-1e-3 <= wire_tip[0] <= 10e-3)),
            # the angles too: the robots' home pose already holds the torch at the procedure's angles, so an untouched
            # start state would pass these two and score 0.25 (E22)
            "work_angle": dict(value_deg=work, range_deg=[40., 50.], ok=_ok(presented and 40. <= work <= 50.)),
            "travel_angle_push": dict(value_deg=travel, range_deg=[0., 15.],
                                      ok=_ok(presented and 0. <= travel <= 15.)),
            "ctwd": dict(value_mm=ctwd*1e3 if math.isfinite(ctwd) else None, range_mm=[17., 21.],
                         ok=_ok(math.isfinite(ctwd) and 17e-3 <= ctwd <= 21e-3)),
            # the gun's hardware (nozzle, contact tip, body) touches nothing; the wire may touch (touch sensing with the
            # wire is how a robot finds a seam)
            "no_contact": dict(contacts=self._gun_contacts(), wire_touched=bool(env.history["wire_touch"]),
                               gun_contact_ticks=env.history["gun_contact_ticks"],
                               ok=_ok(presented and not self._gun_contacts() and env.history["gun_contact_ticks"] == 0)),
            "no_arc": dict(arc_s=env.history["arc_s"], ok=_ok(presented and env.history["arc_s"] == 0.)),
            "settled": dict(tcp_motion_1s_mm=settled, limit_mm=.5,
                            ok=_ok(presented and settled is not None and settled < .5)),
        }
        self._measured = dict(wire_tip=wire_tip, work=work, travel=travel, ctwd=ctwd if math.isfinite(ctwd) else None,
                              settled=settled, faulted=bool(env.history["gun_contact_ticks"] or env.history["arc_s"]))
        return checks

    # ------------------------------------------------------------------ A1
    def _bond_segments(self):
        p = self.env.world.process
        rows = slice_bonds(p, 0, p.g.nx)
        g = p.g
        segs, cur = [], None
        for i, th, ky, solid in rows:
            if not (solid and th > 0):
                continue
            if cur is not None and i == cur[1] + 1:
                cur[1] = i
            else:
                cur = [i, i]
                segs.append(cur)
        return [dict(x0_mm=float(g.xe[a]*1e3), x1_mm=float(g.xe[b + 1]*1e3), length_mm=float((g.xe[b + 1] - g.xe[a])*1e3))
                for a, b in segs]

    def _tacks(self):
        """Tacks: bonded segments less than TACK_MERGE_GAP apart are one tack (a tack whose cold start left an unfused
        stretch is still one tack with incomplete fusion, not two); a tack's length is its fused length (the sum)."""
        tacks = []
        for sgm in self._bond_segments():
            if tacks and sgm["x0_mm"] - tacks[-1]["x1_mm"] < TACK_MERGE_GAP*1e3:
                t = tacks[-1]
                t["x1_mm"] = sgm["x1_mm"]
                t["fused_mm"] += sgm["length_mm"]
                t["segments"].append(sgm)
            else:
                tacks.append(dict(x0_mm=sgm["x0_mm"], x1_mm=sgm["x1_mm"], fused_mm=sgm["length_mm"], segments=[sgm]))
        return tacks

    def _weld_runs(self, rows):
        """Contiguous runs of sections holding weld metal inside the joint (0 <= x <= L_BRACKET), each a list of its
        sections: a tack, a fillet, or a film of metal lying on the joint."""
        runs, cur = [], []
        for r in rows:
            if 0. <= r["x_m"] <= L_BRACKET and r["weld_area_m2"] > 0:
                cur.append(r)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        return runs

    @staticmethod
    def _fused(r):
        """A section an etched macro shows fused: the root melted and both members entered by at least one cell."""
        return bool(r["root_fused"] and r["penetration_base_m"] >= 2.5e-4 and r["penetration_bracket_m"] >= 2.5e-4)

    def _tack_quality(self):
        """D1.1:2000 5.18.2 holds tack welds to "the same quality requirements as the final welds": every run of weld
        metal on the joint is fused to both members along its length (Table 6.1(2), weld/base-metal fusion: "Thorough
        fusion shall exist ... between weld metal and base metal"), its profile has no overlap and no more convexity
        than Fig 5.4 allows, and it is not larger than the 1/4 in fillet it becomes part of. A film of weld metal lying
        unfused on the root is not a tack, and a tack that carries one is a tack with incomplete fusion."""
        p = self.env.world.process
        rows = SC.weld_report(p, (-2e-3, L_BRACKET + 2e-3))
        out = []
        for run in self._weld_runs(rows):
            inner = run[TACK_END_SECTIONS:len(run) - TACK_END_SECTIONS]
            unfused = [r for r in inner if not self._fused(r)]
            fused = [r for r in run if self._fused(r)]
            ov = max((max(r["overlap_base_m"], r["overlap_bracket_m"]) for r in run), default=0.)
            conv_bad = [r for r in fused if r["convexity_m"] > (3.2e-3 if r["face_width_m"] > 7.9e-3 else 1.6e-3)]
            out.append(dict(x0_mm=round((run[0]["x_m"] - run[0]["dx_m"]/2)*1e3, 2),
                            x1_mm=round((run[-1]["x_m"] + run[-1]["dx_m"]/2)*1e3, 2),
                            fused_mm=round(sum(r["dx_m"] for r in fused)*1e3, 2),
                            unfused_mm=round(sum(r["dx_m"] for r in unfused)*1e3, 2),
                            unfused_x_mm=[round(r["x_m"]*1e3, 2) for r in unfused][:40],
                            overlap_max_mm=round(ov*1e3, 3),
                            convexity_max_mm=round(max((r["convexity_m"] for r in fused), default=0.)*1e3, 3),
                            convexity_over_limit_sections=len(conv_bad),
                            size_max_mm=round(max((r["size_m"] for r in fused), default=0.)*1e3, 3),
                            porosity_max=max((r["porosity"] for r in run), default=0.),
                            undercut_max_mm=round(max((max(r["undercut_base_m"], r["undercut_bracket_m"]) for r in run),
                                                      default=0.)*1e3, 3),
                            undercut_over_1mm_mm=round(sum(r["dx_m"] for r in run
                                                           if max(r["undercut_base_m"], r["undercut_bracket_m"]) > 1e-3)
                                                       *1e3, 2)))
        return out

    def _eval_tack(self):
        env, w = self.env, self.env.world
        segs = self._tacks()
        near = [s for s in segs if s["x0_mm"] < 40.]
        far = [s for s in segs if s["x1_mm"] > 110.]
        ok_len = lambda s: 12.7 <= s["fused_mm"] <= 25.4
        sq, off = env.bracket_alignment()
        tmax = env.joint_max_temperature()
        quality = self._tack_quality()
        checks = {
            "two_tacks": dict(tacks=segs, ok=_ok(len(segs) == 2 and len(near) == 1 and len(far) == 1)),
            "tack_lengths": dict(range_mm=[12.7, 25.4], ok=_ok(len(segs) == 2 and all(ok_len(s) for s in segs))),
            # bonded now (a tack that cracked and was re-welded counts; its break stays in the bond's events)
            "bonded": dict(active=w.bond.active, broke_during_episode=w.bond.broken, capacity_n=w.bond.capacity_n,
                           ok=_ok(w.bond.active)),
            "square": dict(angle_error_deg=sq, limit_deg=1., ok=_ok(w.bond.active and abs(sq) <= 1.)),
            # within the fit-up the thermal grid describes (the bond's own guard, bonds.FIT_MM; a hand closing on the
            # bracket moves it 0.54 mm)
            "position": dict(offset_mm=off, limit_mm=FIT_MM, ok=_ok(w.bond.active and off <= FIT_MM)),
            # conditional on the bond: a squareness, a position, a released hand and a cool joint are the qualities of
            # a tacked bracket (otherwise an untouched loose bracket passes these four and scores 0.57)
            "hand_off": dict(touching=env.hand_touches_bracket(),
                             ok=_ok(w.bond.active and not env.hand_touches_bracket())),
            "cooled": dict(max_joint_temperature_c=tmax, limit_c=100., ok=_ok(w.bond.active and tmax < 100.)),
            # the tacks as a final weld (D1.1:2000 5.18.2): conditional on the bond like the four above (no tack,
            # nothing to judge). Every run of weld metal on the joint counts, so a film lying unfused beside a tack
            # fails it
            "tack_fusion": dict(runs=quality, end_sections_allowed=TACK_END_SECTIONS,
                                ok=_ok(w.bond.active and quality and all(q["unfused_mm"] == 0. for q in quality))),
            "tack_profile": dict(overlap_limit_mm=OVERLAP_RESOLUTION*1e3, max_size_mm=TACK_MAX_SIZE*1e3,
                                 porosity_limit=.015, undercut_limit_mm=UNDERCUT_MAX*1e3,
                                 undercut_over_1mm_limit_mm=UNDERCUT_OVER_1MM_LENGTH*1e3,
                                 ok=_ok(w.bond.active and quality and all(
                                     q["overlap_max_mm"] <= OVERLAP_RESOLUTION*1e3 + 1e-6
                                     and q["convexity_over_limit_sections"] == 0
                                     and q["size_max_mm"] <= TACK_MAX_SIZE*1e3 + 1e-6
                                     and q["porosity_max"] <= .015 and q["undercut_max_mm"] <= UNDERCUT_MAX*1e3 + 1e-6
                                     for q in quality)
                                        and sum(q["undercut_over_1mm_mm"] for q in quality)
                                        <= UNDERCUT_OVER_1MM_LENGTH*1e3 + 1e-6)),
        }
        self._measured = dict(tacks=segs, bonded=bool(w.bond.active), square=sq, offset=off,
                              touching=bool(env.hand_touches_bracket()))
        return checks

    def _progress_tack(self):
        """tack_progress on this state; the quality share is over every run of weld metal on the joint (_weld_runs)."""
        m = self._measured
        runs = [r for run in self._weld_runs(SC.weld_report(self.env.world.process, X_READ)) for r in run]
        share = sum(1 for r in runs if self._fused(r) and not _section_defects(r))/len(runs) if runs else 0.
        return tack_progress(m["tacks"], m["bonded"], m["square"], m["offset"], m["touching"], share)

    # ------------------------------------------------------------------ A2
    @staticmethod
    def _beads(rows):
        """Beads: contiguous sections holding weld metal, each cut to its fillet (first to last section with size > 0).
        A film lying ahead of a bead's front is not its fillet; a hole inside a fillet is part of it (a size-0 section
        at a stop would split the bead and hide the crater's bottom)."""
        runs, cur = [], []
        for r in rows:
            if r["weld_area_m2"] > 0:
                cur.append(r)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        out = []
        for run in runs:
            idx = [i for i, r in enumerate(run) if r["size_m"] > 0]
            if idx:
                out.append(run[idx[0]:idx[-1] + 1])
        return out

    @staticmethod
    def _episodes(bead):
        """Index ranges [a, b) of a bead's freeze episodes: the metal one moving pool left, adjacent sections inside the
        joint frozen within EPISODE_GAP_S of each other (a section beyond a bracket end never splits one). A section
        with no freeze time (still freezing) belongs to none."""
        eps, cur = [], None
        for i, r in enumerate(bead):
            t = r.get("freeze_t")
            if t is None:
                if cur:
                    eps.append(cur)
                cur = None
            elif cur and (abs(t - bead[i - 1]["freeze_t"]) <= EPISODE_GAP_S
                          or not (0. <= r["x_m"] <= L_BRACKET and 0. <= bead[i - 1]["x_m"] <= L_BRACKET)):
                cur[1] = i + 1
            else:
                if cur:
                    eps.append(cur)
                cur = [i, i + 1]
        if cur:
            eps.append(cur)
        return eps

    @staticmethod
    def _craters(rows):
        """The weld's craters, wherever it stopped (D1.1:2000 Table 6.1(3): "All craters shall be filled to provide the
        specified weld size"). A pool's last liquid freezes short (pool.py: the crater), and every section records when it
        last froze (freeze_t; a section an arc melts again freezes again). A stop is a section that froze last within
        STOP_RADIUS among the sections of its own freeze episode (_episodes): one at the end of a single pass whichever
        way it ran, one at each stop of several runs, none at a start; a later pass's metal, which froze in its own
        episode, cannot hide an earlier crater it did not remelt, unless it restarted within EPISODE_GAP_S and so merged
        with it. An episode's end where a later pass's metal begins is where that pass stopped remelting, not a stop
        (judging such ends would judge, at full size, the old metal beside every tie-in, which the 10 % undersize rule
        allows). Known false positive: a later pool whose metal reaches an earlier stop from the
        side that pass came from, touching or remelting its last-frozen section, leaves the rest of that crater, toward
        the pass's front, judged only by the 10 % undersize rule. A section at the edge of the sections read, with freeze
        times rising toward it, is not a stop: the pass ran on beyond what was read. The crater judged is
        the stop's own episode within CRATER_LEN / 2 of it, run on to the nearer end of that episode where that lies
        within FRONT_REACH (a bead end or older metal), inside the joint, less the bead's own end face (its last
        CRATER_END_FACE) where the bead ends. Rows without freeze times (constructed): both ends of each bead."""
        later = lambda q, r: q.get("freeze_t") is None or q["freeze_t"] > r["freeze_t"]
        edge = (rows[0]["x_m"], rows[-1]["x_m"]) if rows else ()
        craters = []
        for bead in Evaluator._beads(rows):
            lo, hi = bead[0]["x_m"] - bead[0]["dx_m"]/2, bead[-1]["x_m"] + bead[-1]["dx_m"]/2
            eps = Evaluator._episodes(bead)
            found = []
            for a, b in eps:
                ep = bead[a:b]
                stops = []
                for j, r in enumerate(ep):
                    if any(q["freeze_t"] > r["freeze_t"] for q in ep if abs(q["x_m"] - r["x_m"]) <= STOP_RADIUS + 1e-9):
                        continue
                    if (j == 0 and a > 0 and later(bead[a - 1], r)) or (j == len(ep) - 1 and b < len(bead)
                                                                         and later(bead[b], r)):
                        continue
                    if r["x_m"] in edge:              # the last section read: the pass ran on beyond it
                        craters.append(dict(rows=[], how="ran on past the sections read", stop_mm=[round(r["x_m"]*1e3, 2)],
                                            window_mm=None, episode_mm=[round(ep[0]["x_m"]*1e3, 2),
                                                                        round(ep[-1]["x_m"]*1e3, 2)],
                                            bead_mm=[round(lo*1e3, 2), round(hi*1e3, 2)], min_size_mm=None,
                                            at_x_mm=None, ok=True))
                        continue
                    stops.append(r)
                found.append((ep, stops, "last to freeze"))
            if not eps:
                found.append((bead, [bead[0], bead[-1]], "bead end (no freeze times)"))
            for ep, stops, how in found:
                groups = []                          # stops within CRATER_LEN of each other are one crater (tied times)
                for st in sorted(stops, key=lambda r_: r_["x_m"]):
                    if groups and st["x_m"] - groups[-1][-1]["x_m"] <= CRATER_LEN + 1e-9:
                        groups[-1].append(st)
                    else:
                        groups.append([st])
                for g in groups:
                    x0, x1 = g[0]["x_m"], g[-1]["x_m"]
                    e0, e1 = ep[0]["x_m"] - ep[0]["dx_m"]/2, ep[-1]["x_m"] + ep[-1]["dx_m"]/2
                    # the pass's front: toward the nearer end of its own episode within FRONT_REACH, whether a bead end
                    # or older metal; the end face is left out only where the bead ends
                    up = e1 - x1 <= x0 - e0 and e1 - x1 <= FRONT_REACH + 1e-9
                    down = x0 - e0 < e1 - x1 and x0 - e0 <= FRONT_REACH + 1e-9
                    near = lambda r: (any(abs(r["x_m"] - st["x_m"]) <= CRATER_LEN/2 + 1e-9 for st in g)
                                      or (up and r["x_m"] >= x1) or (down and r["x_m"] <= x0))
                    sel = [r for r in ep if near(r) and 0. <= r["x_m"] <= L_BRACKET
                           and r["x_m"] - lo > CRATER_END_FACE and hi - r["x_m"] > CRATER_END_FACE]
                    low = min(sel, key=lambda r_: r_["size_m"]) if sel else None
                    craters.append(dict(rows=sel, how=how, stop_mm=[round(st["x_m"]*1e3, 2) for st in g],
                                        window_mm=[round(sel[0]["x_m"]*1e3, 2), round(sel[-1]["x_m"]*1e3, 2)] if sel else None,
                                        episode_mm=[round(ep[0]["x_m"]*1e3, 2), round(ep[-1]["x_m"]*1e3, 2)],
                                        bead_mm=[round(lo*1e3, 2), round(hi*1e3, 2)],
                                        min_size_mm=round(low["size_m"]*1e3, 4) if low else None,
                                        at_x_mm=round(low["x_m"]*1e3, 2) if low else None,
                                        ok=bool(low is None or low["size_m"] >= NOMINAL_LEG - 1e-9)))
        return craters

    def _eval_fillet_2f(self, rows=None):
        """The acceptance checks (all must pass) on the sections `rows` (sections.weld_report over X_READ; read from the
        world's weld process when None). Where each is judged: the span the weld must cover (END_ALLOWANCE in from each
        end) for size and throat; every section of the joint for undercut, and every section holding weld metal for
        porosity and convexity; the craters wherever the weld stopped; overlap over the span and every crater. Within
        END_ALLOWANCE of an end and outside every crater (where a weld starts), overlap up to START_OVERLAP_MAX (two
        cells) is accepted: a weld started on cold steel in this model leaves at least that cold lap with every start
        technique tried (E26: nine techniques; the two starts shown to clear the tacks, 0.5 mm; every other saved start
        1.5-12.25 mm), a start geometry not sized against reality. What lies there is reported in the
        evaluation's diagnostics (start_zone)."""
        from .env import TACKS
        if rows is None:
            rows = SC.weld_report(self.env.world.process, X_READ)
        joint = [r for r in rows if 0. <= r["x_m"] <= L_BRACKET]
        inside = _required(rows)
        metal = [r for r in joint if r["weld_area_m2"] > 0]
        s = SC.summarize(rows)
        L = sum(r["dx_m"] for r in inside) or 1.
        size = np.array([r["size_m"] for r in inside]) if inside else np.zeros(1)
        legs = np.array([min(r["leg_vertical_m"], r["leg_horizontal_m"]) for r in inside]) if inside else np.zeros(1)
        dxs = np.array([r["dx_m"] for r in inside]) if inside else np.ones(1)
        # D1.1:2000 Table 6.1(6), undersized welds: "The size of a fillet weld in any continuous weld may be less
        # than the specified nominal size (L) ... the undersize portion ... shall not exceed 10% of the weld length";
        # the size is "the leg lengths of the largest isosceles right triangle that can be inscribed within the fillet
        # weld cross section" (D1.1:2000 Annex B, terms) [read]: it includes a concave face's throat
        frac_ok = float(np.sum(dxs*(size >= NOMINAL_LEG - 1e-9))/L)
        frac_legs = float(np.sum(dxs*(legs >= NOMINAL_LEG - 1e-9))/L)
        on_cells = sum(1 for r in inside if r["weld_area_m2"] > 0 and r.get("geometry") != "profile")
        # sections whose cells hold metal the surface does not explain, or lack metal it claims (sections.py keeps the
        # cells' metal there): a measure of how far the two representations of the weld agree
        disagree = sum(1 for r in inside if r.get("geometry") == "profile"
                       and (r["profile_extra_cells"] or r["profile_missing_cells"]))
        # the leg up the bracket's face ends at the fused cell's edge (the pool does not carry the toe up the face past
        # it), so it steps in 0.25 mm and "size >= 6.35 mm" asks it for 6.50 mm
        v_on_edges = sum(1 for r in inside if r["leg_vertical_m"] > 0
                         and abs(r["leg_vertical_m"]/2.5e-4 - round(r["leg_vertical_m"]/2.5e-4)) < 1e-6)
        craters = self._last_craters = self._craters(rows)
        judged = [c for c in craters if c["rows"]]
        in_crater = {_key(r) for c in judged for r in c["rows"]}
        profiled = [r for r in joint if _span(r) or _key(r) in in_crater]
        start_zone = [r for r in joint if not _span(r) and _key(r) not in in_crater and r["weld_area_m2"] > 0]
        conv_bad = [r for r in metal if r["convexity_m"] > _convexity_limit(r) + CELL_TOL]
        ov_bad = [r for r in profiled if _overlap(r) > OVERLAP_RESOLUTION + CELL_TOL] + \
            [r for r in start_zone if _overlap(r) > START_OVERLAP_MAX + CELL_TOL]
        uc = [_undercut(r) for r in joint]
        uc_over = sum(r["dx_m"] for r, u in zip(joint, uc) if u > UNDERCUT_LIMIT + CELL_TOL)
        # macro sections: the quarter points and one at each end of the fillet, the ends placed off the start state's
        # tacks (midway between the end allowance and the tack's arc path less 1 mm of its fusion: 6 and 144 mm), so
        # the root fusion they show is the fillet's own (at 10 and 140 mm they would sit on the tacks, which were fused
        # before the fillet began)
        fusion_x = [.5*(END_ALLOWANCE + TACKS[0][0] - 1e-3), .25*L_BRACKET, .5*L_BRACKET, .75*L_BRACKET,
                    .5*(TACKS[-1][1] + 1e-3 + L_BRACKET - END_ALLOWANCE)]
        fus = []
        for xq in fusion_x:
            r = min(rows, key=lambda r_: abs(r_["x_m"] - xq)) if rows else None
            fus.append(dict(x_mm=xq*1e3, root_fused=bool(r and r["root_fused"]),
                            pen_base_mm=r["penetration_base_m"]*1e3 if r else 0.,
                            pen_bracket_mm=r["penetration_bracket_m"]*1e3 if r else 0.,
                            ok=bool(r and _fused_section(r))))
        # the weld's quality checks judge a weld: they pass only where one covers the inspected seam (an untouched
        # tacked joint has no porosity, undercut or convexity to fail, and would score 0.5 without this gate)
        covered = float(np.sum(dxs*(size > 0))/L) if inside else 0.
        weld_present = covered >= COVERAGE_FOR_QUALITY
        crater_ok = bool(judged or any(c["how"].startswith("ran on") for c in craters)) and all(c["ok"] for c in judged)
        worst = min(judged, key=lambda c: c["min_size_mm"]) if judged else None
        liquid = sum(r["liquid"] for r in rows)
        freezing = sum(1 for r in rows if r["weld_area_m2"] > 0 and r.get("freeze_t") is None)
        porosity_max = max((r["porosity"] for r in metal), default=0.)
        self.diagnostics = dict(
            leg_difference=dict(max_mm=max((abs(r["leg_vertical_m"] - r["leg_horizontal_m"]) for r in inside),
                                           default=0.)*1e3,
                                note="not an acceptance criterion (D1.1 accepts unequal legs when "
                                     "the smaller meets the size); ASME QW-184 holds it to <= 3.2 mm"),
            start_zone=dict(
                sections_mm=[round(r["x_m"]*1e3, 2) for r in start_zone],
                overlap_max_mm=max((_overlap(r) for r in start_zone), default=0.)*1e3,
                overlap_limit_mm=START_OVERLAP_MAX*1e3,
                overlap_over_one_cell_mm=sum(r["dx_m"] for r in start_zone
                                             if _overlap(r) > OVERLAP_RESOLUTION + CELL_TOL)*1e3,
                note=f"weld metal within {END_ALLOWANCE*1e3:g} mm of an end and outside every crater (a start on cold "
                     f"steel): overlap accepted up to {START_OVERLAP_MAX*1e3:g} mm, not one cell (_eval_fillet_2f)"))
        checks = {
            "continuous_length": dict(from_mm=s.get("continuous_from_m", 0)*1e3 if s.get("continuous_from_m") is not None else None,
                                      to_mm=s.get("continuous_to_m", 0)*1e3 if s.get("continuous_to_m") is not None else None,
                                      # slice edges carry float residue: 5.0 mm short of the low end read
                                      # 0.005000000000000011 and failed
                                      ok=_ok(s.get("continuous_from_m", 1.) <= END_ALLOWANCE + 1e-9
                                               and s.get("continuous_to_m", 0.) >= L_BRACKET - END_ALLOWANCE - 1e-9)),
            # sizes are measured on the pool's free surface (sections.profile_section: legs at the toes, 0.05 mm
            # sampling; size exact on the surface). A section with no surface profile is read on the 0.25 mm cells,
            # where a flat 45 deg face reads ">= 6.35 mm" only from a true leg of 6.50 mm (a leg is read at the edge of
            # the last weld cell whose centre lies under the surface): the count of such sections is printed. The
            # vertical leg steps in 0.25 mm wherever it is read (above)
            "leg_size": dict(fraction_size_ge_6_35mm=frac_ok, fraction_legs_ge_6_35mm=frac_legs,
                             min_size_mm=float(size.min()*1e3), sections_read_on_cells=on_cells,
                             sections_cells_disagree_with_surface=disagree,
                             resolution_mm=dict(vertical_leg=.25, horizontal_leg=.05 if not on_cells else .25),
                             vertical_legs_on_cell_edges=v_on_edges, sections=len(inside),
                             # exactly 10 % undersize read 0.8999999999999999
                             ok=_ok(frac_ok >= .9 - 1e-9 and size.min() >= NOMINAL_LEG - UNDERSIZE)),
            "throat": dict(median_mm=float(np.median([r["throat_m"] for r in inside])*1e3) if inside else 0.,
                           limit_mm=.707*NOMINAL_LEG*1e3,
                           ok=_ok(inside and np.median([r["throat_m"] for r in inside]) >= .707*NOMINAL_LEG)),
            "convexity": dict(max_mm=max((r["convexity_m"] for r in metal), default=0.)*1e3,
                              sections_over_limit=len(conv_bad), weld_coverage=covered,
                              ok=_ok(weld_present and not conv_bad)),
            "undercut": dict(max_mm=max(uc, default=0.)*1e3, over_1mm_length_mm=uc_over*1e3,
                             limit_mm=UNDERCUT_MAX*1e3, over_1mm_limit_mm=UNDERCUT_OVER_1MM_LENGTH*1e3,
                             weld_coverage=covered, ok=_ok(weld_present and max(uc, default=0.) <= UNDERCUT_MAX + CELL_TOL
                                                           and uc_over <= UNDERCUT_OVER_1MM_LENGTH + CELL_TOL)),
            # overlap (cold lap): weld metal lying beyond the fusion boundary on parent metal that never melted is an
            # unacceptable profile in AWS D1.1 (Fig 5.4); one cell (0.25 mm) is the sections' resolution
            "overlap": dict(max_mm=max((_overlap(r) for r in profiled), default=0.)*1e3, limit_mm=OVERLAP_RESOLUTION*1e3,
                            start_zone_max_mm=max((_overlap(r) for r in start_zone), default=0.)*1e3,
                            start_zone_limit_mm=START_OVERLAP_MAX*1e3,
                            length_over_limit_mm=sum(r["dx_m"] for r in ov_bad)*1e3, weld_coverage=covered,
                            ok=_ok(weld_present and not ov_bad)),
            "fusion_sections": dict(sections=fus, ok=_ok(all(f["ok"] for f in fus))),
            "porosity": dict(max_fraction=porosity_max, limit=POROSITY_MAX, weld_coverage=covered,
                             ok=_ok(weld_present and porosity_max <= POROSITY_MAX)),
            # D1.1:2000 Table 6.1(3): "All craters shall be filled to provide the specified weld size": the specified
            # 1/4 in, not the undersize floor, at every place the weld stopped
            # (_craters)
            "crater_filled": dict(min_size_mm=worst["min_size_mm"] if worst else 0., at_x_mm=worst["at_x_mm"] if worst else None,
                                  craters=[{k: v for k, v in c.items() if k != "rows"} for c in craters],
                                  limit_mm=NOMINAL_LEG*1e3, ok=_ok(crater_ok)),
            # solid in its cells and frozen in the pool (weld metal with no freeze time is still freezing: its surface
            # and its crater are not final)
            "solidified": dict(liquid_sections=liquid, freezing_sections=freezing, weld_coverage=covered,
                               ok=_ok(weld_present and liquid == 0 and freezing == 0)),
        }
        return checks

    def judge_fillet(self, rows, judged=None, flags=None):
        """fillet-2f's evaluation of one state's sections: success (every check), the progress reward and the score."""
        checks = {k: {("pass" if kk == "ok" else kk): vv for kk, vv in c.items()}
                  for k, c in self._eval_fillet_2f(rows).items()}
        passed = [c["pass"] for c in checks.values()]
        success = bool(passed) and all(passed)
        reward = fillet_progress(rows, self.start_raw_progress, self._last_craters)
        reward.update(contract=REWARD_CONTRACT, score_rule="1 if success else 0.9 x progress",
                      mean_of_checks=float(np.mean(passed)))
        return dict(success=success, score=1. if success else FAIL_SCORE_SCALE*reward["progress"], checks=checks,
                    progress=reward["progress"], reward=reward, judged=judged or {}, flags=flags or {},
                    diagnostics=dict(self.diagnostics))

    def _evaluate_fillet_2f(self, settle=False):
        return self.judge_fillet(*self._final_rows(settle))

    def _final_rows(self, settle=False):
        """The weld's sections to judge (fillet-2f's reading, shared by every weld task), what was done
        to read them (judged: a still-freezing weld frozen hands-off on a copy for the on-demand evaluation; the time
        against the task's limit) and flags."""
        w = self.env.world
        p = w.process
        rows = SC.weld_report(p, X_READ)
        # still freezing: a section liquid in its cells (the "solidified" check), or weld metal whose slice the pool
        # has not yet frozen (no freeze time: its crater cannot be located yet)
        judged = dict(liquid_sections=int(sum(r["liquid"] for r in rows)),
                      freezing_sections=int(sum(1 for r in rows if r["weld_area_m2"] > 0 and r.get("freeze_t") is None)),
                      hands_off_freeze=None, freeze_pending=False)
        if judged["liquid_sections"] or judged["freezing_sections"]:
            st = p.src.st
            if st.on_cmd or st.arc:
                judged["hands_off_freeze"] = dict(applied=False, reason="the arc is on: the weld is not finished")
            elif settle:
                tip, axis, _ = w.torch_pose()
                o, R = w.weld_frame()
                q, sim_s = freeze_hands_off(p, R.T @ (tip - o), R.T @ axis, self.env.config.rigid_dt)
                rows = SC.weld_report(q, X_READ)
                q.__dict__.clear()
                judged["hands_off_freeze"] = dict(
                    applied=True, simulated_s=round(sim_s, 3), liquid_sections_after=int(sum(r["liquid"] for r in rows)),
                    freezing_sections_after=int(sum(1 for r in rows if r["weld_area_m2"] > 0 and r.get("freeze_t") is None)))
            else:
                judged["freeze_pending"] = True      # an on-demand evaluation (settle) judges it as it freezes
        t = getattr(self.env, "sim_time", None)
        judged.update(time_s=t, within_time_limit=None if t is None else
                      bool(t <= TIME_LIMIT_S[self.env.config.task] + 1e-9))
        flags = dict(bracket_detached=bool(getattr(p, "detached", None)),
                     bond_broken=bool(getattr(getattr(w, "bond", None), "broken", False)))
        return rows, judged, flags

    # ------------------------------------------------------------------ the added weld tasks
    @staticmethod
    def _named(checks):
        return {k: {("pass" if kk == "ok" else kk): vv for kk, vv in c.items()} for k, c in checks.items()}

    def _half_reading(self, rows, new):
        """half-fillet's craters judged (every one touching the half, and every one of the metal the episode laid,
        wherever it lies: the second tack's, untouched, is the fixture's), and the raw progress: the half's fillet credit
        less, per length of the half, the judged craters' deficit beyond it and the start zone's rejected metal (as
        fillet_progress charges what success judges outside its span)."""
        lo, hi = HALF_SPAN
        beyond = _undercut_beyond_allowance(rows)
        craters = [c for c in self._craters(rows) if c["rows"] and any(_in(r, lo, hi) or new(r) for r in c["rows"])]
        in_crater = {_key(r) for c in craters for r in c["rows"]}
        L = sum(r["dx_m"] for r in rows if _in(r, lo, hi))
        charge = 0.
        for r in rows:
            if not 0. <= r["x_m"] <= L_BRACKET or _in(r, lo, hi) or r["weld_area_m2"] <= 0:
                continue
            if _key(r) in in_crater:
                charge += r["dx_m"]*(1. - section_credit(r, beyond)[0])
            elif r["x_m"] < lo and _section_defects(r, beyond, START_OVERLAP_MAX):
                charge += r["dx_m"]
        raw = window_credit(rows, lo, hi, beyond) - (charge/L if L > 0 else 0.)
        return craters, in_crater, beyond, raw

    def judge_half_fillet(self, rows, judged=None, flags=None):
        """half-fillet (A9) on one state's sections. The half (HALF_SPAN) as fillet-2f judges its span; beyond it the
        judged craters' metal at one cell of overlap, the start zone's at START_OVERLAP_MAX, the rest of the metal the
        episode laid for convexity and porosity (fillet-2f judges both on all weld metal); the craters (_half_reading)
        filled to 6.35 mm; undercut on the whole joint; HALF_LEFT_FREE bare; solid. Progress: (raw - raw0) / (1 -
        raw0) less FREE_ZONE_PENALTY x the share of HALF_LEFT_FREE holding metal, clipped to [0, 1]; raw0 the tacked
        start's."""
        lo, hi = HALF_SPAN
        new = episode_metal(self.start_rows)
        craters, in_crater, beyond, raw = self._half_reading(rows, new)
        checks = _window_checks(rows, lo, hi, HALF_FUSION_X, beyond)
        outside = []
        for r in rows:
            if not 0. <= r["x_m"] <= L_BRACKET or _in(r, lo, hi) or r["weld_area_m2"] <= 0:
                continue
            if _key(r) in in_crater:
                bad = _section_defects(r, beyond)
            elif r["x_m"] < lo:
                bad = _section_defects(r, beyond, START_OVERLAP_MAX)
            elif new(r):
                bad = [d for d in _section_defects(r, beyond) if d in ("convexity", "porosity")]
            else:
                bad = []
            if bad:
                outside.append((round(r["x_m"]*1e3, 2), bad))
        worst = min(craters, key=lambda c: c["min_size_mm"]) if craters else None
        checks["sound_beyond_half"] = dict(rejected_sections=outside[:40],
                                           start_zone_overlap_limit_mm=START_OVERLAP_MAX*1e3, ok=_ok(not outside))
        checks["crater_filled"] = dict(min_size_mm=worst["min_size_mm"] if worst else 0.,
                                       craters=[{k: v for k, v in c.items() if k != "rows"} for c in craters],
                                       limit_mm=NOMINAL_LEG*1e3, ok=_ok(craters and all(c["ok"] for c in craters)))
        checks.update(_joint_checks(rows, HALF_LEFT_FREE, "left_half_unwelded"))
        checks = self._named(checks)
        raw0 = self._half_reading(self.start_rows, episode_metal(self.start_rows))[3] if self.start_rows else 0.
        share = free_share(rows, *HALF_LEFT_FREE)
        net = (raw - raw0)/(1. - raw0) if raw0 < 1. else float(raw >= 1.)
        reward = dict(progress=float(np.clip(net - FREE_ZONE_PENALTY*share, 0., 1.)), raw=raw, start_raw=raw0,
                      net_progress=float(net), left_half_metal_share=share)
        return self._scored(all(c["pass"] for c in checks.values()), checks, reward, judged=judged or {},
                            flags=flags or {})

    def judge_stitch_weld_ends(self, rows, judged=None, flags=None):
        """stitch-weld-ends (A10) on one state's sections: each window (STITCH_WINDOWS) as fillet-2f judges its span, a
        crater judged only on its sections inside a window, undercut on the whole joint, STITCH_MIDDLE_FREE bare, solid.
        Progress: the mean over the two windows of (credit - start credit) / (1 - start credit) (window_credit) less
        FREE_ZONE_PENALTY x the share of the middle holding metal, clipped to [0, 1]."""
        beyond = _undercut_beyond_allowance(rows)
        checks = {}
        for (lo, hi), fx, side in zip(STITCH_WINDOWS, STITCH_FUSION_X, ("_right_end", "_left_end")):
            checks.update(_window_checks(rows, lo, hi, fx, beyond, side))
        inside, exempt = [], []
        for c in self._craters(rows):
            rs = [r for r in c["rows"] if any(_in(r, lo, hi) for lo, hi in STITCH_WINDOWS)]
            if rs:
                low = min(r["size_m"] for r in rs)
                inside.append(dict(stop_mm=c["stop_mm"],
                                   judged_mm=[round(rs[0]["x_m"]*1e3, 2), round(rs[-1]["x_m"]*1e3, 2)],
                                   min_size_mm=round(low*1e3, 4), ok=bool(low >= NOMINAL_LEG - 1e-9)))
            elif c["rows"]:
                exempt.append(c["stop_mm"])
        checks["crater_filled"] = dict(craters_in_windows=inside, exempt_outside_windows_stop_mm=exempt,
                                       limit_mm=NOMINAL_LEG*1e3, ok=_ok(all(c["ok"] for c in inside)))
        checks.update(_joint_checks(rows, STITCH_MIDDLE_FREE, "middle_unwelded"))
        checks = self._named(checks)
        start = self.start_rows
        nets = []
        for lo, hi in STITCH_WINDOWS:
            c, c0 = window_credit(rows, lo, hi, beyond), window_credit(start, lo, hi) if start else 0.
            nets.append((c - c0)/(1. - c0) if c0 < 1. else float(c >= 1.))
        share = free_share(rows, *STITCH_MIDDLE_FREE)
        reward = dict(progress=float(np.clip(_mean(nets) - FREE_ZONE_PENALTY*share, 0., 1.)), window_net_progress=nets,
                      middle_metal_share=share)
        return self._scored(all(c["pass"] for c in checks.values()), checks, reward, judged=judged or {},
                            flags=flags or {})

    def judge_fillet_from_both_ends(self, rows, judged=None, flags=None):
        """fillet-from-both-ends (A12) on one state's sections: every fillet-2f check, unchanged, and the two runs
        (both_ends_sequence). Progress: fillet_progress with the credit of a section kept where the run that laid it
        moved toward the middle (toward_middle), net of the tacked start read the same way."""
        checks = self._eval_fillet_2f(rows)
        diagnostics = dict(self.diagnostics)
        runs = weld_runs(rows)
        checks["two_runs_from_the_ends"] = both_ends_sequence(runs)
        checks = self._named(checks)
        start = self.start_rows
        raw0 = fillet_progress(start, eligible=toward_middle(weld_runs(start)))["raw"] if start else None
        reward = fillet_progress(rows, raw0, self._last_craters, eligible=toward_middle(runs))
        return self._scored(all(c["pass"] for c in checks.values()), checks, reward, judged=judged or {},
                            flags=flags or {}, diagnostics=diagnostics)

    def _assembly(self):
        env, w = self.env, self.env.world
        sq, off = env.bracket_alignment()
        return dict(bond_active=bool(w.bond.active), square_deg=float(sq), offset_mm=float(off),
                    hand_touching=bool(env.hand_touches_bracket()))

    def judge_tack_and_weld(self, rows, judged=None, flags=None, bond_active=False, square_deg=0., offset_mm=0.,
                            hand_touching=False):
        """tack-and-weld (A11): every fillet-2f check on the final weld, which incorporates the tacks and is judged as
        one weld (as fillet-2f judges its fillet over the fixture's tacks), and the tacked assembly: bonded, square
        within 1 deg, within FIT_MM of the fit-up, the hand released. The tacks are the means: a bracket held and welded
        without them is judged by the same end state. Progress: fillet_progress (net of the loose start, which holds no
        weld) x the alignment (each of squareness and position 1 within its limit, 0 at twice: the smaller) x
        FAULT_FACTOR while the hand still touches the bracket."""
        checks = self._eval_fillet_2f(rows)
        diagnostics = dict(self.diagnostics)
        checks.update({
            "bonded": dict(active=bool(bond_active), ok=_ok(bond_active)),
            "square": dict(angle_error_deg=square_deg, limit_deg=1., ok=_ok(bond_active and abs(square_deg) <= 1.)),
            "position": dict(offset_mm=offset_mm, limit_mm=FIT_MM, ok=_ok(bond_active and offset_mm <= FIT_MM)),
            "hand_off": dict(touching=bool(hand_touching), ok=_ok(bond_active and not hand_touching)),
        })
        checks = self._named(checks)
        weld = fillet_progress(rows, self.start_raw_progress, self._last_craters)
        align = min(graded(abs(square_deg), 1., 1.), graded(offset_mm, FIT_MM, FIT_MM))
        release = FAULT_FACTOR if hand_touching else 1.
        reward = dict(weld, weld_progress=weld["progress"], alignment=align, release_factor=release,
                      progress=float(weld["progress"]*align*release))
        return self._scored(all(c["pass"] for c in checks.values()), checks, reward, judged=judged or {},
                            flags=flags or {}, diagnostics=diagnostics)
