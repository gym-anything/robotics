"""Task instructions and evaluator. The evaluator reads physical state only: hole records (in
each workpiece's frame), part and tool poses, item insertions, tool states, bit states, and hard claims tracked at
tick rate (rotation of a held workpiece, contact of a held tool with the world, peak insertion force). It never reads
an action. Thresholds are authored; each has an adversarial test (E17). A benchmark task's checks, thresholds and score
are in its verifier.py (tasks/<task>/), which Evaluator.evaluate calls; the other tasks' checks are the methods below.

Every check is a goal (it fails in the untouched start state: the work must be done for it to pass) or a constraint
(it holds in the start state and can only be broken: do not cut the bar, leave the spindle stopped, ...). `score` is the
fraction of goal checks passed; `success` needs every check, goals and constraints. (As the fraction of all checks,
the score of a do-nothing end state would be 0.875 on press-drill.) Exception: ledger-anchor reports a continuous
`progress` read from the work product and `score` = 1 on success, otherwise 0.9 x progress
(tasks/ledger-anchor/verifier.py).

Checks of phenomena the physics does not model cannot fail, and make a task look harder than it is. They are
kept in the code behind the switches below, off until the phenomenon is modelled.
"""

import copy
import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from . import materials as M
from .layout import WALL_AXIS_TOL_DEG, WALL_MARK_TOL_M, task_verifier

# ------------------------------------------------------------------ checks of unmodelled phenomena: off
# press-drill / press-blind "bit never above 550 C": cutting speed heats an HSS bit (holes.py, the tool-chip
# interface rise), but the press runs at a fixed belt step per task (1000 rpm for the 9 mm drill: E7 peaks at 472 C),
# so the operator has no speed choice and the check cannot fail. Off until the belt step is part of the task.
SCORE_BIT_BURN = False
# anchor holes "entry spall at most 2 d": the spall is a constant 1.4 d or 1.8 d set at seating (holes.py).
SCORE_ENTRY_SPALL = False
# timber exit damage factor: the plywood backer under the clamped boards (a free body, held by
# the strap clamps) always backs the exit (holes._exit_backed), so tear-out never occurs there.
SCORE_BACKED_EXIT = False
# exit burr (press-drill, match-bolt, panel-holes): the burr limit is the RCSC 1/16 in (params.BURR_GROSS_M, 1.6 mm)
# and the burr model's largest burr from any push the press allows is 0.49 mm (the params entry's own note); the panel's
# exit is always backed (cap collapse never occurs). The check cannot fail on any task.
SCORE_EXIT_BURR = False
# wall anchors "nut not taken past 1.7 T_inst" (over-torque): scored, because it can fail. The wrench's blows carry a
# hammer's energy and the nut's torque is where they leave it (tools.ImpactWrench). The Adaptive Torque module stops
# the first pull at the installation torque, but every further pull of the trigger on a set anchor adds a blow
# (0.3-1.4 N m: E20 retrigger): the nut passed 1.7 T_inst at the 77th further pull and the stud broke at the 95th;
# and the same wrench without its module (a start state no task has) passes 1.7 T_inst after 2.6 s and
# breaks the stud after 3.1 s (E20 no_module). Set this to False to report the level in the check's note without
# scoring it.
SCORE_OVER_TORQUE = True

# ------------------------------------------------------------------ continuous scores
# a failed attempt scores at most this fraction of its progress, so that every failure stays below every success
FAIL_SCORE_FACTOR = 0.9

INSTRUCTIONS = {
    "take-drill": "Take the cordless drill in both hands, one hand on its grip and the other on its motor housing, and "
                  "hold it ready to drill straight down into the bench, the bit tip 30 to 70 mm above the bench, for 3 "
                  "seconds. Do not run it.",
    "punch-layout": "Centre-punch the four layout marks on the steel bar.",
    "press-drill": "Clamp the bar in the drill-press vise and drill a 9 mm hole through it at the punched mark. Leave the "
                   "spindle stopped and the quill up.",
    "press-blind": "Drill the M8 tap-drill hole at the punched mark in the aluminium block to 15 mm full-diameter depth. "
                   "Do not drill through. Leave the spindle stopped and the quill up.",
    "match-bolt": "Drill the two 9 mm bolt holes through the angle bracket and the flat bar at the marks so that they line "
                  "up, then push the two M8 bolts through both parts.",
    "panel-holes": "Drill four 6 mm holes through the enclosure cover at the four marks. The cover is strapped on a "
                   "plywood backer; a 3 mm and a 6 mm drill are on the bench.",
    "timber-bolt": "Bore a 9/16-inch hole through both boards at the mark, square to their faces (within 1.2 degrees), "
                   "and pass the 1/2-inch carriage bolt through by hand.",
    "anchor-holes": "Drill four 1/2-inch anchor holes at the marked crosses, square to the wall, 108 to 125 mm deep.",
    "rebar-relocate": "Drill the four 1/2-inch anchor holes at the crosses on the two channel lines, square to the wall, "
                      "108 to 125 mm deep. If a hole hits reinforcement, stop without cutting the bar and leave that "
                      "hole unfilled; drill its replacement on the same channel line at least twice the stopped hole's "
                      "depth from it and no more than 150 mm from it.",
    "clean-and-set": "Set the four 1/2-inch wedge anchors in the four predrilled holes: drive each to its nominal embedment "
                     "(95 mm, its washer on the wall), then tighten its nut with the impact wrench on the kit trolley (its "
                     "adaptive-torque module is set for these anchors: it stops with a green light at their installation "
                     "torque, 68 N m). The holes still hold their drilling dust: remove it with the shop vacuum's nozzle "
                     "(switch the vacuum on; the brush is optional) so that the anchors reach their embedment.",
    "ledger-anchor": "The ledger is set level on temporary brackets against the concrete wall. Fasten it to the wall "
                     "with the four 1/2-inch wedge anchors, one at each point marked on the ledger: drill through the "
                     "ledger and 108 to 125 mm into the concrete, clear the drilling dust with the shop vacuum as needed, "
                     "drive each anchor to its nominal embedment (95 mm) with its washer on the ledger, and tighten its "
                     "nut with the impact wrench, whose adaptive-torque module is set for these anchors and stops with a "
                     "green light at their installation torque, 68 N m.",
}


def _check(name, passed, value=None, threshold=None, note=None, kind="goal"):
    c = dict(name=name, passed=bool(passed), kind=kind)
    if value is not None:
        c["value"] = value
    if threshold is not None:
        c["threshold"] = threshold
    if note:
        c["note"] = note
    return c


def _angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b)/(np.linalg.norm(a)*np.linalg.norm(b)), -1., 1.))))


def _ramp(x, a, b):
    """Partial credit: 1 up to the acceptance limit a, falling linearly to 0 at b (one tolerance width beyond it)."""
    return 1. if x <= a else 0. if x >= b else (b - x)/(b - a)


def _in_plane(h, p_local):
    """Distance (m) of a hole's entry from a point, in the plane of the hole's entry face (the workpiece's frame)."""
    r = h.p0 - np.asarray(p_local, float)
    return float(np.linalg.norm(r - h.face_normal*float(r @ h.face_normal)))


class Evaluator:
    def __init__(self, env):
        self.env = env
        self.w = env.world
        self.L = env.layout
        self.task = self.L["task"]
        self.instruction = INSTRUCTIONS[self.task]
        w = self.w
        self.s = dict(start_pose={n: w.body_pose(info["body"]) for n, info in w.scene.parts.items()},
                      max_rel_turn_deg={}, held_since=None, placed_since=None, trigger_max_m=0., bit_touched=False,
                      housing_turn_deg={}, max_bit_temp_c={}, events=[], anchor_max_turn_nm={})
        self._ref_rel = {}
        self._wp_bodies = {n: info["body"] for n, info in w.scene.parts.items() if info["kind"] == "drillable"}

    def snapshot(self):
        return dict(s=self.s, ref=self._ref_rel)

    def restore(self, snap):
        self.s, self._ref_rel = snap["s"], snap["ref"]

    # ------------------------------------------------------------------ hard claims at tick rate
    def tick(self):
        """A benchmark task's verifier.py may track its own claims: track_tick(ev) every tick, track_claims(ev) with
        the claims below every tenth tick."""
        w = self.w
        folder = task_verifier(self.task)
        if hasattr(folder, "track_tick"):
            folder.track_tick(self)
        if w.tick % 10:
            return
        s = self.s
        # rotation of each free workpiece about its own vertical axis relative to its support (vise or bench)
        for name, body in self._wp_bodies.items():
            if w.scene.drillables[name].static:
                continue
            q = w.bq[body]
            yaw = Rotation.from_quat(q[3:]).as_euler("xyz")[2]
            ref = self._ref_rel.setdefault(name, yaw)
            turn = abs(math.degrees(math.atan2(math.sin(yaw - ref), math.cos(yaw - ref))))
            spinning = any(it.state.get("contact_face") or it.slots.get(name) for it in w.hm.items.values()
                           if it.kind == "bit" and it.spindle is not None and abs(it.spindle.speed()) > 1.)
            if spinning:
                s["max_rel_turn_deg"][name] = max(s["max_rel_turn_deg"].get(name, 0.), turn)
            else:
                self._ref_rel[name] = yaw
        for n, it in w.hm.items.items():
            if it.kind == "bit":
                s["max_bit_temp_c"][n] = max(s["max_bit_temp_c"].get(n, 20.), it.state["temp_c"])
        if hasattr(folder, "track_claims"):
            folder.track_claims(self)
        for n, t in w.scene.tools.items():
            if hasattr(t, "housing") and hasattr(t, "item") and getattr(t, "spindle", None) is not None:
                # housing rotation about its own spindle axis while the bit is cutting (a grab twists the tool)
                cutting = any(v[1] in ("cut", "hole") for v in t.item.slots.values()) and abs(t.spindle.speed()) > 1.
                R = Rotation.from_quat(w.bq[t.housing][3:])
                roll = R.as_euler("xyz")[0] if False else None
                ax = R.apply([1., 0., 0.])
                ref = self._ref_rel.get(("housing", n))
                up = R.apply([0., 0., 1.])
                if cutting:
                    if ref is None:
                        self._ref_rel[("housing", n)] = up.tolist()
                    else:
                        ref = np.asarray(ref)
                        u = up - ax*float(up @ ax)
                        r = ref - ax*float(ref @ ax)
                        if np.linalg.norm(u) > 1e-6 and np.linalg.norm(r) > 1e-6:
                            ang = _angle_deg(u, r) if float(u @ r) >= 0 else 180. - _angle_deg(u, r)
                            s["housing_turn_deg"][n] = max(s["housing_turn_deg"].get(n, 0.), ang)
                else:
                    self._ref_rel.pop(("housing", n), None)

    # ------------------------------------------------------------------ helpers
    def holes(self, name):
        return [h for h in self.w.hm.drillables[name].holes if h.kind == "drilled"]

    def nearest(self, name, p_local, kinds=("drilled",)):
        best = None
        for h in self.w.hm.drillables[name].holes:
            if h.kind not in kinds:
                continue
            e = float(np.linalg.norm((h.p0 - np.asarray(p_local))[[i for i in range(3)]] -
                                     h.face_normal*float((h.p0 - np.asarray(p_local)) @ h.face_normal)))
            if best is None or e < best[1]:
                best = (h, e)
        return best

    def common(self):
        """Checks common to every task: no fault, no broken bit, and (except where the task's verifier.py reports it as
        a diagnostic: payload_diagnostic) no arm over its rated payload."""
        w = self.w
        out = [_check("no fault", w.fault is None, w.fault, kind="constraint")]
        broken = [n for n, it in w.hm.items.items() if it.kind == "bit" and it.state["broken"]]
        out.append(_check("no bit broken", not broken, broken, kind="constraint"))
        if hasattr(task_verifier(self.task), "payload_diagnostic"):
            return out
        # an arm that carries more than its rating has left its specification (FR3: 3 kg with its 0.73 kg hand, so
        # 2.27 kg of tool). The world lets it (the real arm does too) and counts the seconds; the task is not passed
        # by a robot working outside its rating
        over = {a: round(t, 2) for a, t in getattr(w, "over_payload_s", {}).items() if t > P.PAYLOAD_OVER_S}
        rated = P.FR3_PAYLOAD_KG - P.FRANKA_HAND_KG
        out.append(_check(f"no arm carried more than its rated payload ({rated:g} kg beyond its hand)", not over,
                          over or 0., f"<= {P.PAYLOAD_OVER_S:g} s over the rating", kind="constraint"))
        return out

    def spindle_stopped_quill_up(self):
        p = self.w.scene.tools["drill_press"]
        w = self.w
        return [_check("spindle switched off and stopped", (not p.s_on) and abs(p.spindle.speed()) < 2*math.pi/60,
                       round(p.spindle.speed()*60/(2*math.pi), 2), "< 1 rpm", kind="constraint"),
                _check("quill within 2 mm of the top", p.j["quill"].pos(w) > -0.002, round(p.j["quill"].pos(w)*1e3, 2),
                       "> -2 mm", kind="constraint"),
                _check("bit clear of the work", not p.item.slots, list(p.item.slots), kind="constraint")]

    # ------------------------------------------------------------------ evaluation
    def evaluate(self):
        """The episode's outcome. A benchmark task's verify(ev) (tasks/<task>/verifier.py) decides: it hands its
        checks to evaluate_checks. Any other task's checks are its method below (_anchor_holes, _clean_and_set,
        _match_bolt)."""
        folder = task_verifier(self.task)
        if folder is not None:
            return folder.verify(self)
        return self.evaluate_checks(getattr(type(self), "_" + self.task.replace("-", "_")))

    def evaluate_checks(self, checks, **hooks):
        """The released verdict on checks(self): the task's own checks, or (checks, progress report) for a task with
        its own progress measure. The keywords are a task's hooks for the contract graders (contract_checks, progress,
        note, terms: catalogue_evaluator.CatalogueEvaluator.evaluate_checks); this verdict does not use them."""
        res = checks(self)
        # a task with its own progress measure returns (checks, progress report)
        own, progress = res if isinstance(res, tuple) else (res, None)
        checks = self.common() + own
        goals = [c for c in checks if c["kind"] == "goal"]
        passed = sum(c["passed"] for c in goals)
        success = all(c["passed"] for c in checks)
        out = dict(task=self.task, success=success, score=round(passed/len(goals), 4) if goals else 0.,
                   goals=dict(passed=passed, total=len(goals)),
                   constraints_ok=all(c["passed"] for c in checks if c["kind"] == "constraint"),
                   checks=checks, instruction=self.instruction)
        if progress is not None:
            out.update(progress)
            out["score"] = 1. if success else round(FAIL_SCORE_FACTOR*progress["progress"], 4)
        diagnostic = getattr(task_verifier(self.task), "payload_diagnostic", None)
        if diagnostic is not None:
            out.setdefault("diagnostics", {})["payload"] = diagnostic(self)
        return out

    def _through_hole_checks(self, name, mark, d_mm, tol_m, label, max_d_mm=None, burr_mm=P.BURR_GROSS_M*1e3,
                             burr_scored=None):
        out = []
        hit = self.nearest(name, mark)
        # through at the diameter asked for: a pilot that runs through under a wider stage that has only begun is not
        # the hole (otherwise a 6 mm stage 0.9 mm deep would read "through")
        if hit is None or not hit[0].through_at(d_mm):
            out.append(_check(f"{label}: through hole at the mark", False, None if hit is None else
                              ("not through" if not hit[0].through else f"through at a smaller diameter only")))
            return out, None
        h, e = hit
        out.append(_check(f"{label}: through hole at the mark", True, "through"))
        out.append(_check(f"{label}: entry within {tol_m*1e3:.1f} mm of the mark", e <= tol_m, round(e*1e3, 3),
                          f"<= {tol_m*1e3:.1f} mm"))
        ang = _angle_deg(h.a, -h.face_normal)
        out.append(_check(f"{label}: axis within 1 deg of the face normal", ang <= 1., round(ang, 3), "<= 1 deg"))
        if max_d_mm:
            dmax = h.d_mm + 2e3*h.max_offset_m
            out.append(_check(f"{label}: diameter at most {max_d_mm} mm", dmax <= max_d_mm and h.d_mm >= d_mm - 1e-6,
                              round(dmax, 3), f"{d_mm}-{max_d_mm} mm"))
        if SCORE_EXIT_BURR if burr_scored is None else burr_scored:
            out.append(_check(f"{label}: exit burr at most {burr_mm} mm", h.burr_m*1e3 <= burr_mm, round(h.burr_m*1e3, 3),
                              f"<= {burr_mm} mm"))
        return out, h

    def _match_bolt(self):
        w = self.w
        out = []
        holes = []
        for k, mk in enumerate(self.L["marks"]):
            o, h = self._through_hole_checks("angle_bracket", mk["local"], 9.0, 0.0005, f"angle hole {k}", 9.25)
            out += o
            holes.append(h)
        bar_through = [h for h in self.holes("flat_bar") if h.through]
        out.append(_check("two through holes in the flat bar", len(bar_through) >= 2, len(bar_through), ">= 2"))
        if all(holes):
            sp = float(np.linalg.norm(holes[0].p0 - holes[1].p0))
            out.append(_check("hole spacing within 0.5 mm of 60 mm", abs(sp - 0.060) <= 0.0005, round(sp*1e3, 3),
                              "60 +- 0.5 mm"))
        for k in range(2):
            it = w.hm.items[f"m8_bolt{k}"]
            through = [dn for dn, v in it.slots.items() if v[1] == "hole"]
            ok = {"angle_bracket", "flat_bar"} <= set(through)
            out.append(_check(f"bolt {k} passes through both parts", ok, sorted(through)))
            out.append(_check(f"bolt {k} pushed in with less than 50 N", it.state["max_axial_n"] < 50.,
                              round(it.state["max_axial_n"], 2), "< 50 N", kind="constraint"))
        for n in ("angle_bracket", "flat_bar"):
            turn = self.s["max_rel_turn_deg"].get(n, 0.)
            out.append(_check(f"{n} not spun (< 2 deg)", turn < 2., round(turn, 3), "< 2 deg", kind="constraint"))
        return out

    # ------------------------------------------------------------------ wall
    def _wall_mark_local(self, mk):
        d = self.w.hm.drillables["wall"]
        q = self.w.bq[d.body]
        R = Rotation.from_quat(q[3:])
        return R.inv().apply(np.array([mk["x"], P.WALL_FACE_Y, mk["z"]]) - q[:3])

    def _anchor_hole(self, mk, label, depth_range=(0.108, 0.125)):
        out = []
        m = self._wall_mark_local(mk)
        hit = self.nearest("wall", m)
        if hit is None or hit[1] > 0.02:
            return [_check(f"{label}: hole at the mark", False)], None
        h, e = hit
        tol = WALL_MARK_TOL_M
        out.append(_check(f"{label}: within {tol*1e3:.0f} mm of the mark", e <= tol, round(e*1e3, 2),
                          f"<= {tol*1e3:.0f} mm"))
        ang = _angle_deg(h.a, -h.face_normal)
        out.append(_check(f"{label}: axis within {WALL_AXIS_TOL_DEG:.0f} deg of the wall normal", ang <= WALL_AXIS_TOL_DEG,
                          round(ang, 2), f"<= {WALL_AXIS_TOL_DEG:.0f} deg"))
        fd = h.full_depth()
        out.append(_check(f"{label}: depth {depth_range[0]*1e3:.0f}-{depth_range[1]*1e3:.0f} mm",
                          depth_range[0] <= fd <= depth_range[1], round(fd*1e3, 1)))
        if SCORE_ENTRY_SPALL:
            out.append(_check(f"{label}: entry spall at most 2 d", h.entry_spall_d_m <= 2*h.d_mm*1e-3,
                              round(h.entry_spall_d_m*1e3, 1), f"<= {2*h.d_mm:.1f} mm"))
        out.append(_check(f"{label}: reinforcement damage at most 0.5 mm", h.rebar_damage_m <= 0.0005,
                          round(h.rebar_damage_m*1e3, 3), "<= 0.5 mm"))
        return out, h

    def _bit_not_chipped(self):
        chipped = [n for n, it in self.w.hm.items.items() if it.kind == "bit" and it.state["chipped"]]
        return _check("bit not chipped on steel", not chipped, chipped, kind="constraint")

    def _anchor_holes(self):
        out = []
        for k, mk in enumerate(self.L["marks"]):
            out += self._anchor_hole(mk, f"hole {k}")[0]
        return out + [self._bit_not_chipped()]

    def _anchor_checks(self, k, h, anchors):
        """Anchor k of a hole is set the way ESR-4266 4.2.3 sets a KB-TZ2, all read from its physical state
        (parts.WedgeAnchor): driven to h_nom (its tip 95 +- 2 mm from the concrete face, ESR Fig. 2, after setting) and
        straight; its nut torqued to T_inst (the installation torque: the largest thread and bearing friction torque
        against which the nut was turned on while the cone gripped, recorded from the solved friction forces,
        parts.WedgeAnchor 'max_turn_torque_nm'; at least 0.9 T_inst, params.T_INST_TOL); the expansion engaged (the
        cone drawn into a clip gripping this hole's wall, still carrying load); and the stud did not turn in its hole
        while the nut was loaded (an anchor that spins: a bad hole). The torque is judged on the installation, as an
        inspector's is, not on what the stud carries at the moment of evaluation: the tension of a set anchor falls
        afterwards without anything being wrong with its setting (in the ledger task each later anchor draws the one
        ledger further into its seat on the wall and takes 4.5 kN of 26.3 kN off the anchors set before it, E12); the
        present tension is given in the check's note. A nut never turned on against 0.9 T_inst fails,
        and one taken past 1.7 T_inst (the setting torque of ACI 355.2's splitting test) fails on that maximum
        (SCORE_OVER_TORQUE: see there for how the wrench gets there). A stud broken by over-torque (parts.WedgeAnchor,
        'failed') keeps the torque it was taken to on record and fails the expansion check. No cleaning sequence is
        judged: dust left in the hole shows as an anchor short of h_nom."""
        w = self.w
        out = []
        key = f"wall/{h.id}"
        best = None
        for a in anchors:
            z = a.state["insertion_m"].get(key)
            if z is not None and (best is None or z > best[1]):
                best = (a, z)
        clean_note = f"vacuum passes {len(h.vacuum_passes)}, brush strokes {len(h.brush_strokes)}, " \
                     f"dust column {h.dust_column_m()*1e3:.1f} mm"
        T = P.KBTZ2["torque"]
        t_lo, t_hi = (1 - P.T_INST_TOL)*T, 1.7*T
        band = f"{t_lo:.0f}-{t_hi:.0f} N m" if SCORE_OVER_TORQUE else f"at least {t_lo:.0f} N m"
        names = (f"hole {k}: anchor driven to h_nom (95 +- 2 mm)", f"hole {k}: anchor straight (within 3 deg)",
                 f"hole {k}: nut torqued to T_inst ({band})",
                 f"hole {k}: expansion engaged (cone drawn into a gripping clip)",
                 f"hole {k}: anchor did not spin (<= {P.ANCHOR_SPIN_MAX_DEG:.0f} deg)")
        if best is None:
            out.append(_check(names[0], False, None, "93-97 mm", note=clean_note))
            out.append(_check(names[2], False, None, "no anchor in the hole"))
            out.append(_check(names[3], False, None, "no anchor in the hole"))
            return out
        a, z = best
        # the nominal embedment is the depth the anchor is HAMMERED to, before its nut is tightened (ESR-4266 4.2.3);
        # setting then draws the stud out by the tension over the anchor's stiffness, 1.0-1.5 mm here. Judged on the
        # deepest the tip has been in this hole, with the anchor still within 3 mm of it (not pulled back out). Read
        # after setting, the depth leaves a correctly driven and set anchor 0.5 mm inside the lower limit
        z_drive = max(float(a.state.get("insertion_max_m", {}).get(key, z)), z)
        seated = (abs(z_drive - P.KBTZ2["h_nom"]) <= 0.002 and z >= z_drive - 0.003
                  and "wall" in a.slots and a.slots["wall"][1] == "hole")
        out.append(_check(names[0], seated, round(z_drive*1e3, 2), "93-97 mm",
                          note=f"driven to {z_drive*1e3:.2f} mm, now at {z*1e3:.2f} mm; {clean_note}"))
        p, R = w.body_pose(a.body)
        ang = _angle_deg(R @ np.array([1., 0., 0.]), w.hm.world_dir(w.hm.drillables["wall"], h.a))
        out.append(_check(names[1], ang <= 3., round(ang, 2), "<= 3 deg"))
        st = a.state.get("setting", {})
        clip = a.state.get("clip") or {}
        here = clip.get("key") == key
        tq = float(st.get("max_turn_torque_nm", 0.)) if here else 0.
        out.append(_check(names[2], t_lo <= tq and (tq <= t_hi or not SCORE_OVER_TORQUE), round(tq, 2), band,
                          note=f"largest friction torque the nut was turned on against with the cone gripping; "
                               f"over-torque level {t_hi:.0f} N m {'EXCEEDED' if tq > t_hi else 'not reached'}"
                               f"{'' if SCORE_OVER_TORQUE else ' (reported, not scored)'}; stud tension: largest "
                               f"{st.get('max_set_n', 0.):.0f} N, now {st.get('thread_n', 0.):.0f} N"
                               f"{'; ANCHOR FAILED (' + str(st.get('failed')) + ')' if st.get('failed') else ''}"))
        engaged = bool(here and st.get("grip") and st.get("cone_n", 0.) > 0.)
        out.append(_check(names[3], engaged, round(float(st.get("s_m", 0.))*1e3, 3) if here else None,
                          "cone drawn in, clip gripping", note=f"cone load {st.get('cone_n', 0.):.0f} N"))
        spin = float(st.get("spin_max_deg", 0.))
        out.append(_check(names[4], spin <= P.ANCHOR_SPIN_MAX_DEG, round(spin, 2), f"<= {P.ANCHOR_SPIN_MAX_DEG:.0f} deg",
                          kind="constraint"))
        return out

    def _clean_and_set(self):
        w = self.w
        out = []
        anchors = [w.hm.items[f"anchor{k}"] for k in range(4)]
        for k, mk in enumerate(self.L["marks"]):
            hit = self.nearest("wall", self._wall_mark_local(mk))
            if hit is None:
                out.append(_check(f"hole {k}: anchor driven to h_nom (95 +- 2 mm)", False, None, "no hole"))
                continue
            out += self._anchor_checks(k, hit[0], anchors)
        return out
