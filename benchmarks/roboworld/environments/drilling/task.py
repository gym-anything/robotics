"""Task instructions and evaluator. The evaluator reads physical state only: hole records (in
each workpiece's frame), part and tool poses, item insertions, tool states, bit states, and hard claims tracked at
tick rate (rotation of a held workpiece, contact of a held tool with the world, peak insertion force). It never reads
an action. Thresholds are authored; each has an adversarial test (E17). ledger-anchor's checks, thresholds and score
are documented in Evaluator._ledger_anchor and the module constants under "ledger-anchor (revised task)".

Every check is a goal (it fails in the untouched start state: the work must be done for it to pass) or a constraint
(it holds in the start state and can only be broken: do not cut the bar, leave the spindle stopped, ...). `score` is the
fraction of goal checks passed; `success` needs every check, goals and constraints. (As the fraction of all checks,
the score of a do-nothing end state would be 0.875 on press-drill.) Exception: ledger-anchor reports a continuous
`progress` read from the work product and `score` = 1 on success, otherwise 0.9 x progress (Evaluator._ledger_anchor).

Checks of phenomena the physics does not model cannot fail, and make a task look harder than it is. They are
kept in the code behind the switches below, off until the phenomenon is modelled.
"""

import copy
import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from . import materials as M
from .layout import WALL_AXIS_TOL_DEG, WALL_MARK_TOL_M

# ------------------------------------------------------------------ checks of unmodelled phenomena: off
# panel-holes "did not grab": the thin-sheet grab (a twist drill catching 1.5 mm sheet like a thread) is not reproduced
# (E6: 1.0 N m, 0.8 deg housing turn), so the housing-turn check cannot fail. The task is described accordingly: it
# is a placing, piloting and enlarging task on a backed sheet (the trade's way of avoiding the grab), not a test of
# handling a drill that grabs.
SCORE_SHEET_GRAB = False
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
# punch-layout "dimple at least 0.3 mm deep": every strike leaves 0.40 to 0.52 mm in this steel over the punch's
# 60-130 N cap settings (tools.CentrePunch: the strike energy is the spring's at that setting), and a dimple exists only after a strike, which the position check already requires; the depth
# check cannot fail on its own.
SCORE_DIMPLE_DEPTH = False
# wall anchors "nut not taken past 1.7 T_inst" (over-torque): scored, because it can fail. The wrench's blows carry a
# hammer's energy and the nut's torque is where they leave it (tools.ImpactWrench). The Adaptive Torque module stops
# the first pull at the installation torque, but every further pull of the trigger on a set anchor adds a blow
# (0.3-1.4 N m: E20 retrigger): the nut passed 1.7 T_inst at the 77th further pull and the stud broke at the 95th;
# and the same wrench without its module (a start state no task has) passes 1.7 T_inst after 2.6 s and
# breaks the stud after 3.1 s (E20 no_module). Set this to False to report the level in the check's note without
# scoring it.
SCORE_OVER_TORQUE = True

# ------------------------------------------------------------------ ledger-anchor (revised task)
# Every other task keeps its checks and its goal-fraction score.
# The rated-payload seconds decide no success here: the task is judged on its work product, not on how the tools were
# carried (one hand or two), and the policy is never told the rating. They are reported under `diagnostics`.
PAYLOAD_DIAGNOSTIC_TASKS = ("ledger-anchor",)
# the search radius of a mark: the openings at a mark are the records within this distance (the wall tasks' radius),
# and the mark's bore is the opening that carries the most work (Evaluator._ledger_openings; the record nearest the
# mark, of any depth, can be an abandoned start instead of the bore carrying the anchor). The radius
# is not a licence for other holes: every other record must be a pit shallower than the bit's point (a running bit's
# touch, no full-diameter hole), a re-entry (its entry within one bit radius of the judged record in the same part: the
# same opening, cut again off axis; the hole module opens a new record for a bit more than CAPTURE_TOL off a hole's
# axis), a start in the ledger that the anchor's washer covers (its whole width within WASHER_R_M of the bore's axis:
# at most 10.55 mm off for this bit), or a blind start in the ledger elsewhere (reported and charged in progress, not
# failed: surface damage in the timber). Any other full-diameter hole fails the job: through the ledger
# beside or away from a mark's opening, or in the concrete. A start that has
# reached the concrete has gone through the ledger beside the bore, and an open hole that close to an anchor's hole
# breaks the environment's own aborted-hole rule (ABORTED_HOLE_DISTANCE_FACTOR, ETAG 001 Part 1 7.3: at least twice
# its depth away unless filled; the scene has no mortar). Justified as "under the washer" alone, the radius would let
# a second full bore 15-19 mm beside an anchor pass; with no depth floor a 0.82 mm pit 30 mm from a mark would fail a
# finished job.
MARK_RADIUS_M = 0.020
# "with its washer on the ledger": the washer's bearing face stands at most this far off the ledger's front face
WASHER_ON_FACE_M = 0.001
# "set level on temporary brackets ... Fasten it": the ledger is fastened where it was set. Its centre may end at most
# this far from its start pose in the wall's plane (a ledger slid 60 mm along its brackets and fastened there does
# not pass). Authored: twice the 5 mm nudge the task accepts, which also covers the ledger's give under
# the anchor blows (0.03-1.04 mm per anchor driven); half the mark search radius.
LEDGER_SHIFT_MAX_M = 0.010
# a failed attempt scores at most this fraction of its progress, so that every failure stays below every success
FAIL_SCORE_FACTOR = 0.9
LEDGER_SCORING = ("ledger-anchor progress v3: score = 1 on success, else 0.9 x progress; progress "
                  "= mean over the four marks of the opening carrying the most work, placement x (bore + level x "
                  "position x (embedment + torque))/3, less a quarter per bar cut off a mark's opening and each other "
                  "defect hole's share of a bore; embedment and torque count only at the embedment reached once torqued")
# tasks whose per-step record carries no score when the state cannot be evaluated (a world fault, a non-finite state,
# an evaluator error) instead of 0, which reads as "nothing done". The backend's step reward is
# then None (BackendStep.reward may be None); its evaluate() already refuses a faulted state.
NO_SCORE_WHEN_UNEVALUABLE = ("ledger-anchor",)

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
        w = self.w
        if self.task == "ledger-anchor":
            self._track_anchor_torque()
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
        if self.task == "take-drill":
            self._track_hold()
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

    def _track_anchor_torque(self):
        """The all-time largest installation-torque record of each anchor, every tick.
        parts.WedgeAnchor resets its record to 0 when the cone loses its grip (clip_left_hole, cone_pulled_through,
        cone_driven_out), so a nut taken past 1.7 T_inst, then struck loose and tightened again, would leave no trace of
        the over-torque in the record alone."""
        rec = self.s.setdefault("anchor_max_turn_nm", {})
        for n, it in self.w.hm.items.items():
            if it.kind == "anchor":
                t = float(it.state.get("setting", {}).get("max_turn_torque_nm", 0.))
                if t > rec.get(n, 0.):
                    rec[n] = t

    def _track_hold(self):
        w = self.w
        drill = w.scene.tools["drill"]
        s = self.s
        s["trigger_max_m"] = max(s["trigger_max_m"], drill.j["trigger"].pos(w))
        if drill.item.slots:
            s["bit_touched"] = True
        tool = set(drill.bodies)
        others = [b for b in range(w.model.body_count) if b not in tool and b not in w.scene.robot_bodies]
        world_f = sum(w.pair_contact_forces(list(tool), others + [-1]).values())
        hands = {}
        for arm in w.rig.instances:
            fingers = [w._mj_to_newton[b] for b in w._arm_fingers[arm]]
            hands[arm] = sum(w.pair_contact_forces(list(tool), fingers).values())
        held = world_f < P.CONTACT_FORCE_TOL_N and all(v > 0.2 for v in hands.values()) and len(hands) == 2
        if held:
            if s["held_since"] is None:
                s["held_since"] = w.sim_time
        else:
            s["held_since"] = None
        s["hold_hands_n"] = {k: round(v, 2) for k, v in hands.items()}
        s["hold_world_n"] = round(world_f, 2)
        # where the hands are: one hand's fingers on the grip shape, the other's on the motor head (contact forces)
        on = self._finger_forces_on(drill, ("grip", "head"))
        arms = list(on)
        placed = len(arms) == 2 and any(on[a]["grip"] > 0.2 and on[b]["head"] > 0.2 for a in arms for b in arms if a != b)
        if placed:
            if s.get("placed_since") is None:
                s["placed_since"] = w.sim_time
        else:
            s["placed_since"] = None
        s["hands_on_n"] = {a: {k: round(v, 2) for k, v in f.items()} for a, f in on.items()}

    def _finger_forces_on(self, tool, parts):
        """{arm: {part: normal force (N) of that arm's finger links on the tool's shape labelled <tool>_<part>}}."""
        w = self.w
        if not hasattr(self, "_shape_ids"):
            lab = list(w.scene.b.shape_label)
            self._shape_ids = {part: {i for i, l in enumerate(lab) if l == f"{tool.name}_{part}"} for part in parts}
        m, d = w.solver.mj_model, w.solver.mj_data
        gs = w._geom_shape
        fingers = {arm: set(w._arm_fingers[arm]) for arm in w.rig.instances}
        out = {arm: {part: 0. for part in parts} for arm in fingers}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            g = d.contact.geom[i]
            sh = [int(gs[x]) if x < len(gs) else -1 for x in g]
            bodies = [int(m.geom_bodyid[x]) for x in g]
            for part, ids in self._shape_ids.items():
                for j in (0, 1):
                    if sh[j] in ids:
                        for arm, fs in fingers.items():
                            if bodies[1 - j] in fs:
                                w._mujoco.mj_contactForce(m, d, i, f6)
                                out[arm][part] += abs(float(f6[0]))
        return out

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
        """Checks common to every task: no fault, no broken bit, and (except in PAYLOAD_DIAGNOSTIC_TASKS, which report
        it as a diagnostic) no arm over its rated payload."""
        w = self.w
        out = [_check("no fault", w.fault is None, w.fault, kind="constraint")]
        broken = [n for n, it in w.hm.items.items() if it.kind == "bit" and it.state["broken"]]
        out.append(_check("no bit broken", not broken, broken, kind="constraint"))
        if self.task in PAYLOAD_DIAGNOSTIC_TASKS:
            return out
        # an arm that carries more than its rating has left its specification (FR3: 3 kg with its 0.73 kg hand, so
        # 2.27 kg of tool). The world lets it (the real arm does too) and counts the seconds; the task is not passed
        # by a robot working outside its rating
        over = {a: round(t, 2) for a, t in getattr(w, "over_payload_s", {}).items() if t > P.PAYLOAD_OVER_S}
        rated = P.FR3_PAYLOAD_KG - P.FRANKA_HAND_KG
        out.append(_check(f"no arm carried more than its rated payload ({rated:g} kg beyond its hand)", not over,
                          over or 0., f"<= {P.PAYLOAD_OVER_S:g} s over the rating", kind="constraint"))
        return out

    def payload_diagnostic(self):
        """The rated-payload seconds as a report, deciding nothing (PAYLOAD_DIAGNOSTIC_TASKS)."""
        secs = {a: round(float(t), 3) for a, t in getattr(self.w, "over_payload_s", {}).items()}
        rated = P.FR3_PAYLOAD_KG - P.FRANKA_HAND_KG
        return dict(name=f"seconds each arm carried more than its rated payload ({rated:g} kg beyond its hand)",
                    seconds_over_rating=secs, allowance_s=P.PAYLOAD_OVER_S,
                    within_rating=not any(t > P.PAYLOAD_OVER_S for t in secs.values()),
                    note="reported, not scored: this task is judged on its work product, not on how the tools were "
                         "carried")

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
        fn = getattr(self, "_" + self.task.replace("-", "_"))
        res = fn()
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
        if self.task in PAYLOAD_DIAGNOSTIC_TASKS:
            out.setdefault("diagnostics", {})["payload"] = self.payload_diagnostic()
        return out

    def _take_drill(self):
        w, s = self.w, self.s
        drill = w.scene.tools["drill"]
        p, R = w.body_pose(drill.item.body)
        axis = R @ np.array([1., 0., 0.])
        tip = p + R @ drill.item.tip
        held_s = 0. if s["held_since"] is None else w.sim_time - s["held_since"]
        placed_s = 0. if s.get("placed_since") is None else w.sim_time - s["placed_since"]
        return [_check("held only by the robot's two hands for 3 s", held_s >= 3., round(held_s, 3), ">= 3 s",
                       note=f"finger forces {s.get('hold_hands_n')}, world contact {s.get('hold_world_n')} N"),
                _check("one hand on the grip and the other on the motor housing for 3 s", placed_s >= 3.,
                       round(placed_s, 3), ">= 3 s", note=f"finger forces on grip / head {s.get('hands_on_n')}"),
                _check("bit points straight down (within 5 deg)", _angle_deg(axis, (0., 0., -1.)) <= 5. and axis[2] < 0,
                       round(_angle_deg(axis, (0., 0., -1.)), 2), "<= 5 deg"),
                _check("tip 30-70 mm above the bench", 0.030 <= tip[2] - P.BENCH_TOP_Z <= 0.070,
                       round((tip[2] - P.BENCH_TOP_Z)*1e3, 1), "30-70 mm"),
                _check("trigger never pressed past 1 mm", s["trigger_max_m"] < 0.001, round(s["trigger_max_m"]*1e3, 3),
                       "< 1 mm", kind="constraint"),
                _check("the bit touched nothing", not s["bit_touched"], s["bit_touched"], kind="constraint")]

    def _punch_layout(self):
        w = self.w
        bar = w.hm.drillables["flat_bar"]
        out = []
        dimples = [h for h in bar.holes if h.kind == "dimple"]
        for k, mk in enumerate(self.L["marks"]):
            m = np.asarray(mk["local"])
            near = sorted(dimples, key=lambda h: np.linalg.norm((h.p0 - m)[:2]))
            if not near:
                out.append(_check(f"mark {k}: dimple within 0.3 mm", False, None, "<= 0.3 mm"))
                continue
            h = near[0]
            e = float(np.linalg.norm((h.p0 - m)[:2]))
            out.append(_check(f"mark {k}: dimple within 0.3 mm", e <= 0.0003, round(e*1e3, 3), "<= 0.3 mm"))
            if SCORE_DIMPLE_DEPTH:
                out.append(_check(f"mark {k}: dimple at least 0.3 mm deep", h.stages[0]["s"] >= 0.0003,
                                  round(h.stages[0]["s"]*1e3, 3), ">= 0.3 mm"))
            stray = [x for x in dimples if x is not h and np.linalg.norm((x.p0 - m)[:2]) < 0.010]
            out.append(_check(f"mark {k}: no stray dimple within 10 mm", not stray, len(stray), kind="constraint"))
        p0, _ = self.s["start_pose"]["flat_bar"]
        p, _ = w.body_pose(w.scene.parts["flat_bar"]["body"])
        out.append(_check("bar moved less than 2 mm", np.linalg.norm(p - p0) < 0.002,
                          round(float(np.linalg.norm(p - p0))*1e3, 2), "< 2 mm", kind="constraint"))
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

    def _press_drill(self):
        out, h = self._through_hole_checks("flat_bar", self.L["marks"][0]["local"], 9.0, 0.0005, "hole A", 9.25)
        turn = self.s["max_rel_turn_deg"].get("flat_bar", 0.)
        out.append(_check("bar turned less than 2 deg while drilling", turn < 2., round(turn, 3), "< 2 deg",
                          kind="constraint"))
        if SCORE_BIT_BURN:
            T = self.s["max_bit_temp_c"].get("drill_press_bit", 20.)
            out.append(_check("bit tip never above 550 C", T < P.HSS_BURN_C, round(T, 1), "< 550 C", kind="constraint"))
        return out + self.spindle_stopped_quill_up()

    def _press_blind(self):
        out = []
        mark = self.L["marks"][0]["local"]
        hit = self.nearest("alu_block", mark)
        if hit is None:
            return [_check("full-diameter depth 15.0-15.5 mm", False, None, "15.0-15.5 mm")] + \
                self.spindle_stopped_quill_up()
        h, e = hit
        fd = h.full_depth(6.8)
        out.append(_check("full-diameter depth 15.0-15.5 mm", 0.0150 <= fd <= 0.0155, round(fd*1e3, 3), "15.0-15.5 mm"))
        out.append(_check("not through", not h.through, h.through))
        out.append(_check("entry within 0.5 mm of the mark", e <= 0.0005, round(e*1e3, 3), "<= 0.5 mm"))
        ang = _angle_deg(h.a, -h.face_normal)
        out.append(_check("axis within 1 deg", ang <= 1., round(ang, 3), "<= 1 deg"))
        if SCORE_BIT_BURN:
            T = self.s["max_bit_temp_c"].get("drill_press_bit", 20.)
            out.append(_check("bit tip never above 550 C", T < P.HSS_BURN_C, round(T, 1), "< 550 C", kind="constraint"))
        return out + self.spindle_stopped_quill_up()

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

    def _panel_holes(self):
        out = []
        for k, mk in enumerate(self.L["marks"]):
            o, h = self._through_hole_checks("cover_panel", mk["local"], 6.0, 0.001, f"hole {k}",
                                             burr_scored=SCORE_EXIT_BURR and SCORE_BACKED_EXIT)
            out += o
            if h is not None:
                # each stage's own wander from where its full diameter began (holes.py wander_m): a 6 mm drill
                # centred in a pilot that wandered cuts a round hole, off the mark (the entry check judges that)
                rnd = (h.d_mm + 2e3*h.wander_m)/h.d_mm
                out.append(_check(f"hole {k}: roundness at most 1.10", rnd <= 1.10, round(rnd, 4), "<= 1.10"))
        if SCORE_SHEET_GRAB:
            for n, ang in self.s["housing_turn_deg"].items():
                out.append(_check(f"{n} did not grab (housing turned < 30 deg)", ang < 30., round(ang, 2), "< 30 deg",
                                  kind="constraint"))
        return out

    def _timber_bolt(self):
        out = []
        mk = self.L["marks"][0]["local"]
        hit = self.nearest("board1", mk)
        if hit is None or not hit[0].through:
            out.append(_check("through hole in the top board", False))
        else:
            h, e = hit
            out.append(_check("through hole in the top board", True))
            out.append(_check("entry within 2 mm of the mark", e <= 0.002, round(e*1e3, 3), "<= 2 mm"))
            ang = _angle_deg(h.a, -h.face_normal)
            tol = P.TIMBER_AXIS_TOL_DEG
            out.append(_check(f"square to the faces (axis within {tol:.2f} deg of the face normal)", ang <= tol,
                              round(ang, 3), f"<= {tol:.2f} deg"))
            out.append(_check("no charred wall (< 200 C)", h.scorch_max_c < P.WOOD_SCORCH_C[0], round(h.scorch_max_c, 1),
                              "< 200 C"))
        b0 = [h for h in self.holes("board0") if h.through]
        out.append(_check("through hole in the bottom board", bool(b0), len(b0)))
        if b0 and SCORE_BACKED_EXIT:
            fd = max(h.exit_fd for h in b0)
            out.append(_check("exit damage factor at most 1.15", fd <= 1.15, round(fd, 3), "<= 1.15"))
        it = self.w.hm.items["carriage_bolt"]
        through = [dn for dn, v in it.slots.items() if v[1] == "hole"]
        out.append(_check("carriage bolt through both boards", {"board0", "board1"} <= set(through), sorted(through)))
        out.append(_check("bolt pushed in with less than 50 N", it.state["max_axial_n"] < 50.,
                          round(it.state["max_axial_n"], 2), "< 50 N", kind="constraint"))
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

    def _reached_bar(self, h):
        """The stopped hole reached the reinforcement: the hole module recorded steel under the bit, or the hole's floor
        lies within the bit's radius of a bar's surface (wall fields)."""
        if any(e.get("event") == "rebar_contact" for e in h.events):
            return True
        d = self.w.hm.drillables["wall"]
        if d.fields is None:
            return False
        floor = h.p0 + h.a*h.depth
        return d.fields.bar_distance(d.to_fields(floor))[0] <= 0.5e-3*h.d_mm

    def _rebar_relocate(self):
        """Every mark but the one over the bar meets the anchor-hole criteria; the hole over the bar was started, reached
        the bar and stopped without cutting it; a replacement on its channel line, at least ABORTED_HOLE_DISTANCE_FACTOR
        x the stopped hole's depth and at most RELOCATE_MAX_M from it (ETAG 001 Part 1 7.3: the aborted hole may stay
        unfilled at that distance), meets the anchor-hole criteria."""
        out = []
        kbar = self.L["rebar_mark"]
        for k, mk in enumerate(self.L["marks"]):
            if k != kbar:
                out += self._anchor_hole(mk, f"hole {k}")[0]
        m = self._wall_mark_local(self.L["marks"][kbar])
        hit = self.nearest("wall", m)
        aborted = hit[0] if hit is not None and hit[1] <= WALL_MARK_TOL_M else None
        out.append(_check(f"hole {kbar} (over the bar): started within {WALL_MARK_TOL_M*1e3:.0f} mm of its mark",
                          aborted is not None))
        if aborted is not None:
            out.append(_check(f"hole {kbar} (over the bar): reached the bar", self._reached_bar(aborted),
                              round(aborted.depth*1e3, 1), "steel under the bit"))
            out.append(_check(f"hole {kbar} (over the bar): the bar was not cut (damage at most 0.5 mm)",
                              aborted.rebar_damage_m <= 0.0005, round(aborted.rebar_damage_m*1e3, 3), "<= 0.5 mm"))
            d_min = P.ABORTED_HOLE_DISTANCE_FACTOR*aborted.depth
            others = [self._wall_mark_local(mk) for k, mk in enumerate(self.L["marks"]) if k != kbar]
            cands, ok = [], []
            for h in self.holes("wall"):
                if h is aborted:
                    continue
                dz = abs(float(h.p0[2] - aborted.p0[2]))
                dx = abs(float(h.p0[0] - aborted.p0[0]))
                if dz <= WALL_MARK_TOL_M and d_min <= dx <= P.RELOCATE_MAX_M and \
                        all(np.linalg.norm((h.p0 - o)[[0, 2]]) > 0.02 for o in others):
                    cands.append(h)
            for h in cands:
                mk2 = dict(x=float(self.w.hm.world_point(self.w.hm.drillables["wall"], h.p0)[0]),
                           z=float(self.w.hm.world_point(self.w.hm.drillables["wall"], h.p0)[2]))
                checks, _ = self._anchor_hole(mk2, "replacement")
                ok.append(all(c["passed"] for c in checks))
            out.append(_check(f"a replacement on the channel line {d_min*1e3:.0f}-{P.RELOCATE_MAX_M*1e3:.0f} mm from the "
                              "stopped hole meeting the anchor-hole criteria", any(ok), len(cands),
                              f"{P.ABORTED_HOLE_DISTANCE_FACTOR:g} x {aborted.depth*1e3:.1f} mm to "
                              f"{P.RELOCATE_MAX_M*1e3:.0f} mm"))
        return out

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

    # ------------------------------------------------------------------ ledger-anchor (revised task)
    def _ledger_marks(self):
        """Each mark where render.py draws it, on the ledger's front face, in the ledger's own frame (from its start
        pose, so a mark moves with the ledger), and that mark's present place in the wall's frame."""
        hm = self.w.hm
        lg, wall = hm.drillables["ledger"], hm.drillables["wall"]
        p0, R0 = self.s["start_pose"]["ledger"]
        marks_l, marks_w = [], []
        for mk in self.L["marks"]:
            m = np.asarray(R0).T @ (np.array([mk["x"], P.WALL_FACE_Y, mk["z"]]) - np.asarray(p0))
            m[1] = float(lg.center[1] - lg.half[1])
            marks_l.append(m)
            marks_w.append(hm.local(wall, hm.world_point(lg, m)))
        return marks_l, marks_w

    def _washer_gap(self, a, lg):
        """How far an anchor's washer bearing face stands off the ledger's front face (m; negative: pressed into it).
        The bearing face's centre as parts.WedgeAnchor finds it: the washer link plus half its thickness along the
        stud towards the tip."""
        w = self.w
        part = w.scene.parts[a.name]
        _, Rs = w.body_pose(part["body"])
        pw, _ = w.body_pose(part["washer"])
        c = pw + (Rs @ np.array([1., 0., 0.]))*0.5*P.WASHER_T_M
        pl, Rl = w.body_pose(lg.body)
        n = Rl @ np.array([0., -1., 0.])
        face = pl + Rl @ np.array([0., float(lg.center[1] - lg.half[1]), 0.])
        return float((c - face) @ n)

    def _ledger_anchor_in(self, k, h, anchors, lg):
        """The anchor checks of mark k's hole h in the concrete, and that anchor's progress terms."""
        w = self.w
        key = f"wall/{h.id}"
        T = P.KBTZ2["torque"]
        t_lo, t_hi = (1 - P.T_INST_TOL)*T, 1.7*T
        h_lo = P.KBTZ2["h_nom"] - 0.002
        band = f"{t_lo:.0f}-{t_hi:.0f} N m" if SCORE_OVER_TORQUE else f"at least {t_lo:.0f} N m"
        names = (f"hole {k}: anchor driven to h_nom (at least {h_lo*1e3:.0f} mm) with its washer on the ledger",
                 f"hole {k}: anchor straight (within 3 deg)", f"hole {k}: nut torqued to T_inst ({band})",
                 f"hole {k}: expansion engaged (cone drawn into a gripping clip)",
                 f"hole {k}: anchor did not spin (<= {P.ANCHOR_SPIN_MAX_DEG:.0f} deg)")
        clean_note = f"vacuum passes {len(h.vacuum_passes)}, brush strokes {len(h.brush_strokes)}, " \
                     f"dust column {h.dust_column_m()*1e3:.1f} mm"
        here = [a for a in anchors if tuple(a.slots.get("wall") or (None, None, None))[1:] == ("hole", h.id)]
        if not here:
            return [_check(names[0], False, None, f"at least {h_lo*1e3:.0f} mm",
                           note=f"no anchor in this hole (a record left by an anchor that has gone counts for "
                                f"nothing); {clean_note}"),
                    _check(names[2], False, None, "no anchor in the hole"),
                    _check(names[3], False, None, "no anchor in the hole")], dict(E=0., Q=0.)
        a = max(here, key=lambda x: x.state["insertion_m"].get(key, -1.))
        z = float(a.state["insertion_m"].get(key, 0.))
        z_drive = max(float(a.state.get("insertion_max_m", {}).get(key, z)), z)
        back_out = z < z_drive - 0.003
        gap = self._washer_gap(a, lg)
        out = [_check(names[0], z_drive >= h_lo and not back_out and gap <= WASHER_ON_FACE_M, round(z_drive*1e3, 2),
                      f"at least {h_lo*1e3:.0f} mm, washer at most {WASHER_ON_FACE_M*1e3:g} mm off the ledger",
                      note=f"{a.name}: driven to {z_drive*1e3:.2f} mm, now at {z*1e3:.2f} mm; washer bearing face "
                           f"{gap*1e3:+.2f} mm off the ledger's face; {clean_note}")]
        _, R = w.body_pose(a.body)
        phi = _angle_deg(R @ np.array([1., 0., 0.]), w.hm.world_dir(w.hm.drillables["wall"], h.a))
        out.append(_check(names[1], phi <= 3., round(phi, 2), "<= 3 deg"))
        st = a.state.get("setting", {})
        here_clip = (a.state.get("clip") or {}).get("key") == key
        tq = float(st.get("max_turn_torque_nm", 0.)) if here_clip else 0.
        tq_ever = max(float(self.s.get("anchor_max_turn_nm", {}).get(a.name, 0.)),
                      float(st.get("max_turn_torque_nm", 0.)))
        over = tq_ever > t_hi
        out.append(_check(names[2], t_lo <= tq and (not over or not SCORE_OVER_TORQUE), round(tq, 2), band,
                          note=f"largest friction torque the nut was turned on against with the cone gripping; the "
                               f"anchor's largest ever {tq_ever:.1f} N m, over-torque level {t_hi:.0f} N m "
                               f"{'EXCEEDED' if over else 'not reached'}"
                               f"{'' if SCORE_OVER_TORQUE else ' (reported, not scored)'}; stud tension: largest "
                               f"{st.get('max_set_n', 0.):.0f} N, now {st.get('thread_n', 0.):.0f} N"
                               f"{'; ANCHOR FAILED (' + str(st.get('failed')) + ')' if st.get('failed') else ''}"))
        engaged = bool(here_clip and st.get("grip") and st.get("cone_n", 0.) > 0.)
        out.append(_check(names[3], engaged, round(float(st.get("s_m", 0.))*1e3, 3) if here_clip else None,
                          "cone drawn in, clip gripping", note=f"cone load {st.get('cone_n', 0.):.0f} N"))
        spin = float(st.get("spin_max_deg", 0.))
        out.append(_check(names[4], spin <= P.ANCHOR_SPIN_MAX_DEG, round(spin, 2), f"<= {P.ANCHOR_SPIN_MAX_DEG:.0f} deg",
                          kind="constraint"))
        # progress terms: the embedment reached (its present depth if it has been drawn back out); the installation
        # torque counted only while the cone grips in this hole (hammering alone turns no nut: no credit) and only for
        # the embedment reached, falling to nothing 2 mm short of h_lo (a torque record means the
        # washer bears on the ledger and the self-locking nut has run down the stud, so a nut tightened short cannot be
        # driven further, and a further drive would reset the record: torque recorded short is never part of a finished
        # job); nothing for an anchor that can no longer be set (stud failed, taken past 1.7 T_inst, or spun: no spare
        # anchor, no reverse on the wrench, and the spin record only grows)
        short = _ramp(h_lo - z_drive, 0., 0.002)        # 1 at h_lo and beyond, 0 two millimetres short
        E = min((z if back_out else z_drive)/h_lo, 1.)*_ramp(phi, 3., 6.)*(short if tq > 0. else 1.)
        Q = min(tq/t_lo, 1.)*short if engaged else 0.
        spun = spin > P.ANCHOR_SPIN_MAX_DEG
        spoiled = bool(st.get("failed")) or over or spun
        if spoiled:
            E = Q = 0.
        return out, dict(anchor=a.name, driven_mm=round(z_drive*1e3, 2), now_mm=round(z*1e3, 2),
                         washer_gap_mm=round(gap*1e3, 3), torque_nm=round(tq, 2), largest_torque_ever_nm=round(tq_ever, 2),
                         spoiled=spoiled, spun=spun, set_short=bool(tq > 0. and z_drive < h_lo), E=E, Q=Q)

    def _ledger_openings(self, k, marks_l, marks_w, anchors, lg):
        """The openings at mark k: each wall record within MARK_RADIUS_M of the mark's
        place on the wall, with the ledger record in line with it (its axis passing within one bit radius of the wall
        record's entry; a through record first, then the nearest the mark), or alone; and each ledger record within the
        radius that has no wall record in line, alone. The hole module can split one physical opening into several
        records (a bit coming back more than CAPTURE_TOL off a started hole's axis seats a new one), so the mark's bore
        is the opening that carries the most work, not the record nearest the mark. Returns [(ledger rec, offset) or
        None, (wall rec, offset) or None]."""
        w, hm = self.w, self.w.hm
        wall = hm.drillables["wall"]
        led = [(h, _in_plane(h, marks_l[k])) for h in self.holes("ledger")]
        led = [x for x in led if x[1] <= MARK_RADIUS_M]
        wal = [(h, _in_plane(h, marks_w[k])) for h in self.holes("wall")]
        wal = [x for x in wal if x[1] <= MARK_RADIUS_M]
        out, paired = [], set()
        for hw in wal:
            q = hm.world_point(wall, hw[0].p0)
            inline = []
            for hl in led:
                p, a = hm.world_point(lg, hl[0].p0), hm.world_dir(lg, hl[0].a)
                r = q - p
                if float(np.linalg.norm(r - a*float(r @ a))) <= 0.5e-3*hw[0].d_mm:
                    inline.append(hl)
            hl = min(inline, key=lambda x: (not x[0].through, x[1])) if inline else None
            if hl is not None:
                paired.add(hl[0].id)
            out.append((hl, hw))
        out += [(hl, None) for hl in led if hl[0].id not in paired]
        return out

    def _ledger_anchor(self):
        """The job: fasten the ledger, set level on its temporary brackets, to the wall with the four KB-TZ2 anchors
        through it at its marks. All read from physical state; for each mark k the
        verifier judges, and the score credits, the same opening: of the openings at the mark (_ledger_openings), the
        one that carries the most work (largest M_k below; ties: through the ledger, then deeper, then nearer):
          - through the ledger, its entry within WALL_MARK_TOL_M of the mark, which is where it is drawn: on the
            ledger's face in the ledger's frame;
          - in the concrete behind it, its axis within WALL_AXIS_TOL_DEG of the wall's normal, 108-125 mm deep to its
            DEEPEST point ('depth of drill hole to deepest point', fischer ETA-19/0520 Table B2.1). 108 mm
            is the anchor's h0; 125 mm is the site's limit, which keeps the hole in front of the far mat's bars, whose
            near surface is at 149.2 mm in this 200 mm wall;
          - the anchor whose clip is in that concrete hole now (a record left by an anchor that has gone counts for
            nothing): driven to at least h_nom - 2 mm (the deepest its tip stayed, params.DRIVEN_DWELL),
            not drawn back out by more than 3 mm, its washer on the ledger (bearing face at most WASHER_ON_FACE_M off
            the ledger's face; no upper bound: driving further needs the timber to crush, which is not modelled);
            straight; the nut turned on against at least 0.9 T_inst while the
            cone gripped and never past 1.7 T_inst over the episode (_track_anchor_torque); expansion
            engaged; not spun.
        Every other drilled record is one of (MARK_RADIUS_M): a pit shallower than the bit's point; a re-entry (entry
        within one bit radius of the judged opening's record in the same part: the same opening); a
        start in the ledger the anchor's washer covers; a blind start in the ledger elsewhere (reported and charged in
        progress, not failed: surface damage a carpenter fills); or a hole that fails the job (through the
        ledger beside or away from a mark's opening, or in the concrete). The ledger fastened where it was set (centre
        within LEDGER_SHIFT_MAX_M of its start pose in the wall's plane) and still level; no bar cut; the bit
        not chipped. The rated payload is a diagnostic (PAYLOAD_DIAGNOSTIC_TASKS). No cleaning sequence, tool, order or
        hand is judged.

        Progress v3: for each mark M_k = p_k (B_k + L S (E_k + Q_k))/3, each mark
        worth a quarter whatever the others' state; progress = max(0, mean M_k - n/4 - sum c), where
          p_k = ramp(entry offset; 3, 6 mm) x ramp(axis; 3, 6 deg): placement
          B_k = min((ledger bored + concrete depth)/(38 + 108 mm), 1) x ramp(concrete depth; 125 mm, the far mat's
                near surface): the bore, to its deepest point
          E_k = min(driven depth/93 mm, 1) x ramp(anchor tilt; 3, 6 deg), and x ramp(93 mm - driven depth; 0, 2 mm) once
                a torque is recorded (an anchor tightened short can no longer be driven)
          Q_k = min(installation torque/61.2 N m, 1) x ramp(93 mm - driven depth; 0, 2 mm) while the cone grips here
          L   = ramp(ledger drop over its length; 3, 6 mm), S = ramp(ledger offset in the wall's plane; 10, 20 mm): the
                fastening holds the ledger as it was set; bores keep their value
          M_k = 0 if the opening cut a bar; E_k = Q_k = 0 for a spoiled anchor (stud failed, over-torqued or spun)
          n   = bars cut by records that are not a mark's opening; c = each other defect record's share of a bore:
                min(depth, 38 mm)/146 mm/12 in the ledger, min(depth, 108 mm)/146 mm/12 in the concrete (a whole stray
                bore costs one operation's 1/12)
        ramp(x; a, b) is 1 up to the acceptance limit a and falls linearly to 0 at b. Every factor is 1 inside
        acceptance and no cost applies to a success, so a success has progress 1. `damage` lists every record's class
        and cost."""
        w, hm = self.w, self.w.hm
        lg = hm.drillables["ledger"]
        anchors = [hm.items[f"anchor{k}"] for k in range(4)]
        marks_l, marks_w = self._ledger_marks()
        tol, ax_tol, R_mark = WALL_MARK_TOL_M, WALL_AXIS_TOL_DEG, MARK_RADIUS_M
        d_lo, d_hi, t_led = P.KBTZ2["h0"], 0.125, P.LEDGER_THICK_M
        far_mat = P.WALL_THICK_M - P.FAR_MAT_COVER_M - P.REBAR_D_M          # near surface of the far mat's bars
        depth_name = f"depth {d_lo*1e3:.0f}-{d_hi*1e3:.0f} mm into the concrete (to its deepest point)"
        p_lg, R_lg = w.body_pose(w.scene.parts["ledger"]["body"])
        p0_lg = np.asarray(self.s["start_pose"]["ledger"][0])
        drop = abs(float((R_lg @ np.array([1., 0., 0.]))[2]))*1.2
        shift = float(np.hypot(*(p_lg - p0_lg)[[0, 2]]))                    # in the wall's plane (x along, z up)
        L, S = _ramp(drop, 0.003, 0.006), _ramp(shift, LEDGER_SHIFT_MAX_M, 2*LEDGER_SHIFT_MAX_M)
        out, rows, judged = [], [], {}
        damage = dict(stray_holes=[], extra_holes_at_marks=[], blind_starts=[], misplaced_bores=[], rebar_cut=[],
                      spoiled_anchors=[], spun_anchors=[], set_short=[], pits=[], covered_starts=[], re_entries=[])

        def opening(k, hL, hW):
            """(checks, progress row, M) of one opening at mark k."""
            lab, chk = f"hole {k}", []
            if hL is None:
                chk.append(_check(f"{lab}: through the ledger at the mark", False, None,
                                  note=f"no hole in the ledger within {R_mark*1e3:.0f} mm of the mark"))
            else:
                chk.append(_check(f"{lab}: through the ledger at the mark", hL[0].through,
                                  "through" if hL[0].through else f"{hL[0].depth*1e3:.1f} mm deep"))
                chk.append(_check(f"{lab}: within {tol*1e3:.0f} mm of the mark on the ledger", hL[1] <= tol,
                                  round(hL[1]*1e3, 2), f"<= {tol*1e3:.0f} mm"))
            terms = dict(E=0., Q=0.)
            if hW is None:
                chk.append(_check(f"{lab}: {depth_name}", False, None, f"{d_lo*1e3:.0f}-{d_hi*1e3:.0f} mm",
                                  note="no hole in the concrete behind the mark"))
                chk.append(_check(f"{lab}: anchor driven to h_nom (at least {(P.KBTZ2['h_nom'] - 0.002)*1e3:.0f} mm) "
                                  f"with its washer on the ledger", False, None, note="no hole"))
            else:
                h = hW[0]
                ang = _angle_deg(h.a, -h.face_normal)
                chk.append(_check(f"{lab}: axis within {ax_tol:.0f} deg of the wall normal", ang <= ax_tol, round(ang, 2),
                                  f"<= {ax_tol:.0f} deg"))
                chk.append(_check(f"{lab}: {depth_name}", d_lo <= h.depth <= d_hi, round(h.depth*1e3, 1),
                                  f"{d_lo*1e3:.0f}-{d_hi*1e3:.0f} mm", note=f"full-diameter depth {h.full_depth()*1e3:.1f} mm"))
                o, terms = self._ledger_anchor_in(k, h, anchors, lg)
                chk += o
            row, M, key = dict(mark=k, ledger_hole=None if hL is None else hL[0].id,
                               wall_hole=None if hW is None else hW[0].id), 0., (0., False, 0., 0.)
            if hL is not None or hW is not None:
                e = hL[1] if hL is not None else hW[1]
                hx = hW[0] if hW is not None else hL[0]
                th = _angle_deg(hx.a, -hx.face_normal)
                p = _ramp(e, tol, 2*tol)*_ramp(th, ax_tol, 2*ax_tol)
                ell = 0. if hL is None else (t_led if hL[0].through else min(hL[0].depth, t_led))
                z = 0. if hW is None else hW[0].depth
                B = min((ell + z)/(t_led + d_lo), 1.)*_ramp(z, d_hi, far_mat)
                cut = hW is not None and hW[0].rebar_damage_m > 0.0005
                M = 0. if cut else p*(B + L*S*(terms["E"] + terms["Q"]))/3.
                row.update(entry_offset_mm=round(e*1e3, 2), axis_deg=round(th, 2), placement=round(p, 4),
                           ledger_bored_mm=round(ell*1e3, 1), concrete_depth_mm=round(z*1e3, 1), bore=round(B, 4),
                           cut=cut, misplaced=p == 0.)
                key = (round(M, 9), hL is not None and hL[0].through, ell + z, -e)
            row.update({x: (round(v, 4) if isinstance(v, float) else v) for x, v in terms.items()}, M=M)
            return chk, row, key

        for k in range(len(marks_l)):
            cands = [opening(k, hL, hW) for hL, hW in self._ledger_openings(k, marks_l, marks_w, anchors, lg)]
            chk, row, _ = max(cands, key=lambda c: c[2]) if cands else opening(k, None, None)
            out += chk
            judged[k] = (None if row["ledger_hole"] is None else next(h for h in self.holes("ledger") if h.id == row["ledger_hole"]),
                         None if row["wall_hole"] is None else next(h for h in self.holes("wall") if h.id == row["wall_hole"]))
            if row.pop("misplaced", False):
                damage["misplaced_bores"].append(k)
            if row.pop("cut", False):
                damage["rebar_cut"].append(dict(mark=k, hole=row["wall_hole"]))
            for flag, dk in (("spoiled", "spoiled_anchors"), ("spun", "spun_anchors"), ("set_short", "set_short")):
                if row.get(flag):
                    damage[dk].append(row["anchor"])
            rows.append(row)
        walls = [h.rebar_damage_m for h in self.holes("wall")]
        out.append(_check("no reinforcement cut (damage at most 0.5 mm in every hole)", max(walls, default=0.) <= 0.0005,
                          round(max(walls, default=0.)*1e3, 3), "<= 0.5 mm", kind="constraint"))
        # every other drilled record
        n_cut, cost, cut_at = 0, 0., []          # wall entries (world) of bars cut off a mark's opening
        for name, pts, side, full in (("wall", marks_w, 1, d_lo), ("ledger", marks_l, 0, t_led)):
            for h in self.holes(name):
                if any(j[side] is h for j in judged.values()):
                    continue
                dists = [_in_plane(h, m) for m in pts]
                k = int(np.argmin(dists))
                rec = dict(workpiece=name, hole=h.id, from_nearest_mark_mm=round(dists[k]*1e3, 1), mark=k,
                           depth_mm=round(h.depth*1e3, 2), through=h.through)
                if h.full_depth() <= 0.:
                    damage["pits"].append(rec)
                    continue
                bore = judged[k][side] if dists[k] <= R_mark else None
                if bore is not None:
                    gap = _in_plane(h, bore.p0)
                    rec["from_bore_mm"] = round(gap*1e3, 1)
                    if gap <= 0.5e-3*h.d_mm:
                        damage["re_entries"].append(rec)
                        continue
                    lbore = judged[k][0]
                    if side == 0 and lbore is not None and _in_plane(h, lbore.p0) + 0.5e-3*h.d_mm <= P.WASHER_R_M:
                        damage["covered_starts"].append(rec)
                        continue
                inline_cut = side == 0 and any(
                    float(np.linalg.norm((q - hm.world_point(lg, h.p0)) - hm.world_dir(lg, h.a)*float(
                        (q - hm.world_point(lg, h.p0)) @ hm.world_dir(lg, h.a)))) <= 0.5e-3*h.d_mm for q in cut_at)
                if side == 1 and h.rebar_damage_m > 0.0005:
                    n_cut += 1
                    rec["cost"] = 0.25
                    cut_at.append(hm.world_point(hm.drillables["wall"], h.p0))
                elif inline_cut:
                    rec["cost"] = 0.                     # the bore that cut a bar costs its quarter once
                else:
                    rec["cost"] = round(min(h.depth, full)/(t_led + d_lo)/12., 5)
                    cost += rec["cost"]
                if side == 0 and not h.through:
                    damage["blind_starts"].append(rec)
                else:
                    damage["extra_holes_at_marks" if bore is not None else "stray_holes"].append(rec)
        bad = len(damage["stray_holes"]) + len(damage["extra_holes_at_marks"])
        out.append(_check("no hole drilled away from the marks or beside a mark's bore", not bad, bad, "0",
                          kind="constraint", note=f"{len(damage['stray_holes'])} away from every mark, "
                                                  f"{len(damage['extra_holes_at_marks'])} beside a mark's bore; not "
                                                  f"failing: {len(damage['blind_starts'])} blind starts in the ledger, "
                                                  f"{len(damage['pits'])} pits, {len(damage['covered_starts'])} starts "
                                                  f"under a washer, {len(damage['re_entries'])} re-entries"))
        out.append(_check(f"ledger fastened where it was set (moved at most {LEDGER_SHIFT_MAX_M*1e3:.0f} mm in the "
                          f"wall's plane)", shift <= LEDGER_SHIFT_MAX_M, round(shift*1e3, 2),
                          f"<= {LEDGER_SHIFT_MAX_M*1e3:.0f} mm", kind="constraint"))
        out.append(_check("ledger still level within 3 mm over its length", drop <= 0.003, round(drop*1e3, 2), "<= 3 mm",
                          kind="constraint"))
        out.append(self._bit_not_chipped())
        damage.update(ledger_drop_mm=round(drop*1e3, 2), ledger_offset_mm=round(shift*1e3, 2), bars_cut_off_bore=n_cut,
                      defect_cost=round(cost, 5),
                      bit_chipped=[n for n, it in hm.items.items() if it.kind == "bit" and it.state["chipped"]],
                      bit_broken=[n for n, it in hm.items.items() if it.kind == "bit" and it.state["broken"]])
        progress = max(0., sum(r["M"] for r in rows)/len(rows) - 0.25*n_cut - cost)
        for r in rows:
            r["M"] = round(r["M"], 4)
        return out, dict(progress=round(progress, 4), scoring=LEDGER_SCORING,
                         progress_terms=dict(level_factor=round(L, 4), position_factor=round(S, 4), marks=rows),
                         damage=damage)
