"""Outcome of take-drill: the drill held by the robot's two hands alone, one on its grip and one on its motor housing,
for 3 s, its bit pointing straight down with the tip 30 to 70 mm above the bench; the trigger never pressed past 1 mm
and the bit touching nothing.

verify(ev) receives the episode's evaluator (task.Evaluator, or a contract grader built on it: catalogue_evaluator,
task_set_v2_evaluator) and returns its evaluation dict (Evaluator.evaluate_checks): checks(ev) below, and under
drilling-tasks-v1 or -v2 the continuous progress(ev) as well. track_claims(ev) records the hold and where the hands
are, every tenth tick (Evaluator.tick)."""
import math

import numpy as np

from drilling import params as P
from drilling.catalogue_evaluator import clip, mean
from drilling.task import _angle_deg, _check, _ramp


def verify(ev):
    return ev.evaluate_checks(checks, progress=progress)


def track_claims(ev):
    w = ev.w
    drill = w.scene.tools["drill"]
    s = ev.s
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
    on = finger_forces_on(ev, drill, ("grip", "head"))
    arms = list(on)
    placed = len(arms) == 2 and any(on[a]["grip"] > 0.2 and on[b]["head"] > 0.2 for a in arms for b in arms if a != b)
    if placed:
        if s.get("placed_since") is None:
            s["placed_since"] = w.sim_time
    else:
        s["placed_since"] = None
    s["hands_on_n"] = {a: {k: round(v, 2) for k, v in f.items()} for a, f in on.items()}


def finger_forces_on(ev, tool, parts):
    """{arm: {part: normal force (N) of that arm's finger links on the tool's shape labelled <tool>_<part>}}."""
    w = ev.w
    if not hasattr(ev, "_shape_ids"):
        lab = list(w.scene.b.shape_label)
        ev._shape_ids = {part: {i for i, l in enumerate(lab) if l == f"{tool.name}_{part}"} for part in parts}
    m, d = w.solver.mj_model, w.solver.mj_data
    gs = w._geom_shape
    fingers = {arm: set(w._arm_fingers[arm]) for arm in w.rig.instances}
    out = {arm: {part: 0. for part in parts} for arm in fingers}
    f6 = np.zeros(6)
    for i in range(d.ncon):
        g = d.contact.geom[i]
        sh = [int(gs[x]) if x < len(gs) else -1 for x in g]
        bodies = [int(m.geom_bodyid[x]) for x in g]
        for part, ids in ev._shape_ids.items():
            for j in (0, 1):
                if sh[j] in ids:
                    for arm, fs in fingers.items():
                        if bodies[1 - j] in fs:
                            w._mujoco.mj_contactForce(m, d, i, f6)
                            out[arm][part] += abs(float(f6[0]))
    return out


def checks(ev):
    w, s = ev.w, ev.s
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


def progress(ev):
    s, w = ev.s, ev.w
    held = 0. if s["held_since"] is None else clip((w.sim_time - s["held_since"])/3.)
    placed = 0. if s["placed_since"] is None else clip((w.sim_time - s["placed_since"])/3.)
    it = w.scene.tools["drill"].item
    if it.state["broken"]:
        return 0.
    tip, axis = w.hm.item_world(it)
    angle = math.degrees(math.acos(float(np.clip(-axis[2]/np.linalg.norm(axis), -1., 1.))))
    height = tip[2] - P.BENCH_TOP_Z
    height_credit = min(clip(height/0.030), _ramp(height, 0.070, 0.150))
    # Fade geometry credit with the hold, so lifting/contact onset does not jump the score.
    return mean((held, placed, held*_ramp(angle, 5., 25.), held*height_credit))
