"""Outcome of head-off: the head cut off behind the gill cover through a joint of the backbone, at most 8 g of fillet
muscle with it, the fish still on the board.

verify(e) evaluates an episode and returns its gar.evaluation.v1 result; e is the episode's evaluator: task.Evaluator
for the released checks (released_checks), or the task-set evaluator of filleting-tasks-v1 or -v2
(task_set_evaluator.py, task_set_v2_evaluator.py), whose cutting grading (_cut_task) reads PRODUCT, released_checks,
product and progress below; KNIFE_PUT_DOWN lets the -v2 grading also accept the knife put down safely."""
import numpy as np

from filleting import layout as LY
from filleting.task import NOTHING_MADE, REF, TIME_LIMIT_S, check
from filleting.task_set_progress import ramp

PRODUCT = "separation"
KNIFE_PUT_DOWN = True


def verify(e):
    if e.contract is None:
        return e.evaluate_checks(released_checks)
    return e.evaluate_graded()


def released_checks(e):
    env, w = e.env, e.env.world
    lab, frags = e.fragments()
    attach = e.bone_attachment(lab)
    hd = e.head_fragment(attach)
    fr = e.frame_fragment(lab, frags, attach)
    ts = w.tissue
    # the backbone between the head and the frame is cut at some joint: the joints head->segment_00,
    # segment_00->segment_01, ... up to the frame's first segment (the anterior vertebrae have a joint each, so the
    # heading cut may take one or more of them with the head)
    on = ts.joints["on"].numpy()
    jname = {jd.get("name", ""): j for j, jd in enumerate(ts.joint_desc)}
    nk = w.assembly["n_knife_bodies"]
    names = [rb["name"] for rb in w.workpiece.rigid_bodies]
    segs = sorted(n for n in names if n.startswith("segment_"))
    def tied_to(n):
        a_ = attach.get(nk + names.index(n), {})
        return max(a_, key=a_.get) if a_ else None
    first = next((k for k, n in enumerate(segs) if fr is not None and tied_to(n) == fr), len(segs) - 1)
    chain = ["head->segment_00"] + [f"{segs[k - 1]}->{segs[k]}" for k in range(1, first + 1)]
    cut_joints = [c for c in chain if c in jname and not on[jname[c]]]
    head_joint_on = not cut_joints
    separated = (not head_joint_on) and (hd is None or hd != fr)
    meat = 0.
    if hd is not None and hd != fr:
        meat = float(frags[hd]["by_ref"][REF["fillet_upper"]] + frags[hd]["by_ref"][REF["fillet_lower"]])
    on_board = e.in_footprint(ts.x.numpy()[e.fragment_nodes(frags[fr])], LY.BOARD).mean() if fr is not None else 0.
    checks = dict([check("head_separated", float(separated), 1., "bool", separated,
                         f"backbone cut at {', '.join(cut_joints) or 'no joint'}; no flesh path from the head to "
                         "the body" if hd != fr else "the head's flesh is joined to the body"),
                   check("meat_with_head", meat*1e3, 8., "g of fillet muscle behind the collar line",
                         separated and meat <= 8e-3, "" if separated else NOTHING_MADE),
                   check("fish_on_board", on_board, .95, "fraction of body nodes over the board", on_board >= .95)])
    e._common_knife(checks)
    dm = e._common_cut(checks, frags)
    checks["time"] = check("time", env.sim_time, TIME_LIMIT_S["A1"], "s", env.sim_time <= TIME_LIMIT_S["A1"])[1]
    return checks, dict(**dm, head_fragment=hd, frame_fragment=fr, fragments=len(frags), meat_with_head_kg=meat)


def product(e, ref, reg):
    """The separation's terminals (source, target): the head's tets and body, and the fillets' tets more than two mesh
    lengths from the head with the posterior vertebra."""
    w, nt = e.env.world, e.graph.nt
    source = np.flatnonzero(ref == REF["head"])
    candidates = np.flatnonzero((ref == REF["fillet_upper"]) | (ref == REF["fillet_lower"]))
    target = e._targets(source, candidates)
    names = [b["name"] for b in w.workpiece.rigid_bodies]
    source = np.append(source, nt + w.assembly["n_knife_bodies"] + names.index("head"))
    # Include the posterior vertebra so an intact backbone cannot vanish from the graph.
    segs = sorted(n for n in names if n.startswith("segment_"))
    if segs:
        target = np.append(target, nt + w.assembly["n_knife_bodies"] + names.index(segs[-1]))
    return source, target


def progress(e, checks, metrics, edges):
    """(raw progress, remaining connections): the head's separation, then the meat it leaves on the body."""
    separated = checks["head_separated"]["value"]
    sever, remaining = e._sever(separated, edges)
    raw = .8*sever + .2*separated*ramp(metrics["meat_with_head_kg"]*1000, 8., 30.)*checks["fish_on_board"]["value"]
    return raw, remaining
