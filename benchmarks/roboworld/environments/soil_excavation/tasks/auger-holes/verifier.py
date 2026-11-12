"""Outcome of auger-holes: five holes, each at least 18 cm deep, 70-95 mm wide at 5 cm depth and centred within 20 mm of
its mark (released: the drill still held; the task contracts: every auger part back above grade).

verify(e, lg, final=True, contract=None) reads the episode for the evaluator e from its particle ledger lg
(task.Ledger). Without a contract it returns the released judgement, (criteria, progress parts) as task.Evaluator.judge
takes it; with contract soil-tasks-v1 or soil-tasks-v2 it returns that contract's reading, (criteria, raw progress,
diagnostics) as task_set.py's and task_set_v2.py's TaskSetEvaluator take it."""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from soil_excavation import layout as L
from soil_excavation.task import _pose, crit, tool_held
from soil_excavation.task_set_v2 import band, clip, ramp


def verify(e, lg, final=True, contract=None):
    if contract is None:
        return _auger_holes(e, lg, final)
    return read(e, lg)


def _auger_holes(e, lg, final):
    res = []
    for k, t in enumerate(e.env.spec["targets"]):
        t = np.asarray(t)
        rect = ((t[0] - 0.08, t[1] - 0.08), (t[0] + 0.08, t[1] + 0.08))
        top, org = lg.heightfield(rect, cell=0.005)
        d, X, Y = lg.depth_at(top, org, t, 0.08)
        deep = d >= 0.05
        dia = 2*math.sqrt(float(deep.sum())*org[2]**2/math.pi)
        if deep.any():
            cx, cy = float(np.mean(X[deep])), float(np.mean(Y[deep]))
            off = math.hypot(cx - t[0], cy - t[1])
            dc, _, _ = lg.depth_at(top, org, (cx, cy), 0.015)
            depth = float(np.mean(dc))
        else:
            off, depth = float("nan"), 0.0
        res.append(dict(depth=depth, dia=dia, off=off,
                        ok=depth >= 0.18 and 0.07 <= dia <= 0.095 and np.isfinite(off) and off <= 0.02))
    held = tool_held(e.env, "drill")
    n_ok = sum(r["ok"] for r in res)
    c = [crit(f"hole_{k + 1}", dict(depth=round(r["depth"], 3), diameter=round(r["dia"], 3),
                                    offset=None if not np.isfinite(r["off"]) else round(r["off"], 3)), r["ok"],
              ">= 0.18 m deep, 70-95 mm wide at 5 cm, within 20 mm of the mark") for k, r in enumerate(res)]
    c.append(crit("drill_held", held, all(held.values()), "both hands on the drill"))
    return c, [n_ok/len(res), float(all(held.values()))]


# soil-tasks-v1's and soil-tasks-v2's reading (the same in task_set.py and task_set_v2.py)
def read(e, lg):
    c, credits = [], []
    for k, mark in enumerate(e.env.spec["targets"]):
        x, y = mark
        top, org = lg.heightfield(((x - .08, y - .08), (x + .08, y + .08)), cell=.005)
        depths, X, Y = lg.depth_at(top, org, mark, .08)
        deep = depths >= .05
        diameter = 2*math.sqrt(float(deep.sum())*org[2]**2/math.pi)
        if deep.any():
            cx, cy = float(X[deep].mean()), float(Y[deep].mean())
            off = math.hypot(cx - x, cy - y)
            dc, _, _ = lg.depth_at(top, org, (cx, cy), .015)
            depth = float(dc.mean())
        else:
            off, depth = float("inf"), 0.
        ok = depth >= .18 and .07 <= diameter <= .095 and off <= .02
        c.append(crit(f"hole_{k + 1}", dict(depth=depth, diameter=diameter, offset=off if np.isfinite(off) else None),
                      ok, ">= 0.18 m deep, 70-95 mm wide at 5 cm, within 20 mm of the mark"))
        credits.append(min(clip(depth/.18), band(diameter, .07, .095, .04, .14), ramp(off, .02, .06)))
    p, r = _pose(e.env, "auger")
    clearance = min(part_lowest_z(part, p, r) for part in e.env.world.scene.parts["auger"]) - L.GRADE
    c.append(crit("auger_clear", clearance, clearance >= 0., "every auger/spindle part at or above grade", "m"))
    raw = float(np.mean(credits))*(.9 + .1*clip(1. + clearance/.05))
    return c, raw, []


def part_lowest_z(part, position, rotation):
    """Exact support height of an authored box/cylinder/capsule/mesh part.

    Use scene.parts (already transformed into the spindle frame), not an assumed
    vertical auger length: a tilted flight and the tip both have to clear grade.
    """
    r = rotation @ Rotation.from_quat(part.quat).as_matrix()
    p = np.asarray(position) + rotation @ np.asarray(part.pos)
    if part.kind == "mesh":
        return float((p + np.asarray(part.mesh[0]) @ r.T)[:, 2].min())
    if part.kind == "box":
        return float(p[2] - np.abs(r[2]) @ np.asarray(part.size))
    radius, half = part.size[:2]
    radial = radius if part.kind == "capsule" else radius*math.sqrt(max(0., 1. - r[2, 2]**2))
    if part.kind not in ("capsule", "cylinder"):
        raise ValueError(f"Unsupported auger geometry {part.kind}")
    return float(p[2] - abs(r[2, 2])*half - radial)
