"""label-tasks-v2: the render-checked eight-task set, selected with `"contract": "label-tasks-v2"`.

The render check ran the eight tasks of the released set (variants.TASK_SET) unchanged: the same
catalogue tasks, start states, instructions, verifiers and reward contracts (task.REWARD_CONTRACTS). The review that
went with it changed no task code; it only took three environment changes back out of its branch (the pad-bond
adhesive-side rule in labelsim.py, refusing commands after a tear, the PhysicsTraceStore trace). Those stay as
released here, under v2 as under v1.

So v2 adds no task behaviour of its own. A v2 selection resolves through the released resolver to the same
description and the same backend options (preset, task, seed), so the session runs the released code path; the
contract is carried in the selection and the task id. tube-label, the audited hardest task, keeps its released legacy
selection (splits/test.json) and takes no v2 contract. Selections without a contract are not routed here
(variants.task_resolve).
"""
from .variants import TASK_SET, task_resolve as released_task_resolve

CONTRACT = "label-tasks-v2"
AUDITED = "tube-label"
TASKS = tuple(s["task"] for s in TASK_SET if s["task"] != AUDITED)
# The eight exact selections, easiest first, as benchmark.json task_set lists them.
SELECTIONS = tuple({**s, "contract": CONTRACT} for s in TASK_SET if s["task"] != AUDITED) + (
    dict(preset="specimen", task=AUDITED),)


def task_resolve(selection):
    contract = selection.get("contract")
    if contract != CONTRACT:
        raise ValueError(f"Unknown contract {contract!r}; choose {CONTRACT!r}, or omit it for the released catalogue")
    out = released_task_resolve({k: v for k, v in selection.items() if k != "contract"})
    v = out["selection"]
    if v["task"] not in TASKS:
        raise ValueError(f"{CONTRACT} covers {list(TASKS)}; {v['task']} has no {CONTRACT} contract "
                         "(omit contract for its released selection)")
    v = dict(v, contract=CONTRACT)
    return dict(out, task_id=f"label-application/v1/{v['preset']}/{v['task']}/{CONTRACT}/seed-{v['seed']}",
                selection=v)
