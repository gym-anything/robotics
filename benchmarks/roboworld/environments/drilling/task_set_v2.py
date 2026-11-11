"""drilling-tasks-v2: the render-checked eight-task catalogue (fix round of the drilling-tasks-v1 set).

What differs from drilling-tasks-v1 (drilling/catalogue.py, unchanged):
- the spec's rebar-relocate is task 7 again (wall preset), where v1 swapped in match-bolt. Its success is the legacy
  verdict, rated payload included; v2 adds its continuous progress (drilling/task_set_v2_evaluator.py). match-bolt
  keeps v2 scoring outside the eight, as the prepared swap;
- every instruction is the legacy text: timber-bolt keeps "(within 1.2 degrees)", the bolt's own passing limit
  (params.TIMBER_AXIS_TOL_DEG = atan(1.59 mm / 76.2 mm)), which v1 dropped.
Legacy selectors, start states and the audited ledger-anchor stay unchanged. drilling.catalogue.task_resolve hands a
selection here only when it names drilling-tasks-v2; NewtonDrillingCatalogueBackend builds the v2 session class
(drilling/task_set_v2_backend.py) only for a launch that names it.
"""

from . import variants as legacy
from .task import INSTRUCTIONS as LEGACY_INSTRUCTIONS

CONTRACT = "drilling-tasks-v2"
LEGACY = "v1"
TASKS = ("take-drill", "press-drill", "punch-layout", "press-blind", "panel-holes",
         "timber-bolt", "rebar-relocate", "ledger-anchor")
WALL = frozenset(("rebar-relocate", "ledger-anchor"))
# match-bolt is outside the eight (the spec's set has rebar-relocate) but keeps its v2 scoring: it is the prepared swap
# if the owner rules that rebar-relocate's rated-payload constraint blocks every robot solution.
ALTERNATES = ("match-bolt",)
PROGRESS_SCORED = frozenset(TASKS[:-1] + ALTERNATES)
NO_SCORE_WHEN_UNEVALUABLE = PROGRESS_SCORED
DIFFICULTY = dict(zip(TASKS, ("entry", "entry", "medium", "medium", "medium", "hard", "hard", "hardest")))
# Every instruction is the legacy text (the spec keeps them all).
INSTRUCTIONS = dict(LEGACY_INSTRUCTIONS)
TASK_SET = tuple(dict(preset="wall" if t in WALL else "bench", task=t, seed=0,
                      **({"contract": CONTRACT} if t in PROGRESS_SCORED else {})) for t in TASKS)


def validate_contract(task, value=None):
    if value is None or value == LEGACY:
        return LEGACY
    if value != CONTRACT or task not in PROGRESS_SCORED:
        raise ValueError(f"Contract {value!r} is not offered for {task!r}; use v1" +
                         (f" or {CONTRACT}" if task in PROGRESS_SCORED else ""))
    return CONTRACT


def instruction(task, contract=LEGACY):
    return (INSTRUCTIONS if validate_contract(task, contract) == CONTRACT else LEGACY_INSTRUCTIONS)[task]


def task_resolve(selection):
    # Delegate all existing seed typing, falsy defaults and preset inference to the old resolver.
    out = legacy.task_resolve(selection)
    task = out["selection"]["task"]
    chosen = validate_contract(task, selection.get("contract"))
    if chosen == LEGACY:
        return out
    v = out["selection"]
    out["task_id"] = f"drilling/v1/{v['preset']}/{task}/{chosen}/seed-{v['seed']}"
    out["description"] = instruction(task, chosen)
    out["selection"]["contract"] = chosen
    out["backend_options"]["contract"] = chosen
    return out


def contract():
    out = legacy.contract()
    out["selectors"] = [*out["selectors"], "contract"]
    out["contracts"] = {t: [LEGACY, CONTRACT] if t in PROGRESS_SCORED else [LEGACY]
                        for names in legacy.PUBLIC.values() for t in names}
    out["task_set"] = [dict(s) for s in TASK_SET]
    out["contract_note"] = "v1 is the unchanged default; ledger-anchor keeps its own progress v3 scoring."
    return out
