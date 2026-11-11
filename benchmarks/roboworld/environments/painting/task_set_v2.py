"""painting-tasks-v2: the render-checked eight-task set (selection, ids, instructions and session wiring).

The seven revised tasks are those of painting-tasks-v1, with the same starts and instructions; only the verifier differs
(painting/task_set_v2_verifier.py): it keeps the legacy untouched, no-runs and roller-release checks, which v1 had
loosened, plus v1's three tightenings (trim sinks, pending wall drops landed first, the named craft pigment). The audited
behind-radiator task keeps its own progress-v1 selection. No placement, start or physics changes: a v2 world is built by
the unchanged PaintingEnv constructor with no contract (its legacy start), then adopt() puts it under the contract.

Nothing released reads this module: painting.variants.task_resolve and NewtonPaintingBackend dispatch here only for a
selection or launch whose contract is painting-tasks-v2, and a session launched without it never calls these.
"""
from . import catalogue
from . import variants

CONTRACT = "painting-tasks-v2"
TASKS = tuple(catalogue.INSTRUCTIONS)          # the seven revised tasks; behind-radiator is outside the contract
SELECTORS = ("preset", "task", "seed", "contract")


def check_task(task):
    if task not in catalogue.INSTRUCTIONS:
        raise ValueError(f"Unknown contract {CONTRACT!r} for {task}; choose {catalogue.selected_contract(task)}")


def describe(task, seed=0):
    """The v2 instruction: v1's, word for word (painting.catalogue.INSTRUCTIONS and its finish-coat sentence)."""
    check_task(task)
    return catalogue.describe(task, seed, catalogue.CONTRACT)


def resolve(preset=None, task=None, seed=0, contract=CONTRACT):
    if contract != CONTRACT:
        raise ValueError(f"resolve() selects {CONTRACT}, not {contract!r}")
    v = variants.resolve(preset, task, seed)
    check_task(v["task"])
    return dict(v, contract=CONTRACT)


def resolve_any(preset=None, task=None, seed=0, contract=None):
    """A v2 session's resolver: the v2 contract here, every other choice exactly as painting.variants.resolve."""
    if contract == CONTRACT:
        return resolve(preset, task, seed, contract)
    return variants.resolve(preset, task, seed, contract)


def selections():
    from .layout import TASKS as ALL
    return [dict(preset=ALL[t]["preset"], task=t, seed=0, **({"contract": CONTRACT} if t in TASKS else {}))
            for t in catalogue.EIGHT_TASKS]


def task_resolve(selection):
    """benchmark.json's resolver (painting.variants.task_resolve) for a painting-tasks-v2 selection."""
    if set(selection) - set(SELECTORS):
        raise ValueError("Selectors are preset, task, seed and contract")
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0), selection.get("contract"))
    return dict(task_id=f"painting/v1/{v['preset']}/{v['task']}/{v['contract']}/seed-{v['seed']}",
                description=describe(v["task"], v["seed"]), backend_options=dict(v), selection=v,
                split=variants.split_of(v["task"], v["seed"]))


def contract():
    """The variation contract a v2 session reports: painting.variants.contract() with the v2 choice and set added."""
    from .layout import TASKS as ALL
    return dict(variants.contract(),
                contract_choices={t: [catalogue.selected_contract(t)] + ([catalogue.CONTRACT, CONTRACT] if t in TASKS
                                                                         else []) for t in ALL},
                task_set=selections(), task_set_v1=catalogue.selections())


# ---------------------------------------------------------------------- the environment under the contract
def adopt(env):
    """Put an environment built by the unchanged PaintingEnv constructor (no contract) under painting-tasks-v2: the
    instruction, the verifier (measured on the same settled start) and the start checkpoint, which then carries the
    verifier's drop clock (PaintingEnv.checkpoint/restore already call the evaluator's checkpoint hooks)."""
    from .task_set_v2_verifier import TaskSetV2Evaluator
    if getattr(env, "contract", None) == CONTRACT or isinstance(getattr(env, "evaluator", None), TaskSetV2Evaluator):
        raise ValueError(f"environment is already under {CONTRACT}")
    if getattr(getattr(env, "config", None), "contract", None) is not None:
        raise ValueError(f"adopt() takes an environment built without a contract, not {env.config.contract!r}")
    with env.lock:
        env.spec = dict(env.spec, instruction=describe(env.spec["task"], env.spec["seed"]))
        env.contract = CONTRACT
        env.evaluator = TaskSetV2Evaluator(env)
        env.initial = env.checkpoint()
        env._eval_key = None
    return env


# ---------------------------------------------------------------------- NewtonPaintingBackend session wiring
# Reached only from a backend launched with contract=painting-tasks-v2 (backend._task_set_v2). The backend's own
# _rebuild builds the unchanged environment; a v2 choice is then adopted. Other choices resolve as the released reset.
def session_adopt(backend):
    from dataclasses import replace
    adopt(backend.env)
    backend.capabilities = replace(backend.capabilities, notes=backend._notes())


def session_variation(backend):
    c = backend.config
    if getattr(backend.env, "contract", None) == CONTRACT:
        return dict(preset=c.preset, task=c.task, seed=c.seed, contract=CONTRACT)
    return variants.resolve(c.preset, c.task, c.seed, c.contract)


def _want(backend, request):
    cur = backend._variation()
    task = request.get("task", cur["task"])
    # the current contract carries over while the task stays; another task starts from its own default
    return resolve_any(request.get("preset", cur["preset"] if "task" not in request else None), task,
                       int(request.get("seed", cur["seed"])),
                       request.get("contract", cur.get("contract") if task == cur["task"] else None))


def session_reset(backend, seed=None, options=None):
    from gym_anything_robotics.errors import UnsupportedActionError
    options = dict(options or {})
    unknown = set(options) - set(SELECTORS)
    if unknown:
        raise UnsupportedActionError(f"Unknown reset options {sorted(unknown)}; selectors are preset, task, seed, contract; "
                                     "nothing changed")
    if seed is not None:
        options["seed"] = seed
    if options:
        try:
            want = _want(backend, options)
        except (ValueError, TypeError) as exc:
            raise UnsupportedActionError(f"Reset refused: {exc}; nothing changed") from exc
        if want != backend._variation():
            if want.get("contract") == CONTRACT:
                backend._rebuild({k: want[k] for k in ("preset", "task", "seed")})
                session_adopt(backend)
            else:
                backend._rebuild(want)
            return backend.env.get_state(), {"continuity": "reset", "variation": backend._variation(), "rebuilt": True}
    return backend.env.reset(), {"continuity": "reset", "variation": backend._variation()}


def session_select(backend, request):
    if set(request) - set(SELECTORS):
        raise ValueError("Selectors are preset, task, seed and contract")
    v = _want(backend, request)
    text = describe(v["task"], v["seed"]) if v.get("contract") == CONTRACT else \
        catalogue.describe(v["task"], v["seed"], v.get("contract"))
    return dict(variation=v, task=text, reset_request=dict(seed=v["seed"], options=dict(preset=v["preset"], task=v["task"],
                                                                                       contract=v.get("contract"))))


def session_contract(backend):
    return dict(**contract(), current=backend._variation())
