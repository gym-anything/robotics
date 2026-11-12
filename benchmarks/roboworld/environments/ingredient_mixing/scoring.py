"""Task-set judging, independent of physics.

The seven opt-in revisions use stir-sauce's failure factor. This module never writes physical state.
Baselines are captured at hand-over, not at the first evaluation. Heat records contain simultaneous portion
readings, never averages of particle temperature peaks. Legacy tasks and stir-sauce bypass the revised judging.
"""
import math
from dataclasses import dataclass

from .config import Config

LEGACY = "v1"
CONTRACT = "ingredient-mixing-tasks-v1"
STIR_SAUCE_CONTRACT = "stir-sauce-v1"
ORDER = ("sauce-blend", "muffin-fold", "dry-whisk", "pudding-whisk", "pancake-whisk", "cake-mixer",
         "reheat-porridge", "stir-sauce")
DIFFICULTY = dict(zip(ORDER, ("entry", "entry", "medium", "medium", "medium", "hard", "hard", "hardest")))
# The render-checked set: a TaskConfig with this contract is judged by scoring_v2.py (dispatched below).
V2 = "ingredient-mixing-tasks-v2"


def _v2(config):
    return getattr(config, "contract", LEGACY) == V2


@dataclass
class TaskConfig(Config):
    """Catalogue-only config extension; the physics Config and its defaults remain frozen."""
    contract: str = LEGACY

    def __post_init__(self):
        super().__post_init__()
        validate_contract(self.contract, self.task)


def task_spec(config):
    if _v2(config):
        from . import scoring_v2
        return scoring_v2.task_spec(config)
    from . import task
    return revise_spec(config, task.task_spec(config))


def instruction(config):
    if _v2(config):
        from . import scoring_v2
        return scoring_v2.instruction(config)
    from . import task
    original = task.task_spec(config)["instruction"]
    return task.instruction(config).replace(original, task_spec(config)["instruction"], 1)


def validate_contract(contract, task):
    if contract == V2:
        from . import scoring_v2
        return scoring_v2.validate_contract(contract, task)
    if contract not in (LEGACY, CONTRACT):
        raise ValueError(f"Unknown contract {contract!r}; choose {LEGACY!r} or {CONTRACT!r}")
    if contract == CONTRACT and task == "stir-sauce":
        raise ValueError("stir-sauce keeps its own contract; select contract=v1")
    return contract


def enabled(config):
    if _v2(config):
        from . import scoring_v2
        return scoring_v2.enabled(config)
    return getattr(config, "contract", LEGACY) == CONTRACT and config.task != "stir-sauce"


def contract_for(config):
    return STIR_SAUCE_CONTRACT if config.task == "stir-sauce" else getattr(config, "contract", LEGACY)


def revise_spec(config, spec):
    """Only judging/instructions change; recipe, props, layout and dynamics stay identical."""
    if not enabled(config):
        return spec
    spec = dict(spec)
    if config.task == "pudding-whisk":
        spec["instruction"] = ("The bowl holds two cups of cold milk with a packet of instant pudding mix heaped in "
                               "the middle. Whisk until the pudding is smooth, then take the whisk out. Keep the "
                               "pudding in the bowl.")
    elif config.task == "cake-mixer":
        spec["instruction"] = ("The bowl holds cake mix with water, oil and eggs. The hand mixer stands on its heel "
                               "beside the bowl, switched off. Beat the batter until smooth. Leave the mixer "
                               "switched off and set down on the counter, out of the bowl. Keep the batter in the bowl.")
    elif config.task == "reheat-porridge":
        spec["instruction"] = ("The saucepan on the hob holds cold cooked porridge; the hob is on. Stir until the "
                               "porridge is hot all the way through (74 C, 165 F, everywhere), without letting it "
                               "catch on the bottom. Then turn the hob off. Keep the porridge in the pan and the pan "
                               "on the hob. A probe thermometer stands in the crock.")
    from .task import task_verifier
    criteria = getattr(task_verifier(config.task), "contract_criteria", None)
    if criteria is not None:
        spec["criteria"] = criteria(spec["criteria"])
    return spec


def clip(x):
    return min(1., max(0., float(x)))


def ramp(x, lo, hi):
    return clip((x - lo)/(hi - lo)) if hi > lo else 0.


def uniform_progress(criterion, baseline):
    if criterion["batch_share_in_the_vessel"] < criterion["share_limit"]:
        return 0.
    terms = []
    starts = list(baseline["cv"].values())
    if starts and all(v is not None and v <= criterion["limit"] for v in starts):
        return 0.  # already uniform at hand-over; no improvement to credit
    for species, cv in criterion["each"].items():
        start = baseline["cv"].get(species)
        limit = criterion["limit"]
        if cv is None or start is None or not math.isfinite(cv) or not math.isfinite(start):
            return 0.
        # Already-uniform ingredients must remain uniform, but do not block credit for mixing the others.
        terms.append(clip(math.log(start/max(cv, 1e-12))/math.log(start/limit)) if start > limit
                     else clip(limit/max(cv, 1e-12)))
    return min(terms, default=0.)


def hot_quality(reading, start_c, limit_c=74., spread_k=15.):
    return ramp(reading["coldest_c"], start_c, limit_c) * (1. - ramp(
        reading["hottest_c"] - reading["coldest_c"], spread_k, 40.))


def hot_ok(reading, limit_c=74., spread_k=15.):
    return (reading.get("batch_share", 0.) >= 0.99 and reading["coldest_c"] >= limit_c
            and reading["hottest_c"] - reading["coldest_c"] <= spread_k)


def best_hot(previous, current, start_c):
    """Keep one coherent reading; later uneven heating must not erase a completed reheat.

    A reading covers >=99% of the batch; heating a little food then returning cold food cannot pass. Success is
    re-derived from the temperatures, not a sticky success flag. Current readings also allow constructed states.
    """
    candidates = [r for r in (previous, current) if r is not None and r.get("batch_share", 0.) >= 0.99]
    if not candidates:
        return None
    return dict(max(candidates, key=lambda r: (hot_ok(r), hot_quality(r, start_c), r["coldest_c"])))


def capture_start(env):
    """Read once after settling; TaskEnv adds the result to the physical start checkpoint without moving anything."""
    if _v2(env.config):
        from . import scoring_v2
        return scoring_v2.capture_start(env)
    if not enabled(env.config) or env.info is None:
        return
    C = evaluate(env.world, env.info, _progress=False)["criteria"]
    baseline = dict(cv=dict(C.get("uniform", {}).get("each", {})),
                    dry=C.get("no_dry_powder", {}).get("value", 0.),
                    lump_quality=1. - ramp(C.get("smooth", {}).get("value", 0.), 0.02, 0.10),
                    coldest_c=C.get("hot_through", {}).get("coldest_c", 5.),
                    batch_mass=float(sum(env.info["batch"])), spacing=env.info["spacing"])
    env.info["task_baseline"] = baseline
    from .task import task_verifier
    hand_over = getattr(task_verifier(env.config.task), "hand_over", None)
    if hand_over is not None:
        hand_over(env, baseline)


def progress(task, C, baseline, bottom_cm2):
    """Continuous material improvement times retention, damage and finishing factors."""
    keep = clip(1. - C["contained"]["value"])
    in_place = C["vessel_in_place"]["ok"]
    close = "tool_out" if "tool_out" in C else "appliance_off" if "appliance_off" in C else None
    from . import scoring
    from .task import task_verifier
    own = getattr(task_verifier(task), "progress_v1", None)
    if own is not None:
        core, damage, done, terms = own(scoring, C, baseline, bottom_cm2)
    else:
        U = uniform_progress(C["uniform"], baseline)
        terms = dict(uniform=U)
        core, damage = U, 1.
        if "no_dry_powder" in C:
            d = C["no_dry_powder"]
            D = clip((baseline["dry"] - d["value"])/(baseline["dry"] - d["limit"])) \
                if baseline["dry"] > d["limit"] else 0.
            terms["wet"] = D
            core = (U + D)/2.
            if "smooth" in C:
                L = 1. - ramp(C["smooth"]["value"], C["smooth"]["limit"], 0.10)
                # A fresh heap often has no lumps. Normalize the combined quality from its actual start instead
                # of awarding L/3 immediately (a plain (U + D + L)/3 would give idle credit).
                L0 = baseline["lump_quality"]
                core = clip((U + D + L - L0)/(3. - L0))
                terms.update(smooth=L, initial_smooth=L0)
        if "not_overmixed" in C:
            work = C["not_overmixed"]
            damage = 1. - ramp(work["value"], work["limit"], 2.*work["limit"])
            terms["not_overmixed"] = damage
        if C["uniform"]["batch_share_in_the_vessel"] < C["uniform"]["share_limit"]:
            core = 0.
        done = all(C[k]["ok"] for k in ("uniform", "no_dry_powder", "smooth", "not_overmixed") if k in C)
    finish = (0.85 + 0.15*bool(in_place)) if close is None else (
        0.75 + 0.10*bool(in_place) + 0.15*bool(C[close]["ok"] and done))
    return dict(progress=clip(core*damage*keep*finish), keep=keep, finish=finish, **terms)


def record_hot_state(hob):
    """Read the same post-heat state as Hob.record_food_peak, from its external instance wrapper.

    Called at every existing material/heat update (10 ms), before species/heat exchange. The extra read is only
    for the explicitly selected porridge contract. It cannot affect the dynamics or stir-sauce's record.
    """
    if _v2(getattr(getattr(hob, "w", None), "config", None)):
        from . import scoring_v2
        return scoring_v2.record_hot_state(hob)
    if getattr(hob, "task_baseline", None) is None or hob.knob.s_level <= 0:
        return
    import numpy as np
    from scipy.spatial.transform import Rotation
    x, temp, mass, flags = hob.w.cont.read("x", "temp", "mass", "flags")
    q = hob.w.bq[hob.pan.body]
    local = (x - q[:3].astype(np.float32)) @ Rotation.from_quat(q[3:]).as_matrix().astype(np.float32)
    inside = (flags == 0) & hob.pan.inside(local, margin=0.002)
    from . import scoring
    from .task import task_verifier
    sample = getattr(task_verifier(hob.w.config.task), "record_hot_sample", None)
    if sample is not None:
        sample(scoring, hob, local[inside], temp[inside], mass[inside])


def evaluate(world, info, *, _progress=True):
    """The task's verifier.py verify (tasks/<task>/), which judges with evaluate_criteria below."""
    from .task import task_verifier
    return task_verifier(world.config.task).verify(world, info, _progress=_progress)


def evaluate_criteria(world, info, *, _progress=True):
    """Separate opt-in judging from the frozen legacy evaluator; never patch its module globals."""
    if _v2(world.config):
        from . import scoring_v2
        return scoring_v2.evaluate(world, info, _progress=_progress)
    from . import task
    from . import scoring
    folder = task.task_verifier(world.config.task)
    if enabled(world.config) and hasattr(folder, "contract_evaluation"):
        out = folder.contract_evaluation(scoring, world, info)
    else:
        # Bake predicates and stir-sauce's own evaluator are reused without modification.
        out = task.evaluate(world, info)
    out["contract"] = contract_for(world.config)
    if enabled(world.config) and _progress:
        if "task_baseline" not in info:
            raise ValueError("Task-set scoring requires the hand-over baseline; rebuild/reset the task")
        bottom = float(world.heat.area.sum())*1e4 if world.heat is not None else 1.
        r = progress(world.config.task, out["criteria"], info["task_baseline"], bottom)
        out["progress"] = r["progress"]
        out["reward"] = dict(r, contract=CONTRACT, score_rule=f"1 if success else {task.FAIL_SCORE_SCALE} x progress")
        out["score"] = 1.0 if out["success"] else task.FAIL_SCORE_SCALE*r["progress"]
    return out
