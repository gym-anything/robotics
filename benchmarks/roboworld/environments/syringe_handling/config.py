"""Launch configuration: task, seed, robot, variant and numerics. Unknown fields fail (unified contract)."""

from dataclasses import dataclass, fields

from . import params as P

LEGACY_CONTRACT = "v1"
EIGHT_TASK_CONTRACT = "syringe-tasks-v1"
VIAL_TO_PAD_CONTRACT = "syringe-vial-to-pad-v1"   # reported by vial-to-pad's own evaluator


def resolve_contract(task, contract=None):
    value = LEGACY_CONTRACT if contract is None else contract
    if value not in (LEGACY_CONTRACT, EIGHT_TASK_CONTRACT):
        raise ValueError(f"Unknown contract {value!r}; choose {LEGACY_CONTRACT!r} or {EIGHT_TASK_CONTRACT!r}")
    if task == "vial-to-pad" and value != LEGACY_CONTRACT:
        raise ValueError("vial-to-pad keeps its own evaluator; select contract='v1' (or omit it)")
    return value

# setup of the bay -> tasks
TASKS = {
    "injection": ("sharps-away", "set-dose", "draw-dose", "inject", "vial-to-pad"),
    "iv-lock": ("flush-lock", "slow-push", "fast-bolus"),
}
SETUP_OF = {t: s for s, ts in TASKS.items() for t in ts}
VARIANTS = {
    "": "the task as listed",
    "upright": "inject / vial-to-pad with the pad strapped to an upright upper-arm model (its skin vertical, facing the robot)",
    "adapter": "draw-dose through a vented needle-free vial adapter instead of the blunt fill needle",
    "half-vial": "draw-dose / vial-to-pad from a vial that holds 5 mL",
    "vacuum-vial": "draw-dose from a vial left 25 kPa under ambient by an earlier draw",
    "subcut": "inject as a subcutaneous dose: 0.3 mL from a 1 mL syringe through a 27 G x 13 mm needle",
    "blocked": "flush-lock with a cannula that is blocked (do not force it)",
    "clamp-open": "flush-lock / slow-push / fast-bolus with the slide clamp found open",
    "g22": "the iv-lock tasks with a 22 G cannula",
    "g18": "the iv-lock tasks with an 18 G cannula",
    "short": "slow-push as 1 mL over one minute",
}
VARIANT_TASKS = {"upright": ("inject", "vial-to-pad"), "adapter": ("draw-dose",), "half-vial": ("draw-dose", "vial-to-pad"),
                 "vacuum-vial": ("draw-dose",), "subcut": ("inject",), "blocked": ("flush-lock",),
                 "clamp-open": TASKS["iv-lock"], "g22": ("flush-lock", "slow-push"), "g18": TASKS["iv-lock"],
                 "short": ("slow-push",)}


@dataclass
class Config:
    preset: str = ""                 # the bay's setup; empty: the task's own (a task has one setup)
    task: str = "set-dose"
    seed: int = 0
    robot: str = "dual-fr3"
    variant: str = ""
    rigid_dt: float = P.DT
    solver_iterations: int = P.SOLVER_ITERATIONS
    solver_impratio: float = P.SOLVER_IMPRATIO
    solver_noslip_iterations: int = P.NOSLIP_ITERATIONS
    solver_noslip_tolerance: float = P.NOSLIP_TOLERANCE
    solver_ls_iterations: int = P.SOLVER_LS_ITERATIONS
    solver_ls_tolerance: float = P.SOLVER_LS_TOLERANCE
    contact_timeconst: float = P.CONTACT_TIMECONST
    contact_solimp: tuple = P.CONTACT_SOLIMP
    contact_capacity: int = 2048
    settle_s: float = 0.5
    record_trace: bool = True
    policy_samples: int = 16
    layout: dict = None              # owner/probe override (never from the public API)
    contract: str = LEGACY_CONTRACT  # a selection without a contract keeps the v1 reward and instruction

    def __post_init__(self):
        self.contract = resolve_contract(self.task, self.contract)
        if self.task not in SETUP_OF:
            raise ValueError(f"Unknown task {self.task!r}; choose {sorted(SETUP_OF)}")
        if not self.preset:
            self.preset = SETUP_OF[self.task]
        if self.preset not in TASKS:
            raise ValueError(f"Unknown preset {self.preset!r}; choose {sorted(TASKS)}")
        if self.task not in TASKS[self.preset]:
            raise ValueError(f"Task {self.task!r} is not in preset {self.preset!r}: {TASKS[self.preset]}")
        if type(self.seed) is not int or not 0 <= self.seed < 10000:
            raise ValueError("seed must be an integer 0..9999")
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {self.variant!r}; choose {sorted(VARIANTS)}")
        if self.variant and self.task not in VARIANT_TASKS[self.variant]:
            raise ValueError(f"Variant {self.variant!r} belongs to {VARIANT_TASKS[self.variant]}, not {self.task!r}")

    @property
    def setup(self):
        return SETUP_OF[self.task]

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**d)
