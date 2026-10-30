"""Launch configuration: preset, task, seed, robot and numerics. Unknown fields fail (unified contract)."""

from dataclasses import dataclass, fields

from . import params as P

# preset -> tasks
TASKS = {
    "bake": ("muffin-fold", "dry-whisk", "pancake-whisk", "pudding-whisk", "cake-mixer", "sauce-blend"),
    "hob": ("stir-sauce", "reheat-porridge"),
}
VARIANTS = {
    "": "the task as listed",
    "mat": "the vessel stands on a silicone mat (friction 0.9)",
    "sleeves": "the hand tools carry square-section non-slip sleeves on their handles (flat faces for a parallel hand)",
    "spelt": "pancake-whisk with the whole-spelt batter of the cited table (K 6.9 Pa s^n, n 0.33)",
    "honey": "sauce-blend with honey, mayonnaise and mustard",
    "pet-meal": "sauce-blend as a pet's meal: canned food with a powdered supplement",
}
# the tasks a variant belongs to. The mat goes under a bowl on the counter: scene.task_layout's own seeded mat is laid
# in the bake preset only, and a silicone mat between an induction zone and the pan it heats is not a kitchen's set-up
# (the hob's heat model has no mat in it either), so the hob tasks do not take it
VARIANT_TASKS = {"mat": TASKS["bake"], "spelt": ("pancake-whisk",), "honey": ("sauce-blend",), "pet-meal": ("sauce-blend",)}


@dataclass
class Config:
    preset: str = "bake"
    task: str = "muffin-fold"
    seed: int = 0
    robot: str = "dual-fr3"
    variant: str = ""
    device: str = "cuda:0"           # the continuum; the rigid world runs on the CPU (MuJoCo)
    rigid_dt: float = P.DT
    cell_m: float = 0.0025           # continuum grid cell (E2/E3: 2.5 mm reference)
    points_per_cell: int = 8
    solver_iterations: int = P.SOLVER_ITERATIONS
    solver_impratio: float = P.SOLVER_IMPRATIO
    solver_noslip_iterations: int = P.NOSLIP_ITERATIONS
    solver_noslip_tolerance: float = P.NOSLIP_TOLERANCE
    solver_ls_iterations: int = P.SOLVER_LS_ITERATIONS
    solver_ls_tolerance: float = P.SOLVER_LS_TOLERANCE
    contact_timeconst: float = P.CONTACT_TIMECONST
    contact_solimp: tuple = P.CONTACT_SOLIMP
    contact_capacity: int = 4096
    settle_s: float = 0.5
    record_trace: bool = True
    policy_samples: int = 16
    material: bool = True            # False: the rigid scene without the mixture (rigid benches)
    layout: dict = None              # owner/probe override (never from the public API)

    def __post_init__(self):
        if self.preset not in TASKS:
            raise ValueError(f"Unknown preset {self.preset!r}; choose {sorted(TASKS)}")
        if self.task not in TASKS[self.preset]:
            raise ValueError(f"Task {self.task!r} is not in preset {self.preset!r}: {TASKS[self.preset]}")
        if type(self.seed) is not int or not 0 <= self.seed < 10000:
            raise ValueError("seed must be an integer 0..9999")
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {self.variant!r}; choose {sorted(VARIANTS)}")
        if self.variant in VARIANT_TASKS and self.task not in VARIANT_TASKS[self.variant]:
            raise ValueError(f"Variant {self.variant!r} belongs to {VARIANT_TASKS[self.variant]}, not {self.task!r}")
        if not 0.00125 <= float(self.cell_m) <= 0.005:
            raise ValueError("cell_m must be within 1.25 mm .. 5 mm")
        if self.points_per_cell not in (1, 8):
            raise ValueError("points_per_cell must be 1 or 8")

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**d)
