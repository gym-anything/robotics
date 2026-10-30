"""Launch configuration: preset, task, seed, robot and numerics. Unknown fields fail (unified contract)."""

from dataclasses import dataclass, fields

# preset -> tasks
TASKS = {
    "service": ("pour-glass", "taste-pour", "split-bottle", "water-tumbler", "carry-glass", "refill-glass"),
    "tasting": ("tasting-flight", "decant-carafe", "dump-and-pour"),
    "lined": ("lined-measure",),
}
RIGID_DT_S = 0.001                   # s, MuJoCo step (the FR3's 1 kHz control rate; barcode_scanning precedent)


@dataclass
class Config:
    preset: str = "service"
    task: str = "pour-glass"
    seed: int = 0
    robot: str = "dual-fr3"
    device: str = "cuda:0"
    rigid_dt: float = RIGID_DT_S
    liquid: bool = True              # False: rigid-only scene (diagnostics; tasks needing liquid refuse to evaluate)
    surface_forces: bool = True      # cohesion + wall adhesion + contact-line friction (liquid.py, params.SURFACE_FORCES)
    gas: bool = True                 # bottle headspace gas (gas.py); False only for ablation probes
    lump_liquid: bool = True         # added-mass partitioning (coupling.py); False only to reproduce the instability
    solver_iterations: int = 50
    solver_impratio: float = 3.      # barcode_scanning value (grip creep vs restitution trade-off)
    solver_noslip_iterations: int = 100
    solver_noslip_tolerance: float = 1e-10
    solver_update_data_interval: int = 0   # MuJoCo keeps its float64 state (barcode E20b: float32 re-sync walked bottles)
    contact_timeconst: float = .004  # s, rigid contact of every non-pad geom (barcode E17)
    contact_solimp: tuple = (.95, .99, .001)
    contact_capacity: int = 6144
    settle_rigid_s: float = 0.5      # rigid bodies settle before the liquid is placed
    liquid_settle_steps: int = 150   # damped liquid preparation steps (liquid.Liquid.prepare)
    liquid_packing_steps: int = 40
    record_trace: bool = True
    policy_samples: int = 24         # path-traced samples per pixel of policy RGB frames

    def __post_init__(self):
        if self.preset not in TASKS:
            raise ValueError(f"Unknown preset {self.preset!r}; choose {sorted(TASKS)}")
        if self.task not in TASKS[self.preset]:
            raise ValueError(f"Task {self.task!r} is not in preset {self.preset!r}: {TASKS[self.preset]}")
        if type(self.seed) is not int or not 0 <= self.seed < 10000:
            raise ValueError("seed must be an integer 0..9999")

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**d)
