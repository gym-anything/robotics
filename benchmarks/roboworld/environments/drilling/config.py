"""Launch configuration: preset, task, seed, robot and numerics. Unknown fields fail (unified contract)."""

from dataclasses import dataclass, fields

from . import params as P

# preset -> tasks (README task table)
TASKS = {
    "bench": ("take-drill", "punch-layout", "press-drill", "press-blind", "match-bolt", "panel-holes", "timber-bolt"),
    "wall": ("anchor-holes", "rebar-relocate", "clean-and-set", "ledger-anchor"),
    "probe": ("probe",),
}


@dataclass
class Config:
    preset: str = "wall"
    task: str = "anchor-holes"
    seed: int = 0
    robot: str = "dual-fr3"
    device: str = "cpu"
    rigid_dt: float = P.DT
    solver_iterations: int = P.SOLVER_ITERATIONS
    solver_impratio: float = P.SOLVER_IMPRATIO
    solver_noslip_iterations: int = P.NOSLIP_ITERATIONS
    solver_noslip_tolerance: float = P.NOSLIP_TOLERANCE
    solver_ls_iterations: int = P.SOLVER_LS_ITERATIONS
    solver_ls_tolerance: float = P.SOLVER_LS_TOLERANCE
    solver_ccd_iterations: int = P.SOLVER_CCD_ITERATIONS
    solver_ccd_tolerance: float = P.SOLVER_CCD_TOLERANCE
    contact_timeconst: float = P.CONTACT_TIMECONST
    contact_solimp: tuple = P.CONTACT_SOLIMP
    contact_capacity: int = 4096
    settle_s: float = 1.0
    record_trace: bool = True
    policy_samples: int = 16
    variant: str = ""                # seeded variants: loose-vise, worn-bit, anti-rotation
    layout: dict = None              # owner/probe override of the generated layout (never from the public API)

    def __post_init__(self):
        if self.preset not in TASKS:
            raise ValueError(f"Unknown preset {self.preset!r}; choose {sorted(TASKS)}")
        if self.task not in TASKS[self.preset]:
            raise ValueError(f"Task {self.task!r} is not in preset {self.preset!r}: {TASKS[self.preset]}")
        if type(self.seed) is not int or not 0 <= self.seed < 10000:
            raise ValueError("seed must be an integer 0..9999")
        if self.variant not in ("", "loose-vise", "worn-bit"):
            raise ValueError(f"Unknown variant {self.variant!r}")

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**d)
