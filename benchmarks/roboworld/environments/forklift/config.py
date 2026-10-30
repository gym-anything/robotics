"""Session configuration: preset, task, seed, robot and engine settings (immutable during a session)."""

from dataclasses import asdict, dataclass, field, fields

from . import params as P
from .robot_setup import ALIASES as _ALIASES

PRESETS = ("warehouse", "dock", "yard", "lab")
TASKS = {
    "warehouse": ("start-and-park", "aisle-cruise", "drive-to-pallet", "carry-to-floor-bay", "carry-loose-stack",
                  "put-away-rack", "feed-conveyor"),
    "dock": ("load-trailer", "load-trailer-chocked", "load-bales", "unload-trailer"),
    "yard": ("ramp-up-loaded", "ramp-down-loaded", "yard-route"),
    "lab": ("free",),                       # probes only: an empty floor (not in the benchmark catalogue)
}
ROBOTS = (*_ALIASES, "none")              # the catalogue's forklift aliases (robot_setup.ALIASES) and the robot-less probes
DEFAULT_ROBOT = "h1-2"
CAMERA_SETS = ("onboard", "onboard+evidence")


@dataclass(frozen=True)
class Config:
    preset: str = "warehouse"
    task: str = "start-and-park"
    seed: int = 0
    robot: str = DEFAULT_ROBOT
    camera_set: str = "onboard"
    device: str = "cpu"
    dt: float = P.DT
    record_trace: bool = True
    contact_timeconst: float = P.CONTACT_TIMECONST
    contact_solimp: tuple = P.CONTACT_SOLIMP
    solver_iterations: int = P.SOLVER_ITERATIONS
    solver_impratio: float = P.IMPRATIO
    solver_noslip_iterations: int = P.NOSLIP_ITERATIONS
    solver_noslip_tolerance: float = 1e-10
    contact_capacity: int = 4096
    extras: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ValueError(f"preset must be one of {PRESETS}")
        if self.task not in TASKS[self.preset]:
            raise ValueError(f"task {self.task!r} is not a {self.preset} task; tasks: {TASKS[self.preset]}")
        if type(self.seed) is not int or not 0 <= self.seed <= 999:
            raise ValueError("seed must be an integer in 0..999")
        if self.robot not in ROBOTS:
            raise ValueError(f"robot must be one of {ROBOTS}")
        if self.camera_set not in CAMERA_SETS:
            raise ValueError(f"camera_set must be one of {CAMERA_SETS}")
        if not 0 < self.dt <= 0.0005 + 1e-12:
            raise ValueError("dt must be in (0, 0.5 ms]: the seated H1-2 and the tyre model are qualified at 0.5 ms")

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        d = dict(d)
        if "contact_solimp" in d:
            d["contact_solimp"] = tuple(d["contact_solimp"])
        return cls(**d)

    def to_dict(self):
        return asdict(self)
