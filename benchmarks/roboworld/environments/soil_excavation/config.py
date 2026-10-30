"""Launch configuration: task, seed, variation selectors, robot and the reused classroom controller options.
Unknown fields fail."""

from dataclasses import dataclass, fields
from .layout import TASKS
from . import params as P


@dataclass
class Config:
    task: str = "penetrate-marked"
    seed: int = 0
    moisture: str = "moist"              # dry | moist | wet (params.MOISTURE)
    topsoil: object = None               # topsoil depth (m), 0.12-0.32; None = the preset's
    stones: object = None                # number of embedded stones; None = the task's seeded default
    clutch: object = 8                   # drill clutch setting 1-11 or None (drill mode); auger task only
    robot: object = "mobile-fr3-duo"     # catalog alias or a qualified gar.robot-setup.v1 object (validated by the backend)
    glides: str = "nylon"                # unused here (classroom World reads it); kept for the shared World
    reflex_profile: str = "contact_x5"   # FR3 collision thresholds (classroom params.REFLEX_PROFILES)
    policy_samples: int = 16
    reflex_stops_base: bool = True
    device: str = "cuda:0"               # the soil's GPU (MuJoCo runs on the CPU)

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"Unknown task {self.task!r}; choose {sorted(TASKS)}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.moisture not in P.MOISTURE:
            raise ValueError(f"moisture must be one of {sorted(P.MOISTURE)}")
        if self.topsoil is not None and (isinstance(self.topsoil, bool) or not 0.12 <= float(self.topsoil) <= 0.32):
            raise ValueError("topsoil must be a depth in 0.12-0.32 m or null")
        if self.stones is not None and (type(self.stones) is not int or not 0 <= self.stones <= 6):
            raise ValueError("stones must be an integer 0-6 or null")
        if self.clutch is not None and (type(self.clutch) is not int or self.clutch not in P.DRILL_CLUTCH):
            raise ValueError("clutch must be an integer 1-11 or null (drill mode)")
        from classroom_arrangement.params import REFLEX_PROFILES
        if self.reflex_profile not in REFLEX_PROFILES:
            raise ValueError(f"reflex_profile must be one of {sorted(REFLEX_PROFILES)}")

    @property
    def preset(self):
        return TASKS[self.task]["preset"]

    contents = 0                         # classroom backend notes read it; no desk contents here

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**d)
