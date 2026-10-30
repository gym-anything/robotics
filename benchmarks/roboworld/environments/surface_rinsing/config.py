"""Launch configuration: task, seed, robot and the reused classroom controller options. Unknown fields fail."""

from dataclasses import dataclass, fields
from .layout import PRESETS, TASKS


@dataclass
class Config:
    task: str = "rinse-glassware"
    seed: int = 0
    robot: object = "mobile-fr3-duo"     # catalog alias or a qualified gar.robot-setup.v1 object (validated by the backend)
    glides: str = "nylon"                # unused here (classroom World reads it); kept for the shared World
    contents: int = 0                    # unused here (the shared classroom backend's notes read it)
    reflex_profile: str = "contact_x5"   # FR3 collision thresholds (classroom params.REFLEX_PROFILES)
    policy_samples: int = 16
    reflex_stops_base: bool = True
    water_device: str = "cuda:0"         # the water system's Warp device (films, packets); the rigid world is CPU MuJoCo

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"Unknown task {self.task!r}; choose {sorted(TASKS)}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        from classroom_arrangement.params import REFLEX_PROFILES
        if self.reflex_profile not in REFLEX_PROFILES:
            raise ValueError(f"reflex_profile must be one of {sorted(REFLEX_PROFILES)}")

    @property
    def preset(self):
        return TASKS[self.task]["preset"]

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**d)
