"""Launch configuration: preset, task, seed, robot and the reused classroom controller options. Unknown fields fail."""

from dataclasses import dataclass, fields
from .layout import PRESETS, TASKS


@dataclass
class Config:
    preset: str = "wall"
    task: str = "roll-strip"
    seed: int = 0
    robot: object = "mobile-fr3-duo"     # catalog alias or a qualified gar.robot-setup.v1 object (validated by the backend)
    glides: str = "nylon"                # unused here (classroom World reads it); kept for the shared World
    reflex_profile: str = "contact_x5"   # FR3 collision thresholds (classroom params.REFLEX_PROFILES)
    policy_samples: int = 16
    reflex_stops_base: bool = True
    contents: int = 0
    contract: str | None = None

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown preset {self.preset!r}; choose {sorted(PRESETS)}")
        if self.task not in PRESETS[self.preset]:
            raise ValueError(f"Task {self.task!r} is not in preset {self.preset!r}: {PRESETS[self.preset]}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        from .catalogue import selected_contract
        selected_contract(self.task, self.contract)
        from classroom_arrangement.params import REFLEX_PROFILES
        if self.reflex_profile not in REFLEX_PROFILES:
            raise ValueError(f"reflex_profile must be one of {sorted(REFLEX_PROFILES)}")

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**d)

    @staticmethod
    def preset_of(task):
        return TASKS[task]["preset"]
