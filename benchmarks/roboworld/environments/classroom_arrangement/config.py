"""Launch configuration: preset, task, seed, robot and variation selectors. Unknown fields fail (unified contract)."""

from dataclasses import dataclass, fields
from .layout import PRESETS

GLIDES = ("nylon", "felt", "steel")


@dataclass
class Config:
    preset: str = "room114"
    task: str = "straighten-row"
    seed: int = 0
    robot: object = "mobile-fr3-duo"   # catalog alias, or a qualified gar.robot-setup.v1 object (backend validates)
    glides: str = "nylon"            # Virco glide option on every desk and chair: nylon, felt, steel
    contents: int = 0                # textbooks in each involved desk's book box (0-3)
    reflex_profile: str = "contact_x5"   # FR3 collision thresholds (params.REFLEX_PROFILES): contact_x5, franka_ros, libfranka_default
    policy_samples: int = 16         # path-traced samples per pixel of a policy frame
    reflex_stops_base: bool = True   # an arm protective stop also stops the base and holds it for 1 s (params.REFLEX_STOPS_BASE)
    contract: str = "v1"             # opt-in task semantics, never a physics selector

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown preset {self.preset!r}; choose {sorted(PRESETS)}")
        if self.task not in PRESETS[self.preset]:
            raise ValueError(f"Task {self.task!r} is not in preset {self.preset!r}: {PRESETS[self.preset]}")
        from .catalogue import validate_contract
        validate_contract(self.task, self.contract)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.glides not in GLIDES:
            raise ValueError(f"glides must be one of {GLIDES}")
        from .params import REFLEX_PROFILES
        if self.reflex_profile not in REFLEX_PROFILES:
            raise ValueError(f"reflex_profile must be one of {sorted(REFLEX_PROFILES)}")
        if type(self.contents) is not int or not 0 <= self.contents <= 3:
            raise ValueError("contents must be an integer 0-3")

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**d)
