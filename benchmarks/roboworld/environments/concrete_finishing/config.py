"""Launch configuration: preset, task, seed, robot and the seed's site conditions. Unknown fields fail (unified
contract)."""

from dataclasses import dataclass, fields
from .layout import PRESETS, VARIANTS


@dataclass
class Config:
    preset: str = "b012"
    task: str = "pad-darby"
    seed: int = 0
    robot: object = "mobile-fr3-duo"   # catalog alias, or a qualified gar.robot-setup.v1 object (backend validates)
    variant: str = "default"           # site conditions (variants.py): default, door, fan, hot, cool, fast-sealant
    reflex_profile: str = "contact_x5"   # FR3 collision thresholds (classroom_arrangement params.REFLEX_PROFILES)
    policy_samples: int = 16           # path-traced samples per pixel of a policy frame
    reflex_stops_base: bool = True     # an arm protective stop also stops the base and holds it for 1 s

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown preset {self.preset!r}; choose {sorted(PRESETS)}")
        if self.task not in PRESETS[self.preset]:
            raise ValueError(f"Task {self.task!r} is not in preset {self.preset!r}: {PRESETS[self.preset]}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        from .variants import resolve
        resolve(self.preset, self.task, self.seed, self.variant)
        from classroom_arrangement.params import REFLEX_PROFILES
        if self.reflex_profile not in REFLEX_PROFILES:
            raise ValueError(f"reflex_profile must be one of {sorted(REFLEX_PROFILES)}")
        if type(self.policy_samples) is not int or not 1 <= self.policy_samples <= 256:
            raise ValueError("policy_samples must be an integer from 1 to 256")

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**d)

    def site(self):
        from .variants import site
        return site(self.task, self.seed, self.variant)
