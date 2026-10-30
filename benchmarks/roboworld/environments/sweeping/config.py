"""Launch configuration: preset, task, seed, robot and variation selectors. Unknown fields fail (unified contract)."""

from dataclasses import dataclass, fields
from .layout import PRESETS, VARIANTS

GLIDES = ("nylon", "felt", "steel")


@dataclass
class Config:
    preset: str = "room112"
    task: str = "gather-open"
    seed: int = 0
    robot: object = "mobile-fr3-duo"   # catalog alias, or a qualified gar.robot-setup.v1 object (backend validates)
    variant: str = "default"           # layout.VARIANTS: which tools are out, which glides the chairs have
    glides: str = "nylon"              # Virco glide option on tables and chairs: nylon, felt, steel
    amount: float = 1.0                # scale of the debris amounts (0.25 to 1.5)
    reflex_profile: str = "contact_x5"   # FR3 collision thresholds (classroom_arrangement params.REFLEX_PROFILES)
    policy_samples: int = 16           # path-traced samples per pixel of a policy frame
    reflex_stops_base: bool = True     # an arm protective stop also stops the base and holds it for 1 s
    device: str = "cuda:0"             # where the bristle and debris solver runs

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown preset {self.preset!r}; choose {sorted(PRESETS)}")
        if self.task not in PRESETS[self.preset]:
            raise ValueError(f"Task {self.task!r} is not in preset {self.preset!r}: {PRESETS[self.preset]}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if self.glides not in GLIDES:
            raise ValueError(f"glides must be one of {GLIDES}")
        if isinstance(self.amount, bool) or not isinstance(self.amount, (int, float)) or not 0.25 <= self.amount <= 1.5:
            raise ValueError("amount must be a number from 0.25 to 1.5")
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
