"""Preserved object parameters; documented numerical migration settings."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Config:
    dt: float = 1 / 960
    device: str = "cuda:0"
    seed: int = 0
    scene: str = "baseline"
    pen_mass: float = .0058
    cap_mass: float = .005
    shaft_radius: float = .0049
    bead_radius: float = .00525
    interference: float = .00015
    bore_radius: float = .0054
    cap_radius: float = .0062
    cap_length: float = .058
    pen_length: float = .140
    bead_u: float = .038
    bead_flat: float = .0003
    lead_deg: float = 22.
    return_deg: float = 50.
    nub_radius: float = .00022
    nub_count: int = 4
    nub_offset: float = .0015
    click_gap: float = .0004
    contact_gap: float = .00004
    nub_stiffness: float = 3e4
    nub_damping: float = 600.
    plastic_friction: float = .25
    sdf_resolution: int = 768
    solver_iterations: int = 30
    contact_capacity: int = 4096
    table_top: float = .75

    def __post_init__(self):
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if self.scene not in ("baseline", "studio"):
            raise ValueError("scene must be baseline or studio")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        for name in ("nub_count", "sdf_resolution", "solver_iterations", "contact_capacity"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("pen_mass", "cap_mass", "shaft_radius", "bead_radius", "interference",
                     "bore_radius", "cap_radius", "cap_length", "pen_length", "bead_u", "bead_flat",
                     "nub_radius", "nub_offset", "click_gap", "contact_gap", "nub_stiffness", "table_top"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.lead_deg < 90 or not 0 < self.return_deg < 90:
            raise ValueError("Ramp angles must be between 0 and 90 degrees")
        if not self.cap_radius > self.bore_radius > self.shaft_radius or self.bead_radius <= self.shaft_radius:
            raise ValueError("Invalid hollow-cap or bead dimensions")
        if any(not math.isfinite(v) or v < 0 for v in (self.nub_damping, self.plastic_friction)):
            raise ValueError("Damping and friction must be finite and nonnegative")

    @property
    def lead_run(self):
        return (self.bead_radius-self.shaft_radius)/math.tan(math.radians(self.lead_deg))

    @property
    def return_run(self):
        return (self.bead_radius-self.shaft_radius)/math.tan(math.radians(self.return_deg))

    @property
    def seat_u(self):
        return self.bead_u + self.bead_flat/2 + self.nub_radius + self.nub_offset

    @property
    def shoulder_u(self):
        return self.seat_u + self.click_gap
