"""Explicit, immutable migration configuration; SI units throughout."""

from dataclasses import dataclass
import math

from football.football_specs import BallSpec


@dataclass(frozen=True)
class Config:
    device: str = "cpu"
    dt: float = 0.002
    iterations: int = 64
    contact_hz: int = 20_000
    net_hz: int = 24_000
    robot: bool = True
    net_layout: str = "enclosure"
    seed: int = 0
    ball_position: tuple = (-11.0, 0.0, BallSpec().radius_m + 0.001)
    ball_velocity: tuple = (0.0, 0.0, 0.0)
    ball_omega: tuple = (0.0, 0.0, 0.0)
    robot_position: tuple = (-11.42, 0.0, 0.80)
    robot_yaw: float = 0.0
    # Catalogue task (variants.py): selects its static props and verifier. Its start state is the five fields
    # above, which variants.apply sets; goal's are the defaults.
    task: str = "goal"

    def __post_init__(self):
        from .variants import validate

        validate(self.task, self.seed)
        if self.device != "cpu":
            raise ValueError(
                "This qualified backend uses Newton/MuJoCo CPU; CUDA is not qualified"
            )
        if not math.isfinite(self.robot_yaw):
            raise ValueError("robot_yaw must be finite")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be positive and finite")
        for name in ("iterations", "contact_hz", "net_hz"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.net_layout not in {"none", "back", "enclosure"}:
            raise ValueError("Unknown net_layout")
        for name in ("ball_position", "ball_velocity", "ball_omega", "robot_position"):
            value = getattr(self, name)
            if len(value) != 3 or not all(math.isfinite(x) for x in value):
                raise ValueError(f"{name} must contain three finite SI values")
