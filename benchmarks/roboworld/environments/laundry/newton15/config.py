"""Explicit SI workcell configuration. No hidden fidelity substitutions."""
from dataclasses import dataclass
import math

GARMENTS = {
    "tee": "cotton_jersey_half_sleeve_tee",
    "shirt": "woven_button_down_shirt",
    "jeans": "denim_five_pocket_jeans",
    "towel": "terry_cotton_bath_towel",
    "asymmetric": "asymmetric_draped_one_sleeve_top",
}


@dataclass(frozen=True)
class Config:
    garment: str = "towel"
    device: str = "cuda:0"
    dt: float = 1/1440
    nonlinear_iterations: int = 20
    linear_iterations: int = 60
    rigid_iterations: int = 30
    table_top: float = .78
    initial_table_clearance: float = .0085
    table_half_x: float = .80
    table_half_y: float = .70
    robot_base_y: float = .78
    contact_capacity: int = 8192
    edge_length: float = .04
    coupling: str = "lagged"
    capture_cloth: bool = True
    capture_rigid: bool = True
    capture_collision: bool = True
    contact_damping: float = 10.
    local_contact_projection: bool = True
    ipc_threads: int | None = None

    def __post_init__(self):
        if any(type(v) is not bool for v in (self.capture_cloth, self.capture_rigid, self.capture_collision, self.local_contact_projection)):
            raise ValueError("capture flags must be boolean")
        if self.garment not in GARMENTS:
            raise ValueError(f"garment must be one of {tuple(GARMENTS)}")
        for name in ("dt", "table_top", "initial_table_clearance", "table_half_x", "table_half_y", "robot_base_y", "edge_length", "contact_damping"):
            v = getattr(self, name)
            if isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("nonlinear_iterations", "linear_iterations", "rigid_iterations", "contact_capacity"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.ipc_threads is not None and (type(self.ipc_threads) is not int or self.ipc_threads < 1):
            raise ValueError("ipc_threads must be a positive integer or None for the IPC default")
        if self.coupling != "lagged":
            raise ValueError("Only the qualified lagged contact path is supported")
        if self.edge_length != .04:
            raise ValueError("The retained P13 reset assets have 40 mm target edges; other resolutions require regeneration")
        if self.linear_iterations > 999:
            raise ValueError("Native PCG capacity is 999 iterations; refusing silent truncation")
        if self.robot_base_y <= self.table_half_y:
            raise ValueError("Robot pedestals must be outside the folding tabletop")
