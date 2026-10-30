"""SI units. Named-product dimensions and unmeasured constitutive priors are separate."""

from dataclasses import dataclass
import math
import re
from pathlib import Path
from .assets import URDF

ROOT = Path(__file__).resolve().parents[1]
G1_URDF = URDF

G1_SHA256 = "555edb931dbe8780a82c778f7c7f53fe7f4f6d73c7e791845631cda9e8042302"

COLORS = {
    "black": (0.025, 0.031, 0.042),
    "red": (0.72, 0.028, 0.045),
    "blue": (0.024, 0.10, 0.63),
    "green": (0.02, 0.34, 0.12),
    "orange": (0.93, 0.26, 0.025),
    "violet": (0.34, 0.045, 0.48),
}


@dataclass(frozen=True)
class Board:
    width: float = 2.4384
    height: float = 1.2192
    bottom: float = 0.9144
    trim: float = 0.022225
    face_y: float = 0.095
    backer: float = 0.0111125

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in self.__dict__.values()):
            raise ValueError("Board dimensions and placement must be finite and positive")
        if 2 * self.trim >= min(self.width, self.height):
            raise ValueError("Board trim consumes the writing surface")

    @property
    def writing_width(self):
        return self.width - 2 * self.trim

    @property
    def writing_height(self):
        return self.height - 2 * self.trim

    @property
    def center_z(self):
        return self.bottom + self.height / 2


@dataclass(frozen=True)
class MaterialPriors:
    """Engineering priors, NOT manufacturer measurements."""

    nib_damping_ns_m: float = 0.45
    nib_friction: float = 0.35
    # The 1.0 N s/m felt prior scaled by sqrt(207 kPa / 2.2 kPa) keeps its damping
    # ratio after the felt modulus was re-sourced. Unscaled, the felt rebounds off
    # the board (restitution ~.98) and the free magnetic duster drops 4.4 mm.
    pad_damping_ns_m: float = 9.70
    pad_friction: float = 0.55
    magnet_surface_force_n: float = 1.4
    magnet_gap_scale_m: float = 0.0015
    ink_reservoir_ul: float = 1500.0
    ink_film_um: float = 3.0
    ink_dry_time_s: float = 12.0
    erasure_distance_m: float = 0.018
    eraser_mass_kg: float = 0.030
    chair_mass_kg: float = 4.7
    desk_mass_kg: float = 12.0

    def __post_init__(self):
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            for v in self.__dict__.values()
        ):
            raise ValueError("Material priors must be finite positive numbers")


@dataclass(frozen=True)
class SceneLayout:
    """Placements, never a rescaling of named physical products."""

    marker_order: tuple = tuple(COLORS)
    marker_offset_x: float = 0.0
    eraser_x: float = 0.62
    furniture_offset_x: float = 0.0
    robot_x: float = -0.60
    robot_y: float = 1.15
    robot_yaw_rad: float = -math.pi / 2

    def __post_init__(self):
        if len(self.marker_order) != 6 or set(self.marker_order) != set(COLORS):
            raise ValueError("marker_order must be a permutation of all six physical markers")
        for name, value in self.__dict__.items():
            if name != "marker_order" and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError("Layout positions must be finite numbers")
        if not (
            -0.10 <= self.marker_offset_x <= 0.10
            and 0.35 <= self.eraser_x <= 0.95
            and abs(self.furniture_offset_x) <= 0.08
            and -0.85 <= self.robot_x <= 0.10
            and 1.02 <= self.robot_y <= 1.35
            and abs(self.robot_yaw_rad + math.pi / 2) <= 0.15
        ):
            raise ValueError("Layout outside the validated classroom placement envelope")


@dataclass(frozen=True)
class Config:
    physics_dt: float = 0.0005
    texture_width: int = 4096
    texture_height: int = 2048
    device: str = "cpu"
    noslip_iterations: int = 10
    deterministic: bool = True
    jacobian: str = "auto"
    constraint_solver: str = "newton"
    seed: int = 0
    robot: bool = True
    furniture: bool = True
    episode_ticks: int = 1200000  # 10 simulated minutes at the default 0.5 ms.
    board: Board = Board()
    materials: MaterialPriors = MaterialPriors()
    layout: SceneLayout = SceneLayout()

    def __post_init__(self):
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be boolean")
        if self.jacobian not in ("auto", "dense", "sparse"):
            raise ValueError("jacobian must be auto, dense, or sparse")
        if self.constraint_solver not in ("newton", "cg"):
            raise ValueError("constraint_solver must be newton or cg")
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", self.device):
            raise ValueError("device must be cpu or an explicit cuda:N device")
        if (
            isinstance(self.noslip_iterations, bool)
            or not isinstance(self.noslip_iterations, int)
            or self.noslip_iterations < 0
        ):
            raise ValueError("noslip_iterations must be a nonnegative integer")
        if self.device != "cpu" and self.noslip_iterations:
            raise ValueError(
                "MuJoCo Warp 3.11 has no noslip postsolver; explicitly select noslip_iterations=0 for GPU/CPU comparison"
            )
        if self.device != "cpu" and self.deterministic and self.constraint_solver != "cg":
            raise ValueError(
                'Verified deterministic GPU configuration requires explicit constraint_solver="cg"; no silent solver fallback'
            )
        if not 0 < self.physics_dt <= 0.002:
            raise ValueError("physics_dt must be positive and <= 0.002 s; no implicit substeps")
        if self.texture_width < 1 or self.texture_height < 1:
            raise ValueError("Texture dimensions must be positive")
        if any(
            isinstance(v, bool) or not isinstance(v, int)
            for v in (self.texture_width, self.texture_height, self.seed, self.episode_ticks)
        ):
            raise ValueError("Texture dimensions and seed must be integers")
        if self.episode_ticks < 1:
            raise ValueError("episode_ticks must be positive")
