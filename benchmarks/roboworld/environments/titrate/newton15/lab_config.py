"""Dimensional laboratory specification; SI throughout, no policy waypoints."""

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class LabConfig:
    seed: int = 0
    device: str = "cuda:0"
    robot: bool = True
    bench_height: float = .900
    bench_width: float = 1.800
    bench_depth: float = .750
    flask_x: float = .150
    flask_y: float = .035
    # IKA C-MAG MS 4: 150 W x 260 D x 105 H mm; ceramic 100 x 100 mm.
    stirrer_height: float = .105
    # Corning 2103-50: OD 14 mm, overall height 750 mm, 2 mm valve bore.
    burette_outer_radius: float = .007
    burette_inner_radius: float = .00525  # reconstructed wall, not manufacturer CAD
    burette_height: float = .750
    valve_bore_radius: float = .001
    valve_plug_radius: float = .007
    valve_plug_half_length: float = .014
    tip_inner_radius: float = .00065  # explicitly estimated drawn tip
    tip_outer_radius: float = .00125
    tip_clearance: float = .008

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0 or type(self.robot) is not bool:
            raise ValueError("Explicit nonnegative integer seed and Boolean robot required")
        for key, value in asdict(self).items():
            if type(value) is float and (not math.isfinite(value) or (value <= 0 and key not in ("flask_x", "flask_y"))):
                raise ValueError(f"Invalid dimensional parameter {key}")
        if not self.tip_inner_radius < self.tip_outer_radius or not self.burette_inner_radius < self.burette_outer_radius:
            raise ValueError("Hollow apparatus requires positive wall thickness")

    @property
    def flask_position(self):
        return (self.flask_x, self.flask_y, self.bench_height+self.stirrer_height+.0002)

    @property
    def tip_position(self):
        return (self.flask_x, self.flask_y, self.flask_position[2]+.114+self.tip_clearance)

    @property
    def valve_position(self):
        return (self.flask_x, self.flask_y, self.tip_position[2]+.055)

    @property
    def camera_eye(self):
        return (.68, -.60, 1.65)

    @property
    def camera_target(self):
        return (self.flask_x, self.flask_y, 1.18)
