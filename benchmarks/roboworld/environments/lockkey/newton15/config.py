"""SI configuration. Engine migration retains the authored five-pin inventory."""

from dataclasses import dataclass, fields
import math
from lockkey import lock_specs as LK


@dataclass(frozen=True)
class Config:
    device: str = "cuda:0"
    hz: int = 1920
    solver_iterations: int = 64
    solver_algorithm: str = "newton"
    scaled_scene_contacts: bool = True
    plug_geometry: str = "complete"
    box_geometry: str = "clear_lid"
    aligned_casing: bool = True
    mechanical_latch: bool = True
    shackle_swivel: bool = True
    shackle_contact_time_constant: float = .0011
    shackle_contact_damping_ratio: float = 1.
    contact_capacity: int = 2048
    bitting: str = "31542"
    wrong_bitting: str = "12345"
    plug_length: float = .028
    bore_clearance: float = .00010
    plug_clearance: float = .00005
    keyway_half_width: float = .00105
    blade_clearance: float = .00010
    root_tare: float = .00020
    ledge_z: float = -.002
    pin_chamfer: float = .00030
    bottom_pin_tip: float = .0012
    contact_gap: float = .00001
    contact_ke: float = 1e7
    contact_kd: float = 500.
    contact_impedance: float = .9999
    contact_time_constant: float = .0021
    contact_damping_ratio: float = 2.
    limit_time_constant: float = .0021
    brass_friction: float = .10
    spring_k: float = 40.
    spring_d: float = .6
    spring_target: float = -.0075
    insertion_travel: float = .012
    # Studio workbench elevation; all task/robot relative positions are retained.
    table_height: float = .75

    @classmethod
    def from_recorded(cls, values):
        """Interpret old recordings without inheriting subsequently upgraded defaults."""
        old_features = dict(solver_algorithm="newton", scaled_scene_contacts=False,
            plug_geometry="legacy", box_geometry="legacy", aligned_casing=False, mechanical_latch=False,
            shackle_swivel=False, shackle_contact_time_constant=.0021, shackle_contact_damping_ratio=2.)
        return cls(**{**old_features, **values})

    def __post_init__(self):
        if self.solver_algorithm not in ("newton", "cg"):
            raise ValueError("solver_algorithm must be newton or cg")
        if self.plug_geometry not in ("legacy", "complete"):
            raise ValueError("plug_geometry must be legacy or complete")
        if self.box_geometry not in ("legacy", "clear_lid"):
            raise ValueError("box_geometry must be legacy or clear_lid")
        for name in ("bitting", "wrong_bitting"):
            v = getattr(self, name)
            if not isinstance(v, str) or len(v) != 5 or any(d not in "1234567" for d in v):
                raise ValueError(f"{name} must specify five KW1 depths 1..7")
        for name in ("hz", "solver_iterations", "contact_capacity"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("scaled_scene_contacts", "aligned_casing", "mechanical_latch", "shackle_swivel"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        signed = {"ledge_z", "spring_target"}
        for f in fields(self):
            if f.type is float:
                v = getattr(self, f.name)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise ValueError(f"{f.name} must be finite")
                if f.name not in signed and v <= 0:
                    raise ValueError(f"{f.name} must be positive")
        if not 0 < self.contact_impedance < 1 or self.blade_clearance >= 2*self.keyway_half_width:
            raise ValueError("Invalid contact impedance or blade clearance")

    @property
    def dt(self):
        return 1. / self.hz

    @property
    def shackle_travel(self):
        # Short leg starts 10 mm inside the casing. The surrounding staple is
        # another 7 mm above it: 18 mm travel leaves 1 mm sweep clearance and
        # retains 6 mm of the long leg. Preserve the old recorded geometry explicitly.
        return .018 if self.shackle_swivel else .011

    @property
    def spine_z(self):
        return LK.PLUG_R - LK.EDP_IN * LK.IN

    @property
    def keyway_top(self):
        return self.spine_z + LK.BLADE_HEIGHT + .0004

    @property
    def face_x(self):
        return LK.FIRST_CUT + 2 * LK.CUT_SPACING

    @property
    def chambers_x(self):
        return [self.face_x - LK.FIRST_CUT - i * LK.CUT_SPACING for i in range(5)]
