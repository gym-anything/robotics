"""Legacy regular-grid backend catalogue with explicit boundaries.

This module describes the original rectangular VBD backend.  The newer
arbitrary-panel constructions live in :mod:`laundry.garment_patterns` and the
official Style3D adapter lives in :mod:`laundry.newton_style3d_garment`.
Keeping these contracts distinct prevents a sewn T-shirt from silently being
instantiated as the legacy rectangular towel mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Mapping

from .specs import MATERIAL_PROFILES, TowelSpec


class GarmentTopology(str, Enum):
    RECTANGULAR_SHEET = "rectangular_sheet"
    SEWN_PANEL_SHELL = "sewn_panel_shell"
    TUBULAR_SHELL = "tubular_shell"
    MULTI_GARMENT_PILE = "multi_garment_pile"


class PhysicsSupport(str, Enum):
    IMPLEMENTED = "implemented"
    RESEARCH_REQUIRED = "research_required"


@dataclass(frozen=True)
class GarmentProfile:
    id: str
    label: str
    topology: GarmentTopology
    physics_support: PhysicsSupport
    material_profile: str | None
    length_m: float | None
    width_m: float | None
    areal_density_kg_m2: float | None
    unloaded_thickness_m: float | None
    supported_tasks: tuple[str, ...]
    calibration_status: str
    implementation_note: str

    def __post_init__(self) -> None:
        if not self.id or not self.label or not self.calibration_status:
            raise ValueError("Garment profiles need identifiers, labels, and status")
        dimensions = (
            self.length_m,
            self.width_m,
            self.areal_density_kg_m2,
            self.unloaded_thickness_m,
        )
        if self.physics_support is PhysicsSupport.IMPLEMENTED:
            if self.topology is not GarmentTopology.RECTANGULAR_SHEET:
                raise ValueError("Only rectangular-sheet garments are implemented")
            if self.material_profile not in MATERIAL_PROFILES:
                raise ValueError("Implemented garment references an unknown material")
            if any(value is None for value in dimensions):
                raise ValueError("Implemented garments require complete dimensions")
        for value in dimensions:
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError("Garment dimensions and densities must be positive")
        if len(set(self.supported_tasks)) != len(self.supported_tasks):
            raise ValueError("Supported task identifiers must be unique")

    @property
    def implemented(self) -> bool:
        return self.physics_support is PhysicsSupport.IMPLEMENTED

    def require_implemented(self) -> None:
        if not self.implemented:
            raise NotImplementedError(
                f"{self.label} requires {self.topology.value} physics: "
                f"{self.implementation_note}"
            )

    def as_towel_spec(self) -> TowelSpec:
        """Return the rectangular Newton-core representation.

        The method name is deliberately concrete: it is unavailable for sewn
        or tubular garments and does not imply that those topologies can be
        approximated faithfully by a rectangle.
        """

        self.require_implemented()
        assert self.length_m is not None
        assert self.width_m is not None
        assert self.areal_density_kg_m2 is not None
        assert self.unloaded_thickness_m is not None
        return TowelSpec(
            length_m=self.length_m,
            width_m=self.width_m,
            areal_density_kg_m2=self.areal_density_kg_m2,
            unloaded_thickness_m=self.unloaded_thickness_m,
            specimen_id=self.id,
            calibration_status=self.calibration_status,
        )


_canonical = TowelSpec()
GARMENT_PROFILES: Mapping[str, GarmentProfile] = MappingProxyType(
    {
        "canonical_cotton_towel": GarmentProfile(
            id="canonical_cotton_towel",
            label="Canonical woven-cotton towel",
            topology=GarmentTopology.RECTANGULAR_SHEET,
            physics_support=PhysicsSupport.IMPLEMENTED,
            material_profile="cotton_towel_prior",
            length_m=_canonical.length_m,
            width_m=_canonical.width_m,
            areal_density_kg_m2=_canonical.areal_density_kg_m2,
            unloaded_thickness_m=_canonical.unloaded_thickness_m,
            supported_tasks=("half_fold",),
            calibration_status=_canonical.calibration_status,
            implementation_note="Orthotropic rectangular VBD mesh is exercised by the P6 diagnostics.",
        ),
        "t_shirt": GarmentProfile(
            id="t_shirt",
            label="Short-sleeve T-shirt",
            topology=GarmentTopology.SEWN_PANEL_SHELL,
            physics_support=PhysicsSupport.RESEARCH_REQUIRED,
            material_profile=None,
            length_m=None,
            width_m=None,
            areal_density_kg_m2=None,
            unloaded_thickness_m=None,
            supported_tasks=(),
            calibration_status=(
                "not available in the legacy rectangular backend; canonical sewn "
                "construction exists but material/seam values remain prior-only"
            ),
            implementation_note=(
                "Use cotton_jersey_half_sleeve_tee through garment_patterns and "
                "the Style3D adapter; it cannot be represented by as_towel_spec()."
            ),
        ),
        "trousers": GarmentProfile(
            id="trousers",
            label="Trousers",
            topology=GarmentTopology.SEWN_PANEL_SHELL,
            physics_support=PhysicsSupport.RESEARCH_REQUIRED,
            material_profile=None,
            length_m=None,
            width_m=None,
            areal_density_kg_m2=None,
            unloaded_thickness_m=None,
            supported_tasks=(),
            calibration_status=(
                "not available in the legacy rectangular backend; canonical five-pocket "
                "jeans construction exists but material/seam values remain prior-only"
            ),
            implementation_note=(
                "Use denim_five_pocket_jeans through garment_patterns and the Style3D "
                "adapter; it cannot be represented by as_towel_spec()."
            ),
        ),
        "hoodie": GarmentProfile(
            id="hoodie",
            label="Hooded sweatshirt",
            topology=GarmentTopology.SEWN_PANEL_SHELL,
            physics_support=PhysicsSupport.RESEARCH_REQUIRED,
            material_profile=None,
            length_m=None,
            width_m=None,
            areal_density_kg_m2=None,
            unloaded_thickness_m=None,
            supported_tasks=(),
            calibration_status="unimplemented; no named specimen or seam calibration",
            implementation_note="Requires multilayer hood, cuffs, ribbing, seams, and drawcords.",
        ),
        "sock": GarmentProfile(
            id="sock",
            label="Sock",
            topology=GarmentTopology.TUBULAR_SHELL,
            physics_support=PhysicsSupport.RESEARCH_REQUIRED,
            material_profile=None,
            length_m=None,
            width_m=None,
            areal_density_kg_m2=None,
            unloaded_thickness_m=None,
            supported_tasks=(),
            calibration_status="unimplemented; no named specimen or knit calibration",
            implementation_note="Requires a closed tubular knit with large-strain constitutive data.",
        ),
        "mixed_pile": GarmentProfile(
            id="mixed_pile",
            label="Mixed laundry pile",
            topology=GarmentTopology.MULTI_GARMENT_PILE,
            physics_support=PhysicsSupport.RESEARCH_REQUIRED,
            material_profile=None,
            length_m=None,
            width_m=None,
            areal_density_kg_m2=None,
            unloaded_thickness_m=None,
            supported_tasks=(),
            calibration_status="unimplemented; component garments are not calibrated",
            implementation_note="Requires multiple validated garments plus robust inter-garment contact.",
        ),
    }
)


def get_garment_profile(profile_id: str) -> GarmentProfile:
    try:
        return GARMENT_PROFILES[profile_id]
    except KeyError as exception:
        raise KeyError(f"Unknown garment profile {profile_id!r}") from exception


def implemented_garment_profiles() -> tuple[GarmentProfile, ...]:
    return tuple(profile for profile in GARMENT_PROFILES.values() if profile.implemented)
