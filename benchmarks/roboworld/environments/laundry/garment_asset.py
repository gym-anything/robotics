"""Construction-level garment assets.

The original laundry environment starts from one regular rectangular grid.  A
real garment is instead made from multiple two-dimensional pattern pieces that
are joined along selected boundary paths.  This module is the engine-neutral
contract for those pieces and joins.  It deliberately separates:

* material-space pattern geometry from an assembled three-dimensional pose;
* sewn joins from open, hemmed, or raw boundaries;
* physical panels from local constructions such as pockets and waistbands;
* executable priors from measured/calibrated specimen data.

Nothing in this module claims that a pattern has passed a dynamics probe.  It
is the validated input that an arbitrary-panel cloth compiler can consume.
All geometry is expressed in metres and all material identifiers resolve to
``MATERIAL_PROFILES``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .garments import GarmentTopology
from .specs import MATERIAL_PROFILES


Point2 = tuple[float, float]
_GEOMETRY_TOLERANCE_M = 1.0e-8


class PathRole(str, Enum):
    """Physical role of a named pattern path."""

    SEAM = "seam"
    HEM = "hem"
    OPENING = "opening"
    RAW = "raw"
    INTERNAL_STITCH = "internal_stitch"
    FOLD_REFERENCE = "fold_reference"
    GRAIN_REFERENCE = "grain_reference"


class SeamType(str, Enum):
    """Construction types retained because they change local mechanics."""

    PLAIN = "plain"
    OVERLOCK = "overlock"
    COVERSTITCH = "coverstitch"
    FLAT_FELLED = "flat_felled"
    BOUND = "bound"
    LAPPED = "lapped"
    PATCH = "patch"
    WAISTBAND = "waistband"
    COLLAR = "collar"
    CUFF = "cuff"


class OpeningKind(str, Enum):
    NECK = "neck"
    ARM = "arm"
    WRIST = "wrist"
    WAIST = "waist"
    LEG = "leg"
    FRONT = "front"
    POCKET = "pocket"
    PERIMETER = "perimeter"


class FeatureKind(str, Enum):
    HEM_BAND = "hem_band"
    RIB_BAND = "rib_band"
    COLLAR = "collar"
    CUFF = "cuff"
    PLACKET = "placket"
    BUTTON = "button"
    BUTTONHOLE = "buttonhole"
    ZIPPER = "zipper"
    POCKET = "pocket"
    POCKET_BAG = "pocket_bag"
    WAISTBAND = "waistband"
    BELT_LOOP = "belt_loop"
    INTERLINING = "interlining"
    DOBBY_BAND = "dobby_band"
    TIE = "tie"


class ProvenanceKind(str, Enum):
    REPOSITORY_AUTHORED = "repository_authored"
    MEASURED_SPECIMEN = "measured_specimen"
    THIRD_PARTY_PATTERN = "third_party_pattern"
    PROCEDURAL = "procedural"


def _is_finite_point(point: Point2) -> bool:
    return len(point) == 2 and all(math.isfinite(float(value)) for value in point)


def _distance(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _cross(a: Point2, b: Point2, c: Point2) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_on_segment(point: Point2, a: Point2, b: Point2, tolerance: float) -> bool:
    if abs(_cross(a, b, point)) > tolerance:
        return False
    return (
        min(a[0], b[0]) - tolerance <= point[0] <= max(a[0], b[0]) + tolerance
        and min(a[1], b[1]) - tolerance <= point[1] <= max(a[1], b[1]) + tolerance
    )


def _orientation(a: Point2, b: Point2, c: Point2, tolerance: float) -> int:
    value = _cross(a, b, c)
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _segments_intersect(
    a: Point2,
    b: Point2,
    c: Point2,
    d: Point2,
    tolerance: float = _GEOMETRY_TOLERANCE_M,
) -> bool:
    o1 = _orientation(a, b, c, tolerance)
    o2 = _orientation(a, b, d, tolerance)
    o3 = _orientation(c, d, a, tolerance)
    o4 = _orientation(c, d, b, tolerance)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and _point_on_segment(c, a, b, tolerance))
        or (o2 == 0 and _point_on_segment(d, a, b, tolerance))
        or (o3 == 0 and _point_on_segment(a, c, d, tolerance))
        or (o4 == 0 and _point_on_segment(b, c, d, tolerance))
    )


def signed_polygon_area(points: Sequence[Point2]) -> float:
    """Return signed polygon area; counter-clockwise polygons are positive."""

    if len(points) < 3:
        return 0.0
    return 0.5 * sum(
        a[0] * b[1] - b[0] * a[1]
        for a, b in zip(points, points[1:] + points[:1])
    )


def point_in_polygon(
    point: Point2,
    polygon: Sequence[Point2],
    *,
    include_boundary: bool = True,
) -> bool:
    """Ray-casting containment test without an optional geometry dependency."""

    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if _point_on_segment(point, a, b, _GEOMETRY_TOLERANCE_M):
            return include_boundary
        if (a[1] > point[1]) != (b[1] > point[1]):
            intersection_x = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if intersection_x > point[0]:
                inside = not inside
    return inside


@dataclass(frozen=True)
class PatternPath:
    """Named polyline in material space."""

    id: str
    points_m: tuple[Point2, ...]
    role: PathRole
    semantic_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Pattern paths require an identifier")
        if len(self.points_m) < 2:
            raise ValueError(f"Path {self.id!r} requires at least two points")
        if any(not _is_finite_point(point) for point in self.points_m):
            raise ValueError(f"Path {self.id!r} contains a non-finite point")
        if any(
            _distance(a, b) <= _GEOMETRY_TOLERANCE_M
            for a, b in zip(self.points_m[:-1], self.points_m[1:])
        ):
            raise ValueError(f"Path {self.id!r} contains a zero-length segment")
        if len(set(self.semantic_tags)) != len(self.semantic_tags):
            raise ValueError(f"Path {self.id!r} has duplicate semantic tags")

    @property
    def length_m(self) -> float:
        return sum(_distance(a, b) for a, b in zip(self.points_m[:-1], self.points_m[1:]))

    def point_at_fraction(self, fraction: float) -> Point2:
        """Interpolate by arclength, not by polyline vertex index."""

        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("Path fraction must lie in [0, 1]")
        if fraction == 0.0:
            return self.points_m[0]
        if fraction == 1.0:
            return self.points_m[-1]
        target = fraction * self.length_m
        traversed = 0.0
        for a, b in zip(self.points_m[:-1], self.points_m[1:]):
            segment_length = _distance(a, b)
            if traversed + segment_length >= target:
                local = (target - traversed) / segment_length
                return (
                    a[0] + local * (b[0] - a[0]),
                    a[1] + local * (b[1] - a[1]),
                )
            traversed += segment_length
        return self.points_m[-1]

    def resample(self, count: int, *, reverse: bool = False) -> tuple[Point2, ...]:
        if count < 2:
            raise ValueError("A path resampling requires at least two points")
        fractions: Iterable[float] = (index / (count - 1) for index in range(count))
        points = tuple(self.point_at_fraction(fraction) for fraction in fractions)
        return tuple(reversed(points)) if reverse else points


@dataclass(frozen=True)
class PanelPattern:
    """One physical cut panel before assembly.

    ``seed_mirror_material_x`` turns a cut piece over only in the authored
    assembly seed. ``physical_winding_reversed`` reverses the corresponding
    physical and material triangle corners passed to Style3D so every shared
    sewn edge has opposite traversal. They are independent, explicit pieces
    of construction metadata rather than render-only normal switches.
    """

    id: str
    label: str
    material_profile: str
    areal_density_kg_m2: float
    unloaded_thickness_m: float
    grain_direction: Point2
    outline_paths: tuple[PatternPath, ...]
    internal_paths: tuple[PatternPath, ...] = ()
    display_color_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
    semantic_tags: tuple[str, ...] = ()
    seed_mirror_material_x: bool = False
    physical_winding_reversed: bool = False

    def __post_init__(self) -> None:
        if not self.id or not self.label:
            raise ValueError("Panels require identifiers and labels")
        if self.material_profile not in MATERIAL_PROFILES:
            raise ValueError(
                f"Panel {self.id!r} references unknown material {self.material_profile!r}"
            )
        if min(self.areal_density_kg_m2, self.unloaded_thickness_m) <= 0.0:
            raise ValueError(f"Panel {self.id!r} density and thickness must be positive")
        if not _is_finite_point(self.grain_direction):
            raise ValueError(f"Panel {self.id!r} has a non-finite grain direction")
        grain_norm = math.hypot(*self.grain_direction)
        if grain_norm <= _GEOMETRY_TOLERANCE_M:
            raise ValueError(f"Panel {self.id!r} grain direction cannot be zero")
        if len(self.outline_paths) < 3:
            raise ValueError(f"Panel {self.id!r} needs at least three outline paths")

        all_paths = self.outline_paths + self.internal_paths
        path_ids = tuple(path.id for path in all_paths)
        if len(set(path_ids)) != len(path_ids):
            raise ValueError(f"Panel {self.id!r} has duplicate path identifiers")
        allowed_outline_roles = {PathRole.SEAM, PathRole.HEM, PathRole.OPENING, PathRole.RAW}
        if any(path.role not in allowed_outline_roles for path in self.outline_paths):
            raise ValueError(f"Panel {self.id!r} has a non-boundary role on its outline")

        for first, second in zip(self.outline_paths, self.outline_paths[1:]):
            if _distance(first.points_m[-1], second.points_m[0]) > _GEOMETRY_TOLERANCE_M:
                raise ValueError(
                    f"Panel {self.id!r} outline is open between {first.id!r} and {second.id!r}"
                )
        if _distance(self.outline_paths[-1].points_m[-1], self.outline_paths[0].points_m[0]) > _GEOMETRY_TOLERANCE_M:
            raise ValueError(f"Panel {self.id!r} outline does not close")

        polygon = self.outline_polygon
        if signed_polygon_area(polygon) <= _GEOMETRY_TOLERANCE_M**2:
            raise ValueError(f"Panel {self.id!r} outline must be counter-clockwise and nonzero")
        self._validate_simple_polygon(polygon)
        for path in self.internal_paths:
            if any(not point_in_polygon(point, polygon) for point in path.points_m):
                raise ValueError(
                    f"Internal path {path.id!r} leaves panel {self.id!r}"
                )
        if any(not 0.0 <= component <= 1.0 for component in self.display_color_rgb):
            raise ValueError(f"Panel {self.id!r} display colour must lie in [0, 1]")
        if len(set(self.semantic_tags)) != len(self.semantic_tags):
            raise ValueError(f"Panel {self.id!r} has duplicate semantic tags")
        if not isinstance(self.seed_mirror_material_x, bool):
            raise ValueError(
                f"Panel {self.id!r} seed mirror flag must be a boolean"
            )
        if not isinstance(self.physical_winding_reversed, bool):
            raise ValueError(
                f"Panel {self.id!r} physical winding flag must be a boolean"
            )

    @staticmethod
    def _validate_simple_polygon(polygon: Sequence[Point2]) -> None:
        count = len(polygon)
        for i in range(count):
            a = polygon[i]
            b = polygon[(i + 1) % count]
            for j in range(i + 1, count):
                if j in {i, (i + 1) % count} or (j + 1) % count == i:
                    continue
                c = polygon[j]
                d = polygon[(j + 1) % count]
                if _segments_intersect(a, b, c, d):
                    raise ValueError("Panel outline self-intersects")

    @property
    def outline_polygon(self) -> tuple[Point2, ...]:
        points: list[Point2] = []
        for path in self.outline_paths:
            points.extend(path.points_m[:-1])
        return tuple(points)

    @property
    def area_m2(self) -> float:
        return signed_polygon_area(self.outline_polygon)

    @property
    def mass_kg(self) -> float:
        return self.area_m2 * self.areal_density_kg_m2

    @property
    def paths(self) -> Mapping[str, PatternPath]:
        return MappingProxyType(
            {path.id: path for path in self.outline_paths + self.internal_paths}
        )

    def path(self, path_id: str) -> PatternPath:
        try:
            return self.paths[path_id]
        except KeyError as exception:
            raise KeyError(f"Panel {self.id!r} has no path {path_id!r}") from exception


@dataclass(frozen=True)
class PathRef:
    panel_id: str
    path_id: str
    start_fraction: float = 0.0
    end_fraction: float = 1.0
    reverse: bool = False

    def __post_init__(self) -> None:
        if not self.panel_id or not self.path_id:
            raise ValueError("Path references require panel and path identifiers")
        if not (
            math.isfinite(self.start_fraction)
            and math.isfinite(self.end_fraction)
            and 0.0 <= self.start_fraction < self.end_fraction <= 1.0
        ):
            raise ValueError("Path reference fractions must satisfy 0 <= start < end <= 1")

    def length_m(self, panels: Mapping[str, PanelPattern]) -> float:
        return (
            panels[self.panel_id].path(self.path_id).length_m
            * (self.end_fraction - self.start_fraction)
        )


@dataclass(frozen=True)
class SeamSpec:
    id: str
    side_a: PathRef
    side_b: PathRef
    seam_type: SeamType
    seam_allowance_m: float
    stitch_spacing_m: float
    line_stiffness_n_m: float
    line_damping_n_s_m: float
    maximum_ease_fraction: float = 0.03
    calibration_status: str = "construction prior; seam coupon not measured"

    def __post_init__(self) -> None:
        if not self.id or not self.calibration_status:
            raise ValueError("Seams require an identifier and evidence status")
        if min(
            self.seam_allowance_m,
            self.stitch_spacing_m,
            self.line_stiffness_n_m,
        ) <= 0.0:
            raise ValueError(f"Seam {self.id!r} dimensions/stiffness must be positive")
        if self.line_damping_n_s_m < 0.0:
            raise ValueError(f"Seam {self.id!r} damping cannot be negative")
        if not 0.0 <= self.maximum_ease_fraction < 1.0:
            raise ValueError(f"Seam {self.id!r} ease fraction must lie in [0, 1)")
        if self.side_a == self.side_b:
            raise ValueError(f"Seam {self.id!r} cannot stitch a path reference to itself")


@dataclass(frozen=True)
class OpeningSpec:
    id: str
    kind: OpeningKind
    boundary: tuple[PathRef, ...]
    closable: bool = False
    semantic_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.boundary:
            raise ValueError("Openings require an identifier and at least one boundary")


@dataclass(frozen=True)
class LocalFeatureSpec:
    id: str
    kind: FeatureKind
    panel_ids: tuple[str, ...]
    path_refs: tuple[PathRef, ...] = ()
    width_m: float | None = None
    layer_count: int = 1
    material_profile: str | None = None
    calibration_status: str = "construction prior; local mechanics not calibrated"
    semantic_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.panel_ids or not self.calibration_status:
            raise ValueError("Local features require an id, host panels, and evidence status")
        if self.width_m is not None and self.width_m <= 0.0:
            raise ValueError(f"Feature {self.id!r} width must be positive")
        if self.layer_count < 1:
            raise ValueError(f"Feature {self.id!r} layer count must be positive")
        if self.material_profile is not None and self.material_profile not in MATERIAL_PROFILES:
            raise ValueError(f"Feature {self.id!r} references an unknown material")


@dataclass(frozen=True)
class AssetProvenance:
    kind: ProvenanceKind
    source_uri: str
    license_spdx: str
    note: str

    def __post_init__(self) -> None:
        if not self.source_uri or not self.license_spdx or not self.note:
            raise ValueError("Asset provenance requires source, license, and a note")


@dataclass(frozen=True)
class GarmentConstruction:
    """Validated set of physical panels and their construction graph."""

    id: str
    label: str
    category: str
    topology: GarmentTopology
    panels: tuple[PanelPattern, ...]
    seams: tuple[SeamSpec, ...]
    openings: tuple[OpeningSpec, ...]
    features: tuple[LocalFeatureSpec, ...]
    supported_tasks: tuple[str, ...]
    provenance: AssetProvenance
    calibration_status: str

    def __post_init__(self) -> None:
        if not self.id or not self.label or not self.category or not self.calibration_status:
            raise ValueError("Garments require identifiers, labels, categories, and evidence status")
        if not self.panels:
            raise ValueError(f"Garment {self.id!r} requires at least one physical panel")
        if self.topology not in {GarmentTopology.RECTANGULAR_SHEET, GarmentTopology.SEWN_PANEL_SHELL}:
            raise ValueError(f"Garment {self.id!r} uses an unsupported construction topology")

        panel_ids = tuple(panel.id for panel in self.panels)
        self._require_unique(panel_ids, "panel")
        self._require_unique((seam.id for seam in self.seams), "seam")
        self._require_unique((opening.id for opening in self.openings), "opening")
        self._require_unique((feature.id for feature in self.features), "feature")
        self._require_unique(self.supported_tasks, "task")
        panels = self.panel_map

        for seam in self.seams:
            self._validate_path_ref(seam.side_a, panels, f"seam {seam.id!r}")
            self._validate_path_ref(seam.side_b, panels, f"seam {seam.id!r}")
            length_a = seam.side_a.length_m(panels)
            length_b = seam.side_b.length_m(panels)
            mismatch = abs(length_a - length_b) / max(length_a, length_b)
            if mismatch > seam.maximum_ease_fraction + 1.0e-12:
                raise ValueError(
                    f"Seam {seam.id!r} path mismatch {mismatch:.3f} exceeds "
                    f"declared ease {seam.maximum_ease_fraction:.3f}"
                )

        for opening in self.openings:
            for path_ref in opening.boundary:
                self._validate_path_ref(path_ref, panels, f"opening {opening.id!r}")
                role = panels[path_ref.panel_id].path(path_ref.path_id).role
                if role not in {PathRole.OPENING, PathRole.HEM}:
                    raise ValueError(
                        f"Opening {opening.id!r} references path {path_ref.path_id!r} "
                        f"with incompatible role {role.value!r}"
                    )

        for feature in self.features:
            for panel_id in feature.panel_ids:
                if panel_id not in panels:
                    raise ValueError(
                        f"Feature {feature.id!r} references missing panel {panel_id!r}"
                    )
            for path_ref in feature.path_refs:
                self._validate_path_ref(path_ref, panels, f"feature {feature.id!r}")

    @staticmethod
    def _require_unique(values: Iterable[str], label: str) -> None:
        materialized = tuple(values)
        if any(not value for value in materialized) or len(set(materialized)) != len(materialized):
            raise ValueError(f"Garment {label} identifiers must be nonempty and unique")

    @staticmethod
    def _validate_path_ref(
        path_ref: PathRef,
        panels: Mapping[str, PanelPattern],
        owner: str,
    ) -> None:
        if path_ref.panel_id not in panels:
            raise ValueError(f"{owner} references missing panel {path_ref.panel_id!r}")
        if path_ref.path_id not in panels[path_ref.panel_id].paths:
            raise ValueError(
                f"{owner} references missing path {path_ref.panel_id!r}:{path_ref.path_id!r}"
            )

    @property
    def panel_map(self) -> Mapping[str, PanelPattern]:
        return MappingProxyType({panel.id: panel for panel in self.panels})

    @property
    def total_panel_area_m2(self) -> float:
        return sum(panel.area_m2 for panel in self.panels)

    @property
    def total_panel_mass_kg(self) -> float:
        return sum(panel.mass_kg for panel in self.panels)

    @property
    def topology_signature(self) -> tuple[object, ...]:
        """Stable structural fingerprint, intentionally independent of labels."""

        return (
            self.topology.value,
            len(self.panels),
            tuple(sorted(len(panel.outline_paths) for panel in self.panels)),
            len(self.seams),
            tuple(sorted(seam.seam_type.value for seam in self.seams)),
            len(self.openings),
            tuple(sorted(opening.kind.value for opening in self.openings)),
            tuple(sorted(feature.kind.value for feature in self.features)),
        )

    def construction_summary(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "id": self.id,
                "label": self.label,
                "category": self.category,
                "topology": self.topology.value,
                "panel_count": len(self.panels),
                "seam_count": len(self.seams),
                "opening_count": len(self.openings),
                "feature_count": len(self.features),
                "total_panel_area_m2": self.total_panel_area_m2,
                "total_panel_mass_kg": self.total_panel_mass_kg,
                "material_profiles": tuple(sorted({panel.material_profile for panel in self.panels})),
                "calibration_status": self.calibration_status,
            }
        )
