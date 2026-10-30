"""Repository-authored canonical garment constructions.

These patterns are dimensioned engineering priors, not scans of commercial
products.  Their purpose is to make topology, construction, mass, and local
material variation explicit so that the simulator can be developed and
falsified honestly.  Every pattern carries repository provenance and every
material family remains labelled prior-only until physical coupons and named
garment specimens are measured.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .garment_asset import (
    AssetProvenance,
    FeatureKind,
    GarmentConstruction,
    LocalFeatureSpec,
    OpeningKind,
    OpeningSpec,
    PanelPattern,
    PathRef,
    PathRole,
    PatternPath,
    ProvenanceKind,
    SeamSpec,
    SeamType,
)
from .garments import GarmentTopology


Point2 = tuple[float, float]
Section = tuple[str, PathRole, tuple[Point2, ...], tuple[str, ...]]


_REPOSITORY_PROVENANCE = AssetProvenance(
    kind=ProvenanceKind.REPOSITORY_AUTHORED,
    source_uri="repo://laundry/garment_patterns.py",
    license_spdx="MIT",
    note=(
        "Procedural research pattern authored for this repository; dimensions are "
        "engineering priors and are not copied from or fitted to a commercial garment."
    ),
)


def _quadratic(start: Point2, control: Point2, end: Point2, segments: int = 5) -> tuple[Point2, ...]:
    if segments < 1:
        raise ValueError("A quadratic path needs at least one segment")
    points: list[Point2] = []
    for index in range(segments + 1):
        t = index / segments
        one_minus_t = 1.0 - t
        points.append(
            (
                one_minus_t * one_minus_t * start[0]
                + 2.0 * one_minus_t * t * control[0]
                + t * t * end[0],
                one_minus_t * one_minus_t * start[1]
                + 2.0 * one_minus_t * t * control[1]
                + t * t * end[1],
            )
        )
    points[0] = start
    points[-1] = end
    return tuple(points)


def _line(start: Point2, end: Point2) -> tuple[Point2, ...]:
    return (start, end)


def _offset_polyline_left(
    points: Sequence[Point2],
    distance_m: float,
) -> tuple[Point2, ...]:
    """Return a mitered constant-distance offset on a path's left side.

    Neck facings are cut by tracing the actual neckline and offsetting that
    curve into the garment.  Translating the curve would incorrectly preserve
    its arclength; a proper parallel curve gains the turning-angle times the
    facing width, which is the pattern length needed to lie around a neck
    without circumferential compression.
    """

    if len(points) < 2 or not math.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("A polyline offset needs two points and a positive distance")
    segment_normals: list[Point2] = []
    for first, second in zip(points[:-1], points[1:]):
        dx, dy = second[0] - first[0], second[1] - first[1]
        length = math.hypot(dx, dy)
        if length <= 1.0e-12:
            raise ValueError("A polyline offset cannot contain a zero-length segment")
        segment_normals.append((-dy / length, dx / length))

    result: list[Point2] = []
    for index, point in enumerate(points):
        if index == 0:
            offset = segment_normals[0]
            scale = distance_m
        elif index == len(points) - 1:
            offset = segment_normals[-1]
            scale = distance_m
        else:
            previous = segment_normals[index - 1]
            following = segment_normals[index]
            summed = (previous[0] + following[0], previous[1] + following[1])
            norm = math.hypot(*summed)
            if norm <= 1.0e-10:
                raise ValueError("A polyline offset has a 180-degree miter")
            offset = (summed[0] / norm, summed[1] / norm)
            cosine = offset[0] * previous[0] + offset[1] * previous[1]
            if cosine <= 1.0e-6:
                raise ValueError("A polyline offset has an unbounded miter")
            scale = distance_m / cosine
        result.append((point[0] + scale * offset[0], point[1] + scale * offset[1]))
    return tuple(result)


def _section(
    path_id: str,
    role: PathRole,
    points: Sequence[Point2],
    *tags: str,
) -> Section:
    return path_id, role, tuple(points), tuple(tags)


def _internal(
    path_id: str,
    role: PathRole,
    points: Sequence[Point2],
    *tags: str,
) -> PatternPath:
    return PatternPath(path_id, tuple(points), role, tuple(tags))


def _panel(
    panel_id: str,
    label: str,
    material: str,
    density: float,
    thickness: float,
    sections: Iterable[Section],
    *,
    internal: Iterable[PatternPath] = (),
    color: tuple[float, float, float],
    tags: tuple[str, ...] = (),
    grain: Point2 = (0.0, 1.0),
    seed_mirror_material_x: bool = False,
    physical_winding_reversed: bool = False,
) -> PanelPattern:
    return PanelPattern(
        id=panel_id,
        label=label,
        material_profile=material,
        areal_density_kg_m2=density,
        unloaded_thickness_m=thickness,
        grain_direction=grain,
        outline_paths=tuple(
            PatternPath(path_id, points, role, semantic_tags)
            for path_id, role, points, semantic_tags in sections
        ),
        internal_paths=tuple(internal),
        display_color_rgb=color,
        semantic_tags=tags,
        seed_mirror_material_x=seed_mirror_material_x,
        physical_winding_reversed=physical_winding_reversed,
    )


def _rectangle_panel(
    panel_id: str,
    label: str,
    width: float,
    height: float,
    material: str,
    density: float,
    thickness: float,
    *,
    roles: tuple[PathRole, PathRole, PathRole, PathRole],
    path_ids: tuple[str, str, str, str] = ("bottom", "right", "top", "left"),
    color: tuple[float, float, float],
    tags: tuple[str, ...] = (),
    internal: Iterable[PatternPath] = (),
    physical_winding_reversed: bool = False,
) -> PanelPattern:
    p0, p1, p2, p3 = (0.0, 0.0), (width, 0.0), (width, height), (0.0, height)
    return _panel(
        panel_id,
        label,
        material,
        density,
        thickness,
        (
            _section(path_ids[0], roles[0], _line(p0, p1)),
            _section(path_ids[1], roles[1], _line(p1, p2)),
            _section(path_ids[2], roles[2], _line(p2, p3)),
            _section(path_ids[3], roles[3], _line(p3, p0)),
        ),
        internal=internal,
        color=color,
        tags=tags,
        physical_winding_reversed=physical_winding_reversed,
    )


def _ref(
    panel: str,
    path: str,
    start: float = 0.0,
    end: float = 1.0,
    *,
    reverse: bool = False,
) -> PathRef:
    return PathRef(panel, path, start, end, reverse)


def _seam(
    seam_id: str,
    side_a: PathRef,
    side_b: PathRef,
    seam_type: SeamType = SeamType.PLAIN,
    *,
    allowance: float = 0.010,
    spacing: float = 0.003,
    stiffness: float = 2.0e4,
    damping: float = 8.0,
    ease: float = 0.03,
) -> SeamSpec:
    return SeamSpec(
        id=seam_id,
        side_a=side_a,
        side_b=side_b,
        seam_type=seam_type,
        seam_allowance_m=allowance,
        stitch_spacing_m=spacing,
        line_stiffness_n_m=stiffness,
        line_damping_n_s_m=damping,
        maximum_ease_fraction=ease,
    )


def _tee_body_panel(panel_id: str, label: str, *, front: bool) -> PanelPattern:
    p0 = (-0.25, 0.0)
    p1 = (0.25, 0.0)
    p2 = (0.25, 0.43)
    p3 = (0.18, 0.65)
    p4 = (0.08, 0.67)
    p5 = (-0.08, 0.67)
    p6 = (-0.18, 0.65)
    p7 = (-0.25, 0.43)
    neck_control = (0.0, 0.555 if front else 0.625)
    return _panel(
        panel_id,
        label,
        "cotton_jersey_prior",
        0.160,
        0.50e-3,
        (
            _section("bottom_hem", PathRole.HEM, _line(p0, p1), "waist"),
            _section("right_side", PathRole.SEAM, _line(p1, p2)),
            _section("right_armhole", PathRole.SEAM, _quadratic(p2, (0.205, 0.555), p3)),
            _section("right_shoulder", PathRole.SEAM, _line(p3, p4)),
            _section("neckline", PathRole.SEAM, _quadratic(p4, neck_control, p5, 7), "neck"),
            _section("left_shoulder", PathRole.SEAM, _line(p5, p6)),
            _section("left_armhole", PathRole.SEAM, _quadratic(p6, (-0.205, 0.555), p7)),
            _section("left_side", PathRole.SEAM, _line(p7, p0)),
        ),
        color=(0.80, 0.22, 0.16),
        tags=("body", "front" if front else "back", "jersey"),
        physical_winding_reversed=not front,
    )


def _tee_sleeve_panel(
    panel_id: str,
    label: str,
    *,
    physical_winding_reversed: bool,
) -> PanelPattern:
    p0, p1 = (-0.17, 0.0), (0.17, 0.0)
    p2, p3, p4 = (0.14, 0.17), (0.0, 0.30), (-0.14, 0.17)
    return _panel(
        panel_id,
        label,
        "cotton_jersey_prior",
        0.160,
        0.50e-3,
        (
            _section("sleeve_hem", PathRole.HEM, _line(p0, p1), "arm_opening"),
            _section("underarm_front", PathRole.SEAM, _line(p1, p2)),
            _section("cap_front", PathRole.SEAM, _quadratic(p2, (0.12, 0.28), p3)),
            _section("cap_back", PathRole.SEAM, _quadratic(p3, (-0.12, 0.28), p4)),
            _section("underarm_back", PathRole.SEAM, _line(p4, p0)),
        ),
        # A flattened sleeve tube has a developable fold from the cap crown to
        # the centre of the open hem.  Constraining the triangulation to this
        # material line permits that fold without forcing triangles to bridge
        # across it; it does not prescribe a runtime crease or stiffness.
        internal=(
            _internal(
                "flat_lay_fold_axis",
                PathRole.FOLD_REFERENCE,
                ((0.0, 0.0), (0.0, 0.30)),
                "reset_landmark",
            ),
        ),
        color=(0.84, 0.25, 0.18),
        tags=("sleeve", "jersey"),
        physical_winding_reversed=physical_winding_reversed,
    )


def build_cotton_tee() -> GarmentConstruction:
    front = _tee_body_panel("front", "Front body", front=True)
    back = _tee_body_panel("back", "Back body", front=False)
    left_sleeve = _tee_sleeve_panel(
        "left_sleeve",
        "Left short sleeve",
        physical_winding_reversed=False,
    )
    right_sleeve = _tee_sleeve_panel(
        "right_sleeve",
        "Right short sleeve",
        physical_winding_reversed=True,
    )
    total_neck = front.path("neckline").length_m + back.path("neckline").length_m
    rib_length = 0.90 * total_neck
    rib = _rectangle_panel(
        "neck_rib",
        "Folded rib-knit neck band",
        rib_length,
        0.026,
        "cotton_rib_prior",
        0.220,
        0.80e-3,
        roles=(PathRole.SEAM, PathRole.SEAM, PathRole.HEM, PathRole.SEAM),
        path_ids=("neck_attach", "closure_right", "neck_opening", "closure_left"),
        color=(0.63, 0.12, 0.09),
        tags=("rib_band", "folded_two_layer"),
        physical_winding_reversed=True,
    )
    front_fraction = front.path("neckline").length_m / total_neck

    seams = [
        _seam("right_side", _ref("front", "right_side"), _ref("back", "right_side"), SeamType.OVERLOCK),
        _seam("left_side", _ref("front", "left_side"), _ref("back", "left_side"), SeamType.OVERLOCK),
        _seam("right_shoulder", _ref("front", "right_shoulder"), _ref("back", "right_shoulder"), SeamType.OVERLOCK),
        _seam("left_shoulder", _ref("front", "left_shoulder"), _ref("back", "left_shoulder"), SeamType.OVERLOCK),
        _seam("left_sleeve_front_cap", _ref("left_sleeve", "cap_front"), _ref("front", "left_armhole", reverse=True), SeamType.OVERLOCK, ease=0.12),
        _seam("left_sleeve_back_cap", _ref("left_sleeve", "cap_back"), _ref("back", "left_armhole"), SeamType.OVERLOCK, ease=0.12),
        _seam("right_sleeve_front_cap", _ref("right_sleeve", "cap_front"), _ref("front", "right_armhole"), SeamType.OVERLOCK, ease=0.12),
        _seam("right_sleeve_back_cap", _ref("right_sleeve", "cap_back"), _ref("back", "right_armhole", reverse=True), SeamType.OVERLOCK, ease=0.12),
        _seam("left_sleeve_underarm", _ref("left_sleeve", "underarm_front"), _ref("left_sleeve", "underarm_back", reverse=True), SeamType.OVERLOCK),
        _seam("right_sleeve_underarm", _ref("right_sleeve", "underarm_front"), _ref("right_sleeve", "underarm_back", reverse=True), SeamType.OVERLOCK),
        # Traverse the closed neckline once: front right-to-left, then back
        # left-to-right.  The rib-band closure joins only the two right-neck
        # endpoints.  Reversing both rib intervals here used to union the
        # left and right shoulder corners into one physical particle.
        _seam("front_neck_rib", _ref("front", "neckline"), _ref("neck_rib", "neck_attach", 0.0, front_fraction), SeamType.COVERSTITCH, allowance=0.007, ease=0.12),
        _seam("back_neck_rib", _ref("back", "neckline", reverse=True), _ref("neck_rib", "neck_attach", front_fraction, 1.0), SeamType.COVERSTITCH, allowance=0.007, ease=0.12),
        _seam("neck_rib_closure", _ref("neck_rib", "closure_right"), _ref("neck_rib", "closure_left", reverse=True), SeamType.OVERLOCK, allowance=0.006),
    ]
    return GarmentConstruction(
        id="cotton_jersey_half_sleeve_tee",
        label="Cotton-jersey half-sleeve T-shirt",
        category="knit_top",
        topology=GarmentTopology.SEWN_PANEL_SHELL,
        panels=(front, back, left_sleeve, right_sleeve, rib),
        seams=tuple(seams),
        openings=(
            OpeningSpec("neck", OpeningKind.NECK, (_ref("neck_rib", "neck_opening"),)),
            OpeningSpec("left_arm", OpeningKind.ARM, (_ref("left_sleeve", "sleeve_hem"),)),
            OpeningSpec("right_arm", OpeningKind.ARM, (_ref("right_sleeve", "sleeve_hem"),)),
            OpeningSpec("waist", OpeningKind.WAIST, (_ref("front", "bottom_hem"), _ref("back", "bottom_hem"))),
        ),
        features=(
            LocalFeatureSpec("folded_neck_rib", FeatureKind.RIB_BAND, ("neck_rib",), width_m=0.026, layer_count=2, material_profile="cotton_rib_prior"),
            LocalFeatureSpec("body_hems", FeatureKind.HEM_BAND, ("front", "back"), (_ref("front", "bottom_hem"), _ref("back", "bottom_hem")), width_m=0.020, layer_count=2),
            LocalFeatureSpec("sleeve_hems", FeatureKind.HEM_BAND, ("left_sleeve", "right_sleeve"), (_ref("left_sleeve", "sleeve_hem"), _ref("right_sleeve", "sleeve_hem")), width_m=0.018, layer_count=2),
        ),
        supported_tasks=("flat_fold", "half_fold", "sleeve_tuck_fold"),
        provenance=_REPOSITORY_PROVENANCE,
        calibration_status="construction-complete engineering prior; no named T-shirt measured",
    )


def _shirt_body_panel(
    panel_id: str,
    label: str,
    *,
    front_half: bool = False,
    physical_winding_reversed: bool = False,
) -> PanelPattern:
    if front_half:
        p0, p1, p2 = (0.0, 0.0), (0.25, 0.0), (0.25, 0.45)
        p3, p4, p5 = (0.18, 0.68), (0.07, 0.70), (0.0, 0.62)
        internal = (
            _internal("pocket_left_mount", PathRole.INTERNAL_STITCH, ((0.075, 0.52), (0.075, 0.38))),
            _internal("pocket_bottom_mount", PathRole.INTERNAL_STITCH, ((0.075, 0.38), (0.195, 0.38))),
            _internal("pocket_right_mount", PathRole.INTERNAL_STITCH, ((0.195, 0.38), (0.195, 0.52))),
            _internal("button_line", PathRole.FOLD_REFERENCE, ((0.025, 0.12), (0.025, 0.57))),
        )
        return _panel(
            panel_id,
            label,
            "cotton_poplin_prior",
            0.120,
            0.30e-3,
            (
                _section("bottom_hem", PathRole.HEM, _line(p0, p1)),
                _section("side", PathRole.SEAM, _line(p1, p2)),
                _section("armhole", PathRole.SEAM, _quadratic(p2, (0.205, 0.57), p3)),
                _section("shoulder", PathRole.SEAM, _line(p3, p4)),
                _section("neckline", PathRole.SEAM, _quadratic(p4, (0.025, 0.665), p5, 4)),
                _section("front_opening", PathRole.OPENING, _line(p5, p0), "placket"),
            ),
            internal=internal,
            color=(0.36, 0.58, 0.82),
            tags=("front_half", "woven"),
            physical_winding_reversed=physical_winding_reversed,
        )

    p0, p1 = (-0.25, 0.0), (0.25, 0.0)
    p2, p3, p4 = (0.25, 0.45), (0.18, 0.68), (0.07, 0.70)
    p5, p6, p7 = (-0.07, 0.70), (-0.18, 0.68), (-0.25, 0.45)
    internal = (
        _internal("yoke_bottom_mount", PathRole.INTERNAL_STITCH, ((-0.18, 0.50), (0.18, 0.50))),
        _internal("yoke_right_mount", PathRole.INTERNAL_STITCH, ((0.18, 0.50), (0.18, 0.65))),
        _internal("yoke_top_mount", PathRole.INTERNAL_STITCH, ((0.18, 0.65), (-0.18, 0.65))),
        _internal("yoke_left_mount", PathRole.INTERNAL_STITCH, ((-0.18, 0.65), (-0.18, 0.50))),
    )
    return _panel(
        panel_id,
        label,
        "cotton_poplin_prior",
        0.120,
        0.30e-3,
        (
            _section("bottom_hem", PathRole.HEM, _line(p0, p1)),
            _section("right_side", PathRole.SEAM, _line(p1, p2)),
            _section("right_armhole", PathRole.SEAM, _quadratic(p2, (0.205, 0.57), p3)),
            _section("right_shoulder", PathRole.SEAM, _line(p3, p4)),
            _section("neckline", PathRole.SEAM, _quadratic(p4, (0.0, 0.66), p5, 5)),
            _section("left_shoulder", PathRole.SEAM, _line(p5, p6)),
            _section("left_armhole", PathRole.SEAM, _quadratic(p6, (-0.205, 0.57), p7)),
            _section("left_side", PathRole.SEAM, _line(p7, p0)),
        ),
        internal=internal,
        color=(0.31, 0.52, 0.77),
        tags=("back", "woven", "doubled_yoke_host"),
        physical_winding_reversed=physical_winding_reversed,
    )


def _shirt_sleeve(
    panel_id: str,
    label: str,
    *,
    physical_winding_reversed: bool,
) -> PanelPattern:
    p0, p1 = (-0.11, 0.0), (0.11, 0.0)
    p2, p3, p4 = (0.17, 0.50), (0.0, 0.68), (-0.17, 0.50)
    return _panel(
        panel_id,
        label,
        "cotton_poplin_prior",
        0.120,
        0.30e-3,
        (
            _section("wrist", PathRole.SEAM, _line(p0, p1)),
            _section("underarm_front", PathRole.SEAM, _line(p1, p2)),
            _section("cap_front", PathRole.SEAM, _quadratic(p2, (0.15, 0.64), p3, 6)),
            _section("cap_back", PathRole.SEAM, _quadratic(p3, (-0.15, 0.64), p4, 6)),
            _section("underarm_back", PathRole.SEAM, _line(p4, p0)),
        ),
        internal=(
            _internal(
                "flat_lay_fold_axis",
                PathRole.FOLD_REFERENCE,
                ((0.0, 0.0), (0.0, 0.68)),
                "reset_landmark",
            ),
        ),
        color=(0.37, 0.60, 0.84),
        tags=("long_sleeve", "woven"),
        physical_winding_reversed=physical_winding_reversed,
    )


def build_woven_shirt() -> GarmentConstruction:
    left_front = _shirt_body_panel("left_front", "Left front", front_half=True)
    right_front = _shirt_body_panel(
        "right_front",
        "Right front",
        front_half=True,
        physical_winding_reversed=True,
    )
    back = _shirt_body_panel("back", "Back body")
    left_sleeve = _shirt_sleeve(
        "left_sleeve",
        "Left long sleeve",
        physical_winding_reversed=True,
    )
    right_sleeve = _shirt_sleeve(
        "right_sleeve",
        "Right long sleeve",
        physical_winding_reversed=False,
    )
    cuff_roles = (PathRole.SEAM, PathRole.SEAM, PathRole.OPENING, PathRole.SEAM)
    left_cuff = _rectangle_panel("left_cuff", "Left cuff", 0.22, 0.06, "cotton_poplin_prior", 0.170, 0.55e-3, roles=cuff_roles, path_ids=("sleeve_attach", "end_right", "wrist_opening", "end_left"), color=(0.28, 0.48, 0.73), tags=("cuff", "interfaced"))
    right_cuff = _rectangle_panel("right_cuff", "Right cuff", 0.22, 0.06, "cotton_poplin_prior", 0.170, 0.55e-3, roles=cuff_roles, path_ids=("sleeve_attach", "end_right", "wrist_opening", "end_left"), color=(0.28, 0.48, 0.73), tags=("cuff", "interfaced"), physical_winding_reversed=True)
    yoke = _rectangle_panel("back_yoke", "Doubled back yoke", 0.36, 0.15, "cotton_poplin_prior", 0.240, 0.60e-3, roles=(PathRole.SEAM,) * 4, path_ids=("bottom", "right", "top", "left"), color=(0.25, 0.45, 0.70), tags=("yoke", "double_layer"))
    pocket = _rectangle_panel("chest_pocket", "Chest patch pocket", 0.12, 0.14, "cotton_poplin_prior", 0.135, 0.42e-3, roles=(PathRole.SEAM, PathRole.SEAM, PathRole.OPENING, PathRole.SEAM), path_ids=("bottom", "right", "opening", "left"), color=(0.33, 0.55, 0.80), tags=("patch_pocket",))

    total_neck = left_front.path("neckline").length_m + right_front.path("neckline").length_m + back.path("neckline").length_m
    stand = _rectangle_panel("collar_stand", "Interfaced collar stand", total_neck, 0.035, "cotton_poplin_prior", 0.210, 0.70e-3, roles=(PathRole.SEAM, PathRole.HEM, PathRole.SEAM, PathRole.HEM), path_ids=("neck_attach", "right_end", "collar_attach", "left_end"), color=(0.24, 0.44, 0.69), tags=("collar_stand", "interfaced"))
    # A real turn-down collar is not a constant-width rectangle wrapped around
    # a curved neck: its outer edge needs additional length or it must stretch
    # and cusp.  The flared cut below keeps the stitched stand edge at the
    # measured neck length while adding 55 mm at each collar point.
    collar_height = 0.075
    collar_flare = 0.055
    c0, c1 = (0.0, 0.0), (total_neck, 0.0)
    c2 = (total_neck + collar_flare, collar_height)
    c3 = (-collar_flare, collar_height)
    collar = _panel(
        "collar",
        "Flared turn-down collar",
        "cotton_poplin_prior",
        0.220,
        0.75e-3,
        (
            _section("stand_attach", PathRole.SEAM, _line(c0, c1)),
            _section("right_end", PathRole.HEM, _line(c1, c2)),
            _section("outer_edge", PathRole.OPENING, _line(c2, c3)),
            _section("left_end", PathRole.HEM, _line(c3, c0)),
        ),
        color=(0.26, 0.46, 0.71),
        tags=("collar", "interfaced", "flared_cut"),
    )
    left_fraction = left_front.path("neckline").length_m / total_neck
    back_fraction = (left_front.path("neckline").length_m + back.path("neckline").length_m) / total_neck

    seams = [
        _seam("left_side", _ref("left_front", "side"), _ref("back", "left_side", reverse=True), SeamType.PLAIN),
        _seam("right_side", _ref("right_front", "side"), _ref("back", "right_side"), SeamType.PLAIN),
        _seam("left_shoulder", _ref("left_front", "shoulder"), _ref("back", "left_shoulder", reverse=True), SeamType.PLAIN),
        _seam("right_shoulder", _ref("right_front", "shoulder"), _ref("back", "right_shoulder"), SeamType.PLAIN),
        _seam("left_cap_front", _ref("left_sleeve", "cap_front"), _ref("left_front", "armhole"), ease=0.10),
        _seam("left_cap_back", _ref("left_sleeve", "cap_back"), _ref("back", "left_armhole"), ease=0.10),
        _seam("right_cap_front", _ref("right_sleeve", "cap_front"), _ref("right_front", "armhole"), ease=0.10),
        _seam("right_cap_back", _ref("right_sleeve", "cap_back"), _ref("back", "right_armhole", reverse=True), ease=0.10),
        _seam("left_underarm", _ref("left_sleeve", "underarm_front"), _ref("left_sleeve", "underarm_back", reverse=True)),
        _seam("right_underarm", _ref("right_sleeve", "underarm_front"), _ref("right_sleeve", "underarm_back", reverse=True)),
        _seam("left_cuff_attach", _ref("left_sleeve", "wrist"), _ref("left_cuff", "sleeve_attach"), SeamType.CUFF),
        _seam("right_cuff_attach", _ref("right_sleeve", "wrist"), _ref("right_cuff", "sleeve_attach"), SeamType.CUFF),
        _seam("left_cuff_closure", _ref("left_cuff", "end_right"), _ref("left_cuff", "end_left", reverse=True), SeamType.PLAIN),
        _seam("right_cuff_closure", _ref("right_cuff", "end_right"), _ref("right_cuff", "end_left", reverse=True), SeamType.PLAIN),
        _seam("left_neck_stand", _ref("left_front", "neckline"), _ref("collar_stand", "neck_attach", 0.0, left_fraction, reverse=True), SeamType.COLLAR),
        _seam("back_neck_stand", _ref("back", "neckline"), _ref("collar_stand", "neck_attach", left_fraction, back_fraction, reverse=True), SeamType.COLLAR),
        _seam("right_neck_stand", _ref("right_front", "neckline"), _ref("collar_stand", "neck_attach", back_fraction, 1.0), SeamType.COLLAR),
        _seam("collar_to_stand", _ref("collar_stand", "collar_attach"), _ref("collar", "stand_attach", reverse=True), SeamType.COLLAR),
        _seam("yoke_bottom", _ref("back_yoke", "bottom"), _ref("back", "yoke_bottom_mount"), SeamType.PLAIN),
        _seam("yoke_right", _ref("back_yoke", "right"), _ref("back", "yoke_right_mount"), SeamType.PLAIN),
        _seam("yoke_top", _ref("back_yoke", "top"), _ref("back", "yoke_top_mount"), SeamType.PLAIN),
        _seam("yoke_left", _ref("back_yoke", "left"), _ref("back", "yoke_left_mount"), SeamType.PLAIN),
        _seam("pocket_bottom", _ref("chest_pocket", "bottom"), _ref("left_front", "pocket_bottom_mount"), SeamType.PATCH),
        _seam("pocket_right", _ref("chest_pocket", "right"), _ref("left_front", "pocket_right_mount"), SeamType.PATCH),
        _seam("pocket_left", _ref("chest_pocket", "left"), _ref("left_front", "pocket_left_mount"), SeamType.PATCH),
    ]
    return GarmentConstruction(
        id="woven_button_down_shirt",
        label="Woven cotton button-down shirt",
        category="woven_shirt",
        topology=GarmentTopology.SEWN_PANEL_SHELL,
        panels=(left_front, right_front, back, left_sleeve, right_sleeve, left_cuff, right_cuff, yoke, pocket, stand, collar),
        seams=tuple(seams),
        openings=(
            OpeningSpec("front_placket", OpeningKind.FRONT, (_ref("left_front", "front_opening"), _ref("right_front", "front_opening")), closable=True),
            OpeningSpec("neck", OpeningKind.NECK, (_ref("collar", "outer_edge"),)),
            OpeningSpec("left_wrist", OpeningKind.WRIST, (_ref("left_cuff", "wrist_opening"),), closable=True),
            OpeningSpec("right_wrist", OpeningKind.WRIST, (_ref("right_cuff", "wrist_opening"),), closable=True),
            OpeningSpec("chest_pocket", OpeningKind.POCKET, (_ref("chest_pocket", "opening"),)),
            OpeningSpec("waist", OpeningKind.WAIST, (_ref("left_front", "bottom_hem"), _ref("right_front", "bottom_hem"), _ref("back", "bottom_hem"))),
        ),
        features=(
            LocalFeatureSpec("left_placket", FeatureKind.PLACKET, ("left_front",), (_ref("left_front", "front_opening"),), width_m=0.030, layer_count=3),
            LocalFeatureSpec("right_placket", FeatureKind.PLACKET, ("right_front",), (_ref("right_front", "front_opening"),), width_m=0.030, layer_count=3),
            LocalFeatureSpec("button_row", FeatureKind.BUTTON, ("right_front",), (_ref("right_front", "button_line"),), layer_count=1),
            LocalFeatureSpec("buttonhole_row", FeatureKind.BUTTONHOLE, ("left_front",), (_ref("left_front", "button_line"),), layer_count=1),
            LocalFeatureSpec("chest_patch_pocket", FeatureKind.POCKET, ("left_front", "chest_pocket"), width_m=0.12, layer_count=2),
            LocalFeatureSpec("doubled_yoke", FeatureKind.INTERLINING, ("back", "back_yoke"), width_m=0.15, layer_count=2),
            LocalFeatureSpec("interfaced_collar", FeatureKind.COLLAR, ("collar", "collar_stand"), width_m=0.075, layer_count=3),
            LocalFeatureSpec("interfaced_cuffs", FeatureKind.CUFF, ("left_cuff", "right_cuff"), width_m=0.06, layer_count=3),
        ),
        supported_tasks=("button_close", "flat_fold", "sleeve_fold", "retail_fold"),
        provenance=_REPOSITORY_PROVENANCE,
        calibration_status="construction-complete engineering prior; no named woven shirt measured",
    )


def _jeans_leg_panel(
    panel_id: str,
    label: str,
    *,
    back: bool,
    mirrored_right_side: bool = False,
) -> PanelPattern:
    if back:
        p0, p1, p2, p3, p4, p5 = (0.0, 0.0), (0.19, 0.0), (0.16, 0.70), (0.28, 0.80), (0.20, 1.02), (0.0, 0.98)
        internal = (
            _internal("patch_left_mount", PathRole.INTERNAL_STITCH, ((0.035, 0.83), (0.035, 0.69))),
            _internal("patch_bottom_mount", PathRole.INTERNAL_STITCH, ((0.035, 0.69), (0.155, 0.69))),
            _internal("patch_right_mount", PathRole.INTERNAL_STITCH, ((0.155, 0.69), (0.155, 0.83))),
            _internal("yoke_bottom_mount", PathRole.INTERNAL_STITCH, ((0.025, 0.85), (0.175, 0.85))),
            _internal("yoke_right_mount", PathRole.INTERNAL_STITCH, ((0.175, 0.85), (0.175, 0.95))),
            _internal("yoke_top_mount", PathRole.INTERNAL_STITCH, ((0.175, 0.95), (0.025, 0.95))),
            _internal("yoke_left_mount", PathRole.INTERNAL_STITCH, ((0.025, 0.95), (0.025, 0.85))),
        )
        color = (0.075, 0.19, 0.40)
    else:
        p0, p1, p2, p3, p4, p5 = (0.0, 0.0), (0.18, 0.0), (0.15, 0.70), (0.235, 0.82), (0.17, 0.98), (0.0, 0.98)
        internal = (
            _internal("pocket_opening", PathRole.OPENING, ((0.025, 0.93), (0.125, 0.85)), "slant_pocket"),
            _internal("coin_left_mount", PathRole.INTERNAL_STITCH, ((0.090, 0.89), (0.090, 0.835))),
            _internal("coin_bottom_mount", PathRole.INTERNAL_STITCH, ((0.090, 0.835), (0.145, 0.835))),
            _internal("coin_right_mount", PathRole.INTERNAL_STITCH, ((0.145, 0.835), (0.145, 0.89))),
        )
        color = (0.09, 0.24, 0.50)
    return _panel(
        panel_id,
        label,
        "denim_twill_prior",
        0.400,
        0.90e-3,
        (
            _section("hem", PathRole.HEM, _line(p0, p1), "leg_opening"),
            _section("inseam", PathRole.SEAM, _line(p1, p2)),
            _section("crotch_curve", PathRole.SEAM, _quadratic(p2, ((p2[0] + p3[0]) * 0.55, 0.73), p3, 4)),
            _section("rise", PathRole.OPENING if not back else PathRole.SEAM, _quadratic(p3, ((p3[0] + p4[0]) * 0.5, 0.91), p4, 4)),
            _section("waist", PathRole.SEAM, _line(p4, p5)),
            _section("outseam", PathRole.SEAM, _line(p5, p0)),
        ),
        internal=internal,
        color=color,
        tags=("back_leg" if back else "front_leg", "denim", "twill"),
        seed_mirror_material_x=mirrored_right_side,
        # The right cut pieces are turned over in the assembly seed.  XORing
        # side and front/back gives one coherent orientation over both leg
        # tubes after their crotch, rise, inseam, and outseam welds.
        physical_winding_reversed=(back != mirrored_right_side),
    )


def _pocket_bag(
    panel_id: str,
    label: str,
    mouth_length: float,
    *,
    physical_winding_reversed: bool,
) -> PanelPattern:
    return _rectangle_panel(
        panel_id,
        label,
        mouth_length,
        0.18,
        "cotton_poplin_prior",
        0.095,
        0.25e-3,
        roles=(PathRole.SEAM, PathRole.SEAM, PathRole.SEAM, PathRole.SEAM),
        path_ids=("bottom", "right", "mouth", "left"),
        color=(0.72, 0.67, 0.53),
        tags=("pocket_bag", "lining"),
        physical_winding_reversed=physical_winding_reversed,
    )


def build_denim_jeans() -> GarmentConstruction:
    front_left = _jeans_leg_panel("front_left", "Left front leg", back=False)
    front_right = _jeans_leg_panel(
        "front_right",
        "Right front leg",
        back=False,
        mirrored_right_side=True,
    )
    back_left = _jeans_leg_panel("back_left", "Left back leg", back=True)
    back_right = _jeans_leg_panel(
        "back_right",
        "Right back leg",
        back=True,
        mirrored_right_side=True,
    )
    panels: list[PanelPattern] = [front_left, front_right, back_left, back_right]
    seams: list[SeamSpec] = []

    for side in ("left", "right"):
        seams.extend(
            (
                _seam(f"{side}_outseam", _ref(f"front_{side}", "outseam"), _ref(f"back_{side}", "outseam"), SeamType.FLAT_FELLED, allowance=0.014, stiffness=3.5e4),
                _seam(f"{side}_inseam", _ref(f"front_{side}", "inseam"), _ref(f"back_{side}", "inseam"), SeamType.FLAT_FELLED, allowance=0.014, stiffness=3.5e4),
            )
        )
    seams.extend(
        (
            _seam("front_crotch_curve", _ref("front_left", "crotch_curve"), _ref("front_right", "crotch_curve"), SeamType.FLAT_FELLED, allowance=0.014),
            _seam("front_lower_rise", _ref("front_left", "rise", 0.0, 0.45), _ref("front_right", "rise", 0.0, 0.45), SeamType.FLAT_FELLED, allowance=0.014),
            _seam("back_crotch_curve", _ref("back_left", "crotch_curve"), _ref("back_right", "crotch_curve"), SeamType.FLAT_FELLED, allowance=0.014),
            _seam("back_rise", _ref("back_left", "rise"), _ref("back_right", "rise"), SeamType.FLAT_FELLED, allowance=0.014),
        )
    )

    waistband_ids: list[str] = []
    for part_id, leg in (("front_left", front_left), ("front_right", front_right), ("back_left", back_left), ("back_right", back_right)):
        width = leg.path("waist").length_m
        waistband_id = f"{part_id}_waistband"
        waistband_ids.append(waistband_id)
        panel = _rectangle_panel(
            waistband_id,
            f"{part_id.replace('_', ' ').title()} waistband",
            width,
            0.042,
            "denim_twill_prior",
            0.650,
            1.60e-3,
            roles=(PathRole.SEAM, PathRole.SEAM, PathRole.OPENING, PathRole.SEAM),
            path_ids=("body_attach", "right_end", "waist_opening", "left_end"),
            color=(0.055, 0.16, 0.35),
            tags=("waistband", "interfaced", "double_layer"),
            physical_winding_reversed=(not leg.physical_winding_reversed),
        )
        panels.append(panel)
        seams.append(_seam(f"{part_id}_waistband_attach", _ref(part_id, "waist"), _ref(waistband_id, "body_attach"), SeamType.WAISTBAND, allowance=0.012, stiffness=4.0e4))
    seams.extend(
        (
            # ``body_attach`` follows each host waist from the inner rise to
            # the outer side seam, so ``right_end`` is the outer end on both
            # front and back pieces.  The former left-to-right weld joined a
            # front fly end to a back side end and displaced the waist by
            # about 230 mm.  Front ``left_end`` edges intentionally remain
            # open at the fly.
            _seam("left_waist_side", _ref("front_left_waistband", "right_end"), _ref("back_left_waistband", "right_end"), SeamType.PLAIN),
            _seam("right_waist_side", _ref("front_right_waistband", "right_end"), _ref("back_right_waistband", "right_end"), SeamType.PLAIN),
            _seam("back_waist_center", _ref("back_left_waistband", "left_end"), _ref("back_right_waistband", "left_end"), SeamType.PLAIN),
        )
    )

    pocket_panel_ids: list[str] = []
    for side in ("left", "right"):
        host = f"front_{side}"
        mouth = front_left.path("pocket_opening").length_m
        outer = _pocket_bag(
            f"{side}_pocket_outer",
            f"{side.title()} pocket outer bag",
            mouth,
            physical_winding_reversed=False,
        )
        inner = _pocket_bag(
            f"{side}_pocket_inner",
            f"{side.title()} pocket inner bag",
            mouth,
            physical_winding_reversed=True,
        )
        panels.extend((outer, inner))
        pocket_panel_ids.extend((outer.id, inner.id))
        seams.extend(
            (
                _seam(f"{side}_pocket_opening_attach", _ref(host, "pocket_opening"), _ref(outer.id, "mouth", reverse=True), SeamType.BOUND),
                _seam(f"{side}_pocket_inner_opening_attach", _ref(host, "pocket_opening"), _ref(inner.id, "mouth", reverse=True), SeamType.PLAIN),
                _seam(f"{side}_pocket_bag_bottom", _ref(outer.id, "bottom"), _ref(inner.id, "bottom"), SeamType.OVERLOCK),
                _seam(f"{side}_pocket_bag_right", _ref(outer.id, "right"), _ref(inner.id, "right"), SeamType.OVERLOCK),
                _seam(f"{side}_pocket_bag_left", _ref(outer.id, "left"), _ref(inner.id, "left"), SeamType.OVERLOCK),
            )
        )

    back_pocket_ids: list[str] = []
    for side in ("left", "right"):
        pocket_id = f"{side}_back_patch_pocket"
        pocket = _rectangle_panel(
            pocket_id,
            f"{side.title()} back patch pocket",
            0.12,
            0.14,
            "denim_twill_prior",
            0.430,
            1.10e-3,
            roles=(PathRole.SEAM, PathRole.SEAM, PathRole.OPENING, PathRole.SEAM),
            path_ids=("bottom", "right", "opening", "left"),
            color=(0.08, 0.21, 0.45),
            tags=("back_patch_pocket", "turned_seam_allowance"),
            internal=(
                _internal(
                    "bottom_turn",
                    PathRole.FOLD_REFERENCE,
                    ((0.005, 0.005), (0.115, 0.005)),
                    "turned_seam_allowance",
                ),
                _internal(
                    "right_turn",
                    PathRole.FOLD_REFERENCE,
                    ((0.115, 0.005), (0.115, 0.135)),
                    "turned_seam_allowance",
                ),
                _internal(
                    "left_turn",
                    PathRole.FOLD_REFERENCE,
                    ((0.005, 0.135), (0.005, 0.005)),
                    "turned_seam_allowance",
                ),
            ),
            physical_winding_reversed=(side == "left"),
        )
        panels.append(pocket)
        back_pocket_ids.append(pocket_id)
        host = f"back_{side}"
        seams.extend(
            (
                _seam(f"{side}_back_pocket_bottom", _ref(pocket_id, "bottom"), _ref(host, "patch_bottom_mount"), SeamType.PATCH),
                _seam(f"{side}_back_pocket_right", _ref(pocket_id, "right"), _ref(host, "patch_right_mount"), SeamType.PATCH),
                _seam(f"{side}_back_pocket_left", _ref(pocket_id, "left"), _ref(host, "patch_left_mount"), SeamType.PATCH),
            )
        )

    coin = _rectangle_panel("coin_pocket", "Coin pocket", 0.055, 0.055, "denim_twill_prior", 0.430, 1.10e-3, roles=(PathRole.SEAM, PathRole.SEAM, PathRole.OPENING, PathRole.SEAM), path_ids=("bottom", "right", "opening", "left"), color=(0.07, 0.20, 0.43), tags=("coin_pocket",), physical_winding_reversed=True)
    panels.append(coin)
    seams.extend(
        (
            _seam("coin_bottom", _ref("coin_pocket", "bottom"), _ref("front_right", "coin_bottom_mount"), SeamType.PATCH),
            _seam("coin_right", _ref("coin_pocket", "right"), _ref("front_right", "coin_right_mount"), SeamType.PATCH),
            _seam("coin_left", _ref("coin_pocket", "left"), _ref("front_right", "coin_left_mount"), SeamType.PATCH),
        )
    )

    yoke_ids: list[str] = []
    for side in ("left", "right"):
        yoke_id = f"back_{side}_yoke"
        yoke = _rectangle_panel(yoke_id, f"Back {side} yoke overlay", 0.15, 0.10, "denim_twill_prior", 0.440, 1.05e-3, roles=(PathRole.SEAM,) * 4, path_ids=("bottom", "right", "top", "left"), color=(0.065, 0.18, 0.39), tags=("back_yoke",), physical_winding_reversed=(side == "left"))
        panels.append(yoke)
        yoke_ids.append(yoke_id)
        host = f"back_{side}"
        for edge in ("bottom", "right", "top", "left"):
            seams.append(_seam(f"{side}_yoke_{edge}", _ref(yoke_id, edge), _ref(host, f"yoke_{edge}_mount"), SeamType.FLAT_FELLED, allowance=0.010))

    rise_upper_fraction = 0.55
    fly_length = front_left.path("rise").length_m * rise_upper_fraction
    fly_shield = _rectangle_panel("fly_shield", "Fly shield", 0.040, fly_length, "denim_twill_prior", 0.720, 1.8e-3, roles=(PathRole.HEM, PathRole.SEAM, PathRole.HEM, PathRole.HEM), path_ids=("bottom", "rise_attach", "top", "free"), color=(0.05, 0.15, 0.33), tags=("fly_shield", "double_layer"), physical_winding_reversed=True)
    fly_facing = _rectangle_panel("fly_facing", "Fly facing", 0.040, fly_length, "denim_twill_prior", 0.650, 1.6e-3, roles=(PathRole.HEM, PathRole.SEAM, PathRole.HEM, PathRole.HEM), path_ids=("bottom", "rise_attach", "top", "free"), color=(0.05, 0.15, 0.33), tags=("fly_facing", "double_layer"))
    panels.extend((fly_shield, fly_facing))
    seams.extend(
        (
            _seam("fly_shield_attach", _ref("front_left", "rise", 0.45, 1.0), _ref("fly_shield", "rise_attach"), SeamType.LAPPED),
            _seam("fly_facing_attach", _ref("front_right", "rise", 0.45, 1.0), _ref("fly_facing", "rise_attach"), SeamType.LAPPED),
        )
    )

    openings = [
        OpeningSpec("fly", OpeningKind.FRONT, (_ref("front_left", "rise", 0.45, 1.0), _ref("front_right", "rise", 0.45, 1.0)), closable=True),
        OpeningSpec("waist", OpeningKind.WAIST, tuple(_ref(panel_id, "waist_opening") for panel_id in waistband_ids), closable=True),
        OpeningSpec("left_leg", OpeningKind.LEG, (_ref("front_left", "hem"), _ref("back_left", "hem"))),
        OpeningSpec("right_leg", OpeningKind.LEG, (_ref("front_right", "hem"), _ref("back_right", "hem"))),
        OpeningSpec("left_front_pocket", OpeningKind.POCKET, (_ref("front_left", "pocket_opening"),)),
        OpeningSpec("right_front_pocket", OpeningKind.POCKET, (_ref("front_right", "pocket_opening"),)),
        OpeningSpec("left_back_pocket", OpeningKind.POCKET, (_ref("left_back_patch_pocket", "opening"),)),
        OpeningSpec("right_back_pocket", OpeningKind.POCKET, (_ref("right_back_patch_pocket", "opening"),)),
        OpeningSpec("coin_pocket", OpeningKind.POCKET, (_ref("coin_pocket", "opening"),)),
    ]
    features = [
        LocalFeatureSpec("four_piece_waistband", FeatureKind.WAISTBAND, tuple(waistband_ids), width_m=0.042, layer_count=3, material_profile="denim_twill_prior"),
        LocalFeatureSpec("front_pocket_bags", FeatureKind.POCKET_BAG, ("front_left", "front_right", *pocket_panel_ids), width_m=0.18, layer_count=2, material_profile="cotton_poplin_prior"),
        LocalFeatureSpec("back_patch_pockets", FeatureKind.POCKET, ("back_left", "back_right", *back_pocket_ids), width_m=0.12, layer_count=2),
        LocalFeatureSpec("coin_pocket_feature", FeatureKind.POCKET, ("front_right", "coin_pocket"), width_m=0.055, layer_count=2),
        LocalFeatureSpec("back_yokes", FeatureKind.INTERLINING, ("back_left", "back_right", *yoke_ids), width_m=0.10, layer_count=2),
        LocalFeatureSpec("zip_fly", FeatureKind.ZIPPER, ("front_left", "front_right", "fly_shield", "fly_facing"), (_ref("front_left", "rise", 0.45, 1.0), _ref("front_right", "rise", 0.45, 1.0)), width_m=0.012, layer_count=4),
        LocalFeatureSpec("waist_button", FeatureKind.BUTTON, ("front_left_waistband", "front_right_waistband"), width_m=0.017, layer_count=1),
        LocalFeatureSpec("belt_loops", FeatureKind.BELT_LOOP, tuple(waistband_ids), width_m=0.010, layer_count=3),
        LocalFeatureSpec("leg_hems", FeatureKind.HEM_BAND, ("front_left", "front_right", "back_left", "back_right"), width_m=0.022, layer_count=3),
    ]
    return GarmentConstruction(
        id="denim_five_pocket_jeans",
        label="Five-pocket denim jeans",
        category="denim_trousers",
        topology=GarmentTopology.SEWN_PANEL_SHELL,
        panels=tuple(panels),
        seams=tuple(seams),
        openings=tuple(openings),
        features=tuple(features),
        supported_tasks=("flat_fold", "leg_align_fold", "retail_fold", "pocket_open"),
        provenance=_REPOSITORY_PROVENANCE,
        calibration_status="construction-complete engineering prior; no named jeans specimen measured",
    )


def build_terry_towel() -> GarmentConstruction:
    width, length = 0.45, 0.70
    base = _rectangle_panel(
        "terry_body",
        "Woven terry body",
        width,
        length,
        "terry_cotton_prior",
        0.450,
        3.0e-3,
        roles=(PathRole.HEM,) * 4,
        path_ids=("short_bottom_hem", "long_right_hem", "short_top_hem", "long_left_hem"),
        color=(0.94, 0.78, 0.24),
        tags=("terry", "loop_pile", "woven_ground"),
        internal=(
            _internal("bottom_dobby", PathRole.INTERNAL_STITCH, ((0.025, 0.075), (0.425, 0.075))),
            _internal("top_dobby", PathRole.INTERNAL_STITCH, ((0.425, 0.625), (0.025, 0.625))),
            _internal("long_fold_axis", PathRole.FOLD_REFERENCE, ((0.225, 0.02), (0.225, 0.68))),
            _internal("short_fold_axis", PathRole.FOLD_REFERENCE, ((0.02, 0.35), (0.43, 0.35))),
        ),
    )
    boundary_refs = tuple(_ref("terry_body", path_id) for path_id in ("short_bottom_hem", "long_right_hem", "short_top_hem", "long_left_hem"))
    return GarmentConstruction(
        id="terry_cotton_bath_towel",
        label="Loop-pile terry cotton bath towel",
        category="household_linen",
        topology=GarmentTopology.RECTANGULAR_SHEET,
        panels=(base,),
        seams=(),
        openings=(OpeningSpec("perimeter", OpeningKind.PERIMETER, boundary_refs),),
        features=(
            LocalFeatureSpec("four_multilayer_hems", FeatureKind.HEM_BAND, ("terry_body",), boundary_refs, width_m=0.025, layer_count=3),
            LocalFeatureSpec("bottom_dobby_band", FeatureKind.DOBBY_BAND, ("terry_body",), (_ref("terry_body", "bottom_dobby"),), width_m=0.035, layer_count=1),
            LocalFeatureSpec("top_dobby_band", FeatureKind.DOBBY_BAND, ("terry_body",), (_ref("terry_body", "top_dobby"),), width_m=0.035, layer_count=1),
        ),
        supported_tasks=("half_fold", "thirds_fold", "hotel_fold", "roll"),
        provenance=_REPOSITORY_PROVENANCE,
        calibration_status="construction-complete engineering prior; terry pile/compression unmeasured",
    )


def _asymmetric_body(panel_id: str, label: str, *, front: bool) -> PanelPattern:
    p0, p1, p2 = (-0.24, 0.0), (0.24, 0.0), (0.24, 0.42)
    p3, p4, p5 = (0.17, 0.62), (0.075, 0.65), (-0.075, 0.65)
    p6, p7 = (-0.17, 0.62), (-0.24, 0.42)
    control = (0.0, 0.49 if front else 0.61)
    internal = ()
    if front:
        internal = (
            _internal("overlay_side_mount", PathRole.INTERNAL_STITCH, ((-0.215, 0.08), (-0.215, 0.40))),
            _internal("overlay_shoulder_mount", PathRole.INTERNAL_STITCH, ((-0.165, 0.605), (-0.075, 0.635))),
            _internal("tie_anchor", PathRole.INTERNAL_STITCH, ((-0.210, 0.25), (-0.185, 0.25))),
        )
    return _panel(
        panel_id,
        label,
        "viscose_crepe_prior",
        0.140,
        0.35e-3,
        (
            _section("bottom_hem", PathRole.HEM, _line(p0, p1)),
            _section("right_side", PathRole.SEAM, _line(p1, p2)),
            _section("right_armhole", PathRole.SEAM, _quadratic(p2, (0.20, 0.53), p3)),
            _section("right_shoulder", PathRole.SEAM, _line(p3, p4)),
            _section("neckline", PathRole.SEAM, _quadratic(p4, control, p5, 6)),
            _section("left_shoulder", PathRole.SEAM, _line(p5, p6)),
            _section("left_armhole", PathRole.SEAM, _quadratic(p6, (-0.20, 0.53), p7)),
            _section("left_side", PathRole.SEAM, _line(p7, p0)),
        ),
        internal=internal,
        color=(0.38, 0.16, 0.54) if front else (0.31, 0.12, 0.46),
        tags=("front_underlayer" if front else "back", "crepe", "draped"),
        physical_winding_reversed=not front,
    )


def _asymmetric_sleeve(
    panel_id: str,
    label: str,
    *,
    long_bell: bool,
    physical_winding_reversed: bool,
) -> PanelPattern:
    if long_bell:
        p0, p1 = (-0.25, 0.0), (0.25, 0.0)
        p2, p3, p4 = (0.14, 0.48), (0.0, 0.62), (-0.14, 0.48)
    else:
        p0, p1 = (-0.16, 0.0), (0.16, 0.0)
        p2, p3, p4 = (0.14, 0.14), (0.0, 0.27), (-0.14, 0.14)
    return _panel(
        panel_id,
        label,
        "viscose_crepe_prior",
        0.140,
        0.35e-3,
        (
            _section("opening_hem", PathRole.HEM, _line(p0, p1)),
            _section("underarm_front", PathRole.SEAM, _line(p1, p2)),
            _section("cap_front", PathRole.SEAM, _quadratic(p2, ((p2[0] + p3[0]) * 0.55, p3[1] - 0.02), p3, 5)),
            _section("cap_back", PathRole.SEAM, _quadratic(p3, ((p3[0] + p4[0]) * 0.55, p3[1] - 0.02), p4, 5)),
            _section("underarm_back", PathRole.SEAM, _line(p4, p0)),
        ),
        internal=(
            _internal(
                "flat_lay_fold_axis",
                PathRole.FOLD_REFERENCE,
                ((0.0, 0.0), p3),
                "reset_landmark",
            ),
        ),
        color=(0.48, 0.20, 0.65) if long_bell else (0.56, 0.25, 0.69),
        tags=("bell_sleeve" if long_bell else "cap_sleeve", "asymmetric"),
        physical_winding_reversed=physical_winding_reversed,
    )


def _asymmetric_neck_facing(
    panel_id: str,
    label: str,
    neckline: PatternPath,
    *,
    physical_winding_reversed: bool,
) -> PanelPattern:
    """Cut one real curved facing piece from its corresponding neckline.

    A front and a back facing are used instead of pretending that a closed
    neck ring is a rectangular strip.  The shoulder joins remain the actual
    22 mm facing width, while the free edge is a parallel curve with the
    longer rest length required outside the neck opening.
    """

    width_m = 0.022
    attach = neckline.points_m
    free = _offset_polyline_left(attach, width_m)
    free_path = tuple(reversed(free))
    return _panel(
        panel_id,
        label,
        "viscose_crepe_prior",
        0.210,
        0.70e-3,
        (
            _section("neck_attach", PathRole.SEAM, attach),
            _section("left_shoulder_join", PathRole.SEAM, _line(attach[-1], free[-1])),
            _section("free_edge", PathRole.OPENING, free_path),
            _section("right_shoulder_join", PathRole.SEAM, _line(free[0], attach[0])),
        ),
        internal=(
            (
                _internal(
                    "left_shoulder_roll",
                    PathRole.FOLD_REFERENCE,
                    _line(attach[-1], free_path[1]),
                    "reset_landmark",
                ),
                _internal(
                    "right_shoulder_roll",
                    PathRole.FOLD_REFERENCE,
                    _line(attach[0], free_path[-2]),
                    "reset_landmark",
                ),
            )
            if panel_id == "back_neck_facing"
            else ()
        ),
        color=(0.28, 0.10, 0.41),
        tags=("neck_facing", "double_layer", "parallel_curve_cut"),
        physical_winding_reversed=physical_winding_reversed,
    )


def build_asymmetric_draped_top() -> GarmentConstruction:
    front = _asymmetric_body("front_under", "Front underlayer", front=True)
    back = _asymmetric_body("back", "Back body", front=False)
    left_sleeve = _asymmetric_sleeve(
        "left_bell_sleeve",
        "Left bell sleeve",
        long_bell=True,
        physical_winding_reversed=False,
    )
    right_sleeve = _asymmetric_sleeve(
        "right_cap_sleeve",
        "Right cap sleeve",
        long_bell=False,
        physical_winding_reversed=True,
    )

    o0, o1, o2, o3, o4, o5 = (-0.215, 0.08), (0.17, 0.02), (0.12, 0.38), (-0.075, 0.635), (-0.165, 0.605), (-0.215, 0.40)
    overlay = _panel(
        "drape_overlay",
        "Asymmetric bias-cut drape overlay",
        "viscose_crepe_prior",
        0.140,
        0.35e-3,
        (
            _section("lower_drape", PathRole.HEM, _quadratic(o0, (0.02, -0.01), o1, 7)),
            _section("diagonal_opening", PathRole.OPENING, _quadratic(o1, (0.11, 0.24), o2, 6), "wrap_edge"),
            _section("neck_drape", PathRole.OPENING, _quadratic(o2, (0.02, 0.52), o3, 7), "cowl_edge"),
            _section("shoulder_attach", PathRole.SEAM, _line(o3, o4)),
            _section("upper_side_attach", PathRole.SEAM, _line(o4, o5)),
            _section("side_attach", PathRole.SEAM, _line(o5, o0)),
        ),
        internal=(_internal("tie_anchor", PathRole.INTERNAL_STITCH, ((-0.210, 0.245), (-0.185, 0.245))),),
        color=(0.58, 0.27, 0.72),
        tags=("bias_cut", "asymmetric_overlay", "drape"),
        grain=(1.0, 1.0),
    )
    left_tie = _rectangle_panel("left_tie", "Left waist tie", 0.025, 0.58, "viscose_crepe_prior", 0.180, 0.55e-3, roles=(PathRole.SEAM, PathRole.HEM, PathRole.HEM, PathRole.HEM), path_ids=("anchor", "long_right", "tip", "long_left"), color=(0.45, 0.18, 0.60), tags=("tie", "folded_strip"))
    right_tie = _rectangle_panel("right_tie", "Right waist tie", 0.025, 0.58, "viscose_crepe_prior", 0.180, 0.55e-3, roles=(PathRole.SEAM, PathRole.HEM, PathRole.HEM, PathRole.HEM), path_ids=("anchor", "long_right", "tip", "long_left"), color=(0.45, 0.18, 0.60), tags=("tie", "folded_strip"))

    front_facing = _asymmetric_neck_facing(
        "front_neck_facing",
        "Shaped front neck facing",
        front.path("neckline"),
        physical_winding_reversed=True,
    )
    back_facing = _asymmetric_neck_facing(
        "back_neck_facing",
        "Shaped back neck facing",
        back.path("neckline"),
        physical_winding_reversed=False,
    )
    seams = (
        _seam("right_side", _ref("front_under", "right_side"), _ref("back", "right_side"), SeamType.PLAIN),
        _seam("left_side", _ref("front_under", "left_side"), _ref("back", "left_side"), SeamType.PLAIN),
        _seam("right_shoulder", _ref("front_under", "right_shoulder"), _ref("back", "right_shoulder"), SeamType.PLAIN),
        _seam("left_shoulder", _ref("front_under", "left_shoulder"), _ref("back", "left_shoulder"), SeamType.PLAIN),
        _seam("left_bell_cap_front", _ref("left_bell_sleeve", "cap_front"), _ref("front_under", "left_armhole", reverse=True), ease=0.10),
        _seam("left_bell_cap_back", _ref("left_bell_sleeve", "cap_back"), _ref("back", "left_armhole"), ease=0.10),
        _seam("right_cap_front", _ref("right_cap_sleeve", "cap_front"), _ref("front_under", "right_armhole"), ease=0.10),
        _seam("right_cap_back", _ref("right_cap_sleeve", "cap_back"), _ref("back", "right_armhole", reverse=True), ease=0.10),
        _seam("left_sleeve_underarm", _ref("left_bell_sleeve", "underarm_front"), _ref("left_bell_sleeve", "underarm_back", reverse=True)),
        _seam("right_sleeve_underarm", _ref("right_cap_sleeve", "underarm_front"), _ref("right_cap_sleeve", "underarm_back", reverse=True)),
        _seam("overlay_shoulder", _ref("drape_overlay", "shoulder_attach"), _ref("front_under", "overlay_shoulder_mount", reverse=True), SeamType.LAPPED, ease=0.10),
        _seam("overlay_side", _ref("drape_overlay", "side_attach"), _ref("front_under", "overlay_side_mount", reverse=True), SeamType.LAPPED, ease=0.10),
        _seam("front_neck_attach", _ref("front_under", "neckline"), _ref("front_neck_facing", "neck_attach"), SeamType.BOUND),
        _seam("back_neck_attach", _ref("back", "neckline"), _ref("back_neck_facing", "neck_attach"), SeamType.BOUND),
        _seam("left_facing_shoulder", _ref("front_neck_facing", "left_shoulder_join"), _ref("back_neck_facing", "left_shoulder_join"), SeamType.PLAIN),
        _seam("right_facing_shoulder", _ref("front_neck_facing", "right_shoulder_join"), _ref("back_neck_facing", "right_shoulder_join"), SeamType.PLAIN),
        _seam("left_tie_attach", _ref("left_tie", "anchor"), _ref("front_under", "tie_anchor"), SeamType.LAPPED),
        _seam("right_tie_attach", _ref("right_tie", "anchor"), _ref("drape_overlay", "tie_anchor"), SeamType.LAPPED),
    )
    return GarmentConstruction(
        id="asymmetric_draped_one_sleeve_top",
        label="Asymmetric draped one-bell-sleeve top",
        category="asymmetric_fashion_top",
        topology=GarmentTopology.SEWN_PANEL_SHELL,
        panels=(front, back, left_sleeve, right_sleeve, overlay, left_tie, right_tie, front_facing, back_facing),
        seams=seams,
        openings=(
            OpeningSpec("neck", OpeningKind.NECK, (_ref("front_neck_facing", "free_edge"), _ref("back_neck_facing", "free_edge"))),
            OpeningSpec("left_arm", OpeningKind.ARM, (_ref("left_bell_sleeve", "opening_hem"),)),
            OpeningSpec("right_arm", OpeningKind.ARM, (_ref("right_cap_sleeve", "opening_hem"),)),
            OpeningSpec("waist", OpeningKind.WAIST, (_ref("front_under", "bottom_hem"), _ref("back", "bottom_hem"))),
            OpeningSpec("drape_front", OpeningKind.FRONT, (_ref("drape_overlay", "diagonal_opening"), _ref("drape_overlay", "neck_drape"))),
        ),
        features=(
            LocalFeatureSpec("bias_drape", FeatureKind.PLACKET, ("front_under", "drape_overlay"), (_ref("drape_overlay", "diagonal_opening"), _ref("drape_overlay", "neck_drape")), width_m=0.22, layer_count=2),
            LocalFeatureSpec("waist_ties", FeatureKind.TIE, ("left_tie", "right_tie"), width_m=0.025, layer_count=2),
            LocalFeatureSpec("neck_facing_feature", FeatureKind.COLLAR, ("front_under", "back", "front_neck_facing", "back_neck_facing"), width_m=0.022, layer_count=2),
            LocalFeatureSpec("asymmetric_sleeves", FeatureKind.CUFF, ("left_bell_sleeve", "right_cap_sleeve"), width_m=0.018, layer_count=2),
        ),
        supported_tasks=("drape_preserving_fold", "tie_untangle", "asymmetric_flat_fold"),
        provenance=_REPOSITORY_PROVENANCE,
        calibration_status="construction-complete engineering prior; no named fashion garment measured",
    )


GARMENT_CONSTRUCTIONS: Mapping[str, GarmentConstruction] = MappingProxyType(
    {
        garment.id: garment
        for garment in (
            build_cotton_tee(),
            build_woven_shirt(),
            build_denim_jeans(),
            build_terry_towel(),
            build_asymmetric_draped_top(),
        )
    }
)


def get_garment_construction(construction_id: str) -> GarmentConstruction:
    try:
        return GARMENT_CONSTRUCTIONS[construction_id]
    except KeyError as exception:
        raise KeyError(f"Unknown garment construction {construction_id!r}") from exception


def construction_summaries() -> tuple[Mapping[str, object], ...]:
    return tuple(garment.construction_summary() for garment in GARMENT_CONSTRUCTIONS.values())
