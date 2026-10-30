"""Construction-aware Isaac scene for the canonical garment catalog.

This is an inspection frontend, not a second cloth simulator.  Every visible
panel is barycentrically attached to the committed P13 welded physical state.
Render-only details (thread, buttons, rivets, zipper teeth, and dobby relief)
are generated from the construction metadata and are explicitly audited as
visual annotations rather than collision geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from .garment_assembly import material_to_physical_vertex_map
from .garment_asset import FeatureKind, GarmentConstruction, PathRef
from .garment_meshing import GarmentPatternMesh, compile_garment_pattern_mesh
from .garment_patterns import GARMENT_CONSTRUCTIONS
from .garment_render_mesh import (
    GarmentRenderSource,
    ThickGarmentPanelTopology,
    build_garment_panel_render_topology,
    deform_thick_garment_panel,
)
from .laundry_scene import (
    _bind,
    _create_camera,
    _cube,
    _cylinder,
    _omnipbr_material,
)


HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
DEFAULT_CATALOG_FRAMES = (
    HERE / "artifacts" / "frames" / "p13_garment_flat_lay_catalog.npz"
)

SHOWCASE_VIEW_MODES = ("front", "back", "construction")

# Front and back are explicitly *layer-isolation* views of the same accepted
# welded state.  They do not invent a second pose or duplicate the cloth.  The
# construction view restores every physical cut panel, including facings,
# pocket bags, fly pieces, and overlays.
_VIEW_PANEL_IDS: Mapping[str, Mapping[str, frozenset[str]]] = {
    "cotton_jersey_half_sleeve_tee": {
        "front": frozenset(("front", "left_sleeve", "right_sleeve", "neck_rib")),
        "back": frozenset(("back", "left_sleeve", "right_sleeve", "neck_rib")),
    },
    "woven_button_down_shirt": {
        "front": frozenset(
            (
                "left_front",
                "right_front",
                "left_sleeve",
                "right_sleeve",
                "left_cuff",
                "right_cuff",
                "chest_pocket",
                "collar_stand",
                "collar",
            )
        ),
        "back": frozenset(
            (
                "back",
                "left_sleeve",
                "right_sleeve",
                "left_cuff",
                "right_cuff",
                "back_yoke",
                "collar_stand",
                "collar",
            )
        ),
    },
    "denim_five_pocket_jeans": {
        "front": frozenset(
            (
                "front_left",
                "front_right",
                "front_left_waistband",
                "front_right_waistband",
                "coin_pocket",
            )
        ),
        "back": frozenset(
            (
                "back_left",
                "back_right",
                "back_left_waistband",
                "back_right_waistband",
                "left_back_patch_pocket",
                "right_back_patch_pocket",
                "back_left_yoke",
                "back_right_yoke",
            )
        ),
    },
    "terry_cotton_bath_towel": {
        "front": frozenset(("terry_body",)),
        "back": frozenset(("terry_body",)),
    },
    "asymmetric_draped_one_sleeve_top": {
        "front": frozenset(
            (
                "front_under",
                "left_bell_sleeve",
                "right_cap_sleeve",
                "drape_overlay",
                "left_tie",
                "right_tie",
            )
        ),
        "back": frozenset(
            (
                "back",
                "left_bell_sleeve",
                "right_cap_sleeve",
                "left_tie",
                "right_tie",
            )
        ),
    },
}


_TEXTILE_TEXTURE_STEMS: Mapping[str, str] = {
    "cotton_jersey_prior": "cotton_jersey",
    "cotton_rib_prior": "cotton_jersey",
    "cotton_poplin_prior": "cotton_poplin",
    "denim_twill_prior": "denim_twill",
    "terry_cotton_prior": "terry_cotton",
    "viscose_crepe_prior": "viscose_crepe",
}

_TEXTILE_ROUGHNESS: Mapping[str, float] = {
    "cotton_jersey_prior": 0.82,
    "cotton_rib_prior": 0.84,
    "cotton_poplin_prior": 0.72,
    "denim_twill_prior": 0.79,
    "terry_cotton_prior": 0.91,
    "viscose_crepe_prior": 0.67,
}


@dataclass
class GarmentPanelVisualHandle:
    panel_id: str
    mesh: UsdGeom.Mesh
    topology: ThickGarmentPanelTopology
    declared_thickness_m: float


@dataclass
class GarmentShowcaseItem:
    construction: GarmentConstruction
    pattern_mesh: GarmentPatternMesh
    render_source: GarmentRenderSource
    root_path: str
    panel_visuals: dict[str, GarmentPanelVisualHandle]
    physical_vertex_count: int
    position_offset_m: np.ndarray
    source_positions_m: np.ndarray
    source_triangle_count: int
    visual_detail_paths: tuple[str, ...]
    visual_detail_views: dict[str, frozenset[str]]
    cameras: dict[str, str]
    bounds_m: np.ndarray


@dataclass
class GarmentShowcaseHandles:
    garments: dict[str, GarmentShowcaseItem]
    materials: dict[str, UsdShade.Material]
    catalog_frame_path: Path
    catalog_frame_sha256: str
    selected_construction_id: str
    selected_view_mode: str


def _safe_path_component(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _textile_material(
    stage: Usd.Stage,
    path: str,
    panel,
) -> UsdShade.Material:
    try:
        stem = _TEXTILE_TEXTURE_STEMS[panel.material_profile]
    except KeyError as exception:
        raise KeyError(
            f"No visual textile family for {panel.material_profile!r}"
        ) from exception
    material = _omnipbr_material(
        stage,
        path,
        tuple(map(float, panel.display_color_rgb)),
        roughness=_TEXTILE_ROUGHNESS[panel.material_profile],
        # Albedo remains the authored panel colour.  UV-driven normal and
        # roughness maps provide fibre structure without recolouring beige
        # pocket bags blue or flattening panel-level construction contrast.
        normal_texture=ASSETS / f"{stem}_normal.png",
        roughness_texture=ASSETS / f"{stem}_roughness.png",
        project_uvw=False,
    )
    shader = UsdShade.Shader.Get(stage, path + "/Shader")
    shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(False)
    return material


def _mesh_from_panel(
    stage: Usd.Stage,
    path: str,
    topology: ThickGarmentPanelTopology,
    positions_m: np.ndarray,
    *,
    thickness_m: float,
    material: UsdShade.Material,
) -> UsdGeom.Mesh:
    points, normals = deform_thick_garment_panel(
        topology, positions_m, thickness_m=thickness_m
    )
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in points])
    mesh.CreateFaceVertexCountsAttr(
        [int(value) for value in topology.face_vertex_counts]
    )
    mesh.CreateFaceVertexIndicesAttr(
        [int(value) for value in topology.face_vertex_indices]
    )
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateNormalsAttr([Gf.Vec3f(*map(float, normal)) for normal in normals])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set([Gf.Vec2f(*map(float, uv)) for uv in topology.uv])
    mesh.CreateExtentAttr(
        [
            Gf.Vec3f(*map(float, np.min(points, axis=0))),
            Gf.Vec3f(*map(float, np.max(points, axis=0))),
        ]
    )
    _bind(mesh, material)
    return mesh


def _path_positions(
    pattern_mesh: GarmentPatternMesh,
    source: GarmentRenderSource,
    positions_m: np.ndarray,
    path_ref: PathRef,
) -> np.ndarray:
    material_vertices = pattern_mesh.path_vertices[
        (path_ref.panel_id, path_ref.path_id)
    ]
    fractions = pattern_mesh.path_fractions[
        (path_ref.panel_id, path_ref.path_id)
    ]
    selected = (
        (fractions >= path_ref.start_fraction - 1.0e-12)
        & (fractions <= path_ref.end_fraction + 1.0e-12)
    )
    material_vertices = material_vertices[selected]
    if path_ref.reverse:
        material_vertices = material_vertices[::-1]
    physical_vertices = source.material_to_physical_vertex[material_vertices]
    result = positions_m[physical_vertices]
    if len(result) < 2:
        raise ValueError(
            f"Construction path {path_ref.panel_id}:{path_ref.path_id} "
            "has fewer than two render samples"
        )
    # Consecutive seam landmarks can weld to the same physical particle at a
    # cut corner.  Curves do not need repeated points.
    keep = np.ones(len(result), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(result, axis=0), axis=1) > 1.0e-10
    result = result[keep]
    if len(result) < 2:
        raise ValueError("A construction path collapsed to one render point")
    return result


def _curve(
    stage: Usd.Stage,
    path: str,
    points_m: np.ndarray,
    *,
    width_m: float,
    lift_m: float,
    material: UsdShade.Material,
) -> str:
    points = np.asarray(points_m, dtype=np.float64).copy()
    points[:, 2] += float(lift_m)
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in points])
    curve.CreateWidthsAttr([float(width_m)])
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    _bind(curve, material)
    return path


def _ribbon(
    stage: Usd.Stage,
    path: str,
    centreline_m: np.ndarray,
    *,
    width_m: float,
    lift_m: float,
    material: UsdShade.Material,
) -> str:
    centreline = np.asarray(centreline_m, dtype=np.float64).copy()
    if len(centreline) < 2:
        raise ValueError("A construction ribbon needs at least two points")
    centreline[:, 2] += float(lift_m)
    tangents = np.empty_like(centreline)
    tangents[0] = centreline[1] - centreline[0]
    tangents[-1] = centreline[-1] - centreline[-2]
    if len(centreline) > 2:
        tangents[1:-1] = centreline[2:] - centreline[:-2]
    lateral = np.column_stack((-tangents[:, 1], tangents[:, 0], np.zeros(len(tangents))))
    lengths = np.linalg.norm(lateral, axis=1)
    fallback = lengths <= 1.0e-10
    lateral[~fallback] /= lengths[~fallback, None]
    lateral[fallback] = (1.0, 0.0, 0.0)
    half = 0.5 * float(width_m)
    points = np.empty((2 * len(centreline), 3), dtype=np.float64)
    points[0::2] = centreline - half * lateral
    points[1::2] = centreline + half * lateral
    faces = [
        (2 * index, 2 * index + 1, 2 * index + 3, 2 * index + 2)
        for index in range(len(centreline) - 1)
    ]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in points])
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(
        [int(value) for face in faces for value in face]
    )
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    _bind(mesh, material)
    return path


def _sample_polyline(points_m: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64)
    if count < 1 or len(points) < 2:
        raise ValueError("Polyline sampling needs a line and a positive count")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 1.0e-12:
        raise ValueError("Cannot sample a zero-length construction path")
    targets = np.linspace(0.08, 0.92, count) * cumulative[-1]
    result = []
    for target in targets:
        segment = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(lengths) - 1)
        fraction = (target - cumulative[segment]) / max(lengths[segment], 1.0e-12)
        result.append((1.0 - fraction) * points[segment] + fraction * points[segment + 1])
    return np.asarray(result)


def _button(
    stage: Usd.Stage,
    path: str,
    position_m: np.ndarray,
    *,
    radius_m: float,
    material: UsdShade.Material,
    hole_material: UsdShade.Material,
) -> tuple[str, ...]:
    position = np.asarray(position_m, dtype=np.float64).copy()
    position[2] += 1.8e-3
    _cylinder(
        stage,
        path,
        radius_m=radius_m,
        height_m=2.2e-3,
        axis="Z",
        position_m=tuple(map(float, position)),
        material=material,
    )
    paths = [path]
    hole_radius = 0.12 * radius_m
    hole_offset = 0.31 * radius_m
    for index, (dx, dy) in enumerate(
        ((-hole_offset, -hole_offset), (-hole_offset, hole_offset), (hole_offset, -hole_offset), (hole_offset, hole_offset))
    ):
        hole_path = path + f"/Hole{index}"
        hole_position = position + (dx, dy, 1.3e-3)
        _cylinder(
            stage,
            hole_path,
            radius_m=hole_radius,
            height_m=0.45e-3,
            axis="Z",
            position_m=tuple(map(float, hole_position)),
            material=hole_material,
        )
        paths.append(hole_path)
    return tuple(paths)


def _construction_details(
    stage: Usd.Stage,
    root_path: str,
    construction: GarmentConstruction,
    pattern_mesh: GarmentPatternMesh,
    source: GarmentRenderSource,
    positions_m: np.ndarray,
    materials: Mapping[str, UsdShade.Material],
) -> tuple[str, ...]:
    detail_root = root_path + "/ConstructionDetails"
    UsdGeom.Scope.Define(stage, detail_root)
    paths: list[str] = []
    is_denim = construction.id == "denim_five_pocket_jeans"
    thread = materials["denim_thread" if is_denim else "thread"]

    # One side of each declared stitch line is enough; drawing both would
    # duplicate exactly welded curves and falsely exaggerate seam width.
    excluded_panel_fragments = ("pocket_inner", "pocket_outer", "fly_shield", "fly_facing")
    for index, seam in enumerate(construction.seams):
        if any(fragment in seam.side_a.panel_id for fragment in excluded_panel_fragments):
            continue
        points = _path_positions(pattern_mesh, source, positions_m, seam.side_a)
        paths.append(
            _curve(
                stage,
                f"{detail_root}/Seam_{index:02}_{_safe_path_component(seam.id)}",
                points,
                width_m=1.35e-3 if is_denim else 0.85e-3,
                lift_m=1.4e-3 if is_denim else 0.9e-3,
                material=thread,
            )
        )

    # Explicit feature paths carry the construction semantics that generic
    # seam curves cannot: plackets, hems, dobby bands, and draped openings.
    for feature_index, feature in enumerate(construction.features):
        for ref_index, path_ref in enumerate(feature.path_refs):
            points = _path_positions(pattern_mesh, source, positions_m, path_ref)
            path = (
                f"{detail_root}/Feature_{feature_index:02}_{ref_index:02}_"
                f"{_safe_path_component(feature.id)}"
            )
            if feature.kind == FeatureKind.DOBBY_BAND:
                paths.append(
                    _ribbon(
                        stage,
                        path,
                        points,
                        width_m=float(feature.width_m or 0.035),
                        lift_m=1.7e-3,
                        material=materials["dobby"],
                    )
                )
            else:
                paths.append(
                    _curve(
                        stage,
                        path,
                        points,
                        width_m=min(float(feature.width_m or 0.002), 3.2e-3),
                        lift_m=1.15e-3,
                        material=thread,
                    )
                )

    if construction.id == "woven_button_down_shirt":
        button_line = _path_positions(
            pattern_mesh,
            source,
            positions_m,
            PathRef("right_front", "button_line"),
        )
        for index, point in enumerate(_sample_polyline(button_line, 7)):
            paths.extend(
                _button(
                    stage,
                    f"{detail_root}/ShirtButton{index}",
                    point,
                    radius_m=5.2e-3,
                    material=materials["shirt_button"],
                    hole_material=materials["button_hole"],
                )
            )
        buttonhole_line = _path_positions(
            pattern_mesh,
            source,
            positions_m,
            PathRef("left_front", "button_line"),
        )
        for index, point in enumerate(_sample_polyline(buttonhole_line, 7)):
            slit = np.asarray((point + (-0.006, 0.0, 0.0), point + (0.006, 0.0, 0.0)))
            paths.append(
                _curve(
                    stage,
                    f"{detail_root}/Buttonhole{index}",
                    slit,
                    width_m=1.15e-3,
                    lift_m=1.5e-3,
                    material=materials["button_hole"],
                )
            )

    if is_denim:
        waist = _path_positions(
            pattern_mesh,
            source,
            positions_m,
            PathRef("front_left_waistband", "waist_opening"),
        )
        paths.extend(
            _button(
                stage,
                f"{detail_root}/WaistButton",
                waist[-1],
                radius_m=7.0e-3,
                material=materials["brass"],
                hole_material=materials["dark_metal"],
            )
        )
        for side in ("left", "right"):
            pocket_mouth = _path_positions(
                pattern_mesh,
                source,
                positions_m,
                PathRef(f"front_{side}", "pocket_opening"),
            )
            for endpoint_index, point in enumerate((pocket_mouth[0], pocket_mouth[-1])):
                rivet_path = f"{detail_root}/{side.title()}Rivet{endpoint_index}"
                rivet_position = point.copy()
                rivet_position[2] += 1.9e-3
                _cylinder(
                    stage,
                    rivet_path,
                    radius_m=3.2e-3,
                    height_m=1.8e-3,
                    axis="Z",
                    position_m=tuple(map(float, rivet_position)),
                    material=materials["brass"],
                )
                paths.append(rivet_path)
        rise = _path_positions(
            pattern_mesh,
            source,
            positions_m,
            PathRef("front_left", "rise", 0.45, 1.0),
        )
        for index, point in enumerate(_sample_polyline(rise, 12)):
            tooth = f"{detail_root}/ZipperTooth{index:02}"
            _cube(
                stage,
                tooth,
                (3.0e-3, 5.0e-3, 1.4e-3),
                tuple(map(float, point + (0.0, 0.0, 2.0e-3))),
                materials["brass"],
            )
            paths.append(tooth)

    return tuple(paths)


_FEATURE_VIEW_OVERRIDES: Mapping[str, Mapping[str, frozenset[str]]] = {
    "cotton_jersey_half_sleeve_tee": {
        "folded_neck_rib": frozenset(("front", "back", "construction")),
    },
    "woven_button_down_shirt": {
        "chest_patch_pocket": frozenset(("front", "construction")),
        "doubled_yoke": frozenset(("back", "construction")),
        "interfaced_collar": frozenset(("front", "back", "construction")),
        "interfaced_cuffs": frozenset(("front", "back", "construction")),
    },
    "denim_five_pocket_jeans": {
        "four_piece_waistband": frozenset(("front", "back", "construction")),
        "front_pocket_bags": frozenset(("front", "construction")),
        "back_patch_pockets": frozenset(("back", "construction")),
        "coin_pocket_feature": frozenset(("front", "construction")),
        "back_yokes": frozenset(("back", "construction")),
        "waist_button": frozenset(("front", "construction")),
        "belt_loops": frozenset(("front", "back", "construction")),
        "leg_hems": frozenset(("front", "back", "construction")),
    },
    "asymmetric_draped_one_sleeve_top": {
        "waist_ties": frozenset(("front", "back", "construction")),
        "neck_facing_feature": frozenset(("construction",)),
        "asymmetric_sleeves": frozenset(("front", "back", "construction")),
    },
}


def _panel_view_modes(
    construction: GarmentConstruction,
    panel_id: str,
) -> frozenset[str]:
    modes = {"construction"}
    for mode in ("front", "back"):
        if panel_id in _VIEW_PANEL_IDS[construction.id][mode]:
            modes.add(mode)
    return frozenset(modes)


def _detail_view_modes(
    construction: GarmentConstruction,
    path: str,
) -> frozenset[str]:
    """Assign a render annotation to the layers that physically own it."""

    if "/Seam_" in path:
        suffix = path.split("/Seam_", 1)[1]
        seam = construction.seams[int(suffix[:2])]
        return frozenset(
            set(_panel_view_modes(construction, seam.side_a.panel_id))
            | set(_panel_view_modes(construction, seam.side_b.panel_id))
        )
    if "/Feature_" in path:
        suffix = path.split("/Feature_", 1)[1]
        feature = construction.features[int(suffix[:2])]
        override = _FEATURE_VIEW_OVERRIDES.get(construction.id, {}).get(feature.id)
        if override is not None:
            return override
        if feature.path_refs:
            modes = {"construction"}
            for path_ref in feature.path_refs:
                modes.update(_panel_view_modes(construction, path_ref.panel_id))
            return frozenset(modes)
        return frozenset(SHOWCASE_VIEW_MODES)
    if any(
        marker in path
        for marker in (
            "/ShirtButton",
            "/Buttonhole",
            "/WaistButton",
            "Rivet",
            "/ZipperTooth",
        )
    ):
        return frozenset(("front", "construction"))
    return frozenset(SHOWCASE_VIEW_MODES)


def _stage_materials(stage: Usd.Stage) -> dict[str, UsdShade.Material]:
    materials = {
        "table": _omnipbr_material(
            stage,
            "/World/Looks/Table",
            (0.70, 0.72, 0.70),
            roughness=0.58,
            albedo_texture=ASSETS / "table_laminate.png",
            project_uvw=True,
            texture_scale=(2.8, 2.8),
        ),
        "floor": _omnipbr_material(
            stage,
            "/World/Looks/Floor",
            (0.38, 0.39, 0.40),
            roughness=0.86,
            albedo_texture=ASSETS / "concrete_floor.png",
            project_uvw=True,
            texture_scale=(0.55, 0.55),
        ),
        "metal": _omnipbr_material(
            stage, "/World/Looks/Metal", (0.22, 0.24, 0.26), roughness=0.28, metallic=0.88
        ),
        "dark_metal": _omnipbr_material(
            stage, "/World/Looks/DarkMetal", (0.06, 0.065, 0.07), roughness=0.33, metallic=0.75
        ),
        "thread": _omnipbr_material(
            stage, "/World/Looks/Thread", (0.88, 0.90, 0.91), roughness=0.75
        ),
        "denim_thread": _omnipbr_material(
            stage, "/World/Looks/DenimThread", (0.93, 0.55, 0.16), roughness=0.70
        ),
        "shirt_button": _omnipbr_material(
            stage, "/World/Looks/ShirtButton", (0.90, 0.93, 0.92), roughness=0.23
        ),
        "button_hole": _omnipbr_material(
            stage, "/World/Looks/ButtonHole", (0.055, 0.065, 0.075), roughness=0.74
        ),
        "brass": _omnipbr_material(
            stage, "/World/Looks/Brass", (0.62, 0.38, 0.12), roughness=0.25, metallic=0.90
        ),
        "dobby": _omnipbr_material(
            stage, "/World/Looks/Dobby", (0.29, 0.43, 0.52), roughness=0.81
        ),
        "spool_red": _omnipbr_material(
            stage, "/World/Looks/SpoolRed", (0.60, 0.07, 0.055), roughness=0.66
        ),
        "spool_gold": _omnipbr_material(
            stage, "/World/Looks/SpoolGold", (0.80, 0.42, 0.08), roughness=0.70
        ),
    }
    return materials


def _laboratory(stage: Usd.Stage, materials: Mapping[str, UsdShade.Material]) -> None:
    UsdGeom.Xform.Define(stage, "/World/Lab")
    _cube(stage, "/World/Lab/Floor", (5.2, 5.2, 0.08), (0.0, 0.0, -0.79), materials["floor"])
    _cube(stage, "/World/Lab/TableTop", (1.72, 1.46, 0.055), (0.0, 0.0, -0.035), materials["table"])
    for x_index, x in enumerate((-0.77, 0.77)):
        for y_index, y in enumerate((-0.64, 0.64)):
            _cube(
                stage,
                f"/World/Lab/Leg_{x_index}_{y_index}",
                (0.055, 0.055, 0.72),
                (x, y, -0.405),
                materials["metal"],
            )
    # A steel edge rail and tick blocks give the garment scale without placing
    # a fake metric overlay on the cloth itself.
    _cube(stage, "/World/Lab/ScaleRail", (1.48, 0.018, 0.012), (0.0, -0.704, 0.001), materials["metal"])
    for index, x in enumerate(np.linspace(-0.70, 0.70, 29)):
        height = 0.020 if index % 5 == 0 else 0.011
        _cube(
            stage,
            f"/World/Lab/Tick{index:02}",
            (0.0018, height, 0.0022),
            (float(x), -0.690 + 0.5 * height, 0.009),
            materials["dark_metal"],
        )
    for index, (x, material) in enumerate(
        ((-0.72, materials["spool_red"]), (0.72, materials["spool_gold"]))
    ):
        _cylinder(
            stage,
            f"/World/Lab/ThreadSpool{index}",
            radius_m=0.035,
            height_m=0.085,
            axis="Z",
            position_m=(x, 0.62, 0.070),
            material=material,
        )
        _cylinder(
            stage,
            f"/World/Lab/ThreadSpool{index}/Core",
            radius_m=0.013,
            height_m=0.092,
            axis="Z",
            position_m=(x, 0.62, 0.070),
            material=materials["dark_metal"],
        )


def _garment_cameras(
    stage: Usd.Stage,
    construction_id: str,
    bounds_m: np.ndarray,
) -> dict[str, str]:
    size = bounds_m[1] - bounds_m[0]
    focal_length = 48.0
    horizontal_aperture = 24.0
    aspect = 1.5
    required_height = 1.17 * max(
        0.5 * size[0] * focal_length / (0.5 * horizontal_aperture),
        0.5 * size[1] * focal_length / (0.5 * horizontal_aperture / aspect),
    )
    height = max(float(required_height), 1.50)
    component = _safe_path_component(construction_id)
    root = f"/World/Cameras/{component}"
    front = _create_camera(
        stage,
        root + "/FrontLayer",
        (0.0, 0.0, height),
        (0.0, 0.0, 0.0),
        focal_length,
    )
    back = _create_camera(
        stage,
        root + "/BackLayer",
        (0.0, 0.0, height),
        (0.0, 0.0, 0.0),
        focal_length,
    )
    oblique_distance = max(1.25, 1.15 * float(size[1]))
    construction = _create_camera(
        stage,
        root + "/Construction",
        (0.78, -oblique_distance, max(0.92, 0.92 * oblique_distance)),
        (0.0, 0.03, 0.015),
        50.0,
    )
    return {"front": front, "back": back, "construction": construction}


def set_selected_garment(
    handles: GarmentShowcaseHandles,
    construction_id: str,
) -> GarmentShowcaseItem:
    if construction_id not in handles.garments:
        raise KeyError(f"Unknown showcase garment {construction_id!r}")
    for garment_id, item in handles.garments.items():
        imageable = UsdGeom.Imageable.Get(
            item.panel_visuals[next(iter(item.panel_visuals))].mesh.GetPrim().GetStage(),
            item.root_path,
        )
        if garment_id == construction_id:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
    handles.selected_construction_id = construction_id
    return handles.garments[construction_id]


def set_garment_view(
    handles: GarmentShowcaseHandles,
    construction_id: str,
    view_mode: str,
) -> GarmentShowcaseItem:
    """Select one garment and expose an honest subset of its physical panels."""

    if view_mode not in SHOWCASE_VIEW_MODES:
        raise KeyError(f"Unknown showcase view {view_mode!r}")
    item = set_selected_garment(handles, construction_id)
    if view_mode == "construction":
        visible_panels = frozenset(item.panel_visuals)
    else:
        visible_panels = _VIEW_PANEL_IDS[construction_id][view_mode]
    stage = next(iter(item.panel_visuals.values())).mesh.GetPrim().GetStage()
    for panel_id, panel in item.panel_visuals.items():
        imageable = UsdGeom.Imageable.Get(stage, panel.mesh.GetPath())
        if panel_id in visible_panels:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
    for path, modes in item.visual_detail_views.items():
        imageable = UsdGeom.Imageable.Get(stage, path)
        if view_mode in modes:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
    handles.selected_view_mode = view_mode
    return item


def build_garment_showcase_scene(
    stage: Usd.Stage,
    *,
    catalog_frame_path: Path = DEFAULT_CATALOG_FRAMES,
    target_edge_length_m: float = 0.040,
    subdivision_factor: int = 2,
    selected_construction_id: str = "cotton_jersey_half_sleeve_tee",
    progress: Callable[[str], None] | None = None,
) -> GarmentShowcaseHandles:
    """Author all five real garment constructions and select one for display."""

    def emit(message: str) -> None:
        if progress is not None:
            progress(message)

    if not catalog_frame_path.is_file():
        raise FileNotFoundError(f"Missing garment catalog frames: {catalog_frame_path}")
    emit("authoring garment-lab stage")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdGeom.Scope.Define(stage, "/World/Cameras")
    UsdGeom.Scope.Define(stage, "/World/Garments")
    materials = _stage_materials(stage)
    _laboratory(stage, materials)
    emit("laboratory and shared materials authored")

    garments: dict[str, GarmentShowcaseItem] = {}
    with np.load(catalog_frame_path) as frames:
        for construction in GARMENT_CONSTRUCTIONS.values():
            construction_id = construction.id
            emit(f"compiling {construction_id}")
            prefix = construction_id
            positions = np.asarray(
                frames[f"{prefix}_positions_m"], dtype=np.float64
            ).copy()
            artifact_triangles = np.asarray(
                frames[f"{prefix}_physical_triangles"], dtype=np.int32
            )
            pattern_mesh = compile_garment_pattern_mesh(
                construction, target_edge_length_m=target_edge_length_m
            )
            mapping = material_to_physical_vertex_map(pattern_mesh)
            physical_triangles = mapping[pattern_mesh.triangles]
            # Style3D reverses the corner order of explicitly authored back-
            # facing cut pieces before ingestion.  Triangle identity and
            # ordering must match; corner winding is allowed to differ.
            compiled_unoriented = np.sort(physical_triangles, axis=1)
            artifact_unoriented = np.sort(artifact_triangles, axis=1)
            compiled_unoriented = compiled_unoriented[
                np.lexsort(
                    (
                        compiled_unoriented[:, 2],
                        compiled_unoriented[:, 1],
                        compiled_unoriented[:, 0],
                    )
                )
            ]
            artifact_unoriented = artifact_unoriented[
                np.lexsort(
                    (
                        artifact_unoriented[:, 2],
                        artifact_unoriented[:, 1],
                        artifact_unoriented[:, 0],
                    )
                )
            ]
            if not np.array_equal(compiled_unoriented, artifact_unoriented):
                raise ValueError(
                    f"P13 artifact topology for {construction_id!r} no longer "
                    "matches the compiled construction"
                )
            expected_physical_count = int(np.max(mapping)) + 1
            if positions.shape != (expected_physical_count, 3):
                raise ValueError(
                    f"P13 artifact for {construction_id!r} has shape {positions.shape}; "
                    f"expected {(expected_physical_count, 3)}"
                )
            source = GarmentRenderSource(construction_id, mapping)
            raw_bounds = np.asarray((np.min(positions, axis=0), np.max(positions, axis=0)))
            xy_centre = 0.5 * (raw_bounds[0, :2] + raw_bounds[1, :2])
            maximum_half_thickness = 0.5 * max(
                panel.unloaded_thickness_m for panel in construction.panels
            )
            offset = np.asarray(
                (-xy_centre[0], -xy_centre[1], 0.004 + maximum_half_thickness - raw_bounds[0, 2]),
                dtype=np.float64,
            )
            display_positions = positions + offset
            bounds = np.asarray(
                (np.min(display_positions, axis=0), np.max(display_positions, axis=0))
            )
            root_path = f"/World/Garments/{_safe_path_component(construction_id)}"
            root = UsdGeom.Xform.Define(stage, root_path)
            root.GetPrim().SetCustomDataByKey("construction_id", construction_id)
            root.GetPrim().SetCustomDataByKey("source_probe", "P13")
            root.GetPrim().SetCustomDataByKey(
                "physics_ownership", "Newton Style3D artifact; visual-only Isaac copy"
            )

            panel_visuals: dict[str, GarmentPanelVisualHandle] = {}
            for panel_index, panel in enumerate(construction.panels):
                material_path = (
                    f"/World/Looks/Garments/{_safe_path_component(construction_id)}/"
                    f"{_safe_path_component(panel.id)}"
                )
                panel_material = _textile_material(stage, material_path, panel)
                materials[f"{construction_id}:{panel.id}"] = panel_material
                render_topology = build_garment_panel_render_topology(
                    pattern_mesh,
                    source,
                    panel.id,
                    subdivision_factor=subdivision_factor,
                    texture_repeats_per_m=10.0,
                )
                mesh_path = (
                    f"{root_path}/Panels/P{panel_index:02}_"
                    f"{_safe_path_component(panel.id)}"
                )
                mesh = _mesh_from_panel(
                    stage,
                    mesh_path,
                    render_topology,
                    display_positions,
                    thickness_m=panel.unloaded_thickness_m,
                    material=panel_material,
                )
                mesh.GetPrim().SetCustomDataByKey("panel_id", panel.id)
                mesh.GetPrim().SetCustomDataByKey(
                    "declared_unloaded_thickness_m", panel.unloaded_thickness_m
                )
                panel_visuals[panel.id] = GarmentPanelVisualHandle(
                    panel_id=panel.id,
                    mesh=mesh,
                    topology=render_topology,
                    declared_thickness_m=panel.unloaded_thickness_m,
                )

            emit(
                f"authored {construction_id}: {len(panel_visuals)} panel shells"
            )

            detail_paths = _construction_details(
                stage,
                root_path,
                construction,
                pattern_mesh,
                source,
                display_positions,
                materials,
            )
            detail_views = {
                path: _detail_view_modes(construction, path) for path in detail_paths
            }
            cameras = _garment_cameras(stage, construction_id, bounds)
            garments[construction_id] = GarmentShowcaseItem(
                construction=construction,
                pattern_mesh=pattern_mesh,
                render_source=source,
                root_path=root_path,
                panel_visuals=panel_visuals,
                physical_vertex_count=expected_physical_count,
                position_offset_m=offset,
                source_positions_m=positions,
                source_triangle_count=len(physical_triangles),
                visual_detail_paths=detail_paths,
                visual_detail_views=detail_views,
                cameras=cameras,
                bounds_m=bounds,
            )
            emit(
                f"finished {construction_id}: {len(detail_paths)} construction details"
            )

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Ambient")
    dome.CreateIntensityAttr(460.0)
    dome.CreateColorAttr(Gf.Vec3f(0.79, 0.85, 0.94))
    key = UsdLux.RectLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(4200.0)
    key.CreateWidthAttr(1.20)
    key.CreateHeightAttr(0.72)
    from .laundry_scene import _look_at  # local import keeps the helper private

    _look_at(key, (-0.55, -0.72, 1.62), (0.0, 0.0, 0.0))
    fill = UsdLux.RectLight.Define(stage, "/World/Lights/Fill")
    fill.CreateIntensityAttr(2500.0)
    fill.CreateWidthAttr(0.90)
    fill.CreateHeightAttr(0.60)
    _look_at(fill, (0.86, -0.18, 1.05), (0.0, 0.04, 0.0))
    rim = UsdLux.RectLight.Define(stage, "/World/Lights/Rim")
    rim.CreateIntensityAttr(1900.0)
    rim.CreateWidthAttr(0.74)
    rim.CreateHeightAttr(0.42)
    _look_at(rim, (0.15, 0.80, 1.18), (0.0, 0.0, 0.0))

    handles = GarmentShowcaseHandles(
        garments=garments,
        materials=materials,
        catalog_frame_path=catalog_frame_path,
        catalog_frame_sha256="sha256:"
        + hashlib.sha256(catalog_frame_path.read_bytes()).hexdigest(),
        selected_construction_id=selected_construction_id,
        selected_view_mode="front",
    )
    set_garment_view(handles, selected_construction_id, "front")
    emit("garment showcase scene complete")
    return handles


def audit_garment_showcase_scene(
    stage: Usd.Stage,
    handles: GarmentShowcaseHandles,
) -> dict[str, object]:
    expected_ids = tuple(GARMENT_CONSTRUCTIONS)
    rigid_body_paths = [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    cases: list[dict[str, object]] = []
    for construction_id, item in handles.garments.items():
        panel_ids = frozenset(item.panel_visuals)
        view_panels = _VIEW_PANEL_IDS.get(construction_id, {})
        front_panels = view_panels.get("front", frozenset())
        back_panels = view_panels.get("back", frozenset())
        render_vertices = 0
        shell_faces = 0
        finite = True
        for panel in item.panel_visuals.values():
            points = np.asarray(panel.mesh.GetPointsAttr().Get(), dtype=np.float64)
            finite = finite and bool(np.all(np.isfinite(points)))
            render_vertices += len(points)
            shell_faces += len(panel.topology.face_vertex_counts)
        cases.append(
            {
                "construction_id": construction_id,
                "physical_vertices": item.physical_vertex_count,
                "physical_triangles": item.source_triangle_count,
                "physical_panels": len(item.construction.panels),
                "render_panel_shells": len(item.panel_visuals),
                "render_vertices": render_vertices,
                "render_shell_faces": shell_faces,
                "declared_features": len(item.construction.features),
                "construction_detail_primitives": len(item.visual_detail_paths),
                "finite": finite,
                "actual_panel_count_retained": (
                    len(item.panel_visuals) == len(item.construction.panels)
                ),
                "front_layer_panel_count": len(front_panels),
                "back_layer_panel_count": len(back_panels),
                "internal_or_overlay_panel_count": len(
                    panel_ids - front_panels - back_panels
                ),
                "view_panel_sets_valid": bool(
                    front_panels
                    and back_panels
                    and front_panels <= panel_ids
                    and back_panels <= panel_ids
                ),
                "detail_view_assignments_valid": bool(
                    set(item.visual_detail_views) == set(item.visual_detail_paths)
                    and all(
                        modes
                        and modes <= frozenset(SHOWCASE_VIEW_MODES)
                        and "construction" in modes
                        for modes in item.visual_detail_views.values()
                    )
                ),
                "three_inspection_views": tuple(item.cameras)
                == SHOWCASE_VIEW_MODES,
            }
        )
    gates = {
        "all_five_canonical_constructions_present": tuple(handles.garments)
        == expected_ids,
        "all_actual_cut_panels_have_render_shells": all(
            case["actual_panel_count_retained"] for case in cases
        ),
        "all_render_points_finite": all(case["finite"] for case in cases),
        "construction_features_visible_for_every_garment": all(
            case["construction_detail_primitives"] > 0 for case in cases
        ),
        "front_and_back_layer_sets_are_valid": all(
            case["view_panel_sets_valid"] for case in cases
        ),
        "detail_annotations_follow_owned_layers": all(
            case["detail_view_assignments_valid"] for case in cases
        ),
        "front_back_construction_views_per_garment": all(
            case["three_inspection_views"] for case in cases
        ),
        "visual_scene_has_no_rigid_body_dynamics": not rigid_body_paths,
        "selected_garment_is_valid": handles.selected_construction_id
        in handles.garments,
        "selected_view_is_valid": handles.selected_view_mode
        in SHOWCASE_VIEW_MODES,
    }
    return {
        "schema_version": "laundry-garment-showcase-audit/1.1",
        "source_probe": "P13",
        "source_artifact": handles.catalog_frame_path.relative_to(HERE).as_posix(),
        "source_artifact_sha256": handles.catalog_frame_sha256,
        "cases": cases,
        "gates": gates,
        "pass": all(gates.values()),
        "rigid_body_api_paths": rigid_body_paths,
        "render_only_annotations": (
            "thread, buttons, buttonholes, rivets, zipper teeth, and dobby relief; "
            "all panel shells remain embedded in the committed welded P13 state; "
            "front/back are named layer-isolation views, not alternate poses"
        ),
        "calibration_status": "construction-complete engineering priors; not garment-calibrated",
    }


__all__ = [
    "DEFAULT_CATALOG_FRAMES",
    "SHOWCASE_VIEW_MODES",
    "GarmentPanelVisualHandle",
    "GarmentShowcaseHandles",
    "GarmentShowcaseItem",
    "audit_garment_showcase_scene",
    "build_garment_showcase_scene",
    "set_garment_view",
    "set_selected_garment",
]
