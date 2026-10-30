"""Professional textile-lab USD scene for LaundryFold visualization.

This module authors render geometry only. Newton remains the sole owner of
cloth, contact, and rigid-fixture dynamics; the visual bridge only copies
committed snapshots into the mesh and fixture transforms declared here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from .mesh import RectangularMesh
from .render_mesh import ThickClothTopology, build_thick_cloth_topology, deform_thick_cloth


HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"


@dataclass
class LaundrySceneHandles:
    cloth_mesh: UsdGeom.Mesh
    cloth_topology: ThickClothTopology
    cloth_thickness_m: float
    body_transform_ops: dict[str, UsdGeom.XformOp]
    cameras: dict[str, str]
    render_materials: dict[str, UsdShade.Material]


def _bind(geometry: UsdGeom.Gprim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(material)


def _omnipbr_material(
    stage: Usd.Stage,
    path: str,
    color: tuple[float, float, float],
    *,
    roughness: float,
    metallic: float = 0.0,
    albedo_texture: Path | None = None,
    normal_texture: Path | None = None,
    roughness_texture: Path | None = None,
    project_uvw: bool = True,
    texture_scale: tuple[float, float] = (1.0, 1.0),
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(
        float(metallic)
    )
    shader.CreateInput(
        "reflection_roughness_constant", Sdf.ValueTypeNames.Float
    ).Set(float(roughness))
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(bool(project_uvw))
    shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(
        Gf.Vec2f(*texture_scale)
    )
    if albedo_texture is not None:
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(albedo_texture.resolve().as_posix())
        )
    if normal_texture is not None:
        shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(normal_texture.resolve().as_posix())
        )
        shader.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(0.62)
    if roughness_texture is not None:
        shader.CreateInput(
            "reflectionroughness_texture", Sdf.ValueTypeNames.Asset
        ).Set(Sdf.AssetPath(roughness_texture.resolve().as_posix()))
        shader.CreateInput(
            "reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float
        ).Set(1.0)
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        shader.ConnectableAPI(), "out"
    )
    return material


def _cube(
    stage: Usd.Stage,
    path: str,
    size_m: tuple[float, float, float],
    position_m: tuple[float, float, float],
    material: UsdShade.Material,
) -> UsdGeom.Cube:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position_m))
    xform.AddScaleOp().Set(Gf.Vec3f(*size_m))
    _bind(cube, material)
    return cube


def _cylinder(
    stage: Usd.Stage,
    path: str,
    *,
    radius_m: float,
    height_m: float,
    axis: str,
    position_m: tuple[float, float, float],
    material: UsdShade.Material,
) -> UsdGeom.Cylinder:
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(radius_m)
    cylinder.CreateHeightAttr(height_m)
    cylinder.CreateAxisAttr(axis)
    UsdGeom.Xformable(cylinder).AddTranslateOp().Set(Gf.Vec3d(*position_m))
    _bind(cylinder, material)
    return cylinder


def _capsule(
    stage: Usd.Stage,
    path: str,
    *,
    radius_m: float,
    height_m: float,
    axis: str,
    material: UsdShade.Material,
) -> UsdGeom.Capsule:
    capsule = UsdGeom.Capsule.Define(stage, path)
    capsule.CreateRadiusAttr(radius_m)
    capsule.CreateHeightAttr(height_m)
    capsule.CreateAxisAttr(axis)
    _bind(capsule, material)
    return capsule


def _look_at(
    xformable: UsdGeom.Xformable,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> None:
    e = Gf.Vec3d(*eye)
    t = Gf.Vec3d(*target)
    forward = (t - e).GetNormalized()
    world_up = Gf.Vec3d(0.0, 0.0, 1.0)
    right = Gf.Cross(forward, world_up)
    if right.GetLength() <= 1.0e-9:
        world_up = Gf.Vec3d(0.0, 1.0, 0.0)
        right = Gf.Cross(forward, world_up)
    right = right.GetNormalized()
    up = Gf.Cross(right, forward)
    matrix = Gf.Matrix4d(
        right[0], right[1], right[2], 0.0,
        up[0], up[1], up[2], 0.0,
        -forward[0], -forward[1], -forward[2], 0.0,
        e[0], e[1], e[2], 1.0,
    )
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(matrix)


def _create_camera(
    stage: Usd.Stage,
    path: str,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    focal_length_mm: float,
) -> str:
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(focal_length_mm)
    camera.CreateHorizontalApertureAttr(24.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    _look_at(camera, eye, target)
    return path


def _create_textured_card(
    stage: Usd.Stage,
    material: UsdShade.Material,
) -> None:
    mesh = UsdGeom.Mesh.Define(stage, "/World/Lab/CalibrationCard")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-0.54, 0.872, 0.12),
            Gf.Vec3f(0.54, 0.872, 0.12),
            Gf.Vec3f(0.54, 0.872, 0.78),
            Gf.Vec3f(-0.54, 0.872, 0.78),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    # Face the operator/cameras at negative Y.
    mesh.CreateFaceVertexIndicesAttr([0, 3, 2, 1])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    uv.Set(
        [
            Gf.Vec2f(0.0, 0.0),
            Gf.Vec2f(1.0, 0.0),
            Gf.Vec2f(1.0, 1.0),
            Gf.Vec2f(0.0, 1.0),
        ]
    )
    _bind(mesh, material)


def _create_static_towel_stack(
    stage: Usd.Stage,
    root: str,
    center: tuple[float, float, float],
    materials: Iterable[UsdShade.Material],
) -> None:
    x, y, z = center
    colors = tuple(materials)
    for layer in range(4):
        offset = 0.004 * ((layer % 2) - 0.5)
        _cube(
            stage,
            f"{root}/Towel{layer}",
            (0.26, 0.19, 0.034),
            (x + offset, y, z + layer * 0.036),
            colors[layer % len(colors)],
        )


def _dynamic_xform(stage: Usd.Stage, path: str) -> tuple[UsdGeom.Xform, UsdGeom.XformOp]:
    xform = UsdGeom.Xform.Define(stage, path)
    xformable = UsdGeom.Xformable(xform)
    xformable.ClearXformOpOrder()
    return xform, xformable.AddTransformOp()


def build_laundry_scene(
    stage: Usd.Stage,
    mesh: RectangularMesh,
    *,
    cloth_thickness_m: float = 0.0008,
    subdivision_factor: int = 3,
) -> LaundrySceneHandles:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    for scope in ("Looks", "Lab", "Dynamic", "Cameras", "Lights"):
        UsdGeom.Scope.Define(stage, f"/World/{scope}")

    materials = {
        "cloth": _omnipbr_material(
            stage,
            "/World/Looks/CanonicalCotton",
            # The authored albedo already carries the sage dye.  A near-white
            # constant avoids multiplying it into rubber-like darkness.
            (0.94, 0.98, 0.95),
            roughness=0.84,
            albedo_texture=ASSETS / "woven_cotton_albedo.png",
            normal_texture=ASSETS / "woven_cotton_normal.png",
            roughness_texture=ASSETS / "woven_cotton_roughness.png",
            project_uvw=False,
        ),
        "table": _omnipbr_material(
            stage,
            "/World/Looks/LabLaminate",
            (0.60, 0.62, 0.64),
            roughness=0.58,
            albedo_texture=ASSETS / "table_laminate.png",
            texture_scale=(0.38, 0.38),
        ),
        "floor": _omnipbr_material(
            stage,
            "/World/Looks/Concrete",
            (0.37, 0.38, 0.39),
            roughness=0.86,
            albedo_texture=ASSETS / "concrete_floor.png",
            texture_scale=(0.55, 0.55),
        ),
        "wall": _omnipbr_material(stage, "/World/Looks/Wall", (0.78, 0.80, 0.80), roughness=0.88),
        "metal": _omnipbr_material(stage, "/World/Looks/Aluminum", (0.48, 0.52, 0.55), roughness=0.27, metallic=0.92),
        "dark_metal": _omnipbr_material(stage, "/World/Looks/DarkMetal", (0.055, 0.065, 0.070), roughness=0.31, metallic=0.88),
        "rubber": _omnipbr_material(stage, "/World/Looks/CompliantRubber", (0.035, 0.055, 0.060), roughness=0.76),
        # A matte polyurethane sleeve reads as an industrial textile roller;
        # the visual radius remains the exact 4 mm contact radius from P6.
        "roller": _omnipbr_material(stage, "/World/Looks/Roller", (0.72, 0.76, 0.74), roughness=0.67, metallic=0.0),
        "cotton_navy": _omnipbr_material(stage, "/World/Looks/CottonNavy", (0.10, 0.18, 0.25), roughness=0.86),
        "cotton_sand": _omnipbr_material(stage, "/World/Looks/CottonSand", (0.65, 0.58, 0.47), roughness=0.88),
        "card": _omnipbr_material(
            stage,
            "/World/Looks/TaskCard",
            (1.0, 1.0, 1.0),
            roughness=0.62,
            albedo_texture=ASSETS / "task_card.png",
            project_uvw=False,
        ),
    }

    # Room and exact finite support surface used by P6.
    _cube(stage, "/World/Lab/Floor", (3.4, 2.8, 0.05), (0.0, 0.0, -0.835), materials["floor"])
    _cube(stage, "/World/Lab/BackWall", (3.4, 0.05, 1.75), (0.0, 0.90, 0.02), materials["wall"])
    _cube(stage, "/World/Lab/LeftWall", (0.05, 1.8, 1.75), (-1.68, 0.0, 0.02), materials["wall"])
    _cube(stage, "/World/Lab/TableTop", (0.602, 0.460, 0.040), (0.049, 0.0, -0.020), materials["table"])
    _cube(stage, "/World/Lab/TableApron", (0.68, 0.50, 0.055), (0.049, 0.0, -0.070), materials["dark_metal"])
    for index, (x, y) in enumerate(((-0.22, -0.18), (-0.22, 0.18), (0.32, -0.18), (0.32, 0.18))):
        _cube(stage, f"/World/Lab/TableLeg{index}", (0.045, 0.045, 0.72), (x, y, -0.43), materials["dark_metal"])

    # Shelving, baskets, and folded reference textiles give the atomic task a
    # believable textile-lab context without adding simulated garments.
    _cube(stage, "/World/Lab/Shelf/Frame", (0.62, 0.035, 1.35), (-1.18, 0.70, -0.08), materials["dark_metal"])
    for level, z in enumerate((-0.61, -0.25, 0.11, 0.47)):
        _cube(stage, f"/World/Lab/Shelf/Level{level}", (0.62, 0.34, 0.025), (-1.18, 0.58, z), materials["metal"])
    _create_static_towel_stack(stage, "/World/Lab/Shelf/StackA", (-1.18, 0.55, 0.145), (materials["cotton_navy"], materials["cotton_sand"]))
    _create_static_towel_stack(stage, "/World/Lab/Shelf/StackB", (-1.18, 0.55, 0.505), (materials["cotton_sand"], materials["cotton_navy"]))
    # Open laundry cart with restrained wire rails.
    _cube(stage, "/World/Lab/Cart/Base", (0.54, 0.40, 0.045), (1.02, 0.33, -0.64), materials["dark_metal"])
    for index, (x, y) in enumerate(((0.77, 0.15), (0.77, 0.51), (1.27, 0.15), (1.27, 0.51))):
        _cylinder(stage, f"/World/Lab/Cart/Post{index}", radius_m=0.010, height_m=0.72, axis="Z", position_m=(x, y, -0.29), material=materials["metal"])
    for index, z in enumerate((-0.48, -0.18, 0.12)):
        _cube(stage, f"/World/Lab/Cart/RailFront{index}", (0.54, 0.012, 0.012), (1.02, 0.15, z), materials["metal"])
        _cube(stage, f"/World/Lab/Cart/RailBack{index}", (0.54, 0.012, 0.012), (1.02, 0.51, z), materials["metal"])

    _create_textured_card(stage, materials["card"])

    # Fixed portal frame around the exact P6 laboratory fixtures.
    for index, (x, y) in enumerate(((-0.43, -0.28), (-0.43, 0.28), (0.43, -0.28), (0.43, 0.28))):
        _cube(stage, f"/World/Lab/Portal/Post{index}", (0.032, 0.032, 0.73), (x, y, 0.345), materials["dark_metal"])
    _cube(stage, "/World/Lab/Portal/RailFront", (0.90, 0.035, 0.045), (0.0, -0.28, 0.72), materials["metal"])
    _cube(stage, "/World/Lab/Portal/RailBack", (0.90, 0.035, 0.045), (0.0, 0.28, 0.72), materials["metal"])

    body_ops: dict[str, UsdGeom.XformOp] = {}
    roots: dict[str, UsdGeom.Xform] = {}
    for name in (
        "gripper_x_carriage",
        "gripper_gantry",
        "gripper_wrist",
        "gripper_positive_pad",
        "gripper_negative_pad",
        "roller_x_carriage",
        "roller_z_carriage",
        "crease_roller",
    ):
        root, op = _dynamic_xform(stage, f"/World/Dynamic/{name}")
        roots[name] = root
        body_ops[name] = op

    _cube(stage, roots["gripper_x_carriage"].GetPath().pathString + "/Trolley", (0.10, 0.54, 0.05), (0.0, 0.0, 0.67), materials["dark_metal"])
    _cube(stage, roots["gripper_gantry"].GetPath().pathString + "/Stem", (0.035, 0.055, 0.52), (0.0, 0.0, 0.29), materials["metal"])
    _cylinder(stage, roots["gripper_wrist"].GetPath().pathString + "/Wrist", radius_m=0.030, height_m=0.072, axis="Y", position_m=(0.0, 0.0, 0.0), material=materials["dark_metal"])
    for name, backing_offset in (
        ("gripper_positive_pad", 0.00375),
        ("gripper_negative_pad", -0.00375),
    ):
        root_path = roots[name].GetPath().pathString
        _cube(stage, root_path + "/Backing", (0.072, 0.414, 0.007), (0.0, 0.0, backing_offset), materials["metal"])
        _cube(stage, root_path + "/Pad", (0.064, 0.410, 0.004), (0.0, 0.0, 0.0), materials["rubber"])

    _cube(stage, roots["roller_x_carriage"].GetPath().pathString + "/Trolley", (0.09, 0.52, 0.045), (0.0, 0.0, 0.61), materials["dark_metal"])
    carrier_path = roots["roller_z_carriage"].GetPath().pathString
    _cube(stage, carrier_path + "/Crossbar", (0.045, 0.46, 0.030), (0.0, 0.0, 0.055), materials["metal"])
    for y in (-0.208, 0.208):
        _cylinder(stage, carrier_path + ("/BearingN" if y > 0 else "/BearingS"), radius_m=0.014, height_m=0.025, axis="Y", position_m=(0.0, y, 0.0), material=materials["dark_metal"])
    _capsule(
        stage,
        roots["crease_roller"].GetPath().pathString + "/Roller",
        radius_m=0.004,
        height_m=0.402,
        axis="Y",
        material=materials["roller"],
    )

    topology = build_thick_cloth_topology(
        mesh,
        subdivision_factor=subdivision_factor,
        # The source tile contains many yarns.  Ten tiles per metre keeps
        # individual yarn-scale contrast above a pixel in the render cameras,
        # while 60/m aliases into a smooth plastic-looking surface.
        texture_repeats_per_m=10.0,
    )
    initial_points, initial_normals = deform_thick_cloth(
        topology, mesh.positions_m, thickness_m=cloth_thickness_m
    )
    cloth = UsdGeom.Mesh.Define(stage, "/World/Dynamic/CanonicalTowel")
    cloth.CreatePointsAttr(
        [Gf.Vec3f(*map(float, point)) for point in initial_points]
    )
    cloth.CreateFaceVertexCountsAttr(
        [int(value) for value in topology.face_vertex_counts]
    )
    cloth.CreateFaceVertexIndicesAttr(
        [int(value) for value in topology.face_vertex_indices]
    )
    cloth.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    cloth.CreateDoubleSidedAttr(False)
    cloth.CreateNormalsAttr(
        [Gf.Vec3f(*map(float, normal)) for normal in initial_normals]
    )
    cloth.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    st = UsdGeom.PrimvarsAPI(cloth).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set([Gf.Vec2f(*map(float, uv)) for uv in topology.uv])
    cloth.CreateExtentAttr(
        [
            Gf.Vec3f(*map(float, np.min(initial_points, axis=0))),
            Gf.Vec3f(*map(float, np.max(initial_points, axis=0))),
        ]
    )
    _bind(cloth, materials["cloth"])

    cameras = {
        # Centerline views keep the portal uprights at the image edges instead
        # of putting the front-right post across the towel.
        "hero": _create_camera(stage, "/World/Cameras/Hero", (0.04, -1.08, 0.50), (0.045, 0.0, 0.095), 40.0),
        "operator": _create_camera(stage, "/World/Cameras/Operator", (0.07, -1.20, 0.55), (0.05, 0.0, 0.10), 38.0),
        "overhead": _create_camera(stage, "/World/Cameras/Overhead", (0.05, -0.01, 1.35), (0.05, 0.0, 0.0), 52.0),
        # Side-on metrology view: the 410 mm roller axis spans the frame and
        # the loaded fold remains visible below it.  The camera sits between
        # the two left portal posts, so neither upright occludes contact.
        "crease": _create_camera(stage, "/World/Cameras/Crease", (-0.60, -0.02, 0.090), (0.024, 0.0, 0.018), 35.0),
        "profile": _create_camera(stage, "/World/Cameras/Profile", (-0.42, -0.62, 0.16), (0.01, 0.0, 0.016), 52.0),
        "workcell": _create_camera(stage, "/World/Cameras/Workcell", (1.55, -1.80, 1.08), (-0.12, 0.12, 0.05), 34.0),
    }

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Ambient")
    dome.CreateIntensityAttr(520.0)
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 0.96))
    key = UsdLux.RectLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(3600.0)
    key.CreateWidthAttr(1.15)
    key.CreateHeightAttr(0.65)
    _look_at(key, (0.25, -0.35, 1.35), (0.05, 0.0, 0.05))
    fill = UsdLux.RectLight.Define(stage, "/World/Lights/Fill")
    fill.CreateIntensityAttr(2100.0)
    fill.CreateWidthAttr(0.85)
    fill.CreateHeightAttr(0.55)
    _look_at(fill, (-0.75, -0.10, 0.85), (0.0, 0.0, 0.08))
    rim = UsdLux.RectLight.Define(stage, "/World/Lights/Rim")
    rim.CreateIntensityAttr(1700.0)
    rim.CreateWidthAttr(0.65)
    rim.CreateHeightAttr(0.45)
    _look_at(rim, (0.65, 0.55, 0.90), (0.08, 0.0, 0.10))

    return LaundrySceneHandles(
        cloth_mesh=cloth,
        cloth_topology=topology,
        cloth_thickness_m=cloth_thickness_m,
        body_transform_ops=body_ops,
        cameras=cameras,
        render_materials=materials,
    )


def audit_laundry_scene(
    stage: Usd.Stage,
    handles: LaundrySceneHandles,
) -> dict[str, object]:
    points = np.asarray(handles.cloth_mesh.GetPointsAttr().Get(), dtype=np.float64)
    rigid_bodies = [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    expected_bodies = {
        "gripper_x_carriage",
        "gripper_gantry",
        "gripper_wrist",
        "gripper_positive_pad",
        "gripper_negative_pad",
        "roller_x_carriage",
        "roller_z_carriage",
        "crease_roller",
    }
    gates = {
        "finite_cloth_points": bool(np.all(np.isfinite(points))),
        "finite_thickness_closed_shell": bool(
            len(points) == 2 * handles.cloth_topology.embedding.vertex_count
        ),
        "all_fixture_links_present": set(handles.body_transform_ops) == expected_bodies,
        "multiple_cameras": len(handles.cameras) >= 4,
        "visual_stage_has_no_rigid_body_dynamics": not rigid_bodies,
        "calibration_card_present": bool(stage.GetPrimAtPath("/World/Lab/CalibrationCard")),
    }
    return {
        "schema_version": "laundry-scene-audit/1.0",
        "gates": gates,
        "pass": all(gates.values()),
        "cloth_render_vertex_count": int(len(points)),
        "cloth_mechanical_vertex_count": int(
            np.max(handles.cloth_topology.embedding.coarse_vertex_indices) + 1
        ),
        "cloth_render_triangle_count": int(
            len(handles.cloth_topology.embedding.surface_triangles) * 2
        ),
        "camera_paths": handles.cameras,
        "visual_fixture_links": sorted(handles.body_transform_ops),
        "rigid_body_api_paths": rigid_bodies,
        "calibration_status": "prior-only / not physically calibrated",
    }


__all__ = [
    "LaundrySceneHandles",
    "audit_laundry_scene",
    "build_laundry_scene",
]
