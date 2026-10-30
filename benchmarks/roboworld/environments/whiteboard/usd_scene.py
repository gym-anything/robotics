"""PBR USD export of the current Newton state. Rendering never steps physics."""

from dataclasses import asdict
import json
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
from .config import COLORS

PALETTE = {
    "wall": ((0.76, 0.77, 0.74), 0.82, 0),
    "ceiling": ((0.89, 0.89, 0.86), 0.91, 0),
    "paint_white": ((0.81, 0.82, 0.80), 0.36, 0),
    "linoleum": ((0.41, 0.44, 0.40), 0.37, 0),
    "steel": ((0.15, 0.18, 0.20), 0.29, 0.8),
    "aluminum": ((0.66, 0.69, 0.72), 0.23, 1),
    "chrome": ((0.72, 0.75, 0.79), 0.12, 1),
    "charcoal": ((0.03, 0.04, 0.045), 0.60, 0),
    "wood": ((0.63, 0.40, 0.21), 0.4, 0),
    "mdf": ((0.39, 0.25, 0.14), 0.85, 0),
    "laminate": ((0.66, 0.57, 0.41), 0.32, 0),
    "dark_plastic": ((0.13, 0.15, 0.16), 0.53, 0),
    "chair_blue": ((0.047, 0.17, 0.27), 0.31, 0),
    "marker_white": ((0.90, 0.90, 0.87), 0.27, 0),
    "board": ((0.965, 0.965, 0.96), 0.13, 0),
    "felt": ((0.09, 0.11, 0.12), 0.97, 0),
    "eraser_blue": ((0.03, 0.13, 0.38), 0.36, 0),
    "glass": ((0.56, 0.71, 0.77), 0.06, 0),
    "dark_glass": ((0.07, 0.14, 0.17), 0.10, 0),
    "diffuser": ((0.85, 0.89, 0.94), 0.28, 0),
    **{name: (rgb, 0.29, 0) for name, rgb in COLORS.items()},
}


def transform(prim, pos, quat=(0, 0, 0, 1)):
    x = UsdGeom.Xformable(prim)
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    x.AddOrientOp().Set(Gf.Quatf(float(quat[3]), Gf.Vec3f(*map(float, quat[:3]))))


def material(stage, name, output, texture=None):
    color, rough, metal = PALETTE[name]
    mat = UsdShade.Material.Define(stage, "/Materials/" + name)
    shader = UsdShade.Shader.Define(stage, f"/Materials/{name}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.48)
    if name == "diffuser":
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.94, 1))
    if name == "glass":
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.20)
    if texture:
        reader = UsdShade.Shader.Define(stage, f"/Materials/{name}/uv")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"/Materials/{name}/texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("textures/" + texture.name)
        )
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    if name == "board" and texture:
        # Keep the portable PreviewSurface above, and explicitly bind the
        # pixel-resolved physical board texture in the RTX/MDL context.
        mdl = UsdShade.Shader.Define(stage, f"/Materials/{name}/rtx_surface")
        mdl.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        mdl.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        mdl.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        mdl.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
        mdl.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(texture.resolve()))
        )
        mdl.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(rough)
        mdl.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
        mdl.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(False)
        mdl.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1, 1))
        mat.CreateSurfaceOutput("mdl").ConnectToSource(mdl.ConnectableAPI(), "out")
    return mat


def textures(output, ink):
    folder = output / "textures"
    folder.mkdir(exist_ok=True)
    Image.fromarray(ink.rgb()).save(folder / "board.png")
    rng = np.random.default_rng(312)
    y, x = np.mgrid[:512, :512]
    noise = rng.normal(0, 1, (512, 512))
    from .tool_appearance import marker_label

    paths = {"board": folder / "board.png", "marker_white": marker_label(folder)}
    for name, base in [
        ("linoleum", (156, 162, 150)),
        ("wood", (186, 146, 99)),
        ("laminate", (187, 174, 147)),
        ("wall", (217, 219, 212)),
    ]:
        if name in ("wood", "laminate"):
            pattern = (
                3 * np.sin(x * 0.10 + 2 * np.sin(y * 0.006))
                + 1.5 * np.sin(x * 0.48 + y * 0.006)
                + noise
            )
        elif name == "linoleum":
            pattern = noise * 3 + 2 * np.sin(x * 0.12) * np.sin(y * 0.09)
        else:
            pattern = noise * 0.6
        rgb = np.clip(np.asarray(base)[None, None, :] + pattern[:, :, None], 0, 255).astype(
            np.uint8
        )
        path = folder / (name + ".png")
        Image.fromarray(rgb).save(path)
        paths[name] = path
    return paths


def mesh(stage, path, vertices, faces, uv=None):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, np.float32)))
    m.CreateFaceVertexCountsAttr([len(f) for f in faces])
    m.CreateFaceVertexIndicesAttr(np.concatenate(faces).astype(np.int32).tolist())
    m.CreateSubdivisionSchemeAttr("none")
    if uv is not None:
        UsdGeom.PrimvarsAPI(m).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying"
        ).Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))
    return m


def bevel_box(stage, path, size, bevel):
    h = np.asarray(size) / 2
    radius = min(bevel, float(h.min()) * 0.42)
    points = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                sign = np.array([sx, sy, sz])
                for axis in range(3):
                    p = h - radius
                    p[axis] = h[axis]
                    points.append(p * sign)
    points = np.asarray(points)
    hull = ConvexHull(points)
    faces, uv = [], []
    for tri, equation in zip(hull.simplices, hull.equations):
        face = tri.copy()
        a, b, c = points[face]
        if np.dot(np.cross(b - a, c - a), equation[:3]) < 0:
            face = face[::-1]
        faces.append(face)
        drop = np.argmax(np.abs(equation[:3]))
        coords = np.delete(points[face], drop, axis=1)
        uv.extend(coords * 2)
    return mesh(stage, path, points, faces, uv)


def smooth_cylinder(stage, path, radius, height):
    import trimesh
    from .tool_appearance import cylindrical_uv

    shape = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
    g = mesh(
        stage,
        path,
        shape.vertices,
        shape.faces,
        cylindrical_uv(shape.vertices, shape.faces, height),
    )
    normals = []
    for face in shape.faces:
        vertices = shape.vertices[face]
        if np.ptp(vertices[:, 2]) < 1e-10:
            normals.extend([(0, 0, float(np.sign(vertices[0, 2])))] * 3)
        else:
            normals.extend([(float(v[0] / radius), float(v[1] / radius), 0.0) for v in vertices])
    g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals, np.float32)))
    g.SetNormalsInterpolation("faceVarying")
    return g


def camera(stage, name, eye, target, focal=30, aperture=36, role="teacher"):
    c = UsdGeom.Camera.Define(stage, "/Cameras/" + name)
    matrix = (
        Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    )
    c.AddTransformOp().Set(matrix)
    c.CreateFocalLengthAttr(focal)
    c.CreateHorizontalApertureAttr(aperture)
    c.CreateVerticalApertureAttr(aperture * 9 / 16)
    c.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
    c.GetPrim().SetCustomDataByKey("observation_role", role)
    return c


def export_scene(env, output):
    output.mkdir(parents=True, exist_ok=True)
    paths = textures(output, env.ink)
    Image.fromarray(env.task.reference_view(env.ink)).save(output / "task_reference.png")
    stage = Usd.Stage.CreateNew(str(output / "classroom.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Classroom")
    stage.SetDefaultPrim(root.GetPrim())
    mats = {name: material(stage, name, output, paths.get(name)) for name in PALETTE}
    from .visual_assets import copy_to, ASSETS
    from .appearance import bind_pbr, bind_glass

    external_maps = copy_to(output)
    for name, source in (
        ("wood", "oak_veneer_01"),
        ("laminate", "oak_veneer_01"),
        ("wall", "beige_wall_001"),
    ):
        bind_pbr(
            stage,
            mats[name],
            external_maps[source],
            ASSETS[source]["tile_m"],
            coated=name in ("wood", "laminate"),
            roughness=0.42 if name == "wood" else 0.35,
        )
    bind_glass(stage, mats["glass"])
    q = env.state_0.body_q.numpy()
    from .deformable_contact import render_meshes

    deformable = render_meshes(env)
    for obj in env.objects:
        parent = "/Classroom/" + obj.name
        xf = UsdGeom.Xform.Define(stage, parent)
        pose = (
            q[env.body_ids[obj.name]]
            if obj.name in env.body_ids
            else [*obj.position, *obj.rotation]
        )
        transform(xf, pose[:3], pose[3:])
        for part in obj.parts:
            position = part.position
            if obj.name in deformable and (
                part.name in ("nib", "pad") or part.name.startswith("wall_")
            ):
                continue
            path = parent + "/" + part.name
            if part.material == "board":
                w, _, h = part.size
                # Facing +Y; UV image top maps to top of the physical board.
                g = mesh(
                    stage,
                    path,
                    [
                        [-w / 2, 0.0005, -h / 2],
                        [-w / 2, 0.0005, h / 2],
                        [w / 2, 0.0005, h / 2],
                        [w / 2, 0.0005, -h / 2],
                    ],
                    [[0, 1, 2, 3]],
                    [[0, 0], [0, 1], [1, 1], [1, 0]],
                )
            elif part.kind == "mesh":
                g = mesh(stage, path, part.vertices, part.faces)
                # Vertex normals smooth the moulded shell, not the physics shape.
                import trimesh

                normals = trimesh.Trimesh(
                    vertices=part.vertices, faces=part.faces, process=False
                ).vertex_normals
                g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals, np.float32)))
                g.SetNormalsInterpolation("vertex")
            elif part.kind == "box":
                g = bevel_box(stage, path, part.size, part.bevel)
            elif obj.name.startswith("marker_") and part.name == "barrel":
                # The collision union deliberately overlaps barrel/end pieces.
                # Render only the exposed white interval so coplanar coloured
                # tail/collar surfaces cannot z-fight. The outer union is unchanged.
                tail = next(p for p in obj.parts if p.name == "tail")
                collar = next(p for p in obj.parts if p.name == "collar")
                low = tail.position[2] + tail.size[1] / 2
                high = collar.position[2] - collar.size[1] / 2
                g = smooth_cylinder(stage, path, part.size[0], high - low)
                position = (*part.position[:2], (high + low) / 2)
            else:
                g = smooth_cylinder(stage, path, part.size[0], part.size[1])
            transform(g, position, part.rotation)
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats[part.material])
        if obj.name in deformable:
            data = deformable[obj.name]
            g = mesh(stage, parent + "/elastic_material", data["vertices"], data["faces"])
            color = "felt" if obj.name == "eraser" else obj.name.split("_", 1)[1]
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats[color])
    # Visual robot geometry comes from the same imported Newton articulation.
    builder = env.builder
    for i in range(len(env.shape_ids), builder.shape_count):
        if not (builder.shape_flags[i] & int(env.newton.ShapeFlags.VISIBLE)):
            continue
        body = builder.shape_body[i]
        if body < 0:
            continue
        kind = builder.shape_type[i]
        local = np.asarray(builder.shape_transform[i])
        pose = q[body]
        rot = Rotation.from_quat(pose[3:])
        p = pose[:3] + rot.apply(local[:3])
        r = (rot * Rotation.from_quat(local[3:])).as_quat()
        path = f"/Classroom/G1/shape_{i}"
        source = builder.shape_source[i]
        scale = np.asarray(builder.shape_scale[i])
        if source is not None and hasattr(source, "vertices"):
            vertices = np.asarray(source.vertices) * scale
            faces = np.asarray(source.indices).reshape(-1, 3)
            g = mesh(stage, path, vertices, faces)
            import trimesh

            normals = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).vertex_normals
            g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals, np.float32)))
            g.SetNormalsInterpolation("vertex")
        elif kind == env.newton.GeoType.SPHERE:
            g = UsdGeom.Sphere.Define(stage, path)
            g.CreateRadiusAttr(scale[0])
        elif kind == env.newton.GeoType.BOX:
            g = bevel_box(stage, path, 2 * scale, 0.003)
        elif kind in (env.newton.GeoType.CYLINDER, env.newton.GeoType.CAPSULE):
            g = (
                UsdGeom.Capsule.Define(stage, path)
                if kind == env.newton.GeoType.CAPSULE
                else UsdGeom.Cylinder.Define(stage, path)
            )
            g.CreateRadiusAttr(scale[0])
            g.CreateHeightAttr(scale[1] * 2)
            g.CreateAxisAttr("Z")
        else:
            continue
        transform(g, p, r)
        color = np.asarray(builder.shape_color[i])
        name = "paint_white" if color.mean() > 0.40 else "charcoal"
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats[name])
    dome = UsdLux.DomeLight.Define(stage, "/Lighting/Sky")
    dome.CreateTextureFileAttr(
        Sdf.AssetPath("textures/" + external_maps["kiara_5_noon"]["hdri"].name)
    )
    dome.CreateTextureFormatAttr("latlong")
    dome.CreateIntensityAttr(350.0)
    dome.AddRotateZOp().Set(90)
    for i, y in enumerate((1.6, 4.0, 6.5)):
        for j, x in enumerate((-1.6, 1.6)):
            light = UsdLux.RectLight.Define(stage, f"/Lighting/Ceiling_{i}_{j}")
            light.CreateWidthAttr(0.30)
            light.CreateHeightAttr(1.15)
            # Approx. 3800 lm over a 0.345 m² Lambertian panel (L = flux/pi/A).
            light.CreateIntensityAttr(3500)
            light.CreateColorAttr(Gf.Vec3f(1.0, 0.94, 0.84))
            transform(light, (x, y, 3.13))
    camera(stage, "classroom", (3.1, 6.5, 2.1), (-0.3, 0.6, 1.3), focal=25)
    camera(stage, "board", (2.3, 3.0, 1.85), (0, 0.08, 1.52), focal=35)
    camera(stage, "tools", (0.20, 1.20, 1.35), (-0.15, 0.13, 0.92), focal=25)
    camera(stage, "duster", (0.75, 0.42, 1.10), (0.62, 0.14, 0.925), focal=50)
    camera(
        stage,
        "nib",
        (0.12, 0.45, env.config.board.center_z + 0.075),
        (0.025, env.config.board.face_y, env.config.board.center_z),
        focal=62,
    )
    camera(stage, "student", (-1.7, 6.7, 1.2), (0, 0.08, 1.53), focal=34)
    camera(
        stage,
        "stroke",
        (0.025, 0.38, env.config.board.center_z),
        (0.025, env.config.board.face_y, env.config.board.center_z),
        focal=62,
    )
    # Onboard optical calibration is separate from privileged review cameras.
    # Mount is resolved from the revision-pinned URDF, not a floating room camera.
    if env.config.robot:
        from .sensors import optical_pose

        eye, look, up = optical_pose(env)
        c = UsdGeom.Camera.Define(stage, "/Cameras/g1_rgb")
        c.AddTransformOp().Set(
            Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(*up)).GetInverse()
        )
        c.CreateFocalLengthAttr(18.0 / np.tan(np.deg2rad(69 / 2)))
        c.CreateHorizontalApertureAttr(36)
        c.CreateVerticalApertureAttr(20.25)
        c.CreateClippingRangeAttr(Gf.Vec2f(0.03, 100))
        c.GetPrim().SetCustomDataByKey("observation_role", "policy")
    stage.GetRootLayer().Save()
    metadata = {
        "state": env.get_state(),
        "config": asdict(env.config),
        "task": {
            "specification": asdict(env.task),
            "description": env.task.description,
        },
        "physics": "Newton 1.5.0 / SolverMuJoCo",
        "renderer_dynamics_steps": 0,
        "geometry": "shared scene plus current Newton G1 body transforms",
        "appearance_assets": ASSETS,
        "review_cameras": ["classroom", "board", "tools", "student"],
        "policy_cameras": ["g1_rgb"] if env.config.robot else [],
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    return output / "classroom.usda"
