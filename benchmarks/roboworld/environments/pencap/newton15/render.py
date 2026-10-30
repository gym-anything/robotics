"""PBR export from actual Newton poses; rendering cannot advance physics."""

from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
from .runtime import require_engine

PALETTE = {
    "floor": ((.22, .24, .23), .78, 0.), "wood": ((.48, .29, .13), .40, 0.),
    "pen": ((.95, .49, .025), .23, 0.), "cap": ((.014, .07, .38), .22, 0.),
    "wall": ((.75, .74, .69), .8, 0.), "steel": ((.16, .18, .19), .32, .8),
    "aluminum": ((.61, .64, .66), .22, 1.), "paper": ((.86, .84, .77), .85, 0.),
    "robot_white": ((.81, .83, .82), .28, 0.), "robot_dark": ((.018, .023, .028), .45, 0.),
    "robot_red": ((.55, .012, .008), .3, 0.), "ceramic": ((.32, .12, .08), .18, 0.),
    "cabinet": ((.33, .42, .45), .38, .2)}


def pose(prim, p, q=(0., 0., 0., 1.)):
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, p)))
    xf.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))


def mesh(stage, path, vertices, faces, *, smooth=True):
    v, f = np.asarray(vertices, np.float32), np.asarray(faces, np.int32).reshape(-1, 3)
    g = UsdGeom.Mesh.Define(stage, path)
    g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    g.CreateFaceVertexCountsAttr([3]*len(f))
    g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel()))
    g.CreateSubdivisionSchemeAttr("none")
    if smooth:
        import trimesh
        normals = trimesh.Trimesh(v, f, process=False).vertex_normals.astype(np.float32)
        g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
        g.SetNormalsInterpolation("vertex")
    return g


def camera(stage, name, eye, target, focal=35., role="diagnostic"):
    c = UsdGeom.Camera.Define(stage, "/Cameras/"+name)
    c.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target),
                                                  Gf.Vec3d(0., 0., 1.)).GetInverse())
    c.CreateHorizontalApertureAttr(36.)
    c.CreateVerticalApertureAttr(20.25)
    c.CreateFocalLengthAttr(focal)
    c.CreateClippingRangeAttr(Gf.Vec2f(.001, 30.))
    c.GetPrim().SetCustomDataByKey("observation_role", role)


def export(env, output):
    newton, _ = require_engine()
    from whiteboard.usd_scene import bevel_box
    from whiteboard.visual_assets import copy_to
    from whiteboard.appearance import bind_pbr
    from titrate.newton15.panda_visual import bind as bind_robot
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output / "scene.usdc"))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Lab").GetPrim())
    mats = {}
    for name, (rgb, rough, metal) in PALETTE.items():
        m = UsdShade.Material.Define(stage, "/Materials/"+name)
        s = UsdShade.Shader.Define(stage, "/Materials/"+name+"/surface")
        s.CreateIdAttr("UsdPreviewSurface")
        s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
        mats[name] = m
    textures = copy_to(output)
    bind_pbr(stage, mats["wood"], textures["oak_veneer_01"], 1.8, coated=True, roughness=.40)
    bind_pbr(stage, mats["wall"], textures["beige_wall_001"], 3., coated=True, roughness=.80)
    b = env.builder
    for i, q in enumerate(env.state.body_q.numpy()):
        pose(UsdGeom.Xform.Define(stage, f"/Lab/body_{i}"), q[:3], q[3:])
    for i in range(b.shape_count):
        if not b.shape_flags[i] & int(newton.ShapeFlags.VISIBLE):
            continue
        body = b.shape_body[i]
        path = f"/Lab/body_{body}/shape_{i}" if body >= 0 else f"/Lab/static_{i}"
        source, scale, kind = b.shape_source[i], np.asarray(b.shape_scale[i]), b.shape_type[i]
        if source is not None and hasattr(source, "vertices"):
            g = mesh(stage, path, np.asarray(source.vertices)*scale, source.indices)
        elif kind == newton.GeoType.BOX:
            g = bevel_box(stage, path, 2*scale, min(.002, scale.min()*.1))
        elif kind == newton.GeoType.SPHERE:
            g = UsdGeom.Sphere.Define(stage, path)
            g.CreateRadiusAttr(float(scale[0]))
        elif kind in (newton.GeoType.CYLINDER, newton.GeoType.CAPSULE):
            g = (UsdGeom.Cylinder if kind == newton.GeoType.CYLINDER else UsdGeom.Capsule).Define(stage, path)
            g.CreateRadiusAttr(float(scale[0]))
            g.CreateHeightAttr(float(2*scale[1]))
            g.CreateAxisAttr("Z")
        else:
            raise ValueError(f"Unsupported visible geometry: {kind}")
        local = np.asarray(b.shape_transform[i])
        pose(g, local[:3], local[3:])
        material = env.materials.get(i, "robot_white")
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats[material])
        if b.shape_label[i] == "pen_shell":
            vertices = np.asarray(source.vertices)
            faces = np.asarray(source.indices).reshape(-1, 3)
            centers = vertices[faces].mean(axis=1)
            # Surface finish only: the pen geometry and contact mesh are intact.
            for finish, mask in (("aluminum", centers[:, 1] < .004),):
                subset = UsdGeom.Subset.CreateGeomSubset(g, finish, UsdGeom.Tokens.face,
                    np.flatnonzero(mask).tolist(), "materialBind")
                UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(mats[finish])
        if i not in env.materials and source is not None and hasattr(source, "vertices"):
            if not bind_robot(stage, g, np.asarray(source.vertices)*scale, source.indices, mats):
                raise RuntimeError(f"Source Panda appearance not mapped: {b.shape_label[i]}")
    for i, position in enumerate(((0., 0., 2.594), (-.9, .3, 2.294))):
        light = UsdLux.RectLight.Define(stage, f"/Lighting/area_{i}")
        light.CreateWidthAttr(.6)
        light.CreateHeightAttr(1.2)
        light.CreateIntensityAttr(1700.)
        light.CreateColorAttr(Gf.Vec3f(1., .96, .90))
        pose(light, position)
    dome = UsdLux.DomeLight.Define(stage, "/Lighting/daylight")
    dome.CreateTextureFileAttr(Sdf.AssetPath(str(textures["kiara_5_noon"]["hdri"])))
    dome.CreateIntensityAttr(100.)
    camera(stage, "room", (1.50, -1.80, 1.75), (0., 0., .95), 32.)
    camera(stage, "bench", (.60, -.68, 1.19), (-.015, 0., .755), 50.)
    camera(stage, "closeup", (-.012, -.14, .86), (-.06, 0., .755), 55.)
    camera(stage, "gripper_left", (.27, -.19, 1.47), (0., .13, 1.30), 45.)
    # A mounted external RGB sensor is added with the studio scene; this is
    # a real-world-available camera type, not fictitious onboard Panda hardware.
    from .scene import CAMERA_EYE, CAMERA_TARGET
    camera(stage, "bench_rgb", CAMERA_EYE, CAMERA_TARGET,
           18./np.tan(np.radians(69./2)), role="policy")
    stage.GetRootLayer().Save()
    return output / "scene.usdc"


def open_renderer(env, output, *, width=1280, height=720, samples=8, interpreter=None):
    # Reuse the existing tested render-only body-pose protocol. It contains no
    # dynamics timeline; no liquid payload or titration mechanics is imported.
    from titrate.newton15.renderer import Renderer
    from .runtime import CACHE
    import json
    if interpreter is None and (CACHE/"runtime.json").is_file():
        interpreter = json.loads((CACHE/"runtime.json").read_text(encoding="utf-8"))["render_interpreter"]
    return Renderer(export(env, output), width=width, height=height, samples=samples,
                    interpreter=interpreter, backend="rtx")
