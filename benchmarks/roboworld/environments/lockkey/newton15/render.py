"""Render actual body poses; diagnostics never alter physics or policy sensing."""

import json
from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
from pencap.newton15.render import pose, mesh, camera, PALETTE
from .runtime import require_engine, CACHE


def export(env, output, *, cutaway=False):
    newton, _ = require_engine()
    from whiteboard.usd_scene import bevel_box
    from whiteboard.visual_assets import copy_to
    from whiteboard.appearance import bind_pbr
    from titrate.newton15.panda_visual import bind as bind_robot
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output/"scene.usdc"))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Lab").GetPrim())
    palette = dict(PALETTE, brass=((.62, .42, .13), .27, .85),
        nickel=((.60, .61, .56), .22, .95), paint_blue=((.028, .10, .16), .31, .35))
    mats = {}
    for name, (rgb, rough, metal) in palette.items():
        m = UsdShade.Material.Define(stage, "/Materials/"+name)
        s = UsdShade.Shader.Define(stage, "/Materials/"+name+"/surface")
        s.CreateIdAttr("UsdPreviewSurface")
        for field, value, typ in (("diffuseColor", Gf.Vec3f(*rgb), Sdf.ValueTypeNames.Color3f),
                ("roughness", rough, Sdf.ValueTypeNames.Float), ("metallic", metal, Sdf.ValueTypeNames.Float)):
            s.CreateInput(field, typ).Set(value)
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
        mats[name] = m
    textures = copy_to(output)
    for name, texture, roughness in (("wood", "oak_veneer_01", .40), ("wall", "beige_wall_001", .80)):
        bind_pbr(stage, mats[name], textures[texture], 2., coated=True, roughness=roughness)
    b = env.builder
    for i, q in enumerate(env.state.body_q.numpy()):
        pose(UsdGeom.Xform.Define(stage, f"/Lab/body_{i}"), q[:3], q[3:])
    for i in range(b.shape_count):
        if not b.shape_flags[i] & int(newton.ShapeFlags.VISIBLE):
            continue
        label, body = b.shape_label[i], b.shape_body[i]
        if cutaway and (label == "shell" or label.startswith("plug_") or label == "lock_casing"):
            continue
        if label == "shell" and not cutaway and hasattr(env, "materials"):
            continue
        path = f"/Lab/body_{body}/shape_{i}" if body >= 0 else f"/Lab/static_{i}"
        source, scale, kind = b.shape_source[i], np.asarray(b.shape_scale[i]), b.shape_type[i]
        if source is not None and hasattr(source, "vertices"):
            g = mesh(stage, path, np.asarray(source.vertices)*scale, source.indices,
                     smooth=label.startswith(("panda", "keypin", "driver")))
            if label == "lock_casing":
                # Averaging normals from the keyway/corner onto broad planar
                # faces makes this rigid casing appear dented. Keep hard edges.
                vertices = np.asarray(source.vertices)*scale
                triangles = vertices[np.asarray(source.indices).reshape(-1, 3)]
                normals = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
                g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(normals, 3, axis=0).astype(np.float32)))
                g.SetNormalsInterpolation("faceVarying")
        elif kind == newton.GeoType.BOX:
            g = bevel_box(stage, path, 2*scale, min(.001, scale.min()*.08))
        elif kind == newton.GeoType.SPHERE:
            g = UsdGeom.Sphere.Define(stage, path)
            g.CreateRadiusAttr(float(scale[0]))
        elif kind in (newton.GeoType.CYLINDER, newton.GeoType.CAPSULE):
            g = (UsdGeom.Cylinder if kind == newton.GeoType.CYLINDER else UsdGeom.Capsule).Define(stage, path)
            g.CreateRadiusAttr(float(scale[0]))
            g.CreateHeightAttr(float(2*scale[1]))
            g.CreateAxisAttr("Z")
        else:
            raise ValueError(f"Unsupported visible shape {label}: {kind}")
        local = np.asarray(b.shape_transform[i])
        pose(g, local[:3], local[3:])
        material = getattr(env, "materials", {}).get(i, "brass")
        if label.startswith("driver"):
            material = "nickel"
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats[material])
        if label.startswith("panda") and source is not None and hasattr(source, "vertices"):
            if not bind_robot(stage, g, np.asarray(source.vertices)*scale, source.indices, mats):
                raise RuntimeError(f"Panda visual material mapping missing: {label}")
    for i, p in enumerate(((0., 0., 2.574), (-.6, .7, 2.324))):
        light = UsdLux.RectLight.Define(stage, f"/Lighting/light_{i}")
        light.CreateWidthAttr(.6)
        light.CreateHeightAttr(1.2)
        light.CreateIntensityAttr(1700.)
        light.CreateColorAttr(Gf.Vec3f(1., .96, .91))
        pose(light, p)
    dome = UsdLux.DomeLight.Define(stage, "/Lighting/daylight")
    dome.CreateTextureFileAttr(Sdf.AssetPath(str(textures["kiara_5_noon"]["hdri"])))
    dome.CreateIntensityAttr(350.)
    z = env.config.table_height
    camera(stage, "room", (1.70, -1.80, 1.90), (.12, 0., z+.22), 32.)
    camera(stage, "bench", (.78, -.70, z+.62), (.0, 0., z+.13), 52.)
    camera(stage, "lock", (.35, -.22, z+.26), (.10, 0., z+.13), 52.)
    camera(stage, "keys", (.34, -.34, z+.21), (.175, -.16, z+.063), 58.)
    camera(stage, "keyway", (.30, -.025, z+.105), (.114, 0., z+.10), 80.)
    # Observer-only view around the initial Panda grasp frame.
    if hasattr(env, "joints"):
        from .kinematics import Kinematics
        from .robot import HOME
        tcp = Kinematics().forward(HOME[:7], z)[:3, 3]
        camera(stage, "grasp", tcp+np.array([.29, -.34, .14]), tcp+np.array([0., 0., .06]), 38.)
    camera(stage, "mechanism", (.025, -.070, .029), (0., 0., .003), 60.)
    camera(stage, "pin_stacks", (.125, -.070, z+.129), (.10, 0., z+.103), 60.)
    from .scene import CAMERA_EYE, CAMERA_TARGET
    camera(stage, "bench_rgb", CAMERA_EYE, CAMERA_TARGET, 18./np.tan(np.radians(69./2)), role="policy")
    stage.GetRootLayer().Save()
    return output/"scene.usdc"


def open_renderer(env, output, *, width=1280, height=720, samples=8, interpreter=None, cutaway=False):
    from titrate.newton15.renderer import Renderer
    if interpreter is None and (CACHE/"runtime.json").is_file():
        interpreter = json.loads((CACHE/"runtime.json").read_text(encoding="utf-8"))["render_interpreter"]
    return Renderer(export(env, output, cutaway=cutaway), width=width, height=height,
                    samples=samples, interpreter=interpreter, backend="rtx")
