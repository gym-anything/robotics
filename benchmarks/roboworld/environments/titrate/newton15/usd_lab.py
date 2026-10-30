"""Render-only laboratory export from the Newton collision/visual model."""

from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
from scipy.spatial.transform import Rotation
from .runtime import require_newton

PALETTE = {
    "epoxy": ((.10, .13, .14), .27, 0.), "floor": ((.47, .50, .49), .7, 0.),
    "wall": ((.80, .82, .79), .85, 0.), "cabinet": ((.72, .76, .75), .36, 0.),
    "steel": ((.58, .62, .65), .21, 1.), "iron": ((.055, .075, .080), .46, .5),
    "glass": ((.98, .99, 1.), .015, 0.), "water": ((1., 1., 1.), .0, 0.),
    "ptfe": ((.93, .94, .90), .35, 0.), "rubber": ((.020, .024, .027), .8, 0.),
    "instrument_blue": ((.015, .10, .29), .29, 0.), "ceramic": ((.92, .93, .90), .18, 0.),
    "polypropylene": ((.77, .81, .79), .4, 0.), "robot_white": ((.81, .83, .82), .28, 0.),
    "robot_dark": ((.027, .031, .035), .41, .1), "ink": ((.014, .017, .020), .6, 0.),
    "robot_red": ((.55, .018, .012), .3, 0.),
    "skirting": ((.23, .28, .28), .6, 0.),
}


def pose(prim, position, quat=(0., 0., 0., 1.)):
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
    xf.AddOrientOp().Set(Gf.Quatf(float(quat[3]), Gf.Vec3f(*map(float, quat[:3]))))


def mesh(stage, path, vertices, faces, smooth=False):
    m = UsdGeom.Mesh.Define(stage, path)
    v, f = np.asarray(vertices, np.float32), np.asarray(faces, np.int32)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    m.CreateFaceVertexCountsAttr([3]*len(f))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.flatten()))
    m.CreateSubdivisionSchemeAttr("none")
    if smooth and len(f):
        import trimesh

        n = trimesh.Trimesh(v, f, process=False).vertex_normals
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n.astype(np.float32)))
        m.SetNormalsInterpolation("vertex")
    return m


def materials(stage):
    result = {}
    for name, (color, roughness, metallic) in PALETTE.items():
        material = UsdShade.Material.Define(stage, f"/Materials/{name}")
        s = UsdShade.Shader.Define(stage, f"/Materials/{name}/surface")
        s.CreateIdAttr("UsdPreviewSurface")
        for field, val, kind in (("diffuseColor", Gf.Vec3f(*color), Sdf.ValueTypeNames.Color3f),
                                  ("roughness", roughness, Sdf.ValueTypeNames.Float),
                                  ("metallic", metallic, Sdf.ValueTypeNames.Float)):
            s.CreateInput(field, kind).Set(val)
        s.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
        if name in ("glass", "water"):
            mdl = UsdShade.Shader.Define(stage, f"/Materials/{name}/rtx_surface")
            mdl.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
            mdl.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
            mdl.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
            mdl.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            mdl.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.333 if name == "water" else 1.474)
            mdl.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
            mdl.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            material.CreateSurfaceOutput("mdl").ConnectToSource(mdl.ConnectableAPI(), "out")
        result[name] = material
    from .liquid_optics import bind
    bind(stage, result["water"])
    return result


def camera(stage, name, eye, target, focal=36, role="teacher"):
    c = UsdGeom.Camera.Define(stage, "/Cameras/"+name)
    c.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
    c.CreateFocalLengthAttr(focal)
    c.CreateHorizontalApertureAttr(36.)
    c.CreateVerticalApertureAttr(20.25)
    c.CreateClippingRangeAttr(Gf.Vec2f(.001, 30.))
    c.GetPrim().SetCustomDataByKey("observation_role", role)


def export(lab, model, state, output, liquid_surface=None):
    newton, _ = require_newton()
    from whiteboard.usd_scene import bevel_box

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output/"laboratory.usdc"))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Lab").GetPrim())
    mats, q, b = materials(stage), state.body_q.numpy(), lab.builder
    from .appearance import apply_room_finishes

    stage.GetRootLayer().customLayerData = {"appearance_assets": str(apply_room_finishes(stage, mats, output))}
    for index, body in enumerate(q):
        prim = UsdGeom.Xform.Define(stage, f"/Lab/body_{index}")
        pose(prim, body[:3], body[3:])
    for i in range(b.shape_count):
        if not b.shape_flags[i] & int(newton.ShapeFlags.VISIBLE):
            continue
        body = b.shape_body[i]
        path = f"/Lab/body_{body}/shape_{i}" if body >= 0 else f"/Lab/static_{i}"
        source, scale, kind = b.shape_source[i], np.asarray(b.shape_scale[i]), b.shape_type[i]
        if source is not None and hasattr(source, "vertices"):
            g = mesh(stage, path, np.asarray(source.vertices)*scale, np.asarray(source.indices).reshape(-1, 3), smooth=True)
        elif kind == newton.GeoType.BOX:
            g = bevel_box(stage, path, 2*scale, min(.002, scale.min()*.15))
        elif kind in (newton.GeoType.CYLINDER, newton.GeoType.CAPSULE):
            g = (UsdGeom.Capsule if kind == newton.GeoType.CAPSULE else UsdGeom.Cylinder).Define(stage, path)
            g.CreateRadiusAttr(float(scale[0]))
            g.CreateHeightAttr(float(2*scale[1]))
            g.CreateAxisAttr("Z")
        elif kind == newton.GeoType.SPHERE:
            g = UsdGeom.Sphere.Define(stage, path)
            g.CreateRadiusAttr(float(scale[0]))
        else:
            raise ValueError(f"Unsupported visible geometry {kind}")
        local = np.asarray(b.shape_transform[i])
        pose(g, local[:3], local[3:])
        material = lab.materials.get(i)
        if material is None:
            # URDF DAE material groups are retained on each Newton Mesh. Using
            # builder.shape_color loses the original black joints and markings.
            color = getattr(source, "color", None)
            if color is None:
                color = b.shape_color[i]
            color = np.asarray(color)[:3]
            material = "robot_white" if color.mean() > .4 else "robot_dark"
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats[material])
        if i not in lab.materials and source is not None and hasattr(source, "vertices"):
            from .panda_visual import bind

            if not bind(stage, g, np.asarray(source.vertices)*scale, source.indices, mats):
                raise ValueError(f"Could not match source Panda visual materials for shape {i}")
    surface = liquid_surface
    water = mesh(stage, "/Lab/liquid", np.empty((0, 3)) if surface is None else surface.vertices,
                  np.empty((0, 3), dtype=int) if surface is None else surface.triangles, smooth=True)
    UsdShade.MaterialBindingAPI.Apply(water.GetPrim()).Bind(mats["water"])
    from .liquid_optics import set_primvar
    set_primvar(water, None if surface is None else surface.pink_molar)
    if surface is not None and surface.pink_volume is not None:
        from .volume_optics import install
        install(stage, surface.pink_volume, output/'optics')
    elif surface is not None and surface.pink_molar is not None:
        raise ValueError('Cannot export boundary-only chemistry without interior optical field')
    from .markings import add_markings

    add_markings(stage, lab, mats, output)
    for index, xy in enumerate(((-.6, -.1), (.6, .4))):
        light = UsdLux.RectLight.Define(stage, f"/Lighting/ceiling_{index}")
        light.CreateWidthAttr(.3)
        light.CreateHeightAttr(1.2)
        light.CreateIntensityAttr(3200.)
        light.CreateColorAttr(Gf.Vec3f(1., .96, .90))
        pose(light, (*xy, 2.649))
    dome = UsdLux.DomeLight.Define(stage, "/Lighting/daylight")
    from .appearance import daylight_asset
    dome.CreateTextureFileAttr(Sdf.AssetPath(daylight_asset(output)))
    dome.CreateIntensityAttr(120.)
    dome.CreateColorAttr(Gf.Vec3f(1.))
    x, y, z = lab.config.flask_position
    camera(stage, "laboratory", (1.58, -1.68, 1.83), (0., .25, 1.29), 27.)
    camera(stage, "apparatus", (.65, -.95, 1.56), (x, y, 1.47), 32.)
    camera(stage, "flask", (x+.15, y-.23, z+.13), (x, y, z+.045), 60.)
    camera(stage, "valve", (x+.08, y-.16, lab.config.valve_position[2]+.04), lab.config.valve_position, 70.)
    # Documentary view spans the real outlet and receiving flask. This is
    # separate from the physical policy camera and changes no scene geometry.
    camera(stage, "dispensing", (x+.20, y-.40, z+.22), (x, y, z+.08), 40.)
    # Documentary close-up of assisted robot/tool contact. It is not
    # another robot-mounted sensor and is never included in policy observations.
    camera(stage, "gripper", (.29, -.47, 1.62), (-.06, -.04, 1.425), 50.)
    camera(stage, "pipetting", (.72, -.16, 1.20), (.47, .12, 1.11), 45.)
    # Documentary full-tool view: source, 10 mL mark, physical piston and vent.
    # Separate from the unchanged bench_rgb policy sensor.
    camera(stage, "pipette_full", (.97, -.65, 1.30), (.47, .12, 1.17), 26.)
    camera(stage, "pipette_tip", (.59, -.10, .984), (.47, .12, .934), 75.)
    camera(stage, "pipette_rack", (.97, -.65, 1.12), (.64, -.25, .945), 55.)
    camera(stage, "bench_rgb", lab.config.camera_eye, lab.config.camera_target, 28., role="policy")
    # Physical external camera mount, calibrated extrinsics; wrist camera is
    # added to the robot model separately, not implied to be Panda hardware.
    stage.GetRootLayer().Save()
    return output/"laboratory.usdc"
