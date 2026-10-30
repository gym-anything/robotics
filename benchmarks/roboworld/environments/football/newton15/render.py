"""PBR training ground exported from physical geometry and current state."""

from pathlib import Path
import json
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
from scipy.spatial.transform import Rotation
from pencap.newton15.render import pose, mesh, camera
from football.seam_geometry import PANELS


def material(stage, name, rgb, roughness=0.5, metallic=0.0):
    m = UsdShade.Material.Define(stage, "/Materials/" + name)
    s = UsdShade.Shader.Define(stage, str(m.GetPath()) + "/surface")
    s.CreateIdAttr("UsdPreviewSurface")
    s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
    return m


def bind(shape, mat):
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)


def box(stage, name, position, size, mat):
    from whiteboard.usd_scene import bevel_box

    g = bevel_box(stage, name, size, min(0.001, min(size) * 0.1))
    pose(g, position)
    bind(g, mat)
    return g


def curves(stage, name, points, counts, diameter, mat):
    g = UsdGeom.BasisCurves.Define(stage, name)
    g.CreateTypeAttr("linear")
    g.CreateWrapAttr("nonperiodic")
    g.CreateCurveVertexCountsAttr(counts)
    g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, np.float32)))
    g.CreateWidthsAttr([diameter])
    g.SetWidthsInterpolation("constant")
    bind(g, mat)
    return g


def ball_panels(stage, parent, radius, mats):
    # Spherical tessellation, not a faceted polyhedron; canonical seams are also
    # used by the aerodynamic orientation calculation.
    UsdGeom.Xform.Define(stage, parent).AddTransformOp().Set(Gf.Matrix4d(1))
    for kind in ("pentagon", "hexagon"):
        vertices, faces = [], []
        for panel in (p for p in PANELS if p.kind == kind):
            c = np.asarray(panel.center)
            ring = [0.982 * np.asarray(v) + 0.018 * c for v in panel.vertices]
            ring = [v / np.linalg.norm(v) for v in ring]
            for k, a in enumerate(ring):
                b = ring[(k + 1) % len(ring)]
                lattice, n = {}, 6
                for i in range(n + 1):
                    for j in range(n + 1 - i):
                        v = (1 - (i + j) / n) * c + i / n * a + j / n * b
                        lattice[i, j] = len(vertices)
                        vertices.append(radius * v / np.linalg.norm(v))
                for i in range(n):
                    for j in range(n - i):
                        faces.append(
                            (lattice[i, j], lattice[i + 1, j], lattice[i, j + 1])
                        )
                        if i + j < n - 1:
                            faces.append(
                                (
                                    lattice[i + 1, j],
                                    lattice[i + 1, j + 1],
                                    lattice[i, j + 1],
                                )
                            )
        g = mesh(stage, parent + "/" + kind, vertices, faces)
        g.CreateNormalsAttr(
            Vt.Vec3fArray.FromNumpy(np.asarray(vertices, np.float32) / radius)
        )
        bind(g, mats["dark"] if kind == "pentagon" else mats["white"])
    core = UsdGeom.Sphere.Define(stage, parent + "/seams")
    core.CreateRadiusAttr(radius - 0.0008)
    bind(core, mats["dark"])


def optical_pose(world):
    q = world.poses[world.body_ids["torso_link"]]
    r = Rotation.from_quat(q[3:])
    mount = r * Rotation.from_euler("y", 0.8307767239493009)
    eye = q[:3] + r.apply((0.0576235, 0.01753, 0.42987))
    return [
        eye.tolist(),
        (eye + mount.apply((1, 0, 0))).tolist(),
        mount.apply((0, 0, 1)).tolist(),
    ]


TASK_MATERIALS = {
    "mannequin": ((0.86, 0.68, 0.04), 0.55, 0.0),
    "sheet": ((0.07, 0.20, 0.60), 0.80, 0.0),
    "edge": ((0.93, 0.91, 0.84), 0.70, 0.0),
    "marker": ((0.95, 0.40, 0.04), 0.60, 0.0),
    "machine": ((0.11, 0.15, 0.22), 0.45, 0.30),
}


def floats(values):
    return tuple(float(v) for v in values)


def task_scene(stage, world, mats):
    """A task's props (static shapes, exported above in white) get their own material; its render-only markers
    (variants.py; no collider, the ball rolls over flat training discs) are drawn. Plain Python floats only."""
    used = {p["material"] for p in world.layout["props"]} | {m["material"] for m in world.layout["markers"]}
    for name in sorted(used - set(mats)):
        rgb, roughness, metallic = TASK_MATERIALS[name]
        mats[name] = material(stage, name, rgb, roughness, metallic)
    b = world.builder
    for i in range(b.shape_count):
        name = world.prop_materials.get(b.shape_label[i])
        if name is not None:
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"/Lab/static_{i}")).Bind(mats[name])
    for marker in world.layout["markers"]:
        path = "/Lab/task_markers/" + marker["name"]
        if marker["shape"] == "disc":
            g = UsdGeom.Cylinder.Define(stage, path)
            g.CreateRadiusAttr(float(marker["radius_m"]))
            g.CreateHeightAttr(float(marker["height_m"]))
            pose(g, floats(marker["position_m"]))
            bind(g, mats[marker["material"]])
        else:
            box(stage, path, floats(marker["position_m"]), floats(marker["size_m"]), mats[marker["material"]])


def export(world, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output / "scene.usdc"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Lab").GetPrim())
    mats = {
        name: material(stage, name, color, rough, metal)
        for name, color, rough, metal in (
            ("white", (0.80, 0.82, 0.79), 0.34, 0.0),
            ("dark", (0.018, 0.023, 0.026), 0.43, 0.0),
            ("cord", (0.61, 0.64, 0.59), 0.85, 0.0),
            ("steel", (0.22, 0.25, 0.24), 0.32, 0.85),
            ("grass", (0.095, 0.19, 0.033), 0.85, 0.0),
            ("line", (0.74, 0.78, 0.65), 0.92, 0.0),
            ("path", (0.30, 0.30, 0.28), 0.95, 0.0),
            ("bench", (0.13, 0.22, 0.23), 0.50, 0.35),
        )
    }
    from whiteboard.appearance import mdl_surface

    shader = mdl_surface(stage, mats["grass"], "OmniPBR")
    from .visual_assets import grass

    maps = grass()
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(False)
    # bevel_box authors two UV cycles/metre; 0.25 therefore gives the asset's
    # documented two-metre tile instead of stretching one tile over the pitch.
    shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(
        Gf.Vec2f(0.25, 0.25)
    )
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(1)
    )
    for name, key in (
        ("diffuse_texture", "diff"),
        ("normalmap_texture", "nor_gl"),
        ("reflectionroughness_texture", "rough"),
    ):
        shader.CreateInput(name, Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(maps[key].resolve()))
        )
    shader.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(0.6)
    b, n = world.builder, world.n
    for i, q in enumerate(world.poses):
        pose(UsdGeom.Xform.Define(stage, f"/Lab/body_{i}"), q[:3], q[3:])
    for i in range(b.shape_count):
        if not b.shape_flags[i] & int(n.ShapeFlags.VISIBLE):
            continue
        body, label = b.shape_body[i], b.shape_label[i]
        if body == world.ball_id:
            continue
        path = f"/Lab/body_{body}/shape_{i}" if body >= 0 else f"/Lab/static_{i}"
        source, scale, kind = (
            b.shape_source[i],
            np.asarray(b.shape_scale[i]),
            b.shape_type[i],
        )
        if source is not None and hasattr(source, "vertices"):
            g = mesh(stage, path, np.asarray(source.vertices) * scale, source.indices)
        elif kind == n.GeoType.BOX:
            from whiteboard.usd_scene import bevel_box

            g = bevel_box(stage, path, 2 * scale, min(0.001, float(scale.min()) * 0.1))
        elif kind in (n.GeoType.CYLINDER, n.GeoType.CAPSULE):
            g = (
                UsdGeom.Cylinder if kind == n.GeoType.CYLINDER else UsdGeom.Capsule
            ).Define(stage, path)
            g.CreateRadiusAttr(float(scale[0]))
            g.CreateHeightAttr(float(2 * scale[1]))
            g.CreateAxisAttr("Z")
        elif kind == n.GeoType.SPHERE:
            g = UsdGeom.Sphere.Define(stage, path)
            g.CreateRadiusAttr(float(scale[0]))
        else:
            raise ValueError(f"Unsupported visual shape {label}: {kind}")
        local = np.asarray(b.shape_transform[i])
        pose(g, local[:3], local[3:])
        # Xform order must be translation, orientation, scale (not scaled translation).
        ops = UsdGeom.Xformable(g).GetOrderedXformOps()
        UsdGeom.Xformable(g).SetXformOpOrder(
            sorted(
                ops,
                key=lambda op: {
                    "xformOp:translate": 0,
                    "xformOp:orient": 1,
                    "xformOp:scale": 2,
                }.get(op.GetOpName(), 3),
            )
        )
        mat = "grass" if label == "pitch" else "white"
        if body >= 0 and np.mean(b.shape_color[i]) <= 0.4:
            mat = "dark"
        bind(g, mats[mat])
    ball_panels(stage, f"/Lab/body_{world.ball_id}/surface", world.ball.radius_m, mats)
    for name, net in world.nets.items():
        points = net.position[np.stack([net.edge_a, net.edge_b], axis=1)].reshape(-1, 3)
        curves(
            stage,
            "/Lab/net_" + name,
            points,
            [2] * len(net.edges),
            net.prior.cord_diameter_m,
            mats["cord"],
        )
    # Markings on this 34 x 24 m training sector. Whole regulation pitch is not claimed.
    for name, pos, size in (
        ("goal_line", (0, 0, 0.0008), (0.12, 24, 0.001)),
        ("area_line", (-5.5, 0, 0.0008), (0.12, 18.32, 0.001)),
        ("area_left", (-2.75, -9.16, 0.0008), (5.5, 0.12, 0.001)),
        ("area_right", (-2.75, 9.16, 0.0008), (5.5, 0.12, 0.001)),
    ):
        box(stage, "/Lab/markings/" + name, pos, size, mats["line"])
    disk = UsdGeom.Cylinder.Define(stage, "/Lab/markings/penalty_mark")
    disk.CreateRadiusAttr(0.11)
    disk.CreateHeightAttr(0.001)
    pose(disk, (-11, 0, 0.0006))
    bind(disk, mats["line"])
    # Catalogue tasks with props or markers only (variants.py); goal has neither and exports exactly as before.
    if world.layout["props"] or world.layout["markers"]:
        task_scene(stage, world, mats)
    # Context outside the active training sector: paved access and perimeter fence.
    box(stage, "/Context/surround", (-15, 0, -0.15), (90, 80, 0.10), mats["grass"])
    box(stage, "/Context/access", (-15, -13, -0.025), (42, 1.6, 0.05), mats["path"])
    segments = []
    for y in (-14.0, 14.0):
        for x in np.arange(-35, 6, 2.5):
            segments.extend(((x, y, 0), (x, y, 2.3)))
        for z in (0.1, 1.1, 2.2):
            segments.extend(((-35, y, z), (5, y, z)))
    curves(
        stage,
        "/Context/fence_posts",
        segments,
        [2] * (len(segments) // 2),
        0.04,
        mats["steel"],
    )
    wire = []
    for y in (-14.0, 14.0):
        for x in np.arange(-35, 5, 0.15):
            wire.extend(((x, y, 0.05), (x, y, 2.2)))
        for z in np.arange(0.05, 2.21, 0.15):
            wire.extend(((-35, y, z), (5, y, z)))
    curves(
        stage, "/Context/fence_mesh", wire, [2] * (len(wire) // 2), 0.003, mats["steel"]
    )
    for k, x in enumerate((-17.0, -20.0, -23.0)):
        box(
            stage,
            f"/Context/bench_{k}/seat",
            (x, -13, 0.45),
            (1.8, 0.42, 0.07),
            mats["bench"],
        )
        box(
            stage,
            f"/Context/bench_{k}/back",
            (x, -13.18, 0.78),
            (1.8, 0.055, 0.45),
            mats["bench"],
        )
        for j, dx in enumerate((-0.65, 0.65)):
            box(
                stage,
                f"/Context/bench_{k}/leg_{j}",
                (x + dx, -13, 0.22),
                (0.05, 0.32, 0.44),
                mats["steel"],
            )
    from .visual_assets import sky, grass_blades

    grass_blades(stage, mats["grass"])
    dome = UsdLux.DomeLight.Define(stage, "/Lighting/daylight")
    dome.CreateTextureFileAttr(Sdf.AssetPath(str(sky())))
    dome.CreateTextureFormatAttr("latlong")
    dome.CreateIntensityAttr(500.0)
    dome.AddRotateZOp().Set(115.0)
    for name, eye, target, focal in (
        ("overview", (-17, -12, 6.2), (-6, 0, 0.55), 30.0),
        ("sideline", (-7, -14, 3.6), (-5, 0, 0.75), 30.0),
        ("contact", (-9.5, -1.55, 0.75), (-11, 0, 0.32), 52.0),
        ("goal", (-3.8, -6.0, 2.5), (1.0, 0, 1.0), 32.0),
    ):
        camera(stage, name, eye, target, focal)
        stage.GetPrimAtPath("/Cameras/" + name).GetAttribute("clippingRange").Set(
            Gf.Vec2f(0.01, 150.0)
        )
    if world.robot:
        eye, look, up = optical_pose(world)
        camera(
            stage, "g1_rgb", eye, look, 18 / np.tan(np.radians(69 / 2)), role="policy"
        )
        stage.GetPrimAtPath("/Cameras/g1_rgb").GetAttribute("xformOp:transform").Set(
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(*up))
            .GetInverse()
        )
    stage.GetRootLayer().Save()
    (output / "nets.json").write_text(
        json.dumps(
            {
                name: np.stack([net.edge_a, net.edge_b], axis=1).tolist()
                for name, net in world.nets.items()
            }
        )
    )
    return output / "scene.usdc"
