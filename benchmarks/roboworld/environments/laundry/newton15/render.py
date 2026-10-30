"""Render-only USD export and a persistent authenticated cloth/rigid bridge."""
from gym_anything_robotics.paths import worker_env
from gym_anything_robotics.render_lifecycle import close_worker, watch_worker

import base64
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import uuid
import numpy as np
from whiteboard.ipc import send, receive
from .runtime import ROOT, CACHE


def export(env, folder, *, shading="smooth", pile=None):
    if pile is None:
        pile = env.config.garment == "towel"
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
    from pencap.newton15.render import PALETTE, mesh, pose, camera
    from whiteboard.usd_scene import bevel_box
    from whiteboard.appearance import bind_pbr
    from whiteboard.visual_assets import copy_to
    from titrate.newton15.panda_visual import bind as robot_material
    from ..garment_render_mesh import build_garment_panel_render_topology
    from ..garment_showcase_scene import _mesh_from_panel
    from .appearance import textile
    from .shading import deform
    from .scene import CAMERA_EYE, CAMERA_TARGET
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(folder/"scene.usdc"))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Lab").GetPrim())
    palette = {**PALETTE, "basket": ((.58, .63, .60), .58, 0.)}
    mats = {}
    for name, (rgb, rough, metal) in palette.items():
        m = UsdShade.Material.Define(stage, "/Materials/"+name)
        s = UsdShade.Shader.Define(stage, "/Materials/"+name+"/surface")
        s.CreateIdAttr("UsdPreviewSurface")
        for key, value, typ in (("diffuseColor", Gf.Vec3f(*rgb), Sdf.ValueTypeNames.Color3f),
                                ("roughness", rough, Sdf.ValueTypeNames.Float),
                                ("metallic", metal, Sdf.ValueTypeNames.Float)):
            s.CreateInput(key, typ).Set(value)
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
        mats[name] = m
    textures = copy_to(folder)
    bind_pbr(stage, mats["wood"], textures["oak_veneer_01"], 1.8, coated=True, roughness=.55)
    bind_pbr(stage, mats["wall"], textures["beige_wall_001"], 3., coated=True, roughness=.8)
    from ..laundry_scene import _omnipbr_material
    from ..garment_showcase_scene import ASSETS
    mats["floor"] = _omnipbr_material(stage, "/Materials/concrete_floor",
        (.38, .39, .40), roughness=.86, albedo_texture=ASSETS/"concrete_floor.png",
        project_uvw=True, texture_scale=(.55, .55))
    b, newton = env.builder, env.sim.newton
    for i, q in enumerate(env.state.body_q.numpy()):
        pose(UsdGeom.Xform.Define(stage, f"/Lab/body_{i}"), q[:3], q[3:])
    for i in range(b.shape_count):
        if not b.shape_flags[i] & int(newton.ShapeFlags.VISIBLE):
            continue
        body, kind, scale = b.shape_body[i], b.shape_type[i], np.asarray(b.shape_scale[i])
        path = f"/Lab/body_{body}/shape_{i}" if body >= 0 else f"/Lab/static_{i}"
        source = b.shape_source[i]
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
            raise ValueError(f"Visible shape {b.shape_label[i]} is unsupported: {kind}")
        local = np.asarray(b.shape_transform[i])
        pose(g, local[:3], local[3:])
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats[env.materials.get(i, "robot_white")])
        if i not in env.materials and source is not None and hasattr(source, "vertices"):
            if not robot_material(stage, g, np.asarray(source.vertices)*scale, source.indices, mats):
                raise ValueError(f"Cannot map Panda appearance: {b.shape_label[i]}")
    panels = {}
    for i, panel in enumerate(env.sim.construction.panels):
        top = build_garment_panel_render_topology(env.sim.pattern_mesh, env.sim.topology,
            panel.id, subdivision_factor=3,
            texture_repeats_per_m=16. if panel.material_profile == "terry_cotton_prior" else 10.)
        path = f"/Lab/Garment/panel_{i}"
        mat = textile(stage, f"/Materials/textile_{i}", panel)
        thickness = .0005 if pile and panel.material_profile == "terry_cotton_prior" else panel.unloaded_thickness_m
        garment_mesh = _mesh_from_panel(stage, path, top, env.state.particle_q.numpy(),
            thickness_m=thickness, material=mat)
        _, normals = deform(top, env.state.particle_q.numpy(),
            thickness_m=thickness, shading=shading)
        garment_mesh.GetNormalsAttr().Set(normals.astype(np.float32))
        panels[path] = (top, thickness)
    for i, center in enumerate(((.25, -.15, 2.65), (-1.15, .7, 2.60))):
        light = UsdLux.RectLight.Define(stage, f"/Lighting/area_{i}")
        light.CreateWidthAttr(.7)
        light.CreateHeightAttr(1.2)
        light.CreateIntensityAttr(700.)
        light.CreateColorAttr(Gf.Vec3f(1., .97, .93))
        pose(light, center)
    dome = UsdLux.DomeLight.Define(stage, "/Lighting/daylight")
    dome.CreateTextureFileAttr(Sdf.AssetPath(str(textures["kiara_5_noon"]["hdri"])))
    dome.CreateIntensityAttr(5.)
    camera(stage, "room", (3.0, 2.05, 2.55), (-.20, -.10, .85), 32.)
    camera(stage, "bench", (1.55, -1.20, 1.95), (0., 0., .85), 40.)
    camera(stage, "cloth", (.85, -.55, 1.40), (0., 0., .83), 48.)
    camera(stage, "overhead", (.001, 0., 2.55), (0., 0., .8), 21.)
    camera(stage, "grasp", (.48, -.86, 1.55), (0., -.32, 1.19), 65.)
    camera(stage, "interaction", (.80, -1.45, 1.70), (0., -.25, 1.02), 32.)
    camera(stage, "edge", (1.65, -.95, 1.05), (.76, 0., .72), 40.)
    camera(stage, "edge_wide", (2.10, -1.65, 1.30), (.72, 0., .52), 35.)
    camera(stage, "bimanual", (1.70, -1.25, 1.90), (0., 0., 1.10), 40.)
    camera(stage, "handling_detail", (.95, .72, 1.50), (.08, .22, 1.12), 35.)
    camera(stage, "fabric", (.25, -.22, 1.13), (0., 0., .80), 70.)
    camera(stage, "bench_rgb", CAMERA_EYE, CAMERA_TARGET, 18./np.tan(np.radians(69/2)), role="policy")
    from .details import Details
    details = Details(env, stage)
    from .pile import TerryPile
    textile_pile = TerryPile(env, stage) if pile else None
    stage.GetRootLayer().Save()
    return folder/"scene.usdc", panels, details, textile_pile


class Renderer:
    def __init__(self, env, folder, *, width=1280, height=720, samples=8, interpreter=None, shading="smooth", pile=None):
        if any(type(v) is not int or v < 1 for v in (width, height, samples)):
            raise ValueError("width, height and samples must be positive integers")
        scene, self.panels, self.details, self.pile = export(env, folder, shading=shading, pile=pile)
        self.shading = shading
        selected = interpreter or os.environ.get("LAUNDRY_RENDER_PYTHON")
        # Read the existing machine renderer installation, never alter it.
        settings = ROOT/"lockkey/.cache/runtime.json"
        if selected is None and settings.is_file():
            selected = json.loads(settings.read_text(encoding="utf-8"))["render_interpreter"]
        if not selected or not Path(selected).is_file():
            raise FileNotFoundError("Set LAUNDRY_RENDER_PYTHON to Isaac Sim 5.1's python.bat")
        self.folder = Path(folder)
        self.log = (self.folder/"worker.log").open("wb")
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(240)
        secret = secrets.token_urlsafe(32)
        variables = dict(os.environ, LAUNDRY_RENDER_PORT=str(listener.getsockname()[1]),
                         LAUNDRY_RENDER_SECRET=secret)
        # Only this offscreen worker opts out of the installed OBS Vulkan hook.
        # Its layer manifest declares this switch; do not alter OBS or registry.
        variables.setdefault("DISABLE_VULKAN_OBS_CAPTURE", "1")
        self.process = None
        try:
            self.process = subprocess.Popen([str(selected), "-m", "laundry.newton15.render_worker",
                "--scene", str(scene.resolve()), "--width", str(width), "--height", str(height),
                "--samples", str(samples)], cwd=ROOT, env=worker_env(variables), stdout=self.log,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.connection, _ = listener.accept()
            self.connection.settimeout(240)
            if not secrets.compare_digest(receive(self.connection).get("secret", ""), secret):
                raise PermissionError("Unrecognized render worker")
            if receive(self.connection).get("phase") != "ready":
                raise RuntimeError("Renderer not ready")
        except BaseException:
            close_worker(self, graceful=False)
            raise
        finally:
            listener.close()
        watch_worker(self)
        self.sequence, self.closed, self.lock = 0, False, threading.Lock()

    def observe(self, tick, body_poses, cameras, particles):
        from .shading import deform
        from titrate.newton15.render_payload import encode
        with self.lock:
            if self.closed:
                raise RuntimeError("Renderer closed")
            meshes = {}
            for path, (topology, thickness) in self.panels.items():
                points, normals = deform(topology, particles, thickness_m=thickness, shading=self.shading)
                meshes[path] = {"points": encode(points, "float32"), "normals": encode(normals, "float32")}
            self.sequence += 1
            send(self.connection, dict(op="observe", sequence=self.sequence, tick=tick,
                body_poses=body_poses.tolist(), cameras=list(cameras), meshes=meshes,
                pile={} if self.pile is None else {path:encode(points, "float32") for path,points in self.pile.sample(particles).items()},
                details=self.details.sample(particles)))
            result = receive(self.connection)
            if not result.get("ok") or result.get("sequence") != self.sequence or result.get("tick") != tick or result.get("physics_steps") != 0:
                raise RuntimeError(f"Invalid renderer response: {result}")
            self.last_timings = result.get("timings", {})
            return result

    def close(self):
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True
