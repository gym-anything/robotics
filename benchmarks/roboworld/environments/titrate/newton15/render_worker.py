"""Persistent Isaac RTX worker. Never runs a dynamics timeline."""

import argparse
import base64
import io
import os
from pathlib import Path
import socket
import time
import traceback
from whiteboard.ipc import send, receive


def main():
    from gym_anything_robotics.render_lifecycle import watch_parent
    watch_parent()
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--samples", type=int, required=True)
    a = p.parse_args()
    connection = socket.create_connection(("127.0.0.1", int(os.environ["TITRATE_RENDER_PORT"])))
    send(connection, {"secret": os.environ.pop("TITRATE_RENDER_SECRET")})
    from .runtime import configure

    configure()
    # Each live renderer owns its Kit settings/log. Sharing one log between
    # independent processes created interleaved output and sparse NUL regions.
    cache = Path(os.environ.pop("TITRATE_RENDER_SESSION"))/"kit"
    cache.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "PathTracing", "width": a.width, "height": a.height,
        "samples_per_pixel_per_frame": a.samples, "max_bounces": 12,
        "multi_gpu": False, "active_gpu": 0,
        "extra_args": ["--portable", f"--portable-root={cache}", "--/app/enableCrashReporter=false",
                       # As in laundry's worker: resolve materials before readback (Isaac 6.0.1 otherwise returns
                       # empty frames while they load). Render-only.
                       "--/rtx/materialDb/syncLoads=true", "--/rtx/hydra/materialSyncLoads=true",
                       "--/rtx-transient/resourcemanager/texturestreaming/async=false",
                       "--/rtx-transient/resourcemanager/enableTextureStreaming=false",
                       f"--/log/file={cache/'worker.log'}"]})
    # Isaac Sim 6.0.1 renders no frames for a stage passed as SimulationApp(open_usd=...) (empty readback on
    # driver 610.57.04); the same stage opened after start-up renders. Render-only; physics unchanged.
    import omni.usd
    if not omni.usd.get_context().open_stage(a.scene):
        raise RuntimeError(f"Could not open render scene {a.scene}")
    # Isaac 6.0.1 path tracing then stays black until the stage is re-synced (as with lockkey's scene);
    # one off/on of the top-level prims before the first frame does it. Render-only.
    _roots = list(omni.usd.get_context().get_stage().GetPseudoRoot().GetChildren())
    for _prim in _roots:
        _prim.SetActive(False)
    app.update()
    for _prim in _roots:
        _prim.SetActive(True)
    try:
        import carb
        import numpy as np
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdRender, Vt
        from whiteboard.render_readback import capture_rgb

        settings = carb.settings.get_settings()
        exposure = {"iso": 400., "fStop": 2.8, "time": 1/120.}
        for key, value in {"/rtx/post/histogram/enabled": False,
                            "/rtx/pathtracing/spp": a.samples, "/rtx/pathtracing/totalSpp": a.samples,
                            "/rtx/pathtracing/optixDenoiser/enabled": True,
                            "/rtx/post/tonemap/op": 6, "/rtx/post/tonemap/filmIso": exposure["iso"],
                            "/rtx/post/tonemap/fNumber": exposure["fStop"],
                            "/rtx/post/tonemap/exposureTime": exposure["time"]}.items():
            settings.set(key, value)
        for _ in range(12):
            app.update()
        omni.timeline.get_timeline_interface().pause()
        stage = omni.usd.get_context().get_stage()
        if any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.IsA(UsdPhysics.Scene) for p in stage.Traverse()):
            raise RuntimeError("Render export must contain no dynamics scene")
        # Isaac 6.0.1 (Kit 110, USD 25.11) deprecates the tonemap ISO/f-number/exposure-time settings
        # above in favour of each camera's exposure:* attributes. Only the camera a render product is
        # created on gets the settings; a product retargeted to another camera (below) uses that
        # camera's own attributes, whose defaults (ISO 100, f/1, 1 s) are ~7.7 EV brighter, so
        # tighten_codex's second policy camera (overview_rgb) comes out white. Author the
        # same exposure on every camera without its own; with Kit's responsivity the first camera's
        # image is unchanged bit for bit. Isaac 5.1 has no such attributes and ignores them. Render-only.
        exposure["responsivity"] = settings.get("/rtx/post/tonemap/responsivity") or 1.
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                for name, value in exposure.items():
                    attr = prim.CreateAttribute("exposure:"+name, Sdf.ValueTypeNames.Float)
                    if not attr.HasAuthoredValue():
                        attr.Set(float(value))
        rep.orchestrator.set_capture_on_play(False)
        product, annotator = None, None
        send(connection, {"phase": "ready"})
        while True:
            request = receive(connection)
            if request["op"] == "close":
                break
            try:
                timings = {}
                started = time.perf_counter()
                if request["op"] != "observe":
                    raise ValueError("Unknown render operation")
                for index, q in enumerate(request.get("body_poses", [])):
                    prim = stage.GetPrimAtPath(f"/Lab/body_{index}")
                    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*q[:3]))
                    prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(q[6], Gf.Vec3f(*q[3:6])))
                timings['body_update_s'] = time.perf_counter()-started
                if "control_cable_indicators" in request:
                    colours={"power":(.06,.65,.11),"sensor":(.90,.42,.01),
                             "pass_":(.06,.65,.11),"fail":(.70,.012,.006)}
                    values=request["control_cable_indicators"]
                    if set(values)!=set(colours) or any(type(v) is not bool for v in values.values()):
                        raise ValueError("Invalid physical indicator states")
                    for signal,on in values.items():
                        shader=stage.GetPrimAtPath("/Materials/indicator_"+signal+"/surface")
                        if not shader:raise ValueError("Indicator missing from exported scene")
                        rgb=colours[signal]
                        shader.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*(v*(1. if on else .15) for v in rgb)))
                        shader.GetAttribute("inputs:emissiveColor").Set(Gf.Vec3f(*(v*(6. if on else 0.) for v in rgb)))
                for name, on in request.get("indicators", {}).items():
                    # Generic exported lamp: its material carries its lit colour.
                    material = stage.GetPrimAtPath("/Materials/"+name)
                    shader = stage.GetPrimAtPath("/Materials/"+name+"/surface")
                    rgb = material.GetCustomDataByKey("indicator_rgb") if material else None
                    if not shader or rgb is None or type(on) is not bool:
                        raise ValueError("Indicator missing from exported scene")
                    glow = float(material.GetCustomDataByKey("indicator_emission") or 6.)
                    shader.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*(v*(1. if on else .15) for v in rgb)))
                    shader.GetAttribute("inputs:emissiveColor").Set(Gf.Vec3f(*(v*(glow if on else 0.) for v in rgb)))
                started = time.perf_counter()
                if "liquid" in request:
                    from .render_payload import mesh_arrays
                    liquid = UsdGeom.Mesh(stage.GetPrimAtPath("/Lab/liquid"))
                    points, triangles, normals, pink = mesh_arrays(request['liquid'])
                    liquid.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
                    liquid.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, np.int32)))
                    liquid.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(triangles.ravel()))
                    liquid.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
                    liquid.SetNormalsInterpolation("vertex")
                    from .liquid_optics import set_primvar
                    set_primvar(liquid, pink)
                timings['liquid_update_s'] = time.perf_counter()-started
                images, retries, product_recoveries = {}, {}, {}
                for name in request["cameras"]:
                    started = time.perf_counter()
                    path = "/Cameras/"+name
                    entry = stage.GetPrimAtPath(path)
                    mounted = entry.GetCustomDataByKey("mounted_camera_path") if entry else None
                    if mounted:
                        if not mounted.startswith("/Lab/body_"):raise ValueError("Invalid mounted camera path")
                        path = mounted
                    camera = UsdGeom.Camera(stage.GetPrimAtPath(path))
                    if not camera:
                        raise ValueError("Unknown camera")
                    camera.GetVerticalApertureAttr().Set(camera.GetHorizontalApertureAttr().Get()*a.height/a.width)
                    if product is None:
                        product = rep.create.render_product(path, (a.width, a.height))
                        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                        annotator.attach([product])
                    else:
                        UsdRender.Product(stage.GetPrimAtPath(product.path)).GetCameraRel().SetTargets([path])
                    render_steps = []
                    def render_only():
                        began = time.perf_counter()
                        rep.orchestrator.step(rt_subframes=2, delta_time=0.)
                        render_steps.append(time.perf_counter()-began)
                    # Replicator step waits for rendering and already requests
                    # two render subframes at the unchanged 32-sample setting.
                    # A second complete capture is not a second physics step,
                    # but needlessly retraces another image on dynamic meshes.
                    product_recoveries[name] = 0
                    try:
                        pixels, retries[name] = capture_rgb(render_only, annotator.get_data,
                            a.width, a.height, flush_attempts=1)
                    except RuntimeError:
                        # The same recorded state can render in a fresh product
                        # after a product gets stuck returning all-black data.
                        # Rebind only the renderer; never reset/step physics.
                        annotator.detach()
                        product.destroy()
                        product = rep.create.render_product(path, (a.width, a.height))
                        annotator = rep.AnnotatorRegistry.get_annotator('rgb')
                        annotator.attach([product])
                        product_recoveries[name] = 1
                        pixels, invalid = capture_rgb(render_only, annotator.get_data,
                            a.width, a.height, flush_attempts=1)
                        retries[name] = 8+invalid
                    timings[name+'_render_steps_s'] = render_steps
                    timings[name+'_capture_s'] = time.perf_counter()-started
                    started = time.perf_counter()
                    out = io.BytesIO()
                    Image.fromarray(pixels).save(out, format="PNG")
                    images[name] = base64.b64encode(out.getvalue()).decode()
                    timings[name+'_encode_s'] = time.perf_counter()-started
                send(connection, {"ok": True, "sequence": request["sequence"], "tick": request["tick"],
                                  "physics_steps": 0, "images": images, "readback_retries": retries,
                                  'render_product_recoveries': product_recoveries,
                                  "stage_times_s": timings,
                                  'renderer_settings': {key: settings.get(key) for key in (
                                      '/rtx/rendermode', '/rtx/pathtracing/spp', '/rtx/pathtracing/totalSpp',
                                      '/omni/replicator/RTSubframes', '/rtx/hydra/TBNFrameMode',
                                      '/rtx-transient/resourcemanager/enableTextureStreaming')}})
            except Exception as exc:
                traceback.print_exc()
                send(connection, {"ok": False, "error": str(exc)})
        if product is not None:
            annotator.detach()
            product.destroy()
    finally:
        connection.close()
        app.close()


if __name__ == "__main__":
    main()
