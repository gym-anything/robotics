"""Render the received physical cloth/robot state; never run a physics timeline."""
import argparse
import base64
import io
import os
from pathlib import Path
import socket
import traceback
import time
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
    connection = socket.create_connection(("127.0.0.1", int(os.environ["LAUNDRY_RENDER_PORT"])))
    send(connection, {"secret": os.environ.pop("LAUNDRY_RENDER_SECRET")})
    kit = Path(a.scene).parent/"kit"
    kit.mkdir(exist_ok=True)
    from isaacsim import SimulationApp
    # The qualified Windows renderer uses D3D12; Vulkan remains available as
    # an explicit override and is the Linux default. This does not select the
    # Newton physics device (CUDA) or change camera/sample settings.
    api=os.environ.get("LAUNDRY_RENDER_GRAPHICS_API", "d3d12" if os.name == "nt" else "vulkan")
    if api not in ("vulkan", "d3d12") or (api == "d3d12" and os.name != "nt"):
        raise ValueError("Choose vulkan, or d3d12 on Windows")
    app = SimulationApp(dict(headless=True, renderer="PathTracing", width=a.width, height=a.height, samples_per_pixel_per_frame=a.samples,
        max_bounces=12, multi_gpu=False, active_gpu=0,
        extra_args=["--portable", f"--portable-root={kit}", "--/app/enableCrashReporter=false",
                    "--/app/vulkan="+("true" if api == "vulkan" else "false"),
                    "--/rtx/materialDb/syncLoads=true", "--/rtx/hydra/materialSyncLoads=true",
                    "--/rtx-transient/resourcemanager/texturestreaming/async=false",
                    "--/rtx-transient/resourcemanager/enableTextureStreaming=false",
                    f"--/log/file={kit/'worker.log'}"]))
    # Isaac Sim 6.0.1 renders no frames for a stage passed as SimulationApp(open_usd=...) (empty readback on
    # driver 610.57.04); the same stage opened after start-up renders. Render-only; physics unchanged.
    import omni.usd
    if not omni.usd.get_context().open_stage(a.scene):
        raise RuntimeError(f"Could not open render scene {a.scene}")
    try:
        import carb
        import numpy as np
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import UsdGeom, UsdPhysics, UsdRender
        from whiteboard.render_readback import capture_rgb
        from .render_updates import apply_recorded_state
        settings = carb.settings.get_settings()
        for key, value in {"/rtx/post/histogram/enabled": False, "/rtx/pathtracing/spp": a.samples,
            "/rtx/pathtracing/totalSpp": a.samples, "/rtx/pathtracing/optixDenoiser/enabled": True,
            "/rtx/post/tonemap/op": 6, "/rtx/post/tonemap/filmIso": 800.,
            "/rtx/post/tonemap/fNumber": 2.8, "/rtx/post/tonemap/exposureTime": 1/120.}.items():
            settings.set(key, value)
        for _ in range(12):
            app.update()
        omni.timeline.get_timeline_interface().pause()
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                camera = UsdGeom.Camera(prim)
                camera.GetVerticalApertureAttr().Set(camera.GetHorizontalApertureAttr().Get()*a.height/a.width)
        if any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.IsA(UsdPhysics.Scene) for p in stage.Traverse()):
            raise RuntimeError("Renderer must not contain a dynamics scene")
        rep.orchestrator.set_capture_on_play(False)
        product, annotator, active_camera = None, None, None
        send(connection, {"phase": "ready"})
        while True:
            request = receive(connection)
            if request["op"] == "close":
                break
            try:
                if request["op"] != "observe":
                    raise ValueError("Unknown render operation")
                start_update = time.perf_counter()
                apply_recorded_state(stage, request)
                timings = {"usd_update_s":time.perf_counter()-start_update}
                images = {}
                for name in request["cameras"]:
                    camera_path = "/Cameras/"+name
                    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
                    if not camera:
                        raise ValueError("Unknown camera")
                    if product is None:
                        product = rep.create.render_product(camera_path, (a.width, a.height))
                        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                        annotator.attach([product])
                        # Resolve materials before the first returned frame.
                        # This is render-only startup at the SAME recorded tick.
                        rep.orchestrator.step(rt_subframes=8, delta_time=0.)
                    elif name != active_camera:
                        UsdRender.Product(stage.GetPrimAtPath(product.path)).GetCameraRel().SetTargets([camera_path])
                    active_camera = name
                    def render_only():
                        rep.orchestrator.step(rt_subframes=2, delta_time=0.)
                    start_render = time.perf_counter()
                    try:
                        pixels, _ = capture_rgb(render_only, annotator.get_data, a.width, a.height, flush_attempts=1)
                    except RuntimeError:
                        # Recreate only the render product at the same recorded
                        # tick; do not advance, reset or restore the simulator.
                        annotator.detach()
                        product.destroy()
                        product = rep.create.render_product(camera_path, (a.width, a.height))
                        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                        annotator.attach([product])
                        pixels, _ = capture_rgb(render_only, annotator.get_data, a.width, a.height, flush_attempts=1)
                    timings[name+"_render_s"] = time.perf_counter()-start_render
                    out = io.BytesIO()
                    # Lossless PNG; use fast compression for live observations.
                    # This changes encoded size, not any RGB sample.
                    Image.fromarray(pixels).save(out, format="PNG", compress_level=1)
                    images[name] = base64.b64encode(out.getvalue()).decode()
                send(connection, dict(ok=True, tick=request["tick"], sequence=request["sequence"],
                                      physics_steps=0, images=images, timings=timings))
            except Exception as exc:
                traceback.print_exc()
                send(connection, dict(ok=False, error=str(exc)))
        if product:
            annotator.detach()
            product.destroy()
    finally:
        connection.close()
        app.close()


if __name__ == "__main__":
    main()
