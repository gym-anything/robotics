"""Render the received physical cloth/robot state; never run a physics timeline."""
import argparse
import base64
import io
import os
from pathlib import Path
import socket
import traceback
import time
import faulthandler
import json
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
    connection = socket.create_connection(("127.0.0.1", int(os.environ["WIPE_RENDER_PORT"])))
    send(connection, {"secret": os.environ.pop("WIPE_RENDER_SECRET")})
    kit = Path(a.scene).parent/"kit"
    kit.mkdir(exist_ok=True)
    from isaacsim import SimulationApp
    # The qualified Windows renderer uses D3D12; Vulkan remains available as
    # an explicit override and is the Linux default. This does not select the
    # Newton physics device (CUDA) or change camera/sample settings.
    api=os.environ.get("WIPE_RENDER_GRAPHICS_API", "d3d12" if os.name == "nt" else "vulkan")
    if api not in ("vulkan", "d3d12") or (api == "d3d12" and os.name != "nt"):
        raise ValueError("Choose vulkan, or d3d12 on Windows")
    streaming=os.environ.get('WIPE_TEXTURE_STREAMING','0')=='1'
    budget=float(os.environ.get('WIPE_TEXTURE_BUDGET','.6'))
    if not 0<budget<=1:raise ValueError('Texture budget must be a GPU-memory fraction in(0,1]')
    app = SimulationApp(dict(headless=True, open_usd=a.scene, create_new_stage=False,
        renderer="PathTracing", width=a.width, height=a.height, samples_per_pixel_per_frame=a.samples,
        max_bounces=12, multi_gpu=False, active_gpu=int(os.environ.get("GAR_RENDER_GPU", "0")),
        disable_viewport_updates=True,
        extra_args=["--portable", f"--portable-root={kit}", "--/app/enableCrashReporter=false",
                    "--/exts/omni.kit.viewport.window/startup/disableWindowOnLoad=true",
                    "--/app/vulkan="+("true" if api == "vulkan" else "false"),
                    "--/rtx/materialDb/syncLoads=true", "--/rtx/hydra/materialSyncLoads=true",
                    "--/rtx-transient/resourcemanager/texturestreaming/async=false",
                    "--/rtx-transient/resourcemanager/enableTextureStreaming="+str(streaming).lower(),
                    f"--/rtx-transient/resourcemanager/texturestreaming/memoryBudget={budget}",
                    f"--/log/file={kit/'worker.log'}"]),
        experience=str(Path(__file__).with_name('render_only.kit')) if os.environ.get('WIPE_MINIMAL_RENDER')=='1' else '')
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
        # With no viewport window, initialize the context's Hydra engine
        # explicitly; a render product alone does not create it in this Kit.
        omni.usd.add_hydra_engine('rtx',omni.usd.get_context())
        settings = carb.settings.get_settings()
        # Explicit diagnostic bound for a missing/stalled assets-loaded event.
        # This does NOT assert that assets finished loading: native warnings and
        # the post-warmup loading status are retained with the captured images.
        asset_wait=os.environ.get('WIPE_ASSET_WAIT_SECONDS')
        if asset_wait is not None:
            import math
            asset_wait=float(asset_wait)
            if not math.isfinite(asset_wait) or not 10<=asset_wait<=120:
                raise ValueError('Diagnostic asset wait must be10..120 seconds')
            settings.set('/exts/omni.replicator.core/maxAssetLoadingTime',asset_wait)
        for key, value in {"/rtx/post/histogram/enabled": False, "/rtx/pathtracing/spp": a.samples,
            "/rtx/pathtracing/totalSpp": a.samples, "/rtx/pathtracing/optixDenoiser/enabled": True,
            "/rtx/post/tonemap/op": 6, "/rtx/post/tonemap/filmIso": 400.,
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
        effective=dict(width=a.width,height=a.height,samples=a.samples,max_bounces=12,
            diagnostic_asset_wait_seconds=asset_wait,
            graphics_api=api,minimal_renderer=os.environ.get('WIPE_MINIMAL_RENDER')=='1',
            brushed_surfaces=os.environ.get('WIPE_BRUSHED_SURFACES')=='1',
            fixture_finish=os.environ.get('WIPE_FIXTURE_FINISH')=='1',
            species_optics=os.environ.get('WIPE_SPECIES_OPTICS')=='1',
            layered_coat=os.environ.get('WIPE_LAYERED_COAT')=='1',
            stable_film_topology=os.environ.get('WIPE_STABLE_FILM_TOPOLOGY')=='1',
            paged_film=os.environ.get('WIPE_PAGED_FILM')=='1',
            texture_streaming=settings.get('/rtx-transient/resourcemanager/enableTextureStreaming'),
            texture_budget_fraction=settings.get('/rtx-transient/resourcemanager/texturestreaming/memoryBudget'),
            note='Streaming chooses view-dependent texture mip levels; no asset resampling, geometry removal or physics changes.')
        (Path(a.scene).parent/'renderer_settings.json').write_text(json.dumps(effective,indent=2)+'\n')
        product, annotator, active_camera = None, None, None
        send(connection, {"phase": "ready"})
        while True:
            request = receive(connection)
            if request["op"] == "close":
                break
            try:
                if request["op"] != "observe":
                    raise ValueError("Unknown render operation")
                faulthandler.dump_traceback_later(90,repeat=False)
                start_update = time.perf_counter()
                apply_recorded_state(stage, request)
                print(f'Render request {request["sequence"]}: state applied',flush=True)
                timings = {"usd_update_s":time.perf_counter()-start_update}
                images = {}
                for name in request["cameras"]:
                    camera_path = "/Cameras/"+name
                    alias=stage.GetPrimAtPath(camera_path)
                    if alias and alias.GetCustomDataByKey('mounted_camera_path'):
                        camera_path=alias.GetCustomDataByKey('mounted_camera_path')
                    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
                    if not camera:
                        raise ValueError("Unknown camera")
                    if product is None:
                        product = rep.create.render_product(camera_path, (a.width, a.height))
                        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                        annotator.attach([product])
                        product.hydra_texture.set_updates_enabled(True)
                        print(f'Render request {request["sequence"]}: warm-up {name}; '
                              f'USD loading={omni.usd.get_context().get_stage_loading_status()}; '
                              f'max asset wait={settings.get("/exts/omni.replicator.core/maxAssetLoadingTime")}',flush=True)
                        if os.environ.get('WIPE_PROFILE_GRAPHICS')=='1':
                            import subprocess
                            memory=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free',
                                '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
                            print(json.dumps(dict(stage='before_first_warmup',
                                gpu_used_free_mib=memory.stdout.strip(),
                                scope='Whole-GPU memory, not per-process')),flush=True)
                        # Resolve materials before the first returned frame.
                        # This is render-only startup at the SAME recorded tick.
                        rep.orchestrator.step(rt_subframes=8, delta_time=0.)
                        print(json.dumps(dict(stage='after_first_warmup',
                            asset_loading_status=omni.usd.get_context().get_stage_loading_status(),
                            diagnostic_asset_wait_seconds=asset_wait)),flush=True)
                        print(f'Render request {request["sequence"]}: warm-up complete',flush=True)
                    elif name != active_camera:
                        UsdRender.Product(stage.GetPrimAtPath(product.path)).GetCameraRel().SetTargets([camera_path])
                        product.hydra_texture.set_camera_path(camera_path)
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
                    print(f'Render request {request["sequence"]}: captured {name}',flush=True)
                if os.environ.get('WIPE_PROFILE_GRAPHICS')=='1' and (request['sequence']==1 or request['sequence']%30==0):
                    import subprocess
                    memory=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
                    print(json.dumps(dict(render_sequence=request['sequence'],tick=request['tick'],
                        gpu_used_free_mib=memory.stdout.strip(),scope='Whole-GPU memory, not per-process')),
                        flush=True)
                send(connection, dict(ok=True, tick=request["tick"], sequence=request["sequence"],
                                      physics_steps=0, images=images, timings=timings))
            except Exception as exc:
                traceback.print_exc()
                send(connection, dict(ok=False, error=str(exc)))
            finally:faulthandler.cancel_dump_traceback_later()
        if product:
            annotator.detach()
            product.destroy()
    finally:
        connection.close()
        app.close()


if __name__ == "__main__":
    main()
