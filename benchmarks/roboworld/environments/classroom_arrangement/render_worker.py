"""Render-only Isaac Sim worker for classroom_arrangement: never runs a physics timeline.

Policy / diagnostic RGB (LDR PNG, path traced, OptiX denoiser on, pinhole cameras) from the body poses sent with
each request. Protocol and lifecycle follow barcode_scanning.render_worker (authenticated local socket, parent watch,
one idle render product parked off-scene so switching cameras never disables a product).
"""
import argparse
import base64
import io
import os
from pathlib import Path
import socket
import time
import traceback
import faulthandler


def main():
    from gym_anything_robotics.render_lifecycle import watch_parent
    from whiteboard.ipc import send, receive
    watch_parent()
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--samples", type=int, required=True)
    a = p.parse_args()
    connection = socket.create_connection(("127.0.0.1", int(os.environ["CLS_RENDER_PORT"])))
    send(connection, {"secret": os.environ.pop("CLS_RENDER_SECRET")})
    kit = Path(a.scene).parent/"kit"
    kit.mkdir(exist_ok=True)
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, open_usd=a.scene, create_new_stage=False, renderer="PathTracing",
                             width=a.width, height=a.height, samples_per_pixel_per_frame=a.samples, max_bounces=6,
                             multi_gpu=False, active_gpu=0, disable_viewport_updates=True,
                             extra_args=["--portable", f"--portable-root={kit}", "--/app/enableCrashReporter=false",
                                         "--/exts/omni.kit.viewport.window/startup/disableWindowOnLoad=true",
                                         "--/app/vulkan=true", "--/rtx/materialDb/syncLoads=true",
                                         "--/rtx/hydra/materialSyncLoads=true",
                                         "--/rtx-transient/resourcemanager/texturestreaming/async=false",
                                         "--/rtx-transient/resourcemanager/enableTextureStreaming=false",
                                         f"--/log/file={kit/'worker.log'}"]),
                        experience=str(Path(__file__).resolve().parents[1]/"wiping"/"render_only.kit"))
    try:
        import carb
        import numpy as np
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics
        omni.usd.add_hydra_engine("rtx", omni.usd.get_context())
        settings = carb.settings.get_settings()
        for key, value in {"/rtx/post/histogram/enabled": False, "/rtx/pathtracing/spp": a.samples,
                           "/rtx/pathtracing/totalSpp": a.samples, "/rtx/pathtracing/optixDenoiser/enabled": True,
                           "/rtx/post/tonemap/op": 6, "/rtx/post/tonemap/filmIso": 400.,
                           "/rtx/post/tonemap/fNumber": 2.8, "/rtx/post/tonemap/exposureTime": 1/100.}.items():
            settings.set(key, value)
        for _ in range(10):
            app.update()
        omni.timeline.get_timeline_interface().pause()
        stage = omni.usd.get_context().get_stage()
        if any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.IsA(UsdPhysics.Scene) for p in stage.Traverse()):
            raise RuntimeError("Renderer must not contain a dynamics scene")
        rep.orchestrator.set_capture_on_play(False)
        layer = stage.GetRootLayer()
        products = {}
        null_cam = UsdGeom.Camera.Define(stage, "/RenderWorkerNull")
        UsdGeom.Xformable(null_cam.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0., 0., -1000.))
        null_cam.GetClippingRangeAttr().Set(Gf.Vec2f(.01, .02))

        def product(key, camera_path, res):
            if key not in products:
                prod = rep.create.render_product(camera_path, res)
                ann = rep.AnnotatorRegistry.get_annotator("rgb")
                ann.attach([prod])
                products[key] = dict(product=prod, annotator=ann, camera=camera_path, res=res, live_camera=camera_path)
            pr = products[key]
            if pr["camera"] != camera_path:
                pr["product"].hydra_texture.set_camera_path(camera_path)
                pr["camera"] = camera_path
                pr["live_camera"] = camera_path
            for k, other in products.items():
                want = other["camera"] if k == key else "/RenderWorkerNull"
                if other["live_camera"] != want:
                    other["product"].hydra_texture.set_camera_path(want)
                    other["live_camera"] = want
            return pr

        def camera_path(name):
            alias = stage.GetPrimAtPath("/Cameras/" + name)
            if not alias:
                raise ValueError(f"Unknown camera {name}")
            return alias.GetCustomDataByKey("mounted_camera_path") or str(alias.GetPath())

        def apply(poses):
            updates = []
            for i, q in enumerate(poses):
                if not np.isfinite(q).all():
                    raise ValueError("Non-finite body pose")
                path = f"/Lab/body_{i}"
                updates.append((path + ".xformOp:translate", Gf.Vec3d(*q[:3])))
                updates.append((path + ".xformOp:orient", Gf.Quatf(float(q[6]), Gf.Vec3f(*map(float, q[3:6])))))
            with Sdf.ChangeBlock():
                for attr, value in updates:
                    spec = layer.GetAttributeAtPath(attr)
                    if not spec:
                        raise ValueError(f"No authored attribute {attr}")
                    if spec.default == value:
                        continue
                    spec.default = value

        def capture(pr, subframes, warm=False):
            ann = pr["annotator"]
            for attempt in range(8):
                rep.orchestrator.step(rt_subframes=(max(8, subframes) if warm else subframes), delta_time=0.)
                data = np.asarray(ann.get_data())
                if data.ndim == 3 and data.shape[:2] == (pr["res"][1], pr["res"][0]) and float(data[..., :3].max()) > 0:
                    return data
            raise RuntimeError("No valid readback after 8 render-only attempts")

        send(connection, {"phase": "ready"})
        warmed = set()
        while True:
            req = receive(connection)
            if req["op"] == "close":
                break
            try:
                faulthandler.dump_traceback_later(120, repeat=False)
                t0 = time.perf_counter()
                timings = {}
                if req["op"] != "observe":
                    raise ValueError("Unknown render operation")
                spp = int(req.get("spp", a.samples))
                settings.set("/rtx/pathtracing/spp", spp); settings.set("/rtx/pathtracing/totalSpp", spp)
                apply(req["body_poses"])
                images = {}
                for name in req["cameras"]:
                    res = tuple(req.get("resolution", (a.width, a.height)))
                    pr = product(f"ldr_{res[0]}x{res[1]}", camera_path(name), res)
                    key = (name, res)
                    t1 = time.perf_counter()
                    rgb = capture(pr, 2, warm=key not in warmed)
                    warmed.add(key)
                    timings[name + "_s"] = time.perf_counter() - t1
                    out = io.BytesIO()
                    Image.fromarray(np.ascontiguousarray(rgb[:, :, :3])).save(out, format="PNG", compress_level=1)
                    images[name] = base64.b64encode(out.getvalue()).decode()
                timings["total_s"] = time.perf_counter() - t0
                send(connection, dict(ok=True, tick=req["tick"], physics_steps=0, images=images, timings=timings))
            except Exception as exc:
                traceback.print_exc()
                send(connection, dict(ok=False, error=f"{type(exc).__name__}: {exc}"))
            finally:
                faulthandler.cancel_dump_traceback_later()
        for pr in products.values():
            pr["annotator"].detach()
            pr["product"].destroy()
    finally:
        connection.close()
        app.close()


if __name__ == "__main__":
    main()
