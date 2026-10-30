"""Render-only Isaac Sim worker for barcode_scanning: never runs a physics timeline.

Two kinds of frames, both from the recorded simulator state sent with each request:
  observe  policy / diagnostic RGB (LDR PNG, OptiX denoiser on, pinhole cameras)
  scan     the scanner imager's linear radiance (HdrColor, float16 RGB, denoiser OFF because denoisers reshape
           edges, thin-lens depth of field on the imager camera); K sub-exposure pose sets are rendered and
           averaged (motion blur over the exposure), the scanner's LEDs/aim set per request
Each request may also set light intensities, prim visibility and texture files (the host display, the
good-read LED). Protocol and lifecycle follow wiping.render_worker (authenticated local socket, parent watch).
"""
import argparse
import base64
import io
import json
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
    p.add_argument("--scan-samples", type=int, required=True)
    a = p.parse_args()
    connection = socket.create_connection(("127.0.0.1", int(os.environ["BCS_RENDER_PORT"])))
    send(connection, {"secret": os.environ.pop("BCS_RENDER_SECRET")})
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
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
        omni.usd.add_hydra_engine("rtx", omni.usd.get_context())
        settings = carb.settings.get_settings()
        for key, value in {"/rtx/post/histogram/enabled": False, "/rtx/pathtracing/spp": a.samples,
                           "/rtx/pathtracing/totalSpp": a.samples, "/rtx/pathtracing/optixDenoiser/enabled": True,
                           "/rtx/post/tonemap/op": 6, "/rtx/post/tonemap/filmIso": 400.,
                           "/rtx/post/tonemap/fNumber": 2.8, "/rtx/post/tonemap/exposureTime": 1/120.,
                           "/rtx/raytracing/fractionalCutoutOpacity": True}.items():   # clear film (render.Stage.film)
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
        # An idle render product is parked on this camera, far below the floor looking down at nothing (every ray
        # misses), instead of having its updates disabled: a product whose updates were disabled and enabled again
        # never returned an image ("No valid readback"), so a policy-camera frame after a scanner frame failed, and every
        # frame after it (measured for each alternative: re-enabling either product failed, re-creating them failed,
        # keeping both live rendered both every step)
        null_cam = UsdGeom.Camera.Define(stage, "/RenderWorkerNull")
        UsdGeom.Xformable(null_cam.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0., 0., -1000.))
        null_cam.GetClippingRangeAttr().Set(Gf.Vec2f(.01, .02))

        def product(key, camera_path, res, hdr):
            if key not in products:
                prod = rep.create.render_product(camera_path, res)
                ann = rep.AnnotatorRegistry.get_annotator("HdrColor" if hdr else "rgb")
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

        def apply(req, poses):
            import numpy as np
            updates = []
            for i, q in enumerate(poses):
                if not np.isfinite(q).all():
                    raise ValueError("Non-finite body pose")
                path = f"/Lab/body_{i}"
                updates.append((path + ".xformOp:translate", Gf.Vec3d(*q[:3])))
                updates.append((path + ".xformOp:orient", Gf.Quatf(float(q[6]), Gf.Vec3f(*map(float, q[3:6])))))
            for path, value in req.get("lights", {}).items():
                updates.append((path + ".inputs:intensity", float(value)))
            for path, visible in req.get("visibility", {}).items():
                updates.append((path + ".visibility", UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible))
            for path, file in req.get("textures", {}).items():
                updates.append((path, Sdf.AssetPath(file)))
            with Sdf.ChangeBlock():
                for attr, value in updates:
                    spec = layer.GetAttributeAtPath(attr)
                    if not spec:
                        raise ValueError(f"No authored attribute {attr}")
                    if spec.default == value:
                        continue            # unchanged: re-authoring it still makes the renderer resync the prim
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
                if req["op"] == "observe":
                    settings.set("/rtx/pathtracing/maxBounces", 6)
                    settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
                    settings.set("/rtx/pathtracing/spp", a.samples); settings.set("/rtx/pathtracing/totalSpp", a.samples)
                    apply(req, req["body_poses"])
                    images = {}
                    for name in req["cameras"]:
                        res = tuple(req.get("resolution", (a.width, a.height)))
                        pr = product(f"ldr_{res[0]}x{res[1]}", camera_path(name), res, False)
                        key = (name, res)
                        t1 = time.perf_counter()
                        rgb = capture(pr, 2, warm=key not in warmed)
                        warmed.add(key)
                        timings[name + "_s"] = time.perf_counter() - t1
                        out = io.BytesIO()
                        Image.fromarray(np.ascontiguousarray(rgb[:, :, :3])).save(out, format="PNG", compress_level=1)
                        images[name] = base64.b64encode(out.getvalue()).decode()
                    send(connection, dict(ok=True, tick=req["tick"], physics_steps=0, images=images, timings=timings))
                elif req["op"] == "scan":
                    # the imager's path depth (params.RENDER_IMAGER_MAX_BOUNCES, sent by render.Renderer.scan): paths
                    # bounced up to 6 times inside the closed rooms and made every scanner sample about twice as dear
                    # as at 3; policy and diagnostic cameras keep 6
                    settings.set("/rtx/pathtracing/maxBounces", int(req.get("max_bounces", 6)))
                    settings.set("/rtx/pathtracing/optixDenoiser/enabled", False)
                    # samples per pixel of the whole frame, split over its motion-blur subframes (their average carries
                    # all of them); RTX clamps a frame to 24 samples unless clampSpp is set, and a second render subframe of an
                    # unchanged scene adds none. Overrides for sample studies: spp (per frame), total_spp, rt_subframes,
                    # clamp_spp
                    spp = max(1, -(-int(req.get("spp", a.scan_samples))//len(req["subframe_poses"])))
                    settings.set("/rtx/pathtracing/clampSpp", int(req.get("clamp_spp", 0)))
                    settings.set("/rtx/pathtracing/spp", spp)
                    settings.set("/rtx/pathtracing/totalSpp", int(req.get("total_spp", spp)))
                    rt = int(req.get("rt_subframes", 1))
                    res = tuple(req["resolution"])
                    pr = product("scan", camera_path(req["camera"]), res, True)
                    acc = None
                    for k, poses in enumerate(req["subframe_poses"]):
                        apply(req if k == 0 else {}, poses)
                        t1 = time.perf_counter()
                        img = capture(pr, rt, warm="scan" not in warmed).astype(np.float32)[..., :3]
                        warmed.add("scan")
                        timings[f"sub{k}_s"] = time.perf_counter() - t1
                        acc = img if acc is None else acc + img
                    acc /= len(req["subframe_poses"])
                    if not np.isfinite(acc).all():
                        raise RuntimeError("Non-finite scanner radiance")
                    blob = base64.b64encode(acc.astype(np.float16).tobytes()).decode()
                    send(connection, dict(ok=True, tick=req["tick"], physics_steps=0, radiance=blob,
                                          shape=list(acc.shape), timings=timings))
                else:
                    raise ValueError("Unknown render operation")
                timings["total_s"] = time.perf_counter() - t0
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
