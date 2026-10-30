"""Isaac Sim render-only worker for the sewing cell: draws recorded states, never runs a physics timeline.

Protocol and Kit setup follow wiping/render_worker.py and welding/render_worker.py (whiteboard.ipc, render_lifecycle
parent watch, the minimal render-only kit, RGB readback); the state application is the sewing frame (sewing/render.py:
robot body poses, the machine's moving parts, the plies' points and normals, the thread curves).
"""
import argparse
import base64
import faulthandler
import io
import json
import os
import socket
import time
import traceback
from pathlib import Path

import numpy as np


def _dec(s, dtype, cols=None):
    a = np.frombuffer(base64.b64decode(s), dtype=dtype)
    return a.reshape(-1, cols) if cols else a


def capture(step_render, read_pixels, width, height, flush_attempts=1):
    """RGB readback (whiteboard.render_readback.capture_rgb's loop). A missing or wrongly shaped buffer fails after
    8 attempts."""
    last = None
    for attempt in range(8):
        step_render()
        raw = np.asarray(read_pixels())
        last = raw.shape
        if raw.ndim == 3 and raw.shape[:2] == (height, width) and raw.shape[2] >= 3 and attempt + 1 >= flush_attempts:
            return raw[:, :, :3].copy(), 0
    raise RuntimeError(f"No RGB readback after 8 render-only attempts; shape={last}")


def apply_state(stage, request):
    from pxr import Gf, Sdf, UsdGeom, Vt
    layer = stage.GetRootLayer()
    updates = []
    poses = dict(request["body_q"])
    poses.update(request.get("xforms") or {})
    for path, q in poses.items():
        q = np.asarray(q, float)
        if not np.isfinite(q).all():
            raise ValueError("Non-finite pose")
        updates.append((path + ".xformOp:translate", Gf.Vec3d(*q[:3])))
        updates.append((path + ".xformOp:orient", Gf.Quatf(float(q[6]), Gf.Vec3f(*map(float, q[3:6])))))
    with Sdf.ChangeBlock():
        for attr, value in updates:
            spec = layer.GetAttributeAtPath(attr)
            if not spec:
                raise ValueError("Missing pose attribute " + attr)
            spec.default = value
    for ply in request.get("plies") or []:
        m = UsdGeom.Mesh(stage.GetPrimAtPath(ply["path"]))
        P = _dec(ply["points"], np.float32, 3)
        N = _dec(ply["normals"], np.float32, 3)
        if not (np.isfinite(P).all() and np.isfinite(N).all()) or len(P) != ply["n"]:
            raise ValueError("Invalid ply points")
        m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P))
        m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(N)) if m.GetNormalsAttr() else m.CreateNormalsAttr(
            Vt.Vec3fArray.FromNumpy(N))
        m.SetNormalsInterpolation("vertex")
        m.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([P.min(0), P.max(0)], np.float32)))
    for th in request.get("threads") or []:
        c = UsdGeom.BasisCurves(stage.GetPrimAtPath(th["path"]))
        if th["points"] is None:
            UsdGeom.Imageable(c).MakeInvisible()
            continue
        P = _dec(th["points"], np.float32, 3)
        if not np.isfinite(P).all() or len(P) < 2:
            raise ValueError("Invalid thread points")
        c.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P))
        c.GetCurveVertexCountsAttr().Set([len(P)])
        c.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([P.min(0) - 1e-3, P.max(0) + 1e-3], np.float32)))
        UsdGeom.Imageable(c).MakeVisible()


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
    connection = socket.create_connection(("127.0.0.1", int(os.environ["WIPE_RENDER_PORT"])))
    send(connection, {"secret": os.environ.pop("WIPE_RENDER_SECRET")})
    kit = Path(a.scene).parent/"kit"
    kit.mkdir(exist_ok=True)
    from isaacsim import SimulationApp
    import wiping
    app = SimulationApp(dict(headless=True, open_usd=a.scene, create_new_stage=False, renderer="PathTracing",
                             width=a.width, height=a.height, samples_per_pixel_per_frame=a.samples, max_bounces=8,
                             multi_gpu=False, active_gpu=0, disable_viewport_updates=True,
                             extra_args=["--portable", f"--portable-root={kit}", "--/app/enableCrashReporter=false",
                                         "--/exts/omni.kit.viewport.window/startup/disableWindowOnLoad=true",
                                         "--/app/vulkan=true", "--/rtx/materialDb/syncLoads=true",
                                         "--/rtx/hydra/materialSyncLoads=true",
                                         "--/rtx-transient/resourcemanager/texturestreaming/async=false",
                                         f"--/log/file={kit/'worker.log'}"]),
                        experience=str(Path(wiping.__file__).with_name("render_only.kit")))
    try:
        import carb
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import UsdGeom, UsdPhysics, UsdRender
        omni.usd.add_hydra_engine("rtx", omni.usd.get_context())
        settings = carb.settings.get_settings()
        # fixed exposure of a camera in a lit workroom (no auto-exposure) [authored]
        for key, value in {"/rtx/post/histogram/enabled": False, "/rtx/pathtracing/spp": a.samples,
                           "/rtx/pathtracing/totalSpp": a.samples, "/rtx/pathtracing/optixDenoiser/enabled": True,
                           "/rtx/post/tonemap/op": 6, "/rtx/post/tonemap/filmIso": 400.,
                           "/rtx/post/tonemap/fNumber": 2.8, "/rtx/post/tonemap/exposureTime": 1/60.}.items():
            settings.set(key, value)
        for _ in range(12):
            app.update()
        omni.timeline.get_timeline_interface().pause()
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                cam = UsdGeom.Camera(prim)
                cam.GetVerticalApertureAttr().Set(cam.GetHorizontalApertureAttr().Get()*a.height/a.width)
        if any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.IsA(UsdPhysics.Scene) for p in stage.Traverse()):
            raise RuntimeError("Renderer must not contain a dynamics scene")
        rep.orchestrator.set_capture_on_play(False)
        product, annotator, active = None, None, None
        send(connection, {"phase": "ready"})
        while True:
            request = receive(connection)
            if request["op"] == "close":
                break
            try:
                if request["op"] != "observe":
                    raise ValueError("Unknown render operation")
                faulthandler.dump_traceback_later(120, repeat=False)
                t0 = time.perf_counter()
                apply_state(stage, request)
                timings = {"usd_update_s": time.perf_counter() - t0}
                images = {}
                for name in request["cameras"]:
                    path = "/Cameras/" + name
                    alias = stage.GetPrimAtPath(path)
                    if alias and alias.GetCustomDataByKey("mounted_camera_path"):
                        path = alias.GetCustomDataByKey("mounted_camera_path")
                    if not UsdGeom.Camera(stage.GetPrimAtPath(path)):
                        raise ValueError("Unknown camera " + name)
                    if product is None:
                        product = rep.create.render_product(path, (a.width, a.height))
                        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                        annotator.attach([product])
                        product.hydra_texture.set_updates_enabled(True)
                        rep.orchestrator.step(rt_subframes=8, delta_time=0.)
                    elif path != active:
                        UsdRender.Product(stage.GetPrimAtPath(product.path)).GetCameraRel().SetTargets([path])
                        product.hydra_texture.set_camera_path(path)
                    active = path
                    ex = request.get("exposure", {}).get(name)
                    if ex:
                        settings.set("/rtx/post/tonemap/exposureTime", float(ex["exposureTime"]))
                        settings.set("/rtx/post/tonemap/fNumber", float(ex["fNumber"]))
                        settings.set("/rtx/post/tonemap/filmIso", float(ex["filmIso"]))
                    t1 = time.perf_counter()
                    pixels, _ = capture(lambda: rep.orchestrator.step(rt_subframes=2, delta_time=0.),
                                        annotator.get_data, a.width, a.height, flush_attempts=1)
                    timings[name + "_render_s"] = time.perf_counter() - t1
                    out = io.BytesIO()
                    Image.fromarray(pixels).save(out, format="PNG", compress_level=1)
                    images[name] = base64.b64encode(out.getvalue()).decode()
                send(connection, dict(ok=True, tick=request["tick"], sequence=request["sequence"], physics_steps=0,
                                      images=images, timings=timings))
            except Exception as exc:
                traceback.print_exc()
                send(connection, dict(ok=False, error=str(exc)))
            finally:
                faulthandler.cancel_dump_traceback_later()
        if product:
            annotator.detach()
            product.destroy()
    finally:
        connection.close()
        app.close()


if __name__ == "__main__":
    main()
