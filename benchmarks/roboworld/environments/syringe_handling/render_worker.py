"""Render-only Isaac Sim worker for syringe_handling: never runs a physics timeline.

Each request carries the recorded body poses and the liquid's meshes of one simulator state (render.liquid_meshes:
one mesh per liquid, each with points, triangles and normals); the worker writes them into the exported stage and
path-traces the requested cameras, each at its own resolution (LDR PNG, OptiX denoiser on). A mesh missing from a
request is hidden. Protocol and lifecycle are ingredient_mixing.render_worker's (authenticated local socket, parent
watch), which follows wine_pouring's.
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

# Exposure (every camera, policy and diagnostic): ISO 400, f/2.8, 1/125 s, the incident-light exposure for 500 lx (an
# 18 % grey under 500 lx has 28.6 nit, EV100 7.84), as in the teaching kitchen: white casework, the white robots and the
# white tray stay below clipping. White balance is in the lights' colour (dressing.LIGHT_RGB).
EXPOSURE = dict(iso=400., f_number=2.8, time_s=1/125.)


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
    connection = socket.create_connection(("127.0.0.1", int(os.environ["SYRINGE_RENDER_PORT"])))
    send(connection, {"secret": os.environ.pop("SYRINGE_RENDER_SECRET")})
    kit = Path(a.scene).parent/"kit"
    kit.mkdir(exist_ok=True)
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, open_usd=a.scene, create_new_stage=False, renderer="PathTracing",
                             width=a.width, height=a.height, samples_per_pixel_per_frame=a.samples, max_bounces=8,
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
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt
        omni.usd.add_hydra_engine("rtx", omni.usd.get_context())
        settings = carb.settings.get_settings()
        for key, value in {"/rtx/post/histogram/enabled": False, "/rtx/pathtracing/spp": a.samples,
                           "/rtx/pathtracing/totalSpp": a.samples, "/rtx/pathtracing/optixDenoiser/enabled": True,
                           "/rtx/pathtracing/maxBounces": 12, "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 12,
                           "/rtx/post/tonemap/op": 6, "/rtx/post/tonemap/filmIso": EXPOSURE["iso"],
                           "/rtx/post/tonemap/fNumber": EXPOSURE["f_number"],
                           "/rtx/post/tonemap/exposureTime": EXPOSURE["time_s"]}.items():
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

        def product(key, camera_path, res):
            if key not in products:
                prod = rep.create.render_product(camera_path, res)
                ann = rep.AnnotatorRegistry.get_annotator("rgb")
                ann.attach([prod])
                products[key] = dict(product=prod, annotator=ann, camera=camera_path, res=res)
            pr = products[key]
            if pr["camera"] != camera_path:
                pr["product"].hydra_texture.set_camera_path(camera_path)
                pr["camera"] = camera_path
            return pr

        def camera_path(name):
            alias = stage.GetPrimAtPath("/Cameras/" + name)
            if not alias:
                raise ValueError(f"Unknown camera {name}")
            return alias.GetCustomDataByKey("mounted_camera_path") or str(alias.GetPath())

        def array(m, key, dtype, width):
            x = np.frombuffer(base64.b64decode(m[key]), dtype)
            x = x.reshape(-1, width) if width > 1 else x
            if dtype == np.float32 and not np.isfinite(x).all():
                raise ValueError(f"Non-finite liquid mesh ({key})")
            return np.ascontiguousarray(x)

        def apply(req):
            updates = []
            for i, q in enumerate(req["body_poses"]):
                if not np.isfinite(q).all():
                    raise ValueError("Non-finite body pose")
                path = f"/Lab/body_{i}"
                updates.append((path + ".xformOp:translate", Gf.Vec3d(*q[:3])))
                updates.append((path + ".xformOp:orient", Gf.Quatf(float(q[6]), Gf.Vec3f(*map(float, q[3:6])))))
            for path, visible in req.get("visibility", {}).items():
                updates.append((path + ".visibility", UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible))
            with Sdf.ChangeBlock():
                for attr, value in updates:
                    spec = layer.GetAttributeAtPath(attr)
                    if not spec:
                        raise ValueError(f"No authored attribute {attr}")
                    if spec.default != value:
                        spec.default = value
            # the mixture's surfaces: replace each mesh with the request's (or hide it when the request has none)
            meshes = req.get("meshes", {})
            for prim in stage.GetPrimAtPath("/Liquid").GetChildren():
                name = prim.GetName()
                g = UsdGeom.Mesh(prim)
                m = meshes.get(name)
                if not m:
                    UsdGeom.Imageable(prim).MakeInvisible()
                    continue
                v, t = array(m, "v", np.float32, 3), array(m, "t", np.int32, 3)
                if not (len(v) and len(t)):
                    UsdGeom.Imageable(prim).MakeInvisible()
                    continue
                if int(t.max()) >= len(v) or int(t.min()) < 0:
                    raise ValueError(f"Liquid mesh {name}: triangle index out of range")
                pv = UsdGeom.PrimvarsAPI(g)
                with Sdf.ChangeBlock():
                    g.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v))
                    g.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(t), 3, np.int32)))
                    g.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.ascontiguousarray(t.ravel())))
                    g.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray([v.min(0), v.max(0)], np.float32)))
                    if "n" in m:
                        g.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(array(m, "n", np.float32, 3)))
                        g.SetNormalsInterpolation("vertex")
                    elif g.GetNormalsAttr():
                        g.GetNormalsAttr().Clear()
                    c = array(m, "c", np.float32, 3) if "c" in m else np.tile(np.float32([.8, .75, .6]), (len(v), 1))
                    gl = array(m, "g", np.float32, 1) if "g" in m else np.zeros(len(v), np.float32)
                    tr = array(m, "x", np.float32, 1) if "x" in m else np.zeros(len(v), np.float32)
                    if not (len(c) == len(gl) == len(tr) == len(v)):
                        raise ValueError(f"Liquid mesh {name}: primvar count differs from the vertex count")
                    pv.GetPrimvar("displayColor").Set(Vt.Vec3fArray.FromNumpy(c))
                    pv.GetPrimvar("gloss").Set(Vt.FloatArray.FromNumpy(gl))
                    pv.GetPrimvar("trans").Set(Vt.FloatArray.FromNumpy(tr))
                UsdGeom.Imageable(prim).MakeVisible()

        def capture(pr, subframes, warm=False):
            ann = pr["annotator"]
            # (a new worker's first readbacks can come back empty for longer than 8 tries while other sessions' renders
            # hold the card; the first capture of a camera now gets 40, every later one 20)
            tries = 40 if warm else 20
            for attempt in range(tries):
                rep.orchestrator.step(rt_subframes=(max(8, subframes) if warm else subframes), delta_time=0.)
                data = np.asarray(ann.get_data())
                if data.ndim == 3 and data.shape[:2] == (pr["res"][1], pr["res"][0]) and float(data[..., :3].max()) > 0:
                    return data
            raise RuntimeError(f"No valid readback after {tries} render-only attempts")

        send(connection, {"phase": "ready"})
        warmed = set()
        while True:
            req = receive(connection)
            if req["op"] == "close":
                break
            try:
                faulthandler.dump_traceback_later(300, repeat=False)
                t0 = time.perf_counter()
                timings = {}
                if req["op"] != "observe":
                    raise ValueError("Unknown render operation")
                apply(req)
                timings["apply_s"] = time.perf_counter() - t0
                images = {}
                for name in req["cameras"]:
                    res = tuple(req.get("resolutions", {}).get(name) or req.get("resolution") or (a.width, a.height))
                    pr = product(f"ldr_{res[0]}x{res[1]}", camera_path(name), res)
                    key = (name, res)
                    t1 = time.perf_counter()
                    rgb = capture(pr, int(req.get("subframes", 2)), warm=key not in warmed)
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
