"""Render-only Isaac Sim worker for drilling: never runs a physics timeline.

Each request carries the recorded body poses and the meshes that changed (box minus holes, burr rings, dust columns,
scorch sleeves and the drawn-only drilling debris: render.WorkpieceCache; the vacuum hose: render.Renderer) of one
simulator state; the worker writes them into the exported stage (creating a debris or hose prim, bound to its named
material, the first time it is sent) and path-traces the requested cameras (LDR PNG, OptiX denoiser on). Protocol and lifecycle follow
wine_pouring.render_worker (authenticated local socket, parent watch).
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

# Camera white balance (every camera): the workshop is lit at 4000 K. Gains in linear sRGB: the 4000 K blackbody's
# chromaticity (CIE xy 0.3805, 0.3768 -> linear sRGB 1.53 : 1 : 0.577) corrected to the power 0.85 (a camera's auto
# white balance under neutral-white LED light; half correction, wine_pouring's choice for a warm bar, left a sepia cast
# in the first workshop renders), scaled to keep a neutral grey's luminance.
WB_GAINS = (0.712, 1.022, 1.630)


def white_balance(rgb8):
    import numpy as np
    x = rgb8.astype(np.float32)/255.
    lin = np.where(x <= 0.04045, x/12.92, ((x + 0.055)/1.055)**2.4)*np.asarray(WB_GAINS, np.float32)
    lin = np.clip(lin, 0., 1.)
    out = np.where(lin <= 0.0031308, lin*12.92, 1.055*lin**(1/2.4) - 0.055)
    return (out*255. + 0.5).astype(np.uint8)


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
    connection = socket.create_connection(("127.0.0.1", int(os.environ["DRL_RENDER_PORT"])))
    send(connection, {"secret": os.environ.pop("DRL_RENDER_SECRET")})
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
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt
        omni.usd.add_hydra_engine("rtx", omni.usd.get_context())
        settings = carb.settings.get_settings()
        for key, value in {"/rtx/post/histogram/enabled": False, "/rtx/pathtracing/spp": a.samples,
                           "/rtx/pathtracing/totalSpp": a.samples, "/rtx/pathtracing/optixDenoiser/enabled": True,
                           "/rtx/pathtracing/maxBounces": 12, "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 12,
                           "/rtx/post/tonemap/op": 6, "/rtx/post/tonemap/filmIso": 400.,
                           "/rtx/post/tonemap/fNumber": 2.8, "/rtx/post/tonemap/exposureTime": 1/60.}.items():
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
            # shape xforms that change (a broken bit's shank, cut back: render.broken_bit_updates)
            for path, x in req.get("xforms", {}).items():
                prim = stage.GetPrimAtPath(path)
                if not prim:
                    raise ValueError(f"No prim {path}")
                xf = UsdGeom.Xformable(prim)
                ops = {op.GetOpType(): op for op in xf.GetOrderedXformOps()}
                ops[UsdGeom.XformOp.TypeTranslate].Set(Gf.Vec3d(*map(float, x["t"])))
                sop = ops.get(UsdGeom.XformOp.TypeScale) or xf.AddScaleOp()
                sop.Set(Gf.Vec3f(*map(float, x["s"])))
            with Sdf.ChangeBlock():
                for attr, value in updates:
                    spec = layer.GetAttributeAtPath(attr)
                    if not spec:
                        raise ValueError(f"No authored attribute {attr}")
                    if spec.default != value:
                        spec.default = value
            # workpiece meshes that changed since the last request (None hides a prim, e.g. no burr yet); a mesh that
            # names a material ("mat": debris, the vacuum hose) is created on first use under its (existing) parent
            for path, m in req.get("meshes", {}).items():
                prim = stage.GetPrimAtPath(path)
                if not prim:
                    if m is None:
                        continue
                    if "mat" not in m:
                        raise ValueError(f"No mesh prim {path}")
                    if not stage.GetPrimAtPath(path.rsplit("/", 1)[0]):
                        raise ValueError(f"No parent prim for {path}")
                    mat = stage.GetPrimAtPath("/Looks/" + m["mat"])
                    if not mat:
                        raise ValueError(f"No material {m['mat']}")
                    prim = UsdGeom.Mesh.Define(stage, path).GetPrim()
                    UsdGeom.Mesh(prim).CreateSubdivisionSchemeAttr("none")
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(mat))
                    if m.get("double"):
                        UsdGeom.Mesh(prim).CreateDoubleSidedAttr(True)
                g = UsdGeom.Mesh(prim)
                if m is None:
                    UsdGeom.Imageable(prim).MakeInvisible()
                    continue
                v = np.frombuffer(base64.b64decode(m["v"]), np.float32).reshape(-1, 3)
                t = np.frombuffer(base64.b64decode(m["t"]), np.int32).reshape(-1, 3)
                if not np.isfinite(v).all():
                    raise ValueError("Non-finite workpiece mesh")
                g.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(v)))
                g.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(t), 3, np.int32)))
                g.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.ascontiguousarray(t.ravel())))
                g.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray([v.min(0), v.max(0)], np.float32)))
                if "uv" in m:
                    uv = np.frombuffer(base64.b64decode(m["uv"]), np.float32).reshape(-1, 2)
                    UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                         UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(uv)))
                if "n" in m:
                    nrm = np.frombuffer(base64.b64decode(m["n"]), np.float32).reshape(-1, 3)
                    g.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(nrm)))
                    g.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
                elif g.GetNormalsAttr():
                    g.GetNormalsAttr().Clear()
                UsdGeom.Imageable(prim).MakeVisible()

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
                    Image.fromarray(white_balance(np.ascontiguousarray(rgb[:, :, :3]))).save(out, format="PNG",
                                                                                            compress_level=1)
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
