"""Render-only Isaac Sim worker for concrete_finishing: never runs a physics timeline.

Each request carries the recorded body poses and the paste surfaces of one simulator state (render.paste_meshes: one
mesh per region, with points, triangles, normals and the per-vertex primvars displayColor, gloss and tex that
ConcreteFinishing.mdl reads); the worker writes them into the exported stage and path-traces the requested cameras
(LDR PNG, OptiX denoiser on). A region missing from a request is hidden. Protocol, lifecycle and the mesh update are
ingredient_mixing.render_worker's (authenticated local socket, parent watch).
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

# Exposure (every camera): ISO 400, f/2.8, 1/60 s, drilling's for its grey basement workshop of concrete and CMU
# (the same kind of room, the same 4000 K light).
EXPOSURE = dict(iso=400., f_number=2.8, time_s=1/60.)
# Camera white balance: drilling's (render_worker.WB_GAINS): the 4000 K blackbody (CIE xy 0.3805, 0.3768 -> linear sRGB
# 1.53 : 1 : 0.577) corrected to the power 0.85, a camera's auto white balance under neutral-white LED light; the first
# stills of this room without it were orange-brown throughout.
from drilling.render_worker import white_balance


def pose_update_mask(previous, current, position_tolerance_m=1e-6, rotation_tolerance_rad=1e-6):
    """Which authored transform components need updating? Inputs use xyz + quaternion xyzw.

    Compare against the actual authored pose, not the preceding requested pose,
    so a series of small physical movements cannot accumulate hidden drift.
    This is used only by the opt-in render diagnostic; physics retains full precision.
    """
    import math
    import numpy as np
    old, new = np.asarray(previous, float), np.asarray(current, float)
    translation = np.linalg.norm(new[:3] - old[:3]) > position_tolerance_m
    norms = np.linalg.norm(old[3:])*np.linalg.norm(new[3:])
    rotation = not norms or abs(float(np.dot(old[3:], new[3:])))/norms < math.cos(rotation_tolerance_rad/2)
    return bool(translation), bool(rotation)


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
    connection = socket.create_connection(("127.0.0.1", int(os.environ["CF_RENDER_PORT"])))
    send(connection, {"secret": os.environ.pop("CF_RENDER_SECRET")})
    kit = Path(a.scene).parent/"kit"
    kit.mkdir(exist_ok=True)
    render_mode = os.environ.get("CF_RENDER_MODE", "PathTracing")
    if render_mode not in ("PathTracing", "RaytracedLighting"):
        raise ValueError("CF_RENDER_MODE must be PathTracing or RaytracedLighting")
    pose_filter = os.environ.get("CF_RENDER_POSE_FILTER", "0")
    if pose_filter not in ("0", "1"):
        raise ValueError("CF_RENDER_POSE_FILTER must be 0 or 1")
    pose_filter = pose_filter == "1"
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, open_usd=a.scene, create_new_stage=False, renderer=render_mode,
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
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
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
        # Diagnostic switch for the installed Replicator device-buffer path.
        # Keep the established CPU default until a measured comparison supports a change.
        readback_device = os.environ.get("CF_RENDER_RGB_DEVICE", "cpu")
        if readback_device not in ("cpu", "cuda"):
            raise ValueError("CF_RENDER_RGB_DEVICE must be cpu or cuda")

        def product(key, camera_path, res):
            if key not in products:
                prod = rep.create.render_product(camera_path, res)
                ann = rep.AnnotatorRegistry.get_annotator("rgb", device=readback_device)
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
                raise ValueError(f"Non-finite paste mesh ({key})")
            return np.ascontiguousarray(x)

        topology = {}                                       # per paste mesh: (triangles, hash) last set

        def apply(req):
            updates = []
            for i, q in enumerate(req["body_poses"]):
                if not np.isfinite(q).all():
                    raise ValueError("Non-finite body pose")
                path = f"/Lab/body_{i}"
                move, rotate = True, True
                if pose_filter:
                    ps = layer.GetAttributeAtPath(path + ".xformOp:translate")
                    qs = layer.GetAttributeAtPath(path + ".xformOp:orient")
                    if not ps or not qs:
                        raise ValueError(f"No authored transform {path}")
                    old_q = qs.default
                    old = list(ps.default) + list(old_q.GetImaginary()) + [old_q.GetReal()]
                    move, rotate = pose_update_mask(old, q)
                if move:
                    updates.append((path + ".xformOp:translate", Gf.Vec3d(*q[:3])))
                if rotate:
                    updates.append((path + ".xformOp:orient", Gf.Quatf(float(q[6]), Gf.Vec3f(*map(float, q[3:6])))))
            for path, visible in req.get("visibility", {}).items():
                updates.append((path + ".visibility", UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible))
            changed = {"body_translation_indices": [], "body_rotation_indices": []}
            with Sdf.ChangeBlock():
                for attr, value in updates:
                    spec = layer.GetAttributeAtPath(attr)
                    if not spec:
                        raise ValueError(f"No authored attribute {attr}")
                    if spec.default != value:
                        spec.default = value
                        for suffix, key in ((".xformOp:translate", "body_translation_indices"),
                                            (".xformOp:orient", "body_rotation_indices")):
                            if attr.startswith("/Lab/body_") and attr.endswith(suffix):
                                changed[key].append(int(attr.split("/Lab/body_", 1)[1].split(".", 1)[0]))
            # the paste surfaces: replace each mesh with the request's (or hide it when the request has none)
            meshes = req.get("meshes", {})
            for name in meshes:
                path = "/Paste/"+name
                if not stage.GetPrimAtPath(path):
                    g = UsdGeom.Mesh.Define(stage, path)
                    g.CreatePointsAttr(); g.CreateFaceVertexCountsAttr(); g.CreateFaceVertexIndicesAttr(); g.CreateExtentAttr()
                    pv = UsdGeom.PrimvarsAPI(g)
                    pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, "vertex")
                    pv.CreatePrimvar("gloss", Sdf.ValueTypeNames.FloatArray, "vertex")
                    pv.CreatePrimvar("tex", Sdf.ValueTypeNames.FloatArray, "vertex")
                    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(UsdShade.Material.Get(stage, "/Looks/paste"))
            for prim in stage.GetPrimAtPath("/Paste").GetChildren():
                name = prim.GetName()
                g = UsdGeom.Mesh(prim)
                m = meshes.get(name)
                if not m:
                    UsdGeom.Imageable(prim).MakeInvisible()
                    continue
                if m.get("same"):                          # unchanged since the last frame: left as it is
                    UsdGeom.Imageable(prim).MakeVisible()
                    continue
                v, t = array(m, "v", np.float32, 3), array(m, "t", np.int32, 3)
                if not (len(v) and len(t)):
                    UsdGeom.Imageable(prim).MakeInvisible()
                    continue
                if int(t.max()) >= len(v) or int(t.min()) < 0:
                    raise ValueError(f"Paste mesh {name}: triangle index out of range")
                pv = UsdGeom.PrimvarsAPI(g)
                with Sdf.ChangeBlock():
                    g.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v))
                    # the topology only when it changed (a reset rebuilds the whole surface in the renderer)
                    tkey = (len(t), hash(t.tobytes()))
                    if topology.get(name) != tkey:
                        g.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(t), 3, np.int32)))
                        g.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.ascontiguousarray(t.ravel())))
                        topology[name] = tkey
                    g.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray([v.min(0), v.max(0)], np.float32)))
                    if "n" in m:
                        g.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(array(m, "n", np.float32, 3)))
                        g.SetNormalsInterpolation("vertex")
                    elif g.GetNormalsAttr():
                        g.GetNormalsAttr().Clear()
                    c = array(m, "c", np.float32, 3) if "c" in m else np.tile(np.float32([.8, .75, .6]), (len(v), 1))
                    gl = array(m, "g", np.float32, 1) if "g" in m else np.zeros(len(v), np.float32)
                    tx = array(m, "x", np.float32, 1) if "x" in m else np.zeros(len(v), np.float32)
                    if not (len(c) == len(gl) == len(tx) == len(v)):
                        raise ValueError(f"Paste mesh {name}: primvar count differs from the vertex count")
                    pv.GetPrimvar("displayColor").Set(Vt.Vec3fArray.FromNumpy(c))
                    pv.GetPrimvar("gloss").Set(Vt.FloatArray.FromNumpy(gl))
                    pv.GetPrimvar("tex").Set(Vt.FloatArray.FromNumpy(tx))
                UsdGeom.Imageable(prim).MakeVisible()
            return changed

        def capture(pr, subframes, warm=False):
            ann = pr["annotator"]
            step_s = readback_s = 0.0
            for attempt in range(8):
                t_step = time.perf_counter()
                rep.orchestrator.step(rt_subframes=(max(8, subframes) if warm else subframes), delta_time=0.)
                step_s += time.perf_counter() - t_step
                t_read = time.perf_counter()
                raw = ann.get_data()
                data = np.asarray(raw.numpy() if hasattr(raw, "numpy") else raw)
                readback_s += time.perf_counter() - t_read
                if data.ndim == 3 and data.shape[:2] == (pr["res"][1], pr["res"][0]) and float(data[..., :3].max()) > 0:
                    return data, dict(step_s=step_s, readback_s=readback_s, attempts=attempt + 1)
            raise RuntimeError("No valid readback after 8 render-only attempts")

        send(connection, {"phase": "ready"})
        warmed = set()
        last_camera_position = {}
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
                changed = apply(req)
                timings["apply_s"] = time.perf_counter() - t0
                timings["meshes_full"] = sum(not m.get("same") for m in req.get("meshes", {}).values())
                timings["meshes_reused"] = sum(bool(m.get("same")) for m in req.get("meshes", {}).values())
                images = {}
                camera_motion = {}
                for name in req["cameras"]:
                    res = tuple(req.get("resolutions", {}).get(name) or req.get("resolution") or (a.width, a.height))
                    pr = product(f"ldr_{res[0]}x{res[1]}", camera_path(name), res)
                    if os.environ.get("CF_RENDER_PROFILE") == "1":
                        transform = UsdGeom.Xformable(stage.GetPrimAtPath(camera_path(name))).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        pos = np.asarray(transform.ExtractTranslation(), dtype=float)
                        previous = last_camera_position.get(name)
                        camera_motion[name] = None if previous is None else float(np.linalg.norm(pos - previous))
                    key = (name, res)
                    t1 = time.perf_counter()
                    rgb, detail = capture(pr, int(req.get("subframes", 2)), warm=key not in warmed)
                    if os.environ.get("CF_RENDER_PROFILE") == "1":
                        last_camera_position[name] = pos
                    timings[name + "_detail"] = detail
                    warmed.add(key)
                    timings[name + "_s"] = time.perf_counter() - t1
                    out = io.BytesIO()
                    Image.fromarray(white_balance(np.ascontiguousarray(rgb[:, :, :3]))).save(out, format="PNG", compress_level=1)
                    images[name] = base64.b64encode(out.getvalue()).decode()
                timings["total_s"] = time.perf_counter() - t0
                if os.environ.get("CF_RENDER_PROFILE") == "1":
                    print("CF_RENDER_TIMING " + json.dumps(dict(device=readback_device, render_mode=render_mode,
                                                              effective_render_mode=settings.get("/rtx/rendermode"),
                                                              pose_filter=pose_filter, pose_updates=changed,
                                                              camera_translation_since_previous_frame_m=camera_motion,
                                                              tick=req["tick"], timings=timings)), flush=True)
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
