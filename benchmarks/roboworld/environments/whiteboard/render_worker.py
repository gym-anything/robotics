"""Render-only worker: persistent stage, physical-pose updates, in-memory ink texture."""

import argparse
import base64
import io
import os
import socket
import traceback
from .ipc import send, receive
from .render_readback import capture_rgb


def main():
    from gym_anything_robotics.render_lifecycle import watch_parent
    watch_parent()
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--samples", type=int, required=True)
    a = p.parse_args()
    conn = socket.create_connection(("127.0.0.1", int(os.environ["WHITEBOARD_RENDER_PORT"])))
    send(conn, {"secret": os.environ.pop("WHITEBOARD_RENDER_SECRET")})
    from .runtime import configure

    cache = configure() / "kit_live"
    cache.mkdir(exist_ok=True)
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "renderer": "PathTracing",
            "width": a.width,
            "height": a.height,
            "samples_per_pixel_per_frame": a.samples,
            "max_bounces": 6,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
            "extra_args": [
                "--/rtx-transient/resourcemanager/enableTextureStreaming=false",
                "--portable",
                f"--portable-root={cache}",
                "--/app/enableCrashReporter=false",
                f"--/log/file={cache / 'worker.log'}",
            ],
        }
    )
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
        import omni.ui as ui
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Gf, Sdf, UsdGeom, UsdShade, UsdRender

        settings = carb.settings.get_settings()
        for name, value in {
            "/rtx/post/histogram/enabled": False,
            "/rtx/pathtracing/spp": a.samples,
            "/rtx/pathtracing/totalSpp": a.samples,
            "/rtx/pathtracing/optixDenoiser/enabled": True,
            "/rtx/post/tonemap/op": 6,
            "/rtx/post/tonemap/filmIso": 400.0,
            "/rtx/post/tonemap/fNumber": 2.8,
            "/rtx/post/tonemap/exposureTime": 1 / 120.0,
        }.items():
            settings.set(name, value)
        for _ in range(12):
            app.update()
        omni.timeline.get_timeline_interface().pause()
        stage = omni.usd.get_context().get_stage()
        if any(
            prim.HasAPI(__import__("pxr.UsdPhysics", fromlist=["RigidBodyAPI"]).RigidBodyAPI)
            for prim in stage.Traverse()
        ):
            raise RuntimeError("Render scene contains a rigid-body physics API")
        provider = ui.DynamicTextureProvider("whiteboard_live_ink")
        dynamic_bound = False
        product, annotator = None, None
        rep.orchestrator.set_capture_on_play(False)
        send(conn, {"phase": "ready"})
        while True:
            request = receive(conn)
            if request["op"] == "close":
                break
            try:
                if request["op"] != "observe":
                    raise ValueError("Unknown renderer operation")
                for path, pose in request["poses"].items():
                    prim = stage.GetPrimAtPath(path)
                    if not prim:
                        raise ValueError(f"Missing physical render prim: {path}")
                    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*pose[:3]))
                    prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(pose[6], Gf.Vec3f(*pose[3:6])))
                for path, points in request.get("elastic_vertices", {}).items():
                    material = UsdGeom.Mesh(stage.GetPrimAtPath(path))
                    if not material:
                        raise ValueError(f"Missing elastic material mesh: {path}")
                    material.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
                if "optical_pose" in request:
                    eye, look, up = request["optical_pose"]
                    stage.GetPrimAtPath("/Cameras/g1_rgb").GetAttribute("xformOp:transform").Set(
                        Gf.Matrix4d()
                        .SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(*up))
                        .GetInverse()
                    )
                if "board_png" in request:
                    pixels = np.asarray(
                        Image.open(io.BytesIO(base64.b64decode(request["board_png"]))).convert(
                            "RGBA"
                        )
                    )
                    provider.set_data_array(
                        np.ascontiguousarray(pixels), [pixels.shape[1], pixels.shape[0]]
                    )
                    if not dynamic_bound:
                        shader = UsdShade.Shader(
                            stage.GetPrimAtPath("/Materials/board/rtx_surface")
                        )
                        shader.GetInput("diffuse_texture").Set(
                            Sdf.AssetPath("dynamic://whiteboard_live_ink")
                        )
                        dynamic_bound = True
                frames, readback_retries = {}, {}
                for name in request["cameras"]:
                    camera = UsdGeom.Camera(stage.GetPrimAtPath("/Cameras/" + name))
                    if not camera:
                        raise ValueError("Unknown camera")
                    camera.GetVerticalApertureAttr().Set(
                        camera.GetHorizontalApertureAttr().Get() * a.height / a.width
                    )
                    if product is None:
                        product = rep.create.render_product("/Cameras/" + name, (a.width, a.height))
                        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                        annotator.attach([product])
                    else:
                        UsdRender.Product(
                            stage.GetPrimAtPath(product.path)
                        ).GetCameraRel().SetTargets(["/Cameras/" + name])
                    pixels, readback_retries[name] = capture_rgb(
                        lambda: rep.orchestrator.step(rt_subframes=2, delta_time=0.0),
                        annotator.get_data,
                        a.width,
                        a.height,
                    )
                    buf = io.BytesIO()
                    Image.fromarray(pixels).save(buf, format="PNG")
                    frames[name] = base64.b64encode(buf.getvalue()).decode()
                send(
                    conn,
                    {
                        "ok": True,
                        "sequence": request["sequence"],
                        "tick": request["tick"],
                        "physics_steps": 0,
                        "width": a.width,
                        "height": a.height,
                        "samples": a.samples,
                        "encoding": "png/base64",
                        "images": frames,
                        "readback_retries": readback_retries,
                    },
                )
            except Exception as exc:
                send(conn, {"ok": False, "error": str(exc)})
                traceback.print_exc()
        if product is not None:
            annotator.detach()
            product.destroy()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        conn.close()
        app.close()


if __name__ == "__main__":
    main()
