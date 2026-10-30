"""Headless Isaac RTX observation worker. No dynamics scene or timeline play."""

import argparse
import base64
import io
import os
from pathlib import Path
import socket
import traceback
from whiteboard.ipc import send, receive


def main():
    from gym_anything_robotics.render_lifecycle import watch_parent
    watch_parent()
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    for name in ("width", "height", "samples"):
        p.add_argument("--" + name, type=int, required=True)
    a = p.parse_args()
    conn = socket.create_connection(
        ("127.0.0.1", int(os.environ["FOOTBALL_RENDER_PORT"]))
    )
    send(conn, {"secret": os.environ.pop("FOOTBALL_RENDER_SECRET")})
    folder = Path(os.environ["FOOTBALL_RENDER_SESSION"]) / "kit"
    folder.mkdir(exist_ok=True)
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "sync_loads": True,
            "renderer": "PathTracing",
            "width": a.width,
            "height": a.height,
            "samples_per_pixel_per_frame": a.samples,
            "max_bounces": 8,
            "multi_gpu": False,
            "active_gpu": 0,
            "extra_args": [
                "--portable",
                f"--portable-root={folder}",
                f"--/log/file={folder / 'worker.log'}",
                "--/app/enableCrashReporter=false",
                # Apply before opening the stage: streaming otherwise exposes
                # low-resolution texture mips in the first observation.
                "--/rtx-transient/resourcemanager/enableTextureStreaming=false",
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
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdPhysics, UsdRender, Vt
        from whiteboard.render_readback import capture_rgb

        settings = carb.settings.get_settings()
        for key, value in {
            "/rtx/post/histogram/enabled": False,
            "/rtx/pathtracing/spp": a.samples,
            "/rtx/pathtracing/totalSpp": a.samples,
            "/rtx/pathtracing/optixDenoiser/enabled": True,
            "/rtx/post/tonemap/op": 6,
            "/rtx/post/tonemap/filmIso": 200.0,
            "/rtx/post/tonemap/fNumber": 4.0,
            "/rtx/post/tonemap/exposureTime": 1 / 120.0,
        }.items():
            settings.set(key, value)
        for _ in range(12):
            app.update()
        omni.timeline.get_timeline_interface().pause()
        stage = omni.usd.get_context().get_stage()
        if any(
            prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Scene)
            for prim in stage.Traverse()
        ):
            raise RuntimeError("Render-only scene contains physics")
        rep.orchestrator.set_capture_on_play(False)
        product, annotator = None, None
        warmed = False
        send(conn, {"phase": "ready"})
        # Edge indices are exported as a sidecar to preserve the exact physical graph.
        import json

        topology = json.loads((Path(a.scene).parent / "nets.json").read_text())
        while True:
            request = receive(conn)
            if request["op"] == "close":
                break
            try:
                if request["op"] != "observe":
                    raise ValueError("Unknown render operation")
                for i, q in enumerate(request["poses"]):
                    prim = stage.GetPrimAtPath(f"/Lab/body_{i}")
                    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*q[:3]))
                    prim.GetAttribute("xformOp:orient").Set(
                        Gf.Quatf(q[6], Gf.Vec3f(*q[3:6]))
                    )
                for name, positions in request["nets"].items():
                    points = np.asarray(positions, np.float32)[
                        np.asarray(topology[name], np.int32)
                    ].reshape(-1, 3)
                    UsdGeom.BasisCurves(
                        stage.GetPrimAtPath("/Lab/net_" + name)
                    ).GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
                surface = request["ball_surface"]
                deformation = surface["state"]
                matrix = np.eye(4)
                matrix[:3, :3] = np.asarray(
                    deformation["deformation_gradient_ball_local"]
                ).T
                matrix[3, :3] = deformation["translation_ball_local_m"]
                stage.GetPrimAtPath(
                    f"/Lab/body_{surface['body']}/surface"
                ).GetAttribute("xformOp:transform").Set(
                    Gf.Matrix4d(*matrix.reshape(-1))
                )
                if "optical_pose" in request:
                    eye, look, up = request["optical_pose"]
                    stage.GetPrimAtPath("/Cameras/g1_rgb").GetAttribute(
                        "xformOp:transform"
                    ).Set(
                        Gf.Matrix4d()
                        .SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(*up))
                        .GetInverse()
                    )
                images, retries = {}, {}
                for name in request["cameras"]:
                    path = "/Cameras/" + name
                    cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
                    if not cam:
                        raise ValueError("Unknown camera")
                    cam.GetVerticalApertureAttr().Set(
                        cam.GetHorizontalApertureAttr().Get() * a.height / a.width
                    )
                    if product is None:
                        product = rep.create.render_product(path, (a.width, a.height))
                        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                        annotator.attach([product])
                    else:
                        UsdRender.Product(
                            stage.GetPrimAtPath(product.path)
                        ).GetCameraRel().SetTargets([path])
                    pixels, retries[name] = capture_rgb(
                        lambda: rep.orchestrator.step(rt_subframes=2, delta_time=0.0),
                        annotator.get_data,
                        a.width,
                        a.height,
                        flush_attempts=6 if not warmed else 1,
                    )
                    warmed = True
                    buf = io.BytesIO()
                    Image.fromarray(pixels).save(buf, format="PNG")
                    images[name] = base64.b64encode(buf.getvalue()).decode()
                send(
                    conn,
                    {
                        "ok": True,
                        "sequence": request["sequence"],
                        "tick": request["tick"],
                        "physics_steps": 0,
                        "images": images,
                        "readback_retries": retries,
                        "width": a.width,
                        "height": a.height,
                        "samples": a.samples,
                    },
                )
            except Exception as exc:
                traceback.print_exc()
                send(conn, {"ok": False, "error": str(exc)})
        if product is not None:
            annotator.detach()
            product.destroy()
    finally:
        conn.close()
        app.close()


if __name__ == "__main__":
    main()
