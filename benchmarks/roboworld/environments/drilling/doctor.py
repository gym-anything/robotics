"""`gar doctor drilling`: what this environment needs beyond Newton/Warp/MuJoCo (no simulation is started).

Physics, the evaluator and camera-less sessions need ruckig (the arms' jerk-limited trajectory generator: drilling/world.py
imports it at module level, so without it no session starts), scipy (rotations, the inverse-kinematics least squares)
and the locked FR3 and Panda descriptions of the shared robot catalogue. RGB observations (the default session has two
cameras) also need the Isaac render interpreter (ISAAC_RENDER_PYTHON; drilling/render.py starts
`python -m drilling.render_worker` with it), usd-core (pxr: the scene export), manifold3d (the drilled workpieces'
meshes), trimesh (normals), pillow (the worker's white balance and the generated textures), the worker's modules
beside drilling, the pinned dressing assets and about 4.6 GB of free GPU memory per session. A camera-less session
(`--backend-options-json '{"cameras": []}'`) runs without any of the second group; its physics is on the CPU, and where
a GPU is present its process still opens a CUDA context of about 0.5 GB (532 MiB per process measured with nvidia-smi:
Warp initialises the device it finds).
"""
from gym_anything_robotics.paths import project_file
import importlib.util
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# files the Isaac render worker (python -m drilling.render_worker, PYTHONPATH = gym_anything_robotics.paths.worker_path()) loads at start-up
WORKER_FILES = ("drilling/render_worker.py", "whiteboard/ipc.py", "whiteboard/usd_scene.py",
                "gym_anything_robotics/render_lifecycle.py")
RENDER_GPU_MIB = 4600.            # one render worker with the two default cameras (measured: 4.2-4.6 GB)
PHYSICS = (("ruckig", "ruckig>=0.19 (the arms' jerk-limited trajectory generator; no session starts without it)"),
           ("scipy", "scipy (rotations, the inverse-kinematics solve, the evaluator)"),
           ("mujoco", "mujoco (collision queries of tools, parts and holes)"))
IMAGES = (("pxr", "usd-core (pxr: the render scene export, drilling/render.py)"),
          ("manifold3d", "manifold3d (the meshes of drilled workpieces, burrs and dust)"),
          ("trimesh", "trimesh (vertex normals of the exported meshes)"),
          ("PIL", "pillow (the render worker's white balance and the generated textures)"))


def gpu_check(query=None):
    """(ok, detail): is there GPU memory for one render worker? `query`: () -> (free MiB, total MiB) for tests;
    default nvidia-smi. Camera-less sessions need no GPU: the physics runs on the CPU (see the module docstring)."""
    try:
        if query is None:
            import subprocess
            line = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10, check=True).stdout.splitlines()[0]
            free, total = (float(v) for v in line.split(","))
        else:
            free, total = query()
    except Exception as exc:                              # noqa: BLE001 - report, do not raise
        return False, (f"no GPU could be queried ({type(exc).__name__}): RGB observations need one; camera-less "
                       "sessions (cameras: []) do not")
    ok = free >= RENDER_GPU_MIB
    return ok, (f"{free:.0f} of {total:.0f} MiB free; one session's render worker takes about {RENDER_GPU_MIB:.0f} MiB"
                + ("" if ok else ": a session with cameras would fail to draw its first image; pause or stop other "
                                 "sessions, or run camera-less"))


def checks(spec, environ=os.environ, find=importlib.util.find_spec, gpu=None):
    out = []

    def add(name, ok, detail):
        out.append(dict(check=name, ok=bool(ok), detail=detail))
    for mod, req in PHYSICS:
        add(f"import {mod}", find(mod) is not None, req)
    try:
        from robots.catalog import ENVIRONMENTS, model, resolve
        item = ENVIRONMENTS["drilling"]
        robot = spec.robot if isinstance(getattr(spec, "robot", None), str) else item["default"]
        rig = item["aliases"].get(robot, robot)
        setup = resolve(rig).to_dict()
        for inst in setup["instances"].values():
            profile = model(inst["model"]).to_dict()["actuation"]["profiles"][inst["controller_profile"]]
            fingers = [v["effort_limit"] for k, v in profile["joints"].items() if "finger" in k]
        hand = model("franka-hand@1").to_dict()["contact"]["fingertip_pads"]["finger_frame_offset_z_m"]
        add(f"robot catalogue entry for {robot}", True,
            f"rig {rig}: {', '.join(sorted({i['model'] for i in setup['instances'].values()}))} with the Franka Hand "
            f"({fingers[0]:g} N per finger, pads at {hand*1e3:g} mm)")
    except Exception as exc:                              # noqa: BLE001
        add("robot catalogue entry", False, f"{type(exc).__name__}: {exc}")
    for mod, req in IMAGES:
        add(f"import {mod} (RGB observations only)", find(mod) is not None, req)
    isaac = environ.get("ISAAC_RENDER_PYTHON")
    add("ISAAC_RENDER_PYTHON (RGB observations only)", bool(isaac) and Path(isaac).is_file(),
        (isaac if isaac and Path(isaac).is_file() else
         f"{isaac}: no such file" if isaac else
         "unset: point it at Isaac Sim's python (sessions with "
         "cameras: [] run without it)"))
    missing = [f for f in WORKER_FILES if not project_file(f).is_file()]
    add("render worker modules beside drilling (RGB observations only)", not missing,
        f"{ROOT}: " + (f"missing {', '.join(missing)} (install the full gym-anything-robotics package)" if missing
                       else "drilling.render_worker, whiteboard.ipc, whiteboard.usd_scene, "
                            "gym_anything_robotics.render_lifecycle present"))
    try:
        from . import visual_assets as va
        add("pinned visual assets (workshop dressing, RGB observations only)", va.MANIFEST.is_file(),
            f"pins {va.MANIFEST.name}; cache {va.CACHE}" + ("" if va.CACHE.is_dir() else
                                                           " (empty: fetched and hash-checked on the first render, "
                                                           "which needs network access once)"))
    except Exception as exc:                              # noqa: BLE001
        add("pinned visual assets (workshop dressing, RGB observations only)", False, f"{type(exc).__name__}: {exc}")
    add("GPU memory for one render worker (RGB observations only)", *gpu_check(gpu))
    workspace = Path(environ.get("GAR_WORKSPACE") or Path.cwd())
    try:
        free = shutil.disk_usage(workspace).free/1024**3
        add("disk for session traces (>= 2 GiB free)", free >= 2.,
            f"{free:.1f} GiB free at {workspace}: a session records its physics trace (up to 512 MiB kept per session) "
            "and action history under .gar/sessions/")
    except OSError as exc:
        add("disk for session traces (>= 2 GiB free)", False, f"{workspace}: {exc}")
    return dict(ok=all(c["ok"] for c in out), checks=out,
                note="RGB observations need the Isaac render interpreter, usd-core, manifold3d, trimesh, pillow and a "
                     "GPU; physics, evaluation and camera-less sessions need only ruckig, scipy and the robot catalogue")
