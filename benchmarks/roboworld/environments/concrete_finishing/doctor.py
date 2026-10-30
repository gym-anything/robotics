"""`gar doctor concrete-finishing`: what this environment needs beyond Newton/Warp/MuJoCo (no simulation).

Concrete finishing is built on the classroom's rigid world and robot (CPU MuJoCo), its wet materials are numpy
height fields (no GPU solver), and its render worker draws the room with drilling's pinned Poly Haven models and
textures. Nothing here starts physics or a renderer: packages are looked up without being imported, cached files are
hashed against their pins, the Isaac interpreter is asked whether it has `isaacsim` and nothing else.

A failed check (ok false) means a session cannot start or cannot draw. A check that passes with a remark (a model not
cached yet, no ffmpeg) names something a first render downloads or only the films need.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKER_GPU_MIB = 3850           # an Isaac render-only worker of this kind of room (sweeping's measured 3,835 MiB)
MODELS = {"mobile-fr3-duo": "mobile-fr3-duo", "mobile-fr3-single": "mobile-fr3-single",
          "finish-mobile-fr3-duo": "mobile-fr3-duo", "finish-mobile-fr3-single": "mobile-fr3-single"}
DRESSING = ("mounted_fluorescent_lights", "power_box_01", "cement_bag", "plastic_crate_02", "WetFloorSign_01", "plastic_broom",
            "korean_fire_extinguisher_01")
TEXTURES = ("concrete_block_wall", "painted_plaster_wall", "concrete", "brushed_concrete", "concrete_floor_worn_001", "plywood")


def checks(spec):
    out = []

    def add(name, ok, detail):
        out.append(dict(check=name, ok=bool(ok), detail=str(detail)))

    for mod, why in (("numpy", "numpy (height fields)"), ("scipy", "scipy (rotations, distance transforms, inverse kinematics)"),
                     ("PIL", "pillow (textures, films)"), ("trimesh", "trimesh (robot meshes)"), ("mujoco", "mujoco (rigid world)"),
                     ("newton", "newton 1.5"), ("warp", "warp-lang (Newton's kernels on the CPU)"),
                     ("OpenGL", "software RGB rasterizer"), ("pxr", "exported USD scene reader")):
        add(f"import {mod}", importlib.util.find_spec(mod) is not None, why)
    for pkg, why in (("classroom_arrangement", "rigid world, robot controllers and inverse kinematics"),
                     ("sweeping", "the render worker's lifecycle and the robot's drawing (sweeping.render.robot_visual)"),
                     ("drilling", "USD stage helper, white balance, pinned room models and textures, procedural services"),
                     ("whiteboard", "render-worker message framing (whiteboard.ipc)"),
                     ("pencap", "geometry catalogue behind GET /v1/geometry (pencap.newton15.catalog)"),
                     ("robots", "robot catalogue, rigs and locked robot assets"),
                     ("gym_anything_robotics", "backend, registry and the render-worker lifecycle")):
        add(f"package {pkg}", importlib.util.find_spec(pkg) is not None, why)
    kit = REPO/"wiping"/"render_only.kit"
    add("render-only kit file (wiping/render_only.kit)", kit.is_file(), kit)
    mdl = REPO/"concrete_finishing"/"ConcreteFinishing.mdl"
    add("paste material (ConcreteFinishing.mdl)", mdl.is_file(), mdl)

    robot = spec.robot if isinstance(getattr(spec, "robot", None), str) else None
    try:
        from robots.catalog import for_environment
        rig = for_environment("concrete-finishing", robot)
        add("robot setup", True, f"{rig.id} ({'default' if robot is None else robot})")
    except Exception as exc:                       # noqa: BLE001
        add("robot setup", False, f"{type(exc).__name__}: {str(exc)[:300]}")
    for model in ([MODELS[robot]] if robot in MODELS else ["mobile-fr3-duo", "mobile-fr3-single"]):
        try:
            from robots import assets
            add(f"{model} robot assets (locked hashes)", True, assets.resolve(model))
        except Exception as exc:                   # noqa: BLE001
            add(f"{model} robot assets (locked hashes)", False, str(exc)[:300])

    isaac = os.environ.get("ISAAC_RENDER_PYTHON")
    present = bool(isaac) and Path(isaac).is_file()
    add("Mesa software EGL (default RGB renderer)", Path('/usr/share/glvnd/egl_vendor.d/50_mesa.json').is_file(),
        "CPU rasterizer of the exported scene; no GPU allocation")
    add("ISAAC_RENDER_PYTHON (optional RTX renderer)", True,
        isaac or "unset: software RGB remains available; source environment.sh for optional RTX")
    if present:
        try:
            q = subprocess.run([isaac, "-c", "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('isaacsim') else 3)"],
                               capture_output=True, text=True, timeout=60)
            add("isaacsim in the render interpreter", q.returncode == 0,
                "found" if q.returncode == 0 else f"not importable there (exit {q.returncode}): {q.stderr.strip()[-200:]}")
        except Exception as exc:                   # noqa: BLE001
            add("isaacsim in the render interpreter", False, f"{type(exc).__name__}: {exc}")
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits", "--id=0"],
                           capture_output=True, text=True, timeout=20)
        name, total, used = (x.strip() for x in q.stdout.strip().splitlines()[0].split(","))
        free = int(total) - int(used)
        add("GPU memory free for an optional RTX worker (advisory)", True,
            f"{name}: {free} MiB free of {total}; a render worker holds about {WORKER_GPU_MIB} MiB; the physics runs on the CPU "
            "(a session with cameras=[] needs no GPU)")
    except Exception as exc:                       # noqa: BLE001
        add("GPU memory free for a render worker", True, f"not read ({type(exc).__name__}: {exc})")
    # drilling's pinned models and textures (visual only): cached files are hashed, missing ones are downloaded and
    # hashed by the first render
    try:
        from drilling import visual_assets as va
        man = json.loads(va.MANIFEST.read_text())
        for name in DRESSING + TEXTURES:
            entry = man.get(name)
            if entry is None:
                add(f"pinned visual asset {name}", False, "not pinned in drilling/visual_assets.json")
                continue
            folder = va.CACHE/name
            files = list(entry["files"].items())
            cached = [rel for rel, _ in files if (folder/rel).is_file()]
            if len(cached) < len(files):
                add(f"pinned visual asset {name}", True, f"{len(cached)} of {len(files)} files cached; the first render downloads "
                                                        "the rest from Poly Haven (network) and checks their hashes")
                continue
            bad = [rel for rel, f in files if hashlib.sha256((folder/rel).read_bytes()).hexdigest() != f["sha256"]]
            add(f"pinned visual asset {name}", not bad, "cached, hashes ok" if not bad else f"HASH DIFFERS: {bad[:3]}")
    except Exception as exc:                       # noqa: BLE001
        add("pinned visual assets (drilling.visual_assets)", False, f"{type(exc).__name__}: {exc}")
    ffmpeg = shutil.which("ffmpeg")
    add("ffmpeg (evidence films only)", True, ffmpeg or "not on PATH: sessions run without it; the probes' films do not")
    return dict(ok=all(c["ok"] for c in out), checks=out,
                note="The physics runs on the CPU (rigid world and paste fields). Routine RGB uses Mesa software OpenGL in the physics runtime; "
                     "the Isaac interpreter is needed only when that optional renderer is selected. Robot assets are verified against their locked hashes; "
                     "drilling's pinned CC0 models and textures are downloaded on the first render and hash-checked")
