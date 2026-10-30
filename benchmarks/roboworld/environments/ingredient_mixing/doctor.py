"""`gar doctor ingredient-mixing`: what this environment needs beyond Newton/Warp/MuJoCo (no simulation).

The mixture runs on a CUDA GPU (Warp); Ruckig plans the arms' joint trajectories; scipy builds the vessels'
signed-distance tables; the Isaac render interpreter serves every RGB observation (policy cameras) and the evidence
films; PIL and ffmpeg make the films. The room's Poly Haven textures and models are the hash-checked pins of the
sibling packages barcode_scanning, drilling, wine_pouring and gift_bags (dressing.PINNED), fetched there on first use;
the sky behind the window is whiteboard's pinned panorama (kitchen.daylight).
"""
import importlib.util
import os
import shutil
from pathlib import Path


def checks(spec):
    out = []

    def add(name, ok, detail):
        out.append(dict(check=name, ok=bool(ok), detail=detail))
    for mod, req in (("scipy", "scipy"), ("PIL", "pillow"), ("warp", "warp-lang>=1.16"), ("ruckig", "ruckig (arm trajectories)"),
                     ("mujoco", "mujoco>=3.11")):
        add(f"import {mod}", importlib.util.find_spec(mod) is not None, req)
    try:
        import warp as wp
        wp.config.quiet = True
        wp.init()
        add("CUDA device for the mixture", wp.is_cuda_available(),
            ", ".join(str(d) for d in wp.get_cuda_devices()) or "no CUDA device")
    except Exception as exc:                       # noqa: BLE001 - report, do not raise
        add("CUDA device for the mixture", False, f"{type(exc).__name__}: {exc}")
    isaac = os.environ.get("ISAAC_RENDER_PYTHON")
    add("ISAAC_RENDER_PYTHON (RGB observations, evidence films)", bool(isaac) and Path(isaac).is_file(),
        isaac or "unset: point it at Isaac Sim's python "
                 "(sessions with cameras=() run without it)")
    add("ffmpeg (evidence films)", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "not on PATH")
    here = Path(__file__).resolve().parent
    add("disk for session traces (>= 1 GiB free)", shutil.disk_usage(here).free > 1024**3,
        f"{shutil.disk_usage(here).free/1024**3:.1f} GiB free")
    try:
        from .dressing import PINNED
        need = tuple(PINNED) + ("whiteboard",)                  # whiteboard: the sky behind the window
        missing = [p for p in need if importlib.util.find_spec(f"{p}.visual_assets") is None]
        add("room dressing assets (pins of sibling packages)", not missing,
            "missing: " + ", ".join(missing) if missing else "pins of " + ", ".join(need))
    except Exception as exc:                       # noqa: BLE001
        add("room dressing assets (pins of sibling packages)", False, f"{type(exc).__name__}: {exc}")
    try:
        from robots.catalog import for_environment
        robot = spec.robot if isinstance(getattr(spec, "robot", None), str) else None
        rig = for_environment("ingredient-mixing", robot)
        add("robot setup", True, f"{rig.id}")
    except Exception as exc:                       # noqa: BLE001
        add("robot setup", False, f"{type(exc).__name__}: {str(exc)[:200]}")
    return dict(ok=all(c_["ok"] for c_ in out), checks=out,
                note="GPU required for the mixture; the render interpreter for RGB observations")
