"""`gar doctor wine-pouring`: what this environment needs beyond Newton/Warp/MuJoCo (no simulation).

The liquid runs on a CUDA GPU (Warp); scipy builds the vessel signed-distance and density maps; the Isaac render
interpreter serves every RGB observation (policy cameras) and the videos; PIL draws labels and marks. Density
maps are cached in wine_pouring/.cache (or WINE_POURING_CACHE): that folder must be writable (about 7 s per container
kind to rebuild). The bar dressing (Poly Haven models and textures) is fetched on first use and hash-checked against
visual_assets.json, which needs network access once.
"""
import importlib.util
import os
import shutil
from pathlib import Path


def cache_dir():
    return Path(os.environ.get("WINE_POURING_CACHE") or Path(__file__).resolve().parent/".cache")


def checks(spec):
    out = []

    def add(name, ok, detail):
        out.append(dict(check=name, ok=bool(ok), detail=detail))
    for mod, req in (("scipy", "scipy"), ("PIL", "pillow"), ("warp", "warp-lang>=1.16")):
        add(f"import {mod}", importlib.util.find_spec(mod) is not None, req)
    try:
        import warp as wp
        wp.config.quiet = True
        wp.init()
        cuda = wp.is_cuda_available()
        add("CUDA device for the liquid", cuda, ", ".join(str(d) for d in wp.get_cuda_devices()) or "no CUDA device")
    except Exception as exc:                       # noqa: BLE001 - report, do not raise
        add("CUDA device for the liquid", False, f"{type(exc).__name__}: {exc}")
    isaac = os.environ.get("ISAAC_RENDER_PYTHON")
    add("ISAAC_RENDER_PYTHON (RGB observations, videos)", bool(isaac) and Path(isaac).is_file(),
        isaac or "unset: point it at Isaac Sim's python "
                 "(sessions with cameras=() run without it)")
    c = cache_dir()
    try:
        c.mkdir(parents=True, exist_ok=True)
        probe = c/".write_test"
        probe.write_text("ok")
        probe.unlink()
        add("density-map cache writable", True, str(c))
    except OSError as exc:
        add("density-map cache writable", False, f"{c}: {exc} (set WINE_POURING_CACHE to a writable folder)")
    add("disk for session traces (>= 1 GiB free)", shutil.disk_usage(c if c.exists() else Path.cwd()).free > 1024**3,
        f"{shutil.disk_usage(c if c.exists() else Path.cwd()).free/1024**3:.1f} GiB free")
    fonts = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
    add("DejaVu fonts (bottle labels, fill-line marks)", fonts.is_file(), str(fonts))
    robot = spec.robot if isinstance(spec.robot, str) else None
    if robot in ("dual-panda", "pour-panda"):
        from robots.catalog import model
        try:
            m = model("panda@1")
            add("Panda description (dual-panda)", True, str(getattr(m, "source", "panda@1")))
        except Exception as exc:                   # noqa: BLE001
            add("Panda description (dual-panda)", False, f"{type(exc).__name__}: {exc}")
    try:
        from . import visual_assets as va
        add("bar dressing assets (Poly Haven, fetched once)", True, f"cache {getattr(va, 'CACHE', 'see visual_assets.py')}")
    except Exception as exc:                       # noqa: BLE001
        add("bar dressing assets (Poly Haven, fetched once)", False, f"{type(exc).__name__}: {exc}")
    return dict(ok=all(c_["ok"] for c_ in out), checks=out,
                note="GPU required for the liquid; the render interpreter for RGB observations")
