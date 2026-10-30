"""`gar doctor percussion`: what this environment needs beyond Newton/Warp/MuJoCo (no simulation).

The physics runs on the CPU (MuJoCo through Newton's SolverMuJoCo). scipy builds the audio observation (resample_poly),
the A-weighting filter and the modal tables; the instruments' mode tables ship in percussion/data with sha256 hashes in
data/MANIFEST.json. The Isaac render interpreter (ISAAC_RENDER_PYTHON) serves every RGB observation and the
videos; ffmpeg encodes the videos. The room dressing (pinned CC0 Poly Haven models and textures) is fetched on first use
into percussion/.cache and hash-checked against visual_assets.json, which needs network access once.
"""
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

PKG = Path(__file__).resolve().parent


def checks(spec):
    out = []

    def add(name, ok, detail):
        out.append(dict(check=name, ok=bool(ok), detail=detail))
    for mod, req in (("scipy", "scipy (audio observation resampling, A-weighting, modal tables)"),
                     ("PIL", "pillow (video captions, dashboard images)"), ("mujoco", "mujoco>=3.3 (CPU physics)")):
        add(f"import {mod}", importlib.util.find_spec(mod) is not None, req)
    man = PKG/"data"/"MANIFEST.json"
    try:
        files = json.loads(man.read_text())["files"]
        bad = [n for n, h in files.items()
               if not (PKG/"data"/n).is_file() or hashlib.sha256((PKG/"data"/n).read_bytes()).hexdigest() != h]
        add("instrument mode tables (data/*.npz, hashed in data/MANIFEST.json)", not bad,
            f"{len(files)} tables" + (f"; missing or changed: {bad}" if bad else ", hashes match"))
    except (OSError, ValueError, KeyError) as exc:
        add("instrument mode tables (data/*.npz, hashed in data/MANIFEST.json)", False, f"{type(exc).__name__}: {exc}")
    isaac = os.environ.get("ISAAC_RENDER_PYTHON")
    add("ISAAC_RENDER_PYTHON (RGB observations, videos)", bool(isaac) and Path(isaac).is_file(),
        isaac or "unset: point it at Isaac Sim's python "
                 "(sessions with cameras=() run without it; render.py raises at the first camera observation)")
    add("ffmpeg (videos with sound)", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "not on PATH")
    cache = PKG/".cache"
    try:
        cache.mkdir(parents=True, exist_ok=True)
        probe = cache/".write_test"
        probe.write_text("ok")
        probe.unlink()
        add("percussion/.cache writable (visual assets, render stages)", True, str(cache))
    except OSError as exc:
        add("percussion/.cache writable (visual assets, render stages)", False, f"{cache}: {exc}")
    try:
        from . import visual_assets as va
        pinned = json.loads(va.MANIFEST.read_text())
        missing = [n for n in list(va.MODELS) + list(va.TEXTURES) if n not in pinned]
        fetched = [n for n in pinned if (va.CACHE/n).is_dir()]
        add("room dressing assets pinned (visual_assets.json)", not missing,
            f"{len(pinned)} pinned, {len(fetched)} in the cache" + (f"; not pinned: {missing}" if missing else "")
            + ("" if len(fetched) == len(pinned) else " (the rest are fetched and hash-checked on first render)"))
    except Exception as exc:                       # noqa: BLE001 - report, do not raise
        add("room dressing assets pinned (visual_assets.json)", False, f"{type(exc).__name__}: {exc}")
    add("disk for session traces and renders (>= 2 GiB free)", shutil.disk_usage(PKG).free > 2*1024**3,
        f"{shutil.disk_usage(PKG).free/1024**3:.1f} GiB free")
    robot = spec.robot if isinstance(getattr(spec, "robot", None), str) else None
    if robot:
        try:
            from .robot_setup import validate
            add(f"robot setup {robot!r}", True, f"qualified as {validate(robot)}")
        except Exception as exc:                   # noqa: BLE001
            add(f"robot setup {robot!r}", False, str(exc))
    return dict(ok=all(c_["ok"] for c_ in out), checks=out,
                note="CPU physics; the Isaac render interpreter for RGB observations and videos")
