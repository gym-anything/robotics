"""`gar doctor filleting`: what this environment needs beyond Newton/Warp/MuJoCo (no simulation).

Checked: the Python packages the tissue solver, the topology code and the renderer import; the Isaac render interpreter
(every RGB observation and film is rendered by it); the FR3 and Panda robot assets (every rig, whichever is selected); the visual
asset cache (the room dressing's pinned Poly Haven models are downloaded into it on first use, otherwise the network);
the wiping package's shared assets (dressing.py reads them through the repository).
"""
import importlib.util
import os
import shutil
from pathlib import Path



def checks(spec):
    out = []

    def add(name, ok, detail, required=True):
        out.append(dict(check=name, ok=bool(ok), detail=detail if required else f"{detail} (optional)",
                        required=required))
    for mod, req in (("scipy", "scipy (sparse graphs, KD-trees)"), ("trimesh", "trimesh (robot link meshes)"),
                     ("PIL", "pillow (frames)"), ("collada", "pycollada (the Franka .dae link meshes)")):
        add(f"import {mod}", importlib.util.find_spec(mod) is not None, req)
    isaac = os.environ.get("ISAAC_RENDER_PYTHON")
    add("ISAAC_RENDER_PYTHON (every RGB observation and film)", bool(isaac) and Path(isaac).is_file(),
        isaac or "unset: point it at Isaac Sim's python; without it "
                 "a session with cameras fails at its first observation (use cameras=() for physics only)")
    # every rig's assets, whichever is selected: gar doctor takes no --robot
    # the default deployment (dual-fr3) needs only the FR3's; the Panda's only for robot=dual-panda
    models = (("fr3", True), ("panda", False))
    for m, req in models:
        try:
            from robots.assets import resolve
            path, ok = str(resolve(m)), True
        except Exception as exc:
            path, ok = (f"{type(exc).__name__}: {exc}; set GAR_ROBOT_ASSETS to a directory holding the robot assets"
                        + (" or TITRATE_PANDA_URDF to the Panda URDF" if m == "panda" else "")), False
        add(f"{m} robot assets (robots/models/{m})" + ("" if req else ", needed only for robot=dual-panda"), ok, path,
            required=req)
    repo = Path(__file__).resolve().parent.parent
    wiping = repo/"wiping"/"assets"
    add("wiping/assets (room dressing textures read through the repository)", wiping.is_dir(),
        str(wiping) + ("" if wiping.is_dir() else ": missing (an installed wheel without the repository)"))
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))/"gar"/"filleting_visual_assets"   # dressing.CACHE
    add("visual asset cache (pinned Poly Haven models; downloaded on first use)", cache.exists(), str(cache),
        required=False)
    return dict(ok=all(c["ok"] for c in out if c["required"]), checks=out,
                note="Wall time: tens to a few hundred wall seconds per simulated second (holding still ~60x, cutting "
                     "~100-200x with one camera on an unshared host, more beside other GPU or CPU jobs; each extra "
                     "policy camera adds ~1.6 s per observation); startup 135-225 s; use client timeouts of at least "
                     "600 s")
