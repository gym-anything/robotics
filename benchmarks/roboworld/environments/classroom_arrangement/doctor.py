"""`gar doctor classroom-arrangement`: what this environment needs beyond Newton/Warp/MuJoCo (no simulation).

The Mobile FR3 Duo description is not bundled: it is built once from franka_description (robot_asset.py) into the
shared robot asset cache and verified against the locked hashes in robots/models/mobile-fr3-*/assets.lock.json.
Isaac Sim's interpreter renders the RGB observations; physics, evaluation and checkpoints run without it.
"""
import importlib.util
import os
from pathlib import Path


def checks(spec):
    out = []

    def add(name, ok, detail):
        out.append(dict(check=name, ok=bool(ok), detail=detail))
    for mod in ("trimesh", "PIL", "scipy", "collada"):
        add(f"import {mod}", importlib.util.find_spec(mod) is not None, "trimesh / pillow / scipy / pycollada (robot DAE visuals)")
    robot = spec.robot if isinstance(spec.robot, str) else "mobile-fr3-duo"
    model = {"mobile-fr3-duo": "mobile-fr3-duo", "mobile-fr3-single": "mobile-fr3-single"}.get(robot, "mobile-fr3-duo")
    try:
        from robots import assets
        path = assets.resolve(model)
        add(f"{model} robot assets (locked)", True, str(path))
    except Exception as exc:
        add(f"{model} robot assets (locked)", False, f"{exc}; build them: python -m classroom_arrangement.robot_asset "
                                                    "--franka-description <checkout at 7aeeddc> --xacro-path <xacro>")
    isaac = os.environ.get("ISAAC_RENDER_PYTHON")
    add("ISAAC_RENDER_PYTHON (RGB observations and diagnostic renders)", bool(isaac) and Path(isaac).is_file(),
        isaac or "unset: point it at the Isaac Sim python interpreter; without it create the session with cameras=()")
    # appearance assets fetched on the first render (pinned SHA-256): an offline host must have them cached
    from classroom_arrangement.render import HDRIS, EXTERIOR_HDRI
    import hashlib
    hdr = Path(__file__).resolve().parent/".cache"/"visual_assets"/f"{EXTERIOR_HDRI}_2k.hdr"
    cached = hdr.is_file() and hashlib.sha256(hdr.read_bytes()).hexdigest() == HDRIS[EXTERIOR_HDRI]["sha256"]
    out.append(dict(check="exterior HDR (first render downloads it)", ok=True,
                    detail=str(hdr) + (" cached, hash ok" if cached else
                                       " not cached: the first render downloads it from Poly Haven (needs network)")))
    try:
        from whiteboard.visual_assets import ASSETS, CACHE
        oak = all((CACHE/f"oak_veneer_01_{k}_1k.jpg").is_file() for k in ("diff", "rough", "nor_gl"))
    except Exception:
        oak = False
    out.append(dict(check="oak veneer textures (whiteboard.visual_assets)", ok=True,
                    detail="cached" if oak else "not cached: the first render downloads them (needs network)"))
    return dict(ok=all(c["ok"] for c in out), checks=out,
                note="Robot assets are verified by hash; the Isaac interpreter is needed only for images; pinned CC0 "
                     "appearance assets are downloaded on the first render and hash-checked")
