"""`gar doctor sewing`: what this environment needs beyond Newton/Warp/MuJoCo (no simulation, no renderer started).

Checked: the Python packages the cell imports (SciPy for the meshes and the G1's IK, Pillow for the room's textures,
imageio-ffmpeg for films); the Isaac render interpreter that every RGB observation uses; the modules of other
environments the renderer imports (whiteboard.ipc and whiteboard.appearance, wiping, label_application.dressing: a
frozen export that leaves one out fails at the first observation); and the
locked robot assets of every offered deployment, resolved and hash-checked against robots/models/*/assets.lock.json
(the G1's in whiteboard/.cache, the Panda's in titrate/.cache, the FR3's in Newton's asset cache, downloaded once on
first use; GAR_ROBOT_ASSETS overrides all three): the selected deployment's (dual-fr3 unless a launch configuration
names another) are required, the others reported.
"""
import importlib.util
import os
from pathlib import Path

# robot alias -> the catalog models whose locked assets the rig loads
MODELS = {"dual-fr3": ("fr3",), "single-fr3": ("fr3",), "dual-panda": ("panda",), "g1": ("g1",)}
CROSS_ENV = ("whiteboard.ipc", "whiteboard.appearance", "wiping", "label_application.dressing")


def checks(spec):
    out = []

    def add(name, ok, detail, required=True):
        out.append(dict(check=name, ok=bool(ok), detail=detail, required=required))
    for mod, req in (("scipy", "scipy (meshes, the G1's IK)"), ("PIL", "Pillow (room textures, stills)"),
                     ("imageio_ffmpeg", "imageio-ffmpeg (films)")):
        add(f"import {mod}", importlib.util.find_spec(mod) is not None, req)
    for mod in CROSS_ENV:
        try:
            found = importlib.util.find_spec(mod) is not None
        except ModuleNotFoundError:
            found = False
        add(f"module {mod} (renderer)", found,
            "another environment's module the renderer imports; an exported copy of the code must carry it "
            "(git archive ... whiteboard wiping label_application)")
    isaac = os.environ.get("ISAAC_RENDER_PYTHON")
    add("ISAAC_RENDER_PYTHON (RGB observations)", bool(isaac) and Path(isaac).is_file(),
        isaac or "unset: point it at Isaac Sim's python")
    # every offered deployment's assets (gar doctor takes no robot option): the selected one (by default dual-fr3) is
    # required, the others are reported with the option they serve
    robot = spec.robot if isinstance(spec.robot, str) else None
    from .robot_setup import ALIASES
    selected = robot if robot in MODELS else next((a for a, rig in ALIASES.items() if rig == robot), "dual-fr3")
    seen = {}
    for alias, names in MODELS.items():
        for name in names:
            if name in seen:
                continue
            req = name in MODELS.get(selected, ())
            users = ", ".join(a for a, n in MODELS.items() if name in n)
            try:
                from robots.assets import resolve
                p = resolve(name)
                seen[name] = Path(p).is_file()
                add(f"{name} assets ({users})", seen[name], str(p), required=req)
            except Exception as exc:
                seen[name] = False
                where = {"g1": "whiteboard/.cache/assets/unitree_ros (provisioned by the whiteboard environment)",
                         "panda": "titrate/.cache/assets/franka_description or TITRATE_PANDA_URDF",
                         "fr3": "Newton's asset cache (newton.utils.download_asset('franka_emika_panda'), the network "
                                "once)"}.get(name, "")
                if os.environ.get("GAR_ROBOT_ASSETS"):
                    where = f"GAR_ROBOT_ASSETS/{name} (set: {os.environ['GAR_ROBOT_ASSETS']})"
                add(f"{name} assets ({users})", False, f"{type(exc).__name__}: {exc}; expected in {where}; needed for "
                    f"--robot {users}", required=req)
    return dict(ok=all(c["ok"] for c in out if c["required"]), checks=out,
                note="imports, files and variables only; nothing is simulated or rendered")
