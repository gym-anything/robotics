"""`gar doctor barcode-scanning`: what this environment needs beyond Newton/Warp/MuJoCo (no simulation).

zxing-cpp decodes the scanner's frames (and encodes the printed symbols), trimesh builds convex hulls, and the
Isaac render interpreter is REQUIRED here: every scanner read comes from a rendered imager frame, so the
environment cannot run without it (unlike label-application, where it only served RGB observations).
"""
import importlib.util
import os
from pathlib import Path


def checks(spec):
    out = []

    def add(name, ok, detail):
        out.append(dict(check=name, ok=bool(ok), detail=detail))
    for mod, req in (("zxingcpp", "zxing-cpp>=3.1"), ("trimesh", "trimesh"), ("PIL", "pillow"), ("scipy", "scipy")):
        add(f"import {mod}", importlib.util.find_spec(mod) is not None, req + " (pip install 'gym-anything-robotics[barcode-scanning]')")
    isaac = os.environ.get("ISAAC_RENDER_PYTHON")
    add("ISAAC_RENDER_PYTHON (scanner imager frames and RGB observations)", bool(isaac) and Path(isaac).is_file(),
        isaac or "unset: point it at Isaac Sim's python")
    fonts = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    add("DejaVu fonts (label and packaging artwork)", fonts.is_file(), str(fonts))
    robot = spec.robot if isinstance(spec.robot, str) else None
    if robot in ("dual-panda", "scan-panda"):
        urdf = os.environ.get("TITRATE_PANDA_URDF")
        add("TITRATE_PANDA_URDF (dual-panda rig)", bool(urdf) and Path(urdf).is_file(), urdf or "unset")
    if robot in ("g1", "scan-g1"):
        # an installed wheel has no repository cache: the locked unitree_ros package is found under
        # $GAR_ROBOT_ASSETS/g1
        try:
            from robots.assets import resolve
            urdf, ok = str(resolve("g1")), True
        except Exception as exc:
            urdf, ok = f"{exc}; set GAR_ROBOT_ASSETS to a directory whose g1/ is the locked unitree_ros package", False
        add("G1 assets (g1 rig; GAR_ROBOT_ASSETS/g1 from an installed wheel)", ok, urdf)
    return dict(ok=all(c["ok"] for c in out), checks=out,
                note="The Isaac render interpreter is required: scanner reads are decoded from rendered frames")
