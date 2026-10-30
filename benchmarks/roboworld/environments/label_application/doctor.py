"""`gar doctor label-application`: what this environment needs beyond Newton/Warp/MuJoCo (no simulation).

Checked: the declared Python dependencies (python-barcode draws the label symbols at creation, zxing-cpp
decodes rendered views for the verifier, pypardiso solves the label's linear systems) and that pypardiso
can load MKL; the Isaac render interpreter used for RGB observations and reader views; and, for the
dual-panda rig, the Panda description shared through the robot catalog.
"""
import importlib.util
import os
from pathlib import Path


def checks(spec):
    out = []

    def add(name, ok, detail):
        out.append(dict(check=name, ok=bool(ok), detail=detail))
    for mod, req in (("barcode", "python-barcode>=0.16"), ("zxingcpp", "zxing-cpp>=3.1"), ("pypardiso", "pypardiso>=0.4")):
        add(f"import {mod}", importlib.util.find_spec(mod) is not None,
            req + " (pip install 'gym-anything-robotics[label-application]')")
    try:
        import pypardiso
        pypardiso.PyPardisoSolver()
        add("pypardiso loads MKL", True, os.environ.get("PYPARDISO_MKL_RT") or "found under sys.prefix")
    except Exception as exc:
        add("pypardiso loads MKL", False, f"{type(exc).__name__}: {exc}; set PYPARDISO_MKL_RT to libmkl_rt.so "
            "when the interpreter's prefix has no MKL (e.g. a venv layered over the runtime)")
    isaac = os.environ.get("ISAAC_RENDER_PYTHON")
    add("ISAAC_RENDER_PYTHON (RGB observations, reader views)", bool(isaac) and Path(isaac).is_file(),
        isaac or "unset: point it at Isaac Sim's python; "
        "cameras=() runs without it")
    robot = spec.robot if isinstance(spec.robot, str) else None
    if robot in ("dual-panda", "label-panda"):
        urdf = os.environ.get("TITRATE_PANDA_URDF")
        add("TITRATE_PANDA_URDF (dual-panda rig)", bool(urdf) and Path(urdf).is_file(), urdf or "unset")
    required = [c for c in out if not c["check"].startswith("ISAAC")]
    return dict(ok=all(c["ok"] for c in required), checks=out,
                note="ISAAC_RENDER_PYTHON is reported but not required: the environment runs without cameras")
