"""Dependency diagnostics only; never starts physics, a GPU context or a renderer."""
import importlib.metadata as metadata
import importlib.util as util
import os
from pathlib import Path
import shutil


def checks(spec=None):
    required = {"newton": ("newton", "1.5.0"), "warp-lang": ("warp", None),
                "mujoco": ("mujoco", "3.11.0"), "numpy": ("numpy", None),
                "scipy": ("scipy", None), "ruckig": ("ruckig", None), "Pillow": ("PIL", None), "PyOpenGL": ("OpenGL", None)}
    packages, issues = {}, []
    for distribution, (module, exact) in required.items():
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            version = None
        found = util.find_spec(module) is not None
        packages[distribution] = dict(version=version, import_available=found, required_version=exact)
        if not found or (exact is not None and version != exact):
            issues.append(f"{distribution}: requires {exact or 'an importable installation'}, found {version}")
    executable = os.environ.get("ISAAC_RENDER_PYTHON")
    renderer = dict(executable=executable, executable_exists=bool(executable and Path(executable).is_file()),
                    worker=str(Path(__file__).with_name("render_worker.py")), gpu_execution_checked=False)
    mesa=Path('/usr/share/glvnd/egl_vendor.d/50_mesa.json')
    renderer.update(default_engine='mujoco',software_egl_vendor=str(mesa),software_egl_available=mesa.is_file(),isaac_optional=True)
    if not mesa.is_file():issues.append('Default CPU RGB needs Mesa EGL/llvmpipe and its GLVND vendor file')
    rows=[dict(check=k,ok=bool(v['import_available'] and (not v['required_version'] or v['version']==v['required_version'])),
               detail=f"version {v['version']}; required {v['required_version'] or 'importable'}") for k,v in packages.items()]
    rows.append(dict(check='Mesa EGL vendor',ok=mesa.is_file(),detail=str(mesa)))
    font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    rows.append(dict(check='Printed cue font',ok=font.is_file(),detail=str(font)))
    if not font.is_file():issues.append('Default physical labels need fonts-dejavu-core (DejaVuSans.ttf)')
    return dict(ok=not issues, checks=rows, packages=packages, renderer=renderer, issues=issues,
                optional_media_tools={k: shutil.which(k) for k in ("ffmpeg", "ffprobe")},
                scope="Package/module and Mesa vendor-file presence; optional Isaac executable only; no simulation or GPU execution.")


if __name__ == "__main__":
    import json
    print(json.dumps(checks(), indent=2))
