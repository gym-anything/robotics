"""Where this project's packages are installed.

The core packages live in ``src/`` and the RoboWorld environments in ``benchmarks/roboworld/environments/``, and a
wheel install puts them all in site-packages. Code that needs another package's files, or starts a render worker in a
different Python interpreter, finds the packages by import instead of assuming one folder holds them all.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

CORE_PACKAGES = ("gym_anything_robotics", "robots")
ENVIRONMENT_PACKAGES = ("percussion", "label_application", "titrate", "laundry", "sewing", "deposit_cans", "ingredient_mixing", "soil_excavation", "football", "syringe_handling", "surface_rinsing", "wine_pouring", "lockkey", "welding", "barcode_scanning", "painting", "gift_bags", "concrete_finishing", "classroom_arrangement", "pencap", "filleting", "whiteboard", "drilling", "tighten_codex", "forklift", "wiping", "sweeping", "control_cable")


def package_dir(name: str) -> Path:
    """The folder of an installed top-level package, found without importing it."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(f"package {name!r} is not installed (pip install -e . from the repository root)")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def project_file(relative: str | os.PathLike[str]) -> Path:
    """A file given as ``<package>/<path inside it>``, for example ``wiping/render_only.kit``."""
    first, *rest = Path(relative).parts
    return package_dir(first).joinpath(*rest)


def worker_path() -> str:
    """A folder of links to this project's packages only, for a render interpreter's PYTHONPATH.

    Isaac Sim runs the render workers in its own Python. Putting the packages' parent folder on that interpreter's
    path would also expose the Newton runtime's compiled modules (numpy for another Python version) when the packages
    are installed in site-packages, so the worker gets links to these packages and nothing else.
    """
    names = CORE_PACKAGES + ENVIRONMENT_PACKAGES
    targets = {name: package_dir(name) for name in names}
    key = hashlib.sha256("\n".join(f"{n}={t}" for n, t in sorted(targets.items())).encode()).hexdigest()[:16]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    folder = cache / "gar" / "worker_path" / key
    folder.mkdir(parents=True, exist_ok=True)
    for name, target in targets.items():
        link = folder / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    return str(folder)


def worker_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """A copy of ``env`` (default: this process's environment) whose PYTHONPATH is :func:`worker_path`.

    PYTHONSAFEPATH keeps ``python -m`` from also putting the working folder on the path, because the launchers start
    workers in the folder that holds the packages, which is site-packages in a non-editable install.
    """
    result = dict(os.environ if env is None else env)
    result["PYTHONPATH"] = worker_path()
    result["PYTHONSAFEPATH"] = "1"
    return result
