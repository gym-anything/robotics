"""Benchmark task folders: ``<environment package>/tasks/<task>/`` with ``task.json``, ``setup.py`` and ``verifier.py``.

Each benchmark task of an environment has its own folder, named after the task:

* ``task.json`` describes the task: its id, environment, split, selection and the instruction the agent receives, and
  names the entry points below (``hooks.setup`` and ``success.spec.program``).
* ``setup.py`` holds the code that builds the task's start state (``setup``).
* ``verifier.py`` holds the code that checks the task's outcome (``verify``).

The environment package keeps the code its tasks share (scene, physics, rendering, the evaluator machinery) and calls
into the folder of the task it runs. What the entry points receive and return is the environment's own interface; the
docstring of each file says what it is. Tasks of an environment that are not benchmark tasks stay in its package code.

Folders are loaded from their files, so a folder is named exactly like its task (hyphens included) and its code imports
the environment package by absolute name.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import ModuleType

from .paths import package_dir

SCHEMA = "gar.task-folder.v1"
PARTS = ("setup", "verifier")
FILES = ("task.json", "setup.py", "verifier.py")
_lock = threading.RLock()


def root(package: str) -> Path:
    """The ``tasks`` folder of an environment package."""
    return package_dir(package) / "tasks"


def names(package: str) -> tuple[str, ...]:
    """The task folders of an environment package, sorted by name."""
    folder = root(package)
    return tuple(sorted(p.parent.name for p in folder.glob("*/task.json"))) if folder.is_dir() else ()


def exists(package: str, name: str) -> bool:
    return (root(package) / name / "task.json").is_file()


def folder(package: str, name: str) -> Path:
    path = root(package) / name
    if not (path / "task.json").is_file():
        raise KeyError(f"{package} has no task folder {name!r}; it has {list(names(package))}")
    return path


def load(package: str, name: str) -> dict:
    """The task's ``task.json``."""
    return json.loads((folder(package, name) / "task.json").read_text())


def module(package: str, name: str, part: str) -> ModuleType:
    """The task's ``setup.py`` or ``verifier.py``, loaded once per process."""
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")
    key = f"{package}.tasks.{name}.{part}"
    with _lock:
        loaded = sys.modules.get(key)
        if loaded is not None:
            return loaded
        path = folder(package, name) / f"{part}.py"
        spec = importlib.util.spec_from_file_location(key, path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[key] = loaded
        try:
            spec.loader.exec_module(loaded)
        except BaseException:
            del sys.modules[key]
            raise
        return loaded


def setup(package: str, name: str) -> ModuleType:
    return module(package, name, "setup")


def verifier(package: str, name: str) -> ModuleType:
    return module(package, name, "verifier")


def entry(package: str, name: str, hook: str):
    """The function ``task.json`` names for ``hook`` ("setup" or "success")."""
    card = load(package, name)
    reference = card["hooks"]["setup"] if hook == "setup" else card["success"]["spec"]["program"]
    filename, function = reference.split("::")
    return getattr(module(package, name, Path(filename).stem), function)


__all__ = ["FILES", "PARTS", "SCHEMA", "entry", "exists", "folder", "load", "module", "names", "root", "setup",
           "verifier"]
