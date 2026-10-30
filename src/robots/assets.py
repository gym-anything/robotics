"""Pinned assets shared by all environments; verification never blesses new hashes."""
from gym_anything_robotics.paths import package_dir

import hashlib
import json
import os
from pathlib import Path
from .catalog import ROOT, model

_verified = {}


def install_hint(name):
    """How to install a model's locked assets, for every catalogue model: its own installer
    (robots/models/<name>/install.py) when it has one, else the provisioning step of the environment that pins it.
    Installers honour GAR_ROBOT_ASSETS (<GAR_ROBOT_ASSETS>/<name>)."""
    installer = ROOT / "models" / name / "install.py"
    if installer.is_file():
        how = f"run python -m robots.models.{name}.install"
    else:
        how = {"g1": "run python -m whiteboard.assets (unitree_ros files into whiteboard/.cache/assets)",
               "panda": "run python -m titrate.newton15.assets --isaac-root ISAAC_SIM_ROOT, or set TITRATE_PANDA_URDF",
               "fr3": "Newton downloads the pinned franka_emika_panda package on first use (network once)"}.get(
                   name, f"install the files listed in robots/models/{name}/assets.lock.json")
    where = os.environ.get("GAR_ROBOT_ASSETS")
    if where:
        how += f"; GAR_ROBOT_ASSETS is set, so they are read from {Path(where).expanduser() / name}"
    return how


def verify(name, directory):
    directory = Path(directory).resolve()
    lock = json.loads((ROOT / "models" / name / "assets.lock.json").read_text())
    entries = lock["files"]
    if isinstance(entries, list):
        entries = {entry["path"]: entry for entry in entries}
    signature = []
    for relative, entry in entries.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"{name} asset missing or outside package: {relative} (in {directory}); to install the "
                             f"locked {name} assets: {install_hint(name)}")
        stat = path.stat()
        signature.append((relative, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    key = (name, str(directory))
    if _verified.get(key) != signature:
        for relative, entry in entries.items():
            path = directory / relative
            if (
                path.stat().st_size != entry["bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
            ):
                raise ValueError(f"{name} asset differs from pinned source: {relative} (in {directory}); to "
                                 f"reinstall the locked {name} assets: {install_hint(name)}")
        _verified[key] = signature
    return directory


def resolve(name, *, path=None):
    spec = model(name).to_dict()
    if path is not None:
        source = Path(path).resolve()
        directory = source
        for _ in Path(spec["asset"]["path"]).parts:
            directory = directory.parent
        verify(name, directory)
        if directory / spec["asset"]["path"] != source:
            raise ValueError("Explicit URDF does not match locked package layout")
        return source
    if value := os.environ.get("GAR_ROBOT_ASSETS"):
        directory = Path(value).expanduser() / name
    elif name == "fr3":
        import newton.utils

        directory = newton.utils.download_asset("franka_emika_panda")
    elif name == "panda":
        if value := os.environ.get("TITRATE_PANDA_URDF"):
            return resolve(name, path=value)
        directory = package_dir("titrate") / ".cache/assets/franka_description"
    elif name == "g1":
        directory = package_dir("whiteboard") / ".cache/assets/unitree_ros"
    else:
        directory = Path.home() / ".cache/gar/robots" / name
        if not directory.is_dir():
            raise ValueError(
                f"Install the locked {name} assets in {directory} or set GAR_ROBOT_ASSETS: {install_hint(name)}"
            )
    return verify(name, directory) / spec["asset"]["path"]
