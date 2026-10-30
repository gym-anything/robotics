"""Pinned Panda asset installation without a machine-specific runtime path."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT/"assets"/"panda_manifest.json"
INSTALLED = ROOT.parent/".cache"/"assets"/"franka_description"
RELATIVE_URDF = Path("exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest(urdf):
    """Maintainer helper: inventory only; never accepts new hashes at runtime."""
    urdf = Path(urdf).resolve()
    root = urdf.parent.parent
    paths = {urdf, root/"package.xml"}
    for node in ET.parse(urdf).findall(".//mesh"):
        name = node.attrib["filename"]
        prefix = "package://franka_description/"
        path = (root/name[len(prefix):] if name.startswith(prefix) else urdf.parent/name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Robot resource leaves its package")
        paths.add(path)
    return {"robot": "Franka Panda", "description_version": "0.7.1",
        "source": "Isaac Sim 5.1 bundled franka_description; source package declares Apache 2.0",
        "urdf": urdf.relative_to(root).as_posix(),
        "files": {p.relative_to(root).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted(paths)}}


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, spec in manifest["files"].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
            raise ValueError(f"Panda asset missing or differs from pinned source: {name}")
    return root/manifest["urdf"]


def resolve_panda():
    if value := os.environ.get("TITRATE_PANDA_URDF"):
        path = Path(value).expanduser().resolve()
        result = verify(path.parent.parent)
        if result != path:
            raise ValueError("Explicit Panda path does not match the manifest's URDF path")
        return result
    return verify(INSTALLED)


def install(isaac_root):
    source = (Path(isaac_root).expanduser().resolve()/RELATIVE_URDF).parent.parent
    verify(source)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    INSTALLED.mkdir(parents=True, exist_ok=True)
    for name, spec in manifest["files"].items():
        target = (INSTALLED/name).resolve()
        if not target.is_relative_to(INSTALLED.resolve()):
            raise ValueError("Invalid asset destination")
        if target.exists():
            if digest(target) != spec["sha256"]:
                raise ValueError(f"Refusing to overwrite a modified installed asset: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/name, target)
    return verify(INSTALLED)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isaac-root", type=Path, required=True)
    a = p.parse_args()
    print(install(a.isaac_root))


if __name__ == "__main__":
    main()
