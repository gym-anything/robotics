"""Provision only the revision- and content-pinned G1 files used by this scene."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
ASSET_ROOT = ROOT / ".cache" / "assets" / "unitree_ros"
URDF = ASSET_ROOT / "robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf"


def provision(*, workers=4):
    lock = json.loads((ROOT / "assets.lock.json").read_text())
    prefix = f"https://raw.githubusercontent.com/unitreerobotics/unitree_ros/{lock['revision']}/"

    def fetch(entry):
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid asset manifest path")
        path = ASSET_ROOT.joinpath(*relative.parts)
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]:
            return {"path": entry["path"], "cached": True}
        request = Request(
            prefix + relative.as_posix(), headers={"User-Agent": "GAR-whiteboard-assets"}
        )
        with urlopen(request, timeout=90) as response:
            content = response.read(entry["bytes"] + 1)
        if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Asset content verification failed: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".download")
        temporary.write_bytes(content)
        os.replace(temporary, path)
        return {"path": entry["path"], "cached": False}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        result = list(pool.map(fetch, lock["files"]))
    return {"revision": lock["revision"], "root": str(ASSET_ROOT), "files": result}


def verify():
    lock = json.loads((ROOT / "assets.lock.json").read_text())
    for item in lock["files"]:
        path = ASSET_ROOT / item["path"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(
                f"Missing/modified G1 asset: {path}. Run python -m whiteboard.assets"
            )
    return {"revision": lock["revision"], "verified_files": len(lock["files"])}


if __name__ == "__main__":
    print(json.dumps(provision(), indent=2))
