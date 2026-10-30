"""Pinned CC0 appearance assets; no collision geometry or physics parameters."""

from functools import lru_cache
import hashlib
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache/visual_assets"
LICENSE = "https://polyhaven.com/license"
ASSETS = {
    "oak_veneer_01": {
        "author": "Jenelle van Heerden",
        "tile_m": 1.8,
        "diff": "c14681f72f59f17bff568cae7a837641443b607b899f0c279cd64a5638f877a7",
        "rough": "ca2af5649b628ef05f196793d68cdca874c12ceb6e9ae566976546d9dd5a2604",
        "nor_gl": "1297acf7b4476d5ccd83cc8b27be96aa6110f7a9c3a40be0bd767bf81b17db01",
    },
    "beige_wall_001": {
        "author": "Dimitrios Savva; Rico Cilliers",
        "tile_m": 3.0,
        "diff": "72104704d0b9bd3713c5f6238971252b63eccf6e61263059c674caa8f3d0fa6d",
        "rough": "fc8328f47f4d2d82d5311a75bcb5c8cd8979d171a0ee097ad9e44ca9c5a00d60",
        "nor_gl": "cec747be50f91da0693986080b6f59ef868d53cb2ab7d577982165ece63ed75e",
    },
    "kiara_5_noon": {
        "author": "Greg Zaal",
        "hdri": "f5709ff2bc42a29a655f3ca3964d76522b7a61846d668d41a3306e204b271a8c",
    },
}


@lru_cache(maxsize=7)
def asset(name, kind):
    digest = ASSETS[name][kind]
    if kind == "hdri":
        filename = name + "_2k.hdr"
        url = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/" + filename
    else:
        filename = f"{name}_{kind}_1k.jpg"
        url = f"https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/{name}/{filename}"
    path = CACHE / filename
    CACHE.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(
            Request(url, headers={"User-Agent": "GymAnythingWhiteboard/1.0"}), timeout=60
        ) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Downloaded appearance asset hash mismatch: " + filename)
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Appearance asset hash mismatch: " + str(path))
    return path


def copy_to(output):
    folder = output / "textures"
    folder.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, definition in ASSETS.items():
        result[name] = {}
        for kind in ("diff", "rough", "nor_gl", "hdri"):
            if kind not in definition:
                continue
            source = asset(name, kind)
            target = folder / source.name
            if not target.exists():
                shutil.copyfile(source, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != definition[kind]:
                raise ValueError("Exported appearance asset hash mismatch: " + str(target))
            result[name][kind] = target
    return result


if __name__ == "__main__":
    for name, definition in ASSETS.items():
        for kind in ("diff", "rough", "nor_gl", "hdri"):
            if kind in definition:
                print(asset(name, kind))
