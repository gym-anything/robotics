"""Small licensed appearance assets, content checked and cached outside Git."""

import hashlib
import os
from pathlib import Path
from urllib.request import Request, urlopen

SKY_URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/suburban_football_field_2k.hdr"
SKY_MD5 = "d16e4847727300ad4ec020a4a45d0437"
GRASS = {
    "diff": "0dbc071e91d6905edfcfbe8eb785a1ab",
    "nor_gl": "8279e096e204ea326d57a318869f99df",
    "rough": "05f6bb1383e12b327727d313c5899cf2",
}


def grass():
    folder = (
        Path(os.environ.get("FOOTBALL_CACHE", Path.home() / ".cache/football-newton15"))
        / "appearance"
    )
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, expected in GRASS.items():
        filename = f"leafy_grass_{name}_1k.jpg"
        path = folder / filename
        if not path.exists():
            url = (
                "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/leafy_grass/"
                + filename
            )
            with urlopen(
                Request(url, headers={"User-Agent": "GAR-football/1.0"}), timeout=60
            ) as response:
                data = response.read()
            if hashlib.md5(data).hexdigest() != expected:
                raise RuntimeError("Grass asset hash changed")
            path.write_bytes(data)
        if hashlib.md5(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Grass cache checksum mismatch")
        paths[name] = path
    return paths


def sky():
    folder = (
        Path(os.environ.get("FOOTBALL_CACHE", Path.home() / ".cache/football-newton15"))
        / "appearance"
    )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "suburban_football_field_2k.hdr"
    if not path.exists():
        with urlopen(
            Request(SKY_URL, headers={"User-Agent": "GAR-football/1.0"}), timeout=60
        ) as response:
            data = response.read()
        if hashlib.md5(data).hexdigest() != SKY_MD5:
            raise RuntimeError("Football HDRI content differs from Poly Haven manifest")
        path.write_bytes(data)
    if hashlib.md5(path.read_bytes()).hexdigest() != SKY_MD5:
        raise RuntimeError("Football HDRI cache checksum mismatch")
    return path


def grass_blades(stage, mat):
    """Render-only subgrid grass: the pitch model owns homogenized compliance.

    Deterministic blade normals and scale add local depth without adding an
    unrelated hair/soil physics solver. No displacement of the contact plane.
    """
    import numpy as np
    from .render import mesh, bind

    rng = np.random.default_rng(144)
    count = 300_000
    xy = rng.uniform([-34, -12], [3, 12], (count, 2))
    angle = rng.uniform(0, 2 * np.pi, count)
    half = rng.uniform(0.0006, 0.0014, count)
    side = np.c_[np.cos(angle) * half, np.sin(angle) * half, np.zeros(count)]
    root = np.c_[xy, np.full(count, 0.0005)]
    tip = (
        root + np.c_[rng.normal(0, 0.004, (count, 2)), rng.uniform(0.009, 0.023, count)]
    )
    vertices = np.stack([root - side, root + side, tip], axis=1).reshape(-1, 3)
    shape = mesh(
        stage,
        "/Lab/subgrid_grass",
        vertices,
        np.arange(len(vertices)).reshape(-1, 3),
        smooth=False,
    )
    shape.CreateDoubleSidedAttr(True)
    bind(shape, mat)
