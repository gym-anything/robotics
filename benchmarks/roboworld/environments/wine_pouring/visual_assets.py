"""Pinned CC0 appearance assets (Poly Haven models and textures). Visual only: no collision
geometry or physics parameter comes from here.

Every downloaded file is checked against the SHA-256 frozen in `visual_assets.json`; a missing
pin is an error unless `python -m wine_pouring.visual_assets --pin` is run deliberately
(which downloads, hashes and records the files). Licence: CC0, https://polyhaven.com/license
"""
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache/visual_assets"
MANIFEST = ROOT / "visual_assets.json"
LICENSE = "CC0 1.0 (https://polyhaven.com/license)"
API = "https://api.polyhaven.com/files/"

# model name -> role in the scene (all visual dressing; nothing here has collision or physics)
MODELS = {
    "bar_chair_round_01": "bar stools on the guest side",
    "wine_barrel_01": "wine barrel by the back bar",
    "wine_bottles_01": "bottles on the back-bar shelves",
    "hanging_industrial_lamp": "pendant lamps over the bar (the sphere lights carry the light)",
    "potted_plant_02": "plant at the end of the bar",
    "wooden_display_shelves_01": "back-bar shelving",
    "fancy_picture_frame_01": "picture on the back wall",
    "brass_candleholders": "candles on the guest side of the bar",
}
# texture name -> tile size in metres
TEXTURES = {
    "dark_wooden_planks": 2.0,
    "brick_wall_10": 2.5,
    "marble_01": 1.2,
    "natural_walnut_veneer": 1.0,
    "grey_plaster": 3.0,
}
TEX_MAPS = ("diff", "rough", "nor_gl")


def _get(url):
    with urlopen(Request(url, headers={"User-Agent": "GymAnythingWinePouring/1.0"}), timeout=120) as r:
        return r.read()


def _manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def _files_for(name, kind):
    """(relative path, url) for a model's 1k USD + included textures, or a texture's 1k JPG maps."""
    info = json.loads(_get(API + name))
    if kind == "model":
        usd = info["usd"]["1k"]["usd"]
        out = [(Path(usd["url"]).name, usd["url"])]
        out += [(rel, f["url"]) for rel, f in usd.get("include", {}).items()]
        return out
    keys = {"diff": "Diffuse", "rough": "Rough", "nor_gl": "nor_gl"}
    return [(f"{name}_{m}_1k.jpg", info[keys[m]]["1k"]["jpg"]["url"]) for m in TEX_MAPS]


def fetch(name, kind, pin=False):
    """Return the local folder of a pinned asset, downloading and verifying as needed."""
    man = _manifest()
    folder = CACHE / name
    entry = man.get(name)
    if entry is None:
        if not pin:
            raise KeyError(f"visual asset {name} is not pinned in {MANIFEST.name}")
        entry = {"kind": kind, "files": {}}
        for rel, url in _files_for(name, kind):
            data = _get(url)
            p = folder / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            entry["files"][rel] = {"url": url, "sha256": hashlib.sha256(data).hexdigest()}
        man[name] = entry
        MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True))
    for rel, f in entry["files"].items():
        p = folder / rel
        if not p.exists():
            data = _get(f["url"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        if hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]:
            raise ValueError(f"visual asset hash mismatch: {p}")
    return folder


def model(name):
    """Path of the pinned model's USD file."""
    folder = fetch(name, "model")
    return next(folder / rel for rel in _manifest()[name]["files"] if rel.endswith((".usd", ".usdc")))


def texture(name):
    """{map: path} of a pinned texture."""
    folder = fetch(name, "texture")
    return {m: folder / f"{name}_{m}_1k.jpg" for m in TEX_MAPS}


if __name__ == "__main__":
    pin = "--pin" in sys.argv
    for n in MODELS:
        print(n, fetch(n, "model", pin=pin))
    for n in TEXTURES:
        print(n, fetch(n, "texture", pin=pin))
