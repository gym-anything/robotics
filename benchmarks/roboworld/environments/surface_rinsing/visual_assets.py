"""Pinned CC0 appearance assets for the dish room (Poly Haven models and textures). Visual only: no collision geometry
or physics parameter comes from here; every model sits outside the robot's working volume (render-only dressing).

Every downloaded file is checked against the SHA-256 frozen in `visual_assets.json`; a missing pin is an error unless
`python -m surface_rinsing.visual_assets --pin` is run deliberately (download, hash, record). The fetch logic is
percussion/visual_assets.py's. Licence: CC0, https://polyhaven.com/license
"""
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def _cache_dir():
    """Inside the package where it is writable (the checkout); else the user's cache (an installed wheel in a read-only
    site-packages); RINSING_ASSET_CACHE overrides both."""
    import os
    if os.environ.get("RINSING_ASSET_CACHE"):
        return Path(os.environ["RINSING_ASSET_CACHE"])
    inside = ROOT/".cache"/"visual_assets"
    try:
        inside.mkdir(parents=True, exist_ok=True)
        probe = inside/".write_test"
        probe.touch()
        probe.unlink()
        return inside
    except OSError:
        return Path.home()/".cache"/"gar"/"rinsing-visual-assets"


CACHE = _cache_dir()
MANIFEST = ROOT/"visual_assets.json"
LICENSE = "CC0 1.0 (https://polyhaven.com/license)"
API = "https://api.polyhaven.com/files/"

MODELS = {
    "mounted_fluorescent_lights": "ceiling light fittings over the stations (render-only; the light is a USD light)",
    "bleach_bottle": "sanitizer jug on the shelf above the pot sink (render-only)",
    "multi_cleaner_bottle": "spray cleaner on the clean drainboard's shelf (render-only)",
    "plastic_jerrycan": "detergent concentrate by the pot sink's leg (render-only)",
    "plastic_crate_01": "produce crate on the dunnage shelf (render-only)",
    "plastic_crate_02": "crate on the dunnage shelf (render-only)",
    "plastic_broom": "broom leaning in the can-wash corner (render-only)",
    "dustpan": "dustpan by the broom (render-only)",
    "trashbag": "tied trash bag by the back door (render-only)",
    "plastic_container": "food container on the clean dish table (render-only)",
    "industrial_storage_cart": "utility cart against the south wall (render-only)",
}
TEXTURES = {
    "brown_floor_tiles": 1.7,
    "long_white_tiles": 1.27,
    "concrete_block_wall": 2.0,
}
TEX_MAPS = ("diff", "rough", "nor_gl")


def _get(url):
    with urlopen(Request(url, headers={"User-Agent": "GymAnythingPainting/1.0"}), timeout=120) as r:
        return r.read()


def _manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def _files_for(name, kind):
    info = json.loads(_get(API + name))
    if kind == "model":
        usd = info["usd"]["1k"]["usd"]
        out = [(Path(usd["url"]).name, usd["url"])]
        out += [(rel, f["url"]) for rel, f in usd.get("include", {}).items()]
        return out
    keys = {"diff": "Diffuse", "rough": "Rough", "nor_gl": "nor_gl"}
    return [(f"{name}_{m}_1k.jpg", info[keys[m]]["1k"]["jpg"]["url"]) for m in TEX_MAPS]


def fetch(name, kind, pin=False):
    man = _manifest()
    folder = CACHE/name
    entry = man.get(name)
    if entry is None:
        if not pin:
            raise KeyError(f"visual asset {name} is not pinned in {MANIFEST.name}")
        entry = {"kind": kind, "files": {}}
        for rel, url in _files_for(name, kind):
            data = _get(url)
            p = folder/rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            entry["files"][rel] = {"url": url, "sha256": hashlib.sha256(data).hexdigest()}
        man[name] = entry
        MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True))
    for rel, f in entry["files"].items():
        p = folder/rel
        if not p.exists():
            data = _get(f["url"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        if hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]:
            raise ValueError(f"visual asset hash mismatch: {p}")
    return folder


def model(name):
    folder = fetch(name, "model")
    return next(folder/rel for rel in _manifest()[name]["files"] if rel.endswith((".usd", ".usdc")))


def texture(name):
    folder = fetch(name, "texture")
    return {m: folder/f"{name}_{m}_1k.jpg" for m in TEX_MAPS}


if __name__ == "__main__":
    pin = "--pin" in sys.argv
    for n in MODELS:
        print(n, fetch(n, "model", pin=pin))
    for n in TEXTURES:
        print(n, fetch(n, "texture", pin=pin))
