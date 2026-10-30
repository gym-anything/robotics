"""Pinned CC0 appearance assets (Poly Haven models, textures, HDRI) for the front yard. Visual only: no collision
geometry or physics parameter comes from here; every model is render-only dressing outside the work volume, and the
soil's colour comes from its particles' materials (render.py), the textures only add surface detail.

Every downloaded file is checked against the SHA-256 frozen in `visual_assets.json`; a missing pin is an error unless
`python -m soil_excavation.visual_assets --pin` is run deliberately (download, hash, record). Fetch logic: painting's
(percussion's). Licence: CC0, https://polyhaven.com/license
"""
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
CACHE = ROOT/".cache"/"visual_assets"
MANIFEST = ROOT/"visual_assets.json"
LICENSE = "CC0 1.0 (https://polyhaven.com/license)"
API = "https://api.polyhaven.com/files/"

MODELS = {
    "shrub_01": "foundation shrubs set back in the bed (render-only)",
    "shrub_02": "foundation shrub (render-only)",
    "shrub_04": "the #1 container shrub's foliage on the root ball (drawn on the root-ball body; its mass is the "
                "root ball part 'shrub', tools.rootball)",
    "garden_hose_wall_mounted_01": "hose reel on the house wall (render-only)",
    "garden_gloves_01": "work gloves on the walkway (render-only)",
    "planter_pot_clay": "a terracotta pot by the door (render-only)",
}
TEXTURES = {
    "brushed_concrete": 1.5,          # walkway (broom finish), foundation
    "leafy_grass": 1.0,               # lawn
    "wood_chip_path": 1.0,            # shredded bark mulch
    "brown_mud_02": 0.6,              # soil surface relief (normal map) and colour detail
    "weathered_plank_siding": 2.0,    # house front
}
HDRI = dict(name="residential_garden", note="sky and ambient light of a residential garden")
TEX_MAPS = ("diff", "rough", "nor_gl")


def _get(url):
    with urlopen(Request(url, headers={"User-Agent": "GymAnythingSoil/1.0"}), timeout=120) as r:
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
    if kind == "hdri":
        return [(f"{name}_2k.hdr", info["hdri"]["2k"]["hdr"]["url"])]
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


def hdri():
    folder = fetch(HDRI["name"], "hdri")
    return folder/f"{HDRI['name']}_2k.hdr"


if __name__ == "__main__":
    pin = "--pin" in sys.argv
    for n in MODELS:
        print(n, fetch(n, "model", pin=pin))
    for n in TEXTURES:
        print(n, fetch(n, "texture", pin=pin))
    print(HDRI["name"], fetch(HDRI["name"], "hdri", pin=pin))
