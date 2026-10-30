"""Pinned CC0 appearance assets for the welding shop (Poly Haven models, textures and an HDRI). Visual only: no
collision geometry or physics parameter comes from here (props within a robot's reach get colliders in scene.py).

Every downloaded file is checked against the SHA-256 frozen in `visual_assets.json`; a missing pin is an error unless
`python -m welding.visual_assets --pin` is run deliberately (download, hash, record). Licence: CC0,
https://polyhaven.com/license. Method: label_application/visual_assets.py.
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

# model name -> role in the shop (all visual dressing outside every robot's reach; welding/shop.py places them)
MODELS = {
    "hanging_industrial_lamp": "high-bay pendants under the purlins (disk lights carry the light)",
    "steel_frame_shelves_01": "stock shelving behind the cell (plastic bins on it)",
    "plastic_crate_01": "parts bins on the shelves",
    "metal_toolbox": "toolboxes on the consumables shelf and the fitter's bench",
    "cardboard_box_01": "consumables boxes on the top shelf",
    "drawer_cabinet": "drawer cabinet at the back wall",
    "industrial_storage_cart": "storage cart at the back-right corner",
    "portable_welding_cart": "oxy-fuel cart (green oxygen, red acetylene cylinders, regulators, hoses, torch) at the right wall",
    "korean_fire_extinguisher_01": "fire extinguisher by the roll-up door (required near hot work)",
    "metal_office_desk": "the programming desk by the front wall",
    "metal_stool_01": "stools at the desk and by the cell",
    "metal_trash_can": "scrap skip at the front wall",
    "barrel_03": "drums at the front wall",
    "wooden_crate_01": "crates at the right wall",
    "bench_vice_01": "vise on the fitter's bench",
    "cross_pein_hammer": "hammer on the fitter's bench",
    "adjustable_wrench": "wrench on the fitter's bench",
    "combination_wrench": "wrench on the fitter's bench",
    "drill_press_01": "bench drill press on its stand",
    "metal_tool_chest": "roll-away tool chest at the right wall",
    "tool_cart": "tool cart beside the cell, outside the left screen",
}
TEXTURES = {
    "hangar_concrete_floor": 4.0,      # tile size m
    "concrete_block_wall": 2.0,        # the block dado
    "box_profile_metal_sheet": 1.5,    # cladding relief (its red paint replaced by a grey constant)
    "corrugated_iron_02": 1.5,         # roof deck
    "painted_metal_shutter": 1.0,      # roll-up door
    "rusty_metal_02": .8,              # rusty bar stock
}
HDRIS = {"machine_shop_02": "the sky outside the window openings (dome light; the shop is closed)"}
TEX_MAPS = ("diff", "rough", "nor_gl")


def _get(url):
    with urlopen(Request(url, headers={"User-Agent": "GymAnythingWelding/1.0"}), timeout=180) as r:
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
        f = info["hdri"]["2k"]["hdr"]
        return [(Path(f["url"]).name, f["url"])]
    keys = {"diff": "Diffuse", "rough": "Rough", "nor_gl": "nor_gl"}
    return [(f"{name}_{m}_1k.jpg", info[keys[m]]["1k"]["jpg"]["url"]) for m in TEX_MAPS]


def fetch(name, kind, pin=False):
    """Local folder of a pinned asset, downloading and verifying as needed."""
    man = _manifest()
    folder = CACHE/name
    entry = man.get(name)
    if entry is None:
        if not pin:
            raise KeyError(f"visual asset {name} is not pinned in {MANIFEST.name}")
        entry = {"kind": kind, "files": {}, "license": LICENSE, "source": f"https://polyhaven.com/a/{name}"}
        for rel, url in _files_for(name, kind):
            data = _get(url)
            p = folder/rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            entry["files"][rel] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
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


def hdri(name):
    folder = fetch(name, "hdri")
    return next(folder/rel for rel in _manifest()[name]["files"])


if __name__ == "__main__":
    pin = "--pin" in sys.argv
    for n in MODELS:
        print(n, fetch(n, "model", pin=pin))
    for n in TEXTURES:
        print(n, fetch(n, "texture", pin=pin))
    for n in HDRIS:
        print(n, fetch(n, "hdri", pin=pin))
