"""Pinned CC0 appearance assets (Poly Haven models and textures). Visual only: no collision
geometry or physics parameter comes from here.

Every downloaded file is checked against the SHA-256 frozen in `visual_assets.json`; a missing
pin is an error unless `python -m barcode_scanning.visual_assets --pin` is run deliberately
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
    "steel_frame_shelves_01": "stockroom wall shelving (background)",
    "steel_frame_shelves_02": "stockroom wall shelving (background)",
    "cardboard_box_01": "shipping cases on floor and background shelves",
    "plastic_crate_01": "tote crates (background)",
    "plastic_crate_03": "tote crates on the floor",
    "hand_truck": "hand truck by the dock door",
    "metal_trash_can": "waste bin",
    "mounted_fluorescent_lights": "ceiling fixtures (the RectLights carry the light)",
    "korean_fire_extinguisher_01": "wall fire extinguisher",
    "fire_alarm": "wall fire alarm call point",
    "wall_clock": "wall clock",
    "rollershutter_door": "receiving dock roller shutter",
    "clipboard": "count sheet clipboard",
    "security_camera_01": "ceiling security camera",
    "ladder_sectioned_01": "step ladder against the wall",
    "plastic_broom": "broom",
    "bleach_bottle": "cleaning supplies on background shelves",
    "multi_cleaner_bottle": "cleaning supplies on background shelves",
    "plastic_bottle_gallon": "gallon jugs on background shelves",
    "russian_food_cans_01": "canned goods on background shelves",
    "wooden_display_shelves_01": "store display shelving (counter preset)",
    "wooden_bookshelf_worn": "library bookshelves (library preset)",
    "book_encyclopedia_set_01": "books on library shelves",
    "decorative_book_set_01": "books on library shelves",
    "desk_lamp_arm_01": "circulation desk lamp",
}
# texture name -> tile size in metres
TEXTURES = {
    "hangar_concrete_floor": 4.0,
    "concrete_block_wall": 2.0,
    "beige_wall_002": 3.0,
    "large_grey_tiles": 2.0,
    "laminate_floor_02": 2.0,
    "white_oak_veneer": 1.0,
}
TEX_MAPS = ("diff", "rough", "nor_gl")


def _get(url):
    with urlopen(Request(url, headers={"User-Agent": "GymAnythingBarcodeScanning/1.0"}), timeout=120) as r:
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
