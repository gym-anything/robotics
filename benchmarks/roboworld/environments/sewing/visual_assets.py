"""Pinned CC0 appearance assets for the sewing sample room (Poly Haven models, textures and an HDRI). Visual only: no
collision geometry or physics parameter comes from here (everything within a robot's reach has a collider in
scene.py). The machine head itself is authored (sewing/assets/build_machine.py): no CC0 industrial lockstitch exists.

Every downloaded file is checked against the SHA-256 frozen in `visual_assets.json`; a missing pin is an error unless
`python -m sewing.visual_assets --pin` is run deliberately (download, hash, record). Licence: CC0,
https://polyhaven.com/license. Method: label_application/visual_assets.py, as welding.
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

# model name -> role in the sample room (all visual dressing, outside every robot's sweep)
MODELS = {
    "WoodenTable_01": "pinned earlier; the cutting table is now authored in room.py (not referenced)",
    "measuring_tape_01": "tape measure on the cutting table",
    "wicker_basket_01": "basket of cut parts by the cutting table",
    "steel_frame_shelves_01": "shelving with fabric bolts and thread boxes at the wall",
    "drawer_cabinet": "notions cabinet (thread, needles, bobbins)",
    "wall_clock": "wall clock",
    "metal_stool_01": "stools at the other workstations",
    "cardboard_box_01": "box of fabric offcuts",
    "mounted_fluorescent_lights": "ceiling fixtures (rect lights carry the light)",
}
TEXTURES = {
    "rough_linen": 0.05,               # plain-weave close-up: the muslin's weave (tile size m)
    "white_oak_veneer": 0.6,           # sewing table top (melamine wood-grain)
    "laminate_floor_02": 1.5,
    "beige_wall_002": 2.0,
}
HDRIS = {"artist_workshop": "a real craft workshop interior: distant lighting and reflections (superseded as the dome by "
                             "the sky once the room was closed)",
         "kloofendal_48d_partly_cloudy_puresky": "the sky seen through the room's windows (an upper floor); the "
                                                 "daylight enters only through them"}
TEX_MAPS = ("diff", "rough", "nor_gl")


def _get(url):
    with urlopen(Request(url, headers={"User-Agent": "GymAnythingSewing/1.0"}), timeout=180) as r:
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
