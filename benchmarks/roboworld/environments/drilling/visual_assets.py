"""Pinned CC0 appearance assets (Poly Haven models and textures). Visual only: no collision
geometry or physics parameter comes from here.

Every downloaded file is checked against the SHA-256 frozen in `visual_assets.json`; a missing
pin is an error unless `python -m drilling.visual_assets --pin` is run deliberately
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

# model name -> role in the scene. Drill_01 and drill_press_01 draw the two physical tools whose colliders were fitted
# to them (tools.CordlessDrill, tools.DrillPress); everything else is visual dressing without collision or physics.
MODELS = {
    "Drill_01": "the cordless drill's housing, grip and chuck (the physical tool's appearance)",
    "drill_press_01": "the drill press (static parts, feed handle, depth limiter, buttons)",
    "metal_tool_chest": "top tool chest on its roller cabinet by the east wall",
    "steel_frame_shelves_01": "stock shelving on the west wall and beside the wall bay",
    "metal_toolbox": "toolboxes on the shelving and on the bench's lower shelf",
    "korean_fire_extinguisher_01": "fire extinguisher by the door",
    "mounted_fluorescent_lights": "ceiling LED troffers (the rect lights carry the light)",
    "cardboard_box_01": "stock boxes on the shelves",
    "WetFloorSign_01": "caution sign near the wall bay",
    "plastic_broom": "broom leaning on the wall",
    "measuring_tape_01": "tape measure on the bench",
    "cross_pein_hammer": "spare hammer on the bench",
    "adjustable_wrench": "wrench on the bench",
    "power_box_01": "electrical panel on the south wall",
    "wooden_crate_01": "crate of offcuts",
    "industrial_wall_lamp": "task light over the drill press",
    "bench_vice_01": "bench vise bolted at the far end of the workbench",
    "pliers": "pegboard: combination pliers",
    "tongue_groove_pliers": "pegboard: tongue-and-groove pliers",
    "screwdrivers_02": "pegboard: two screwdrivers",
    "combination_wrench": "pegboard: combination wrenches (three sizes)",
    "pipe_wrench": "pegboard: pipe wrench",
    "flathead_screwdriver": "pegboard: flat screwdriver",
    "handsaw_wood": "pegboard: hand saw",
    "caged_hanging_light": "caged hanging lights over the bench and the wall bay's corner",
    "cement_bag": "cement bags beside the wall bay",
    "plastic_crate_02": "stacked crates; crates of anchor boxes on shelves and under the bench",
    "plastic_crate_03": "crate of anchor boxes beside the wall bay",
    "ladder_sectioned_01": "ladder against the west wall",
    "lubricant_spray": "spray can on the bench",
    "small_oil_can_01": "oil can on the bench",
    "spray_paint_bottles": "spray cans on the wall bay's shelving",
    "tool_cart": "rolling tool cart east of the wall bay",
    "oil_tin": "oil tin on the wall bay's shelving",
}
# texture name -> tile size in metres
TEXTURES = {
    "concrete_floor_painted": 3.0,
    "concrete_floor_worn_001": 3.0,
    "concrete": 2.4,
    "painted_plaster_wall": 2.5,
    "brushed_concrete": 2.0,
    "concrete_block_wall": 2.0,
    "painted_concrete": 2.5,
    "plywood": 1.2,
    "metal_plate": 1.0,
    "blue_metal_plate": 1.0,
}
TEX_MAPS = ("diff", "rough", "nor_gl")


def _get(url):
    with urlopen(Request(url, headers={"User-Agent": "GymAnythingDrilling/1.0"}), timeout=120) as r:
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
    """{map: path} of a pinned texture (or of a generated one: PROCEDURAL)."""
    if name in PROCEDURAL:
        return procedural(name)
    folder = fetch(name, "texture")
    return {m: folder / f"{name}_{m}_1k.jpg" for m in TEX_MAPS}


# ------------------------------------------------------------------ generated textures (no download, no licence)
# Douglas fir, flat-sawn (the ledger and the timber boards). Poly Haven has no single-board softwood lumber texture
# (its wood textures are hardwood veneers or multi-plank walls with seams), and a plywood veneer reads as
# a reddish hardwood. Colours were measured on The Wood Database's flat-sawn Douglas-fir sample
# photograph (https://www.wood-database.com/douglas-fir/, pseudotsuga-menziesii 300x600: brightest quarter of the pixels
# = earlywood sRGB (196, 160, 123), darkest tenth = latewood (171, 119, 66)); Douglas fir's earlywood-to-latewood
# transition is abrupt and its latewood band is wide (Wood Handbook FPL-GTR-282, ch. 2). Ring width 2.5-4 mm (6-10
# rings per inch, construction-grade lumber: authored). The tile (1.2 m, the ledger's length) is centred on each board
# by render.workpiece_uv, so no board face shows its edge.
PROCEDURAL = {"douglas_fir_flatsawn": dict(tile_m=1.2, px=3072, early=(196, 160, 123), late=(171, 119, 66),
                                            ring_mm=(2.5, 4.0), latewood=0.33, pith_m=0.07, taper_m=0.035, seed=11)}


def procedural(name):
    """{map: path} of a generated texture (diffuse only), written once to the cache and reused."""
    spec = PROCEDURAL[name]
    path = CACHE / "procedural" / f"{name}_diff_{hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _douglas_fir(spec).save(path)
    return {"diff": path, "rough": None, "nor_gl": None}


def _douglas_fir(spec):
    """Flat-sawn softwood face: the growth rings of a log whose axis runs along u (image x), cut `pith_m` from the pith
    at the face's centre line; the saw cut runs out of the log by `taper_m` over the tile (log taper and sweep), which
    draws the nested cathedral arches; small smooth warps keep the rings irregular."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(spec["seed"])
    N, T = spec["px"], spec["tile_m"]
    x = (np.arange(N) + 0.5)/N
    U, V = np.meshgrid(x, x)                                  # U along the grain (image x), V across (image y)
    um, vm = (U - 0.5)*T, (V - 0.5)*T                         # metres from the tile centre
    # warps: their slope along the grain stays below the taper's (0.029 < 0.035 per tile), so every ring keeps one apex
    # (open cathedral arches; a steeper warp closed them into chains of ellipses along the board)
    warp = sum(a*np.sin(2*np.pi*(fu*U + fv*V) + rng.uniform(0, 2*np.pi)) for a, fu, fv in
               ((0.0030, 0.3, 0.4), (0.0010, 1.1, -0.9), (0.0006, 2.3, 2.3), (0.0003, 4.1, -5.1), (0.0004, 0.2, 9.0)))
    r = np.sqrt(spec["pith_m"]**2 + (vm + 0.25*warp)**2) + spec["taper_m"]*(um/T) + warp + 0.05   # metres from pith
    widths = rng.uniform(spec["ring_mm"][0], spec["ring_mm"][1], 6000)*1e-3
    bounds = np.concatenate([[0.], np.cumsum(widths)])
    k = np.clip(np.searchsorted(bounds, r) - 1, 0, len(widths) - 1)
    phase = (r - bounds[k])/widths[k]                          # 0 at the ring's start (earlywood) .. 1 at its end
    lw = 1. - spec["latewood"]*rng.uniform(0.8, 1.2, len(widths))[k]
    late = np.clip((phase - lw)/0.05, 0., 1.)                 # abrupt transition into the latewood
    early_grad = 0.07*phase*(1. - late)                       # earlywood darkens slightly towards the transition
    lin = lambda c: np.where(np.asarray(c)/255. <= 0.04045, np.asarray(c)/255./12.92,
                             ((np.asarray(c)/255. + 0.055)/1.055)**2.4)
    e, l_ = lin(spec["early"]), lin(spec["late"])
    col = e[None, None, :]*(1. - late[..., None]) + l_[None, None, :]*late[..., None]
    col = col*(1. - early_grad[..., None])
    # tracheid streaks (long along the grain) and a broad colour drift
    streak = rng.normal(0., 1., (48, N))
    pos = np.arange(N)/N*47
    i0, fr = pos.astype(int), pos % 1.
    st = streak[i0]*(1 - fr)[:, None] + streak[i0 + 1]*fr[:, None]                   # (N along u, N across v)
    col = col*(1. + 0.03*st.T[..., None])
    drift = 1. + 0.05*np.sin(2*np.pi*(0.7*U + rng.uniform()))*np.sin(2*np.pi*(1.3*V + rng.uniform()))
    col = np.clip(col*drift[..., None], 0., 1.)
    srgb = np.where(col <= 0.0031308, col*12.92, 1.055*col**(1/2.4) - 0.055)
    return Image.fromarray((srgb*255 + 0.5).astype(np.uint8))


if __name__ == "__main__":
    pin = "--pin" in sys.argv
    for n in MODELS:
        print(n, fetch(n, "model", pin=pin))
    for n in TEXTURES:
        print(n, fetch(n, "texture", pin=pin))
