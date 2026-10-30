"""Hash-pinned, locally served viewer dependencies; no runtime CDN requests."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

FILES = {
    "three.module.js": (
        "build/three.module.js",
        "c8211c69345d2e9949dc7a8ac969380497aa0600a5a8ac6a459c8cd02dd9cb8a",
    ),
    "three.core.js": (
        "build/three.core.js",
        "eb077d2417f61d3e6d9264c317cabc4ea35769ed6b0ab533067292a550784c20",
    ),
    "OrbitControls.js": (
        "examples/jsm/controls/OrbitControls.js",
        "b97879c748170baadeb3fb84cea1ffdf4674e283dc06042f34e2acb95a76042c",
    ),
    "LICENSE": ("LICENSE", "bfe119ea4fd413f5f7ca3fcd63adb0c4a073ed39daa2fe7d3e6b769e21272601"),
}
ROOT = Path(__file__).resolve().parent / ".cache/web_vendor"


def provision():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, (source, digest) in FILES.items():
        path = ROOT / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        with urlopen(
            "https://raw.githubusercontent.com/mrdoob/three.js/r180/" + source, timeout=60
        ) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("Viewer dependency hash mismatch: " + name)
        path.write_bytes(data)
    print("Verified Three.js r180 modules and MIT license:", ROOT)


if __name__ == "__main__":
    provision()
