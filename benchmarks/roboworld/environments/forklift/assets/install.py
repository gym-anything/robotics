"""Install the locked third-party visual assets of the forklift environment (never blesses new hashes).

Sources
- NVIDIA Isaac Sim 5.1 content (the "Forklift C" truck and the Simple_Warehouse kit), from NVIDIA's public asset
  server `ROOT_URL`. Used at runtime by the render-only Isaac worker and by the truck builder (meshes); not committed.
  Terms: NVIDIA Isaac Sim / Omniverse content licence (see forklift/ATTRIBUTIONS.md).
- The yard sky HDR from NVIDIA's Skies collection (`extra_sources.json`), same terms.
- Everything else the scene draws (`extra_sources*.json`, one file per visual module): single files by URL
  (Poly Haven CC0 texture maps), and whole folders of NVIDIA's public content bucket by prefix (SimReady props with their
  materials and textures), cached under `nvidia/<bucket key>`.

Usage
  python -m forklift.assets.install            # download what is missing into the cache, verify every file
  python -m forklift.assets.install --fetch-extra   # (maintainers) download the extra_sources*.json entries, unlocked
  python -m forklift.assets.install --lock     # (maintainers) write assets.lock.json from the files in the cache
The cache is `$GAR_FORKLIFT_ASSETS` or `forklift/.cache/assets`.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
LOCK = HERE / "assets.lock.json"
ROOT_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/"
POLYHAVEN = "https://dl.polyhaven.org/file/ph-assets/"

# What the environment uses. NVIDIA entries are paths below ROOT_URL; the cache mirrors them.
FORKLIFT_PREFIX = "Isaac/Robots/IsaacSim/ForkliftC/"
WAREHOUSE_PREFIX = "Isaac/Environments/Simple_Warehouse/"


# Build-time tool (not a runtime dependency): CoACD convex decomposition of the cab interior for collision geometry
# (collision = visual in the region the robot's legs and hands share with the cab). Installed into a private folder,
# never into a shared interpreter. MIT licence (https://colin97.github.io/CoACD/).
TOOLS = {
    "coacd": dict(url="https://files.pythonhosted.org/packages/e4/2e/a05a4dc7a4aaa32fda7d409b32b0f12da7e421713041608ce52f2248626a/"
                      "coacd-1.0.5-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
                  sha256="70439f5170dfb8b249615650c13203c65459d34b2524fca8ef6658992b12355c", version="1.0.5"),
}


def tools_dir():
    """Private folder of the build-time tools: a checkout's forklift/.cache/pydeps where it already exists, else
    `.tools` in the asset cache (an installed copy's package folder is not writable; the cache is)."""
    legacy = PKG / ".cache" / "pydeps"
    return legacy if legacy.is_dir() else cache_dir() / ".tools"


def install_tools():
    import subprocess
    import tempfile
    for name, t in TOOLS.items():
        if (tools_dir() / name).is_dir():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            whl = Path(tmp) / t["url"].rsplit("/", 1)[1]
            _fetch(t["url"], whl)
            if _sha(whl) != t["sha256"]:
                raise SystemExit(f"{name} wheel hash mismatch")
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(tools_dir()), str(whl)],
                           check=True)
    return tools_dir()


def cache_dir():
    value = os.environ.get("GAR_FORKLIFT_ASSETS")
    return Path(value).expanduser() if value else PKG / ".cache" / "assets"


def _list(prefix):
    """All keys below an NVIDIA prefix (the bucket allows anonymous ListObjectsV2)."""
    keys, token = [], None
    while True:
        url = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/?list-type=2&prefix=" + \
              urllib.parse.quote("Assets/Isaac/5.1/" + prefix)
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        text = urllib.request.urlopen(url, timeout=60).read().decode()
        for k, s in zip(re.findall(r"<Key>([^<]+)</Key>", text), re.findall(r"<Size>([^<]+)</Size>", text)):
            keys.append((k[len("Assets/Isaac/5.1/"):], int(s)))
        if "<IsTruncated>true" not in text:
            return keys
        token = re.findall(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text)[0]


BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"


def _list_bucket(prefix):
    """All (key, size) below a prefix of NVIDIA's public content bucket (keys relative to the bucket root)."""
    import urllib.parse
    keys, token = [], None
    while True:
        url = BUCKET + "?list-type=2&prefix=" + urllib.parse.quote(prefix)
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        text = urllib.request.urlopen(url, timeout=60).read().decode()
        for k, sz in zip(re.findall(r"<Key>([^<]+)</Key>", text), re.findall(r"<Size>([^<]+)</Size>", text)):
            if not k.endswith("/") and "/.thumbs/" not in k:
                keys.append((k, int(sz)))
        if "<IsTruncated>true" not in text:
            return keys
        token = re.findall(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text)[0]


def extra_entries(listing=True):
    """{cache path: url} of every extra_sources*.json entry. A {"path", "url"} entry is one file; a {"prefix"} entry is
    a whole folder of NVIDIA's bucket (cached as nvidia/<key>; listed from the server when `listing`)."""
    import urllib.parse
    out = {}
    for f in sorted(HERE.glob("extra_sources*.json")):
        for e in json.loads(f.read_text()).get("files", []):
            if "prefix" in e:
                if listing:
                    for k, _ in _list_bucket(e["prefix"]):
                        out["nvidia/" + k] = BUCKET + urllib.parse.quote(k)
            else:
                out[e["path"]] = e["url"]
    return out


def fetch_extra():
    """Download every extra entry that is not in the cache yet (hashes are pinned later by --lock)."""
    root, n = cache_dir(), 0
    for path, url in sorted(extra_entries().items()):
        p = root / path
        if not p.is_file():
            print("fetch", path, flush=True)
            _fetch(url, p)
            n += 1
    print(f"{n} extra files fetched into {root}")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "gym-anything-robotics/forklift-assets"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def write_lock():
    import urllib.parse  # noqa: F401
    entries = {}
    for prefix in (FORKLIFT_PREFIX, WAREHOUSE_PREFIX):
        for key, size in _list(prefix):
            if "/.thumbs/" in key or key.endswith(".last_generated") or "rotated_in_progress" in key \
                    or "updatemat" in key or "forklift_b" in key or key.endswith(("full_warehouse.usd",
                                                                                 "warehouse_multiple_shelves.usd")):
                continue
            entries[key] = dict(url=ROOT_URL + key, bytes=size)
    for path, url in extra_entries().items():
        entries[path] = dict(url=url, bytes=None)
    root = cache_dir()
    for key, e in entries.items():
        p = root / key
        if not p.is_file():
            raise SystemExit(f"missing in cache, run install first: {key}")
        e["bytes"] = p.stat().st_size
        e["sha256"] = _sha(p)
    LOCK.write_text(json.dumps(dict(schema="forklift.assets.lock.v1", root_url=ROOT_URL, files=entries), indent=1,
                               sort_keys=True) + "\n")
    print(f"wrote {LOCK} ({len(entries)} files)")


def install(verify_only=False):
    import urllib.parse  # noqa: F401
    root = cache_dir()
    if LOCK.is_file():
        lock = json.loads(LOCK.read_text())["files"]
    else:                                   # bootstrap: list the server (hashes are then written with --lock)
        lock = {k: dict(url=ROOT_URL + k, bytes=s) for k, s in _list(FORKLIFT_PREFIX)
                if "/.thumbs/" not in k and not k.endswith(".last_generated") and "rotated_in_progress" not in k
                and "updatemat" not in k and "forklift_b" not in k}
        for path, url in extra_entries().items():
            lock[path] = dict(url=url, bytes=None)
    bad = []
    for key, e in sorted(lock.items()):
        p = root / key
        if not p.is_file() or (e.get("bytes") and p.stat().st_size != e["bytes"]):
            if verify_only:
                bad.append(key)
                continue
            print("fetch", key, flush=True)
            _fetch(e["url"], p)
        if e.get("sha256") and _sha(p) != e["sha256"]:
            bad.append(key)
    if bad:
        raise SystemExit("assets differ from the lock or are missing: " + ", ".join(bad[:10]))
    print(f"{len(lock)} asset files verified in {root}")
    if not verify_only:
        install_tools()
    return root


def fetch_prefix(prefix):
    """Download every file below an NVIDIA prefix into the cache (bootstrap before --lock)."""
    import urllib.parse  # noqa: F401
    root = cache_dir()
    n = 0
    for key, size in _list(prefix):
        if "/.thumbs/" in key:
            continue
        p = root / key
        if p.is_file() and p.stat().st_size == size:
            continue
        _fetch(ROOT_URL + key, p)
        n += 1
    return n


def resolve(key):
    """Absolute path of a locked asset (verified by size; run install for the full hash check). NVIDIA Isaac content by
    its path below ROOT_URL; extras by their cache path (a Poly Haven map by its `path`, a bucket folder's file as
    nvidia/<bucket key>)."""
    root = cache_dir()
    p = root / key
    if not p.is_file():
        raise FileNotFoundError(f"forklift asset {key} is not installed: run python -m forklift.assets.install")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--fetch-extra", action="store_true")
    a = ap.parse_args()
    if a.fetch_extra:
        fetch_extra()
    elif a.lock:
        write_lock()
    else:
        install(verify_only=a.verify)
    sys.exit(0)
