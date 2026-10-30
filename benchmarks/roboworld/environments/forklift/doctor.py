"""`gar doctor forklift`: what this environment needs beyond Newton/Warp/MuJoCo (no simulation, no renderer started).

Checked:
- the forklift assets against forklift/assets/assets.lock.json (the NVIDIA Forklift C and Simple_Warehouse content the
  truck builder reads at every launch, and the textures and props of the scene): every locked file present with its
  locked size, and the SHA-256 of a fixed sample of them (the full hash check is
  `python -m forklift.assets.install --verify`). Cache: $GAR_FORKLIFT_ASSETS or forklift/.cache/assets;
- the cab's collision decomposition, shipped in forklift/data and locked by its SHA-256 (geometry.CAB_HULLS_SHA256,
  since a regenerated CoACD decomposition differs run to run);
- the Unitree H1-2's locked files (robots/models/h1_2, resolved and hash-checked by robots.assets; ~/.cache/gar/robots/
  h1_2 or $GAR_ROBOT_ASSETS/h1_2);
- every offered robot's rig and builder (robot_setup.ALIASES: the shared URDF builder, or a model's own builder such as
  the driving-robot kit's robots/models/drive_kit/newton.py), imported;
- the Python packages a world build imports (SciPy, OpenUSD's pxr, trimesh, Pillow);
- the head camera's renderer: ISAAC_RENDER_PYTHON (the Isaac interpreter of every RGB observation), and the modules the
  render worker imports from elsewhere (whiteboard.ipc, gym_anything_robotics.render_lifecycle) and the render-only
  kit file it starts with (wiping/render_only.kit).
A failed required check carries the command that fixes it; the result's hint lists them.

  python -m forklift.doctor [--json]       # the same checks; exit 1 when a required one fails
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

HASH_SAMPLE = 48           # locked files whose SHA-256 the doctor checks (evenly spaced in path order; about 0.5 s)
FIX_ASSETS = "python -m forklift.assets.install"
FIX_H1_2 = "python -m robots.models.h1_2.install"


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def forklift_assets():
    """(ok, detail, fix) of the forklift asset cache against the lock: presence and size of every file, and the hash of
    a fixed sample."""
    from forklift.assets.install import LOCK, cache_dir
    root = cache_dir()
    files = json.loads(LOCK.read_text())["files"]
    missing, wrong_size = [], []
    for key, e in sorted(files.items()):
        p = root / key
        if not p.is_file():
            missing.append(key)
        elif e.get("bytes") and p.stat().st_size != e["bytes"]:
            wrong_size.append(key)
    keys = sorted(files)
    step = max(1, len(keys) // HASH_SAMPLE)
    sample = [k for k in keys[::step][:HASH_SAMPLE] if k not in missing and k not in wrong_size]
    bad_hash = [k for k in sample if files[k].get("sha256") and _sha(root / k) != files[k]["sha256"]]
    where = f"{root}" + (" ($GAR_FORKLIFT_ASSETS)" if os.environ.get("GAR_FORKLIFT_ASSETS") else "")
    if not (missing or wrong_size or bad_hash):
        return True, f"{len(files)} locked files present with their sizes in {where}; {len(sample)} hashes checked", None
    parts = [f"{n} {what}" for n, what in ((len(missing), "missing"), (len(wrong_size), "of the wrong size"),
                                           (len(bad_hash), "of the sample differing from the lock")) if n]
    first = (missing + wrong_size + bad_hash)[0]
    return False, (f"{', '.join(parts)} of {len(files)} locked files in {where} (first: {first}); run {FIX_ASSETS} "
                   f"(downloads what is missing, about 2.5 GB in all, and verifies every hash)"), FIX_ASSETS


def h1_2_assets():
    """(ok, detail, fix) of the H1-2's locked files (robots.assets resolves and hash-checks them)."""
    try:
        from robots.assets import resolve
        return True, str(resolve("h1_2")), None
    except Exception as exc:
        where = os.environ.get("GAR_ROBOT_ASSETS")
        fix = FIX_H1_2 + ("; it installs into $GAR_ROBOT_ASSETS/h1_2" if where else "; it installs into ~/.cache/gar/robots/h1_2")
        return False, f"{type(exc).__name__}: {exc}", fix


def offered_robots():
    """[(alias, rig, model, ok, detail, fix)] for every robot forklift offers (robot_setup.ALIASES): the rig resolves
    from the catalogue, and the code that builds it imports: the shared URDF builder for a URDF model, the model's own
    Newton builder (robots/models/<model>/newton.py) for an authored one such as the driving-robot kit."""
    import importlib
    from forklift.robot_setup import ALIASES
    out = []
    for alias, rig in ALIASES.items():
        try:
            import robots
            inst = next(iter(robots.resolve(rig).to_dict()["instances"].values()))
            mid = inst["model"].split("@")[0]
            fmt = robots.model(inst["model"]).to_dict()["asset"]["format"]
            mod = "robots.backends.newton" if fmt == "urdf" else f"robots.models.{mid}.newton"
            importlib.import_module(mod)
            out.append((alias, rig, mid, True, f"rig {rig}, model {mid} ({fmt}); builder {mod} imports", None))
        except Exception as exc:
            out.append((alias, rig, None, False, f"{type(exc).__name__}: {exc}",
                        "install the whole gym-anything-robotics tree (the wheel must carry every robot's builder "
                        "package, e.g. robots.models.drive_kit)"))
    return out


def checks(spec):
    out = []

    def add(name, ok, detail, fix=None, required=True):
        out.append(dict(check=name, ok=bool(ok), detail=detail, required=required, **({"fix": fix} if fix and not ok else {})))

    ok, detail, fix = forklift_assets()
    add("forklift assets (assets.lock.json)", ok, detail, fix)
    from forklift.geometry import cab_hulls_status
    ok, detail = cab_hulls_status()
    add("cab collision decomposition (forklift/data/cab_hulls_coacd.npz, hash-locked)", ok, detail,
        "restore forklift/data/cab_hulls_coacd.npz from the package (git checkout -- "
        "benchmarks/roboworld/environments/forklift/data/cab_hulls_coacd.npz)")
    ok, detail, fix = h1_2_assets()
    add("robot assets: h1_2 (Unitree H1-2 with Inspire hands, the seated operator)", ok, detail, fix)
    for alias, rig, mid, ok, detail, fix in offered_robots():
        add(f"robot builder: --robot {alias}", ok, detail, fix)
    for mod, what in (("scipy", "SciPy (poses, the operator's posture solve)"), ("pxr", "OpenUSD (the truck's meshes)"),
                      ("trimesh", "trimesh (collision hulls)"), ("PIL", "Pillow (scene textures)")):
        add(f"import {mod}", importlib.util.find_spec(mod) is not None, what,
            "install it into this interpreter (pip install scipy usd-core trimesh pillow)")
    isaac = os.environ.get("ISAAC_RENDER_PYTHON")
    add("ISAAC_RENDER_PYTHON (head camera; every RGB observation)", bool(isaac) and Path(isaac).is_file(),
        isaac or "unset: a session with the head camera (the default) fails at its first observation",
        "set ISAAC_RENDER_PYTHON to Isaac Sim 5.1's python")
    for mod in ("whiteboard.ipc", "gym_anything_robotics.render_lifecycle"):
        try:
            found = importlib.util.find_spec(mod) is not None
        except ModuleNotFoundError:
            found = False
        add(f"module {mod} (render worker)", found, "imported by the render worker and its client (forklift/render.py)",
            "install the whole gym-anything-robotics tree (the wheel carries whiteboard and gym_anything_robotics)")
    kit = None
    try:
        spec_w = importlib.util.find_spec("wiping")
        if spec_w is not None and spec_w.submodule_search_locations:
            kit = Path(list(spec_w.submodule_search_locations)[0]) / "render_only.kit"
    except ModuleNotFoundError:
        pass
    add("wiping/render_only.kit (render worker experience)", kit is not None and kit.is_file(), str(kit),
        "install the whole gym-anything-robotics tree (the wheel carries wiping's render_only.kit)")
    required_failed = [c for c in out if c["required"] and not c["ok"]]
    ok = not required_failed
    hint = None if ok else "Run: " + "; ".join(dict.fromkeys(c["fix"] for c in required_failed if c.get("fix")))
    return dict(ok=ok, checks=out, hint=hint,
                note="imports, files and variables only; nothing is simulated or rendered. Wall time and the client "
                     "timeout: the session's capabilities (hardware.wall_time, hardware.client_timeout_s)")


def main(argv=None):
    import argparse
    import sys
    from types import SimpleNamespace
    ap = argparse.ArgumentParser(prog="python -m forklift.doctor", description="the checks of `gar doctor forklift`")
    ap.add_argument("--json", action="store_true")
    res = checks(SimpleNamespace(robot=None))
    if ap.parse_args(argv).json:
        print(json.dumps(res, indent=1))
    else:
        for c in res["checks"]:
            print(f"{'ok  ' if c['ok'] else 'FAIL'} {c['check']}: {c['detail']}" + (f"\n     fix: {c['fix']}" if c.get("fix") else ""))
        print("OK" if res["ok"] else res["hint"])
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
