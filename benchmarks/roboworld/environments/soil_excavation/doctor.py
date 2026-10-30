"""Readiness checks.

`checks(spec)` is what `gar doctor soil-excavation` calls: imports, files, environment variables and the GPU's free
memory only -- no simulator, no renderer, no network. `python -m soil_excavation.doctor [--render]` is the deeper
smoke test: pinned assets fetched and hash-checked, every preset's start state built, settled and stepped for 0.5 s
(1000 ticks) with its wall/sim ratio, faults and protective stops, optionally one head-camera render.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
SESSION_GPU_MIB = 9000          # one session: soil world ~5 GB + render worker ~4 GB


def _gpu_free_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return int(out[0])
    except Exception:
        return None


def checks(spec):
    out = []

    def add(name, ok, detail):
        out.append(dict(check=name, ok=bool(ok), detail=detail))
    for mod, why in (("numpy", "state"), ("scipy", "rotations, soil geometry"), ("PIL", "images"),
                     ("trimesh", "tool meshes"), ("warp", "soil kernels"), ("mujoco", "rigid world"), ("newton", "physics")):
        add(f"import {mod}", importlib.util.find_spec(mod) is not None, why)
    robot = getattr(spec, "robot", None)
    model = robot if isinstance(robot, str) and robot in ("mobile-fr3-duo", "mobile-fr3-single") else "mobile-fr3-duo"
    try:
        from robots import assets
        add(f"{model} robot description (locked)", True, str(assets.resolve(model)))
    except Exception as exc:
        add(f"{model} robot description (locked)", False, f"{exc}; build it with python -m classroom_arrangement.robot_asset")
    isaac = os.environ.get("ISAAC_RENDER_PYTHON")
    add("ISAAC_RENDER_PYTHON (RGB observations, videos)", bool(isaac) and Path(isaac).is_file(),
        isaac or "unset: point it at the Isaac Sim python interpreter; without it create the session with cameras=()")
    try:
        pinned = json.loads((PKG/"visual_assets.json").read_text())
        cache = PKG/".cache"/"visual_assets"
        missing = [n for n, e in pinned.items() for rel in e["files"] if not (cache/n/rel).is_file()]
        add("pinned CC0 dressing (visual_assets.json)", True,
            f"{len(pinned)} assets pinned; " + ("all cached" if not missing else
                                                f"{len(missing)} files not cached: the first render downloads and hash-checks them"))
    except Exception as exc:
        add("pinned CC0 dressing (visual_assets.json)", False, repr(exc))
    for rel in ("benchmark.json", "variations.json", "soil_surfaces.mdl"):
        p = PKG/rel
        add(f"package file {rel}", p.is_file(), str(p))
    free = _gpu_free_mib()
    add("GPU memory for one session (soil world + render worker)", free is not None and free >= SESSION_GPU_MIB,
        "nvidia-smi unavailable" if free is None else f"{free} MiB free, {SESSION_GPU_MIB} MiB needed with a camera "
        "(about 5000 without)")
    return dict(ok=all(c["ok"] for c in out), checks=out,
                note="imports, files, variables and free GPU memory only; `python -m soil_excavation.doctor` builds and "
                     "steps every preset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--presets", default=None, help="comma list of presets (default: all six)")
    a = ap.parse_args()
    ok = True
    v = os.environ.get("ISAAC_RENDER_PYTHON")
    print(f"ISAAC_RENDER_PYTHON: {v or 'MISSING (set it to the Isaac Sim python interpreter)'}")
    from soil_excavation import visual_assets as va
    for n in list(va.MODELS):
        try:
            va.fetch(n, "model")
        except Exception as exc:
            print("asset", n, "FAILED", exc); ok = False
    for n in list(va.TEXTURES):
        try:
            va.fetch(n, "texture")
        except Exception as exc:
            print("asset", n, "FAILED", exc); ok = False
    print("pinned assets: checked")
    from soil_excavation.config import Config
    from soil_excavation.env import SoilEnv
    from soil_excavation.layout import PRESETS, TASKS
    first = {p: next(t for t, v in TASKS.items() if v["preset"] == p) for p in PRESETS}
    presets = a.presets.split(",") if a.presets else list(first)
    for preset in presets:
        task = first[preset]
        t0 = time.perf_counter()
        env = SoilEnv(Config(task=task))
        t1 = time.perf_counter()
        env.act(ticks=1000, state=False)
        t2 = time.perf_counter()
        w = env.world
        print(f"{preset} ({task}): {w.soil.n} particles, build+settle {t1 - t0:.1f} s, 0.5 s step {t2 - t1:.1f} s "
              f"({(t2 - t1)/0.5:.0f}x), fault {w.fault}, stops {len(w.reflexes)}", flush=True)
        ok &= w.fault is None and not w.reflexes
        if a.render and preset == presets[0]:
            import tempfile
            from soil_excavation.render import export, Renderer
            with tempfile.TemporaryDirectory() as d:
                scene, meta = export(env, Path(d))
                r = Renderer(Path(d), scene, meta, width=640, height=360, samples=8)
                try:
                    imgs = r.observe(env, ["head_rgb"], cell=0.5*w.soil.h)
                finally:
                    r.close()
                print("head_rgb render:", {k: len(v) for k, v in imgs.items()} if imgs else "FAILED")
                ok &= bool(imgs)
        w.soil = None
        del env
    print("DOCTOR", "OK" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
