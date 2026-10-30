"""Readiness checks.

`checks(spec)` is what `gar doctor painting` calls: imports, files and environment variables only -- no simulator, no
renderer, no network. `python -m painting.doctor [--render]` is the deeper smoke test: pinned assets fetched and
hash-checked, every preset built, settled and stepped for 0.5 s (1000 ticks), optionally one head-camera render."""
import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent


def checks(spec):
    out = []

    def add(name, ok, detail):
        out.append(dict(check=name, ok=bool(ok), detail=detail))
    for mod in ("numpy", "scipy", "PIL", "trimesh", "mujoco", "newton"):
        add(f"import {mod}", importlib.util.find_spec(mod) is not None, "paint fields, contact, images, physics")
    robot = getattr(spec, "robot", None)
    model = robot if isinstance(robot, str) and robot in ("mobile-fr3-duo", "mobile-fr3-single", "g1") else "mobile-fr3-duo"
    try:
        from robots import assets
        add(f"{model} robot description (locked)", True, str(assets.resolve(model)))
    except Exception as exc:
        add(f"{model} robot description (locked)", False, f"{exc}; build it with python -m classroom_arrangement.robot_asset")
    isaac = os.environ.get("ISAAC_RENDER_PYTHON")
    add("ISAAC_RENDER_PYTHON (RGB observations, videos)", bool(isaac) and Path(isaac).is_file(),
        isaac or "unset: point it at the Isaac Sim python interpreter; without it create the session with cameras=()")
    man = PKG/"visual_assets.json"
    try:
        pinned = json.loads(man.read_text())
        cache = PKG/".cache"/"visual_assets"
        missing = [n for n, e in pinned.items() for rel in e["files"] if not (cache/n/rel).is_file()]
        add("pinned CC0 dressing (visual_assets.json)", True,
            f"{len(pinned)} assets pinned; " + ("all cached" if not missing else
                                                f"{len(missing)} files not cached: the first render downloads and hash-checks them"))
    except Exception as exc:
        add("pinned CC0 dressing (visual_assets.json)", False, repr(exc))
    for rel in ("benchmark.json", "variations.json"):
        p = PKG/rel
        add(f"task catalogue {rel}", p.is_file(), str(p))
    return dict(ok=all(c["ok"] for c in out), checks=out,
                note="imports, files and variables only; `python -m painting.doctor` builds and steps every preset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    ok = True
    for var in ("ISAAC_RENDER_PYTHON",):
        v = os.environ.get(var)
        print(f"{var}: {v or 'MISSING (set it to the Isaac Sim python interpreter)'}")
        ok &= bool(v)
    from painting import visual_assets as va
    for n in list(va.MODELS) + list(va.TEXTURES):
        try:
            va.fetch(n, "model" if n in va.MODELS else "texture")
        except Exception as exc:
            print("asset", n, "FAILED", exc); ok = False
    print("pinned assets: checked")
    from painting.config import Config
    from painting.env import PaintingEnv
    for task in ("roll-strip", "fill-circle", "spray-stencil"):
        t0 = time.perf_counter()
        env = PaintingEnv(Config(preset=Config.preset_of(task), task=task))
        t1 = time.perf_counter()
        env.act(ticks=1000)
        t2 = time.perf_counter()
        led = env.paint.ledger_residual()["relative"]
        print(f"{task}: build+settle {t1 - t0:.1f} s, 0.5 s step {t2 - t1:.2f} s ({(t2 - t1)/0.5:.1f}x), ledger {led:.1e}, "
              f"stops {len(env.world.reflexes)}")
        ok &= led < 1e-9 and not env.world.reflexes
        if a.render and task == "roll-strip":
            import tempfile
            from painting.render import export, Renderer
            d = tempfile.mkdtemp(prefix="painting-doctor-")
            scene, meta = export(env, d)
            r = Renderer(d, scene, meta, width=320, height=180, samples=4)
            try:
                img = r.observe(env, ("head_rgb",))["head_rgb"]
                print(f"render: head_rgb {len(img)} bytes PNG")
            finally:
                r.close()
    print("OK" if ok else "PROBLEMS FOUND")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
