"""Readiness checks.

`checks(spec)` is what `gar doctor surface-rinsing` calls: imports, files, GPU visibility and environment variables only --
no simulator, no renderer, no network. `python -m surface_rinsing.doctor [--render]` is the deeper smoke test: pinned
assets fetched and hash-checked, every preset built, settled and stepped for 0.5 s with the water system, ledger checked,
optionally one head-camera render."""
import argparse
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
    for mod in ("numpy", "scipy", "PIL", "trimesh", "mujoco", "newton", "warp"):
        add(f"import {mod}", importlib.util.find_spec(mod) is not None, "rigid world, water kernels, geometry, images")
    try:
        import warp as wp
        wp.config.log_level = wp.LOG_WARNING
        wp.init()
        cuda = [d for d in wp.get_devices() if d.is_cuda]
        add("CUDA device for the water system (Warp)", bool(cuda), ", ".join(str(d) for d in cuda) or
            "no CUDA device: the water kernels need cuda:0")
    except Exception as exc:
        add("CUDA device for the water system (Warp)", False, repr(exc))
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
    man = PKG/"visual_assets.json"
    try:
        pinned = json.loads(man.read_text())
        cache = PKG/".cache"/"visual_assets"
        missing = [n for n, e in pinned.items() for rel in e["files"] if not (cache/n/rel).is_file()]
        add("pinned CC0 dressing and textures (visual_assets.json)", True,
            f"{len(pinned)} assets pinned; " + ("all cached" if not missing else
                                                f"{len(missing)} files not cached: the first render downloads and hash-checks them"))
    except Exception as exc:
        add("pinned CC0 dressing and textures (visual_assets.json)", False, repr(exc))
    for rel in ("benchmark.json", "variations.json", "SurfaceRinsing.mdl"):
        p = PKG/rel
        add(f"package file {rel}", p.is_file(), str(p))
    return dict(ok=all(c["ok"] for c in out), checks=out,
                note="imports, files, GPU and variables only; `python -m surface_rinsing.doctor` builds and steps every preset")


def smoke(render=False):
    from .config import Config
    from .env import RinsingEnv
    from .layout import PRESETS
    rows = []
    for preset, tasks in PRESETS.items():
        t0 = time.time()
        env = RinsingEnv(Config(task=tasks[0]))
        w = env.world
        t1 = time.time()
        for _ in range(1000):
            w.step()
        led = env.water.ledger_residual()
        rows.append(dict(preset=preset, task=tasks[0], build_settle_s=round(t1 - t0, 1), step_0p5s_wall_s=round(time.time() - t1, 1),
                         settle=env.settle_record, reflexes=len(w.reflexes), fault=w.fault, ledger_relative=led["relative"],
                         cells=int(env.water.surf.n)))
        print(json.dumps(rows[-1]), flush=True)
        if render:
            from .render import export, Renderer
            out = Path("/tmp")/f"rinsing-doctor-{preset}"
            scene, meta = export(env, out)
            r = Renderer(out, scene, meta)
            try:
                img = r.observe(env, ["head_rgb"])["head_rgb"]
                (out/"head_rgb.png").write_bytes(img)
                rows[-1]["render"] = str(out/"head_rgb.png")
            finally:
                r.close()
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    print(json.dumps(checks(None), indent=1))
    print(json.dumps(smoke(a.render), indent=1))
