"""`gar doctor welding`: what this environment needs beyond Newton/Warp/MuJoCo (no simulation).

Checked: the Python packages the process model, the bead surface and the renderer import; a CUDA device for the
thermal solver (Warp); the Isaac render interpreter (every RGB observation and film); the robot assets of every offered
deployment -- the UR10e and the FR3 (the default deployment `ur10e-fr3` needs both; `python -m robots.models.ur10e.install`
installs the UR10e's, Newton downloads the FR3's), the Panda (`dual-panda`) and the Unitree G1 (`g1`), each checked
against its pinned lock (robots/assets.py) and reported as optional unless the launch selects the deployment that needs
it; the pinned visual assets (downloaded on first use, else from the network); the tacked start-state cache (optional:
built on first use). The result also says, for every deployment, whether its robot assets are present (deployments),
and gives a hint naming what to do when a required check fails.

Which deployment the verdict is for: `gar doctor welding` takes no robot option (GAR's doctor builds its launch spec
from the environment alone), so it judges the default ur10e-fr3 and reports the other deployments' assets as optional
rows; `python -m welding.doctor --robot ALIAS` runs the same checks for the named deployment (exit 1 if it cannot
start).
"""
import importlib.util
import os
from pathlib import Path

# robot model -> (the welding deployments that need it, how to provision it). The Panda and G1 assets are git-ignored
# caches shared with the titration and whiteboard environments (robots/assets.py resolves them there, or under
# $GAR_ROBOT_ASSETS/<model> when that is set)
ROBOT_ASSETS = {
    "ur10e": (("ur10e-fr3", "ur10e"), "run python -m robots.models.ur10e.install"),
    "fr3": (("ur10e-fr3", "dual-fr3", "single-fr3"),
            "Newton downloads the pinned franka_emika_panda package on first use (needs the network once)"),
    "panda": (("dual-panda",),
              "run python -m titrate.newton15.assets --isaac-root ISAAC_SIM_5.1_ROOT (copies the 22 pinned "
              "franka_description files from Isaac Sim's URDF importer into titrate/.cache/assets; the root holds "
              "exts/; the pin is the Windows release's CRLF text files, so a Linux pip isaacsim copy needs its two "
              "text files converted first), or set TITRATE_PANDA_URDF to a verified "
              "copy's panda_arm_hand.urdf"),
    "g1": (("g1",),
           "run python -m whiteboard.assets (downloads the 51 pinned unitree_ros files into whiteboard/.cache/assets), "
           "or set GAR_ROBOT_ASSETS to a directory whose g1/ is that locked package"),
}


def _selected(spec):
    robot = getattr(spec, "robot", None)
    if isinstance(robot, dict):              # a setup object: its rig id -> the alias it runs as
        try:
            from robots.catalog import ENVIRONMENTS
            ids = {v: k for k, v in ENVIRONMENTS["welding"]["aliases"].items()}
            return ids.get(robot.get("id"))
        except Exception:
            return None
    return robot


def checks(spec):
    out = []

    def add(name, ok, detail, required=True):
        out.append(dict(check=name, ok=bool(ok), detail=detail if required else f"{detail} (optional)",
                        required=required))
    for mod, req in (("scipy", "scipy (sparse, ndimage, optimize)"), ("trimesh", "trimesh (robot link meshes)"),
                     ("PIL", "pillow (frames)"), ("collada", "pycollada (UR10e and Franka .dae link meshes)"),
                     ("imageio_ffmpeg", "imageio-ffmpeg (films)")):
        add(f"import {mod}", importlib.util.find_spec(mod) is not None, req)
    try:
        import warp as wp
        wp.config.quiet = True
        cuda = wp.get_cuda_device_count() > 0
        add("CUDA device (thermal solver)", cuda, "found" if cuda else "no CUDA device: the weld process needs a GPU")
    except Exception as exc:
        add("CUDA device (thermal solver)", False, f"warp failed: {exc}")
    isaac = os.environ.get("ISAAC_RENDER_PYTHON")
    add("ISAAC_RENDER_PYTHON (every RGB observation and film)", bool(isaac) and Path(isaac).is_file(),
        isaac or "unset: point it at the Isaac Sim python interpreter; a session with cameras fails at its first "
                 "observation (cameras=() for physics only)")
    selected = _selected(spec) or "ur10e-fr3"
    present = {}
    for m, (deployments, how) in ROBOT_ASSETS.items():
        try:
            from robots.assets import resolve
            path, ok = str(resolve(m)), True
        except Exception as exc:
            path, ok = f"{exc}; {how}", False
        present[m] = ok
        # required for the deployment the launch selects (the default ur10e-fr3 when none is given); the others are
        # optional rows that name the deployments that cannot start without them
        add(f"robot assets: {m} (deployments {', '.join(deployments)})", ok, path, required=selected in deployments)
    needs = {}
    for m, (deployments, _) in ROBOT_ASSETS.items():
        for d in deployments:
            needs.setdefault(d, []).append(m)
    deployments = {d: dict(robot_assets_present=all(present[m] for m in ms),
                           missing=[m for m in ms if not present[m]]) for d, ms in needs.items()}
    try:
        from welding import visual_assets as VA
        missing = [n for n in list(VA.MODELS) + list(VA.TEXTURES) + list(VA.HDRIS) if not (VA.CACHE/n).is_dir()]
        add("visual assets cache", not missing, "all pinned assets present" if not missing else
            f"{len(missing)} not cached yet (downloaded and hash-checked on first render): {', '.join(missing)}",
            required=False)
    except Exception as exc:
        add("visual assets manifest", False, str(exc))
    from welding.env import _cache_dir, physics_hash
    h = physics_hash()
    cached = sorted(p.name for p in _cache_dir().glob(f"tacked_{h}*.pkl"))      # one per physics source (env.py)
    add("tacked start-state cache", bool(cached), (f"{len(cached)} cached for this physics source" if cached else
        "none for this physics source: the first fillet-2f/position-torch session builds it once (157 s on a quiet "
        "GPU, up to 45 min on a saturated one), every deployment and catalogue entry then places it on its own world, about 10 s"),
        required=False)
    ok = all(c["ok"] for c in out if c["required"])
    failed = [c["check"] for c in out if c["required"] and not c["ok"]]
    if ok:
        hint = None
    elif all(f.startswith("robot assets") for f in failed):
        hint = (f"The {selected} deployment's robot assets are missing: run the provisioning command in each failed "
                f"robot-assets row, then rerun this check")
    else:
        hint = ("Fix the failed required rows (each row's detail says how), then rerun this check")
    return dict(ok=ok, checks=out, deployment=selected, deployments=deployments, hint=hint,
                note="Use client timeouts of at least 3600 s "
                     "(the backend declares client_timeout_s 3600; the gar CLI uses it). The verdict is for deployment "
                     f"{selected}; for another one run python -m welding.doctor --robot ALIAS. A session with cameras "
                     "also starts an Isaac render worker that holds about 4.2 GiB of GPU memory while warm")


def main(argv=None):
    """python -m welding.doctor [--robot ALIAS] [--json]: the checks of `gar doctor welding` for a named deployment."""
    import argparse
    import json
    from types import SimpleNamespace
    ap = argparse.ArgumentParser(prog="python -m welding.doctor", description=main.__doc__)
    ap.add_argument("--robot", default=None, help="deployment alias (default: ur10e-fr3)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.robot is not None:
        from robots.catalog import ENVIRONMENTS
        aliases = ENVIRONMENTS["welding"]["aliases"]
        if args.robot not in aliases:
            ap.error(f"--robot must be one of {', '.join(aliases)}")
    res = checks(SimpleNamespace(robot=args.robot))
    if args.json:
        print(json.dumps(res, indent=1))
    else:
        print(f"welding ({res['deployment']})  DEPENDENCIES {'OK' if res['ok'] else 'MISSING'}")
        for c in res["checks"]:
            print(f"  {'yes' if c['ok'] else 'NO ':3}  {c['check']}: {c['detail']}")
        print("  deployments with their robot assets present: " +
              (", ".join(d for d, v in res["deployments"].items() if v["robot_assets_present"]) or "none") +
              "; missing: " + (", ".join(f"{d} ({', '.join(v['missing'])})" for d, v in res["deployments"].items()
                                         if not v["robot_assets_present"]) or "none"))
        if res["hint"]:
            print(res["hint"])
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
