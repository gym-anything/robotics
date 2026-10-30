"""`gar doctor gift-bags`: what this environment needs beyond Newton/Warp/MuJoCo (no simulation).

ipctk (IPC Toolkit Python bindings) supplies contact candidates, barrier/friction derivatives and CCD for the
paper contact world; pypardiso (MKL PARDISO) solves its Newton systems; scipy builds the sparse systems and the
evaluator's hulls; threadpoolctl caps the BLAS/OpenMP pools at the solver thread budget; trimesh loads the robot
meshes; usd-core (pxr) writes the render scene. The Isaac render interpreter serves RGB observations (physics and
evaluation run without it); its worker imports wiping, laundry, whiteboard and gym_anything_robotics modules from the
directory that holds gift_bags (gift_bags/render.py sets the worker's PYTHONPATH to it).
"""
from gym_anything_robotics.paths import project_file
import ast
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# files the Isaac render worker (python -m wiping.render_worker, PYTHONPATH = gym_anything_robotics.paths.worker_path()) loads at start-up
WORKER_FILES = ("wiping/render_worker.py", "wiping/render_updates.py", "wiping/render_payload.py",
                "wiping/render_only.kit", "laundry/__init__.py", "laundry/newton15/details.py",
                "titrate/newton15/render_payload.py", "whiteboard/ipc.py", "whiteboard/render_readback.py",
                "gym_anything_robotics/render_lifecycle.py")


def cpu_quota(root="/sys/fs/cgroup"):
    """(CPUs, source) this process may use: the cgroup v2 cpu.max quota (v1 cfs quota as a fallback), else the CPU
    affinity mask. nproc/os.cpu_count report the host's cores, not the quota."""
    root = Path(root)
    try:
        quota, period = (root/"cpu.max").read_text().split()[:2]
        if quota != "max":
            return float(quota)/float(period), f"{root/'cpu.max'} ({quota} / {period})"
    except (OSError, ValueError):
        pass
    try:
        quota = int((root/"cpu"/"cpu.cfs_quota_us").read_text())
        period = int((root/"cpu"/"cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return quota/period, f"{root/'cpu'/'cpu.cfs_quota_us'} ({quota} / {period})"
    except (OSError, ValueError):
        pass
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return float(n), "CPU affinity mask (no cgroup quota)"


def default_threads():
    """The GIFT_BAGS_THREADS default written in gift_bags/ipcworld.py (read, not imported: importing it applies the
    thread cap to this process)."""
    tree = ast.parse(Path(__file__).with_name("ipcworld.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SOLVER_THREADS" for t in node.targets):
            for call in ast.walk(node.value):
                if isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "get" and len(call.args) == 2 \
                        and isinstance(call.args[1], ast.Constant):
                    return int(call.args[1].value)
    raise ValueError("SOLVER_THREADS default not found in gift_bags/ipcworld.py")


def thread_check(environ=os.environ, quota=None):
    """(ok, detail) for the solver thread budget against the CPU quota."""
    cpus, source = quota or cpu_quota()
    raw = environ.get("GIFT_BAGS_THREADS")
    try:
        threads = int(raw) if raw is not None else default_threads()
    except ValueError:
        return False, f"GIFT_BAGS_THREADS={raw!r} is not an integer (gift_bags/ipcworld.py would fail to import)"
    origin = "set in the environment" if raw is not None else "default in gift_bags/ipcworld.py"
    detail = f"GIFT_BAGS_THREADS={threads} ({origin}); CPU quota {cpus:g} from {source}"
    preset = {k: environ[k] for k in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS") if k in environ}
    if preset:
        detail += "; already set (ipcworld.py keeps them): " + ", ".join(f"{k}={v}" for k, v in preset.items())
    if threads < 1:
        return False, detail + ": must be >= 1"
    if threads > cpus:
        return False, detail + (": more solver threads than the quota allows (oversubscription made PARDISO up to "
                                "100x slower); lower GIFT_BAGS_THREADS")
    over = [k for k, v in preset.items() if v.isdigit() and int(v) > cpus]
    if over and importlib.util.find_spec("threadpoolctl") is None:
        return False, detail + f": {', '.join(over)} exceed the quota and threadpoolctl is missing to cap them"
    return True, detail + ". Other processes share the quota: leave headroom when several simulations run"


def gpu_check(environ=os.environ, query=None):
    """(ok, detail): is there GPU memory for a session? A session keeps the robot model on the GPU (about 0.7 GB)
    unless GIFT_BAGS_DEVICE=cpu, and a render worker (RGB observations) takes about 4 GB. On a card filled by other
    jobs a launch dies with a CUDA out-of-memory error; this says so beforehand. `query`: () -> (free MiB, total MiB)
    for tests; default nvidia-smi."""
    device = environ.get("GIFT_BAGS_DEVICE", "cuda:0")
    try:
        if query is None:
            import subprocess
            line = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10, check=True).stdout.splitlines()[0]
            free, total = (float(v) for v in line.split(","))
        else:
            free, total = query()
    except Exception as exc:
        if device == "cpu":
            return True, f"GIFT_BAGS_DEVICE=cpu: physics needs no GPU; images do, and no GPU could be queried ({type(exc).__name__})"
        return False, (f"no GPU could be queried ({type(exc).__name__}): set GIFT_BAGS_DEVICE=cpu to run physics and "
                       "evaluation without one (no images)")
    need_sim, need_render = 700., 4000.
    head = f"{free:.0f} of {total:.0f} MiB free"
    if device == "cpu":
        ok, tail = True, "GIFT_BAGS_DEVICE=cpu: the robot model is kept off the GPU"
    else:
        ok = free >= need_sim
        tail = (f"a session needs about {need_sim:.0f} MiB for the robot model" if ok else
                f"under the {need_sim:.0f} MiB a session's robot model needs: a launch would fail with a CUDA "
                "out-of-memory error; set GIFT_BAGS_DEVICE=cpu or free the card")
    render = ("" if free >= need_render + (0. if device == "cpu" else need_sim) else
              f"; under the {need_render:.0f} MiB a render worker needs: RGB observations would fail")
    return ok, f"{head}; {tail}{render}"


def checks(spec):
    out = []

    def add(name, ok, detail):
        out.append(dict(check=name, ok=bool(ok), detail=detail))
    for mod, req in (("ipctk", "ipctk>=1.6 (IPC Toolkit bindings)"), ("pypardiso", "pypardiso>=0.4 (MKL PARDISO)"),
                     ("scipy", "scipy"), ("PIL", "pillow (artwork)"),
                     ("threadpoolctl", "threadpoolctl (caps BLAS/OpenMP at the solver thread budget; without it "
                                       "gift_bags/ipcworld.py skips the cap silently)"),
                     ("trimesh", "trimesh (robot meshes, gift_bags/rig.py)"),
                     ("pxr", "usd-core (pxr: render scene export, gift_bags/render.py)")):
        add(f"import {mod}", importlib.util.find_spec(mod) is not None, req)
    isaac = os.environ.get("ISAAC_RENDER_PYTHON")
    add("ISAAC_RENDER_PYTHON (RGB observations only)", bool(isaac) and Path(isaac).is_file(),
        isaac or "unset: point it at Isaac Sim's python")
    missing = [f for f in WORKER_FILES if not project_file(f).is_file()]
    add("render worker modules beside gift_bags (RGB observations only)", not missing,
        f"{ROOT}: " + (f"missing {', '.join(missing)} (install the full gym-anything-robotics package)" if missing
                       else "wiping, laundry, titrate.newton15, whiteboard, gym_anything_robotics present"))
    cache = ROOT/".cache"/"visual_assets"
    add("pinned visual assets cache (render dressing)", cache.is_dir(),
        f"{cache} (python -m gift_bags.visual_assets fetches and checks the SHA-256 pins)")
    add("solver threads within the CPU quota", *thread_check())
    add("GPU memory for the robot model and for images", *gpu_check())
    # start states: a preset whose cached start state is unsound cannot be reset
    try:
        from .variants import availability
        for preset, st in availability(refresh=True).items():
            ss = st.get("start_state") or {}
            ready, total = st.get("start_states_ready", (None, None))
            rows = st.get("start_states") or []
            where = {w_: sum(r["where"] == w_ for r in rows) for w_ in ("shipped", "cache", "missing")}
            listing = (f"{ready} of {total} start state(s) ready ({where['shipped']} shipped with the code, {where['cache']} in "
                       f"the local cache, {where['missing']} missing: a launch that needs a missing one builds it first; "
                       "list: python -m gift_bags.prebuild)") if total else "no start state needed"
            detail = st["reason"] if not st["available"] else (
                (f"sound (key {ss['key']}); " if ss.get("cached") else "") + listing
                + (". " + st["note"] if st.get("note") and where["missing"] else ""))
            add(f"start state of preset {preset}", st["available"], detail)
    except Exception as exc:
        add("start states", False, f"check failed: {type(exc).__name__}: {exc}")
    return dict(ok=all(c["ok"] for c in out), checks=out,
                note="RGB observations need the Isaac render interpreter; physics and evaluation do not")
