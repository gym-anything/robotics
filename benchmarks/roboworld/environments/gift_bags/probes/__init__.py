"""Probes. Benches write their own evidence JSON (`save_bench`), stamped with the command line that produced it
and the code revision, so no evidence file is assembled by hand."""
import json
import subprocess
import sys
import time
from pathlib import Path

BENCHES = Path(__file__).resolve().parents[1]/"evidence"/"benches"


# every module the scene, the world or the evaluator imports, so that two records with one digest come from the same
# code. Among them are dressing.py (the props' collision boxes are authored there, beside their pictures), startcache.py
# (which start-state file a world loads) and robot_setup.py (which rig a name resolves to).
PHYSICS_FILES = ("ipcworld.py", "params.py", "bagnet.py", "handles.py", "wrapmap.py", "giftbox.py", "tape_bonds.py",
                 "carton.py", "g1.py", "variants.py", "world.py", "rig.py", "scene.py", "task.py", "env.py",
                 "kinematics.py", "config.py", "dressing.py", "startcache.py", "robot_setup.py")
# imported by a physics source without taking part in the physics: the start-state build table (readiness only)
# and the artwork and scanned models (pictures)
NOT_PHYSICS = ("prebuild.py", "artwork.py", "visual_assets.py")


def physics_digest():
    """SHA-256 (first 12 hex) over the world's physics sources as they are on disk now: the same digest means the same
    physics whatever the git state (an uncommitted tree included)."""
    import hashlib
    pkg = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for f in PHYSICS_FILES:
        h.update(f.encode()); h.update((pkg/f).read_bytes())
    return h.hexdigest()[:12]


def loaded_physics():
    """({physics file: first 10 hex of its SHA-256}, first 16 hex of ipcworld.py's) as the files were when this
    process first imported the gift_bags package (gift_bags.AT_IMPORT), i.e. as the run loaded them."""
    import gift_bags
    src = gift_bags.AT_IMPORT.get("sources", {})
    return {f: src[f][:10] for f in PHYSICS_FILES if f in src}, (src.get("ipcworld.py") or "")[:16]


def physics_file_hashes():
    """{physics source file: first 10 hex of its SHA-256}: lets two records with different digests be compared file by
    file (which sources differed when each was made)."""
    import hashlib
    pkg = Path(__file__).resolve().parents[1]
    return {f: hashlib.sha256((pkg/f).read_bytes()).hexdigest()[:10] for f in PHYSICS_FILES}


def code_stamp():
    root = Path(__file__).resolve().parents[2]
    run = lambda *a: subprocess.run(a, cwd=root, capture_output=True, text=True).stdout.strip()
    import hashlib
    from .. import params as Pm
    sha = hashlib.sha256((Path(__file__).resolve().parents[1]/"ipcworld.py").read_bytes()).hexdigest()[:16]
    out = dict(git=run("git", "rev-parse", "--short", "HEAD"),
               gift_bags_uncommitted=bool(run("git", "status", "--porcelain", "--", "gift_bags/*.py", "gift_bags/probes",
                                              "gift_bags/demos")),
               physics_digest=physics_digest(), physics_files=physics_file_hashes(),
               # the contact world's physics version and whether this ipcworld.py is registered under it (params.py):
               # digests differ between records for any source edit; records of one version share the contact physics
               contact_world=dict(version=Pm.CONTACT_WORLD_VERSION, ipcworld_sha=sha,
                                  registered=sha in Pm.CONTACT_WORLD_SHA.get(Pm.CONTACT_WORLD_VERSION, [])))
    # What the process LOADED: this module may be imported long after the run began (a demo that imports it only to
    # write its record). The package itself noted its sources' hashes at its first import; if a physics file on disk
    # now differs from that, the stamp says so and names the loaded hashes, which are the ones that count.
    try:
        loaded, loaded_ipc = loaded_physics()
        changed = sorted(f for f, h in loaded.items() if out["physics_files"].get(f) != h)
        out["loaded_at_package_import"] = dict(
            physics_files_equal_to_this_stamp=not changed, files_that_differ=changed,
            ipcworld_sha=loaded_ipc, ipcworld_registered=loaded_ipc in Pm.CONTACT_WORLD_SHA.get(Pm.CONTACT_WORLD_VERSION, []),
            physics_files=loaded if changed else None)
    except Exception as exc:
        out["loaded_at_package_import"] = dict(error=repr(exc))
    return out


# The stamp is taken when this module is imported, i.e. when the probe or demo starts: a long run must be stamped with
# the sources it loaded, not with the tree as it is hours later when the record is written.
try:
    STAMP_AT_START = dict(code_stamp(), stamped="at process start")
except Exception as exc:                                   # never let custody code break a run
    STAMP_AT_START = dict(error=repr(exc))


def _cpu_stat():
    """(wall clock s, scheduler periods, throttled periods, throttled s) of this container's CPU quota, or None."""
    try:
        d = dict(l.split() for l in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines())
        return (time.time(), int(d["nr_periods"]), int(d["nr_throttled"]), int(d["throttled_usec"])/1e6)
    except Exception:
        return None


try:                                                        # the baseline is the package's first import, not this module's
    import gift_bags as _gb
    _CPU_AT_START = _gb.AT_IMPORT.get("cpu") or _cpu_stat()
except Exception:
    _CPU_AT_START = _cpu_stat()


def cpu_quota_stamp():
    """How starved of CPU this run was. The container's CPU quota is shared by every session in it; when it is
    exhausted the kernel stops all of them for the rest of each scheduler period, and a run's wall clock grows while
    its work does not. A wall/sim figure is an upper bound unless `throttled_fraction_of_periods` is near 0;
    `process_cpu_s_per_wall_s` against the thread count says how much of its threads the process really had."""
    now = _cpu_stat()
    if now is None or _CPU_AT_START is None:
        return None
    wall = now[0] - _CPU_AT_START[0]
    per = now[1] - _CPU_AT_START[1]
    return dict(wall_s_since_process_start=round(wall, 1), process_cpu_s=round(time.process_time(), 1),
                process_cpu_s_per_wall_s=round(time.process_time()/wall, 2) if wall > 0 else None,
                scheduler_periods=per, throttled_fraction_of_periods=round((now[2] - _CPU_AT_START[2])/per, 3) if per else None,
                container_throttled_s=round(now[3] - _CPU_AT_START[3], 1))


def thread_stamp():
    """How many solver threads this process ran with. The contact world is bit-reproducible on one thread and not on
    several (two one-thread runs of one command agreed in every logged number; runs at the default did
    not), so a result near a limit must be read with this beside it."""
    import os
    mod = sys.modules.get("gift_bags.ipcworld")
    n = getattr(mod, "SOLVER_THREADS", None)
    env = {k: os.environ.get(k) for k in ("GIFT_BAGS_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                          "OMP_WAIT_POLICY", "KMP_BLOCKTIME", "GIFT_BAGS_DEVICE", "GIFT_BAGS_FORCE_RULE",
                                          "GIFT_BAGS_FORCE_RTOL")}
    try:
        load = round(os.getloadavg()[0], 1)
    except OSError:
        load = None
    return dict(solver_threads=n, bit_reproducible=(n == 1) if n is not None else None, environment=env,
                robot_model_device=os.environ.get("GIFT_BAGS_DEVICE", "cuda:0"), load_1min_at_write=load,
                cpu_quota=cpu_quota_stamp())


def run_stamp():
    """The stamp taken at process start, plus the digest at write time when the tree changed while the run was going,
    and the solver thread count."""
    out = dict(STAMP_AT_START)
    out["threads"] = thread_stamp()
    try:
        now = physics_digest()
        if now != out.get("physics_digest"):
            out["physics_digest_when_written"] = now
            out["tree_changed_during_run"] = True
    except Exception:
        pass
    return out


def stamped(data):
    """A run record with the command line, the code revision, the dirty flag and the physics digest (at start)."""
    return dict(script=" ".join(["python", *sys.argv]), code=run_stamp(), **data)


def write_record(path, data, **dump):
    """Write a stamped evidence JSON (every probe and demo summary goes through here)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamped(data), indent=1, default=dump.get("default", float)))
    return path


def save_bench(name, bench, data):
    BENCHES.mkdir(parents=True, exist_ok=True)
    out = dict(bench=bench, script=" ".join(["python", "gift_bags/probes/" + Path(sys.argv[0]).name, *sys.argv[1:]]),
               code=run_stamp(), **data)                      # physics_digest inside code (taken at process start)
    p = BENCHES/f"{name}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", p, flush=True)
    return p
