"""Evidence custody: hash every committed evidence file (JSON, PNG, JPG, MP4, MD) into
`barcode_scanning/evidence/MANIFEST.json`, with the git revision, the hashes of the core files that decide the
physics, sensing and scoring, and, per file, whether it was written after the last change to any core file (evidence
older than the code it describes is listed, not hidden).

A file's time is when its run FINISHED, not the code the run started with: a run launched before a change and
finishing after it looks current by time. Runs therefore record `code_at_launch` (`launch_stamp()`, the core hashes
when the process started); the manifest compares every stamp it finds with the current core files and lists the
evidence launched on other code.

  python -m barcode_scanning.manifest
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEEP = {".json", ".png", ".jpg", ".mp4", ".md", ".pdf"}
CORE = ("world.py", "scene.py", "stock.py", "env.py", "device.py", "decoder.py", "optics.py", "render.py", "housing.py",
        "params.py", "config.py", "task.py", "scoring.py", "catalog.py", "artwork.py", "packaging.py", "dressing.py",
        "scripted.py", "probes/fixtures.py", "render_worker.py", "skills.py", "grasping.py", "collide.py", "frontend.py")


def core_hashes():
    return {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest()[:16] for f in CORE}


def launch_stamp():
    """Call when a run starts; save the result with its evidence as `code_at_launch`."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return dict(utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), git_head=head, core_sha256_16=core_hashes())


def _stamps(o, depth=0):
    if isinstance(o, dict):
        if "code_at_launch" in o and isinstance(o["code_at_launch"], dict):
            yield o["code_at_launch"]
        if depth < 2:
            for v in o.values():
                yield from _stamps(v, depth + 1)


def tracked_like(p):
    """Files Git would keep (the .gitignore excludes raw recordings, frames, sessions and render exports)."""
    return subprocess.run(["git", "check-ignore", "-q", str(p)], cwd=ROOT.parent).returncode != 0


def build():
    core = core_hashes()
    last_core = max((ROOT/f).stat().st_mtime for f in CORE)
    files, stale, other_code, stamped = {}, {}, {}, 0
    for p in sorted((ROOT/"evidence").rglob("*")):
        if not (p.is_file() and p.suffix in KEEP and p.name != "MANIFEST.json" and tracked_like(p)):
            continue
        rel = str(p.relative_to(ROOT.parent))
        after = p.stat().st_mtime >= last_core
        files[rel] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size,
                          written_utc=datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
                          after_last_core_change=after)
        if not after:
            stale[rel] = files[rel]["written_utc"]
        if p.suffix == ".json":
            try:
                found = list(_stamps(json.loads(p.read_text())))
            except (ValueError, UnicodeDecodeError):
                found = []
            if found:
                stamped += 1
                changed = sorted({f for st in found for f, h in st.get("core_sha256_16", {}).items() if core.get(f) != h})
                files[rel]["code_at_launch_matches"] = not changed
                if changed:
                    other_code[rel] = changed
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "barcode_scanning"], cwd=ROOT.parent,
                                capture_output=True, text=True).stdout.strip())
    man = dict(built=datetime.now(timezone.utc).isoformat(timespec="seconds"), git_head=head, working_tree_dirty=dirty,
               core_sha256_16=core, last_core_change_utc=datetime.fromtimestamp(last_core, timezone.utc).isoformat(timespec="seconds"),
               written_before_last_core_change=stale, json_with_launch_stamp=stamped,
               launched_on_other_core_code=other_code, files=files, count=len(files),
               note="raw recordings (*.npz), per-frame PNGs, session folders and render exports are excluded by .gitignore "
                    "and kept in the workspace and the durable store")
    (ROOT/"evidence"/"MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(f"{len(files)} files hashed; {len(stale)} written before the last core change; {stamped} JSON files carry a "
          f"launch stamp, {len(other_code)} of them launched on other core code")
    return man


if __name__ == "__main__":
    build()
