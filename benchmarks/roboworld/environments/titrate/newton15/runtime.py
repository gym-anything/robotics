"""Strict runtime identity and workspace-local caches, independent of whiteboard."""

from importlib.metadata import version
from functools import lru_cache
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
PINNED = {"newton": "1.5.0", "warp-lang": "1.16.0"}


def configure():
    cache = ROOT / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("TEMP", "TMP", "CUDA_CACHE_PATH", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        directory = cache / name.lower()
        directory.mkdir(exist_ok=True)
        os.environ[name] = str(directory)
    os.environ["PIP_CACHE_DIR"] = str(cache / "pip")
    return cache


@lru_cache(maxsize=1)
def require_newton():
    found = {name: version(name) for name in PINNED}
    if found != PINNED:
        raise RuntimeError(f"Titration requires {PINNED}; found {found}. No fallback.")
    cache = configure()
    import warp as wp

    wp.config.kernel_cache_dir = str(cache / "warp")
    import newton

    newton.use_coord_layout_targets = True
    return newton, wp


def require_disk(path, reserve_gib=3.0):
    path = Path(path)
    existing = path
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < reserve_gib * 1024**3:
        raise OSError(f"Disk reserve: {free / 1024**3:.2f} GiB available; {reserve_gib} required")
